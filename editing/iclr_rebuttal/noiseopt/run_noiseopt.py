#!/usr/bin/env python3
"""Noise-optimisation baseline (simple DNO / ReNO style) for the "why audio?" question.

Same scenario settings as the paper run (common.load_paper_args), same Qwen2.5-VL critic call, same frozen LTX-2.3, but
the ONLY optimised variable is the initial Gaussian video noise of the Retake region. Text and audio conditioning stay
at their base values. Additive: main code and packages/ are untouched; two names in ltx_pipelines.retake are patched.

  eps     video noise, initialised from the seed's own draw (so iteration 1 is the seed's prompt-only Retake path)
  loss    Qwen NLL only (same passes as the paper config; no LPIPS / temporal / L2, as the LoRA control)
  update  Adam on eps, then the Retake-region part of eps is rescaled to norm sqrt(n) (stays on the Gaussian shell)
  grad    through ALL denoising steps by a manual step-wise reverse pass: forward without gradients storing each step's
          latents, then one step at a time backward (DiT block checkpointing on), so one step's graph is alive at a time.

Best iterate = lowest Qwen NLL, and the eps that was actually scored is the one kept (cloned BEFORE the update; the main
loop keeps the state one update later, see FINDINGS section 12).

Outputs in --output-dir: baseline_video.mp4 (prompt-only render, same seed), mode_noise/best_optimized_video_noise.mp4,
mode_noise/best_noise.pt (eps, seed noise, mask), noise_log.csv, iter_frames/iter_NNN.mp4 (every --iter-every), config.json,
timing.json (+ OOM.json on OOM).

Usage: python noiseopt/run_noiseopt.py --scenario dog_yawning --output-dir DIR [--noise-lr 0.02] [--set iterations=2]
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import math
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # editing/iclr_rebuttal
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "multicritic"))
from common import args_to_cli, critic_spec, load_paper_args  # noqa: E402
from run_opt import _parse_val  # noqa: E402

log = logging.getLogger("noiseopt")
MODE = "noise"


class Ctx:
    eps: torch.Tensor | None = None      # float32 leaf; None = use the seeded draw and create it
    create: bool = False                 # create eps from the seeded draw at the next video noising
    seed_noise: torch.Tensor | None = None
    mask: torch.Tensor | None = None
    trace: dict | None = None


def make_noiser(generator):
    """Replacement for retake.GaussianNoiser: 1st call of a render = video (retake.py noises video, then audio).

    The seeded draw is always taken, so the audio noise is the same draw as in the unpatched pipeline."""
    calls = {"n": 0}

    def noiser(latent_state, noise_scale: float = 1.0):
        lat = latent_state.latent
        noise = torch.randn(*lat.shape, device=lat.device, dtype=lat.dtype, generator=generator)
        if calls["n"] == 0:
            if Ctx.create:
                Ctx.seed_noise = noise.detach().float().clone()
                Ctx.eps = Ctx.seed_noise.clone().requires_grad_(True)
                Ctx.mask = latent_state.denoise_mask.detach().clone()
                Ctx.create = False
            if Ctx.eps is not None:
                assert Ctx.eps.shape == noise.shape, (Ctx.eps.shape, noise.shape)
                noise = Ctx.eps.to(dtype=lat.dtype)
        calls["n"] += 1
        m = latent_state.denoise_mask * noise_scale
        while m.dim() < noise.dim():
            m = m.unsqueeze(-1)
        return replace(latent_state, latent=(noise * m + lat * (1 - m)).to(lat.dtype))

    return noiser


def full_grad_euler(sigmas, video_state, audio_state, stepper, denoise_fn):
    """Euler loop for the optimisation renders: forward WITHOUT gradients, storing only each step's latents.

    Returns a detached, grad-requiring final video latent; adjoint() then pushes the gradient back one step at a time.
    (Nested torch checkpoints do not work here: every inner DiT-block checkpoint keeps its inputs alive, 75 GB at step 30.)"""
    import ltx_pipelines.utils.samplers as sm
    v0, a0 = video_state, audio_state

    def one(v_lat, a_lat, i):
        vs, as_ = replace(v0, latent=v_lat), replace(a0, latent=a_lat)
        dv, da = denoise_fn(vs, as_, sigmas, i)
        dv = sm.post_process_latent(dv, vs.denoise_mask, vs.clean_latent)
        da = sm.post_process_latent(da, as_.denoise_mask, as_.clean_latent)
        return stepper.step(v_lat, dv, sigmas, i), stepper.step(a_lat, da, sigmas, i)

    vs, avs = [v0.latent.detach()], [a0.latent.detach()]
    with torch.no_grad():
        for i in range(max(int(sigmas.shape[0]) - 1, 0)):
            nv, na = one(vs[-1], avs[-1], i)
            vs.append(nv.detach()); avs.append(na.detach())
    v_final = vs[-1].clone().requires_grad_(True)
    Ctx.trace = {"one": one, "v": vs[:-1], "a": avs[:-1], "v_final": v_final, "v_start": v0.latent}
    return replace(v0, latent=v_final), replace(a0, latent=avs[-1])


def adjoint():
    """Backward through the denoising steps from the final-latent gradient to eps. Audio's final gradient is zero (the
    audio decode is detached), but the audio latents still carry gradient between steps through the joint model."""
    t = Ctx.trace
    gv, ga = t["v_final"].grad, None
    assert gv is not None, "no gradient on the final latent"
    for i in reversed(range(len(t["v"]))):
        vi = t["v"][i].clone().requires_grad_(True)
        ai = t["a"][i].clone().requires_grad_(True)
        nv, na = t["one"](vi, ai, i)
        outs, grads = [nv], [gv]
        if ga is not None:
            outs.append(na); grads.append(ga)
        torch.autograd.backward(outs, grads)
        gv = vi.grad.detach()
        ga = ai.grad.detach() if ai.grad is not None else None
        del vi, ai, nv, na, outs, grads
    t["v_start"].backward(gradient=gv)
    Ctx.trace = None


def region(x, mask):
    m = mask
    while m.dim() < x.dim():
        m = m.unsqueeze(-1)
    return m.expand_as(x) > 0.5


@torch.no_grad()
def project(eps, mask):
    r = region(eps, mask)
    eps[r] *= math.sqrt(int(r.sum())) / eps[r].norm().clamp_min(1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--noise-lr", type=float, default=0.02)
    ap.add_argument("--iter-every", type=int, default=1, help="save the critic-scored video every N iterations")
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--set", action="append", default=[], help="key=value override of a paper arg")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

    overrides = {k: _parse_val(v) for k, v in (kv.split("=", 1) for kv in a.set)}
    overrides["output_dir"] = a.output_dir
    ns = load_paper_args(a.scenario, overrides)
    out = Path(a.output_dir)
    cfg = {"scenario": a.scenario, "overrides": overrides, "noise_lr": a.noise_lr, "critic": critic_spec(ns),
           "method": "Adam on the initial video noise, Retake-region norm projected to sqrt(n), Qwen NLL only, "
                     "gradient through all denoising steps", "paper_cli": args_to_cli(ns)}
    if a.dry_run:
        print(json.dumps(cfg, indent=2, default=str))
        return
    (out / "iter_frames").mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2, default=str))

    from optimize_lora_critic import _rubric_overrides
    from run_multicritic import save_mp4
    import ltx_pipelines.retake as R
    from ltx_pipelines.utils.constants import detect_params
    from motion_opt.core import _parse_loras, build_cached_source_latents, build_guiders_for_mode, compute_target_shape
    from motion_opt.models import build_retake_pipeline, resolve_quantization_policy
    from motion_opt.multimodal_loop import render_baseline_video, render_final_video, render_with_injected_latents
    from motion_opt.multimodal_loop_qwen import pre_encode_base_contexts
    from motion_opt.qwen_loss import build_qwen_model, build_qwen_rubric_inputs, compute_qwen_video_loss
    from motion_opt.runtime import build_retake_kwargs, prepare_retake_input_video

    torch.manual_seed(ns.seed)
    torch.cuda.manual_seed_all(ns.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")

    height, width, num_frames, frame_rate = compute_target_shape(ns.src_video, ns.height, ns.width, ns.num_frames, ns.frame_rate)
    duration = num_frames / frame_rate
    quant = resolve_quantization_policy(ns.retake_quantization if ns.retake_quantization is not None else ns.quantization)
    retake_input = str(prepare_retake_input_video(args=ns, is_main=True, output_dir=out, height=height, width=width,
                                                  num_frames=num_frames, frame_rate=frame_rate))
    params = detect_params(ns.checkpoint_path)
    vg, ag = build_guiders_for_mode(args=ns, params=params, use_low_memory_guidance=ns.low_memory_guidance)
    if getattr(vg, "skip_step", 0) or getattr(ag, "skip_step", 0):
        # the guider caches its last output for skipped steps, which a checkpoint recompute would read wrongly
        raise RuntimeError(f"skip_step != 0 is not supported here: {vg.skip_step} {ag.skip_step}")
    pipeline = build_retake_pipeline(checkpoint_path=ns.checkpoint_path, gemma_root=ns.gemma_root, loras=_parse_loras(ns.loras),
                                     device=device, quant_policy=quant, gradient_checkpointing=ns.gradient_checkpointing)
    cached_video, base_audio, wave_sr = build_cached_source_latents(pipeline=pipeline, src_video=retake_input, height=height,
                                                                    width=width, num_frames=num_frames, audio_sr=ns.audio_sr,
                                                                    device=device)
    pos, neg = pre_encode_base_contexts(pipeline=pipeline, pos_prompt=ns.edit_prompt, neg_prompt=ns.negative_prompt, device=device)
    qwen, proc = build_qwen_model(ns.qwen_model, device=device, gradient_checkpointing=ns.gradient_checkpointing)
    q_inputs, yes_id, no_id = build_qwen_rubric_inputs(processor=proc, edit_prompt=ns.edit_prompt,
                                                       num_frames=ns.qwen_max_frames + (ns.qwen_max_frames % 2),
                                                       img_size=ns.qwen_img_size, device=device,
                                                       motion_question=ns.qwen_motion_question,
                                                       gradient_rubric=ns.qwen_gradient_rubric)
    kwargs = build_retake_kwargs(args=ns, frame_rate=frame_rate, duration=duration, video_guider_params=vg, audio_guider_params=ag)
    final_kwargs = dict(kwargs)
    if ns.final_retake_num_inference_steps is not None:
        final_kwargs["num_inference_steps"] = ns.final_retake_num_inference_steps
    fvg, fag = build_guiders_for_mode(args=ns, params=params, use_low_memory_guidance=False)
    final_kwargs["video_guider_params"], final_kwargs["audio_guider_params"] = fvg, fag

    if not a.no_baseline:
        render_baseline_video(pipeline=pipeline, src_video=retake_input, cached_video_latent=cached_video, base_audio_latent=base_audio,
                              base_pos_context=pos, base_neg_context=neg, retake_kwargs=final_kwargs,
                              output_path=out / "baseline_video.mp4", num_frames=num_frames, frame_rate=frame_rate, audio_sr=wave_sr)

    orig_noiser, orig_euler = R.GaussianNoiser, R.euler_denoising_loop
    R.GaussianNoiser = lambda generator: make_noiser(generator)
    R.euler_denoising_loop = full_grad_euler
    Ctx.create, Ctx.eps = True, None

    rubric = _rubric_overrides(ns)
    accum = max(1, int(getattr(ns, "qwen_grad_accum_steps", 1)))
    mode0 = getattr(ns, "qwen_sample_mode", "linspace")
    qkw = dict(qwen_model=qwen, cached_inputs=q_inputs, yes_token_id=yes_id, no_token_id=no_id, max_frames=ns.qwen_max_frames,
               img_size=ns.qwen_img_size, contiguous_start_frame=getattr(ns, "qwen_contiguous_start_frame", 0),
               rubric_weight_overrides=rubric)
    opt, sched = None, None
    best = {"nll": float("inf"), "iter": 0, "eps": None}
    iters, t_start, status = [], time.time(), "ok"
    lf = open(out / "noise_log.csv", "w", newline="")
    w = csv.writer(lf)
    w.writerow(["iter", "qwen_nll", "qwen_yes_prob", "grad_norm", "delta_rms_region", "cos_to_seed_region", "iter_s", "peak_gb", "is_best"])
    torch.cuda.reset_peak_memory_stats()
    try:
        for it in range(1, ns.iterations + 1):
            torch.cuda.synchronize(); t0 = time.time()
            if Ctx.eps is not None and Ctx.eps.grad is not None:
                Ctx.eps.grad = None
            frames = render_with_injected_latents(pipeline=pipeline, src_video=retake_input, injected_audio_latent=base_audio,
                                                  cached_video_latent=cached_video, retake_kwargs=kwargs, max_frames=ns.max_eval_frames,
                                                  frame_stride=ns.frame_stride, resize_to=None,
                                                  audio_opt_last_steps=0,  # 0 -> calls R.euler_denoising_loop = full_grad_euler
                                                  eval_sample_start=0, pos_context=pos, neg_context=neg, inject_text_context=True)
            leaf = frames.detach().requires_grad_(True)
            total = torch.zeros((), device=frames.device)
            for p in range(accum):
                l = compute_qwen_video_loss(frames_chw=leaf, backward=False, return_details=False,
                                            sample_mode=mode0 if p == 0 else "contiguous_random", **qkw)
                (l / accum).backward()
                total = total + l.detach()
            frames.backward(gradient=leaf.grad)
            del frames
            torch.cuda.empty_cache()
            adjoint()
            nll = float(total) / accum
            eps = Ctx.eps
            if eps.grad is None or float(eps.grad.norm()) == 0.0:
                raise RuntimeError("no gradient reached the noise")
            grad_norm = float(eps.grad.norm())
            if ns.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_([eps], max_norm=ns.grad_clip)
            if opt is None:
                opt = torch.optim.Adam([eps], lr=a.noise_lr)
                if getattr(ns, "lr_schedule", "constant") == "cosine" and ns.iterations > 1:
                    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ns.iterations, eta_min=a.noise_lr * 0.01)
            is_best = nll < best["nll"]
            if is_best:  # the eps that produced this score, before the update
                best.update(nll=nll, iter=it, eps=eps.detach().clone())
            r = region(eps, Ctx.mask)
            d = eps.detach()[r] - Ctx.seed_noise[r]
            rms = float(d.pow(2).mean().sqrt())
            cos = float(torch.nn.functional.cosine_similarity(eps.detach()[r], Ctx.seed_noise[r], dim=0))
            if a.iter_every > 0 and (it == 1 or it % a.iter_every == 0):
                try:
                    save_mp4(leaf, out / "iter_frames" / f"iter_{it:03d}.mp4", a.fps)
                except Exception as e:  # logging must never kill the run
                    log.warning("iter video failed: %s", e)
            opt.step()
            project(eps, Ctx.mask)
            if sched is not None:
                sched.step()
            torch.cuda.synchronize(); dt = time.time() - t0
            iters.append(dt)
            peak = torch.cuda.max_memory_allocated() / 1e9
            w.writerow([it, f"{nll:.6f}", f"{math.exp(-nll):.6f}", f"{grad_norm:.4f}", f"{rms:.5f}", f"{cos:.5f}", f"{dt:.2f}", f"{peak:.2f}", int(is_best)])
            lf.flush()
            log.info("iter %d/%d nll=%.4f P=%.4f grad=%.3f rms=%.4f cos=%.4f %.1fs peak=%.1fGB%s", it, ns.iterations, nll, math.exp(-nll),
                     grad_norm, rms, cos, dt, peak, " *" if is_best else "")
            del leaf, total
            gc.collect(); torch.cuda.empty_cache()
            if ns.early_stopping > 0 and it - best["iter"] >= ns.early_stopping:
                log.info("early stop at %d (best %d)", it, best["iter"])
                break
        peak_loop = torch.cuda.max_memory_allocated() / 1e9
        md = out / f"mode_{MODE}"
        md.mkdir(exist_ok=True)
        torch.save({"eps": best["eps"].cpu(), "seed_noise": Ctx.seed_noise.cpu(), "mask": Ctx.mask.cpu(), "best_iter": best["iter"],
                    "qwen_nll": best["nll"]}, md / "best_noise.pt")
        R.euler_denoising_loop = orig_euler
        Ctx.eps = best["eps"]
        t1 = time.time()
        render_final_video(mode=MODE, best={"delta_v": None, "audio_latent": None}, pipeline=pipeline, src_video=retake_input,
                           cached_video_latent=cached_video, base_audio_latent=base_audio, base_pos_context=pos, base_neg_context=neg,
                           retake_kwargs=final_kwargs, output_dir=md, num_frames=num_frames, frame_rate=frame_rate, audio_sr=wave_sr,
                           audio_opt_last_steps=ns.audio_opt_last_steps, skip_baseline=True)
        final_s = time.time() - t1
    except torch.OutOfMemoryError as e:
        status, peak_loop, final_s = "OOM", torch.cuda.max_memory_allocated() / 1e9, None
        (out / "OOM.json").write_text(json.dumps({"error": str(e)[:2000], "traceback": traceback.format_exc()[-4000:],
                                                  "iterations_done": len(iters)}, indent=2))
    finally:
        lf.close()
        R.GaussianNoiser, R.euler_denoising_loop = orig_noiser, orig_euler
    post = iters[1:] or iters
    (out / "timing.json").write_text(json.dumps({"status": status, "n_iterations": len(iters), "iter_seconds": iters,
                                                 "mean_iter_s_post_warmup": sum(post) / len(post) if post else None,
                                                 "best_iter": best["iter"], "best_p": math.exp(-best["nll"]) if best["iter"] else None,
                                                 "final_render_s": final_s, "peak_mem_gb_loop": peak_loop,
                                                 "wall_total_s": time.time() - t_start, "gpu": torch.cuda.get_device_name(0)}, indent=2))
    if status == "OOM":
        sys.exit(3)


if __name__ == "__main__":
    main()
