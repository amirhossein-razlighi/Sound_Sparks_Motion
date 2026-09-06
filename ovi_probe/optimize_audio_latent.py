#!/usr/bin/env python3
"""Phase 3: optimize the Ovi audio-stream latent with our Qwen motion critic.

The Ovi analog of our LTX method: freeze everything (twin-DiT, VAEs, T5),
SDEdit-edit the source clip, and optimize ONLY the audio latent that seeds the
joint denoise. Gradients flow

    Qwen NLL on decoded frames -> Wan VAE decode -> last-K joint denoise steps
    -> audio->video cross-attention -> z_aud (the optimized parameter)

exactly mirroring the paper's audio-conditioning optimization, in a second,
architecturally different AV model. Reuses editing/src/motion_opt/qwen_loss.py
UNCHANGED.

Run from ovi_probe/Ovi (configs are cwd-relative):
    cd ovi_probe/Ovi && python ../optimize_audio_latent.py \
        --src-video ../../input_videos/cook_spinach_main.mp4 \
        --prompt "The man pets the dog. <AUDCAP>...<ENDAUDCAP>" \
        --edit-prompt "The man pets the dog." \
        --out-dir /scratch/amirrz/Ovi_exp/outputs/opt/man_pets_dog
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE + "/Ovi")
sys.path.insert(0, os.path.abspath(_HERE + "/../editing/src"))

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
log = logging.getLogger("ovi_opt")

from motion_opt.qwen_loss import (  # noqa: E402  (our module, unchanged)
    build_qwen_model,
    build_qwen_rubric_inputs,
    compute_qwen_video_loss,
)

# sibling module (same directory)
sys.path.insert(0, _HERE)
from sdedit_edit import (  # noqa: E402
    AUDIO_SR,
    FPS,
    encode_source,
    encode_text,
    load_audio_16k,
    load_engine,
    load_video_frames,
    sdedit_generate,
)
from ovi.utils.io_utils import save_video  # noqa: E402

DEFAULT_QWEN = os.environ.get("QWEN_ROOT", "/project/def-amahdavi/amirrz/HF/models/Qwen2.5-VL-7B-Instruct")


def render(engine, args, z_aud, with_grad: bool, decode_audio: bool = False,
           decode_video: bool = True):
    return sdedit_generate(
        engine,
        text_prompt=args.prompt,
        z_vid_src=args._z_vid,
        z_aud_start=z_aud,
        strength=args.strength,
        seed=args.seed,
        sample_steps=args.sample_steps,
        solver_name=args.solver,
        with_grad=with_grad,
        text_embeddings=args._text_embs,
        decode_video=decode_video,
        decode_audio=decode_audio,
        # Audio as per-step conditioning (LTX analog) in ALL renders, so the
        # last-K gradient window is valid and baseline/final are consistent.
        pin_audio=True,
        grad_last_steps=args.grad_last_steps if with_grad else 0,
    )


def frames_for_qwen(video_cfhw: torch.Tensor) -> torch.Tensor:
    """[3,F,H,W] in [-1,1] (grad ok) -> [F,3,H,W] in [0,1]."""
    return ((video_cfhw + 1.0) / 2.0).clamp(0.0, 1.0).permute(1, 0, 2, 3).contiguous()


def segmented_decode_grad(engine, vid_lat: torch.Tensor, g_frames: torch.Tensor,
                          seg: int = 2, ctx: int = 2) -> torch.Tensor:
    """Approximate dL/d(vid_lat) given dL/d(frames) via short segment decodes.

    The full 121-frame decode graph is ~70 GB — cannot exist on the H100 at
    all. And chunking the *input* of one big decode bounds nothing (autograd
    records whole-tensor ops as soon as one conv mixes detached and live
    latents). So instead each segment of `seg` latent frames (+`ctx` leading
    context frames, because the VAE is causal) is decoded as an INDEPENDENT
    short clip, its slice of the frame-gradient is backpropped, and the latent
    gradients are accumulated. Disjoint frame chunks' contributions add, so
    the only approximation is the causal-init transient at each segment start,
    absorbed by the context frames. (Pass 1 — the loss value itself — remains
    exact on the true full decode.)

    Frame map (causal Wan VAE): latent 0 -> 1 frame, latent g>=1 -> 4 frames,
    so latents [s,e) cover global frames [(s-1)*4+1, (e-1)*4+1) for s>=1.
    """
    vdev = next(engine.vae_model_video.model.parameters()).device
    n_lat = vid_lat.shape[1]
    lat_det = vid_lat.detach()
    G = torch.zeros_like(lat_det, dtype=torch.float32)
    s = 0
    while s < n_lat:
        e = min(s + seg, n_lat)
        a = max(0, s - ctx)
        live = lat_det[:, a:e].to(vdev).clone().requires_grad_(True)
        with torch.enable_grad():
            dec = engine.vae_model_video.wrapped_decode(live.unsqueeze(0)).squeeze(0)
        if s == 0:
            f_lo_loc, f_lo_glob = 0, 0
        else:
            f_lo_loc = (s - a - 1) * 4 + 1
            f_lo_glob = (s - 1) * 4 + 1
        f_hi_loc = (e - a - 1) * 4 + 1
        f_hi_glob = (e - 1) * 4 + 1
        sel = dec[:, f_lo_loc:f_hi_loc]
        sel.backward(g_frames[:, f_lo_glob:f_hi_glob].to(device=sel.device, dtype=sel.dtype))
        G[:, a:e] += live.grad.float().to(G.device)
        del dec, sel, live
        torch.cuda.empty_cache()
        s = e
    return G


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src-video", required=True)
    p.add_argument("--prompt", required=True, help="Ovi prompt incl. <AUDCAP>...<ENDAUDCAP>")
    p.add_argument("--edit-prompt", required=True, help="Plain edit sentence for the Qwen rubric")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--strength", type=float, default=0.6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample-steps", type=int, default=30)
    p.add_argument("--solver", default="unipc")
    p.add_argument("--iterations", type=int, default=25)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--spsa-c", type=float, default=0.1,
                   help="SPSA perturbation scale (latents are ~unit-scale).")
    p.add_argument("--opt-method", default="grad", choices=["grad", "spsa"],
                   help="grad = true gradients (needs 2 GPUs: VAE decode-grad "
                        "on cuda:1). spsa = zeroth-order, single GPU.")
    p.add_argument("--grad-last-steps", type=int, default=2,
                   help="Grad only through the last K denoise steps (needs "
                        "pin_audio; LTX audio_opt_last_steps analog).")
    p.add_argument("--decode-seg", type=int, default=2,
                   help="Latent frames per decode-grad segment (grad method).")
    p.add_argument("--decode-ctx", type=int, default=2,
                   help="Leading causal-context latent frames per segment.")
    p.add_argument("--latent-reg-weight", type=float, default=0.01)
    p.add_argument("--early-stopping", type=int, default=8)
    p.add_argument("--qwen-model", default=DEFAULT_QWEN)
    p.add_argument("--qwen-max-frames", type=int, default=24)
    p.add_argument("--qwen-img-size", type=int, default=224)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")

    # ---- Engine + source latents ----
    engine = load_engine()
    # Freeze the fusion model hard (we optimize an input, not weights).
    # NOTE: keep model.eval() — Ovi's inner per-block checkpointing must stay
    # OFF. Nesting it inside our outer per-step checkpoint made the inner
    # saved inputs accumulate across 30 blocks x 2 CFG calls x all steps
    # (OOM at 78 GB). With only the outer per-step checkpoint, the forward
    # drops activations progressively and each step's backward recomputes
    # just that one step's graph.
    engine.model.requires_grad_(False)
    if args.opt_method == "grad":
        # Last-K window mode: NO outer checkpoints, so Ovi's per-block
        # checkpointing can be enabled safely (the earlier OOM came from
        # NESTING it inside per-step checkpoints). Bounds the grad-step graph
        # to block inputs: 30 x 2 CFG x K steps x ~96 MB (~12 GB at K=2).
        # train() is behavior-neutral for this DiT (no dropout/BN).
        engine.model.gradient_checkpointing = True
        engine.model.train()
    else:
        engine.model.eval()
    video = load_video_frames(args.src_video, target_area=engine.target_area)
    audio = load_audio_16k(args.src_video)
    z_vid, z_aud_src = encode_source(engine, video, audio)
    args._z_vid = z_vid

    # Two-GPU gradient mode: park the video VAE on cuda:1 so the segmented
    # decode-grad has a whole H100 to itself (~4 GB of graph per decoded
    # frame; 13-frame segments ~55 GB). All no-grad decodes route there too.
    if args.opt_method == "grad":
        assert torch.cuda.device_count() >= 2, \
            "--opt-method grad needs 2 GPUs (VAE decode-grad on cuda:1)"
        v = engine.vae_model_video
        v.model = v.model.to("cuda:1")
        v.scale = [t.to("cuda:1") for t in v.scale]
        v.device = "cuda:1"
        log.info("video VAE moved to cuda:1")

    # Encode the prompt once and offload T5 (~11 GB) — the optimization loop
    # reuses the cached embeddings every iteration.
    args._text_embs = encode_text(engine, args.prompt)
    engine.offload_to_cpu(engine.text_model.model)
    torch.cuda.empty_cache()
    log.info("T5 offloaded; VRAM now %.1f GB", torch.cuda.memory_allocated() / 1e9)

    # ---- Qwen critic (ours, unchanged) ----
    log.info("loading Qwen critic from %s", args.qwen_model)
    qwen_model, qwen_processor = build_qwen_model(
        args.qwen_model, device=device, gradient_checkpointing=True)
    qwen_num_frames = args.qwen_max_frames + (args.qwen_max_frames % 2)
    cached_inputs, yes_id, no_id = build_qwen_rubric_inputs(
        processor=qwen_processor,
        edit_prompt=args.edit_prompt,
        num_frames=qwen_num_frames,
        img_size=args.qwen_img_size,
        device=device,
        motion_question=None,
        gradient_rubric="motion",
    )
    rubric_overrides = {"motion": 1.0, "entities": 0.0, "overall": 0.0}

    # Qwen (~17 GB) lives on CPU and visits the GPU only while scoring — the
    # denoise-backward and chunked-decode passes need every GB of headroom.
    qwen_model.to("cpu")
    torch.cuda.empty_cache()

    def _vram(tag: str) -> None:
        log.info("[vram] %-18s alloc=%.1f GB  peak=%.1f GB", tag,
                 torch.cuda.memory_allocated() / 1e9,
                 torch.cuda.max_memory_allocated() / 1e9)

    # SPSA needs a SMOOTH scalar, but the bf16 Qwen quantizes each nll reading
    # to ~0.12-nat steps (bf16 logit grid at |logit|~5-20) — a single reading
    # is too coarse to resolve the effect of a small latent perturbation.
    # Fix: average fp32 readings over FIXED frame windows (deterministic ->
    # common random numbers preserved). K windows dither the quantization to
    # ~K-fold finer resolution and probe different parts of the clip.
    _WINDOWS = [("linspace", 0), ("contiguous", 0), ("contiguous", 48), ("contiguous", 97)]

    def qwen_nll(video_cfhw, backward: bool = False):
        """Mean nll over the fixed windows. Takes the VIDEO-layout tensor
        [3,F,H,W] in [-1,1] (may be a grad leaf). With backward=True, each
        window builds a FRESH transform graph from the leaf and backprops
        (loss/N) immediately (LTX qwen_grad_accum pattern) — graphs are freed
        window by window and the leaf's .grad accumulates the gradient of the
        MEAN. A shared transform graph across windows would crash with
        'backward through the graph a second time'."""
        qwen_model.to(device)
        n = len(_WINDOWS)
        acc, details = 0.0, None
        for mode, start in _WINDOWS:
            with torch.enable_grad() if backward else torch.no_grad():
                qfr = frames_for_qwen(video_cfhw)  # fresh graph per window
                loss, details = compute_qwen_video_loss(
                    frames_chw=qfr,
                    qwen_model=qwen_model,
                    cached_inputs=cached_inputs,
                    yes_token_id=yes_id,
                    no_token_id=no_id,
                    max_frames=args.qwen_max_frames,
                    img_size=args.qwen_img_size,
                    backward=False,
                    return_details=True,
                    sample_mode=mode,
                    contiguous_start_frame=start,
                    rubric_weight_overrides=rubric_overrides,
                )
                if backward:
                    (loss / n).backward()
            acc += float(loss)
            del qfr, loss
        qwen_model.to("cpu")
        torch.cuda.empty_cache()
        return acc / n, details

    # ---- Baseline: SDEdit with the SOURCE audio latent, no optimization ----
    log.info("rendering no-opt SDEdit baseline...")
    with torch.no_grad():
        base = render(engine, args, z_aud_src, with_grad=False, decode_audio=True)
        base_frames = base["video"]
        base_loss, base_det = qwen_nll(base_frames, backward=False)
    base_yes = math.exp(-float(base_loss))
    log.info("baseline qwen: nll=%.4f yes_prob=%.4f", float(base_loss), base_yes)
    save_video(os.path.join(args.out_dir, "sdedit_baseline.mp4"),
               base["video"].cpu().float().numpy(), base["audio"],
               sample_rate=AUDIO_SR, fps=FPS)
    del base, base_frames
    torch.cuda.empty_cache()

    # ---- Optimize the audio latent ----
    # Default method: SPSA (zeroth-order). Full backprop through the denoise
    # chain works (28 GB peak, per-step checkpointed) but the Wan VAE's
    # streaming decoder costs multi-GB of autograd graph PER FRAME at Ovi's
    # mandatory 720^2-area resolution — differentiable decode of even 13
    # frames exceeds the H100. SPSA needs only forward renders (all no_grad),
    # estimating the gradient from paired perturbations with common random
    # numbers (same diffusion noise -> the score difference isolates the
    # audio-latent effect).
    z_aud = torch.nn.Parameter(z_aud_src.clone())
    opt = torch.optim.Adam([z_aud], lr=args.lr)
    best = {"loss": float("inf"), "z": z_aud_src.clone(), "iter": 0, "yes": base_yes}

    def score(z, want_frames: bool = False):
        """Total objective (qwen nll + reg) at z, all no-grad."""
        with torch.no_grad():
            o = render(engine, args, z, with_grad=False)
            fr = o["video"]
            l, _ = qwen_nll(fr, backward=False)
            reg = args.latent_reg_weight * torch.mean((z - z_aud_src) ** 2)
        nll = float(l)
        del o, fr
        torch.cuda.empty_cache()
        return nll, float(reg)

    csv_path = os.path.join(args.out_dir, "opt_log.csv")
    with open(csv_path, "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["iter", "qwen_nll", "yes_prob", "reg", "total", "grad_norm", "is_best", "sec"])

        for it in range(1, args.iterations + 1):
            t0 = time.time()
            torch.cuda.reset_peak_memory_stats()
            opt.zero_grad(set_to_none=True)

            if args.opt_method == "spsa":
                # Rademacher perturbation, deterministic per iteration.
                gen = torch.Generator(device=z_aud.device)
                gen.manual_seed(args.seed * 10007 + it)
                delta = (torch.randint(0, 2, z_aud.shape, generator=gen,
                                       device=z_aud.device, dtype=torch.int8)
                         .float() * 2.0 - 1.0)

                zp = (z_aud.detach() + args.spsa_c * delta)
                zm = (z_aud.detach() - args.spsa_c * delta)
                nll_p, reg_p = score(zp)
                nll_m, reg_m = score(zm)
                g_scale = ((nll_p + reg_p) - (nll_m + reg_m)) / (2.0 * args.spsa_c)
                z_aud.grad = g_scale * delta  # SPSA estimate (Rademacher: 1/delta = delta)
                gn = float(z_aud.grad.norm())
                opt.step()
                # Center evaluation at the updated point (tracks real progress).
                nll, reg = score(z_aud.detach())
            else:
                # True gradients (two-GPU): denoise-with-grad on GPU0
                # (per-step checkpointed), Qwen backprop to frames on GPU0,
                # segmented decode-grad on the VAE's GPU1.
                out = render(engine, args, z_aud, with_grad=True, decode_video=False)
                vid_lat = out["z_vid_final"]  # [C,Fl,H,W] on GPU0, grad-carrying
                _vram("after render")

                vdev = next(engine.vae_model_video.model.parameters()).device
                with torch.no_grad():
                    frames_full = engine.vae_model_video.wrapped_decode(
                        vid_lat.detach().unsqueeze(0).to(vdev)).squeeze(0).to(device)
                frames_leaf = frames_full.detach().clone().requires_grad_(True)
                loss_t, _det = qwen_nll(frames_leaf, backward=True)
                # qwen_nll backprops (loss/n) per window -> .grad IS the mean's grad
                g_frames = frames_leaf.grad
                del frames_full, frames_leaf
                torch.cuda.empty_cache()
                _vram("after pass1/qwen")

                G_lat = segmented_decode_grad(engine, vid_lat, g_frames,
                                              seg=args.decode_seg, ctx=args.decode_ctx)
                _vram("after pass2/decode")
                vid_lat.backward(G_lat.to(device=vid_lat.device, dtype=vid_lat.dtype))
                _vram("after chain bwd")
                del g_frames, G_lat

                reg_t = args.latent_reg_weight * torch.mean((z_aud - z_aud_src) ** 2)
                if reg_t.requires_grad:
                    reg_t.backward()
                gn = float(z_aud.grad.norm()) if z_aud.grad is not None else 0.0
                opt.step()
                nll, reg = float(loss_t), float(reg_t)
                del out, vid_lat, loss_t
                torch.cuda.empty_cache()

            yes = math.exp(-nll)
            total = nll + reg
            is_best = total < best["loss"]
            if is_best:
                best.update(loss=total, z=z_aud.detach().clone(), iter=it, yes=yes)
            dt = time.time() - t0
            w.writerow([it, nll, yes, reg, total, gn, int(is_best), f"{dt:.1f}"])
            fcsv.flush()
            log.info("iter %2d/%d nll=%.4f yes=%.4f reg=%.5f |g|=%.3e %s (%.1fs, peak %.1f GB)",
                     it, args.iterations, nll, yes, reg, gn,
                     "*BEST*" if is_best else "", dt,
                     torch.cuda.max_memory_allocated() / 1e9)

            if it - best["iter"] >= args.early_stopping:
                log.info("early stop at iter %d (best@%d)", it, best["iter"])
                break

    # ---- Final render with best latent ----
    log.info("final render with best audio latent (iter %d, yes=%.4f)", best["iter"], best["yes"])
    with torch.no_grad():
        fin = render(engine, args, best["z"], with_grad=False, decode_audio=True)
        fin_frames = fin["video"]
        fin_loss, _ = qwen_nll(fin_frames, backward=False)
    fin_yes = math.exp(-float(fin_loss))
    save_video(os.path.join(args.out_dir, "optimized.mp4"),
               fin["video"].cpu().float().numpy(), fin["audio"],
               sample_rate=AUDIO_SR, fps=FPS)
    torch.save(best["z"].cpu(), os.path.join(args.out_dir, "best_audio_latent.pt"))

    summary = {
        "baseline_yes_prob": base_yes,
        "optimized_yes_prob": fin_yes,
        "best_iter": best["iter"],
        "strength": args.strength,
        "optimizer": "SPSA (zeroth-order, common random numbers)",
        "sample_steps": args.sample_steps,
    }
    import json
    with open(os.path.join(args.out_dir, "summary.json"), "w") as fj:
        json.dump(summary, fj, indent=2)
    log.info("DONE. baseline yes=%.4f -> optimized yes=%.4f  (%s)",
             base_yes, fin_yes, args.out_dir)


if __name__ == "__main__":
    main()
