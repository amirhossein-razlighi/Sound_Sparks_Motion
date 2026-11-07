#!/usr/bin/env python3
"""Multi-critic proof of concept (reviewer: "does multi-critic optimisation show the method is not reward hacking?").

Runs the paper's optimisation for one scenario with its own run_config (common.load_paper_args, as run_opt.py), with
ONE change: the in-loop critic loss becomes a weighted mean of
    Qwen2.5-VL-7B  (the paper's critic, unchanged code path, same question, same frames, same passes)
    InternVL3-8B   (second critic, own GPU, same question; same frames as each Qwen pass, thinned to <= 16)
  loss = w_q * NLL_qwen + w_i * NLL_internvl   (default 0.5 / 0.5, so the loss scale and the regulariser balance stay
  comparable with the single-critic run).
Everything else (LTX, latents, regularisers, LPIPS/temporal, schedule, best-checkpoint rule, previews) is untouched:
the loop module's `compute_qwen_video_loss` name is replaced by a wrapper (no main-code edits). With --critics qwen25
the wrapper only logs (control arm A: identical code path, single critic).

RNG: the Qwen pass's frame indices are recomputed with the CUDA RNG state the Qwen call saw, and the post-call state is
restored, so the Qwen frame windows of arm B are drawn exactly as in arm A.

Extra outputs in <output-dir>/multicritic/:
  passes.csv      one row per critic call: iter, pass, grad, sample_mode, frames given to each critic, NLL and
                  P(yes) of every critic, combined loss, seconds per critic
  iter_frames/    the rendered video the critics scored at every iteration (pass 0), mp4 (the current iterate, not
                  only the best-so-far previews that the loop writes every 5 iterations to visualizations/)
  config.json     critic setup (question, weights, devices, frame rule, token lists)
Plus timing.json as run_opt.py.

Usage:
  python run_multicritic.py --scenario red_car_door_opens --output-dir DIR [--critics qwen25,internvl3]
         [--weights 0.5,0.5] [--internvl-device cuda:1] [--internvl-max-frames 16] [--set key=val ...]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # editing/iclr_rebuttal
from common import args_to_cli, critic_spec, load_paper_args  # noqa: E402
from run_opt import _parse_val  # noqa: E402

_TP = 2  # Qwen temporal patch size (qwen_loss._TEMPORAL_PATCH_SIZE)


class _GradScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w):
        ctx.w = w
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return g * ctx.w, None


def qwen_indices(n, max_frames, sample_mode, contiguous_start_frame, device):
    """Exact replica of compute_qwen_video_loss's frame selection (must be called with the same RNG state)."""
    from motion_opt.qwen_loss import _normal_sample_indices
    num = min(max_frames, n) if max_frames > 0 else n
    if num % _TP:
        num -= num % _TP
    num = max(num, _TP)
    if sample_mode == "linspace":
        return torch.linspace(0, n - 1, num, device=device).round().long()
    if sample_mode == "normal":
        return _normal_sample_indices(n, num, device)
    if sample_mode == "contiguous":
        s = min(max(int(contiguous_start_frame), 0), max(n - num, 0))
        return torch.arange(s, s + num, device=device).long()
    if sample_mode == "contiguous_random":
        lo = min(max(int(contiguous_start_frame), 0), max(n - num, 0))
        hi = max(n - num, lo)
        s = lo if hi <= lo else int(torch.randint(lo, hi + 1, (1,), device=device).item())
        return torch.arange(s, s + num, device=device).long()
    raise ValueError(sample_mode)


def save_mp4(frames_chw: torch.Tensor, path: Path, fps: float):
    import av
    import numpy as np
    x = (frames_chw.detach().float().clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    h, w = x.shape[1] // 2 * 2, x.shape[2] // 2 * 2
    c = av.open(str(path), "w")
    s = c.add_stream("libx264", rate=int(round(fps)))
    s.width, s.height, s.pix_fmt = w, h, "yuv420p"
    s.options = {"crf": "16"}
    for f in x:
        for p in s.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(f[:h, :w]), format="rgb24")):
            c.mux(p)
    for p in s.encode():
        c.mux(p)
    c.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--critics", default="qwen25,internvl3")
    ap.add_argument("--weights", default="0.5,0.5")
    ap.add_argument("--internvl-device", default="cuda:1")
    ap.add_argument("--internvl-max-frames", type=int, default=16)
    ap.add_argument("--fps", type=float, default=24.0, help="fps of the per-iteration mp4s only")
    ap.add_argument("--set", action="append", default=[], help="key=value override of a paper arg")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    critics = a.critics.split(",")
    assert critics[0] == "qwen25" and set(critics) <= {"qwen25", "internvl3"}, critics
    use_ivl = "internvl3" in critics
    wq, wi = (float(x) for x in a.weights.split(",")) if use_ivl else (1.0, 0.0)

    overrides = {k: _parse_val(v) for k, v in (kv.split("=", 1) for kv in a.set)}
    overrides["output_dir"] = a.output_dir
    ns = load_paper_args(a.scenario, overrides)
    spec = critic_spec(ns)
    out = Path(a.output_dir)
    mc = out / "multicritic"
    (mc / "iter_frames").mkdir(parents=True, exist_ok=True)
    cfg = {"scenario": a.scenario, "overrides": overrides, "critics": critics, "weights": {"qwen25": wq, "internvl3": wi},
           "question": spec["question"], "qwen_critic": spec, "internvl_device": a.internvl_device,
           "internvl_frames": f"the Qwen pass's frames, uniformly thinned to <= {a.internvl_max_frames}",
           "equivalent_cli": ["python", "editing/optimize_qwen_vl.py", *args_to_cli(ns)]}
    if a.dry_run:
        print(json.dumps(cfg, indent=2))
        return

    import optimize_qwen_vl as oq
    import motion_opt.multimodal_loop_qwen as mlq

    ivl = None
    if use_ivl:
        from internvl_critic import InternVLCritic
        ivl = InternVLCritic(spec["question"], device=a.internvl_device, max_frames=a.internvl_max_frames)
        cfg.update(internvl_yes_tokens=ivl.yes_tokens, internvl_no_tokens=ivl.no_tokens, internvl_size=ivl.size,
                   internvl_gradient_checkpointing=ivl.gc)
    (mc / "config.json").write_text(json.dumps(cfg, indent=2))

    # ---------------- critic wrapper ----------------
    st = {"iter": 0, "pass": 0}  # loop iterations start at 1 (multimodal_loop_qwen: range(1, num_iters + 1))
    pf = open(mc / "passes.csv", "w", newline="")
    pw = csv.writer(pf)
    pw.writerow(["iter", "pass", "grad", "sample_mode", "qwen_frames", "internvl_frames", "qwen_nll", "qwen_yes_prob",
                 "internvl_nll", "internvl_yes_prob", "combined_loss", "qwen_s", "internvl_s", "wall"])
    orig_loss = mlq.compute_qwen_video_loss

    def mc_loss(frames_chw, qwen_model, cached_inputs, yes_token_id, no_token_id, max_frames=8, img_size=224,
                backward=False, return_details=False, sample_mode="linspace", contiguous_start_frame=0,
                rubric_weight_overrides=None):
        dev = frames_chw.device
        grad = torch.is_grad_enabled() and frames_chw.requires_grad
        cpu0, cu0 = torch.get_rng_state(), torch.cuda.get_rng_state(dev)
        kw = dict(cached_inputs=cached_inputs, yes_token_id=yes_token_id, no_token_id=no_token_id,
                  max_frames=max_frames, img_size=img_size, return_details=True, sample_mode=sample_mode,
                  contiguous_start_frame=contiguous_start_frame, rubric_weight_overrides=rubric_weight_overrides)
        leaf = None
        t0 = time.time()
        if backward and grad and ivl is not None:
            # accumulate both critics' gradients on a detached leaf, then push once through the render graph
            leaf = frames_chw.detach().requires_grad_(True)
            lq, det = orig_loss(frames_chw=_GradScale.apply(leaf, wq), qwen_model=qwen_model, backward=True, **kw)
        else:
            lq, det = orig_loss(frames_chw=frames_chw, qwen_model=qwen_model, backward=backward, **kw)
        torch.cuda.synchronize(dev); tq = time.time() - t0
        cpu1, cu1 = torch.get_rng_state(), torch.cuda.get_rng_state(dev)
        torch.set_rng_state(cpu0); torch.cuda.set_rng_state(cu0, dev)
        qidx = qwen_indices(frames_chw.shape[0], max_frames, sample_mode, contiguous_start_frame, dev)
        torch.set_rng_state(cpu1); torch.cuda.set_rng_state(cu1, dev)
        qnll = float(lq.detach().float()) if not det else sum(d["weight"] * d["nll"] for d in det)

        inll, ti, iidx = float("nan"), 0.0, []
        combined = lq
        if ivl is not None:
            iidx = ivl.pick(qidx)
            t1 = time.time()
            src = leaf if leaf is not None else frames_chw
            li = ivl.nll(src[iidx.to(dev)])
            inll = float(li.detach())
            if leaf is not None:
                (wi * li).backward()
                frames_chw.backward(gradient=leaf.grad)
                combined = (wq * lq.detach().float() + wi * li.detach().to(dev))
            elif backward:  # no grad but backward requested (cannot happen in the loop) -> values only
                combined = wq * lq.detach().float() + wi * li.detach().to(dev)
            else:
                combined = wq * lq.float() + wi * li.to(dev)
            torch.cuda.synchronize(ivl.device); ti = time.time() - t1
            det = list(det) + [{"name": "internvl3", "weight": wi, "nll": inll, "yes_prob": float(torch.exp(-li.detach()))}]

        pw.writerow([st["iter"], st["pass"], int(grad), sample_mode, " ".join(map(str, qidx.tolist())),
                     " ".join(map(str, [int(i) for i in iidx])), f"{qnll:.6f}", f"{torch.exp(torch.tensor(-qnll)).item():.6f}",
                     f"{inll:.6f}", f"{torch.exp(torch.tensor(-inll)).item():.6f}" if ivl is not None else "",
                     f"{float(combined.detach()):.6f}", f"{tq:.2f}", f"{ti:.2f}", f"{time.time():.1f}"])
        pf.flush()
        if st["pass"] == 0 and grad:
            try:
                save_mp4(frames_chw, mc / "iter_frames" / f"iter_{st['iter']:03d}.mp4", a.fps)
            except Exception as e:  # logging must never kill the run
                print("iter_frames save failed:", e)
        st["pass"] += 1
        return (combined, det) if return_details else combined

    mlq.compute_qwen_video_loss = mc_loss

    # ---------------- timing (as run_opt.py) ----------------
    rec = {"render_starts": [], "preview_intervals": [], "final_render": None}
    orig_render, orig_final, orig_final_top = mlq.render_with_injected_latents, mlq.render_final_video, oq.render_final_video

    def timed_render(*args, **kw):
        torch.cuda.synchronize()
        rec["render_starts"].append(time.time())
        st["iter"] += 1; st["pass"] = 0
        return orig_render(*args, **kw)

    def timed_preview(*args, **kw):
        torch.cuda.synchronize(); t0 = time.time()
        try:
            return orig_final(*args, **kw)
        finally:
            torch.cuda.synchronize(); rec["preview_intervals"].append((t0, time.time()))

    def timed_final(*args, **kw):
        rec["loop_end"] = time.time()
        rec["peak_mem_gb_loop"] = torch.cuda.max_memory_allocated() / 1e9
        torch.cuda.synchronize(); t0 = time.time()
        try:
            return orig_final_top(*args, **kw)
        finally:
            torch.cuda.synchronize(); rec["final_render"] = time.time() - t0

    mlq.render_with_injected_latents = timed_render
    mlq.render_final_video = timed_preview
    oq.render_final_video = timed_final

    t_start, status = time.time(), "ok"
    try:
        torch.cuda.reset_peak_memory_stats()
        oq.run(ns)
    except torch.OutOfMemoryError as e:
        status = "OOM"
        (out / "OOM.json").write_text(json.dumps({"error": str(e)[:2000], "traceback": traceback.format_exc()[-4000:],
                                                  "iterations_started": len(rec["render_starts"])}, indent=2))
    finally:
        pf.close()
        starts, end = rec["render_starts"], rec.get("loop_end", time.time())
        bounds = starts + [end]
        iters = [bounds[i + 1] - bounds[i] - sum(b - x for x, b in rec["preview_intervals"] if x >= bounds[i] and b <= bounds[i + 1])
                 for i in range(len(starts))]
        post = iters[1:] if len(iters) > 1 else iters
        timing = {"status": status, "n_iterations": len(iters), "iter_seconds": iters,
                  "mean_iter_s_post_warmup": (sum(post) / len(post)) if post else None,
                  "preview_seconds_total": sum(b - x for x, b in rec["preview_intervals"]),
                  "final_render_s": rec["final_render"], "wall_total_s": time.time() - t_start,
                  "peak_mem_gb_loop": rec.get("peak_mem_gb_loop"), "peak_mem_gb_total": torch.cuda.max_memory_allocated() / 1e9,
                  "peak_mem_gb_internvl": (torch.cuda.max_memory_allocated(ivl.device) / 1e9) if ivl is not None else None,
                  "gpu": torch.cuda.get_device_name(0)}
        (out / "timing.json").write_text(json.dumps(timing, indent=2))
    if status == "OOM":
        sys.exit(3)


if __name__ == "__main__":
    main()
