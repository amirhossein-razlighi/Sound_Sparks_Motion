#!/usr/bin/env python3
"""P3/P4 - run the paper's optimisation with a few overrides, plus timing/memory logging.

Args = paper run's run_config.json (see common.load_paper_args) + --set overrides,
then optimize_qwen_vl.run(args) unchanged. Instrumentation is by wrapping two
functions in the loop module's namespace (no main-code edits):
  * render_with_injected_latents -> one call per optimisation iteration; the
    interval between consecutive call starts is one iteration (render + critic +
    perceptual + backward + Adam step); preview renders inside the interval are
    subtracted.
  * render_final_video -> previews (inside loop) and the final render.
Writes <output-dir>/timing.json and, on CUDA OOM, <output-dir>/OOM.json (then exits 3).

Usage:
  python run_opt.py --scenario dog_yawning --output-dir DIR --set audio_opt_last_steps=4 --set seed=1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import torch

from common import args_to_cli, critic_spec, load_paper_args


def _parse_val(v: str):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--set", action="append", default=[], help="key=value override of a paper arg")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    overrides = {}
    for kv in a.set:
        k, v = kv.split("=", 1)
        overrides[k] = _parse_val(v)
    overrides["output_dir"] = a.output_dir
    ns = load_paper_args(a.scenario, overrides)
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "iclr_overrides.json").write_text(json.dumps(
        {"scenario": a.scenario, "overrides": overrides, "critic": critic_spec(ns),
         "equivalent_cli": ["python", "editing/optimize_qwen_vl.py", *args_to_cli(ns)]}, indent=2))
    if a.dry_run:
        print(" ".join(args_to_cli(ns)))
        return

    import optimize_qwen_vl as oq
    import motion_opt.multimodal_loop_qwen as mlq

    rec = {"render_starts": [], "preview_intervals": [], "final_render": None}
    orig_render = mlq.render_with_injected_latents
    orig_final = mlq.render_final_video
    orig_final_top = oq.render_final_video

    def timed_render(*args, **kw):
        torch.cuda.synchronize()
        rec["render_starts"].append(time.time())
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

    t_start = time.time()
    status = "ok"
    try:
        torch.cuda.reset_peak_memory_stats()
        oq.run(ns)
    except torch.OutOfMemoryError as e:
        status = "OOM"
        (out / "OOM.json").write_text(json.dumps({
            "error": str(e)[:2000], "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
            "iterations_started": len(rec["render_starts"]), "traceback": traceback.format_exc()[-4000:]}, indent=2))
    finally:
        starts = rec["render_starts"]
        end = rec.get("loop_end", time.time())
        bounds = starts + [end]
        iters = []
        for i in range(len(starts)):
            lo, hi = bounds[i], bounds[i + 1]
            prev = sum(b - a_ for a_, b in rec["preview_intervals"] if a_ >= lo and b <= hi)
            iters.append(hi - lo - prev)
        post = iters[1:] if len(iters) > 1 else iters
        timing = {
            "status": status,
            "n_iterations": len(iters),
            "iter_seconds": iters,
            "mean_iter_s_post_warmup": (sum(post) / len(post)) if post else None,
            "preview_seconds_total": sum(b - a_ for a_, b in rec["preview_intervals"]),
            "final_render_s": rec["final_render"],
            "wall_total_s": time.time() - t_start,
            "peak_mem_gb_loop": rec.get("peak_mem_gb_loop"),
            "peak_mem_gb_total": torch.cuda.max_memory_allocated() / 1e9,
            "gpu": torch.cuda.get_device_name(0),
        }
        (out / "timing.json").write_text(json.dumps(timing, indent=2))
    if status == "OOM":
        sys.exit(3)


if __name__ == "__main__":
    main()
