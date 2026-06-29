#!/usr/bin/env python3
"""Critic-independent objective metrics for the audio-init ablation (rebuttal P2).

Given a finished run directory (the `--output-dir` of optimize_qwen_vl.py), this
compares the OPTIMIZED video against the BASELINE video using signals that do NOT
depend on the Qwen optimization critic, so the numbers cannot be dismissed as
circular VLM bias:

  * motion_strength        — mean RAFT optical-flow magnitude (principled motion)
  * motion_proxy_framediff — mean |Δ| between consecutive frames (no weights; always available)
  * source_preservation    — LPIPS(optimized, baseline), full clip + static prefix
  * temporal_flicker       — mean consecutive-frame LPIPS (lower = smoother)
  * clip_edit / clip_static / clip_edit_minus_static — CLIP prompt alignment

It also pulls the final Qwen yes-probability from best_params_<mode>.pt for
convenience (clearly labelled as the in-loop critic, reported but not load-bearing).

Every metric is computed in its own try/except: a missing model (e.g. RAFT weights
offline) degrades that one metric to null with an error string, never kills the run.

Writes `<output-dir>/metrics.json`. Standalone — imports the repo's own helpers,
changes no optimization logic.

Usage
-----
    python editing/scripts/eval_metrics.py \\
        --output-dir results/rebuttal/dog_yawning/zero \\
        --mode both \\
        [--edit-prompt "..."] [--static-prompt "..."] \\
        [--raft-weights /path/to/raft_large.pth] [--device cuda]

If --edit-prompt / --static-prompt are omitted they are read from
<output-dir>/run_config.json (written by every run).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from motion_opt.core import (
    compute_raft_flows,
    decode_video_frames_rgb,
    frames_rgb_uint8_to_chw_float,
    load_raft_components,
)

log = logging.getLogger("eval_metrics")


# --------------------------------------------------------------------------- IO

def _decode_chw(video_path: Path, max_frames: int, device: torch.device) -> torch.Tensor:
    """Decode a video to an [N, 3, H, W] float tensor in [0, 1]."""
    frames = decode_video_frames_rgb(
        str(video_path),
        max_frames if max_frames > 0 else None,
        1,        # frame_stride
        None,     # resize_to
        0,        # sample_start
    )
    if not frames:
        raise RuntimeError(f"No frames decoded from {video_path}")
    return frames_rgb_uint8_to_chw_float(frames, device)


def _pad_to_multiple(frames: torch.Tensor, m: int = 8) -> torch.Tensor:
    """RAFT needs H, W divisible by 8. Reflect-pad if needed."""
    h, w = frames.shape[-2:]
    ph, pw = (-h) % m, (-w) % m
    if ph == 0 and pw == 0:
        return frames
    return torch.nn.functional.pad(frames, (0, pw, 0, ph), mode="reflect")


# ----------------------------------------------------------------------- metrics

def _motion_raft(frames: torch.Tensor, raft, transforms) -> float:
    """Mean per-frame optical-flow magnitude over the clip."""
    flows = compute_raft_flows(_pad_to_multiple(frames), raft, transforms)  # [N-1, 2, H, W]
    if flows.shape[0] == 0:
        return 0.0
    mag = torch.linalg.norm(flows, dim=1)  # [N-1, H, W]
    return float(mag.mean().item())


def _motion_proxy_framediff(frames: torch.Tensor) -> float:
    """Weight-free motion proxy: mean abs difference between consecutive frames."""
    if frames.shape[0] < 2:
        return 0.0
    return float((frames[1:] - frames[:-1]).abs().mean().item())


def _lpips(a: torch.Tensor, b: torch.Tensor, backbone: str, max_frames: int) -> float:
    from motion_opt.perceptual_loss import compute_source_lpips_loss
    return float(compute_source_lpips_loss(a, b, backbone=backbone, max_frames=max_frames).item())


def _temporal_flicker(frames: torch.Tensor, backbone: str, max_pairs: int) -> float:
    from motion_opt.perceptual_loss import compute_temporal_consistency_loss
    # No src reference -> raw consecutive-frame LPIPS (absolute flicker level).
    return float(compute_temporal_consistency_loss(frames, None, backbone=backbone, max_pairs=max_pairs).item())


def _clip_sims(frames: torch.Tensor, edit_prompt: str, static_prompt: str,
               clip_name: str, device: torch.device, batch_size: int) -> dict:
    from motion_opt.clip_loss import (
        build_clip_model,
        compute_clip_dual_prompt_frame_similarities,
        encode_text_for_clip,
    )
    clip_model, clip_tok = build_clip_model(clip_name, device)
    edit_emb = encode_text_for_clip(edit_prompt, clip_model, clip_tok, device)
    static_emb = encode_text_for_clip(static_prompt, clip_model, clip_tok, device)
    scores = compute_clip_dual_prompt_frame_similarities(
        frames, static_emb, edit_emb, clip_model, batch_size=batch_size
    )
    n = max(len(scores["edit"]), 1)
    edit_mean = sum(scores["edit"]) / n
    static_mean = sum(scores["static"]) / n
    return {
        "clip_edit": edit_mean,
        "clip_static": static_mean,
        "clip_edit_minus_static": edit_mean - static_mean,
    }


def _final_qwen_yes_prob(output_dir: Path, mode: str) -> float | None:
    p = output_dir / f"mode_{mode}" / f"best_params_{mode}.pt"
    if not p.exists():
        return None
    try:
        d = torch.load(p, map_location="cpu")
        return float(d.get("qwen_score")) if d.get("qwen_score") is not None else None
    except Exception:
        return None


# -------------------------------------------------------------------------- main

def _resolve_prompts(args, output_dir: Path) -> tuple[str, str]:
    edit_prompt, static_prompt = args.edit_prompt, args.static_prompt
    cfg_path = output_dir / "run_config.json"
    if (edit_prompt is None or static_prompt is None) and cfg_path.exists():
        run_args = json.loads(cfg_path.read_text()).get("args", {})
        edit_prompt = edit_prompt or run_args.get("edit_prompt")
        static_prompt = static_prompt or run_args.get("static_prompt")
    # Fall back to a sensible static prompt mirroring optimize_qwen_vl.py's default.
    if static_prompt in (None, ""):
        static_prompt = f"A static frame before the edit: {edit_prompt} has not happened yet."
    return edit_prompt, static_prompt


def run(args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(args.output_dir).expanduser().resolve()

    baseline_path = Path(args.baseline_video) if args.baseline_video else output_dir / "baseline_video.mp4"
    optimized_path = (
        Path(args.optimized_video) if args.optimized_video
        else output_dir / f"mode_{args.mode}" / f"best_optimized_video_{args.mode}.mp4"
    )
    if not baseline_path.exists():
        raise FileNotFoundError(f"baseline video not found: {baseline_path}")
    if not optimized_path.exists():
        raise FileNotFoundError(f"optimized video not found: {optimized_path}")

    edit_prompt, static_prompt = _resolve_prompts(args, output_dir)
    log.info("Device=%s  edit=%r  static=%r", device, edit_prompt, static_prompt)

    base = _decode_chw(baseline_path, args.max_frames, device)
    opt = _decode_chw(optimized_path, args.max_frames, device)
    log.info("Decoded baseline=%s optimized=%s", tuple(base.shape), tuple(opt.shape))

    results: dict = {
        "output_dir": str(output_dir),
        "mode": args.mode,
        "baseline_video": str(baseline_path),
        "optimized_video": str(optimized_path),
        "edit_prompt": edit_prompt,
        "static_prompt": static_prompt,
        "n_frames_baseline": int(base.shape[0]),
        "n_frames_optimized": int(opt.shape[0]),
        "metrics": {},
        "errors": {},
    }
    M, E = results["metrics"], results["errors"]

    # 1) Motion — weight-free proxy (always available)
    try:
        M["motion_proxy_framediff_optimized"] = _motion_proxy_framediff(opt)
        M["motion_proxy_framediff_baseline"] = _motion_proxy_framediff(base)
        M["motion_proxy_framediff_delta"] = M["motion_proxy_framediff_optimized"] - M["motion_proxy_framediff_baseline"]
    except Exception as ex:  # pragma: no cover
        E["motion_proxy"] = repr(ex)

    # 2) Motion — RAFT optical flow (principled; needs weights)
    try:
        with torch.no_grad():
            raft, transforms = load_raft_components(device, args.raft_model, args.raft_weights)
            mo = _motion_raft(opt, raft, transforms)
            mb = _motion_raft(base, raft, transforms)
        M["motion_strength_optimized"] = mo
        M["motion_strength_baseline"] = mb
        M["motion_strength_delta"] = mo - mb
        M["motion_strength_ratio"] = (mo / mb) if mb > 1e-8 else None
        del raft, transforms
        if device.type == "cuda":
            torch.cuda.empty_cache()
    except Exception as ex:
        E["motion_raft"] = repr(ex)
        log.warning("RAFT motion skipped: %s", ex)

    # 3) Source preservation — LPIPS(optimized, baseline)
    try:
        with torch.no_grad():
            M["source_preservation_lpips"] = _lpips(opt, base, args.lpips_backbone, args.max_frames or 16)
            # static prefix: first `retake_start_frames`-ish region should be ~unchanged
            k = min(args.static_prefix_frames, base.shape[0], opt.shape[0])
            if k >= 1:
                M["source_preservation_lpips_static_prefix"] = _lpips(
                    opt[:k], base[:k], args.lpips_backbone, max(k, 1)
                )
    except Exception as ex:
        E["lpips"] = repr(ex)
        log.warning("LPIPS skipped: %s", ex)

    # 4) Temporal flicker
    try:
        with torch.no_grad():
            M["temporal_flicker_optimized"] = _temporal_flicker(opt, args.lpips_backbone, args.max_temporal_pairs)
            M["temporal_flicker_baseline"] = _temporal_flicker(base, args.lpips_backbone, args.max_temporal_pairs)
    except Exception as ex:
        E["temporal"] = repr(ex)
        log.warning("Temporal flicker skipped: %s", ex)

    # 5) CLIP prompt alignment
    try:
        with torch.no_grad():
            opt_clip = _clip_sims(opt, edit_prompt, static_prompt, args.clip_model, device, args.clip_batch_size)
            base_clip = _clip_sims(base, edit_prompt, static_prompt, args.clip_model, device, args.clip_batch_size)
        M["clip_edit_optimized"] = opt_clip["clip_edit"]
        M["clip_edit_baseline"] = base_clip["clip_edit"]
        M["clip_edit_minus_static_optimized"] = opt_clip["clip_edit_minus_static"]
        M["clip_edit_minus_static_baseline"] = base_clip["clip_edit_minus_static"]
        M["clip_edit_alignment_gain"] = opt_clip["clip_edit_minus_static"] - base_clip["clip_edit_minus_static"]
    except Exception as ex:
        E["clip"] = repr(ex)
        log.warning("CLIP alignment skipped: %s", ex)

    # 6) Final Qwen yes-prob (in-loop critic; reported, NOT load-bearing)
    M["final_qwen_yes_prob_critic"] = _final_qwen_yes_prob(output_dir, args.mode)

    out_json = output_dir / "metrics.json"
    out_json.write_text(json.dumps(results, indent=2, sort_keys=True))
    log.info("Wrote %s", out_json)
    print(json.dumps(results["metrics"], indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", required=True, help="Run output dir (contains baseline_video.mp4 + mode_<mode>/).")
    p.add_argument("--mode", default="both", help="Optimization mode subdir (both/audio/text).")
    p.add_argument("--baseline-video", default=None, help="Override baseline video path.")
    p.add_argument("--optimized-video", default=None, help="Override optimized video path.")
    p.add_argument("--edit-prompt", default=None, help="Edit prompt (else read from run_config.json).")
    p.add_argument("--static-prompt", default=None, help="Static prompt (else read from run_config.json).")
    p.add_argument("--device", default=None, help="cuda / cpu (auto if unset).")
    p.add_argument("--max-frames", type=int, default=0, help="Max frames to decode (0 = all).")
    p.add_argument("--static-prefix-frames", type=int, default=8, help="Frames treated as the static prefix.")
    p.add_argument("--raft-model", default="raft_large", choices=["raft_large", "raft_small"])
    p.add_argument("--raft-weights", default=None, help="Local RAFT weights .pth (for offline nodes).")
    p.add_argument("--lpips-backbone", default="alex", choices=["alex", "vgg"])
    p.add_argument("--max-temporal-pairs", type=int, default=16)
    p.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    p.add_argument("--clip-batch-size", type=int, default=8)
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
