#!/usr/bin/env python3
"""Probe Qwen2.5-VL for forward and backward non-determinism.

Runs the same Qwen forward+backward pass N times on IDENTICAL pixel values
and measures:
  - P("yes") drift  -> forward non-determinism
  - gradient cosine similarity to run 0  -> backward non-determinism
  - gradient L2 norm drift

Does this under three conditions (one at a time, set --mode):
  1. baseline   : torch defaults, cudnn.deterministic=False
  2. deterministic : cudnn.deterministic=True, cudnn.benchmark=False
  3. seeded      : same as deterministic + manual_seed before every call

Usage (on the cluster, after activating the venv):
    python editing/scripts/probe_qwen_nondeterminism.py \\
        --qwen-model /project/def-amahdavi/amirrz/HF/models/Qwen2.5-VL-7B-Instruct \\
        --video input_videos/a_red_ferrari_standing_still_in_the.mp4 \\
        --edit-prompt "A red car door opens." \\
        --n-repeats 20 \\
        --mode baseline

    # Run all three modes in sequence:
    for mode in baseline deterministic seeded; do
        python editing/scripts/probe_qwen_nondeterminism.py \\
            --qwen-model /project/.../Qwen2.5-VL-7B-Instruct \\
            --video input_videos/... \\
            --edit-prompt "A red car door opens." \\
            --n-repeats 20 \\
            --mode $mode 2>&1 | tee probe_${mode}.log
    done
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Make editing/src importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from audio_latent_opt.qwen_loss import (
    build_qwen_model,
    build_qwen_rubric_inputs,
    QWEN_IMG_SIZE,
    _TEMPORAL_PATCH_SIZE,
)


def load_video_frames(video_path: str, n_frames: int = 8, img_size: int = QWEN_IMG_SIZE) -> torch.Tensor:
    """Load a video and extract n_frames via linspace sampling.
    Returns float32 tensor [n_frames, 3, img_size, img_size] in [0, 1].
    """
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        raise RuntimeError(f"Could not read video: {video_path}")

    indices = torch.linspace(0, total - 1, n_frames).round().long().tolist()
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"Could not read frame {idx} from {video_path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        frames.append(frame)
    cap.release()

    arr = torch.from_numpy(
        __import__("numpy").stack(frames, axis=0).astype("float32") / 255.0
    )  # [T, H, W, 3]
    return arr.permute(0, 3, 1, 2).contiguous()  # [T, 3, H, W]


def run_single_pass(
    frames_fixed: torch.Tensor,
    qwen_model,
    cached_inputs: dict,
    yes_token_id: int,
    no_token_id: int,
) -> tuple[float, torch.Tensor]:
    """One forward+backward pass on a fresh clone of frames_fixed.
    Returns (yes_prob, grad_flat).
    """
    # Fresh leaf so .grad doesn't accumulate across calls
    frames = frames_fixed.clone().requires_grad_(True)

    # --- replicate _frames_to_pixel_values from qwen_loss.py ---
    from audio_latent_opt.qwen_loss import _frames_to_pixel_values

    pixel_values = _frames_to_pixel_values(frames).to(dtype=torch.bfloat16)

    # Use the first rubric item's inputs (motion question) for simplicity
    if "rubric_items" in cached_inputs:
        inputs = cached_inputs["rubric_items"][0]["inputs"]
    else:
        inputs = cached_inputs

    outputs = qwen_model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        pixel_values_videos=pixel_values,
        video_grid_thw=inputs.get("video_grid_thw"),
    )

    last_logits = outputs.logits[0, -1, :].float()
    yes_logit = last_logits[yes_token_id]
    no_logit = last_logits[no_token_id]
    yes_prob = torch.softmax(torch.stack([yes_logit, no_logit]), dim=0)[0]
    loss = -torch.log(yes_prob + 1e-8)
    loss.backward()

    grad_flat = frames.grad.detach().clone().float().view(-1)
    return float(yes_prob.detach()), grad_flat


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def run_experiment(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Mode: {args.mode}")
    print(f"  Device: {device}")
    print(f"{'='*60}")

    # Apply mode settings
    if args.mode == "baseline":
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    elif args.mode in ("deterministic", "seeded"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    # Load model
    print("Loading Qwen2.5-VL...")
    qwen_model, processor = build_qwen_model(args.qwen_model, device, gradient_checkpointing=False)

    # Build cached inputs (text only, no pixel values)
    print("Building cached inputs...")
    n_frames = args.n_frames
    if n_frames % _TEMPORAL_PATCH_SIZE != 0:
        n_frames += _TEMPORAL_PATCH_SIZE - (n_frames % _TEMPORAL_PATCH_SIZE)

    cached_inputs, yes_token_id, no_token_id = build_qwen_rubric_inputs(
        processor=processor,
        edit_prompt=args.edit_prompt,
        num_frames=n_frames,
        img_size=QWEN_IMG_SIZE,
        device=device,
    )

    # Load and fix frames — these NEVER change across repeats
    print(f"Loading video frames from {args.video}...")
    frames_fixed = load_video_frames(args.video, n_frames=n_frames).to(device)
    print(f"  frames shape: {frames_fixed.shape}  dtype: {frames_fixed.dtype}")
    print(f"  pixel min={frames_fixed.min():.4f}  max={frames_fixed.max():.4f}")

    # Collect N measurements
    yes_probs = []
    grad_norms = []
    cosine_sims = []  # cosine similarity to run 0's gradient
    ref_grad = None

    print(f"\nRunning {args.n_repeats} identical passes...")
    for i in range(args.n_repeats):
        if args.mode == "seeded":
            # Re-seed before every call to isolate any RNG dependency
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)

        yes_prob, grad_flat = run_single_pass(
            frames_fixed, qwen_model, cached_inputs, yes_token_id, no_token_id
        )

        grad_norm = float(grad_flat.norm().item())
        if ref_grad is None:
            ref_grad = grad_flat
            cos = 1.0
        else:
            cos = cosine_sim(ref_grad, grad_flat)

        yes_probs.append(yes_prob)
        grad_norms.append(grad_norm)
        cosine_sims.append(cos)

        print(f"  run {i:02d}:  yes_prob={yes_prob:.6f}  grad_norm={grad_norm:.6f}  cos_sim_to_run0={cos:.6f}")

    # Summary statistics
    import statistics
    print(f"\n--- Summary ({args.mode}) ---")
    print(f"yes_prob :  mean={statistics.mean(yes_probs):.6f}  stdev={statistics.stdev(yes_probs):.2e}  "
          f"min={min(yes_probs):.6f}  max={max(yes_probs):.6f}  range={max(yes_probs)-min(yes_probs):.2e}")
    print(f"grad_norm:  mean={statistics.mean(grad_norms):.6f}  stdev={statistics.stdev(grad_norms):.2e}")
    print(f"cos_sim  :  mean={statistics.mean(cosine_sims[1:]):.6f}  "
          f"min={min(cosine_sims[1:]):.6f}  max={max(cosine_sims[1:]):.6f}")
    print(f"  → cos_sim < 0.9999 means gradient DIRECTION differs between runs")
    print(f"  → cos_sim < 0 means gradient is OPPOSITE to run 0 (!)")


def main():
    p = argparse.ArgumentParser(description="Probe Qwen2.5-VL for forward/backward non-determinism")
    p.add_argument("--qwen-model", required=True, help="Path to Qwen2.5-VL model directory")
    p.add_argument("--video", required=True, help="Source video to extract fixed frames from")
    p.add_argument("--edit-prompt", default="A red car door opens.", help="Edit prompt for the yes/no question")
    p.add_argument("--n-frames", type=int, default=8, help="Number of frames to sample (must be even)")
    p.add_argument("--n-repeats", type=int, default=20, help="Number of identical passes to run")
    p.add_argument(
        "--mode",
        choices=["baseline", "deterministic", "seeded"],
        default="baseline",
        help=(
            "baseline: torch defaults (cudnn.deterministic=False). "
            "deterministic: cudnn.deterministic=True. "
            "seeded: deterministic + re-seed before every call."
        ),
    )
    args = p.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
