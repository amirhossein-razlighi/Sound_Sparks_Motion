"""Perceptual quality preservation losses for VLM-guided video editing.

Addresses the core failure mode of VLM-guided optimization: the optimizer
finds adversarial perturbations that increase P("yes") without producing
genuine visual edits.  These losses anchor the optimization to the manifold
of natural videos.

Three complementary losses:
  1. **Source LPIPS** — penalizes perceptual deviation from the source video,
     preventing quality degradation and artifact introduction.
  2. **Temporal consistency** — penalizes frame-to-frame perceptual jumps,
     reducing flickering and temporal incoherence.
  3. **Latent magnitude penalty** — soft constraint on how far audio/text
     parameters drift from initialization (adaptive version of L2 reg).

Memory budget:
  - LPIPS-AlexNet: ~30 MB weights, ~200 MB activations for 33 frames
  - Total overhead: < 500 MB
"""
from __future__ import annotations

import logging
from typing import Literal

import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LPIPS model loading (lazy, singleton)
# ---------------------------------------------------------------------------

_lpips_model_cache: dict[str, torch.nn.Module] = {}


def _get_lpips_model(
    backbone: Literal["alex", "vgg"] = "alex",
    device: torch.device | str = "cuda",
) -> torch.nn.Module:
    """Return a frozen LPIPS model, cached per backbone."""
    key = f"{backbone}_{device}"
    if key not in _lpips_model_cache:
        import lpips
        model = lpips.LPIPS(net=backbone, verbose=False).to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        _lpips_model_cache[key] = model
        log.info("LPIPS-%s loaded on %s (~%.0f MB)", backbone, device,
                 sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6)
    return _lpips_model_cache[key]


# ---------------------------------------------------------------------------
# 1. Source LPIPS preservation loss
# ---------------------------------------------------------------------------

def compute_source_lpips_loss(
    gen_frames: torch.Tensor,
    src_frames: torch.Tensor,
    backbone: Literal["alex", "vgg"] = "alex",
    max_frames: int = 16,
    sample_mode: Literal["linspace", "stride"] = "linspace",
) -> torch.Tensor:
    """Perceptual distance between generated and source video frames.

    Measures how far the optimized video has drifted from the source in
    perceptual feature space.  High values indicate artifact introduction
    or identity/background corruption.

    Args:
        gen_frames: [N, 3, H, W] float [0, 1] — generated video frames.
        src_frames: [M, 3, H, W] float [0, 1] — source video frames.
            If M != N, we sample matching frames from src_frames.
        backbone: LPIPS backbone. "alex" is ~30 MB; "vgg" is ~60 MB.
        max_frames: Subsample to this many frames for memory efficiency.
        sample_mode: How to pick the subset of frames.

    Returns:
        Scalar mean LPIPS distance (lower = more preserved).
    """
    device = gen_frames.device
    lpips_model = _get_lpips_model(backbone, device)

    n_gen = gen_frames.shape[0]
    n_src = src_frames.shape[0]

    # Subsample frames
    n_eval = min(max_frames, n_gen, n_src)
    if sample_mode == "linspace":
        gen_idx = torch.linspace(0, n_gen - 1, n_eval, device=device).round().long()
        src_idx = torch.linspace(0, n_src - 1, n_eval, device=device).round().long()
    else:
        stride = max(n_gen // n_eval, 1)
        gen_idx = torch.arange(0, n_gen, stride, device=device)[:n_eval]
        src_idx = torch.arange(0, n_src, stride, device=device)[:n_eval]

    gen_sub = gen_frames[gen_idx]
    src_sub = src_frames[src_idx].to(device)

    # LPIPS expects [-1, 1]
    gen_norm = gen_sub * 2.0 - 1.0
    src_norm = src_sub * 2.0 - 1.0

    # Resize to 256x256 for LPIPS (standard evaluation size)
    if gen_norm.shape[-1] != 256 or gen_norm.shape[-2] != 256:
        gen_norm = F.interpolate(gen_norm, size=(256, 256), mode="bilinear", align_corners=False)
        src_norm = F.interpolate(src_norm, size=(256, 256), mode="bilinear", align_corners=False)

    # Compute per-frame LPIPS and mean
    # Process in small batches to limit memory
    batch_size = 8
    losses = []
    for i in range(0, n_eval, batch_size):
        j = min(i + batch_size, n_eval)
        loss_batch = lpips_model(gen_norm[i:j], src_norm[i:j])
        losses.append(loss_batch.mean())

    return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# 2. Temporal consistency loss
# ---------------------------------------------------------------------------

def compute_temporal_consistency_loss(
    gen_frames: torch.Tensor,
    src_frames: torch.Tensor | None = None,
    backbone: Literal["alex", "vgg"] = "alex",
    max_pairs: int = 12,
) -> torch.Tensor:
    """Penalize frame-to-frame perceptual jumps (temporal flickering).

    Computes the mean LPIPS distance between consecutive generated frames.
    If src_frames is provided, we compute the *excess* temporal variation:
    the generated video's frame-to-frame distance minus the source video's,
    so natural motion isn't penalized.

    Args:
        gen_frames: [N, 3, H, W] float [0, 1].
        src_frames: [M, 3, H, W] float [0, 1] or None.
        backbone: LPIPS backbone.
        max_pairs: Subsample to this many consecutive pairs.

    Returns:
        Scalar temporal consistency loss (lower = smoother).
    """
    device = gen_frames.device
    lpips_model = _get_lpips_model(backbone, device)

    n = gen_frames.shape[0]
    if n < 2:
        return torch.tensor(0.0, device=device)

    # Sample consecutive pairs
    n_pairs = min(max_pairs, n - 1)
    pair_idx = torch.linspace(0, n - 2, n_pairs, device=device).round().long()

    def _consecutive_lpips(frames: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        f1 = frames[indices]
        f2 = frames[indices + 1]
        f1_norm = F.interpolate(f1 * 2.0 - 1.0, size=(256, 256), mode="bilinear", align_corners=False)
        f2_norm = F.interpolate(f2 * 2.0 - 1.0, size=(256, 256), mode="bilinear", align_corners=False)

        batch_size = 8
        losses = []
        for i in range(0, len(indices), batch_size):
            j = min(i + batch_size, len(indices))
            loss_batch = lpips_model(f1_norm[i:j], f2_norm[i:j])
            losses.append(loss_batch.mean())
        return torch.stack(losses).mean()

    gen_temporal = _consecutive_lpips(gen_frames, pair_idx)

    if src_frames is not None and src_frames.shape[0] >= 2:
        n_src = src_frames.shape[0]
        src_pair_idx = torch.linspace(0, n_src - 2, n_pairs, device=device).round().long()
        with torch.no_grad():
            src_temporal = _consecutive_lpips(src_frames.to(device), src_pair_idx)
        # Penalize excess temporal variation (allow natural motion level)
        return F.relu(gen_temporal - src_temporal)

    return gen_temporal


# ---------------------------------------------------------------------------
# 3. Adaptive regularization schedule
# ---------------------------------------------------------------------------

def adaptive_reg_weight(
    base_weight: float,
    iteration: int,
    total_iterations: int,
    schedule: Literal["linear_warmup", "cosine_increase", "constant"] = "cosine_increase",
    min_multiplier: float = 0.1,
    max_multiplier: float = 3.0,
) -> float:
    """Compute iteration-dependent regularization weight.

    Key insight: early iterations should explore freely (low reg), while
    later iterations should be increasingly constrained to prevent
    adversarial drift (high reg).  This is the opposite of typical
    learning rate scheduling.

    Args:
        base_weight: The base regularization weight.
        iteration: Current iteration (1-indexed).
        total_iterations: Total number of iterations.
        schedule: Scheduling strategy.
        min_multiplier: Multiplier at iteration 1.
        max_multiplier: Multiplier at final iteration.

    Returns:
        Adjusted weight = base_weight * multiplier.
    """
    if schedule == "constant" or total_iterations <= 1:
        return base_weight

    progress = (iteration - 1) / max(total_iterations - 1, 1)  # 0 → 1

    if schedule == "linear_warmup":
        multiplier = min_multiplier + (max_multiplier - min_multiplier) * progress
    elif schedule == "cosine_increase":
        import math
        # Cosine curve from min to max (smooth acceleration)
        multiplier = min_multiplier + (max_multiplier - min_multiplier) * (
            1.0 - math.cos(math.pi * progress)
        ) / 2.0
    else:
        multiplier = 1.0

    return base_weight * multiplier


# ---------------------------------------------------------------------------
# Combined perceptual quality loss (convenience wrapper)
# ---------------------------------------------------------------------------

def compute_perceptual_quality_loss(
    gen_frames: torch.Tensor,
    src_frames: torch.Tensor,
    *,
    lpips_weight: float = 1.0,
    temporal_weight: float = 0.5,
    backbone: Literal["alex", "vgg"] = "alex",
    max_lpips_frames: int = 16,
    max_temporal_pairs: int = 12,
    backward: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combined perceptual quality loss with detailed breakdown.

    Args:
        gen_frames: [N, 3, H, W] float [0, 1].
        src_frames: [M, 3, H, W] float [0, 1].
        lpips_weight: Weight for source LPIPS preservation.
        temporal_weight: Weight for temporal consistency.
        backbone: LPIPS backbone.
        max_lpips_frames: Max frames for LPIPS computation.
        max_temporal_pairs: Max frame pairs for temporal loss.
        backward: If True, backpropagate inside this function.

    Returns:
        (total_loss, details_dict) where details_dict has individual losses.
    """
    details = {}

    total = torch.tensor(0.0, device=gen_frames.device)

    if lpips_weight > 0:
        lpips_loss = compute_source_lpips_loss(
            gen_frames, src_frames,
            backbone=backbone,
            max_frames=max_lpips_frames,
        )
        weighted_lpips = lpips_weight * lpips_loss
        details["lpips_raw"] = float(lpips_loss.detach().item())
        details["lpips_weighted"] = float(weighted_lpips.detach().item())
        total = total + weighted_lpips

    if temporal_weight > 0:
        temporal_loss = compute_temporal_consistency_loss(
            gen_frames, src_frames,
            backbone=backbone,
            max_pairs=max_temporal_pairs,
        )
        weighted_temporal = temporal_weight * temporal_loss
        details["temporal_raw"] = float(temporal_loss.detach().item())
        details["temporal_weighted"] = float(weighted_temporal.detach().item())
        total = total + weighted_temporal

    details["perceptual_total"] = float(total.detach().item())

    if backward and total.requires_grad:
        total.backward()
        total = total.detach()

    return total, details


# ---------------------------------------------------------------------------
# Source frame caching helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def cache_source_frames(
    pipeline,
    src_video: str,
    cached_video_latent: torch.Tensor,
    base_audio_latent: torch.Tensor,
    base_pos_context,
    base_neg_context,
    retake_kwargs: dict,
    max_frames: int,
    frame_stride: int,
    eval_sample_start: int,
) -> torch.Tensor:
    """Render source video frames once and cache them for LPIPS comparison.

    Runs the retake pipeline with unoptimized parameters to get the
    "clean" baseline frames that we'll compare against during optimization.

    Returns:
        [N, 3, H, W] float [0, 1] tensor on the same device as the pipeline.
    """
    from .multimodal_loop import render_with_injected_latents

    src_frames = render_with_injected_latents(
        pipeline=pipeline,
        src_video=src_video,
        injected_audio_latent=base_audio_latent,
        cached_video_latent=cached_video_latent,
        retake_kwargs=retake_kwargs,
        max_frames=max_frames,
        frame_stride=frame_stride,
        resize_to=None,
        audio_opt_last_steps=0,  # no grad needed
        eval_sample_start=eval_sample_start,
        pos_context=base_pos_context,
        neg_context=base_neg_context,
        inject_text_context=True,
    )
    return src_frames.detach()
