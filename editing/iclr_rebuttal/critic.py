"""No-grad replica of the optimisation loop's Qwen2.5-VL critic call.

Reproduces multimodal_loop_qwen.gradient_optimize_multimodal_qwen's scoring:
  * rubric restricted to `qwen_gradient_rubric` (paper: "motion" question only)
  * pass 0 uses the run's `qwen_sample_mode`; passes 1..P-1 use
    `contiguous_random` (exactly as the loop does when qwen_grad_accum_steps > 1)
  * score = exp(-mean_pass_nll)  (== the loop's logged `qwen_yes_prob`)

The random windows of passes >= 1 are drawn from torch's global RNG, which we
re-seed to a fixed value before every video so all candidates see the same
windows (common random numbers -> fair comparison across candidates).
"""
from __future__ import annotations

import math
from pathlib import Path

import torch

import common  # noqa: F401  (puts editing/ and editing/src on sys.path)

from motion_opt.core import decode_video_frames_rgb
from motion_opt.qwen_loss import build_qwen_model, build_qwen_rubric_inputs, compute_qwen_video_loss

CRN_SEED = 1234  # fixed window seed for passes >= 1


def load_video_chw(path: str | Path, device, max_frames: int | None = None) -> torch.Tensor:
    path = Path(path)
    if path.suffix.lower() == ".gif":
        import imageio.v3 as iio
        import numpy as np

        arr = iio.imread(str(path), index=None)  # [T,H,W,C]
        frames = [np.asarray(f)[..., :3] for f in arr]
    else:
        frames = decode_video_frames_rgb(str(path), max_frames=max_frames, frame_stride=1, resize_to=None)
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    t = torch.stack([torch.from_numpy(f.copy()).permute(2, 0, 1).float().div(255.0) for f in frames])
    return t.to(device)


class LoopCritic:
    def __init__(self, ns, device, qwen_model=None, processor=None):
        if qwen_model is None:
            qwen_model, processor = build_qwen_model(ns.qwen_model, device=device, gradient_checkpointing=False)
        self.model, self.processor, self.ns, self.device = qwen_model, processor, ns, device
        self.passes = max(1, int(getattr(ns, "qwen_grad_accum_steps", 1) or 1))
        nf = ns.qwen_max_frames + (ns.qwen_max_frames % 2)
        self.cached, self.yes_id, self.no_id = build_qwen_rubric_inputs(
            processor=processor,
            edit_prompt=ns.edit_prompt,
            num_frames=nf,
            img_size=ns.qwen_img_size,
            device=device,
            motion_question=ns.qwen_motion_question,
            gradient_rubric=ns.qwen_gradient_rubric,
        )
        rub = ns.qwen_gradient_rubric
        self.overrides = None if rub == "full" else {"motion": 0.0, "entities": 0.0, "overall": 0.0, str(rub): 1.0}

    @torch.no_grad()
    def score_frames(self, frames: torch.Tensor) -> dict:
        torch.manual_seed(CRN_SEED)
        nlls = []
        for i in range(self.passes):
            mode = self.ns.qwen_sample_mode if i == 0 else "contiguous_random"
            loss = compute_qwen_video_loss(
                frames_chw=frames,
                qwen_model=self.model,
                cached_inputs=self.cached,
                yes_token_id=self.yes_id,
                no_token_id=self.no_id,
                max_frames=self.ns.qwen_max_frames,
                img_size=self.ns.qwen_img_size,
                backward=False,
                return_details=False,
                sample_mode=mode,
                contiguous_start_frame=self.ns.qwen_contiguous_start_frame,
                rubric_weight_overrides=self.overrides,
            )
            nlls.append(float(loss.detach().float().item()))
        mean_nll = sum(nlls) / len(nlls)
        return {
            "yes_prob": math.exp(-mean_nll),
            "mean_nll": mean_nll,
            "pass_nll": nlls,
            "pass_yes": [math.exp(-x) for x in nlls],
            "n_frames": int(frames.shape[0]),
        }

    def score_video(self, path) -> dict:
        frames = load_video_chw(path, self.device)
        out = self.score_frames(frames)
        del frames
        return out
