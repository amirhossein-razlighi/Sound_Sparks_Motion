from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PathsConfig:
    src_video: str
    output_dir: Path
    checkpoint_path: str
    gemma_root: str


@dataclass(frozen=True)
class ObjectiveConfig:
    flow_weight: float
    mag_curve_weight: float
    latent_reg_weight: float
    flow_width: int
    flow_height: int
    max_eval_frames: int
    frame_stride: int


@dataclass(frozen=True)
class OptimizerConfig:
    iterations: int
    lr: float
    grad_clip: float


@dataclass(frozen=True)
class RuntimeConfig:
    seed: int
    retake_start_frames: int
    retake_num_inference_steps: int
    final_retake_num_inference_steps: int | None
    save_final_videos: bool

