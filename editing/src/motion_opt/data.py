from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PreparedLatents:
    cached_video_latent: torch.Tensor
    base_audio_latent: torch.Tensor
    base_audio_latent_fp32: torch.Tensor
    waveform_sr: int


@dataclass
class FlowTargets:
    target_flows: torch.Tensor
    raft_model: torch.nn.Module
    raft_transforms: object
