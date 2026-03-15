#!/usr/bin/env python3
"""
Optimize audio latent for targeted video motion edits.

This script performs gradient-based optimization over the audio latent that
conditions LTX retake generation, while freezing all LTX model weights.

High-level flow
---------------
1) Build a target motion signal from a target video (provided directly or
   generated once via TI2V from first frame + target prompt).
2) Encode source video/audio once and cache source video latent.
3) Optimize a low-dimensional parameter vector that perturbs audio latent.
4) For each optimization step, run Retake and score generated video with a
    differentiable RAFT optical-flow objective against target motion.
5) Save the best latent + rendered videos.

Notes
-----
- Optimization uses true backpropagation through the generated video frames,
    the frozen RAFT flow network, and into latent coefficients.
- Only the initial audio latent is changed. All LTX modules stay frozen.
- Objective uses frozen RAFT optical flow from torchvision.

Example
-------
python editing/optimize_audio_embedding.py \
  --src-video /path/to/source.mp4 \
  --edit-prompt "A dog is in the scene." \
  --target-prompt "The dog jumps energetically." \
  --output-dir ./audio_latent_opt \
    --iterations 10 \
    --lr 0.05
"""
from __future__ import annotations

import argparse
import csv
import gc
import logging
import os
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Iterator

import av
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchaudio

# ---------------------------------------------------------------------------
# Make sure editing/ is on sys.path (for running without install)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# LTX-2 imports
from ltx_core.components.guiders import MultiModalGuiderParams
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.quantization import QuantizationPolicy
from ltx_core.types import Audio, VideoPixelShape
import ltx_pipelines.retake as _retake_module
import ltx_pipelines.ti2vid_one_stage as _ti2vid_module
import ltx_pipelines.utils.samplers as _samplers_module
from ltx_pipelines.retake import RetakePipeline
from ltx_pipelines.ti2vid_one_stage import TI2VidOneStagePipeline
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.constants import detect_params
from ltx_pipelines.utils.media_io import (
    _prepare_audio_stream,
    _write_audio,
    decode_audio_from_file,
    encode_video,
    get_videostream_metadata,
    load_video_conditioning,
)

log = logging.getLogger(__name__)


def _is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def _barrier() -> None:
    if _is_distributed():
        dist.barrier()


def _rank0_print(msg: str) -> None:
    if not _is_distributed() or dist.get_rank() == 0:
        print(msg)


def init_distributed_and_device() -> tuple[int, int, torch.device]:
    """Initialize distributed process group from env vars when present.

    Supports torchrun/srun launches where RANK/WORLD_SIZE/LOCAL_RANK are set.
    """
    use_dist = (
        dist.is_available()
        and "RANK" in os.environ
        and "WORLD_SIZE" in os.environ
        and int(os.environ.get("WORLD_SIZE", "1")) > 1
    )

    rank = 0
    world_size = 1
    local_rank = 0
    if use_dist:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))

        visible_gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if visible_gpu_count <= 0:
            raise RuntimeError(
                "Distributed launch requested but no CUDA devices are visible. "
                f"RANK={rank} WORLD_SIZE={world_size} LOCAL_RANK={local_rank}."
            )
        if local_rank < 0 or local_rank >= visible_gpu_count:
            raise RuntimeError(
                "Invalid LOCAL_RANK to visible GPU mapping. "
                f"RANK={rank} WORLD_SIZE={world_size} LOCAL_RANK={local_rank} visible_gpus={visible_gpu_count} "
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}. "
                "Set torchrun --nproc_per_node to the number of visible GPUs in the allocation."
            )

        # Set device before NCCL init so collectives use the correct mapping.
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", device_id=local_rank)

    if torch.cuda.is_available():
        if use_dist:
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    return rank, world_size, device

# ---------------------------------------------------------------------------
# Defaults (cluster-friendly, can be overridden via CLI)
# ---------------------------------------------------------------------------
_CKPT_ROOT = "/project/def-amahdavi/amirrz/LTX-2/checkpoints"
DEFAULT_CHECKPOINT = f"{_CKPT_ROOT}/ltx-2.3-22b-dev.safetensors"
DEFAULT_GEMMA_ROOT = "/project/def-amahdavi/amirrz/HF/models/gemma-3-12b-it-qat-q4_0-unquantized/"


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def align_waveform_length(waveform: torch.Tensor, n_target: int) -> torch.Tensor:
    n = waveform.shape[-1]
    if n >= n_target:
        return waveform[..., :n_target]
    pad = torch.zeros(*waveform.shape[:-1], n_target - n, device=waveform.device, dtype=waveform.dtype)
    return torch.cat([waveform, pad], dim=-1)


def compute_target_shape(
    video_path: str,
    height_override: int | None,
    width_override: int | None,
    num_frames_override: int | None,
    frame_rate_override: float | None,
) -> tuple[int, int, int, float]:
    """Determine (height, width, num_frames, fps) from source video or overrides.

    Single-stage pipeline requires H/W multiples of 32 and num_frames = 8k + 1.
    """
    fps_src, n_frames_src, w_src, h_src = get_videostream_metadata(video_path)

    fps = frame_rate_override if frame_rate_override is not None else fps_src
    n_fr = num_frames_override if num_frames_override is not None else n_frames_src
    height = height_override if height_override is not None else h_src
    width = width_override if width_override is not None else w_src

    height = max(32, (height // 32) * 32)
    width = max(32, (width // 32) * 32)

    if (n_fr - 1) % 8 != 0:
        n_fr = ((n_fr - 1) // 8) * 8 + 1
    n_fr = max(9, n_fr)

    return height, width, n_fr, float(fps)


def extract_first_frame_png(video_path: str, out_path: str) -> None:
    src = av.open(video_path)
    try:
        vs = next(s for s in src.streams if s.type == "video")
        for frame in src.decode(vs):
            frame.to_image().save(out_path, format="PNG")
            return
    finally:
        src.close()
    raise RuntimeError(f"No video frame found in {video_path!r}")


def save_audio_wav(waveform: torch.Tensor, sr: int, path: str) -> None:
    torchaudio.save(path, waveform.detach().cpu().float(), sr, backend="soundfile")


def write_temp_video_with_audio(
    src_video_path: str,
    target_frames: int,
    target_height: int,
    target_width: int,
    fps: float,
    waveform: torch.Tensor,
    sr: int,
    output_path: str,
) -> None:
    """Write MP4 with resized source video frames and supplied audio waveform."""
    src = av.open(src_video_path)
    dst = av.open(output_path, mode="w")

    try:
        vs_in = next(s for s in src.streams if s.type == "video")
        vs_out = dst.add_stream("libx264", rate=int(round(fps)))
        vs_out.width = target_width
        vs_out.height = target_height
        vs_out.pix_fmt = "yuv420p"
        vs_out.options = {"crf": "18", "preset": "veryfast"}

        w = waveform.detach().cpu().float()
        if w.shape[0] == 1:
            w = w.expand(2, -1).contiguous()
        elif w.shape[0] > 2:
            w = w[:2].contiguous()
        as_out = _prepare_audio_stream(dst, sr)

        from fractions import Fraction as _Fraction

        time_base = _Fraction(1, int(round(fps)))
        frame_idx = 0
        last_frame = None

        for av_frame in src.decode(vs_in):
            if frame_idx >= target_frames:
                break
            out = av_frame.reformat(width=target_width, height=target_height, format="yuv420p")
            out.pts = frame_idx
            out.time_base = time_base
            for pkt in vs_out.encode(out):
                dst.mux(pkt)
            last_frame = out
            frame_idx += 1

        while last_frame is not None and frame_idx < target_frames:
            pad = av.VideoFrame(width=target_width, height=target_height, format="yuv420p")
            pad.pts = frame_idx
            pad.time_base = time_base
            for i in range(len(last_frame.planes)):
                # PyAV VideoPlane does not expose .shape in all versions.
                # Copy raw plane bytes directly using memoryview.
                src_plane = memoryview(last_frame.planes[i])
                dst_plane = memoryview(pad.planes[i])
                n = min(len(src_plane), len(dst_plane))
                dst_plane[:n] = src_plane[:n]
            for pkt in vs_out.encode(pad):
                dst.mux(pkt)
            frame_idx += 1

        for pkt in vs_out.encode():
            dst.mux(pkt)

        _write_audio(dst, as_out, Audio(waveform=w, sampling_rate=sr))
    finally:
        src.close()
        dst.close()


# ---------------------------------------------------------------------------
# Video / optical flow objective (RAFT)
# ---------------------------------------------------------------------------

def decode_video_frames_rgb(
    video_path: str,
    max_frames: int | None,
    frame_stride: int,
    resize_to: tuple[int, int] | None,
    sample_start: int = 0,
) -> list[np.ndarray]:
    """Decode RGB frames from a video file as uint8 HxWx3 arrays."""
    out: list[np.ndarray] = []
    src = av.open(video_path)
    try:
        vs = next(s for s in src.streams if s.type == "video")
        frame_idx = 0
        sampled_idx = 0
        for frame in src.decode(vs):
            if frame_idx % frame_stride != 0:
                frame_idx += 1
                continue
            if sampled_idx < sample_start:
                sampled_idx += 1
                frame_idx += 1
                continue
            arr = np.asarray(frame.to_image().convert("RGB"), dtype=np.uint8)
            out.append(arr)
            sampled_idx += 1
            frame_idx += 1
            if max_frames is not None and len(out) >= max_frames:
                break
    finally:
        src.close()
    return out


def decode_video_mask_frames(
    video_path: str,
    max_frames: int | None,
    frame_stride: int,
    resize_to: tuple[int, int] | None,
    threshold: float,
    sample_start: int = 0,
) -> list[np.ndarray]:
    """Decode a binary mask video as float HxW arrays in {0,1}."""
    out: list[np.ndarray] = []
    src = av.open(video_path)
    try:
        vs = next(s for s in src.streams if s.type == "video")
        frame_idx = 0
        sampled_idx = 0
        for frame in src.decode(vs):
            if frame_idx % frame_stride != 0:
                frame_idx += 1
                continue
            if sampled_idx < sample_start:
                sampled_idx += 1
                frame_idx += 1
                continue

            arr = np.asarray(frame.to_image().convert("L"), dtype=np.float32) / 255.0
            if resize_to is not None and (arr.shape[1], arr.shape[0]) != resize_to:
                arr_t = torch.from_numpy(arr)[None, None]
                arr = (
                    F.interpolate(arr_t, size=(resize_to[1], resize_to[0]), mode="nearest")[0, 0].numpy()
                )
            out.append((arr >= threshold).astype(np.float32, copy=False))
            sampled_idx += 1
            frame_idx += 1
            if max_frames is not None and len(out) >= max_frames:
                break
    finally:
        src.close()
    return out


def flatten_video_chunks(
    video_iter: Iterator[torch.Tensor],
    max_frames: int | None,
    frame_stride: int,
    resize_to: tuple[int, int] | None,
    sample_start: int = 0,
) -> torch.Tensor:
    """Collect generated frames into a float tensor [F, 3, H, W] in [0, 1].

    This function keeps gradients if input chunks are floating tensors.
    """
    frames: list[torch.Tensor] = []
    seen = 0
    sampled_seen = 0
    for chunk in video_iter:
        # Expected by default decode patch: [F, 3, H, W] float in [0,1].
        # Fallback for uint8 [F, H, W, 3] if no patch was applied.
        if chunk.dim() != 4:
            raise RuntimeError(f"Unexpected decoded chunk shape: {tuple(chunk.shape)}")

        if chunk.shape[1] == 3:
            frame_tensor = chunk
        elif chunk.shape[-1] == 3:
            frame_tensor = chunk.permute(0, 3, 1, 2)
        else:
            raise RuntimeError(f"Cannot infer channel axis from chunk shape: {tuple(chunk.shape)}")

        if frame_tensor.dtype == torch.uint8:
            frame_tensor = frame_tensor.float() / 255.0
        else:
            frame_tensor = frame_tensor.float().clamp(0.0, 1.0)

        for frame in frame_tensor:
            if seen % frame_stride != 0:
                seen += 1
                continue
            if sampled_seen < sample_start:
                sampled_seen += 1
                seen += 1
                continue
            frames.append(frame)
            sampled_seen += 1
            seen += 1
            if max_frames is not None and len(frames) >= max_frames:
                stacked = torch.stack(frames, dim=0)
                if resize_to is not None:
                    stacked = F.interpolate(
                        stacked,
                        size=(resize_to[1], resize_to[0]),
                        mode="bilinear",
                        align_corners=False,
                    )
                return stacked

    if len(frames) == 0:
        return torch.empty(0, 3, 0, 0)

    stacked = torch.stack(frames, dim=0)
    if resize_to is not None:
        stacked = F.interpolate(
            stacked,
            size=(resize_to[1], resize_to[0]),
            mode="bilinear",
            align_corners=False,
        )
    return stacked


def load_raft_components(
    device: torch.device,
    model_name: str,
    weights_path: str | None = None,
):
    """Load frozen RAFT model and input transforms from torchvision.

    If ``weights_path`` is provided, the checkpoint is loaded from disk and no
    network download is attempted.
    """
    try:
        from torchvision.models.optical_flow import (  # type: ignore
            Raft_Large_Weights,
            Raft_Small_Weights,
            raft_large,
            raft_small,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "torchvision optical flow (RAFT) is required. Install torchvision with optical-flow support."
        ) from exc

    if model_name == "raft_small":
        weights = Raft_Small_Weights.DEFAULT
        if weights_path is None:
            raft = raft_small(weights=weights, progress=True)
        else:
            raft = raft_small(weights=None, progress=False)
    else:
        weights = Raft_Large_Weights.DEFAULT
        if weights_path is None:
            raft = raft_large(weights=weights, progress=True)
        else:
            raft = raft_large(weights=None, progress=False)

    if weights_path is not None:
        ckpt_path = Path(weights_path).expanduser().resolve()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"RAFT checkpoint not found: {ckpt_path}")
        state = torch.load(str(ckpt_path), map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        raft.load_state_dict(state)
        log.info("Loaded RAFT weights from local file: %s", ckpt_path)

    raft = raft.to(device).eval()
    for p in raft.parameters():
        p.requires_grad_(False)

    return raft, weights.transforms()


def frames_rgb_uint8_to_chw_float(frames_rgb: list[np.ndarray], device: torch.device) -> torch.Tensor:
    if len(frames_rgb) == 0:
        return torch.empty(0, 3, 0, 0, device=device)
    arr = np.stack(frames_rgb, axis=0)
    t = torch.from_numpy(arr).to(device=device, dtype=torch.float32) / 255.0
    return t.permute(0, 3, 1, 2).contiguous()


def mask_frames_to_nchw_float(mask_frames: list[np.ndarray], device: torch.device) -> torch.Tensor:
    if len(mask_frames) == 0:
        return torch.empty(0, 1, 0, 0, device=device)
    arr = np.stack(mask_frames, axis=0)
    return torch.from_numpy(arr).to(device=device, dtype=torch.float32).unsqueeze(1).contiguous()


def compute_raft_flows(
    frames_chw: torch.Tensor,
    raft_model: torch.nn.Module,
    raft_transforms,
) -> torch.Tensor:
    """Compute consecutive-frame optical flow with RAFT.

    Input: frames_chw [F,3,H,W] in [0,1]
    Output: flow [F-1,2,H,W]
    """
    if frames_chw.shape[0] < 2:
        return torch.empty(0, 2, frames_chw.shape[-2], frames_chw.shape[-1], device=frames_chw.device)

    prev = frames_chw[:-1]
    nxt = frames_chw[1:]
    prev_in, nxt_in = raft_transforms(prev, nxt)
    preds = raft_model(prev_in, nxt_in)
    return preds[-1]


def flow_objective_torch(
    gen_frames_chw: torch.Tensor,
    target_flows: torch.Tensor,
    raft_model: torch.nn.Module,
    raft_transforms,
    flow_weight: float,
    mag_curve_weight: float,
    roi_frame_masks: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable objective: RAFT flow field + flow magnitude curve."""
    gen_flows = compute_raft_flows(gen_frames_chw, raft_model, raft_transforms)
    if gen_flows.shape[0] == 0 or target_flows.shape[0] == 0:
        inf = torch.tensor(float("inf"), device=gen_frames_chw.device)
        return inf, inf, inf

    n = min(gen_flows.shape[0], target_flows.shape[0])
    if roi_frame_masks is not None and roi_frame_masks.shape[0] >= 2:
        n = min(n, roi_frame_masks.shape[0] - 1)
    gen = gen_flows[:n]
    tgt = target_flows[:n]

    if gen.shape[-2:] != tgt.shape[-2:]:
        scale_x = tgt.shape[-1] / gen.shape[-1]
        scale_y = tgt.shape[-2] / gen.shape[-2]
        gen = F.interpolate(gen, size=tgt.shape[-2:], mode="bilinear", align_corners=False)
        # Out-of-place scaling to avoid version-counter invalidation on a grad-tracked tensor.
        scale = torch.tensor([scale_x, scale_y], device=gen.device, dtype=gen.dtype).view(1, 2, 1, 1)
        gen = gen * scale

    flow_masks = None
    if roi_frame_masks is not None:
        flow_masks = torch.maximum(roi_frame_masks[:n], roi_frame_masks[1 : n + 1])
        if flow_masks.shape[-2:] != tgt.shape[-2:]:
            flow_masks = F.interpolate(flow_masks, size=tgt.shape[-2:], mode="nearest")
        flow_masks = flow_masks.clamp(0.0, 1.0)

    if flow_masks is None:
        flow_mse = F.mse_loss(gen, tgt)

        gen_mag = torch.linalg.norm(gen, dim=1).mean(dim=(1, 2))
        tgt_mag = torch.linalg.norm(tgt, dim=1).mean(dim=(1, 2))
        mag_curve_mse = F.mse_loss(gen_mag, tgt_mag)
    else:
        if float(flow_masks.sum().item()) <= 0.0:
            inf = torch.tensor(float("inf"), device=gen_frames_chw.device)
            return inf, inf, inf
        weight_sum = flow_masks.sum().clamp_min(1.0)
        flow_mse = ((gen - tgt).pow(2) * flow_masks).sum() / (weight_sum * gen.shape[1])

        mask_2d = flow_masks.squeeze(1)
        mask_area = mask_2d.sum(dim=(1, 2)).clamp_min(1.0)
        gen_mag_map = torch.linalg.norm(gen, dim=1)
        tgt_mag_map = torch.linalg.norm(tgt, dim=1)
        gen_mag = (gen_mag_map * mask_2d).sum(dim=(1, 2)) / mask_area
        tgt_mag = (tgt_mag_map * mask_2d).sum(dim=(1, 2)) / mask_area
        mag_curve_mse = F.mse_loss(gen_mag, tgt_mag)

    total = flow_weight * flow_mse + mag_curve_weight * mag_curve_mse
    return total, flow_mse, mag_curve_mse


# ---------------------------------------------------------------------------
# LTX setup / rendering helpers
# ---------------------------------------------------------------------------

def _parse_loras(raw_loras: list[list[str]]) -> list[LoraPathStrengthAndSDOps]:
    out: list[LoraPathStrengthAndSDOps] = []
    for entry in raw_loras:
        lora_path = str(Path(entry[0]).expanduser().resolve())
        strength = float(entry[1]) if len(entry) > 1 else 1.0
        out.append(LoraPathStrengthAndSDOps(lora_path, strength, LTXV_LORA_COMFY_RENAMING_MAP))
    return out


def build_guiders_for_mode(
    args: argparse.Namespace,
    params,
    use_low_memory_guidance: bool,
) -> tuple[MultiModalGuiderParams, MultiModalGuiderParams]:
    if use_low_memory_guidance:
        video_guider_params = MultiModalGuiderParams(
            cfg_scale=args.cfg_scale if args.cfg_scale is not None else 1.0,
            stg_scale=0.0,
            stg_blocks=params.video_guider_params.stg_blocks,
            rescale_scale=0.0,
            modality_scale=args.a2v_scale if args.a2v_scale is not None else 1.0,
        )
        audio_guider_params = MultiModalGuiderParams(
            cfg_scale=args.audio_cfg_scale if args.audio_cfg_scale is not None else 1.0,
            stg_scale=0.0,
            stg_blocks=params.audio_guider_params.stg_blocks,
            rescale_scale=0.0,
        )
    else:
        video_guider_params = MultiModalGuiderParams(
            cfg_scale=args.cfg_scale if args.cfg_scale is not None else params.video_guider_params.cfg_scale,
            stg_scale=params.video_guider_params.stg_scale,
            stg_blocks=params.video_guider_params.stg_blocks,
            rescale_scale=params.video_guider_params.rescale_scale,
            modality_scale=args.a2v_scale if args.a2v_scale is not None else params.video_guider_params.modality_scale,
        )
        audio_guider_params = MultiModalGuiderParams(
            cfg_scale=args.audio_cfg_scale if args.audio_cfg_scale is not None else params.audio_guider_params.cfg_scale,
            stg_scale=params.audio_guider_params.stg_scale,
            stg_blocks=params.audio_guider_params.stg_blocks,
            rescale_scale=params.audio_guider_params.rescale_scale,
        )

    return video_guider_params, audio_guider_params


@torch.inference_mode()
def maybe_generate_target_video(
    args: argparse.Namespace,
    device: torch.device,
    height: int,
    width: int,
    num_frames: int,
    frame_rate: float,
    video_guider_params: MultiModalGuiderParams,
    audio_guider_params: MultiModalGuiderParams,
    quant_policy: QuantizationPolicy | None,
    loras: list[LoraPathStrengthAndSDOps],
    output_dir: Path,
) -> Path:
    if args.target_video is not None:
        path = Path(args.target_video).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"--target-video not found: {path}")
        return path

    if not args.target_prompt:
        raise ValueError("Provide either --target-video or --target-prompt.")

    target_path = output_dir / "target_motion_video.mp4"
    if target_path.exists() and not args.regenerate_target:
        log.info("Using existing generated target video: %s", target_path)
        return target_path

    with tempfile.TemporaryDirectory(prefix="ltx_target_gt_") as tmp_dir:
        first_frame_png = os.path.join(tmp_dir, "first_frame.png")
        extract_first_frame_png(args.src_video, first_frame_png)

        images = [ImageConditioningInput(path=first_frame_png, frame_idx=0, strength=1.0)]

        pipeline = TI2VidOneStagePipeline(
            checkpoint_path=args.checkpoint_path,
            gemma_root=args.gemma_root,
            loras=tuple(loras),
            device=device,
            quantization=quant_policy,
        )

        orig_decode = _ti2vid_module.vae_decode_video

        def _tiled_decode(latent, decoder, tiling_config=None, generator=None):
            return orig_decode(latent, decoder, TilingConfig.default(), generator)

        _ti2vid_module.vae_decode_video = _tiled_decode
        try:
            video_iter, audio = pipeline(
                prompt=args.target_prompt,
                negative_prompt=args.negative_prompt,
                seed=args.seed,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=frame_rate,
                num_inference_steps=args.ti2v_num_inference_steps,
                video_guider_params=video_guider_params,
                audio_guider_params=audio_guider_params,
                images=images,
                enhance_prompt=args.enhance_prompt,
            )
        finally:
            _ti2vid_module.vae_decode_video = orig_decode

        n_chunks = get_video_chunks_number(num_frames, TilingConfig.default())
        encode_video(
            video=video_iter,
            fps=int(round(frame_rate)),
            audio=audio,
            output_path=str(target_path),
            video_chunks_number=n_chunks,
        )

        del pipeline, video_iter, audio
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    log.info("Generated target-motion video -> %s", target_path)
    return target_path


def build_cached_source_latents(
    pipeline: RetakePipeline,
    src_video: str,
    height: int,
    width: int,
    num_frames: int,
    audio_sr: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Encode source video/audio once and return (video_latent, audio_latent, sr)."""
    tiling = TilingConfig.default()

    # Video latent (cached to avoid repeated VAE encode)
    video_encoder = pipeline.model_ledger.video_encoder()
    pixel_video = load_video_conditioning(
        video_path=src_video,
        height=height,
        width=width,
        frame_cap=num_frames,
        dtype=torch.bfloat16,
        device=device,
    )
    with torch.inference_mode():
        video_latent = video_encoder.tiled_encode(pixel_video, tiling).to(device)

    del video_encoder, pixel_video
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Audio latent (base point for optimization)
    src_fps, _, _, _ = get_videostream_metadata(src_video)
    duration = float(num_frames) / float(src_fps)
    audio_in = decode_audio_from_file(src_video, device, max_duration=duration)
    if audio_in is None:
        raise RuntimeError(f"Source video has no audio stream: {src_video}")

    waveform = audio_in.waveform.squeeze(0).float()
    if audio_in.sampling_rate != audio_sr:
        waveform = torchaudio.functional.resample(waveform, orig_freq=audio_in.sampling_rate, new_freq=audio_sr)
        waveform_sr = audio_sr
    else:
        waveform_sr = audio_in.sampling_rate

    n_audio_samples = int(duration * waveform_sr)
    waveform = align_waveform_length(waveform, n_audio_samples)

    output_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=src_fps)

    audio_encoder = pipeline.model_ledger.audio_encoder()
    with torch.inference_mode():
        base_audio_latent = _retake_module._encode_audio_for_retake(
            audio_encoder=audio_encoder,
            waveform=waveform,
            waveform_sr=waveform_sr,
            output_shape=output_shape,
            dtype=torch.bfloat16,
        ).to(device)
    # Keep this as a graph-free anchor across optimization iterations.
    base_audio_latent = base_audio_latent.detach()

    del audio_encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return video_latent, base_audio_latent, waveform_sr


def render_with_injected_audio_latent(
    pipeline: RetakePipeline,
    src_video: str,
    injected_audio_latent: torch.Tensor,
    cached_video_latent: torch.Tensor,
    retake_kwargs: dict,
    max_frames: int,
    frame_stride: int,
    resize_to: tuple[int, int],
    audio_opt_last_steps: int,
    eval_sample_start: int,
) -> torch.Tensor:
    """Run Retake with forced audio latent and return frames [F,3,H,W] in [0,1].

    This path patches video decode to keep float outputs so gradients can flow.
    """
    orig_video_encode = _retake_module._encode_video_for_retake
    orig_audio_encode = _retake_module._encode_audio_for_retake
    orig_video_decode = _retake_module.vae_decode_video
    orig_video_encoder_getter = pipeline.model_ledger.video_encoder
    orig_audio_encoder_getter = pipeline.model_ledger.audio_encoder
    orig_audio_decode = _retake_module.vae_decode_audio
    orig_euler_loop = _retake_module.euler_denoising_loop

    def _late_step_grad_euler_loop(sigmas, video_state, audio_state, stepper, denoise_fn):
        total_steps = max(int(sigmas.shape[0]) - 1, 0)
        if audio_opt_last_steps <= 0 or audio_opt_last_steps >= total_steps:
            return orig_euler_loop(sigmas, video_state, audio_state, stepper, denoise_fn)

        grad_start_step = total_steps - audio_opt_last_steps
        for step_idx in range(total_steps):
            if step_idx < grad_start_step:
                # Truncate graph in early denoising and keep gradients only for late steps.
                with torch.no_grad():
                    denoised_video, denoised_audio = denoise_fn(video_state, audio_state, sigmas, step_idx)
                    denoised_video = _samplers_module.post_process_latent(
                        denoised_video,
                        video_state.denoise_mask,
                        video_state.clean_latent,
                    )
                    denoised_audio = _samplers_module.post_process_latent(
                        denoised_audio,
                        audio_state.denoise_mask,
                        audio_state.clean_latent,
                    )
                    next_video = stepper.step(video_state.latent, denoised_video, sigmas, step_idx).detach()
                    next_audio = stepper.step(audio_state.latent, denoised_audio, sigmas, step_idx).detach()
            else:
                denoised_video, denoised_audio = denoise_fn(video_state, audio_state, sigmas, step_idx)
                denoised_video = _samplers_module.post_process_latent(
                    denoised_video,
                    video_state.denoise_mask,
                    video_state.clean_latent,
                )
                denoised_audio = _samplers_module.post_process_latent(
                    denoised_audio,
                    audio_state.denoise_mask,
                    audio_state.clean_latent,
                )
                next_video = stepper.step(video_state.latent, denoised_video, sigmas, step_idx)
                next_audio = stepper.step(audio_state.latent, denoised_audio, sigmas, step_idx)

            video_state = replace(video_state, latent=next_video)
            audio_state = replace(audio_state, latent=next_audio)

        return (video_state, audio_state)

    def _cached_video_encode(video_encoder, video_path, output_shape, dtype, device):  # noqa: ARG001
        return cached_video_latent

    def _forced_audio_encode(audio_encoder, waveform, waveform_sr, output_shape, dtype):  # noqa: ARG001
        return injected_audio_latent

    def _float_decode_video(latent, video_decoder, tiling_config=None, generator=None):
        def _to_frames(frames_bcfhw: torch.Tensor) -> torch.Tensor:
            # Keep float range [0,1] and preserve autograd graph.
            return ((frames_bcfhw[0] + 1.0) / 2.0).clamp(0.0, 1.0).permute(1, 0, 2, 3).contiguous()

        if tiling_config is not None:
            for frames in video_decoder.tiled_decode(latent, tiling_config, generator=generator):
                yield _to_frames(frames)
        else:
            yield _to_frames(video_decoder(latent, generator=generator))

    _retake_module._encode_video_for_retake = _cached_video_encode
    _retake_module._encode_audio_for_retake = _forced_audio_encode
    _retake_module.vae_decode_video = _float_decode_video
    pipeline.model_ledger.video_encoder = lambda: None
    pipeline.model_ledger.audio_encoder = lambda: None
    _retake_module.euler_denoising_loop = _late_step_grad_euler_loop
    # Detach audio latent before decode: severs the shared transformer
    # checkpoint nodes from the audio (discarded) path, so that when the
    # decoded_audio tensor goes out of scope its graph doesn't free nodes
    # that the video gradient path still needs.
    def _detached_audio_decode(latent, audio_decoder, vocoder):
        with torch.no_grad():
            return orig_audio_decode(latent.detach(), audio_decoder, vocoder)
    _retake_module.vae_decode_audio = _detached_audio_decode
    try:
        video_iter, _ = pipeline(video_path=src_video, **retake_kwargs)
        return flatten_video_chunks(
            video_iter=video_iter,
            max_frames=max_frames,
            frame_stride=frame_stride,
            resize_to=resize_to,
            sample_start=eval_sample_start,
        )
    finally:
        _retake_module._encode_video_for_retake = orig_video_encode
        _retake_module._encode_audio_for_retake = orig_audio_encode
        _retake_module.vae_decode_video = orig_video_decode
        pipeline.model_ledger.video_encoder = orig_video_encoder_getter
        pipeline.model_ledger.audio_encoder = orig_audio_encoder_getter
        _retake_module.vae_decode_audio = orig_audio_decode
        _retake_module.euler_denoising_loop = orig_euler_loop


def render_and_save_video_with_latent(
    pipeline: RetakePipeline,
    src_video: str,
    injected_audio_latent: torch.Tensor,
    cached_video_latent: torch.Tensor,
    retake_kwargs: dict,
    output_path: Path,
    fps: float,
    num_frames: int,
    audio_sr: int,
    audio_opt_last_steps: int,
) -> None:
    """Render a full video with injected latent and save as MP4.

    The original source audio is muxed back in to simplify visual comparison.
    """
    orig_video_encode = _retake_module._encode_video_for_retake
    orig_audio_encode = _retake_module._encode_audio_for_retake
    orig_euler_loop = _retake_module.euler_denoising_loop

    def _late_step_grad_euler_loop(sigmas, video_state, audio_state, stepper, denoise_fn):
        total_steps = max(int(sigmas.shape[0]) - 1, 0)
        if audio_opt_last_steps <= 0 or audio_opt_last_steps >= total_steps:
            return orig_euler_loop(sigmas, video_state, audio_state, stepper, denoise_fn)

        grad_start_step = total_steps - audio_opt_last_steps
        for step_idx in range(total_steps):
            if step_idx < grad_start_step:
                with torch.no_grad():
                    denoised_video, denoised_audio = denoise_fn(video_state, audio_state, sigmas, step_idx)
                    denoised_video = _samplers_module.post_process_latent(
                        denoised_video,
                        video_state.denoise_mask,
                        video_state.clean_latent,
                    )
                    denoised_audio = _samplers_module.post_process_latent(
                        denoised_audio,
                        audio_state.denoise_mask,
                        audio_state.clean_latent,
                    )
                    next_video = stepper.step(video_state.latent, denoised_video, sigmas, step_idx).detach()
                    next_audio = stepper.step(audio_state.latent, denoised_audio, sigmas, step_idx).detach()
            else:
                denoised_video, denoised_audio = denoise_fn(video_state, audio_state, sigmas, step_idx)
                denoised_video = _samplers_module.post_process_latent(
                    denoised_video,
                    video_state.denoise_mask,
                    video_state.clean_latent,
                )
                denoised_audio = _samplers_module.post_process_latent(
                    denoised_audio,
                    audio_state.denoise_mask,
                    audio_state.clean_latent,
                )
                next_video = stepper.step(video_state.latent, denoised_video, sigmas, step_idx)
                next_audio = stepper.step(audio_state.latent, denoised_audio, sigmas, step_idx)

            video_state = replace(video_state, latent=next_video)
            audio_state = replace(audio_state, latent=next_audio)

        return (video_state, audio_state)

    def _cached_video_encode(video_encoder, video_path, output_shape, dtype, device):  # noqa: ARG001
        return cached_video_latent

    def _forced_audio_encode(audio_encoder, waveform, waveform_sr, output_shape, dtype):  # noqa: ARG001
        return injected_audio_latent

    _retake_module._encode_video_for_retake = _cached_video_encode
    _retake_module._encode_audio_for_retake = _forced_audio_encode
    _retake_module.euler_denoising_loop = _late_step_grad_euler_loop
    try:
        video_iter, _ = pipeline(video_path=src_video, **retake_kwargs)

        src_audio = decode_audio_from_file(src_video, pipeline.device, max_duration=num_frames / fps)
        if src_audio is None:
            out_audio = None
        else:
            wave = src_audio.waveform.squeeze(0).float()
            if src_audio.sampling_rate != audio_sr:
                wave = torchaudio.functional.resample(wave, orig_freq=src_audio.sampling_rate, new_freq=audio_sr)
            wave = align_waveform_length(wave, int((num_frames / fps) * audio_sr))
            if wave.shape[0] == 1:
                wave = wave.expand(2, -1).contiguous()
            elif wave.shape[0] > 2:
                wave = wave[:2].contiguous()
            out_audio = Audio(waveform=wave.cpu(), sampling_rate=audio_sr)

        encode_video(
            video=video_iter,
            fps=int(round(fps)),
            audio=out_audio,
            output_path=str(output_path),
            video_chunks_number=get_video_chunks_number(num_frames, TilingConfig.default()),
        )
    finally:
        _retake_module._encode_video_for_retake = orig_video_encode
        _retake_module._encode_audio_for_retake = orig_audio_encode
        _retake_module.euler_denoising_loop = orig_euler_loop


def prepare_target_motion_video(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s %(message)s", datefmt="%H:%M:%S")

    rank, _, device = init_distributed_and_device()
    is_main = rank == 0
    if not is_main:
        logging.getLogger().setLevel(logging.WARNING)

    output_dir = Path(args.output_dir).expanduser().resolve()
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    _barrier()

    loras = _parse_loras(args.loras)

    def _resolve_quantization_policy(name: str | None) -> QuantizationPolicy | None:
        if name == "fp8-cast":
            return QuantizationPolicy.fp8_cast()
        if name == "fp8-scaled-mm":
            try:
                return QuantizationPolicy.fp8_scaled_mm()
            except ImportError:
                log.warning(
                    "fp8-scaled-mm requested but tensorrt_llm is unavailable; falling back to fp8-cast."
                )
                return QuantizationPolicy.fp8_cast()
        return None

    ti2v_quant_name = args.ti2v_quantization if args.ti2v_quantization is not None else args.quantization
    ti2v_quant_policy = _resolve_quantization_policy(ti2v_quant_name)

    # Always generate target videos at original resolution to prevent crop mismatch
    # vs Retake squeezed geometries.
    orig_height, orig_width, num_frames, frame_rate = compute_target_shape(
        args.src_video,
        None,
        None,
        args.num_frames,
        args.frame_rate,
    )

    params = detect_params(args.checkpoint_path)
    video_guider_params, audio_guider_params = build_guiders_for_mode(
        args=args,
        params=params,
        use_low_memory_guidance=args.ti2v_low_memory_guidance,
    )

    if is_main:
        target_video_path = maybe_generate_target_video(
            args=args,
            device=device,
            height=orig_height,
            width=orig_width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            video_guider_params=video_guider_params,
            audio_guider_params=audio_guider_params,
            quant_policy=ti2v_quant_policy,
            loras=loras,
            output_dir=output_dir,
        )
        log.info("Prepared target motion video: %s", target_video_path)

    _barrier()
    if _is_distributed():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------

def optimize_audio_latent(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s %(message)s", datefmt="%H:%M:%S")

    rank, world_size, device = init_distributed_and_device()
    is_main = rank == 0
    if not is_main:
        logging.getLogger().setLevel(logging.WARNING)

    if world_size > 1 and not args.distributed_shard_transformer:
        raise ValueError(
            "Multi-process launch detected but --distributed-shard-transformer is disabled. "
            "Enable --distributed-shard-transformer for true multi-GPU gradient optimization."
        )
    if args.audio_opt_last_steps < 0:
        raise ValueError("--audio-opt-last-steps must be >= 0")
    if args.final_audio_opt_last_steps is not None and args.final_audio_opt_last_steps < 0:
        raise ValueError("--final-audio-opt-last-steps must be >= 0")

    output_dir = Path(args.output_dir).expanduser().resolve()
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    _barrier()

    loras = _parse_loras(args.loras)
    if loras:
        log.info("LoRAs: %s", [(l.path, l.strength) for l in loras])

    def _resolve_quantization_policy(name: str | None) -> QuantizationPolicy | None:
        if name == "fp8-cast":
            return QuantizationPolicy.fp8_cast()
        if name == "fp8-scaled-mm":
            try:
                return QuantizationPolicy.fp8_scaled_mm()
            except ImportError:
                log.warning(
                    "fp8-scaled-mm requested but tensorrt_llm is unavailable; falling back to fp8-cast."
                )
                return QuantizationPolicy.fp8_cast()
        return None

    # Backward-compatible behavior: --quantization applies to both unless stage-specific
    # overrides are provided.
    ti2v_quant_name = args.ti2v_quantization if args.ti2v_quantization is not None else args.quantization
    retake_quant_name = args.retake_quantization if args.retake_quantization is not None else args.quantization
    ti2v_quant_policy = _resolve_quantization_policy(ti2v_quant_name)
    retake_quant_policy = _resolve_quantization_policy(retake_quant_name)

    height, width, num_frames, frame_rate = compute_target_shape(
        args.src_video,
        args.height,
        args.width,
        args.num_frames,
        args.frame_rate,
    )
    duration = num_frames / frame_rate

    # Build a normalized retake input video so metadata matches cached latent
    # dimensions and model constraints (H/W multiples of 32, frames=8k+1).
    # Keep early preprocessing on CPU to avoid fragmenting GPU memory before
    # loading the very large TI2V transformer for target generation.
    src_audio_for_input = decode_audio_from_file(args.src_video, torch.device("cpu"), max_duration=duration)
    if src_audio_for_input is None:
        raise RuntimeError(f"Source video has no audio stream: {args.src_video}")

    src_wave = src_audio_for_input.waveform.squeeze(0).float()
    if src_audio_for_input.sampling_rate != args.audio_sr:
        src_wave = torchaudio.functional.resample(
            src_wave,
            orig_freq=src_audio_for_input.sampling_rate,
            new_freq=args.audio_sr,
        )
    src_wave = align_waveform_length(src_wave, int(duration * args.audio_sr))

    retake_input_video = output_dir / "retake_input_prepared.mp4"
    if is_main:
        write_temp_video_with_audio(
            src_video_path=args.src_video,
            target_frames=num_frames,
            target_height=height,
            target_width=width,
            fps=frame_rate,
            waveform=src_wave,
            sr=args.audio_sr,
            output_path=str(retake_input_video),
        )
        log.info("Prepared retake input video -> %s", retake_input_video)
    _barrier()

    # Free temporary preprocessing tensors before loading large model weights.
    del src_audio_for_input, src_wave
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    params = detect_params(args.checkpoint_path)
    # Memory-safe guidance is useful for Retake optimization, but TI2V target
    # generation usually needs full guidance quality.
    video_guider_params, audio_guider_params = build_guiders_for_mode(
        args=args,
        params=params,
        use_low_memory_guidance=args.low_memory_guidance,
    )

    # 1) Target-motion source
    if args.target_video is not None:
        target_video_path = Path(args.target_video).expanduser().resolve()
    else:
        target_video_path = output_dir / "target_motion_video.mp4"

    if is_main:
        # Always generate target videos at original resolution to prevent crop mismatch
        orig_height, orig_width, _, _ = compute_target_shape(
            args.src_video,
            None,
            None,
            args.num_frames,
            args.frame_rate,
        )
        target_video_path = maybe_generate_target_video(
            args=args,
            device=device,
            height=orig_height,
            width=orig_width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            video_guider_params=video_guider_params,
            audio_guider_params=audio_guider_params,
            quant_policy=ti2v_quant_policy,
            loras=loras,
            output_dir=output_dir,
        )
    _barrier()
    if not target_video_path.exists():
        raise FileNotFoundError(f"Target-motion video not found after preparation: {target_video_path}")

    resize_to = (args.flow_width, args.flow_height)

    eval_sample_start = max(0, int(args.eval_start_frame // max(args.frame_stride, 1)))

    # If ROI mask is provided and no explicit eval start is requested, choose
    # the sampled temporal window with maximal mask coverage.
    if args.roi_mask_video is not None and args.eval_start_frame < 0:
        roi_mask_path_for_search = Path(args.roi_mask_video).expanduser().resolve()
        if not roi_mask_path_for_search.exists():
            raise FileNotFoundError(f"--roi-mask-video not found: {roi_mask_path_for_search}")
        search_masks = decode_video_mask_frames(
            video_path=str(roi_mask_path_for_search),
            max_frames=None,
            frame_stride=args.frame_stride,
            resize_to=resize_to,
            threshold=args.roi_mask_threshold,
            sample_start=0,
        )
        if len(search_masks) > 0:
            coverage = np.array([float(m.mean()) for m in search_masks], dtype=np.float32)
            window = args.max_eval_frames if args.max_eval_frames is not None else len(coverage)
            window = max(1, min(window, len(coverage)))
            window_sums = np.convolve(coverage, np.ones(window, dtype=np.float32), mode="valid")
            eval_sample_start = int(np.argmax(window_sums)) if len(window_sums) > 0 else 0
            log.info(
                "Auto-selected eval sampled-window start=%d (frame~%d) from ROI coverage.",
                eval_sample_start,
                eval_sample_start * args.frame_stride,
            )

    # Clamp eval start so generated Retake frames still provide at least 2
    # sampled frames (needed for optical flow). Target videos can be longer than
    # generated clips, so ROI-driven starts may otherwise overshoot generation.
    sampled_total_generated = ((num_frames - 1) // max(args.frame_stride, 1)) + 1
    max_eval_start_for_flow = max(0, sampled_total_generated - 2)
    if eval_sample_start > max_eval_start_for_flow:
        log.warning(
            "Eval sampled-window start=%d exceeds generated clip capacity (%d sampled frames). "
            "Clamping to %d.",
            eval_sample_start,
            sampled_total_generated,
            max_eval_start_for_flow,
        )
        eval_sample_start = max_eval_start_for_flow

    target_frames_rgb = decode_video_frames_rgb(
        video_path=str(target_video_path),
        max_frames=args.max_eval_frames,
        frame_stride=args.frame_stride,
        resize_to=None,
        sample_start=eval_sample_start,
    )
    roi_frame_masks = None
    if args.roi_mask_video is not None:
        roi_mask_path = Path(args.roi_mask_video).expanduser().resolve()
        if not roi_mask_path.exists():
            raise FileNotFoundError(f"--roi-mask-video not found: {roi_mask_path}")
        roi_mask_frames = decode_video_mask_frames(
            video_path=str(roi_mask_path),
            max_frames=args.max_eval_frames,
            frame_stride=args.frame_stride,
            resize_to=resize_to,
            threshold=args.roi_mask_threshold,
            sample_start=eval_sample_start,
        )
        roi_frame_masks = mask_frames_to_nchw_float(roi_mask_frames, device=device)
        if roi_frame_masks.shape[0] < 2:
            raise RuntimeError("ROI mask video must provide at least 2 usable frames.")
        roi_coverage = float(roi_frame_masks.mean().item())
        if roi_coverage <= 0.0:
            raise RuntimeError(
                "ROI mask video has zero active pixels after decoding/thresholding. "
                "Lower --roi-mask-threshold, regenerate masks, or verify object prompt/tracking."
            )
        if roi_coverage < 1e-3:
            log.warning(
                "ROI mask coverage is extremely low (%.4f%%). Loss signal may be weak.",
                roi_coverage * 100.0,
            )
        log.info(
            "Loaded ROI masks from %s with %.2f%% average coverage",
            roi_mask_path,
            roi_coverage * 100.0,
        )

    target_frames = frames_rgb_uint8_to_chw_float(target_frames_rgb, device=device)
    if target_frames.shape[0] == 0:
        raise RuntimeError("Target frames are empty. Increase --max-eval-frames or check target video.")
    if resize_to is not None:
        target_frames = F.interpolate(
            target_frames,
            size=(resize_to[1], resize_to[0]),
            mode="bilinear",
            align_corners=False,
        )

    raft_model, raft_transforms = load_raft_components(
        device=device,
        model_name=args.raft_model,
        weights_path=args.raft_weights_path,
    )
    
    lpips_model = None
    source_frames_lpips = None
    if args.lpips_weight > 0.0:
        import lpips
        log.info("Loading LPIPS model...")
        lpips_model = lpips.LPIPS(net="vgg").to(device)
        for param in lpips_model.parameters():
            param.requires_grad = False
        lpips_model.eval()

        source_frames_rgb = decode_video_frames_rgb(
            video_path=str(retake_input_video),
            max_frames=args.max_eval_frames,
            frame_stride=args.frame_stride,
            resize_to=None,
            sample_start=eval_sample_start,
        )
        source_frames_t = frames_rgb_uint8_to_chw_float(source_frames_rgb, device=device)
        if resize_to is not None:
            source_frames_t = F.interpolate(
                source_frames_t,
                size=(resize_to[1], resize_to[0]),
                mode="bilinear",
                align_corners=False,
            )
        # Normalize to [-1, 1] for LPIPS
        source_frames_lpips = source_frames_t * 2.0 - 1.0

    with torch.no_grad():
        target_flows = compute_raft_flows(target_frames, raft_model, raft_transforms).detach()
    if target_flows.shape[0] == 0:
        raise RuntimeError("Target RAFT flow is empty. Need at least 2 target frames.")

    log.info("Target video: %s", target_video_path)
    log.info("Target flow fields: %d", target_flows.shape[0])

    # 2) Build Retake pipeline and cache source latents
    pipeline = RetakePipeline(
        checkpoint_path=args.checkpoint_path,
        gemma_root=args.gemma_root,
        loras=tuple(loras),
        device=device,
        quantization=retake_quant_policy,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    if world_size > 1 and args.distributed_shard_transformer:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        cached_transformer = pipeline.model_ledger.transformer()
        param_dtypes_before = {
            p.dtype
            for p in cached_transformer.parameters()
            if torch.is_floating_point(p)
        }
        if len(param_dtypes_before) > 1:
            # FSDP flatten requires uniform floating dtype across managed params.
            # Some quantized/checkpoint-loaded paths may leave a subset in fp32.
            target_dtype = pipeline.dtype
            if is_main:
                log.warning(
                    "Transformer has mixed floating dtypes before FSDP (%s). Casting to %s for uniform flatten.",
                    sorted(str(d) for d in param_dtypes_before),
                    target_dtype,
                )
            cached_transformer = cached_transformer.to(dtype=target_dtype)

        velocity_model = getattr(cached_transformer, "velocity_model", None)
        if args.gradient_checkpointing and velocity_model is not None and hasattr(velocity_model, "set_gradient_checkpointing"):
            velocity_model.set_gradient_checkpointing(True)
        cached_transformer.requires_grad_(False)
        try:
            sharded_transformer = FSDP(
                cached_transformer,
                use_orig_params=True,
                device_id=torch.cuda.current_device() if torch.cuda.is_available() else None,
                limit_all_gathers=True,
            )
        except torch.OutOfMemoryError as exc:
            raise RuntimeError(
                "FSDP init OOM while sharding transformer. This is a peak-memory issue during flatten/shard. "
                "Use lower-memory transformer weights for Retake (recommended: --retake-quantization fp8-cast), "
                "reduce num_frames/resolution, or increase number of GPUs."
            ) from exc
        pipeline.model_ledger.transformer = lambda: sharded_transformer
        if is_main:
            log.info("Using FSDP-sharded Retake transformer across %d ranks", world_size)

    cached_video_latent, base_audio_latent, waveform_sr = build_cached_source_latents(
        pipeline=pipeline,
        src_video=str(retake_input_video),
        height=height,
        width=width,
        num_frames=num_frames,
        audio_sr=args.audio_sr,
        device=device,
    )

    log.info("Cached video latent shape: %s", tuple(cached_video_latent.shape))
    log.info("Base audio latent shape: %s", tuple(base_audio_latent.shape))

    retake_kwargs = dict(
        prompt=args.edit_prompt,
        start_time=args.retake_start_frames / frame_rate,
        end_time=duration,
        seed=args.seed,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.retake_num_inference_steps,
        video_guider_params=video_guider_params,
        audio_guider_params=audio_guider_params,
        regenerate_video=True,
        regenerate_audio=False,
        enhance_prompt=args.enhance_prompt,
        tiling_config=TilingConfig.default(),
    )

    if args.audio_opt_last_steps > 0:
        log.info(
            "Late-step audio optimization enabled: gradients kept only for last %d/%d denoising steps",
            args.audio_opt_last_steps,
            retake_kwargs["num_inference_steps"],
        )

    # 3) Directly optimise the audio latent in the full latent space
    base_audio_latent_fp32 = base_audio_latent.float().detach()
    audio_latent = torch.nn.Parameter(base_audio_latent_fp32.clone())
    optimizer = torch.optim.Adam([audio_latent], lr=args.lr)

    csv_path = output_dir / "optimization_log.csv"
    with (csv_path.open("w", newline="") if is_main else open(os.devnull, "w", newline="")) as f_csv:
        writer = csv.writer(f_csv)
        if is_main:
            writer.writerow([
                "iter",
                "total_loss",
                "flow_mse",
                "mag_curve_mse",
                "latent_reg",
                "lpips",
                "grad_norm",
                "is_best",
            ])

        best = {
            "loss": float("inf"),
            "latent": audio_latent.detach().clone(),
            "flow_mse": float("inf"),
            "mag_mse": float("inf"),
            "lpips": float("inf"),
        }

        num_iters = args.iterations
        best_latent_path = output_dir / "best_audio_latent.pt"
        if args.resume and best_latent_path.exists():
            log.info(f"Resuming from existing latent at {best_latent_path}, skipping optimization.")
            loaded = torch.load(best_latent_path, map_location="cpu").to(device=audio_latent.device, dtype=audio_latent.dtype)
            audio_latent.data.copy_(loaded)
            best["latent"] = audio_latent.detach().clone()
            num_iters = 0
            
            # Restore metrics if available
            params_path = output_dir / "best_latent_params.pt"
            if params_path.exists():
                saved_params = torch.load(params_path, map_location="cpu")
                best["loss"] = saved_params.get("best_loss", float("inf"))
                best["flow_mse"] = saved_params.get("best_flow_mse", float("inf"))
                best["mag_mse"] = saved_params.get("best_mag_mse", float("inf"))
                best["lpips"] = saved_params.get("best_lpips", float("inf"))

        for it in range(1, num_iters + 1):
            optimizer.zero_grad(set_to_none=True)

            injected_latent = audio_latent.to(dtype=base_audio_latent.dtype)
            gen_frames = render_with_injected_audio_latent(
                pipeline=pipeline,
                src_video=str(retake_input_video),
                injected_audio_latent=injected_latent,
                cached_video_latent=cached_video_latent,
                retake_kwargs=retake_kwargs,
                max_frames=args.max_eval_frames,
                frame_stride=args.frame_stride,
                resize_to=resize_to,
                audio_opt_last_steps=args.audio_opt_last_steps,
                eval_sample_start=eval_sample_start,
            )
            if gen_frames.shape[0] < 2:
                raise RuntimeError("Generated video has fewer than 2 frames; cannot compute flow objective.")

            flow_total, flow_mse_t, mag_mse_t = flow_objective_torch(
                gen_frames_chw=gen_frames,
                target_flows=target_flows,
                raft_model=raft_model,
                raft_transforms=raft_transforms,
                flow_weight=args.flow_weight,
                mag_curve_weight=args.mag_curve_weight,
                roi_frame_masks=roi_frame_masks,
            )
            
            lpips_loss_t = torch.tensor(0.0, device=device)
            if args.lpips_weight > 0.0 and lpips_model is not None:
                # Limit LPIPS computation to matched frames
                n_frames = min(gen_frames.shape[0], source_frames_lpips.shape[0])
                gen_lpips = gen_frames[:n_frames] * 2.0 - 1.0 # [0, 1] to [-1, 1]
                src_lpips = source_frames_lpips[:n_frames]
                
                # lpips_model returns [N, 1, 1, 1]
                lpips_loss_val = lpips_model(gen_lpips, src_lpips).mean()
                lpips_loss_t = args.lpips_weight * lpips_loss_val

            latent_reg_t = args.latent_reg_weight * torch.mean((audio_latent - base_audio_latent_fp32) ** 2)
            total_t = flow_total + latent_reg_t + lpips_loss_t
            total_t.backward()

            if world_size > 1 and audio_latent.grad is not None:
                dist.all_reduce(audio_latent.grad, op=dist.ReduceOp.SUM)
                audio_latent.grad.div_(world_size)

            grad_norm = float(audio_latent.grad.norm().item()) if audio_latent.grad is not None else 0.0
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_([audio_latent], max_norm=args.grad_clip)
            optimizer.step()

            total = float(total_t.detach().item())
            flow_mse = float(flow_mse_t.detach().item())
            mag_mse = float(mag_mse_t.detach().item())
            latent_reg = float(latent_reg_t.detach().item())
            lpips_val = float(lpips_loss_t.detach().item())

            is_best = total < best["loss"]
            if is_main:
                writer.writerow([it, total, flow_mse, mag_mse, latent_reg, lpips_val, grad_norm, int(is_best)])
                f_csv.flush()

            if is_best:
                best["loss"] = total
                best["latent"] = audio_latent.detach().clone()
                best["flow_mse"] = flow_mse
                best["mag_mse"] = mag_mse
                best["lpips"] = lpips_val

            log.info(
                "iter=%d total=%.6f flow=%.6f mag=%.6f reg=%.6f lpips=%.6f grad=%.6f best=%.6f",
                it,
                total,
                flow_mse,
                mag_mse,
                latent_reg,
                lpips_val,
                grad_norm,
                best["loss"],
            )

    best_latent = best["latent"].to(dtype=base_audio_latent.dtype)

    if is_main:
        torch.save(
            {
                "best_loss": best["loss"],
                "best_flow_mse": best["flow_mse"],
                "best_mag_mse": best["mag_mse"],
                "best_lpips": best.get("lpips", 0.0),
                "world_size": world_size,
            },
            output_dir / "best_latent_params.pt",
        )

        # Save optimized and baseline audio latents for later experiments
        torch.save(base_audio_latent.detach().cpu(), output_dir / "base_audio_latent.pt")
        torch.save(best_latent.detach().cpu(), output_dir / "best_audio_latent.pt")

    # Save final videos (can use a higher step budget than optimization)
    final_retake_kwargs = dict(retake_kwargs)
    if args.final_retake_num_inference_steps is not None:
        final_retake_kwargs["num_inference_steps"] = args.final_retake_num_inference_steps
    
    # Rebuild guiders to ensure we use max quality for final renders
    # (i.e. not memory-safe forced cfg=1.0) if optimization used lower quality.
    final_video_guiders, final_audio_guiders = build_guiders_for_mode(
        args=args,
        params=params,
        use_low_memory_guidance=False,
    )
    final_retake_kwargs["video_guider_params"] = final_video_guiders
    final_retake_kwargs["audio_guider_params"] = final_audio_guiders

    final_audio_opt_last_steps = (
        args.final_audio_opt_last_steps if args.final_audio_opt_last_steps is not None else args.audio_opt_last_steps
    )

    if is_main and args.save_final_videos and world_size == 1:
        # Check if we need to upscale back to original dimensions for final render
        orig_height, orig_width, _, _ = compute_target_shape(
            args.src_video,
            height_override=None,
            width_override=None,
            num_frames_override=args.num_frames,
            frame_rate_override=args.frame_rate,
        )

        final_height = height
        final_width = width
        final_retake_input_video = retake_input_video
        final_cached_video_latent = cached_video_latent

        if orig_height != height or orig_width != width:
            log.info("Restoring original source dimensions (%dx%d) for final video renders...", orig_width, orig_height)
            final_height, final_width = orig_height, orig_width
            final_retake_input_video = output_dir / "retake_input_prepared_hires.mp4"

            # Re-read audio briefly to mux
            src_audio_for_input = decode_audio_from_file(args.src_video, torch.device("cpu"), max_duration=duration)
            if src_audio_for_input is not None:
                src_wave = src_audio_for_input.waveform.squeeze(0).float()
                if src_audio_for_input.sampling_rate != args.audio_sr:
                    src_wave = torchaudio.functional.resample(
                        src_wave,
                        orig_freq=src_audio_for_input.sampling_rate,
                        new_freq=args.audio_sr,
                    )
                src_wave = align_waveform_length(src_wave, int(duration * args.audio_sr))
            else:
                src_wave = None

            write_temp_video_with_audio(
                src_video_path=args.src_video,
                target_frames=num_frames,
                target_height=final_height,
                target_width=final_width,
                fps=frame_rate,
                waveform=src_wave,
                sr=args.audio_sr,
                output_path=str(final_retake_input_video),
            )
            final_cached_video_latent, _, _ = build_cached_source_latents(
                pipeline=pipeline,
                src_video=str(final_retake_input_video),
                height=final_height,
                width=final_width,
                num_frames=num_frames,
                audio_sr=args.audio_sr,
                device=device,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Save final videos
        best_video_path = output_dir / "best_optimized_video.mp4"
        render_and_save_video_with_latent(
            pipeline=pipeline,
            src_video=str(final_retake_input_video),
            injected_audio_latent=best_latent,
            cached_video_latent=final_cached_video_latent,
            retake_kwargs=final_retake_kwargs,
            output_path=best_video_path,
            fps=frame_rate,
            num_frames=num_frames,
            audio_sr=waveform_sr,
            audio_opt_last_steps=final_audio_opt_last_steps,
        )

        baseline_video_path = output_dir / "baseline_unoptimized_video.mp4"
        render_and_save_video_with_latent(
            pipeline=pipeline,
            src_video=str(final_retake_input_video),
            injected_audio_latent=base_audio_latent,
            cached_video_latent=final_cached_video_latent,
            retake_kwargs=final_retake_kwargs,
            output_path=baseline_video_path,
            fps=frame_rate,
            num_frames=num_frames,
            audio_sr=waveform_sr,
            audio_opt_last_steps=final_audio_opt_last_steps,
        )

        if args.transfer_prompt:
            transfer_kwargs = dict(final_retake_kwargs)
            transfer_kwargs["prompt"] = args.transfer_prompt
            transfer_video_path = output_dir / "transfer_prompt_with_best_latent.mp4"
            render_and_save_video_with_latent(
                pipeline=pipeline,
                src_video=str(final_retake_input_video),
                injected_audio_latent=best_latent,
                cached_video_latent=final_cached_video_latent,
                retake_kwargs=transfer_kwargs,
                output_path=transfer_video_path,
                fps=frame_rate,
                num_frames=num_frames,
                audio_sr=waveform_sr,
                audio_opt_last_steps=final_audio_opt_last_steps,
            )
    elif is_main and args.save_final_videos and world_size > 1:
        log.warning("Skipping final video rendering in distributed mode. Re-run single-GPU with saved best latent to render.")

    # For convenience: save source audio track used for muxing
    if is_main:
        src_audio = decode_audio_from_file(args.src_video, device, max_duration=duration)
        if src_audio is not None:
            wave = src_audio.waveform.squeeze(0).float()
            if src_audio.sampling_rate != args.audio_sr:
                wave = torchaudio.functional.resample(wave, orig_freq=src_audio.sampling_rate, new_freq=args.audio_sr)
            wave = align_waveform_length(wave, int(duration * args.audio_sr))
            save_audio_wav(wave, args.audio_sr, str(output_dir / "source_audio_used.wav"))

        try:
            from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
            log.info("Decoding optimized audio latents to .wav files...")
            
            audio_decoder = pipeline.model_ledger.audio_decoder().to(device)
            vocoder = pipeline.model_ledger.vocoder().to(device)
            
            with torch.no_grad():
                best_audio_decoded = vae_decode_audio(
                    best_latent.to(device).to(audio_decoder.dtype),
                    audio_decoder,
                    vocoder,
                )
                save_audio_wav(
                    best_audio_decoded.waveform.squeeze(0),
                    best_audio_decoded.sampling_rate,
                    str(output_dir / "best_optimized_audio.wav")
                )
                
                base_audio_decoded = vae_decode_audio(
                    base_audio_latent.to(device).to(audio_decoder.dtype),
                    audio_decoder,
                    vocoder,
                )
                save_audio_wav(
                    base_audio_decoded.waveform.squeeze(0),
                    base_audio_decoded.sampling_rate,
                    str(output_dir / "baseline_unoptimized_audio.wav")
                )
            
            del audio_decoder, vocoder
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            log.warning("Could not decode and save audio latents: %s", e)

        log.info("Done. Best total loss: %.6f", best["loss"])
        log.info("Saved outputs in: %s", output_dir)

    _barrier()
    if _is_distributed():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Core I/O
    p.add_argument("--src-video", required=True, help="Source video used for retake and latent optimization.")
    p.add_argument("--edit-prompt", required=True, help="Prompt used during optimization retake generation.")
    p.add_argument("--output-dir", required=True, help="Directory where outputs/logs are written.")

    # Target motion source: either provide video or generate via TI2V prompt
    p.add_argument("--target-video", default=None, help="Optional existing target-motion video.")
    p.add_argument(
        "--target-prompt",
        default=None,
        help=(
            "If --target-video is not provided, generate target video once with TI2V using "
            "this prompt and source first frame."
        ),
    )
    p.add_argument(
        "--regenerate-target",
        action="store_true",
        help="Regenerate target-motion video even if output_dir/target_motion_video.mp4 already exists.",
    )
    p.add_argument(
        "--prepare-target-only",
        action="store_true",
        help="Generate or reuse the target-motion video and exit without optimization.",
    )

    # Optional transfer test
    p.add_argument(
        "--transfer-prompt",
        default=None,
        help="Optional prompt to test whether optimized latent transfers its motion style.",
    )

    # Objective settings
    p.add_argument("--flow-weight", type=float, default=1.0, help="Weight for dense flow field MSE.")
    p.add_argument("--mag-curve-weight", type=float, default=0.25, help="Weight for mean flow-magnitude curve MSE.")
    p.add_argument("--lpips-weight", type=float, default=0.1, help="Weight for LPIPS perceptual loss against source frames.")
    p.add_argument("--resume", action="store_true", help="If best_audio_latent.pt exists in output dir, load it and skip the optimization loop.")
    p.add_argument("--latent-reg-weight", type=float, default=0.05, help="Weight for latent drift regularization.")
    p.add_argument("--flow-width", type=int, default=512, help="Flow objective width (frames are resized to this).")
    p.add_argument("--flow-height", type=int, default=320, help="Flow objective height (frames are resized to this).")
    p.add_argument("--max-eval-frames", type=int, default=33, help="Max frames used in objective computation.")
    p.add_argument("--frame-stride", type=int, default=1, help="Use every N-th frame when computing objective.")
    p.add_argument(
        "--eval-start-frame",
        type=int,
        default=-1,
        help=(
            "Start frame for objective sampling in original frame units. "
            "Set <0 to auto-select based on ROI mask coverage when --roi-mask-video is used."
        ),
    )
    p.add_argument(
        "--roi-mask-video",
        type=str,
        default=None,
        help=(
            "Optional binary mask video aligned with the target video. "
            "When provided, RAFT loss is applied only inside the masked region."
        ),
    )
    p.add_argument(
        "--roi-mask-threshold",
        type=float,
        default=0.5,
        help="Threshold used to binarize --roi-mask-video frames after grayscale conversion.",
    )

    # Optimization settings (gradient-based Adam)
    p.add_argument("--iterations", type=int, default=30, help="Number of optimization iterations.")
    p.add_argument("--lr", type=float, default=0.015, help="Adam learning rate for latent coefficients.")
    p.add_argument("--grad-clip", type=float, default=0.5, help="Clip alpha gradient norm (<=0 disables clip).")
    p.add_argument(
        "--audio-opt-last-steps",
        type=int,
        default=6,
        help=(
            "Keep gradients only through the final K denoising steps during optimization "
            "(0 = full-step gradients)."
        ),
    )
    p.add_argument(
        "--final-audio-opt-last-steps",
        type=int,
        default=None,
        help=(
            "Optional override for final render videos. If omitted, uses --audio-opt-last-steps."
        ),
    )
    p.add_argument(
        "--distributed-shard-transformer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable FSDP transformer sharding for multi-GPU gradient optimization. "
            "Launch with torchrun/srun for this mode."
        ),
    )
    p.add_argument(
        "--save-final-videos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render final videos at the end (auto-disabled when distributed).",
    )
    p.add_argument(
        "--raft-model",
        type=str,
        default="raft_large",
        choices=["raft_large", "raft_small"],
        help="Frozen RAFT backbone used in the flow objective.",
    )
    p.add_argument(
        "--raft-weights-path",
        type=str,
        default=None,
        help=(
            "Optional local .pth checkpoint for RAFT (offline mode). If omitted, "
            "torchvision uses its default weight download/cache behavior."
        ),
    )

    # Prompt / diffusion controls
    p.add_argument("--negative-prompt", default="", help="Negative prompt used in generation.")
    p.add_argument("--enhance-prompt", action="store_true", help="Enable Gemma prompt enhancement.")
    p.add_argument("--num-inference-steps", type=int, default=40, help="Denoising steps.")
    p.add_argument(
        "--ti2v-num-inference-steps",
        type=int,
        default=None,
        help="Optional denoising steps override for TI2V target generation stage.",
    )
    p.add_argument(
        "--retake-num-inference-steps",
        type=int,
        default=None,
        help="Optional denoising steps override for Retake optimization stage.",
    )
    p.add_argument(
        "--final-retake-num-inference-steps",
        type=int,
        default=None,
        help=(
            "Optional denoising steps used only for final output renders "
            "(best/baseline/transfer). If omitted, uses retake optimization steps."
        ),
    )
    p.add_argument("--seed", type=int, default=42, help="Sampling seed for generation calls.")
    p.add_argument("--retake-start-frames", type=int, default=1, help="Frames to keep fixed at start.")

    # Shape
    p.add_argument("--height", type=int, default=None, help="Override output height (multiple of 32).")
    p.add_argument("--width", type=int, default=None, help="Override output width (multiple of 32).")
    p.add_argument("--num-frames", type=int, default=None, help="Override frame count (8k+1).")
    p.add_argument("--frame-rate", type=float, default=None, help="Override frame rate.")

    # Audio
    p.add_argument("--audio-sr", type=int, default=44100, help="Target audio sample rate.")

    # Model
    p.add_argument("--checkpoint-path", default=DEFAULT_CHECKPOINT)
    p.add_argument("--gemma-root", default=DEFAULT_GEMMA_ROOT)
    p.add_argument("--quantization", default=None, choices=["fp8-cast", "fp8-scaled-mm"])
    p.add_argument(
        "--ti2v-quantization",
        default=None,
        choices=["fp8-cast", "fp8-scaled-mm"],
        help="Optional quantization override for target TI2V generation stage.",
    )
    p.add_argument(
        "--retake-quantization",
        default=None,
        choices=["fp8-cast", "fp8-scaled-mm"],
        help="Optional quantization override for Retake optimization stage.",
    )
    p.add_argument(
        "--lora",
        dest="loras",
        nargs="+",
        metavar=("PATH", "STRENGTH"),
        action="append",
        default=[],
        help="LoRA path and optional strength. Can be passed multiple times.",
    )

    # Guidance
    p.add_argument("--cfg-scale", type=float, default=None, help="Video CFG scale (auto if omitted).")
    p.add_argument("--audio-cfg-scale", type=float, default=None, help="Audio CFG scale (auto if omitted).")
    p.add_argument("--a2v-scale", type=float, default=None, help="Audio-to-video modality scale (auto if omitted).")
    p.add_argument(
        "--low-memory-guidance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use memory-safe guider settings (cfg=1, stg=0, rescale=0 by default) "
            "to avoid extra transformer passes during latent optimization."
        ),
    )
    p.add_argument(
        "--ti2v-low-memory-guidance",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use memory-safe guidance for TI2V target generation as well. Disabled by default "
            "because it can noticeably degrade target-video quality."
        ),
    )
    p.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable activation checkpointing in Retake transformer to reduce VRAM during latent optimization.",
    )

    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.ti2v_num_inference_steps is None:
        args.ti2v_num_inference_steps = args.num_inference_steps
    if args.retake_num_inference_steps is None:
        args.retake_num_inference_steps = args.num_inference_steps
    if args.prepare_target_only:
        prepare_target_motion_video(args)
        return
    optimize_audio_latent(args)


if __name__ == "__main__":
    main()
