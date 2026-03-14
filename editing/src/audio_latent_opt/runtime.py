from __future__ import annotations

import gc
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio

from .distributed import barrier
from .metrics import decode_video_frames_rgb, frames_rgb_uint8_to_chw_float, load_raft_components, compute_raft_flows
from .models import build_guiders

# Reuse stable helpers from legacy script.
from editing.optimize_audio_embedding import (
    build_cached_source_latents,
    decode_audio_from_file,
    maybe_generate_target_video,
    write_temp_video_with_audio,
)

log = logging.getLogger(__name__)


def prepare_retake_input_video(*, args, is_main: bool, output_dir: Path, height: int, width: int, num_frames: int, frame_rate: float) -> Path:
    duration = num_frames / frame_rate

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

    from .helpers import align_waveform_length

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

    barrier()

    del src_audio_for_input, src_wave
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    return retake_input_video


def prepare_target_flow(*, args, device: torch.device, output_dir: Path, is_main: bool, height: int, width: int, num_frames: int, frame_rate: float, video_guider_params, audio_guider_params, quant_policy, loras):
    if args.target_video is not None:
        target_video_path = Path(args.target_video).expanduser().resolve()
    else:
        target_video_path = output_dir / "target_motion_video.mp4"

    if is_main:
        from .helpers import compute_target_shape
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
            quant_policy=quant_policy,
            loras=loras,
            output_dir=output_dir,
        )

    barrier()
    if not target_video_path.exists():
        raise FileNotFoundError(f"Target-motion video not found after preparation: {target_video_path}")

    resize_to = (args.flow_width, args.flow_height)
    target_frames_rgb = decode_video_frames_rgb(
        video_path=str(target_video_path),
        max_frames=args.max_eval_frames,
        frame_stride=args.frame_stride,
        resize_to=None,
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
    with torch.no_grad():
        target_flows = compute_raft_flows(target_frames, raft_model, raft_transforms).detach()
    if target_flows.shape[0] == 0:
        raise RuntimeError("Target RAFT flow is empty. Need at least 2 target frames.")

    return target_video_path, resize_to, target_flows, raft_model, raft_transforms


def build_retake_kwargs(*, args, frame_rate: float, duration: float, video_guider_params, audio_guider_params) -> dict:
    return dict(
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
        tiling_config=__import__("ltx_core.model.video_vae", fromlist=["TilingConfig"]).TilingConfig.default(),
    )


def get_guiders(args, checkpoint_path: str):
    return build_guiders(args, checkpoint_path)
