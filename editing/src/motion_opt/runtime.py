from __future__ import annotations

import gc
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from .distributed import barrier
from .metrics import decode_video_frames_rgb, frames_rgb_uint8_to_chw_float, load_raft_components, compute_raft_flows
from .models import build_guiders
from .core import (
    decode_audio_from_file,
    decode_video_mask_frames,
    get_videostream_metadata,
    mask_frames_to_nchw_float,
    maybe_generate_target_video,
    write_temp_video_with_audio,
)

log = logging.getLogger(__name__)


def _retake_ready_source_reason(
    *,
    src_video: Path,
    height: int,
    width: int,
    num_frames: int,
    frame_rate: float,
    audio_sr: int,
    duration: float,
) -> tuple[bool, str]:
    try:
        src_fps, src_frames, src_width, src_height = get_videostream_metadata(str(src_video))
    except Exception as exc:
        return False, f"metadata probe failed: {exc}"

    if src_width != width or src_height != height:
        return False, f"size is {src_width}x{src_height}, expected {width}x{height}"
    if src_frames != num_frames:
        return False, f"frame count is {src_frames}, expected {num_frames}"
    if abs(float(src_fps) - float(frame_rate)) > 1e-3:
        return False, f"fps is {src_fps:.6g}, expected {frame_rate:.6g}"

    src_audio = decode_audio_from_file(str(src_video), torch.device("cpu"), max_duration=duration)
    if src_audio is None:
        return False, "missing audio stream"
    if int(src_audio.sampling_rate) != int(audio_sr):
        return False, f"audio sr is {src_audio.sampling_rate}, expected {audio_sr}"

    expected_samples = int(duration * audio_sr)
    actual_samples = int(src_audio.waveform.shape[-1])
    if actual_samples + 2 < expected_samples:
        return False, f"audio has {actual_samples} samples, expected at least {expected_samples}"

    return True, "matches retake-ready shape/audio"


def _refresh_retake_input_link(*, src_video: Path, retake_input_video: Path) -> None:
    try:
        if retake_input_video.exists() or retake_input_video.is_symlink():
            try:
                if retake_input_video.samefile(src_video):
                    return
            except FileNotFoundError:
                pass
            if retake_input_video.is_dir():
                log.warning("Cannot replace retake input artifact directory: %s", retake_input_video)
                return
            retake_input_video.unlink()
        retake_input_video.symlink_to(src_video)
    except OSError:
        log.warning("Could not symlink retake input artifact to %s", src_video, exc_info=True)


def prepare_retake_input_video(*, args, is_main: bool, output_dir: Path, height: int, width: int, num_frames: int, frame_rate: float) -> Path:
    duration = num_frames / frame_rate
    src_video_path = Path(args.src_video).expanduser().resolve()
    retake_input_video = output_dir / "retake_input_prepared.mp4"

    is_ready, ready_reason = _retake_ready_source_reason(
        src_video=src_video_path,
        height=height,
        width=width,
        num_frames=num_frames,
        frame_rate=frame_rate,
        audio_sr=args.audio_sr,
        duration=duration,
    )
    if is_ready:
        if is_main:
            _refresh_retake_input_link(src_video=src_video_path, retake_input_video=retake_input_video)
            log.info("Source video is already retake-ready (%s); using without preprocessing: %s", ready_reason, src_video_path)
        barrier()
        return src_video_path
    if is_main:
        log.info("Preparing retake input video because source is not retake-ready (%s).", ready_reason)

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
    eval_sample_start = max(0, int(args.eval_start_frame // max(args.frame_stride, 1)))

    roi_frame_masks = None
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

    sampled_total_generated = ((num_frames - 1) // max(args.frame_stride, 1)) + 1
    max_eval_start_for_flow = max(0, sampled_total_generated - 2)
    if eval_sample_start > max_eval_start_for_flow:
        log.warning(
            "Eval sampled-window start=%d exceeds generated clip capacity (%d sampled frames). Clamping to %d.",
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
            log.warning("ROI mask coverage is extremely low (%.4f%%). Loss signal may be weak.", roi_coverage * 100.0)
        log.info("Loaded ROI masks from %s with %.2f%% average coverage", roi_mask_path, roi_coverage * 100.0)

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

    return target_video_path, resize_to, target_flows, raft_model, raft_transforms, roi_frame_masks, eval_sample_start


def build_retake_kwargs(*, args, frame_rate: float, duration: float, video_guider_params, audio_guider_params) -> dict:
    from ltx_core.model.video_vae import TilingConfig

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
        tiling_config=TilingConfig.default(),
    )


def get_guiders(args, checkpoint_path: str):
    return build_guiders(args, checkpoint_path)
