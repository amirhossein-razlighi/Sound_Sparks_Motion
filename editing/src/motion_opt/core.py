from __future__ import annotations

import argparse
import tempfile
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Iterator

import av
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

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

        time_base = Fraction(1, int(round(fps)))
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


def decode_video_frames_rgb(
    video_path: str,
    max_frames: int | None,
    frame_stride: int,
    resize_to: tuple[int, int] | None,
    sample_start: int = 0,
) -> list[np.ndarray]:
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
                arr = F.interpolate(arr_t, size=(resize_to[1], resize_to[0]), mode="nearest")[0, 0].numpy()
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
    frames: list[torch.Tensor] = []
    seen = 0
    sampled_seen = 0
    for chunk in video_iter:
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
                    stacked = F.interpolate(stacked, size=(resize_to[1], resize_to[0]), mode="bilinear", align_corners=False)
                return stacked

    if len(frames) == 0:
        return torch.empty(0, 3, 0, 0)

    stacked = torch.stack(frames, dim=0)
    if resize_to is not None:
        stacked = F.interpolate(stacked, size=(resize_to[1], resize_to[0]), mode="bilinear", align_corners=False)
    return stacked


def load_raft_components(device: torch.device, model_name: str, weights_path: str | None = None):
    from torchvision.models.optical_flow import Raft_Large_Weights, Raft_Small_Weights, raft_large, raft_small

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


def compute_raft_flows(frames_chw: torch.Tensor, raft_model: torch.nn.Module, raft_transforms) -> torch.Tensor:
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


def _parse_loras(raw_loras: list[list[str]]) -> list[LoraPathStrengthAndSDOps]:
    out: list[LoraPathStrengthAndSDOps] = []
    for entry in raw_loras:
        lora_path = str(Path(entry[0]).expanduser().resolve())
        strength = float(entry[1]) if len(entry) > 1 else 1.0
        out.append(LoraPathStrengthAndSDOps(lora_path, strength, LTXV_LORA_COMFY_RENAMING_MAP))
    return out


def build_guiders_for_mode(args: argparse.Namespace, params, use_low_memory_guidance: bool) -> tuple[MultiModalGuiderParams, MultiModalGuiderParams]:
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
        return target_path

    with tempfile.TemporaryDirectory(prefix="ltx_target_gt_") as tmp_dir:
        first_frame_png = str(Path(tmp_dir) / "first_frame.png")
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
    tiling = TilingConfig.default()

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
            return ((frames_bcfhw[0] + 1.0) / 2.0).clamp(0.0, 1.0).permute(1, 0, 2, 3).contiguous()

        needed_latent_t = min(latent.shape[2], (max_frames // 8) + 2)
        latent_local = latent[:, :, :needed_latent_t, :, :]

        if tiling_config is not None:
            for frames in video_decoder.tiled_decode(latent_local, tiling_config, generator=generator):
                yield _to_frames(frames)
        else:
            yield _to_frames(video_decoder(latent_local, generator=generator))

    _retake_module._encode_video_for_retake = _cached_video_encode
    _retake_module._encode_audio_for_retake = _forced_audio_encode
    _retake_module.vae_decode_video = _float_decode_video
    pipeline.model_ledger.video_encoder = lambda: None
    pipeline.model_ledger.audio_encoder = lambda: None
    _retake_module.euler_denoising_loop = _late_step_grad_euler_loop

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
        with torch.no_grad():
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

        with torch.no_grad():
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
