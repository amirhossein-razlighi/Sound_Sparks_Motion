from __future__ import annotations

import argparse
import gc
import logging
from pathlib import Path

import torch
import torch.nn.functional as F

from .distributed import barrier, init_distributed_and_device, shutdown_distributed
from .helpers import compute_target_shape, save_audio_wav
from .loop import gradient_optimize_audio_latent
from .models import build_retake_pipeline, resolve_quantization_policy
from .runtime import build_retake_kwargs, get_guiders, prepare_retake_input_video, prepare_target_flow

from .core import (
    _parse_loras,
    build_guiders_for_mode,
    build_cached_source_latents,
    decode_audio_from_file,
    frames_rgb_uint8_to_chw_float,
    render_and_save_video_with_latent,
    render_with_injected_audio_latent,
    align_waveform_length,
    decode_video_frames_rgb,
    write_temp_video_with_audio,
    maybe_generate_target_video,
)
from ltx_pipelines.utils.constants import detect_params

log = logging.getLogger(__name__)

_CKPT_ROOT = "/project/def-amahdavi/amirrz/LTX-2/checkpoints"
DEFAULT_CHECKPOINT = f"{_CKPT_ROOT}/ltx-2.3-22b-dev.safetensors"
DEFAULT_GEMMA_ROOT = "/project/def-amahdavi/amirrz/HF/models/gemma-3-12b-it-qat-q4_0-unquantized/"


def prepare_target_motion_video(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s %(message)s", datefmt="%H:%M:%S")

    rank, _, device = init_distributed_and_device()
    is_main = rank == 0
    if not is_main:
        logging.getLogger().setLevel(logging.WARNING)

    output_dir = Path(args.output_dir).expanduser().resolve()
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    barrier()

    loras = _parse_loras(args.loras)

    ti2v_quant_name = args.ti2v_quantization if args.ti2v_quantization is not None else args.quantization
    ti2v_quant_policy = resolve_quantization_policy(ti2v_quant_name)

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

    barrier()
    shutdown_distributed()


def run(args: argparse.Namespace) -> None:
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
    barrier()

    loras = _parse_loras(args.loras)
    if loras:
        log.info("LoRAs: %s", [(l.path, l.strength) for l in loras])

    ti2v_quant_name = args.ti2v_quantization if args.ti2v_quantization is not None else args.quantization
    retake_quant_name = args.retake_quantization if args.retake_quantization is not None else args.quantization
    ti2v_quant_policy = resolve_quantization_policy(ti2v_quant_name)
    retake_quant_policy = resolve_quantization_policy(retake_quant_name)

    height, width, num_frames, frame_rate = compute_target_shape(
        args.src_video,
        args.height,
        args.width,
        args.num_frames,
        args.frame_rate,
    )
    duration = num_frames / frame_rate

    retake_input_video = prepare_retake_input_video(
        args=args,
        is_main=is_main,
        output_dir=output_dir,
        height=height,
        width=width,
        num_frames=num_frames,
        frame_rate=frame_rate,
    )

    video_guider_params, audio_guider_params = get_guiders(args, args.checkpoint_path)
    target_video_path, resize_to, target_flows, raft_model, raft_transforms, roi_frame_masks, eval_sample_start = prepare_target_flow(
        args=args,
        device=device,
        output_dir=output_dir,
        is_main=is_main,
        height=height,
        width=width,
        num_frames=num_frames,
        frame_rate=frame_rate,
        video_guider_params=video_guider_params,
        audio_guider_params=audio_guider_params,
        quant_policy=ti2v_quant_policy,
        loras=loras,
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
        source_frames_lpips = source_frames_t * 2.0 - 1.0

    pipeline = build_retake_pipeline(
        checkpoint_path=args.checkpoint_path,
        gemma_root=args.gemma_root,
        loras=loras,
        device=device,
        quant_policy=retake_quant_policy,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    if world_size > 1 and args.distributed_shard_transformer:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        cached_transformer = pipeline.model_ledger.transformer()
        param_dtypes_before = {p.dtype for p in cached_transformer.parameters() if torch.is_floating_point(p)}
        if len(param_dtypes_before) > 1:
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
                "FSDP init OOM while sharding transformer. Use lower-memory Retake weights "
                "(recommended: --retake-quantization fp8-cast), reduce num_frames/resolution, "
                "or increase number of GPUs."
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

    retake_kwargs = build_retake_kwargs(
        args=args,
        frame_rate=frame_rate,
        duration=duration,
        video_guider_params=video_guider_params,
        audio_guider_params=audio_guider_params,
    )

    base_audio_latent_fp32 = base_audio_latent.float().detach()

    best = gradient_optimize_audio_latent(
        args=args,
        is_main=is_main,
        world_size=world_size,
        output_dir=output_dir,
        base_audio_latent=base_audio_latent,
        base_audio_latent_fp32=base_audio_latent_fp32,
        cached_video_latent=cached_video_latent,
        retake_input_video=str(retake_input_video),
        pipeline=pipeline,
        retake_kwargs=retake_kwargs,
        resize_to=resize_to,
        target_flows=target_flows,
        raft_model=raft_model,
        raft_transforms=raft_transforms,
        render_with_injected_audio_latent=render_with_injected_audio_latent,
        roi_frame_masks=roi_frame_masks,
        source_frames_lpips=source_frames_lpips,
        lpips_model=lpips_model,
        eval_sample_start=eval_sample_start,
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
        torch.save(base_audio_latent.detach().cpu(), output_dir / "base_audio_latent.pt")
        torch.save(best_latent.detach().cpu(), output_dir / "best_audio_latent.pt")

        final_retake_kwargs = dict(retake_kwargs)
        if args.final_retake_num_inference_steps is not None:
            final_retake_kwargs["num_inference_steps"] = args.final_retake_num_inference_steps

        params = detect_params(args.checkpoint_path)
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

        if args.save_final_videos and world_size == 1:
            orig_height, orig_width, _, _ = compute_target_shape(
                args.src_video,
                height_override=None,
                width_override=None,
                num_frames_override=args.num_frames,
                frame_rate_override=args.frame_rate,
            )

            final_retake_input_video = retake_input_video
            final_cached_video_latent = cached_video_latent

            if orig_height != height or orig_width != width:
                log.info("Restoring original source dimensions (%dx%d) for final video renders...", orig_width, orig_height)
                final_retake_input_video = output_dir / "retake_input_prepared_hires.mp4"

                src_audio_for_input = decode_audio_from_file(args.src_video, torch.device("cpu"), max_duration=duration)
                src_wave = None
                if src_audio_for_input is not None:
                    src_wave = src_audio_for_input.waveform.squeeze(0).float()
                    if src_audio_for_input.sampling_rate != args.audio_sr:
                        import torchaudio

                        src_wave = torchaudio.functional.resample(
                            src_wave,
                            orig_freq=src_audio_for_input.sampling_rate,
                            new_freq=args.audio_sr,
                        )
                    src_wave = align_waveform_length(src_wave, int(duration * args.audio_sr))

                write_temp_video_with_audio(
                    src_video_path=args.src_video,
                    target_frames=num_frames,
                    target_height=orig_height,
                    target_width=orig_width,
                    fps=frame_rate,
                    waveform=src_wave,
                    sr=args.audio_sr,
                    output_path=str(final_retake_input_video),
                )
                final_cached_video_latent, _, _ = build_cached_source_latents(
                    pipeline=pipeline,
                    src_video=str(final_retake_input_video),
                    height=orig_height,
                    width=orig_width,
                    num_frames=num_frames,
                    audio_sr=args.audio_sr,
                    device=device,
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            render_and_save_video_with_latent(
                pipeline=pipeline,
                src_video=str(final_retake_input_video),
                injected_audio_latent=best_latent,
                cached_video_latent=final_cached_video_latent,
                retake_kwargs=final_retake_kwargs,
                output_path=output_dir / "best_optimized_video.mp4",
                fps=frame_rate,
                num_frames=num_frames,
                audio_sr=waveform_sr,
                audio_opt_last_steps=final_audio_opt_last_steps,
            )
            render_and_save_video_with_latent(
                pipeline=pipeline,
                src_video=str(final_retake_input_video),
                injected_audio_latent=base_audio_latent,
                cached_video_latent=final_cached_video_latent,
                retake_kwargs=final_retake_kwargs,
                output_path=output_dir / "baseline_unoptimized_video.mp4",
                fps=frame_rate,
                num_frames=num_frames,
                audio_sr=waveform_sr,
                audio_opt_last_steps=final_audio_opt_last_steps,
            )

            if args.transfer_prompt:
                transfer_kwargs = dict(final_retake_kwargs)
                transfer_kwargs["prompt"] = args.transfer_prompt
                render_and_save_video_with_latent(
                    pipeline=pipeline,
                    src_video=str(final_retake_input_video),
                    injected_audio_latent=best_latent,
                    cached_video_latent=final_cached_video_latent,
                    retake_kwargs=transfer_kwargs,
                    output_path=output_dir / "transfer_prompt_with_best_latent.mp4",
                    fps=frame_rate,
                    num_frames=num_frames,
                    audio_sr=waveform_sr,
                    audio_opt_last_steps=final_audio_opt_last_steps,
                )
        elif args.save_final_videos and world_size > 1:
            log.warning("Skipping final video rendering in distributed mode. Re-run single-GPU with saved best latent to render.")

        src_audio = decode_audio_from_file(args.src_video, device, max_duration=duration)
        if src_audio is not None:
            wave = src_audio.waveform.squeeze(0).float()
            if src_audio.sampling_rate != args.audio_sr:
                import torchaudio

                wave = torchaudio.functional.resample(wave, orig_freq=src_audio.sampling_rate, new_freq=args.audio_sr)
            wave = align_waveform_length(wave, int(duration * args.audio_sr))
            save_audio_wav(wave, args.audio_sr, str(output_dir / "source_audio_used.wav"))

        try:
            from ltx_core.model.audio_vae import decode_audio as vae_decode_audio

            log.info("Decoding optimized audio latents to .wav files...")
            audio_decoder = pipeline.model_ledger.audio_decoder().to(device)
            vocoder = pipeline.model_ledger.vocoder().to(device)

            with torch.no_grad():
                decoder_dtype = next(audio_decoder.parameters()).dtype if hasattr(audio_decoder, "parameters") else torch.float32
                best_audio_decoded = vae_decode_audio(
                    best_latent.to(device).to(decoder_dtype),
                    audio_decoder,
                    vocoder,
                )
                save_audio_wav(
                    best_audio_decoded.waveform.squeeze(0),
                    best_audio_decoded.sampling_rate,
                    str(output_dir / "best_optimized_audio.wav"),
                )

                base_audio_decoded = vae_decode_audio(
                    base_audio_latent.to(device).to(decoder_dtype),
                    audio_decoder,
                    vocoder,
                )
                save_audio_wav(
                    base_audio_decoded.waveform.squeeze(0),
                    base_audio_decoded.sampling_rate,
                    str(output_dir / "baseline_unoptimized_audio.wav"),
                )

            del audio_decoder, vocoder
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            log.warning("Could not decode and save audio latents: %s", e)

        log.info("Done. Best total loss: %.6f", best["loss"])
        log.info("Saved outputs in: %s", output_dir)

    barrier()
    shutdown_distributed()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--src-video", required=True, help="Source video used for retake and latent optimization.")
    p.add_argument("--edit-prompt", required=True, help="Prompt used during optimization retake generation.")
    p.add_argument("--output-dir", required=True, help="Directory where outputs/logs are written.")

    p.add_argument("--target-video", default=None, help="Optional existing target-motion video.")
    p.add_argument(
        "--target-prompt",
        default=None,
        help=(
            "If --target-video is not provided, generate target video once with TI2V using "
            "this prompt and source first frame."
        ),
    )
    p.add_argument("--regenerate-target", action="store_true", help="Regenerate target video even if it already exists.")
    p.add_argument("--prepare-target-only", action="store_true", help="Generate/reuse target video and exit.")
    p.add_argument("--transfer-prompt", default=None, help="Optional prompt for transfer test with optimized latent.")

    p.add_argument("--flow-weight", type=float, default=1.0)
    p.add_argument("--mag-curve-weight", type=float, default=0.25)
    p.add_argument("--lpips-weight", type=float, default=0.1)
    p.add_argument("--lpips-max-frames", type=int, default=8)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--latent-reg-weight", type=float, default=0.05)
    p.add_argument("--flow-width", type=int, default=512)
    p.add_argument("--flow-height", type=int, default=320)
    p.add_argument("--max-eval-frames", type=int, default=33)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--eval-start-frame", type=int, default=-1)
    p.add_argument("--roi-mask-video", type=str, default=None)
    p.add_argument("--roi-mask-threshold", type=float, default=0.5)

    p.add_argument("--iterations", type=int, default=30)
    p.add_argument("--lr", type=float, default=0.015)
    p.add_argument("--grad-clip", type=float, default=0.5)
    p.add_argument("--audio-opt-last-steps", type=int, default=6)
    p.add_argument("--final-audio-opt-last-steps", type=int, default=None)
    p.add_argument("--distributed-shard-transformer", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--save-final-videos", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--raft-model", type=str, default="raft_large", choices=["raft_large", "raft_small"])
    p.add_argument("--raft-weights-path", type=str, default=None)

    p.add_argument("--negative-prompt", default="")
    p.add_argument("--enhance-prompt", action="store_true")
    p.add_argument("--num-inference-steps", type=int, default=40)
    p.add_argument("--ti2v-num-inference-steps", type=int, default=None)
    p.add_argument("--retake-num-inference-steps", type=int, default=None)
    p.add_argument("--final-retake-num-inference-steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--retake-start-frames", type=int, default=1)

    p.add_argument("--height", type=int, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--num-frames", type=int, default=None)
    p.add_argument("--frame-rate", type=float, default=None)

    p.add_argument("--audio-sr", type=int, default=44100)

    p.add_argument("--checkpoint-path", default=DEFAULT_CHECKPOINT)
    p.add_argument("--gemma-root", default=DEFAULT_GEMMA_ROOT)
    p.add_argument("--quantization", default=None, choices=["fp8-cast", "fp8-scaled-mm"])
    p.add_argument("--ti2v-quantization", default=None, choices=["fp8-cast", "fp8-scaled-mm"])
    p.add_argument("--retake-quantization", default=None, choices=["fp8-cast", "fp8-scaled-mm"])
    p.add_argument(
        "--lora",
        dest="loras",
        nargs="+",
        metavar=("PATH", "STRENGTH"),
        action="append",
        default=[],
        help="LoRA path and optional strength. Can be passed multiple times.",
    )

    p.add_argument("--cfg-scale", type=float, default=None)
    p.add_argument("--audio-cfg-scale", type=float, default=None)
    p.add_argument("--a2v-scale", type=float, default=None)
    p.add_argument("--low-memory-guidance", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ti2v-low-memory-guidance", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)

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
    run(args)


if __name__ == "__main__":
    main()
