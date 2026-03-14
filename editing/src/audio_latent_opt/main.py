from __future__ import annotations

import argparse
import gc
import logging
from pathlib import Path

import torch

from .distributed import barrier, init_distributed_and_device, shutdown_distributed
from .helpers import compute_target_shape, save_audio_wav
from .loop import gradient_optimize_audio_latent
from .models import build_retake_pipeline, resolve_quantization_policy
from .runtime import build_retake_kwargs, get_guiders, prepare_retake_input_video, prepare_target_flow

# Reuse stable functions from legacy script for rendering and parser parity.
from editing.optimize_audio_embedding import (
    _parse_loras,
    build_cached_source_latents,
    build_parser as build_legacy_parser,
    decode_audio_from_file,
    render_and_save_video_with_latent,
    render_with_injected_audio_latent,
    align_waveform_length,
)

log = logging.getLogger(__name__)


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
    _, resize_to, target_flows, raft_model, raft_transforms = prepare_target_flow(
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
        velocity_model = getattr(cached_transformer, "velocity_model", None)
        if args.gradient_checkpointing and velocity_model is not None and hasattr(velocity_model, "set_gradient_checkpointing"):
            velocity_model.set_gradient_checkpointing(True)
        cached_transformer.requires_grad_(False)
        sharded_transformer = FSDP(cached_transformer, use_orig_params=True)
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
    )

    best_latent = best["latent"].to(dtype=base_audio_latent.dtype)

    if is_main:
        torch.save(
            {
                "best_loss": best["loss"],
                "best_flow_mse": best["flow_mse"],
                "best_mag_mse": best["mag_mse"],
                "world_size": world_size,
            },
            output_dir / "best_latent_params.pt",
        )
        torch.save(base_audio_latent.detach().cpu(), output_dir / "base_audio_latent.pt")
        torch.save(best_latent.detach().cpu(), output_dir / "best_audio_latent.pt")

        final_retake_kwargs = dict(retake_kwargs)
        if args.final_retake_num_inference_steps is not None:
            final_retake_kwargs["num_inference_steps"] = args.final_retake_num_inference_steps

        if args.save_final_videos and world_size == 1:
            render_and_save_video_with_latent(
                pipeline=pipeline,
                src_video=str(retake_input_video),
                injected_audio_latent=best_latent,
                cached_video_latent=cached_video_latent,
                retake_kwargs=final_retake_kwargs,
                output_path=output_dir / "best_optimized_video.mp4",
                fps=frame_rate,
                num_frames=num_frames,
                audio_sr=waveform_sr,
            )
            render_and_save_video_with_latent(
                pipeline=pipeline,
                src_video=str(retake_input_video),
                injected_audio_latent=base_audio_latent,
                cached_video_latent=cached_video_latent,
                retake_kwargs=final_retake_kwargs,
                output_path=output_dir / "baseline_unoptimized_video.mp4",
                fps=frame_rate,
                num_frames=num_frames,
                audio_sr=waveform_sr,
            )

            if args.transfer_prompt:
                transfer_kwargs = dict(final_retake_kwargs)
                transfer_kwargs["prompt"] = args.transfer_prompt
                render_and_save_video_with_latent(
                    pipeline=pipeline,
                    src_video=str(retake_input_video),
                    injected_audio_latent=best_latent,
                    cached_video_latent=cached_video_latent,
                    retake_kwargs=transfer_kwargs,
                    output_path=output_dir / "transfer_prompt_with_best_latent.mp4",
                    fps=frame_rate,
                    num_frames=num_frames,
                    audio_sr=waveform_sr,
                )

        src_audio = decode_audio_from_file(args.src_video, device, max_duration=duration)
        if src_audio is not None:
            wave = src_audio.waveform.squeeze(0).float()
            if src_audio.sampling_rate != args.audio_sr:
                import torchaudio

                wave = torchaudio.functional.resample(wave, orig_freq=src_audio.sampling_rate, new_freq=args.audio_sr)
            wave = align_waveform_length(wave, int(duration * args.audio_sr))
            save_audio_wav(wave, args.audio_sr, str(output_dir / "source_audio_used.wav"))

        log.info("Done. Best total loss: %.6f", best["loss"])
        log.info("Saved outputs in: %s", output_dir)

    barrier()
    shutdown_distributed()


def build_parser() -> argparse.ArgumentParser:
    # Reuse the legacy parser to preserve identical CLI surface.
    return build_legacy_parser()


def main() -> None:
    args = build_parser().parse_args()
    if args.ti2v_num_inference_steps is None:
        args.ti2v_num_inference_steps = args.num_inference_steps
    if args.retake_num_inference_steps is None:
        args.retake_num_inference_steps = args.num_inference_steps
    run(args)


if __name__ == "__main__":
    main()
