#!/usr/bin/env python3
"""Generate missing source audio before running Qwen-guided editing.

This is a small preprocessing utility for the reviewer-facing no-audio case:
given a source video with no audio stream, use LTX RetakePipeline to generate
audio conditioned on the whole source video, then mux that generated waveform
onto the source frames. The resulting video can be passed to
editing/optimize_qwen_vl.py without changing the main Qwen entrypoint.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
from fractions import Fraction
from pathlib import Path

import av
import torch
import torchaudio

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "src"))

from motion_opt.core import (  # noqa: E402
    _parse_loras,
    align_waveform_length,
    build_guiders_for_mode,
    compute_target_shape,
    decode_audio_from_file,
    save_audio_wav,
    write_temp_video_with_audio,
)
from motion_opt.models import build_retake_pipeline, resolve_quantization_policy  # noqa: E402
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number  # noqa: E402
from ltx_core.types import Audio  # noqa: E402
from ltx_pipelines.utils.constants import detect_params  # noqa: E402
from ltx_pipelines.utils.media_io import encode_video  # noqa: E402

_CKPT_ROOT = "${CKPT_ROOT}"
DEFAULT_CHECKPOINT = f"{_CKPT_ROOT}/ltx-2.3-22b-dev.safetensors"
DEFAULT_GEMMA_ROOT = "${GEMMA_ROOT}"

log = logging.getLogger(__name__)


def _env_snapshot() -> dict[str, str]:
    keys = [
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
        "SLURM_SUBMIT_DIR",
        "CUDA_VISIBLE_DEVICES",
        "PYTORCH_ALLOC_CONF",
        "PYTORCH_CUDA_ALLOC_CONF",
    ]
    return {key: os.environ[key] for key in keys if key in os.environ}


def _resolve_prompt(args: argparse.Namespace) -> str:
    return (args.audio_prompt or args.static_prompt or args.edit_prompt or "").strip()


def _normalize_waveform(audio: Audio, *, target_sr: int, target_samples: int) -> torch.Tensor:
    wave = audio.waveform.detach().float().cpu()
    if wave.dim() == 3 and wave.shape[0] == 1:
        wave = wave.squeeze(0)
    elif wave.dim() == 1:
        wave = wave.unsqueeze(0)
    elif wave.dim() != 2:
        raise RuntimeError(f"Unexpected generated audio shape: {tuple(wave.shape)}")

    if audio.sampling_rate != target_sr:
        wave = torchaudio.functional.resample(
            wave,
            orig_freq=audio.sampling_rate,
            new_freq=target_sr,
        )

    wave = align_waveform_length(wave, target_samples)
    return wave.clamp(-1.0, 1.0).contiguous()


def _write_path_file(output_dir: Path, path: Path) -> None:
    (output_dir / "prepared_source_path.txt").write_text(str(path), encoding="utf-8")


def _write_manifest(output_dir: Path, payload: dict) -> None:
    with (output_dir / "audio_bootstrap_config.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _write_video_only_conditioning_clip(
    *,
    src_video_path: str,
    target_frames: int,
    target_height: int,
    target_width: int,
    fps: float,
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

        if last_frame is None:
            raise RuntimeError(f"No video frames decoded from {src_video_path}")

        while frame_idx < target_frames:
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
    finally:
        src.close()
        dst.close()


@torch.inference_mode()
def run(args: argparse.Namespace) -> Path:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    src_video = Path(args.src_video).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    height, width, num_frames, frame_rate = compute_target_shape(
        str(src_video),
        args.height,
        args.width,
        args.num_frames,
        args.frame_rate,
    )
    duration = num_frames / frame_rate
    target_samples = int(duration * args.audio_sr)

    source_audio = decode_audio_from_file(str(src_video), torch.device("cpu"), max_duration=duration)
    source_had_audio = source_audio is not None
    if source_had_audio and not args.force_generate_audio:
        log.info("Source already has audio; leaving source unchanged: %s", src_video)
        _write_path_file(output_dir, src_video)
        _write_manifest(
            output_dir,
            {
                "args": vars(args),
                "argv": sys.argv,
                "env": _env_snapshot(),
                "source_had_audio": True,
                "generated_audio": False,
                "prepared_source_video": str(src_video),
                "target_shape": {
                    "height": height,
                    "width": width,
                    "num_frames": num_frames,
                    "frame_rate": frame_rate,
                    "duration": duration,
                },
            },
        )
        return src_video
    del source_audio

    prompt = _resolve_prompt(args)
    if not prompt:
        raise ValueError("Provide --audio-prompt, --static-prompt, or --edit-prompt for audio generation.")

    conditioning_video = output_dir / "conditioning_video_no_audio.mp4"
    _write_video_only_conditioning_clip(
        src_video_path=str(src_video),
        target_frames=num_frames,
        target_height=height,
        target_width=width,
        fps=frame_rate,
        output_path=str(conditioning_video),
    )
    log.info("Prepared video-only conditioning clip -> %s", conditioning_video)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    quant_policy = resolve_quantization_policy(args.quantization)
    params = detect_params(args.checkpoint_path)
    video_guider_params, audio_guider_params = build_guiders_for_mode(
        args=args,
        params=params,
        use_low_memory_guidance=args.low_memory_guidance,
    )

    log.info("Generating source audio conditioned on full video:")
    log.info("  source : %s", conditioning_video)
    log.info("  prompt : %s", prompt)
    log.info("  shape  : %dx%d, %d frames @ %.3f fps (%.2fs)", width, height, num_frames, frame_rate, duration)

    pipeline = build_retake_pipeline(
        checkpoint_path=args.checkpoint_path,
        gemma_root=args.gemma_root,
        loras=_parse_loras(args.loras),
        device=device,
        quant_policy=quant_policy,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    tiling = TilingConfig.default()
    video_iter, generated_audio = pipeline(
        video_path=str(conditioning_video),
        prompt=prompt,
        start_time=0.0,
        end_time=duration,
        seed=args.seed,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.num_inference_steps,
        video_guider_params=video_guider_params,
        audio_guider_params=audio_guider_params,
        regenerate_video=False,
        regenerate_audio=True,
        enhance_prompt=args.enhance_prompt,
        tiling_config=tiling,
    )

    generated_wave = _normalize_waveform(
        generated_audio,
        target_sr=args.audio_sr,
        target_samples=target_samples,
    )

    generated_audio_wav = output_dir / "generated_source_audio.wav"
    save_audio_wav(generated_wave, args.audio_sr, str(generated_audio_wav))
    log.info("Saved generated source audio -> %s", generated_audio_wav)

    ltx_preview_video: Path | None = None
    if args.save_ltx_preview:
        preview_audio = generated_wave
        if preview_audio.shape[0] == 1:
            preview_audio = preview_audio.expand(2, -1).contiguous()
        elif preview_audio.shape[0] > 2:
            preview_audio = preview_audio[:2].contiguous()
        ltx_preview_video = output_dir / "ltx_audio_generation_preview.mp4"
        encode_video(
            video=video_iter,
            fps=int(round(frame_rate)),
            audio=Audio(waveform=preview_audio, sampling_rate=args.audio_sr),
            output_path=str(ltx_preview_video),
            video_chunks_number=get_video_chunks_number(num_frames, tiling),
        )
        log.info("Saved optional LTX preview video -> %s", ltx_preview_video)

    prepared_source_video = output_dir / args.output_video_name
    write_temp_video_with_audio(
        src_video_path=str(conditioning_video),
        target_frames=num_frames,
        target_height=height,
        target_width=width,
        fps=frame_rate,
        waveform=generated_wave,
        sr=args.audio_sr,
        output_path=str(prepared_source_video),
    )
    log.info("Saved Qwen-ready source video -> %s", prepared_source_video)

    _write_path_file(output_dir, prepared_source_video)
    _write_manifest(
        output_dir,
        {
            "args": vars(args),
            "argv": sys.argv,
            "env": _env_snapshot(),
            "source_had_audio": source_had_audio,
            "generated_audio": True,
            "audio_prompt": prompt,
            "conditioning_video": str(conditioning_video),
            "prepared_source_video": str(prepared_source_video),
            "generated_audio_wav": str(generated_audio_wav),
            "ltx_preview_video": str(ltx_preview_video) if ltx_preview_video is not None else None,
            "target_shape": {
                "height": height,
                "width": width,
                "num_frames": num_frames,
                "frame_rate": frame_rate,
                "duration": duration,
            },
        },
    )

    del pipeline, generated_audio, video_iter
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    return prepared_source_video


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--src-video", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--edit-prompt", default="")
    parser.add_argument("--static-prompt", default="")
    parser.add_argument(
        "--audio-prompt",
        default="",
        help="Prompt for source-audio generation. Defaults to static prompt, then edit prompt.",
    )
    parser.add_argument("--negative-prompt", default="blurry, low quality, artifacts, distorted")
    parser.add_argument("--output-video-name", default="source_with_generated_audio.mp4")
    parser.add_argument("--force-generate-audio", action="store_true")
    parser.add_argument("--save-ltx-preview", action="store_true")

    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--frame-rate", type=float, default=None)
    parser.add_argument("--audio-sr", type=int, default=44100)

    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enhance-prompt", action="store_true")

    parser.add_argument("--cfg-scale", type=float, default=None)
    parser.add_argument("--audio-cfg-scale", type=float, default=None)
    parser.add_argument("--a2v-scale", type=float, default=None)
    parser.add_argument("--low-memory-guidance", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--quantization", default=None, choices=["fp8-cast", "fp8-scaled-mm"])
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--checkpoint-path", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--gemma-root", default=DEFAULT_GEMMA_ROOT)
    parser.add_argument(
        "--lora",
        dest="loras",
        nargs="+",
        metavar=("PATH", "STRENGTH"),
        action="append",
        default=[],
    )

    return parser


def main() -> None:
    prepared_path = run(build_parser().parse_args())
    print(prepared_path)


if __name__ == "__main__":
    main()
