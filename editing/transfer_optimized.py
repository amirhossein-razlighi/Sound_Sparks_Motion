#!/usr/bin/env python3
"""Transfer optimized latents from one video to a new target video.

Takes the best_audio_latent_{mode}.pt and/or best_text_delta_{mode}.pt saved
by optimize_qwen_vl.py and applies them to a completely different video.

Hypothesis: if the optimization found a genuine motion direction in latent
space (e.g., "jumping" or "yawning"), it should transfer across subjects.

Example
-------
    # 1. Optimize on a dog-yawning video:
    python editing/optimize_qwen_vl.py --src-video dog.mp4 --edit-prompt "A dog yawning" ...

    # 2. Transfer the result to a cat video:
    python editing/transfer_optimized.py \\
        --target-video cat.mp4 \\
        --opt-dir results/QwenVL/a_dog_yawning/both/mode_both \\
        --mode both \\
        --edit-prompt "A cat yawning" \\
        --output-dir results/transfer/dog_to_cat_yawn
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_CKPT_ROOT = "${CKPT_ROOT}"
DEFAULT_CHECKPOINT = f"{_CKPT_ROOT}/ltx-2.3-22b-dev.safetensors"
DEFAULT_GEMMA_ROOT = "${GEMMA_ROOT}/"
DEFAULT_QWEN_ROOT = "${QWEN_ROOT}"

sys.path.insert(0, str(Path(__file__).parent / "src"))

from motion_opt.core import (
    _parse_loras,
    build_cached_source_latents,
    build_guiders_for_mode,
    compute_target_shape,
    align_waveform_length,
)
from motion_opt.models import build_retake_pipeline, resolve_quantization_policy
from motion_opt.multimodal_loop import pre_encode_base_contexts
from motion_opt.runtime import build_retake_kwargs, prepare_retake_input_video
from motion_opt.clip_loss import (
    build_clip_model,
    compute_clip_dual_prompt_frame_similarities,
    encode_text_for_clip,
)
from motion_opt.metrics import decode_video_frames_rgb

import ltx_pipelines.retake as _retake_module
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.types import Audio
from ltx_core.text_encoders.gemma.embeddings_processor import EmbeddingsProcessorOutput
from ltx_pipelines.utils.constants import detect_params
from ltx_pipelines.utils.media_io import decode_audio_from_file, encode_video

import torchaudio

log = logging.getLogger(__name__)

try:
    import wandb as _wandb
except ImportError:
    _wandb = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _interpolate_audio_latent(latent: torch.Tensor, target_t: int) -> torch.Tensor:
    """Interpolate audio latent along the time axis to match target_t."""
    t_src = latent.shape[2]
    if t_src == target_t:
        return latent
    x = latent.permute(0, 1, 3, 2)  # [1, C, F, T]
    x = F.interpolate(x, size=(x.shape[2], target_t), mode="bilinear", align_corners=False)
    return x.permute(0, 1, 3, 2)


def _build_out_audio(src_audio_path: str, pipeline, waveform_sr: int, duration: float) -> Audio | None:
    src_audio = decode_audio_from_file(src_audio_path, pipeline.device, max_duration=duration)
    if src_audio is None:
        return None
    wave = src_audio.waveform.squeeze(0).float()
    if src_audio.sampling_rate != waveform_sr:
        wave = torchaudio.functional.resample(wave, orig_freq=src_audio.sampling_rate, new_freq=waveform_sr)
    wave = align_waveform_length(wave, int(duration * waveform_sr))
    if wave.shape[0] == 1:
        wave = wave.expand(2, -1).contiguous()
    elif wave.shape[0] > 2:
        wave = wave[:2].contiguous()
    return Audio(waveform=wave.cpu(), sampling_rate=waveform_sr)


def _render_video(
    pipeline,
    retake_input_video: str,
    cached_video_latent: torch.Tensor,
    audio_for_render: torch.Tensor,
    pos_context: EmbeddingsProcessorOutput,
    base_neg_context: EmbeddingsProcessorOutput,
    retake_kwargs: dict,
    out_audio: Audio | None,
    out_path: Path,
    num_frames: int,
    frame_rate: float,
) -> None:
    def _cached_video(video_encoder, video_path, output_shape, dtype, device):  # noqa: ARG001
        return cached_video_latent

    def _injected_audio(audio_encoder, waveform, waveform_sr, output_shape, dtype):  # noqa: ARG001
        return audio_for_render

    def _patched_encode_prompts(prompts, model_ledger, **kwargs):  # noqa: ARG001
        return [pos_context, base_neg_context]

    orig_video = _retake_module._encode_video_for_retake
    orig_audio = _retake_module._encode_audio_for_retake
    orig_prompts = _retake_module.encode_prompts
    _retake_module._encode_video_for_retake = _cached_video
    _retake_module._encode_audio_for_retake = _injected_audio
    _retake_module.encode_prompts = _patched_encode_prompts
    try:
        with torch.no_grad():
            video_iter, _ = pipeline(video_path=retake_input_video, **retake_kwargs)
        encode_video(
            video=video_iter,
            fps=int(round(frame_rate)),
            audio=out_audio,
            output_path=str(out_path),
            video_chunks_number=get_video_chunks_number(num_frames, TilingConfig.default()),
        )
        log.info("Saved: %s", out_path)
    finally:
        _retake_module._encode_video_for_retake = orig_video
        _retake_module._encode_audio_for_retake = orig_audio
        _retake_module.encode_prompts = orig_prompts


def _video_to_chw(video_path: Path, max_frames: int, device: torch.device) -> torch.Tensor:
    frames = decode_video_frames_rgb(str(video_path), max_frames=max_frames or None, frame_stride=1, resize_to=None)
    return torch.stack([
        torch.from_numpy(f).permute(2, 0, 1).float().div(255.0) for f in frames
    ]).to(device)


def _score_with_qwen(
    video_path: Path,
    edit_prompt: str,
    qwen_model_name: str,
    device: torch.device,
    max_frames: int = 16,
    img_size: int = 224,
) -> dict[str, float]:
    """Run Qwen2.5-VL inference (no grad) on the transfer result and return scores."""
    from motion_opt.qwen_loss import (
        build_qwen_model,
        build_qwen_rubric_inputs,
        compute_qwen_video_loss,
    )

    log.info("Scoring transfer result with Qwen2.5-VL...")
    qwen_model, qwen_processor = build_qwen_model(qwen_model_name, device=device, gradient_checkpointing=False)

    qwen_num_frames = max_frames if max_frames % 2 == 0 else max_frames + 1
    cached_inputs, yes_id, no_id = build_qwen_rubric_inputs(
        processor=qwen_processor,
        edit_prompt=edit_prompt,
        num_frames=qwen_num_frames,
        img_size=img_size,
        device=device,
    )

    frames = _video_to_chw(video_path, max_frames=qwen_num_frames, device=device)

    with torch.no_grad():
        loss, details = compute_qwen_video_loss(
            frames_chw=frames,
            qwen_model=qwen_model,
            cached_inputs=cached_inputs,
            yes_token_id=yes_id,
            no_token_id=no_id,
            max_frames=qwen_num_frames,
            img_size=img_size,
            backward=False,
            return_details=True,
            sample_mode="linspace",
        )

    scores = {"qwen_nll": float(loss.item()), "qwen_yes_prob": float((-loss).exp().item())}
    for item in details:
        scores[f"{item['name']}_yes_prob"] = float(item["yes_prob"])
        scores[f"{item['name']}_nll"] = float(item["nll"])

    del qwen_model, qwen_processor, frames, cached_inputs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return scores


def _write_clip_diagnostics(
    baseline_path: Path,
    transfer_path: Path,
    edit_prompt: str,
    static_prompt: str,
    output_dir: Path,
    clip_model_name: str,
    max_frames: int,
    batch_size: int,
    device: torch.device,
    wandb_run,
) -> dict[str, float]:
    """Per-frame CLIP similarity for baseline vs. transfer. Returns summary stats."""
    log.info("Computing CLIP similarity diagnostics...")
    clip_model, clip_tokenizer = build_clip_model(clip_model_name, device)
    static_emb = encode_text_for_clip(static_prompt, clip_model, clip_tokenizer, device)
    edit_emb = encode_text_for_clip(edit_prompt, clip_model, clip_tokenizer, device)

    diag_dir = output_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    csv_path = diag_dir / "clip_similarity.csv"

    rows = []
    series = {}
    summary: dict[str, float] = {}

    for label, vpath in [("baseline", baseline_path), ("transfer", transfer_path)]:
        if not vpath.exists():
            log.warning("Skipping CLIP diagnostics for %s: file not found", label)
            continue
        frames = _video_to_chw(vpath, max_frames=max_frames or 0, device=device)
        with torch.no_grad():
            sc = compute_clip_dual_prompt_frame_similarities(frames, static_emb, edit_emb, clip_model, batch_size=batch_size)
        series[label] = sc
        for fi, (ss, es) in enumerate(zip(sc["static"], sc["edit"])):
            rows.append({"video": label, "frame": fi, "static_sim": ss, "edit_sim": es, "edit_minus_static": es - ss})
        summary[f"{label}_mean_edit_sim"] = float(sum(sc["edit"]) / len(sc["edit"])) if sc["edit"] else 0.0
        summary[f"{label}_mean_edit_minus_static"] = float(
            sum(e - s for e, s in zip(sc["edit"], sc["static"])) / len(sc["edit"])
        ) if sc["edit"] else 0.0
        del frames

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video", "frame", "static_sim", "edit_sim", "edit_minus_static"])
        writer.writeheader()
        writer.writerows(rows)

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        colors = {"baseline": "#888888", "transfer": "#2196F3"}
        for label, sc in series.items():
            x = list(range(len(sc["static"])))
            c = colors.get(label, "black")
            axes[0].plot(x, sc["edit"], color=c, linewidth=1.5, label=f"{label}: edit")
            axes[0].plot(x, sc["static"], color=c, linestyle="--", alpha=0.5, label=f"{label}: static")
            diff = [e - s for e, s in zip(sc["edit"], sc["static"])]
            axes[1].plot(x, diff, color=c, linewidth=1.5, label=label)

        axes[0].set_title("Per-frame CLIP similarity")
        axes[0].set_xlabel("Frame")
        axes[0].set_ylabel("CLIP cosine similarity")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].set_title("edit_sim − static_sim  (higher = edit transferred)")
        axes[1].set_xlabel("Frame")
        axes[1].set_ylabel("Δ similarity")
        axes[1].axhline(0, color="red", linestyle="--", alpha=0.4)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        plot_path = diag_dir / "clip_similarity.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        log.info("Saved CLIP diagnostics plot: %s", plot_path)

        if wandb_run is not None:
            try:
                wandb_run.log({"diagnostics/clip_similarity_plot": _wandb.Image(str(plot_path))})
            except Exception:
                pass
    except Exception:
        log.warning("Failed to plot CLIP diagnostics", exc_info=True)

    del clip_model, clip_tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    opt_dir = Path(args.opt_dir).expanduser().resolve()
    mode = args.mode

    # ---- Save run config ----
    config = {"args": vars(args), "env": {k: os.environ.get(k) for k in [
        "SLURM_JOB_ID", "SLURM_JOB_NAME", "CUDA_VISIBLE_DEVICES",
    ] if os.environ.get(k)}}
    (output_dir / "transfer_config.json").write_text(json.dumps(config, indent=2, sort_keys=True))

    # ---- W&B ----
    wandb_run = None
    if _wandb is not None and os.environ.get("WANDB_DISABLED", "").lower() not in {"1", "true"}:
        try:
            wandb_run = _wandb.init(
                project=os.environ.get("WANDB_PROJECT", "ltx-transfer"),
                name=f"transfer-{mode}-{Path(args.target_video).stem[:20]}",
                config=vars(args),
                dir=os.environ.get("WANDB_DIR"),
            )
        except Exception:
            log.warning("W&B init failed", exc_info=True)

    try:
        # ---- Load optimized tensors ----
        audio_latent_path = opt_dir / f"best_audio_latent_{mode}.pt"
        text_delta_path = opt_dir / f"best_text_delta_{mode}.pt"
        saved_audio_latent = None
        saved_text_delta = None

        if mode in ("audio", "both"):
            if not audio_latent_path.exists():
                raise FileNotFoundError(f"Audio latent not found: {audio_latent_path}")
            saved_audio_latent = torch.load(audio_latent_path, map_location="cpu")
            log.info("Loaded audio latent: shape=%s", tuple(saved_audio_latent.shape))

        if mode in ("text", "both"):
            if not text_delta_path.exists():
                raise FileNotFoundError(f"Text delta not found: {text_delta_path}")
            saved_text_delta = torch.load(text_delta_path, map_location="cpu")
            log.info("Loaded text delta: shape=%s", tuple(saved_text_delta.shape))

        # Also load source optimization score for comparison
        src_params_path = opt_dir / f"best_params_{mode}.pt"
        src_qwen_score = None
        if src_params_path.exists():
            src_params = torch.load(src_params_path, map_location="cpu")
            src_qwen_score = float(src_params.get("qwen_score", 0))
            log.info("Source optimization best qwen_score=%.4f", src_qwen_score)

        # ---- Resolve shape from target video ----
        height, width, num_frames, frame_rate = compute_target_shape(
            args.target_video, args.height, args.width, args.num_frames, args.frame_rate,
        )
        duration = num_frames / frame_rate
        log.info("Target: %dx%d, %d frames @ %.1f fps (%.2fs)", width, height, num_frames, frame_rate, duration)

        retake_quant = resolve_quantization_policy(args.quantization)
        args.src_video = args.target_video
        retake_input_video = prepare_retake_input_video(
            args=args, is_main=True, output_dir=output_dir,
            height=height, width=width, num_frames=num_frames, frame_rate=frame_rate,
        )

        # ---- Load pipeline ----
        params = detect_params(args.checkpoint_path)
        video_guider_params, audio_guider_params = build_guiders_for_mode(
            args=args, params=params, use_low_memory_guidance=False,
        )
        log.info("Loading RetakePipeline...")
        pipeline = build_retake_pipeline(
            checkpoint_path=args.checkpoint_path,
            gemma_root=args.gemma_root,
            loras=_parse_loras(args.loras),
            device=device,
            quant_policy=retake_quant,
            gradient_checkpointing=False,
        )

        # ---- Encode target video/audio ----
        cached_video_latent, base_audio_latent, waveform_sr = build_cached_source_latents(
            pipeline=pipeline, src_video=str(retake_input_video),
            height=height, width=width, num_frames=num_frames,
            audio_sr=args.audio_sr, device=device,
        )
        log.info("Target audio latent shape: %s", tuple(base_audio_latent.shape))

        # ---- Encode text context ----
        base_pos_context, base_neg_context = pre_encode_base_contexts(
            pipeline=pipeline, pos_prompt=args.edit_prompt,
            neg_prompt=args.negative_prompt, device=device,
        )

        # ---- Adapt audio latent to target T ----
        if saved_audio_latent is not None:
            target_t = base_audio_latent.shape[2]
            src_t = saved_audio_latent.shape[2]
            if src_t != target_t:
                log.info("Audio latent T mismatch: %d → %d, interpolating...", src_t, target_t)
                saved_audio_latent = _interpolate_audio_latent(saved_audio_latent, target_t)
            audio_for_render = saved_audio_latent.to(device=device, dtype=base_audio_latent.dtype)
        else:
            audio_for_render = base_audio_latent

        # ---- Adapt text delta to target seq len ----
        if saved_text_delta is not None:
            target_seq = base_pos_context.video_encoding.shape[1]
            src_seq = saved_text_delta.shape[1]
            if src_seq != target_seq:
                log.info("Text delta seq mismatch: %d → %d, interpolating...", src_seq, target_seq)
                delta = saved_text_delta.float().permute(0, 2, 1)
                delta = F.interpolate(delta, size=target_seq, mode="linear", align_corners=False)
                delta = delta.permute(0, 2, 1)
            else:
                delta = saved_text_delta.float()
            pos_context = EmbeddingsProcessorOutput(
                video_encoding=base_pos_context.video_encoding + delta.to(device=device, dtype=base_pos_context.video_encoding.dtype),
                audio_encoding=base_pos_context.audio_encoding,
                attention_mask=base_pos_context.attention_mask,
            )
            log.info("Applied text delta (shape %s)", tuple(delta.shape))
        else:
            pos_context = base_pos_context

        # ---- Build retake kwargs ----
        retake_kwargs = build_retake_kwargs(
            args=args, frame_rate=frame_rate, duration=duration,
            video_guider_params=video_guider_params, audio_guider_params=audio_guider_params,
        )
        retake_kwargs["num_inference_steps"] = args.num_inference_steps

        out_audio = _build_out_audio(str(retake_input_video), pipeline, waveform_sr, duration)

        # ---- Render baseline (target video with UNMODIFIED latents, edit prompt text only) ----
        baseline_path = output_dir / "baseline_video.mp4"
        log.info("Rendering baseline (unmodified latents)...")
        _render_video(
            pipeline=pipeline,
            retake_input_video=str(retake_input_video),
            cached_video_latent=cached_video_latent,
            audio_for_render=base_audio_latent,
            pos_context=base_pos_context,
            base_neg_context=base_neg_context,
            retake_kwargs=retake_kwargs,
            out_audio=out_audio,
            out_path=baseline_path,
            num_frames=num_frames,
            frame_rate=frame_rate,
        )
        if wandb_run is not None and baseline_path.exists():
            try:
                wandb_run.log({"media/video/baseline": _wandb.Video(str(baseline_path), format="mp4", caption="Baseline")}, step=0)
            except Exception:
                pass

        # ---- Render transfer result ----
        transfer_path = output_dir / f"transfer_{mode}.mp4"
        log.info("Rendering transfer result (mode=%s)...", mode)
        _render_video(
            pipeline=pipeline,
            retake_input_video=str(retake_input_video),
            cached_video_latent=cached_video_latent,
            audio_for_render=audio_for_render,
            pos_context=pos_context,
            base_neg_context=base_neg_context,
            retake_kwargs=retake_kwargs,
            out_audio=out_audio,
            out_path=transfer_path,
            num_frames=num_frames,
            frame_rate=frame_rate,
        )
        if wandb_run is not None and transfer_path.exists():
            try:
                wandb_run.log({"media/video/transfer": _wandb.Video(str(transfer_path), format="mp4", caption=f"Transfer ({mode})")}, step=1)
            except Exception:
                pass

        # ---- Qwen scoring ----
        results: dict = {
            "mode": mode,
            "opt_dir": str(opt_dir),
            "target_video": args.target_video,
            "edit_prompt": args.edit_prompt,
        }
        if src_qwen_score is not None:
            results["src_optimization_qwen_score"] = src_qwen_score

        if args.qwen_model and transfer_path.exists():
            try:
                qwen_scores = _score_with_qwen(
                    transfer_path, args.edit_prompt, args.qwen_model,
                    device, max_frames=args.qwen_eval_frames, img_size=224,
                )
                results.update(qwen_scores)
                log.info("Qwen transfer score: yes_prob=%.4f  nll=%.4f", qwen_scores["qwen_yes_prob"], qwen_scores["qwen_nll"])
                if src_qwen_score is not None:
                    log.info("  vs. source optimization: %.4f  (delta: %+.4f)",
                             src_qwen_score, qwen_scores["qwen_yes_prob"] - src_qwen_score)
                if wandb_run is not None:
                    wandb_run.summary.update({f"transfer/{k}": v for k, v in qwen_scores.items()})
                    if src_qwen_score is not None:
                        wandb_run.summary["transfer/qwen_score_delta_vs_source"] = qwen_scores["qwen_yes_prob"] - src_qwen_score
            except Exception:
                log.warning("Qwen scoring failed", exc_info=True)

        # ---- CLIP diagnostics ----
        if args.clip_diag and baseline_path.exists() and transfer_path.exists():
            static_prompt = args.static_prompt or f"A static scene before: {args.edit_prompt}"
            clip_summary = _write_clip_diagnostics(
                baseline_path=baseline_path,
                transfer_path=transfer_path,
                edit_prompt=args.edit_prompt,
                static_prompt=static_prompt,
                output_dir=output_dir,
                clip_model_name=args.clip_model,
                max_frames=args.clip_max_frames,
                batch_size=8,
                device=device,
                wandb_run=wandb_run,
            )
            results.update(clip_summary)
            log.info("CLIP: transfer mean_edit_sim=%.4f  baseline mean_edit_sim=%.4f",
                     clip_summary.get("transfer_mean_edit_sim", 0),
                     clip_summary.get("baseline_mean_edit_sim", 0))
            if wandb_run is not None:
                wandb_run.summary.update({f"transfer/clip_{k}": v for k, v in clip_summary.items()})

        # ---- Save summary ----
        summary_path = output_dir / "transfer_results.json"
        summary_path.write_text(json.dumps(results, indent=2))
        log.info("Results saved to %s", summary_path)

        # ---- Print summary ----
        log.info("")
        log.info("=" * 60)
        log.info("TRANSFER SUMMARY")
        log.info("=" * 60)
        log.info("  Mode        : %s", mode)
        log.info("  Edit prompt : %s", args.edit_prompt)
        log.info("  Opt dir     : %s", opt_dir)
        if src_qwen_score is not None:
            log.info("  Src Qwen yes_prob (optimization) : %.4f", src_qwen_score)
        if "qwen_yes_prob" in results:
            log.info("  Transfer Qwen yes_prob           : %.4f", results["qwen_yes_prob"])
        for key in ("motion_yes_prob", "entities_yes_prob", "overall_yes_prob"):
            if key in results:
                log.info("    %s : %.4f", key, results[key])
        if "transfer_mean_edit_sim" in results:
            log.info("  CLIP edit sim (transfer)  : %.4f", results["transfer_mean_edit_sim"])
        if "baseline_mean_edit_sim" in results:
            log.info("  CLIP edit sim (baseline)  : %.4f", results["baseline_mean_edit_sim"])
        log.info("  Output      : %s", output_dir)
        log.info("")

    finally:
        if wandb_run is not None:
            wandb_run.finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--target-video", required=True)
    p.add_argument("--opt-dir", required=True,
                   help="mode_audio / mode_text / mode_both dir from an optimize_qwen_vl run "
                        "(contains best_audio_latent_{mode}.pt etc).")
    p.add_argument("--mode", required=True, choices=["text", "audio", "both"])
    p.add_argument("--edit-prompt", required=True)
    p.add_argument("--output-dir", required=True)

    # Evaluation
    p.add_argument("--qwen-model", default=DEFAULT_QWEN_ROOT,
                   help="Qwen2.5-VL model path for scoring the transfer result. "
                        "Set to empty string to skip Qwen evaluation.")
    p.add_argument("--qwen-eval-frames", type=int, default=16,
                   help="Frames to pass to Qwen for inference scoring (no grad, memory-light).")
    p.add_argument("--clip-diag", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    p.add_argument("--clip-max-frames", type=int, default=0, help="0 = all frames")
    p.add_argument("--static-prompt", default="",
                   help="Reference prompt for CLIP diagnostics (before-state description).")

    # Video shape
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--num-frames", type=int, default=None)
    p.add_argument("--frame-rate", type=float, default=None)
    p.add_argument("--audio-sr", type=int, default=44100)

    # Pipeline
    p.add_argument("--negative-prompt", default="")
    p.add_argument("--enhance-prompt", action="store_true")
    p.add_argument("--num-inference-steps", type=int, default=30)
    p.add_argument("--retake-num-inference-steps", type=int, default=None)
    p.add_argument("--final-retake-num-inference-steps", type=int, default=None)
    p.add_argument("--retake-start-frames", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cfg-scale", type=float, default=None)
    p.add_argument("--audio-cfg-scale", type=float, default=None)
    p.add_argument("--a2v-scale", type=float, default=None)
    p.add_argument("--low-memory-guidance", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--quantization", default=None, choices=["fp8-cast", "fp8-scaled-mm"])
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--checkpoint-path", default=DEFAULT_CHECKPOINT)
    p.add_argument("--gemma-root", default=DEFAULT_GEMMA_ROOT)
    p.add_argument("--lora", dest="loras", nargs="+", metavar=("PATH", "STRENGTH"),
                   action="append", default=[])

    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.retake_num_inference_steps is None:
        args.retake_num_inference_steps = args.num_inference_steps
    args.ti2v_num_inference_steps = args.num_inference_steps
    run(args)


if __name__ == "__main__":
    main()
