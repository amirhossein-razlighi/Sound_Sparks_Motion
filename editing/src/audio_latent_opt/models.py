from __future__ import annotations

from ltx_core.components.guiders import MultiModalGuiderParams
from ltx_core.quantization import QuantizationPolicy
from ltx_pipelines.retake import RetakePipeline
from ltx_pipelines.utils.constants import detect_params


def resolve_quantization_policy(name: str | None) -> QuantizationPolicy | None:
    if name == "fp8-cast":
        return QuantizationPolicy.fp8_cast()
    if name == "fp8-scaled-mm":
        try:
            return QuantizationPolicy.fp8_scaled_mm()
        except ImportError:
            return QuantizationPolicy.fp8_cast()
    return None


def build_guiders(args, checkpoint_path: str) -> tuple[MultiModalGuiderParams, MultiModalGuiderParams]:
    params = detect_params(checkpoint_path)
    if args.low_memory_guidance:
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


def build_retake_pipeline(*, checkpoint_path: str, gemma_root: str, loras, device, quant_policy, gradient_checkpointing: bool) -> RetakePipeline:
    return RetakePipeline(
        checkpoint_path=checkpoint_path,
        gemma_root=gemma_root,
        loras=tuple(loras),
        device=device,
        quantization=quant_policy,
        gradient_checkpointing=gradient_checkpointing,
    )
