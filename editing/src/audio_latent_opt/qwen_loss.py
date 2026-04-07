"""Qwen2.5-VL video-text alignment loss.

Replaces the X-CLIP loss with a large generative video-language model.
The loss measures: how likely is Qwen2.5-VL to answer "yes" when asked
"Does this video show: {edit_prompt}?"

    loss = 1 - P("yes" | video_frames, question)

Gradient flows through Qwen2.5-VL's visual encoder back to the pixel values,
which are computed differentiably (resize → normalize → patchify) from the
generated frames.

Pixel format follows Qwen2VLImageProcessor._preprocess exactly:
  - Normalise: (x - 0.5) / 0.5   (frames_chw is assumed to be in [0, 1])
  - Extract 14×14 spatial patches, 2-frame temporal patches, 2×2 spatial merge
  - pixel_values_videos: [grid_t * gh * gw, C * 2 * 4 * 14²] = [N, 4704]
  - video_grid_thw: [(grid_t, gh, gw)]  where gh = H // (14*2), gw = W // (14*2)

For 224×224 images and 8 frames: N = 4 * 8 * 8 = 256, feature_dim = 4704.
"""
from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)

# Qwen2.5-VL image normalization (same as CLIP)
_QWEN_MEAN = [0.5, 0.5, 0.5]
_QWEN_STD = [0.5, 0.5, 0.5]

# Visual architecture constants (match Qwen2.5-VL-7B config)
_PATCH_SIZE = 14
_TEMPORAL_PATCH_SIZE = 2
_MERGE_SIZE = 2  # 2×2 spatial patch merge → effective token covers 28×28 pixels

# Safe image size: divisible by patch_size * merge_size = 28
# 224 = 8 * 28  →  8×8 = 64 spatial tokens per temporal chunk
QWEN_IMG_SIZE = 224


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def build_qwen_model(model_name: str, device: torch.device, gradient_checkpointing: bool = True):
    """Load a frozen Qwen2.5-VL model for video-text scoring.

    Returns (model, processor).

    Args:
        model_name: HuggingFace model ID, e.g. "Qwen/Qwen2.5-VL-7B-Instruct".
        device: Target device.
        gradient_checkpointing: Enable gradient checkpointing on the LLM layers
            to reduce activation memory during the backward pass.
    """
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    log.info("Loading Qwen2.5-VL model: %s ...", model_name)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map=str(device),
    ).eval()

    for p in model.parameters():
        p.requires_grad_(False)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        log.info("Qwen2.5-VL gradient checkpointing enabled.")

    # Force a fixed image resolution to keep grid_thw predictable.
    # min_pixels = max_pixels = 224*224 tells the processor not to rescale.
    processor = AutoProcessor.from_pretrained(
        model_name,
        min_pixels=QWEN_IMG_SIZE * QWEN_IMG_SIZE,
        max_pixels=QWEN_IMG_SIZE * QWEN_IMG_SIZE,
    )

    log.info("Qwen2.5-VL loaded. dtype=bfloat16, img_size=%d", QWEN_IMG_SIZE)
    return model, processor


# ---------------------------------------------------------------------------
# One-time input preparation (no grad, done before optimization)
# ---------------------------------------------------------------------------

@torch.no_grad()
def build_qwen_inputs(
    processor,
    edit_prompt: str,
    num_frames: int,
    img_size: int,
    device: torch.device,
) -> tuple[dict, int, int]:
    """Build frozen text inputs and yes/no token IDs.

    Uses dummy (black) PIL frames so the processor computes the correct
    input_ids with the right number of <video_pad> placeholders.
    We will replace pixel_values_videos with differentiable tensors at
    runtime.

    Returns:
        cached_inputs: dict with "input_ids", "attention_mask", "video_grid_thw"
            (all on *device*, dtype fixed, no grad).
        yes_token_id: vocabulary index for the first token of "yes".
        no_token_id:  vocabulary index for the first token of "no".
    """
    import numpy as np
    from PIL import Image

    # num_frames must be even (temporal_patch_size = 2)
    if num_frames % _TEMPORAL_PATCH_SIZE != 0:
        num_frames = num_frames + (_TEMPORAL_PATCH_SIZE - num_frames % _TEMPORAL_PATCH_SIZE)
        log.warning("Rounded num_frames up to %d to satisfy temporal_patch_size=2.", num_frames)

    dummy_frame = Image.fromarray(np.zeros((img_size, img_size, 3), dtype=np.uint8))
    dummy_video = [dummy_frame] * num_frames

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": dummy_video, "fps": 1},
                {
                    "type": "text",
                    "text": (
                        f'Does this video show: "{edit_prompt}"? '
                        "Answer only 'yes' or 'no'."
                    ),
                },
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        videos=[dummy_video],
        return_tensors="pt",
        padding=True,
    )

    cached = {
        "input_ids": inputs["input_ids"].to(device),
        "attention_mask": inputs["attention_mask"].to(device),
    }
    if "video_grid_thw" in inputs:
        cached["video_grid_thw"] = inputs["video_grid_thw"].to(device)

    yes_token_id = processor.tokenizer.encode("yes", add_special_tokens=False)[-1]
    no_token_id = processor.tokenizer.encode("no", add_special_tokens=False)[-1]

    log.info(
        "Qwen inputs built. input_ids shape: %s, yes_id=%d, no_id=%d",
        tuple(cached["input_ids"].shape), yes_token_id, no_token_id,
    )
    return cached, yes_token_id, no_token_id


# ---------------------------------------------------------------------------
# Differentiable patchify (replicates Qwen2VLImageProcessor._preprocess)
# ---------------------------------------------------------------------------

def _frames_to_pixel_values(
    frames: torch.Tensor,
    patch_size: int = _PATCH_SIZE,
    temporal_patch_size: int = _TEMPORAL_PATCH_SIZE,
    merge_size: int = _MERGE_SIZE,
) -> torch.Tensor:
    """Convert [T, C, H, W] normalised frames → pixel_values_videos.

    This exactly mirrors the reshape/permute done inside
    Qwen2VLImageProcessor._preprocess so that our differentiable tensor
    matches what the model expects.

    Output shape: [grid_t * (H//(p*m)) * (W//(p*m)),
                   C * temporal_patch_size * merge_size² * patch_size²]
    For T=8, H=W=224, defaults: [256, 4704]
    """
    T, channel, H, W = frames.shape

    grid_t = T // temporal_patch_size
    grid_h = H // patch_size
    grid_w = W // patch_size

    # --- Normalize: (x - 0.5) / 0.5 = 2x - 1 ---
    mean = torch.tensor(_QWEN_MEAN, device=frames.device, dtype=frames.dtype).view(1, 3, 1, 1)
    std = torch.tensor(_QWEN_STD, device=frames.device, dtype=frames.dtype).view(1, 3, 1, 1)
    frames = (frames - mean) / std  # [T, C, H, W]

    # Reshape to expose patch grid axes
    x = frames.reshape(
        grid_t,
        temporal_patch_size,
        channel,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    )
    # Permute: (grid_t, gh//m, gw//m, m, m, C, t_patch, patch_h, patch_w)
    x = x.permute(0, 3, 6, 4, 7, 2, 1, 5, 8).contiguous()

    # Flatten to [num_tokens, patch_dim]
    num_tokens = grid_t * (grid_h // merge_size) * (grid_w // merge_size)
    patch_dim = channel * temporal_patch_size * (merge_size ** 2) * (patch_size ** 2)
    return x.reshape(num_tokens, patch_dim)


# ---------------------------------------------------------------------------
# Differentiable loss
# ---------------------------------------------------------------------------

def compute_qwen_video_loss(
    frames_chw: torch.Tensor,
    qwen_model,
    cached_inputs: dict,
    yes_token_id: int,
    no_token_id: int,
    max_frames: int = 8,
    img_size: int = QWEN_IMG_SIZE,
) -> torch.Tensor:
    """Differentiable Qwen2.5-VL video-text alignment loss.

    Scores the generated video against the cached yes/no question and
    returns  1 - P("yes" | video, question).  Lower = better match.

    Args:
        frames_chw: [N, 3, H, W] float in [0, 1].  Must be part of the
            autograd graph (generated during a gradient-enabled forward).
        qwen_model: Frozen Qwen2VLForConditionalGeneration.
        cached_inputs: dict from build_qwen_inputs() — input_ids,
            attention_mask, video_grid_thw.
        yes_token_id: vocab index for "yes".
        no_token_id:  vocab index for "no".
        max_frames: Number of frames to sample (must be even).
        img_size: Spatial size to resize frames to (must be divisible by 28).

    Returns:
        Scalar loss in [0, 1].  Lower = frames more consistent with prompt.
    """
    n = frames_chw.shape[0]

    # Sample an even number of frames
    num_frames = min(max_frames, n) if max_frames > 0 else n
    if num_frames % _TEMPORAL_PATCH_SIZE != 0:
        num_frames -= num_frames % _TEMPORAL_PATCH_SIZE
    num_frames = max(num_frames, _TEMPORAL_PATCH_SIZE)

    idx = torch.linspace(0, n - 1, num_frames, device=frames_chw.device).round().long()
    frames = frames_chw[idx]  # [T, 3, H, W]

    # Resize to model's fixed spatial resolution (differentiable)
    frames = F.interpolate(
        frames,
        size=(img_size, img_size),
        mode="bilinear",
        align_corners=False,
    )

    # Convert to the format Qwen2.5-VL expects (differentiable)
    pixel_values_videos = _frames_to_pixel_values(frames)  # [num_tokens, patch_dim]

    # Cast to bfloat16 to match model weights
    pixel_values_videos = pixel_values_videos.to(dtype=torch.bfloat16)

    # Forward pass — gradient flows through visual encoder → LLM → logits
    outputs = qwen_model(
        input_ids=cached_inputs["input_ids"],
        attention_mask=cached_inputs["attention_mask"],
        pixel_values_videos=pixel_values_videos,
        video_grid_thw=cached_inputs.get("video_grid_thw"),
    )

    # Logits at the last token position (where model would generate yes/no)
    last_logits = outputs.logits[0, -1, :].float()  # [vocab_size]

    yes_logit = last_logits[yes_token_id]
    no_logit = last_logits[no_token_id]

    # Soft yes-probability over {yes, no}
    yes_prob = torch.softmax(torch.stack([yes_logit, no_logit]), dim=0)[0]
    # return 1.0 - yes_prob
    return -torch.log(yes_prob + 1e-8)  # log loss, more stable gradients when yes_prob ~ 0


# ---------------------------------------------------------------------------
# Re-export attention extraction (implemented in attn_vis to avoid circular
# imports; callers can import from either module)
# ---------------------------------------------------------------------------
def extract_qwen_attention_maps(frames_chw, qwen_model, cached_inputs, **kwargs):
    """Convenience re-export — see attn_vis.extract_qwen_attention_maps."""
    from .attn_vis import extract_qwen_attention_maps as _impl
    return _impl(frames_chw, qwen_model, cached_inputs, **kwargs)
