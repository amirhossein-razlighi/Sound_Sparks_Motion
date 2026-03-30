"""Video-CLIP loss: measures how consistent generated frames are with an edit prompt.

Used as the primary optimization objective for all three experiment modes:
  - text token optimization
  - audio latent optimization
  - joint (both)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

# Standard CLIP normalization constants
_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def build_clip_model(model_name: str, device: torch.device):
    """Load a HuggingFace CLIP model with frozen weights.

    Returns (clip_model, tokenizer).
    """
    from transformers import CLIPModel, CLIPTokenizerFast

    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tokenizer = CLIPTokenizerFast.from_pretrained(model_name)
    return model, tokenizer


@torch.no_grad()
def encode_text_for_clip(
    prompt: str,
    clip_model,
    tokenizer,
    device: torch.device,
) -> torch.Tensor:
    """Encode a text prompt into a normalized CLIP embedding.

    Returns:
        Tensor of shape [1, D] (float32, L2-normalized).
    """
    inputs = tokenizer(
        [prompt],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77,
    ).to(device)
    text_features = clip_model.get_text_features(**inputs).float()
    return F.normalize(text_features, dim=-1)


def compute_clip_video_loss(
    frames_chw: torch.Tensor,
    text_embedding: torch.Tensor,
    clip_model,
    max_frames: int = 16,
) -> torch.Tensor:
    """Differentiable CLIP alignment loss for video frames.

    Measures how well generated frames align with the edit prompt embedding.

    loss = 1 - mean_cosine_similarity(CLIP(frames), text_embedding)

    Gradient flows back through CLIP's visual encoder to the input frames,
    and then through the diffusion decoder to whichever parameter is being
    optimized (audio latent, text delta, or both).

    Args:
        frames_chw: [N, 3, H, W] float tensor in [0, 1]. Must be part of the
            autograd graph (i.e. generated during a gradient-enabled forward pass).
        text_embedding: [1, D] normalized float32 tensor (pre-computed, no grad).
        clip_model: Frozen HuggingFace CLIPModel.
        max_frames: If N > max_frames, uniformly subsample before encoding.

    Returns:
        Scalar loss in [0, 2]. Lower = more consistent with the edit prompt.
    """
    n = frames_chw.shape[0]
    if max_frames > 0 and n > max_frames:
        idx = torch.linspace(0, n - 1, max_frames, device=frames_chw.device).round().long()
        frames_chw = frames_chw[idx]

    # Resize to CLIP input size (differentiable)
    pixel_values = F.interpolate(
        frames_chw,
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
    )

    # Normalize with CLIP channel stats
    mean = torch.tensor(_CLIP_MEAN, device=pixel_values.device, dtype=pixel_values.dtype).view(1, 3, 1, 1)
    std = torch.tensor(_CLIP_STD, device=pixel_values.device, dtype=pixel_values.dtype).view(1, 3, 1, 1)
    pixel_values = (pixel_values - mean) / std

    # CLIP image features — gradient flows through get_image_features
    image_features = clip_model.get_image_features(pixel_values=pixel_values).float()  # [N, D]
    image_features = F.normalize(image_features, dim=-1)

    # Cosine similarity against text embedding
    text_emb = text_embedding.to(device=image_features.device, dtype=image_features.dtype)  # [1, D]
    cosine_sim = (image_features * text_emb).sum(dim=-1)  # [N]

    return 1.0 - cosine_sim.mean()
