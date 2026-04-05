"""Video-CLIP loss: measures how consistent generated frames are with an edit prompt.

Uses X-CLIP (microsoft/xclip-base-patch32) which applies cross-frame attention
so the video embedding is temporally aware, unlike standard CLIP which scores
each frame independently.

Used as the primary optimization objective for all three experiment modes:
  - text token optimization
  - audio latent optimization
  - joint (both)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

# X-CLIP uses the same normalization constants as CLIP
_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

# X-CLIP base was trained with 8 frames
_XCLIP_NUM_FRAMES = 8


def build_clip_model(model_name: str, device: torch.device):
    """Load a HuggingFace X-CLIP model with frozen weights.

    For video-aware losses use 'microsoft/xclip-base-patch32' (default).
    Falls back gracefully to standard CLIPModel if a CLIP model name is given.

    Returns (model, tokenizer).
    """
    from transformers import AutoTokenizer

    if "xclip" in model_name.lower():
        from transformers import XCLIPModel
        model = XCLIPModel.from_pretrained(model_name).to(device).eval()
    else:
        from transformers import CLIPModel
        model = CLIPModel.from_pretrained(model_name).to(device).eval()

    for p in model.parameters():
        p.requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


@torch.no_grad()
def encode_text_for_clip(
    prompt: str,
    clip_model,
    tokenizer,
    device: torch.device,
) -> torch.Tensor:
    """Encode a text prompt into a normalized embedding.

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
    max_frames: int = _XCLIP_NUM_FRAMES,
) -> torch.Tensor:
    """Differentiable video-CLIP alignment loss.

    For X-CLIP models: all sampled frames are encoded jointly via cross-frame
    attention, producing a single temporally-aware video embedding.

    For standard CLIP models (fallback): frames are encoded independently and
    the embeddings are mean-pooled (original behaviour).

    loss = 1 - cosine_similarity(video_embedding, text_embedding)

    Gradient flows back through the visual encoder to the input frames,
    and then through the diffusion decoder to whichever parameter is being
    optimized (audio latent, text delta, or both).

    Args:
        frames_chw: [N, 3, H, W] float tensor in [0, 1]. Must be part of the
            autograd graph (i.e. generated during a gradient-enabled forward pass).
        text_embedding: [1, D] normalized float32 tensor (pre-computed, no grad).
        clip_model: Frozen HuggingFace XCLIPModel (or CLIPModel for fallback).
        max_frames: Number of frames to sample. X-CLIP base expects 8.

    Returns:
        Scalar loss in [0, 2]. Lower = more consistent with the edit prompt.
    """
    n = frames_chw.shape[0]

    # Uniformly subsample to max_frames (X-CLIP requires a fixed clip length)
    num_frames = min(max_frames, n) if max_frames > 0 else n
    idx = torch.linspace(0, n - 1, num_frames, device=frames_chw.device).round().long()
    frames_chw = frames_chw[idx]  # [T, 3, H, W]

    # Resize to 224x224 (differentiable)
    pixel_values = F.interpolate(
        frames_chw,
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
    )

    # Normalize with CLIP/X-CLIP channel stats
    mean = torch.tensor(_CLIP_MEAN, device=pixel_values.device, dtype=pixel_values.dtype).view(1, 3, 1, 1)
    std = torch.tensor(_CLIP_STD, device=pixel_values.device, dtype=pixel_values.dtype).view(1, 3, 1, 1)
    pixel_values = (pixel_values - mean) / std  # [T, 3, 224, 224]

    from transformers import XCLIPModel
    if isinstance(clip_model, XCLIPModel):
        # X-CLIP expects exactly model_num_frames frames — it hardcodes this in
        # its message-token reshape: msg.view(batch, self.num_frames, hidden).
        # Passing any other count causes a shape mismatch at runtime.
        model_num_frames = clip_model.config.vision_config.num_frames  # e.g. 8
        if pixel_values.shape[0] != model_num_frames:
            idx = torch.linspace(0, pixel_values.shape[0] - 1, model_num_frames,
                                 device=pixel_values.device).round().long()
            pixel_values = pixel_values[idx]
        # Call X-CLIP components manually to avoid a transformers version bug
        # where mit() returns a plain tuple instead of BaseModelOutputWithPooling,
        # causing get_video_features() to crash on mit_outputs.pooler_output.
        flat = pixel_values  # [T, 3, 224, 224]  (batch=1, so B*T = T)
        vision_out = clip_model.vision_model(pixel_values=flat)
        # vision_out[1] is the pooled CLS embedding: [T, vision_hidden]
        frame_embeds = clip_model.visual_projection(vision_out[1])  # [T, proj_dim]
        frame_embeds = frame_embeds.unsqueeze(0)  # [1, T, proj_dim]
        mit_out = clip_model.mit(frame_embeds)
        # Handle both ModelOutput (.pooler_output) and plain tuple ([1])
        video_features = (mit_out.pooler_output if hasattr(mit_out, "pooler_output")
                          else mit_out[1]).float()  # [1, D]
        video_features = F.normalize(video_features, dim=-1)
        text_emb = text_embedding.to(device=video_features.device, dtype=video_features.dtype)
        cosine_sim = (video_features * text_emb).sum(dim=-1)  # [1]
    else:
        # Fallback: standard CLIP, frame-independent (original behaviour)
        image_features = clip_model.get_image_features(pixel_values=pixel_values).float()  # [T, D]
        image_features = F.normalize(image_features, dim=-1)
        text_emb = text_embedding.to(device=image_features.device, dtype=image_features.dtype)
        cosine_sim = (image_features * text_emb).sum(dim=-1)  # [T]

    return 1.0 - cosine_sim.mean()