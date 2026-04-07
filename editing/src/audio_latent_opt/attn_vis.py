"""Attention map visualization for LTX-2 video editing.

Two complementary views:
  1. Qwen2.5-VL self-attention  — where the *scorer* looks in the video when
     deciding yes/no. Extracted from the last Qwen decoder layers.
  2. LTX cross-attention hooks  — where audio / text drives video generation
     inside the diffusion transformer:
       - audio_to_video_attn : audio latent tokens → video latent tokens
       - attn2               : text embedding tokens → video latent tokens

All extraction is done under torch.no_grad() and is only triggered at
visualize_every_iters steps to keep H100 memory pressure low.
"""
from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colormap helpers (plasma, no matplotlib dependency)
# ---------------------------------------------------------------------------

# Five key colours of the plasma colormap sampled at t ∈ {0, 0.25, 0.5, 0.75, 1}
_PLASMA_KEYPOINTS = np.array([
    [0.050383, 0.029803, 0.527975],
    [0.461214, 0.097688, 0.585136],
    [0.798216, 0.280197, 0.469538],
    [0.973381, 0.585016, 0.255155],
    [0.940015, 0.975158, 0.131326],
], dtype=np.float32)


def _plasma(x: np.ndarray) -> np.ndarray:
    """Plasma colormap. x: float32 in [0, 1] → float32 [..., 3] in [0, 1]."""
    x = np.clip(x, 0.0, 1.0)
    n = len(_PLASMA_KEYPOINTS) - 1
    idx_f = x * n
    lo = np.floor(idx_f).astype(np.int32).clip(0, n - 1)
    hi = (lo + 1).clip(0, n)
    t = (idx_f - lo)[..., np.newaxis]
    return _PLASMA_KEYPOINTS[lo] + t * (_PLASMA_KEYPOINTS[hi] - _PLASMA_KEYPOINTS[lo])


# ---------------------------------------------------------------------------
# Frame helpers
# ---------------------------------------------------------------------------

def frames_to_numpy(frames: torch.Tensor) -> np.ndarray:
    """[N, 3, H, W] float in [0,1] → [N, H, W, 3] uint8."""
    arr = frames.detach().float().cpu().permute(0, 2, 3, 1).numpy()
    return (arr.clip(0.0, 1.0) * 255).astype(np.uint8)


def overlay_heatmap(
    frames: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.55,
) -> np.ndarray:
    """Blend a spatial attention heatmap onto video frames.

    Args:
        frames:  [T, H, W, 3] uint8 video frames.
        heatmap: float32 array broadcastable to [T, H, W].  Values in [0, 1].
        alpha:   Heatmap opacity (0 = invisible, 1 = solid colour).

    Returns:
        [T, H, W, 3] uint8 blended frames.
    """
    T, H, W, _ = frames.shape

    # Ensure heatmap is [T, H, W]
    hmap = np.asarray(heatmap, dtype=np.float32)
    if hmap.ndim == 2:
        hmap = np.broadcast_to(hmap[np.newaxis], (T, H, W))
    elif hmap.ndim != 3:
        raise ValueError(f"heatmap must be 2-D or 3-D, got shape {hmap.shape}")

    # Upsample to frame resolution using bilinear (via torch)
    hmap_t = torch.from_numpy(hmap).unsqueeze(0).unsqueeze(0)  # [1, 1, T, H', W']  or [1,1,H',W']
    if hmap_t.shape[-2:] != (H, W) or hmap_t.shape[2] != T:
        if hmap_t.dim() == 4:
            hmap_t = F.interpolate(hmap_t, size=(H, W), mode="bilinear", align_corners=False)
        else:
            hmap_t = F.interpolate(hmap_t, size=(T, H, W), mode="trilinear", align_corners=False)
    hmap_np = hmap_t.squeeze().numpy()  # [T, H, W] or [H, W]
    if hmap_np.ndim == 2:
        hmap_np = np.broadcast_to(hmap_np[np.newaxis], (T, H, W))

    # Normalise per-frame so the brightest spot is always full colour
    hmin = hmap_np.min(axis=(1, 2), keepdims=True)
    hmax = hmap_np.max(axis=(1, 2), keepdims=True)
    safe_range = np.where((hmax - hmin) > 1e-6, hmax - hmin, 1.0)
    hmap_norm = (hmap_np - hmin) / safe_range  # [T, H, W] in [0,1]

    colour = _plasma(hmap_norm)  # [T, H, W, 3] float32

    base = frames.astype(np.float32) / 255.0
    blended = (1.0 - alpha) * base + alpha * colour
    return (blended.clip(0.0, 1.0) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# W&B logging helper
# ---------------------------------------------------------------------------

def log_attn_frames_to_wandb(
    wandb_run,
    tag: str,
    frames_np: np.ndarray,
    heatmap: np.ndarray,
    step: int,
    caption: str = "",
    fps: int = 8,
) -> None:
    """Overlay heatmap on frames and log both the overlay video and a thumbnail
    to wandb.

    Args:
        wandb_run: Active wandb run (or None — no-op).
        tag: W&B media key prefix, e.g. "both/attn_qwen_iter_10".
        frames_np: [T, H, W, 3] uint8 frames.
        heatmap: float32 [T, H, W] or [H, W] heatmap.
        step: W&B step index.
        caption: Caption string for the video.
        fps: Playback fps for the logged video.
    """
    if wandb_run is None:
        return
    try:
        import wandb

        overlay = overlay_heatmap(frames_np, heatmap)  # [T, H, W, 3] uint8

        # Video: [T, H, W, C] → wandb expects [T, C, H, W]
        video_arr = overlay.transpose(0, 3, 1, 2)  # [T, 3, H, W]
        wandb_run.log(
            {
                tag: wandb.Video(video_arr, fps=fps, format="mp4", caption=caption),
                f"{tag}_thumb": wandb.Image(overlay[len(overlay) // 2], caption=caption),
            },
            step=step,
        )
    except Exception:
        log.warning("Failed to log attention video to W&B (tag=%s)", tag, exc_info=True)


# ---------------------------------------------------------------------------
# Qwen2.5-VL attention extraction
# ---------------------------------------------------------------------------

_TEMPORAL_PATCH_SIZE = 2
_MERGE_SIZE = 2


def _get_video_pad_token_id(qwen_model) -> int:
    """Return the video pad token ID from Qwen2.5-VL model config."""
    cfg = qwen_model.config
    for attr in ("video_token_id", "image_token_id"):
        if hasattr(cfg, attr):
            return int(getattr(cfg, attr))
    return 151656  # <|video_pad|> default for Qwen2.5-VL


@torch.no_grad()
def extract_qwen_attention_maps(
    frames_chw: torch.Tensor,
    qwen_model,
    cached_inputs: dict,
    max_frames: int = 8,
    img_size: int = 224,
    num_layers_to_avg: int = 4,
) -> dict | None:
    """Extract spatial attention maps from Qwen2.5-VL.

    Computes how much the model's yes/no generation token attends to each video
    patch token.  Averaged over the last ``num_layers_to_avg`` decoder layers
    and all attention heads.

    Returns dict with:
        "attn_spatial": float32 numpy [grid_t, grid_h, grid_w]
        "grid_thw": (grid_t, grid_h, grid_w)
        "num_frames": int — how many frames were actually used
    or None on failure.

    Memory note: ``output_attentions=True`` allocates ~140 MB on top of the
    resident model.  All tensors are freed before returning.
    """
    try:
        from .qwen_loss import _frames_to_pixel_values

        n = frames_chw.shape[0]
        num_frames = min(max_frames, n) if max_frames > 0 else n
        if num_frames % _TEMPORAL_PATCH_SIZE != 0:
            num_frames -= num_frames % _TEMPORAL_PATCH_SIZE
        num_frames = max(num_frames, _TEMPORAL_PATCH_SIZE)

        idx = torch.linspace(0, n - 1, num_frames, device=frames_chw.device).round().long()
        frames = frames_chw[idx].detach()

        frames = F.interpolate(
            frames,
            size=(img_size, img_size),
            mode="bilinear",
            align_corners=False,
        )
        pixel_values_videos = _frames_to_pixel_values(frames).to(dtype=torch.bfloat16)

        # output_attentions=True is incompatible with gradient checkpointing
        was_gc = getattr(qwen_model, "is_gradient_checkpointing", False)
        if was_gc:
            qwen_model.gradient_checkpointing_disable()

        try:
            outputs = qwen_model(
                input_ids=cached_inputs["input_ids"],
                attention_mask=cached_inputs["attention_mask"],
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=cached_inputs.get("video_grid_thw"),
                output_attentions=True,
            )
        finally:
            if was_gc:
                qwen_model.gradient_checkpointing_enable()

        # outputs.attentions: tuple of [1, num_heads, seq_len, seq_len] per layer
        attentions = outputs.attentions

        # Find video pad token positions in the sequence
        video_pad_id = _get_video_pad_token_id(qwen_model)
        input_ids_1d = cached_inputs["input_ids"][0]  # [seq_len]
        vis_positions = (input_ids_1d == video_pad_id).nonzero(as_tuple=True)[0]

        if vis_positions.numel() == 0:
            log.warning("No video pad tokens found — cannot extract Qwen attention maps.")
            return None

        last_pos = input_ids_1d.shape[0] - 1  # generation position

        # Average attention to visual tokens from the last `num_layers_to_avg` layers
        attn_list = []
        for layer_attn in attentions[-num_layers_to_avg:]:
            # layer_attn: [1, H, seq, seq] → select row at last_pos, columns at vis_positions
            a = layer_attn[0, :, last_pos, :][:, vis_positions].float()  # [H, num_vis]
            attn_list.append(a.mean(dim=0))  # [num_vis]
        attn_vis = torch.stack(attn_list, dim=0).mean(dim=0).cpu().numpy()  # [num_vis]

        # Determine spatial grid from video_grid_thw
        if "video_grid_thw" in cached_inputs:
            thw = cached_inputs["video_grid_thw"][0]  # [grid_t, grid_h, grid_w]
            grid_t = int(thw[0])
            grid_h = int(thw[1])
            grid_w = int(thw[2])
        else:
            grid_t = num_frames // _TEMPORAL_PATCH_SIZE
            grid_h = img_size // (14 * _MERGE_SIZE)  # patch_size=14, merge=2 → 28
            grid_w = grid_h

        expected = grid_t * grid_h * grid_w
        if attn_vis.shape[0] != expected:
            log.warning(
                "Qwen vis-token count mismatch: got %d, expected %d",
                attn_vis.shape[0], expected,
            )
            return None

        attn_spatial = attn_vis.reshape(grid_t, grid_h, grid_w).astype(np.float32)

        del outputs, attentions
        torch.cuda.empty_cache()

        return {
            "attn_spatial": attn_spatial,  # [grid_t, grid_h, grid_w]
            "grid_thw": (grid_t, grid_h, grid_w),
            "num_frames": num_frames,
        }

    except Exception:
        log.warning("Qwen attention map extraction failed", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# LTX cross-attention capture (audio→video and text→video)
# ---------------------------------------------------------------------------

class LTXAttentionCapture:
    """Context manager that registers forward hooks on LTX cross-attention.

    Hooks fire on every transformer block call during the denoising loop.
    Attention weights are averaged over heads, batch, and denoising steps.

    Usage::

        with LTXAttentionCapture(pipeline, block_fraction=0.5) as cap:
            render_final_video(...)
        audio_attn = cap.get("audio_to_video")  # float32 numpy [Tv]
        text_attn  = cap.get("text_to_video")   # float32 numpy [Tv]

    Memory note: only one transformer block is hooked (the middle block by
    default) to stay within H100 VRAM budget.  All intermediate tensors are
    freed immediately inside the hook.
    """

    def __init__(self, pipeline, block_fraction: float = 0.5):
        """
        Args:
            pipeline: The RetakePipeline instance.
            block_fraction: Which block to hook.  0.5 = middle block.
        """
        self._pipeline = pipeline
        self._block_fraction = block_fraction
        self._hooks: list = []
        self._storage: dict[str, Any] = {}
        self._ltx_model = None

    # ------------------------------------------------------------------
    def _find_ltx_model(self):
        """Find the LTXModel (has .transformer_blocks) inside the pipeline."""
        for _, m in self._pipeline.named_modules():
            if hasattr(m, "transformer_blocks"):
                return m
        return None

    def _make_hook(self, key: str):
        storage = self._storage

        def hook(module, args, kwargs, output):
            # args[0] = x (query source), kwargs['context'] = key/value source
            context = kwargs.get("context")
            if context is None:
                return  # self-attention — skip
            x = args[0]
            with torch.no_grad():
                try:
                    dtype = module.to_q.weight.dtype
                    x_f = x.detach().to(dtype=dtype)
                    c_f = context.detach().to(dtype=dtype)

                    q = module.q_norm(module.to_q(x_f))   # [B, Tq, H*dh]
                    k = module.k_norm(module.to_k(c_f))   # [B, Tk, H*dh]

                    B, Tq, D = q.shape
                    h = module.heads
                    dh = D // h
                    scale = math.sqrt(dh)

                    q_r = q.view(B, Tq, h, dh).permute(0, 2, 1, 3).float()  # [B,h,Tq,dh]
                    k_r = k.view(B, k.shape[1], h, dh).permute(0, 2, 1, 3).float()

                    # Full attention matrix → per-video-token attention mass
                    # [B,h,Tq,Tk] softmax over Tk → sum over Tk is always 1.
                    # We want: for each Tq, how much attention does it have overall.
                    # Use the row-sum (=1 by softmax) — so just use the *logit spread*
                    # instead: take max logit per query as attention "intensity".
                    logits = (q_r @ k_r.transpose(-2, -1)) / scale  # [B,h,Tq,Tk]
                    # max logit → proxy for "how strongly does this video token
                    # align with ANY context token"
                    max_logit = logits.max(dim=-1).values.float()  # [B, h, Tq]
                    # Softmax over the query positions → relative importance
                    importance = max_logit.softmax(dim=-1).mean(dim=[0, 1]).cpu()  # [Tq]

                    prev = storage.get(key)
                    storage[key] = importance if prev is None else prev + importance
                    storage[f"{key}_n"] = storage.get(f"{key}_n", 0) + 1
                    storage[f"{key}_Tq"] = int(Tq)

                    del q, k, q_r, k_r, logits, max_logit, importance
                except Exception:
                    pass  # silent fail — visualization is best-effort

        return hook

    # ------------------------------------------------------------------
    def __enter__(self) -> "LTXAttentionCapture":
        self._storage.clear()
        self._hooks.clear()

        ltx = self._find_ltx_model()
        if ltx is None:
            log.warning("LTXAttentionCapture: could not find LTXModel in pipeline — skipping hooks.")
            return self
        self._ltx_model = ltx

        n_blocks = len(ltx.transformer_blocks)
        block_idx = int(n_blocks * self._block_fraction)
        block_idx = max(0, min(block_idx, n_blocks - 1))
        block = ltx.transformer_blocks[block_idx]

        log.info("LTXAttentionCapture: hooking block %d / %d", block_idx, n_blocks)

        for attn_name, storage_key in [
            ("audio_to_video_attn", "audio_to_video"),
            ("attn2",               "text_to_video"),
        ]:
            attn_module = getattr(block, attn_name, None)
            if attn_module is None:
                log.warning("Block %d has no attribute '%s' — skipping.", block_idx, attn_name)
                continue
            handle = attn_module.register_forward_hook(
                self._make_hook(storage_key),
                with_kwargs=True,
            )
            self._hooks.append(handle)

        return self

    def __exit__(self, *_):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ------------------------------------------------------------------
    def get(self, key: str) -> np.ndarray | None:
        """Return averaged attention map for *key* as float32 numpy [Tq]."""
        raw = self._storage.get(key)
        if raw is None:
            return None
        n = max(self._storage.get(f"{key}_n", 1), 1)
        return (raw / n).numpy().astype(np.float32)

    def reshape_to_video_grid(
        self,
        key: str,
        cached_video_latent: torch.Tensor,
    ) -> np.ndarray | None:
        """Reshape 1-D attention vector to [T_l, H_l, W_l] using latent shape.

        Args:
            key: Storage key ("audio_to_video" or "text_to_video").
            cached_video_latent: Source video latent tensor.  Shape can be
                [1, C, T_l, H_l, W_l] or [1, C, T_l*H_l*W_l].

        Returns:
            float32 numpy [T_l, H_l, W_l] normalised to [0,1], or None.
        """
        attn = self.get(key)
        if attn is None:
            return None

        try:
            shape = cached_video_latent.shape  # infer spatial grid
            if len(shape) == 5:  # [1, C, T_l, H_l, W_l]
                T_l, H_l, W_l = int(shape[2]), int(shape[3]), int(shape[4])
            else:
                return attn[np.newaxis, np.newaxis, :]  # fallback: keep 1-D

            expected = T_l * H_l * W_l
            if attn.shape[0] != expected:
                log.warning(
                    "LTX attention Tq=%d ≠ T_l*H_l*W_l=%d — cannot reshape.", attn.shape[0], expected
                )
                return attn[np.newaxis, np.newaxis, :]

            grid = attn.reshape(T_l, H_l, W_l)
            # Normalise to [0, 1]
            mn, mx = grid.min(), grid.max()
            if mx > mn:
                grid = (grid - mn) / (mx - mn)
            return grid

        except Exception:
            log.warning("LTXAttentionCapture.reshape_to_video_grid failed", exc_info=True)
            return None
