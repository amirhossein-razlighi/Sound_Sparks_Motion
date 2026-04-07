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
from pathlib import Path
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
        heatmap: float32 array of shape [H', W'] or [T', H', W'].
                 Any spatial/temporal resolution — upsampled to match frames.
        alpha:   Heatmap opacity (0 = invisible, 1 = solid colour).

    Returns:
        [T, H, W, 3] uint8 blended frames.
    """
    T, H, W, _ = frames.shape

    hmap = np.asarray(heatmap, dtype=np.float32)
    if hmap.ndim == 2:
        # 2-D spatial map — add a dummy temporal dim so trilinear can handle it.
        # After upsampling the single "frame" is copied across all T output frames.
        hmap = hmap[np.newaxis]          # [1, H', W']
    elif hmap.ndim != 3:
        raise ValueError(f"heatmap must be 2-D or 3-D, got shape {hmap.shape}")
    # hmap is now [T', H', W']

    # Upsample to [T, H, W] via trilinear interpolation.
    # F.interpolate with mode="trilinear" expects [N, C, D, H, W].
    hmap_t = torch.from_numpy(np.ascontiguousarray(hmap)).unsqueeze(0).unsqueeze(0).float()
    # shape: [1, 1, T', H', W']
    if hmap_t.shape[2:] != torch.Size([T, H, W]):
        hmap_t = F.interpolate(hmap_t, size=(T, H, W), mode="trilinear", align_corners=False)
    hmap_np = hmap_t.squeeze(0).squeeze(0).numpy()  # [T, H, W]

    # Normalise per-frame so the brightest spot always uses the full colour range.
    hmin = hmap_np.min(axis=(1, 2), keepdims=True)
    hmax = hmap_np.max(axis=(1, 2), keepdims=True)
    safe_range = np.where((hmax - hmin) > 1e-6, hmax - hmin, 1.0)
    hmap_norm = (hmap_np - hmin) / safe_range  # [T, H, W] in [0, 1]

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
    save_dir: Path | str | None = None,
) -> None:
    """Overlay heatmap on frames, optionally save it locally, and log it to W&B.

    Args:
        wandb_run: Active wandb run (or None — no-op).
        tag: W&B media key prefix, e.g. "media/attention/both/qwen_optimized".
        frames_np: [T, H, W, 3] uint8 frames.
        heatmap: float32 [T, H, W] or [H, W] heatmap.
        step: W&B step index.
        caption: Caption string for the video.
        fps: Playback fps for the logged video.
        save_dir: Optional local directory for overlay MP4 + thumbnail PNG.
    """
    overlay = overlay_heatmap(frames_np, heatmap)  # [T, H, W, 3] uint8
    safe_name = tag.replace("/", "__")
    video_path = None
    thumb_path = None

    if save_dir is not None:
        try:
            import imageio.v2 as imageio

            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            video_path = save_dir / f"{safe_name}.mp4"
            thumb_path = save_dir / f"{safe_name}_thumb.png"
            imageio.mimsave(video_path, list(overlay), fps=fps, macro_block_size=1)
            imageio.imwrite(thumb_path, overlay[len(overlay) // 2])
            log.info("Saved attention overlay to %s", video_path)
        except Exception:
            log.warning("Failed to save attention overlay locally (tag=%s)", tag, exc_info=True)

    if wandb_run is None:
        return

    try:
        import wandb

        if video_path is not None and video_path.exists():
            video = wandb.Video(str(video_path), format="mp4", caption=caption)
        else:
            # Raw array fallback requires wandb[media]/moviepy.
            video_arr = overlay.transpose(0, 3, 1, 2)  # [T, 3, H, W]
            video = wandb.Video(video_arr, fps=fps, format="mp4", caption=caption)
        thumb = (
            wandb.Image(str(thumb_path), caption=caption)
            if thumb_path is not None and thumb_path.exists()
            else wandb.Image(overlay[len(overlay) // 2], caption=caption)
        )
        wandb_run.log(
            {
                tag: video,
                f"{tag}_thumb": thumb,
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


def _get_qwen_decoder_layers(qwen_model):
    """Return the Qwen text decoder layers across common Transformers layouts."""
    candidates = [
        ("model.language_model.layers", getattr(getattr(qwen_model, "model", None), "language_model", None)),
        ("model.layers", getattr(qwen_model, "model", None)),
        ("language_model.layers", getattr(qwen_model, "language_model", None)),
    ]
    for name, module in candidates:
        layers = getattr(module, "layers", None)
        if layers is not None:
            return layers, name
    raise AttributeError("Could not find Qwen decoder layers on model.language_model.layers or model.layers")


def _iter_qwen_attention_configs(qwen_model):
    """Yield unique config objects whose _attn_implementation controls Qwen text attention."""
    seen: set[int] = set()
    for cfg in (
        getattr(qwen_model, "config", None),
        getattr(getattr(qwen_model, "config", None), "text_config", None),
        getattr(getattr(qwen_model, "model", None), "config", None),
        getattr(getattr(getattr(qwen_model, "model", None), "language_model", None), "config", None),
    ):
        if cfg is None or not hasattr(cfg, "_attn_implementation"):
            continue
        if id(cfg) in seen:
            continue
        seen.add(id(cfg))
        yield cfg


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
    patch token, averaged over the last ``num_layers_to_avg`` decoder layers
    and all attention heads.

    Implementation notes
    --------------------
    Qwen2.5-VL defaults to ``_attn_implementation="sdpa"`` whose
    ``sdpa_attention_forward`` always returns ``None`` for attention weights —
    ``output_attentions=True`` is silently ignored.  We therefore:

    1. Temporarily switch to ``"eager"`` so the standard matmul path runs and
       returns real ``[B, H, S, S]`` tensors.
    2. Register forward hooks on only the last ``num_layers_to_avg`` decoder
       layers instead of using ``output_attentions=True``.  This avoids
       allocating all 28 full attention matrices (~3 GB) and keeps memory low.
    3. Each hook immediately reduces ``[B, H, S, S]`` → ``[num_vis_tokens]``
       and discards the rest.

    Returns dict with:
        "attn_spatial": float32 numpy [grid_t, grid_h, grid_w]
        "grid_thw":     (grid_t, grid_h, grid_w)
        "num_frames":   int
    or None on failure.
    """
    try:
        from .qwen_loss import _frames_to_pixel_values

        # ---- Prepare pixel values ----
        n = frames_chw.shape[0]
        num_frames = min(max_frames, n) if max_frames > 0 else n
        if num_frames % _TEMPORAL_PATCH_SIZE != 0:
            num_frames -= num_frames % _TEMPORAL_PATCH_SIZE
        num_frames = max(num_frames, _TEMPORAL_PATCH_SIZE)

        idx = torch.linspace(0, n - 1, num_frames, device=frames_chw.device).round().long()
        frames = F.interpolate(
            frames_chw[idx].detach(),
            size=(img_size, img_size),
            mode="bilinear",
            align_corners=False,
        )
        pixel_values_videos = _frames_to_pixel_values(frames).to(dtype=torch.bfloat16)

        # ---- Find visual token positions ----
        video_pad_id = _get_video_pad_token_id(qwen_model)
        input_ids_1d = cached_inputs["input_ids"][0]  # [seq_len]
        vis_positions = (input_ids_1d == video_pad_id).nonzero(as_tuple=True)[0]
        if vis_positions.numel() == 0:
            log.warning("No video pad tokens found in Qwen input_ids — skipping attn maps.")
            return None
        last_pos = int(input_ids_1d.shape[0]) - 1  # generation token position

        # ---- Register hooks on the last N decoder layers ----
        # Hooks capture self_attn output = (attn_output, attn_weights) under eager mode.
        # attn_weights: [B, num_heads, seq_len, seq_len]
        # We immediately slice out [0, :, last_pos, vis_positions] and average over heads.
        captured: list[torch.Tensor] = []

        def _make_hook(vis_pos, lpos):
            def hook(module, inputs, output):
                # output is (attn_output, attn_weights)
                if not isinstance(output, (tuple, list)) or len(output) < 2:
                    return
                attn_weights = output[1]
                if attn_weights is None:
                    return
                if attn_weights.ndim != 4:
                    log.debug("Unexpected Qwen attention weight shape: %s", tuple(attn_weights.shape))
                    return
                # [B, H, seq, seq] → [H, num_vis]
                a = attn_weights[0, :, lpos, :][:, vis_pos].float()
                captured.append(a.mean(dim=0).detach().cpu())  # [num_vis]
                del attn_weights, a
            return hook

        decoder_layers, decoder_layers_name = _get_qwen_decoder_layers(qwen_model)
        n_layers = len(decoder_layers)
        hook_indices = list(range(max(0, n_layers - num_layers_to_avg), n_layers))
        log.debug("Qwen attention hooks using %s indices %s", decoder_layers_name, hook_indices)
        hooks = []
        for i in hook_indices:
            h = decoder_layers[i].self_attn.register_forward_hook(
                _make_hook(vis_positions, last_pos)
            )
            hooks.append(h)

        # ---- Temporarily switch to eager attention ----
        # sdpa_attention_forward always returns None for attn_weights;
        # eager_attention_forward (lines 169-179 in modeling_qwen2_5_vl.py) returns them.
        attn_configs = list(_iter_qwen_attention_configs(qwen_model))
        orig_impls = [(cfg, cfg._attn_implementation) for cfg in attn_configs]
        was_gc = getattr(qwen_model, "is_gradient_checkpointing", False)
        try:
            for cfg, _ in orig_impls:
                cfg._attn_implementation = "eager"
            if was_gc:
                qwen_model.gradient_checkpointing_disable()

            qwen_model(
                input_ids=cached_inputs["input_ids"],
                attention_mask=cached_inputs["attention_mask"],
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=cached_inputs.get("video_grid_thw"),
                output_attentions=False,
                use_cache=False,
            )
        finally:
            for cfg, impl in orig_impls:
                cfg._attn_implementation = impl
            if was_gc:
                qwen_model.gradient_checkpointing_enable()
            for h in hooks:
                h.remove()

        if not captured:
            log.warning("Qwen attention hooks captured nothing — no weights available.")
            return None

        # ---- Average over captured layers ----
        attn_vis = torch.stack(captured, dim=0).mean(dim=0).numpy()  # [num_vis_tokens]

        # ---- Determine spatial grid ----
        if "video_grid_thw" in cached_inputs:
            thw = cached_inputs["video_grid_thw"][0]
            # Qwen stores video_grid_thw as the raw ViT patch grid. The LLM
            # sequence sees tokens after the 2x2 spatial merge.
            grid_t = int(thw[0])
            grid_h = int(thw[1]) // _MERGE_SIZE
            grid_w = int(thw[2]) // _MERGE_SIZE
        else:
            grid_t = num_frames // _TEMPORAL_PATCH_SIZE
            grid_h = img_size // (14 * _MERGE_SIZE)  # patch_size=14, merge_size=2 → 28
            grid_w = grid_h

        expected = grid_t * grid_h * grid_w
        if attn_vis.shape[0] != expected:
            log.warning(
                "Qwen vis-token count mismatch: got %d, expected grid %d×%d×%d=%d",
                attn_vis.shape[0], grid_t, grid_h, grid_w, expected,
            )
            return None

        attn_spatial = attn_vis.reshape(grid_t, grid_h, grid_w).astype(np.float32)
        torch.cuda.empty_cache()

        return {
            "attn_spatial": attn_spatial,   # [grid_t, grid_h, grid_w]
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
    """Context manager that captures LTX cross-attention during a render call.

    ``RetakePipeline`` instantiates the transformer fresh on every call via
    ``model_ledger.transformer()`` and deletes it afterwards — there is no
    persistent reference to hook from the outside.  This class works around
    that by **monkey-patching** ``model_ledger.transformer`` so that when the
    pipeline creates the model we immediately register forward hooks on the
    desired block, then restore the original method when the context exits.

    Hooks fire on every denoising step and accumulate a lightweight
    per-video-token attention score.  Only one block is hooked (middle by
    default) to stay within H100 VRAM budget.

    Usage::

        with LTXAttentionCapture(pipeline, block_fraction=0.5) as cap:
            render_final_video(...)   # hooks fire inside here
        audio_attn = cap.get("audio_to_video")  # float32 numpy [Tv]
        text_attn  = cap.get("text_to_video")   # float32 numpy [Tv]
    """

    def __init__(self, pipeline, block_fraction: float = 0.5):
        self._pipeline = pipeline
        self._block_fraction = block_fraction
        self._hooks: list = []
        self._storage: dict[str, Any] = {}
        self._orig_transformer_fn = None
        self._hook_error_logged: set[str] = set()

    # ------------------------------------------------------------------
    def _make_hook(self, key: str):
        storage = self._storage

        def hook(module, args, kwargs, output):
            context = kwargs.get("context")
            if context is None and len(args) > 1 and torch.is_tensor(args[1]):
                context = args[1]
            if context is None:
                return  # self-attention — skip
            x = args[0]
            with torch.no_grad():
                try:
                    x_f = x.detach()
                    c_f = context.detach()

                    q = module.q_norm(module.to_q(x_f))   # [B, Tq, H*dh]
                    k = module.k_norm(module.to_k(c_f))   # [B, Tk, H*dh]

                    B, Tq, D = q.shape
                    h = module.heads
                    dh = D // h
                    scale = math.sqrt(dh)

                    q_r = q.view(B, Tq, h, dh).permute(0, 2, 1, 3).float()
                    k_r = k.view(B, k.shape[1], h, dh).permute(0, 2, 1, 3).float()

                    logits = (q_r @ k_r.transpose(-2, -1)) / scale  # [B,h,Tq,Tk]
                    # Max logit per query → proxy for "how strongly does this
                    # video token align with any context token"
                    max_logit = logits.max(dim=-1).values.float()   # [B, h, Tq]
                    importance = max_logit.softmax(dim=-1).mean(dim=[0, 1]).cpu()  # [Tq]

                    prev = storage.get(key)
                    storage[key] = importance if prev is None else prev + importance
                    storage[f"{key}_n"] = storage.get(f"{key}_n", 0) + 1
                    storage[f"{key}_Tq"] = int(Tq)

                    del q, k, q_r, k_r, logits, max_logit, importance
                except Exception:
                    if key not in self._hook_error_logged:
                        log.warning("LTXAttentionCapture hook failed for %s", key, exc_info=True)
                        self._hook_error_logged.add(key)

        return hook

    def _register_hooks_on_model(self, ltx_model) -> None:
        # model_ledger.transformer() returns X0Model; the actual LTXModel
        # (which owns transformer_blocks) is nested at .velocity_model
        if not hasattr(ltx_model, "transformer_blocks") and hasattr(ltx_model, "velocity_model"):
            ltx_model = ltx_model.velocity_model
        n_blocks = len(ltx_model.transformer_blocks)
        block_idx = int(n_blocks * self._block_fraction)
        block_idx = max(0, min(block_idx, n_blocks - 1))
        block = ltx_model.transformer_blocks[block_idx]

        log.info("LTXAttentionCapture: hooking block %d / %d", block_idx, n_blocks)
        for attn_name, storage_key in [
            ("audio_to_video_attn", "audio_to_video"),
            ("attn2",               "text_to_video"),
        ]:
            attn_module = getattr(block, attn_name, None)
            if attn_module is None:
                log.warning("Block %d has no '%s' — skipping.", block_idx, attn_name)
                continue
            handle = attn_module.register_forward_hook(
                self._make_hook(storage_key),
                with_kwargs=True,
            )
            self._hooks.append(handle)

    # ------------------------------------------------------------------
    def __enter__(self) -> "LTXAttentionCapture":
        self._storage.clear()
        self._hooks.clear()
        self._hook_error_logged.clear()

        ledger = getattr(self._pipeline, "model_ledger", None)
        if ledger is None:
            log.warning("LTXAttentionCapture: pipeline has no model_ledger — skipping.")
            return self

        # Patch model_ledger.transformer so we can intercept the freshly-
        # created model before the denoising loop starts.
        orig_fn = ledger.transformer
        capture = self  # close over self

        def patched_transformer():
            model = orig_fn()
            capture._register_hooks_on_model(model)
            return model

        self._orig_transformer_fn = orig_fn
        ledger.transformer = patched_transformer
        return self

    def __exit__(self, *_):
        # Restore the original factory method
        ledger = getattr(self._pipeline, "model_ledger", None)
        if ledger is not None and self._orig_transformer_fn is not None:
            ledger.transformer = self._orig_transformer_fn
            self._orig_transformer_fn = None
        # Remove any hooks still attached (the model may already be deleted,
        # but remove() is a no-op in that case)
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
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
