"""Per-frame LTX cross-attention visualization helpers.

This module intentionally lives next to, rather than inside, ``attn_vis.py``.
The optimization-time visualization averages attention over latent time and
logs compact videos.  These helpers keep the temporal axis so ablations can
inspect what audio/text cross-attention emphasizes in each rendered frame.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ltx_core.model.transformer.rope import apply_rotary_emb

from .attn_vis import _plasma, overlay_heatmap

log = logging.getLogger(__name__)


class PerFrameLTXAttentionCapture:
    """Capture audio/text cross-attention as per-video-token importance.

    The LTX retake pipeline constructs a fresh transformer inside every render.
    Like ``LTXAttentionCapture``, this context manager patches the pipeline's
    transformer factory, registers hooks once the model exists, and restores the
    original factory afterward.

    For each hooked cross-attention module we recompute the attention logits
    from the hook inputs and reduce them to one scalar per video query token.
    The scalar can be:

    - ``output_norm``: L2 norm of the exact attention output before output
      projection. This is the default and is a useful proxy for how strongly
      the conditioning modality updates each video token.
    - ``max_prob``: largest softmax probability over context tokens.
    - ``neg_entropy``: concentration of the softmax distribution.
    - ``max_logit``: maximum scaled query-key logit.

    The stored vector is averaged over selected blocks, denoising calls, batch,
    and heads, but *not* over video time.
    """

    def __init__(
        self,
        pipeline,
        block_fractions: tuple[float, ...] = (0.5,),
        importance: str = "output_norm",
        query_chunk_size: int = 1024,
    ) -> None:
        if importance not in {"output_norm", "max_prob", "neg_entropy", "max_logit"}:
            raise ValueError(
                "importance must be one of: output_norm, max_prob, neg_entropy, max_logit"
            )
        self._pipeline = pipeline
        self._block_fractions = block_fractions
        self._importance = importance
        self._query_chunk_size = max(int(query_chunk_size), 1)
        self._hooks: list[Any] = []
        self._storage: dict[str, Any] = {}
        self._orig_transformer_fn = None
        self._hook_error_logged: set[str] = set()
        self.hooked_blocks: list[int] = []

    def _reduce_attention(
        self,
        module,
        x: torch.Tensor,
        context: torch.Tensor,
        mask: torch.Tensor | None,
        pe: torch.Tensor | None,
        k_pe: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return a CPU float vector [Tq] for one attention call."""
        with torch.no_grad():
            q = module.q_norm(module.to_q(x.detach()))
            k = module.k_norm(module.to_k(context.detach()))

            if pe is not None:
                q = apply_rotary_emb(q, pe, module.rope_type)
                k = apply_rotary_emb(k, pe if k_pe is None else k_pe, module.rope_type)

            B, Tq, D = q.shape
            heads = int(module.heads)
            dim_head = D // heads
            scale = math.sqrt(dim_head)

            q = q.view(B, Tq, heads, dim_head).permute(0, 2, 1, 3).float()
            k = k.view(B, k.shape[1], heads, dim_head).permute(0, 2, 1, 3).float()
            v = None
            if self._importance == "output_norm":
                v = module.to_v(context.detach())
                v = v.view(B, v.shape[1], heads, dim_head).permute(0, 2, 1, 3).float()

            mask_f = None
            if mask is not None:
                mask_f = mask.detach().float()
                if mask_f.ndim == 2:
                    mask_f = mask_f.unsqueeze(0).unsqueeze(0)
                elif mask_f.ndim == 3:
                    mask_f = mask_f.unsqueeze(1)

            scores: list[torch.Tensor] = []
            for start in range(0, Tq, self._query_chunk_size):
                end = min(start + self._query_chunk_size, Tq)
                logits = (q[:, :, start:end] @ k.transpose(-2, -1)) / scale
                if mask_f is not None:
                    logits = logits + mask_f[..., start:end, :] if mask_f.shape[-2] == Tq else logits + mask_f

                if self._importance == "max_logit":
                    score = logits.max(dim=-1).values
                else:
                    probs = logits.softmax(dim=-1)
                    if self._importance == "max_prob":
                        score = probs.max(dim=-1).values
                    elif self._importance == "neg_entropy":
                        entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
                        denom = math.log(max(probs.shape[-1], 2))
                        score = 1.0 - entropy / denom
                    else:
                        assert v is not None
                        attn_out = probs @ v
                        score = attn_out.norm(dim=-1)
                scores.append(score.mean(dim=(0, 1)).detach().cpu())

            return torch.cat(scores, dim=0)

    def _make_hook(self, key: str):
        storage = self._storage

        def hook(module, args, kwargs, output):  # noqa: ARG001
            context = kwargs.get("context")
            if context is None and len(args) > 1 and torch.is_tensor(args[1]):
                context = args[1]
            if context is None:
                return
            x = args[0]
            try:
                importance = self._reduce_attention(
                    module=module,
                    x=x,
                    context=context,
                    mask=kwargs.get("mask"),
                    pe=kwargs.get("pe"),
                    k_pe=kwargs.get("k_pe"),
                )
                prev = storage.get(key)
                storage[key] = importance if prev is None else prev + importance
                storage[f"{key}_n"] = storage.get(f"{key}_n", 0) + 1
                storage[f"{key}_Tq"] = int(importance.shape[0])
            except Exception:
                if key not in self._hook_error_logged:
                    log.warning("Per-frame attention hook failed for %s", key, exc_info=True)
                    self._hook_error_logged.add(key)

        return hook

    def _register_hooks_on_model(self, ltx_model) -> None:
        if not hasattr(ltx_model, "transformer_blocks") and hasattr(ltx_model, "velocity_model"):
            ltx_model = ltx_model.velocity_model
        n_blocks = len(ltx_model.transformer_blocks)
        selected: list[int] = []
        for frac in self._block_fractions:
            idx = int(n_blocks * float(frac))
            idx = max(0, min(idx, n_blocks - 1))
            if idx not in selected:
                selected.append(idx)
        self.hooked_blocks = selected

        for block_idx in selected:
            block = ltx_model.transformer_blocks[block_idx]
            log.info("PerFrameLTXAttentionCapture: hooking block %d / %d", block_idx, n_blocks)
            for attn_name, storage_key in [
                ("audio_to_video_attn", "audio_to_video"),
                ("attn2", "text_to_video"),
            ]:
                attn_module = getattr(block, attn_name, None)
                if attn_module is None:
                    log.warning("Block %d has no '%s' - skipping.", block_idx, attn_name)
                    continue
                handle = attn_module.register_forward_hook(
                    self._make_hook(storage_key),
                    with_kwargs=True,
                )
                self._hooks.append(handle)

    def __enter__(self) -> "PerFrameLTXAttentionCapture":
        self._storage.clear()
        self._hooks.clear()
        self._hook_error_logged.clear()
        self.hooked_blocks = []

        ledger = getattr(self._pipeline, "model_ledger", None)
        if ledger is None:
            log.warning("PerFrameLTXAttentionCapture: pipeline has no model_ledger.")
            return self

        orig_fn = ledger.transformer
        capture = self

        def patched_transformer():
            model = orig_fn()
            capture._register_hooks_on_model(model)
            return model

        self._orig_transformer_fn = orig_fn
        ledger.transformer = patched_transformer
        return self

    def __exit__(self, *_):
        ledger = getattr(self._pipeline, "model_ledger", None)
        if ledger is not None and self._orig_transformer_fn is not None:
            ledger.transformer = self._orig_transformer_fn
            self._orig_transformer_fn = None
        for handle in self._hooks:
            try:
                handle.remove()
            except Exception:
                pass
        self._hooks.clear()

    def get_vector(self, key: str) -> np.ndarray | None:
        raw = self._storage.get(key)
        if raw is None:
            return None
        n = max(int(self._storage.get(f"{key}_n", 1)), 1)
        return (raw / n).numpy().astype(np.float32)

    def get_grid(self, key: str, cached_video_latent: torch.Tensor) -> np.ndarray | None:
        """Return normalized [latent_t, latent_h, latent_w] attention grid."""
        vec = self.get_vector(key)
        if vec is None:
            return None
        shape = cached_video_latent.shape
        if len(shape) != 5:
            log.warning("Cannot infer video grid from cached_video_latent shape %s", tuple(shape))
            return None
        latent_t, latent_h, latent_w = int(shape[2]), int(shape[3]), int(shape[4])
        expected = latent_t * latent_h * latent_w
        if vec.shape[0] != expected:
            log.warning(
                "%s attention length %d does not match latent grid %dx%dx%d=%d",
                key,
                vec.shape[0],
                latent_t,
                latent_h,
                latent_w,
                expected,
            )
            return None
        grid = vec.reshape(latent_t, latent_h, latent_w)
        mn = float(grid.min())
        mx = float(grid.max())
        if mx > mn:
            grid = (grid - mn) / (mx - mn)
        return grid.astype(np.float32)

    def counts(self) -> dict[str, int]:
        return {
            key[:-2]: int(value)
            for key, value in self._storage.items()
            if key.endswith("_n")
        }


def frames_to_uint8_np(frames_chw: torch.Tensor) -> np.ndarray:
    """[T, 3, H, W] float [0,1] -> [T, H, W, 3] uint8."""
    arr = frames_chw.detach().float().cpu().permute(0, 2, 3, 1).numpy()
    return (arr.clip(0.0, 1.0) * 255).astype(np.uint8)


def upsample_heatmap_to_frames(
    heatmap: np.ndarray,
    num_frames: int,
    height: int,
    width: int,
    normalize_per_frame: bool = True,
) -> np.ndarray:
    """Upsample [T', H', W'] heatmap to [T, H, W]."""
    hmap = np.asarray(heatmap, dtype=np.float32)
    if hmap.ndim == 2:
        hmap = hmap[np.newaxis]
    if hmap.ndim != 3:
        raise ValueError(f"heatmap must be [H,W] or [T,H,W], got {hmap.shape}")

    hmap_t = torch.from_numpy(np.ascontiguousarray(hmap))[None, None].float()
    if hmap_t.shape[2:] != torch.Size([num_frames, height, width]):
        hmap_t = F.interpolate(
            hmap_t,
            size=(num_frames, height, width),
            mode="trilinear",
            align_corners=False,
        )
    out = hmap_t[0, 0].numpy()
    if normalize_per_frame:
        mn = out.min(axis=(1, 2), keepdims=True)
        mx = out.max(axis=(1, 2), keepdims=True)
        out = (out - mn) / np.where((mx - mn) > 1e-6, mx - mn, 1.0)
    return out.astype(np.float32)


def save_per_frame_attention_pngs(
    *,
    frames_np: np.ndarray,
    heatmap_grid: np.ndarray,
    output_dir: Path,
    modality: str,
    every_k: int,
    alpha: float,
    metadata: dict[str, Any] | None = None,
) -> list[Path]:
    """Save overlay and raw heatmap PNGs every K rendered frames."""
    import imageio.v2 as imageio

    output_dir = Path(output_dir)
    overlay_dir = output_dir / modality / "overlay_png"
    heatmap_dir = output_dir / modality / "heatmap_png"
    raw_dir = output_dir / modality / "raw"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    np.save(raw_dir / f"{modality}_latent_grid.npy", np.asarray(heatmap_grid, dtype=np.float32))

    if metadata is not None:
        (output_dir / modality / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True)
        )

    num_frames, height, width, _ = frames_np.shape
    heatmap_full = upsample_heatmap_to_frames(heatmap_grid, num_frames, height, width)
    overlay_full = overlay_heatmap(frames_np, heatmap_full, alpha=alpha)
    color_full = (_plasma(heatmap_full) * 255).astype(np.uint8)

    saved: list[Path] = []
    every_k = max(int(every_k), 1)
    for frame_idx in range(0, num_frames, every_k):
        overlay_path = overlay_dir / f"frame_{frame_idx:04d}_{modality}_overlay.png"
        heatmap_path = heatmap_dir / f"frame_{frame_idx:04d}_{modality}_heatmap.png"
        imageio.imwrite(overlay_path, overlay_full[frame_idx])
        imageio.imwrite(heatmap_path, color_full[frame_idx])
        saved.extend([overlay_path, heatmap_path])

    return saved
