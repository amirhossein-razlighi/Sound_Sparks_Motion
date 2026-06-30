"""Trainable-LoRA capacity control for the rebuttal ablation.

Injects PEFT LoRA adapters into the frozen LTX-2 DiT transformer so the SAME
Qwen-critic optimization can tune *extra free parameters* (LoRA) instead of the
text/audio conditioning latents. Used to answer the reviewers' question of
whether the gain is just added capacity: if a capacity-matched LoRA — optimized
with the same critic, losses, iterations, and LR — cannot reproduce the edit,
the audio-conditioning pathway is doing something a generic free-parameter
bump cannot.

No main code is modified. We build the DiT once, wrap it with LoRA, and
monkey-patch ``pipeline.model_ledger.transformer`` to return our wrapped
instance on every render — the same patching style used by
``core.render_with_injected_latents`` for the encoders. Because the fp8 policy
only monkey-patches ``nn.Linear.forward`` (it does not replace the Linear
class), PEFT attaches a normal trainable bf16/fp32 LoRA branch in parallel to
the frozen fp8 base weights.
"""
from __future__ import annotations

import logging

import torch

log = logging.getLogger(__name__)

# Attention Q/K/V and output projections inside every transformer block — the
# same targets the project's own LoRA trainer uses by default.
DEFAULT_LORA_TARGETS = ("to_q", "to_k", "to_v", "to_out.0")


def inject_lora_into_pipeline(
    pipeline,
    *,
    rank: int,
    alpha: int,
    dropout: float = 0.0,
    target_modules=None,
):
    """Build the DiT once, wrap it with LoRA, and patch the ledger getter.

    Returns ``(peft_transformer, trainable_params, restore_fn)``. Call
    ``restore_fn()`` to put the original ``model_ledger.transformer`` getter
    back when finished.
    """
    from peft import LoraConfig, get_peft_model

    targets = list(target_modules) if target_modules else list(DEFAULT_LORA_TARGETS)

    # Build the transformer once (the ledger otherwise builds a fresh instance
    # per render, which would discard our adapters).
    base_transformer = pipeline.model_ledger.transformer()

    lora_cfg = LoraConfig(
        r=int(rank),
        lora_alpha=int(alpha),
        lora_dropout=float(dropout),
        target_modules=targets,
        init_lora_weights=True,  # lora_B = 0 -> starts as a no-op (== base model)
    )
    peft_transformer = get_peft_model(base_transformer, lora_cfg)

    # PEFT initialises adapter weights in the *base layer's* dtype. Under the
    # fp8-cast policy the base Linear weights are float8_e4m3fn, which has no
    # matmul kernel ("addmm_cuda not implemented for Float8_e4m3fn") and cannot be
    # optimized. Cast the trainable LoRA params to the model's compute dtype (the
    # activations entering lora_A are bf16), so the parallel branch runs and Adam
    # can update it.
    # Detect the model's compute dtype from a *frozen base* float param (e.g. a
    # norm), NOT from an adapter param — the activations run in this dtype.
    compute_dtype = torch.bfloat16
    for p in peft_transformer.parameters():
        if (not p.requires_grad) and p.dtype in (torch.bfloat16, torch.float16, torch.float32):
            compute_dtype = p.dtype
            break
    for p in peft_transformer.parameters():
        if p.requires_grad and p.dtype != compute_dtype:
            p.data = p.data.to(compute_dtype)
    log.info("LoRA adapter compute dtype: %s", compute_dtype)

    # PEFT freezes the base and marks only LoRA params trainable.
    trainable = [p for p in peft_transformer.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in peft_transformer.parameters())
    log.info(
        "LoRA injected: rank=%d alpha=%d dropout=%.2f targets=%s  "
        "trainable=%.3fM / %.1fM params (%.4f%%)",
        rank, alpha, dropout, targets,
        n_train / 1e6, n_total / 1e6, 100.0 * n_train / max(n_total, 1),
    )
    if n_train == 0:
        raise RuntimeError(
            f"LoRA injection produced 0 trainable params — target_modules {targets} "
            "matched nothing in the transformer."
        )

    # CRITICAL: RetakePipeline.__call__ runs `transformer.requires_grad_(False)` on
    # every render (it normally optimizes only the input latents, so it freezes the
    # net). That call would also freeze our LoRA params -> zero grad. Override the
    # method on our instance so it always keeps LoRA params trainable while still
    # freezing the base, regardless of what the pipeline requests.
    def _requires_grad_keep_lora(mode: bool = True):
        for name, p in peft_transformer.named_parameters():
            p.requires_grad_(True if "lora_" in name else bool(mode))
        return peft_transformer

    peft_transformer.requires_grad_ = _requires_grad_keep_lora

    # Every render goes through model_ledger.transformer(); return our adapted
    # instance instead of building a fresh (adapter-less) one each call.
    orig_getter = pipeline.model_ledger.transformer
    pipeline.model_ledger.transformer = lambda: peft_transformer

    def restore() -> None:
        pipeline.model_ledger.transformer = orig_getter

    return peft_transformer, trainable, restore


def lora_state_dict(peft_transformer) -> dict:
    """Return just the LoRA adapter weights (small), on CPU."""
    from peft import get_peft_model_state_dict

    sd = get_peft_model_state_dict(peft_transformer)
    return {k: v.detach().cpu().clone() for k, v in sd.items()}


def set_lora_state(peft_transformer, state: dict | None) -> None:
    """Load LoRA adapter weights back into the transformer (in place)."""
    if not state:
        return
    from peft import set_peft_model_state_dict

    device = next(peft_transformer.parameters()).device
    set_peft_model_state_dict(
        peft_transformer,
        {k: v.to(device) for k, v in state.items()},
    )


def save_lora(peft_transformer, path) -> None:
    torch.save(lora_state_dict(peft_transformer), str(path))


__all__ = [
    "DEFAULT_LORA_TARGETS",
    "inject_lora_into_pipeline",
    "lora_state_dict",
    "set_lora_state",
    "save_lora",
]
