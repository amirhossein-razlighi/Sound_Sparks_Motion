#!/usr/bin/env python3
"""Sequential (max_workers=1, xet-disabled) MiniMax-H3 download for the
thread-limited login node. Scoped to the Ref2VA diffusers layout — skips
FL2VA/, Ref2VA/ (SGLang copies) and the FL2VA `transformer/` (66 GB); if the
ModularPipeline later insists on `transformer/`, rerun with EXTRA=transformer.
Re-runnable: complete files are skipped.
"""
import logging
import os
import time

from huggingface_hub import snapshot_download

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

OUT = "/scratch/amirrz/H3_exp/ckpt"

patterns = [
    "model_index.json",
    "modular_model_index.json",
    "transformer_ref/*",
    "text_encoder/*",
    "vae/*",
    "audio_vae/*",
    "scheduler/*",
    "audio_scheduler/*",
    "processor/*",
    "tokenizer/*",
    "scripts/*",
    "docs/*",
    "LICENSE",
]
extra = os.environ.get("EXTRA")
if extra:
    patterns += [f"{e}/*" for e in extra.split(",")]

t0 = time.time()
snapshot_download(
    repo_id="MiniMaxAI/MiniMax-H3",
    local_dir=OUT,
    allow_patterns=patterns,
    max_workers=1,
)
logging.info("H3 DOWNLOAD COMPLETE in %.0fs", time.time() - t0)
