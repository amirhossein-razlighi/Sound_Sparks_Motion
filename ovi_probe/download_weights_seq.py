#!/usr/bin/env python3
"""Sequential (max_workers=1) Ovi weights download for thread-limited login nodes.

Mirrors Ovi/download_weights.py (720x720_5s only) but avoids the thread pool
that dies with "can't start new thread" under the login node's ulimit.
Re-runnable: snapshot_download skips files already complete.
"""
import logging
import time

from huggingface_hub import snapshot_download

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

OUT = "/scratch/amirrz/Ovi_exp/ckpts"

JOBS = [
    ("Wan-AI/Wan2.2-TI2V-5B", f"{OUT}/Wan2.2-TI2V-5B",
     ["google/*", "models_t5_umt5-xxl-enc-bf16.pth", "Wan2.2_VAE.pth"]),
    ("hkchengrex/MMAudio", f"{OUT}/MMAudio",
     ["ext_weights/best_netG.pt", "ext_weights/v1-16.pth"]),
    ("chetwinlow1/Ovi", f"{OUT}/Ovi",
     ["model.safetensors"]),
]

for repo_id, local_dir, patterns in JOBS:
    logging.info("Downloading %s -> %s", repo_id, local_dir)
    t0 = time.time()
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        allow_patterns=patterns,
        max_workers=1,
    )
    logging.info("DONE %s in %.1fs", repo_id, time.time() - t0)

logging.info("ALL DOWNLOADS COMPLETE")
