#!/usr/bin/env python3
"""Phase-1 smoke test: stock Ovi T2V+audio generation, one clip.

Validates: weights load, engine runs end-to-end on one GPU, output MP4 with
audio is written. Run from ovi_probe/Ovi (their configs are cwd-relative):

    cd ovi_probe/Ovi && python ../smoke_t2v.py
"""
import logging
import os
import sys
import time

import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/Ovi")

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

from ovi.ovi_fusion_engine import OviFusionEngine  # noqa: E402
from ovi.utils.io_utils import save_video  # noqa: E402

CKPT_DIR = "/scratch/amirrz/Ovi_exp/ckpts"
OUT_DIR = "/scratch/amirrz/Ovi_exp/outputs/smoke"

PROMPT = (
    "A man gently pets a golden retriever dog sitting beside him on a porch. "
    "The dog wags its tail happily. "
    "<AUDCAP>Soft rustling of fur, a dog panting contentedly, calm ambient outdoor sounds.<ENDAUDCAP>"
)

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    config = OmegaConf.create({
        "ckpt_dir": CKPT_DIR,
        "model_name": "720x720_5s",
        "mode": "t2v",
        "cpu_offload": False,
        "fp8": False,
        "qint8": False,
    })

    t0 = time.time()
    engine = OviFusionEngine(config=config, device=0, target_dtype=torch.bfloat16)
    print(f"[smoke] engine loaded in {time.time()-t0:.1f}s, "
          f"VRAM {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

    t0 = time.time()
    result = engine.generate(
        text_prompt=PROMPT,
        image_path=None,
        video_frame_height_width=[512, 992],
        seed=42,
        solver_name="unipc",
        sample_steps=50,
        video_guidance_scale=4.0,
        audio_guidance_scale=3.0,
        slg_layer=11,
        video_negative_prompt="jitter, bad hands, blur, distortion",
        audio_negative_prompt="robotic, muffled, echo, distorted",
    )
    if result is None:
        print("[smoke] FAILED: generate returned None (see traceback above)", flush=True)
        sys.exit(1)

    video, audio, _ = result
    print(f"[smoke] generated in {time.time()-t0:.1f}s  video {video.shape}  audio {audio.shape}", flush=True)

    out = os.path.join(OUT_DIR, "smoke_t2v_seed42.mp4")
    save_video(out, video, audio, sample_rate=16000, fps=24)
    print(f"[smoke] SUCCESS -> {out}", flush=True)


if __name__ == "__main__":
    main()
