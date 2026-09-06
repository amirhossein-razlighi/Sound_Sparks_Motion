#!/usr/bin/env python3
"""Critic sanity check: score existing mp4s with our Qwen motion rubric.

    VIDEOS="a.mp4 b.mp4" EDIT_PROMPT="A red rose blooming." \
      /scratch/amirrz/H3_exp/venv/bin/python h3_probe/score_videos.py
Prints the 4-window dithered mean nll and yes-prob per video. Used to verify
the critic is sensible on H3 outputs (e.g. a visibly blooming rose scoring
yes=0.04 would indicate a critic blind spot for slow/gradual motion).
"""
import logging
import math
import os
import sys

import cv2
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(_HERE + "/../editing/src"))
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
log = logging.getLogger("score")

QWEN = "/project/def-amahdavi/amirrz/HF/models/Qwen2.5-VL-7B-Instruct"
VIDEOS = os.environ["VIDEOS"].split()
EDIT_PROMPT = os.environ["EDIT_PROMPT"]


def read_video(path):
    cap = cv2.VideoCapture(path)
    fr = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(fr).astype(np.float32) / 255.0  # [F,H,W,3]


def main():
    from motion_opt.qwen_loss import build_qwen_model, build_qwen_rubric_inputs, compute_qwen_video_loss
    dev = torch.device("cuda")
    qwen, proc = build_qwen_model(QWEN, device=dev, gradient_checkpointing=False)
    ci, yes_id, no_id = build_qwen_rubric_inputs(processor=proc, edit_prompt=EDIT_PROMPT, num_frames=24,
                                                 img_size=224, device=dev, motion_question=None,
                                                 gradient_rubric="motion")
    windows = [("linspace", 0), ("contiguous", 0), ("contiguous", 50), ("contiguous", 100)]
    ov = {"motion": 1.0, "entities": 0.0, "overall": 0.0}
    print(f"\nEDIT_PROMPT = {EDIT_PROMPT!r}")
    for v in VIDEOS:
        vid = read_video(v)
        fr = torch.from_numpy(vid).permute(0, 3, 1, 2).float().clamp(0, 1).to(dev)
        acc = 0.0
        with torch.no_grad():
            for mode, start in windows:
                if start >= fr.shape[0]:
                    start = max(fr.shape[0] - 24, 0)
                loss, _ = compute_qwen_video_loss(frames_chw=fr, qwen_model=qwen, cached_inputs=ci,
                                                  yes_token_id=yes_id, no_token_id=no_id, max_frames=24,
                                                  img_size=224, backward=False, return_details=True,
                                                  sample_mode=mode, contiguous_start_frame=start,
                                                  rubric_weight_overrides=ov)
                acc += float(loss)
        nll = acc / len(windows)
        print(f"{os.path.basename(os.path.dirname(v))}/{os.path.basename(v)}  frames={fr.shape[0]}  "
              f"nll={nll:.4f}  yes={math.exp(-nll):.4f}")
    print("SCORE DONE")


if __name__ == "__main__":
    main()
