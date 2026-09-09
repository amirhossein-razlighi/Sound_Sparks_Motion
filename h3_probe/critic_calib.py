#!/usr/bin/env python3
"""Critic sensitivity calibration on the goldfish iteration previews.

Scores a list of videos with the Qwen motion critic under several configurations
(frame size, crop, fp32 logit head, question) so we can pick one that (a) sees
the jump in fullmethod iter_02 / the H3 t2va reference clip and (b) keeps the
baseline low. 4 windows per video (linspace + 3 contiguous), no grad.

env: VIDEOS (";"-separated paths), OUT (json path), EDIT ("The goldfish jumps out of the fish tank into the air.")
"""
import json
import logging
import math
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(_HERE + "/../editing/src"))
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
log = logging.getLogger("calib")
QWEN = os.environ.get("H3_QWEN", "/project/def-amahdavi/amirrz/HF/models/Qwen2.5-VL-7B-Instruct")
VIDEOS = [v for v in os.environ["VIDEOS"].split(";") if v]
OUT = os.environ.get("OUT", "/scratch/amirrz/H3_exp/outputs/critic_calib.json")
EDIT = os.environ.get("EDIT", "The goldfish jumps out of the fish tank into the air.")
ALT = os.environ.get("ALT_EDIT", "A goldfish leaps out of the water into the air above the tank.")


class FP32Head(torch.nn.Module):
    """Only the last position is used by the loss: compute it in fp32 (bf16 logits quantize the nll to ~0.1 nat)."""

    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, x):
        w = self.head.weight
        b = self.head.bias
        return torch.nn.functional.linear(x[:, -1:].float(), w.float(), None if b is None else b.float())


def load_frames(path):
    import subprocess
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                            "-of", "csv=p=0", path], capture_output=True, text=True).stdout.strip().split(",")
    w, h = int(probe[0]), int(probe[1])
    raw = subprocess.run(["ffmpeg", "-v", "error", "-threads", "2", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True).stdout
    fr = np.frombuffer(raw, dtype=np.uint8).reshape(-1, h, w, 3)
    return torch.from_numpy(fr.copy()).permute(0, 3, 1, 2).float() / 255.0


def main():
    from motion_opt.qwen_loss import build_qwen_model, build_qwen_rubric_inputs, compute_qwen_video_loss
    dev = torch.device("cuda")
    qwen, proc = build_qwen_model(QWEN, device=dev, gradient_checkpointing=False)
    bf16_head = qwen.lm_head
    ov = {"motion": 1.0, "entities": 0.0, "overall": 0.0}
    NF = int(os.environ.get("FRAMES", "24"))
    wins = [("linspace", 0)] + [("contiguous", st) for st in (0, 20, 40, 60, 80, 100)]
    # configs: question variants x image configs, all with the fp32 head. Questions may use {edit_prompt}.
    QUESTIONS = {
        "q0_default": None,
        "q1_plain": 'Does this video show: "{edit_prompt}"? Answer only \'yes\' or \'no\'.',
        "q2_anypoint": "Does a goldfish jump out of the water at any point in this video? Answer only 'yes' or 'no'.",
        "q3_inair": "Is a goldfish ever in the air above the water surface in this video? Answer only 'yes' or 'no'.",
        "q4_leap": "Does a fish leap out of the fish tank in this video? Answer only 'yes' or 'no'.",
        "q5_brief": "Watch the whole clip. Does one of the goldfish jump up out of the water, even briefly? Answer only 'yes' or 'no'.",
        "q6_jumping": "Does this video show a goldfish jumping? Answer only 'yes' or 'no'.",
    }
    if os.environ.get("QUESTIONS_JSON"):
        QUESTIONS = json.load(open(os.environ["QUESTIONS_JSON"]))
    IMGS = {"224": (224, None)}
    if os.environ.get("WITH_CROP", "0") == "1":
        IMGS["448crop"] = (448, (0.15, 0.0, 0.85, 1.0))
    configs = [(f"{iname}_{qname}", img, crop, True, q) for qname, q in QUESTIONS.items() for iname, (img, crop) in IMGS.items()]
    from transformers import AutoProcessor
    procs = {}
    inputs = {}
    for name, img, crop, fp32, q in configs:
        if img not in procs:
            procs[img] = proc if img == 224 else AutoProcessor.from_pretrained(QWEN, min_pixels=img * img, max_pixels=img * img)
        if (q, img) not in inputs:
            inputs[(q, img)] = build_qwen_rubric_inputs(processor=procs[img], edit_prompt=EDIT, num_frames=NF, img_size=img, device=dev,
                                                        motion_question=q, gradient_rubric="motion")
    results = {}
    for vp in VIDEOS:
        fr = load_frames(vp).to(dev)
        F_, _, Hh, Ww = fr.shape
        row = {}
        for name, img, crop, fp32, q in configs:
            qwen.lm_head = FP32Head(bf16_head) if fp32 else bf16_head
            ci, yes_id, no_id = inputs[(q, img)]
            f = fr
            if crop is not None:
                x0, y0, x1, y1 = crop
                f = fr[:, :, int(y0 * Hh):int(y1 * Hh), int(x0 * Ww):int(x1 * Ww)]
            vals = []
            with torch.no_grad():
                for mode, start in wins:
                    loss, _ = compute_qwen_video_loss(frames_chw=f, qwen_model=qwen, cached_inputs=ci, yes_token_id=yes_id,
                                                      no_token_id=no_id, max_frames=NF, img_size=img, backward=False,
                                                      return_details=True, sample_mode=mode, contiguous_start_frame=start,
                                                      rubric_weight_overrides=ov)
                    vals.append(float(loss))
            pw = [math.exp(-v) for v in vals]
            row[name] = {"nll_4win": float(np.mean(vals)), "nll_lin": vals[0], "yes_4win": math.exp(-float(np.mean(vals))),
                         "yes_lin": math.exp(-vals[0]), "yes_windows": pw, "yes_any": 1 - float(np.prod([1 - x for x in pw[1:]])),
                         "yes_max": max(pw)}
            log.info("%-60s %-18s 4win nll=%.3f yes=%.5f | lin nll=%.3f yes=%.5f", os.path.basename(os.path.dirname(vp)) + "/" + os.path.basename(vp),
                     name, row[name]["nll_4win"], row[name]["yes_4win"], vals[0], math.exp(-vals[0]))
        results[vp] = row
        json.dump(results, open(OUT, "w"), indent=1)
    qwen.lm_head = bf16_head
    log.info("=== summary (yes 4-window) ===")
    names = [c[0] for c in configs]
    for key in ("yes_lin", "yes_max", "yes_any"):
        log.info("--- %s (windows: linspace + contiguous starts 0/20/40/60/80/100, %d frames)", key, NF)
        for n in names:
            log.info("%-22s " + " ".join(f"{results[vp][n][key]:9.5f}" for vp in results), n)
    for n in names:
        log.info("per-window yes %-14s " + " | ".join(" ".join(f"{x:.4f}" for x in results[vp][n]["yes_windows"]) for vp in results), n)
    log.info("columns = " + ", ".join(os.path.basename(vp) for vp in results))
    log.info("CALIB DONE")


if __name__ == "__main__":
    main()
