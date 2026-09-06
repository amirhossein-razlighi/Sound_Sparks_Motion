#!/usr/bin/env python3
"""Diagnostic: can Ovi+SDEdit do ANY editing (appearance), or is prompt
leverage absent in this regime?

Sweeps 3 sources x appearance prompts x strengths {0.5, 0.7, 0.85}, plus a
prompt-leverage control (same source+strength, wildly different prompt). If
appearance edits fail at every strength AND the wild prompt barely changes the
output, the SDEdit regime gives the text negligible leverage — explaining why
audio-latent optimization cannot summon motion, independent of the optimizer.

Run from ovi_probe/Ovi:  python ../appearance_test.py
"""
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE + "/Ovi")
sys.path.insert(0, _HERE)

import logging  # noqa: E402
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
log = logging.getLogger("appearance")

from sdedit_edit import (  # noqa: E402
    AUDIO_SR, FPS, encode_source, load_audio_16k, load_engine,
    load_video_frames, sdedit_generate,
)
from ovi.utils.io_utils import save_video  # noqa: E402

REPO = os.path.abspath(_HERE + "/..")
OUT = "/scratch/amirrz/Ovi_exp/outputs/appearance_test"

SOURCES = {
    "cook": (f"{REPO}/input_videos/cook_spinach_main.mp4",
             "The man wears a bright red shirt while cooking. "
             "<AUDCAP>Calm indoor kitchen ambience.<ENDAUDCAP>"),
    "dogchair": (f"{REPO}/input_videos/dog_sitting_on_chair_main.mp4",
                 "A black dog sitting on a chair. "
                 "<AUDCAP>Quiet room ambience.<ENDAUDCAP>"),
    "rose": (f"{REPO}/input_videos/a_red_rose_bud_in_a_green_main.mp4",
             "A blue rose bud in a green grass field. "
             "<AUDCAP>Gentle breeze, soft outdoor nature ambience.<ENDAUDCAP>"),
}
STRENGTHS = [float(x) for x in os.environ.get("STRENGTHS", "0.5 0.7 0.85").split()]
ONLY = os.environ.get("SOURCES_ONLY", "").split()  # subset of source names
# Leverage control: content completely unrelated to the source.
WILD_PROMPT = ("A campfire burning on a beach at sunset, waves in the background. "
               "<AUDCAP>Crackling fire, ocean waves.<ENDAUDCAP>")


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    engine = load_engine()

    for name, (src, prompt) in SOURCES.items():
        if ONLY and name not in ONLY:
            continue
        video = load_video_frames(src, target_area=engine.target_area)
        audio = load_audio_16k(src)
        z_vid, z_aud = encode_source(engine, video, audio)

        outputs = {}
        for st in STRENGTHS:
            t0 = time.time()
            with torch.no_grad():
                r = sdedit_generate(
                    engine, text_prompt=prompt, z_vid_src=z_vid, z_aud_start=z_aud,
                    strength=st, seed=42, sample_steps=30)
            vid = r["video"].cpu().float().numpy()
            outputs[st] = vid
            save_video(os.path.join(OUT, f"{name}_appearance_s{st:.2f}.mp4"),
                       vid, r["audio"], sample_rate=AUDIO_SR, fps=FPS)
            log.info("%s appearance s=%.2f done (%.1fs)", name, st, time.time() - t0)

        # Leverage control: same everything, wild prompt, at a reference strength.
        ref_st = 0.7 if 0.7 in outputs else STRENGTHS[len(STRENGTHS) // 2]
        with torch.no_grad():
            rw = sdedit_generate(
                engine, text_prompt=WILD_PROMPT, z_vid_src=z_vid, z_aud_start=z_aud,
                strength=ref_st, seed=42, sample_steps=30)
        wild = rw["video"].cpu().float().numpy()
        save_video(os.path.join(OUT, f"{name}_WILD_s{ref_st:.2f}.mp4"),
                   wild, rw["audio"], sample_rate=AUDIO_SR, fps=FPS)

        # Quantify prompt leverage: mean |difference| between the two renders
        # (identical source, noise, seed — ONLY the prompt differs).
        d = float(np.mean(np.abs(outputs[ref_st] - wild)))
        log.info("[LEVERAGE] %s: mean|appearance(%.2f) - wild(%.2f)| = %.4f "
                 "(range [-1,1]; ~0 => prompt has no leverage)", name, ref_st, ref_st, d)

        del outputs, wild, z_vid, z_aud, video, audio
        torch.cuda.empty_cache()

    log.info("APPEARANCE TEST DONE -> %s", OUT)


if __name__ == "__main__":
    main()
