#!/usr/bin/env python3
"""TRILLIUM: per-run review sheet for per-item tuning (coordinator 19:05).
Rows = INPUT / BASELINE (Retake) / OURS; columns = frame 0, K (first free frame), mid, last, and the frame of the largest
frame-to-frame change in OURS (plus the frame before it); also prints the mean frame change after K for
input / baseline / ours (motion check for stop / still / slow-down edits). Writes <run_dir>/review.png and prints the spike frame + magnitudes.
  python trillium_review_sheet.py <name> <cfg>
"""
import json, sys
from pathlib import Path
import av, numpy as np
from PIL import Image, ImageDraw, ImageFont

R = Path("/scratch/amirrz/iclr_rebuttal/ivebench"); FONT = "/cvmfs/soft.computecanada.ca/gentoo/2023/x86-64-v3/usr/share/fonts/dejavu/DejaVuSans.ttf"
name, cfg = sys.argv[1], sys.argv[2]; d = R / "runs" / name / cfg


def dec(p):
    c = av.open(str(p)); f = np.stack([x.to_ndarray(format="rgb24") for x in c.decode(video=0)]); c.close(); return f


inp, base, ours = dec(R / "inputs" / f"{name}.mp4"), dec(d / "baseline_video.mp4"), dec(d / "best_optimized_video_both.mp4")
K = int(json.loads((d / "run_config.json").read_text())["args"].get("retake_start_frames", 5))
n = min(len(inp), len(base), len(ours))
ch = lambda v: np.r_[0, np.abs(np.diff(v[:n].astype(np.float32).mean(-1), axis=0)).mean((1, 2))]
co, cb, ci = ch(ours), ch(base), ch(inp)
sp = int(np.argmax(co[1:]) + 1)
cols = sorted({0, min(K, n - 1), n // 2, n - 1, max(sp - 1, 0), sp})
w, h = 320, 180; font = ImageFont.truetype(FONT, 13)
S = Image.new("RGB", (90 + w * len(cols), 3 * h), (30, 30, 30)); dr = ImageDraw.Draw(S)
for r, (lab, v) in enumerate((("INPUT", inp), ("BASELINE", base), ("OURS", ours))):
    dr.text((6, r * h + h // 2 - 8), lab, fill=(255, 255, 0), font=font)
    for c, t in enumerate(cols):
        im = Image.fromarray(v[min(t, len(v) - 1)]).resize((w, h)); di = ImageDraw.Draw(im)
        tag = f"{lab} f{t}" + (" K" if t == K else "") + (" SPIKE" if t == sp else "")
        di.rectangle([0, 0, w, 17], fill=(0, 0, 0)); di.text((3, 1), tag, fill=(255, 255, 255), font=font)
        S.paste(im, (90 + c * w, r * h))
S.save(d / "review.png")
print(json.dumps(dict(name=name, cfg=cfg, K=K, n=n, ours_spike_frame=sp, ours_spike=round(float(co[sp]), 2),
                      ours_median_change=round(float(np.median(co[1:])), 2), base_max_change=round(float(cb[1:].max()), 2),
                      input_max_change=round(float(ci[1:].max()), 2),
                      mean_change_after_K=dict(input=round(float(ci[K + 1:].mean()), 3), baseline=round(float(cb[K + 1:].mean()), 3),
                                               ours=round(float(co[K + 1:].mean()), 3)),
                      review=str(d / "review.png"))))
