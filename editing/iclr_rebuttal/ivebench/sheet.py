#!/usr/bin/env python3
"""Inspection sheet for one IVEBench item: rows = input, then per config (Retake baseline, ours); columns = frames
0, K-1, K+2, 25%, 50%, 75%, last, and the two frames around the largest frame-to-frame change of that row's video.
Also prints per-video: max |frame diff| / median (jump score) and its frame index.  sheet.py <name> <out.jpg> [cfgs]"""
import json, sys
from pathlib import Path
import av, numpy as np
from PIL import Image, ImageDraw
R = Path("/scratch/amirrz/iclr_rebuttal/ivebench"); name, outp = sys.argv[1], sys.argv[2]
cfgs = sys.argv[3].split(",") if len(sys.argv) > 3 else sorted(p.name for p in (R / "runs" / name).iterdir() if (p / "best_optimized_video_both.mp4").exists())
def load(p):
    c = av.open(str(p)); f = [x.to_ndarray(format="rgb24") for x in c.decode(video=0)]; c.close(); return f
rows = [("input", load(R / "inputs" / f"{name}.mp4"), 5)]
for cfg in cfgs:
    d = R / "runs" / name / cfg; K = json.load(open(d / "run_config.json"))["args"]["retake_start_frames"]
    rows += [(f"{cfg} retake", load(d / "baseline_video.mp4"), K), (f"{cfg} OURS", load(d / "best_optimized_video_both.mp4"), K)]
tw, th = 192, 108; cols = 9
sheet = Image.new("RGB", (cols * tw + 110, len(rows) * th), "black"); dr = ImageDraw.Draw(sheet)
for r, (lab, fr, K) in enumerate(rows):
    T = len(fr); g = np.stack([np.asarray(Image.fromarray(x).convert("L").resize((128, 72)), np.float32) for x in fr])
    df = np.abs(np.diff(g, axis=0)).mean((1, 2)); j = int(df.argmax()); score = float(df.max() / (np.median(df) + 1e-6))
    idx = [0, max(K - 1, 0), min(K + 2, T - 1), T // 4, T // 2, 3 * T // 4, T - 1, j, min(j + 1, T - 1)]
    for c, i in enumerate(idx):
        sheet.paste(Image.fromarray(fr[i]).resize((tw, th)), (110 + c * tw, r * th))
        dr.text((112 + c * tw, r * th + 2), f"f{i}", fill="yellow")
    dr.text((4, r * th + 30), lab, fill="white"); dr.text((4, r * th + 50), f"jump {score:.1f}x@{j}", fill="orange")
    print(f"{lab:12s} T={T} K={K} jump={score:.1f}x at f{j}->{j+1}")
sheet.save(outp, quality=85); print(outp)
