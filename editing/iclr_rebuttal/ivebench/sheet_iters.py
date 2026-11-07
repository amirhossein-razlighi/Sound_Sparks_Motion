#!/usr/bin/env python3
"""Per-run iteration sheet: rows = input, Retake baseline, every saved best-so-far iteration (iter_005, 010, ...), final ours;
columns = 9 frames spread over the clip (0, K-1, then 7 evenly spaced up to the last) + jump score. Extracts the iteration
mp4s from run_full.tar into <extract_dir> (kept, so a picked iteration can be copied out).
  sheet_iters.py <name> <cfg> <out.jpg> <extract_dir>"""
import json, subprocess, sys
from pathlib import Path
import av, numpy as np
from PIL import Image, ImageDraw
R = Path("/scratch/amirrz/iclr_rebuttal/ivebench"); name, cfg, outp, ex = sys.argv[1:5]
d = R / "runs" / name / cfg; ex = Path(ex); ex.mkdir(parents=True, exist_ok=True)
if (d / "run_full.tar").exists():
    subprocess.run(["tar", "-xf", str(d / "run_full.tar"), "-C", str(ex), "--wildcards", "*/visualizations/iter_*/best_optimized_video_both.mp4"], check=False)
its = sorted(ex.glob("*/mode_both/visualizations/iter_*/best_optimized_video_both.mp4"))
K = json.load(open(d / "run_config.json"))["args"]["retake_start_frames"]
def load(p):
    c = av.open(str(p)); f = [x.to_ndarray(format="rgb24") for x in c.decode(video=0)]; c.close(); return f
rows = [("input", R / "inputs" / f"{name}.mp4"), ("retake", d / "baseline_video.mp4")] + [(p.parent.name, p) for p in its] + [("FINAL ours", d / "best_optimized_video_both.mp4")]
tw, th = 176, 99; cols = 9
sheet = Image.new("RGB", (cols * tw + 120, len(rows) * th), "black"); dr = ImageDraw.Draw(sheet)
for r, (lab, p) in enumerate(rows):
    fr = load(p); T = len(fr)
    g = np.stack([np.asarray(Image.fromarray(x).convert("L").resize((128, 72)), np.float32) for x in fr])
    df = np.abs(np.diff(g, axis=0)).mean((1, 2)); j = int(df.argmax()); score = float(df.max() / (np.median(df) + 1e-6))
    idx = [0, max(K - 1, 0)] + [int(round(x)) for x in np.linspace(K + 2, T - 1, 7)]
    for c, i in enumerate(idx):
        sheet.paste(Image.fromarray(fr[min(i, T - 1)]).resize((tw, th)), (120 + c * tw, r * th)); dr.text((122 + c * tw, r * th + 2), f"f{i}", fill="yellow")
    dr.text((4, r * th + 30), lab, fill="white"); dr.text((4, r * th + 50), f"jump {score:.1f}x@{j}", fill="orange")
    print(f"{lab:12s} jump={score:.1f}x@f{j}  {p}")
sheet.save(outp, quality=85); print(outp)
