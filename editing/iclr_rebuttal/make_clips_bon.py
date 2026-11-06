#!/usr/bin/env python3
"""Side-by-side clips source | OURS | best-of-3 pick (current selection) for all 12 scenarios."""
import csv, subprocess
from pathlib import Path
from common import TWELVE, COMPARISONS, SCENARIOS
F = "/cvmfs/soft.computecanada.ca/gentoo/2023/x86-64-v3/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"
R = Path("/scratch/amirrz/iclr_rebuttal"); out = R / "samples_input_ours_bon3"; out.mkdir(exist_ok=True)
sel = {r["scenario"]: r for r in csv.DictReader(open(R / "best_of_n/summary.csv"))}
for k, s in enumerate(TWELVE, 1):
    c = SCENARIOS[s][1]; prompt = open(COMPARISONS / c / "prompt.txt").read().strip().replace("'", "").replace(":", " ")
    src = sel[s]["bon3_src"]; lab = "best-of-3 = paper Retake baseline" if src == "paper_baseline" else f"best-of-3 = Retake {src.replace('_', ' ')}"
    vids = [(COMPARISONS / c / "input_video.mp4", "source"), (COMPARISONS / c / "ours.mp4", "OURS"), (R / "best_of_n/selected" / s / "bon3.mp4", lab)]
    a, fl = [], []
    for i, (v, l) in enumerate(vids):
        a += ["-i", str(v)]
        fl.append(f"[{i}:v]fps=25,scale=512:320,setsar=1,drawtext=fontfile={F}:text='{l}':x=6:y=6:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.6[v{i}]")
    fl.append(f"[v0][v1][v2]hstack=inputs=3,pad=iw:ih+30:0:30:black,drawtext=fontfile={F}:text='{s}  |  {prompt}':x=8:y=6:fontsize=18:fontcolor=yellow[o]")
    subprocess.run(["ffmpeg", "-v", "error", "-y", *a, "-filter_complex", ";".join(fl), "-map", "[o]", "-t", "3.56", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "22", str(out / f"{k:02d}_{s}.mp4")], check=True)
print("done", len(list(out.glob("*.mp4"))))
