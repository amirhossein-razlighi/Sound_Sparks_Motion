#!/usr/bin/env python3
"""Transfer sample clips: target | prompt-only Retake on target | transferred latent (top-6 and bottom-2 by held-out gain)."""
import csv, statistics as st, subprocess
from pathlib import Path
F = "/cvmfs/soft.computecanada.ca/gentoo/2023/x86-64-v3/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"
R = Path("/scratch/amirrz/iclr_rebuttal"); out = R / "samples_transfer"; out.mkdir(exist_ok=True)
rows = list(csv.DictReader(open(R / "transfer/pairs.csv")))
def g(r):
    v = [float(r[f"{m}_gain"]) for m in ("qwen25", "qwen3vl", "internvl3") if r.get(f"{m}_gain") not in ("", None, "None")]
    return st.mean(v) if v else -9
rows.sort(key=g, reverse=True)
pick = [("top", r) for r in rows[:6]] + [("low", r) for r in rows[-2:]]
tg = {(t["latent"], t["target"]): t for t in csv.DictReader(open(R / "transfer/targets.csv"))}
for k, (tag, r) in enumerate(pick, 1):
    t = tg[(r["latent"], r["target"])]
    sc = lambda kind: " ".join(f"{m[:5]} {float(r[f'{m}_{kind}']):.2f}" for m in ("qwen25", "qwen3vl", "internvl3") if r.get(f"{m}_{kind}") not in ("", None, "None"))
    vids = [(t["target_video"], "target video"), (r["baseline_video"], "prompt-only Retake  " + sc("baseline")), (r["transfer_video"], "TRANSFERRED latent  " + sc("transfer"))]
    prompt = r["edit_prompt"].replace("'", "").replace(":", " ")
    a, fl = [], []
    for i, (v, l) in enumerate(vids):
        a += ["-i", str(v)]
        fl.append(f"[{i}:v]fps=25,scale=512:320:force_original_aspect_ratio=decrease,pad=512:320:(ow-iw)/2:(oh-ih)/2,setsar=1,"
                  f"drawtext=fontfile={F}:text='{l}':x=6:y=6:fontsize=15:fontcolor=white:box=1:boxcolor=black@0.6[v{i}]")
    fl.append(f"[v0][v1][v2]hstack=inputs=3,pad=iw:ih+30:0:30:black,drawtext=fontfile={F}:text='{tag}  latent={r['latent']}  ->  {prompt}':x=8:y=6:fontsize=17:fontcolor=yellow[o]")
    subprocess.run(["ffmpeg", "-v", "error", "-y", *a, "-filter_complex", ";".join(fl), "-map", "[o]", "-t", "3.6", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "22", str(out / f"{k:02d}_{tag}_{r['latent']}__{r['target']}.mp4")], check=True)
print(sorted(p.name for p in out.glob("*.mp4")))
