#!/usr/bin/env python3
"""Transfer clips: latent's SOURCE video | NEW target video | TRANSFER result, all 33 pairs, ranked by held-out gain."""
import csv, json, statistics as st, subprocess
from pathlib import Path
from common import PAPER_RESULTS
from p5_setup import LATENTS
F = "/cvmfs/soft.computecanada.ca/gentoo/2023/x86-64-v3/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"
R = Path("/scratch/amirrz/iclr_rebuttal"); out = R / "samples_transfer_srcview"; out.mkdir(exist_ok=True)
rows = list(csv.DictReader(open(R / "transfer/pairs.csv")))
def g(r):
    v = [float(r[f"{m}_gain"]) for m in ("qwen25", "qwen3vl", "internvl3") if r.get(f"{m}_gain") not in ("", None, "None")]
    return st.mean(v) if v else -9
rows.sort(key=g, reverse=True)
tg = {(t["latent"], t["target"]): t for t in csv.DictReader(open(R / "transfer/targets.csv"))}
# display-only: the transferred latent for shouting is normal_wo_lpips (paper's transfer latent), but its own source
# edit is artifacty; show the normal_w_lpips run of the same source+prompt in panel 1 instead (user's choice).
SRC_PANEL = {"groom_angry_shouting": "a_angry_man_is_shouting./normal_w_lpips/both"}
import sys
ONLY = [a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--only=")]
esc = lambda s: s.replace("'", "").replace(":", " ").replace("%", " pct")
for k, r in enumerate(rows, 1):
    if ONLY and r["latent"] not in ONLY:
        continue
    t = tg[(r["latent"], r["target"])]
    src = PAPER_RESULTS / SRC_PANEL.get(r["latent"], LATENTS[r["latent"]]) / "mode_both" / "best_optimized_video_both.mp4"  # tuned result on the latent source
    sc = " ".join(f"{m[:5]} {float(r[f'{m}_transfer']):.2f}" for m in ("qwen25", "qwen3vl", "internvl3") if r.get(f"{m}_transfer") not in ("", None, "None"))
    vids = [(src, f"TUNED ON SOURCE (ours)  {esc(r['source_prompt'])}"), (t["target_video"], "NEW TARGET video"), (r["transfer_video"], f"TRANSFER result  {sc}")]
    a, fl = [], []
    for i, (v, l) in enumerate(vids):
        a += ["-i", str(v)]
        fl.append(f"[{i}:v]fps=25,scale=512:320:force_original_aspect_ratio=decrease,pad=512:320:(ow-iw)/2:(oh-ih)/2,setsar=1,"
                  f"drawtext=fontfile={F}:text='{l}':x=6:y=6:fontsize=15:fontcolor=white:box=1:boxcolor=black@0.6[v{i}]")
    fl.append(f"[v0][v1][v2]hstack=inputs=3,pad=iw:ih+30:0:30:black,drawtext=fontfile={F}:text='rank {k:02d}  target prompt  {esc(r['edit_prompt'])}':x=8:y=6:fontsize=17:fontcolor=yellow[o]")
    subprocess.run(["ffmpeg", "-v", "error", "-y", *a, "-filter_complex", ";".join(fl), "-map", "[o]", "-t", "3.6", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "22", str(out / f"rank{k:02d}_{r['latent']}__{r['target']}.mp4")], check=True)
print("done", len(list(out.glob("*.mp4"))))
