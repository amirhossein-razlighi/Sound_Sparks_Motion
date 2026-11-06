#!/usr/bin/env python3
"""Export the three panels of each transfer pair as separate original files (no re-encode), one dir per pair."""
import csv, shutil, statistics as st
from pathlib import Path
from common import PAPER_RESULTS
from p5_setup import LATENTS
SRC_PANEL = {"groom_angry_shouting": "a_angry_man_is_shouting./normal_w_lpips/both"}  # same display choice as make_clips_xfer_src.py
R = Path("/scratch/amirrz/iclr_rebuttal"); out = R / "transfer_panels_separate"; out.mkdir(exist_ok=True)
rows = list(csv.DictReader(open(R / "transfer/pairs.csv")))
def g(r):
    v = [float(r[f"{m}_gain"]) for m in ("qwen25", "qwen3vl", "internvl3") if r.get(f"{m}_gain") not in ("", None, "None")]
    return st.mean(v) if v else -9
rows.sort(key=g, reverse=True)
tg = {(t["latent"], t["target"]): t for t in csv.DictReader(open(R / "transfer/targets.csv"))}
for k, r in enumerate(rows, 1):
    t = tg[(r["latent"], r["target"])]
    d = out / f"rank{k:02d}_{r['latent']}__{r['target']}"; d.mkdir(exist_ok=True)
    src = PAPER_RESULTS / SRC_PANEL.get(r["latent"], LATENTS[r["latent"]]) / "mode_both" / "best_optimized_video_both.mp4"
    for name, p in (("1_tuned_on_source", src), ("2_new_target", Path(t["target_video"])), ("3_transfer_result", Path(r["transfer_video"]))):
        shutil.copy2(p, d / f"{name}{p.suffix}")
    sc = lambda kind: "  ".join(f"{m} {float(r[f'{m}_{kind}']):.2f}" for m in ("qwen25", "qwen3vl", "internvl3") if r.get(f"{m}_{kind}") not in ("", None, "None"))
    (d / "info.txt").write_text(f"source prompt (latent tuned for): {r['source_prompt']}\ntarget prompt: {r['edit_prompt']}\n"
                                f"panel1: {src}\npanel2: {t['target_video']}\npanel3: {r['transfer_video']}\n"
                                f"held-out P(yes) transfer: {sc('transfer')}\nheld-out P(yes) prompt-only Retake: {sc('baseline')}\n")
print(len(rows), "dirs ->", out)
