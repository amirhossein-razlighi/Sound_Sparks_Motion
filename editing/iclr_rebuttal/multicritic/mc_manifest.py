#!/usr/bin/env python3
"""Manifest of every video of the multi-critic proof of concept, for heldout_score.py (qwen25 / qwen3vl / internvl3,
16 uniform frames, held-out protocol) and the fidelity/motion metrics.

Per scenario: input (the prepared input the methods saw), retake (prompt-only LTX Retake = the run's baseline_video,
same seed), paper_ours (the paper's video), A_final / B_final (best checkpoint, final render), A_prev_iterNNN /
B_prev_iterNNN (the loop's best-so-far previews, every 5 iterations), A_it_NNN / B_it_NNN (the video the critics scored at
each iteration). Writes $ROOT/manifest.csv (columns id,group,scenario,method,video,edit_prompt).
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import SCENARIOS, COMPARISONS, load_paper_args, paper_run_dir  # noqa: E402

ROOT = Path("/scratch/amirrz/iclr_rebuttal/multicritic")
ARMS = {"A": "A_qwen25_only", "B": "B_qwen25+internvl3"}
rows = []
for s in sys.argv[1].split(",") if len(sys.argv) > 1 else ["red_car_door_opens", "man_pets_dog", "dog_yawning", "robot_waving"]:
    if not (ROOT / s).exists():
        continue
    ns = load_paper_args(s, {"output_dir": "/tmp/unused"})
    p = ns.edit_prompt if hasattr(ns, "edit_prompt") else ns.prompt
    add = lambda m, v, g="final": rows.append(dict(id=f"mc/{s}/{m}", group=g, scenario=s, method=m, video=str(v), edit_prompt=p))
    add("input", paper_run_dir(s) / "retake_input_prepared.mp4")
    add("paper_ours", COMPARISONS / SCENARIOS[s][1] / "ours.mp4")
    for a, d in ARMS.items():
        r = ROOT / s / d
        if not (r / "mode_both" / "best_optimized_video_both.mp4").exists():
            continue
        if a == "A" or not any(x["method"] == "retake" and x["scenario"] == s for x in rows):
            add("retake" if a == "A" else "retake_B", r / "baseline_video.mp4")
        add(f"{a}_final", r / "mode_both" / "best_optimized_video_both.mp4")
        for v in sorted((r / "mode_both" / "visualizations").glob("iter_*/best_optimized_video_both.mp4")):
            add(f"{a}_prev_{v.parent.name}", v, "preview")
        for v in sorted((r / "multicritic" / "iter_frames").glob("iter_*.mp4")):
            add(f"{a}_it_{v.stem.split('_')[1]}", v, "iterate")
with open(ROOT / "manifest.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(len(rows), "rows ->", ROOT / "manifest.csv")
