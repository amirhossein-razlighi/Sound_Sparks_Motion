#!/usr/bin/env python3
"""Manifest of every IVEBench output that exists (ours per config + its prompt-only Retake baseline) for heldout_score.py.
edit_prompt = IVEBench target_prompt (the string the method was given). With --phrase every row is duplicated with id suffix
"@phrase" and edit_prompt = IVEBench target_phrase (short action phrase; honest edit question, used for selection).
Usage: build_manifest.py [--root R] [--out CSV] [--phrase]"""
import argparse, csv, json, os
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("--root", default=os.environ.get("IVE_ROOT", "/scratch/amirrz/iclr_rebuttal/ivebench"))
ap.add_argument("--out", default=None); ap.add_argument("--phrase", action="store_true"); a = ap.parse_args()
R = Path(a.root); items = {Path(x["src_video_name"]).stem: x for x in json.load(open(R / "subset_subject_motion_short.json"))}
rows = []
for name, it in items.items():
    for cfg_dir in sorted((R / "runs" / name).glob("*")):
        pairs = [("ours", "best_optimized_video_both.mp4"), ("retake", "baseline_video.mp4")] + \
                [(f"ours-{p.stem}", p.name) for p in sorted(cfg_dir.glob("iter_*.mp4"))]
        for kind, f in pairs:
            if (cfg_dir / f).exists():
                rows.append(dict(id=f"ive/{name}/{cfg_dir.name}/{kind}", group="ivebench", scenario=name, method=f"{kind}_{cfg_dir.name}",
                                 video=str(cfg_dir / f), edit_prompt=it["target_prompt"].strip(), critic_from=""))
    rows.append(dict(id=f"ive/{name}/source", group="ivebench", scenario=name, method="source", video=str(R / "inputs" / f"{name}.mp4"),
                     edit_prompt=it["target_prompt"].strip(), critic_from=""))
if a.phrase:
    rows += [dict(r, id=r["id"] + "@phrase", edit_prompt=items[r["scenario"]]["target_phrase"].strip()) for r in list(rows)]
out = Path(a.out or R / "manifest_ivebench.csv")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(len(rows), "rows ->", out)
