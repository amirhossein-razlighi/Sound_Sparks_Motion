#!/usr/bin/env python3
"""Merge everything known per IVEBench item into ivebench_summary.csv + ivebench_summary.md (CPU):
frame-check notes (Rorqual notes.csv, Trillium notes.json/runs.csv), held-out scores per config (qwen25/qwen3vl/internvl3,
ours vs Retake, incl. eye-picked iterations), official IVEBench per-video scores if eval CSVs exist."""
import csv, json, glob, statistics as st
from collections import defaultdict
from pathlib import Path
R = Path("/scratch/amirrz/iclr_rebuttal/ivebench"); G = Path("/home/amirrz/my_codes/Sound_Sparks_Motion/results_trillium/ivebench")
items = [Path(x["src_video_name"]).stem for x in json.load(open(R / "subset_subject_motion_short.json"))]
meta = {Path(x["src_video_name"]).stem: x for x in json.load(open(R / "subset_subject_motion_short.json"))}
notes = defaultdict(list)
for r in csv.DictReader(open(R / "notes.csv")):
    notes[r["name"]].append(f"[R {r['cfg']}] retake={r['verdict_retake']} ours={r['verdict_ours']}: {r['notes']}")
if (G / "notes.json").exists():
    for k, v in json.load(open(G / "notes.json")).items():
        n, c = k.split(":"); notes[n].append(f"[T {c}] {v}")
H = defaultdict(dict)   # H[(name, method)][model] = yes_prob
for m in ("qwen25", "qwen3vl", "internvl3"):
    f = R / "heldout" / f"{m}.csv"
    if f.exists():
        for r in csv.DictReader(open(f)):
            if r.get("yes_prob") not in ("", None):
                H[(r["scenario"], r["method"])][m] = float(r["yes_prob"])
rows = []
for n in items:
    cfgs = sorted(p.name for p in (R / "runs" / n).glob("*") if (p / "best_optimized_video_both.mp4").exists()) if (R / "runs" / n).exists() else []
    row = dict(item=n, edit=meta[n]["edit_prompt"], subcat=meta[n]["subcategory"], n_frames=json.load(open(R / "inputs" / f"{n}.json"))["n"],
               configs=" ".join(cfgs), source_q25=H[(n, "source")].get("qwen25", ""))
    for c in cfgs:
        for kind in ["ours", "retake"] + [k.split("_", 1)[1] for (nn, k) in H if nn == n and k.startswith("ours-iter") and k.endswith("_" + c)]:
            pass
    sc = []
    for (nn, meth), d in H.items():
        if nn == n and meth != "source":
            sc.append(f"{meth}: " + "/".join(f"{d.get(m, float('nan')):.2f}" for m in ("qwen25", "qwen3vl", "internvl3")))
    row["heldout(q25/q3/ivl3)"] = "; ".join(sorted(sc))
    row["notes"] = " || ".join(notes[n])
    rows.append(row)
with open(R / "ivebench_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
done = [r for r in rows if r["configs"]]
print(f"{len(done)}/{len(rows)} items with >=1 finished run ->", R / "ivebench_summary.csv")
