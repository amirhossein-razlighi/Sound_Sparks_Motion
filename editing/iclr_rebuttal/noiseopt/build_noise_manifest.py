#!/usr/bin/env python3
"""Manifests for the noise-optimization metric bundle: every finished noise run (final render of the best iterate),
plus input, paper ours and LTX-2 Retake rows from comparisons.csv, so all rows go through one pipeline."""
import csv, glob, os
R = "/scratch/amirrz/iclr_rebuttal"
cmp_rows = list(csv.DictReader(open(f"{R}/manifests/comparisons.csv")))
fields = list(cmp_rows[0].keys())
scen = ["turtle_neck", "dog_yawning", "red_car_door_opens", "man_pets_dog"]
rows = []
for s in scen:
    ref = next(r for r in cmp_rows if r["scenario"] == s and r["method"] == "ours")
    for m in ("input", "ours", "ltx2_retake"):
        rows.append(next(r for r in cmp_rows if r["scenario"] == s and r["method"] == m))
    for d in sorted(glob.glob(f"{R}/noiseopt/{s}/lr*")):
        v = f"{d}/mode_noise/best_optimized_video_noise.mp4"
        if not os.path.exists(f"{d}/timing.json") or not os.path.exists(v):
            continue
        tag = os.path.basename(d)
        rows.append({**ref, "id": f"noise/{s}/{tag}", "group": "noiseopt", "method": f"noise_{tag}", "video": v})
os.makedirs(f"{R}/noiseopt/metrics/logs", exist_ok=True)
for name, keep in (("noiseopt_with_input", lambda r: True), ("noiseopt", lambda r: r["method"] != "input")):
    with open(f"{R}/manifests/{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fields); w.writeheader(); w.writerows(r for r in rows if keep(r))
    print(name, sum(1 for r in rows if keep(r)))
