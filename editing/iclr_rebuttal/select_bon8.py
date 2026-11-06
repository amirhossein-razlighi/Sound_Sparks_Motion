#!/usr/bin/env python3
"""Best-of-8 at the in-loop critic: pool = paper Retake baseline (seed 42) + Rorqual seeds 0,1 + Trillium seeds 2-6.
Writes best_of_n/bon8_selection.csv (CPU only). Held-out comparison is done separately once all held-out CSVs exist."""
import csv, json
from pathlib import Path
from common import OUT_ROOT, TWELVE
ROOT = Path(__file__).resolve().parents[2]
rows = []
for s in TWELVE:
    a = json.loads((OUT_ROOT / "best_of_n" / s / "scores.json").read_text())
    b = json.loads((ROOT / "results_trillium/best_of_n" / s / "scores.json").read_text())
    y = {"paper_baseline(s42)": a["reference"]["paper_baseline"]["yes_prob"]}
    y.update({f"seed_{k:02d}": a["seeds"][str(k)]["yes_prob"] for k in (0, 1)})
    y.update({f"seed_{k:02d}": b["seeds"][str(k)]["yes_prob"] for k in (2, 3, 4, 5, 6)})
    pool = list(y)
    best3 = max(pool[:3], key=y.get); best8 = max(pool, key=y.get)
    ours = a["reference"]["paper_ours"]["yes_prob"]
    rows.append(dict(scenario=s, ours=round(ours, 3), bon3_pick=best3, bon3=round(y[best3], 3), bon8_pick=best8, bon8=round(y[best8], 3),
                     n_seeds_beating_ours=sum(v > ours for v in y.values()), **{k: round(v, 3) for k, v in y.items()}))
out = OUT_ROOT / "best_of_n" / "bon8_selection.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
m = lambda k: sum(r[k] for r in rows) / len(rows)
for r in rows:
    print(f"{r['scenario']:28s} ours {r['ours']:.3f}  bon3 {r['bon3']:.3f} ({r['bon3_pick']})  bon8 {r['bon8']:.3f} ({r['bon8_pick']})  beat-ours {r['n_seeds_beating_ours']}/8")
print(f"MEAN in-loop critic: ours {m('ours'):.3f}  bon3 {m('bon3'):.3f}  bon8 {m('bon8'):.3f}  ->", out)
