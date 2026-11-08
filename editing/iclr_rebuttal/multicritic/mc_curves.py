#!/usr/bin/env python3
"""Per-iteration curves for the multi-critic proof of concept (no pandas; run with a python that has matplotlib,
e.g. /scratch/amirrz/iclr_rebuttal/venv_vbench/bin/python). Reads report/per_iteration.csv and finals.csv (best iter).
Writes report/curves.png: one row per scenario; columns = in-loop Qwen2.5-VL (full-clip pass, the paper's critic, both
arms optimise it), InternVL3 on the in-loop iterate (16-frame protocol; in the loop for B only), and InternVL3 in-loop
pass-0 value for B (what the second critic saw). Arm A blue, arm B orange; repeats dashed/dotted; circle = best iter."""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REP = Path("/scratch/amirrz/iclr_rebuttal/multicritic/report")
COL = {"A": "#2a78d6", "B": "#eb6834"}
LS = {"": "-", "_r2": "--", "_r3": ":"}
LAB = {"A": "A: Qwen2.5-VL only", "B": "B: Qwen2.5-VL + InternVL3"}
rows = list(csv.DictReader(open(REP / "per_iteration.csv")))
D = defaultdict(lambda: defaultdict(list))
for r in rows:
    D[r["scenario"]][r["arm"]].append(r)
best = {}
for r in csv.DictReader(open(REP / "finals.csv")):
    if r["video"].endswith("_final") and r.get("best_iter"):
        best[(r["scenario"], r["video"][:-6])] = int(float(r["best_iter"]))
f = lambda v: float(v) if v not in ("", "nan", None) else float("nan")
panels = [("inloop_qwen_p_pass0", "Qwen2.5-VL, in-loop full-clip pass\n(the paper's critic; both arms optimise it)"),
          ("inloop_internvl_p_pass0", "InternVL3, in-loop pass 0\n(the second critic; logged for B only)"),
          ("heldout_internvl3", "InternVL3 on the iterate, 16-frame protocol\n(same protocol for A and B)")]
scen = sorted(D)
fig, ax = plt.subplots(len(scen), 3, figsize=(13, 2.4 * len(scen)), squeeze=False, sharex=True, sharey=True)
for i, s in enumerate(scen):
    for j, (k, t) in enumerate(panels):
        x = ax[i, j]
        for a in sorted(D[s]):
            rr = sorted(D[s][a], key=lambda r: int(r["iter"]))
            xs, ys = [int(r["iter"]) for r in rr], [f(r[k]) for r in rr]
            if all(y != y for y in ys):
                continue
            x.plot(xs, ys, color=COL[a[0]], lw=2 if "_r" not in a else 1.2, ls=LS.get(a[1:], "-"),
                   label=LAB[a[0]] + (f" rep {a[3:]}" if "_r" in a else ""))
            b = best.get((s, a))
            if b in xs:
                x.plot([b], [ys[xs.index(b)]], "o", ms=7, mfc="white", mec=COL[a[0]], mew=2)
        x.set_ylim(-0.02, 1.02); x.grid(alpha=0.25, lw=0.5)
        for sp in ("top", "right"):
            x.spines[sp].set_visible(False)
        if i == 0:
            x.set_title(t, fontsize=9)
        if j == 0:
            x.set_ylabel(f"{s}\nP(yes)", fontsize=8)
            x.legend(fontsize=6.5, loc="best", frameon=False)
        if i == len(scen) - 1:
            x.set_xlabel("iteration (circle = selected best iteration)", fontsize=8)
fig.tight_layout(); fig.savefig(REP / "curves.png", dpi=110); plt.close(fig)
print("wrote", REP / "curves.png")
