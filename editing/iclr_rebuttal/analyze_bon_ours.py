#!/usr/bin/env python3
"""Best-of-N on ours vs ours vs prompt-only best-of-N. CPU only. Reads critic_summary.csv and Trillium metric CSVs; writes per-pick rows and mean tables."""
import csv
from pathlib import Path
import pandas as pd

R = Path("/scratch/amirrz/iclr_rebuttal"); T = Path("/home/amirrz/my_codes/Sound_Sparks_Motion/results_trillium")
M = T / "best_of_n_ours/metrics"; OUT = R / "best_of_n_ours"
cs = pd.read_csv(OUT / "critic_summary.csv").set_index("scenario")
fc = pd.read_csv(T / "best_of_n_ours/frame_checks.csv")
fc["k"] = fc.seed.astype(str).str.replace("s", "", regex=False).astype(int)


def col(pick): return "ours_s%02d" % int(pick[1:])


def load(f, c):
    d = pd.read_csv(M / f)
    return d.pivot_table(index="scenario", columns="method", values=c)


rows = []
tabs = {m: load(f"{m}.csv", "yes_prob") for m in ("qwen25", "qwen3vl", "internvl3")}
tabs |= {k: load("flow_warp.csv", k) for k in ("flow_mag", "warp_err")}
tabs |= {k: load("source_fidelity.csv", k) for k in ("first_frame_fidelity", "aligned_fidelity")}
for s, r in cs.iterrows():
    for label, pick in [("seed42", "s42")] + [(f"bo{n}", r[f"ours_bo{n}_pick"]) for n in (3, 4, 8)]:
        d = {"scenario": s, "set": label, "pick": pick, "critic": r["ours_s%s" % pick[1:]]}
        f = fc[(fc.scenario == s) & (fc.k == int(pick[1:]))].iloc[0]
        d["frame_check"], d["note"] = f.edit_visible, f.note
        for k, t in tabs.items(): d[k] = t.loc[s, col(pick)]
        rows.append(d)
p = pd.DataFrame(rows); p.to_csv(OUT / "picks_all_metrics.csv", index=False)
print(p.groupby("set")[["critic", *tabs]].mean(numeric_only=True).round(4).T.to_string())
print({k: int((p[p.set == k].qwen3vl >= .5).sum()) for k in p.set.unique()})
for m in ("qwen25", "qwen3vl", "internvl3"):
    print(m, {k: int((p[p.set == k][m] >= .5).sum()) for k in p.set.unique()})
print(p[p.set == "bo8"][["scenario", "pick", "critic", "qwen25", "qwen3vl", "internvl3", "frame_check", "flow_mag", "aligned_fidelity"]].round(3).to_string())
