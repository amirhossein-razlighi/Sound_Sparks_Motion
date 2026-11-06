#!/usr/bin/env python3
"""P1 analysis: best-of-N selection from the 24-seed pool (CPU only).

Pool order: [42, 0, 1, ..., 22]. best-of-n (prefix) = argmax critic yes-prob over
the first n seeds (so best-of-1 = seed 42 = the paper's Retake baseline seed).
We also report the exact expectation of max-of-n over uniformly random n-subsets
of the 24 seeds (order-statistics formula), which removes the arbitrariness of
the prefix order.

Copies to best_of_n/selected/<scenario>/: bon4/8/16/24.mp4, ours.mp4 (paper),
retake_seed42.mp4 (our seed-42 render), paper_retake_baseline.mp4.
Writes best_of_n/summary.csv and best_of_n/summary.md.
"""
from __future__ import annotations

import csv
import json
import shutil
from math import comb
from pathlib import Path

from common import OUT_ROOT, TWELVE, paper_baseline_video, paper_ours_video

ORDER = [42] + list(range(23))
NS_ALL = [1, 3, 4, 8, 16, 24]  # evaluated up to the pool size that exists (user: start with N=3)
OURS_SECONDS = 589.6  # paper end-to-end LTX-2 edit cost (turtle, 1x H100); see Table 7 caveats


def expected_max(vals: list[float], n: int) -> float:
    v = sorted(vals)
    N = len(v)
    return sum(v[i] * comb(i, n - 1) for i in range(n - 1, N)) / comb(N, n)


def main():
    root = OUT_ROOT / "best_of_n"
    rows = []
    for s in TWELVE:
        sp = root / s / "scores.json"
        if not sp.exists():
            continue
        d = json.loads(sp.read_text())
        seeds = [x for x in ORDER if str(x) in d["seeds"]]
        NS = [n for n in NS_ALL if n <= len(seeds)]
        if len(seeds) not in NS:
            NS.append(len(seeds))
        y = {x: d["seeds"][str(x)]["yes_prob"] for x in seeds}
        cost = {x: d["seeds"][str(x)]["render_s"] + d["seeds"][str(x)]["score_s"] for x in seeds}
        sel = root / "selected" / s
        sel.mkdir(parents=True, exist_ok=True)
        r = {"scenario": s, "ours_yes": d["reference"]["paper_ours"]["yes_prob"],
             "paper_baseline_yes": d["reference"]["paper_baseline"]["yes_prob"],
             "seed42_yes": y[42], "seed42_mad_vs_paper": d["seeds"]["42"].get("mean_abs_diff_vs_paper_baseline"),
             "mean_render_s": sum(d["seeds"][str(x)]["render_s"] for x in seeds) / len(seeds),
             "mean_score_s": sum(d["seeds"][str(x)]["score_s"] for x in seeds) / len(seeds)}
        for n in NS:
            pre = seeds[:n]
            best = max(pre, key=lambda x: y[x])
            r[f"bon{n}_seed"] = best
            r[f"bon{n}_yes"] = y[best]
            r[f"bon{n}_cost_s"] = sum(cost[x] for x in pre)
            r[f"E_bon{n}_yes"] = expected_max(list(y.values()), n)
            if n > 1:
                shutil.copy2(root / s / "candidates" / f"seed_{best:02d}.mp4", sel / f"bon{n}.mp4")
        r["pool_frac_seeds_beating_ours"] = sum(v > r["ours_yes"] for v in y.values()) / len(y)
        shutil.copy2(paper_ours_video(s), sel / "ours.mp4")
        shutil.copy2(root / s / "candidates" / "seed_42.mp4", sel / "retake_seed42.mp4")
        shutil.copy2(paper_baseline_video(s), sel / "paper_retake_baseline.mp4")
        (sel / "selection.json").write_text(json.dumps({k: v for k, v in r.items()}, indent=2, default=str))
        rows.append(r)
    if not rows:
        print("no complete scenarios yet"); return
    with open(root / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    k = len(rows)
    NS = sorted({int(key[3:-4]) for r in rows for key in r if key.startswith("bon") and key.endswith("_yes")})
    NS = [n for n in NS if all(f"bon{n}_yes" in r for r in rows)]
    mean = lambda key: sum(float(r[key]) for r in rows) / k
    lines = [f"# P1 best-of-N (critic = exact in-loop Qwen2.5-VL call), {k}/12 scenarios", "",
             "| | ours | Retake s42 (paper) | " + " | ".join(f"BoN-{n}" for n in NS) + " |", "|---|---|---|" + "---|" * len(NS),
             f"| mean critic yes (prefix) | {mean('ours_yes'):.3f} | {mean('paper_baseline_yes'):.3f} | " +
             " | ".join(f"{mean(f'bon{n}_yes'):.3f}" for n in NS) + " |",
             "| mean critic yes (E over subsets) | | | " + " | ".join(f"{mean(f'E_bon{n}_yes'):.3f}" for n in NS) + " |",
             "| #scen ours > BoN (prefix) | | | " + " | ".join(str(sum(r['ours_yes'] > r[f'bon{n}_yes'] for r in rows)) for n in NS) + " |",
             "| mean wall-clock s (render+score) | " + f"{OURS_SECONDS:.0f} (paper) | | " +
             " | ".join(f"{mean(f'bon{n}_cost_s'):.0f}" for n in NS) + " |", "",
             f"Mean forward-only render {mean('mean_render_s'):.1f} s, critic scoring {mean('mean_score_s'):.1f} s per candidate.",
             "Selection uses the critic only; whether the critic-selected sample is a *good edit* must be judged by GPT/humans (P2 held-out scores, rsync list)."]
    (root / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
