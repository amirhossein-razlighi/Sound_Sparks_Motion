#!/usr/bin/env python3
"""Aggregate audio-init ablation metrics into a master CSV + markdown table.

Walks <results-root>/<scenario>/<variant>/metrics.json (written by
eval_metrics.py) and produces:

  * <results-root>/aggregate_long.csv  — one row per (scenario, variant), all metrics
  * <results-root>/aggregate_summary.md — per-variant means across scenarios,
    with the three random seeds collapsed into a single mean±std row.

The summary groups variants into: source (ours), zero, random (s42/s1/s2 pooled).
This is the table to translate into the rebuttal response.

Pure stdlib — no pandas. Run on the login node after the GPU runs finish.

Usage:
    python editing/scripts/rebuttal/aggregate_metrics.py --results-root results/rebuttal
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

# Metrics surfaced in the markdown summary (CSV always carries everything).
HEADLINE = [
    ("motion_strength_optimized", "motion(RAFT)↑"),
    ("motion_strength_delta", "Δmotion(RAFT)↑"),
    ("motion_proxy_framediff_optimized", "motion(framediff)↑"),
    ("source_preservation_lpips", "src-LPIPS↓"),
    ("temporal_flicker_optimized", "flicker↓"),
    ("clip_edit_alignment_gain", "CLIP-align-gain↑"),
    ("final_qwen_yes_prob_critic", "Qwen-yes(critic)"),
]

GROUPS = {"source": ["source"], "zero": ["zero"], "random": ["random_s42", "random_s1", "random_s2"]}


def _fmt(x: float | None, nd: int = 4) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def _mean_std(vals: list[float]) -> tuple[float | None, float | None]:
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return None, None
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return m, 0.0
    var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
    return m, math.sqrt(var)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="results/rebuttal")
    args = ap.parse_args()

    root = Path(args.results_root).expanduser().resolve()
    rows: list[dict] = []
    all_metric_keys: set[str] = set()

    for mj in sorted(root.glob("*/*/metrics.json")):
        variant = mj.parent.name
        scenario = mj.parent.parent.name
        try:
            data = json.loads(mj.read_text())
        except Exception as ex:
            print(f"WARN: could not read {mj}: {ex}")
            continue
        metrics = data.get("metrics", {})
        all_metric_keys.update(metrics.keys())
        rows.append({"scenario": scenario, "variant": variant, **metrics})

    if not rows:
        print(f"No metrics.json found under {root}. Run the ablation + eval first.")
        return

    metric_cols = sorted(all_metric_keys)

    # ---- long CSV -----------------------------------------------------------
    long_csv = root / "aggregate_long.csv"
    with long_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "variant", *metric_cols])
        for r in sorted(rows, key=lambda r: (r["scenario"], r["variant"])):
            w.writerow([r["scenario"], r["variant"], *[r.get(c, "") for c in metric_cols]])
    print(f"Wrote {long_csv}  ({len(rows)} rows)")

    # ---- summary by group (mean across scenarios; random pools 3 seeds) -----
    scenarios = sorted({r["scenario"] for r in rows})
    lines: list[str] = []
    lines.append("# Audio-init ablation — summary\n")
    lines.append(f"Scenarios ({len(scenarios)}): {', '.join(scenarios)}\n")
    lines.append("Means across scenarios. `random` pools seeds s42/s1/s2 (mean±std).\n")

    header = "| metric | " + " | ".join(GROUPS.keys()) + " |"
    sep = "|" + "---|" * (len(GROUPS) + 1)
    lines.append(header)
    lines.append(sep)

    by_gv: dict[str, dict[str, list[float]]] = {g: {} for g in GROUPS}
    for key, _label in HEADLINE:
        for g, variant_names in GROUPS.items():
            vals = [r.get(key) for r in rows if r["variant"] in variant_names]
            by_gv[g][key] = [v for v in vals if isinstance(v, (int, float))]

    for key, label in HEADLINE:
        cells = []
        for g in GROUPS:
            m, s = _mean_std(by_gv[g][key])
            if g == "random" and m is not None:
                cells.append(f"{_fmt(m)} ± {_fmt(s)}")
            else:
                cells.append(_fmt(m))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("Arrows: ↑ higher is better, ↓ lower is better. "
                 "`Qwen-yes(critic)` is the in-loop optimization critic — reported for "
                 "context, NOT independent evidence.\n")

    summary_md = root / "aggregate_summary.md"
    summary_md.write_text("\n".join(lines))
    print(f"Wrote {summary_md}\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
