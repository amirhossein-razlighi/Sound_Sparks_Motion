#!/usr/bin/env python3
"""TRILLIUM: (re)build results_trillium/opt_runs.csv + copy metrics.json for every finished OPT task in done.txt.

  python trillium_report.py --lane /scratch/amirrz/iclr_rebuttal/lane_main --out-root /scratch/amirrz/iclr_rebuttal/out
Idempotent: rows are keyed by (exp, scenario, tag, xformers); later attempts of the same task replace earlier rows.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
COLS = ["exp", "scenario", "tag", "status", "best_iter", "best_total_loss", "best_qwen_yes", "n_iters",
        "mean_iter_s_post_warmup", "peak_mem_gb_loop", "wall_total_s", "output_dir", "gpu", "baseline_mad_vs_paper", "xformers"]


def mad_vs_paper(out: Path, scenario: str) -> str:
    try:
        import numpy as np
        import imageio.v3 as iio
        from common import paper_run_dir
        a = iio.imread(str(out / "baseline_video.mp4")).astype(np.float32)
        b = iio.imread(str(paper_run_dir(scenario) / "baseline_video.mp4")).astype(np.float32)
        n = min(len(a), len(b))
        return f"{np.abs(a[:n] - b[:n]).mean():.3f}"
    except Exception as ex:  # noqa: BLE001
        return f"err:{type(ex).__name__}"


def row_for(exp: str, scen: str, tag: str, rc: str, out: Path, mode: str, xf: str) -> dict:
    r = {c: "" for c in COLS}
    r.update(exp=exp, scenario=scen, tag=tag, output_dir=f"trillium:{out}", xformers=xf)
    t = json.loads((out / "timing.json").read_text()) if (out / "timing.json").exists() else {}
    oom = (out / "OOM.json").exists()
    r["status"] = "OOM" if oom else ("ok" if rc == "0" and t.get("status") == "ok" else f"fail_rc{rc}")
    for k in ("mean_iter_s_post_warmup", "peak_mem_gb_loop", "wall_total_s", "gpu"):
        v = t.get(k)
        r[k] = f"{v:.3f}" if isinstance(v, float) else ("" if v is None else v)
    log = out / f"mode_{mode}" / f"optimization_log_qwen_{mode}.csv"
    if log.exists():
        rows = list(csv.DictReader(log.open()))
        it_rows = [x for x in rows if x.get("iter", "").strip().lstrip("-").isdigit() and int(x["iter"]) > 0]
        r["n_iters"] = len(it_rows)
        best = [x for x in it_rows if str(x.get("is_best", "")).strip().lower() in ("1", "true", "yes")]
        if best:
            b = best[-1]
            r.update(best_iter=b["iter"], best_total_loss=b["total_loss"], best_qwen_yes=b["qwen_yes_prob"])
    seed = json.loads((out / "run_config.json").read_text())["args"].get("seed") if (out / "run_config.json").exists() else None
    r["baseline_mad_vs_paper"] = mad_vs_paper(out, scen) if seed == 42 else "n/a(seed!=42)"
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--xformers", choices=["0", "1"], required=True, help="whether this lane ran with xformers")
    a = ap.parse_args()
    dst = REPO / "results_trillium"
    csv_path = dst / "opt_runs.csv"
    existing = {}
    if csv_path.exists():
        for r in csv.DictReader(csv_path.open()):
            existing[(r["exp"], r["scenario"], r["tag"], r.get("xformers", ""))] = r
    for line in (Path(a.lane) / "done.txt").read_text().splitlines():
        rc, _, task = line.partition("|")
        parts = task.split()
        if rc == "PREEMPTED" or len(parts) < 4 or parts[0] != "OPT":
            continue
        exp, scen, tag = parts[1:4]
        mode = next((kv.split("=", 1)[1] for kv in parts[4:] if kv.startswith("opt_mode=")), "both")
        out = Path(a.out_root) / exp / scen / tag
        existing[(exp, scen, tag, a.xformers)] = row_for(exp, scen, tag, rc, out, mode, a.xformers)
        if (out / "metrics.json").exists():
            sub = dst / exp / scen / (tag if a.xformers == "1" else tag + ".noxformers")
            sub.mkdir(parents=True, exist_ok=True)
            shutil.copy2(out / "metrics.json", sub / "metrics.json")
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for k in sorted(existing):
            w.writerow({c: existing[k].get(c, "") for c in COLS})
    print(csv_path.read_text())


if __name__ == "__main__":
    main()
