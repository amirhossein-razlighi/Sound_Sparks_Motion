#!/usr/bin/env python3
"""Re-score videos with the *in-loop* Qwen2.5-VL critic (critic.py), using for each
video the critic settings of the run_config in its manifest `critic_from` column.

Also reports the value the run itself logged (best_params_<mode>.pt `qwen_score`,
which was measured on the training-time render, i.e. low-memory guidance and the
latents *before* the last Adam step), so logged-vs-final-video can be compared.

Output: <out>/loop_qwen25.csv (resumable)
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from critic import LoopCritic


def ns_from_run_config(path: str):
    from optimize_qwen_vl import build_parser
    ns = build_parser().parse_args(["--src-video", "x", "--edit-prompt", "x", "--output-dir", "x"])
    for k, v in json.loads(Path(path).read_text())["args"].items():
        if hasattr(ns, k):
            setattr(ns, k, v)
    if ns.qwen_max_frames % 2:
        ns.qwen_max_frames += 1
    from common import apply_model_path_env
    apply_model_path_env(ns)
    return ns


def logged_score(video: str):
    v = Path(video)
    for p in sorted(v.parent.glob("best_params_*.pt")):
        try:
            d = torch.load(p, map_location="cpu")
            return float(d.get("qwen_score")), float(d.get("qwen_loss"))
        except Exception:
            pass
    return "", ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, action="append")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = [r for m in a.manifest for r in csv.DictReader(open(m)) if r.get("critic_from")]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dst = out / "loop_qwen25.csv"
    done = {r["id"] for r in csv.DictReader(open(dst))} if dst.exists() else set()
    todo = [r for r in rows if r["id"] not in done]
    if not todo:
        return
    from motion_opt.qwen_loss import build_qwen_model
    dev = torch.device("cuda")
    first = ns_from_run_config(todo[0]["critic_from"])
    model, proc = build_qwen_model(first.qwen_model, device=dev, gradient_checkpointing=False)
    critics: dict[str, LoopCritic] = {}
    new = not dst.exists()
    with open(dst, "a", newline="") as f:
        w = None
        for r in todo:
            cf = r["critic_from"]
            try:
                if cf not in critics:
                    critics[cf] = LoopCritic(ns_from_run_config(cf), dev, model, proc)
                c = critics[cf]
                res = c.score_video(r["video"])
                spec = f"{c.ns.qwen_max_frames}f/{c.ns.qwen_sample_mode}/{c.passes}p"
                ls, ll = logged_score(r["video"])
                row = {**r, "loop_yes_prob": res["yes_prob"], "loop_mean_nll": res["mean_nll"],
                       "pass_yes": json.dumps([round(x, 5) for x in res["pass_yes"]]), "critic_spec": spec,
                       "n_frames": res["n_frames"], "upsampled_to": res["upsampled_to"],
                       "logged_best_yes": ls, "logged_best_total_loss": ll, "error": ""}
            except Exception as e:
                row = {**r, "loop_yes_prob": "", "loop_mean_nll": "", "pass_yes": "", "critic_spec": "",
                       "n_frames": "", "upsampled_to": "",
                       "logged_best_yes": "", "logged_best_total_loss": "", "error": repr(e)[:300]}
            if w is None:
                w = csv.DictWriter(f, fieldnames=list(row))
                if new:
                    w.writeheader()
            w.writerow(row); f.flush()
            print(r["id"], row["loop_yes_prob"], row["logged_best_yes"], row["error"], flush=True)


if __name__ == "__main__":
    main()
