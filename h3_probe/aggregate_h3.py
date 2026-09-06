#!/usr/bin/env python3
"""Aggregate H3 fair-eval results (eval_40.json per arm) into a markdown table.

    python3 h3_probe/aggregate_h3.py  [> h3_probe/results/H3_SUMMARY.md]
"""
import glob
import json
import os

ROOT = "/scratch/amirrz/H3_exp/outputs"
rows = []
for f in sorted(glob.glob(os.path.join(ROOT, "opt_*", "eval_40.json"))):
    r = json.load(open(f))
    arm = r["arm"]
    rc_path = os.path.join(os.path.dirname(f), "run_config.json")
    edit = json.load(open(rc_path))["edit_prompt"] if os.path.exists(rc_path) else "The man pets the dog."
    mode = "both" if r.get("text_delta_l2", 0) > 0 else "audio"
    rows.append((arm, edit, mode, r["baseline_yes"], r["optimized_yes"],
                 r["audio_latent_l2_shift"], r["text_delta_l2"], r["baseline_nll"], r["optimized_nll"]))

print("| arm | edit prompt | mode | baseline yes | optimized yes | Δyes (rel) | nll base→opt (Δ nats) | ‖Δz_audio‖ | ‖δ_text‖ |")
print("|---|---|---|---|---|---|---|---|---|")
for arm, edit, mode, b, o, za, dt, nb, no in rows:
    rel = (o - b) / max(b, 1e-9) * 100
    print(f"| {arm} | {edit} | {mode} | {b:.4f} | {o:.4f} | {o-b:+.4f} ({rel:+.0f}%) | {nb:.3f}→{no:.3f} ({no-nb:+.3f}) | {za:.2f} | {dt:.2f} |")
if rows:
    import statistics as st
    bs = [r[3] for r in rows]; os_ = [r[4] for r in rows]
    print(f"\n**mean baseline yes = {st.mean(bs):.4f} → mean optimized yes = {st.mean(os_):.4f}**  "
          f"(improved in {sum(o > b for b, o in zip(bs, os_))}/{len(rows)} arms)")
