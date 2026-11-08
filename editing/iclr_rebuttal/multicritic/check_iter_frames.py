#!/usr/bin/env python3
"""Faithfulness check of multicritic/iter_frames/*.mp4: re-score the saved per-iteration videos with the loop critic
(critic.LoopCritic, pass 0 = the run's own sample mode on the full clip) and compare with the pass-0 NLL logged during the
run (passes.csv). Equal up to mp4 compression -> the saved videos are exactly what the critic scored.
Also re-scores baseline_video.mp4 (the normal render of the initial latents) for comparison with iter_001 (render gap).
Writes $ROOT/iter_frames_check.csv."""
import csv
import math
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_paper_args  # noqa: E402
from critic import LoopCritic  # noqa: E402

ROOT = Path("/scratch/amirrz/iclr_rebuttal/multicritic")
dev = torch.device("cuda")
rows, model = [], None
for s, run in (("man_pets_dog", "A_qwen25_only"), ("dog_yawning", "A_qwen25_only"), ("turtle_neck", "A_qwen25_only"),
               ("red_car_door_opens", "A_qwen25_only")):
    r = ROOT / s / run
    ns = load_paper_args(s, {"output_dir": "/tmp/unused"})
    c = LoopCritic(ns, dev) if model is None else LoopCritic(ns, dev, *model)
    model = (c.model, c.processor)
    P = pd.read_csv(r / "multicritic" / "passes.csv")
    P = P[(P["pass"] == 0) & (P.grad == 1)].set_index("iter")
    for it in (1, 5, 10):
        f = r / "multicritic" / "iter_frames" / f"iter_{it:03d}.mp4"
        if not f.exists() or it not in P.index:
            continue
        o = c.score_video(f)
        rows.append(dict(scenario=s, video=f.name, logged_pass0_nll=float(P.loc[it, "qwen_nll"]),
                         rescored_pass0_nll=o["pass_nll"][0], logged_p=math.exp(-float(P.loc[it, "qwen_nll"])),
                         rescored_p=o["pass_yes"][0]))
        print(rows[-1], flush=True)
    o = c.score_video(r / "baseline_video.mp4")
    rows.append(dict(scenario=s, video="baseline_video.mp4 (normal render, initial latents)", logged_pass0_nll="",
                     rescored_pass0_nll=o["pass_nll"][0], logged_p="", rescored_p=o["pass_yes"][0]))
    print(rows[-1], flush=True)
with open(ROOT / "iter_frames_check.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
