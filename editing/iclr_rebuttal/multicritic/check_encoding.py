#!/usr/bin/env python3
"""Robustness of the in-loop critic scores to (near-invisible) video encoding, every iteration of every run.

During optimisation the critics score float RGB frames of the in-loop render. multicritic/iter_frames/iter_NNN.mp4 is the
same frames encoded as h264 crf 16, yuv420p, 8 bit (visually lossless). For every logged iteration (pass 0 = the run's
own sample mode on the full clip) this re-scores the mp4 with
  * the loop's Qwen2.5-VL critic (critic.LoopCritic, pass 0 only, same frame indices as logged), and
  * InternVL3 (internvl_critic.InternVLCritic, same thinned indices as logged in arm B; for arm A the indices the
    wrapper would have used: the Qwen pass-0 indices thinned to <= 16)
and compares with the logged values. A score that collapses after encoding was carried by pixel detail the eye does not
see, i.e. the signature of an adversarial / critic-specific gain.
Writes $ROOT/encoding_check.csv (one row per run x iteration)."""
import csv
import json
import math
import sys
from pathlib import Path

import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from common import load_paper_args  # noqa: E402
from critic import LoopCritic, load_video_chw  # noqa: E402
from internvl_critic import InternVLCritic  # noqa: E402
from motion_opt.qwen_loss import compute_qwen_video_loss  # noqa: E402

ROOT = Path("/scratch/amirrz/iclr_rebuttal/multicritic")
dev = torch.device("cuda")
ivl = None
qm = None
out = []
runs = sorted(r for r in ROOT.glob("*/[AB]_*") if r.is_dir() and ".old" not in r.name and (r / "multicritic" / "passes.csv").exists())
for r in runs:
    cfg = json.loads((r / "multicritic" / "config.json").read_text())
    ov = dict(cfg["overrides"]); ov["output_dir"] = "/tmp/unused"
    ns = load_paper_args(cfg["scenario"], ov)
    c = LoopCritic(ns, dev) if qm is None else LoopCritic(ns, dev, *qm)
    qm = (c.model, c.processor)
    if ivl is None:
        ivl = InternVLCritic(cfg["question"], device="cuda:0")
    ivl.question = cfg["question"]; ivl._ids.clear()
    P = pd.read_csv(r / "multicritic" / "passes.csv")
    P = P[(P["pass"] == 0) & (P.grad == 1)]
    for _, row in P.iterrows():
        it = int(row["iter"])
        f = r / "multicritic" / "iter_frames" / f"iter_{it:03d}.mp4"
        if not f.exists():
            continue
        fr = load_video_chw(f, dev)
        with torch.no_grad():
            torch.manual_seed(1234)
            q = float(compute_qwen_video_loss(frames_chw=fr, qwen_model=c.model, cached_inputs=c.cached, yes_token_id=c.yes_id,
                                              no_token_id=c.no_id, max_frames=ns.qwen_max_frames, img_size=ns.qwen_img_size,
                                              backward=False, return_details=False, sample_mode=row["sample_mode"],
                                              contiguous_start_frame=ns.qwen_contiguous_start_frame,
                                              rubric_weight_overrides=c.overrides))
            qidx = torch.tensor([int(x) for x in str(row["qwen_frames"]).split()])
            iidx = ivl.pick(qidx)
            i_nll = float(ivl.nll(fr[iidx.to(dev)]))
        o = dict(scenario=r.parent.name, run=r.name, arm=r.name[0], iter=it,
                 logged_qwen_p=math.exp(-row["qwen_nll"]), encoded_qwen_p=math.exp(-q),
                 logged_internvl_p=math.exp(-row["internvl_nll"]) if str(row["internvl_nll"]) != "nan" else float("nan"),
                 encoded_internvl_p=math.exp(-i_nll))
        out.append(o)
    print(r, len(P), "iterations", flush=True)
    with open(ROOT / "encoding_check.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
print("wrote", ROOT / "encoding_check.csv", len(out))
