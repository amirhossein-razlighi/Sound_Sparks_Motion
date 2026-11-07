#!/usr/bin/env python3
"""Gate before any multi-critic optimisation: the differentiable InternVL3 path must reproduce the official HF
processor path, and gradients must reach the frames.

For input / ours / Retake of each scenario: 16 linspace frames (and the scenario's own Qwen frame count, thinned to
<= 16, from the first frames) scored with the in-loop question two ways:
  ref  = official processor (uint8 frames -> InternVLVideoProcessor) + same yes/no rule
  diff = InternVLCritic.pixel_values (torch bicubic, antialias) on float frames in [0,1]
Also: one backward per video -> grad norm on the frames, peak memory, seconds.
Writes <out>/calibration.csv and prints a summary; exit 1 if any |P_ref - P_diff| > 0.05 or a grad is 0/non-finite.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import av
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from common import SCENARIOS, COMPARISONS, critic_spec, load_paper_args  # noqa: E402
from internvl_critic import InternVLCritic  # noqa: E402


def frames_of(p):
    c = av.open(str(p)); f = [x.to_ndarray(format="rgb24") for x in c.decode(video=0)]; c.close(); return np.stack(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default="red_car_door_opens,man_pets_dog,dog_yawning,robot_waving")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows, crit = [], None
    for s in a.scenarios.split(","):
        ns = load_paper_args(s, {"output_dir": "/tmp/unused"}); q = critic_spec(ns)["question"]
        if crit is None:
            crit = InternVLCritic(q, device=a.device)
        crit.question = q; crit._ids.clear()
        for m, fn in (("input", "input_video.mp4"), ("ours", "ours.mp4"), ("retake", "baseline_ltx2.mp4")):
            V = frames_of(COMPARISONS / SCENARIOS[s][1] / fn)
            for tag, idx in (("lin16", np.linspace(0, len(V) - 1, 16).round().astype(int)),):
                sub = V[idx]
                with torch.no_grad():
                    msgs = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": q}]}]
                    text = crit.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                    o = crit.proc(text=[text], videos=[sub], return_tensors="pt")
                    pv_ref = o["pixel_values"].to(crit.device, torch.bfloat16)
                    assert torch.equal(o["input_ids"].to(crit.device), crit._inputs(len(idx))["input_ids"])
                    x = torch.from_numpy(sub).permute(0, 3, 1, 2).float().div(255).to(crit.device)
                    pv_diff = crit.pixel_values(x)
                    pix_mad = float((pv_ref.float() - pv_diff.float()).abs().mean())
                    inp = crit._inputs(len(idx))
                    lg = crit.model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"], pixel_values=pv_ref).logits[0, -1].float()
                    lp = torch.log_softmax(lg, -1)
                    ly, ln = torch.logsumexp(lp[crit.yes], 0), torch.logsumexp(lp[crit.no], 0)
                    p_ref = float(torch.exp(ly - torch.logaddexp(ly, ln)))
                torch.cuda.reset_peak_memory_stats(crit.device); t0 = time.time()
                xg = x.clone().requires_grad_(True)
                nll = crit.nll(xg); nll.backward()
                torch.cuda.synchronize(crit.device)
                g = xg.grad
                r = dict(scenario=s, method=m, frames=tag, p_ref=p_ref, p_diff=float(torch.exp(-nll.detach())),
                         pixel_mad_normalised=pix_mad, grad_norm=float(g.norm()), grad_finite=bool(torch.isfinite(g).all()),
                         fwd_bwd_s=time.time() - t0, peak_gb=torch.cuda.max_memory_allocated(crit.device) / 1e9)
                rows.append(r); print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    with open(out / "calibration.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    d = np.array([abs(r["p_ref"] - r["p_diff"]) for r in rows])
    bad = [r for r in rows if abs(r["p_ref"] - r["p_diff"]) > 0.05 or not r["grad_finite"] or r["grad_norm"] == 0]
    print(f"max |p_ref-p_diff| = {d.max():.4f}, mean {d.mean():.4f}; peak GB {max(r['peak_gb'] for r in rows):.1f}; bad {len(bad)}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
