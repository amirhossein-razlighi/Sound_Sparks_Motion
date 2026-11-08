#!/usr/bin/env python3
"""Critic-independent metrics for every final/preview video in $ROOT/manifest.csv (GPU or CPU):
  dino_first / dino_aligned : DINOv2-small CLS cosine to the prepared input (first frame; mean over 16 time-aligned
                              frames) - source preservation, same rule as source_fidelity.py
  motion                    : mean |frame difference| (grey, 256x144, [0,1]) - how much the video moves
  mad_vs_input              : mean |pixel difference| to the input, time-aligned (all frames, 256x144)
  mad_vs_retake             : same vs the prompt-only Retake render of that scenario
Writes $ROOT/metrics.csv."""
import csv
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

ROOT = Path("/scratch/amirrz/iclr_rebuttal/multicritic")
dev = "cuda" if torch.cuda.is_available() else "cpu"
proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
m = AutoModel.from_pretrained("facebook/dinov2-small").to(dev).eval()


def frames(p):
    c = av.open(str(p)); f = [x.to_ndarray(format="rgb24") for x in c.decode(video=0)]; c.close(); return f


def small(f):
    return np.stack([np.asarray(Image.fromarray(x).resize((256, 144), Image.BILINEAR)) for x in f]).astype(np.float32) / 255


@torch.no_grad()
def emb(fr):
    idx = np.linspace(0, len(fr) - 1, 16).round().astype(int)
    x = proc(images=[Image.fromarray(fr[i]) for i in idx], return_tensors="pt").to(dev)
    return torch.nn.functional.normalize(m(**x).last_hidden_state[:, 0], dim=-1)


rows = [r for r in csv.DictReader(open(ROOT / "manifest.csv")) if r["group"] in ("final", "preview")]
ref = {}
for r in rows:
    if r["method"] in ("input", "retake"):
        f = frames(r["video"]); ref[(r["scenario"], r["method"])] = (emb(f), small(f))
out = []
for r in rows:
    f = frames(r["video"]); s = small(f); e = emb(f)
    E_in, S_in = ref[(r["scenario"], "input")]
    o = dict(id=r["id"], scenario=r["scenario"], method=r["method"], n_frames=len(f),
             dino_first=float(e[0] @ E_in[0]), dino_aligned=float((e * E_in).sum(-1).mean()),
             motion=float(np.abs(np.diff(s.mean(-1), axis=0)).mean()))
    n = min(len(s), len(S_in)); o["mad_vs_input"] = float(np.abs(s[:n] - S_in[:n]).mean())
    if (r["scenario"], "retake") in ref:
        S_r = ref[(r["scenario"], "retake")][1]; n = min(len(s), len(S_r)); o["mad_vs_retake"] = float(np.abs(s[:n] - S_r[:n]).mean())
    out.append(o); print(o, flush=True)
with open(ROOT / "metrics.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["id", "scenario", "method", "n_frames", "dino_first", "dino_aligned", "motion", "mad_vs_input", "mad_vs_retake"])
    w.writeheader(); w.writerows(out)
print("wrote", ROOT / "metrics.csv", len(out))

# ---- render gap: in-loop differentiable render (what the critics score) vs the normal render of the same latents ----
# iteration 1 uses the initial latents, so iter_001 (in-loop) vs baseline_video.mp4 (prompt-only Retake render, same
# seed) isolates the render-path difference; also the paper's prompt-only Retake for reference.
import re as _re
gap = []
for sdir in sorted(d for d in ROOT.iterdir() if d.is_dir()):
    for r in sorted(sdir.glob("[AB]_*")):
        it1, base = r / "multicritic" / "iter_frames" / "iter_001.mp4", r / "baseline_video.mp4"
        if ".old" in r.name or not (it1.exists() and base.exists()):
            continue
        a, b = small(frames(it1)), small(frames(base)); n = min(len(a), len(b))
        d = np.abs(a[:n] - b[:n]).mean(axis=(1, 2, 3))
        gap.append(dict(scenario=sdir.name, run=r.name, mad_mean=float(d.mean()), mad_first_frame=float(d[0]),
                        mad_last_frame=float(d[-1]), mad_max_frame=float(d.max())))
        print(gap[-1], flush=True)
if gap:
    with open(ROOT / "render_gap.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(gap[0])); w.writeheader(); w.writerows(gap)
