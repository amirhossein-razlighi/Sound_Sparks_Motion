#!/usr/bin/env python3
"""Source fidelity (DINOv2-small CLS cosine) of every comparison video vs its source: first-frame identity and
time-aligned similarity over 16 uniform frames. Complements VBench, which only measures self-consistency."""
import csv, statistics as st, sys
from collections import defaultdict
import numpy as np, torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from select_frames_auto import decode_all
dev = "cuda" if torch.cuda.is_available() else "cpu"
proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small"); m = AutoModel.from_pretrained("facebook/dinov2-small").to(dev).eval()
rows = list(csv.DictReader(open(sys.argv[1])))
src = {r["scenario"]: r["video"] for r in rows if r["method"] == "input"}
@torch.no_grad()
def emb(frames):
    x = proc(images=[Image.fromarray(f) for f in frames], return_tensors="pt").to(dev)
    return torch.nn.functional.normalize(m(**x).last_hidden_state[:, 0], dim=-1)
def frames16(p):
    fr = decode_all(p); idx = np.linspace(0, len(fr) - 1, 16).round().astype(int); return [fr[i] for i in idx]
S = {s: emb(frames16(v)) for s, v in src.items()}
out = []
for r in rows:
    if r["method"] == "input": continue
    E = emb(frames16(r["video"])); s = r["scenario"]
    out.append(dict(scenario=s, method=r["method"], first_frame_fidelity=float(E[0] @ S[s][0]), aligned_fidelity=float((E * S[s]).sum(-1).mean())))
with open(sys.argv[2], "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
g = defaultdict(list)
for o in out: g[o["method"]].append(o)
for k, v in sorted(g.items(), key=lambda kv: -st.mean(x["first_frame_fidelity"] for x in kv[1])):
    print(k, round(st.mean(x["first_frame_fidelity"] for x in v), 3), round(st.mean(x["aligned_fidelity"] for x in v), 3))
