#!/usr/bin/env python3
"""P7 (part 1): RAFT flow magnitude + flow-warp error for every video of one or more manifests.

All videos are resized so the long side is 512 px (keeps aspect; our clips are 512x320) and
evaluated at their native frame rate / length, so numbers are comparable across methods.

  flow_mag     : mean |F_{t->t+1}| in pixels (RAFT-large, torchvision, cuDNN off - see eval_metrics fix)
  warp_err     : mean L1 (RGB in [0,1]) between frame t and frame t+1 backward-warped by F_{t->t+1},
                 over non-occluded pixels (forward-backward consistency mask, Sundaram et al. 2010
                 thresholds); lower = temporally more consistent
  occl_frac    : mean fraction of pixels masked as occluded
Output: <out>/flow_warp.csv (resumable), one row per manifest row.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F

from critic import load_video_chw

LONG = 512


def resize(fr: torch.Tensor) -> torch.Tensor:
    h, w = fr.shape[-2:]
    s = LONG / max(h, w)
    nh, nw = int(round(h * s / 8)) * 8, int(round(w * s / 8)) * 8
    return F.interpolate(fr, size=(nh, nw), mode="bilinear", align_corners=False).contiguous()


def backwarp(img, flow):
    n, _, h, w = img.shape
    yy, xx = torch.meshgrid(torch.arange(h, device=img.device), torch.arange(w, device=img.device), indexing="ij")
    gx = (xx[None] + flow[:, 0]) / (w - 1) * 2 - 1
    gy = (yy[None] + flow[:, 1]) / (h - 1) * 2 - 1
    return F.grid_sample(img, torch.stack([gx, gy], -1), mode="bilinear", padding_mode="border", align_corners=True)


@torch.no_grad()
def flows(raft, tf, a, b, bs=8):
    out = []
    for i in range(0, a.shape[0], bs):
        x, y = tf(a[i:i + bs], b[i:i + bs])
        with torch.backends.cudnn.flags(enabled=False):
            out.append(raft(x.contiguous(), y.contiguous())[-1])
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
    dev = torch.device("cuda")
    w = Raft_Large_Weights.DEFAULT
    raft, tf = raft_large(weights=w).to(dev).eval(), w.transforms()
    rows = [r for m in a.manifest for r in csv.DictReader(open(m))]
    dst = Path(a.out) / "flow_warp.csv"
    dst.parent.mkdir(parents=True, exist_ok=True)
    done = {r["id"] for r in csv.DictReader(open(dst))} if dst.exists() else set()
    new = not dst.exists()
    with open(dst, "a", newline="") as f:
        wr = None
        for r in rows:
            if r["id"] in done:
                continue
            try:
                fr = resize(load_video_chw(r["video"], dev))
                x0, x1 = fr[:-1], fr[1:]
                fw = flows(raft, tf, x0, x1)
                bw = flows(raft, tf, x1, x0)
                warped = backwarp(x1, fw)
                bw_w = backwarp(bw, fw)
                s = fw + bw_w
                occ = (s.pow(2).sum(1) > 0.01 * (fw.pow(2).sum(1) + bw_w.pow(2).sum(1)) + 0.5)
                valid = (~occ).float().unsqueeze(1)
                err = ((x0 - warped).abs() * valid).sum((1, 2, 3)) / (valid.sum((1, 2, 3)) * 3).clamp_min(1)
                res = dict(flow_mag=float(fw.norm(dim=1).mean()), warp_err=float(err.mean()),
                           occl_frac=float(occ.float().mean()), n_frames=int(fr.shape[0]),
                           res=f"{fr.shape[-1]}x{fr.shape[-2]}", error="")
            except Exception as e:
                res = dict(flow_mag="", warp_err="", occl_frac="", n_frames="", res="", error=repr(e)[:300])
            row = {k: r[k] for k in ("id", "group", "scenario", "method", "video")} | res
            if wr is None:
                wr = csv.DictWriter(f, fieldnames=list(row))
                if new:
                    wr.writeheader()
            wr.writerow(row); f.flush()
            print(r["id"], res.get("flow_mag"), res.get("warp_err"), res["error"], flush=True)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
