#!/usr/bin/env python3
"""VBench Dynamic Degree WITHOUT the final yes/no threshold (venv_vbench).

Uses VBench's own DynamicDegree class unchanged (RAFT raft-things, VBench frame sampling at ~8 fps, per frame pair the mean
of the top-5% optical-flow magnitudes = `max_rad`, threshold thres = 6 * min(H,W)/256). VBench then says "dynamic" iff at
least round(4*n/16) pairs exceed thres. We report, per video: dyn_mean = mean(max_rad)/thres (continuous, 1.0 = at VBench's
threshold), frac_above = fraction of pairs above thres, vbench_flag = VBench's own boolean (identical rule).
  p7_dyn_cont.py --manifest M.csv [--manifest ...] --out CSV
"""
import argparse, csv
import torch
from easydict import EasyDict as edict
from vbench.dynamic_degree import DynamicDegree
from vbench.third_party.RAFT.core.utils_core.utils import InputPadder

ap = argparse.ArgumentParser(); ap.add_argument("--manifest", action="append", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--raft", default="/scratch/amirrz/iclr_rebuttal/vbench_cache/raft_model/models/raft-things.pth"); a = ap.parse_args()
dd = DynamicDegree(edict({"model": a.raft, "small": False, "mixed_precision": False, "alternate_corr": False}), torch.device("cuda"))
rows = []
for m in a.manifest:
    for r in csv.DictReader(open(m)):
        v = r["video"]
        if not v.endswith(".mp4"):
            continue
        with torch.no_grad():
            fr = dd.get_frames(v); dd.set_params(frame=fr[0], count=len(fr)); sc = []
            for i1, i2 in zip(fr[:-1], fr[1:]):
                p = InputPadder(i1.shape); i1, i2 = p.pad(i1, i2)
                _, f = dd.model(i1, i2, iters=20, test_mode=True); sc.append(dd.get_score(i1, f))
        th = dd.params["thres"]
        rows.append(dict(id=r["id"], group=r.get("group", ""), scenario=r["scenario"], method=r["method"], n_pairs=len(sc), thres=th,
                         dyn_mean=sum(sc) / len(sc) / th, frac_above=sum(s > th for s in sc) / len(sc), vbench_flag=dd.check_move(sc)))
        print(rows[-1]["id"], f"{rows[-1]['dyn_mean']:.3f}", rows[-1]["vbench_flag"], flush=True)
with open(a.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(len(rows), "->", a.out)
