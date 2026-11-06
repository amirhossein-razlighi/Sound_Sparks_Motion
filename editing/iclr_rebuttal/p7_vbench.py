#!/usr/bin/env python3
"""P7 (part 2): VBench subject/background consistency, temporal flickering, motion smoothness.

Run with the separate VBench venv (/scratch/amirrz/iclr_rebuttal/venv_vbench, VBench 0.1.5, offline
`local=True` weights in VBENCH_CACHE_DIR). For each method we build a folder of mp4s
(<scenario>.mp4; gifs converted with ffmpeg), then call VBench in `custom_input` mode.
Output: <out>/vbench/<method>/... (VBench's own json) and <out>/vbench_per_video.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path

DIMS = ["subject_consistency", "background_consistency", "temporal_flickering", "motion_smoothness"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    from vbench import VBench
    import vbench as vb
    out = Path(a.out) / "vbench"
    rows = [r for m in a.manifest for r in csv.DictReader(open(m))]
    by_method: dict[str, list[dict]] = {}
    for r in rows:
        by_method.setdefault(f"{r['group']}__{r['method']}", []).append(r)
    per_video = []
    for meth, rs in sorted(by_method.items()):
        vdir = out / "videos" / meth
        vdir.mkdir(parents=True, exist_ok=True)
        for r in rs:
            dst = vdir / f"{r['scenario']}.mp4"
            if dst.exists():
                continue
            src = Path(r["video"])
            if src.suffix.lower() == ".mp4":
                os.symlink(src, dst)
            else:
                # UniEdit gifs: 2 panels (reconstruction | edit, 1030 = 512+6+512 px) -> keep the right panel
                vf = "crop=(iw-6)/2:ih:(iw+6)/2:0," if src.name == "uniedit.gif" else ""
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-pix_fmt", "yuv420p",
                                "-vf", vf + "scale=trunc(iw/2)*2:trunc(ih/2)*2", str(dst)], check=True)
        res_json = out / "results" / meth
        res_json.mkdir(parents=True, exist_ok=True)
        if not list(res_json.glob("*_eval_results.json")):
            bench = VBench(torch.device("cuda"), str(Path(vb.__file__).parent / "VBench_full_info.json"), str(res_json))
            bench.evaluate(videos_path=str(vdir), name=meth, dimension_list=DIMS, local=True, mode="custom_input")
        res = json.loads(next(res_json.glob("*_eval_results.json")).read_text())
        for dim in DIMS:
            for v in res[dim][1]:
                per_video.append(dict(method=meth, scenario=Path(v["video_path"]).stem, dim=dim,
                                      score=v["video_results"]))
        print(meth, {d: res[d][0] for d in DIMS}, flush=True)
    with open(Path(a.out) / "vbench_per_video.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "scenario", "dim", "score"]); w.writeheader(); w.writerows(per_video)


if __name__ == "__main__":
    main()
