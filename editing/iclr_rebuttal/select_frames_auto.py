#!/usr/bin/env python3
"""Method-agnostic, motion-weighted frame selection for VLM critics (CPU).

For every video: m_t = mean |f_t - f_{t-1}| on 128-px grayscale frames (t >= 1). Frames are sampled by the inverse CDF
of the cumulative change (plus a small uniform floor so static stretches still get some frames), always including the
first and last frame -> N frames concentrated where the video actually changes. Same rule for every method.
Output JSON: {manifest id: [frame indices]} (indices into the full decoded clip; UniEdit gif crop as elsewhere).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from critic import uniedit_panel


def decode_all(path: str) -> list[np.ndarray]:
    p = Path(path)
    if p.suffix.lower() == ".gif":
        import imageio.v3 as iio
        return uniedit_panel([np.asarray(f)[..., :3] for f in iio.imread(str(p), index=None)], p)
    import av
    c = av.open(str(p)); fr = [f.to_ndarray(format="rgb24") for f in c.decode(c.streams.video[0])]; c.close()
    return fr


def motion_weighted(frames: list[np.ndarray], n: int, floor: float = 0.15) -> list[int]:
    T = len(frames)
    if T <= n:
        return list(range(T)) + [T - 1] * (n - T)
    g = [np.asarray(Image.fromarray(f).convert("L").resize((128, 80)), dtype=np.float32) / 255 for f in frames]
    m = np.array([0.0] + [np.abs(g[t] - g[t - 1]).mean() for t in range(1, T)])
    w = m / max(m.sum(), 1e-8)
    w = (1 - floor) * w + floor / T
    cdf = np.cumsum(w); cdf /= cdf[-1]
    q = np.linspace(0, 1, n)[1:-1]
    mid = np.searchsorted(cdf, q).clip(1, T - 2)
    idx = sorted(set([0, T - 1, *mid.tolist()]))
    while len(idx) < n:  # fill duplicates collapsed by set() with the largest gaps
        gaps = np.diff(idx); k = int(np.argmax(gaps)); idx.insert(k + 1, idx[k] + gaps[k] // 2); idx = sorted(set(idx))
        if len(idx) >= T:
            break
    return idx[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=16)
    a = ap.parse_args()
    res = {}
    for m in a.manifest:
        for r in csv.DictReader(open(m)):
            fr = decode_all(r["video"])
            res[r["id"]] = motion_weighted(fr, a.n)
    Path(a.out).write_text(json.dumps(res))
    print(len(res), "videos ->", a.out)


if __name__ == "__main__":
    main()
