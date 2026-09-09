#!/usr/bin/env python3
"""Critic-independent, viewer-independent timeline of a clip: per-frame mean |frame diff| (motion) and mean brightness,
binned into 12 segments, plus the frame of the largest change.  Login-node light (ffmpeg -threads 1, 128 px grey).
Usage: motion_timeline.py video [video ...]"""
import subprocess, sys, numpy as np
def load(p, w=128):
    out = subprocess.run(["ffmpeg", "-v", "error", "-threads", "1", "-i", p, "-vf", f"scale={w}:-2,format=gray", "-f", "rawvideo", "-"],
                         capture_output=True).stdout
    h = int(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", p],
                           capture_output=True, text=True).stdout.strip().split(",")[1]) * w // int(subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width", "-of", "csv=p=0", p], capture_output=True, text=True).stdout.strip())
    h -= h % 2
    a = np.frombuffer(out, np.uint8); n = a.size // (w * h); return a[: n * w * h].reshape(n, h, w).astype(np.float32)
for p in sys.argv[1:]:
    v = load(p); d = np.abs(np.diff(v, axis=0)).mean(axis=(1, 2)); b = v.mean(axis=(1, 2))
    bins = np.array_split(np.arange(len(d)), 12)
    md = [d[i].mean() for i in bins]; mb = [b[i].mean() for i in bins]
    print(f"{p.split('/')[-1]:24s} frames={len(v):3d} motion/bin: " + " ".join(f"{x:4.1f}" for x in md) +
          f" | peak diff at frame {int(d.argmax())+1} ({d.max():.1f}) | bright/bin: " + " ".join(f"{x:3.0f}" for x in mb))
