#!/usr/bin/env python3
"""Trim generated sources to the retake geometry (89 frames = 3.7 s) for every candidate whose src ends in _89f.mp4.
Usage: trim_sources.py candidates_rN.json   (login node; ffmpeg -threads 1; skips files that already exist)."""
import json, os, subprocess, sys
c = {k: v for k, v in json.load(open(sys.argv[1])).items() if not k.startswith("_")}
for k, v in c.items():
    s89, w89 = v["src"], v["wav"]
    if not s89.endswith("_89f.mp4"):
        continue
    src = s89.replace("_89f.mp4", ".mp4"); wav = w89.replace("_89f.wav", ".wav")
    if not os.path.exists(src):
        print("[missing]", k, src); continue
    if not os.path.exists(s89):
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-threads", "1", "-i", src, "-frames:v", "89", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", "-an", s89], check=True)
    if os.path.exists(wav) and not os.path.exists(w89):
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-threads", "1", "-i", wav, "-t", "3.7083", w89], check=True)
    print("[ok]", k)
