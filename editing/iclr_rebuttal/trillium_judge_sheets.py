#!/usr/bin/env python3
"""TRILLIUM (additive): blind pairwise-judging packets, ours vs best-of-8 pick, with the SOURCE shown as reference.

For each scenario and each order (1: A=ours, 2: A=bon8), writes <out>/<hashed id>/ (id = sha256(salt|scenario|order)[:10]):
  sheet_1..4.png  time-aligned contact sheets, rows = SOURCE / A / B, 6 frames per sheet (every 4th frame of the clip
                  plus the frames of largest change in A or B, so sudden jumps cannot fall between samples)
  motion.png      per-frame change curve (mean |f_t - f_{t-1}|) of SOURCE, A, B -> abrupt events show as spikes
  prompt.txt      the edit instruction
The key (hashed id -> scenario, order, which of A/B is ours) is written to --key, OUTSIDE the judge folder.
Frame indices printed on every tile. CPU only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = "/cvmfs/soft.computecanada.ca/gentoo/2023/x86-64-v3/usr/share/fonts/dejavu/DejaVuSans.ttf"


def decode(path: str) -> np.ndarray:
    import av
    c = av.open(path)
    f = np.stack([x.to_ndarray(format="rgb24") for x in c.decode(video=0)])
    c.close()
    return f


def change(v: np.ndarray) -> np.ndarray:
    g = v.astype(np.float32).mean(-1)
    return np.r_[0.0, np.abs(np.diff(g, axis=0)).mean((1, 2))]


def pick_frames(n: int, ca: np.ndarray, cb: np.ndarray, k_extra: int = 1) -> list[int]:
    base = set(range(0, n, 4)) | {n - 1}
    for c in (ca, cb):  # add the k largest-change frames of each edited video (and the frame before)
        for t in np.argsort(c)[::-1][:k_extra]:
            base |= {int(t), max(int(t) - 1, 0)}
    return sorted(base)


def tile(img: np.ndarray, label: str, w: int, h: int, font) -> Image.Image:
    im = Image.fromarray(img).resize((w, h), Image.BILINEAR)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, w, 18], fill=(0, 0, 0))
    d.text((4, 1), label, fill=(255, 255, 255), font=font)
    return im


def sheets(src, a, b, idx, out: Path, w=256, h=160, per=6):
    font = ImageFont.truetype(FONT, 13)
    rows = [("SOURCE", src), ("A", a), ("B", b)]
    paths = []
    for si in range(0, len(idx), per):
        chunk = idx[si:si + per]
        S = Image.new("RGB", (w * per + 70, h * 3), (40, 40, 40))
        d = ImageDraw.Draw(S)
        for r, (name, v) in enumerate(rows):
            d.text((6, r * h + h // 2 - 8), name, fill=(255, 255, 0), font=ImageFont.truetype(FONT, 16))
            for c, t in enumerate(chunk):
                S.paste(tile(v[min(t, len(v) - 1)], f"{name} frame {t}", w, h, font), (70 + c * w, r * h))
        p = out / f"sheet_{si // per + 1}.png"
        S.save(p)
        paths.append(p.name)
    return paths


def motion_plot(cs, ca, cb, out: Path, W=1000, H=300, pad=50):
    """PIL line plot (no matplotlib in the venv): per-frame change of SOURCE (gray), A (blue), B (red)."""
    font = ImageFont.truetype(FONT, 14)
    im = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(im)
    n = len(cs); ymax = max(float(np.max(cs)), float(np.max(ca)), float(np.max(cb)), 1e-6) * 1.05
    X = lambda t: pad + (W - 2 * pad) * t / max(n - 1, 1)
    Y = lambda v: H - pad - (H - 2 * pad) * v / ymax
    d.line([(pad, H - pad), (W - pad, H - pad)], fill="black"); d.line([(pad, pad), (pad, H - pad)], fill="black")
    for t in range(0, n, 8):
        d.line([(X(t), H - pad), (X(t), H - pad + 5)], fill="black"); d.text((X(t) - 6, H - pad + 7), str(t), fill="black", font=font)
    for c, lab, col, ly in ((cs, "SOURCE", (128, 128, 128), 8), (ca, "A", (31, 119, 180), 26), (cb, "B", (214, 39, 40), 44)):
        d.line([(X(t), Y(float(v))) for t, v in enumerate(c)], fill=col, width=2)
        d.text((W - pad - 70, ly), lab, fill=col, font=font)
    d.text((pad, 8), "per-frame change, mean |f_t - f_(t-1)|  (spikes = abrupt events); x = frame index", fill="black", font=font)
    im.save(out / "motion.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", default="/scratch/amirrz/iclr_rebuttal/bon8_package")
    ap.add_argument("--out", default="/scratch/amirrz/iclr_rebuttal/judge_blind", help="judge-visible folder (no key)")
    ap.add_argument("--key", default="/scratch/amirrz/iclr_rebuttal/judge_key.json", help="key file, OUTSIDE --out")
    ap.add_argument("--salt", default="trillium-bon8-2026-11-06")
    a = ap.parse_args()
    import hashlib
    P, O = Path(a.package), Path(a.out)
    assert not Path(a.key).resolve().is_relative_to(O.resolve()), "key must live outside the judge folder"
    key = {}
    for d in sorted(x for x in P.iterdir() if x.is_dir()):
        bon = sorted(d.glob("seeds_best_of8__*.mp4"))
        if not bon:
            print(d.name, "skip (bon8 pick missing)")
            continue
        src, ours, bon8 = decode(str(d / "input.mp4")), decode(str(d / "ours_best.mp4")), decode(str(bon[0]))
        prompt = (d / "prompt.txt").read_text().splitlines()[0].split(":", 1)[1].strip()
        for order, (A, B, a_is) in ((1, (ours, bon8, "ours")), (2, (bon8, ours, "bon8"))):
            hid = hashlib.sha256(f"{a.salt}|{d.name}|{order}".encode()).hexdigest()[:10]
            od = O / hid; od.mkdir(parents=True, exist_ok=True)
            cs, ca, cb = change(src), change(A), change(B)
            n = min(len(src), len(A), len(B))
            idx = pick_frames(n, ca[:n], cb[:n])
            files = sheets(src, A, B, idx, od)
            motion_plot(cs[:n], ca[:n], cb[:n], od)
            (od / "prompt.txt").write_text(prompt + "\n")
            key[hid] = {"scenario": d.name, "order": order, "A": a_is, "B": "bon8" if a_is == "ours" else "ours",
                        "bon8_file": bon[0].name, "frames": idx}
            print(d.name, f"order{order}", hid, len(idx), "frames", len(files), "sheets")
    Path(a.key).write_text(json.dumps(key, indent=1))
    print("key ->", a.key, len(key), "packets")


if __name__ == "__main__":
    main()
