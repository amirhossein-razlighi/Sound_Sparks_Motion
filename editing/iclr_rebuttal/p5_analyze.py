#!/usr/bin/env python3
"""P5 analysis: per (latent, target) pair -> predictors + transfer outcome CSV.

Predictors (computed here, offline models from the HF cache):
  * text_cos      : cosine(CLIP ViT-L/14 text emb(source edit prompt), emb(target edit prompt))
  * dino_ff_cos   : cosine(DINOv2-small CLS of first frame of source video, of target video)
  * category      : subject-category match label (from targets.csv, hand-assigned)
Outcomes (from heldout/<model>.csv rows with group=transfer):
  * <model>_transfer, <model>_baseline, <model>_gain   for qwen25 / qwen3vl / internvl3
  * success_<model> = transfer >= 0.5 and baseline < 0.5   (flip), and gain > 0
Writes transfer/pairs.csv and prints success rates + Spearman correlations.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from common import OUT_ROOT, PAPER_RESULTS
from p5_setup import LATENTS

MODELS = ["qwen25", "qwen3vl", "internvl3"]


def first_frame(path: str):
    import av
    from PIL import Image
    c = av.open(path)
    f = next(c.decode(video=0)).to_image().convert("RGB")
    c.close()
    return f


def main():
    root = OUT_ROOT / "transfer"
    tg = list(csv.DictReader(open(root / "targets.csv")))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPTokenizer
    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(dev).eval()
    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    dproc = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    dino = AutoModel.from_pretrained("facebook/dinov2-small").to(dev).eval()

    @torch.no_grad()
    def temb(t):
        e = clip.get_text_features(**tok([t], return_tensors="pt", padding=True).to(dev))
        return torch.nn.functional.normalize(e, dim=-1)[0]

    @torch.no_grad()
    def iemb(path):
        x = dproc(images=first_frame(path), return_tensors="pt").to(dev)
        return torch.nn.functional.normalize(dino(**x).last_hidden_state[:, 0], dim=-1)[0]

    src_video = {k: str(PAPER_RESULTS / v / "retake_input_prepared.mp4") for k, v in LATENTS.items()}
    scores = {}
    for m in MODELS:
        p = OUT_ROOT / "heldout" / f"{m}.csv"
        if p.exists():
            for r in csv.DictReader(open(p)):
                if r["group"] == "transfer" and r["yes_prob"] != "":
                    scores[(m, r["scenario"], r["method"])] = float(r["yes_prob"])
    out = []
    for t in tg:
        key = f"{t['latent']}__{t['target']}"
        row = {k: t[k] for k in ("latent", "target", "source_prompt", "edit_prompt", "subject_category_match", "style", "note")}
        row["text_cos"] = float(temb(t["source_prompt"]) @ temb(t["edit_prompt"]))
        row["dino_ff_cos"] = float(iemb(src_video[t["latent"]]) @ iemb(t["target_video"]))
        row["transfer_video"] = str(Path(t["out_dir"]) / "transfer_both.mp4")
        row["baseline_video"] = str(Path(t["out_dir"]) / "baseline_video.mp4")
        for m in MODELS:
            a, b = scores.get((m, key, "transfer")), scores.get((m, key, "baseline"))
            row[f"{m}_transfer"], row[f"{m}_baseline"] = a, b
            row[f"{m}_gain"] = (a - b) if a is not None and b is not None else None
            row[f"{m}_flip"] = (a >= 0.5 and b < 0.5) if a is not None and b is not None else None
        out.append(row)
    with open(root / "pairs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    from scipy.stats import spearmanr
    lines = ["# P5 transfer summary", ""]
    for m in MODELS:
        g = [r for r in out if r[f"{m}_gain"] is not None]
        if not g:
            continue
        gains = np.array([r[f"{m}_gain"] for r in g])
        lines.append(f"## {m}: n={len(g)} pairs; mean gain {gains.mean():+.3f}; gain>0 in {np.mean(gains > 0):.0%}; "
                     f"transfer>=0.5 in {np.mean([r[f'{m}_transfer'] >= 0.5 for r in g]):.0%} vs baseline "
                     f"{np.mean([r[f'{m}_baseline'] >= 0.5 for r in g]):.0%}")
        for lat in LATENTS:
            gl = [r for r in g if r["latent"] == lat]
            if gl:
                lines.append(f"- {lat}: gain>0 {sum(r[f'{m}_gain'] > 0 for r in gl)}/{len(gl)}, mean gain {np.mean([r[f'{m}_gain'] for r in gl]):+.3f}")
        for pred in ("text_cos", "dino_ff_cos"):
            rho, p = spearmanr([r[pred] for r in g], gains)
            lines.append(f"- Spearman(gain, {pred}) = {rho:+.2f} (p={p:.2f})")
        for cat in sorted({r["subject_category_match"] for r in g}):
            gc = [r[f"{m}_gain"] for r in g if r["subject_category_match"] == cat]
            lines.append(f"- category={cat}: n={len(gc)}, mean gain {np.mean(gc):+.3f}, gain>0 {np.mean(np.array(gc) > 0):.0%}")
        lines.append("")
    (root / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
