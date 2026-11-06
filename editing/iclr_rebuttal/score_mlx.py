#!/usr/bin/env python3
"""Held-out VLM critic on Apple Silicon (MLX). Same protocol as the cluster scorer
(editing/iclr_rebuttal/heldout_score.py):

  * input: the 16 pre-extracted frames in <bundle>/<frames_dir>/00..15.jpg (uniform over the clip,
    long side 512 px; identical frames to what the cluster VLMs see)
  * prompt: the `question` column of manifest.csv, verbatim, as ONE user turn with the 16 frames
    as images (Gemma 3 has no native video input; multi-image is its standard video recipe),
    chat template applied, generation prompt added
  * yes_prob = softmax([logsumexp(logits[yes ids]), logsumexp(logits[no ids])])[0] at the first
    answer position, yes ids = single-token encodings of {yes, Yes, YES, " yes", " Yes", " YES"},
    no ids likewise; also report yesno_mass (prob. mass on those ids) and the argmax token.

If the installed mlx-vlm version has a different API, adapt ONLY the marked lines in `forward_logits`
(input preparation / forward call); do not change frames, question, token sets or the formula.

Usage:  python score_mlx.py --bundle . --model mlx-community/gemma-3-12b-it-4bit --out results_mac/gemma3_12b_4bit.csv
        [--limit 3]   (smoke test on the first 3 rows)
"""
from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image


def variant_ids(tok, words):
    ids = set()
    for w in words:
        for form in (w, " " + w):
            t = tok.encode(form, add_special_tokens=False)
            if len(t) == 1:
                ids.add(int(t[0]))
    return sorted(ids)


def forward_logits(model, processor, config, images, question):
    # ---- mlx-vlm API specific (adapt here if needed) -------------------------------------------
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import prepare_inputs
    prompt = apply_chat_template(processor, config, question, num_images=len(images))
    inputs = prepare_inputs(processor, images=images, prompts=prompt,
                            image_token_index=getattr(model.config, "image_token_index", None))
    input_ids = inputs["input_ids"]
    pixel_values = inputs.get("pixel_values")
    mask = inputs.get("attention_mask")
    extra = {k: v for k, v in inputs.items() if k not in ("input_ids", "pixel_values", "attention_mask")}
    out = model(input_ids, pixel_values, mask=mask, **extra)
    logits = out.logits if hasattr(out, "logits") else out
    # ---------------------------------------------------------------------------------------------
    return logits[0, -1].astype(mx.float32), int(input_ids.shape[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=".")
    ap.add_argument("--model", default="mlx-community/gemma-3-12b-it-4bit")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    from mlx_vlm import load
    model, processor = load(a.model)
    config = model.config
    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    yes, no = variant_ids(tok, ["yes", "Yes", "YES"]), variant_ids(tok, ["no", "No", "NO"])
    print("yes ids", yes, "no ids", no, flush=True)
    rows = list(csv.DictReader(open(Path(a.bundle) / "manifest.csv")))
    if a.limit:
        rows = rows[: a.limit]
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = {r["id"] for r in csv.DictReader(open(out))} if out.exists() else set()
    new = not out.exists()
    with open(out, "a", newline="") as f:
        w = None
        for r in rows:
            if r["id"] in done:
                continue
            t0 = time.time()
            try:
                fd = Path(a.bundle) / r["frames_dir"]
                images = [Image.open(fd / f"{i:02d}.jpg").convert("RGB") for i in range(16)]
                logits, n_tok = forward_logits(model, processor, config, images, r["question"])
                lp = logits - mx.logsumexp(logits)
                ly = mx.logsumexp(lp[mx.array(yes)]).item(); ln = mx.logsumexp(lp[mx.array(no)]).item()
                m = max(ly, ln)
                res = dict(yes_prob=math.exp(ly - m) / (math.exp(ly - m) + math.exp(ln - m)),
                           yesno_mass=math.exp(ly) + math.exp(ln),
                           argmax_token=tok.decode([int(mx.argmax(logits).item())]), n_input_tokens=n_tok, error="")
            except Exception as e:
                res = dict(yes_prob="", yesno_mass="", argmax_token="", n_input_tokens="", error=repr(e)[:300])
            res["seconds"] = round(time.time() - t0, 1)
            row = {k: r[k] for k in ("id", "group", "scenario", "method", "edit_prompt")} | {"model": a.model} | res
            if w is None:
                w = csv.DictWriter(f, fieldnames=list(row))
                if new:
                    w.writeheader()
            w.writerow(row); f.flush()
            print(r["id"], res["yes_prob"], res["yesno_mass"], res["argmax_token"], res["seconds"], res["error"], flush=True)
            mx.clear_cache() if hasattr(mx, "clear_cache") else None


if __name__ == "__main__":
    main()
