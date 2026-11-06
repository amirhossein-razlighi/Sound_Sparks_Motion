#!/usr/bin/env python3
"""Pairwise VLM judge: ours vs another video, both orders (A/B swapped), two questions.

For each pair (ours, other) and each question we ask the VLM with Video A and Video B in one user turn and read
P(A) vs P(B) at the first answer position (logsumexp over single-token variants of "A"/"B"). We run order 1
(A = ours) and order 2 (A = other) and report p_ours = mean(P(A | order1), P(B | order2)) - position bias cancels.

Frames: 16 uniform per video, long side 512 (same sampler as heldout_score.py; UniEdit gif crop not needed here).
Output: <out>/pairwise_<model>.csv (resumable): one row per (pair, question) with both orders.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch

from heldout_score import MODELS, read_frames

Q = {
    "overall": ('Both videos are edits of the same source video with the instruction: "{p}". Which video performs the '
                'requested action or state change better, while keeping the subject and scene of the source? '
                'Answer with a single letter: A or B.'),
    "natural": ('Both videos are edits of the same source video with the instruction: "{p}". In which video is the motion '
                'more natural and physically plausible, with smooth progression and without sudden jumps, abrupt '
                'appearance changes or visual artifacts? Answer with a single letter: A or B.'),
}


def ids(tok, words):
    out = set()
    for w in words:
        for f in (w, " " + w):
            t = tok.encode(f, add_special_tokens=False)
            if len(t) == 1:
                out.add(t[0])
    return sorted(out)


class Judge:
    def __init__(self, name, device="cuda"):
        from transformers import AutoProcessor
        self.name, self.device = name, device
        path = MODELS[name]
        if name == "qwen25":
            from transformers import Qwen2_5_VLForConditionalGeneration as M
            self.proc = AutoProcessor.from_pretrained(path, min_pixels=128 * 28 * 28, max_pixels=512 * 512)
        elif name == "qwen3vl":
            from transformers import Qwen3VLForConditionalGeneration as M
            self.proc = AutoProcessor.from_pretrained(path)
        else:
            from transformers import InternVLForConditionalGeneration as M
            self.proc = AutoProcessor.from_pretrained(path)
        self.model = M.from_pretrained(path, dtype=torch.bfloat16, device_map=device).eval()
        self.tok = self.proc.tokenizer
        self.A, self.B = ids(self.tok, ["A"]), ids(self.tok, ["B"])

    @torch.no_grad()
    def pa(self, va, vb, question):
        fa, fpsa, Ta, ia = va
        fb, fpsb, Tb, ib = vb
        content = [{"type": "text", "text": "Video A:"}, {"type": "video"}, {"type": "text", "text": "Video B:"},
                   {"type": "video"}, {"type": "text", "text": question}]
        text = self.proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
        vids = [np.stack(fa), np.stack(fb)]
        if self.name == "qwen25":
            inp = self.proc(text=[text], videos=vids, fps=[len(fa) / (Ta / fpsa), len(fb) / (Tb / fpsb)], return_tensors="pt")
        elif self.name == "qwen3vl":
            from transformers.video_utils import VideoMetadata
            md = [VideoMetadata(total_num_frames=Ta, fps=fpsa, frames_indices=ia, duration=Ta / fpsa),
                  VideoMetadata(total_num_frames=Tb, fps=fpsb, frames_indices=ib, duration=Tb / fpsb)]
            inp = self.proc(text=[text], videos=vids, video_metadata=md, do_sample_frames=False, return_tensors="pt")
        else:
            inp = self.proc(text=[text], videos=vids, return_tensors="pt")
        inp = inp.to(self.device)
        for k, v in inp.items():
            if torch.is_tensor(v) and v.is_floating_point():
                inp[k] = v.to(torch.bfloat16)
        lp = torch.log_softmax(self.model(**inp).logits[0, -1].float(), -1)
        la, lb = torch.logsumexp(lp[self.A], 0), torch.logsumexp(lp[self.B], 0)
        return float(torch.softmax(torch.stack([la, lb]), 0)[0]), float(math.exp(torch.logaddexp(la, lb).item()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="CSV: pair_id,scenario,edit_prompt,ours,other,other_method")
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default="qwen25,internvl3")
    a = ap.parse_args()
    pairs = list(csv.DictReader(open(a.pairs)))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cache = {}

    def frames(p):
        if p not in cache:
            cache[p] = read_frames(p)
        return cache[p]

    for name in a.models.split(","):
        dst = out / f"pairwise_{name}.csv"
        done = {(r["pair_id"], r["question"]) for r in csv.DictReader(open(dst))} if dst.exists() else set()
        todo = [(p, q) for p in pairs for q in Q if (p["pair_id"], q) not in done]
        if not todo:
            continue
        j = Judge(name)
        print(f"[{name}] A ids {j.A} B ids {j.B}; {len(todo)} items", flush=True)
        new = not dst.exists()
        with open(dst, "a", newline="") as f:
            w = None
            for p, q in todo:
                qt = Q[q].format(p=p["edit_prompt"])
                try:
                    vo, vx = frames(p["ours"]), frames(p["other"])
                    a1, m1 = j.pa(vo, vx, qt)   # A = ours
                    a2, m2 = j.pa(vx, vo, qt)   # A = other
                    res = dict(p_ours_order1=a1, p_ours_order2=1 - a2, p_ours=(a1 + (1 - a2)) / 2,
                               position_bias_A=(a1 + a2) / 2, ab_mass=min(m1, m2), error="")
                except Exception as e:
                    res = dict(p_ours_order1="", p_ours_order2="", p_ours="", position_bias_A="", ab_mass="", error=repr(e)[:300])
                row = {**p, "question": q, "model": name, **res}
                if w is None:
                    w = csv.DictWriter(f, fieldnames=list(row))
                    if new:
                        w.writeheader()
                w.writerow(row); f.flush()
                print(name, p["pair_id"], q, res.get("p_ours"), res["error"], flush=True)
                torch.cuda.empty_cache()
        del j
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
