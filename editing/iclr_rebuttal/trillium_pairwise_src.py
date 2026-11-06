#!/usr/bin/env python3
"""TRILLIUM (additive): pairwise VLM judge WITH the source video as reference (extends pairwise_judge.py, unchanged).

Per pair (ours, other) and question: the VLM sees "Source video", "Video A", "Video B" in one turn and we read P(A) vs P(B)
at the first answer position. Both orders (A = ours, then A = other); p_ours = mean(P(A|order1), P(B|order2)).
Frames: --n-frames uniform per video (default 24, denser than the 16 of pairwise_judge.py), long side 512 as there.
Pairs CSV: pair_id,scenario,edit_prompt,source,ours,other,other_method
Output: <out>/pairwise_src_<model>.csv (resumable).
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch

from heldout_score import read_frames
from pairwise_judge import Judge

PRE = 'The first video is the SOURCE. Videos A and B are two edits of this source with the instruction: "{p}". '
Q = {
    "overall": PRE + ('Which edit performs the requested action better while keeping the subject, identity and scene of the '
                      'source unchanged otherwise? Answer with a single letter: A or B.'),
    "natural": PRE + ('In which edit is the motion more natural and physically plausible, progressing smoothly, without sudden '
                      'jumps, objects or people suddenly appearing, abrupt appearance changes, identity drift or visual '
                      'artifacts? Answer with a single letter: A or B.'),
    "faithful": PRE + ('Which edit stays more faithful to the source video (same subject, appearance, framing, camera and '
                       'background), apart from the requested change? Answer with a single letter: A or B.'),
}


class SrcJudge(Judge):
    @torch.no_grad()
    def pa3(self, vs, va, vb, question):
        V = [vs, va, vb]
        content = [{"type": "text", "text": "Source video:"}, {"type": "video"}, {"type": "text", "text": "Video A:"},
                   {"type": "video"}, {"type": "text", "text": "Video B:"}, {"type": "video"}, {"type": "text", "text": question}]
        text = self.proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
        vids = [np.stack(v[0]) for v in V]
        if self.name == "qwen25":
            inp = self.proc(text=[text], videos=vids, fps=[len(f) / (T / fps) for f, fps, T, _ in V], return_tensors="pt")
        elif self.name == "qwen3vl":
            from transformers.video_utils import VideoMetadata
            md = [VideoMetadata(total_num_frames=T, fps=fps, frames_indices=ix, duration=T / fps) for _, fps, T, ix in V]
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
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default="qwen3vl,internvl3")
    ap.add_argument("--n-frames", type=int, default=24)
    a = ap.parse_args()
    pairs = list(csv.DictReader(open(a.pairs)))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cache = {}

    def frames(p):
        if p not in cache:
            cache[p] = read_frames(p, n_frames=a.n_frames)
        return cache[p]

    for name in a.models.split(","):
        dst = out / f"pairwise_src_{name}.csv"
        done = {(r["pair_id"], r["question"]) for r in csv.DictReader(open(dst))} if dst.exists() else set()
        todo = [(p, q) for p in pairs for q in Q if (p["pair_id"], q) not in done]
        if not todo:
            continue
        j = SrcJudge(name)
        print(f"[{name}] A ids {j.A} B ids {j.B}; {len(todo)} items; {a.n_frames} frames/video", flush=True)
        new = not dst.exists()
        with open(dst, "a", newline="") as f:
            w = None
            for p, q in todo:
                qt = Q[q].format(p=p["edit_prompt"])
                try:
                    vs, vo, vx = frames(p["source"]), frames(p["ours"]), frames(p["other"])
                    a1, m1 = j.pa3(vs, vo, vx, qt)   # A = ours
                    a2, m2 = j.pa3(vs, vx, vo, qt)   # A = other
                    o1, o2 = a1, 1 - a2
                    res = dict(p_ours_order1=o1, p_ours_order2=o2, p_ours=(o1 + o2) / 2, position_bias_A=(a1 + a2) / 2,
                               consistent=("ours" if o1 > .5 and o2 > .5 else "other" if o1 < .5 and o2 < .5 else "inconsistent"),
                               ab_mass=min(m1, m2), n_frames=a.n_frames, error="")
                except Exception as e:  # noqa: BLE001
                    res = dict(p_ours_order1="", p_ours_order2="", p_ours="", position_bias_A="", consistent="", ab_mass="",
                               n_frames=a.n_frames, error=repr(e)[:300])
                row = {**p, "question": q, "model": name, **res}
                if w is None:
                    w = csv.DictWriter(f, fieldnames=list(row))
                    if new:
                        w.writeheader()
                w.writerow(row); f.flush()
                print(name, p["pair_id"], q, res.get("p_ours"), res.get("consistent"), res["error"], flush=True)
                torch.cuda.empty_cache()
        del j
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
