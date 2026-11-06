#!/usr/bin/env python3
"""P2 - score videos with VLM verifiers (inference only, no gradients).

Same protocol for every model / method / video:
  * 16 frames, uniform linspace over the whole clip (gif or mp4)
  * frames resized so the long side is 512 px (aspect kept)
  * question: 'Does this video clearly show the action or state change described by the
    edit prompt: "<p>"? Answer yes or no.'   (chat template, generation prompt added)
  * yes_prob = softmax over {logsumexp(yes-variant logits), logsumexp(no-variant logits)}
    at the first answer position; we also log the mass that yes/no variants get among
    all tokens (format sanity) and the argmax token.

Models: qwen25 (the in-loop critic family, but THIS protocol, not the loop's),
        qwen3vl (Qwen3-VL-8B-Instruct, held-out), internvl3 (InternVL3-8B-hf, held-out).

Input: manifest CSV with columns  id,group,scenario,method,video,edit_prompt
Output: <out>/<model>.csv  (one row per manifest row), resumable.
"""
from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
import torch

import os
_HF = os.environ.get("HELDOUT_MODEL_ROOT", "/project/def-amahdavi/amirrz/HF/models")
MODELS = {
    "qwen25": f"{_HF}/Qwen2.5-VL-7B-Instruct",
    "qwen3vl": f"{_HF}/Qwen3-VL-8B-Instruct",
    "internvl3": f"{_HF}/InternVL3-8B-hf",
}
QUESTION = ('Does this video clearly show the action or state change described by the edit prompt: '
            '"{p}"? Answer yes or no.')
N_FRAMES = 16
LONG_SIDE = 512


def read_frames(path: str) -> tuple[list[np.ndarray], float, int, list[int]]:
    """Return 16 uniformly sampled RGB uint8 frames, source fps, total frames, indices."""
    p = Path(path)
    if p.suffix.lower() == ".gif":
        import imageio.v3 as iio
        arr = iio.imread(str(p), index=None)
        frames = [np.asarray(f)[..., :3] for f in arr]
        try:
            meta = iio.immeta(str(p))
            fps = 1000.0 / float(meta.get("duration", 40))
        except Exception:
            fps = 25.0
    else:
        import av
        c = av.open(str(p))
        vs = c.streams.video[0]
        fps = float(vs.average_rate) if vs.average_rate else 25.0
        frames = [f.to_ndarray(format="rgb24") for f in c.decode(vs)]
        c.close()
    T = len(frames)
    idx = np.linspace(0, T - 1, N_FRAMES).round().astype(int).tolist()
    from PIL import Image
    out = []
    for i in idx:
        im = Image.fromarray(frames[i])
        w, h = im.size
        s = LONG_SIDE / max(w, h)
        nw, nh = max(28, int(round(w * s / 28)) * 28), max(28, int(round(h * s / 28)) * 28)
        out.append(np.asarray(im.resize((nw, nh), Image.BICUBIC)))
    return out, fps, T, idx


def _variant_ids(tok, words):
    ids = set()
    for w in words:
        for form in (w, " " + w):
            t = tok.encode(form, add_special_tokens=False)
            if len(t) == 1:
                ids.add(t[0])
    return sorted(ids)


class Scorer:
    def __init__(self, name: str, device="cuda"):
        from transformers import AutoProcessor
        self.name, self.device = name, device
        path = MODELS[name]
        if name == "qwen25":
            from transformers import Qwen2_5_VLForConditionalGeneration as M
            self.proc = AutoProcessor.from_pretrained(path, min_pixels=128 * 28 * 28, max_pixels=LONG_SIDE * LONG_SIDE)
        elif name == "qwen3vl":
            from transformers import Qwen3VLForConditionalGeneration as M
            self.proc = AutoProcessor.from_pretrained(path)
        elif name == "internvl3":
            from transformers import InternVLForConditionalGeneration as M
            self.proc = AutoProcessor.from_pretrained(path)
        else:
            raise ValueError(name)
        self.model = M.from_pretrained(path, dtype=torch.bfloat16, device_map=device).eval()
        tok = self.proc.tokenizer
        self.yes = _variant_ids(tok, ["yes", "Yes", "YES"])
        self.no = _variant_ids(tok, ["no", "No", "NO"])
        self.tok = tok

    def _inputs(self, frames, fps, T, idx, prompt):
        msgs = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": QUESTION.format(p=prompt)}]}]
        text = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        video = np.stack(frames)  # [T,H,W,C] uint8
        if self.name == "qwen25":
            eff_fps = N_FRAMES / (T / fps)
            return self.proc(text=[text], videos=[video], fps=[eff_fps], return_tensors="pt")
        if self.name == "qwen3vl":
            from transformers.video_utils import VideoMetadata
            md = VideoMetadata(total_num_frames=T, fps=fps, frames_indices=idx, duration=T / fps)
            return self.proc(text=[text], videos=[video], video_metadata=[md], do_sample_frames=False,
                             do_resize=False, return_tensors="pt")
        return self.proc(text=[text], videos=[video], return_tensors="pt")

    @torch.no_grad()
    def score(self, video: str, prompt: str) -> dict:
        frames, fps, T, idx = read_frames(video)
        inp = self._inputs(frames, fps, T, idx, prompt).to(self.device)
        for k, v in inp.items():
            if torch.is_tensor(v) and v.is_floating_point():
                inp[k] = v.to(torch.bfloat16)
        logits = self.model(**inp).logits[0, -1].float()
        lp = torch.log_softmax(logits, -1)
        ly, ln = torch.logsumexp(lp[self.yes], 0), torch.logsumexp(lp[self.no], 0)
        yes_prob = float(torch.softmax(torch.stack([ly, ln]), 0)[0])
        return {
            "yes_prob": yes_prob,
            "yesno_mass": float(math.exp(torch.logaddexp(ly, ln).item())),
            "argmax_token": self.tok.decode([int(logits.argmax())]),
            "src_frames": T,
            "src_fps": round(fps, 3),
            "n_input_tokens": int(inp["input_ids"].shape[1]),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default="qwen25,qwen3vl,internvl3")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.manifest)))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in a.models.split(","):
        dst = out / f"{name}.csv"
        done = {r["id"] for r in csv.DictReader(open(dst))} if dst.exists() else set()
        todo = [r for r in rows if r["id"] not in done]
        if not todo:
            print(f"[{name}] all {len(rows)} done"); continue
        t0 = time.time()
        sc = Scorer(name)
        print(f"[{name}] loaded in {time.time()-t0:.0f}s; yes ids={sc.yes} no ids={sc.no}; {len(todo)} videos", flush=True)
        new = not dst.exists()
        with open(dst, "a", newline="") as f:
            w = None
            for r in todo:
                try:
                    res = sc.score(r["video"], r["edit_prompt"])
                    res["error"] = ""
                except Exception as e:  # keep going; record the failure
                    res = {"yes_prob": "", "yesno_mass": "", "argmax_token": "", "src_frames": "", "src_fps": "",
                           "n_input_tokens": "", "error": repr(e)[:300]}
                row = {**r, **res}
                if w is None:
                    w = csv.DictWriter(f, fieldnames=list(row))
                    if new:
                        w.writeheader()
                w.writerow(row); f.flush()
                print(f"[{name}] {r['id']}: {res['yes_prob']} {res['error']}", flush=True)
        del sc
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
