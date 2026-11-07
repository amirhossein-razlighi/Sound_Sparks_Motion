"""Differentiable InternVL3-8B yes/no critic (second in-loop critic for the multi-critic proof of concept).

Same question text as the run's Qwen2.5-VL critic. Loss = -log P(yes | video, question) with P(yes) = softmax over
{logsumexp(yes-variant logits), logsumexp(no-variant logits)} at the answer position (the held-out scorer's rule,
heldout_score.py). Frames -> InternVL video input exactly as the HF processor does it (one 448x448 tile per frame,
bicubic resize, /255, ImageNet mean/std), but in torch so gradients flow back to the rendered frames. The token
sequence ("Frame1: <img><IMG_CONTEXT>*256</img> ...") is built once per frame count with the official processor on a
dummy video and cached.

Device: the critic lives on its own GPU (default cuda:1); frames are moved there with .to() (autograd-safe), the
returned loss is moved back to the frames' device.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_PATH = "/project/def-amahdavi/amirrz/HF/models/InternVL3-8B-hf"


def _variant_ids(tok, words):
    ids = set()
    for w in words:
        for form in (w, " " + w):
            t = tok.encode(form, add_special_tokens=False)
            if len(t) == 1:
                ids.add(t[0])
    return sorted(ids)


class InternVLCritic:
    def __init__(self, question: str, path: str = DEFAULT_PATH, device: str = "cuda:1", max_frames: int = 16,
                 gradient_checkpointing: bool = True):
        from transformers import AutoProcessor, InternVLForConditionalGeneration
        self.device = torch.device(device)
        self.question, self.max_frames = question, int(max_frames)
        self.proc = AutoProcessor.from_pretrained(path)
        vp = self.proc.video_processor
        self.size = (int(vp.size["height"]), int(vp.size["width"]))
        assert int(vp.resample) == 3, "expected bicubic"
        self.mean = torch.tensor(vp.image_mean, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(vp.image_std, device=self.device).view(1, 3, 1, 1)
        self.model = InternVLForConditionalGeneration.from_pretrained(path, dtype=torch.bfloat16,
                                                                      device_map=str(self.device)).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.gc = False
        if gradient_checkpointing:
            try:
                self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                # HF only checkpoints in train mode; every dropout in this config is 0.0 (attention_dropout, dropout,
                # hidden_dropout_prob, projection_dropout), so train mode is numerically the same as eval here.
                self.model.train()
                self.gc = True
            except Exception:
                pass
        tok = self.proc.tokenizer
        self.yes = torch.tensor(_variant_ids(tok, ["yes", "Yes", "YES"]), device=self.device)
        self.no = torch.tensor(_variant_ids(tok, ["no", "No", "NO"]), device=self.device)
        self.yes_tokens = [tok.decode([int(i)]) for i in self.yes]
        self.no_tokens = [tok.decode([int(i)]) for i in self.no]
        self._ids: dict[int, dict] = {}

    def _inputs(self, n: int) -> dict:
        if n not in self._ids:
            msgs = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": self.question}]}]
            text = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            dummy = np.zeros((n, 64, 64, 3), dtype=np.uint8)
            o = self.proc(text=[text], videos=[dummy], return_tensors="pt")
            assert o["pixel_values"].shape[0] == n, o["pixel_values"].shape  # one tile per frame
            self._ids[n] = {"input_ids": o["input_ids"].to(self.device),
                            "attention_mask": o["attention_mask"].to(self.device)}
        return self._ids[n]

    def pick(self, idx: torch.Tensor) -> torch.Tensor:
        """Frame indices actually given to InternVL: the Qwen pass's indices, uniformly thinned to <= max_frames."""
        idx = idx.to("cpu").long()
        if len(idx) <= self.max_frames:
            return idx
        sub = torch.linspace(0, len(idx) - 1, self.max_frames).round().long()
        return idx[sub]

    def pixel_values(self, frames_chw: torch.Tensor) -> torch.Tensor:
        """frames [T,3,H,W] in [0,1] (any device, may require grad) -> [T,3,448,448] normalised bf16 on self.device."""
        x = frames_chw.to(self.device, dtype=torch.float32)
        x = F.interpolate(x, size=self.size, mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
        return ((x - self.mean) / self.std).to(torch.bfloat16)

    def nll(self, frames_chw: torch.Tensor) -> torch.Tensor:
        """-log P(yes) for the given frames (all of them; select frames before calling). Differentiable."""
        inp = self._inputs(frames_chw.shape[0])
        out = self.model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                         pixel_values=self.pixel_values(frames_chw))
        lp = torch.log_softmax(out.logits[0, -1].float(), -1)
        ly, ln = torch.logsumexp(lp[self.yes], 0), torch.logsumexp(lp[self.no], 0)
        return -(ly - torch.logaddexp(ly, ln))
