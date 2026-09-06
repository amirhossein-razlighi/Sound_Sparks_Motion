"""Attention-mass maps for MiniMax-H3 (single-stream self-attention over one packed sequence).

For every *generated video* token (query) we compute, at chosen denoising steps and over all heads
and layers, the softmax attention mass that lands on each key group of the packed sequence:

    audio_ref  - the audio-reference rows (H3's audio conditioning latent = what we optimize)
    audio_gen  - the generated audio rows
    text_txt   - the prompt's text tokens (Qwen3-VL conditioner rows tagged 1)
    text_vis   - the prompt's vision rows (the reference video as seen by the conditioner, tag 0)
    video_ref  - the reference-video rows (the source clip, noise-augmented)
    video_gen  - the generated video rows themselves

This is the H3 analogue of the LTX audio-to-video / text-to-video attention maps: the masses are
exact (full softmax over all keys, no approximation) and are computed under no_grad from the same
q/k the model uses (post-norm, post-rotary), so they never touch autograd. The capture is installed
by monkeypatching `MiniMaxH3AttnProcessor.__call__`; when STATE["on"] is False the patch is a no-op.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import torch

GROUPS = ("audio_ref", "audio_gen", "text_txt", "text_vis", "video_ref", "video_gen")
STATE = {"on": False, "group_id": None, "q_idx": None, "acc": None, "nlayers": 0, "installed": False,
         "chunk": 2048, "err": None}


def install(layout: dict, text_token_tags: torch.Tensor, ncv: int, nca: int) -> None:
    """layout: dict with text_indices / video_indices / audio_indices (sequence positions)."""
    ti, vi, ai = layout["text_indices"].cpu(), layout["video_indices"].cpu(), layout["audio_indices"].cpu()
    tt = text_token_tags.cpu()
    S = int(ti.numel() + vi.numel() + ai.numel())
    gid = torch.full((S,), -1, dtype=torch.long)
    gid[ai[:nca]] = 0
    gid[ai[nca:]] = 1
    gid[ti[tt == 1]] = 2
    gid[ti[tt == 0]] = 3
    gid[vi[:ncv]] = 4
    gid[vi[ncv:]] = 5
    assert int((gid < 0).sum()) == 0, "every sequence position must belong to a group"
    STATE["group_id"] = gid
    STATE["q_idx"] = vi[ncv:].clone()
    STATE["counts"] = [int((gid == g).sum()) for g in range(len(GROUPS))]
    if STATE["installed"]:
        return
    from diffusers.models.transformers import transformer_minimax_h3 as tm
    orig = tm.MiniMaxH3AttnProcessor.__call__

    def patched(self, attn, hidden_states, rotary_emb=None, attention_mask=None):
        if STATE["on"] and STATE["err"] is None:
            try:
                with torch.no_grad():
                    _capture(attn, hidden_states.detach(), rotary_emb, tm)
            except Exception as e:  # never break the render
                STATE["err"] = repr(e)
        return orig(self, attn, hidden_states, rotary_emb, attention_mask)

    tm.MiniMaxH3AttnProcessor.__call__ = patched
    STATE["installed"] = True


def begin() -> None:
    STATE["acc"] = None
    STATE["nlayers"] = 0
    STATE["err"] = None
    STATE["on"] = True


def end() -> dict | None:
    STATE["on"] = False
    if STATE["acc"] is None or STATE["nlayers"] == 0:
        return None
    return {"mass": (STATE["acc"] / STATE["nlayers"]).detach().cpu(), "nlayers": STATE["nlayers"], "counts": STATE["counts"],
            "err": STATE["err"]}


@torch.no_grad()
def _capture(attn, h, rotary_emb, tm) -> None:
    """Replicates the processor's q/k path, then accumulates per-query attention mass per key group."""
    h = h.detach()
    if h.shape[1] != STATE["group_id"].numel():
        return  # not the packed sequence (e.g. the token refiner attends over the text rows only)
    if getattr(attn, "fused_projections", False):
        q, k, _ = attn.to_qkv(h).chunk(3, dim=-1)
    else:
        q, k = attn.to_q(h), attn.to_k(h)
    q = attn.norm_q(q.unflatten(-1, (attn.heads, -1)))
    k = attn.norm_k(k.unflatten(-1, (attn.heads, -1)))
    if rotary_emb is not None:
        q = tm._apply_rotary_emb(q, *rotary_emb)
        k = tm._apply_rotary_emb(k, *rotary_emb)
    dev = h.device
    gid = STATE["group_id"].to(dev)
    q_idx = STATE["q_idx"].to(dev)
    qg = q[0, q_idx]                       # [Nq, H, D]
    kall = k[0]                            # [S, H, D]
    H, D = qg.shape[1], qg.shape[2]
    scale = D ** -0.5
    acc = torch.zeros(qg.shape[0], len(GROUPS), device=dev, dtype=torch.float32)
    ch = STATE["chunk"]
    for hh in range(H):
        kh = kall[:, hh, :]                # [S, D]
        for s in range(0, qg.shape[0], ch):
            qh = qg[s:s + ch, hh, :]       # [nq, D]
            logits = (qh @ kh.transpose(0, 1)).float() * scale     # [nq, S]
            p = torch.softmax(logits, dim=-1)
            acc[s:s + ch].index_add_(1, gid, p)
    acc /= H
    if STATE["acc"] is None:
        STATE["acc"] = acc
    else:
        STATE["acc"] += acc.to(STATE["acc"].device)
    STATE["nlayers"] += 1


# ----------------------------------------------------------------------------- rendering
_PLASMA = np.array([[0.050383, 0.029803, 0.527975], [0.461214, 0.097688, 0.585136], [0.798216, 0.280197, 0.469538],
                    [0.973381, 0.585016, 0.255155], [0.940015, 0.975158, 0.131326]])


def _cmap(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0, 1)
    t = x * (len(_PLASMA) - 1)
    i = np.clip(np.floor(t).astype(int), 0, len(_PLASMA) - 2)
    f = (t - i)[..., None]
    return _PLASMA[i] * (1 - f) + _PLASMA[i + 1] * f


def maps_from_mass(mass: torch.Tensor, nlf: int, lh: int, lw: int, patch) -> np.ndarray:
    """[N_gen_rows, G] -> [G, nlf//pt, lh//ph, lw//pw] (row order is frame-major, then y, then x)."""
    pt, ph, pw = patch
    g = mass.shape[1]
    return mass.numpy().T.reshape(g, nlf // pt, lh // ph, lw // pw)


def motion_map(frames: torch.Tensor, n_lat: int, gh: int, gw: int) -> np.ndarray:
    """|frame_t - frame_{t-1}| (mean over RGB) pooled to the latent grid: [n_lat, gh, gw]."""
    f = frames.detach().float().cpu()                        # [F,3,H,W]
    d = (f[1:] - f[:-1]).abs().mean(1)                       # [F-1,H,W]
    Fm = d.shape[0]
    out = np.zeros((n_lat, gh, gw), dtype=np.float32)
    pooled = torch.nn.functional.adaptive_avg_pool2d(d[None], (gh, gw))[0].numpy()   # [F-1,gh,gw]
    for lf in range(n_lat):
        a = int(round(lf / max(1, n_lat - 1) * (Fm - 1)))
        lo, hi = max(0, a - 2), min(Fm, a + 3)
        out[lf] = pooled[lo:hi].mean(0)
    return out


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    a, b = a - a.mean(), b - b.mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / den) if den > 0 else 0.0


def render(result: dict, frames: torch.Tensor, nlf: int, lh: int, lw: int, patch, out_png: str, tag: str = "",
           n_cols: int = 6, groups_to_draw=("audio_ref", "text_txt", "text_vis", "video_ref")) -> dict:
    """Writes <out_png> (rows: frame, motion, then one heat-overlay row per group) and returns summary stats."""
    from PIL import Image, ImageDraw
    mass = result["mass"]
    maps = maps_from_mass(mass, nlf, lh, lw, patch)          # [G, T, gh, gw]
    G, T, gh, gw = maps.shape
    fr = frames.detach().float().cpu()                       # [F,3,H,W] in [0,1]
    Fn, _, Hh, Ww = fr.shape
    mot = motion_map(fr, T, gh, gw)
    stats = {"tag": tag, "nlayers": result["nlayers"], "counts": dict(zip(GROUPS, result["counts"])),
             "mean_mass": {g: float(maps[i].mean()) for i, g in enumerate(GROUPS)},
             "corr_with_motion": {g: _corr(maps[i], mot) for i, g in enumerate(GROUPS)},
             "err": result.get("err")}
    # per-group normalization across the whole clip (so frames are comparable within a row and across iterations)
    cols = [int(round(c)) for c in np.linspace(0, Fn - 1, n_cols)]
    tw, th = 256, int(round(256 * Hh / Ww))
    rows_img = []
    def frame_img(i):
        return Image.fromarray((fr[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)).resize((tw, th), Image.BILINEAR)
    rows_img.append([frame_img(i) for i in cols])
    def heat_row(m3, cols_lat, norm_max):
        out = []
        for i, lf in zip(cols, cols_lat):
            hm = m3[lf] / max(norm_max, 1e-8)
            hm_img = Image.fromarray((_cmap(hm) * 255).astype(np.uint8)).resize((tw, th), Image.BILINEAR)
            base = frame_img(i).convert("RGB")
            out.append(Image.blend(base, hm_img, 0.6))
        return out
    cols_lat = [int(round(i / max(1, Fn - 1) * (T - 1))) for i in cols]
    rows_img.append(heat_row(mot, cols_lat, float(np.percentile(mot, 99))))
    labels = ["frame", "motion |dI|"]
    for g in groups_to_draw:
        gi = GROUPS.index(g)
        rows_img.append(heat_row(maps[gi], cols_lat, float(np.percentile(maps[gi], 99))))
        labels.append(f"{g}  mean={stats['mean_mass'][g]:.3f}  corr(motion)={stats['corr_with_motion'][g]:+.2f}")
    W = tw * n_cols + 150
    Hout = th * len(rows_img)
    canvas = Image.new("RGB", (W, Hout), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    for r, (row, lab) in enumerate(zip(rows_img, labels)):
        for c, im in enumerate(row):
            canvas.paste(im, (150 + c * tw, r * th))
        for k, line in enumerate([lab[:24], lab[24:48], lab[48:72]]):
            if line:
                draw.text((4, r * th + 4 + 12 * k), line, fill=(230, 230, 230))
    draw.text((4, Hout - 14), f"{tag} cols=frames {cols}", fill=(200, 200, 200))
    canvas.save(out_png)
    return stats


def selftest() -> None:
    """CPU shape/consistency test with a tiny fake attention module (no model needed)."""
    import types
    torch.manual_seed(0)
    S, Ht, D = 60, 2, 8
    ti, ai, vi = torch.arange(0, 10), torch.arange(10, 16), torch.arange(16, 60)   # 10 text, 6 audio, 44 video
    tt = torch.tensor([0] * 6 + [1] * 4)
    layout = {"text_indices": ti, "audio_indices": ai, "video_indices": vi}
    ncv, nca = 20, 3
    install(layout, tt, ncv, nca)
    STATE["chunk"] = 5
    lin = lambda: torch.nn.Linear(Ht * D, Ht * D, bias=False)
    attn = types.SimpleNamespace(fused_projections=False, to_q=lin(), to_k=lin(), heads=Ht,
                                 norm_q=torch.nn.Identity(), norm_k=torch.nn.Identity())
    from diffusers.models.transformers import transformer_minimax_h3 as tm
    begin()
    h = torch.randn(1, S, Ht * D)
    _capture(attn, h, None, tm)
    _capture(attn, h, None, tm)
    _capture(attn, torch.randn(1, 10, Ht * D), None, tm)   # refiner-like short sequence: must be ignored
    res = end()
    m = res["mass"]
    assert m.shape == (44 - ncv, 6), m.shape
    assert torch.allclose(m.sum(1), torch.ones(m.shape[0]), atol=1e-4), m.sum(1)
    assert res["nlayers"] == 2 and res["counts"] == [3, 3, 4, 6, 20, 24], res["counts"]
    # rendering with a fake geometry: 24 gen rows = nlf 2 x (4x3)
    frames = torch.rand(9, 3, 32, 48)
    st = render(res, frames, nlf=2, lh=8, lw=6, patch=(1, 2, 2), out_png="/tmp/h3_attn_selftest.png", tag="selftest", n_cols=3)
    assert set(st["mean_mass"]) == set(GROUPS)
    print("h3_attn selftest OK:", {k: round(v, 3) for k, v in st["mean_mass"].items()})


if __name__ == "__main__":
    selftest()
