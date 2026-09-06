#!/usr/bin/env python3
"""OUR METHOD ON H3: critic-guided optimization of the PINNED audio latent.

Variable: the generated-audio latent that is pinned at every denoising step
(`pin_audio`). It is parameterized in a small, semantically meaningful
subspace so that a zeroth-order optimizer can move it:

    z(w) = z_src + sum_k w_k * (z_bank_k - z_src),   w in R^K, w0 = 0

where z_src is the source soundtrack's latent and z_bank_k are latents of K
candidate sounds H3 itself generated for this scene (the true event sound
plus distractors: other vocalisations, another animal, room tone). w = 0 is
exactly the baseline. The objective is our unchanged Qwen motion-critic nll
(4-window mean) + L2 anchor on w. SPSA (paired +/-c Rademacher probes, Adam),
common random numbers (same seed / eps / text / references for every render).

The critic must discover WHICH sound sparks the motion and HOW MUCH of it;
the decoded soundtrack of the result is the sound it found.

env: SLUG (cat_yawns), BANK (comma list of bank names; wavs at
inputs/sweep/<name>_h3sound.wav), OUT_DIR, ITERS=8, SPSA_C=0.35, LR=0.25,
REG_W=0.02, TEXT=edit|neutral, STEPS=16.
"""
import csv
import importlib.util
import json
import logging
import math
import os
import subprocess
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(_HERE + "/../editing/src"))
sys.path.insert(0, _HERE)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
log = logging.getLogger("optpin")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_bs = _load("baseline_sweep")
_at = _load("audio_transfer_test")
import pin_audio  # noqa: E402

CKPT, QWEN, SW = _bs.CKPT, _bs.QWEN, _bs.SW
SEED, NUM_FRAMES, HEIGHT, WIDTH = _bs.SEED, _bs.NUM_FRAMES, _bs.HEIGHT, _bs.WIDTH
STEPS = int(os.environ.get("STEPS", "16"))
SLUG = os.environ.get("SLUG", "cat_yawns")
BANK = [b for b in os.environ.get("BANK", "").replace(";", ",").split(",") if b]
OUT = os.environ.get("OUT_DIR", f"/scratch/amirrz/H3_exp/outputs/optpin_{SLUG}")
ITERS = int(os.environ.get("ITERS", "8"))
SPSA_C = float(os.environ.get("SPSA_C", "0.35"))
LR = float(os.environ.get("LR", "0.25"))
REG_W = float(os.environ.get("REG_W", "0.02"))
TEXT = os.environ.get("TEXT", "edit")
W_MAX = float(os.environ.get("W_MAX", "1.5"))
SCEN = {s[0]: s for s in _at.SCEN}


def main():
    os.makedirs(OUT, exist_ok=True)
    assert BANK, "BANK must list the candidate sound names"
    _, src, wav_src, _, edit, scene = SCEN[SLUG]
    prompt = _bs.grammar_prompt(edit, scene) if TEXT == "edit" else _at.neutral_prompt(scene)
    json.dump({"slug": SLUG, "bank": BANK, "text": TEXT, "steps": STEPS, "iters": ITERS, "spsa_c": SPSA_C, "lr": LR,
               "reg_w": REG_W, "w_max": W_MAX, "seed": SEED, "edit_prompt": edit},
              open(os.path.join(OUT, "run_config.json"), "w"), indent=2)

    from diffusers import ModularPipeline
    from diffusers.modular_pipelines import ComponentsManager
    from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3AudioReference, MiniMaxH3VideoReference
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import audio_latent_num_frames
    from diffusers.utils import export_to_video
    import soundfile as sf
    from motion_opt.qwen_loss import build_qwen_model, build_qwen_rubric_inputs, compute_qwen_video_loss

    cm = ComponentsManager()
    cm.enable_auto_cpu_offload(device="cuda")
    pipe = ModularPipeline.from_pretrained(CKPT, workflow="ref2va", components_manager=cm)
    pipe.load_components(torch_dtype=torch.bfloat16)
    dev = torch.device("cuda")
    qwen, proc = build_qwen_model(QWEN, device=dev, gradient_checkpointing=False)
    ci, yes_id, no_id = build_qwen_rubric_inputs(processor=proc, edit_prompt=edit, num_frames=24, img_size=224,
                                                 device=dev, motion_question=None, gradient_rubric="motion")
    qwen.to("cpu")
    torch.cuda.empty_cache()
    windows = [("linspace", 0), ("contiguous", 0), ("contiguous", 50), ("contiguous", 100)]
    ov = {"motion": 1.0, "entities": 0.0, "overall": 0.0}
    n_aud = audio_latent_num_frames(NUM_FRAMES)
    pin_audio.install_pin()
    refs = [MiniMaxH3VideoReference.from_file(src), MiniMaxH3AudioReference.from_file(wav_src)]

    z_src = pin_audio.target_rows(pipe, wav_src, n_aud, dev)
    dirs = []
    for b in BANK:
        zb = pin_audio.target_rows(pipe, f"{SW}/{b}_h3sound.wav", n_aud, dev)
        d = zb - z_src
        log.info("bank %-18s |d|=%.1f", b, float(d.norm()))
        dirs.append(d)
    D = torch.stack(dirs)  # [K, rows, C]
    K = len(BANK)

    def z_of(w):
        return z_src + torch.einsum("k,krc->rc", w.float(), D)

    def score(vid):
        fr = torch.from_numpy(np.ascontiguousarray(vid)).permute(0, 3, 1, 2).float().clamp(0, 1).to(dev)
        qwen.to(dev)
        acc = 0.0
        with torch.no_grad():
            for mode, start in windows:
                loss, _ = compute_qwen_video_loss(frames_chw=fr, qwen_model=qwen, cached_inputs=ci, yes_token_id=yes_id,
                                                  no_token_id=no_id, max_frames=24, img_size=224, backward=False,
                                                  return_details=True, sample_mode=mode, contiguous_start_frame=start,
                                                  rubric_weight_overrides=ov)
                acc += float(loss)
        qwen.to("cpu")
        torch.cuda.empty_cache()
        return acc / len(windows)

    def render(w, steps=STEPS):
        pin_audio.set_target(z_of(w), seed=SEED)
        t0 = time.time()
        try:
            state = pipe(prompt=prompt, references=refs, num_frames=NUM_FRAMES, height=HEIGHT, width=WIDTH,
                         num_inference_steps=steps, generator=torch.Generator("cuda").manual_seed(SEED),
                         output_type="np")
        finally:
            pin_audio.set_target(None)
        vids = state.values.get("videos")
        vid = np.asarray(vids[0] if isinstance(vids, (list, tuple)) else vids)
        if vid.ndim == 5:
            vid = vid[0]
        aud = state.values.get("audio")
        wav = None
        if aud is not None:
            aud = aud[0] if isinstance(aud, (list, tuple)) else aud
            wav = np.squeeze(np.asarray(torch.as_tensor(aud).float().cpu()))
            if wav.ndim == 2 and wav.shape[0] in (1, 2):
                wav = wav.T
        log.info("  render(%d steps) %.0fs", steps, time.time() - t0)
        return vid, wav

    def save(vid, wav, name):
        path = os.path.join(OUT, name)
        export_to_video([np.asarray(f) for f in vid], path, fps=24)
        if wav is not None:
            sf.write(path[:-4] + ".wav", wav, 32000)
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-threads", "2", "-i", path, "-i", path[:-4] + ".wav",
                            "-c:v", "copy", "-c:a", "aac", "-shortest", path[:-4] + "_av.mp4"], check=False)

    def objective(w):
        vid, wav = render(w)
        nll = score(vid)
        return nll + REG_W * float((w ** 2).sum()), nll, vid, wav

    # ---- baseline: w = 0 (pinned source sound) ----
    w = torch.zeros(K)
    f0, base_nll, vid, wav = objective(w)
    save(vid, wav, "baseline.mp4")
    log.info("BASELINE nll=%.4f yes=%.5f", base_nll, math.exp(-base_nll))
    best = {"f": f0, "nll": base_nll, "w": w.clone(), "iter": 0}
    m = torch.zeros(K)
    v = torch.zeros(K)
    g = torch.Generator().manual_seed(SEED + 1)
    csvf = open(os.path.join(OUT, "opt_log.csv"), "w", newline="")
    cw = csv.writer(csvf)
    cw.writerow(["iter", "f_plus", "f_minus", "nll_plus", "nll_minus", "best_f", "best_nll"] + [f"w_{b}" for b in BANK])
    for it in range(1, ITERS + 1):
        delta = (torch.randint(0, 2, (K,), generator=g).float() * 2 - 1)
        wp = (w + SPSA_C * delta).clamp(-0.2, W_MAX)
        wm = (w - SPSA_C * delta).clamp(-0.2, W_MAX)
        fp, np_, vidp, wavp = objective(wp)
        fm, nm_, vidm, wavm = objective(wm)
        for f_, n_, w_, vid_, wav_ in ((fp, np_, wp, vidp, wavp), (fm, nm_, wm, vidm, wavm)):
            if f_ < best["f"]:
                best = {"f": f_, "nll": n_, "w": w_.clone(), "iter": it}
                save(vid_, wav_, "best_probe.mp4")
        ghat = (fp - fm) / (2 * SPSA_C) * delta
        m = 0.9 * m + 0.1 * ghat
        v = 0.999 * v + 0.001 * ghat ** 2
        w = (w - LR * m / (v.sqrt() + 1e-8)).clamp(-0.2, W_MAX)
        cw.writerow([it, fp, fm, np_, nm_, best["f"], best["nll"]] + [float(x) for x in w])
        csvf.flush()
        log.info("iter %d/%d  f+=%.4f f-=%.4f  best=%.4f@%d  w=%s", it, ITERS, fp, fm, best["f"], best["iter"],
                 ", ".join(f"{b}={float(x):+.2f}" for b, x in zip(BANK, w)))
    # ---- final: best w ----
    fw = best["w"]
    vid, wav = render(fw)
    fin_nll = score(vid)
    save(vid, wav, "optimized_final.mp4")
    torch.save({"w_best": fw, "w_last": w, "bank": BANK, "z_src": z_src, "D": D}, os.path.join(OUT, "latents.pt"))
    log.info("DONE. baseline nll=%.4f (yes=%.5f) -> final nll=%.4f (yes=%.5f) with w=%s", base_nll,
             math.exp(-base_nll), fin_nll, math.exp(-fin_nll),
             ", ".join(f"{b}={float(x):+.2f}" for b, x in zip(BANK, fw)))


if __name__ == "__main__":
    main()
