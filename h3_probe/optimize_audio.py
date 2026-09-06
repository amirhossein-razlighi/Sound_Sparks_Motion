#!/usr/bin/env python3
"""SPSA optimization of the H3 audio-REFERENCE latent with our Qwen critic,
inside the grammar-aligned [video editing] regime where the text edit is
suppressed by reconstruction pressure.

The paper's claim, ported: when text loses, can the optimized audio
conditioning tip the motion? The audio reference's normalized latent rows
(~[T,32], unit scale) are the optimization variable, injected by monkeypatching
`pipe.audio_vae.encode` to return a fixed posterior. Everything else (video
reference, structured prompt, seed/noise) is held fixed -> common random
numbers, so score differences isolate the audio effect.

Zeroth-order (SPSA, Rademacher, paired +/-c probes; 2 renders/iter), because
33B full-attention backprop is not happening on this hardware. Scoring = our
UNCHANGED motion_opt.qwen_loss, multi-window fp32-dithered mean nll (bf16
scores quantize at ~0.12 nats - learned on the Ovi probe).

    /scratch/amirrz/H3_exp/venv/bin/python h3_probe/optimize_audio.py
"""
import csv
import logging
import math
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(_HERE + "/../editing/src"))

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
log = logging.getLogger("h3opt")

CKPT = "/scratch/amirrz/H3_exp/ckpt"
OUT_DIR = os.environ.get("OUT_DIR", "/scratch/amirrz/H3_exp/outputs/opt_mpd")
SRC = os.environ.get("SRC", "/home/amirrz/my_codes/Sound_Sparks_Motion/input_videos/cook_spinach_main.mp4")
AUDIO_WAV = os.environ.get("AUDIO_WAV", "/scratch/amirrz/H3_exp/inputs/mpd_source.wav")
PROMPT_FILE = os.environ.get("PROMPT_FILE", "/scratch/amirrz/H3_exp/inputs/edit_prompt_mpd_audio.txt")
EDIT_PROMPT = os.environ.get("EDIT_PROMPT", "The man pets the dog.")
QWEN = "/project/def-amahdavi/amirrz/HF/models/Qwen2.5-VL-7B-Instruct"

STEPS_OPT = int(os.environ.get("STEPS_OPT", "16"))
STEPS_FINAL = int(os.environ.get("STEPS_FINAL", "40"))
ITERS = int(os.environ.get("ITERS", "10"))
SPSA_C = float(os.environ.get("SPSA_C", "0.15"))
LR = float(os.environ.get("LR", "0.05"))
REG_W = float(os.environ.get("REG_W", "0.01"))
EARLY = int(os.environ.get("EARLY", "6"))
SEED = 42
NUM_FRAMES, HEIGHT, WIDTH = 124, 448, 768


class _FixedPosterior:
    def __init__(self, latent):
        self._l = latent

    def mode(self):
        return self._l


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(SEED)

    from diffusers import ModularPipeline
    from diffusers.modular_pipelines import ComponentsManager
    from diffusers.modular_pipelines.minimax_h3 import (
        MiniMaxH3AudioReference,
        MiniMaxH3VideoReference,
    )

    prompt = open(PROMPT_FILE).read().strip()
    import json as _json
    _json.dump({"src": SRC, "audio_wav": AUDIO_WAV, "prompt_file": PROMPT_FILE,
                "edit_prompt": EDIT_PROMPT, "steps_opt": STEPS_OPT, "iters": ITERS,
                "spsa_c": SPSA_C, "lr": LR}, open(os.path.join(OUT_DIR, "run_config.json"), "w"), indent=2)

    cm = ComponentsManager()
    cm.enable_auto_cpu_offload(device="cuda")
    pipe = ModularPipeline.from_pretrained(CKPT, workflow="ref2va", components_manager=cm)
    pipe.load_components(torch_dtype=torch.bfloat16)
    log.info("H3 pipeline loaded")

    refs = [MiniMaxH3VideoReference.from_file(SRC), MiniMaxH3AudioReference.from_file(AUDIO_WAV)]

    # ---- Source audio latent (the block's own math, once) ----
    audio_vae = pipe.audio_vae
    a_mean = torch.tensor(audio_vae.config.latents_mean).view(1, 1, -1)
    a_std = torch.tensor(audio_vae.config.latents_std).view(1, 1, -1)
    aref = refs[1]
    wav = torch.as_tensor(aref.audio)
    if wav.ndim == 1:
        wav = wav[None]
    with torch.no_grad():
        post = audio_vae.encode(wav.to("cuda")[:, None].to(next(audio_vae.parameters()).dtype),
                                return_dict=False)[0]
        lat = post.mode().float().cpu().transpose(1, 2)          # [B, T, C]
    z_src = ((lat - a_mean) / a_std).clone()                     # normalized, [B, T, C]
    log.info("audio latent (normalized): %s  (%d dims)", tuple(z_src.shape), z_src.numel())

    orig_encode = audio_vae.encode
    inject = {"z": None}

    def patched_encode(x, return_dict=False):
        if inject["z"] is None:
            return orig_encode(x, return_dict=return_dict)
        denorm = (inject["z"] * a_std + a_mean).transpose(1, 2).to(x.device, dtype=torch.float32)
        return (_FixedPosterior(denorm),)

    audio_vae.encode = patched_encode

    # ---- Qwen critic (ours, unchanged) ----
    from motion_opt.qwen_loss import build_qwen_model, build_qwen_rubric_inputs, compute_qwen_video_loss
    qwen_model, qwen_processor = build_qwen_model(QWEN, device=torch.device("cuda"), gradient_checkpointing=False)
    cached_inputs, yes_id, no_id = build_qwen_rubric_inputs(
        processor=qwen_processor, edit_prompt=EDIT_PROMPT, num_frames=24, img_size=224,
        device=torch.device("cuda"), motion_question=None, gradient_rubric="motion")
    qwen_model.to("cpu")
    torch.cuda.empty_cache()
    _WINDOWS = [("linspace", 0), ("contiguous", 0), ("contiguous", 50), ("contiguous", 100)]
    overrides = {"motion": 1.0, "entities": 0.0, "overall": 0.0}

    def score_frames(video_np) -> float:
        """video_np [F,H,W,3] float 0..1 -> dithered mean nll (fp32)."""
        fr = torch.from_numpy(np.ascontiguousarray(video_np)).permute(0, 3, 1, 2).float().clamp(0, 1).cuda()
        qwen_model.to("cuda")
        acc = 0.0
        with torch.no_grad():
            for mode, start in _WINDOWS:
                loss, _ = compute_qwen_video_loss(
                    frames_chw=fr, qwen_model=qwen_model, cached_inputs=cached_inputs,
                    yes_token_id=yes_id, no_token_id=no_id, max_frames=24, img_size=224,
                    backward=False, return_details=True, sample_mode=mode,
                    contiguous_start_frame=start, rubric_weight_overrides=overrides)
                acc += float(loss)
        qwen_model.to("cpu")
        torch.cuda.empty_cache()
        return acc / len(_WINDOWS)

    def render(z, steps) -> np.ndarray:
        inject["z"] = z
        t0 = time.time()
        state = pipe(prompt=prompt, references=refs, num_frames=NUM_FRAMES,
                     height=HEIGHT, width=WIDTH, num_inference_steps=steps,
                     generator=torch.Generator("cuda").manual_seed(SEED), output_type="np")
        vids = state.values.get("videos")
        vid = np.asarray(vids[0] if isinstance(vids, (list, tuple)) else vids)
        if vid.ndim == 5:
            vid = vid[0]
        auds = (state.values.get("audio") if state.values.get("audio") is not None else state.values.get("audios"))
        last_audio["a"] = None if auds is None else (auds[0] if isinstance(auds, (list, tuple)) else auds)
        last_audio["sr"] = int(state.values.get("sampling_rate", 32000) or 32000)
        log.info("  render(%d steps) %.1fs", steps, time.time() - t0)
        return vid

    last_audio = {"a": None, "sr": 32000}

    def save_vid(vid, name):
        """Silent mp4 (what the critic/eval scripts read) + generated audio as
        <stem>.wav and a muxed <stem>_av.mp4 for listening."""
        from diffusers.utils import export_to_video
        path = os.path.join(OUT_DIR, name)
        export_to_video([np.asarray(f) for f in vid], path, fps=24)
        aud = last_audio["a"]
        if aud is None:
            return
        try:
            import subprocess
            import soundfile as sf
            wav = np.asarray(torch.as_tensor(aud).float().cpu()) if torch.is_tensor(aud) else np.asarray(aud)
            wav = np.squeeze(wav)
            if wav.ndim == 2 and wav.shape[0] in (1, 2):
                wav = wav.T
            stem = os.path.splitext(path)[0]
            sf.write(stem + ".wav", wav, last_audio["sr"])
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-threads", "2", "-i", path, "-i", stem + ".wav",
                            "-c:v", "copy", "-c:a", "aac", "-shortest", stem + "_av.mp4"], check=False)
        except Exception as e:  # audio is a convenience, never fail the run
            log.warning("audio save failed for %s: %s", name, e)

    # ---- Baseline (source audio latent, opt-steps) ----
    base_vid = render(z_src, STEPS_OPT)
    base_nll = score_frames(base_vid)
    log.info("BASELINE nll=%.4f yes=%.5f", base_nll, math.exp(-base_nll))
    save_vid(base_vid, "baseline_optsteps.mp4")

    # ---- SPSA ----
    z = z_src.clone()
    m = torch.zeros_like(z)  # Adam state
    v = torch.zeros_like(z)
    best = {"nll": base_nll, "z": z_src.clone(), "iter": 0}
    csvf = open(os.path.join(OUT_DIR, "opt_log.csv"), "w", newline="")
    w = csv.writer(csvf)
    w.writerow(["iter", "nll_p", "nll_m", "best_nll", "grad_scale", "sec"])

    for it in range(1, ITERS + 1):
        t0 = time.time()
        gen = torch.Generator().manual_seed(SEED * 10007 + it)
        delta = (torch.randint(0, 2, z.shape, generator=gen).float() * 2 - 1)
        f_p = score_frames(render(z + SPSA_C * delta, STEPS_OPT)) + REG_W * float(((z + SPSA_C * delta - z_src) ** 2).mean())
        f_m = score_frames(render(z - SPSA_C * delta, STEPS_OPT)) + REG_W * float(((z - SPSA_C * delta - z_src) ** 2).mean())
        g = ((f_p - f_m) / (2 * SPSA_C)) * delta
        # Adam
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        mhat = m / (1 - 0.9 ** it)
        vhat = v / (1 - 0.999 ** it)
        z = z - LR * mhat / (vhat.sqrt() + 1e-8)
        cur = min(f_p, f_m)
        if cur < best["nll"]:
            best.update(nll=cur, z=(z + SPSA_C * delta if f_p < f_m else z - SPSA_C * delta).clone(), iter=it)
        dt = time.time() - t0
        w.writerow([it, f_p, f_m, best["nll"], float((f_p - f_m) / (2 * SPSA_C)), f"{dt:.0f}"])
        csvf.flush()
        log.info("iter %2d/%d  f+=%.4f f-=%.4f  best=%.4f@%d  (%.0fs)",
                 it, ITERS, f_p, f_m, best["nll"], best["iter"], dt)
        if it - best["iter"] >= EARLY:
            log.info("early stop")
            break
    csvf.close()

    # ---- Final: current z and best probe, at final steps ----
    fin_vid = render(z, STEPS_FINAL)
    fin_nll = score_frames(fin_vid)
    save_vid(fin_vid, "optimized_final.mp4")
    torch.save({"z": z, "z_best": best["z"], "z_src": z_src}, os.path.join(OUT_DIR, "latents.pt"))
    log.info("DONE. baseline nll=%.4f (yes=%.5f) -> final nll=%.4f (yes=%.5f), best-probe nll=%.4f",
             base_nll, math.exp(-base_nll), fin_nll, math.exp(-fin_nll), best["nll"])


if __name__ == "__main__":
    main()
