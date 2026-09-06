#!/usr/bin/env python3
"""Phase-1 smoke test: MiniMax-H3 Ref2VA with a source video as reference.

Validates: components load on this hardware (auto CPU offload), a ref2va
generation runs end-to-end with our source clip as the video reference, and
the output (video+audio) is written. Deliberately defensive about the
post-release API: dumps the returned state's keys before saving.

    /scratch/amirrz/H3_exp/venv/bin/python h3_probe/smoke_ref2va.py
"""
import logging
import os
import sys
import time

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
log = logging.getLogger("h3smoke")

CKPT = "/scratch/amirrz/H3_exp/ckpt"
OUT_DIR = "/scratch/amirrz/H3_exp/outputs/smoke"
SRC = os.environ.get("SRC", "/home/amirrz/my_codes/Sound_Sparks_Motion/input_videos/cook_spinach_main.mp4")
PROMPT = os.environ.get(
    "PROMPT",
    "Referencing <Video 1>: the same kitchen scene with the same man in the gray "
    "apron and the same white dog beside him. The man reaches out and gently pets "
    "the dog's head; the dog leans in happily. overall_soundscape: soft rustling "
    "of fur, calm indoor kitchen ambience.",
)
if os.environ.get("PROMPT_FILE"):
    PROMPT = open(os.environ["PROMPT_FILE"]).read().strip()   # commas break sbatch --export
STEPS = int(os.environ.get("STEPS", "40"))
NUM_FRAMES = int(os.environ.get("NUM_FRAMES", "124"))  # 5.2 s; H3 needs 17n+5 in [120, 360]
HEIGHT = int(os.environ.get("HEIGHT", "448"))          # multiple of 32; short edge
WIDTH = int(os.environ.get("WIDTH", "768"))
SEED = int(os.environ.get("SEED", "42"))
AUDIO_REF = os.environ.get("AUDIO_REF", "")   # optional audio reference wav
TAG = os.environ.get("TAG", "ref2va")         # output name tag


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    from diffusers import ModularPipeline
    from diffusers.modular_pipelines import ComponentsManager
    from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3AudioReference, MiniMaxH3VideoReference

    t0 = time.time()
    cm = ComponentsManager()
    cm.enable_auto_cpu_offload(device="cuda")
    pipe = ModularPipeline.from_pretrained(CKPT, workflow="ref2va", components_manager=cm)
    pipe.load_components(torch_dtype=torch.bfloat16)
    log.info("pipeline loaded in %.1fs", time.time() - t0)

    refs = [MiniMaxH3VideoReference.from_file(SRC)]
    if AUDIO_REF:
        refs.append(MiniMaxH3AudioReference.from_file(AUDIO_REF))
        log.info("audio reference: %s", AUDIO_REF)
    log.info("references: %d", len(refs))

    t0 = time.time()
    state = pipe(
        prompt=PROMPT,
        references=refs,
        num_frames=NUM_FRAMES,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=STEPS,
        generator=torch.Generator("cuda").manual_seed(SEED),
        output_type="np",
    )
    log.info("generation done in %.1fs", time.time() - t0)
    log.info("state type: %s", type(state))
    try:
        keys = list(state.values.keys()) if hasattr(state, "values") else list(vars(state).keys())
        log.info("state keys: %s", keys)
    except Exception:
        log.info("could not enumerate state keys")

    videos = getattr(state, "videos", None)
    if videos is None and hasattr(state, "values"):
        videos = state.values.get("videos")
    audios = getattr(state, "audios", None)
    if audios is None and hasattr(state, "values"):
        for k in ("audios", "audio", "waveforms"):
            if state.values.get(k) is not None:
                audios = state.values.get(k)
                log.info("audio found under key %r", k)
                break

    assert videos is not None, "no 'videos' in pipeline output"
    vid = np.asarray(videos[0] if isinstance(videos, (list, tuple)) else videos)
    if vid.ndim == 5:  # [B, F, H, W, 3] -> first batch item
        vid = vid[0]
    log.info("video array: %s dtype=%s", vid.shape, vid.dtype)

    # Save via diffusers' export util (expects a list of [H,W,3] frames).
    from diffusers.utils import export_to_video
    frames = [np.asarray(f) for f in vid]
    out_path = os.path.join(OUT_DIR, f"smoke_{TAG}.mp4")
    export_to_video(frames, out_path, fps=24)
    log.info("wrote %s", out_path)

    if audios is not None:
        try:
            import soundfile as sf
        except ImportError:
            sf = None
        aud = audios[0] if isinstance(audios, (list, tuple)) else audios
        aud = np.asarray(torch.as_tensor(aud).float().cpu()) if torch.is_tensor(aud) else np.asarray(aud)
        log.info("audio array: %s", aud.shape)
        np.save(os.path.join(OUT_DIR, f"smoke_{TAG}_audio.npy"), aud)
        sr = int(state.values.get("sampling_rate", 32000)) if hasattr(state, "values") else 32000
        if sf is not None:
            wav = np.squeeze(aud)
            if wav.ndim == 2 and wav.shape[0] in (1, 2):
                wav = wav.T
            sf.write(os.path.join(OUT_DIR, f"smoke_{TAG}_audio.wav"), wav, sr)
            log.info("wrote smoke_audio.wav (sr=%d)", sr)
            import subprocess
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-threads", "2",
                            "-i", out_path, "-i", os.path.join(OUT_DIR, f"smoke_{TAG}_audio.wav"),
                            "-c:v", "copy", "-c:a", "aac", "-shortest",
                            os.path.join(OUT_DIR, f"smoke_{TAG}_av.mp4")], check=False)

    log.info("SMOKE SUCCESS")


if __name__ == "__main__":
    main()
