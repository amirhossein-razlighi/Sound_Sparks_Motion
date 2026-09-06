#!/usr/bin/env python3
"""Generate H3's OWN sound of each target event (text-to-audio-video, no
references) and save it as a 32 kHz mono wav for the pinned-audio test:
    /scratch/amirrz/H3_exp/inputs/sweep/<slug>_h3sound.wav
The t2va video is kept too (<OUT>/<slug>__t2va.mp4 + _av.mp4) for inspection.
"""
import importlib.util
import logging
import os
import subprocess
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
log = logging.getLogger("t2va")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_bs = _load("baseline_sweep")
_pt = _load("pin_test")
CKPT, SW = _bs.CKPT, _bs.SW
STEPS, SEED, NUM_FRAMES, HEIGHT, WIDTH = _bs.STEPS, _bs.SEED, _bs.NUM_FRAMES, _bs.HEIGHT, _bs.WIDTH
OUT = os.environ.get("OUT_DIR", "/scratch/amirrz/H3_exp/outputs/t2va_sounds")
ONLY = [s for s in os.environ.get("ONLY", "").split(",") if s]


def main():
    os.makedirs(OUT, exist_ok=True)
    from diffusers import ModularPipeline
    from diffusers.modular_pipelines import ComponentsManager
    from diffusers.utils import export_to_video
    import soundfile as sf

    cm = ComponentsManager()
    cm.enable_auto_cpu_offload(device="cuda")
    pipe = None
    for wf in ("t2va", None):
        try:
            pipe = (ModularPipeline.from_pretrained(CKPT, workflow=wf, components_manager=cm) if wf
                    else ModularPipeline.from_pretrained(CKPT, components_manager=cm))
            log.info("pipeline loaded with workflow=%s", wf)
            break
        except Exception as e:
            log.warning("workflow=%s failed: %s", wf, e)
    assert pipe is not None
    pipe.load_components(torch_dtype=torch.bfloat16)
    # Only the Ref2VA transformer is on disk; the t2va denoiser looks for `transformer`, so alias it.
    if getattr(pipe, "transformer", None) is None:
        tr = getattr(pipe, "transformer_ref", None)
        if tr is None:
            from diffusers import MiniMaxH3Transformer3DModel
            tr = MiniMaxH3Transformer3DModel.from_pretrained(CKPT, subfolder="transformer_ref", torch_dtype=torch.bfloat16)
            log.info("loaded transformer_ref weights explicitly")
        pipe.update_components(transformer=tr)
        log.info("aliased transformer <- transformer_ref")
    assert getattr(pipe, "transformer", None) is not None, "no transformer available for t2va"

    import json as _json
    prompts = _pt.T2VA
    if os.environ.get("PROMPTS_JSON"):  # {name: prompt} -> inputs/sweep/<name>_h3sound.wav
        prompts = _json.load(open(os.environ["PROMPTS_JSON"]))
    for slug, prompt in prompts.items():
        if ONLY and slug not in ONLY:
            continue
        wav_out = f"{SW}/{slug}_h3sound.wav"
        if os.path.exists(wav_out):
            log.info("[skip] %s", slug)
            continue
        state = pipe(prompt=prompt, num_frames=NUM_FRAMES, height=HEIGHT, width=WIDTH, num_inference_steps=STEPS,
                     generator=torch.Generator("cuda").manual_seed(SEED), output_type="np")
        vids = state.values.get("videos")
        vid = np.asarray(vids[0] if isinstance(vids, (list, tuple)) else vids)
        if vid.ndim == 5:
            vid = vid[0]
        aud = state.values.get("audio")
        assert aud is not None, "no audio in t2va output"
        aud = aud[0] if isinstance(aud, (list, tuple)) else aud
        wav = np.asarray(torch.as_tensor(aud).float().cpu()) if torch.is_tensor(aud) else np.asarray(aud)
        wav = np.squeeze(wav)
        if wav.ndim == 2 and wav.shape[0] in (1, 2):
            wav = wav.T
        sr = int(state.values.get("sampling_rate", 32000) or 32000)
        path = os.path.join(OUT, f"{slug}__t2va.mp4")
        export_to_video([np.asarray(f) for f in vid], path, fps=24)
        sf.write(path[:-4] + ".wav", wav, sr)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-threads", "2", "-i", path, "-i", path[:-4] + ".wav",
                        "-c:v", "copy", "-c:a", "aac", "-shortest", path[:-4] + "_av.mp4"], check=False)
        mono = wav.mean(1) if wav.ndim == 2 else wav
        sf.write(wav_out, mono.astype(np.float32), sr)
        log.info("SOUND %s -> %s (rms=%.4f, %.1fs)", slug, wav_out, float(np.sqrt(np.mean(mono ** 2))), len(mono) / sr)
    log.info("T2VA SOUNDS DONE")


if __name__ == "__main__":
    main()
