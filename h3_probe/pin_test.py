#!/usr/bin/env python3
"""Mechanism test: does a PINNED audio latent spark motion in H3?

For scenarios where H3's text editing failed, render with identical seed /
steps / references / text and vary only the pinned generated-audio latent:

  arm                 text     pinned generated audio
  edit__nopin         edit     (none: H3 generates its own)      <- baseline (sweep)
  edit__pin_src       edit     source audio                      <- control
  edit__pin_ltx       edit     LTX-optimized audio (our method)
  neutral__pin_ltx    neutral  LTX-optimized audio               <- text never names the motion
  edit__pin_h3snd     edit     H3's own t2va sound of the event  (if t2va works)
  neutral__pin_h3snd  neutral  H3's own t2va sound of the event

Audio *reference* rows stay = source audio in every arm, so the pinned rows are
the only difference. Each render: silent mp4 + .wav (decoded soundtrack, i.e.
the pinned sound) + _av.mp4, and a Qwen motion-critic score.
"""
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
log = logging.getLogger("pintest")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_bs = _load("baseline_sweep")
_at = _load("audio_transfer_test")
import pin_audio  # noqa: E402

grammar_prompt, neutral_prompt = _bs.grammar_prompt, _at.neutral_prompt
CKPT, QWEN, SW = _bs.CKPT, _bs.QWEN, _bs.SW
STEPS, SEED, NUM_FRAMES, HEIGHT, WIDTH = _bs.STEPS, _bs.SEED, _bs.NUM_FRAMES, _bs.HEIGHT, _bs.WIDTH
OUT = os.environ.get("OUT_DIR", "/scratch/amirrz/H3_exp/outputs/pin_test")
ONLY = [s for s in os.environ.get("ONLY", "").split(",") if s]
ARMS = [s for s in os.environ.get("ARMS", "").replace(";", ",").split(",") if s]  # optional arm-name filter (";" or "," separated; sbatch --export splits on ",")

# slug -> free-text t2va prompt for H3's own sound of the event
T2VA = {
    "goldfish": "A goldfish leaps out of a small glass fish tank into the air with a loud splash of water, then drops back into the tank with a second splash. Close-up, indoor, quiet room.",
    "cat_yawns": "A cat sitting on an armchair opens its mouth wide in a long, audible yawn with a small squeak, then closes it. Quiet living room.",
    "boy_crouches": "A boy standing in a swimming pool crouches down and slaps the water surface with his hands, making a splash. Outdoor pool sounds.",
    "turtle_extends_neck": "A tortoise on a rock slowly stretches its neck out of its shell with a soft scraping sound, then holds it out. Quiet pond ambience.",
}
SCEN = {s[0]: s for s in _at.SCEN}
_RV = _bs.RV
SCEN.update({  # extra failing/weak scenarios from the sweep (slug, src, src wav, ltx wav, edit, scene)
    "man_shouts": ("man_shouts", f"{_RV}/transfers/man_shouts/retake_input_prepared.mp4", f"{SW}/man_shouts.wav",
                   f"{SW}/man_shouts_ltxopt.wav", "The man shouts loudly.", "a man facing the camera"),
})
T2VA_EXTRA = {
    "man_shouts": "A man in a grey t-shirt standing in a plain grey room faces the camera and shouts loudly, a long forceful yell, then stops. Indoor room acoustics.",
    "red_bird_opens_wings": "A red macaw perched on a mossy branch in a forest suddenly spreads and flaps its wings twice with a loud wing-beat and a short squawk, then folds them.",
}
T2VA.update(T2VA_EXTRA)


def main():
    os.makedirs(OUT, exist_ok=True)
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
    qwen.to("cpu")
    torch.cuda.empty_cache()
    windows = [("linspace", 0), ("contiguous", 0), ("contiguous", 50), ("contiguous", 100)]
    ov = {"motion": 1.0, "entities": 0.0, "overall": 0.0}
    n_aud = audio_latent_num_frames(NUM_FRAMES)
    log.info("num_audio_latents=%d audio_channels=%s", n_aud, pipe.audio_channels)
    pin_audio.install_pin()

    rp = os.path.join(OUT, "results.json")
    results = json.load(open(rp)) if os.path.exists(rp) else {}

    def score(vid, edit):
        ci, yes_id, no_id = build_qwen_rubric_inputs(processor=proc, edit_prompt=edit, num_frames=24, img_size=224,
                                                     device=dev, motion_question=None, gradient_rubric="motion")
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

    def unpack(state):
        vids = state.values.get("videos")
        vid = np.asarray(vids[0] if isinstance(vids, (list, tuple)) else vids)
        if vid.ndim == 5:
            vid = vid[0]
        auds = state.values.get("audio")
        wav = None
        if auds is not None:
            aud = auds[0] if isinstance(auds, (list, tuple)) else auds
            wav = np.asarray(torch.as_tensor(aud).float().cpu()) if torch.is_tensor(aud) else np.asarray(aud)
            wav = np.squeeze(wav)
            if wav.ndim == 2 and wav.shape[0] in (1, 2):
                wav = wav.T
        return vid, wav, int(state.values.get("sampling_rate", 32000) or 32000)

    def save(vid, wav, sr, path):
        export_to_video([np.asarray(f) for f in vid], path, fps=24)
        if wav is None:
            return
        stem = os.path.splitext(path)[0]
        sf.write(stem + ".wav", wav, sr)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-threads", "2", "-i", path, "-i", stem + ".wav",
                        "-c:v", "copy", "-c:a", "aac", "-shortest", stem + "_av.mp4"], check=False)

    def h3_sound(slug):
        """H3's own t2va sound of the event -> 32 kHz mono wav path (cached), or None."""
        p = f"{SW}/{slug}_h3sound.wav"
        if os.path.exists(p):
            return p
        try:
            pin_audio.set_target(None)
            state = pipe(prompt=T2VA[slug], num_frames=NUM_FRAMES, height=HEIGHT, width=WIDTH,
                         num_inference_steps=STEPS, generator=torch.Generator("cuda").manual_seed(SEED),
                         output_type="np")
            vid, wav, sr = unpack(state)
            if wav is None:
                return None
            save(vid, wav, sr, os.path.join(OUT, f"{slug}__t2va_sound_source.mp4"))
            mono = wav.mean(1) if wav.ndim == 2 else wav
            sf.write(p, mono.astype(np.float32), sr)
            return p
        except Exception as e:
            log.exception("t2va sound generation failed for %s: %s", slug, e)
            return None

    for slug in (ONLY or ["goldfish", "cat_yawns", "boy_crouches", "turtle_extends_neck"]):
        _, src, wav_src, wav_ltx, edit, scene = SCEN[slug]
        refs = [MiniMaxH3VideoReference.from_file(src), MiniMaxH3AudioReference.from_file(wav_src)]
        wav_h3 = h3_sound(slug) if slug in T2VA else None
        rows = {"src": pin_audio.target_rows(pipe, wav_src, n_aud, dev),
                "ltx": pin_audio.target_rows(pipe, wav_ltx, n_aud, dev)}
        if wav_h3:
            rows["h3snd"] = pin_audio.target_rows(pipe, wav_h3, n_aud, dev)
        arms = [("edit__nopin", grammar_prompt(edit, scene), None),
                ("edit__pin_src", grammar_prompt(edit, scene), "src"),
                ("edit__pin_ltx", grammar_prompt(edit, scene), "ltx"),
                ("neutral__pin_ltx", neutral_prompt(scene), "ltx")]
        refs_h3 = None
        if wav_h3:
            refs_h3 = [MiniMaxH3VideoReference.from_file(src), MiniMaxH3AudioReference.from_file(wav_h3)]
            arms += [("edit__pin_h3snd", grammar_prompt(edit, scene), "h3snd"),
                     ("neutral__pin_h3snd", neutral_prompt(scene), "h3snd"),
                     # reference audio = the event sound as well, so reference and pinned rows agree
                     ("neutral__pinref_h3snd", neutral_prompt(scene), "h3snd:ref")]
        for arm, prompt, pin in arms:
            key = f"{slug}__{arm}"
            if ARMS and arm not in ARMS:
                continue
            use_refs = refs_h3 if (pin and pin.endswith(":ref")) else refs
            pin = None if pin is None else pin.split(":")[0]
            path = os.path.join(OUT, key + ".mp4")
            if key in results and os.path.exists(path):
                log.info("[skip] %s", key)
                continue
            t0 = time.time()
            try:
                pin_audio.set_target(None if pin is None else rows[pin], seed=SEED)
                state = pipe(prompt=prompt, references=use_refs, num_frames=NUM_FRAMES, height=HEIGHT, width=WIDTH,
                             num_inference_steps=STEPS, generator=torch.Generator("cuda").manual_seed(SEED),
                             output_type="np")
                vid, wav, sr = unpack(state)
                save(vid, wav, sr, path)
                nll = score(vid, edit)
                results[key] = {"slug": slug, "arm": arm, "pin": pin, "edit": edit, "nll": nll, "yes": math.exp(-nll),
                                "sec": round(time.time() - t0)}
                log.info("RESULT %-34s yes=%.4f nll=%.3f (%.0fs)", key, math.exp(-nll), nll, time.time() - t0)
            except Exception as e:
                log.exception("FAILED %s: %s", key, e)
                results[key] = {"slug": slug, "arm": arm, "error": str(e)[:300]}
            finally:
                pin_audio.set_target(None)
            json.dump(results, open(rp, "w"), indent=2)

    log.info("=== summary (yes-prob) ===")
    for k, v in results.items():
        log.info("%-40s %s", k, f"{v['yes']:.4f}" if "yes" in v else "ERR")
    log.info("PIN TEST DONE")


if __name__ == "__main__":
    main()
