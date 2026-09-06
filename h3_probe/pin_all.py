#!/usr/bin/env python3
"""Sound-sparks-motion on H3, all benchmark scenarios: edit text fixed, pinned
generated-audio latent = H3's own sound of the event.

Per scenario (h3_probe/pin_all_scenarios.json) two renders with identical
seed / steps / references / structured edit prompt:
    <slug>__edit__nopin       H3 generates its own soundtrack        (baseline)
    <slug>__edit__pin_h3snd   generated audio rows pinned to
                              inputs/sweep/<slug>_h3sound.wav        (ours)
Both saved as silent mp4 + .wav + _av.mp4 and scored by the Qwen motion critic.

env: ONLY (comma list of slugs), OUT_DIR, ARMS (";"-separated arm filter).
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
_REPO = os.path.abspath(_HERE + "/..")
sys.path.insert(0, os.path.join(_REPO, "editing/src"))
sys.path.insert(0, _HERE)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
log = logging.getLogger("pinall")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_bs = _load("baseline_sweep")
import pin_audio  # noqa: E402

grammar_prompt = _bs.grammar_prompt
CKPT, QWEN, SW = _bs.CKPT, _bs.QWEN, _bs.SW
STEPS, SEED, NUM_FRAMES, HEIGHT, WIDTH = _bs.STEPS, _bs.SEED, _bs.NUM_FRAMES, _bs.HEIGHT, _bs.WIDTH
OUT = os.environ.get("OUT_DIR", "/scratch/amirrz/H3_exp/outputs/pin_all")
ONLY = [s for s in os.environ.get("ONLY", "").replace(";", ",").split(",") if s]
ARMS = [s for s in os.environ.get("ARMS", "").replace(";", ",").split(",") if s]
SCEN = {k: v for k, v in json.load(open(os.path.join(_HERE, "pin_all_scenarios.json"))).items() if not k.startswith("_")}


def keep_audio_prompt(prompt: str) -> str:
    """Same grammar, but <Audio 2> is to be kept verbatim and the action synchronized to its events."""
    a = "<Audio 2>: reference - the target video's sound follows the content and timing of <Audio 2>; on-screen actions stay synchronized with it."
    b = ("<Audio 2>: fully_preserved - the target video's soundtrack is exactly <Audio 2>, unchanged; the on-screen "
         "action happens exactly when its sound happens in <Audio 2> and is synchronized to it.")
    assert a in prompt
    return prompt.replace(a, b).replace("overall_soundscape: The soundscape follows <Audio 2>, with the natural sounds of the described action.",
                                        "overall_soundscape: Exactly <Audio 2>; the described action is heard in it and the visuals match its timing.")


def resolve_src(p):
    p = p if os.path.isabs(p) else os.path.join(_REPO, p)
    if not os.path.exists(p) and p.endswith("_main.mp4"):
        alt = p[:-9] + ".mp4"
        if os.path.exists(alt):
            return alt
    return p


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
        aud = state.values.get("audio")
        wav = None
        if aud is not None:
            aud = aud[0] if isinstance(aud, (list, tuple)) else aud
            wav = np.squeeze(np.asarray(torch.as_tensor(aud).float().cpu()))
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

    for slug, sc in SCEN.items():
        if ONLY and slug not in ONLY:
            continue
        src = resolve_src(sc["src"])
        wav_h3 = f"{SW}/{slug}_h3sound.wav"
        if not os.path.exists(wav_h3):
            log.warning("no H3 sound for %s (%s) - skipping", slug, wav_h3)
            continue
        edit, scene = sc["edit"], sc["scene"]
        prompt = grammar_prompt(edit, scene)
        refs = [MiniMaxH3VideoReference.from_file(src), MiniMaxH3AudioReference.from_file(sc["wav"])]
        # "no tricks" arms: the event sound is simply the audio REFERENCE (plain H3 inference)
        refs_h3 = [MiniMaxH3VideoReference.from_file(src), MiniMaxH3AudioReference.from_file(wav_h3)]
        prompt_keep = keep_audio_prompt(prompt)
        rows_h3 = pin_audio.target_rows(pipe, wav_h3, n_aud, dev)
        for arm, prompt_a, refs_a, pin in (("edit__nopin", prompt, refs, None),
                                           ("edit__pin_h3snd", prompt, refs, rows_h3),
                                           ("edit__ref_h3snd", prompt, refs_h3, None),
                                           ("edit__refkeep_h3snd", prompt_keep, refs_h3, None)):
            if ARMS and arm not in ARMS:
                continue
            key = f"{slug}__{arm}"
            path = os.path.join(OUT, key + ".mp4")
            if key in results and os.path.exists(path):
                log.info("[skip] %s", key)
                continue
            t0 = time.time()
            try:
                pin_audio.set_target(pin, seed=SEED)
                state = pipe(prompt=prompt_a, references=refs_a, num_frames=NUM_FRAMES, height=HEIGHT, width=WIDTH,
                             num_inference_steps=STEPS, generator=torch.Generator("cuda").manual_seed(SEED),
                             output_type="np")
                vid, wav, sr = unpack(state)
                save(vid, wav, sr, path)
                nll = score(vid, edit)
                results[key] = {"slug": slug, "arm": arm, "edit": edit, "src": src, "nll": nll, "yes": math.exp(-nll),
                                "sec": round(time.time() - t0)}
                log.info("RESULT %-42s yes=%.4f nll=%.3f (%.0fs)", key, math.exp(-nll), nll, time.time() - t0)
            except Exception as e:
                log.exception("FAILED %s: %s", key, e)
                results[key] = {"slug": slug, "arm": arm, "error": str(e)[:300]}
            finally:
                pin_audio.set_target(None)
            json.dump(results, open(rp, "w"), indent=2)

    log.info("=== summary: baseline -> pinned (critic yes) ===")
    for slug in SCEN:
        vals = [results.get(f"{slug}__{a}", {}).get("yes", float("nan")) for a in ("edit__nopin", "edit__pin_h3snd", "edit__ref_h3snd", "edit__refkeep_h3snd")]
        if any(v == v for v in vals):
            log.info("%-28s nopin=%.4f pin=%.4f ref=%.4f refkeep=%.4f", slug, *vals)
    log.info("PIN ALL DONE")


if __name__ == "__main__":
    main()
