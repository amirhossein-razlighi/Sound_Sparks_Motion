#!/usr/bin/env python3
"""Does SOUND spark motion in H3?  Controlled audio-reference substitution.

For each scenario where H3's text editing FAILED (baseline sweep), render with
the same seed / steps / frame grid and vary ONLY the audio reference and the
text prompt:

  arm            text prompt                    audio reference
  edit_srcaud    edit grammar (as in sweep)     source audio        <- sweep baseline (re-rendered here for a same-job control)
  edit_ltxaud    edit grammar (as in sweep)     LTX-optimized audio (our method's output on LTX-2)
  neutral_srcaud NO edit sentence; "motion      source audio        <- control
                 follows <Audio 2>"
  neutral_ltxaud NO edit sentence; "motion      LTX-optimized audio <- "sound sparks motion" arm
                 follows <Audio 2>"

If the motion appears in *_ltxaud but not *_srcaud, the audio pathway is
causal in H3 and the audio optimized by our method carries the motion across
backbones.  Every render is scored by the Qwen motion critic and saved with
H3's generated soundtrack (<slug>__<arm>.mp4 silent, .wav, _av.mp4 muxed).

    /scratch/amirrz/H3_exp/venv/bin/python h3_probe/audio_transfer_test.py
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
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
log = logging.getLogger("audxfer")

_spec = importlib.util.spec_from_file_location("baseline_sweep", os.path.join(_HERE, "baseline_sweep.py"))
_bs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bs)
grammar_prompt = _bs.grammar_prompt
CKPT, QWEN, RV, SW = _bs.CKPT, _bs.QWEN, _bs.RV, _bs.SW
STEPS, SEED, NUM_FRAMES, HEIGHT, WIDTH = _bs.STEPS, _bs.SEED, _bs.NUM_FRAMES, _bs.HEIGHT, _bs.WIDTH
OUT = os.environ.get("OUT_DIR", "/scratch/amirrz/H3_exp/outputs/audio_transfer")
ONLY = [s for s in os.environ.get("ONLY", "").split(",") if s]

# (slug, source video, source wav, LTX-optimized wav, edit sentence, scene) — scene strings identical to the sweep.
SCEN = [
    ("goldfish", f"{RV}/goldfish/retake_input_prepared.mp4", f"{SW}/goldfish.wav", f"{SW}/goldfish_ltxopt.wav",
     "The goldfish jumps out of the fish tank into the air.", "a goldfish in a fish tank"),
    ("turtle_extends_neck", f"{RV}/turtle_extends_neck/retake_input_prepared.mp4", f"{SW}/turtle_extends_neck.wav",
     f"{SW}/turtle_extends_neck_ltxopt.wav", "The turtle extends its neck forward out of its shell.", "a turtle"),
    ("cat_yawns", f"{RV}/transfers/cat_yawns/retake_input_prepared.mp4", f"{SW}/cat_yawns.wav", f"{SW}/cat_yawns_ltxopt.wav",
     "The cat yawns widely.", "a cat sitting"),
    ("boy_crouches", f"{RV}/boy_crouches/retake_input_prepared.mp4", f"{SW}/boy_crouches.wav", f"{SW}/boy_crouches_ltxopt.wav",
     "The boy crouches slightly and touches the water.", "a boy standing near water"),
    ("red_bird_opens_wings", f"{RV}/transfers/red_bird_opens_wings/retake_input_prepared.mp4", f"{SW}/red_bird_opens_wings.wav",
     f"{SW}/red_bird_opens_wings_ltxopt.wav", "The red bird opens its wings.", "a red bird perched"),
]


def neutral_prompt(scene: str) -> str:
    """Same grammar, NO mention of the edit: the subject's motion is left to the soundtrack."""
    return f"""subject_definitions:
<Video 1> is the source video to edit: a fixed-camera shot of {scene}.
<Subject 1> is the main subject of <Video 1>: {scene}.
<Audio 2> is the soundtrack accompanying the target video.

summary:
[video editing] The target video is an edited version of <Video 1>. The camera position, framing, scene, lighting, and pacing of <Video 1> are kept identical. The subject's actions follow the sounds in <Audio 2>.

retention_analysis:
<Audio 2>: reference - the target video's sound follows the content and timing of <Audio 2>; on-screen actions stay synchronized with it.
<Video 1> (camera, framing, scene layout, lighting, and temporal structure): fully_preserved - the fixed camera, background, lighting, and overall pacing are kept identical to the source.
<Subject 1> (appears throughout): partially_preserved - appearance and position are identical to <Video 1>; its actions are whatever the sounds in <Audio 2> imply.

detailed_description:
[Shot 1] Fixed camera, framing identical to <Video 1>. The scene of <Video 1>: {scene}. The subject moves in sync with the sounds heard in <Audio 2>. Everything else - the background, the lighting, the framing - remains exactly as in <Video 1>.

overall_soundscape: The soundscape follows <Audio 2>.

non_diegetic_music: None."""


def main():
    os.makedirs(OUT, exist_ok=True)
    from diffusers import ModularPipeline
    from diffusers.modular_pipelines import ComponentsManager
    from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3AudioReference, MiniMaxH3VideoReference
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

    def save(vid, state, path):
        export_to_video([np.asarray(f) for f in vid], path, fps=24)
        auds = (state.values.get("audio") if state.values.get("audio") is not None else state.values.get("audios"))
        if auds is None:
            return
        aud = auds[0] if isinstance(auds, (list, tuple)) else auds
        wav = np.asarray(torch.as_tensor(aud).float().cpu()) if torch.is_tensor(aud) else np.asarray(aud)
        wav = np.squeeze(wav)
        if wav.ndim == 2 and wav.shape[0] in (1, 2):
            wav = wav.T
        stem = os.path.splitext(path)[0]
        sf.write(stem + ".wav", wav, int(state.values.get("sampling_rate", 32000) or 32000))
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-threads", "2", "-i", path, "-i", stem + ".wav",
                        "-c:v", "copy", "-c:a", "aac", "-shortest", stem + "_av.mp4"], check=False)

    for slug, src, wav_src, wav_ltx, edit, scene in SCEN:
        if ONLY and slug not in ONLY:
            continue
        arms = [("edit_srcaud", grammar_prompt(edit, scene), wav_src),
                ("edit_ltxaud", grammar_prompt(edit, scene), wav_ltx),
                ("neutral_srcaud", neutral_prompt(scene), wav_src),
                ("neutral_ltxaud", neutral_prompt(scene), wav_ltx)]
        for arm, prompt, wav in arms:
            key = f"{slug}__{arm}"
            path = os.path.join(OUT, key + ".mp4")
            if key in results and os.path.exists(path):
                log.info("[skip] %s", key)
                continue
            t0 = time.time()
            try:
                refs = [MiniMaxH3VideoReference.from_file(src), MiniMaxH3AudioReference.from_file(wav)]
                state = pipe(prompt=prompt, references=refs, num_frames=NUM_FRAMES, height=HEIGHT, width=WIDTH,
                             num_inference_steps=STEPS, generator=torch.Generator("cuda").manual_seed(SEED),
                             output_type="np")
                vids = state.values.get("videos")
                vid = np.asarray(vids[0] if isinstance(vids, (list, tuple)) else vids)
                if vid.ndim == 5:
                    vid = vid[0]
                save(vid, state, path)
                nll = score(vid, edit)
                results[key] = {"slug": slug, "arm": arm, "edit": edit, "wav": wav, "nll": nll, "yes": math.exp(-nll),
                                "sec": round(time.time() - t0)}
                log.info("RESULT %-36s yes=%.4f nll=%.3f (%.0fs)", key, math.exp(-nll), nll, time.time() - t0)
            except Exception as e:
                log.exception("FAILED %s: %s", key, e)
                results[key] = {"slug": slug, "arm": arm, "error": str(e)[:300]}
            json.dump(results, open(rp, "w"), indent=2)

    log.info("=== summary (yes-prob) ===")
    for slug, *_ in SCEN:
        row = "  ".join(f"{a}={results.get(f'{slug}__{a}', {}).get('yes', float('nan')):.4f}"
                        for a in ("edit_srcaud", "edit_ltxaud", "neutral_srcaud", "neutral_ltxaud"))
        log.info("%-22s %s", slug, row)
    log.info("AUDIO TRANSFER DONE")


if __name__ == "__main__":
    main()
