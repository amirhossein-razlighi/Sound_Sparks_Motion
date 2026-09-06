#!/usr/bin/env python3
"""Fair evaluation of the H3 optimization arms at FULL steps (40), apples-to-apples.

For each arm dir (with latents.pt from optimize_audio.py / optimize_av.py):
  baseline_40.mp4   z_src, no text delta   -> scored
  optimized_40.mp4  z (+ dt if present)     -> scored
Same seed/noise/prompt/refs; only the optimized conditioning differs. Also
settles attribution: if baseline_40 is frame-aligned, alignment comes from the
audio reference, not the step count.

    ARMS="opt_mpd_both opt_mpd" /scratch/amirrz/H3_exp/venv/bin/python h3_probe/eval_arms.py
"""
import json
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
log = logging.getLogger("h3eval")

CKPT = "/scratch/amirrz/H3_exp/ckpt"
OUT_ROOT = "/scratch/amirrz/H3_exp/outputs"
SRC = "/home/amirrz/my_codes/Sound_Sparks_Motion/input_videos/cook_spinach_main.mp4"
AUDIO_WAV = "/scratch/amirrz/H3_exp/inputs/mpd_source.wav"
PROMPT_FILE = "/scratch/amirrz/H3_exp/inputs/edit_prompt_mpd_audio.txt"
EDIT_PROMPT = "The man pets the dog."
QWEN = "/project/def-amahdavi/amirrz/HF/models/Qwen2.5-VL-7B-Instruct"
STEPS = int(os.environ.get("STEPS", "40"))
SEED = 42
NUM_FRAMES, HEIGHT, WIDTH = 124, 448, 768
ARMS = os.environ.get("ARMS", "opt_mpd_both opt_mpd").split()


class _FixedPosterior:
    def __init__(self, latent):
        self._l = latent

    def mode(self):
        return self._l


def main() -> None:
    from diffusers import ModularPipeline
    from diffusers.modular_pipelines import ComponentsManager
    from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3AudioReference, MiniMaxH3VideoReference
    import diffusers.modular_pipelines.minimax_h3.encoders as h3_enc
    from diffusers.utils import export_to_video
    from motion_opt.qwen_loss import build_qwen_model, build_qwen_rubric_inputs, compute_qwen_video_loss

    cm = ComponentsManager()
    cm.enable_auto_cpu_offload(device="cuda")
    pipe = ModularPipeline.from_pretrained(CKPT, workflow="ref2va", components_manager=cm)
    pipe.load_components(torch_dtype=torch.bfloat16)

    audio_vae = pipe.audio_vae
    a_mean = torch.tensor(audio_vae.config.latents_mean).view(1, 1, -1)
    a_std = torch.tensor(audio_vae.config.latents_std).view(1, 1, -1)
    orig_encode = audio_vae.encode
    inject = {"z": None}

    def patched_encode(x, return_dict=False):
        if inject["z"] is None:
            return orig_encode(x, return_dict=return_dict)
        denorm = (inject["z"] * a_std + a_mean).transpose(1, 2).to(x.device, dtype=torch.float32)
        return (_FixedPosterior(denorm),)

    audio_vae.encode = patched_encode

    _orig_gpe = h3_enc.get_qwen3vl_prompt_embeds
    text_cache = {"base": None}
    inject_text = {"delta": None}

    def patched_gpe(*a, **k):
        if text_cache["base"] is None:
            text_cache["base"] = _orig_gpe(*a, **k)
        base = text_cache["base"]
        d = inject_text["delta"]
        return base if d is None else (base + d.to(base.device, base.dtype))

    h3_enc.get_qwen3vl_prompt_embeds = patched_gpe

    qwen_model, qwen_processor = build_qwen_model(QWEN, device=torch.device("cuda"), gradient_checkpointing=False)
    qwen_model.to("cpu")
    torch.cuda.empty_cache()
    rub = {}  # per-arm rubric inputs (edit prompt differs per scenario)
    ctx = {"prompt": None, "refs": None}
    windows = [("linspace", 0), ("contiguous", 0), ("contiguous", 50), ("contiguous", 100)]
    overrides = {"motion": 1.0, "entities": 0.0, "overall": 0.0}

    def score(video_np):
        cached_inputs, yes_id, no_id = rub["cur"]
        fr = torch.from_numpy(np.ascontiguousarray(video_np)).permute(0, 3, 1, 2).float().clamp(0, 1).cuda()
        qwen_model.to("cuda")
        acc = 0.0
        with torch.no_grad():
            for mode, start in windows:
                loss, _ = compute_qwen_video_loss(
                    frames_chw=fr, qwen_model=qwen_model, cached_inputs=cached_inputs,
                    yes_token_id=yes_id, no_token_id=no_id, max_frames=24, img_size=224,
                    backward=False, return_details=True, sample_mode=mode,
                    contiguous_start_frame=start, rubric_weight_overrides=overrides)
                acc += float(loss)
        qwen_model.to("cpu")
        torch.cuda.empty_cache()
        return acc / len(windows)

    def render(z, dt):
        inject["z"], inject_text["delta"] = z, dt
        t0 = time.time()
        state = pipe(prompt=ctx["prompt"], references=ctx["refs"], num_frames=NUM_FRAMES, height=HEIGHT, width=WIDTH,
                     num_inference_steps=STEPS, generator=torch.Generator("cuda").manual_seed(SEED),
                     output_type="np")
        vids = state.values.get("videos")
        vid = np.asarray(vids[0] if isinstance(vids, (list, tuple)) else vids)
        if vid.ndim == 5:
            vid = vid[0]
        log.info("  render %d steps: %.0fs", STEPS, time.time() - t0)
        return vid

    for arm in ARMS:
        d = os.path.join(OUT_ROOT, arm)
        lp = os.path.join(d, "latents.pt")
        if not os.path.exists(lp):
            log.warning("[skip] %s has no latents.pt", arm)
            continue
        L = torch.load(lp, map_location="cpu")
        rc = json.load(open(os.path.join(d, "run_config.json"))) if os.path.exists(os.path.join(d, "run_config.json")) else \
            {"src": SRC, "audio_wav": AUDIO_WAV, "prompt_file": PROMPT_FILE, "edit_prompt": EDIT_PROMPT}
        ctx["prompt"] = open(rc["prompt_file"]).read().strip()
        ctx["refs"] = [MiniMaxH3VideoReference.from_file(rc["src"]), MiniMaxH3AudioReference.from_file(rc["audio_wav"])]
        text_cache["base"] = None  # new prompt/refs -> re-encode text once
        rub["cur"] = build_qwen_rubric_inputs(
            processor=qwen_processor, edit_prompt=rc["edit_prompt"], num_frames=24, img_size=224,
            device=torch.device("cuda"), motion_question=None, gradient_rubric="motion")
        log.info("arm %s: edit_prompt=%r", arm, rc["edit_prompt"])
        z_src, z_opt = L["z_src"], L["z"]
        dt_opt = L.get("dt")
        log.info("=== arm %s  (text residual: %s) ===", arm, "yes" if dt_opt is not None else "no")

        vb = render(z_src, None)
        nb = score(vb)
        export_to_video([np.asarray(f) for f in vb], os.path.join(d, f"baseline_{STEPS}.mp4"), fps=24)
        vo = render(z_opt, dt_opt)
        no = score(vo)
        export_to_video([np.asarray(f) for f in vo], os.path.join(d, f"optimized_{STEPS}.mp4"), fps=24)

        res = {"arm": arm, "steps": STEPS, "baseline_nll": nb, "baseline_yes": math.exp(-nb),
               "optimized_nll": no, "optimized_yes": math.exp(-no),
               "audio_latent_l2_shift": float((z_opt - z_src).norm()),
               "text_delta_l2": float(dt_opt.norm()) if dt_opt is not None else 0.0}
        json.dump(res, open(os.path.join(d, f"eval_{STEPS}.json"), "w"), indent=2)
        log.info("RESULT %s: baseline yes=%.4f -> optimized yes=%.4f  (nll %.4f -> %.4f)",
                 arm, res["baseline_yes"], res["optimized_yes"], nb, no)

    log.info("EVAL DONE")


if __name__ == "__main__":
    main()
