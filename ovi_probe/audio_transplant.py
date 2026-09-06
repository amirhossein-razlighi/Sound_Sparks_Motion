#!/usr/bin/env python3
"""Audio-latent transplant probe: does a semantically-matched audio latent
steer video motion in Ovi, with NO optimization?

  1. Generate a clip with Ovi in (near-)T2V mode under the EDIT prompt
     (strength=1.0 SDEdit = denoise from noise) and harvest its final AUDIO
     latent — audio that semantically matches the target motion.
  2. SDEdit the SOURCE video as usual, but seed the audio stream with
     z_mix = (1-alpha) * z_src_audio + alpha * z_edit_audio, alpha in a sweep.
     alpha=0 is the no-op baseline; alpha=1 is a full transplant.
  3. Qwen-score every variant (same motion rubric as the paper).

If the target motion strengthens with alpha while the video stream's init,
noise, text and weights stay IDENTICAL, the audio pathway is steering video
motion — the paper's claim, demonstrated on a second architecture without
any gradient machinery.

Run from ovi_probe/Ovi:
    cd ovi_probe/Ovi && python ../audio_transplant.py \
        --src-video ../../input_videos/cook_spinach_main.mp4 \
        --prompt "The man pets the dog. <AUDCAP>...<ENDAUDCAP>" \
        --edit-prompt "The man pets the dog." \
        --out-dir /scratch/amirrz/Ovi_exp/outputs/transplant/man_pets_dog
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE + "/Ovi")
sys.path.insert(0, os.path.abspath(_HERE + "/../editing/src"))
sys.path.insert(0, _HERE)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
log = logging.getLogger("transplant")

from motion_opt.qwen_loss import (  # noqa: E402  (our module, unchanged)
    build_qwen_model,
    build_qwen_rubric_inputs,
    compute_qwen_video_loss,
)
from sdedit_edit import (  # noqa: E402
    AUDIO_SR,
    FPS,
    encode_source,
    encode_text,
    load_audio_16k,
    load_engine,
    load_video_frames,
    sdedit_generate,
)
from ovi.utils.io_utils import save_video  # noqa: E402

DEFAULT_QWEN = os.environ.get("QWEN_ROOT", "/project/def-amahdavi/amirrz/HF/models/Qwen2.5-VL-7B-Instruct")


def frames_for_qwen(video_cfhw: torch.Tensor) -> torch.Tensor:
    return ((video_cfhw + 1.0) / 2.0).clamp(0.0, 1.0).permute(1, 0, 2, 3).contiguous()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src-video", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--edit-prompt", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--strength", type=float, default=0.7)
    p.add_argument("--alphas", default="0.0,0.5,1.0",
                   help="Blend factors source->transplant audio latent.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample-steps", type=int, default=30)
    p.add_argument("--pin-audio", action="store_true",
                   help="Re-impose the audio latent at every denoise step "
                        "(retake-style pinning; the LTX mechanism analog).")
    p.add_argument("--qwen-model", default=DEFAULT_QWEN)
    p.add_argument("--qwen-max-frames", type=int, default=24)
    p.add_argument("--qwen-img-size", type=int, default=224)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    alphas = [float(a) for a in args.alphas.split(",")]

    engine = load_engine()
    engine.model.requires_grad_(False).eval() if hasattr(engine.model, "requires_grad_") else None

    video = load_video_frames(args.src_video, target_area=engine.target_area)
    audio = load_audio_16k(args.src_video)
    z_vid, z_aud_src = encode_source(engine, video, audio)
    text_embs = encode_text(engine, args.prompt)
    engine.offload_to_cpu(engine.text_model.model)
    torch.cuda.empty_cache()

    # ---- Qwen critic ----
    qwen_model, qwen_processor = build_qwen_model(
        args.qwen_model, device=device, gradient_checkpointing=True)
    qwen_num_frames = args.qwen_max_frames + (args.qwen_max_frames % 2)
    cached_inputs, yes_id, no_id = build_qwen_rubric_inputs(
        processor=qwen_processor, edit_prompt=args.edit_prompt,
        num_frames=qwen_num_frames, img_size=args.qwen_img_size, device=device,
        motion_question=None, gradient_rubric="motion")
    qwen_model.to("cpu")
    torch.cuda.empty_cache()

    def qwen_score(frames) -> float:
        qwen_model.to(device)
        with torch.no_grad():
            loss, _ = compute_qwen_video_loss(
                frames_chw=frames, qwen_model=qwen_model, cached_inputs=cached_inputs,
                yes_token_id=yes_id, no_token_id=no_id,
                max_frames=args.qwen_max_frames, img_size=args.qwen_img_size,
                backward=False, return_details=True, sample_mode="linspace",
                rubric_weight_overrides={"motion": 1.0, "entities": 0.0, "overall": 0.0})
        qwen_model.to("cpu")
        torch.cuda.empty_cache()
        return float(loss)

    # ---- Step 1: harvest an edit-matched audio latent (near-T2V, strength 1) ----
    log.info("harvesting edit-matched audio latent (strength=1.0 generation)...")
    t0 = time.time()
    with torch.no_grad():
        gen_out = sdedit_generate(
            engine, text_prompt=args.prompt,
            z_vid_src=torch.zeros_like(z_vid),   # content irrelevant at strength 1
            z_aud_start=torch.zeros_like(z_aud_src),
            strength=1.0, seed=args.seed + 777,   # independent generation
            sample_steps=args.sample_steps, text_embeddings=text_embs,
            decode_video=True, decode_audio=True)
    z_aud_edit = gen_out["z_aud_final"].float().detach().clone()
    log.info("harvested in %.1fs; |z_edit|=%.2f |z_src|=%.2f",
             time.time() - t0, z_aud_edit.norm(), z_aud_src.norm())
    save_video(os.path.join(args.out_dir, "edit_audio_donor.mp4"),
               gen_out["video"].detach().cpu().float().numpy(), gen_out["audio"],
               sample_rate=AUDIO_SR, fps=FPS)
    del gen_out
    torch.cuda.empty_cache()

    # ---- Step 2+3: alpha sweep, everything else identical ----
    results = {}
    for alpha in alphas:
        z_mix = (1.0 - alpha) * z_aud_src + alpha * z_aud_edit
        with torch.no_grad():
            out = sdedit_generate(
                engine, text_prompt=args.prompt,
                z_vid_src=z_vid, z_aud_start=z_mix,
                strength=args.strength, seed=args.seed,
                sample_steps=args.sample_steps, text_embeddings=text_embs,
                decode_video=True, decode_audio=True,
                pin_audio=args.pin_audio)
            nll = qwen_score(frames_for_qwen(out["video"]))
        yes = math.exp(-nll)
        results[alpha] = {"qwen_nll": nll, "yes_prob": yes}
        tag = f"alpha{alpha:.2f}" + ("_pinned" if args.pin_audio else "")
        save_video(os.path.join(args.out_dir, f"transplant_{tag}.mp4"),
                   out["video"].detach().cpu().float().numpy(), out["audio"],
                   sample_rate=AUDIO_SR, fps=FPS)
        log.info("alpha=%.2f  qwen_nll=%.4f  yes_prob=%.4f", alpha, nll, yes)
        del out
        torch.cuda.empty_cache()

    with open(os.path.join(args.out_dir, "transplant_summary.json"), "w") as f:
        json.dump({"alphas": results, "strength": args.strength,
                   "seed": args.seed, "edit_prompt": args.edit_prompt}, f, indent=2)
    log.info("TRANSPLANT DONE: %s", {f"{a:.2f}": f"{r['yes_prob']:.4f}" for a, r in results.items()})


if __name__ == "__main__":
    main()
