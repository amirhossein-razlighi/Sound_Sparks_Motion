#!/usr/bin/env python3
"""Stage-2 refinement of the paper-demo videos (LTX two-stage, second half).

For each dir with an OURS video (+ prompt.txt): VAE-encode -> 2x latent
upsample -> renoise to sigma0 -> distilled CFG-free Euler refinement (the
official STAGE_2 schedule) -> decode -> mux the ORIGINAL optimized soundtrack.
This is the half of the official pipeline our plain upscale skipped: the
distilled model re-projects the upscaled latent onto the video manifold,
regenerating detail (hands, edges, flicker) the upsampler can only sharpen.

Env knobs:
  SIGMAS    comma sigma list (default the official "0.909375,0.725,0.421875,0.0";
            gentler content-preserving variant: "0.421875,0.0")
  OUT_NAME  final 1152x720 output name (default highres_refined.mp4)
  DIRS      space-separated dir list (default: all of results/videos/* and
            results/videos/transfers/* that have prompt.txt + an OURS video)
  FORCE=1   redo existing outputs
"""
from __future__ import annotations

import glob
import logging
import os
import subprocess
import sys
import time

import cv2
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
log = logging.getLogger("refine")

# Import order matters: ltx_pipelines.retake fully initializes ltx_core.loader
# BEFORE anything touches ltx_core.quantization (fp8_cast <-> loader circular).
import ltx_pipelines.retake  # noqa: E402,F401

from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps  # noqa: E402
from ltx_core.model.upsampler import upsample_video  # noqa: E402
from ltx_core.model.video_vae import TilingConfig, decode_video, get_video_chunks_number  # noqa: E402
from ltx_core.types import VideoPixelShape  # noqa: E402
from ltx_pipelines.retake import _encode_audio_for_retake, _encode_video_for_retake  # noqa: E402
from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT, STAGE_2_DISTILLED_SIGMA_VALUES  # noqa: E402
from ltx_core.components.diffusion_steps import EulerDiffusionStep  # noqa: E402
from ltx_core.components.noisers import GaussianNoiser  # noqa: E402
from ltx_pipelines.utils.helpers import (  # noqa: E402
    PipelineComponents,
    denoise_audio_video,
    encode_prompts,
    simple_denoising_func,
)
from ltx_pipelines.utils.samplers import euler_denoising_loop  # noqa: E402

from motion_opt.models import resolve_quantization_policy  # noqa: E402
from ltx_pipelines.utils.media_io import decode_audio_from_file, encode_video  # noqa: E402
from ltx_pipelines.utils.model_ledger import ModelLedger  # noqa: E402

CKPT = os.path.expandvars("${CKPT_ROOT}/ltx-2.3-22b-dev.safetensors")
UPSAMPLER = os.path.expandvars("${CKPT_ROOT}/ltx-2.3-spatial-upscaler-x2-1.0.safetensors")
DISTILLED = os.environ.get("DISTILLED_LORA") or os.path.expandvars("${CKPT_ROOT}/ltx-2.3-22b-distilled-lora-384.safetensors")
GEMMA = os.path.expandvars("${GEMMA_ROOT}/")


def video_meta(path):
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    cap.release()
    return n, w, h, fps


def find_src(d):
    for pat in ("best_optimized_*.mp4", "transfer_both.mp4"):
        m = sorted(glob.glob(os.path.join(d, pat)))
        if m:
            return m[0]
    return None


def main() -> None:
    device = torch.device("cuda")
    dtype = torch.bfloat16
    sigmas_env = os.environ.get("SIGMAS")
    sigma_list = ([float(x) for x in sigmas_env.split(",")] if sigmas_env
                  else list(STAGE_2_DISTILLED_SIGMA_VALUES))
    out_name = os.environ.get("OUT_NAME", "highres_refined.mp4")
    force = os.environ.get("FORCE", "0") == "1"

    for p in (CKPT, UPSAMPLER, DISTILLED):
        assert os.path.exists(p), f"missing: {p}"

    dirs_env = os.environ.get("DIRS")
    if dirs_env:
        dirs = dirs_env.split()
    else:
        dirs = sorted(d for d in glob.glob("results/videos/*/") + glob.glob("results/videos/transfers/*/")
                      if find_src(d) and os.path.exists(os.path.join(d, "prompt.txt")))
    log.info("sigmas=%s  out=%s  dirs=%d", sigma_list, out_name, len(dirs))

    quant = resolve_quantization_policy("fp8-cast")
    ledger = ModelLedger(dtype=dtype, device=device, checkpoint_path=CKPT,
                         gemma_root_path=GEMMA, spatial_upsampler_path=UPSAMPLER,
                         quantization=quant)
    distilled_ledger = ledger.with_additional_loras(
        loras=[LoraPathStrengthAndSDOps(DISTILLED, 1.0, LTXV_LORA_COMFY_RENAMING_MAP)])
    upsampler = ledger.spatial_upsampler()
    components = PipelineComponents(dtype=dtype, device=device)
    stepper = EulerDiffusionStep()
    tiling = TilingConfig.default()

    transformer = None  # built lazily once, after first Gemma encode is freed

    for d in dirs:
        src = find_src(d)
        out_hd = os.path.join(d, out_name)
        out_2x = os.path.join(d, out_name.replace(".mp4", "_2x.mp4"))
        if not force and os.path.exists(out_hd):
            log.info("[skip] %s", d)
            continue
        prompt = open(os.path.join(d, "prompt.txt")).read().strip()
        n, w, h, fps = video_meta(src)
        t0 = time.time()
        log.info("[%s] %dx%d %df '%s'", os.path.basename(os.path.normpath(d)), w, h, n, prompt[:60])

        # ---- contexts (Gemma; loaded+freed inside encode_prompts) ----
        ctx_p, _ctx_n = encode_prompts([prompt, DEFAULT_NEGATIVE_PROMPT], ledger)
        v_ctx, a_ctx = ctx_p.video_encoding, ctx_p.audio_encoding

        # ---- encode source video + audio latents ----
        shape1 = VideoPixelShape(batch=1, frames=n, height=h, width=w, fps=fps)
        encoder = ledger.video_encoder()
        with torch.no_grad():
            latent = _encode_video_for_retake(encoder, src, shape1, dtype, device)
            up2 = upsample_video(latent, encoder, upsampler)
        audio = decode_audio_from_file(src, device, max_duration=n / fps)
        shape2 = VideoPixelShape(batch=1, frames=n, height=h * 2, width=w * 2, fps=fps)
        with torch.no_grad():
            audio_latent = _encode_audio_for_retake(
                ledger.audio_encoder(), audio.waveform.to(device), audio.sampling_rate, shape2, dtype)

        if transformer is None:
            transformer = distilled_ledger.transformer()

        # ---- stage-2 distilled refinement (CFG-free, official schedule) ----
        sigmas = torch.tensor(sigma_list, dtype=torch.float32, device=device)
        noiser = GaussianNoiser(generator=torch.Generator(device=device).manual_seed(42))

        def loop_fn(sigmas, video_state, audio_state, stepper):
            return euler_denoising_loop(
                sigmas=sigmas, video_state=video_state, audio_state=audio_state,
                stepper=stepper,
                denoise_fn=simple_denoising_func(
                    video_context=v_ctx, audio_context=a_ctx, transformer=transformer))

        with torch.no_grad():
            video_state, _audio_state = denoise_audio_video(
                output_shape=shape2, conditionings=[], noiser=noiser, sigmas=sigmas,
                stepper=stepper, denoising_loop_fn=loop_fn, components=components,
                dtype=dtype, device=device, noise_scale=float(sigma_list[0]),
                initial_video_latent=up2, initial_audio_latent=audio_latent)

            it = decode_video(video_state.latent, ledger.video_decoder(), tiling)
            encode_video(video=it, fps=int(round(fps)), audio=audio, output_path=out_2x,
                         video_chunks_number=get_video_chunks_number(n, tiling))

        subprocess.run(["ffmpeg", "-y", "-v", "error", "-threads", "2", "-i", out_2x,
                        "-vf", "scale=1152:720:flags=lanczos", "-c:v", "libx264", "-crf", "16",
                        "-pix_fmt", "yuv420p", "-c:a", "copy", out_hd], check=True)
        log.info("  wrote %s (%.1fs)", out_hd, time.time() - t0)
        del latent, up2, audio_latent, video_state, ctx_p, v_ctx, a_ctx
        torch.cuda.empty_cache()

    log.info("REFINE DONE (%d dirs)", len(dirs))


if __name__ == "__main__":
    main()
