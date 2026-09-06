#!/usr/bin/env python3
"""Upscale the paper-demo videos with the LTX spatial (latent) upsampler.

For every subdir of --videos-root containing best_optimized_*.mp4 (OURS):
  1. VAE-encode the video (512x320) to latents.
  2. Apply the LTX-2.3 x2 latent upsampler once  -> decode -> highres_2x.mp4 (1024x640).
  3. Apply it twice (4x latent)                  -> decode -> lanczos to 1728x1080
     -> highres.mp4 (true 1080p; 1728/1080 = 1.6 = source aspect, no crop).
  4. The original (optimized) audio track is muxed back into both.

Uses only the vendored LTX components (ModelLedger VAE + spatial_upsampler +
upsample_video), same encode/decode conventions as the pipelines. Resumable:
skips a dir whose outputs already exist (FORCE=1 redoes).

    python editing/scripts/upsample_highres.py \
        --videos-root results/videos --checkpoint $CKPT_ROOT/ltx-2.3-22b-dev.safetensors \
        --upsampler $CKPT_ROOT/ltx-2.3-spatial-upscaler-x2-1.0.safetensors
"""
from __future__ import annotations

import argparse
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
log = logging.getLogger("highres")

from ltx_core.model.upsampler import upsample_video  # noqa: E402
from ltx_core.model.video_vae import TilingConfig, decode_video, get_video_chunks_number  # noqa: E402
from ltx_core.types import VideoPixelShape  # noqa: E402
from ltx_pipelines.retake import _encode_video_for_retake  # noqa: E402
from ltx_pipelines.utils.media_io import decode_audio_from_file, encode_video  # noqa: E402
from ltx_pipelines.utils.model_ledger import ModelLedger  # noqa: E402


def video_meta(path: str) -> tuple[int, int, int, float]:
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    cap.release()
    return n, w, h, fps


def decode_and_save(latent: torch.Tensor, ledger: ModelLedger, out_path: str,
                    num_frames: int, fps: float, audio) -> None:
    tiling = TilingConfig.default()
    with torch.no_grad():
        it = decode_video(latent, ledger.video_decoder(), tiling)
        encode_video(
            video=it,
            fps=int(round(fps)),
            audio=audio,
            output_path=out_path,
            video_chunks_number=get_video_chunks_number(num_frames, tiling),
        )
    log.info("  wrote %s", out_path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--videos-root", default="results/videos")
    p.add_argument("--checkpoint", default=os.path.expandvars("${CKPT_ROOT}/ltx-2.3-22b-dev.safetensors"))
    p.add_argument("--upsampler", default=os.path.expandvars("${CKPT_ROOT}/ltx-2.3-spatial-upscaler-x2-1.0.safetensors"))
    p.add_argument("--pattern", default="best_optimized_*.mp4")
    args = p.parse_args()

    force = os.environ.get("FORCE", "0") == "1"
    device = torch.device("cuda")
    dtype = torch.bfloat16

    assert os.path.exists(args.checkpoint), f"checkpoint missing: {args.checkpoint}"
    assert os.path.exists(args.upsampler), f"upsampler missing: {args.upsampler}"

    ledger = ModelLedger(
        dtype=dtype, device=device,
        checkpoint_path=args.checkpoint,
        spatial_upsampler_path=args.upsampler,
    )
    upsampler = ledger.spatial_upsampler()

    dirs = sorted(d for d in glob.glob(os.path.join(args.videos_root, "*"))
                  if os.path.isdir(d) and glob.glob(os.path.join(d, args.pattern)))
    log.info("found %d dirs with %s", len(dirs), args.pattern)

    for d in dirs:
        src = sorted(glob.glob(os.path.join(d, args.pattern)))[0]
        out_2x = os.path.join(d, "highres_2x.mp4")
        out_hd = os.path.join(d, "highres.mp4")
        if not force and os.path.exists(out_hd) and os.path.exists(out_2x):
            log.info("[skip] %s (outputs exist)", d)
            continue

        t0 = time.time()
        n, w, h, fps = video_meta(src)
        log.info("[%s] %s  %dx%d %df @%.1ffps", os.path.basename(d), os.path.basename(src), w, h, n, fps)
        audio = decode_audio_from_file(src, device, max_duration=n / fps)

        encoder = ledger.video_encoder()
        shape = VideoPixelShape(batch=1, frames=n, height=h, width=w, fps=fps)
        with torch.no_grad():
            latent = _encode_video_for_retake(encoder, src, shape, dtype, device)
            up2 = upsample_video(latent, encoder, upsampler)          # 2x

        decode_and_save(up2, ledger, out_2x, n, fps, audio)           # 1024x640

        # Verdict from visual QA: a second upsampler pass smears detail
        # (out-of-distribution on its own output). Best deliverable = the
        # native 2x pass + mild 1.125x lanczos to exact 720p.
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-threads", "2", "-i", out_2x,
             "-vf", "scale=1152:720:flags=lanczos", "-c:v", "libx264", "-crf", "16",
             "-pix_fmt", "yuv420p", "-c:a", "copy", out_hd],
            check=True)
        log.info("  wrote %s  (%.1fs total)", out_hd, time.time() - t0)

        del latent, up2
        torch.cuda.empty_cache()

    log.info("ALL DONE (%d dirs)", len(dirs))


if __name__ == "__main__":
    main()
