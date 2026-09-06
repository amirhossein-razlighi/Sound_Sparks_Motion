#!/usr/bin/env python3
"""SDEdit-style audio-video editing with a frozen Ovi model (Phase 2).

Ports our editing setup to Ovi (twin-DiT, cross-modal-fusion AV model):
  1. Encode the source video (Wan2.2 VAE) and source audio (MMAudio VAE).
  2. Flow-noise both latents to an intermediate sigma (SDEdit strength).
  3. Jointly denoise with the frozen Ovi fusion model under the EDIT prompt.
  4. Decode and save video+audio.

This file is also the foundation for Phase 3: `sdedit_generate` takes the
starting latents as arguments, runs OUTSIDE inference_mode, and can make the
last K denoising steps grad-carrying (grad_last_steps > 0) — gradients then
flow from the decoded frames back into `z_aud_start` through the audio->video
cross-attention.

Run from ovi_probe/Ovi (their configs are cwd-relative):
    cd ovi_probe/Ovi && python ../sdedit_edit.py \
        --src-video ../../input_videos/cook_spinach_main.mp4 \
        --prompt "A man pets the dog. <AUDCAP>Soft petting sounds...<ENDAUDCAP>" \
        --strength 0.6 --out-dir /scratch/amirrz/Ovi_exp/outputs/sdedit_test
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/Ovi")

import logging  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
log = logging.getLogger("sdedit")

from omegaconf import OmegaConf  # noqa: E402
from ovi.ovi_fusion_engine import OviFusionEngine  # noqa: E402
from ovi.utils.io_utils import save_video  # noqa: E402
from ovi.utils.processing_utils import snap_hw_to_multiple_of_32  # noqa: E402

CKPT_DIR = "/scratch/amirrz/Ovi_exp/ckpts"
AUDIO_SR = 16000
NUM_FRAMES = 121  # (31-1)*4+1 for the 5s model
FPS = 24


# ---------------------------------------------------------------------------
# Engine / IO
# ---------------------------------------------------------------------------

def load_engine(model_name: str = "720x720_5s") -> OviFusionEngine:
    config = OmegaConf.create({
        "ckpt_dir": CKPT_DIR,
        "model_name": model_name,
        "mode": "t2v",
        "cpu_offload": False,
        "fp8": False,
        "qint8": False,
    })
    t0 = time.time()
    engine = OviFusionEngine(config=config, device=0, target_dtype=torch.bfloat16)
    log.info("engine loaded in %.1fs, VRAM %.1f GB", time.time() - t0,
             torch.cuda.memory_allocated() / 1e9)
    return engine


def load_video_frames(path: str, target_area: int, num_frames: int = NUM_FRAMES) -> torch.Tensor:
    """Read a video, temporally resample to num_frames, resize to ~target_area.

    Returns [1, 3, F, H, W] float in [-1, 1] (Wan VAE input convention).
    """
    import cv2

    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames read from {path}")
    log.info("source video: %d frames %sx%s", len(frames), frames[0].shape[1], frames[0].shape[0])

    # Temporal resample (index interpolation) to exactly num_frames.
    idx = np.linspace(0, len(frames) - 1, num_frames).round().astype(int)
    frames = [frames[i] for i in idx]

    h0, w0 = frames[0].shape[:2]
    h, w = snap_hw_to_multiple_of_32(h0, w0, area=target_area)
    frames = [cv2.resize(f, (w, h), interpolation=cv2.INTER_LANCZOS4) for f in frames]

    arr = np.stack(frames).astype(np.float32) / 127.5 - 1.0  # [F,H,W,3]
    t = torch.from_numpy(arr).permute(3, 0, 1, 2).unsqueeze(0)  # [1,3,F,H,W]
    log.info("video tensor: %s (resized %dx%d -> %dx%d)", tuple(t.shape), w0, h0, w, h)
    return t


def load_audio_16k(path: str, seconds: float = NUM_FRAMES / FPS) -> torch.Tensor:
    """Extract mono 16k audio from the video file, pad/trim to `seconds`.

    Returns [1, samples] float in [-1, 1].
    """
    import torchaudio

    n_target = int(seconds * AUDIO_SR)
    try:
        wav, sr = torchaudio.load(path)
    except Exception:
        log.warning("no audio track in %s -> zeros", path)
        return torch.zeros(1, n_target)
    wav = wav.mean(dim=0, keepdim=True)
    if sr != AUDIO_SR:
        wav = torchaudio.functional.resample(wav, sr, AUDIO_SR)
    if wav.shape[1] < n_target:
        # loop-pad (repeats content; keeps energy) then trim
        reps = math.ceil(n_target / wav.shape[1])
        wav = wav.repeat(1, reps)
    wav = wav[:, :n_target]
    log.info("audio tensor: %s (%.2fs @ %dHz)", tuple(wav.shape), seconds, AUDIO_SR)
    return wav


# ---------------------------------------------------------------------------
# Latent encode
# ---------------------------------------------------------------------------

def encode_source(engine: OviFusionEngine, video: torch.Tensor, audio: torch.Tensor):
    """Encode source video+audio to Ovi's latent spaces.

    Returns (z_vid [C,Fl,Hl,Wl], z_aud [L,C]) fp32 on GPU, detached.
    """
    device = torch.device(f"cuda:{engine.device}" if isinstance(engine.device, int) else engine.device)
    with torch.no_grad():
        zv = engine.vae_model_video.wrapped_encode(
            video.to(device=device, dtype=torch.bfloat16))  # [1,C,Fl,Hl,Wl]
        # Audio encode must run fp32: the mel converter uses torch.stft and
        # cuFFT has no bf16 kernels (the stock engine never encodes audio, so
        # it never hits this). We upcast the audio VAE, call its internals
        # directly (avoiding the bf16 autocast in wrapped_encode), then cast
        # back so the decode path stays identical to stock.
        va = engine.vae_model_audio.float()
        mel = va.mel_converter(audio.to(device=device, dtype=torch.float32))
        za = va.tod.encode(mel).mean  # [1,C,L]
        engine.vae_model_audio = va.bfloat16()
    # .clone() is required: the MMAudio encode runs under inference_mode, and
    # inference tensors cannot be used later in autograd graphs (Phase 3).
    zv = zv.squeeze(0).float().clone()
    za = za.squeeze(0).float().clone()  # [C,L]
    if za.shape[0] != engine.audio_latent_channel and za.shape[1] == engine.audio_latent_channel:
        pass  # already [L,C]
    else:
        za = za.transpose(0, 1).contiguous()  # [C,L] -> [L,C]
    log.info("z_vid %s  z_aud %s  (expected vid C=%d len=%d, aud L=%d C=%d)",
             tuple(zv.shape), tuple(za.shape),
             engine.video_latent_channel, engine.video_latent_length,
             engine.audio_latent_length, engine.audio_latent_channel)
    assert zv.shape[0] == engine.video_latent_channel, f"video latent C mismatch: {zv.shape}"
    assert za.shape == (engine.audio_latent_length, engine.audio_latent_channel), \
        f"audio latent shape mismatch: {za.shape}"
    return zv, za


# ---------------------------------------------------------------------------
# SDEdit sampling (no inference_mode; grad-ready)
# ---------------------------------------------------------------------------

def encode_text(engine: OviFusionEngine, text_prompt: str,
                video_negative_prompt: str = "jitter, bad hands, blur, distortion",
                audio_negative_prompt: str = "robotic, muffled, echo, distorted"):
    """Encode (pos, vneg, aneg) once. Cache + offload T5 for optimization loops."""
    device = torch.device(f"cuda:{engine.device}" if isinstance(engine.device, int) else engine.device)
    text_prompt = engine.text_formatter(text_prompt)
    with torch.no_grad():
        embs = engine.text_model(
            [text_prompt, video_negative_prompt, audio_negative_prompt],
            engine.text_model.device)
    return [e.to(engine.target_dtype).to(device) for e in embs]


def sdedit_generate(
    engine: OviFusionEngine,
    *,
    text_prompt: str,
    z_vid_src: torch.Tensor,          # [C,Fl,Hl,Wl] fp32
    z_aud_start: torch.Tensor,        # [L,C] the (possibly optimized) audio latent
    strength: float = 0.6,
    seed: int = 42,
    sample_steps: int = 50,
    solver_name: str = "unipc",       # supplies the (shifted) sigma grid only
    shift: float = 5.0,
    video_guidance_scale: float = 4.0,
    audio_guidance_scale: float = 3.0,
    slg_layer: int = 11,
    video_negative_prompt: str = "jitter, bad hands, blur, distortion",
    audio_negative_prompt: str = "robotic, muffled, echo, distorted",
    with_grad: bool = False,          # Phase 3: grad through ALL SDEdit steps
    text_embeddings=None,             # optional precomputed (pos, vneg, aneg)
    decode_video: bool = True,
    decode_audio: bool = True,
    decode_latent_frames: int | None = None,  # decode only first N latent frames
    pin_audio: bool = False,          # re-impose z_aud_start at every step
    grad_last_steps: int = 0,         # >0: grad only through the last K steps
                                      # (requires pin_audio; LTX-style window)
):
    """Joint SDEdit denoise from flow-noised source latents under the edit prompt.

    Stepping is pure flow-Euler on the solver's shifted sigma grid
    (x_{i+1} = x_i + (sigma_{i+1}-sigma_i) * v_pred). A pure stepper is
    required for gradient checkpointing: scheduler.step mutates internal
    multistep state and would corrupt on checkpoint recompute.

    with_grad=False: everything under no_grad (Phase 2 baseline).
    with_grad=True : every SDEdit step is wrapped in torch.utils.checkpoint
    (stores only step-boundary latents; recomputes in backward), and the video
    decode carries grad. In Ovi the audio latent is the denoised STATE (it
    enters only at the first step), so the chain from the decoded frames back
    to z_aud_start must be unbroken through ALL steps — last-K windows would
    sever it.
    """
    device = torch.device(f"cuda:{engine.device}" if isinstance(engine.device, int) else engine.device)
    dtype = engine.target_dtype

    scheduler_v, _ = engine.get_scheduler_time_steps(
        sampling_steps=sample_steps, device=device, solver_name=solver_name, shift=shift)

    # --- pick start index: first sigma <= strength (sigmas descend ~1 -> 0) ---
    sigmas = scheduler_v.sigmas.tolist()  # len steps+1, ends at 0
    k0 = next((i for i, s in enumerate(sigmas[:-1]) if s <= strength), 0)
    sigma_start = float(sigmas[k0])
    run_sigmas = sigmas[k0:]              # [sigma_start, ..., 0]
    n_run = len(run_sigmas) - 1
    log.info("SDEdit: strength=%.2f -> start idx %d/%d (sigma=%.4f), %d Euler steps, grad=%s",
             strength, k0, sample_steps, sigma_start, n_run, with_grad)

    if text_embeddings is None:
        text_embeddings = encode_text(engine, text_prompt,
                                      video_negative_prompt, audio_negative_prompt)
    emb_pos, emb_vneg, emb_aneg = text_embeddings

    # --- flow-noise the source latents to sigma_start:  x_t = (1-s)x0 + s*eps ---
    gen = torch.Generator(device=device).manual_seed(seed)
    eps_v = torch.randn(z_vid_src.shape, device=device, dtype=torch.float32, generator=gen)
    eps_a = torch.randn(z_aud_start.shape, device=device, dtype=torch.float32, generator=gen)

    vid = ((1.0 - sigma_start) * z_vid_src.to(device) + sigma_start * eps_v).to(dtype)
    aud = ((1.0 - sigma_start) * z_aud_start.to(device) + sigma_start * eps_a).to(dtype)

    max_seq_len_audio = aud.shape[0]
    _ph, _pw = engine.model.video_model.patch_size[1], engine.model.video_model.patch_size[2]
    max_seq_len_video = vid.shape[1] * vid.shape[2] * vid.shape[3] // (_ph * _pw)

    pos_args = dict(audio_context=[emb_pos], vid_context=[emb_pos],
                    vid_seq_len=max_seq_len_video, audio_seq_len=max_seq_len_audio,
                    first_frame_is_clean=False)
    neg_args = dict(audio_context=[emb_aneg], vid_context=[emb_vneg],
                    vid_seq_len=max_seq_len_video, audio_seq_len=max_seq_len_audio,
                    first_frame_is_clean=False, slg_layer=slg_layer)

    def _pos_call(vid, aud, t_scalar):
        """Pure: positive CFG call. Checkpointed SEPARATELY from the negative
        call — backward recompute of BOTH 11B passes of one step at once
        (~50 GB) overflows the H100; one at a time is ~25 GB."""
        timestep_input = torch.full((1,), t_scalar, device=device)
        pv, pa = engine.model(vid=[vid], audio=[aud], t=timestep_input, **pos_args)
        return pv[0], pa[0]

    def _neg_call(vid, aud, t_scalar):
        timestep_input = torch.full((1,), t_scalar, device=device)
        pv, pa = engine.model(vid=[vid], audio=[aud], t=timestep_input, **neg_args)
        return pv[0], pa[0]

    def _euler_step(vid, aud, t_scalar, dt, ckpt: bool):
        if ckpt:
            pv_pos, pa_pos = torch.utils.checkpoint.checkpoint(
                _pos_call, vid, aud, t_scalar, use_reentrant=False)
            pv_neg, pa_neg = torch.utils.checkpoint.checkpoint(
                _neg_call, vid, aud, t_scalar, use_reentrant=False)
        else:
            pv_pos, pa_pos = _pos_call(vid, aud, t_scalar)
            pv_neg, pa_neg = _neg_call(vid, aud, t_scalar)
        pv = pv_neg + video_guidance_scale * (pv_pos - pv_neg)
        pa = pa_neg + audio_guidance_scale * (pa_pos - pa_neg)
        # flow matching ODE: x_{sigma+dt} = x + dt * v_pred  (dt < 0 here)
        return vid + dt * pv, aud + dt * pa

    # Last-K gradient window (LTX audio_opt_last_steps analog). Valid ONLY with
    # pin_audio: pinning re-injects z_aud_start at every step, so the gradient
    # reaches it through the last K steps alone — no need for an unbroken chain
    # from the initial noising (which needs per-step checkpoints whose backward
    # recompute of a single 11B call is >53 GB and cannot fit). In window mode
    # the grad steps run UNCHECKPOINTED; Ovi's per-block checkpointing (enabled
    # by the caller via model.train() + model.gradient_checkpointing) bounds
    # the graph to ~30 block-inputs x 2 CFG calls x K steps (~12 GB at K=2).
    window_mode = with_grad and grad_last_steps > 0
    if window_mode:
        assert pin_audio, "grad_last_steps requires pin_audio (audio must be per-step conditioning)"
    grad_from = n_run - min(grad_last_steps, n_run) if window_mode else 0

    t_loop = time.time()
    with torch.amp.autocast('cuda', enabled=dtype != torch.float32, dtype=dtype):
        for i in range(n_run):
            sig, sig_next = run_sigmas[i], run_sigmas[i + 1]
            t_scalar = sig * scheduler_v.config.num_train_timesteps
            dt = sig_next - sig
            step_grad = with_grad and (not window_mode or i >= grad_from)
            if step_grad:
                # window mode: no outer checkpoint (block ckpt inside model);
                # full-chain mode: split-CFG per-call checkpoints.
                vid, aud = _euler_step(vid, aud, t_scalar, dt, ckpt=not window_mode)
            else:
                with torch.no_grad():
                    vid, aud = _euler_step(vid, aud, t_scalar, dt, ckpt=False)
                vid, aud = vid.detach(), aud.detach()
            if pin_audio:
                # Retake-style pinning (LTX's post_process_latent analog):
                # re-impose the injected audio latent at the new noise level,
                # with the SAME eps draw, so the audio stream is per-step
                # conditioning instead of freely-evolving state. At sigma=0
                # this leaves exactly z_aud_start.
                aud = ((1.0 - sig_next) * z_aud_start.to(device) + sig_next * eps_a).to(dtype)
                # keep the pin grad-carrying only where the next step is a grad
                # step (or this is the final pin feeding the output latent)
                next_is_grad = with_grad and (not window_mode or (i + 1) >= grad_from)
                if not next_is_grad:
                    aud = aud.detach()
        log.info("denoise loop done in %.1fs", time.time() - t_loop)

        out = {"z_vid_final": vid, "z_aud_final": aud}

        if decode_video:
            lat = vid.unsqueeze(0)
            if decode_latent_frames is not None and decode_latent_frames < lat.shape[2]:
                # Wan2.2 VAE is causal-3D: decoding a temporal prefix is valid.
                # Used by the grad path to bound the decode graph's memory
                # (the loss then sees the first (N-1)*4+1 frames).
                lat = lat[:, :, :decode_latent_frames]
            # The video VAE may live on a second GPU (two-GPU optimization);
            # route the latent there and bring frames back.
            vdev = next(engine.vae_model_video.model.parameters()).device
            gen_video = engine.vae_model_video.wrapped_decode(lat.to(vdev))  # [1,3,F,H,W] fp32
            out["video"] = gen_video.squeeze(0).to(lat.device)  # grad-carrying if with_grad
        if decode_audio:
            with torch.no_grad():
                a = aud.detach().unsqueeze(0).transpose(1, 2)  # [1,C,L]
                gen_audio = engine.vae_model_audio.wrapped_decode(a)
            out["audio"] = gen_audio.squeeze().float().detach().cpu().numpy()

    return out


# ---------------------------------------------------------------------------
# CLI (Phase 2: baseline SDEdit run, no optimization)
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src-video", required=True)
    p.add_argument("--prompt", required=True,
                   help="Edit prompt; include <AUDCAP>...<ENDAUDCAP> for the audio caption.")
    p.add_argument("--strength", type=float, default=0.6,
                   help="SDEdit strength in (0,1]: fraction of noise injected (higher = more edit freedom).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample-steps", type=int, default=50)
    p.add_argument("--solver", default="unipc")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--tag", default="sdedit")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    engine = load_engine()

    video = load_video_frames(args.src_video, target_area=engine.target_area)
    audio = load_audio_16k(args.src_video)
    z_vid, z_aud = encode_source(engine, video, audio)

    # Sanity: pure reconstruction (decode the encoded source, no denoising).
    with torch.no_grad():
        recon = engine.vae_model_video.wrapped_decode(
            z_vid.unsqueeze(0).to(engine.target_dtype)).squeeze(0).cpu().float().numpy()
        ra = engine.vae_model_audio.wrapped_decode(
            z_aud.unsqueeze(0).transpose(1, 2).to(engine.target_dtype)).squeeze().cpu().float().numpy()
    save_video(os.path.join(args.out_dir, "vae_roundtrip.mp4"), recon, ra,
               sample_rate=AUDIO_SR, fps=FPS)
    log.info("saved VAE round-trip (upper bound on fidelity)")

    result = sdedit_generate(
        engine,
        text_prompt=args.prompt,
        z_vid_src=z_vid,
        z_aud_start=z_aud,
        strength=args.strength,
        seed=args.seed,
        sample_steps=args.sample_steps,
        solver_name=args.solver,
        grad_last_steps=0,
    )
    out_path = os.path.join(
        args.out_dir, f"{args.tag}_s{args.strength:.2f}_seed{args.seed}.mp4")
    save_video(out_path, result["video"].detach().cpu().float().numpy(),
               result["audio"], sample_rate=AUDIO_SR, fps=FPS)
    log.info("SUCCESS -> %s", out_path)


if __name__ == "__main__":
    main()
