"""Pin H3's *generated* audio rows to a chosen audio latent (audio->video mode).

H3 is a single-stream omni DiT: the packed sequence holds text rows, the video
reference rows (noise-augmented, fixed), the audio *reference* rows (fixed,
t=1 = clean), the generated video rows and the generated audio rows. The
generated audio rows are an evolving latent stepped down the audio schedule
(shift 3) - the exact analogue of the LTX audio latent our method optimizes.

`install_pin` monkeypatches the loop's scheduler step so that after every
step the generated audio rows are overwritten with the forward-noised target
latent at the *next* sigma (`x_t = t*x0 + (1-t)*eps`, fixed eps). The video
rows must then stay consistent with that sound through joint attention at
every step, and the decoded soundtrack IS the pinned sound. Nothing else in
the pipeline changes; `set_target(None)` restores normal generation.
"""
from __future__ import annotations

import math

import numpy as np
import torch

_STATE = {"rows": None, "eps": None, "seed": 0, "pin_until": None, "installed": False}


def target_rows(pipe, wav_path: str, num_audio_latents: int, device) -> torch.Tensor:
    """Encode a (mono or stereo) 32 kHz wav into normalized, channel-major generated-audio rows
    of exactly the pipeline's shape [audio_channels * num_audio_latents, audio_latent_channels]."""
    import soundfile as sf
    wav, sr = sf.read(wav_path, dtype="float32", always_2d=True)  # [N, ch]
    assert sr == 32000, f"{wav_path}: expected 32 kHz, got {sr}"
    wav = wav.mean(1)
    need = (num_audio_latents + 4) * 800  # 800-sample hop, small margin
    wav = np.tile(wav, int(math.ceil(need / max(1, len(wav)))))[:need]
    audio_vae = pipe.audio_vae
    mean = torch.tensor(audio_vae.config.latents_mean).view(1, 1, -1)
    std = torch.tensor(audio_vae.config.latents_std).view(1, 1, -1)
    x = torch.from_numpy(wav)[None, None].to(device, next(audio_vae.parameters()).dtype)
    with torch.no_grad():
        lat = audio_vae.encode(x, return_dict=False)[0].mode().float().cpu().transpose(1, 2)  # [1, T, C]
    z = ((lat - mean) / std)[0]
    assert z.shape[0] >= num_audio_latents, (z.shape, num_audio_latents)
    z = z[:num_audio_latents]
    ch = int(pipe.audio_channels)
    return z.repeat(ch, 1).contiguous()  # channel-major: ch0 rows, then ch1 rows


def set_target(rows: torch.Tensor | None, seed: int = 0, pin_until: int | None = None) -> None:
    """rows: [audio_channels*T, C] normalized target (None = no pinning).
    pin_until: pin only through this step index (None = every step incl. the last -> clean target)."""
    _STATE.update(rows=None if rows is None else rows.detach().float().cpu(), eps=None, seed=seed, pin_until=pin_until)


def install_pin() -> None:
    if _STATE["installed"]:
        return
    import diffusers.modular_pipelines.minimax_h3.denoise as dn
    cls = dn.MiniMaxH3LoopSchedulerStep
    orig = cls.__call__

    def patched(self, components, block_state, i, t):
        components, block_state = orig(self, components, block_state, i, t)
        rows = _STATE["rows"]
        if rows is None:
            return components, block_state
        if _STATE["pin_until"] is not None and i + 1 > _STATE["pin_until"]:
            return components, block_state
        n = int(block_state.num_condition_audio_rows)
        gen = block_state.audio_latents[n:]
        assert gen.shape == rows.shape, f"pinned rows {tuple(rows.shape)} != generated rows {tuple(gen.shape)}"
        ts = block_state.audio_timesteps
        t_next = float(ts[i + 1]) if i + 1 < len(ts) else 1.0  # H3 convention: t=1 is clean
        if _STATE["eps"] is None:
            g = torch.Generator(device=gen.device).manual_seed(int(_STATE["seed"]))
            _STATE["eps"] = torch.randn(gen.shape, generator=g, device=gen.device, dtype=torch.float32)
        x0 = rows.to(gen.device, torch.float32)
        block_state.audio_latents[n:] = components.audio_scheduler.scale_noise(x0, t_next, _STATE["eps"]).to(gen.dtype)
        return components, block_state

    cls.__call__ = patched
    _STATE["installed"] = True
