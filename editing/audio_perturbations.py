"""
Audio perturbation functions for the audio-edit → video experiment.

Each perturbation takes a waveform tensor of shape (C, T) (float32, values in [-1, 1])
and the sample rate, and returns a modified waveform of the same shape.

Registry: call `get_all_perturbations()` to get the full list of Perturbation objects,
or `get_perturbation_by_name(name)` to retrieve a single one.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torchaudio
from scipy import signal as scipy_signal

# ---------------------------------------------------------------------------
# Perturbation dataclass
# ---------------------------------------------------------------------------


@dataclass
class Perturbation:
    name: str
    description: str
    fn: Callable[[torch.Tensor, int], torch.Tensor]

    def apply(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        """Apply the perturbation.  Input: (C, T) float32 in [-1, 1]."""
        return self.fn(waveform, sr)


# ---------------------------------------------------------------------------
# Individual perturbation functions
# All accept (waveform: Tensor[C, T], sr: int) → Tensor[C, T]
# ---------------------------------------------------------------------------


def identity(waveform: torch.Tensor, sr: int) -> torch.Tensor:
    """No change — baseline reference."""
    return waveform.clone()


def add_gaussian_noise(waveform: torch.Tensor, sr: int, snr_db: float) -> torch.Tensor:
    """Add white Gaussian noise at the specified SNR (dB)."""
    signal_power = waveform.pow(2).mean() + 1e-10
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = torch.randn_like(waveform) * noise_power.sqrt()
    return (waveform + noise).clamp(-1.0, 1.0)


def replace_with_silence(waveform: torch.Tensor, sr: int) -> torch.Tensor:
    """Replace audio with complete silence (zeros)."""
    return torch.zeros_like(waveform)


def replace_with_white_noise(waveform: torch.Tensor, sr: int) -> torch.Tensor:
    """Replace audio with pure white noise at the same RMS level as the original."""
    rms = waveform.pow(2).mean().sqrt() + 1e-10
    noise = torch.randn_like(waveform) * rms
    return noise.clamp(-1.0, 1.0)


def lowpass_filter(waveform: torch.Tensor, sr: int, cutoff_hz: float) -> torch.Tensor:
    """Apply a 4th-order Butterworth low-pass filter."""
    nyq = sr / 2.0
    norm_cutoff = min(cutoff_hz / nyq, 0.99)
    sos = scipy_signal.butter(4, norm_cutoff, btype="low", output="sos")
    arr = waveform.cpu().numpy()
    filtered = scipy_signal.sosfilt(sos, arr, axis=-1)
    return torch.from_numpy(filtered.astype(np.float32)).to(waveform.device).clamp(-1.0, 1.0)


def highpass_filter(waveform: torch.Tensor, sr: int, cutoff_hz: float) -> torch.Tensor:
    """Apply a 4th-order Butterworth high-pass filter."""
    nyq = sr / 2.0
    norm_cutoff = min(cutoff_hz / nyq, 0.99)
    sos = scipy_signal.butter(4, norm_cutoff, btype="high", output="sos")
    arr = waveform.cpu().numpy()
    filtered = scipy_signal.sosfilt(sos, arr, axis=-1)
    return torch.from_numpy(filtered.astype(np.float32)).to(waveform.device).clamp(-1.0, 1.0)


def bandpass_filter(waveform: torch.Tensor, sr: int, low_hz: float, high_hz: float) -> torch.Tensor:
    """Retain only the frequency band between low_hz and high_hz."""
    nyq = sr / 2.0
    low = min(low_hz / nyq, 0.98)
    high = min(high_hz / nyq, 0.99)
    sos = scipy_signal.butter(4, [low, high], btype="bandpass", output="sos")
    arr = waveform.cpu().numpy()
    filtered = scipy_signal.sosfilt(sos, arr, axis=-1)
    return torch.from_numpy(filtered.astype(np.float32)).to(waveform.device).clamp(-1.0, 1.0)


def phase_randomize(waveform: torch.Tensor, sr: int) -> torch.Tensor:
    """Destroy temporal structure: randomize phase spectrum, keep magnitude.

    The resulting audio has the same spectral power as the original but all
    phase coherence across frequencies (i.e., temporal events / onsets) is
    lost.  Sounds like Gaussian noise coloured to match the source spectrum.
    """
    result = torch.zeros_like(waveform)
    n = waveform.shape[-1]
    for c in range(waveform.shape[0]):
        f = torch.fft.rfft(waveform[c].float())
        magnitude = f.abs()
        random_phase = torch.rand(f.shape, device=waveform.device) * 2.0 * math.pi
        f_new = magnitude * torch.exp(1j * random_phase)
        result[c] = torch.fft.irfft(f_new, n=n).clamp(-1.0, 1.0).to(waveform.dtype)
    return result


def time_reverse(waveform: torch.Tensor, sr: int) -> torch.Tensor:
    """Reverse the audio waveform in time."""
    return waveform.flip(-1).clone()


def time_shuffle(waveform: torch.Tensor, sr: int, chunk_sec: float = 1.0) -> torch.Tensor:
    """Shuffle fixed-length chunks in random order, destroying temporal continuity.

    Args:
        chunk_sec: chunk length in seconds (default 1 second).
    """
    chunk_size = max(1, int(sr * chunk_sec))
    n = waveform.shape[-1]
    n_full_chunks = n // chunk_size

    if n_full_chunks < 2:
        return waveform.clone()

    # Build shuffled result from full chunks + keep any tail unchanged
    order = torch.randperm(n_full_chunks)
    chunks = [waveform[..., i * chunk_size:(i + 1) * chunk_size] for i in order]
    shuffled = torch.cat(chunks, dim=-1)

    # Re-attach leftover tail
    tail = waveform[..., n_full_chunks * chunk_size:]
    return torch.cat([shuffled, tail], dim=-1)


def partial_mute(waveform: torch.Tensor, sr: int, mute_fraction: float = 0.5) -> torch.Tensor:
    """Zero out a contiguous random segment covering `mute_fraction` of the audio."""
    result = waveform.clone()
    n = waveform.shape[-1]
    mute_n = max(1, int(n * mute_fraction))
    start = torch.randint(0, max(1, n - mute_n), (1,)).item()
    result[..., start:start + mute_n] = 0.0
    return result


def random_segment_mute(waveform: torch.Tensor, sr: int, num_segments: int = 5) -> torch.Tensor:
    """Randomly mute `num_segments` short segments spread over the audio."""
    result = waveform.clone()
    n = waveform.shape[-1]
    seg_len = n // (num_segments * 3)  # each muted window ≈ 1/3 of an equal-length slot
    if seg_len < 1:
        return result
    for _ in range(num_segments):
        start = torch.randint(0, max(1, n - seg_len), (1,)).item()
        result[..., start:start + seg_len] = 0.0
    return result


def pitch_shift_semitones(waveform: torch.Tensor, sr: int, semitones: float) -> torch.Tensor:
    """Crude pitch shift via resampling (preserves duration by trimming/padding).

    Shifts pitch up by `semitones` (positive = higher pitch, negative = lower).
    Uses integer approximation of the resampling ratio, so the shift may be
    slightly inexact; acceptable for a research exploration.
    """
    rate = 2.0 ** (semitones / 12.0)
    # We want the signal to sound at a different pitch: resample at orig_freq=sr*rate,
    # new_freq=sr (which effectively speeds up or slows down at sr, shifting pitch).
    orig_freq = max(1, round(sr * rate))
    resampled = torchaudio.functional.resample(waveform.float(), orig_freq=orig_freq, new_freq=sr)
    n_orig = waveform.shape[-1]
    if resampled.shape[-1] >= n_orig:
        out = resampled[..., :n_orig]
    else:
        pad = torch.zeros(*waveform.shape[:-1], n_orig - resampled.shape[-1], device=waveform.device)
        out = torch.cat([resampled, pad], dim=-1)
    return out.to(waveform.dtype).clamp(-1.0, 1.0)


def add_echo(waveform: torch.Tensor, sr: int, delay_sec: float = 0.3, decay: float = 0.5) -> torch.Tensor:
    """Add a single echo at `delay_sec` with amplitude `decay`."""
    delay_samples = int(sr * delay_sec)
    result = waveform.clone()
    if delay_samples < waveform.shape[-1]:
        result[..., delay_samples:] += decay * waveform[..., :waveform.shape[-1] - delay_samples]
    return result.clamp(-1.0, 1.0)


def speed_up(waveform: torch.Tensor, sr: int, factor: float = 1.5) -> torch.Tensor:
    """Speed up the audio by `factor`× (pitch rises + duration shortens), then loop-pad to original length."""
    orig_freq = max(1, round(sr * factor))
    resampled = torchaudio.functional.resample(waveform.float(), orig_freq=orig_freq, new_freq=sr)
    n_orig = waveform.shape[-1]
    if resampled.shape[-1] >= n_orig:
        return resampled[..., :n_orig].to(waveform.dtype).clamp(-1.0, 1.0)
    # Loop the shortened audio to fill original length
    repeats = math.ceil(n_orig / max(1, resampled.shape[-1]))
    tiled = resampled.repeat(1, repeats)[..., :n_orig]
    return tiled.to(waveform.dtype).clamp(-1.0, 1.0)


def slow_down(waveform: torch.Tensor, sr: int, factor: float = 0.67) -> torch.Tensor:
    """Slow down the audio (pitch drops + duration stretches), then trim to original length."""
    orig_freq = max(1, round(sr * factor))
    resampled = torchaudio.functional.resample(waveform.float(), orig_freq=orig_freq, new_freq=sr)
    n_orig = waveform.shape[-1]
    if resampled.shape[-1] >= n_orig:
        return resampled[..., :n_orig].to(waveform.dtype).clamp(-1.0, 1.0)
    pad = torch.zeros(*waveform.shape[:-1], n_orig - resampled.shape[-1], device=waveform.device)
    return torch.cat([resampled, pad], dim=-1).to(waveform.dtype).clamp(-1.0, 1.0)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_ALL_PERTURBATIONS: list[Perturbation] = [
    Perturbation("identity",         "No change (baseline)",                 lambda w, s: identity(w, s)),
    Perturbation("noise_snr20",      "Add Gaussian noise at SNR = 20 dB",   lambda w, s: add_gaussian_noise(w, s, 20.0)),
    Perturbation("noise_snr10",      "Add Gaussian noise at SNR = 10 dB",   lambda w, s: add_gaussian_noise(w, s, 10.0)),
    Perturbation("noise_snr5",       "Add Gaussian noise at SNR = 5 dB",    lambda w, s: add_gaussian_noise(w, s, 5.0)),
    Perturbation("noise_snr0",       "Add Gaussian noise at SNR = 0 dB",    lambda w, s: add_gaussian_noise(w, s, 0.0)),
    Perturbation("silence",          "Complete silence (zero signal)",       lambda w, s: replace_with_silence(w, s)),
    Perturbation("white_noise",      "Replace with white noise (same RMS)",  lambda w, s: replace_with_white_noise(w, s)),
    Perturbation("lowpass_1000hz",   "Low-pass filter at 1 kHz",            lambda w, s: lowpass_filter(w, s, 1000.0)),
    Perturbation("lowpass_500hz",    "Low-pass filter at 500 Hz",           lambda w, s: lowpass_filter(w, s, 500.0)),
    Perturbation("highpass_4000hz",  "High-pass filter at 4 kHz",           lambda w, s: highpass_filter(w, s, 4000.0)),
    Perturbation("bandpass_speech",  "Band-pass 300–3400 Hz (phone band)",  lambda w, s: bandpass_filter(w, s, 300.0, 3400.0)),
    Perturbation("phase_randomize",  "Randomize phase (same power spectrum)",lambda w, s: phase_randomize(w, s)),
    Perturbation("time_reverse",     "Reverse audio in time",               lambda w, s: time_reverse(w, s)),
    Perturbation("time_shuffle_1s",  "Shuffle 1-second chunks randomly",    lambda w, s: time_shuffle(w, s, 1.0)),
    Perturbation("time_shuffle_0.5s","Shuffle 0.5-second chunks randomly",  lambda w, s: time_shuffle(w, s, 0.5)),
    Perturbation("partial_mute_50",  "Mute a random 50% contiguous segment", lambda w, s: partial_mute(w, s, 0.5)),
    Perturbation("partial_mute_80",  "Mute a random 80% contiguous segment", lambda w, s: partial_mute(w, s, 0.8)),
    Perturbation("random_mute_segs", "Randomly mute 5 short segments",      lambda w, s: random_segment_mute(w, s, 5)),
    Perturbation("pitch_up_4",       "Pitch shift up 4 semitones",          lambda w, s: pitch_shift_semitones(w, s, 4.0)),
    Perturbation("pitch_up_12",      "Pitch shift up 12 semitones (octave)", lambda w, s: pitch_shift_semitones(w, s, 12.0)),
    Perturbation("pitch_down_4",     "Pitch shift down 4 semitones",        lambda w, s: pitch_shift_semitones(w, s, -4.0)),
    Perturbation("pitch_down_12",    "Pitch shift down 12 semitones (octave)",lambda w, s: pitch_shift_semitones(w, s, -12.0)),
    Perturbation("echo",             "Add echo (300 ms, 50% decay)",        lambda w, s: add_echo(w, s, 0.3, 0.5)),
    Perturbation("speed_up",         "Speed up 1.5× (with pitch rise)",     lambda w, s: speed_up(w, s, 1.5)),
    Perturbation("slow_down",        "Slow down 0.67× (with pitch drop)",   lambda w, s: slow_down(w, s, 0.67)),
]

_PERTURBATION_MAP: dict[str, Perturbation] = {p.name: p for p in _ALL_PERTURBATIONS}


def get_all_perturbations() -> list[Perturbation]:
    """Return all registered perturbations."""
    return list(_ALL_PERTURBATIONS)


def get_perturbation_by_name(name: str) -> Perturbation:
    """Retrieve a perturbation by name. Raises KeyError if not found."""
    if name not in _PERTURBATION_MAP:
        available = ", ".join(_PERTURBATION_MAP.keys())
        raise KeyError(f"Unknown perturbation '{name}'. Available: {available}")
    return _PERTURBATION_MAP[name]


def list_perturbation_names() -> list[str]:
    """Return all perturbation names in registration order."""
    return [p.name for p in _ALL_PERTURBATIONS]
