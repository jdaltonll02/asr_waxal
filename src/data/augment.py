"""On-the-fly audio augmentation for CTC training.

Three independent augmentations applied stochastically per utterance:
  - Speed perturbation ±10%  (resampling trick, preserves pitch)
  - Additive white/pink Gaussian noise at random SNR 15–35 dB
  - Simulated room impulse response via random exponential decay convolution

All ops work on 16 kHz float32 numpy arrays and are safe to call from
multiple DataLoader workers (no shared state).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve


def _speed_perturb(audio: np.ndarray, sr: int = 16000, rng: np.random.Generator = None) -> np.ndarray:
    """Resample to simulate ±10% speed change, then resample back to sr."""
    import librosa
    if rng is None:
        rng = np.random.default_rng()
    factor = rng.uniform(0.9, 1.1)
    if abs(factor - 1.0) < 0.01:
        return audio
    new_sr = int(sr * factor)
    return librosa.resample(audio, orig_sr=new_sr, target_sr=sr)


def _add_noise(audio: np.ndarray, rng: np.random.Generator = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    snr_db = rng.uniform(15.0, 35.0)
    signal_power = np.mean(audio ** 2) + 1e-9
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.standard_normal(len(audio)).astype(np.float32) * np.sqrt(noise_power)
    return audio + noise


def _rir_convolve(audio: np.ndarray, sr: int = 16000, rng: np.random.Generator = None) -> np.ndarray:
    """Simulate a small room by convolving with a synthetic exponential-decay RIR."""
    if rng is None:
        rng = np.random.default_rng()
    rt60 = rng.uniform(0.05, 0.4)  # reverberation time in seconds
    rir_len = int(rt60 * sr)
    if rir_len < 2:
        return audio
    t = np.arange(rir_len, dtype=np.float32) / sr
    decay = np.exp(-6.9 * t / rt60)  # -60 dB at rt60
    rir = rng.standard_normal(rir_len).astype(np.float32) * decay
    rir /= np.abs(rir).max() + 1e-9
    convolved = fftconvolve(audio, rir)[:len(audio)].astype(np.float32)
    # Normalize to original RMS to avoid level drift
    orig_rms = np.sqrt(np.mean(audio ** 2) + 1e-9)
    conv_rms = np.sqrt(np.mean(convolved ** 2) + 1e-9)
    return convolved * (orig_rms / conv_rms)


def augment_audio(
    audio: np.ndarray,
    sr: int = 16000,
    p_speed: float = 0.5,
    p_noise: float = 0.5,
    p_rir: float = 0.3,
    seed: int | None = None,
) -> np.ndarray:
    """Apply stochastic augmentations to a single audio array."""
    rng = np.random.default_rng(seed)
    if rng.random() < p_speed:
        audio = _speed_perturb(audio, sr, rng)
    if rng.random() < p_noise:
        audio = _add_noise(audio, rng)
    if rng.random() < p_rir:
        audio = _rir_convolve(audio, sr, rng)
    return audio
