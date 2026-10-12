"""Chroma feature extraction - 12-bin pitch class profile.

Chroma is invariant to octave and (with normalization) to key, so it
finds "same chord progression" matches that log-mel misses.
"""
from __future__ import annotations
import numpy as np

from .features import _hann, _mel_filterbank

def chroma(samples, sample_rate, n_fft=4096, hop=1024):
    x = samples.astype(np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    win = _hann(n_fft).astype(np.float32)
    n_frames = 1 + max(0, (len(x) - n_fft) // hop)
    if n_frames <= 0:
        return np.zeros((0, 12), dtype=np.float32)

    freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    with np.errstate(divide="ignore", invalid="ignore"):
        midi = 69 + 12 * np.log2(np.where(freqs > 0, freqs, 1e-9) / 440.0)
    pc = np.mod(np.round(midi).astype(int), 12)
    valid = (freqs >= 55) & (freqs <= 2000)

    out = np.zeros((n_frames, 12), dtype=np.float32)
    for i in range(n_frames):
        seg = x[i * hop: i * hop + n_fft] * win
        spec = np.abs(np.fft.rfft(seg, n=n_fft))
        for p in range(12):
            mask = valid & (pc == p)
            if mask.any():
                out[i, p] = spec[mask].sum()
    # L2 normalize each frame
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    return out / norms
