"""Log-mel feature extraction (numpy only)."""
from __future__ import annotations
import numpy as np

def _hann(n: int) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / max(1, n - 1))

def _mel_filterbank(n_fft, sr, n_mels, fmin=20.0, fmax=None):
    if fmax is None:
        fmax = sr / 2.0

    def hz_to_mel(f): return 2595.0 * np.log10(1.0 + f / 700.0)
    def mel_to_hz(m): return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mel_pts = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_pts = mel_to_hz(mel_pts)
    n_bins = n_fft // 2 + 1
    bin_pts = np.clip(
        np.floor((n_fft + 1) * hz_pts / sr).astype(int), 0, n_bins - 1
    )

    fb = np.zeros((n_mels, n_bins), dtype=np.float32)
    for i in range(n_mels):
        l, c, r = bin_pts[i], bin_pts[i + 1], bin_pts[i + 2]
        if c <= l: c = l + 1
        if r <= c: r = c + 1
        if c > l:
            fb[i, l:c] = np.linspace(0.0, 1.0, c - l, endpoint=False)
        if r > c and c < n_bins:
            hi = min(r, n_bins)
            fb[i, c:hi] = np.linspace(1.0, 0.0, hi - c, endpoint=False)
    return fb

def log_mel(samples, sample_rate, n_fft=2048, hop=512, n_mels=64):
    x = samples.astype(np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)

    win = _hann(n_fft).astype(np.float32)
    n_frames = 1 + max(0, (len(x) - n_fft) // hop)
    if n_frames <= 0:
        return np.zeros((0, n_mels), dtype=np.float32)

    fb = _mel_filterbank(n_fft, sample_rate, n_mels)
    out = np.empty((n_frames, n_mels), dtype=np.float32)
    for i in range(n_frames):
        seg = x[i * hop: i * hop + n_fft] * win
        spec = np.abs(np.fft.rfft(seg, n=n_fft)) ** 2
        out[i] = fb @ spec.astype(np.float32)

    return np.log1p(out)
