"""Sample-accurate waveform verification.

For a candidate (src, dst, length), compute normalized cross-correlation
at the exact sample resolution. Peak NCC near 1.0 means the waveforms
are near-identical (delta-encodable). Peak NCC near 0 means they only
share a spectral envelope, not the actual signal.
"""
from __future__ import annotations
import numpy as np

def ncc_peak(src: np.ndarray, dst: np.ndarray,
             max_lag: int = 4096) -> tuple[float, int]:
    """Return (peak_ncc, lag). Positive lag = dst is delayed copy of src."""
    if len(src) < 32 or len(dst) < 32:
        return 0.0, 0

    s = src.astype(np.float64)
    d = dst.astype(np.float64)
    s -= s.mean()
    d -= d.mean()

    s_energy = np.sqrt((s * s).sum())
    d_energy = np.sqrt((d * d).sum())
    if s_energy < 1e-8 or d_energy < 1e-8:
        return 0.0, 0

    # Limit search to a window to keep it cheap.
    lo = max(0, -(len(d) - 1))
    hi = min(len(s) - 1, max_lag)
    lo = max(lo, -max_lag)

    best_r = -1.0
    best_lag = 0
    step = 1
    # Coarse pass first
    for lag in range(lo, hi + 1, max(1, step)):
        if lag >= 0:
            a = s[lag:]
            b = d[:len(a)]
        else:
            b = d[-lag:]
            a = s[:len(b)]
        n = min(len(a), len(b), 32768)
        if n < 64:
            continue
        a = a[:n]
        b = b[:n]
        num = (a * b).sum()
        den = np.sqrt((a * a).sum() * (b * b).sum())
        if den < 1e-8:
            continue
        r = num / den
        if r > best_r:
            best_r = r
            best_lag = lag
    return float(best_r), int(best_lag)

def waveform_score(samples: np.ndarray, src: int, dst: int,
                   length: int, max_lag: int = 2048) -> float:
    """Peak NCC over a bounded lag search around the given alignment."""
    if length <= 0:
        return 0.0
    # Sample a middle slice for speed (up to 64k frames).
    probe_len = min(length, 65536)
    off = (length - probe_len) // 2
    a = samples[src + off: src + off + probe_len]
    b = samples[dst + off: dst + off + probe_len]
    if a.ndim > 1:
        a = a.mean(axis=1)
    if b.ndim > 1:
        b = b.mean(axis=1)
    r, _ = ncc_peak(a.astype(np.float32), b.astype(np.float32),
                    max_lag=max_lag)
    return r
