"""Two-stage alignment between two PCM arrays.

Lag convention: tgt[lag + i] ~= gain * ref[i]
    lag > 0 : tgt has extra content at the front
    lag < 0 : ref has extra content at the front
"""
from __future__ import annotations
import numpy as np

def _mono_float(arr):
    x = arr.astype(np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x

def _lowpass_decimate(x, factor):
    if factor <= 1:
        return x.copy()
    csum = np.concatenate([[0.0], np.cumsum(x)])
    y = (csum[factor:] - csum[:-factor]) / factor
    return y[::factor]

def _fft_xcorr_lag(a, b):
    """Return L such that b[L + i] ~= a[i].

    The FFT gives a circular cross-correlation. The linear peak lives
    at index p in [0, len(a)+len(b)-2]; indices past that are padding.
    The correct unwrap threshold is len(a), not nfft // 2, because the
    valid negative-lag region is bounded by len(a) and the valid
    positive-lag region wraps to the top of the array.
    """
    n = len(a) + len(b) - 1
    nfft = 1 << max(1, (n - 1).bit_length())
    A = np.fft.rfft(a, nfft)
    B = np.fft.rfft(b, nfft)
    z = np.fft.irfft(A * np.conj(B), nfft)
    peak = int(np.argmax(z))
    if peak >= len(a):
        peak -= nfft
    return -peak

def _slices(lag, n_ref, n_tgt):
    if lag >= 0:
        return lag, 0, min(n_ref, n_tgt - lag)
    else:
        return 0, -lag, min(n_ref + lag, n_tgt)

def _ncc_slice(a, b):
    n = len(a)
    if n < 256:
        return -1.0
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if den < 1e-8:
        return -1.0
    return float((a * b).sum() / den)

def _ncc_full(ref, tgt, lag):
    tgt_s, ref_s, n = _slices(lag, len(ref), len(tgt))
    if n < 256:
        return -1.0
    return _ncc_slice(ref[ref_s:ref_s + n], tgt[tgt_s:tgt_s + n])

def _gain_on_aligned(ref, tgt, lag):
    tgt_s, ref_s, n = _slices(lag, len(ref), len(tgt))
    if n < 256:
        return 1.0
    a = ref[ref_s:ref_s + n]
    b = tgt[tgt_s:tgt_s + n]
    a = a - a.mean()
    b = b - b.mean()
    den = float((a * a).sum())
    if den < 1e-8:
        return 1.0
    return float((a * b).sum() / den)

def match_files(ref, tgt, *, coarse=8, probe_len=65536):
    """Return (lag, gain, ncc). ncc is full-length at the chosen lag."""
    r = _mono_float(ref)
    t = _mono_float(tgt)

    if len(r) < 4096 or len(t) < 4096:
        return 0, 1.0, _ncc_full(r, t, 0)

    r_ds = _lowpass_decimate(r, coarse)
    t_ds = _lowpass_decimate(t, coarse)
    if len(r_ds) < 32 or len(t_ds) < 32:
        return 0, 1.0, _ncc_full(r, t, 0)

    coarse_lag = coarse * _fft_xcorr_lag(r_ds, t_ds)

    # Probe is taken from the middle of the OVERLAP implied by coarse_lag,
    # not the middle of the reference. Without this, an excerpt shorter
    # than the reference can never find its own alignment.
    tgt_s, ref_s, ov = _slices(coarse_lag, len(r), len(t))
    if ov < 4096:
        return int(coarse_lag), _gain_on_aligned(r, t, coarse_lag), \
               _ncc_full(r, t, coarse_lag)

    pl = min(probe_len, ov)
    mid = (ov - pl) // 2
    p_ref = ref_s + mid
    ref_probe = r[p_ref:p_ref + pl]

    search_half = max(coarse * 8, 128)
    lo = max(-p_ref, coarse_lag - search_half)
    hi = min(len(t) - p_ref - pl, coarse_lag + search_half)

    best_lag = coarse_lag
    best_score = -2.0
    for lag in range(lo, hi + 1):
        t_start = p_ref + lag
        score = _ncc_slice(ref_probe, t[t_start:t_start + pl])
        if score > best_score:
            best_score = score
            best_lag = lag

    full_ncc = _ncc_full(r, t, best_lag)
    gain = _gain_on_aligned(r, t, best_lag)
    return int(best_lag), float(gain), float(full_ncc)
