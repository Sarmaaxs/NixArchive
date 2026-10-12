"""Bit-width aware modular arithmetic.

Residuals and reconstructions are wrapped to the source's sample width
so they always fit in the target FLAC subtype (PCM_16 or PCM_24)."""
from __future__ import annotations
import numpy as np

def _bits(sampwidth: int) -> int:
    return sampwidth * 8

def wrap_signed(values, sampwidth: int) -> np.ndarray:
    bits = _bits(sampwidth)
    mod = 1 << bits
    half = 1 << (bits - 1)
    v = values.astype(np.int64) & (mod - 1)
    v = np.where(v >= half, v - mod, v)
    return v.astype(np.int32)

def residual(target, reference, sampwidth: int) -> np.ndarray:
    diff = target.astype(np.int64) - reference.astype(np.int64)
    return wrap_signed(diff, sampwidth)

def reconstruct(reference, residual_arr, sampwidth: int) -> np.ndarray:
    s = reference.astype(np.int64) + residual_arr.astype(np.int64)
    return wrap_signed(s, sampwidth)
