"""Sample-accurate offset refinement via cross-correlation."""
from __future__ import annotations
import numpy as np

def refine_offset(src, dst, search_radius=4096):
    if len(src) < 2 or len(dst) < 2:
        return 0

    s = src.astype(np.float64)
    d = dst.astype(np.float64)
    s -= s.mean()
    d -= d.mean()

    corr = np.correlate(s, d, mode="full")
    center = len(d) - 1
    lo = max(0, center - search_radius)
    hi = min(len(corr), center + search_radius + 1)
    idx = int(np.argmax(np.abs(corr[lo:hi]))) + lo
    return int(idx - center)
