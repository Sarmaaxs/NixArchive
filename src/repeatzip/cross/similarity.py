"""Fast file-level similarity - for choosing a reference."""
from __future__ import annotations
import numpy as np

def quick_ncc(a: np.ndarray, b: np.ndarray, probe: int = 131072) -> float:
    """Zero-mean NCC on a prefix slice. Range [-1, 1]. Cheap."""
    n = min(len(a), len(b), probe)
    if n < 2048:
        return 0.0
    x = a[:n].astype(np.float64).reshape(n, -1).mean(axis=1)
    y = b[:n].astype(np.float64).reshape(n, -1).mean(axis=1)
    x -= x.mean()
    y -= y.mean()
    den = float(np.sqrt((x * x).sum() * (y * y).sum()))
    if den < 1e-8:
        return 0.0
    return float((x * y).sum() / den)
