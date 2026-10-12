"""See package docs."""
from __future__ import annotations
from typing import List, Tuple

import numpy as np

from repeatzip.io.wav_io import read_wav
from repeatzip.models import BlockSpec
from .features import log_mel
from .aligner import refine_offset

def _exact_length(samples: np.ndarray, src: int, dst: int,
                  max_len: int) -> int:
    if max_len <= 0:
        return 0
    a = samples[src:src + max_len]
    b = samples[dst:dst + max_len]
    if a.shape != b.shape:
        return 0
    diff = (a != b)
    if diff.ndim > 1:
        diff = diff.any(axis=1)
    bad = np.flatnonzero(diff)
    return int(bad[0]) if bad.size else max_len

def _find_runs(X: np.ndarray, min_len: int, min_sep: int,
               threshold: float, max_len: int,
               verbose: bool = False) -> List[Tuple[int, int, int]]:
    """Return list of (src_hop, dst_hop, length_hops) seeds."""
    N = X.shape[0]
    runs = []
    used = np.zeros(N, dtype=bool)

    BLOCK = 128
    for cs in range(0, N, BLOCK):
        ce = min(cs + BLOCK, N)
        sim = X[cs:ce] @ X.T

        for li in range(ce - cs):
            gi = cs + li
            if used[gi]:
                continue

            lo_j = gi + min_sep
            if lo_j >= N:
                continue

            # Reject matches whose offset is exactly 1 hop (adjacent frames
            # of a stationary signal always look identical).
            js = np.arange(lo_j, N)
            offsets = js - gi
            valid = offsets > 1
            if not valid.any():
                continue
            sims = sim[li, lo_j:]
            sims = np.where(valid, sims, -np.inf)
            best_j = lo_j + int(np.argmax(sims))
            if sims[best_j - lo_j] < threshold:
                continue

            d_max = min(N - gi, N - best_j, max_len)
            if d_max < min_len:
                continue

            diag = np.einsum("ij,ij->i",
                             X[gi:gi + d_max], X[best_j:best_j + d_max])
            below = np.where(diag < threshold)[0]
            length = int(below[0]) if len(below) > 0 else d_max

            if length < min_len:
                continue

            used[gi:gi + length] = True
            runs.append((gi, best_j, length))

    return runs

def _de_nest(runs):
    runs = sorted(runs, key=lambda r: -r[2])
    kept = []
    for i, j, l in runs:
        conflict = False
        for ki, kj, kl in kept:
            if not (j + l <= kj or j >= kj + kl):
                conflict = True; break
            if not (i + l <= kj or i >= kj + kl):
                conflict = True; break
            if not (j + l <= ki or j >= ki + kl):
                conflict = True; break
        if not conflict:
            kept.append((i, j, l))
    return kept

def detect_blocks(
    wav_path: str,
    *,
    min_block_sec: float = 2.0,
    max_block_sec: float = 60.0,
    similarity_threshold: float = 0.75,
    hop: int = 1024,
    n_fft: int = 4096,
    n_mels: int = 64,
    verbose: bool = False,
) -> List[BlockSpec]:
    samples, fmt = read_wav(wav_path)
    sr = fmt.sample_rate

    mono = (samples.mean(axis=1).astype(np.float32)
            if samples.ndim > 1 else samples.astype(np.float32))

    feats = log_mel(mono, sr, n_fft=n_fft, hop=hop, n_mels=n_mels)
    N = feats.shape[0]
    if N < 8:
        return []

    # Per-bin z-score, then L2-normalize each feature row.
    feats = feats - feats.mean(axis=0, keepdims=True)
    feats = feats / (feats.std(axis=0, keepdims=True) + 1e-8)
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    X = (feats / norms).astype(np.float32)

    min_len_hops = max(2, int(min_block_sec * sr / hop))
    max_len_hops = int(max_block_sec * sr / hop)
    # Require separation of at least the min block length, plus 4 hops,
    # so we never match a frame against its own neighborhood.
    min_sep_hops = min_len_hops + 4

    if verbose:
        print(f"[detect] {N} hops ({N * hop / sr:.1f}s), "
              f"min_len={min_len_hops} hops, "
              f"min_sep={min_sep_hops} hops, "
              f"threshold={similarity_threshold}")

    runs = _find_runs(X, min_len_hops, min_sep_hops,
                      similarity_threshold, max_len_hops,
                      verbose=verbose)

    if verbose:
        print(f"[detect] {len(runs)} raw run(s)")

    if not runs:
        return []

    kept = _de_nest(runs)

    if verbose:
        print(f"[detect] {len(kept)} non-overlapping run(s)")

    specs: List[BlockSpec] = []
    for i, j, l in kept:
        src_s = i * hop
        dst_s = j * hop
        length_s = l * hop

        # Refine dst offset with cross-correlation on the raw waveform.
        win = min(length_s, 16384)
        if src_s + win <= len(mono) and dst_s + win <= len(mono):
            lag = refine_offset(mono[src_s:src_s + win],
                                mono[dst_s:dst_s + win],
                                search_radius=hop * 2)
            dst_s = max(0, dst_s + lag)

        if dst_s + length_s > len(mono):
            length_s = len(mono) - dst_s
        if length_s < min_block_sec * sr:
            continue

        specs.append(BlockSpec(
            src_start_s=src_s / sr,
            src_end_s=(src_s + length_s) / sr,
            dst_start_s=dst_s / sr,
        ))

    specs.sort(key=lambda b: b.dst_start_s)
    final = []
    prev_end = -1.0
    for b in specs:
        if b.dst_start_s >= prev_end - 1e-9:
            final.append(b)
            prev_end = b.dst_start_s + (b.src_end_s - b.src_start_s)

    return final
