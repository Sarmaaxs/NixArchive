"""Find near-duplicate WAV recordings in a folder tree.

Scans a directory for WAV files, extracts a compact spectral
signature from a middle segment of each, and groups files whose
signatures cluster tightly. Candidate groups are confirmed by
waveform alignment on those same segments. The output is a report
of where compression would help; nothing is encoded.

    nixarchive scan ~/Music
    nixarchive scan ~/Music --csv results.csv --threshold 0.9
"""
from __future__ import annotations
import os
import sys
import time
import wave
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from repeatzip.errors import RepeatzipError
from repeatzip.analysis.features import _mel_filterbank, _hann
from repeatzip.cross.align import match_files


PROBE_DURATION_S = 30.0
PROBE_FFT = 2048
PROBE_MELS = 16
PROBE_FRAMES = 8
SIG_DIM = PROBE_MELS * PROBE_FRAMES

DEFAULT_SIM_THRESHOLD = 0.85
DEFAULT_NCC_THRESHOLD = 0.70
DEFAULT_MAX_FILES = 5000
DEFAULT_MIN_DURATION_S = 2.0
SAME_RATE_TOLERANCE = 1.0
MAX_DURATION_RATIO = 4.0


@dataclass
class ScannedFile:
    path: str
    size_bytes: int
    sample_rate: int
    channels: int
    sampwidth: int
    frames: int
    duration_s: float
    signature: np.ndarray = field(
        default_factory=lambda: np.zeros(SIG_DIM, dtype=np.float32))


@dataclass
class Edge:
    i: int
    j: int
    ncc: float
    lag: int
    gain: float


@dataclass
class Cluster:
    members: List[int]
    primary: int
    edges: List[Edge]
    total_bytes: int


@dataclass
class ScanResult:
    root: str
    files: List[ScannedFile]
    clusters: List[Cluster]
    elapsed_s: float
    n_candidates: int
    n_confirmed: int


# ---------------------------------------------------------------------------
# WAV reading (partial)
# ---------------------------------------------------------------------------

def _read_meta(path: str):
    try:
        with wave.open(path, "rb") as wf:
            if wf.getcomptype() != "NONE":
                return None
            return {
                "sr": wf.getframerate(),
                "ch": wf.getnchannels(),
                "sw": wf.getsampwidth(),
                "n": wf.getnframes(),
            }
    except Exception:
        return None


def _read_probe(path: str, duration_s: float = PROBE_DURATION_S):
    """Return (samples, sr, ch, sw) from the middle of a WAV, or None."""
    try:
        with wave.open(path, "rb") as wf:
            if wf.getcomptype() != "NONE":
                return None
            sw = wf.getsampwidth()
            if sw not in (2, 3):
                return None
            sr = wf.getframerate()
            ch = wf.getnchannels()
            n = wf.getnframes()
            want = int(duration_s * sr)
            if want >= n:
                start = 0
                count = n
            else:
                start = (n - want) // 2
                count = want
            wf.setpos(start)
            raw = wf.readframes(count)
    except Exception:
        return None

    if sw == 2:
        arr = np.frombuffer(raw, dtype="<i2").astype(np.int32)
    else:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        as_u32 = (b[:, 0].astype(np.uint32)
                  | (b[:, 1].astype(np.uint32) << 8)
                  | (b[:, 2].astype(np.uint32) << 16))
        as_i32 = as_u32.astype(np.int32)
        sign = (as_u32 & 0x800000).astype(bool)
        as_i32[sign] -= 0x1000000
        arr = as_i32

    samples = arr.reshape(-1, ch)
    return samples, sr, ch, sw


def _fingerprint(samples: np.ndarray, sr: int) -> np.ndarray:
    """Fixed 128-d signature: 16 mel bands across 8 temporal frames."""
    x = samples.astype(np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    n = len(x)
    if n < PROBE_FRAMES * PROBE_FFT:
        return np.zeros(SIG_DIM, dtype=np.float32)

    fb = _mel_filterbank(PROBE_FFT, sr, PROBE_MELS)
    win = _hann(PROBE_FFT).astype(np.float32)
    hop = PROBE_FFT // 2

    edges = np.linspace(0, n, PROBE_FRAMES + 1).astype(int)
    sig = np.zeros((PROBE_FRAMES, PROBE_MELS), dtype=np.float32)

    for f in range(PROBE_FRAMES):
        seg = x[edges[f]:edges[f + 1]]
        if len(seg) < PROBE_FFT:
            seg = np.pad(seg, (0, PROBE_FFT - len(seg)))
        n_w = max(1, (len(seg) - PROBE_FFT) // hop + 1)
        acc = np.zeros(PROBE_MELS, dtype=np.float32)
        for w in range(n_w):
            frame = seg[w * hop:w * hop + PROBE_FFT]
            if len(frame) < PROBE_FFT:
                frame = np.pad(frame, (0, PROBE_FFT - len(frame)))
            spec = np.abs(np.fft.rfft(frame * win, n=PROBE_FFT)) ** 2
            acc += fb @ spec.astype(np.float32)
        sig[f] = np.log1p(acc / n_w)

    flat = sig.ravel()
    nrm = float(np.linalg.norm(flat))
    return flat / nrm if nrm > 0 else flat


def _find_wavs(root: str) -> List[str]:
    out: List[str] = []
    skip = {"node_modules", "venv", ".venv", "build", "dist",
            "__pycache__", ".git", ".pytest_cache"}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if not d.startswith(".") and d not in skip]
        for fn in filenames:
            if fn.lower().endswith(".wav"):
                out.append(os.path.join(dirpath, fn))
    out.sort()
    return out


# ---------------------------------------------------------------------------
# scanner
# ---------------------------------------------------------------------------

def scan_folder(root: str, *,
                sim_threshold: float = DEFAULT_SIM_THRESHOLD,
                ncc_threshold: float = DEFAULT_NCC_THRESHOLD,
                max_files: int = DEFAULT_MAX_FILES,
                min_duration_s: float = DEFAULT_MIN_DURATION_S,
                duration_s: float = PROBE_DURATION_S,
                verbose: bool = False) -> ScanResult:
    t0 = time.time()
    paths = _find_wavs(root)
    if not paths:
        raise RepeatzipError(f"no WAV files in {root}")
    if len(paths) > max_files:
        raise RepeatzipError(
            f"found {len(paths)} WAV files (limit {max_files}); "
            f"raise --max-files or narrow the folder")

    if verbose:
        print(f"[scan] {len(paths)} WAV files", file=sys.stderr)

    # ---- pass 1: fingerprint ----
    files: List[ScannedFile] = []
    for idx, p in enumerate(paths):
        if verbose and idx and idx % 200 == 0:
            print(f"[scan] fingerprint {idx}/{len(paths)}",
                  file=sys.stderr)
        meta = _read_meta(p)
        if meta is None:
            continue
        if meta["sr"] == 0 or meta["sw"] not in (2, 3):
            continue
        dur = meta["n"] / meta["sr"]
        if dur < min_duration_s:
            continue
        probe = _read_probe(p, duration_s)
        if probe is None:
            continue
        samples, sr, ch, sw = probe
        sig = _fingerprint(samples, sr)
        files.append(ScannedFile(
            path=p, size_bytes=os.path.getsize(p),
            sample_rate=sr, channels=ch, sampwidth=sw,
            frames=meta["n"], duration_s=dur, signature=sig,
        ))

    if verbose:
        print(f"[scan] {len(files)} files fingerprinted",
              file=sys.stderr)

    if len(files) < 2:
        return ScanResult(root=root, files=files, clusters=[],
                          elapsed_s=time.time() - t0,
                          n_candidates=0, n_confirmed=0)

    # ---- pass 2: candidate pairs by signature similarity ----
    sigs = np.stack([f.signature for f in files])
    n = len(files)
    candidates: List[Tuple[int, int, float]] = []
    for i in range(n):
        sims = sigs[i + 1:] @ sigs[i]
        idx = np.where(sims >= sim_threshold)[0]
        for rel in idx:
            j = i + 1 + int(rel)
            if abs(files[i].sample_rate - files[j].sample_rate) > SAME_RATE_TOLERANCE:
                continue
            hi = max(files[i].duration_s, files[j].duration_s)
            lo = max(1e-6, min(files[i].duration_s, files[j].duration_s))
            if hi / lo > MAX_DURATION_RATIO:
                continue
            candidates.append((i, j, float(sims[rel])))

    if verbose:
        print(f"[scan] {len(candidates)} candidate pairs",
              file=sys.stderr)

    # ---- pass 3: confirm by waveform alignment ----
    probe_cache: dict = {}

    def _probe(idx: int):
        if idx not in probe_cache:
            r = _read_probe(files[idx].path, duration_s)
            probe_cache[idx] = r[0] if r else None
            if len(probe_cache) > 200:
                probe_cache.pop(next(iter(probe_cache)))
        return probe_cache[idx]

    edges: List[Edge] = []
    for i, j, _ in candidates:
        pi = _probe(i)
        pj = _probe(j)
        if pi is None or pj is None:
            continue
        try:
            lag, gain, ncc = match_files(pi, pj)
        except Exception:
            continue
        if ncc >= ncc_threshold:
            edges.append(Edge(i=i, j=j, ncc=float(ncc),
                              lag=int(lag), gain=float(gain)))

    if verbose:
        print(f"[scan] {len(edges)} pairs confirmed", file=sys.stderr)

    # ---- pass 4: union-find clusters ----
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for e in edges:
        union(e.i, e.j)

    groups: dict = {}
    for idx in range(n):
        groups.setdefault(find(idx), []).append(idx)

    clusters: List[Cluster] = []
    for members in groups.values():
        if len(members) < 2:
            continue
        primary = max(members, key=lambda k: files[k].size_bytes)
        mset = set(members)
        cedges = [e for e in edges if e.i in mset and e.j in mset]
        total = sum(files[k].size_bytes for k in members)
        clusters.append(Cluster(members=members, primary=primary,
                                edges=cedges, total_bytes=total))

    clusters.sort(key=lambda c: -c.total_bytes)

    return ScanResult(
        root=root, files=files, clusters=clusters,
        elapsed_s=time.time() - t0,
        n_candidates=len(candidates), n_confirmed=len(edges),
    )


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def _human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1024 or u == units[-1]:
            return f"{int(x)} B" if u == "B" else f"{x:.1f} {u}"
        x /= 1024
    return f"{x:.1f} TB"


def _ncc_to_primary(c: Cluster, k: int) -> float:
    best = 0.0
    for e in c.edges:
        if (e.i == k and e.j == c.primary) or \
           (e.j == k and e.i == c.primary):
            best = max(best, e.ncc)
    return best


def print_report(result: ScanResult) -> None:
    total_bytes = sum(f.size_bytes for f in result.files)
    n_files = len(result.files)
    cluster_bytes = sum(c.total_bytes for c in result.clusters)
    cluster_files = sum(len(c.members) for c in result.clusters)

    print()
    print("NixScan")
    print("=" * 60)
    print(f"Scanned:  {result.root}")
    print(f"Files:    {n_files} WAV files, {_human_bytes(total_bytes)}")
    print(f"Duration: {result.elapsed_s:.1f}s")
    print()

    if not result.clusters:
        print("No near-duplicate clusters found.")
        return

    print(f"Candidates tested:  {result.n_candidates}")
    print(f"Pairs confirmed:    {result.n_confirmed}")
    print(f"Clusters found:     {len(result.clusters)}")
    print(f"Files in clusters:  {cluster_files} / {n_files}")
    print(f"Bytes in clusters:  {_human_bytes(cluster_bytes)} / "
          f"{_human_bytes(total_bytes)} "
          f"({100 * cluster_bytes / max(1, total_bytes):.1f}%)")
    print()

    for c_idx, c in enumerate(result.clusters, 1):
        print(f"Cluster {c_idx} -- {len(c.members)} files, "
              f"{_human_bytes(c.total_bytes)}")
        prim = result.files[c.primary]
        print(f"  Primary: {os.path.relpath(prim.path, result.root)}")
        others = [(k, _ncc_to_primary(c, k))
                  for k in c.members if k != c.primary]
        others.sort(key=lambda t: -t[1])
        for k, ncc in others:
            rel = os.path.relpath(result.files[k].path, result.root)
            short = rel if len(rel) <= 44 else "..." + rel[-41:]
            print(f"    - {short:<44} NCC {ncc:.3f}  "
                  f"{_human_bytes(result.files[k].size_bytes)}")
        print()

    print("If these clusters archive like typical near-duplicates,")
    print("expect roughly 50-70% reduction. Run:")
    print("  nixarchive archive <folder> -o out.rpzarc -v")
    print("on any cluster's folder to see the real number.")
    print()


def write_csv(result: ScanResult, path: str) -> None:
    with open(path, "w") as f:
        f.write("cluster,primary,file,size_bytes,ncc_to_primary\n")
        for ci, c in enumerate(result.clusters, 1):
            prim_rel = os.path.relpath(
                result.files[c.primary].path, result.root)
            for k in c.members:
                if k == c.primary:
                    continue
                rel = os.path.relpath(result.files[k].path, result.root)
                f.write(f"{ci},{prim_rel},{rel},"
                        f"{result.files[k].size_bytes},"
                        f"{_ncc_to_primary(c, k):.4f}\n")
