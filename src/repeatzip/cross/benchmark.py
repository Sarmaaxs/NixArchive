"""Benchmark a folder against itself.

Produces three numbers every buyer asks for:

    original       total bytes of the input WAVs
    flac_baseline  total bytes if each WAV were compressed independently
    bundle         total bytes of the .rpzarc

The delta advantage is bundle vs flac_baseline, not bundle vs original.
"""
from __future__ import annotations
import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import List

import numpy as np

from repeatzip.io.flac_io import HAVE_FLAC, encode_flac
from repeatzip.io.wav_io import read_wav
from .archive import archive, extract, _read_bundle

@dataclass
class FileRow:
    name: str
    original_bytes: int
    flac_bytes: int
    bundle_bytes: int
    mode: str
    reference: str
    lag: int
    gain: float
    ncc: float
    ratio_vs_flac: float
    roundtrip_ok: bool

@dataclass
class BenchResult:
    rows: List[FileRow] = field(default_factory=list)
    total_original: int = 0
    total_flac: int = 0
    total_bundle: int = 0
    elapsed_archive: float = 0.0
    elapsed_extract: float = 0.0
    all_roundtrip_ok: bool = True

def benchmark_folder(input_dir: str, *, ref_ncc: float = 0.5,
                     workers: int = 0, verbose: bool = True) -> BenchResult:
    if not HAVE_FLAC:
        raise RuntimeError("soundfile required")

    names = sorted(n for n in os.listdir(input_dir)
                   if n.lower().endswith(".wav"))
    if not names:
        raise RuntimeError(f"no .wav files in {input_dir}")

    # ---- Stage 1: measure plain FLAC baseline ----
    flac_sizes = {}
    for n in names:
        arr, fmt = read_wav(os.path.join(input_dir, n))
        flac_sizes[n] = len(encode_flac(arr, fmt))

    # ---- Stage 2: build bundle ----
    tmpdir = tempfile.mkdtemp(prefix="rpz_bench_")
    bundle_path = os.path.join(tmpdir, "bench.rpzarc")
    t0 = time.time()
    archive(input_dir, bundle_path, ref_ncc=ref_ncc,
            workers=workers, verbose=False)
    elapsed_archive = time.time() - t0

    # ---- Stage 3: extract to verify ----
    out_dir = os.path.join(tmpdir, "out")
    t0 = time.time()
    extract(bundle_path, out_dir, verbose=False)
    elapsed_extract = time.time() - t0

    # ---- Stage 4: compare ----
    meta, payload = _read_bundle(bundle_path)
    by_name = {e["name"]: e for e in meta["entries"]}

    result = BenchResult()
    result.elapsed_archive = elapsed_archive
    result.elapsed_extract = elapsed_extract

    for n in names:
        orig_path = os.path.join(input_dir, n)
        out_path = os.path.join(out_dir, n)
        orig_bytes = os.path.getsize(orig_path)

        if not os.path.exists(out_path):
            result.all_roundtrip_ok = False
            ok = False
        else:
            with open(orig_path, "rb") as f:
                a = f.read()
            with open(out_path, "rb") as f:
                b = f.read()
            ok = a == b
            if not ok:
                result.all_roundtrip_ok = False

        e = by_name[n]
        mode = e.get("kind", "solo")
        ref = e.get("reference") or "-"
        lag = int(e.get("lag", 0))
        gain = float(e.get("gain", 1.0))

        result.rows.append(FileRow(
            name=n,
            original_bytes=orig_bytes,
            flac_bytes=flac_sizes[n],
            bundle_bytes=int(e["payload_len"]),
            mode=mode,
            reference=ref,
            lag=lag,
            gain=gain,
            ncc=0.0,
            ratio_vs_flac=flac_sizes[n] / max(1, e["payload_len"]),
            roundtrip_ok=ok,
        ))
        result.total_original += orig_bytes
        result.total_flac += flac_sizes[n]
        result.total_bundle += int(e["payload_len"])

    return result

def print_report(result: BenchResult, input_dir: str) -> None:
    print()
    print(f"folder: {input_dir}")
    print(f"files : {len(result.rows)}")
    print()
    print(f"{'file':<32} {'original':>13} {'flac':>12} "
          f"{'bundle':>12} {'mode':>7} {'xFLAC':>7} {'ok':>3}")
    print("-" * 92)
    for r in result.rows:
        name = r.name if len(r.name) <= 30 else r.name[:27] + "..."
        print(f"{name:<32} {r.original_bytes:>13,} {r.flac_bytes:>12,} "
              f"{r.bundle_bytes:>12,} {r.mode:>7} "
              f"{r.ratio_vs_flac:>6.2f}x {'Y' if r.roundtrip_ok else 'N':>3}")
    print("-" * 92)
    print(f"{'TOTAL':<32} {result.total_original:>13,} "
          f"{result.total_flac:>12,} {result.total_bundle:>12,}")
    print()
    pct_vs_orig = 100.0 * (1 - result.total_bundle / result.total_original)
    pct_vs_flac = 100.0 * (1 - result.total_bundle / result.total_flac)
    print(f"vs original WAVs : {pct_vs_orig:6.2f}% smaller")
    print(f"vs independent FLAC : {pct_vs_flac:6.2f}% smaller  "
          f"(this is the delta advantage)")
    print()
    print(f"archive time : {result.elapsed_archive:.2f} s")
    print(f"extract time : {result.elapsed_extract:.2f} s")
    print(f"round-trip   : "
          f"{'ALL OK' if result.all_roundtrip_ok else 'FAILURES PRESENT'}")
    print()

    n_delta = sum(1 for r in result.rows if r.mode == "delta")
    n_solo = sum(1 for r in result.rows if r.mode == "solo")
    print(f"modes: {n_delta} delta, {n_solo} solo")

def write_csv(result: BenchResult, path: str) -> None:
    with open(path, "w") as f:
        f.write("name,original_bytes,flac_bytes,bundle_bytes,mode,"
                "reference,lag,gain,ratio_vs_flac,roundtrip_ok\n")
        for r in result.rows:
            f.write(f"{r.name},{r.original_bytes},{r.flac_bytes},"
                    f"{r.bundle_bytes},{r.mode},{r.reference},{r.lag},"
                    f"{r.gain:.6f},{r.ratio_vs_flac:.4f},"
                    f"{int(r.roundtrip_ok)}\n")
