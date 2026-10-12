from __future__ import annotations
import os
import multiprocessing as mp
from typing import List

import numpy as np

def _worker_decode_one(args):
    name, path = args
    from repeatzip.io.wav_io import read_wav
    from repeatzip.io.wav_header import read_wav_parts
    samples, fmt = read_wav(path)
    prefix, trailer = read_wav_parts(path)
    return name, samples, fmt, prefix, trailer

def load_wavs_parallel(input_dir: str, names: List[str], workers: int = 0):
    if workers <= 0:
        workers = max(1, min(os.cpu_count() or 1, len(names)))
    if workers == 1 or len(names) < 4:
        out = {}
        for n in names:
            out[n] = _worker_decode_one((n, os.path.join(input_dir, n)))
        return out
    tasks = [(n, os.path.join(input_dir, n)) for n in names]
    with mp.Pool(processes=workers) as pool:
        results = pool.map(_worker_decode_one, tasks)
    return {r[0]: r for r in results}

def default_workers() -> int:
    return max(1, (os.cpu_count() or 1))
