from __future__ import annotations
import hashlib
import zlib
from typing import List, Optional

import numpy as np

from repeatzip.errors import RpzFormatError
from repeatzip.io.wav_io import read_wav, sec_to_frame
from repeatzip.io import rpz_format, cdc
from repeatzip.io.rpz_format import CODEC_RAW, CODEC_FLAC
from repeatzip.io.flac_io import HAVE_FLAC, encode_flac
from repeatzip.io.wav_header import read_wav_parts
from repeatzip.models import BlockSpec, Pointer
from repeatzip.codec import arith
from repeatzip.analysis.waveform_match import waveform_score

def _longest_exact_run(samples, src, dst, length, min_length):
    if length <= 0:
        return None
    a = samples[src:src+length]
    b = samples[dst:dst+length]
    if a.shape != b.shape:
        return None
    diff = (a != b)
    if diff.ndim > 1:
        diff = diff.any(axis=1)
    if not diff.any():
        return (0, 0, length)
    best_start, best_len, cur_start, cur_len = -1, 0, 0, 0
    for k, d in enumerate(diff):
        if not d:
            if cur_len == 0:
                cur_start = k
            cur_len += 1
            if cur_len > best_len:
                best_len, best_start = cur_len, cur_start
        else:
            cur_len = 0
    if best_len < min_length:
        return None
    return (best_start, best_start, best_len)

def _flac_size(arr_2d, fmt):
    """Measure actual FLAC size for a block. Returns None if unavailable."""
    if not HAVE_FLAC:
        return None
    try:
        return len(encode_flac(arr_2d.astype(np.int16, copy=False), fmt))
    except Exception:
        return None

def _mean_abs(arr) -> float:
    if arr.size == 0:
        return 0.0
    return float(np.abs(arr.astype(np.int64)).mean())

def encode(input_wav, output_rpz, blocks=None, *,
           auto_detect=False, codec="auto",
           allow_delta=True, delta_min_gain=0.10,
           waveform_min_ncc=0.90,
           detector_threshold=0.75, verbose=False):
    samples, fmt = read_wav(input_wav)
    total_frames = samples.shape[0]
    ch = fmt.channels
    sr = fmt.sample_rate

    if auto_detect and not blocks:
        from repeatzip.analysis import detect_blocks
        blocks = detect_blocks(input_wav,
                               similarity_threshold=detector_threshold,
                               verbose=verbose)
        if verbose:
            print(f"[encode] auto-detected {len(blocks)} block(s)")
    blocks = blocks or []
    decisions = []   # (mode, src, dst, length, residual_or_None)
    claimed = []     # (dst_start, dst_end) in original coords

    for b in sorted(blocks, key=lambda x: x.dst_start_s):
        s = sec_to_frame(b.src_start_s, fmt)
        e = sec_to_frame(b.src_end_s, fmt)
        d = sec_to_frame(b.dst_start_s, fmt)
        length = e - s

        if length <= 0 or d < 0 or d + length > total_frames:
            if verbose: print(f"[encode] skip out-of-range: {b}")
            continue

        if any(not (d + length <= cs or d >= ce) for cs, ce in claimed):
            if verbose: print(f"[encode] skip dst-overlaps-claim: {b}")
            continue
        min_run = int(0.25 * sr)
        run = _longest_exact_run(samples, s, d, length, min_run)
        if run is not None:
            s2, d2, L = s + run[0], d + run[1], run[2]
            if not any(not (d2 + L <= cs or d2 >= ce) for cs, ce in claimed):
                decisions.append(("copy", s2, d2, L, None))
                claimed.append((d2, d2 + L))
                if verbose:
                    print(f"[encode] COPY  src={s2/sr:.3f}s dst={d2/sr:.3f}s "
                          f"len={L/sr:.3f}s")
                continue
        if allow_delta:
            if any(not (s + length <= cs or s >= ce) for cs, ce in claimed):
                if verbose: print(f"[encode] skip delta src-overlaps: {b}")
                continue

            # Waveform-level gate first: don't even try delta on audio that
            # is only spectrally similar. NCC near 1.0 = same signal.
            ncc = waveform_score(samples, s, d, length)
            if ncc < waveform_min_ncc:
                if verbose:
                    print(f"[encode] DELTA rejected: NCC={ncc:.3f} "
                          f"< {waveform_min_ncc}")
                continue

            src_blk = samples[s:s+length]
            dst_blk = samples[d:d+length]
            residual = arith.residual(dst_blk, src_blk, fmt.sampwidth)

            sz_dst = _flac_size(dst_blk, fmt)
            sz_res = _flac_size(residual, fmt)
            if sz_dst and sz_res:
                gain = 1.0 - (sz_res / sz_dst)
            else:
                sig_mad = _mean_abs(dst_blk)
                res_mad = _mean_abs(residual)
                gain = 1.0 - res_mad / max(sig_mad, 1.0)

            if gain >= delta_min_gain:
                decisions.append(("delta", s, d, length, residual))
                claimed.append((d, d + length))
                if verbose:
                    print(f"[encode] DELTA src={s/sr:.3f}s dst={d/sr:.3f}s "
                          f"len={length/sr:.3f}s NCC={ncc:.3f} "
                          f"gain={gain:.3f}")
                continue
            elif verbose:
                print(f"[encode] DELTA rejected: NCC={ncc:.3f} "
                      f"gain={gain:.3f} < {delta_min_gain}")

        if verbose:
            print(f"[encode] skip (no copy run, delta not worth): {b}")
    copy_drops = sorted([(d, d + L) for (m, s, d, L, _) in decisions
                         if m == "copy"])

    regions = sorted([(d, d + L, m, r) for (m, s, d, L, r) in decisions],
                     key=lambda x: x[0])

    base_chunks = []
    orig_pos = 0
    for rs, re_, kind, payload in regions:
        if rs > orig_pos:
            base_chunks.append(samples[orig_pos:rs])
        if kind == "delta":
            base_chunks.append(payload)
        orig_pos = re_
    if orig_pos < total_frames:
        base_chunks.append(samples[orig_pos:total_frames])

    base = np.concatenate(base_chunks, axis=0) if base_chunks else samples[:0]
    base_frames = base.shape[0]
    def orig_to_base(f: int) -> int:
        dropped = 0
        for ds, de in copy_drops:
            if ds >= f:
                break
            dropped += min(de, f) - ds
        return f - dropped

    pointers = []
    for mode, s, d, L, _ in decisions:
        pointers.append(Pointer(
            dst_frame=orig_to_base(d),
            src_frame=orig_to_base(s),
            length=L,
            mode=mode,
        ))
    base_bytes = base.astype(np.int32, copy=False).tobytes()
    unique, manifest = cdc.compress(base_bytes)
    use_cdc = len(unique) < len(base_bytes) * 0.995
    payload_src = unique if use_cdc else base_bytes
    if not use_cdc:
        manifest = None

    if verbose:
        n_copy = sum(1 for m, *_ in decisions if m == "copy")
        n_delta = sum(1 for m, *_ in decisions if m == "delta")
        print(f"[encode] blocks: {n_copy} COPY, {n_delta} DELTA "
              f"(of {len(blocks)} candidates)")
        print(f"[encode] base  : {samples.nbytes} -> {len(base_bytes)} B")
        if use_cdc:
            print(f"[encode] CDC   : {len(base_bytes)} -> {len(unique)} B "
                  f"(chunks={len(manifest)})")

    arr = np.frombuffer(payload_src, dtype=np.int32).reshape(-1, ch)
    chosen_codec, payload = CODEC_RAW, payload_src

    if codec == "flac":
        if not HAVE_FLAC:
            raise RpzFormatError("FLAC requested but soundfile missing")
        payload = encode_flac(arr, fmt)
        chosen_codec = CODEC_FLAC
    elif codec == "auto":
        if HAVE_FLAC:
            try:
                fb = encode_flac(arr, fmt)
                if len(fb) < len(payload_src):
                    payload, chosen_codec = fb, CODEC_FLAC
            except RpzFormatError:
                pass
    elif codec != "raw":
        raise RpzFormatError(f"Unknown codec: {codec!r}")

    md5_hex = hashlib.md5(samples.tobytes()).hexdigest()
    header_prefix, wav_trailer = read_wav_parts(input_wav)

    rpz_format.pack(
        path=output_rpz, fmt=fmt,
        total_frames=total_frames, base_frames=base_frames,
        payload=payload, pointers=pointers,
        codec=chosen_codec, md5_hex=md5_hex,
        header_prefix=header_prefix,
        wav_trailer=wav_trailer,
        cdc_manifest=manifest,
        cdc_total_bytes=len(base_bytes) if manifest else 0,
        cdc_unique_bytes=len(unique) if manifest else 0,
    )

    if verbose:
        orig = samples.nbytes
        final = len(payload)
        print(f"[encode] codec  : {rpz_format.CODEC_NAMES[chosen_codec]}")
        print(f"[encode] final  : {final:>12} B ({final/orig:.1%} of original)")
        print(f"[encode] -> {output_rpz}")
