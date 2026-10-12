"""Multi-file archive with cross-file delta coding.

Layout (.rpzarc):
    [4]  magic = b"RPZA"
    [1]  version
    [1]  reserved
    [2]  entry count
    [8]  metadata length
    [8]  total payload length
    [N]  metadata JSON
    [M]  payload blobs (concatenated)
"""
from __future__ import annotations
import base64
import hashlib
import json
import os
import struct
from dataclasses import dataclass, asdict
from typing import List, Optional

import numpy as np

from repeatzip.errors import RepeatzipError
from repeatzip.io import flac_io
from repeatzip.io.flac_io import HAVE_FLAC, encode_flac, decode_flac
from repeatzip.io.wav_io import read_wav, samples_to_pcm_bytes
from repeatzip.io.wav_header import read_wav_parts, write_wav_parts
from .align import match_files
from .delta_codec import encode_against, decode_against
from .parallel import load_wavs_parallel
from .similarity import quick_ncc


MAGIC = b"RPZA"
VERSION = 2
HDR = struct.Struct("<4sBBHQQ")


@dataclass
class Entry:
    name: str
    kind: str
    codec: str
    original_bytes: int
    frames: int
    sample_rate: int
    channels: int
    sampwidth: int
    payload_offset: int
    payload_len: int
    wav_header_b64: str
    wav_trailer_b64: str = ""
    reference: Optional[str] = None
    lag: int = 0
    gain: float = 1.0
    overlap_frames: int = 0
    tail_frames: int = 0
    leading_frames: int = 0
    delta_mode: str = "delta"
    original_sha256: str = ""

    def to_json(self):
        return asdict(self)


def archive(input_dir: str, output_path: str, *, ref_ncc=0.5,
            workers: int = 0, verbose=False) -> None:
    if not HAVE_FLAC:
        raise RepeatzipError("soundfile required for archive mode")
    if not (0.0 <= ref_ncc <= 1.0):
        raise RepeatzipError(f"ref_ncc must be in [0, 1], got {ref_ncc}")

    names = sorted(n for n in os.listdir(input_dir)
                   if n.lower().endswith(".wav"))
    if not names:
        raise RepeatzipError(f"No .wav files in {input_dir}")

    if verbose:
        print(f"[archive] {len(names)} file(s) in {input_dir}")

    loaded = load_wavs_parallel(input_dir, names, workers=workers)
    cache = {name: (arr, fmt, prefix, trailer)
             for name, (_, arr, fmt, prefix, trailer) in loaded.items()}

    entries: List[Entry] = []
    payload_blobs: List[bytes] = []
    cursor = 0
    ref_pool: List[str] = []

    for name in names:
        samples, fmt, prefix, trailer = cache[name]
        orig_path = os.path.join(input_dir, name)
        orig_bytes = os.path.getsize(orig_path)
        with open(orig_path, "rb") as _fh:
            orig_sha = hashlib.sha256(_fh.read()).hexdigest()

        best_ref = None
        best_ncc = ref_ncc
        best_lag = 0
        best_gain = 1.0

        tgt_rms = float(np.sqrt((samples.astype(np.float64) ** 2).mean()))
        if verbose and len(ref_pool) > 8:
            print(f"          {name}: testing {len(ref_pool)} reference candidate(s)")
        for cand in ref_pool:
            cand_samples, cand_fmt, _, _ = cache[cand]
            if (cand_fmt.sample_rate != fmt.sample_rate
                    or cand_fmt.channels != fmt.channels
                    or cand_fmt.sampwidth != fmt.sampwidth):
                continue
            cand_rms = float(np.sqrt((cand_samples.astype(np.float64) ** 2).mean()))
            if cand_rms > 0 and tgt_rms > 0:
                ratio = max(cand_rms, tgt_rms) / min(cand_rms, tgt_rms)
                if ratio > 100.0:
                    continue
            try:
                lag, gain, ncc = match_files(cand_samples, samples)
            except Exception as exc:
                if verbose:
                    print(f"          align failed for {cand}: {exc}")
                continue
            if ncc > best_ncc:
                best_ncc = ncc
                best_ref = cand
                best_lag = lag
                best_gain = gain

        payload_solo = encode_flac(samples, fmt)
        size_solo = len(payload_solo)

        payload_delta = None
        info_delta = None
        if best_ref is not None:
            ref_samples, ref_fmt, _, _ = cache[best_ref]
            if verbose:
                print(f"          alignment: lag={best_lag} "
                      f"ncc={best_ncc:.3f} gain={best_gain:.5f}")
            payload_delta, info_delta = encode_against(
                ref_samples, samples, fmt,
                lag=best_lag, gain=best_gain)

        if payload_delta is not None and len(payload_delta) < size_solo:
            kind = "delta"
            payload = payload_delta
            info = info_delta
            ref = best_ref
            mode_str = (f"delta vs {best_ref} "
                        f"(NCC={best_ncc:.3f} lag={info.get('lag', 0)} "
                        f"gain={info.get('gain', 1.0):.4f})")
        else:
            kind = "solo"
            payload = payload_solo
            info = {"mode": "raw", "lag": 0, "gain": 1.0,
                    "overlap_frames": 0, "leading_frames": 0,
                    "trailing_frames": int(samples.shape[0])}
            ref = None
            mode_str = "solo"

        entries.append(Entry(
            name=name,
            kind=kind,
            codec="flac",
            original_bytes=orig_bytes,
            frames=int(samples.shape[0]),
            sample_rate=fmt.sample_rate,
            channels=fmt.channels,
            sampwidth=fmt.sampwidth,
            payload_offset=cursor,
            payload_len=len(payload),
            wav_header_b64=base64.b64encode(prefix).decode("ascii"),
            wav_trailer_b64=base64.b64encode(trailer).decode("ascii"),
            reference=ref,
            lag=int(info.get("lag", 0)),
            gain=float(info.get("gain", 1.0)),
            overlap_frames=int(info.get("overlap_frames", 0)),
            tail_frames=int(info.get("trailing_frames", 0)),
            leading_frames=int(info.get("leading_frames", 0)),
            delta_mode=info.get("mode", "raw"),
            original_sha256=orig_sha,
        ))
        payload_blobs.append(payload)
        cursor += len(payload)

        if kind == "solo":
            ref_pool.append(name)

        if verbose:
            ratio = orig_bytes / len(payload) if payload else 0
            print(f"[archive] {name:>40}  {orig_bytes:>12,} B  "
                  f"-> {len(payload):>12,} B  ({ratio:5.2f}x)  {mode_str}")

    metadata = {
        "format": "repeatzip-archive",
        "version": VERSION,
        "entries": [e.to_json() for e in entries],
    }
    meta_bytes = json.dumps(metadata, separators=(",", ":")).encode("utf-8")

    header = HDR.pack(MAGIC, VERSION, 0, len(entries),
                      len(meta_bytes), sum(len(b) for b in payload_blobs))

    with open(output_path, "wb") as f:
        f.write(header)
        f.write(meta_bytes)
        for blob in payload_blobs:
            f.write(blob)

    total_orig = sum(e.original_bytes for e in entries)
    total_out = os.path.getsize(output_path)
    if verbose:
        print()
        print(f"[archive] total original : {total_orig:>12,} B "
              f"({total_orig/1e6:.2f} MB)")
        print(f"[archive] total bundle   : {total_out:>12,} B "
              f"({total_out/1e6:.2f} MB)")
        print(f"[archive] savings        : "
              f"{100*(1 - total_out/total_orig):.2f}%")
        print(f"[archive] ratio          : {total_orig/total_out:.3f}x")


MAX_META_LEN = 1 << 30        # 1 GiB
MAX_PAYLOAD_LEN = 1 << 40     # 1 TiB
MAX_FILE_BYTES = 1 << 41      # 2 TiB hard ceiling


def _read_bundle(path):
    file_size = os.path.getsize(path)
    if file_size < HDR.size:
        raise RepeatzipError("bundle too small")
    if file_size > MAX_FILE_BYTES:
        raise RepeatzipError(
            f"bundle size {file_size} exceeds limit {MAX_FILE_BYTES}")

    with open(path, "rb") as f:
        hdr_bytes = f.read(HDR.size)
        magic, version, _, count, meta_len, payload_len = HDR.unpack_from(
            hdr_bytes, 0)
        if magic != MAGIC:
            raise RepeatzipError(f"bad archive magic: {magic!r}")
        if version not in (1, 2):
            raise RepeatzipError(f"unsupported archive version {version}")
        if meta_len > MAX_META_LEN:
            raise RepeatzipError(
                f"metadata length {meta_len} exceeds limit")
        if payload_len > MAX_PAYLOAD_LEN:
            raise RepeatzipError(
                f"payload length {payload_len} exceeds limit")

        declared_total = HDR.size + meta_len + payload_len
        if declared_total != file_size:
            if declared_total > file_size:
                raise RepeatzipError(
                    f"declared size {declared_total} exceeds file "
                    f"size {file_size}")
            raise RepeatzipError(
                f"trailing data: declared size {declared_total}, "
                f"file size {file_size}")

        meta_bytes = f.read(meta_len)
        if len(meta_bytes) != meta_len:
            raise RepeatzipError("metadata truncated")
        pay = f.read(payload_len)
        if len(pay) != payload_len:
            raise RepeatzipError("payload truncated")

    try:
        meta = json.loads(meta_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RepeatzipError(f"bad archive metadata: {exc}") from exc

    entries = meta.get("entries", [])
    if not isinstance(entries, list):
        raise RepeatzipError("metadata entries must be a list")
    if len(entries) != count:
        raise RepeatzipError(
            f"header entry count {count} does not match metadata "
            f"entries {len(entries)}")

    for e in entries:
        try:
            po = int(e["payload_offset"])
            pl = int(e["payload_len"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RepeatzipError(f"bad entry offsets: {exc}") from exc
        if po < 0 or pl < 0 or po + pl > payload_len:
            raise RepeatzipError(
                f"entry {e.get('name', '?')!r} payload out of range")
        e.setdefault("delta_mode",
                     "delta" if e.get("reference") else "raw")
        e.setdefault("lag", 0)
        e.setdefault("gain", 1.0)
        e.setdefault("leading_frames", 0)
        e.setdefault("wav_trailer_b64", "")
        e.setdefault("original_sha256", "")
    return meta, pay


def _safe_output_path(output_dir: str, name: str) -> str:
    if not name or "\x00" in name:
        raise RepeatzipError(f"invalid entry name: {name!r}")
    if name.startswith("."):
        raise RepeatzipError(f"hidden filenames not allowed: {name!r}")
    if "/" in name or "\\" in name:
        raise RepeatzipError(
            f"entry name must be a flat filename: {name!r}")
    if os.path.isabs(name):
        raise RepeatzipError(
            f"entry name must be relative: {name!r}")
    dest = os.path.realpath(output_dir)
    full = os.path.realpath(os.path.join(output_dir, name))
    if os.path.commonpath([dest, full]) != dest:
        raise RepeatzipError(
            f"entry {name!r} escapes extraction directory")
    return full


def extract(bundle_path: str, output_dir: str, *, verbose=False) -> None:
    meta, payload = _read_bundle(bundle_path)
    os.makedirs(output_dir, exist_ok=True)

    for _e in meta["entries"]:
        _safe_output_path(output_dir, _e["name"])

    def _payload_for(e):
        return payload[e["payload_offset"]:
                       e["payload_offset"] + e["payload_len"]]

    from repeatzip.models import AudioFormat
    decoded = {}
    remaining = list(meta["entries"])

    while remaining:
        progressed = False
        for e in list(remaining):
            if e["kind"] == "solo":
                arr, _ = decode_flac(_payload_for(e))
                decoded[e["name"]] = arr
            elif e["kind"] == "delta":
                ref_name = e["reference"]
                if ref_name not in decoded:
                    continue
                fmt = AudioFormat(
                    sample_rate=e["sample_rate"],
                    channels=e["channels"],
                    sampwidth=e["sampwidth"],
                )
                arr = decode_against(
                    decoded[ref_name], _payload_for(e),
                    {"mode": e.get("delta_mode", "delta"),
                     "lag": e.get("lag", 0),
                     "gain": e.get("gain", 1.0),
                     "overlap_frames": e["overlap_frames"],
                     "leading_frames": e.get("leading_frames", 0),
                     "trailing_frames": e.get("tail_frames", 0)},
                    fmt,
                )
                decoded[e["name"]] = arr
            else:
                raise RepeatzipError(f"Unknown entry kind: {e['kind']}")

            prefix = base64.b64decode(e["wav_header_b64"])
            tb64 = e.get("wav_trailer_b64", "")
            trailer = base64.b64decode(tb64) if tb64 else b""
            pcm = samples_to_pcm_bytes(decoded[e["name"]], e["sampwidth"])

            stored = e.get("original_sha256", "")
            if stored:
                full_bytes = prefix + pcm + trailer
                got = hashlib.sha256(full_bytes).hexdigest()
                if got != stored:
                    raise RepeatzipError(
                        f"integrity check failed for {e['name']!r}: "
                        f"expected {stored[:12]}, got {got[:12]}")

            out_path = _safe_output_path(output_dir, e["name"])
            write_wav_parts(out_path, pcm, prefix, trailer)

            if verbose:
                print(f"[extract] {e['name']}  {e['kind']}"
                      + (f" vs {e['reference']}" if e["reference"] else ""))
            remaining.remove(e)
            progressed = True

        if not progressed:
            names = [e["name"] for e in remaining]
            raise RepeatzipError(
                f"Circular or unresolvable delta dependency: {names}")
