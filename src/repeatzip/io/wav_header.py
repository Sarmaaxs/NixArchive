"""Capture and restore the WAV container around the PCM payload.

A WAV file is:
    [prefix][PCM data][trailer]

The prefix is everything up to and including the `data` chunk size
field. The trailer is everything after the PCM (ID3 tags, LIST chunks,
padding). Both must be preserved verbatim for byte-identical restore.
"""
from __future__ import annotations
import struct
from typing import Tuple

def read_wav_parts(path: str) -> Tuple[bytes, bytes]:
    """Return (prefix, trailer).

    prefix  ends right after the `data` chunk size field (so PCM follows).
    trailer is everything after the declared PCM payload.
    """
    with open(path, "rb") as f:
        raw = f.read()

    if len(raw) < 12 or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError("Not a RIFF/WAVE file")

    pos = 12
    while pos + 8 <= len(raw):
        cid = raw[pos:pos+4]
        csz = struct.unpack_from("<I", raw, pos+4)[0]
        chunk_start = pos + 8
        chunk_end = chunk_start + csz

        if cid == b"data":
            prefix = raw[:chunk_start]
            trailer = raw[chunk_end:] if chunk_end <= len(raw) else b""
            return prefix, trailer

        pos = chunk_end + (csz & 1)

    raise ValueError("No data chunk found")

def write_wav_parts(path: str, pcm_bytes: bytes,
                    prefix: bytes, trailer: bytes) -> None:
    """Write [prefix][PCM][trailer] verbatim.

    No size fields are touched. The prefix carries whatever RIFF size
    and data size the original encoder wrote, so the output is byte
    identical to the input for well-formed files.
    """
    with open(path, "wb") as f:
        f.write(prefix)
        f.write(pcm_bytes)
        f.write(trailer)

# Kept for callers that only need the header bytes.
def read_header_prefix(path: str) -> Tuple[bytes, int]:
    prefix, _ = read_wav_parts(path)
    return prefix, len(prefix)
