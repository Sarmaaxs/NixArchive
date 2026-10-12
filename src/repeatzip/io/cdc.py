"""Fixed-size chunk deduplication.

Finds byte-identical regions ANYWHERE in the stream, not just at
detector-identified block boundaries. Catches digital silence,
copied drum samples, loops shorter than the detector's minimum,
and any other literal byte duplication.

Manifest encoding:
    0        -> fresh chunk: take next CHUNK_SIZE bytes from `unique`
    -N (N>0) -> ref: copy CHUNK_SIZE bytes from output at offset N-1
"""
from __future__ import annotations
import hashlib

CHUNK_SIZE = 16384   # 8192 stereo frames @ 16-bit = ~186 ms

def compress(pcm_bytes: bytes):
    """Return (unique_bytes, manifest)."""
    total = len(pcm_bytes)
    seen = {}
    unique = bytearray()
    manifest = []
    pos = 0
    while pos < total:
        chunk = pcm_bytes[pos:pos+CHUNK_SIZE]
        if len(chunk) == CHUNK_SIZE:
            h = hashlib.blake2b(chunk, digest_size=16).digest()
            if h in seen:
                manifest.append(-(seen[h] + 1))
            else:
                seen[h] = pos
                manifest.append(0)
                unique += chunk
        else:
            manifest.append(0)
            unique += chunk
        pos += CHUNK_SIZE
    return bytes(unique), manifest

def decompress(unique: bytes, manifest, total_bytes: int) -> bytes:
    out = bytearray()
    fresh_cursor = 0
    for entry in manifest:
        remaining = total_bytes - len(out)
        if remaining <= 0:
            break
        this_len = min(CHUNK_SIZE, remaining)
        if entry == 0:
            out += unique[fresh_cursor:fresh_cursor + this_len]
            fresh_cursor += this_len
        else:
            ref = -entry - 1
            out += out[ref:ref + this_len]
    return bytes(out)
