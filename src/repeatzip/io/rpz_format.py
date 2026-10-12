"""See package docs."""
from __future__ import annotations
import base64
import json
import struct
from dataclasses import dataclass, asdict
from typing import List, Tuple

from repeatzip.errors import RpzFormatError
from repeatzip.models import AudioFormat, Pointer

MAGIC = b"RPZ\x01"
VERSION = 3
HEADER_SIZE = 64
FLAG_HAS_MD5 = 0x01

CODEC_RAW = 0
CODEC_FLAC = 1
CODEC_NAMES = {CODEC_RAW: "raw", CODEC_FLAC: "flac"}

_HEADER_STRUCT = struct.Struct("<4sBBBBIBB2xQQQQ16x")
_HEADER_STRUCT_V1 = struct.Struct("<4sBBHIBB2xQQQQ16x")

@dataclass
class RpzHeader:
    fmt: AudioFormat
    total_frames: int
    base_frames: int
    metadata_len: int
    payload_len: int
    codec: int = CODEC_RAW
    flags: int = 0
    version: int = VERSION

    @property
    def codec_name(self) -> str:
        return CODEC_NAMES.get(self.codec, f"unknown({self.codec})")

def pack(path, fmt, total_frames, base_frames, payload, pointers,
         codec=CODEC_RAW, md5_hex=None, header_prefix=None,
         wav_trailer=None,
         cdc_manifest=None, cdc_total_bytes=0, cdc_unique_bytes=0):
    metadata = {
        "format": "repeatzip",
        "version": VERSION,
        "codec": CODEC_NAMES.get(codec, "raw"),
        "pointers": [asdict(p) for p in pointers],
    }
    if header_prefix:
        metadata["wav_header_prefix_b64"] = base64.b64encode(
            header_prefix).decode("ascii")
    if wav_trailer:
        metadata["wav_trailer_b64"] = base64.b64encode(
            wav_trailer).decode("ascii")
    if cdc_manifest is not None:
        metadata["cdc"] = {
            "total_bytes": int(cdc_total_bytes),
            "unique_bytes": int(cdc_unique_bytes),
            "manifest": [int(x) for x in cdc_manifest],
        }

    meta_bytes = json.dumps(metadata, separators=(",", ":")).encode("utf-8")

    flags = 0
    trailer = b""
    if md5_hex:
        flags |= FLAG_HAS_MD5
        trailer = bytes.fromhex(md5_hex)
        if len(trailer) != 16:
            raise RpzFormatError("MD5 must be 16 bytes")

    header = _HEADER_STRUCT.pack(
        MAGIC, VERSION, codec, flags, 0,
        fmt.sample_rate, fmt.channels, fmt.sampwidth,
        total_frames, base_frames,
        len(meta_bytes), len(payload),
    )

    with open(path, "wb") as f:
        f.write(header)
        f.write(meta_bytes)
        f.write(payload)
        f.write(trailer)

def unpack(path):
    with open(path, "rb") as f:
        raw = f.read()
    if len(raw) < HEADER_SIZE:
        raise RpzFormatError("File too small.")
    if raw[:4] != MAGIC:
        raise RpzFormatError(f"Bad magic: {raw[:4]!r}")
    version = raw[4]
    if version == 1:
        return _unpack_v1(raw)
    if version in (2, 3):
        return _unpack_v2_v3(raw, version)
    raise RpzFormatError(f"Unsupported .rpz version {version}")

def _unpack_v2_v3(raw, version):
    (magic, _v, codec, flags, _,
     sr, ch, sw, total_frames, base_frames,
     meta_len, payload_len) = _HEADER_STRUCT.unpack_from(raw, 0)

    offset = HEADER_SIZE
    meta_end = offset + meta_len
    payload_end = meta_end + payload_len
    trailer_end = payload_end + (16 if flags & FLAG_HAS_MD5 else 0)
    if trailer_end > len(raw):
        raise RpzFormatError("Container truncated.")

    metadata = json.loads(raw[offset:meta_end].decode("utf-8"))
    payload = raw[meta_end:payload_end]
    md5_trailer = raw[payload_end:trailer_end]

    header = RpzHeader(
        fmt=AudioFormat(sample_rate=sr, channels=ch, sampwidth=sw),
        total_frames=total_frames,
        base_frames=base_frames,
        metadata_len=meta_len,
        payload_len=payload_len,
        codec=codec,
        flags=flags,
        version=version,
    )
    return header, metadata, payload, md5_trailer

def _unpack_v1(raw):
    (magic, _v, flags, _,
     sr, ch, sw, total_frames, base_frames,
     meta_len, audio_len) = _HEADER_STRUCT_V1.unpack_from(raw, 0)
    offset = HEADER_SIZE
    meta_end = offset + meta_len
    payload_end = meta_end + audio_len
    trailer_end = payload_end + (16 if flags & FLAG_HAS_MD5 else 0)
    if trailer_end > len(raw):
        raise RpzFormatError("Container truncated.")
    metadata = json.loads(raw[offset:meta_end].decode("utf-8"))
    payload = raw[meta_end:payload_end]
    md5_trailer = raw[payload_end:trailer_end]
    header = RpzHeader(
        fmt=AudioFormat(sample_rate=sr, channels=ch, sampwidth=sw),
        total_frames=total_frames,
        base_frames=base_frames,
        metadata_len=meta_len,
        payload_len=audio_len,
        codec=CODEC_RAW,
        flags=flags,
        version=1,
    )
    return header, metadata, payload, md5_trailer
