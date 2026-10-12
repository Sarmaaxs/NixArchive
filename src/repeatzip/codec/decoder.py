from __future__ import annotations
import base64
import hashlib
import numpy as np

from repeatzip.errors import ReconstructError
from repeatzip.io.wav_io import write_wav, samples_to_pcm_bytes
from repeatzip.io import rpz_format, cdc
from repeatzip.io.rpz_format import CODEC_RAW, CODEC_FLAC, FLAG_HAS_MD5
from repeatzip.io.flac_io import decode_flac
from repeatzip.io.wav_header import write_wav_parts
from repeatzip.codec import arith

def decode(input_rpz, output_wav, *, verbose=False):
    header, metadata, payload, md5_trailer = rpz_format.unpack(input_rpz)
    ch = header.fmt.channels

    if header.codec == CODEC_RAW:
        payload_bytes = payload
    elif header.codec == CODEC_FLAC:
        arr, _ = decode_flac(payload)
        payload_bytes = arr.astype(np.int32, copy=False).tobytes()
    else:
        raise ReconstructError(f"Unknown codec id: {header.codec}")

    cdc_info = metadata.get("cdc")
    if cdc_info:
        base_bytes = cdc.decompress(
            payload_bytes, cdc_info["manifest"], cdc_info["total_bytes"])
    else:
        base_bytes = payload_bytes

    base = np.frombuffer(base_bytes, dtype=np.int32).reshape(-1, ch)
    if base.shape[0] != header.base_frames:
        raise ReconstructError(
            f"base_frames mismatch: header={header.base_frames}, "
            f"decoded={base.shape[0]}")

    pointers = sorted(metadata["pointers"], key=lambda p: p["dst_frame"])
    chunks = []
    cursor = 0
    n_copy = n_delta = 0

    for p in pointers:
        d = int(p["dst_frame"]); s = int(p["src_frame"]); L = int(p["length"])
        mode = p.get("mode", "copy")

        if d < cursor:
            raise ReconstructError("Pointer dst behind cursor")
        if s + L > base.shape[0]:
            raise ReconstructError("Pointer src out of range")

        chunks.append(base[cursor:d])
        cursor = d

        if mode == "copy":
            chunks.append(base[s:s + L])
            n_copy += 1
        elif mode == "delta":
            if cursor + L > base.shape[0]:
                raise ReconstructError("Delta residual runs past base")
            residual = base[cursor:cursor + L]
            source = base[s:s + L]
            recon = arith.reconstruct(source, residual, header.fmt.sampwidth)
            chunks.append(recon)
            cursor += L
            n_delta += 1
        else:
            raise ReconstructError(f"Unknown pointer mode: {mode!r}")

    chunks.append(base[cursor:])
    restored = np.concatenate(chunks, axis=0) if chunks else base

    if restored.shape[0] != header.total_frames:
        raise ReconstructError(
            f"Frame count mismatch: got {restored.shape[0]}, "
            f"expected {header.total_frames}")

    if header.flags & FLAG_HAS_MD5:
        got = hashlib.md5(restored.tobytes()).hexdigest()
        want = md5_trailer.hex()
        if got != want:
            raise ReconstructError(
                f"Checksum mismatch:\n  expected {want}\n  got      {got}")

    prefix_b64 = metadata.get("wav_header_prefix_b64")
    trailer_b64 = metadata.get("wav_trailer_b64")
    if prefix_b64:
        prefix = base64.b64decode(prefix_b64)
        trailer = base64.b64decode(trailer_b64) if trailer_b64 else b""
        pcm_bytes = samples_to_pcm_bytes(restored, header.fmt.sampwidth)
        write_wav_parts(output_wav, pcm_bytes, prefix, trailer)
    else:
        write_wav(output_wav, restored, header.fmt)

    if verbose:
        print(f"[decode] codec: {header.codec_name}  v{header.version}")
        print(f"[decode] pointers: {n_copy} copy, {n_delta} delta")
        if cdc_info:
            print(f"[decode] CDC: {len(payload_bytes)} -> "
                  f"{cdc_info['total_bytes']} B")
        print(f"[decode] {restored.shape[0]} frames -> {output_wav} (md5 ok)")
