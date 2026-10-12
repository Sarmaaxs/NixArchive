from __future__ import annotations
import wave
from typing import Tuple
import numpy as np

from repeatzip.errors import WavFormatError
from repeatzip.models import AudioFormat

_SUPPORTED = {2, 3}  # 16-bit and 24-bit PCM

def read_wav(path: str) -> Tuple[np.ndarray, AudioFormat]:
    """Return (samples[frames, channels] int32, AudioFormat).

    For 16-bit input, samples are int32 values in [-32768, 32767].
    For 24-bit input, samples are int32 values in [-8388608, 8388607].
    We always expose int32 so downstream code handles both uniformly.
    """
    with wave.open(path, "rb") as wf:
        if wf.getcomptype() != "NONE":
            raise WavFormatError("Compressed WAV codecs are not supported.")
        sampwidth = wf.getsampwidth()
        if sampwidth not in _SUPPORTED:
            raise WavFormatError(
                f"Only 16/24-bit PCM supported (got {sampwidth * 8}-bit)."
            )
        fmt = AudioFormat(
            sample_rate=wf.getframerate(),
            channels=wf.getnchannels(),
            sampwidth=sampwidth,
        )
        raw = wf.readframes(wf.getnframes())

    if sampwidth == 2:
        arr = np.frombuffer(raw, dtype="<i2").astype(np.int32)
    else:  # 24-bit little-endian, packed 3 bytes per sample
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        # Reconstruct signed 24-bit: little-endian, sign-extend top byte.
        as_u32 = (b[:, 0].astype(np.uint32)
                  | (b[:, 1].astype(np.uint32) << 8)
                  | (b[:, 2].astype(np.uint32) << 16))
        as_i32 = as_u32.astype(np.int32)
        # Sign-extend bit 23
        sign = (as_u32 & 0x800000).astype(bool)
        as_i32[sign] -= 0x1000000
        arr = as_i32

    samples = arr.reshape(-1, fmt.channels).copy()
    return samples, fmt

def write_wav(path: str, samples: np.ndarray, fmt: AudioFormat) -> None:
    arr = samples.astype(np.int64, copy=False)
    if fmt.sampwidth == 2:
        pcm = np.clip(arr, -32768, 32767).astype("<i2").tobytes()
    elif fmt.sampwidth == 3:
        u = arr & 0xFFFFFF
        packed = np.empty((u.size, 3), dtype=np.uint8)
        packed[:, 0] = (u & 0xFF).astype(np.uint8)
        packed[:, 1] = ((u >> 8) & 0xFF).astype(np.uint8)
        packed[:, 2] = ((u >> 16) & 0xFF).astype(np.uint8)
        pcm = packed.tobytes()
    else:
        raise WavFormatError(f"Cannot write {fmt.sampwidth * 8}-bit")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(fmt.channels)
        wf.setsampwidth(fmt.sampwidth)
        wf.setframerate(fmt.sample_rate)
        wf.writeframes(pcm)

def sec_to_frame(sec: float, fmt: AudioFormat) -> int:
    return int(round(sec * fmt.sample_rate))

def samples_to_pcm_bytes(samples_int32, sampwidth):
    """Convert internal int32 samples to raw PCM bytes for the given width."""
    arr = samples_int32.astype(np.int64, copy=False)
    if sampwidth == 2:
        return np.clip(arr, -32768, 32767).astype("<i2").tobytes()
    elif sampwidth == 3:
        u = arr & 0xFFFFFF
        packed = np.empty((u.size, 3), dtype=np.uint8)
        packed[:, 0] = (u & 0xFF).astype(np.uint8)
        packed[:, 1] = ((u >> 8) & 0xFF).astype(np.uint8)
        packed[:, 2] = ((u >> 16) & 0xFF).astype(np.uint8)
        return packed.tobytes()
    else:
        raise ValueError(f"Unsupported sampwidth {sampwidth}")
