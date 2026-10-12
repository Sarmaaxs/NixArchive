from __future__ import annotations
import io
from typing import Tuple

import numpy as np

from repeatzip.errors import RpzFormatError
from repeatzip.models import AudioFormat

try:
    import soundfile as sf
    HAVE_FLAC = True
except Exception:
    sf = None
    HAVE_FLAC = False

def encode_flac(samples: np.ndarray, fmt: AudioFormat) -> bytes:
    if not HAVE_FLAC:
        raise RpzFormatError("soundfile not available.")
    if fmt.sampwidth not in (2, 3):
        raise RpzFormatError("Only 16 or 24-bit PCM supported.")
    if fmt.channels not in (1, 2):
        raise RpzFormatError("Only mono or stereo supported.")

    buf = io.BytesIO()
    if fmt.sampwidth == 2:
        sf.write(buf, samples.astype(np.int16, copy=False),
                 fmt.sample_rate, format="FLAC", subtype="PCM_16")
    else:
        # soundfile packs the low 24 bits of an int32 into PCM_24.
        sf.write(buf, samples.astype(np.int32, copy=False),
                 fmt.sample_rate, format="FLAC", subtype="PCM_24")
    return buf.getvalue()

def decode_flac(data: bytes) -> Tuple[np.ndarray, AudioFormat]:
    if not HAVE_FLAC:
        raise RpzFormatError("soundfile not available.")

    buf = io.BytesIO(data)
    try:
        info = sf.info(buf)
        buf.seek(0)
        if info.subtype == "PCM_24":
            arr, sr = sf.read(buf, dtype="int32", always_2d=True)
            sw = 3
        else:
            arr, sr = sf.read(buf, dtype="int16", always_2d=True)
            arr = arr.astype(np.int32)
            sw = 2
    except Exception as e:
        raise RpzFormatError(f"FLAC decode failed: {e}") from e

    fmt = AudioFormat(sample_rate=int(sr),
                      channels=int(arr.shape[1]),
                      sampwidth=sw)
    return np.ascontiguousarray(arr), fmt
