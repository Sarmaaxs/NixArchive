from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class BlockSpec:
    src_start_s: float
    src_end_s: float
    dst_start_s: float

@dataclass(frozen=True)
class Pointer:
    """Splice instruction.

    mode="copy"  : output[dst:dst+L] = base[src:src+L]           (byte-exact)
    mode="delta" : output[dst:dst+L] = base[src:src+L]
                                       + base[dst:dst+L]          (modular int16)
                   The base stream holds the residual at [dst, dst+L).
    """
    dst_frame: int
    src_frame: int
    length: int
    mode: str = "copy"

@dataclass(frozen=True)
class AudioFormat:
    sample_rate: int
    channels: int
    sampwidth: int
