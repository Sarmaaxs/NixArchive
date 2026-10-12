"""Programmatic API."""
from __future__ import annotations
import os
import tempfile
import shutil
from typing import Iterable, List, Optional

from repeatzip.models import BlockSpec

class Archive:
    """Handle a .rpzarc bundle as an object."""

    def __init__(self, path: str):
        self.path = path

    @classmethod
    def pack(cls, input_dir: str, output_path: str, *,
             ref_ncc: float = 0.5, progress=None) -> "Archive":
        from repeatzip.cross.archive import archive
        archive(input_dir, output_path, ref_ncc=ref_ncc, verbose=False)
        return cls(output_path)

    def extract(self, output_dir: str) -> None:
        from repeatzip.cross.archive import extract
        extract(self.path, output_dir, verbose=False)

    def size(self) -> int:
        return os.path.getsize(self.path)

    def entries(self):
        from repeatzip.cross.archive import _read_bundle
        meta, _ = _read_bundle(self.path)
        return meta["entries"]

class Compressor:
    """Single-file encode/decode."""

    @staticmethod
    def encode(input_wav: str, output_rpz: str, *,
               auto_detect: bool = True,
               blocks: Optional[List[BlockSpec]] = None,
               codec: str = "auto",
               threshold: float = 0.75,
               delta_min_gain: float = 0.10,
               waveform_min_ncc: float = 0.90) -> None:
        from repeatzip.codec import encode
        encode(input_wav, output_rpz, blocks=blocks,
               auto_detect=auto_detect, codec=codec,
               delta_min_gain=delta_min_gain,
               waveform_min_ncc=waveform_min_ncc,
               detector_threshold=threshold,
               verbose=False)

    @staticmethod
    def decode(input_rpz: str, output_wav: str) -> None:
        from repeatzip.codec import decode
        decode(input_rpz, output_wav, verbose=False)

def compress_file(input_wav: str, output_rpz: str, **kw) -> int:
    Compressor.encode(input_wav, output_rpz, **kw)
    return os.path.getsize(output_rpz)

def extract_file(input_rpz: str, output_wav: str) -> int:
    Compressor.decode(input_rpz, output_wav)
    return os.path.getsize(output_wav)

def pack_folder(input_dir: str, output_rpzarc: str, **kw) -> int:
    Archive.pack(input_dir, output_rpzarc, **kw)
    return os.path.getsize(output_rpzarc)

def unpack_folder(input_rpzarc: str, output_dir: str) -> None:
    Archive(input_rpzarc).extract(output_dir)
