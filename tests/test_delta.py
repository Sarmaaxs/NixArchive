from __future__ import annotations
import wave
import numpy as np
import pytest

from repeatzip.codec import encode, decode
from repeatzip.models import BlockSpec
from repeatzip.io import rpz_format

def make_similar_wav(path, sr=44100):
    t = np.arange(sr * 2) / sr
    seg_a = (np.sin(2 * np.pi * 440 * t) * 8000).astype(np.int16)
    # seg_b: seg_a attenuated to 95%  -  similar, not byte-identical
    seg_b = (seg_a * 0.95).astype(np.int16)
    full = np.concatenate([seg_a, np.zeros(sr, dtype=np.int16), seg_b])
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes(full.tobytes())
    return full

def read_frames(p):
    with wave.open(str(p), "rb") as wf:
        return wf.readframes(wf.getnframes())

def test_delta_roundtrip(tmp_path):
    wav = tmp_path / "sim.wav"
    make_similar_wav(wav)
    blocks = [BlockSpec(0.0, 2.0, 3.0)]
    rpz = tmp_path / "sim.rpz"
    out = tmp_path / "sim_out.wav"

    encode(str(wav), str(rpz), blocks=blocks, codec="raw", allow_delta=True)
    _, meta, _, _ = rpz_format.unpack(str(rpz))
    modes = [p.get("mode", "copy") for p in meta["pointers"]]
    assert "delta" in modes, f"expected delta pointer, got modes={modes}"

    decode(str(rpz), str(out))
    assert read_frames(wav) == read_frames(out)

def test_delta_disabled_roundtrip(tmp_path):
    wav = tmp_path / "sim2.wav"
    make_similar_wav(wav)
    blocks = [BlockSpec(0.0, 2.0, 3.0)]
    rpz = tmp_path / "sim2.rpz"
    out = tmp_path / "sim2_out.wav"
    encode(str(wav), str(rpz), blocks=blocks, codec="raw", allow_delta=False)
    decode(str(rpz), str(out))
    assert read_frames(wav) == read_frames(out)
