from __future__ import annotations
import wave

import numpy as np
import pytest

from repeatzip.codec import encode, decode
from repeatzip.models import BlockSpec
from repeatzip.analysis import detect_blocks

def make_test_wav(path, sr=44100):
    rng = np.random.default_rng(0)
    seg_len = sr * 4
    t = np.arange(seg_len) / sr
    segment = (
        np.sin(2 * np.pi * 220 * t) * 8000 +
        np.sin(2 * np.pi * 440 * t) * 4000 +
        np.sin(2 * np.pi * 880 * t) * 2000
    ).astype(np.int16)

    intro = (rng.standard_normal(sr * 2) * 800).astype(np.int16)
    middle = (rng.standard_normal(sr * 2) * 800).astype(np.int16)

    full = np.concatenate([intro, segment, middle,
                           segment, middle, segment])

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(full.tobytes())
    return full

def read_frames(p):
    with wave.open(str(p), "rb") as wf:
        return wf.readframes(wf.getnframes())

def test_roundtrip_manual_blocks(tmp_path):
    wav = tmp_path / "manual.wav"
    make_test_wav(wav)
    blocks = [BlockSpec(2.0, 6.0, 8.0), BlockSpec(2.0, 6.0, 14.0)]
    rpz = tmp_path / "manual.rpz"
    out = tmp_path / "manual_out.wav"
    encode(str(wav), str(rpz), blocks=blocks, codec="raw")
    decode(str(rpz), str(out))
    assert read_frames(wav) == read_frames(out)

def test_roundtrip_auto_detect(tmp_path):
    wav = tmp_path / "auto.wav"
    make_test_wav(wav)
    blocks = detect_blocks(str(wav))
    assert len(blocks) >= 1
    rpz = tmp_path / "auto.rpz"
    out = tmp_path / "auto_out.wav"
    encode(str(wav), str(rpz), blocks=blocks, codec="raw")
    decode(str(rpz), str(out))
    assert read_frames(wav) == read_frames(out)

def test_roundtrip_no_blocks(tmp_path):
    wav = tmp_path / "noblocks.wav"
    make_test_wav(wav)
    rpz = tmp_path / "noblocks.rpz"
    out = tmp_path / "noblocks_out.wav"
    encode(str(wav), str(rpz), blocks=[], codec="raw")
    decode(str(rpz), str(out))
    assert read_frames(wav) == read_frames(out)

def test_container_is_smaller_with_dedup(tmp_path):
    wav = tmp_path / "size.wav"
    make_test_wav(wav)
    rpz_off = tmp_path / "off.rpz"
    rpz_on = tmp_path / "on.rpz"
    encode(str(wav), str(rpz_off), blocks=[], codec="raw")
    encode(str(wav), str(rpz_on),
           blocks=[BlockSpec(2.0, 6.0, 8.0), BlockSpec(2.0, 6.0, 14.0)],
           codec="raw")
    assert rpz_on.stat().st_size < rpz_off.stat().st_size

def test_rejects_non_16bit(tmp_path):
    import wave as wv
    p = tmp_path / "8bit.wav"
    with wv.open(str(p), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(1)
        wf.setframerate(8000)
        wf.writeframes(b"\x80" * 8000)
    with pytest.raises(Exception):
        encode(str(p), str(tmp_path / "x.rpz"))
