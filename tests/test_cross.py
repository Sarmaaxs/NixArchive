from __future__ import annotations
import os
import wave
import numpy as np
import pytest

from repeatzip.cross.archive import archive, extract

def _write(path, samples, sr=44100):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes(samples.astype(np.int16).tobytes())

def _read(path):
    with wave.open(str(path), "rb") as wf:
        return wf.readframes(wf.getnframes())

def test_archive_delta_roundtrip(tmp_path):
    src = tmp_path / "in"
    src.mkdir()

    rng = np.random.default_rng(0)
    base = (rng.standard_normal(44100 * 5) * 3000).astype(np.int16)
    _write(src / "a_orig.wav", base)
    _write(src / "b_loud.wav", (base * 1.05).astype(np.int16))
    _write(src / "c_quiet.wav", (base * 0.9).astype(np.int16))

    bundle = tmp_path / "pack.rpzarc"
    archive(str(src), str(bundle), ref_ncc=0.3, verbose=False)

    assert bundle.exists()
    out = tmp_path / "out"
    extract(str(bundle), str(out), verbose=False)

    for n in ("a_orig.wav", "b_loud.wav", "c_quiet.wav"):
        assert _read(src / n) == _read(out / n), f"mismatch on {n}"

def test_archive_savings(tmp_path):
    """Five slightly-scaled versions of a 20-second tone. The group
    should compress dramatically smaller than N independent FLACs."""
    src = tmp_path / "in"
    src.mkdir()

    sr = 44100
    t = np.arange(sr * 20) / sr
    base = (
        np.sin(2 * np.pi * 220 * t) * 6000
        + np.sin(2 * np.pi * 440 * t) * 3000
        + np.sin(2 * np.pi * 660 * t) * 1500
    ).astype(np.int16)

    for i, scale in enumerate([1.0, 1.02, 1.04, 1.06, 1.08]):
        _write(src / f"v{i}.wav", (base * scale).astype(np.int16))

    bundle = tmp_path / "p.rpzarc"
    archive(str(src), str(bundle), ref_ncc=0.3, verbose=False)

    total_orig = sum(os.path.getsize(src / f) for f in os.listdir(src))
    # Should be dramatically smaller than the group of originals.
    assert os.path.getsize(bundle) < total_orig * 0.35, (
        f"bundle {os.path.getsize(bundle)} vs original {total_orig}")
