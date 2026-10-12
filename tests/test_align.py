from __future__ import annotations
import wave
import time

import numpy as np
import pytest

from repeatzip.cross.archive import archive, extract, _read_bundle
from repeatzip.cross.align import match_files


def _write(path, samples, sr=44100):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes(samples.astype(np.int16).tobytes())


def _read(path):
    with wave.open(str(path), "rb") as wf:
        return wf.readframes(wf.getnframes())


def _entries(bundle):
    meta, _ = _read_bundle(str(bundle))
    return meta["entries"]


# ---------- unit tests for alignment ----------

def test_lag_detection_positive_multiple_of_coarse():
    rng = np.random.default_rng(0)
    ref = (rng.standard_normal(44100 * 6) * 4000).astype(np.int16)
    intro = (rng.standard_normal(44100 * 2) * 4000).astype(np.int16)
    tgt = np.concatenate([intro, ref])

    lag, gain, ncc = match_files(ref, tgt)
    expected = 44100 * 2
    assert abs(lag - expected) <= 4, f"lag={lag} expected~{expected}"
    assert ncc > 0.95


def test_lag_detection_negative_multiple_of_coarse():
    rng = np.random.default_rng(0)
    tgt = (rng.standard_normal(44100 * 6) * 4000).astype(np.int16)
    intro = (rng.standard_normal(44100 * 2) * 4000).astype(np.int16)
    ref = np.concatenate([intro, tgt])

    lag, gain, ncc = match_files(ref, tgt)
    expected = -44100 * 2
    assert abs(lag - expected) <= 4, f"lag={lag} expected~{expected}"
    assert ncc > 0.95


def test_lag_detection_odd_88203():
    """Lag deliberately not a multiple of the coarse factor."""
    rng = np.random.default_rng(4)
    n = 44100 * 8
    ref = (rng.standard_normal(n) * 4000).astype(np.int16)
    intro = (rng.standard_normal(88203) * 4000).astype(np.int16)
    tgt = np.concatenate([intro, ref])

    lag, gain, ncc = match_files(ref, tgt)
    assert abs(lag - 88203) <= 4, f"lag={lag} expected 88203"
    assert ncc > 0.95


def test_gain_detection():
    rng = np.random.default_rng(0)
    ref = (rng.standard_normal(44100 * 4) * 5000).astype(np.int16)
    tgt = (ref.astype(np.float64) * 0.95).astype(np.int16)

    lag, gain, ncc = match_files(ref, tgt)
    assert abs(lag) <= 4
    assert abs(gain - 0.95) < 0.02


def test_large_offset_4sec_excerpt():
    """4-second excerpt taken 20 seconds into a 60-second reference."""
    rng = np.random.default_rng(7)
    sr = 44100
    ref = (rng.standard_normal(sr * 60) * 4000).astype(np.int16)
    offset = sr * 20
    tgt = ref[offset:offset + sr * 4]

    lag, gain, ncc = match_files(ref, tgt)
    # Under our convention, tgt[i] ~= ref[i + offset], so lag = -offset.
    assert abs(lag + offset) <= 8, f"lag={lag} expected~{-offset}"
    assert ncc > 0.95


def test_match_files_speed():
    """4-minute noise pair should align in under 15 s."""
    rng = np.random.default_rng(11)
    sr = 44100
    n = sr * 240
    ref = (rng.standard_normal(n) * 3000).astype(np.int16)
    tgt = (ref.astype(np.float64) * 0.97).astype(np.int16)

    t0 = time.time()
    lag, gain, ncc = match_files(ref, tgt)
    elapsed = time.time() - t0
    assert elapsed < 15.0, f"match_files took {elapsed:.1f}s, expected <15s"
    assert abs(lag) <= 4


# ---------- integration: delta path must actually be used ----------

def test_delta_used_for_gain_shift(tmp_path):
    src = tmp_path / "in"; src.mkdir()
    rng = np.random.default_rng(1)
    base = (rng.standard_normal(44100 * 5) * 5000).astype(np.int16)
    _write(src / "a.wav", base)
    _write(src / "b.wav", (base.astype(np.float64) * 0.95).astype(np.int16))

    bundle = tmp_path / "p.rpzarc"
    archive(str(src), str(bundle), ref_ncc=0.3, verbose=False)

    entries = _entries(bundle)
    deltas = [e for e in entries if e["delta_mode"] == "delta"]
    assert len(deltas) >= 1, f"delta path not exercised: {entries}"

    out = tmp_path / "out"
    extract(str(bundle), str(out), verbose=False)
    assert _read(src / "a.wav") == _read(out / "a.wav")
    assert _read(src / "b.wav") == _read(out / "b.wav")


def test_delta_used_for_intro_trim(tmp_path):
    src = tmp_path / "in"; src.mkdir()
    rng = np.random.default_rng(2)
    body = (rng.standard_normal(44100 * 8) * 4000).astype(np.int16)
    intro = (rng.standard_normal(44100 * 2) * 800).astype(np.int16)

    _write(src / "a_body_only.wav", body)
    _write(src / "b_with_intro.wav", np.concatenate([intro, body]))

    bundle = tmp_path / "p.rpzarc"
    archive(str(src), str(bundle), ref_ncc=0.3, verbose=False)

    entries = _entries(bundle)
    deltas = [e for e in entries if e["delta_mode"] == "delta"]
    assert len(deltas) >= 1, f"delta path not exercised: {entries}"
    d = deltas[0]
    assert d["leading_frames"] > 0 or d["lag"] != 0, \
        f"alignment metadata missing: {d}"

    out = tmp_path / "out"
    extract(str(bundle), str(out), verbose=False)
    assert _read(src / "a_body_only.wav") == _read(out / "a_body_only.wav")
    assert _read(src / "b_with_intro.wav") == _read(out / "b_with_intro.wav")


def test_solo_when_files_unrelated(tmp_path):
    src = tmp_path / "in"; src.mkdir()
    rng = np.random.default_rng(3)
    _write(src / "a.wav",
           (rng.standard_normal(44100 * 4) * 4000).astype(np.int16))
    _write(src / "b.wav",
           (rng.standard_normal(44100 * 4) * 4000).astype(np.int16))

    bundle = tmp_path / "p.rpzarc"
    archive(str(src), str(bundle), ref_ncc=0.9, verbose=False)

    entries = _entries(bundle)
    deltas = [e for e in entries if e["delta_mode"] == "delta"]
    assert len(deltas) == 0, f"unexpected delta: {entries}"

    out = tmp_path / "out"
    extract(str(bundle), str(out), verbose=False)
    for n in ("a.wav", "b.wav"):
        assert _read(src / n) == _read(out / n)


def test_excerpt_at_55s_plus_3_samples():
    """Excerpt at 55s+3 frames of a 60s reference. Tests both bugs:
    the fine search must find the offset, and the FFT unwrap must not
    flip the sign."""
    rng = np.random.default_rng(42)
    sr = 44100
    ref = (rng.standard_normal(sr * 60) * 3000).astype(np.int16)
    offset = sr * 55 + 3
    tgt = ref[offset:offset + sr * 4]

    lag, gain, ncc = match_files(ref, tgt)
    # tgt[i] = ref[offset + i], so lag = -offset.
    expected = -offset
    assert abs(lag - expected) <= 4, f"lag={lag} expected~{expected}"
    assert ncc > 0.95, f"ncc={ncc}"


def test_excerpt_near_end_of_long_reference():
    """Excerpt at 55s of 60s, must survive sign fix."""
    rng = np.random.default_rng(13)
    sr = 44100
    ref = (rng.standard_normal(sr * 60) * 3000).astype(np.int16)
    tgt = ref[sr * 55:sr * 55 + sr * 4]

    lag, gain, ncc = match_files(ref, tgt)
    assert abs(lag + sr * 55) <= 4, f"lag={lag} expected~{-sr*55}"
    assert ncc > 0.95


def test_reference_is_excerpt_of_target():
    """Reverse case: the reference is a short excerpt of a long target."""
    rng = np.random.default_rng(14)
    sr = 44100
    full = (rng.standard_normal(sr * 60) * 3000).astype(np.int16)
    ref = full[sr * 55:sr * 55 + sr * 4]

    lag, gain, ncc = match_files(ref, full)
    assert abs(lag - sr * 55) <= 4, f"lag={lag} expected~{sr*55}"
    assert ncc > 0.95
