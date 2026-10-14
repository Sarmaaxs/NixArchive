from __future__ import annotations
import os
import wave

import numpy as np
import pytest

from repeatzip.scan import scan_folder


def _write(path, samples, sr=44100):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.astype(np.int16).tobytes())


def _make_base(sr=44100, seconds=6.0):
    rng = np.random.default_rng(0)
    n = int(sr * seconds)
    t = np.arange(n) / sr
    tone = (np.sin(2 * np.pi * 220 * t) * 6000
            + np.sin(2 * np.pi * 440 * t) * 3000
            + np.sin(2 * np.pi * 660 * t) * 1500)
    noise = rng.standard_normal(n) * 400
    return (tone + noise).astype(np.int16)


def test_scan_finds_gain_and_trim_cluster(tmp_path):
    sr = 44100
    base = _make_base(sr, seconds=6.0)

    _write(tmp_path / "a.wav", base)
    _write(tmp_path / "b.wav", (base.astype(np.float64) * 0.9).astype(np.int16))
    _write(tmp_path / "c.wav", base[sr:])  # 1s trimmed from the front

    result = scan_folder(str(tmp_path), verbose=False)

    assert len(result.clusters) == 1, (
        f"expected 1 cluster, got {len(result.clusters)}: "
        f"{[c.members for c in result.clusters]}"
    )
    assert len(result.clusters[0].members) == 3


def test_scan_no_clusters_for_unrelated(tmp_path):
    sr = 44100
    n = sr * 6
    t = np.arange(n) / sr
    rng = np.random.default_rng(1)

    _write(tmp_path / "a.wav", (np.sin(2 * np.pi * 220 * t) * 6000).astype(np.int16))
    _write(tmp_path / "b.wav", (np.sin(2 * np.pi * 3000 * t) * 6000).astype(np.int16))
    _write(tmp_path / "c.wav", (rng.standard_normal(n) * 4000).astype(np.int16))

    result = scan_folder(str(tmp_path), verbose=False)
    assert result.clusters == [], (
        f"unexpected clusters: {[c.members for c in result.clusters]}"
    )


def test_scan_empty_folder_raises(tmp_path):
    from repeatzip.errors import RepeatzipError
    with pytest.raises(RepeatzipError):
        scan_folder(str(tmp_path))


def test_scan_skips_truncated_wav(tmp_path):
    import os
    # Valid WAV
    base = _make_base(44100, 6.0)
    _write(tmp_path / "good.wav", base)
    # Truncated WAV that will make wave.open raise EOFError
    bad = tmp_path / "bad.wav"
    with open(bad, "wb") as f:
        f.write(b"RIFF\x00\x00\x00\x00WAVE")  # header only, no chunks

    result = scan_folder(str(tmp_path), verbose=False)
    # Should not raise; should have scanned exactly one file.
    assert len(result.files) == 1
    assert result.files[0].path.endswith("good.wav")
