from __future__ import annotations
import json
import wave

import numpy as np
import pytest

from repeatzip.cross.archive import (
    archive, extract, _read_bundle, _safe_output_path, HDR,
)
from repeatzip.errors import RepeatzipError


def _write_silent_wav(path, sr=44100, seconds=1):
    n = sr * seconds
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes(np.zeros(n, dtype=np.int16).tobytes())


def test_extract_rejects_path_traversal(tmp_path):
    src = tmp_path / "in"; src.mkdir()
    _write_silent_wav(src / "a.wav")

    bundle = tmp_path / "p.rpzarc"
    archive(str(src), str(bundle), verbose=False)

    raw = bytearray(bundle.read_bytes())
    magic, version, _, count, meta_len, payload_len = HDR.unpack_from(raw, 0)
    off = HDR.size
    meta = json.loads(bytes(raw[off:off + meta_len]).decode("utf-8"))
    meta["entries"][0]["name"] = "../evil.wav"
    new_meta = json.dumps(meta, separators=(",", ":")).encode("utf-8")
    new_raw = bytearray()
    new_raw += HDR.pack(magic, version, 0, count, len(new_meta), payload_len)
    new_raw += new_meta
    new_raw += bytes(raw[off + meta_len:off + meta_len + payload_len])
    bad = tmp_path / "bad.rpzarc"
    bad.write_bytes(bytes(new_raw))

    out = tmp_path / "out"
    with pytest.raises(RepeatzipError):
        extract(str(bad), str(out), verbose=False)

    assert not (tmp_path / "evil.wav").exists()


def test_safe_output_path_rejects_absolute(tmp_path):
    with pytest.raises(RepeatzipError):
        _safe_output_path(str(tmp_path), "/etc/passwd")


def test_safe_output_path_rejects_parent(tmp_path):
    with pytest.raises(RepeatzipError):
        _safe_output_path(str(tmp_path), "../../etc/passwd")


def test_safe_output_path_rejects_nested(tmp_path):
    with pytest.raises(RepeatzipError):
        _safe_output_path(str(tmp_path), "sub/file.wav")


def test_read_bundle_rejects_bad_meta_len(tmp_path):
    src = tmp_path / "in"; src.mkdir()
    _write_silent_wav(src / "a.wav")
    bundle = tmp_path / "p.rpzarc"
    archive(str(src), str(bundle), verbose=False)

    raw = bytearray(bundle.read_bytes())
    magic, version, _, count, meta_len, payload_len = HDR.unpack_from(raw, 0)
    HDR.pack_into(raw, 0, magic, version, 0, count, len(raw) * 2, payload_len)
    bad = tmp_path / "bad.rpzarc"
    bad.write_bytes(bytes(raw))

    with pytest.raises(RepeatzipError):
        _read_bundle(str(bad))


def test_verify_detects_corruption(tmp_path):
    src = tmp_path / "in"; src.mkdir()
    _write_silent_wav(src / "a.wav")
    bundle = tmp_path / "p.rpzarc"
    archive(str(src), str(bundle), verbose=False)

    raw = bytearray(bundle.read_bytes())
    magic, version, _, count, meta_len, payload_len = HDR.unpack_from(raw, 0)
    flip_at = HDR.size + meta_len + payload_len // 2
    raw[flip_at] ^= 0xFF
    bad = tmp_path / "bad.rpzarc"
    bad.write_bytes(bytes(raw))

    from repeatzip.cross.verify import verify_bundle
    result = verify_bundle(str(bad))
    assert not result["ok"], f"expected verify to fail, got {result}"


def test_read_bundle_rejects_trailing_data(tmp_path):
    src = tmp_path / "in"; src.mkdir()
    _write_silent_wav(src / "a.wav")
    bundle = tmp_path / "p.rpzarc"
    archive(str(src), str(bundle), verbose=False)

    raw = bundle.read_bytes()
    bad = tmp_path / "bad.rpzarc"
    bad.write_bytes(raw + b"\x00" * 16)

    with pytest.raises(RepeatzipError, match="trailing"):
        _read_bundle(str(bad))


def test_read_bundle_rejects_count_mismatch(tmp_path):
    src = tmp_path / "in"; src.mkdir()
    _write_silent_wav(src / "a.wav")
    bundle = tmp_path / "p.rpzarc"
    archive(str(src), str(bundle), verbose=False)

    raw = bytearray(bundle.read_bytes())
    magic, version, _, count, meta_len, payload_len = HDR.unpack_from(raw, 0)
    HDR.pack_into(raw, 0, magic, version, 0, count + 5, meta_len, payload_len)
    bad = tmp_path / "bad.rpzarc"
    bad.write_bytes(bytes(raw))

    with pytest.raises(RepeatzipError, match="entry count"):
        _read_bundle(str(bad))


def test_extract_detects_tampered_payload(tmp_path):
    src = tmp_path / "in"; src.mkdir()
    _write_silent_wav(src / "a.wav")
    bundle = tmp_path / "p.rpzarc"
    archive(str(src), str(bundle), verbose=False)

    raw = bytearray(bundle.read_bytes())
    magic, version, _, count, meta_len, payload_len = HDR.unpack_from(raw, 0)
    flip_at = HDR.size + meta_len + payload_len // 2
    raw[flip_at] ^= 0xFF
    bad = tmp_path / "bad.rpzarc"
    bad.write_bytes(bytes(raw))

    out = tmp_path / "out"
    with pytest.raises(Exception):
        extract(str(bad), str(out), verbose=False)


def test_read_bundle_rejects_missing_sha256_in_v2(tmp_path):
    src = tmp_path / "in"; src.mkdir()
    _write_silent_wav(src / "a.wav")
    bundle = tmp_path / "p.rpzarc"
    archive(str(src), str(bundle), verbose=False)

    raw = bytearray(bundle.read_bytes())
    magic, version, _, count, meta_len, payload_len = HDR.unpack_from(raw, 0)
    off = HDR.size
    meta = json.loads(bytes(raw[off:off + meta_len]).decode("utf-8"))
    meta["entries"][0]["original_sha256"] = ""
    new_meta = json.dumps(meta, separators=(",", ":")).encode("utf-8")
    new_raw = bytearray()
    new_raw += HDR.pack(magic, version, 0, count, len(new_meta), payload_len)
    new_raw += new_meta
    new_raw += bytes(raw[off + meta_len:off + meta_len + payload_len])
    bad = tmp_path / "bad.rpzarc"
    bad.write_bytes(bytes(new_raw))

    with pytest.raises(RepeatzipError, match="sha256"):
        _read_bundle(str(bad))


def test_extract_refuses_entry_without_hash(tmp_path):
    src = tmp_path / "in"; src.mkdir()
    _write_silent_wav(src / "a.wav")
    bundle = tmp_path / "p.rpzarc"
    archive(str(src), str(bundle), verbose=False)

    raw = bytearray(bundle.read_bytes())
    magic, version, _, count, meta_len, payload_len = HDR.unpack_from(raw, 0)
    off = HDR.size
    meta = json.loads(bytes(raw[off:off + meta_len]).decode("utf-8"))
    # Rewrite as v1 so the parser accepts empty sha256.
    meta["entries"][0]["original_sha256"] = ""
    new_meta = json.dumps(meta, separators=(",", ":")).encode("utf-8")
    new_raw = bytearray()
    new_raw += HDR.pack(magic, 1, 0, count, len(new_meta), payload_len)
    new_raw += new_meta
    new_raw += bytes(raw[off + meta_len:off + meta_len + payload_len])
    bad = tmp_path / "bad_v1.rpzarc"
    bad.write_bytes(bytes(new_raw))

    out = tmp_path / "out"
    with pytest.raises(Exception):
        extract(str(bad), str(out), verbose=False)
