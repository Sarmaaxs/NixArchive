from __future__ import annotations
import hashlib
import json
import struct

import pytest

from repeatzip.codec import encode, decode
from repeatzip.models import BlockSpec
from repeatzip.io.rpz_format import CODEC_FLAC
from repeatzip.io import rpz_format
from tests.test_roundtrip import make_test_wav, read_frames

def _have_sf():
    try:
        import soundfile  # noqa: F401
        return True
    except Exception:
        return False

pytestmark = pytest.mark.skipif(not _have_sf(),
                                reason="soundfile not installed")

def test_flac_roundtrip_explicit(tmp_path):
    wav = tmp_path / "f.wav"
    make_test_wav(wav)
    blocks = [BlockSpec(2.0, 6.0, 8.0), BlockSpec(2.0, 6.0, 14.0)]
    rpz = tmp_path / "f.rpz"
    out = tmp_path / "f_out.wav"
    encode(str(wav), str(rpz), blocks=blocks, codec="flac")
    hdr, _, _, _ = rpz_format.unpack(str(rpz))
    assert hdr.codec == CODEC_FLAC
    assert hdr.version == 3
    decode(str(rpz), str(out))
    assert read_frames(wav) == read_frames(out)

def test_auto_picks_smaller(tmp_path):
    wav = tmp_path / "auto.wav"
    make_test_wav(wav)
    blocks = [BlockSpec(2.0, 6.0, 8.0), BlockSpec(2.0, 6.0, 14.0)]
    raw = tmp_path / "raw.rpz"
    auto = tmp_path / "auto.rpz"
    encode(str(wav), str(raw), blocks=blocks, codec="raw")
    encode(str(wav), str(auto), blocks=blocks, codec="auto")
    assert auto.stat().st_size <= raw.stat().st_size
    out = tmp_path / "auto_out.wav"
    decode(str(auto), str(out))
    assert read_frames(wav) == read_frames(out)

def test_v1_container_still_decodes(tmp_path):
    wav = tmp_path / "v1.wav"
    make_test_wav(wav)

    from repeatzip.io.wav_io import read_wav
    arr, fmt = read_wav(str(wav))
    pcm = arr.tobytes()
    md5_hex = hashlib.md5(arr.tobytes()).hexdigest()

    meta = json.dumps(
        {"format": "repeatzip", "version": 1, "pointers": []},
        separators=(",", ":"),
    ).encode()
    v1_hdr = struct.Struct("<4sBBHIBB2xQQQQ16x")
    header = v1_hdr.pack(
        b"RPZ\x01", 1, 0x01, 0,
        fmt.sample_rate, fmt.channels, fmt.sampwidth,
        arr.shape[0], arr.shape[0],
        len(meta), len(pcm),
    )
    v1_path = tmp_path / "v1.rpz"
    with open(v1_path, "wb") as f:
        f.write(header); f.write(meta); f.write(pcm)
        f.write(bytes.fromhex(md5_hex))

    out = tmp_path / "v1_out.wav"
    decode(str(v1_path), str(out))
    assert read_frames(wav) == read_frames(out)
