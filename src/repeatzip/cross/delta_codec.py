"""Delta coding with alignment and gain prediction.

Payload = FLAC([leading_raw | residual | trailing_raw])

Reconstruction: decoder computes predicted = round(gain * ref_slice),
then output = arith.reconstruct(predicted, residual). The encoder
verifies the round-trip by actually decoding and comparing; if any
candidate fails, it falls back to plain delta and then to raw.
"""
from __future__ import annotations
import numpy as np

from repeatzip.io.flac_io import encode_flac, decode_flac
from repeatzip.codec import arith

def _slices(lag: int, n_ref: int, n_tgt: int):
    """(tgt_start, ref_start, overlap) for the requested lag."""
    if lag >= 0:
        return lag, 0, min(n_ref, n_tgt - lag)
    else:
        return 0, -lag, min(n_ref + lag, n_tgt)

def _build_payload(residual, leading, trailing, fmt) -> bytes:
    parts = []
    if len(leading):
        parts.append(leading.astype(np.int32, copy=False))
    parts.append(residual.astype(np.int32, copy=False))
    if len(trailing):
        parts.append(trailing.astype(np.int32, copy=False))
    combined = np.concatenate(parts, axis=0)
    return encode_flac(combined, fmt)

def _try_params(ref, tgt, fmt, lag, gain):
    tgt_s, ref_s, overlap = _slices(lag, len(ref), len(tgt))
    if overlap < 512:
        return None

    r_slice = ref[ref_s:ref_s + overlap]
    t_slice = tgt[tgt_s:tgt_s + overlap]

    predicted = np.round(gain * r_slice.astype(np.float64)).astype(np.int32)
    residual = arith.residual(t_slice, predicted, fmt.sampwidth)

    leading = tgt[:tgt_s]
    trailing = tgt[tgt_s + overlap:]
    payload = _build_payload(residual, leading, trailing, fmt)

    return {
        "payload": payload,
        "mode": "delta",
        "lag": int(lag),
        "gain": float(gain),
        "overlap_frames": int(overlap),
        "leading_frames": int(tgt_s),
        "trailing_frames": int(len(trailing)),
    }

def encode_against(ref, tgt, fmt, lag: int = 0, gain: float = 1.0):
    """Encode tgt as delta against ref. Returns (payload, info).

    Tries the proposed (lag, gain) first, then plain delta (0, 1.0),
    then raw. Every candidate is verified by a real decode round-trip.
    """
    candidates = []
    if lag != 0 or gain != 1.0:
        candidates.append((lag, gain))
    candidates.append((0, 1.0))

    best = None
    for L, G in candidates:
        info = _try_params(ref, tgt, fmt, L, G)
        if info is None:
            continue
        payload = info.pop("payload")
        try:
            decoded = decode_against(ref, payload, info, fmt)
        except Exception:
            continue
        if decoded.shape != tgt.shape or not np.array_equal(decoded, tgt):
            continue
        if best is None or len(payload) < len(best[0]):
            best = (payload, info)

    if best is not None:
        return best

    payload = encode_flac(tgt, fmt)
    info = {"mode": "raw", "lag": 0, "gain": 1.0,
            "overlap_frames": 0, "leading_frames": 0,
            "trailing_frames": int(len(tgt))}
    return payload, info

def decode_against(ref, payload: bytes, info: dict, fmt):
    if info.get("mode") == "raw":
        arr, _ = decode_flac(payload)
        return arr

    lag = int(info["lag"])
    gain = float(info["gain"])
    overlap = int(info["overlap_frames"])
    leading_len = int(info["leading_frames"])
    trailing_len = int(info["trailing_frames"])

    combined, _ = decode_flac(payload)
    leading = combined[:leading_len]
    residual = combined[leading_len:leading_len + overlap]
    trailing = combined[leading_len + overlap:
                        leading_len + overlap + trailing_len]

    ref_s = 0 if lag >= 0 else -lag
    r_slice = ref[ref_s:ref_s + overlap]
    predicted = np.round(gain * r_slice.astype(np.float64)).astype(np.int32)
    recon = arith.reconstruct(predicted, residual, fmt.sampwidth)

    parts = []
    if leading_len:
        parts.append(leading[:leading_len])
    parts.append(recon)
    if trailing_len:
        parts.append(trailing[:trailing_len])
    return np.concatenate(parts, axis=0)
