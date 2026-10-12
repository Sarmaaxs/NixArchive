from __future__ import annotations
import base64
import hashlib
from typing import Dict, List

import numpy as np

from repeatzip.io.flac_io import decode_flac
from repeatzip.io.wav_io import samples_to_pcm_bytes
from repeatzip.models import AudioFormat
from .delta_codec import decode_against


def verify_bundle(bundle_path: str, *, verbose=False) -> Dict:
    from .archive import _read_bundle
    meta, payload = _read_bundle(bundle_path)

    entries = meta["entries"]

    def pay(e):
        return payload[e["payload_offset"]:
                       e["payload_offset"] + e["payload_len"]]

    decoded_samples = {}
    decoded_hashes = {}
    remaining = list(entries)
    errors: List[str] = []

    while remaining:
        progressed = False
        for e in list(remaining):
            name = e["name"]
            try:
                if e["kind"] == "solo":
                    arr, _ = decode_flac(pay(e))
                elif e["kind"] == "delta":
                    ref = e["reference"]
                    if ref not in decoded_samples:
                        continue
                    fmt = AudioFormat(
                        sample_rate=e["sample_rate"],
                        channels=e["channels"],
                        sampwidth=e["sampwidth"],
                    )
                    arr = decode_against(
                        decoded_samples[ref], pay(e),
                        {"mode": e.get("delta_mode", "delta"),
                         "lag": e.get("lag", 0),
                         "gain": e.get("gain", 1.0),
                         "overlap_frames": e["overlap_frames"],
                         "leading_frames": e.get("leading_frames", 0),
                         "trailing_frames": e.get("tail_frames", 0)},
                        fmt,
                    )
                else:
                    errors.append(f"{name}: unknown kind {e['kind']!r}")
                    remaining.remove(e)
                    progressed = True
                    continue

                decoded_samples[name] = arr

                prefix = base64.b64decode(e["wav_header_b64"])
                tb64 = e.get("wav_trailer_b64", "")
                trailer = base64.b64decode(tb64) if tb64 else b""
                pcm = samples_to_pcm_bytes(arr, e["sampwidth"])
                full_bytes = prefix + pcm + trailer
                sha = hashlib.sha256(full_bytes).hexdigest()
                decoded_hashes[name] = sha[:16]

                stored = e.get("original_sha256", "")
                if stored:
                    if sha != stored:
                        errors.append(
                            f"{name}: integrity hash mismatch "
                            f"({sha[:12]} vs {stored[:12]})")
                    elif verbose:
                        print(f"  {name:>40}  {e['kind']:<6}"
                              + (f"  vs {e['reference']}"
                                 if e["reference"] else "")
                              + f"  sha256={sha[:16]} OK")
                elif verbose:
                    print(f"  {name:>40}  {e['kind']:<6}"
                          + (f"  vs {e['reference']}"
                             if e["reference"] else "")
                          + f"  sha256={sha[:16]} (no stored hash)")

                remaining.remove(e)
                progressed = True
            except Exception as exc:
                errors.append(f"{name}: {exc}")
                remaining.remove(e)
                progressed = True

        if not progressed:
            names = [e["name"] for e in remaining]
            errors.append(f"unresolvable chain: {names}")
            break

    return {"ok": not errors, "entries": len(entries),
            "errors": errors, "hashes": decoded_hashes}
