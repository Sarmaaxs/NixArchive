"""Human-readable summary of what the encoder will do to a file."""
from __future__ import annotations
from typing import List

from repeatzip.io.wav_io import read_wav, sec_to_frame
from repeatzip.models import BlockSpec

def report(wav_path: str, blocks: List[BlockSpec]) -> None:
    """Print a table showing which blocks the encoder would accept."""
    samples, fmt = read_wav(wav_path)
    total_frames = samples.shape[0]
    sr = fmt.sample_rate

    print(f"file        : {wav_path}")
    print(f"format      : {fmt.sample_rate} Hz, {fmt.channels} ch, "
          f"{fmt.sampwidth*8}-bit PCM")
    print(f"duration    : {total_frames/sr:.2f} s "
          f"({total_frames} frames)")
    print(f"original    : {total_frames * fmt.channels * fmt.sampwidth:,} B")
    print()
    print(f"{'#':>3}  {'src (s)':>15}  {'dst (s)':>15}  "
          f"{'len (s)':>10}  {'bytes':>12}  status")
    print("-" * 90)

    kept = []
    dropped_ranges = []
    total_saved = 0

    for idx, b in enumerate(blocks):
        s = sec_to_frame(b.src_start_s, fmt)
        e = sec_to_frame(b.src_end_s, fmt)
        d = sec_to_frame(b.dst_start_s, fmt)
        length = e - s

        status = "OK"
        saved = 0
        if length <= 0:
            status = "empty"
        elif d < 0 or d + length > total_frames:
            status = "dst out of range"
        else:
            a = samples[s:s + length]
            bb = samples[d:d + length]
            if a.shape != bb.shape:
                status = "shape mismatch"
            else:
                diff = (a != bb)
                if diff.ndim > 1:
                    diff = diff.any(axis=1)
                if diff.any():
                    # find longest contiguous equal run
                    best = cur = 0
                    for k in range(length):
                        if not diff[k]:
                            cur += 1
                            best = max(best, cur)
                        else:
                            cur = 0
                    min_run = int(0.25 * sr)
                    if best >= min_run:
                        status = f"trim to {best/sr:.3f}s"
                        saved = best * fmt.channels * fmt.sampwidth
                    else:
                        status = f"no run >= 0.25s (best {best} frames)"
                else:
                    saved = length * fmt.channels * fmt.sampwidth

        if saved > 0:
            kept.append(b)
            total_saved += saved

        print(f"{idx:>3}  {b.src_start_s:>7.3f}-{b.src_end_s:<7.3f}  "
              f"{b.dst_start_s:>7.3f}          "
              f"{length/sr:>9.3f}  "
              f"{length*fmt.channels*fmt.sampwidth:>12,}  {status}")

    print("-" * 90)
    print(f"blocks offered  : {len(blocks)}")
    print(f"blocks usable   : {len(kept)}")
    print(f"raw dedup saved : {total_saved:,} B "
          f"({100*total_saved/(total_frames*fmt.channels*fmt.sampwidth):.2f}%)")
    print()
    print("note: the encoder will re-verify each block; final savings may")
    print("      differ slightly. Run `compress -v` for the authoritative "
          "report.")
