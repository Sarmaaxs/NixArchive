from __future__ import annotations
import argparse
import json
import os
import sys
from dataclasses import asdict

from repeatzip import __version__
from repeatzip.errors import RepeatzipError
from repeatzip.codec import encode, decode

def _parse_blocks(spec: str):
    from repeatzip.models import BlockSpec
    out = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split(":")
        if len(parts) != 3:
            raise argparse.ArgumentTypeError(
                f"--blocks entries must be src_start:src_end:dst_start, "
                f"got {chunk!r}"
            )
        try:
            a, b, c = (float(x) for x in parts)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Bad numbers in {chunk!r}")
        out.append(BlockSpec(a, b, c))
    return out

def _cmd_analyze(args):
    from repeatzip.analysis import detect_blocks
    blocks = detect_blocks(
        args.input,
        min_block_sec=args.min_length,
        similarity_threshold=args.threshold,
        verbose=args.verbose,
    )
    if args.json:
        print(json.dumps([asdict(b) for b in blocks], indent=2))
    else:
        if not blocks:
            print("No repeating blocks found.")
        for i, b in enumerate(blocks):
            print(f"  [{i:>3}] src {b.src_start_s:8.3f}s -> "
                  f"{b.src_end_s:8.3f}s "
                  f"({b.src_end_s - b.src_start_s:6.3f}s)  "
                  f"repeat @ {b.dst_start_s:8.3f}s")
        print(f"\n{len(blocks)} block(s).")
    return 0

def _cmd_compress(args):
    if args.blocks:
        blocks = _parse_blocks(args.blocks)
    else:
        blocks = None

    if args.output is None:
        # Keep the original extension so `song.wav` -> `song.wav.rpz`
        args.output = args.input + ".rpz"

    if os.path.exists(args.output) and not args.force:
        print(f"error: {args.output} exists (use --force)", file=sys.stderr)
        return 2

    encode(
        input_wav=args.input,
        output_rpz=args.output,
        blocks=blocks,
        auto_detect=args.auto,
        codec=args.codec,
        allow_delta=not args.no_delta,
        delta_min_gain=args.delta_gain,
        waveform_min_ncc=args.waveform_ncc,
        verbose=args.verbose,
        detector_threshold=args.threshold,
    )
    return 0

def _cmd_decompress(args):
    if args.output is None:
        # `song.wav.rpz` -> `song.wav.rpz.restored.wav`
        args.output = args.input + ".restored.wav"

    if os.path.exists(args.output) and not args.force:
        print(f"error: {args.output} exists (use --force)", file=sys.stderr)
        return 2

    decode(args.input, args.output, verbose=args.verbose)
    return 0

def _cmd_report(args):
    from repeatzip.analysis import detect_blocks
    from repeatzip.analysis.report import report

    blocks = detect_blocks(
        args.input,
        min_block_sec=args.min_length,
        similarity_threshold=args.threshold,
        verbose=args.verbose,
    )
    report(args.input, blocks)
    return 0

def _cmd_archive(args):
    from repeatzip.cross.archive import archive
    if args.output is None:
        args.output = os.path.basename(os.path.abspath(args.input)) + ".rpzarc"
    archive(args.input, args.output,
            ref_ncc=args.ref_ncc,
            workers=args.workers,
            verbose=args.verbose)
    return 0

def _cmd_extract(args):
    from repeatzip.cross.archive import extract
    if args.output is None:
        base = os.path.basename(args.input)
        if base.endswith(".rpzarc"):
            base = base[:-7]
        args.output = base + ".extracted"
    extract(args.input, args.output, verbose=args.verbose)
    return 0

def _cmd_delta(args):
    """Encode TARGET as a delta against REFERENCE; also verify."""
    import time
    from repeatzip.cross.archive import archive, extract
    import tempfile, shutil

    # Pack both into an archive: ref first (solo), then target (delta)
    tmp = tempfile.mkdtemp(prefix="repeatzip_delta_")
    try:
        shutil.copy(args.reference, os.path.join(tmp, "000_reference.wav"))
        shutil.copy(args.target, os.path.join(tmp, "001_target.wav"))
        out = args.output or (args.target + ".rpzarc")
        archive(tmp, out, ref_ncc=0.0, verbose=True)
        if args.verify:
            outdir = tempfile.mkdtemp(prefix="repeatzip_verify_")
            extract(out, outdir, verbose=False)
            ok = True
            for f in ("000_reference.wav", "001_target.wav"):
                a = os.path.join(tmp, f)
                b = os.path.join(outdir, f)
                if open(a, "rb").read() != open(b, "rb").read():
                    ok = False
                    print(f"  MISMATCH: {f}")
            print("  VERIFY:", "OK" if ok else "FAILED")
            shutil.rmtree(outdir, ignore_errors=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0

def _cmd_verify(args):
    from repeatzip.cross.verify import verify_bundle
    result = verify_bundle(args.input, verbose=args.verbose)
    for e in result["errors"]:
        print(f"  ERROR: {e}")
    print(f"entries   : {result['entries']}")
    print(f"status    : {'OK' if result['ok'] else 'FAILED'}")
    return 0 if result["ok"] else 1

def _cmd_benchmark(args):
    from repeatzip.cross.benchmark import (
        benchmark_folder, print_report, write_csv,
    )
    result = benchmark_folder(
        args.input,
        ref_ncc=args.ref_ncc,
        workers=args.workers,
        verbose=args.verbose,
    )
    print_report(result, args.input)
    if args.csv:
        write_csv(result, args.csv)
        print(f"csv written: {args.csv}")
    return 0 if result.all_roundtrip_ok else 2

def build_parser():
    p = argparse.ArgumentParser(
        prog="repeatzip",
        description="Lossless WAV compression via repeating-block deduplication.",
    )
    p.add_argument("--version", action="version",
                   version=f"repeatzip {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("analyze", help="Detect repeating blocks and print them.")
    a.add_argument("input", help="Input WAV file")
    a.add_argument("--min-length", type=float, default=2.0,
                   help="Minimum block length in seconds (default: 2.0)")
    a.add_argument("--threshold", type=float, default=0.85,
                   help="Similarity threshold 0-1 (default: 0.85)")
    a.add_argument("--json", action="store_true",
                   help="Emit JSON instead of a table")
    a.add_argument("-v", "--verbose", action="store_true")
    a.set_defaults(func=_cmd_analyze)

    r = sub.add_parser(
        "report",
        help="Show exactly which blocks the encoder would accept.",
    )
    r.add_argument("input", help="Input WAV file")
    r.add_argument("--min-length", type=float, default=2.0)
    r.add_argument("--threshold", type=float, default=0.75)
    r.add_argument("-v", "--verbose", action="store_true")
    r.set_defaults(func=_cmd_report)

    c = sub.add_parser("compress",
                       help="Deduplicate and write a .rpz container.")
    c.add_argument("input", help="Input WAV file")
    c.add_argument("-o", "--output", default=None,
                   help="Output .rpz path (default: <input>.rpz)")
    c.add_argument("--auto", action="store_true",
                   help="Auto-detect repeating blocks")
    c.add_argument("--blocks", default=None,
                   help="Manual blocks: src0:src1:dst,src0:src1:dst (seconds)")
    c.add_argument("--codec", choices=["auto", "raw", "flac"], default="auto",
                   help="Base payload codec (default: auto)")
    c.add_argument("--no-delta", action="store_true",
                   help="Disable delta encoding of near-identical blocks")
    c.add_argument("--threshold", type=float, default=0.75,
                   help="Detector similarity threshold (default: 0.75)")
    c.add_argument("--delta-gain", type=float, default=0.10,
                   help="Min fraction of bytes saved by delta (default: 0.10)")
    c.add_argument("--waveform-ncc", type=float, default=0.90,
                   help="Min waveform NCC for delta (default: 0.90)")
    c.add_argument("--force", action="store_true",
                   help="Overwrite output if it exists")
    c.add_argument("-v", "--verbose", action="store_true")
    c.set_defaults(func=_cmd_compress)

    d = sub.add_parser("decompress",
                       help="Rebuild the original WAV from a .rpz file.")
    d.add_argument("input", help="Input .rpz file")
    d.add_argument("-o", "--output", default=None,
                   help="Output WAV (default: <input>.restored.wav)")
    d.add_argument("--force", action="store_true",
                   help="Overwrite output if it exists")
    d.add_argument("-v", "--verbose", action="store_true")
    d.set_defaults(func=_cmd_decompress)

    arc = sub.add_parser("archive",
                          help="Pack a folder of WAVs with cross-file delta.")
    arc.add_argument("input", help="Folder containing .wav files")
    arc.add_argument("-o", "--output", default=None,
                     help="Output .rpzarc path")
    arc.add_argument("--ref-ncc", type=float, default=0.5,
                     help="Min file NCC to accept as delta reference")
    arc.add_argument("--workers", type=int, default=0,
                     help="Parallel load workers (0 = all cores)")
    arc.add_argument("-v", "--verbose", action="store_true")
    arc.set_defaults(func=_cmd_archive)

    ext = sub.add_parser("extract",
                         help="Unpack a .rpzarc bundle back to WAVs.")
    ext.add_argument("input", help="Input .rpzarc file")
    ext.add_argument("-o", "--output", default=None,
                     help="Output folder")
    ext.add_argument("-v", "--verbose", action="store_true")
    ext.set_defaults(func=_cmd_extract)

    dlt = sub.add_parser("delta",
                         help="Encode TARGET as delta against REFERENCE.")
    dlt.add_argument("reference", help="Reference WAV")
    dlt.add_argument("target", help="Target WAV to encode")
    dlt.add_argument("-o", "--output", default=None,
                     help="Output .rpzarc path")
    dlt.add_argument("--no-verify", dest="verify", action="store_false",
                     default=True)
    dlt.add_argument("-v", "--verbose", action="store_true")
    dlt.set_defaults(func=_cmd_delta)

    vf = sub.add_parser("verify",
                        help="Check bundle integrity without extracting.")
    vf.add_argument("input", help="Input .rpzarc file")
    vf.add_argument("-v", "--verbose", action="store_true")
    vf.set_defaults(func=_cmd_verify)

    bm = sub.add_parser("benchmark",
                        help="Measure savings on a folder of related WAVs.")
    bm.add_argument("input", help="Folder containing .wav files")
    bm.add_argument("--ref-ncc", type=float, default=0.5)
    bm.add_argument("--workers", type=int, default=0)
    bm.add_argument("--csv", default=None,
                    help="Write per-file results to a CSV file")
    bm.add_argument("-v", "--verbose", action="store_true")
    bm.set_defaults(func=_cmd_benchmark)

    return p

def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except RepeatzipError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130

if __name__ == "__main__":
    raise SystemExit(main())
