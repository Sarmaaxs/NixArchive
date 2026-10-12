# NixArchive

Lossless compression for WAV file collections that contain related recordings.

## The problem

Archives of audio sessions often contain multiple versions of the same material: a master and its radio edit, a mix and its instrumental, dozens of velocity layers of the same drum hit. FLAC compresses each file independently and has no way to exploit the fact that the files share most of their content. Byte-level deduplication tools fail too, because a single-sample gain change breaks every byte match.

## What this does

NixArchive packs a folder of related WAV files into a single `.rpzarc` bundle. It finds files that share waveform content, computes the alignment and gain offset between them, and stores the difference. Reconstruction is bit-exact. A file that differs from its reference only by a trim or a volume adjustment compresses to a fraction of what FLAC would produce on its own.

## Measured results

Three versions of a real commercial master: original, 3 dB quieter, and one second shorter.

| | Size |
|---|---|
| Original WAVs | 120.7 MB |
| Independent FLAC | 68.5 MB |
| NixArchive bundle | **25.1 MB** |

63.4% smaller than independent FLAC. All three files round-trip byte-exact.

On files with no waveform overlap, the tool correctly stores them independently and produces the same output FLAC would.

## Install

Requires Python 3.10 or later and libsndfile.

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -e ".[flac]"

Debian and Ubuntu:

    sudo apt install libsndfile1

## Usage

Pack a folder:

    nixarchive archive <folder> -o <output.rpzarc>

Extract a bundle:

    nixarchive extract <output.rpzarc> -o <folder>

Verify integrity without extracting:

    nixarchive verify <output.rpzarc>

Measure savings on a folder before committing to it:

    nixarchive benchmark <folder> --csv results.csv

## Commands

| Command | Purpose |
|---|---|
| `archive` | Pack a folder of WAVs into a bundle |
| `extract` | Rebuild original files from a bundle |
| `verify` | Check bundle integrity without writing files |
| `benchmark` | Report savings per file and overall |
| `compress` | Single-file compression using internal repetition |
| `decompress` | Rebuild a single file |
| `delta` | Encode one file as a difference against another |
| `analyze` | Show repeating regions detected in one file |
| `report` | Show which regions the encoder would accept |

## How it works

For each file, NixArchive searches previously-loaded files for the best alignment (sample offset) and gain (linear scale) match. The search uses a coarse FFT pass on decimated envelopes followed by a fine normalized-cross-correlation scan at sample resolution.

If a match clears the similarity threshold, the file is encoded as:

    residual = target[lag:] - round(gain * reference)

The residual and any leading or trailing raw frames are FLAC-compressed. If no match works, the file is stored solo. The encoder verifies every candidate by decoding it and comparing to the target; a candidate that does not reconstruct exactly is rejected.

## Format

`.rpzarc` version 2:

    4 bytes    magic "RPZA"
    1 byte     version
    1 byte     reserved
    2 bytes    entry count
    8 bytes    metadata length
    8 bytes    payload length
    N bytes    metadata (JSON)
    M bytes    concatenated FLAC payloads

Each entry records the file name, storage mode (solo or delta), codec, original size, frame count, sample rate, channels, sample width, payload offset and length, the original WAV header and trailer as base64, and for delta entries the reference name, alignment lag, gain, and region sizes.

## Limitations

The tool only helps when files share waveform samples. Files that have been through different lossy codecs, different sample rates, or different mastering chains will not match, and the tool will store them independently.

Other current limits:

- All processing is in-memory; no streaming
- Archive format caps at 65,535 entries
- Deltas are one level deep; a delta cannot itself be a reference
- Reference selection is O(N²) in the number of files; a cheap RMS prefilter helps but large folders are slow

## License

Proprietary. See LICENSE.
