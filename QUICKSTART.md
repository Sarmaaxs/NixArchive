# Quick start

Two commands to see what the tool does.

## Install

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -e ".[flac]"

## Test with your own files

Put three or more related WAV files in a folder, then:

    repeatzip archive that_folder/ -o bundle.rpzarc --verbose
    repeatzip verify  bundle.rpzarc --verbose
    repeatzip extract bundle.rpzarc -o restored/

Compare the size of `bundle.rpzarc` to the total size of the input
folder. Compare the files in `restored/` to the originals. They are
byte-identical.

## What to look for in the output

Each entry in the archive log shows one of two modes:

    solo    - this file was stored independently
    delta   - this file was encoded against another

The delta entries are where the savings come from. A well-matched pair
typically lands between 3x and 1000x compression depending on how
similar the two files are.
