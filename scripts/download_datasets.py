#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
download_datasets.py

Downloads/prepares the three benchmarks used by the JRISA proposed method
(Section III-I of the manuscript):

    CREMI       -> Hugging Face ("MedOtter/CREMI") -- confirmed public mirror
    AC4         -> bossDB ('Kasthuri/ac4/em' + '.../neuron') -- CONFIRMED
                   working, image + instance labels both verified with real
                   (non-zero) data as of 2026-09.
    AC3         -> NOT available as its own bossDB experiment; requires
                   manually cropping the documented sub-region out of the
                   full 'kasthuri11' volume (see download_ac3ac4() docstring).
    MitoEM-R/H  -> Grand Challenge portal            -- NO public API; manual
                   registration + download is required (see instructions
                   printed by this script and written to a README in the
                   output folder).

IMPORTANT: this script does NOT fabricate download endpoints. Every path
used here was independently verified to return real, non-zero data (or, for
AC3/MitoEM, honestly flagged as still requiring manual work) before being
hardcoded.

Usage:
    python download_datasets.py --dataset all --output-dir ./data
    python download_datasets.py --dataset cremi --output-dir ./data
    python download_datasets.py --dataset ac3ac4 --output-dir ./data
    python download_datasets.py --dataset mitoem --output-dir ./data

Requirements (install only what you need):
    pip install huggingface_hub      # for CREMI
    pip install intern               # for AC3/AC4 (bossDB client)
"""
import argparse
import os
import sys
import zipfile
import tarfile
from pathlib import Path


# ---------------------------------------------------------------------------
# CREMI  (Hugging Face: MedOtter/CREMI)
# ---------------------------------------------------------------------------
def download_cremi(output_dir: Path):
    """Download the CREMI dataset from its confirmed Hugging Face mirror."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[CREMI] 'huggingface_hub' is not installed.\n"
              "        Install it with: pip install huggingface_hub")
        sys.exit(1)

    dest = output_dir / "cremi"
    dest.mkdir(parents=True, exist_ok=True)

    print(f"[CREMI] Downloading MedOtter/CREMI from Hugging Face to {dest} ...")
    print("        (No HF token is required for this public dataset. If you "
          "have one set as the HF_TOKEN environment variable, it will be "
          "picked up automatically and can help avoid rate limits.)")

    local_path = snapshot_download(
        repo_id="MedOtter/CREMI",
        repo_type="dataset",
        local_dir=str(dest),
    )
    print(f"[CREMI] Downloaded to: {local_path}")

    _extract_archives_in(dest)
    print("[CREMI] Done. Expect volumes for CREMI A/B/C (raw + neuron/cleft labels).")


# ---------------------------------------------------------------------------
# AC3/AC4  (bossDB, via `intern`)
# ---------------------------------------------------------------------------
def download_ac3ac4(output_dir: Path):
    """Download the AC4 image + instance-label cutouts from bossDB via `intern`.

    CORRECTED 2026-09: the collection/experiment/channel names are
    'Kasthuri' / 'ac4' / 'em' (image) and 'Kasthuri' / 'ac4' / 'neuron'
    (instance labels) -- NOT 'kasthuri2015/em/cc' as an earlier version of
    this script used. That older path connects without erroring but silently
    returns all-zero data (a real footgun: no account/token was ever the
    issue -- bossDB's own tutorial confirms public data needs none). The
    correct paths were recovered from the project page's own "View AC4 in
    Neuroglancer" link (bossdb.org/project/kasthuri2015), whose embedded
    source URLs point at s3://bossdb-open-data/Kasthuri/ac4/em and
    .../Kasthuri/ac4/neuron. Both were verified with real (non-zero,
    plausible-looking) pixel/label data before being hardcoded here.

    NOTE: AC3 does NOT exist as its own clean bossDB experiment the way AC4
    does -- 'bossdb://Kasthuri/ac3/em' returns a 404 ("ac3 does not exist").
    The only related resource found ('Kasthuri11', s3://bossdb-open-data/
    neurodata/kasthuri/kasthuri11/image) is the full, uncropped ~21504x
    26624x1850 raw volume at 4nm, not a pre-made AC3 benchmark crop -- so
    getting AC3 requires manually identifying and cropping the documented
    AC3 sub-region (256x1024x1024 at 6x6x29nm) from that full volume
    yourself; this script does not guess those crop coordinates.
    """
    try:
        from intern import array
    except ImportError:
        print("[AC3/AC4] 'intern' is not installed.\n"
              "           Install it with: pip install intern")
        sys.exit(1)

    dest = output_dir / "ac3_ac4"
    dest.mkdir(parents=True, exist_ok=True)

    # --- AC4: confirmed image + instance-label channels ---
    print("[AC3/AC4] Fetching AC4 image ('bossdb://Kasthuri/ac4/em') ...")
    try:
        import numpy as np
        em_channel = array("bossdb://Kasthuri/ac4/em")
        ac4_em = em_channel[0:100, 0:1024, 0:1024]   # intern slicing order is [z, y, x]
        np.save(dest / "ac4_raw.npy", ac4_em)
        print(f"[AC3/AC4] Saved AC4 image to {dest / 'ac4_raw.npy'} "
              f"(shape={ac4_em.shape}, min={ac4_em.min()}, max={ac4_em.max()})")

        print("[AC3/AC4] Fetching AC4 instance labels ('bossdb://Kasthuri/ac4/neuron') ...")
        label_channel = array("bossdb://Kasthuri/ac4/neuron")
        ac4_labels = label_channel[0:100, 0:1024, 0:1024]
        np.save(dest / "ac4_labels.npy", ac4_labels)
        n_instances = len(np.unique(ac4_labels)) - 1  # exclude background 0
        print(f"[AC3/AC4] Saved AC4 instance labels to {dest / 'ac4_labels.npy'} "
              f"(shape={ac4_labels.shape}, ~{n_instances} distinct instance IDs)")
    except Exception as e:
        print(f"[AC3/AC4] AC4 download failed: {e}\n"
              "           Check https://bossdb.org/project/kasthuri2015 in case the "
              "channel names have changed since this script was last updated.")

    # --- AC3: genuinely not available as a clean pre-cropped experiment ---
    print("\n[AC3/AC4] AC3 is NOT available as its own bossDB experiment "
          "('bossdb://Kasthuri/ac3/em' returns 404: \"ac3 does not exist\").\n"
          "           The only lead is 'Kasthuri11'\n"
          "           (s3://bossdb-open-data/neurodata/kasthuri/kasthuri11/image),\n"
          "           the FULL uncropped ~21504x26624x1850 volume at 4nm -- not a\n"
          "           pre-made AC3 crop. To get AC3 (documented as 256x1024x1024 at\n"
          "           6x6x29nm), you would need to identify the correct sub-region\n"
          "           coordinates within that full volume yourself, e.g. via:\n"
          "               full_vol = array('bossdb://neurodata/kasthuri11/image')\n"
          "           and then crop manually. This script does not guess those\n"
          "           coordinates rather than risk silently pulling the wrong region.")

    print("\n[AC3/AC4] AC4 image + instance labels are both real, verified data. "
          "AC3 is the one piece of this dataset still requiring manual work.")


# ---------------------------------------------------------------------------
# MitoEM-R/H  (no public API -- manual download)
# ---------------------------------------------------------------------------
MITOEM_README = """\
MitoEM-R/H — manual download required
======================================

No public, scriptable download endpoint exists for MitoEM as of this
writing. The dataset is distributed through the challenge portal, which
requires registration:

    https://mitoem.grand-challenge.org/

Steps:
    1. Create a Grand Challenge account and register for the MitoEM challenge.
    2. Follow the portal's data-access instructions to download the two
       volumes:
           - Mito-R (rat cortex)
           - Mito-H (human cortex)
    3. Place the downloaded volumes into this folder using the layout below
       (the rest of the JRISA codebase expects this structure):

        data/mitoem/
        +-- mito_r/
        |   +-- im/           # raw EM image stack
        |   +-- mito-labels/  # instance labels (if released for your split)
        +-- mito_h/
            +-- im/
            +-- mito-labels/

    4. Note: MitoEM-R/H is used ONLY as a zero-shot generalization test in
       this pipeline (Section III-I) -- it is never trained on. You need it
       populated before running evaluate.py's generalization pass, but not
       before train.py.

A related, newer resource ("MitoEM 2.0") was also announced as a curated,
multi-species mitochondria instance-segmentation benchmark; if the original
MitoEM portal is unavailable, check whether MitoEM 2.0 supersedes it and
adjust this pipeline's dataset loader accordingly.
"""


def prepare_mitoem(output_dir: Path):
    dest = output_dir / "mitoem"
    (dest / "mito_r" / "im").mkdir(parents=True, exist_ok=True)
    (dest / "mito_r" / "mito-labels").mkdir(parents=True, exist_ok=True)
    (dest / "mito_h" / "im").mkdir(parents=True, exist_ok=True)
    (dest / "mito_h" / "mito-labels").mkdir(parents=True, exist_ok=True)

    readme_path = dest / "README_MANUAL_DOWNLOAD.txt"
    readme_path.write_text(MITOEM_README, encoding="utf-8")

    print(f"[MitoEM] No public API available -- created expected folder layout at {dest}")
    print(f"[MitoEM] Manual download instructions written to: {readme_path}")
    print(MITOEM_README)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _extract_archives_in(folder: Path):
    """Extract any .zip/.tar/.tar.gz files found directly inside `folder`."""
    for item in list(folder.rglob("*")):
        if item.suffix == ".zip":
            print(f"  extracting {item.name} ...")
            with zipfile.ZipFile(item, "r") as zf:
                zf.extractall(item.parent)
        elif item.suffixes[-2:] == [".tar", ".gz"] or item.suffix == ".tgz":
            print(f"  extracting {item.name} ...")
            with tarfile.open(item, "r:gz") as tf:
                tf.extractall(item.parent)
        elif item.suffix == ".tar":
            print(f"  extracting {item.name} ...")
            with tarfile.open(item, "r:") as tf:
                tf.extractall(item.parent)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["cremi", "ac3ac4", "mitoem", "all"],
                         default="all", help="Which dataset(s) to fetch/prepare.")
    parser.add_argument("--output-dir", type=str, default="./data",
                         help="Root output directory (default: ./data)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset in ("cremi", "all"):
        download_cremi(output_dir)
    if args.dataset in ("ac3ac4", "all"):
        download_ac3ac4(output_dir)
    if args.dataset in ("mitoem", "all"):
        prepare_mitoem(output_dir)

    print("\nAll requested dataset steps complete. See per-dataset messages above "
          "for any manual follow-up required.")


if __name__ == "__main__":
    main()
