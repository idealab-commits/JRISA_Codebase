#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
prepare_data.py -- ONE command that gets every dataset ready for training.

    python scripts/prepare_data.py

Run it from the project root (the folder that contains scripts/ and src/).
It is safe to re-run: any step whose output already exists is skipped.

What it does, in order:

  STEP 1  CREMI  download   Hugging Face "MedOtter/CREMI" -> data/cremi/train/sample_{A,B,C}.hdf
  STEP 2  CREMI  convert    reads volumes/raw + volumes/labels/clefts from each .hdf
                            - clefts value 2^64-1 (CREMI's "unlabeled" marker) -> 0 (background)
                            - remaining cleft ids remapped to 1..K
                            - split along z, per volume, exactly as the manuscript (Sec. III-I):
                                z  0:70  -> data/cremi_train/
                                z 70:100 -> data/cremi_val/
                                z 100:125-> data/cremi_test/
  STEP 3  AC4    download   bossDB  Kasthuri/ac4/em (image) + Kasthuri/ac4/neuron (labels)
                            -> data/ac3_ac4/ac4_{raw,labels,sem_gt}.npy
  STEP 4  MitoEM download   rat + human, slices 400-499 (the labelled val range, = [26]'s test
                            split), read straight out of the remote zips so only ~1.5-2.5 GB per
                            volume is downloaded instead of the full 14-15 GB archives
                            -> data/mitoem/mito_{r,h}_{raw,labels,sem_gt}.npy
  STEP 5  CHECK             prints shape / instance count / foreground % for every file and
                            flags anything that looks wrong (e.g. an all-zero volume)

Every volume is saved as up to three files, which is what src/train.py and src/evaluate.py read:
    <name>_raw.npy      uint8          image
    <name>_labels.npy   int32/uint16   instance ids, 0 = background, 1..K = objects
    <name>_sem_gt.npy   uint8          1 where labels > 0, else 0   (skipped with --no-sem-gt)

Options:
    --skip-cremi   --skip-ac4   --skip-mitoem      skip a dataset
    --no-sem-gt                                    write only _raw.npy and _labels.npy
    --mitoem-slices START END                      MitoEM z-range (default 400 500)
    --data-dir DIR                                 default: ./data

AC4 + MitoEM, raw + labels only (what run_data_setup.bat does):
    python scripts/prepare_data.py --skip-cremi --no-sem-gt
"""
import argparse
import sys
from pathlib import Path

import numpy as np

CREMI_SAMPLES = ["A", "B", "C"]
CREMI_SENTINEL = np.uint64(2**64 - 1)          # CREMI "unlabeled" marker in clefts
CREMI_SPLITS = {"cremi_train": (0, 70), "cremi_val": (70, 100), "cremi_test": (100, 125)}

AC4_IMAGE_URI = "bossdb://Kasthuri/ac4/em"
AC4_LABEL_URI = "bossdb://Kasthuri/ac4/neuron"
AC4_SHAPE = (100, 1024, 1024)


def banner(text):
    print("\n" + "=" * 72 + f"\n{text}\n" + "=" * 72, flush=True)


def need(module, pip_name=None):
    try:
        return __import__(module)
    except ImportError:
        print(f"  ERROR: '{module}' is not installed. Run:  pip install {pip_name or module}")
        sys.exit(1)


def remap_dense(labels, compact=False):
    """0 stays 0; every other distinct id becomes 1..K. Done slice by slice to keep memory low.
    Output is int32, or uint16 when compact=True and K fits (halves the size of large volumes)."""
    ids = np.array([0], dtype=labels.dtype)
    for z in range(labels.shape[0]):
        ids = np.union1d(ids, np.unique(labels[z]))
    count = len(ids) - 1
    dtype = np.uint16 if (compact and count < 65535) else np.int32
    out = np.empty(labels.shape, dtype=dtype)
    for z in range(labels.shape[0]):
        out[z] = np.searchsorted(ids, labels[z])
    return out, count


def save_volume(folder, name, raw, labels, with_sem_gt=True):
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / f"{name}_raw.npy", raw.astype(np.uint8))
    np.save(folder / f"{name}_labels.npy", labels.astype(np.int32))
    if with_sem_gt:
        np.save(folder / f"{name}_sem_gt.npy", (labels > 0).astype(np.uint8))


def outputs_exist(folder, name, with_sem_gt=True):
    suffixes = ("raw", "labels", "sem_gt") if with_sem_gt else ("raw", "labels")
    return all((folder / f"{name}_{s}.npy").exists() for s in suffixes)


# ---------------------------------------------------------------------------
# STEP 1 + 2 : CREMI
# ---------------------------------------------------------------------------
def prepare_cremi(data_dir: Path, with_sem_gt=True):
    banner("STEP 1  CREMI download (Hugging Face: MedOtter/CREMI)")
    src_dir = data_dir / "cremi"
    hdf_files = [src_dir / "train" / f"sample_{s}.hdf" for s in CREMI_SAMPLES]

    if all(f.exists() for f in hdf_files):
        print("  already downloaded -- skipping")
    else:
        hf = need("huggingface_hub")
        hf.snapshot_download(repo_id="MedOtter/CREMI", repo_type="dataset", local_dir=str(src_dir))
        missing = [f.name for f in hdf_files if not f.exists()]
        if missing:
            print(f"  ERROR: download finished but these files are missing: {missing}")
            sys.exit(1)
        print("  downloaded sample_A/B/C.hdf")

    banner("STEP 2  CREMI convert: clefts labels, sentinel -> 0, z-split 70/30/25 per volume")
    h5py = need("h5py")
    for s in CREMI_SAMPLES:
        name = f"sample_{s}"
        if all(outputs_exist(data_dir / split, name, with_sem_gt) for split in CREMI_SPLITS):
            print(f"  {name}: already converted -- skipping")
            continue

        with h5py.File(src_dir / "train" / f"{name}.hdf", "r") as f:
            raw = f["volumes/raw"][:]
            clefts = f["volumes/labels/clefts"][:]

        n_sentinel = int((clefts == CREMI_SENTINEL).sum())
        clefts[clefts == CREMI_SENTINEL] = 0
        labels, k = remap_dense(clefts)
        del clefts
        print(f"  {name}: raw {raw.shape}, {k} cleft instances, "
              f"{n_sentinel:,} unlabeled voxels set to background")

        if raw.shape[0] != 125:
            print(f"  WARNING: expected 125 z-slices, got {raw.shape[0]} -- split may not match the paper")

        for split, (z0, z1) in CREMI_SPLITS.items():
            save_volume(data_dir / split, name, raw[z0:z1], labels[z0:z1], with_sem_gt)
        print(f"  {name}: saved to " + ", ".join(CREMI_SPLITS))


# ---------------------------------------------------------------------------
# STEP 3 : AC4
# ---------------------------------------------------------------------------
def prepare_ac4(data_dir: Path, with_sem_gt=True):
    banner("STEP 3  AC4 download (bossDB: Kasthuri/ac4/em + Kasthuri/ac4/neuron)")
    out_dir = data_dir / "ac3_ac4"

    stale = out_dir / "ac4_em.npy"
    if stale.exists():
        stale.unlink()
        print("  removed old ac4_em.npy (from the earlier, broken path that returned all zeros)")

    if outputs_exist(out_dir, "ac4", with_sem_gt):
        print("  already downloaded -- skipping")
    else:
        intern = need("intern")
        from intern import array
        z, y, x = AC4_SHAPE
        print(f"  fetching image  {AC4_IMAGE_URI} ...", flush=True)
        raw = array(AC4_IMAGE_URI)[0:z, 0:y, 0:x]
        print(f"  fetching labels {AC4_LABEL_URI} ...", flush=True)
        neuron = array(AC4_LABEL_URI)[0:z, 0:y, 0:x]
        labels, k = remap_dense(neuron)
        del neuron
        save_volume(out_dir, "ac4", raw, labels, with_sem_gt)
        print(f"  saved ac4: image {raw.shape} (values {raw.min()}-{raw.max()}), {k} instances")

    print("\n  NOTE 1: AC3 is not published on bossDB as its own volume (Kasthuri/ac3 -> 404),\n"
          "          so only AC4 is downloaded. AC4 is the TEST volume in the paper's protocol;\n"
          "          there is no AC3 training volume here.\n"
          "  NOTE 2: these labels are bossDB's 'neuron' segmentation channel. The papers report\n"
          "          AC3/AC4 synapse results; a synapse channel was not found on bossDB.")


# ---------------------------------------------------------------------------
# STEP 4 : MitoEM
# ---------------------------------------------------------------------------
# Sources are the ones linked from the official challenge page (mitoem.grand-challenge.org).
# Rat images + both label sets are on Hugging Face (pytc/EM30, pytc/MitoEM). The human images
# on Hugging Face are a PADDED variant (1040 slices) that does not line up 1:1 with the labels,
# so the unpadded human images are read from the original Dropbox release instead.
# Both were checked: slice 450 image and label are both 4096x4096, and labelled mitochondria
# are darker than the background (~88 vs ~138), as real EM mitochondria are.
MITOEM_SOURCES = {
    "mito_r": ("https://huggingface.co/datasets/pytc/EM30/resolve/main/EM30-R-im.zip",
               "https://huggingface.co/datasets/pytc/MitoEM/resolve/main/EM30-R-mito-train-val-v2.zip"),
    "mito_h": ("https://www.dropbox.com/s/z41qtu4y735j95e/EM30-H-im.zip?dl=1",
               "https://huggingface.co/datasets/pytc/MitoEM/resolve/main/EM30-H-mito-train-val-v2.zip"),
}


class _RemoteZipReader:
    """Keeps one remote zip open and reads image members from it, reopening on network errors."""

    def __init__(self, url):
        self.url, self.zip = url, None

    def _open(self):
        from remotezip import RemoteZip
        if self.zip is not None:
            try:
                self.zip.close()
            except Exception:
                pass
        self.zip = RemoteZip(self.url)

    def read(self, member, retries=3):
        import io
        from PIL import Image
        for attempt in range(1, retries + 1):
            try:
                if self.zip is None:
                    self._open()
                return np.array(Image.open(io.BytesIO(self.zip.read(member))))
            except KeyError:
                raise RuntimeError(f"{member} is not in {self.url} -- the archive layout may have changed")
            except Exception as e:
                if attempt == retries:
                    raise RuntimeError(f"could not read {member} from {self.url}: {e}")
                print(f"    network error on {member}, retry {attempt} ({e})", flush=True)
                self.zip = None

    def close(self):
        if self.zip is not None:
            self.zip.close()


def prepare_mitoem(data_dir: Path, z0=400, z1=500, with_sem_gt=True):
    """Reads only the needed slices straight out of the remote zips (the full image zips are
    14-15 GB each; this downloads roughly 1.5-2.5 GB per volume for 100 slices)."""
    banner(f"STEP 4  MitoEM-R/H download (slices {z0}-{z1 - 1}, read directly from the remote zips)")
    need("remotezip")
    need("PIL", "pillow")
    out_dir = data_dir / "mitoem"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, (im_url, lb_url) in MITOEM_SOURCES.items():
        if outputs_exist(out_dir, name, with_sem_gt):
            print(f"  {name}: already downloaded -- skipping")
            continue
        n = z1 - z0
        raw = labels = None
        im_zip, lb_zip = _RemoteZipReader(im_url), _RemoteZipReader(lb_url)
        try:
            for k, z in enumerate(range(z0, z1)):
                label_dir = "mito-train-v2" if z < 400 else "mito-val-v2"
                im = im_zip.read(f"im/im{z:04d}.png")
                lb = lb_zip.read(f"{label_dir}/seg{z:04d}.tif")
                if raw is None:
                    raw = np.empty((n,) + im.shape, dtype=np.uint8)
                    labels = np.empty((n,) + lb.shape, dtype=lb.dtype)
                raw[k], labels[k] = im, lb
                if (k + 1) % 10 == 0 or k + 1 == n:
                    print(f"  {name}: {k + 1}/{n} slices", flush=True)
        finally:
            im_zip.close()
            lb_zip.close()

        dense, count = remap_dense(labels, compact=True)
        del labels
        np.save(out_dir / f"{name}_raw.npy", raw)
        np.save(out_dir / f"{name}_labels.npy", dense)
        if with_sem_gt:
            np.save(out_dir / f"{name}_sem_gt.npy", (dense > 0).astype(np.uint8))
        print(f"  {name}: saved image {raw.shape}, {count} mitochondria instances")
        del raw, dense

    print(f"\n  NOTE: slices {z0}-{z1 - 1} are the official labelled validation range (mito-val-v2),\n"
          f"        which is the 100-slice test split reference [26] reports on. MitoEM is used only\n"
          f"        for the zero-shot generalization test (Sec. III-I) -- training does not use it.")


# ---------------------------------------------------------------------------
# STEP 5 : CHECK
# ---------------------------------------------------------------------------
def check(data_dir: Path, require_sem_gt=True):
    banner("STEP 5  CHECK -- every prepared volume")
    folders = ["cremi_train", "cremi_val", "cremi_test", "ac3_ac4", "mitoem"]
    problems = 0
    print(f"  {'file':38s} {'shape':18s} {'instances':>9s} {'foreground':>11s}  status")
    for folder in folders:
        for raw_path in sorted((data_dir / folder).glob("*_raw.npy")):
            name = raw_path.name[:-len("_raw.npy")]
            raw = np.load(raw_path, mmap_mode="r")
            labels = np.load(raw_path.parent / f"{name}_labels.npy", mmap_mode="r")
            ids, fg_voxels, raw_max = set(), 0, 0
            for z in range(labels.shape[0]):
                ids.update(np.unique(labels[z]).tolist())
                fg_voxels += int(np.count_nonzero(labels[z]))
                raw_max = max(raw_max, int(raw[z].max()))
            k = len(ids - {0})
            fg = fg_voxels / labels.size * 100
            issues = []
            if raw_max == 0:
                issues.append("IMAGE IS ALL ZEROS")
            if raw.shape != labels.shape:
                issues.append("image/label shape mismatch")
            if k == 0:
                issues.append("NO INSTANCES")
            if require_sem_gt and not (raw_path.parent / f"{name}_sem_gt.npy").exists():
                issues.append("missing sem_gt")
            status = "OK" if not issues else "PROBLEM: " + "; ".join(issues)
            problems += bool(issues)
            print(f"  {folder + '/' + name:38s} {str(tuple(raw.shape)):18s} {k:9d} {fg:10.2f}%  {status}")

    if problems:
        print(f"\n  {problems} problem(s) found -- fix these before training.")
    else:
        print("\n  All prepared volumes look valid.")
    if any((data_dir / "cremi_train").glob("*_raw.npy")):
        print("\n  NEXT STEP -- training on CREMI:\n"
              "    python src/train.py --data-dir ./data/cremi_train --val-data-dir ./data/cremi_val "
              "--dataset cremi --config-name full --seed 0 --out-dir ./runs/cremi_full_seed0 "
              "--max-epochs 600 --iterations-per-epoch 100 --early-stop-patience 50")
    return problems


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--skip-cremi", action="store_true")
    p.add_argument("--skip-ac4", action="store_true")
    p.add_argument("--skip-mitoem", action="store_true")
    p.add_argument("--no-sem-gt", action="store_true",
                   help="write only <name>_raw.npy and <name>_labels.npy (skip <name>_sem_gt.npy)")
    p.add_argument("--mitoem-slices", type=int, nargs=2, default=[400, 500], metavar=("START", "END"),
                   help="MitoEM z-range to download, END exclusive (default 400 500: the labelled val range)")
    a = p.parse_args()

    data_dir = Path(a.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    sem = not a.no_sem_gt

    if not a.skip_cremi:
        prepare_cremi(data_dir, with_sem_gt=sem)
    if not a.skip_ac4:
        prepare_ac4(data_dir, with_sem_gt=sem)
    if not a.skip_mitoem:
        z0, z1 = a.mitoem_slices
        if not (0 <= z0 < z1 <= 500):
            p.error("--mitoem-slices must be within 0-500 (labels exist only for slices 0-499)")
        prepare_mitoem(data_dir, z0, z1, with_sem_gt=sem)
    sys.exit(1 if check(data_dir, require_sem_gt=sem) else 0)


if __name__ == "__main__":
    main()
