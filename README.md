# JRISA -- Joint Re-Isotropic Instance-Aware Segmentation

Reference implementation of the proposed method described in the manuscript
(`EM_Segmentation_Full_Manuscript.docx`, Section III). Implements Algorithms
1-3 exactly, with equations cited inline in the code.

## Status / honesty notes

- This code is **written and syntax-checked**, but has **not been run
  end-to-end** (no GPU/dataset was available in the environment it was
  written in). Before committing to full-scale training, run the pilot
  smoke test below on your own machine.
- The Adaptive Frequency Gate's wavelet high-frequency extraction (Eq. 11)
  uses a documented practical approximation (box-blur residual) rather than
  a full 3D Haar decomposition -- see the docstring in `src/model.py`'s
  `WaveletHF` class if exact fidelity matters to you.
- AC3's exact bossDB annotation channel was not confirmed during research --
  see `scripts/download_datasets.py` and the annotations-risk discussion
  before assuming labels will download automatically.

## Setup

```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
# Install the CUDA-enabled PyTorch build matching your GPU driver:
# https://pytorch.org/get-started/locally/
```

## 0. Get all data in one step (recommended)

Double-click `run_data_setup.bat`, or from the project root run:

```bash
python scripts/prepare_data.py
```

This downloads CREMI and AC4, converts them, applies the manuscript's z-splits
(CREMI 70/30/25 per volume), writes `<name>_raw.npy` / `_labels.npy` /
`_sem_gt.npy` for every volume, and ends with a check table that flags any
broken volume (e.g. all zeros). It is safe to re-run; finished steps are
skipped. When it finishes you can go straight to training (step 4). Steps 1-2
below describe the older manual route.

## 1. Download datasets

```bash
python scripts/download_datasets.py --dataset all --output-dir ./data
```

See the script's output for per-dataset caveats (CREMI is automatic; AC3/AC4
partially automatic with a manual annotation-channel step; MitoEM is fully
manual via the Grand Challenge portal).

## 2. Expected data layout

Each `src/train.py` / `src/evaluate.py` call expects a directory containing
`<name>_raw.npy` and `<name>_labels.npy` pairs (instance-labeled ground
truth, integer IDs, 0=background):

```
data/ac3_ac4/
  ac3_raw.npy       ac3_labels.npy      # training volume
  ac4_raw.npy       ac4_labels.npy      # test volume (held out)
```

Convert whatever format your downloaded annotations arrive in (HDF5,
Neuroglancer precomputed, TIFF stacks, etc.) into this `.npy` layout before
proceeding -- this is a deliberately simple, explicit interchange format so
data bugs are easy to inspect (`np.load(...).shape`, `np.unique(...)`).

## 3. Pilot smoke test (run this BEFORE full-scale training)

Verify convergence behavior and catch bugs cheaply:

```bash
python src/train.py --data-dir ./data/ac3_ac4 --dataset ac3ac4 \
    --config-name full --seed 0 --out-dir ./runs/pilot \
    --max-epochs 10 --iterations-per-epoch 20 --early-stop-patience 100
```

If this runs without error and `train_loss` decreases across epochs in
`runs/pilot/train_log.jsonl`, the pipeline is working.

## 4. Full training (Algorithm 1)

```bash
python src/train.py --data-dir ./data/ac3_ac4 --val-data-dir ./data/ac3_ac4_val \
    --dataset ac3ac4 --config-name full --seed 0 \
    --out-dir ./runs/ac3ac4_full_seed0 \
    --max-epochs 600 --iterations-per-epoch 100 --early-stop-patience 50
```

Repeat for each config in `{iso_only, ani_only, ani_iso_semantic_only, full,
full_afg}` x each seed in your chosen seed set (see the manuscript's
mixed n=3/n=5 seed-budget discussion for a compute-constrained strategy).

## 5. Inference (Algorithm 2)

```bash
python src/infer.py --checkpoint ./runs/ac3ac4_full_seed0/best.pt \
    --volume ./data/ac3_ac4/ac4_raw.npy --dataset ac3ac4 \
    --out-dir ./predictions/full_seed0/ac4
```

## 6. Evaluation (Algorithm 3)

```bash
python src/evaluate.py --pred-root ./predictions --gt-dir ./data/ac3_ac4 \
    --configs iso_only ani_only ani_iso_semantic_only full full_afg \
    --seeds 0 1 2 3 4 --samples ac4 --out results_ac3ac4.json
```

Produces per-config Dice/AD-score/precision/recall/accuracy (mean +/- std
over seeds), the six-way association-error breakdown, and paired t-test
p-values between adjacent ablation configurations -- ready to feed into
`draw_results_template.py`'s `METRICS` / `ASSOC_ERROR` dicts.

## Repo layout

```
JRISA_Codebase/
  requirements.txt
  scripts/
    download_datasets.py   # dataset acquisition (Section III-I)
  src/
    model.py      # Encoder, Decoder, PI module, AFG, dual heads (Sections III-D..G)
    losses.py      # Eqs. 3,4,5,6,7,8,9,15
    decode.py      # Eq. 10, marker-controlled watershed
    metrics.py      # Eqs. 16,17,18 + six-way association taxonomy
    data.py      # BCD label generation + patch dataset
    train.py      # Algorithm 1
    infer.py      # Algorithm 2
    evaluate.py      # Algorithm 3
```
