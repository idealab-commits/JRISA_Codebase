# -*- coding: utf-8 -*-
"""
run_all.py -- the whole JRISA pipeline in one command.

    python run_all.py                                  # CREMI, config "full", seed 0, all stages
    python run_all.py --dataset ac4 --configs full full_afg --seeds 0 1 2
    python run_all.py --stages check data              # only some stages

Stages (in order): check data split pilot train infer evaluate
Every stage can be re-run safely: finished training runs are skipped (DONE marker),
interrupted runs resume from last.pt.

NOTE: each dataset is evaluated on its OWN test split. CREMI labels are synaptic clefts,
AC4 labels are neurons, MitoEM labels are mitochondria -- testing a model trained on one
on another is not a meaningful number.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
DATA = ROOT / "data"
PY = sys.executable

# dataset -> (train.py/infer.py --dataset name, train dir, val dir, test dir, test sample names)
DATASETS = {
    "cremi": ("cremi", DATA / "cremi_train", DATA / "cremi_val", DATA / "cremi_test", None),
    "ac4": ("ac3ac4", DATA / "splits" / "ac4_train", DATA / "splits" / "ac4_val", DATA / "splits" / "ac4_test", ["ac4"]),
    "mitoem": ("mitoem", DATA / "splits" / "mitoem_train", DATA / "splits" / "mitoem_val", DATA / "splits" / "mitoem_test", None),
}
Z_SPLIT = ((0, 60), (60, 80), (80, 100))   # train / val / test slices for AC4 and MitoEM


def run(cmd):
    print("\n>>> " + " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=SRC if "src" in str(cmd[1]) else ROOT)
    if r.returncode != 0:
        sys.exit(f"Command failed with exit code {r.returncode}. Fix the error above and re-run; finished work is kept.")


def stage_check(args):
    print("== CHECK ==")
    try:
        import torch
    except ImportError:
        sys.exit("PyTorch is not installed. Install a CUDA build from https://pytorch.org first.")
    print(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: no CUDA GPU -- training will be far too slow on CPU.")
    missing = []
    for mod in ("numpy", "scipy", "h5py", "skimage", "huggingface_hub", "intern", "remotezip", "PIL"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        print(f"Installing missing packages ({', '.join(missing)}) from requirements.txt ...")
        run([PY, "-m", "pip", "install", "-r", ROOT / "requirements.txt"])
    free_gb = shutil.disk_usage(ROOT).free / 1e9
    print(f"Free disk: {free_gb:.1f} GB" + ("  WARNING: < 20 GB" if free_gb < 20 else ""))


def stage_data(args):
    print("== DATA ==")
    cmd = [PY, ROOT / "scripts" / "prepare_data.py", "--data-dir", DATA]
    if args.dataset != "cremi":
        cmd.append("--skip-cremi")
    if args.dataset != "ac4":
        cmd.append("--skip-ac4")
    if args.dataset != "mitoem":
        cmd.append("--skip-mitoem")
    run(cmd)


def _center_crop(a, size):
    if size <= 0 or a.shape[1] <= size:
        return a
    y0, x0 = (a.shape[1] - size) // 2, (a.shape[2] - size) // 2
    return a[:, y0:y0 + size, x0:x0 + size]


def stage_split(args):
    """CREMI already comes split by prepare_data.py. AC4 and MitoEM are one volume each,
    so they are cut along z into train/val/test (MitoEM also centre-cropped to fit in RAM)."""
    print("== SPLIT ==")
    if args.dataset == "cremi":
        print("CREMI is already split (cremi_train / cremi_val / cremi_test).")
        return
    if args.dataset == "ac4":
        sources = [(DATA / "ac3_ac4", "ac4")]
    else:
        sources = [(DATA / "mitoem", "mito_r"), (DATA / "mitoem", "mito_h")]
    _, *dirs, _ = DATASETS[args.dataset]
    for folder, name in sources:
        raw_p, lab_p = folder / f"{name}_raw.npy", folder / f"{name}_labels.npy"
        if not raw_p.exists():
            print(f"  [skip] {raw_p} not found")
            continue
        raw = np.load(raw_p, mmap_mode="r")
        lab = np.load(lab_p, mmap_mode="r")
        for out_dir, (z0, z1) in zip(dirs, Z_SPLIT):
            out_dir.mkdir(parents=True, exist_ok=True)
            if (out_dir / f"{name}_labels.npy").exists():
                continue
            crop = args.crop if args.dataset == "mitoem" else 0
            np.save(out_dir / f"{name}_raw.npy", np.ascontiguousarray(_center_crop(raw[z0:z1], crop)))
            np.save(out_dir / f"{name}_labels.npy", np.ascontiguousarray(_center_crop(lab[z0:z1], crop)))
            print(f"  {name} z{z0}:{z1} -> {out_dir}")


def _train_cmd(args, cfg, seed, out_dir, epochs, iters):
    ds_name, train_dir, val_dir, _, _ = DATASETS[args.dataset]
    cmd = [PY, SRC / "train.py", "--data-dir", train_dir, "--val-data-dir", val_dir,
           "--dataset", ds_name, "--config-name", cfg, "--seed", seed, "--out-dir", out_dir,
           "--max-epochs", epochs, "--iterations-per-epoch", iters,
           "--early-stop-patience", args.patience]
    if (out_dir / "last.pt").exists():
        cmd += ["--resume", out_dir / "last.pt"]
    return cmd


def stage_pilot(args):
    """10 short epochs: the loss must be finite and go down before committing ~3 days of GPU."""
    print("== PILOT ==")
    out_dir = ROOT / "runs" / f"pilot_{args.dataset}"
    if (out_dir / "PASSED").exists():
        print("Pilot already passed.")
        return
    shutil.rmtree(out_dir, ignore_errors=True)
    run(_train_cmd(args, args.configs[0], 0, out_dir, 10, 20))
    losses = [json.loads(l)["train_loss"] for l in open(out_dir / "train_log.jsonl")]
    print("pilot train_loss:", ", ".join(f"{x:.4f}" for x in losses))
    if not all(np.isfinite(losses)) or np.mean(losses[-3:]) >= np.mean(losses[:3]):
        sys.exit("PILOT FAILED: loss is not finite or not decreasing. Do not start full training.")
    (out_dir / "PASSED").write_text("ok")
    print("Pilot passed.")


def run_dir(args, cfg, seed):
    return ROOT / "runs" / f"{args.dataset}_{cfg}_seed{seed}"


def stage_train(args):
    print("== TRAIN ==")
    for cfg in args.configs:
        for seed in args.seeds:
            out_dir = run_dir(args, cfg, seed)
            if (out_dir / "DONE").exists():
                print(f"[done] {out_dir.name}")
                continue
            run(_train_cmd(args, cfg, seed, out_dir, args.max_epochs, args.iterations_per_epoch))
            (out_dir / "DONE").write_text("ok")


def _test_samples(args):
    _, _, _, test_dir, names = DATASETS[args.dataset]
    return names or sorted(p.name[:-len("_raw.npy")] for p in test_dir.glob("*_raw.npy"))


def stage_infer(args):
    print("== INFER ==")
    ds_name, _, _, test_dir, _ = DATASETS[args.dataset]
    for cfg in args.configs:
        for seed in args.seeds:
            rd = run_dir(args, cfg, seed)
            ckpt = rd / "best.pt" if (rd / "best.pt").exists() else rd / "last.pt"
            if not ckpt.exists():
                print(f"  [skip] no checkpoint in {rd}")
                continue
            for sample in _test_samples(args):
                out = ROOT / "predictions" / args.dataset / f"{cfg}_seed{seed}" / sample
                if (out / "instances.npy").exists():
                    continue
                run([PY, SRC / "infer.py", "--checkpoint", ckpt, "--volume", test_dir / f"{sample}_raw.npy",
                     "--dataset", ds_name, "--out-dir", out])


def stage_evaluate(args):
    print("== EVALUATE ==")
    _, _, _, test_dir, _ = DATASETS[args.dataset]
    out = ROOT / "results" / f"{args.dataset}_results.json"
    out.parent.mkdir(exist_ok=True)
    run([PY, SRC / "evaluate.py", "--pred-root", ROOT / "predictions" / args.dataset, "--gt-dir", test_dir,
         "--configs", *args.configs, "--seeds", *args.seeds, "--samples", *_test_samples(args), "--out", out])
    res = json.load(open(out))
    print(f"\n{'config':<24}{'Dice':>8}{'Prec':>8}{'Rec':>8}{'Acc':>8}")
    for cfg, s in res["configs"].items():
        print(f"{cfg:<24}" + "".join(f"{s[k]['mean']:>8.3f}" for k in ("dice", "precision", "recall", "accuracy")))
    print(f"Full results: {out}")


STAGES = {"check": stage_check, "data": stage_data, "split": stage_split, "pilot": stage_pilot,
          "train": stage_train, "infer": stage_infer, "evaluate": stage_evaluate}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=list(DATASETS), default="cremi")
    p.add_argument("--configs", nargs="+", default=["full"],
                   choices=["iso_only", "ani_only", "ani_iso_semantic_only", "full", "full_afg"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--stages", nargs="+", choices=list(STAGES), default=list(STAGES))
    p.add_argument("--max-epochs", type=int, default=600)
    p.add_argument("--iterations-per-epoch", type=int, default=100)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--crop", type=int, default=1024, help="MitoEM centre crop (pixels) to fit in RAM")
    args = p.parse_args()
    for name in STAGES:
        if name in args.stages:
            STAGES[name](args)
    print("\nAll requested stages finished.")


if __name__ == "__main__":
    main()
