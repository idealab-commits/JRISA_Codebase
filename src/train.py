# -*- coding: utf-8 -*-
"""
train.py -- Algorithm 1 (JRISA Joint Re-Isotropic Instance-Aware Training).

Usage:
    python train.py --data-dir ./data/ac3_ac4 --dataset ac3ac4 \\
        --out-dir ./runs/ac3ac4_full_seed0 --config-name full --seed 0

Implements the exact procedure of Algorithm 1: SGD momentum 0.99, poly LR
schedule, initial rate 0.01, up to 600 epochs with early stopping (see the
manuscript discussion on legitimate epoch reduction -- this does NOT deviate
from "matching [26]'s training budget", it just avoids wasting compute once
converged).

--iterations-per-epoch defaults to 100 rather than an unverified 250 --
this is a training-loop convention, not a baseline-matched parameter (see
prior discussion): reducing it shortens wall-clock time per epoch without
changing the *number* of epochs reported.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from model import JRISA
from losses import total_loss
from data import EMVolumeDataset

# Ablation configurations, Section III-L. Each maps to constructor flags.
CONFIGS = {
    "iso_only":     dict(use_ani=False, use_iso=True,  use_inst=True,  afg=False),
    "ani_only":     dict(use_ani=True,  use_iso=False, use_inst=True,  afg=False),
    "ani_iso_semantic_only": dict(use_ani=True, use_iso=True, use_inst=False, afg=False),  # = ReIsoSeg [26] scope
    "full":         dict(use_ani=True,  use_iso=True,  use_inst=True,  afg=False),
    "full_afg":     dict(use_ani=True,  use_iso=True,  use_inst=True,  afg=True),
}


def loss_weights(cfg, base_gamma=1.0, base_lam=0.5):
    """Turns a config's switches into lambda / gamma of Eq. (15).
    lambda = 0 trains the anisotropic branch only, lambda = 1 the isotropic branch only,
    gamma = 0 drops the instance terms (semantic only, ReIsoSeg [26]'s scope)."""
    if cfg["use_ani"] and not cfg["use_iso"]:
        lam = 0.0
    elif cfg["use_iso"] and not cfg["use_ani"]:
        lam = 1.0
    else:
        lam = base_lam
    gamma = base_gamma if cfg["use_inst"] else 0.0
    return lam, gamma


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def poly_lr(base_lr, epoch, max_epochs, power=0.9):
    return base_lr * (1 - epoch / max_epochs) ** power


def load_volumes(data_dir):
    """Expects data_dir/*.npy pairs: <name>_raw.npy and <name>_labels.npy.
    See README.md for the exact expected layout per dataset."""
    data_dir = Path(data_dir)
    raw_files = sorted(data_dir.glob("*_raw.npy"))
    if not raw_files:
        raise FileNotFoundError(
            f"No '*_raw.npy' files found in {data_dir}. Run scripts/"
            f"download_datasets.py and any dataset-specific preprocessing "
            f"first (see README.md)."
        )
    volumes = []
    for raw_path in raw_files:
        label_path = raw_path.parent / raw_path.name.replace("_raw.npy", "_labels.npy")
        if not label_path.exists():
            raise FileNotFoundError(
                f"Missing label file {label_path} for {raw_path.name}. This "
                f"is exactly the 'annotations problem' flagged earlier -- "
                f"confirm your annotation source actually provides "
                f"instance-labeled ground truth before proceeding."
            )
        volumes.append({"raw": np.load(raw_path), "labels": np.load(label_path)})
    return volumes


def validate(model, val_loader, device, lam, gamma):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in val_loader:
            image = batch["image"].to(device)
            targets = {k: batch[k].to(device) for k in ("sem", "inst_b", "inst_c", "inst_d")}
            outputs = model(image, training_mode=True)
            loss, _ = total_loss(outputs, targets, lam=lam, gamma=gamma)
            losses.append(loss.item())
    model.train()
    return float(np.mean(losses)) if losses else float("inf")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True, help="Directory with <name>_raw.npy / <name>_labels.npy")
    p.add_argument("--val-data-dir", default=None, help="Optional separate validation volume directory")
    p.add_argument("--dataset", choices=["ac3ac4", "cremi", "mitoem"], required=True)
    p.add_argument("--config-name", choices=list(CONFIGS.keys()), default="full")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-epochs", type=int, default=600)
    p.add_argument("--iterations-per-epoch", type=int, default=100,
                   help="Training-loop convention, not baseline-matched -- see module docstring.")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--base-lr", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.99)
    p.add_argument("--gamma", type=float, default=1.0, help="Instance-loss weight, Eq. (15)")
    p.add_argument("--early-stop-patience", type=int, default=50,
                   help="Stop if val loss hasn't improved in this many epochs.")
    p.add_argument("--resume", default=None, help="Path to a checkpoint to resume from")
    args = p.parse_args()

    set_seed(args.seed)

    patch_size = (16, 384, 320) if args.dataset == "cremi" else (16, 320, 320)
    z_sampling = (2, 2, 2) if args.dataset == "cremi" else (1, 2, 2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("WARNING: no CUDA device found -- training will be extremely slow "
              "on CPU (see the earlier feasibility discussion). Proceeding anyway.")

    cfg = CONFIGS[args.config_name]
    lam, gamma = loss_weights(cfg, base_gamma=args.gamma)
    print(f"Loss weights for {args.config_name}: lambda={lam} gamma={gamma}")
    model = JRISA(in_channels=1, num_sem_classes=2, z_sampling=z_sampling,
                  afg_enabled=cfg["afg"]).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    start_epoch = 0
    best_val = float("inf")
    epochs_without_improve = 0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt.get("best_val", best_val)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    train_volumes = load_volumes(args.data_dir)
    train_ds = EMVolumeDataset(train_volumes, patch_size=patch_size,
                                patches_per_epoch=args.iterations_per_epoch * args.batch_size,
                                augment=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                               num_workers=4, pin_memory=(device.type == "cuda"))

    val_loader = None
    if args.val_data_dir:
        val_volumes = load_volumes(args.val_data_dir)
        val_ds = EMVolumeDataset(val_volumes, patch_size=patch_size,
                                  patches_per_epoch=max(20, args.iterations_per_epoch // 5),
                                  augment=False)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print(f"Config={args.config_name} seed={args.seed} device={device} "
          f"patch={patch_size} iters/epoch={args.iterations_per_epoch}")

    t_start = time.time()
    for epoch in range(start_epoch, args.max_epochs):
        lr = poly_lr(args.base_lr, epoch, args.max_epochs)
        for g in optimizer.param_groups:
            g["lr"] = lr

        epoch_losses = []
        for batch in train_loader:
            image = batch["image"].to(device, non_blocking=True)
            targets = {k: batch[k].to(device, non_blocking=True)
                       for k in ("sem", "inst_b", "inst_c", "inst_d")}

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                outputs = model(image, training_mode=True)
                loss, loss_parts = total_loss(outputs, targets, lam=lam, gamma=gamma)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_losses.append(loss.item())

        train_loss = float(np.mean(epoch_losses))
        val_loss = validate(model, val_loader, device, lam, gamma) if val_loader else None
        elapsed = time.time() - t_start

        log_entry = {"epoch": epoch, "lr": lr, "train_loss": train_loss,
                      "val_loss": val_loss, "elapsed_sec": elapsed}
        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        print(f"[epoch {epoch}] lr={lr:.5f} train_loss={train_loss:.4f} "
              f"val_loss={val_loss} elapsed={elapsed/3600:.2f}h")

        ckpt = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "epoch": epoch, "best_val": best_val, "args": vars(args)}
        torch.save(ckpt, out_dir / "last.pt")

        if val_loss is not None:
            if val_loss < best_val - 1e-4:
                best_val = val_loss
                epochs_without_improve = 0
                torch.save(ckpt, out_dir / "best.pt")
            else:
                epochs_without_improve += 1
                if epochs_without_improve >= args.early_stop_patience:
                    print(f"Early stopping at epoch {epoch} "
                          f"(no val improvement in {args.early_stop_patience} epochs).")
                    break

    print(f"Training complete. Total time: {(time.time()-t_start)/3600:.2f} hours.")


if __name__ == "__main__":
    main()
