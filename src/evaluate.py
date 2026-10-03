# -*- coding: utf-8 -*-
"""
evaluate.py -- Algorithm 3 (Ablation and Dual-Level Evaluation Protocol).

Aggregates per-seed predictions into the metric tables from Section III-K,
runs paired t-tests between adjacent ablation configurations (Section III-L),
and writes a JSON results file ready to feed into the two results-figure
templates built earlier (qualitative/quantitative).

Usage:
    python evaluate.py --pred-root ./predictions --gt-dir ./data/ac3_ac4 \\
        --configs iso_only ani_only ani_iso_semantic_only full full_afg \\
        --seeds 0 1 2 3 4 --out results_ac3ac4.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

from metrics import dice_score, ad_score, hungarian_match, precision_recall_accuracy, classify_associations


def evaluate_one_prediction(pred_dir, gt_dir, sample_name):
    sem_pred = np.load(Path(pred_dir) / "sem_pred.npy")
    inst_pred = np.load(Path(pred_dir) / "instances.npy")
    inst_gt = np.load(Path(gt_dir) / f"{sample_name}_labels.npy")
    sem_gt_path = Path(gt_dir) / f"{sample_name}_sem_gt.npy"
    # The semantic target is "any labelled object", so derive it when no _sem_gt file was saved.
    sem_gt = np.load(sem_gt_path) if sem_gt_path.exists() else (inst_gt > 0)

    dsc = dice_score(sem_pred > 0, sem_gt > 0)
    ad = ad_score(sem_pred > 0, sem_gt > 0)

    matched, unmatched_pred, unmatched_gt = hungarian_match(inst_pred, inst_gt, T=0.75)
    precision, recall, accuracy = precision_recall_accuracy(
        len(matched), len(matched) + len(unmatched_pred), len(matched) + len(unmatched_gt)
    )
    _, assoc_pct = classify_associations(inst_pred, inst_gt, T=0.75)

    return {
        "dice": dsc, "ad_score": ad,
        "precision": precision, "recall": recall, "accuracy": accuracy,
        "association_pct": assoc_pct,
    }


def aggregate_config(pred_root, gt_dir, config_name, seeds, sample_names):
    per_seed_scores = {"dice": [], "ad_score": [], "precision": [], "recall": [], "accuracy": []}
    assoc_accum = []

    for seed in seeds:
        seed_scores = {k: [] for k in per_seed_scores}
        for sample in sample_names:
            pred_dir = Path(pred_root) / f"{config_name}_seed{seed}" / sample
            if not pred_dir.exists():
                print(f"  [warn] missing prediction dir {pred_dir}, skipping")
                continue
            res = evaluate_one_prediction(pred_dir, gt_dir, sample)
            for k in seed_scores:
                seed_scores[k].append(res[k])
            assoc_accum.append(res["association_pct"])
        for k in per_seed_scores:
            per_seed_scores[k].append(float(np.mean(seed_scores[k])) if seed_scores[k] else float("nan"))

    summary = {}
    for k, vals in per_seed_scores.items():
        vals = np.array(vals)
        summary[k] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals)), "per_seed": vals.tolist()}

    if assoc_accum:
        keys = assoc_accum[0].keys()
        summary["association_pct"] = {k: float(np.mean([a[k] for a in assoc_accum])) for k in keys}

    return summary, per_seed_scores


def paired_ttest(scores_a, scores_b):
    a, b = np.array(scores_a), np.array(scores_b)
    n = min(len(a), len(b))
    if n < 2:
        return None
    stat, p = stats.ttest_rel(a[:n], b[:n])
    return float(p)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred-root", required=True)
    p.add_argument("--gt-dir", required=True)
    p.add_argument("--configs", nargs="+", required=True,
                   help="Ablation ordering, e.g. iso_only ani_only ani_iso_semantic_only full full_afg")
    p.add_argument("--seeds", nargs="+", type=int, required=True)
    p.add_argument("--samples", nargs="+", required=True,
                   help="Sample/volume names to evaluate on, e.g. ac4")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    results = {"configs": {}, "significance": {}}

    per_config_dice = {}
    for cfg in args.configs:
        print(f"Evaluating config: {cfg}")
        summary, per_seed = aggregate_config(args.pred_root, args.gt_dir, cfg, args.seeds, args.samples)
        results["configs"][cfg] = summary
        per_config_dice[cfg] = per_seed["dice"]

    # paired t-test between adjacent configs in the given ablation ordering (Section III-L)
    for i in range(len(args.configs) - 1):
        a, b = args.configs[i], args.configs[i + 1]
        p_val = paired_ttest(per_config_dice[a], per_config_dice[b])
        results["significance"][f"{a}_vs_{b}"] = {
            "p_value": p_val,
            "significant_at_0.05": (p_val is not None and p_val <= 0.05),
        }

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote results to {args.out}")


if __name__ == "__main__":
    main()
