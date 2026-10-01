# -*- coding: utf-8 -*-
"""
metrics.py -- dual-level evaluation protocol (Section III-K).

    Eq. (16) IoU
    Eq. (17) Hungarian assignment cost
    Eq. (18) precision / recall / accuracy
    Six-way association-error taxonomy (Section III-K)
    Semantic: Dice (via 1 - dice_loss), AD-score
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import find_objects


def iou(mask_a, mask_b):
    """Eq. (16): IoU(A,B) = |A ∩ B| / |A ∪ B|"""
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return inter / union if union > 0 else 0.0


def _instance_masks(label_vol):
    """Returns {instance_id: boolean_mask} for a label volume, skipping 0 (background)."""
    ids = np.unique(label_vol)
    ids = ids[ids != 0]
    return {int(i): (label_vol == i) for i in ids}


def dice_score(pred_mask, gt_mask, eps=1e-5):
    """Semantic Dice (1 - Eq. 3's Dice loss, at the binary-mask level)."""
    inter = np.logical_and(pred_mask, gt_mask).sum()
    return (2.0 * inter + eps) / (pred_mask.sum() + gt_mask.sum() + eps)


def ad_score(pred_mask, gt_mask):
    """Average-distance score: mean symmetric surface distance between
    predicted and ground-truth masks, following [26]/[27]. Requires
    scikit-image's distance_transform via scipy.ndimage."""
    from scipy.ndimage import distance_transform_edt

    def surface_dists(a, b):
        b_dist = distance_transform_edt(~b)
        surface_a = a & ~_erode(a)
        if surface_a.sum() == 0:
            return np.array([0.0])
        return b_dist[surface_a]

    def _erode(mask):
        from scipy.ndimage import binary_erosion
        return binary_erosion(mask)

    d_ab = surface_dists(pred_mask, gt_mask)
    d_ba = surface_dists(gt_mask, pred_mask)
    return float(np.mean(np.concatenate([d_ab, d_ba])))


def hungarian_match(pred_labels, gt_labels, T=0.75, N=None):
    """Eq. (17): C(i,j) = -[IoU(g_j,p_i) >= T] - IoU(g_j,p_i)/(2N)

    Returns: matched pairs [(pred_id, gt_id, iou)], unmatched_pred_ids, unmatched_gt_ids
    """
    pred_masks = _instance_masks(pred_labels)
    gt_masks = _instance_masks(gt_labels)
    pred_ids, gt_ids = list(pred_masks.keys()), list(gt_masks.keys())

    if N is None:
        N = max(len(pred_ids), len(gt_ids), 1)

    if len(pred_ids) == 0 or len(gt_ids) == 0:
        return [], pred_ids, gt_ids

    cost = np.zeros((len(pred_ids), len(gt_ids)))
    iou_table = np.zeros_like(cost)
    for i, pid in enumerate(pred_ids):
        p_bbox_slices = find_objects(pred_labels == pid)
        for j, gid in enumerate(gt_ids):
            v = iou(pred_masks[pid], gt_masks[gid])
            iou_table[i, j] = v
            cost[i, j] = -float(v >= T) - v / (2 * N)

    row_ind, col_ind = linear_sum_assignment(cost)

    matched, matched_pred, matched_gt = [], set(), set()
    for i, j in zip(row_ind, col_ind):
        if iou_table[i, j] >= T:
            matched.append((pred_ids[i], gt_ids[j], float(iou_table[i, j])))
            matched_pred.add(pred_ids[i])
            matched_gt.add(gt_ids[j])

    unmatched_pred = [p for p in pred_ids if p not in matched_pred]
    unmatched_gt = [g for g in gt_ids if g not in matched_gt]
    return matched, unmatched_pred, unmatched_gt


def precision_recall_accuracy(n_matched, n_pred, n_gt):
    """Eq. (18): precision = TP/(TP+FP), recall = TP/(TP+FN), accuracy = TP/(TP+FP+FN)"""
    tp = n_matched
    fp = n_pred - n_matched
    fn = n_gt - n_matched
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return precision, recall, accuracy


def classify_associations(pred_labels, gt_labels, T=0.75, overlap_T=0.10):
    """Six-way association-error taxonomy, Section III-K:
       one-to-one, over-segmentation, under-segmentation, missing,
       background, many-to-many.

    A pragmatic, documented implementation of [2]'s set-based definitions:
    for each gt/pred instance, count how many "significant" overlap partners
    it has (IoU with any candidate >= overlap_T, a looser threshold than the
    T=0.75 matching threshold, used only to detect fragmentation/merging).
    """
    pred_masks = _instance_masks(pred_labels)
    gt_masks = _instance_masks(gt_labels)

    gt_to_preds = {g: [] for g in gt_masks}
    pred_to_gts = {p: [] for p in pred_masks}
    for g, gm in gt_masks.items():
        for p, pm in pred_masks.items():
            if iou(gm, pm) >= overlap_T:
                gt_to_preds[g].append(p)
                pred_to_gts[p].append(g)

    counts = {"one_to_one": 0, "over_segmentation": 0, "under_segmentation": 0,
              "missing": 0, "background": 0, "many_to_many": 0}

    for g, ps in gt_to_preds.items():
        if len(ps) == 0:
            counts["missing"] += 1
        elif len(ps) == 1 and len(pred_to_gts[ps[0]]) == 1:
            counts["one_to_one"] += 1
        elif len(ps) > 1 and all(len(pred_to_gts[p]) == 1 for p in ps):
            counts["over_segmentation"] += 1
        elif len(ps) >= 1 and any(len(pred_to_gts[p]) > 1 for p in ps):
            counts["many_to_many"] += 1

    for p, gs in pred_to_gts.items():
        if len(gs) == 0:
            counts["background"] += 1
        elif len(gs) > 1 and all(len(gt_to_preds[g]) == 1 for g in gs):
            counts["under_segmentation"] += 1

    total = sum(counts.values())
    pct = {k: (v / total * 100.0 if total > 0 else 0.0) for k, v in counts.items()}
    return counts, pct
