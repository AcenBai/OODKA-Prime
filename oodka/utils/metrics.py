"""Segmentation evaluation metrics: Dice, Precision, Recall, HD95, Accuracy."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk


def dice_no_ignore(
    pred: np.ndarray,
    gt: np.ndarray,
    class_ids: Sequence[int],
) -> Tuple[Dict[int, Optional[float]], float, float]:
    """Dice in raw space (no ignore label)."""
    if pred.shape != gt.shape:
        raise ValueError(f"pred.shape={pred.shape} != gt.shape={gt.shape}")

    dice_pc: Dict[int, Optional[float]] = {}
    dices_gt_present: List[float] = []
    dices_any_present: List[float] = []

    for c in class_ids:
        p = pred == c
        g = gt == c
        denom = int(p.sum()) + int(g.sum())
        if denom == 0:
            dice_pc[int(c)] = None
            continue
        d = (2.0 * int((p & g).sum())) / float(denom)
        dice_pc[int(c)] = float(d)
        dices_any_present.append(float(d))
        if int(g.sum()) > 0:
            dices_gt_present.append(float(d))

    mean_gt = float(np.mean(dices_gt_present)) if dices_gt_present else 0.0
    mean_any = float(np.mean(dices_any_present)) if dices_any_present else 0.0
    return dice_pc, mean_gt, mean_any


def dice_ignore_minus_one(
    pred: np.ndarray,
    gt: np.ndarray,
    class_ids: Sequence[int],
) -> Tuple[Dict[int, Optional[float]], float, float]:
    """Dice in preprocessed space where GT may contain -1 (ignore)."""
    if pred.shape != gt.shape:
        raise ValueError(f"pred.shape={pred.shape} != gt.shape={gt.shape}")
    valid = gt != -1

    dice_pc: Dict[int, Optional[float]] = {}
    dices_gt_present: List[float] = []
    dices_any_present: List[float] = []

    for c in class_ids:
        p = (pred == c) & valid
        g = (gt == c) & valid
        denom = int(p.sum()) + int(g.sum())
        if denom == 0:
            dice_pc[int(c)] = None
            continue
        d = (2.0 * int((p & g).sum())) / float(denom)
        dice_pc[int(c)] = float(d)
        dices_any_present.append(float(d))
        if int(g.sum()) > 0:
            dices_gt_present.append(float(d))

    mean_gt = float(np.mean(dices_gt_present)) if dices_gt_present else 0.0
    mean_any = float(np.mean(dices_any_present)) if dices_any_present else 0.0
    return dice_pc, mean_gt, mean_any


def precision_recall_hd95_no_ignore(
    pred: np.ndarray,
    gt: np.ndarray,
    class_ids: Sequence[int],
    spacing_xyz: Tuple[float, float, float],
) -> Tuple[
    Dict[int, Optional[float]],
    Dict[int, Optional[float]],
    Dict[int, Optional[float]],
]:
    """Per-class precision, recall, HD95 in raw NIfTI space."""
    if pred.shape != gt.shape:
        raise ValueError(f"pred.shape={pred.shape} != gt.shape={gt.shape}")

    precision_pc: Dict[int, Optional[float]] = {}
    recall_pc: Dict[int, Optional[float]] = {}
    hd95_pc: Dict[int, Optional[float]] = {}

    for c in class_ids:
        ic = int(c)
        pred_fg = pred == c
        gt_fg = gt == c
        pred_sum = int(pred_fg.sum())
        gt_sum = int(gt_fg.sum())
        tp = int((pred_fg & gt_fg).sum())

        if pred_sum + gt_sum == 0:
            precision_pc[ic] = recall_pc[ic] = hd95_pc[ic] = None
            continue

        precision_pc[ic] = float(tp / pred_sum) if pred_sum > 0 else 0.0
        recall_pc[ic] = float(tp / gt_sum) if gt_sum > 0 else 0.0

        if pred_sum == 0 or gt_sum == 0:
            hd95_pc[ic] = None
            continue

        pred_img = sitk.GetImageFromArray(pred_fg.astype(np.uint8))
        gt_img = sitk.GetImageFromArray(gt_fg.astype(np.uint8))
        pred_img.SetSpacing(spacing_xyz)
        gt_img.SetSpacing(spacing_xyz)

        pred_surface = sitk.LabelContour(pred_img)
        gt_surface = sitk.LabelContour(gt_img)
        pred_surface_np = sitk.GetArrayViewFromImage(pred_surface) > 0
        gt_surface_np = sitk.GetArrayViewFromImage(gt_surface) > 0

        if not pred_surface_np.any() or not gt_surface_np.any():
            hd95_pc[ic] = None
            continue

        pred_dist = sitk.Abs(
            sitk.SignedMaurerDistanceMap(pred_img, squaredDistance=False, useImageSpacing=True)
        )
        gt_dist = sitk.Abs(
            sitk.SignedMaurerDistanceMap(gt_img, squaredDistance=False, useImageSpacing=True)
        )
        all_dists = np.concatenate([
            sitk.GetArrayViewFromImage(gt_dist)[pred_surface_np],
            sitk.GetArrayViewFromImage(pred_dist)[gt_surface_np],
        ])
        hd95_pc[ic] = float(np.percentile(all_dists, 95)) if all_dists.size > 0 else None

    return precision_pc, recall_pc, hd95_pc


def raw_per_class_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    class_ids: Sequence[int],
    spacing_xyz: Tuple[float, float, float],
) -> Tuple[
    Dict[int, Optional[float]],  # sensitivity
    Dict[int, Optional[float]],  # specificity
    Dict[int, Optional[float]],  # precision
    Dict[int, Optional[float]],  # recall
    Dict[int, Optional[float]],  # accuracy
    Dict[int, Optional[float]],  # hd95
]:
    """Full one-vs-rest metrics: sensitivity, specificity, precision, recall, accuracy, HD95."""
    if pred.shape != gt.shape:
        raise ValueError(f"pred.shape={pred.shape} != gt.shape={gt.shape}")
    n_vox = int(pred.size)

    sensitivity_pc: Dict[int, Optional[float]] = {}
    specificity_pc: Dict[int, Optional[float]] = {}
    precision_pc: Dict[int, Optional[float]] = {}
    recall_pc: Dict[int, Optional[float]] = {}
    accuracy_pc: Dict[int, Optional[float]] = {}
    hd95_pc: Dict[int, Optional[float]] = {}

    for c in class_ids:
        ic = int(c)
        pred_fg = pred == c
        gt_fg = gt == c
        tp = int((pred_fg & gt_fg).sum())
        fp = int((pred_fg & ~gt_fg).sum())
        fn = int((~pred_fg & gt_fg).sum())
        tn = n_vox - tp - fp - fn
        pred_sum, gt_sum = tp + fp, tp + fn

        if pred_sum + gt_sum == 0:
            for d in (sensitivity_pc, specificity_pc, precision_pc, recall_pc, accuracy_pc, hd95_pc):
                d[ic] = None
            continue

        precision_pc[ic] = float(tp / pred_sum) if pred_sum > 0 else 0.0
        rec = float(tp / gt_sum) if gt_sum > 0 else 0.0
        recall_pc[ic] = rec
        sensitivity_pc[ic] = rec
        specificity_pc[ic] = float(tn / (tn + fp)) if (tn + fp) > 0 else None
        accuracy_pc[ic] = float((tp + tn) / n_vox) if n_vox > 0 else None

        if pred_sum == 0 or gt_sum == 0:
            hd95_pc[ic] = None
            continue

        pred_img = sitk.GetImageFromArray(pred_fg.astype(np.uint8))
        gt_img = sitk.GetImageFromArray(gt_fg.astype(np.uint8))
        pred_img.SetSpacing(spacing_xyz)
        gt_img.SetSpacing(spacing_xyz)

        pred_surface = sitk.LabelContour(pred_img)
        gt_surface = sitk.LabelContour(gt_img)
        ps_np = sitk.GetArrayViewFromImage(pred_surface) > 0
        gs_np = sitk.GetArrayViewFromImage(gt_surface) > 0

        if not ps_np.any() or not gs_np.any():
            hd95_pc[ic] = None
            continue

        pd = sitk.Abs(sitk.SignedMaurerDistanceMap(pred_img, squaredDistance=False, useImageSpacing=True))
        gd = sitk.Abs(sitk.SignedMaurerDistanceMap(gt_img, squaredDistance=False, useImageSpacing=True))
        all_dists = np.concatenate([
            sitk.GetArrayViewFromImage(gd)[ps_np],
            sitk.GetArrayViewFromImage(pd)[gs_np],
        ])
        hd95_pc[ic] = float(np.percentile(all_dists, 95)) if all_dists.size > 0 else None

    return sensitivity_pc, specificity_pc, precision_pc, recall_pc, accuracy_pc, hd95_pc


def class_mean_accuracy(
    accuracy_per_class: Dict[int, Optional[float]],
    class_ids: Sequence[int],
) -> float:
    vals = [float(accuracy_per_class[int(c)]) for c in class_ids if accuracy_per_class.get(int(c)) is not None]
    return float(np.mean(vals)) if vals else float("nan")
