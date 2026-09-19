#!/usr/bin/env python3
"""Split ROI-v2 predicted scar+edema into scar vs edema without extra training.

The V2 model emits one pathology class. This script keeps LV / RV / normal
myocardium as predicted, and only rewrites voxels where the model said
scar+edema. Split rules are unsupervised except the LGE polarity prior
(higher intensity -> scar). An oracle using GT labels *inside* the predicted
pathology mask is included as an upper bound.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk


# Dataset011 original: 1=scar, 2=edema, 3=LV, 4=normal myo, 5=RV
# V2 pred:             1=LV, 2=RV, 3=normal myo, 4=scar+edema
# Split output:        1=LV, 2=RV, 3=normal myo, 4=scar, 5=edema
SPLIT_NAMES = {
    1: "LV",
    2: "RV",
    3: "normal_myo",
    4: "scar",
    5: "edema",
}
MIN_SPLIT_VOXELS = 16


def remap_gt(array: np.ndarray) -> np.ndarray:
    output = np.zeros_like(array, dtype=np.int16)
    output[array == 3] = 1
    output[array == 5] = 2
    output[array == 4] = 3
    output[array == 1] = 4
    output[array == 2] = 5
    return output


def dice(pred: np.ndarray, target: np.ndarray, class_id: int) -> Optional[float]:
    p = pred == class_id
    g = target == class_id
    denom = int(p.sum()) + int(g.sum())
    if denom == 0:
        return None
    return 2.0 * int((p & g).sum()) / float(denom)


def otsu_threshold(values: np.ndarray, bins: int = 64) -> Optional[float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2 or float(values.max()) <= float(values.min()):
        return None
    hist, edges = np.histogram(values, bins=bins)
    weights = hist.astype(np.float64)
    total = weights.sum()
    if total <= 0:
        return None
    weights /= total
    centers = 0.5 * (edges[:-1] + edges[1:])
    omega = np.cumsum(weights)
    mu = np.cumsum(weights * centers)
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    sigma_b = np.full_like(omega, np.nan)
    valid = denom > 1e-12
    sigma_b[valid] = (mu_t * omega[valid] - mu[valid]) ** 2 / denom[valid]
    if not np.isfinite(sigma_b).any():
        return None
    idx = int(np.nanargmax(sigma_b))
    return float(0.5 * (edges[idx] + edges[idx + 1]))


def kmeans2_high_mask(values: np.ndarray, n_iter: int = 25) -> Optional[np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2 or float(values.max()) <= float(values.min()):
        return None
    low, high = np.percentile(values, [20.0, 80.0])
    if high <= low:
        low, high = float(values.min()), float(values.max())
    for _ in range(n_iter):
        assign_high = np.abs(values - high) < np.abs(values - low)
        if assign_high.any():
            high = float(values[assign_high].mean())
        if (~assign_high).any():
            low = float(values[~assign_high].mean())
        if high < low:
            low, high = high, low
            assign_high = ~assign_high
    return assign_high


def apply_threshold(
    image: np.ndarray,
    pathology: np.ndarray,
    threshold: Optional[float],
) -> np.ndarray:
    high = np.zeros_like(pathology)
    if threshold is None:
        return high
    high[pathology] = image[pathology] > threshold
    return high


def split_all_scar(image: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return pred == 4


def split_all_edema(image: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return np.zeros(pred.shape, dtype=bool)


def split_otsu_volume(image: np.ndarray, pred: np.ndarray) -> np.ndarray:
    pathology = pred == 4
    values = image[pathology]
    if values.size < MIN_SPLIT_VOXELS:
        return pathology
    return apply_threshold(image, pathology, otsu_threshold(values))


def split_otsu_slice(image: np.ndarray, pred: np.ndarray) -> np.ndarray:
    pathology = pred == 4
    high = np.zeros_like(pathology)
    for z in range(pred.shape[0]):
        mask = pathology[z]
        if int(mask.sum()) < MIN_SPLIT_VOXELS:
            high[z] = mask
            continue
        threshold = otsu_threshold(image[z][mask])
        if threshold is None:
            high[z] = mask
        else:
            high[z] = mask & (image[z] > threshold)
    return high


def split_kmeans_volume(image: np.ndarray, pred: np.ndarray) -> np.ndarray:
    pathology = pred == 4
    values = image[pathology]
    if values.size < MIN_SPLIT_VOXELS:
        return pathology
    assign = kmeans2_high_mask(values)
    high = np.zeros_like(pathology)
    if assign is None:
        return pathology
    high[pathology] = assign
    return high


def split_kmeans_slice(image: np.ndarray, pred: np.ndarray) -> np.ndarray:
    pathology = pred == 4
    high = np.zeros_like(pathology)
    for z in range(pred.shape[0]):
        mask = pathology[z]
        if int(mask.sum()) < MIN_SPLIT_VOXELS:
            high[z] = mask
            continue
        assign = kmeans2_high_mask(image[z][mask])
        if assign is None:
            high[z] = mask
        else:
            high[z][mask] = assign
    return high


def split_nsd(k: float) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    def _split(image: np.ndarray, pred: np.ndarray) -> np.ndarray:
        pathology = pred == 4
        remote = pred == 3
        if int(pathology.sum()) < MIN_SPLIT_VOXELS:
            return pathology
        if int(remote.sum()) < MIN_SPLIT_VOXELS:
            return split_otsu_volume(image, pred)
        remote_values = image[remote].astype(np.float64)
        threshold = float(remote_values.mean() + k * remote_values.std())
        return apply_threshold(image, pathology, threshold)

    _split.__name__ = f"nsd_{k:g}"
    return _split


def split_fwhm_remote(image: np.ndarray, pred: np.ndarray) -> np.ndarray:
    pathology = pred == 4
    remote = pred == 3
    if int(pathology.sum()) < MIN_SPLIT_VOXELS:
        return pathology
    peak = float(image[pathology].max())
    if int(remote.sum()) >= MIN_SPLIT_VOXELS:
        base = float(image[remote].mean())
    else:
        base = float(image[pathology].min())
    return apply_threshold(image, pathology, 0.5 * (peak + base))


def split_nearest_lv_vs_remote(image: np.ndarray, pred: np.ndarray) -> np.ndarray:
    pathology = pred == 4
    lv = pred == 1
    remote = pred == 3
    if int(pathology.sum()) < MIN_SPLIT_VOXELS:
        return pathology
    if int(lv.sum()) < MIN_SPLIT_VOXELS or int(remote.sum()) < MIN_SPLIT_VOXELS:
        return split_otsu_volume(image, pred)
    mu_lv = float(image[lv].mean())
    mu_remote = float(image[remote].mean())
    values = image[pathology].astype(np.float64)
    high = np.zeros_like(pathology)
    high[pathology] = np.abs(values - mu_lv) <= np.abs(values - mu_remote)
    return high


def split_oracle(gt_split: np.ndarray) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    def _split(image: np.ndarray, pred: np.ndarray) -> np.ndarray:
        pathology = pred == 4
        high = np.zeros_like(pathology)
        high[pathology] = gt_split[pathology] == 4
        # False-positive pathology (not GT scar/edema) follows intensity.
        unknown = pathology & (gt_split != 4) & (gt_split != 5)
        if unknown.any():
            known = pathology & ~unknown
            if int(known.sum()) >= MIN_SPLIT_VOXELS:
                threshold = otsu_threshold(image[known])
            else:
                threshold = otsu_threshold(image[pathology])
            if threshold is not None:
                high[unknown] = image[unknown] > threshold
        return high

    _split.__name__ = "oracle_gt_inside_pred"
    return _split


def write_split(pred: np.ndarray, high_scar: np.ndarray) -> np.ndarray:
    output = pred.astype(np.int16, copy=True)
    pathology = pred == 4
    output[pathology] = 5
    output[high_scar] = 4
    return output


def summarize_cases(
    rows: List[dict],
    class_ids: Tuple[int, ...] = (1, 2, 3, 4, 5),
) -> dict:
    present_macros = [
        row["macro_gt_present"]
        for row in rows
        if row["macro_gt_present"] is not None
    ]
    per_class = {}
    for class_id in class_ids:
        values = [
            row["dice"][str(class_id)]
            for row in rows
            if str(class_id) in row["gt_present"]
            and row["dice"][str(class_id)] is not None
        ]
        per_class[SPLIT_NAMES[class_id]] = float(np.mean(values)) if values else None
    return {
        "n_cases": len(rows),
        "mean_dice_gt_present": float(np.mean(present_macros)) if present_macros else None,
        "mean_dice_per_class_gt_present": per_class,
    }


def evaluate_split(pred_split: np.ndarray, gt_split: np.ndarray) -> dict:
    scores = {}
    present = []
    present_ids = []
    for class_id in range(1, 6):
        score = dice(pred_split, gt_split, class_id)
        scores[str(class_id)] = score
        if (gt_split == class_id).any():
            present_ids.append(str(class_id))
            if score is not None:
                present.append(score)
    return {
        "dice": scores,
        "gt_present": present_ids,
        "macro_gt_present": float(np.mean(present)) if present else None,
    }


def coverage_stats(pred: np.ndarray, gt_split: np.ndarray, image: np.ndarray) -> dict:
    pathology = pred == 4
    gt_scar = gt_split == 4
    gt_edema = gt_split == 5
    gt_path = gt_scar | gt_edema
    pred_n = int(pathology.sum())
    return {
        "pred_pathology_voxels": pred_n,
        "gt_scar_voxels": int(gt_scar.sum()),
        "gt_edema_voxels": int(gt_edema.sum()),
        "scar_recall_in_pred_pathology": (
            float((pathology & gt_scar).sum() / gt_scar.sum()) if gt_scar.any() else None
        ),
        "edema_recall_in_pred_pathology": (
            float((pathology & gt_edema).sum() / gt_edema.sum()) if gt_edema.any() else None
        ),
        "union_recall": (
            float((pathology & gt_path).sum() / gt_path.sum()) if gt_path.any() else None
        ),
        "pred_pathology_composition": {
            "scar": float((pathology & gt_scar).sum() / pred_n) if pred_n else None,
            "edema": float((pathology & gt_edema).sum() / pred_n) if pred_n else None,
            "other": float((pathology & ~gt_path).sum() / pred_n) if pred_n else None,
        },
        "intensity": {
            name: {
                "n": int(mask.sum()),
                "mean": float(image[mask].mean()) if mask.any() else None,
                "median": float(np.median(image[mask])) if mask.any() else None,
            }
            for name, mask in (
                ("gt_scar", gt_scar),
                ("gt_edema", gt_edema),
                ("gt_normal_myo", gt_split == 3),
                ("pred_pathology", pathology),
                ("pred_normal_myo", pred == 3),
            )
        },
    }


def polarity_on_overlap(
    image: np.ndarray,
    pred: np.ndarray,
    gt_split: np.ndarray,
) -> Optional[dict]:
    overlap = (pred == 4) & ((gt_split == 4) | (gt_split == 5))
    if int(overlap.sum()) < MIN_SPLIT_VOXELS:
        return None
    values = image[overlap].astype(np.float64)
    is_scar = gt_split[overlap] == 4
    if not is_scar.any() or is_scar.all():
        return None
    threshold = otsu_threshold(values)
    if threshold is None:
        return None
    pred_high_scar = values > threshold
    acc_high_scar = float((pred_high_scar == is_scar).mean())
    acc_high_edema = 1.0 - acc_high_scar
    return {
        "n_overlap": int(overlap.sum()),
        "overlap_accuracy_high_is_scar": acc_high_scar,
        "overlap_accuracy_high_is_edema": acc_high_edema,
        "preferred_polarity": "high_is_scar" if acc_high_scar >= acc_high_edema else "high_is_edema",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred_dir",
        default=(
            "/data4/baihexiang/SegMan/spatial_combination/experiments/"
            "lge_roi_v2_z1_b18_100ep_rerun_20260817/eval_test_best/pred_nii"
        ),
    )
    parser.add_argument(
        "--gt_dir",
        default=(
            "/data4/baihexiang/SegMan/OODKA/nnUNet/nnUNetFrame/DATASET/"
            "nnUNet_raw/nnUNet_raw_data/Dataset011_MYO_LGE_BC_OOD/labelsTs"
        ),
    )
    parser.add_argument(
        "--image_dir",
        default=(
            "/data4/baihexiang/SegMan/OODKA/nnUNet/nnUNetFrame/DATASET/"
            "nnUNet_raw/nnUNet_raw_data/Dataset011_MYO_LGE_BC_OOD/imagesTs"
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=(
            "/data4/baihexiang/SegMan/spatial_combination/experiments/"
            "lge_roi_v2_z1_b18_100ep_rerun_20260817/analysis/"
            "unsupervised_scar_edema_split"
        ),
    )
    args = parser.parse_args()

    methods: List[Tuple[str, Callable]] = [
        ("all_scar", split_all_scar),
        ("all_edema", split_all_edema),
        ("otsu_volume", split_otsu_volume),
        ("otsu_slice", split_otsu_slice),
        ("kmeans2_volume", split_kmeans_volume),
        ("kmeans2_slice", split_kmeans_slice),
        ("nsd_2", split_nsd(2.0)),
        ("nsd_3", split_nsd(3.0)),
        ("nsd_5", split_nsd(5.0)),
        ("fwhm_remote", split_fwhm_remote),
        ("nearest_lv_vs_remote", split_nearest_lv_vs_remote),
    ]

    case_ids = [
        name[:-7] if name.endswith(".nii.gz") else name
        for name in sorted(os.listdir(args.pred_dir))
        if name.endswith(".nii.gz")
    ]
    method_rows: Dict[str, List[dict]] = {name: [] for name, _ in methods}
    method_rows["oracle_gt_inside_pred"] = []
    coverages = []
    polarities = []

    for case_id in case_ids:
        pred = sitk.GetArrayFromImage(
            sitk.ReadImage(os.path.join(args.pred_dir, case_id + ".nii.gz"))
        )
        gt = remap_gt(
            sitk.GetArrayFromImage(
                sitk.ReadImage(os.path.join(args.gt_dir, case_id + ".nii.gz"))
            )
        )
        image = sitk.GetArrayFromImage(
            sitk.ReadImage(os.path.join(args.image_dir, case_id + "_0000.nii.gz"))
        ).astype(np.float32)
        if pred.shape != gt.shape or pred.shape != image.shape:
            raise ValueError(f"{case_id}: shape mismatch {pred.shape} {gt.shape} {image.shape}")

        coverages.append({"case_id": case_id, **coverage_stats(pred, gt, image)})
        polarity = polarity_on_overlap(image, pred, gt)
        if polarity is not None:
            polarities.append({"case_id": case_id, **polarity})

        oracle_fn = split_oracle(gt)
        for name, fn in methods + [("oracle_gt_inside_pred", oracle_fn)]:
            high = fn(image, pred)
            pred_split = write_split(pred, high)
            row = {"case_id": case_id, **evaluate_split(pred_split, gt)}
            method_rows[name].append(row)

    summaries = {
        name: summarize_cases(rows) for name, rows in method_rows.items()
    }
    coverage_means = {
        "scar_recall_in_pred_pathology": float(
            np.mean(
                [
                    row["scar_recall_in_pred_pathology"]
                    for row in coverages
                    if row["scar_recall_in_pred_pathology"] is not None
                ]
            )
        ),
        "edema_recall_in_pred_pathology": float(
            np.mean(
                [
                    row["edema_recall_in_pred_pathology"]
                    for row in coverages
                    if row["edema_recall_in_pred_pathology"] is not None
                ]
            )
        ),
        "union_recall": float(
            np.mean(
                [
                    row["union_recall"]
                    for row in coverages
                    if row["union_recall"] is not None
                ]
            )
        ),
        "pred_pathology_composition": {
            key: float(
                np.mean(
                    [
                        row["pred_pathology_composition"][key]
                        for row in coverages
                        if row["pred_pathology_composition"][key] is not None
                    ]
                )
            )
            for key in ("scar", "edema", "other")
        },
        "intensity_mean": {
            key: float(
                np.mean(
                    [
                        row["intensity"][key]["mean"]
                        for row in coverages
                        if row["intensity"][key]["mean"] is not None
                    ]
                )
            )
            for key in (
                "gt_scar",
                "gt_edema",
                "gt_normal_myo",
                "pred_pathology",
                "pred_normal_myo",
            )
        },
        "overlap_polarity": {
            "n_cases": len(polarities),
            "mean_accuracy_high_is_scar": (
                float(np.mean([row["overlap_accuracy_high_is_scar"] for row in polarities]))
                if polarities
                else None
            ),
            "fraction_preferring_high_is_scar": (
                float(
                    np.mean(
                        [
                            row["preferred_polarity"] == "high_is_scar"
                            for row in polarities
                        ]
                    )
                )
                if polarities
                else None
            ),
        },
    }

    os.makedirs(args.output_dir, exist_ok=True)
    payload = {
        "note": (
            "Unsupervised split of ROI-v2 predicted scar+edema on LGE intensity. "
            "LV/RV/normal myocardium are copied from the original prediction. "
            "High intensity is assigned to scar."
        ),
        "n_cases": len(case_ids),
        "coverage": coverage_means,
        "methods": summaries,
        "cases": {
            "coverage": coverages,
            "polarity": polarities,
            "dice": method_rows,
        },
    }
    output_path = os.path.join(args.output_dir, "summary.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    table = {
        key: value
        for key, value in payload.items()
        if key != "cases"
    }
    print(json.dumps(table, indent=2))
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
