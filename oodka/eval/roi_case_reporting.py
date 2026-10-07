"""Per-case label restoration, scoring, and NIfTI export for ROI evaluation."""

from __future__ import annotations

import os

import numpy as np
import SimpleITK as sitk

from ..models.prompts import MYOPS_LGE_ROI_FINAL_GROUPS
from ..utils.metrics import dice_no_ignore, precision_recall_hd95_no_ignore
from ..utils.postprocessing import keep_largest_component_per_class


def remap_gt(
    array: np.ndarray, groups=MYOPS_LGE_ROI_FINAL_GROUPS
) -> np.ndarray:
    """Map raw dataset labels to consecutive evaluation class IDs."""
    output = np.zeros(array.shape, dtype=np.int16)
    for class_id, source_ids in enumerate(groups, start=1):
        output[np.isin(array, source_ids)] = class_id
    return output


def _write_prediction(prediction, label_path: str, output_path: str) -> None:
    output = sitk.GetImageFromArray(prediction.astype(np.int16))
    output.CopyInformation(sitk.ReadImage(label_path))
    sitk.WriteImage(output, output_path)


def score_independent_case(
    *,
    case_id: str,
    final_prompt_logits: np.ndarray,
    target: np.ndarray,
    spacing: tuple,
    preprocessor,
    properties: dict,
    raw_shape: tuple[int, ...],
    threshold: float,
    postprocess: str,
    independent_dirs: dict[int, str],
    independent_predicted: np.ndarray,
    independent_target: np.ndarray,
    label_path: str,
    ending: str,
) -> dict:
    """Score non-exclusive binary masks and write one NIfTI per class."""
    class_count = final_prompt_logits.shape[0]
    threshold_logit = float(np.log(threshold / (1.0 - threshold)))
    row = {"case_id": case_id}
    present_scores = []
    for class_id in range(1, class_count + 1):
        binary = preprocessor.prompt_logits_to_raw_segmentation(
            final_prompt_logits[class_id - 1 : class_id] - threshold_logit,
            {0: 1},
            properties,
        ) > 0
        if tuple(binary.shape) != raw_shape:
            raise ValueError(f"{case_id}: restored={binary.shape}, raw={raw_shape}")
        if postprocess == "largest_per_class":
            binary = keep_largest_component_per_class(
                binary.astype(np.int16), (1,)
            ) > 0
        binary_target = target == class_id
        denominator = int(binary.sum() + binary_target.sum())
        raw_dice = (
            2.0 * int((binary & binary_target).sum()) / denominator
            if denominator > 0 else None
        )
        dice_value = raw_dice if binary_target.any() else None
        row[f"dice_{class_id}"] = dice_value
        if dice_value is not None:
            present_scores.append(dice_value)
        precision, recall, hd95 = precision_recall_hd95_no_ignore(
            binary.astype(np.int16), binary_target.astype(np.int16),
            (1,), spacing,
        )
        row[f"prec_{class_id}"] = precision.get(1)
        row[f"rec_{class_id}"] = recall.get(1)
        row[f"hd95_{class_id}"] = hd95.get(1)
        independent_predicted[class_id] += int(binary.sum())
        independent_target[class_id] += int(binary_target.sum())
        _write_prediction(
            binary, label_path,
            os.path.join(independent_dirs[class_id], case_id + ending),
        )
    row["dice_mean_gt"] = (
        float(np.mean(present_scores)) if present_scores else None
    )
    # Keep CSV column order identical to the historical evaluator.
    ordered = {"case_id": row.pop("case_id"), "dice_mean_gt": row.pop("dice_mean_gt")}
    ordered.update(row)
    return ordered


def score_exclusive_case(
    *,
    case_id: str,
    final_prompt_logits: np.ndarray,
    target: np.ndarray,
    spacing: tuple,
    preprocessor,
    properties: dict,
    raw_shape: tuple[int, ...],
    postprocess: str,
    confusion: np.ndarray,
    pred_dir: str,
    label_path: str,
    ending: str,
) -> dict:
    """Restore one exclusive segmentation, update confusion, and export it."""
    class_count = final_prompt_logits.shape[0]
    prediction = preprocessor.prompt_logits_to_raw_segmentation(
        final_prompt_logits,
        {index: index + 1 for index in range(class_count)},
        properties,
    )
    if tuple(prediction.shape) != raw_shape:
        raise ValueError(
            f"{case_id}: restored={prediction.shape}, raw={raw_shape}"
        )
    if postprocess == "largest_per_class":
        prediction = keep_largest_component_per_class(
            prediction, tuple(range(1, class_count + 1))
        )
    width = class_count + 1
    encoded = target.astype(np.int64) * width + prediction.astype(np.int64)
    confusion += np.bincount(
        encoded.ravel(), minlength=width * width
    ).reshape(width, width)
    dice_pc, dice_mean, _ = dice_no_ignore(
        prediction, target, tuple(range(1, class_count + 1))
    )
    precision, recall, hd95 = precision_recall_hd95_no_ignore(
        prediction, target, tuple(range(1, class_count + 1)), spacing
    )
    row = {"case_id": case_id, "dice_mean_gt": dice_mean}
    for class_id in range(1, class_count + 1):
        row[f"dice_{class_id}"] = dice_pc.get(class_id)
        row[f"prec_{class_id}"] = precision.get(class_id)
        row[f"rec_{class_id}"] = recall.get(class_id)
        row[f"hd95_{class_id}"] = hd95.get(class_id)
    _write_prediction(prediction, label_path, os.path.join(pred_dir, case_id + ending))
    return row
