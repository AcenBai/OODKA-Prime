#!/usr/bin/env python3
"""Evaluate existing five-label LGE predictions after scar/edema merging."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import SimpleITK as sitk


def remap(array: np.ndarray) -> np.ndarray:
    output = np.zeros_like(array, dtype=np.int16)
    output[array == 3] = 1
    output[array == 5] = 2
    output[array == 4] = 3
    output[(array == 1) | (array == 2)] = 4
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    case_rows = []
    for filename in sorted(name for name in os.listdir(args.pred_dir) if name.endswith(".nii.gz")):
        pred = remap(sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(args.pred_dir, filename))))
        target = remap(sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(args.gt_dir, filename))))
        scores = {}
        present = []
        for class_id in range(1, 5):
            pred_mask = pred == class_id
            target_mask = target == class_id
            denominator = int(pred_mask.sum() + target_mask.sum())
            score = 2.0 * int((pred_mask & target_mask).sum()) / max(1, denominator)
            scores[str(class_id)] = score
            if target_mask.any():
                present.append(score)
        case_rows.append(
            {
                "case_id": filename.removesuffix(".nii.gz"),
                "dice": scores,
                "gt_present": [str(class_id) for class_id in range(1, 5) if (target == class_id).any()],
                "macro_gt_present": float(np.mean(present)) if present else None,
            }
        )
    summary = {
        "n_cases": len(case_rows),
        "class_names": {"1": "LV", "2": "RV", "3": "normal_myo", "4": "scar_edema_on_myo"},
        "mean_of_case_macro_gt_present": float(
            np.mean([row["macro_gt_present"] for row in case_rows if row["macro_gt_present"] is not None])
        ),
        "mean_dice_per_class_gt_present": {
            str(class_id): float(
                np.mean(
                    [
                        row["dice"][str(class_id)]
                        for row in case_rows
                        if str(class_id) in row["gt_present"]
                    ]
                )
            )
            for class_id in range(1, 5)
        },
        "cases": case_rows,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps({key: value for key, value in summary.items() if key != "cases"}, indent=2))


if __name__ == "__main__":
    main()
