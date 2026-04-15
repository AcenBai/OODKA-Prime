"""nnUNet 2D baseline evaluation on test / OOD cases."""

from __future__ import annotations

import csv
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

from ..config import EvalConfig, ensure_nnunet_on_path
from ..utils.io_utils import (
    read_nifti_as_zyx,
    read_nifti_as_zyx_with_spacing,
    discover_case_ids_from_dir,
    find_raw_image_files,
    maybe_mkdir_p,
)
from ..utils.metrics import (
    dice_no_ignore,
    raw_per_class_metrics,
)


def evaluate_nnunet_2d(cfg: EvalConfig):
    """Run nnUNet 2D inference on test cases, export predictions, compute metrics."""
    ensure_nnunet_on_path()
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from batchgenerators.utilities.file_and_folder_operations import load_json

    device_t = __import__("torch").device(cfg.device)

    predictor = nnUNetPredictor(
        tile_step_size=cfg.tile_step_size,
        use_gaussian=True,
        use_mirroring=False,
        device=device_t,
        verbose=False,
    )
    predictor.initialize_from_trained_model_folder(
        cfg.nnunet_model_dir,
        use_folds=(cfg.fold,),
        checkpoint_name="checkpoint_best.pth",
    )

    dataset_json = load_json(cfg.dataset_json_path)
    fe = dataset_json.get("file_ending", ".nii.gz")
    labels_dict = dataset_json.get("labels", {})
    class_ids = sorted(int(v) for k, v in labels_dict.items() if int(v) > 0)

    test_ids = discover_case_ids_from_dir(cfg.labelsTs_dir, fe)
    if not test_ids:
        test_ids = discover_case_ids_from_dir(cfg.imagesTs_dir, fe, strip_modality=True)
    if not test_ids:
        raise FileNotFoundError(f"No test cases found in {cfg.imagesTs_dir} / {cfg.labelsTs_dir}")
    if cfg.case_limit > 0:
        test_ids = test_ids[:cfg.case_limit]
    print(f"Evaluating {len(test_ids)} test cases")

    pred_dir = os.path.join(cfg.out_dir, "predictions")
    maybe_mkdir_p(pred_dir)

    all_rows = []
    for case_id in tqdm(test_ids, desc="nnUNet eval"):
        image_files = find_raw_image_files(cfg.imagesTs_dir, case_id, fe)
        if not image_files:
            image_files = find_raw_image_files(cfg.imagesTr_dir, case_id, fe)
        label_path = os.path.join(cfg.labelsTs_dir, case_id + fe)
        if not os.path.isfile(label_path):
            label_path = os.path.join(cfg.labelsTr_dir, case_id + fe)

        gt_arr, spacing_xyz = read_nifti_as_zyx_with_spacing(label_path)
        gt_arr = gt_arr.astype(np.int16)

        import torch
        img_arrays = [sitk.GetArrayFromImage(sitk.ReadImage(f)).astype(np.float32) for f in image_files]
        data_np = np.stack(img_arrays, axis=0) if len(img_arrays) > 1 else img_arrays[0][None]

        pred_logits = predictor.predict_single_npy_array(
            data_np, {"spacing": list(reversed(spacing_xyz))}
        )
        if isinstance(pred_logits, tuple):
            pred_logits = pred_logits[0]
        pred_seg = np.argmax(pred_logits, axis=0).astype(np.int16)

        # Export prediction NIfTI
        ref_img = sitk.ReadImage(label_path)
        pred_img = sitk.GetImageFromArray(pred_seg)
        pred_img.CopyInformation(ref_img)
        sitk.WriteImage(pred_img, os.path.join(pred_dir, case_id + fe))

        # Metrics
        dice_pc, dice_gt, dice_any = dice_no_ignore(pred_seg, gt_arr, class_ids)
        sens, spec, prec, rec, acc, hd95 = raw_per_class_metrics(
            pred_seg, gt_arr, class_ids, spacing_xyz
        )

        row = {"case_id": case_id, "dice_mean_gt": dice_gt, "dice_mean_any": dice_any}
        for c in class_ids:
            row[f"dice_{c}"] = dice_pc.get(c)
            row[f"prec_{c}"] = prec.get(c)
            row[f"rec_{c}"] = rec.get(c)
            row[f"hd95_{c}"] = hd95.get(c)
        all_rows.append(row)

    # Save results
    csv_path = os.path.join(cfg.out_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)

    summary = {
        "n_cases": len(test_ids),
        "mean_dice_gt_present": float(np.mean([r["dice_mean_gt"] for r in all_rows])),
    }
    for c in class_ids:
        vals = [r[f"dice_{c}"] for r in all_rows if r[f"dice_{c}"] is not None]
        summary[f"dice_{c}_mean"] = float(np.mean(vals)) if vals else None
    with open(os.path.join(cfg.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {cfg.out_dir}")
    print(f"Mean Dice (GT-present classes): {summary['mean_dice_gt_present']:.4f}")
    return summary
