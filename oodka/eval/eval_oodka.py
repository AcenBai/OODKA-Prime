"""OODKA sliding-window evaluation on test / OOD cases."""

from __future__ import annotations

import csv
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk
import torch
from tqdm import tqdm

from ..config import EvalConfig, ensure_nnunet_on_path
from ..preprocess.biomedparse_preprocessor import preprocess_case_online
from ..train.forward import predict_patch_logits_per_class
from ..utils.io_utils import (
    discover_case_ids_from_dir,
    find_raw_image_files,
    read_nifti_as_zyx_with_spacing,
    maybe_mkdir_p,
)
from ..utils.metrics import dice_no_ignore, precision_recall_hd95_no_ignore
from ..utils.postprocessing import keep_largest_component_per_class


def _pad_to_min_shape(arr: np.ndarray, min_shape_zyx, pad_value):
    z, y, x = arr.shape[-3:]
    pz = max(0, min_shape_zyx[0] - z)
    py = max(0, min_shape_zyx[1] - y)
    px = max(0, min_shape_zyx[2] - x)
    p = ((pz // 2, pz - pz // 2), (py // 2, py - py // 2), (px // 2, px - px // 2))
    if arr.ndim == 4:
        return np.pad(arr, ((0, 0),) + tuple(p), constant_values=pad_value), p
    return np.pad(arr, p, constant_values=pad_value), p


def _crop_from_pad(arr: np.ndarray, pad, orig_shape):
    pz0, _ = pad[0]; py0, _ = pad[1]; px0, _ = pad[2]
    z, y, x = orig_shape
    if arr.ndim == 4:
        return arr[:, pz0:pz0+z, py0:py0+y, px0:px0+x]
    return arr[pz0:pz0+z, py0:py0+y, px0:px0+x]


def evaluate_oodka_sliding_window(
    cfg: EvalConfig,
    model_nnunet,
    model_biomedparse,
    fusion_modules: Dict[str, torch.nn.Module],
    prompt_features: dict,
    prompt_to_class_id: Dict[int, int],
    P: int,
    patch_size: List[int],
):
    """Full-volume sliding window OODKA evaluation."""
    ensure_nnunet_on_path()
    from nnunetv2.inference.sliding_window_prediction import compute_steps_for_sliding_window, compute_gaussian
    from batchgenerators.utilities.file_and_folder_operations import load_json

    device = torch.device(cfg.device)
    dataset_json = load_json(cfg.dataset_json_path)
    fe = dataset_json.get("file_ending", ".nii.gz")
    labels_dict = dataset_json.get("labels", {})
    class_ids = sorted(int(v) for k, v in labels_dict.items() if int(v) > 0)

    test_ids = discover_case_ids_from_dir(cfg.labelsTs_dir, fe)
    if not test_ids:
        test_ids = discover_case_ids_from_dir(cfg.imagesTs_dir, fe, strip_modality=True)
    if cfg.case_limit > 0:
        test_ids = test_ids[:cfg.case_limit]

    maybe_mkdir_p(cfg.out_dir)
    pred_dir = os.path.join(cfg.out_dir, "pred_nii")
    maybe_mkdir_p(pred_dir)

    patch_size_zyx = tuple(int(x) for x in patch_size)
    gaussian = compute_gaussian(
        patch_size_zyx, sigma_scale=1.0/8, value_scaling_factor=10,
        device=torch.device("cpu"),
    ).cpu().numpy().astype(np.float32)

    # Set all fusion modules to eval
    for m in fusion_modules.values():
        m.eval()

    all_rows = []
    for case_id in tqdm(test_ids, desc="OODKA eval"):
        image_files = find_raw_image_files(cfg.imagesTs_dir, case_id, fe)
        if not image_files:
            image_files = find_raw_image_files(cfg.imagesTr_dir, case_id, fe)
        label_path = os.path.join(cfg.labelsTs_dir, case_id + fe)
        if not os.path.isfile(label_path):
            label_path = os.path.join(cfg.labelsTr_dir, case_id + fe)

        gt_arr, spacing_xyz = read_nifti_as_zyx_with_spacing(label_path)
        gt_arr = gt_arr.astype(np.int16)

        # Online preprocess
        data_nn, seg_nn, data_bp, seg_bp, props_nn, props_bp = preprocess_case_online(
            image_files=image_files,
            seg_file=label_path,
            plans_path=cfg.plans_path,
            dataset_json_path=cfg.dataset_json_path,
            norm_mode=cfg.norm_mode,
            window_level=cfg.window_level,
            window_width=cfg.window_width,
            low_percentile=cfg.low_percentile,
            high_percentile=cfg.high_percentile,
        )

        nn_data = data_nn.astype(np.float32)
        bp_data = data_bp.astype(np.float32)
        Z, Y, X = nn_data.shape[1], nn_data.shape[2], nn_data.shape[3]

        nn_data_p, pad = _pad_to_min_shape(nn_data, patch_size_zyx, 0.0)
        bp_data_p, _ = _pad_to_min_shape(bp_data, patch_size_zyx, 0.0)
        Zp, Yp, Xp = nn_data_p.shape[-3:]

        steps = compute_steps_for_sliding_window((Zp, Yp, Xp), patch_size_zyx, float(cfg.tile_step_size))
        slicers = [
            (slice(z0, z0 + patch_size_zyx[0]), slice(y0, y0 + patch_size_zyx[1]), slice(x0, x0 + patch_size_zyx[2]))
            for z0 in steps[0] for y0 in steps[1] for x0 in steps[2]
        ]

        logits_sum = np.zeros((P, Zp, Yp, Xp), dtype=np.float32)
        wsum = np.zeros((Zp, Yp, Xp), dtype=np.float32)
        is_multi = bp_data_p.ndim == 4

        with torch.no_grad():
            for slz, sly, slx in slicers:
                nn_patch = nn_data_p[:, slz, sly, slx]
                if is_multi:
                    bp_patch_t = torch.from_numpy(bp_data_p[:, slz, sly, slx][None]).float()
                else:
                    bp_patch_t = torch.from_numpy(bp_data_p[slz, sly, slx][None]).float()
                nn_patch_t = torch.from_numpy(nn_patch[None]).float()

                plogits = predict_patch_logits_per_class(
                    nnunet_patch_5d=nn_patch_t,
                    biomedparse_patch_4d=bp_patch_t,
                    patch_size=patch_size,
                    prompt_features=prompt_features,
                    P=P,
                    model_nnunet=model_nnunet,
                    model_biomedparse=model_biomedparse,
                    fusion_modules=fusion_modules,
                    device=device,
                ).cpu().numpy().astype(np.float32)

                logits_sum[:, slz, sly, slx] += plogits * gaussian[None]
                wsum[slz, sly, slx] += gaussian

        fused = logits_sum / np.clip(wsum[None], 1e-6, None)
        fused = _crop_from_pad(fused, pad, (Z, Y, X))

        full_logits = np.concatenate([np.zeros((1, Z, Y, X), dtype=np.float32), fused], axis=0)
        pred_compact = np.argmax(full_logits, axis=0).astype(np.int16)

        pred_orig = np.zeros((Z, Y, X), dtype=np.int16)
        for pi in range(P):
            pred_orig[pred_compact == (pi + 1)] = int(prompt_to_class_id[pi])

        # Export to original NIfTI space
        from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape
        try:
            ref_img = sitk.ReadImage(label_path)
            pred_resized = convert_predicted_logits_to_segmentation_with_correct_shape(
                np.stack([1 - full_logits.max(axis=0, keepdims=True).squeeze(0)] + list(fused), axis=0),
                props_nn,
                dataset_json,
                plans_manager=None, configuration_manager=None,
                return_probabilities=False,
            )
        except Exception:
            pred_resized = pred_orig

        ref_img = sitk.ReadImage(label_path)
        out_img = sitk.GetImageFromArray(pred_resized.astype(np.int16) if isinstance(pred_resized, np.ndarray) else pred_orig)
        out_img.CopyInformation(ref_img)
        sitk.WriteImage(out_img, os.path.join(pred_dir, case_id + fe))

        pred_seg = keep_largest_component_per_class(pred_orig, class_ids)

        dice_pc, dice_gt, dice_any = dice_no_ignore(pred_seg, gt_arr, class_ids)
        prec_pc, rec_pc, hd95_pc = precision_recall_hd95_no_ignore(pred_seg, gt_arr, class_ids, spacing_xyz)

        row = {"case_id": case_id, "dice_mean_gt": dice_gt}
        for c in class_ids:
            row[f"dice_{c}"] = dice_pc.get(c)
            row[f"prec_{c}"] = prec_pc.get(c)
            row[f"rec_{c}"] = rec_pc.get(c)
            row[f"hd95_{c}"] = hd95_pc.get(c)
        all_rows.append(row)

    # Save
    csv_path = os.path.join(cfg.out_dir, "metrics.csv")
    if all_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)

    summary = {
        "n_cases": len(test_ids),
        "mean_dice_gt_present": float(np.mean([r["dice_mean_gt"] for r in all_rows])) if all_rows else 0.0,
    }
    for c in class_ids:
        vals = [r[f"dice_{c}"] for r in all_rows if r.get(f"dice_{c}") is not None]
        summary[f"dice_{c}_mean"] = float(np.mean(vals)) if vals else None
    with open(os.path.join(cfg.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nOODKA eval complete. Results: {cfg.out_dir}")
    print(f"Mean Dice: {summary['mean_dice_gt_present']:.4f}")
    return summary
