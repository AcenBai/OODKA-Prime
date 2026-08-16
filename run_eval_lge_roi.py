#!/usr/bin/env python3
"""Two-pass predicted-ROI inference for the four-class LGE experiment."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from tqdm import tqdm

from oodka.config import EvalConfig
from oodka.data.aligned_preprocessing import AlignedBiomedParsePreprocessor
from oodka.data.lge_roi import (
    ROIGenerator,
    hard_switch_foreground_logits,
    restore_roi_logits,
    roi_diagnostics,
)
from oodka.data.slice_dataset import make_biomedparse_block
from oodka.models.prompts import (
    MYOPS_LGE_ROI_ANATOMY_PROMPTS,
    MYOPS_LGE_ROI_FINAL_NAMES,
    MYOPS_LGE_ROI_REFINEMENT_PROMPTS,
    MYOPS_LGE_ROI_V2_REFINEMENT_PROMPTS,
)
from oodka.train.forward import predict_block_logits_per_class
from oodka.train.model_builder import (
    build_fusion_modules,
    build_prompt_features,
    load_frozen_biomedparse,
)
from oodka.utils.io_utils import (
    discover_case_ids_from_dir,
    find_raw_image_files,
    maybe_mkdir_p,
    read_nifti_as_zyx_with_spacing,
)
from oodka.utils.metrics import dice_no_ignore, precision_recall_hd95_no_ignore


def _remap_gt(array: np.ndarray) -> np.ndarray:
    output = np.zeros(array.shape, dtype=np.int16)
    output[array == 3] = 1
    output[array == 5] = 2
    output[array == 4] = 3
    output[(array == 1) | (array == 2)] = 4
    return output


def _hierarchical_foreground_logits(
    anatomy_logits: torch.Tensor,
    refinement_logits: torch.Tensor,
    *,
    selected_logit: float = 20.0,
    rejected_logit: float = -20.0,
) -> torch.Tensor:
    """Encode strict coarse-to-fine decisions as four foreground logits."""
    if anatomy_logits.ndim != 5 or anatomy_logits.shape[1:3] != (3, 1):
        raise ValueError("anatomy_logits must be [B,3,1,H,W]")
    if refinement_logits.ndim != 5 or refinement_logits.shape[1:3] != (2, 1):
        raise ValueError("refinement_logits must be [B,2,1,H,W]")
    if (
        anatomy_logits.shape[0] != refinement_logits.shape[0]
        or anatomy_logits.shape[-2:] != refinement_logits.shape[-2:]
    ):
        raise ValueError(
            "Anatomy and refinement logits must share batch/spatial shape"
        )

    anatomy = anatomy_logits[:, :, 0]
    refinement = refinement_logits[:, :, 0]
    background = torch.zeros_like(anatomy[:, :1])
    coarse = torch.cat([background, anatomy], dim=1).argmax(dim=1)
    fine = refinement.argmax(dim=1) + 3  # normal=3, scar-edema=4
    final_label = torch.where(coarse == 3, fine, coarse)

    foreground = anatomy.new_full(
        (anatomy.shape[0], 4, *anatomy.shape[-2:]), rejected_logit
    )
    for class_id in range(1, 5):
        foreground[:, class_id - 1] = torch.where(
            final_label == class_id,
            foreground.new_tensor(selected_logit),
            foreground[:, class_id - 1],
        )
    return foreground[:, :, None]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--split", choices=("train", "val", "test"), default="test"
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--case_limit", type=int, default=0)
    parser.add_argument(
        "--decision",
        choices=("auto", "flat", "hierarchical", "independent", "spatial"),
        default="auto",
        help="Final cross-pass decision rule.",
    )
    parser.add_argument(
        "--independent_threshold",
        type=float,
        default=0.5,
        help="Sigmoid threshold for independent non-exclusive masks.",
    )
    args = parser.parse_args()
    if not 0.0 < args.independent_threshold < 1.0:
        raise ValueError("--independent_threshold must be in (0,1)")

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    checkpoint_format = checkpoint.get("format")
    if checkpoint_format not in {"oodka_lge_roi_v1", "oodka_lge_roi_v2"}:
        raise ValueError("Checkpoint is not an OODKA LGE ROI model")
    is_v2 = checkpoint_format == "oodka_lge_roi_v2"
    decision = args.decision
    if decision == "auto":
        decision = "spatial" if is_v2 else "flat"
    if is_v2 and decision != "spatial":
        raise ValueError("V2 checkpoints require --decision spatial (or auto)")
    if not is_v2 and decision == "spatial":
        raise ValueError("Spatial hard switching requires a V2 checkpoint")
    saved = checkpoint["config"]
    cfg = EvalConfig(
        dataset_name="Dataset011_MYO_LGE_BC_OOD",
        fold=int(saved.get("fold", 0)),
        block_z=1,
        batch_size=args.batch_size,
        image_size=int(saved.get("image_size", 256)),
        norm_mode="mri",
        pseudo_rgb_mode="center_repeat",
        device=args.device,
        split=args.split,
        out_dir=args.out_dir,
        use_aligned_biomedparse_preprocessing=True,
    )
    cfg.resolve_paths()
    with open(cfg.dataset_json_path, encoding="utf-8") as handle:
        dataset_json = json.load(handle)
    ending = dataset_json.get("file_ending", ".nii.gz")
    if args.split in {"train", "val"}:
        with open(cfg.splits_final_json, encoding="utf-8") as handle:
            case_ids = json.load(handle)[cfg.fold][args.split]
        images_dir, labels_dir = cfg.imagesTr_dir, cfg.labelsTr_dir
    else:
        case_ids = discover_case_ids_from_dir(cfg.labelsTs_dir, ending)
        images_dir, labels_dir = cfg.imagesTs_dir, cfg.labelsTs_dir
    if args.case_limit:
        case_ids = case_ids[: args.case_limit]

    model = load_frozen_biomedparse(device)
    anatomy_features = build_prompt_features(
        model, MYOPS_LGE_ROI_ANATOMY_PROMPTS, device
    )
    refinement_prompts = (
        MYOPS_LGE_ROI_V2_REFINEMENT_PROMPTS
        if is_v2 else MYOPS_LGE_ROI_REFINEMENT_PROMPTS
    )
    refinement_features = build_prompt_features(
        model, refinement_prompts, device
    )
    modules = build_fusion_modules(
        None,
        model,
        3,
        device,
        text_dim=int(anatomy_features["class_emb"].shape[-1]),
        route_prior_p_mean=float(saved.get("route_prior_p_mean", 0.7)),
        route_prior_concentration=float(
            saved.get("route_prior_concentration", 10.0)
        ),
        route_spatial_basis_grid_size=int(
            saved.get("route_spatial_basis_grid_size", 8)
        ),
        route_spatial_basis_sigma=float(
            saved.get("route_spatial_basis_sigma", 0.0)
        ),
    )
    for name, module in modules.items():
        module.load_state_dict(checkpoint[name])
        module.eval()
    preprocessor = AlignedBiomedParsePreprocessor(
        plans_path=cfg.plans_path,
        dataset_json_path=cfg.dataset_json_path,
        configuration_name="2d",
        low_percentile=float(saved.get("low_percentile", 1.0)),
        high_percentile=float(saved.get("high_percentile", 99.0)),
    )
    roi_generator = ROIGenerator(
        threshold=float(saved.get("roi_threshold", 0.3)),
        expand=float(saved.get("roi_expand", 1.25)),
        fallback=str(saved.get("roi_fallback", "full")),
    )
    maybe_mkdir_p(args.out_dir)
    pred_dir = os.path.join(args.out_dir, "pred_nii")
    maybe_mkdir_p(pred_dir)
    independent_dirs = {}
    if decision == "independent":
        for class_id, class_name in MYOPS_LGE_ROI_FINAL_NAMES.items():
            class_dir = os.path.join(args.out_dir, "pred_binary", class_name)
            maybe_mkdir_p(class_dir)
            independent_dirs[class_id] = class_dir
    rows = []
    all_rois = []
    coverage_values = []
    confusion = np.zeros((5, 5), dtype=np.int64)
    total_myo_intersection = 0
    total_myo_predicted = 0
    total_myo_target = 0
    total_myo_empty_slices = 0
    probability_inside = []
    probability_outside = []
    independent_predicted = np.zeros(5, dtype=np.int64)
    independent_target = np.zeros(5, dtype=np.int64)

    for case_id in tqdm(case_ids, desc=f"LGE ROI {args.split}"):
        image_files = find_raw_image_files(images_dir, case_id, ending)
        label_path = os.path.join(labels_dir, case_id + ending)
        raw_ref = sitk.ReadImage(image_files[0])
        raw_shape = tuple(reversed(raw_ref.GetSize()))
        bp_u8, aligned_seg, properties = preprocessor.run_case(
            image_files, label_path, modality=0
        )
        spatial_shape = bp_u8.shape
        final_prompt_logits = np.zeros((4, *spatial_shape), dtype=np.float32)
        for start in range(0, spatial_shape[0], args.batch_size):
            z_indices = list(
                range(start, min(spatial_shape[0], start + args.batch_size))
            )
            full_blocks = torch.stack(
                [
                    make_biomedparse_block(
                        bp_u8,
                        [z],
                        cfg.image_size,
                        pseudo_rgb_mode="center_repeat",
                    )
                    for z in z_indices
                ]
            )
            valid = torch.ones((len(z_indices), 1), dtype=torch.bool)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                anatomy = predict_block_logits_per_class(
                    full_blocks,
                    valid,
                    (cfg.image_size, cfg.image_size),
                    anatomy_features,
                    3,
                    model,
                    modules,
                    device,
                )
                rois = [
                    roi_generator.from_probability(torch.sigmoid(value))
                    for value in anatomy[:, 2, 0].detach()
                ]
                anatomy_probabilities = torch.sigmoid(anatomy[:, 2, 0]).cpu()
                roi_blocks = []
                for index, roi in enumerate(rois):
                    crop = full_blocks[
                        index, 0, :, roi.y0 : roi.y1, roi.x0 : roi.x1
                    ]
                    roi_blocks.append(
                        F.interpolate(
                            crop[None],
                            size=(cfg.image_size, cfg.image_size),
                            mode="bilinear",
                            align_corners=False,
                        )[0][None]
                    )
                roi_blocks = torch.stack(roi_blocks)
                refinement = predict_block_logits_per_class(
                    roi_blocks,
                    valid,
                    (cfg.image_size, cfg.image_size),
                    refinement_features,
                    4 if is_v2 else 2,
                    model,
                    modules,
                    device,
                )
                restored = restore_roi_logits(
                    refinement,
                    rois,
                    (cfg.image_size, cfg.image_size),
                )
                if decision == "spatial":
                    foreground_scores = hard_switch_foreground_logits(
                        anatomy, restored, rois
                    )
                elif decision == "hierarchical":
                    foreground_scores = _hierarchical_foreground_logits(
                        anatomy, restored
                    )
                else:
                    foreground_scores = torch.cat(
                        [anatomy[:, 0:2], restored], dim=1
                    )
                resized = F.interpolate(
                    foreground_scores[:, :, 0],
                    size=spatial_shape[1:],
                    mode="bilinear",
                    align_corners=False,
                ).float().cpu().numpy()
            for index, z in enumerate(z_indices):
                final_prompt_logits[:, z] = resized[index]
                roi = rois[index]
                all_rois.append(roi)
                total_myo = np.isin(aligned_seg[z], (1, 2, 4))
                total_resized = F.interpolate(
                    torch.from_numpy(total_myo)[None, None].float(),
                    size=(cfg.image_size, cfg.image_size),
                    mode="nearest",
                )[0, 0].bool()
                threshold_mask = anatomy_probabilities[index] >= float(
                    saved.get("roi_threshold", 0.3)
                )
                total_myo_intersection += int((threshold_mask & total_resized).sum())
                total_myo_predicted += int(threshold_mask.sum())
                total_myo_target += int(total_resized.sum())
                total_myo_empty_slices += int(not threshold_mask.any())
                if total_resized.any():
                    probability_inside.append(
                        float(anatomy_probabilities[index][total_resized].mean())
                    )
                if (~total_resized).any():
                    probability_outside.append(
                        float(anatomy_probabilities[index][~total_resized].mean())
                    )
                total = int(total_resized.sum())
                inside = int(
                    total_resized[roi.y0 : roi.y1, roi.x0 : roi.x1].sum()
                )
                coverage_values.append(inside / total if total else 1.0)

        gt_raw, spacing = read_nifti_as_zyx_with_spacing(label_path)
        target = _remap_gt(gt_raw)
        if decision == "independent":
            threshold_logit = float(
                np.log(
                    args.independent_threshold
                    / (1.0 - args.independent_threshold)
                )
            )
            row = {"case_id": case_id}
            present_scores = []
            for class_id in range(1, 5):
                binary = preprocessor.prompt_logits_to_raw_segmentation(
                    final_prompt_logits[class_id - 1 : class_id]
                    - threshold_logit,
                    {0: 1},
                    properties,
                ) > 0
                if tuple(binary.shape) != raw_shape:
                    raise ValueError(
                        f"{case_id}: restored={binary.shape}, raw={raw_shape}"
                    )
                binary_target = target == class_id
                denominator = int(binary.sum() + binary_target.sum())
                raw_dice = (
                    2.0 * int((binary & binary_target).sum()) / denominator
                    if denominator > 0
                    else None
                )
                dice_value = raw_dice if binary_target.any() else None
                row[f"dice_{class_id}"] = dice_value
                if dice_value is not None:
                    present_scores.append(dice_value)
                precision, recall, hd95 = precision_recall_hd95_no_ignore(
                    binary.astype(np.int16),
                    binary_target.astype(np.int16),
                    (1,),
                    spacing,
                )
                row[f"prec_{class_id}"] = precision.get(1)
                row[f"rec_{class_id}"] = recall.get(1)
                row[f"hd95_{class_id}"] = hd95.get(1)
                independent_predicted[class_id] += int(binary.sum())
                independent_target[class_id] += int(binary_target.sum())
                output = sitk.GetImageFromArray(binary.astype(np.int16))
                output.CopyInformation(sitk.ReadImage(label_path))
                sitk.WriteImage(
                    output,
                    os.path.join(independent_dirs[class_id], case_id + ending),
                )
            row["dice_mean_gt"] = (
                float(np.mean(present_scores)) if present_scores else None
            )
            # Keep a stable CSV column order shared with the exclusive path.
            ordered = {"case_id": row.pop("case_id"), "dice_mean_gt": row.pop("dice_mean_gt")}
            ordered.update(row)
            rows.append(ordered)
            continue

        prediction = preprocessor.prompt_logits_to_raw_segmentation(
            final_prompt_logits,
            {0: 1, 1: 2, 2: 3, 3: 4},
            properties,
        )
        if tuple(prediction.shape) != raw_shape:
            raise ValueError(
                f"{case_id}: restored={prediction.shape}, raw={raw_shape}"
            )
        encoded = target.astype(np.int64) * 5 + prediction.astype(np.int64)
        confusion += np.bincount(encoded.ravel(), minlength=25).reshape(5, 5)
        dice_pc, dice_mean, _ = dice_no_ignore(
            prediction, target, (1, 2, 3, 4)
        )
        precision, recall, hd95 = precision_recall_hd95_no_ignore(
            prediction, target, (1, 2, 3, 4), spacing
        )
        row = {"case_id": case_id, "dice_mean_gt": dice_mean}
        for class_id in range(1, 5):
            row[f"dice_{class_id}"] = dice_pc.get(class_id)
            row[f"prec_{class_id}"] = precision.get(class_id)
            row[f"rec_{class_id}"] = recall.get(class_id)
            row[f"hd95_{class_id}"] = hd95.get(class_id)
        rows.append(row)
        output = sitk.GetImageFromArray(prediction.astype(np.int16))
        output.CopyInformation(sitk.ReadImage(label_path))
        sitk.WriteImage(output, os.path.join(pred_dir, case_id + ending))

    with open(os.path.join(args.out_dir, "metrics.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "n_cases": len(rows),
        "decision": decision,
        "checkpoint_format": checkpoint_format,
        "mean_dice_gt_present": float(np.mean([r["dice_mean_gt"] for r in rows])),
        "class_names": MYOPS_LGE_ROI_FINAL_NAMES,
    }
    for class_id in range(1, 5):
        values = [r[f"dice_{class_id}"] for r in rows if r[f"dice_{class_id}"] is not None]
        summary[f"dice_{class_id}_mean"] = float(np.mean(values)) if values else None
    roi_summary = roi_diagnostics(
        all_rois, (cfg.image_size, cfg.image_size)
    )
    roi_summary["total_myo_gt_recall_mean"] = float(np.mean(coverage_values))
    roi_summary["threshold_mask_dice"] = (
        2.0 * total_myo_intersection
        / max(1, total_myo_predicted + total_myo_target)
    )
    roi_summary["threshold_mask_precision"] = (
        total_myo_intersection / max(1, total_myo_predicted)
    )
    roi_summary["threshold_mask_recall"] = (
        total_myo_intersection / max(1, total_myo_target)
    )
    roi_summary["threshold_empty_slice_count"] = total_myo_empty_slices
    roi_summary["probability_inside_gt_mean"] = float(
        np.mean(probability_inside) if probability_inside else 0.0
    )
    roi_summary["probability_outside_gt_mean"] = float(
        np.mean(probability_outside) if probability_outside else 0.0
    )
    summary["roi"] = roi_summary
    if decision == "independent":
        summary["independent_threshold"] = args.independent_threshold
        summary["independent_predicted_voxels"] = independent_predicted.tolist()
        summary["independent_target_voxels"] = independent_target.tolist()
    else:
        summary["confusion_gt_rows_pred_columns"] = confusion.tolist()
        summary["gt_voxel_fraction"] = (
            confusion.sum(axis=1) / max(1, confusion.sum())
        ).tolist()
        summary["pred_voxel_fraction"] = (
            confusion.sum(axis=0) / max(1, confusion.sum())
        ).tolist()
    with open(os.path.join(args.out_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
