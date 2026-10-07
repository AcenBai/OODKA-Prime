#!/usr/bin/env python3
"""Two-pass predicted-ROI inference for LGE and whole-heart models."""

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
from oodka.data.roi_geometry import (
    restore_roi_logits,
    roi_diagnostics,
    transform_roi_tensor,
)
from oodka.data.roi_policy import ROIGenerator
from oodka.data.slice_dataset import make_biomedparse_block
from oodka.eval.roi_block_diagnostics import collect_block_diagnostics
from oodka.eval.roi_case_reporting import (
    remap_gt as _remap_gt,
    score_exclusive_case,
    score_independent_case,
)
from oodka.eval.roi_checkpoint import resolve_roi_eval_profile
from oodka.eval.roi_inference import (
    _hierarchical_foreground_logits,
    combine_roi_foreground_logits,
    select_block_rois,
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
        "--roi_source",
        choices=("predicted", "ground_truth", "full"),
        default="predicted",
        help="Diagnostic source of the second-pass crop.",
    )
    parser.add_argument(
        "--roi_transform",
        choices=("checkpoint", "resize", "pad", "letterbox"),
        default="checkpoint",
        help=(
            "Crop-to-canvas geometry. 'checkpoint' uses the training setting "
            "and falls back to historical resize for old checkpoints."
        ),
    )
    parser.add_argument(
        "--decision",
        choices=(
            "auto", "flat", "hierarchical", "independent", "spatial",
            "refinement",
        ),
        default="auto",
        help="Final cross-pass decision rule.",
    )
    parser.add_argument(
        "--postprocess",
        choices=("none", "largest_per_class"),
        default="none",
        help="Optional per-class largest-component filtering.",
    )
    parser.add_argument(
        "--independent_threshold",
        type=float,
        default=0.5,
        help="Sigmoid threshold for independent non-exclusive masks.",
    )
    parser.add_argument(
        "--block_diagnostics_jsonl",
        default="",
        help=(
            "Optional JSONL export with one record per inference block: ROI "
            "coordinates, per-class ROI coverage/composition, and model-grid "
            "confusions. Empty disables the diagnostic without changing inference."
        ),
    )
    args = parser.parse_args()
    if not 0.0 < args.independent_threshold < 1.0:
        raise ValueError("--independent_threshold must be in (0,1)")

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    profile = resolve_roi_eval_profile(
        checkpoint, decision=args.decision, roi_transform=args.roi_transform
    )
    checkpoint_format = profile.checkpoint_format
    is_whs = profile.is_whs
    is_flat = profile.is_flat
    decision = profile.decision
    saved = checkpoint["config"]
    roi_transform = profile.roi_transform
    dataset_name = profile.dataset_name
    cfg = EvalConfig(
        dataset_name=dataset_name,
        fold=int(saved.get("fold", 0)),
        block_z=int(saved.get("block_z", 1)),
        batch_size=args.batch_size,
        image_size=int(saved.get("image_size", 256)),
        norm_mode=str(saved.get("norm_mode", "mri")),
        pseudo_rgb_mode=str(saved.get("pseudo_rgb_mode", "center_repeat")),
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
    anatomy_prompts = profile.anatomy_prompts
    refinement_prompts = profile.refinement_prompts
    anatomy_features = build_prompt_features(model, anatomy_prompts, device)
    final_groups = profile.final_groups
    final_names = profile.final_names
    final_count = len(final_groups)
    refinement_features = build_prompt_features(
        model, refinement_prompts, device
    )
    modules = build_fusion_modules(
        None,
        model,
        len(anatomy_prompts),
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
        norm_mode=str(saved.get("norm_mode", "mri")),
        window_level=float(saved.get("window_level", 40.0)),
        window_width=float(saved.get("window_width", 400.0)),
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
        for class_id, class_name in final_names.items():
            class_dir = os.path.join(args.out_dir, "pred_binary", class_name)
            maybe_mkdir_p(class_dir)
            independent_dirs[class_id] = class_dir
    rows = []
    all_rois = []
    coverage_values = []
    confusion = np.zeros((final_count + 1, final_count + 1), dtype=np.int64)
    total_myo_intersection = 0
    total_myo_predicted = 0
    total_myo_target = 0
    total_myo_empty_slices = 0
    probability_inside = []
    probability_outside = []
    independent_predicted = np.zeros(final_count + 1, dtype=np.int64)
    independent_target = np.zeros(final_count + 1, dtype=np.int64)
    block_diagnostics = []

    task_label = "WHS ROI" if is_whs else f"LGE {'flat' if is_flat else 'ROI'}"
    for case_id in tqdm(case_ids, desc=f"{task_label} {args.split}"):
        image_files = find_raw_image_files(images_dir, case_id, ending)
        label_path = os.path.join(labels_dir, case_id + ending)
        raw_ref = sitk.ReadImage(image_files[0])
        raw_shape = tuple(reversed(raw_ref.GetSize()))
        bp_u8, aligned_seg, properties = preprocessor.run_case(
            image_files, label_path, modality=0
        )
        spatial_shape = bp_u8.shape
        final_prompt_logits = np.zeros((final_count, *spatial_shape), dtype=np.float32)
        block_starts = list(range(0, spatial_shape[0], cfg.block_z))
        for block_batch_start in range(0, len(block_starts), args.batch_size):
            current_starts = block_starts[
                block_batch_start : block_batch_start + args.batch_size
            ]
            full_blocks_list = []
            valid_rows = []
            for z_start in current_starts:
                valid_count = min(cfg.block_z, spatial_shape[0] - z_start)
                centers = list(range(z_start, z_start + valid_count))
                centers.extend([centers[-1]] * (cfg.block_z - valid_count))
                full_blocks_list.append(
                    make_biomedparse_block(
                        bp_u8,
                        centers,
                        cfg.image_size,
                        pseudo_rgb_mode=cfg.pseudo_rgb_mode,
                    )
                )
                valid_rows.append(
                    torch.arange(cfg.block_z) < valid_count
                )
            full_blocks = torch.stack(full_blocks_list)
            valid = torch.stack(valid_rows)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                anatomy = predict_block_logits_per_class(
                    full_blocks,
                    valid,
                    (cfg.image_size, cfg.image_size),
                    anatomy_features,
                    len(anatomy_prompts),
                    model,
                    modules,
                    device,
                )
                if is_flat:
                    foreground_scores = anatomy
                    rois = []
                    anatomy_probabilities = None
                else:
                    rois = select_block_rois(
                        anatomy, valid, current_starts,
                        roi_source=args.roi_source,
                        generator=roi_generator,
                        checkpoint=checkpoint,
                        final_groups=final_groups,
                        aligned_seg=aligned_seg,
                        image_size=cfg.image_size,
                        device=device,
                    )
                    anatomy_probabilities = torch.sigmoid(
                        anatomy[
                            :, int(checkpoint.get("roi_prompt_index", 2))
                        ]
                    ).cpu()
                    roi_blocks = transform_roi_tensor(
                        full_blocks,
                        rois,
                        transform=roi_transform,
                        mode="bilinear",
                    )
                    refinement = predict_block_logits_per_class(
                        roi_blocks,
                        valid,
                        (cfg.image_size, cfg.image_size),
                        refinement_features,
                        len(refinement_prompts),
                        model,
                        modules,
                        device,
                    )
                    restored = restore_roi_logits(
                        refinement,
                        rois,
                        (cfg.image_size, cfg.image_size),
                        transform=roi_transform,
                    )
                    foreground_scores = combine_roi_foreground_logits(
                        anatomy,
                        restored,
                        rois,
                        decision=decision,
                        outside_prompt_mapping=checkpoint.get(
                            "outside_prompt_mapping", ((0, 0), (1, 1))
                        ),
                    )
                if args.block_diagnostics_jsonl and not is_flat:
                    block_diagnostics.extend(collect_block_diagnostics(
                        case_id=case_id,
                        current_starts=current_starts,
                        valid=valid,
                        rois=rois,
                        aligned_seg=aligned_seg,
                        foreground_scores=foreground_scores,
                        restored=restored,
                        final_groups=final_groups,
                        image_size=cfg.image_size,
                        roi_transform=roi_transform,
                    ))
                block_count, _, block_z = foreground_scores.shape[:3]
                resized = F.interpolate(
                    foreground_scores.permute(0, 2, 1, 3, 4).reshape(
                        block_count * block_z, final_count,
                        cfg.image_size, cfg.image_size,
                    ),
                    size=spatial_shape[1:],
                    mode="bilinear",
                    align_corners=False,
                ).reshape(
                    block_count, block_z, final_count,
                    *spatial_shape[1:],
                ).float().cpu().numpy()
            for block_index, (z_start, block_valid) in enumerate(
                zip(current_starts, valid)
            ):
                valid_count = int(block_valid.sum())
                final_prompt_logits[
                    :, z_start : z_start + valid_count
                ] = resized[block_index, :valid_count].transpose(1, 0, 2, 3)
                if is_flat:
                    continue
                roi = rois[block_index]
                all_rois.append(roi)
                foreground_source_ids = tuple(
                    int(source_id) for source_id in checkpoint.get(
                        "roi_source_labels",
                        tuple(
                            source_id
                            for group in final_groups
                            for source_id in group
                        ),
                    )
                )
                total_myo = np.isin(
                    aligned_seg[z_start : z_start + valid_count],
                    foreground_source_ids,
                )
                total_resized = F.interpolate(
                    torch.from_numpy(total_myo)[:, None].float(),
                    size=(cfg.image_size, cfg.image_size),
                    mode="nearest",
                )[:, 0].bool()
                threshold_mask = anatomy_probabilities[
                    block_index, :valid_count
                ] >= float(
                    saved.get("roi_threshold", 0.3)
                )
                total_myo_intersection += int((threshold_mask & total_resized).sum())
                total_myo_predicted += int(threshold_mask.sum())
                total_myo_target += int(total_resized.sum())
                total_myo_empty_slices += int(
                    (~threshold_mask.flatten(1).any(dim=1)).sum()
                )
                if total_resized.any():
                    probability_inside.append(
                        float(
                            anatomy_probabilities[
                                block_index, :valid_count
                            ][total_resized].mean()
                        )
                    )
                if (~total_resized).any():
                    probability_outside.append(
                        float(
                            anatomy_probabilities[
                                block_index, :valid_count
                            ][~total_resized].mean()
                        )
                    )
                total = int(total_resized.sum())
                inside = int(
                    total_resized[
                        :, roi.y0 : roi.y1, roi.x0 : roi.x1
                    ].sum()
                )
                coverage_values.append(inside / total if total else 1.0)

        gt_raw, spacing = read_nifti_as_zyx_with_spacing(label_path)
        target = _remap_gt(gt_raw, final_groups)
        if decision == "independent":
            rows.append(score_independent_case(
                case_id=case_id,
                final_prompt_logits=final_prompt_logits,
                target=target,
                spacing=spacing,
                preprocessor=preprocessor,
                properties=properties,
                raw_shape=raw_shape,
                threshold=args.independent_threshold,
                postprocess=args.postprocess,
                independent_dirs=independent_dirs,
                independent_predicted=independent_predicted,
                independent_target=independent_target,
                label_path=label_path,
                ending=ending,
            ))
            continue

        rows.append(score_exclusive_case(
            case_id=case_id,
            final_prompt_logits=final_prompt_logits,
            target=target,
            spacing=spacing,
            preprocessor=preprocessor,
            properties=properties,
            raw_shape=raw_shape,
            postprocess=args.postprocess,
            confusion=confusion,
            pred_dir=pred_dir,
            label_path=label_path,
            ending=ending,
        ))
    with open(os.path.join(args.out_dir, "metrics.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "n_cases": len(rows),
        "decision": decision,
        "roi_source": args.roi_source,
        "roi_transform": roi_transform,
        "postprocess": args.postprocess,
        "checkpoint_format": checkpoint_format,
        "mean_dice_gt_present": float(np.mean([r["dice_mean_gt"] for r in rows])),
        "class_names": final_names,
    }
    for class_id in range(1, final_count + 1):
        values = [r[f"dice_{class_id}"] for r in rows if r[f"dice_{class_id}"] is not None]
        summary[f"dice_{class_id}_mean"] = float(np.mean(values)) if values else None
    if not is_flat:
        roi_summary = roi_diagnostics(
            all_rois,
            (cfg.image_size, cfg.image_size),
            transform=roi_transform,
        )
        roi_summary["total_myo_gt_recall_mean"] = float(
            np.mean(coverage_values)
        )
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
    if args.block_diagnostics_jsonl:
        diagnostic_path = os.path.abspath(args.block_diagnostics_jsonl)
        maybe_mkdir_p(os.path.dirname(diagnostic_path))
        with open(diagnostic_path, "w", encoding="utf-8") as handle:
            for record in block_diagnostics:
                handle.write(json.dumps(record) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
