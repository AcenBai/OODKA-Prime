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
from oodka.data.lge_roi import (
    ROICoordinates,
    ROIGenerator,
    hard_switch_foreground_logits,
    restore_roi_logits,
    roi_diagnostics,
)
from oodka.data.slice_dataset import make_biomedparse_block
from oodka.models.prompts import (
    MYOPS_LGE_ROI_ANATOMY_PROMPTS,
    MYOPS_LGE_ROI_FINAL_GROUPS,
    MYOPS_LGE_ROI_FINAL_NAMES,
    MYOPS_LGE_ROI_REFINEMENT_PROMPTS,
    MYOPS_LGE_ROI_SPLIT_FINAL_GROUPS,
    MYOPS_LGE_ROI_SPLIT_FINAL_NAMES,
    MYOPS_LGE_ROI_V2_REFINEMENT_PROMPTS,
    MYOPS_LGE_ROI_V3_REFINEMENT_PROMPTS,
    WHS_CT_ROI_LOCALIZATION_PROMPTS,
    WHS_CT_ROI_REFINEMENT_PROMPTS,
    WHS_CT_GV_LOCALIZATION_PROMPTS,
    WHS_CT_GV_REFINEMENT_PROMPTS,
    WHS_MRI_ROI_LOCALIZATION_PROMPTS,
    WHS_MRI_ROI_REFINEMENT_PROMPTS,
    WHS_MRI_GV_LOCALIZATION_PROMPTS,
    WHS_MRI_GV_REFINEMENT_PROMPTS,
    WHS_ROI_REFINEMENT_GROUPS,
)
from oodka.eval.selective_refinement import (
    keep_largest_component_per_class,
    selective_prompt_overwrite,
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


def _remap_gt(array: np.ndarray, groups=MYOPS_LGE_ROI_FINAL_GROUPS) -> np.ndarray:
    output = np.zeros(array.shape, dtype=np.int16)
    for class_id, source_ids in enumerate(groups, start=1):
        output[np.isin(array, source_ids)] = class_id
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
        raise ValueError("Anatomy and refinement logits must share batch/spatial shape")

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


def _parse_float_grid(value: str, *, name: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise ValueError(f"{name} must be a comma-separated float list") from error
    if not values:
        raise ValueError(f"{name} must not be empty")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--case_limit", type=int, default=0)
    parser.add_argument(
        "--case_ids",
        default="",
        help="Optional comma-separated case ids, preserving the requested order.",
    )
    parser.add_argument(
        "--roi_source",
        choices=("predicted", "ground_truth", "full"),
        default="predicted",
        help="Diagnostic source of the second-pass crop.",
    )
    parser.add_argument(
        "--decision",
        choices=(
            "auto",
            "flat",
            "hierarchical",
            "independent",
            "spatial",
            "refinement",
        ),
        default="auto",
        help="Final cross-pass decision rule.",
    )
    parser.add_argument(
        "--independent_threshold",
        type=float,
        default=0.5,
        help="Sigmoid threshold for independent non-exclusive masks.",
    )
    parser.add_argument(
        "--selective_global_pred_dir",
        default="",
        help=(
            "Optional raw-space global seven-class NIfTI directory. For WHS-GV "
            "models, sweep conservative AO/PA overwrites on top of these labels."
        ),
    )
    parser.add_argument(
        "--selective_thresholds",
        default="0.5",
        help="Comma-separated local sigmoid confidence thresholds.",
    )
    parser.add_argument(
        "--selective_ambiguity_margins",
        default="0.0",
        help="Comma-separated minimum top1-minus-top2 probability margins.",
    )
    parser.add_argument(
        "--selective_postprocess",
        choices=("none", "keep_largest_per_class", "both"),
        default="both",
    )
    parser.add_argument(
        "--selective_overwrite_scope",
        choices=("any", "background_children", "both"),
        default="both",
        help=(
            "Which global labels local AO/PA may replace. background_children "
            "strictly protects LV/RV/LA/RA/Myo."
        ),
    )
    parser.add_argument(
        "--selective_save_predictions",
        action="store_true",
        help="Save fused NIfTIs; requires a single threshold and ambiguity margin.",
    )
    args = parser.parse_args()
    if not 0.0 < args.independent_threshold < 1.0:
        raise ValueError("--independent_threshold must be in (0,1)")

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    checkpoint_format = checkpoint.get("format")
    if checkpoint_format not in {
        "oodka_lge_roi_v1",
        "oodka_lge_roi_v2",
        "oodka_lge_roi_v3_split5",
        "oodka_lge_flat_v1",
        "oodka_whs_roi_v1",
        "oodka_whs_gv_roi_v1",
        "oodka_whs_gv_selective_v1",
    }:
        raise ValueError("Checkpoint is not a supported OODKA ROI model")
    is_whs_gv = checkpoint_format == "oodka_whs_gv_roi_v1"
    is_whs_selective = checkpoint_format == "oodka_whs_gv_selective_v1"
    uses_gv_roi = is_whs_gv or is_whs_selective
    is_whs = checkpoint_format in {"oodka_whs_roi_v1", "oodka_whs_gv_roi_v1"}
    is_whs = is_whs or is_whs_selective
    is_split = checkpoint_format == "oodka_lge_roi_v3_split5"
    is_v2 = checkpoint_format in {"oodka_lge_roi_v2", "oodka_lge_roi_v3_split5"}
    is_flat = checkpoint_format == "oodka_lge_flat_v1"
    selective_enabled = bool(args.selective_global_pred_dir)
    selective_thresholds = _parse_float_grid(
        args.selective_thresholds, name="--selective_thresholds"
    )
    selective_ambiguity_margins = _parse_float_grid(
        args.selective_ambiguity_margins,
        name="--selective_ambiguity_margins",
    )
    if selective_enabled and not uses_gv_roi:
        raise ValueError("Selective AO/PA fusion requires a WHS-GV checkpoint")
    for threshold in selective_thresholds:
        if not 0.0 < threshold < 1.0:
            raise ValueError("Selective thresholds must be in (0,1)")
    for margin in selective_ambiguity_margins:
        if not 0.0 <= margin < 1.0:
            raise ValueError("Selective ambiguity margins must be in [0,1)")
    if args.selective_save_predictions and (
        len(selective_thresholds) != 1 or len(selective_ambiguity_margins) != 1
    ):
        raise ValueError(
            "--selective_save_predictions requires one threshold and one margin"
        )
    decision = args.decision
    if decision == "auto":
        decision = (
            "refinement"
            if is_whs_selective
            else (
                "spatial"
                if is_whs_gv
                else "refinement" if is_whs else "spatial" if is_v2 else "flat"
            )
        )
    if is_whs_gv and decision != "spatial":
        raise ValueError("WHS-GV checkpoints require --decision spatial (or auto)")
    if is_whs_selective and decision != "refinement":
        raise ValueError(
            "Selective WHS-GV checkpoints require --decision refinement (or auto)"
        )
    if is_whs and not is_whs_gv and decision != "refinement":
        raise ValueError("WHS whole-heart checkpoints require refinement (or auto)")
    if is_v2 and decision != "spatial":
        raise ValueError("V2 checkpoints require --decision spatial (or auto)")
    if not is_v2 and not is_whs and decision == "spatial":
        raise ValueError("Spatial hard switching requires a V2 checkpoint")
    if is_flat and decision != "flat":
        raise ValueError("Flat checkpoints require --decision flat (or auto)")
    saved = checkpoint["config"]
    dataset_name = (
        str(saved.get("dataset_name")) if is_whs else "Dataset011_MYO_LGE_BC_OOD"
    )
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
    if args.case_ids:
        requested = [
            value.strip() for value in args.case_ids.split(",") if value.strip()
        ]
        available = set(case_ids)
        missing = [value for value in requested if value not in available]
        if missing:
            raise ValueError(f"Requested cases are absent from {args.split}: {missing}")
        case_ids = requested
    if args.case_limit:
        case_ids = case_ids[: args.case_limit]

    model = load_frozen_biomedparse(device)
    if is_whs_selective:
        anatomy_prompts = checkpoint["prompt_texts"]["localization"]
        refinement_prompts = checkpoint["prompt_texts"]["refinement"]
    elif is_whs:
        if dataset_name == "Dataset009_CT_OOD":
            anatomy_prompts = (
                WHS_CT_GV_LOCALIZATION_PROMPTS
                if is_whs_gv
                else WHS_CT_ROI_LOCALIZATION_PROMPTS
            )
            refinement_prompts = (
                WHS_CT_GV_REFINEMENT_PROMPTS
                if is_whs_gv
                else WHS_CT_ROI_REFINEMENT_PROMPTS
            )
        elif dataset_name == "Dataset010_WHS_MRI_OOD":
            anatomy_prompts = (
                WHS_MRI_GV_LOCALIZATION_PROMPTS
                if is_whs_gv
                else WHS_MRI_ROI_LOCALIZATION_PROMPTS
            )
            refinement_prompts = (
                WHS_MRI_GV_REFINEMENT_PROMPTS
                if is_whs_gv
                else WHS_MRI_ROI_REFINEMENT_PROMPTS
            )
        else:
            raise ValueError(f"Unsupported WHS dataset: {dataset_name}")
    else:
        anatomy_prompts = (
            MYOPS_LGE_ROI_V2_REFINEMENT_PROMPTS
            if is_flat
            else MYOPS_LGE_ROI_ANATOMY_PROMPTS
        )
        refinement_prompts = (
            MYOPS_LGE_ROI_V3_REFINEMENT_PROMPTS
            if is_split
            else (
                MYOPS_LGE_ROI_V2_REFINEMENT_PROMPTS
                if is_v2
                else MYOPS_LGE_ROI_REFINEMENT_PROMPTS
            )
        )
    anatomy_features = build_prompt_features(model, anatomy_prompts, device)
    final_groups = (
        tuple(
            tuple(int(value) for value in group)
            for group in checkpoint["refinement_groups"]
        )
        if is_whs_selective
        else (
            WHS_ROI_REFINEMENT_GROUPS
            if is_whs
            else (
                MYOPS_LGE_ROI_SPLIT_FINAL_GROUPS
                if is_split
                else MYOPS_LGE_ROI_FINAL_GROUPS
            )
        )
    )
    final_names = (
        {1: "AO", 2: "PA"}
        if is_whs_selective
        else (
            {
                1: "LV",
                2: "RV",
                3: "LA",
                4: "RA",
                5: "Myo",
                6: "AO",
                7: "PA",
            }
            if is_whs
            else (
                MYOPS_LGE_ROI_SPLIT_FINAL_NAMES
                if is_split
                else MYOPS_LGE_ROI_FINAL_NAMES
            )
        )
    )
    final_count = len(final_groups)
    selective_class_count = 7
    selective_local_prompt_indices = (0, 1) if is_whs_selective else (5, 6)
    refinement_features = build_prompt_features(model, refinement_prompts, device)
    modules = build_fusion_modules(
        None,
        model,
        len(anatomy_prompts),
        device,
        text_dim=int(anatomy_features["class_emb"].shape[-1]),
        route_prior_p_mean=float(saved.get("route_prior_p_mean", 0.7)),
        route_prior_concentration=float(saved.get("route_prior_concentration", 10.0)),
        route_spatial_basis_grid_size=int(
            saved.get("route_spatial_basis_grid_size", 8)
        ),
        route_spatial_basis_sigma=float(saved.get("route_spatial_basis_sigma", 0.0)),
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
    selective_rows = []
    selective_global_rows = []
    selective_postprocesses = (
        ("none", "keep_largest_per_class")
        if args.selective_postprocess == "both"
        else (args.selective_postprocess,)
    )
    selective_pred_dirs = {}
    selective_scopes = (
        ("any", "background_children")
        if args.selective_overwrite_scope == "both"
        else (args.selective_overwrite_scope,)
    )
    if selective_enabled and args.selective_save_predictions:
        for scope in selective_scopes:
            for postprocess in selective_postprocesses:
                directory = os.path.join(
                    args.out_dir, f"selective_pred_{scope}_{postprocess}"
                )
                maybe_mkdir_p(directory)
                selective_pred_dirs[(scope, postprocess)] = directory

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
                valid_rows.append(torch.arange(cfg.block_z) < valid_count)
            full_blocks = torch.stack(full_blocks_list)
            valid = torch.stack(valid_rows)
            with torch.autocast(
                "cuda", dtype=torch.float16, enabled=device.type == "cuda"
            ):
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
                    if args.roi_source == "predicted":
                        block_probability = (
                            torch.sigmoid(
                                anatomy[:, int(checkpoint.get("roi_prompt_index", 2))]
                            )
                            .masked_fill(~valid.to(device)[:, :, None, None], 0.0)
                            .amax(dim=1)
                        )
                        rois = [
                            roi_generator.from_probability(value)
                            for value in block_probability.detach()
                        ]
                    elif args.roi_source == "ground_truth":
                        foreground_source_ids = tuple(
                            int(source_id)
                            for source_id in checkpoint.get(
                                "roi_source_labels",
                                tuple(
                                    source_id
                                    for group in final_groups
                                    for source_id in group
                                ),
                            )
                        )
                        rois = []
                        for z_start, block_valid in zip(current_starts, valid):
                            valid_count = int(block_valid.sum())
                            target_mask = np.isin(
                                aligned_seg[z_start : z_start + valid_count],
                                foreground_source_ids,
                            ).any(axis=0)
                            target_resized = F.interpolate(
                                torch.from_numpy(target_mask)[None, None].float(),
                                size=(cfg.image_size, cfg.image_size),
                                mode="nearest",
                            )[0, 0]
                            rois.append(roi_generator.from_probability(target_resized))
                    else:
                        rois = [
                            ROICoordinates(
                                0,
                                0,
                                cfg.image_size,
                                cfg.image_size,
                                fallback=False,
                            )
                            for _ in current_starts
                        ]
                    anatomy_probabilities = torch.sigmoid(
                        anatomy[:, int(checkpoint.get("roi_prompt_index", 2))]
                    ).cpu()
                    roi_blocks = []
                    for index, roi in enumerate(rois):
                        crop = full_blocks[
                            index, :, :, roi.y0 : roi.y1, roi.x0 : roi.x1
                        ]
                        roi_blocks.append(
                            F.interpolate(
                                crop,
                                size=(cfg.image_size, cfg.image_size),
                                mode="bilinear",
                                align_corners=False,
                            )
                        )
                    roi_blocks = torch.stack(roi_blocks)
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
                    )
                    if decision == "refinement":
                        foreground_scores = restored
                    elif decision == "spatial":
                        foreground_scores = hard_switch_foreground_logits(
                            anatomy,
                            restored,
                            rois,
                            checkpoint.get("outside_prompt_mapping", ((0, 0), (1, 1))),
                        )
                    elif decision == "hierarchical":
                        foreground_scores = _hierarchical_foreground_logits(
                            anatomy, restored
                        )
                    else:
                        foreground_scores = torch.cat(
                            [anatomy[:, 0:2], restored], dim=1
                        )
                block_count, _, block_z = foreground_scores.shape[:3]
                resized = (
                    F.interpolate(
                        foreground_scores.permute(0, 2, 1, 3, 4).reshape(
                            block_count * block_z,
                            final_count,
                            cfg.image_size,
                            cfg.image_size,
                        ),
                        size=spatial_shape[1:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    .reshape(
                        block_count,
                        block_z,
                        final_count,
                        *spatial_shape[1:],
                    )
                    .float()
                    .cpu()
                    .numpy()
                )
            for block_index, (z_start, block_valid) in enumerate(
                zip(current_starts, valid)
            ):
                valid_count = int(block_valid.sum())
                final_prompt_logits[:, z_start : z_start + valid_count] = resized[
                    block_index, :valid_count
                ].transpose(1, 0, 2, 3)
                if is_flat:
                    continue
                roi = rois[block_index]
                all_rois.append(roi)
                foreground_source_ids = tuple(
                    int(source_id)
                    for source_id in checkpoint.get(
                        "roi_source_labels",
                        tuple(
                            source_id for group in final_groups for source_id in group
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
                ] >= float(saved.get("roi_threshold", 0.3))
                total_myo_intersection += int((threshold_mask & total_resized).sum())
                total_myo_predicted += int(threshold_mask.sum())
                total_myo_target += int(total_resized.sum())
                total_myo_empty_slices += int(
                    (~threshold_mask.flatten(1).any(dim=1)).sum()
                )
                if total_resized.any():
                    probability_inside.append(
                        float(
                            anatomy_probabilities[block_index, :valid_count][
                                total_resized
                            ].mean()
                        )
                    )
                if (~total_resized).any():
                    probability_outside.append(
                        float(
                            anatomy_probabilities[block_index, :valid_count][
                                ~total_resized
                            ].mean()
                        )
                    )
                total = int(total_resized.sum())
                inside = int(total_resized[:, roi.y0 : roi.y1, roi.x0 : roi.x1].sum())
                coverage_values.append(inside / total if total else 1.0)

        gt_raw, spacing = read_nifti_as_zyx_with_spacing(label_path)
        target = _remap_gt(gt_raw, final_groups)
        if selective_enabled:
            local_raw_logits = preprocessor.prompt_values_to_raw_geometry(
                final_prompt_logits[list(selective_local_prompt_indices)],
                properties,
            )
            selective_target = _remap_gt(gt_raw, WHS_ROI_REFINEMENT_GROUPS)
            if tuple(local_raw_logits.shape[1:]) != raw_shape:
                raise ValueError(
                    f"{case_id}: local logits={local_raw_logits.shape[1:]}, "
                    f"raw={raw_shape}"
                )
            global_path = os.path.join(args.selective_global_pred_dir, case_id + ending)
            if not os.path.isfile(global_path):
                raise FileNotFoundError(global_path)
            global_labels = sitk.GetArrayFromImage(sitk.ReadImage(global_path)).astype(
                np.int16, copy=False
            )
            if tuple(global_labels.shape) != raw_shape:
                raise ValueError(
                    f"{case_id}: global={global_labels.shape}, raw={raw_shape}"
                )
            global_tensor = torch.from_numpy(global_labels.astype(np.int64))
            local_tensor = torch.from_numpy(local_raw_logits)
            target_tensor = torch.from_numpy(selective_target.astype(np.int64))
            for postprocess in selective_postprocesses:
                global_evaluated = (
                    keep_largest_component_per_class(
                        global_labels,
                        tuple(range(1, selective_class_count + 1)),
                    )
                    if postprocess == "keep_largest_per_class"
                    else global_labels
                )
                global_dice_pc, global_dice_mean, _ = dice_no_ignore(
                    global_evaluated,
                    selective_target,
                    tuple(range(1, selective_class_count + 1)),
                )
                global_row = {
                    "case_id": case_id,
                    "postprocess": postprocess,
                    "dice_mean_gt": global_dice_mean,
                }
                for class_id in range(1, selective_class_count + 1):
                    global_row[f"dice_{class_id}"] = global_dice_pc.get(class_id)
                selective_global_rows.append(global_row)
            for confidence_threshold in selective_thresholds:
                for ambiguity_margin in selective_ambiguity_margins:
                    for scope in selective_scopes:
                        eligible_ids = (
                            (0, 6, 7) if scope == "background_children" else None
                        )
                        fused_tensor, overwrite_stats = selective_prompt_overwrite(
                            global_tensor,
                            local_tensor,
                            (6, 7),
                            confidence_threshold=confidence_threshold,
                            ambiguity_margin=ambiguity_margin,
                            eligible_global_class_ids=eligible_ids,
                        )
                        fused_unprocessed = fused_tensor.numpy().astype(
                            np.int16, copy=False
                        )
                        changed = fused_tensor != global_tensor
                        before_correct = global_tensor == target_tensor
                        after_correct = fused_tensor == target_tensor
                        beneficial = int(
                            (changed & ~before_correct & after_correct).sum().item()
                        )
                        harmful = int(
                            (changed & before_correct & ~after_correct).sum().item()
                        )
                        for postprocess in selective_postprocesses:
                            fused = (
                                keep_largest_component_per_class(
                                    fused_unprocessed,
                                    tuple(range(1, selective_class_count + 1)),
                                )
                                if postprocess == "keep_largest_per_class"
                                else fused_unprocessed
                            )
                            dice_pc_selective, dice_mean_selective, _ = dice_no_ignore(
                                fused,
                                selective_target,
                                tuple(range(1, selective_class_count + 1)),
                            )
                            selective_row = {
                                "case_id": case_id,
                                "confidence_threshold": confidence_threshold,
                                "ambiguity_margin": ambiguity_margin,
                                "overwrite_scope": scope,
                                "postprocess": postprocess,
                                "dice_mean_gt": dice_mean_selective,
                                "proposed_voxels": overwrite_stats.proposed_voxels,
                                "accepted_voxels": overwrite_stats.accepted_voxels,
                                "changed_voxels": overwrite_stats.changed_voxels,
                                "ambiguous_voxels": overwrite_stats.ambiguous_voxels,
                                "ineligible_voxels": overwrite_stats.ineligible_voxels,
                                "beneficial_changes": beneficial,
                                "harmful_changes": harmful,
                            }
                            for class_id in range(1, selective_class_count + 1):
                                selective_row[f"dice_{class_id}"] = (
                                    dice_pc_selective.get(class_id)
                                )
                            selective_rows.append(selective_row)
                            if args.selective_save_predictions:
                                output = sitk.GetImageFromArray(fused)
                                output.CopyInformation(sitk.ReadImage(label_path))
                                sitk.WriteImage(
                                    output,
                                    os.path.join(
                                        selective_pred_dirs[(scope, postprocess)],
                                        case_id + ending,
                                    ),
                                )
        if decision == "independent":
            threshold_logit = float(
                np.log(args.independent_threshold / (1.0 - args.independent_threshold))
            )
            row = {"case_id": case_id}
            present_scores = []
            for class_id in range(1, final_count + 1):
                binary = (
                    preprocessor.prompt_logits_to_raw_segmentation(
                        final_prompt_logits[class_id - 1 : class_id] - threshold_logit,
                        {0: 1},
                        properties,
                    )
                    > 0
                )
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
            ordered = {
                "case_id": row.pop("case_id"),
                "dice_mean_gt": row.pop("dice_mean_gt"),
            }
            ordered.update(row)
            rows.append(ordered)
            continue

        prediction = preprocessor.prompt_logits_to_raw_segmentation(
            final_prompt_logits,
            {index: index + 1 for index in range(final_count)},
            properties,
        )
        if tuple(prediction.shape) != raw_shape:
            raise ValueError(f"{case_id}: restored={prediction.shape}, raw={raw_shape}")
        width = final_count + 1
        encoded = target.astype(np.int64) * width + prediction.astype(np.int64)
        confusion += np.bincount(encoded.ravel(), minlength=width * width).reshape(
            width, width
        )
        dice_pc, dice_mean, _ = dice_no_ignore(
            prediction, target, tuple(range(1, final_count + 1))
        )
        precision, recall, hd95 = precision_recall_hd95_no_ignore(
            prediction, target, tuple(range(1, final_count + 1)), spacing
        )
        row = {"case_id": case_id, "dice_mean_gt": dice_mean}
        for class_id in range(1, final_count + 1):
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
        "roi_source": args.roi_source,
        "checkpoint_format": checkpoint_format,
        "mean_dice_gt_present": float(np.mean([r["dice_mean_gt"] for r in rows])),
        "class_names": final_names,
    }
    for class_id in range(1, final_count + 1):
        values = [
            r[f"dice_{class_id}"] for r in rows if r[f"dice_{class_id}"] is not None
        ]
        summary[f"dice_{class_id}_mean"] = float(np.mean(values)) if values else None
    if not is_flat:
        roi_summary = roi_diagnostics(all_rois, (cfg.image_size, cfg.image_size))
        roi_summary["total_myo_gt_recall_mean"] = float(np.mean(coverage_values))
        roi_summary["threshold_mask_dice"] = (
            2.0
            * total_myo_intersection
            / max(1, total_myo_predicted + total_myo_target)
        )
        roi_summary["threshold_mask_precision"] = total_myo_intersection / max(
            1, total_myo_predicted
        )
        roi_summary["threshold_mask_recall"] = total_myo_intersection / max(
            1, total_myo_target
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
    if selective_enabled:
        sweep_csv = os.path.join(args.out_dir, "selective_sweep.csv")
        with open(sweep_csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=selective_rows[0].keys())
            writer.writeheader()
            writer.writerows(selective_rows)
        global_csv = os.path.join(args.out_dir, "selective_global.csv")
        with open(global_csv, "w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=selective_global_rows[0].keys(),
            )
            writer.writeheader()
            writer.writerows(selective_global_rows)
        sweep_summary = []
        for confidence_threshold in selective_thresholds:
            for ambiguity_margin in selective_ambiguity_margins:
                for scope in selective_scopes:
                    for postprocess in selective_postprocesses:
                        selected = [
                            row
                            for row in selective_rows
                            if row["confidence_threshold"] == confidence_threshold
                            and row["ambiguity_margin"] == ambiguity_margin
                            and row["overwrite_scope"] == scope
                            and row["postprocess"] == postprocess
                        ]
                        aggregate = {
                            "confidence_threshold": confidence_threshold,
                            "ambiguity_margin": ambiguity_margin,
                            "overwrite_scope": scope,
                            "postprocess": postprocess,
                            "n_cases": len(selected),
                            "mean_dice_gt_present": float(
                                np.mean([row["dice_mean_gt"] for row in selected])
                            ),
                            "proposed_voxels": int(
                                sum(row["proposed_voxels"] for row in selected)
                            ),
                            "accepted_voxels": int(
                                sum(row["accepted_voxels"] for row in selected)
                            ),
                            "changed_voxels": int(
                                sum(row["changed_voxels"] for row in selected)
                            ),
                            "ambiguous_voxels": int(
                                sum(row["ambiguous_voxels"] for row in selected)
                            ),
                            "ineligible_voxels": int(
                                sum(row["ineligible_voxels"] for row in selected)
                            ),
                            "beneficial_changes": int(
                                sum(row["beneficial_changes"] for row in selected)
                            ),
                            "harmful_changes": int(
                                sum(row["harmful_changes"] for row in selected)
                            ),
                        }
                        for class_id in range(1, selective_class_count + 1):
                            values = [
                                row[f"dice_{class_id}"]
                                for row in selected
                                if row[f"dice_{class_id}"] is not None
                            ]
                            aggregate[f"dice_{class_id}_mean"] = (
                                float(np.mean(values)) if values else None
                            )
                        sweep_summary.append(aggregate)
        sweep_summary.sort(
            key=lambda value: value["mean_dice_gt_present"], reverse=True
        )
        with open(
            os.path.join(args.out_dir, "selective_sweep_summary.json"),
            "w",
        ) as handle:
            json.dump(
                {
                    "global_pred_dir": args.selective_global_pred_dir,
                    "child_class_ids": [6, 7],
                    "global_baselines": [
                        {
                            "postprocess": postprocess,
                            "n_cases": len(
                                [
                                    row
                                    for row in selective_global_rows
                                    if row["postprocess"] == postprocess
                                ]
                            ),
                            "mean_dice_gt_present": float(
                                np.mean(
                                    [
                                        row["dice_mean_gt"]
                                        for row in selective_global_rows
                                        if row["postprocess"] == postprocess
                                    ]
                                )
                            ),
                            **{
                                f"dice_{class_id}_mean": float(
                                    np.mean(
                                        [
                                            row[f"dice_{class_id}"]
                                            for row in selective_global_rows
                                            if row["postprocess"] == postprocess
                                            and row[f"dice_{class_id}"] is not None
                                        ]
                                    )
                                )
                                for class_id in range(1, selective_class_count + 1)
                            },
                        }
                        for postprocess in selective_postprocesses
                    ],
                    "results": sweep_summary,
                },
                handle,
                indent=2,
            )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
