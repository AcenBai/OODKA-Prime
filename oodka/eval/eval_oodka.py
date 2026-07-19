"""OODKA evaluation with non-overlapping contiguous 2.5D slice blocks."""

from __future__ import annotations

import csv
import json
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ..config import EvalConfig, ensure_nnunet_on_path
from ..data.slice_dataset import (
    make_biomedparse_block,
    normalize_biomedparse_volume,
)
from ..train.forward import predict_block_logits_per_class
from ..utils.io_utils import (
    discover_case_ids_from_dir,
    find_raw_image_files,
    maybe_mkdir_p,
    read_nifti_as_zyx_with_spacing,
)
from ..utils.metrics import dice_no_ignore, precision_recall_hd95_no_ignore
from ..utils.postprocessing import keep_largest_component_per_class


def _full_bbox(shape: Sequence[int]) -> List[List[int]]:
    return [[0, int(size)] for size in shape]


def _normalize_bbox(bbox) -> List[List[int]] | None:
    if bbox is None:
        return None
    return [[int(value) for value in axis] for axis in bbox]


def _preprocess_nnunet_case(
    image_files: Sequence[str],
    *,
    plans_manager,
    configuration_manager,
    dataset_json: dict,
    raw_shape: Tuple[int, int, int],
    require_no_crop: bool,
) -> np.ndarray:
    """Run only nnUNet's official preprocessing in memory for one case."""
    from nnunetv2.preprocessing.preprocessors.default_preprocessor import (
        DefaultPreprocessor,
    )

    preprocessor = DefaultPreprocessor(verbose=False)
    data, _seg, properties = preprocessor.run_case(
        list(image_files),
        None,
        plans_manager,
        configuration_manager,
        dataset_json,
    )
    shape_before_crop = tuple(
        int(value) for value in properties.get("shape_before_cropping", raw_shape)
    )
    bbox = _normalize_bbox(properties.get("bbox_used_for_cropping"))
    if shape_before_crop != raw_shape:
        raise ValueError(
            f"nnUNet shape_before_cropping={shape_before_crop} != raw shape={raw_shape}"
        )
    if require_no_crop and bbox != _full_bbox(shape_before_crop):
        raise ValueError(
            f"nnUNet crop {bbox} does not cover full shape {shape_before_crop}; "
            "Z-slice correspondence is unsafe"
        )

    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 4:
        raise ValueError(f"Expected nnUNet data [C,Z,H,W], got {data.shape}")
    if data.shape[1] != raw_shape[0]:
        raise ValueError(
            f"nnUNet preprocessed Z={data.shape[1]} != raw Z={raw_shape[0]}"
        )
    return data


def _resize_nnunet_slices(data_zchw: np.ndarray, image_size: int) -> torch.Tensor:
    return F.interpolate(
        torch.from_numpy(np.ascontiguousarray(data_zchw)).float(),
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )


def _make_block_batch(
    nn_data: np.ndarray,
    bp_u8: np.ndarray,
    starts: Sequence[int],
    *,
    block_z: int,
    image_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
    """Materialize B independent blocks while keeping each block's Z contiguous."""
    nn_blocks = []
    bp_blocks = []
    valid_masks = []
    valid_counts = []
    total_z = int(bp_u8.shape[0])

    for z_start in starts:
        valid_count = min(block_z, total_z - int(z_start))
        valid_counts.append(valid_count)
        centers = list(range(int(z_start), int(z_start) + valid_count))
        centers.extend([centers[-1]] * (block_z - valid_count))

        nn_valid = nn_data[:, z_start : z_start + valid_count].transpose(1, 0, 2, 3)
        if valid_count < block_z:
            nn_valid = np.concatenate(
                [nn_valid, np.repeat(nn_valid[-1:], block_z - valid_count, axis=0)],
                axis=0,
            )
        nn_blocks.append(_resize_nnunet_slices(nn_valid, image_size))
        bp_blocks.append(make_biomedparse_block(bp_u8, centers, image_size))
        valid_masks.append(torch.arange(block_z) < valid_count)

    return (
        torch.stack(nn_blocks),
        torch.stack(bp_blocks),
        torch.stack(valid_masks),
        valid_counts,
    )


def _block_logits_to_raw_labels(
    logits: torch.Tensor,
    raw_hw: Tuple[int, int],
    prompt_to_class_id: Dict[int, int],
) -> np.ndarray:
    """Resize per-slice logits to raw H/W, then map prompt indices to labels."""
    # [P,Z,H,W] -> [Z,P,H,W], treating Z as the interpolation batch axis.
    logits_zp = logits.permute(1, 0, 2, 3)
    logits_zp = F.interpolate(
        logits_zp,
        size=raw_hw,
        mode="bilinear",
        align_corners=False,
    )
    background = torch.zeros(
        (logits_zp.shape[0], 1, raw_hw[0], raw_hw[1]),
        device=logits_zp.device,
        dtype=logits_zp.dtype,
    )
    compact = torch.cat([background, logits_zp], dim=1).argmax(dim=1).cpu().numpy()
    labels = np.zeros(compact.shape, dtype=np.int16)
    for prompt_index, class_id in prompt_to_class_id.items():
        labels[compact == int(prompt_index) + 1] = int(class_id)
    return labels


def evaluate_oodka_blocks(
    cfg: EvalConfig,
    model_nnunet,
    model_biomedparse,
    fusion_modules: Dict[str, torch.nn.Module],
    prompt_features: dict,
    prompt_to_class_id: Dict[int, int],
    P: int,
):
    """Evaluate every real slice exactly once using contiguous Z blocks."""
    ensure_nnunet_on_path()
    from batchgenerators.utilities.file_and_folder_operations import load_json
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    device = torch.device(cfg.device)
    dataset_json = load_json(cfg.dataset_json_path)
    plans_manager = PlansManager(load_json(cfg.plans_path))
    configuration_manager = plans_manager.get_configuration(cfg.nnunet_configuration)
    file_ending = dataset_json.get("file_ending", ".nii.gz")
    labels_dict = dataset_json.get("labels", {})
    class_ids = sorted(int(value) for value in labels_dict.values() if int(value) > 0)

    test_ids = discover_case_ids_from_dir(cfg.labelsTs_dir, file_ending)
    if not test_ids:
        test_ids = discover_case_ids_from_dir(
            cfg.imagesTs_dir, file_ending, strip_modality=True
        )
    if not test_ids:
        raise FileNotFoundError(
            f"No test cases found in {cfg.imagesTs_dir} / {cfg.labelsTs_dir}"
        )
    if cfg.case_limit > 0:
        test_ids = test_ids[: cfg.case_limit]

    maybe_mkdir_p(cfg.out_dir)
    pred_dir = os.path.join(cfg.out_dir, "pred_nii")
    maybe_mkdir_p(pred_dir)
    for module in fusion_modules.values():
        module.eval()

    all_rows = []
    for case_id in tqdm(test_ids, desc="OODKA block eval"):
        image_files = find_raw_image_files(cfg.imagesTs_dir, case_id, file_ending)
        if not image_files:
            image_files = find_raw_image_files(cfg.imagesTr_dir, case_id, file_ending)
        if not image_files:
            raise FileNotFoundError(f"No raw image files found for {case_id}")
        if not 0 <= cfg.biomedparse_modality < len(image_files):
            raise IndexError(
                f"{case_id}: biomedparse_modality={cfg.biomedparse_modality}, "
                f"but only {len(image_files)} modalities were found"
            )

        label_path = os.path.join(cfg.labelsTs_dir, case_id + file_ending)
        if not os.path.isfile(label_path):
            label_path = os.path.join(cfg.labelsTr_dir, case_id + file_ending)
        if not os.path.isfile(label_path):
            raise FileNotFoundError(f"Ground-truth label not found for {case_id}")

        raw_ref = sitk.ReadImage(image_files[cfg.biomedparse_modality])
        raw_image = np.asarray(sitk.GetArrayFromImage(raw_ref))
        raw_shape = tuple(int(value) for value in raw_image.shape)
        gt_arr, spacing_xyz = read_nifti_as_zyx_with_spacing(label_path)
        if tuple(gt_arr.shape) != raw_shape:
            raise ValueError(
                f"{case_id}: GT shape={gt_arr.shape} != raw shape={raw_shape}"
            )

        nn_data = _preprocess_nnunet_case(
            image_files,
            plans_manager=plans_manager,
            configuration_manager=configuration_manager,
            dataset_json=dataset_json,
            raw_shape=raw_shape,
            require_no_crop=cfg.require_no_crop,
        )
        bp_u8 = normalize_biomedparse_volume(
            raw_image,
            norm_mode=cfg.norm_mode,
            window_level=cfg.window_level,
            window_width=cfg.window_width,
            low_percentile=cfg.low_percentile,
            high_percentile=cfg.high_percentile,
        )

        pred_seg = np.zeros(raw_shape, dtype=np.int16)
        all_starts = list(range(0, raw_shape[0], cfg.block_z))
        for batch_start in range(0, len(all_starts), cfg.batch_size):
            starts = all_starts[batch_start : batch_start + cfg.batch_size]
            nn_blocks, bp_blocks, valid_z, valid_counts = _make_block_batch(
                nn_data,
                bp_u8,
                starts,
                block_z=cfg.block_z,
                image_size=cfg.image_size,
            )
            block_logits = predict_block_logits_per_class(
                nnunet_images=nn_blocks,
                biomedparse_images=bp_blocks,
                valid_z=valid_z,
                output_size=(cfg.image_size, cfg.image_size),
                prompt_features=prompt_features,
                P=P,
                model_nnunet=model_nnunet,
                model_biomedparse=model_biomedparse,
                fusion_modules=fusion_modules,
                device=device,
            )
            for block_index, (z_start, valid_count) in enumerate(
                zip(starts, valid_counts)
            ):
                raw_labels = _block_logits_to_raw_labels(
                    block_logits[block_index, :, :valid_count],
                    raw_shape[1:],
                    prompt_to_class_id,
                )
                pred_seg[z_start : z_start + valid_count] = raw_labels

        pred_seg = keep_largest_component_per_class(pred_seg, class_ids)
        label_ref = sitk.ReadImage(label_path)
        out_img = sitk.GetImageFromArray(pred_seg)
        out_img.CopyInformation(label_ref)
        sitk.WriteImage(out_img, os.path.join(pred_dir, case_id + file_ending))

        dice_pc, dice_gt, _dice_any = dice_no_ignore(
            pred_seg, gt_arr.astype(np.int16), class_ids
        )
        prec_pc, rec_pc, hd95_pc = precision_recall_hd95_no_ignore(
            pred_seg,
            gt_arr.astype(np.int16),
            class_ids,
            spacing_xyz,
        )
        row = {"case_id": case_id, "dice_mean_gt": dice_gt}
        for class_id in class_ids:
            row[f"dice_{class_id}"] = dice_pc.get(class_id)
            row[f"prec_{class_id}"] = prec_pc.get(class_id)
            row[f"rec_{class_id}"] = rec_pc.get(class_id)
            row[f"hd95_{class_id}"] = hd95_pc.get(class_id)
        all_rows.append(row)

    csv_path = os.path.join(cfg.out_dir, "metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)

    summary = {
        "n_cases": len(test_ids),
        "mean_dice_gt_present": float(
            np.mean([row["dice_mean_gt"] for row in all_rows])
        ),
    }
    for class_id in class_ids:
        values = [
            row[f"dice_{class_id}"]
            for row in all_rows
            if row.get(f"dice_{class_id}") is not None
        ]
        summary[f"dice_{class_id}_mean"] = float(np.mean(values)) if values else None
    with open(
        os.path.join(cfg.out_dir, "summary.json"), "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2)

    print(f"\nOODKA block evaluation complete. Results: {cfg.out_dir}")
    print(f"Mean Dice: {summary['mean_dice_gt_present']:.4f}")
    return summary
