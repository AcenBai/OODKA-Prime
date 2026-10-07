"""Optional per-block ROI geometry and confusion diagnostics."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ..data.roi_geometry import ROICoordinates, roi_placement


def collect_block_diagnostics(
    *,
    case_id: str,
    current_starts: Sequence[int],
    valid: torch.Tensor,
    rois: Sequence[ROICoordinates],
    aligned_seg: np.ndarray,
    foreground_scores: torch.Tensor,
    restored: torch.Tensor,
    final_groups: Sequence[Sequence[int]],
    image_size: int,
    roi_transform: str,
) -> list[dict]:
    """Report model-grid coverage/composition without changing inference."""
    final_count = len(final_groups)
    records = []
    background = torch.zeros_like(foreground_scores[:, :1])
    final_prediction_model = torch.cat(
        [background, foreground_scores], dim=1
    ).argmax(dim=1).detach().cpu()
    local_prediction_model = torch.cat(
        [background, restored], dim=1
    ).argmax(dim=1).detach().cpu()
    for block_index, (z_start, block_valid, roi) in enumerate(
        zip(current_starts, valid, rois)
    ):
        valid_count = int(block_valid.sum())
        target_source = torch.from_numpy(
            aligned_seg[z_start : z_start + valid_count]
        )[:, None].float()
        target_resized = F.interpolate(
            target_source,
            size=(image_size, image_size),
            mode="nearest",
        )[:, 0].long()
        target_model = torch.zeros_like(target_resized)
        for output_id, source_ids in enumerate(final_groups, start=1):
            for source_id in source_ids:
                target_model[target_resized == int(source_id)] = output_id
        roi_mask = torch.zeros_like(target_model, dtype=torch.bool)
        roi_mask[:, roi.y0 : roi.y1, roi.x0 : roi.x1] = True
        pred_final = final_prediction_model[block_index, :valid_count]
        pred_local = local_prediction_model[block_index, :valid_count]
        width = final_count + 1

        def _confusion_for(prediction, mask):
            encoded = (
                target_model[mask].to(torch.int64) * width
                + prediction[mask].to(torch.int64)
            )
            return torch.bincount(
                encoded, minlength=width * width
            ).reshape(width, width).tolist()

        class_voxels = {}
        class_inside = {}
        class_coverage = {}
        roi_voxels = int(roi_mask.sum())
        placement = roi_placement(
            roi,
            (image_size, image_size),
            roi_transform,
        )
        for class_id in range(1, final_count + 1):
            class_mask = target_model == class_id
            total = int(class_mask.sum())
            inside = int((class_mask & roi_mask).sum())
            class_voxels[str(class_id)] = total
            class_inside[str(class_id)] = inside
            class_coverage[str(class_id)] = (
                inside / total if total else None
            )
        records.append(
            {
                "case_id": case_id,
                "z_start": int(z_start),
                "valid_count": valid_count,
                "roi": {
                    "x0": int(roi.x0), "y0": int(roi.y0),
                    "x1": int(roi.x1), "y1": int(roi.y1),
                    "fallback": bool(roi.fallback),
                    "area_fraction": float(
                        (roi.x1 - roi.x0) * (roi.y1 - roi.y0)
                        / (image_size * image_size)
                    ),
                    "transform": roi_transform,
                    "canvas_top": int(placement.top),
                    "canvas_left": int(placement.left),
                    "canvas_height": int(placement.height),
                    "canvas_width": int(placement.width),
                    "scale_x": float(
                        placement.width / max(1, roi.width)
                    ),
                    "scale_y": float(
                        placement.height / max(1, roi.height)
                    ),
                },
                "class_voxels": class_voxels,
                "class_voxels_inside_roi": class_inside,
                "class_coverage": class_coverage,
                "gt_class_fraction_inside_roi": {
                    str(class_id): (
                        class_inside[str(class_id)] / roi_voxels
                        if roi_voxels else 0.0
                    )
                    for class_id in range(1, final_count + 1)
                },
                "confusion_final_full_block": _confusion_for(
                    pred_final, torch.ones_like(roi_mask)
                ),
                "confusion_final_inside_roi": _confusion_for(
                    pred_final, roi_mask
                ),
                "confusion_local_inside_roi": _confusion_for(
                    pred_local, roi_mask
                ),
            }
        )
    return records
