"""Pure block-level ROI selection and two-pass output composition."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ..data.roi_geometry import ROICoordinates
from ..data.roi_policy import ROIGenerator, hard_switch_foreground_logits


def select_block_rois(
    anatomy_logits: torch.Tensor,
    valid_z: torch.Tensor,
    block_starts: Sequence[int],
    *,
    roi_source: str,
    generator: ROIGenerator,
    checkpoint: dict,
    final_groups: Sequence[Sequence[int]],
    aligned_seg: np.ndarray,
    image_size: int,
    device: torch.device,
) -> list[ROICoordinates]:
    """Choose one XY box per Z block using the historical eval semantics."""
    if roi_source == "predicted":
        block_probability = torch.sigmoid(
            anatomy_logits[:, int(checkpoint.get("roi_prompt_index", 2))]
        ).masked_fill(
            ~valid_z.to(device)[:, :, None, None], 0.0
        ).amax(dim=1)
        return [
            generator.from_probability(value)
            for value in block_probability.detach()
        ]
    if roi_source == "ground_truth":
        source_ids = tuple(
            int(source_id) for source_id in checkpoint.get(
                "roi_source_labels",
                tuple(source_id for group in final_groups for source_id in group),
            )
        )
        rois = []
        for z_start, block_valid in zip(block_starts, valid_z):
            valid_count = int(block_valid.sum())
            target_mask = np.isin(
                aligned_seg[z_start : z_start + valid_count], source_ids
            ).any(axis=0)
            target_resized = F.interpolate(
                torch.from_numpy(target_mask)[None, None].float(),
                size=(image_size, image_size),
                mode="nearest",
            )[0, 0]
            rois.append(generator.from_probability(target_resized))
        return rois
    if roi_source == "full":
        return [
            ROICoordinates(0, 0, image_size, image_size, fallback=False)
            for _ in block_starts
        ]
    raise ValueError(f"Unknown ROI source: {roi_source!r}")


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


def combine_roi_foreground_logits(
    anatomy_logits: torch.Tensor,
    restored_roi_logits: torch.Tensor,
    rois: Sequence[ROICoordinates],
    *,
    decision: str,
    outside_prompt_mapping: Sequence[tuple[int, int]],
) -> torch.Tensor:
    """Apply the checkpoint's cross-pass decision to model-grid logits."""
    if decision == "refinement":
        return restored_roi_logits
    if decision == "spatial":
        return hard_switch_foreground_logits(
            anatomy_logits, restored_roi_logits, rois, outside_prompt_mapping
        )
    if decision == "hierarchical":
        return _hierarchical_foreground_logits(anatomy_logits, restored_roi_logits)
    # Historical independent/flat composition is retained for old checkpoints.
    return torch.cat([anatomy_logits[:, 0:2], restored_roi_logits], dim=1)
