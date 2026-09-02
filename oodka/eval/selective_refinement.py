"""Conservative prompt-conditioned updates to a global segmentation.

The local branch proposes one of a configurable set of child prompts.  A
proposal is accepted only when its confidence and, optionally, its separation
from the runner-up pass fixed thresholds.  Every unaccepted voxel is copied
bit-for-bit from the global prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class SelectiveOverwriteStats:
    proposed_voxels: int
    accepted_voxels: int
    changed_voxels: int
    ambiguous_voxels: int
    ineligible_voxels: int
    background_vetoed_voxels: int


def selective_prompt_overwrite(
    global_labels: torch.Tensor,
    local_logits: torch.Tensor,
    child_class_ids: Sequence[int],
    *,
    confidence_threshold: float,
    ambiguity_margin: float = 0.0,
    roi_mask: torch.Tensor | None = None,
    eligible_global_class_ids: Sequence[int] | None = None,
    background_confidence_threshold: float | None = None,
    background_ambiguity_margin: float | None = None,
    background_write_mask: torch.Tensor | None = None,
    background_class_id: int = 0,
) -> tuple[torch.Tensor, SelectiveOverwriteStats]:
    """Return a global segmentation updated by confident local child prompts.

    Args:
        global_labels: Integer labels with arbitrary spatial rank ``[*S]``.
        local_logits: Independent prompt logits with shape ``[C,*S]``.
        child_class_ids: Dataset class id corresponding to every local channel.
        confidence_threshold: Minimum winning sigmoid probability.
        ambiguity_margin: Minimum probability gap between the top two children.
        roi_mask: Optional boolean ``[*S]`` support. Outside it, global labels
            are always retained.
        eligible_global_class_ids: Optional labels that the local branch may
            replace. For example, ``(0, 6, 7)`` protects non-vessel anatomy.
        background_confidence_threshold: Optional stricter confidence required
            when creating a child label from global background.
        background_ambiguity_margin: Optional stricter child separation when
            creating a child label from global background.
        background_write_mask: Optional anatomical support for additions from
            global background. Child-to-child corrections are unaffected.
        background_class_id: Dataset id representing global background.
    """
    if local_logits.ndim < 2:
        raise ValueError("local_logits must have shape [C,*spatial]")
    if tuple(local_logits.shape[1:]) != tuple(global_labels.shape):
        raise ValueError(
            "local/global spatial shapes differ: "
            f"{tuple(local_logits.shape[1:])} vs {tuple(global_labels.shape)}"
        )
    child_ids = tuple(int(value) for value in child_class_ids)
    if len(child_ids) != int(local_logits.shape[0]) or not child_ids:
        raise ValueError("child_class_ids must contain one id per local channel")
    if len(set(child_ids)) != len(child_ids):
        raise ValueError("child_class_ids must be unique")
    if not 0.0 < confidence_threshold < 1.0:
        raise ValueError("confidence_threshold must be in (0,1)")
    if not 0.0 <= ambiguity_margin < 1.0:
        raise ValueError("ambiguity_margin must be in [0,1)")
    if roi_mask is not None and tuple(roi_mask.shape) != tuple(global_labels.shape):
        raise ValueError("roi_mask must match global_labels")
    if background_write_mask is not None and tuple(
        background_write_mask.shape
    ) != tuple(global_labels.shape):
        raise ValueError("background_write_mask must match global_labels")
    background_threshold = (
        float(confidence_threshold)
        if background_confidence_threshold is None
        else float(background_confidence_threshold)
    )
    background_margin = (
        float(ambiguity_margin)
        if background_ambiguity_margin is None
        else float(background_ambiguity_margin)
    )
    if not 0.0 < background_threshold < 1.0:
        raise ValueError("background_confidence_threshold must be in (0,1)")
    if not 0.0 <= background_margin < 1.0:
        raise ValueError("background_ambiguity_margin must be in [0,1)")

    probabilities = torch.sigmoid(local_logits.float())
    winning_probability, winning_index = probabilities.max(dim=0)
    is_background = global_labels == int(background_class_id)
    confidence_required = torch.where(
        is_background,
        winning_probability.new_tensor(background_threshold),
        winning_probability.new_tensor(float(confidence_threshold)),
    )
    proposed = winning_probability >= confidence_required

    if probabilities.shape[0] > 1:
        top_two = probabilities.topk(k=2, dim=0).values
        margin_required = torch.where(
            is_background,
            winning_probability.new_tensor(background_margin),
            winning_probability.new_tensor(float(ambiguity_margin)),
        )
        separated = (top_two[0] - top_two[1]) >= margin_required
    else:
        separated = torch.ones_like(proposed)
    ambiguous = proposed & ~separated
    accepted = proposed & separated
    if roi_mask is not None:
        accepted &= roi_mask.to(device=accepted.device, dtype=torch.bool)
    background_vetoed = torch.zeros_like(accepted)
    if background_write_mask is not None:
        write_allowed = background_write_mask.to(
            device=accepted.device, dtype=torch.bool
        )
        background_vetoed = accepted & is_background & ~write_allowed
        accepted &= ~is_background | write_allowed
    ineligible = torch.zeros_like(accepted)
    if eligible_global_class_ids is not None:
        eligible = torch.zeros_like(accepted)
        for class_id in eligible_global_class_ids:
            eligible |= global_labels == int(class_id)
        ineligible = accepted & ~eligible
        accepted &= eligible

    ids = torch.as_tensor(
        child_ids,
        dtype=global_labels.dtype,
        device=winning_index.device,
    )
    proposals = ids[winning_index]
    output = global_labels.clone()
    output[accepted] = proposals[accepted]
    changed = accepted & (output != global_labels)
    return output, SelectiveOverwriteStats(
        proposed_voxels=int(proposed.sum().item()),
        accepted_voxels=int(accepted.sum().item()),
        changed_voxels=int(changed.sum().item()),
        ambiguous_voxels=int(ambiguous.sum().item()),
        ineligible_voxels=int(ineligible.sum().item()),
        background_vetoed_voxels=int(background_vetoed.sum().item()),
    )


def dilate_mask_in_plane(mask: np.ndarray, iterations: int) -> np.ndarray:
    """Dilate a ``[Z,Y,X]`` mask within each slice without mixing Z planes."""
    from scipy import ndimage

    value = np.asarray(mask, dtype=bool)
    if value.ndim != 3:
        raise ValueError(f"mask must be [Z,Y,X], got {value.shape}")
    if int(iterations) < 0:
        raise ValueError("iterations must be non-negative")
    if int(iterations) == 0:
        return value.copy()
    structure = np.zeros((3, 3, 3), dtype=bool)
    structure[1] = True
    return ndimage.binary_dilation(
        value,
        structure=structure,
        iterations=int(iterations),
    )


def keep_largest_component_per_class(
    segmentation: np.ndarray,
    class_ids: Sequence[int],
) -> np.ndarray:
    """Keep the largest 3-D connected component of every requested class."""
    from scipy import ndimage

    output = np.asarray(segmentation).copy()
    structure = ndimage.generate_binary_structure(output.ndim, 1)
    for class_id in class_ids:
        mask = output == int(class_id)
        if not mask.any():
            continue
        components, count = ndimage.label(mask, structure=structure)
        if count <= 1:
            continue
        sizes = np.bincount(components.ravel())
        sizes[0] = 0
        keep = int(sizes.argmax())
        output[mask & (components != keep)] = 0
    return output
