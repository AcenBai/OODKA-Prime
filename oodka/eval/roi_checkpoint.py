"""Checkpoint-specific ROI task definitions and compatible eval decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..models.prompts import (
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


SUPPORTED_ROI_FORMATS = frozenset({
    "oodka_lge_roi_v1", "oodka_lge_roi_v2",
    "oodka_lge_roi_v3_split5", "oodka_lge_flat_v1",
    "oodka_whs_roi_v1", "oodka_whs_gv_roi_v1",
})


@dataclass(frozen=True)
class ROIEvalProfile:
    """All task-specific choices needed by the shared ROI evaluator."""

    checkpoint_format: str
    dataset_name: str
    decision: str
    roi_transform: str
    is_whs: bool
    is_whs_gv: bool
    is_flat: bool
    anatomy_prompts: dict
    refinement_prompts: dict
    final_groups: Sequence[Sequence[int]]
    final_names: dict


def resolve_roi_eval_profile(
    checkpoint: dict,
    *,
    decision: str = "auto",
    roi_transform: str = "checkpoint",
) -> ROIEvalProfile:
    """Resolve legacy checkpoint formats without changing their eval policy."""
    checkpoint_format = checkpoint.get("format")
    if checkpoint_format not in SUPPORTED_ROI_FORMATS:
        raise ValueError("Checkpoint is not a supported OODKA ROI model")
    is_whs_gv = checkpoint_format == "oodka_whs_gv_roi_v1"
    is_whs = checkpoint_format in {"oodka_whs_roi_v1", "oodka_whs_gv_roi_v1"}
    is_split = checkpoint_format == "oodka_lge_roi_v3_split5"
    is_v2 = checkpoint_format in {"oodka_lge_roi_v2", "oodka_lge_roi_v3_split5"}
    is_flat = checkpoint_format == "oodka_lge_flat_v1"
    if decision == "auto":
        decision = (
            "spatial" if is_whs_gv else "refinement" if is_whs
            else "spatial" if is_v2 else "flat"
        )
    if is_whs_gv and decision != "spatial":
        raise ValueError("WHS-GV checkpoints require --decision spatial (or auto)")
    if is_whs and not is_whs_gv and decision != "refinement":
        raise ValueError("WHS whole-heart checkpoints require refinement (or auto)")
    if is_v2 and decision != "spatial":
        raise ValueError("V2 checkpoints require --decision spatial (or auto)")
    if not is_v2 and not is_whs and decision == "spatial":
        raise ValueError("Spatial hard switching requires a V2 checkpoint")
    if is_flat and decision != "flat":
        raise ValueError("Flat checkpoints require --decision flat (or auto)")

    saved = checkpoint["config"]
    resolved_transform = (
        str(saved.get("roi_transform", "resize"))
        if roi_transform == "checkpoint" else roi_transform
    )
    dataset_name = (
        str(saved.get("dataset_name")) if is_whs
        else "Dataset011_MYO_LGE_BC_OOD"
    )
    if is_whs:
        if dataset_name == "Dataset009_CT_OOD":
            anatomy_prompts = (
                WHS_CT_GV_LOCALIZATION_PROMPTS if is_whs_gv
                else WHS_CT_ROI_LOCALIZATION_PROMPTS
            )
            refinement_prompts = (
                WHS_CT_GV_REFINEMENT_PROMPTS if is_whs_gv
                else WHS_CT_ROI_REFINEMENT_PROMPTS
            )
        elif dataset_name == "Dataset010_WHS_MRI_OOD":
            anatomy_prompts = (
                WHS_MRI_GV_LOCALIZATION_PROMPTS if is_whs_gv
                else WHS_MRI_ROI_LOCALIZATION_PROMPTS
            )
            refinement_prompts = (
                WHS_MRI_GV_REFINEMENT_PROMPTS if is_whs_gv
                else WHS_MRI_ROI_REFINEMENT_PROMPTS
            )
        else:
            raise ValueError(f"Unsupported WHS dataset: {dataset_name}")
        final_groups = WHS_ROI_REFINEMENT_GROUPS
        final_names = {1: "LV", 2: "RV", 3: "LA", 4: "RA", 5: "Myo", 6: "AO", 7: "PA"}
    else:
        anatomy_prompts = (
            MYOPS_LGE_ROI_V2_REFINEMENT_PROMPTS if is_flat
            else MYOPS_LGE_ROI_ANATOMY_PROMPTS
        )
        refinement_prompts = (
            MYOPS_LGE_ROI_V3_REFINEMENT_PROMPTS if is_split
            else MYOPS_LGE_ROI_V2_REFINEMENT_PROMPTS if is_v2
            else MYOPS_LGE_ROI_REFINEMENT_PROMPTS
        )
        final_groups = (
            MYOPS_LGE_ROI_SPLIT_FINAL_GROUPS if is_split
            else MYOPS_LGE_ROI_FINAL_GROUPS
        )
        final_names = (
            MYOPS_LGE_ROI_SPLIT_FINAL_NAMES if is_split
            else MYOPS_LGE_ROI_FINAL_NAMES
        )
    return ROIEvalProfile(
        checkpoint_format=checkpoint_format,
        dataset_name=dataset_name,
        decision=decision,
        roi_transform=resolved_transform,
        is_whs=is_whs,
        is_whs_gv=is_whs_gv,
        is_flat=is_flat,
        anatomy_prompts=anatomy_prompts,
        refinement_prompts=refinement_prompts,
        final_groups=final_groups,
        final_names=final_names,
    )
