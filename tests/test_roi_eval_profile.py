"""Checkpoint compatibility rules for the shared CT/MRI/LGE ROI evaluator."""

import pytest

from oodka.eval.roi_checkpoint import resolve_roi_eval_profile


@pytest.mark.parametrize(
    ("checkpoint_format", "dataset_name", "decision", "class_count"),
    [
        ("oodka_lge_roi_v1", "Dataset011_MYO_LGE_BC_OOD", "flat", 4),
        ("oodka_lge_roi_v2", "Dataset011_MYO_LGE_BC_OOD", "spatial", 4),
        ("oodka_lge_roi_v3_split5", "Dataset011_MYO_LGE_BC_OOD", "spatial", 5),
        ("oodka_lge_flat_v1", "Dataset011_MYO_LGE_BC_OOD", "flat", 4),
        ("oodka_whs_roi_v1", "Dataset009_CT_OOD", "refinement", 7),
        ("oodka_whs_roi_v1", "Dataset010_WHS_MRI_OOD", "refinement", 7),
        ("oodka_whs_gv_roi_v1", "Dataset010_WHS_MRI_OOD", "spatial", 7),
    ],
)
def test_roi_checkpoint_resolves_legacy_policies(
    checkpoint_format, dataset_name, decision, class_count
):
    profile = resolve_roi_eval_profile(
        {"format": checkpoint_format, "config": {"dataset_name": dataset_name}},
    )
    assert profile.dataset_name == dataset_name
    assert profile.decision == decision
    assert profile.roi_transform == "resize"
    assert len(profile.final_groups) == class_count


def test_roi_checkpoint_transform_is_overridable():
    checkpoint = {
        "format": "oodka_whs_roi_v1",
        "config": {"dataset_name": "Dataset009_CT_OOD", "roi_transform": "pad"},
    }
    assert resolve_roi_eval_profile(checkpoint).roi_transform == "pad"
    assert resolve_roi_eval_profile(checkpoint, roi_transform="letterbox").roi_transform == "letterbox"


@pytest.mark.parametrize(
    ("checkpoint_format", "decision"),
    [
        ("oodka_whs_roi_v1", "spatial"),
        ("oodka_whs_gv_roi_v1", "refinement"),
        ("oodka_lge_roi_v2", "flat"),
    ],
)
def test_incompatible_roi_decisions_remain_rejected(checkpoint_format, decision):
    with pytest.raises(ValueError):
        resolve_roi_eval_profile(
            {
                "format": checkpoint_format,
                "config": {"dataset_name": "Dataset010_WHS_MRI_OOD"},
            },
            decision=decision,
        )
