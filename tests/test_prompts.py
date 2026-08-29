import pytest

from oodka.models.prompts import (
    MYOPS_LGE_ROI_V3_REFINEMENT_GROUPS,
    MYOPS_LGE_ROI_V3_REFINEMENT_PROMPTS,
    WHS_CT_ROI_LOCALIZATION_PROMPTS,
    WHS_CT_ROI_REFINEMENT_PROMPTS,
    WHS_CT_GV_LOCALIZATION_PROMPTS,
    WHS_CT_GV_REFINEMENT_PROMPTS,
    WHS_MRI_ROI_LOCALIZATION_PROMPTS,
    WHS_MRI_ROI_REFINEMENT_PROMPTS,
    WHS_MRI_GV_LOCALIZATION_PROMPTS,
    WHS_MRI_GV_REFINEMENT_PROMPTS,
    WHS_GV_LOCALIZATION_GROUPS,
    WHS_GV_OUTSIDE_PROMPT_MAPPING,
    WHS_ROI_LOCALIZATION_GROUPS,
    WHS_ROI_REFINEMENT_GROUPS,
    build_text_prompts_for_dataset,
)


def test_dataset011_uses_five_lge_specific_prompts():
    prompts, mapping = build_text_prompts_for_dataset(
        dataset_name="Dataset011_MYO_LGE_BC_OOD"
    )

    assert sorted(prompts) == ["1", "2", "3", "4", "5"]
    assert mapping == {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}
    assert all("LGE short-axis cardiac MRI" in text for text in prompts.values())
    assert "scar tissue" in prompts["1"]
    assert "edematous tissue" in prompts["2"]
    assert "without scar or edema" in prompts["4"]


def test_unknown_dataset_does_not_silently_use_ct_prompts():
    with pytest.raises(KeyError, match="No prompt registry entry"):
        build_text_prompts_for_dataset(dataset_name="Dataset999_UNKNOWN")


def test_whs_two_pass_prompts_localize_union_then_refine_seven_classes():
    assert WHS_ROI_LOCALIZATION_GROUPS == ((1, 2, 3, 4, 5, 6, 7),)
    assert WHS_ROI_REFINEMENT_GROUPS == tuple((value,) for value in range(1, 8))

    for localization, refinement, modality in (
        (
            WHS_CT_ROI_LOCALIZATION_PROMPTS,
            WHS_CT_ROI_REFINEMENT_PROMPTS,
            "CT",
        ),
        (
            WHS_MRI_ROI_LOCALIZATION_PROMPTS,
            WHS_MRI_ROI_REFINEMENT_PROMPTS,
            "MRI",
        ),
    ):
        assert list(localization) == ["1"]
        assert modality in localization["1"]
        assert "whole heart" in localization["1"]
        assert "pulmonary artery trunk" in localization["1"]
        assert sorted(refinement) == [str(value) for value in range(1, 8)]


def test_whs_gv_prompts_merge_ao_pa_and_preserve_five_global_classes():
    assert WHS_GV_LOCALIZATION_GROUPS == (
        (1,), (2,), (3,), (4,), (5,), (6, 7)
    )
    assert WHS_GV_OUTSIDE_PROMPT_MAPPING == tuple(
        (index, index) for index in range(5)
    )
    for localization, refinement, modality in (
        (WHS_CT_GV_LOCALIZATION_PROMPTS, WHS_CT_GV_REFINEMENT_PROMPTS, "CT"),
        (WHS_MRI_GV_LOCALIZATION_PROMPTS, WHS_MRI_GV_REFINEMENT_PROMPTS, "MRI"),
    ):
        assert sorted(localization) == [str(value) for value in range(1, 7)]
        assert modality in localization["6"]
        assert "ascending aorta" in localization["6"]
        assert "pulmonary artery trunk" in localization["6"]
        assert sorted(refinement) == [str(value) for value in range(1, 8)]


def test_roi_v3_has_five_separate_deployable_prompts():
    assert len(MYOPS_LGE_ROI_V3_REFINEMENT_PROMPTS) == 5
    assert MYOPS_LGE_ROI_V3_REFINEMENT_GROUPS[-2:] == ((1,), (2,))
    assert "pathological myocardial scar tissue" in MYOPS_LGE_ROI_V3_REFINEMENT_PROMPTS["4"]
    assert "pathological myocardial edema" in MYOPS_LGE_ROI_V3_REFINEMENT_PROMPTS["5"]
