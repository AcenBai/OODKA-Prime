import pytest

from oodka.models.prompts import (
    MYOPS_LGE_ROI_V3_REFINEMENT_GROUPS,
    MYOPS_LGE_ROI_V3_REFINEMENT_PROMPTS,
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


def test_roi_v3_has_five_separate_deployable_prompts():
    assert len(MYOPS_LGE_ROI_V3_REFINEMENT_PROMPTS) == 5
    assert MYOPS_LGE_ROI_V3_REFINEMENT_GROUPS[-2:] == ((1,), (2,))
    assert "pathological myocardial scar tissue" in MYOPS_LGE_ROI_V3_REFINEMENT_PROMPTS["4"]
    assert "pathological myocardial edema" in MYOPS_LGE_ROI_V3_REFINEMENT_PROMPTS["5"]
