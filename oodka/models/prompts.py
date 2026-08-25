"""Text prompt definitions for each dataset / task."""

from __future__ import annotations

from typing import Dict, Tuple


WHS_CT_PROMPTS = {
    "1": "CT imaging of the left ventricular blood cavity of the heart",
    "2": "CT imaging of the right ventricular blood cavity of the heart",
    "3": "CT imaging of the left atrial blood cavity of the heart",
    "4": "CT imaging of the right atrial blood cavity of the heart",
    "5": "CT imaging of the myocardium of the left ventricle",
    "6": "CT imaging of the ascending aorta in the thorax",
    "7": "CT imaging of the pulmonary artery trunk in the thorax",
}

WHS_MRI_PROMPTS = {
    "1": "MRI of the left ventricular blood cavity of the heart",
    "2": "MRI of the right ventricular blood cavity of the heart",
    "3": "MRI of the left atrial blood cavity of the heart",
    "4": "MRI of the right atrial blood cavity of the heart",
    "5": "MRI of the myocardium of the left ventricle",
    "6": "MRI of the ascending aorta in the thorax",
    "7": "MRI of the pulmonary artery trunk in the thorax",
}

# Two-pass whole-heart ROI experiments. The localization branch predicts one
# binary union of all seven foreground structures; the refinement branch then
# resolves the seven original classes inside the expanded predicted ROI.
WHS_CT_ROI_LOCALIZATION_PROMPTS = {
    "1": (
        "CT imaging of the whole heart, including the left and right "
        "ventricular blood cavities, left and right atrial blood cavities, "
        "left ventricular myocardium, ascending aorta, and pulmonary artery "
        "trunk"
    ),
}

WHS_MRI_ROI_LOCALIZATION_PROMPTS = {
    "1": (
        "MRI of the whole heart, including the left and right ventricular "
        "blood cavities, left and right atrial blood cavities, left "
        "ventricular myocardium, ascending aorta, and pulmonary artery trunk"
    ),
}

WHS_CT_ROI_REFINEMENT_PROMPTS = dict(WHS_CT_PROMPTS)
WHS_MRI_ROI_REFINEMENT_PROMPTS = dict(WHS_MRI_PROMPTS)

WHS_ROI_LOCALIZATION_GROUPS = ((1, 2, 3, 4, 5, 6, 7),)
WHS_ROI_REFINEMENT_GROUPS = tuple((class_id,) for class_id in range(1, 8))

MYOPS_LGE_PROMPTS = {
    "1": (
        "LGE short-axis cardiac MRI showing scar tissue within the "
        "left ventricular myocardium"
    ),
    "2": (
        "LGE short-axis cardiac MRI showing edematous tissue within the "
        "left ventricular myocardium"
    ),
    "3": (
        "LGE short-axis cardiac MRI showing the blood pool within the "
        "left ventricular cavity"
    ),
    "4": (
        "LGE short-axis cardiac MRI showing normal left ventricular "
        "myocardium without scar or edema"
    ),
    "5": (
        "LGE short-axis cardiac MRI showing the blood pool within the "
        "right ventricular cavity"
    ),
}

# Two-pass LGE ROI experiment. ``total_myo`` is an auxiliary localization
# query and is never emitted as a final segmentation class.
MYOPS_LGE_ROI_ANATOMY_PROMPTS = {
    "1": (
        "LGE short-axis cardiac MRI showing the blood pool within the "
        "left ventricular cavity"
    ),
    "2": (
        "LGE short-axis cardiac MRI showing the blood pool within the "
        "right ventricular cavity"
    ),
    "3": (
        "LGE short-axis cardiac MRI showing the complete left ventricular "
        "myocardium, including normal myocardium, scar, and edema"
    ),
}

MYOPS_LGE_ROI_REFINEMENT_PROMPTS = {
    "1": (
        "LGE short-axis cardiac MRI showing normal left ventricular "
        "myocardium"
    ),
    "2": (
        "LGE short-axis cardiac MRI showing pathological myocardial tissue "
        "consisting of scar or edema within the left ventricular myocardium"
    ),
}

# V2 predicts every deployable class inside the myocardium-derived ROI so
# cavity and myocardial prompts are calibrated in the same forward pass.
MYOPS_LGE_ROI_V2_REFINEMENT_PROMPTS = {
    "1": MYOPS_LGE_ROI_ANATOMY_PROMPTS["1"],
    "2": MYOPS_LGE_ROI_ANATOMY_PROMPTS["2"],
    "3": MYOPS_LGE_ROI_REFINEMENT_PROMPTS["1"],
    "4": MYOPS_LGE_ROI_REFINEMENT_PROMPTS["2"],
}

# V3 keeps the V2 cavity-aware ROI refinement, but separates the two
# pathological tissues so their local competition can be measured directly.
MYOPS_LGE_ROI_V3_REFINEMENT_PROMPTS = {
    "1": MYOPS_LGE_ROI_ANATOMY_PROMPTS["1"],
    "2": MYOPS_LGE_ROI_ANATOMY_PROMPTS["2"],
    "3": MYOPS_LGE_ROI_REFINEMENT_PROMPTS["1"],
    "4": (
        "LGE short-axis cardiac MRI showing pathological myocardial scar "
        "tissue within the left ventricular myocardium"
    ),
    "5": (
        "LGE short-axis cardiac MRI showing pathological myocardial edema "
        "within the left ventricular myocardium"
    ),
}

# Original Dataset011 labels grouped for each branch.
MYOPS_LGE_ROI_ANATOMY_GROUPS = ((3,), (5,), (1, 2, 4))
MYOPS_LGE_ROI_REFINEMENT_GROUPS = ((4,), (1, 2))
MYOPS_LGE_ROI_V2_REFINEMENT_GROUPS = ((3,), (5,), (4,), (1, 2))
MYOPS_LGE_ROI_V3_REFINEMENT_GROUPS = ((3,), (5,), (4,), (1,), (2,))

# Final deployment labels: background=0, LV=1, RV=2, normal-MYO=3,
# scar+edema=4.
MYOPS_LGE_ROI_FINAL_GROUPS = ((3,), (5,), (4,), (1, 2))
MYOPS_LGE_ROI_FINAL_NAMES = {
    1: "LV",
    2: "RV",
    3: "normal_myo",
    4: "scar_edema_on_myo",
}
MYOPS_LGE_ROI_SPLIT_FINAL_GROUPS = MYOPS_LGE_ROI_V3_REFINEMENT_GROUPS
MYOPS_LGE_ROI_SPLIT_FINAL_NAMES = {
    1: "LV",
    2: "RV",
    3: "normal_myo",
    4: "scar_on_myo",
    5: "edema_on_myo",
}

DATASET_PROMPT_REGISTRY: Dict[str, Dict[str, str]] = {
    "Dataset009_CT_OOD": WHS_CT_PROMPTS,
    "Dataset010_WHS_MRI_OOD": WHS_MRI_PROMPTS,
    "Dataset011_MYO_LGE_BC_OOD": MYOPS_LGE_PROMPTS,
}


def build_text_prompts_for_dataset(
    dataset_info: Dict = None,
    dataset_name: str = "Dataset009_CT_OOD",
) -> Tuple[Dict[str, str], Dict[int, int]]:
    """
    Returns:
        text_prompts: {"1": "...", "2": "...", ...}
        prompt_to_class_id: {0: 1, 1: 2, ...}
    """
    if dataset_name not in DATASET_PROMPT_REGISTRY:
        known = ", ".join(sorted(DATASET_PROMPT_REGISTRY))
        raise KeyError(
            f"No prompt registry entry for {dataset_name!r}. Known: {known}"
        )
    prompts = DATASET_PROMPT_REGISTRY[dataset_name]
    ids = sorted(int(k) for k in prompts.keys())
    prompt_to_class_id = {i: ids[i] for i in range(len(ids))}
    return prompts, prompt_to_class_id
