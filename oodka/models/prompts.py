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

DATASET_PROMPT_REGISTRY: Dict[str, Dict[str, str]] = {
    "Dataset009_CT_OOD": WHS_CT_PROMPTS,
    "Dataset010_WHS_MRI_OOD": WHS_MRI_PROMPTS,
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
    prompts = DATASET_PROMPT_REGISTRY.get(dataset_name, WHS_CT_PROMPTS)
    ids = sorted(int(k) for k in prompts.keys())
    prompt_to_class_id = {i: ids[i] for i in range(len(ids))}
    return prompts, prompt_to_class_id
