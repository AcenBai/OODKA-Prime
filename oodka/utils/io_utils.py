"""File I/O helpers for NIfTI and general filesystem operations."""

from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
import SimpleITK as sitk


def maybe_mkdir_p(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def read_nifti_as_zyx(path: str) -> np.ndarray:
    img = sitk.ReadImage(path)
    return np.asarray(sitk.GetArrayFromImage(img))


def read_nifti_as_zyx_with_spacing(
    path: str,
) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    img = sitk.ReadImage(path)
    arr = np.asarray(sitk.GetArrayFromImage(img))
    spacing_xyz = tuple(float(x) for x in img.GetSpacing())
    return arr, spacing_xyz


def strip_modality_suffix(stem: str) -> str:
    """CaseXXXX_0000 -> CaseXXXX"""
    if len(stem) >= 5 and stem[-5] == "_" and stem[-4:].isdigit():
        return stem[:-5]
    return stem


def discover_case_ids_from_dir(
    directory: str, file_ending: str, strip_modality: bool = False
) -> List[str]:
    if not directory or not os.path.isdir(directory):
        return []
    ids = set()
    for fn in os.listdir(directory):
        if not fn.endswith(file_ending):
            continue
        stem = fn[: -len(file_ending)]
        if strip_modality:
            stem = strip_modality_suffix(stem)
        ids.add(stem)
    return sorted(ids)


def find_raw_image_files(
    images_dir: str, case_id: str, file_ending: str
) -> List[str]:
    """Find all modality files for a case: CaseXXXX_0000.nii.gz, ..."""
    if not images_dir or not os.path.isdir(images_dir):
        return []
    out = []
    prefix = case_id + "_"
    for fn in os.listdir(images_dir):
        if not fn.endswith(file_ending):
            continue
        if fn == case_id + file_ending or fn.startswith(prefix):
            out.append(os.path.join(images_dir, fn))
    out.sort()
    return out
