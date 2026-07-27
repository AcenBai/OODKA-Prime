"""Connected-component refinement utilities for segmentation masks."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def keep_largest_foreground_component(seg: np.ndarray) -> np.ndarray:
    """Keep only the largest CC of all foreground labels (>0) combined."""
    from acvl_utils.morphology.morphology_helper import remove_all_but_largest_component

    if seg.ndim != 3:
        raise ValueError(f"expected 3D seg, got {seg.shape}")
    fg = seg > 0
    if not fg.any():
        return seg
    keep = remove_all_but_largest_component(fg)
    out = seg.copy()
    out[fg & ~keep] = 0
    return out


def keep_largest_component_per_class(
    seg: np.ndarray, class_ids: Sequence[int]
) -> np.ndarray:
    """For each class id, keep only its largest CC."""
    from acvl_utils.morphology.morphology_helper import remove_all_but_largest_component

    out = seg.copy()
    for c in class_ids:
        m = out == c
        if not m.any():
            continue
        keep = remove_all_but_largest_component(m)
        out[m & ~keep] = 0
    return out
