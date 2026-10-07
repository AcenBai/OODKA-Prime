"""Compatibility imports for the historical LGE ROI module.

New code should import from roi_geometry, roi_policy, roi_augmentation, or
roi_cache. Keep this path for existing experiments and external callers.
"""

from .roi_geometry import (
    ROICoordinates,
    ROIPlacement,
    ROI_TRANSFORMS,
    crop_and_resize_batch,
    restore_roi_logits,
    roi_diagnostics,
    roi_placement,
    transform_roi_tensor,
)
from .roi_policy import (
    ROIGenerator,
    hard_switch_foreground_logits,
    jitter_roi,
    oracle_block_rois,
    remap_grouped_labels,
    roi_prompt_visibility,
)
from .roi_augmentation import augment_lge_batch
from .roi_cache import ROICache
