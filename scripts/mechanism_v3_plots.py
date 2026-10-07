"""Compatibility exports for mechanism-v3 visualization modules.

New plotting code should live in the focused modules below. Existing imports
from this historical module and visualize_mechanism_v3.py remain supported.
"""


from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/oodka_mechanism_v3_mpl")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F

from oodka.config import TrainConfig
from oodka.data.slice_dataset import FullSliceBlockDataset
from oodka.models.feature_extraction import (
    extract_biomedparse_backbone_features_2p5d,
    extract_nnunet_features,
)
from oodka.models.ot.cost import _coordinates
from oodka.models.prompts import build_text_prompts_for_dataset
from oodka.models.prompts import (
    MYOPS_LGE_ROI_ANATOMY_PROMPTS,
    MYOPS_LGE_ROI_V2_REFINEMENT_PROMPTS,
)
from oodka.data.lge_roi import (
    ROIGenerator,
    crop_and_resize_batch,
    remap_grouped_labels,
)
from oodka.train.forward import (
    _compute_detached_pixel_error_maps,
    _predict_all_prompt_logits,
    _run_pixel_decoder,
)
from oodka.train.model_builder import (
    build_fusion_modules,
    build_prompt_features,
    load_frozen_backbones,
)
from oodka.utils.io_utils import find_raw_image_files

from scripts.mechanism_v3_common import (
    LEVELS,
    CLASS_COLORS,
    DECODER_LABELS,
    _save_json,
    _sha256,
    _git_commit,
    _select_slice,
    _resize_map,
    _rms_5d,
    _rms_4d,
    _token_rms,
    _robust_max,
    _symmetric_limit,
    _ct_limits,
    _draw_gt,
    _heat,
    _share,
    _flatten_feature_slice,
    _mass_image,
    _token_image,
    _row_expected,
    _aggregate_transport,
    _transport_log_density,
    _transport_log_conditional,
    _shared_pca_rgb,
    _normalized_joint_pca_rgb,
)

from scripts.mechanism_v3_transport import (
    _transport_direction_diagnostics,
    _plot_kd_direction_alignment,
    _morton_order,
    _conditional_transport,
    _transport_entropy,
    _select_transport_queries,
    _plot_semantic_sorted_transport,
    _plot_query_transport_atlas,
    _plot_p_query_transport_atlas,
    _select_s_uot_queries,
    _plot_s_uot_rejection_atlas,
    _plot_barycentric_flow,
    _plot_spatial_transport_suite,
)

from scripts.mechanism_v3_representation import (
    _received_weighted_pca_rgb,
    _route_field,
    _feature_cost_components,
    _plot_representation,
    _plot_token_layout,
)

from scripts.mechanism_v3_ot import (
    _plot_p_ot,
    _plot_s_ot,
)

from scripts.mechanism_v3_decision import (
    _plot_decision,
)

__all__ = [name for name in globals() if not name.startswith("__")]
