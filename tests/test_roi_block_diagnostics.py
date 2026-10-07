"""Per-block diagnostics are separate from the inference decision path."""

import numpy as np
import torch

from oodka.data.roi_geometry import ROICoordinates
from oodka.eval.roi_block_diagnostics import collect_block_diagnostics


def test_block_diagnostic_reports_roi_geometry_and_coverage():
    roi = ROICoordinates(1, 1, 3, 3)
    labels = np.zeros((1, 4, 4), dtype=np.int16)
    labels[0, 1:3, 1:3] = 1
    scores = torch.zeros((1, 1, 1, 4, 4))
    restored = torch.ones_like(scores)
    records = collect_block_diagnostics(
        case_id="case",
        current_starts=[0],
        valid=torch.tensor([[True]]),
        rois=[roi],
        aligned_seg=labels,
        foreground_scores=scores,
        restored=restored,
        final_groups=((1,),),
        image_size=4,
        roi_transform="pad",
    )
    assert len(records) == 1
    assert records[0]["class_coverage"]["1"] == 1.0
    assert records[0]["roi"]["area_fraction"] == 0.25
    assert records[0]["roi"]["canvas_height"] == 2
