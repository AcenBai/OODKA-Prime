"""The three ROI sources and composition rules remain explicit."""

import numpy as np
import torch

from oodka.data.roi_geometry import ROICoordinates
from oodka.data.roi_policy import ROIGenerator
from oodka.eval.roi_inference import (
    combine_roi_foreground_logits,
    select_block_rois,
)


def test_roi_sources_choose_predicted_oracle_or_full_block():
    logits = torch.full((1, 1, 2, 16, 16), -10.0)
    logits[0, 0, 0, 3:6, 4:7] = 10.0
    valid = torch.tensor([[True, False]])
    labels = np.zeros((2, 16, 16), dtype=np.int16)
    labels[0, 8:11, 9:12] = 1
    generator = ROIGenerator(threshold=0.5, expand=1.0, min_size=1)
    common = dict(
        generator=generator,
        checkpoint={"roi_prompt_index": 0, "roi_source_labels": (1,)},
        final_groups=((1,),),
        aligned_seg=labels,
        image_size=16,
        device=torch.device("cpu"),
    )
    predicted = select_block_rois(
        logits, valid, [0], roi_source="predicted", **common
    )
    oracle = select_block_rois(
        logits, valid, [0], roi_source="ground_truth", **common
    )
    full = select_block_rois(
        logits, valid, [0], roi_source="full", **common
    )
    assert predicted == [ROICoordinates(4, 3, 7, 6)]
    assert oracle == [ROICoordinates(9, 8, 12, 11)]
    assert full == [ROICoordinates(0, 0, 16, 16)]


def test_refinement_decision_keeps_second_pass_logits():
    anatomy = torch.zeros(1, 1, 1, 4, 4)
    refinement = torch.ones(1, 1, 1, 4, 4)
    result = combine_roi_foreground_logits(
        anatomy, refinement, [ROICoordinates(0, 0, 4, 4)],
        decision="refinement", outside_prompt_mapping=(),
    )
    assert result is refinement
