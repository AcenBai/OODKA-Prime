"""Compatibility entrypoint for the split mechanism-v3 plotting code."""

import numpy as np
import torch

from scripts import (
    mechanism_v3_common,
    mechanism_v3_decision,
    mechanism_v3_ot,
    mechanism_v3_plots,
    mechanism_v3_representation,
    mechanism_v3_transport,
    mechanism_v3_transport_alignment,
    visualize_mechanism_v3,
)


def test_plotting_facade_keeps_the_public_analysis_helpers():
    assert mechanism_v3_plots._plot_decision is mechanism_v3_decision._plot_decision
    assert mechanism_v3_plots._plot_p_ot is mechanism_v3_ot._plot_p_ot
    assert mechanism_v3_plots._plot_s_ot is mechanism_v3_ot._plot_s_ot
    assert mechanism_v3_plots._plot_representation is mechanism_v3_representation._plot_representation
    assert mechanism_v3_plots._plot_spatial_transport_suite is mechanism_v3_transport._plot_spatial_transport_suite
    assert mechanism_v3_plots._plot_kd_direction_alignment is mechanism_v3_transport_alignment._plot_kd_direction_alignment
    assert visualize_mechanism_v3._plot_decision is mechanism_v3_decision._plot_decision
    assert "_plot_decision" in mechanism_v3_plots.__all__


def test_shared_numeric_helpers_still_use_the_same_scaling():
    assert mechanism_v3_common._robust_max([np.array([0.0, 2.0])], 100.0) == 2.0
    plan = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4)
    aggregated = mechanism_v3_common._aggregate_transport(
        plan, (2, 2), (2, 2), max_side=2
    )
    np.testing.assert_array_equal(aggregated, plan[0].numpy())
