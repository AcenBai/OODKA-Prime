import numpy as np
import pytest
import torch

from oodka.eval.correction_gate import (
    FEATURE_NAMES,
    CorrectionGate,
    ProtectedFusionMetricContext,
    build_correction_feature_context,
)
from oodka.utils.metrics import dice_no_ignore


def _context() -> object:
    local_logits = np.full((2, 2, 3, 4), -8.0, dtype=np.float32)
    local_logits[0, 0, 1, 1] = 4.0
    local_logits[1, 0, 1, 2] = 5.0
    local_logits[0, 1, 2, 3] = 3.0
    gv_logit = np.full((2, 3, 4), -8.0, dtype=np.float32)
    gv_logit[0, 1, 1] = 6.0
    fallback = np.zeros((2, 3, 4), dtype=np.uint8)
    fallback[1] = 1
    return build_correction_feature_context(
        local_logits,
        gv_logit,
        fallback,
        candidate_min_probability=0.8,
        gv_core_threshold=0.7,
    )


def test_correction_features_do_not_require_global_predictions() -> None:
    context = _context()
    indices = context.candidate_indices()
    assert len(indices) == 3
    features = context.features(indices)
    assert features.shape == (3, len(FEATURE_NAMES))
    assert np.isfinite(features).all()
    assert ((features >= 0.0) & (features <= 1.0)).all()
    assert set(context.proposal_labels(indices)) == {6, 7}


def test_global_labels_only_protect_the_final_candidate_set() -> None:
    context = _context()
    labels = np.zeros(context.shape, dtype=np.int16)
    unrestricted = context.candidate_indices()
    labels.ravel()[unrestricted[0]] = 1
    labels.ravel()[unrestricted[1]] = 6
    labels.ravel()[unrestricted[2]] = 7
    protected = context.candidate_indices(labels)
    assert protected.tolist() == unrestricted[1:].tolist()

    with pytest.raises(ValueError, match="eligible_labels"):
        context.candidate_indices(np.zeros((2, 3, 5), dtype=np.int16))


def test_gv_distance_is_computed_independently_per_slice() -> None:
    context = _context()
    assert context.gv_distance[0, 1, 1] == pytest.approx(0.0)
    assert context.gv_distance[0, 1, 2] == pytest.approx(1.0 / 64.0)
    assert np.all(context.gv_distance[1] == 1.0)
    plane_one = np.flatnonzero(np.indices(context.shape)[0].ravel() == 1)[:1]
    features = context.features(plane_one)
    feature_index = {name: index for index, name in enumerate(FEATURE_NAMES)}
    assert features[0, feature_index["z_relative_to_gv_support"]] == 1.0
    assert features[0, feature_index["y_relative_to_gv_support"]] == 0.5
    assert features[0, feature_index["x_relative_to_gv_support"]] == 0.5


def test_correction_gate_returns_one_logit_per_candidate() -> None:
    model = CorrectionGate(hidden_dims=(8, 4), dropout=0.0)
    output = model(torch.zeros(5, len(FEATURE_NAMES)))
    assert output.shape == (5,)


def test_sparse_protected_metrics_equal_full_volume_dice() -> None:
    rng = np.random.default_rng(42)
    target = rng.integers(0, 8, size=(3, 5, 7), dtype=np.int16)
    global_labels = rng.integers(0, 8, size=target.shape, dtype=np.int16)
    indices = np.flatnonzero(np.isin(global_labels, (0, 6, 7)))
    proposals = rng.choice(np.asarray((6, 7), dtype=np.int16), size=len(indices))
    accept = rng.random(len(indices)) > 0.35

    fused = global_labels.copy()
    fused.ravel()[indices[accept]] = proposals[accept]
    expected_dice, expected_mean, _ = dice_no_ignore(
        fused,
        target,
        tuple(range(1, 8)),
    )
    metrics = ProtectedFusionMetricContext.from_arrays(
        global_labels,
        target,
        tuple(range(1, 8)),
    ).evaluate(indices, proposals, accept)

    assert metrics.dice_per_class == expected_dice
    assert metrics.mean_dice_gt_present == pytest.approx(expected_mean)
    changed = fused != global_labels
    assert metrics.changed_voxels == int(changed.sum())
    assert metrics.beneficial_changes == int(
        (changed & (global_labels != target) & (fused == target)).sum()
    )
    assert metrics.harmful_changes == int(
        (changed & (global_labels == target) & (fused != target)).sum()
    )
