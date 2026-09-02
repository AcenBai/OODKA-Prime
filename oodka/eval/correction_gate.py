"""Learned accept/reject controller for protected AO/PA refinement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from scipy import ndimage
from torch import nn


FEATURE_NAMES = (
    "local_ao_probability",
    "local_pa_probability",
    "local_winner_probability",
    "local_ao_pa_margin",
    "gv_union_probability",
    "local_ao_mean5",
    "local_pa_mean5",
    "gv_union_mean5",
    "distance_to_gv_core",
    "fallback",
    "z_relative_to_gv_support",
    "y_relative_to_gv_support",
    "x_relative_to_gv_support",
    "z_normalized",
    "y_normalized",
    "x_normalized",
)


class CorrectionGate(nn.Module):
    """Small per-voxel MLP predicting whether a local proposal is trustworthy."""

    def __init__(
        self,
        input_dim: int = len(FEATURE_NAMES),
        hidden_dims: Sequence[int] = (32, 16),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = int(input_dim)
        for hidden in hidden_dims:
            layers.extend(
                [
                    nn.Linear(current, int(hidden)),
                    nn.LayerNorm(int(hidden)),
                    nn.SiLU(),
                    nn.Dropout(float(dropout)),
                ]
            )
            current = int(hidden)
        layers.append(nn.Linear(current, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def _sigmoid(array: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(array, dtype=np.float32), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-value))


def _distance_to_core(core: np.ndarray, cap: float = 64.0) -> np.ndarray:
    """Return capped in-plane distance to a high-confidence GV seed."""
    output = np.empty(core.shape, dtype=np.float32)
    for z_index, plane in enumerate(core):
        if plane.any():
            distance = ndimage.distance_transform_edt(~plane)
            output[z_index] = np.minimum(distance, cap) / cap
        else:
            output[z_index].fill(1.0)
    return output


def _support_bounds(
    core: np.ndarray,
) -> tuple[tuple[int, int] | None, np.ndarray, np.ndarray]:
    """Return volume-z and per-slice in-plane bounds of GV support."""
    z_count = core.shape[0]
    y_bounds = np.full((z_count, 2), -1, dtype=np.int32)
    x_bounds = np.full((z_count, 2), -1, dtype=np.int32)
    populated = []
    for z_index, plane in enumerate(core):
        y_index, x_index = np.nonzero(plane)
        if len(y_index) == 0:
            continue
        populated.append(z_index)
        y_bounds[z_index] = (int(y_index.min()), int(y_index.max()))
        x_bounds[z_index] = (int(x_index.min()), int(x_index.max()))
    z_bounds = (min(populated), max(populated)) if populated else None
    return z_bounds, y_bounds, x_bounds


@dataclass
class CorrectionFeatureContext:
    """Cached dense evidence with sparse feature extraction by flat index."""

    fallback_mask: np.ndarray
    p_ao: np.ndarray
    p_pa: np.ndarray
    p_gv: np.ndarray
    ao_mean5: np.ndarray
    pa_mean5: np.ndarray
    gv_mean5: np.ndarray
    gv_distance: np.ndarray
    gv_z_bounds: tuple[int, int] | None
    gv_y_bounds: np.ndarray
    gv_x_bounds: np.ndarray
    candidate_min_probability: float

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.p_ao.shape)

    def candidate_indices(
        self,
        eligible_labels: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return confident local proposals, optionally protected by global labels."""
        confident = np.maximum(self.p_ao, self.p_pa) >= float(
            self.candidate_min_probability
        )
        if eligible_labels is not None:
            labels = np.asarray(eligible_labels)
            if tuple(labels.shape) != self.shape:
                raise ValueError("eligible_labels must match the evidence shape")
            confident &= np.isin(labels, (0, 6, 7))
        return np.flatnonzero(confident)

    def proposal_labels(self, indices: np.ndarray) -> np.ndarray:
        ao = self.p_ao.ravel()[indices]
        pa = self.p_pa.ravel()[indices]
        return np.where(ao >= pa, 6, 7).astype(np.int16)

    def features(self, indices: np.ndarray) -> np.ndarray:
        indices = np.asarray(indices, dtype=np.int64)
        ao = self.p_ao.ravel()[indices]
        pa = self.p_pa.ravel()[indices]
        gv = self.p_gv.ravel()[indices]
        z_index, y_index, x_index = np.unravel_index(indices, self.shape)
        z_den = max(1, self.shape[0] - 1)
        y_den = max(1, self.shape[1] - 1)
        x_den = max(1, self.shape[2] - 1)
        relative_z = np.full(len(indices), 0.5, dtype=np.float32)
        if self.gv_z_bounds is not None:
            z_start, z_end = self.gv_z_bounds
            relative_z = np.clip(
                (z_index.astype(np.float32) - z_start) / max(1, z_end - z_start),
                0.0,
                1.0,
            )
        relative_y = np.full(len(indices), 0.5, dtype=np.float32)
        relative_x = np.full(len(indices), 0.5, dtype=np.float32)
        y_bounds = self.gv_y_bounds[z_index]
        x_bounds = self.gv_x_bounds[z_index]
        valid_plane = y_bounds[:, 0] >= 0
        relative_y[valid_plane] = np.clip(
            (y_index[valid_plane] - y_bounds[valid_plane, 0])
            / np.maximum(1, y_bounds[valid_plane, 1] - y_bounds[valid_plane, 0]),
            0.0,
            1.0,
        )
        relative_x[valid_plane] = np.clip(
            (x_index[valid_plane] - x_bounds[valid_plane, 0])
            / np.maximum(1, x_bounds[valid_plane, 1] - x_bounds[valid_plane, 0]),
            0.0,
            1.0,
        )
        return np.stack(
            [
                ao,
                pa,
                np.maximum(ao, pa),
                np.abs(ao - pa),
                gv,
                self.ao_mean5.ravel()[indices],
                self.pa_mean5.ravel()[indices],
                self.gv_mean5.ravel()[indices],
                self.gv_distance.ravel()[indices],
                self.fallback_mask.ravel()[indices].astype(np.float32),
                relative_z,
                relative_y,
                relative_x,
                z_index.astype(np.float32) / z_den,
                y_index.astype(np.float32) / y_den,
                x_index.astype(np.float32) / x_den,
            ],
            axis=1,
        ).astype(np.float32, copy=False)


@dataclass(frozen=True)
class SparseFusionMetrics:
    dice_per_class: dict[int, float | None]
    mean_dice_gt_present: float
    changed_voxels: int
    beneficial_changes: int
    harmful_changes: int


@dataclass
class ProtectedFusionMetricContext:
    """Exact Dice accounting from sparse protected AO/PA overwrites."""

    global_labels: np.ndarray
    target: np.ndarray
    class_ids: tuple[int, ...]
    predicted_counts: dict[int, int]
    target_counts: dict[int, int]
    intersections: dict[int, int]

    @classmethod
    def from_arrays(
        cls,
        global_labels: np.ndarray,
        target: np.ndarray,
        class_ids: Sequence[int],
    ) -> "ProtectedFusionMetricContext":
        global_value = np.asarray(global_labels)
        target_value = np.asarray(target)
        if global_value.shape != target_value.shape:
            raise ValueError("global_labels and target must have the same shape")
        ids = tuple(int(value) for value in class_ids)
        return cls(
            global_labels=global_value,
            target=target_value,
            class_ids=ids,
            predicted_counts={
                class_id: int((global_value == class_id).sum()) for class_id in ids
            },
            target_counts={
                class_id: int((target_value == class_id).sum()) for class_id in ids
            },
            intersections={
                class_id: int(
                    ((global_value == class_id) & (target_value == class_id)).sum()
                )
                for class_id in ids
            },
        )

    def evaluate(
        self,
        indices: np.ndarray,
        proposals: np.ndarray,
        accept: np.ndarray,
    ) -> SparseFusionMetrics:
        indices = np.asarray(indices, dtype=np.int64)
        proposals = np.asarray(proposals)
        accept = np.asarray(accept, dtype=bool)
        if proposals.shape != indices.shape or accept.shape != indices.shape:
            raise ValueError("indices, proposals, and accept must have the same shape")
        accepted_indices = indices[accept]
        new_labels = proposals[accept]
        old_labels = self.global_labels.ravel()[accepted_indices]
        truth = self.target.ravel()[accepted_indices]
        changed = old_labels != new_labels
        old_labels = old_labels[changed]
        new_labels = new_labels[changed]
        truth = truth[changed]

        predicted_counts = self.predicted_counts.copy()
        intersections = self.intersections.copy()
        for class_id in self.class_ids:
            predicted_counts[class_id] += int((new_labels == class_id).sum())
            predicted_counts[class_id] -= int((old_labels == class_id).sum())
            intersections[class_id] += int(
                ((new_labels == class_id) & (truth == class_id)).sum()
            )
            intersections[class_id] -= int(
                ((old_labels == class_id) & (truth == class_id)).sum()
            )

        dice_per_class: dict[int, float | None] = {}
        gt_present = []
        for class_id in self.class_ids:
            denominator = predicted_counts[class_id] + self.target_counts[class_id]
            dice = (
                2.0 * intersections[class_id] / denominator if denominator > 0 else None
            )
            dice_per_class[class_id] = dice
            if self.target_counts[class_id] > 0:
                gt_present.append(float(dice))
        before_correct = old_labels == truth
        after_correct = new_labels == truth
        return SparseFusionMetrics(
            dice_per_class=dice_per_class,
            mean_dice_gt_present=float(np.mean(gt_present)) if gt_present else 0.0,
            changed_voxels=int(changed.sum()),
            beneficial_changes=int((~before_correct & after_correct).sum()),
            harmful_changes=int((before_correct & ~after_correct).sum()),
        )


def build_correction_feature_context(
    local_logits: np.ndarray,
    gv_logit: np.ndarray,
    fallback_mask: np.ndarray,
    *,
    candidate_min_probability: float = 0.5,
    gv_core_threshold: float = 0.9,
) -> CorrectionFeatureContext:
    """Build bounded semantic, contextual, and spatial gate features."""
    local = np.asarray(local_logits)
    if local.ndim != 4 or local.shape[0] != 2:
        raise ValueError(f"local_logits must be [2,Z,Y,X], got {local.shape}")
    evidence_shape = tuple(local.shape[1:])
    if tuple(np.shape(gv_logit)) != evidence_shape:
        raise ValueError("gv_logit must match local_logits spatial shape")
    if tuple(np.shape(fallback_mask)) != evidence_shape:
        raise ValueError("fallback_mask must match local_logits spatial shape")
    if not 0.0 < candidate_min_probability < 1.0:
        raise ValueError("candidate_min_probability must be in (0,1)")
    if not 0.0 < gv_core_threshold < 1.0:
        raise ValueError("gv_core_threshold must be in (0,1)")

    p_ao = _sigmoid(local[0])
    p_pa = _sigmoid(local[1])
    p_gv = _sigmoid(gv_logit)
    filter_size = (1, 5, 5)
    ao_mean5 = ndimage.uniform_filter(p_ao, size=filter_size, mode="nearest")
    pa_mean5 = ndimage.uniform_filter(p_pa, size=filter_size, mode="nearest")
    gv_mean5 = ndimage.uniform_filter(p_gv, size=filter_size, mode="nearest")
    gv_core = p_gv >= float(gv_core_threshold)
    gv_distance = _distance_to_core(gv_core)
    gv_z_bounds, gv_y_bounds, gv_x_bounds = _support_bounds(gv_core)
    return CorrectionFeatureContext(
        fallback_mask=np.asarray(fallback_mask, dtype=bool),
        p_ao=p_ao,
        p_pa=p_pa,
        p_gv=p_gv,
        ao_mean5=ao_mean5,
        pa_mean5=pa_mean5,
        gv_mean5=gv_mean5,
        gv_distance=gv_distance,
        gv_z_bounds=gv_z_bounds,
        gv_y_bounds=gv_y_bounds,
        gv_x_bounds=gv_x_bounds,
        candidate_min_probability=float(candidate_min_probability),
    )


@torch.no_grad()
def predict_gate_probabilities(
    model: CorrectionGate,
    context: CorrectionFeatureContext,
    indices: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 262144,
) -> np.ndarray:
    """Predict sparse accept probabilities without materializing all features."""
    model.eval()
    output = np.empty(len(indices), dtype=np.float32)
    for start in range(0, len(indices), int(batch_size)):
        end = min(len(indices), start + int(batch_size))
        features = torch.from_numpy(context.features(indices[start:end])).to(device)
        output[start:end] = torch.sigmoid(model(features)).cpu().numpy()
    return output


def load_gate_checkpoint(
    path: str,
    device: torch.device,
) -> tuple[CorrectionGate, dict]:
    checkpoint = torch.load(path, map_location=device)
    if checkpoint.get("format") != "oodka_correction_gate_v1":
        raise ValueError(f"Unsupported correction-gate checkpoint: {path}")
    if tuple(checkpoint["feature_names"]) != FEATURE_NAMES:
        raise ValueError("Correction-gate feature schema mismatch")
    model = CorrectionGate(
        input_dim=len(checkpoint["feature_names"]),
        hidden_dims=checkpoint["hidden_dims"],
        dropout=float(checkpoint["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint
