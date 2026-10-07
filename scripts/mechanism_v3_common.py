"""Shared numerical and plotting helpers for mechanism-v3 figures."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Iterable, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/oodka_mechanism_v3_mpl")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]


LEVELS = (2, 3, 4, 5)
CLASS_COLORS = (
    "#00ff3b",
    "#ff2d2d",
    "#00c8ff",
    "#ffd400",
    "#d12dff",
    "#ff7a00",
    "#ffffff",
)
DECODER_LABELS = {
    2: "mask_features / res2 (128×128)",
    3: "multi_scale / res3 (64×64)",
    4: "multi_scale / res4 (32×32)",
    5: "multi_scale / res5 (16×16)",
}


def _save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _select_slice(gt: np.ndarray, mode: str, slice_index: int | None) -> int:
    if slice_index is not None:
        if not 0 <= slice_index < gt.shape[0]:
            raise ValueError(
                f"slice_index={slice_index} outside [0,{gt.shape[0]})"
            )
        return int(slice_index)
    foreground_area = (gt > 0).reshape(gt.shape[0], -1).sum(axis=1)
    if mode == "largest_foreground":
        return int(foreground_area.argmax())
    all_classes = np.array(
        [
            all(np.any(gt[z] == class_id) for class_id in range(1, 8))
            for z in range(gt.shape[0])
        ]
    )
    if not all_classes.any():
        raise RuntimeError("No slice contains all seven foreground classes")
    candidates = np.flatnonzero(all_classes)
    return int(candidates[np.argmax(foreground_area[candidates])])


def _resize_map(value: torch.Tensor, output_hw: tuple[int, int]) -> np.ndarray:
    resized = F.interpolate(
        value[None, None].float(),
        size=output_hw,
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    return resized.detach().cpu().numpy()


def _rms_5d(
    feature: torch.Tensor, z_index: int, output_hw: tuple[int, int]
) -> np.ndarray:
    """Channel-RMS map for [B,C,Z,H,W]."""
    if feature.ndim != 5:
        raise ValueError(f"Expected [B,C,Z,H,W], got {feature.shape}")
    rms = feature.float().square().mean(dim=1).add(1e-8).sqrt()[0, z_index]
    return _resize_map(rms, output_hw)


def _rms_4d(
    feature: torch.Tensor, z_index: int, output_hw: tuple[int, int]
) -> np.ndarray:
    """Channel-RMS map for pixel-decoder [B*Z,C,H,W]."""
    if feature.ndim != 4:
        raise ValueError(f"Expected [B*Z,C,H,W], got {feature.shape}")
    rms = feature[z_index].float().square().mean(dim=0).add(1e-8).sqrt()
    return _resize_map(rms, output_hw)


def _token_rms(tokens: torch.Tensor) -> np.ndarray:
    return (
        tokens.float().square().mean(dim=-1).add(1e-8).sqrt().detach().cpu().numpy()
    )


def _robust_max(values: Iterable[np.ndarray], percentile: float = 99.5) -> float:
    flattened = [np.asarray(value, dtype=np.float32).reshape(-1) for value in values]
    if not flattened:
        return 1.0
    merged = np.concatenate(flattened)
    finite = merged[np.isfinite(merged)]
    if finite.size == 0:
        return 1.0
    return max(float(np.percentile(finite, percentile)), 1e-8)


def _symmetric_limit(values: Iterable[np.ndarray], percentile: float = 99.5) -> float:
    return _robust_max((np.abs(value) for value in values), percentile)


def _ct_limits(image: np.ndarray) -> tuple[float, float]:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return 0.0, 1.0
    return float(np.percentile(finite, 1.0)), float(np.percentile(finite, 99.0))


def _draw_gt(
    axis: plt.Axes,
    image: np.ndarray,
    gt: np.ndarray,
    label_names: dict[int, str],
    *,
    class_id: int | None = None,
    legend: bool = False,
) -> None:
    lo, hi = _ct_limits(image)
    axis.imshow(image, cmap="gray", vmin=lo, vmax=hi)
    ids = [class_id] if class_id is not None else list(range(1, 8))
    handles = []
    for cid in ids:
        mask = gt == cid
        if not mask.any():
            continue
        color = CLASS_COLORS[cid - 1]
        axis.contour(mask, levels=[0.5], colors=[color], linewidths=1.25)
        if class_id is not None:
            axis.contourf(
                mask,
                levels=[0.5, 1.5],
                colors=[color],
                alpha=0.22,
            )
        handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                lw=2,
                label=label_names.get(cid, str(cid)),
            )
        )
    if legend and handles:
        axis.legend(handles=handles, fontsize=7, loc="lower left", framealpha=0.75)
    axis.axis("off")


def _heat(
    axis: plt.Axes,
    value: np.ndarray,
    *,
    title: str,
    vmax: float,
    cmap: str = "magma",
    vmin: float = 0.0,
):
    image = axis.imshow(value, cmap=cmap, vmin=vmin, vmax=vmax)
    axis.set_title(title, fontsize=10)
    axis.axis("off")
    return image


def _share(
    axis: plt.Axes,
    value: np.ndarray,
    *,
    title: str,
):
    image = axis.imshow(value, cmap="viridis", vmin=0.0, vmax=1.0)
    axis.set_title(title, fontsize=10)
    axis.axis("off")
    return image


def _flatten_feature_slice(feature: torch.Tensor, z_index: int) -> torch.Tensor:
    """Return one feature slice as [1,C,H,W]."""
    if feature.ndim != 5 or feature.shape[0] != 1:
        raise ValueError(f"Expected [1,C,Z,H,W], got {feature.shape}")
    return feature[:, :, z_index]


def _mass_image(value: torch.Tensor, grid: tuple[int, int]) -> np.ndarray:
    """Dimensionless mass density; uniform mass equals one."""
    mass = value[0].detach().float().cpu().numpy()
    return mass.reshape(grid) * mass.size


def _token_image(value: np.ndarray, grid: tuple[int, int]) -> np.ndarray:
    return np.asarray(value).reshape(grid)


def _row_expected(
    transport: torch.Tensor, value: torch.Tensor
) -> np.ndarray:
    """Expected pairwise value per base token under row-normalized transport."""
    row = transport.sum(dim=-1).clamp_min(1e-12)
    expected = (transport * value).sum(dim=-1) / row
    return expected[0].detach().cpu().numpy()


def _aggregate_transport(
    transport: torch.Tensor,
    base_grid: tuple[int, int],
    expert_grid: tuple[int, int],
    max_side: int = 16,
) -> np.ndarray:
    """Block-sum a square spatial transport into a readable matrix."""
    hb, wb = base_grid
    he, we = expert_grid
    tb, tw = min(hb, max_side), min(wb, max_side)
    te, tx = min(he, max_side), min(we, max_side)
    if hb % tb or wb % tw or he % te or we % tx:
        raise ValueError(
            f"Cannot evenly aggregate {base_grid} x {expert_grid} to "
            f"{(tb, tw)} x {(te, tx)}"
        )
    fb_h, fb_w = hb // tb, wb // tw
    fe_h, fe_w = he // te, we // tx
    value = transport[0].reshape(hb, wb, he, we)
    value = value.reshape(tb, fb_h, tw, fb_w, te, fe_h, tx, fe_w)
    value = value.sum(dim=(1, 3, 5, 7))
    return value.reshape(tb * tw, te * tx).detach().cpu().numpy()


def _transport_log_density(aggregated: np.ndarray) -> np.ndarray:
    total = max(float(aggregated.sum()), 1e-12)
    relative_uniform = aggregated / total * aggregated.size
    return np.log10(relative_uniform + 1e-6)


def _transport_log_conditional(aggregated: np.ndarray) -> np.ndarray:
    rows = aggregated.sum(axis=1, keepdims=True)
    conditional = aggregated / np.maximum(rows, 1e-12)
    relative_uniform = conditional * aggregated.shape[1]
    return np.log10(relative_uniform + 1e-6)


def _shared_pca_rgb(
    expert_tokens: torch.Tensor,
    teacher_tokens: torch.Tensor,
    student_tokens: torch.Tensor,
    grid: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use one PCA and one component range for expert/teacher/student tokens."""
    values = torch.cat(
        [
            expert_tokens[0].float(),
            teacher_tokens[0].float(),
            student_tokens[0].float(),
        ],
        dim=0,
    ).detach().cpu()
    values = values - values.mean(dim=0, keepdim=True)
    torch.manual_seed(0)
    _u, _s, vectors = torch.pca_lowrank(values, q=3, center=False, niter=4)
    projected = values @ vectors[:, :3]
    lo = torch.quantile(projected, 0.01, dim=0)
    hi = torch.quantile(projected, 0.99, dim=0)
    projected = ((projected - lo) / (hi - lo).clamp_min(1e-8)).clamp(0.0, 1.0)
    count = expert_tokens.shape[1]
    arrays = projected.reshape(3, count, 3).numpy()
    return tuple(array.reshape(*grid, 3) for array in arrays)


def _normalized_joint_pca_rgb(
    token_groups: Sequence[torch.Tensor],
    grids: Sequence[tuple[int, int]],
) -> tuple[np.ndarray, ...]:
    """Project normalized token directions through one shared PCA basis."""
    if len(token_groups) != len(grids) or not token_groups:
        raise ValueError("token_groups and grids must be non-empty and aligned")
    normalized = []
    counts = []
    for tokens, grid in zip(token_groups, grids):
        if tokens.ndim != 3 or tokens.shape[0] != 1:
            raise ValueError(f"Expected token tensor [1,N,C], got {tokens.shape}")
        if tokens.shape[1] != grid[0] * grid[1]:
            raise ValueError(f"Token count {tokens.shape[1]} does not match {grid}")
        value = F.normalize(tokens[0].float(), dim=-1, eps=1e-8)
        normalized.append(value.detach().cpu())
        counts.append(value.shape[0])
    values = torch.cat(normalized, dim=0)
    centered = values - values.mean(dim=0, keepdim=True)
    torch.manual_seed(0)
    _u, _s, vectors = torch.pca_lowrank(
        centered, q=3, center=False, niter=4
    )
    projected = centered @ vectors[:, :3]
    lo = torch.quantile(projected, 0.01, dim=0)
    hi = torch.quantile(projected, 0.99, dim=0)
    projected = ((projected - lo) / (hi - lo).clamp_min(1e-8)).clamp(0.0, 1.0)
    arrays = torch.split(projected, counts, dim=0)
    return tuple(
        array.numpy().reshape(*grid, 3)
        for array, grid in zip(arrays, grids)
    )
