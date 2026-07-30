#!/usr/bin/env python3
"""Generate comparable representation, OT, and routing-decision visualizations.

The script analyzes one selected 2.5D block and writes a case-specific tree:

    <output_root>/<case>_z<slice>/
        representation/
        ot/
        decision/
        manifest.json

Energy plots use channel-RMS rather than channel-L2 so tensors with different
channel counts remain numerically comparable. Color limits are shared within
each semantically compatible feature family and recorded in JSON.
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


def _received_weighted_pca_rgb(
    expert_tokens: torch.Tensor,
    teacher_tokens: torch.Tensor,
    student_tokens: torch.Tensor,
    received: torch.Tensor,
    grid: tuple[int, int],
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray, dict]:
    """Shared PCA with received-mass weighting for teacher/student token rows."""
    expert = expert_tokens[0].float().detach().cpu()
    teacher = teacher_tokens[0].float().detach().cpu()
    student = student_tokens[0].float().detach().cpu()
    received_cpu = received[0].float().detach().cpu().clamp_min(0.0)
    count = expert.shape[0]

    positive = received_cpu[received_cpu > 0]
    if positive.numel():
        scale = torch.quantile(positive, 0.99).clamp_min(1e-12)
        visibility = torch.log1p(received_cpu / scale * 9.0) / np.log(10.0)
        visibility = visibility.clamp(0.0, 1.0)
    else:
        visibility = torch.zeros_like(received_cpu)

    values = torch.cat((expert, teacher, student), dim=0)
    weights = torch.cat(
        (
            torch.ones_like(received_cpu),
            visibility,
            visibility,
        ),
        dim=0,
    ).clamp_min(1e-6)
    mean = (values * weights[:, None]).sum(dim=0, keepdim=True)
    mean = mean / weights.sum().clamp_min(1e-8)
    centered = values - mean
    weighted = centered * weights.sqrt()[:, None]
    torch.manual_seed(0)
    _u, singular, vectors = torch.pca_lowrank(
        weighted, q=3, center=False, niter=4
    )
    projected = centered @ vectors[:, :3]
    lo = torch.quantile(projected, 0.01, dim=0)
    hi = torch.quantile(projected, 0.99, dim=0)
    projected = ((projected - lo) / (hi - lo).clamp_min(1e-8)).clamp(0.0, 1.0)
    arrays = projected.reshape(3, count, 3).numpy()
    total_variance = weighted.square().sum().clamp_min(1e-12)
    explained = (singular[:3].square() / total_variance).numpy()
    info = {
        "method": "received-mass-weighted shared PCA",
        "visibility_transform": (
            "log1p(9 * received_mass / positive_P99), clipped to [0, 1]"
        ),
        "received_positive_p99": float(
            torch.quantile(positive, 0.99).item() if positive.numel() else 0.0
        ),
        "explained_variance_ratio": explained.tolist(),
    }
    return (
        tuple(array.reshape(*grid, 3) for array in arrays),
        visibility.numpy().reshape(grid),
        info,
    )


def _route_field(
    axis: plt.Axes,
    image: np.ndarray,
    gt: np.ndarray,
    label_names: dict[int, str],
    transport: torch.Tensor,
    base_grid: tuple[int, int],
    expert_grid: tuple[int, int],
    weight: torch.Tensor,
    *,
    title: str,
    top_k: int = 28,
) -> None:
    _draw_gt(axis, image, gt, label_names)
    hb, wb = base_grid
    he, we = expert_grid
    coords_b = _coordinates(
        hb,
        wb,
        device=transport.device,
        dtype=transport.dtype,
    )
    coords_e = _coordinates(
        he,
        we,
        device=transport.device,
        dtype=transport.dtype,
    )
    row = transport[0].sum(dim=-1).clamp_min(1e-12)
    source = torch.matmul(transport[0], coords_e) / row[:, None]
    score = weight[0].detach().float()
    k = min(top_k, score.numel())
    indices = torch.topk(score, k=k, largest=True).indices
    start = coords_b[indices].detach().cpu().numpy()
    end = source[indices].detach().cpu().numpy()
    score_np = score[indices].cpu().numpy()
    height, width = image.shape
    start_xy = np.column_stack(
        (
            (start[:, 1] + 1.0) * 0.5 * (width - 1),
            (start[:, 0] + 1.0) * 0.5 * (height - 1),
        )
    )
    end_xy = np.column_stack(
        (
            (end[:, 1] + 1.0) * 0.5 * (width - 1),
            (end[:, 0] + 1.0) * 0.5 * (height - 1),
        )
    )
    delta = end_xy - start_xy
    normalized = (score_np - score_np.min()) / (
        max(float(score_np.max() - score_np.min()), 1e-8)
    )
    axis.quiver(
        start_xy[:, 0],
        start_xy[:, 1],
        delta[:, 0],
        delta[:, 1],
        normalized,
        cmap="plasma",
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.004,
        headwidth=3.5,
        headlength=4.5,
        alpha=0.9,
    )
    axis.scatter(
        start_xy[:, 0],
        start_xy[:, 1],
        c=normalized,
        cmap="plasma",
        s=9,
        edgecolors="none",
    )
    axis.set_title(title, fontsize=10)


def _feature_cost_components(
    base_tokens: torch.Tensor,
    expert_tokens: torch.Tensor,
    grid: tuple[int, int],
    semantic: torch.Tensor | None,
    *,
    feature_weight: float,
    coordinate_weight: float,
    coordinate_radius: float,
    semantic_weight: float,
) -> dict[str, torch.Tensor]:
    base = F.normalize(base_tokens.detach().float(), dim=-1, eps=1e-6)
    expert = F.normalize(expert_tokens.detach().float(), dim=-1, eps=1e-6)
    feature = (
        1.0 - torch.bmm(base, expert.transpose(1, 2))
    ).clamp_min(0.0)
    h, w = grid
    coords = _coordinates(h, w, device=base.device, dtype=base.dtype)
    coordinate = (
        torch.cdist(coords, coords) - coordinate_radius
    ).clamp_min(0.0).square().unsqueeze(0)
    components = {
        "feature": feature_weight * feature,
        "coordinate": coordinate_weight * coordinate,
    }
    if semantic is not None and semantic_weight:
        pooled = F.adaptive_avg_pool2d(semantic.float(), grid)
        tokens = pooled.flatten(2).transpose(1, 2)
        tokens = F.normalize(tokens, p=1, dim=-1, eps=1e-6)
        sem = (
            1.0 - torch.bmm(tokens, tokens.transpose(1, 2))
        ).clamp_min(0.0)
        components["semantic"] = semantic_weight * sem
    return components


def _plot_representation(
    output_dir: Path,
    image: np.ndarray,
    gt: np.ndarray,
    label_names: dict[int, str],
    raw_maps: dict[int, dict[str, np.ndarray]],
    branch_maps: dict[int, dict[str, np.ndarray]],
    decoder_maps: dict[int, dict[str, np.ndarray]],
    scales_override: dict | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    if scales_override is None:
        raw_vmax = _robust_max(
            raw_maps[level][name]
            for level in LEVELS
            for name in ("expert", "student")
        )
        branch_vmax = _robust_max(
            branch_maps[level][name]
            for level in LEVELS
            for name in ("expert_p", "expert_s", "student_p", "student_s")
        )
        decoder_vmax = _robust_max(
            decoder_maps[level][name]
            for level in LEVELS
            for name in ("p", "s")
        )
    else:
        raw_vmax = float(scales_override["raw"]["vmax"])
        branch_vmax = float(scales_override["decomposed"]["vmax"])
        decoder_vmax = float(scales_override["pixel_decoder"]["vmax"])
    scales = {
        "energy_definition": "sqrt(mean(channel^2))",
        "percentile_clip": 99.5,
        "raw": {"vmin": 0.0, "vmax": raw_vmax},
        "decomposed": {"vmin": 0.0, "vmax": branch_vmax},
        "pixel_decoder": {"vmin": 0.0, "vmax": decoder_vmax},
        "share": {"vmin": 0.0, "vmax": 1.0},
    }
    if scales_override is None or "student_relative" not in scales_override:
        raise ValueError(
            "mechanism v3 requires per-level student_relative P1/P99 scales"
        )
    relative_maps: dict[int, np.ndarray] = {}
    scales["student_relative"] = {}
    for level in LEVELS:
        window = scales_override["student_relative"][f"res{level}"]
        p1 = float(window["p1"])
        p99 = float(window["p99"])
        if p99 <= p1:
            raise ValueError(f"Invalid Student relative scale for res{level}")
        relative_maps[level] = np.clip(
            (raw_maps[level]["student"] - p1) / (p99 - p1),
            0.0,
            1.0,
        )
        scales["student_relative"][f"res{level}"] = {
            "p1": p1,
            "p99": p99,
            "normalization": "clip((RMS - P1) / (P99 - P1), 0, 1)",
            "scope": "same level jointly across both selected cases",
        }
    _save_json(output_dir / "color_scales.json", scales)

    figure, axes = plt.subplots(
        4,
        4,
        figsize=(16, 16),
        constrained_layout=True,
    )
    last = None
    last_relative = None
    for row, level in enumerate(LEVELS):
        _draw_gt(
            axes[row, 0],
            image,
            gt,
            label_names,
            legend=row == 0,
        )
        axes[row, 0].set_title(f"res{level}: CT + GT", fontsize=10)
        last = _heat(
            axes[row, 1],
            raw_maps[level]["expert"],
            title="Expert raw RMS (absolute)",
            vmax=raw_vmax,
        )
        _heat(
            axes[row, 2],
            raw_maps[level]["student"],
            title="Student raw RMS (absolute)",
            vmax=raw_vmax,
        )
        last_relative = _heat(
            axes[row, 3],
            relative_maps[level],
            title=f"Student raw RMS (relative; res{level} P1–P99)",
            vmax=1.0,
            cmap="magma",
        )
    figure.colorbar(
        last,
        ax=axes[:, 1:3],
        shrink=0.72,
        label="Absolute channel-RMS energy",
    )
    figure.colorbar(
        last_relative,
        ax=axes[:, 3],
        shrink=0.72,
        label="Student within-level relative RMS [0, 1]",
    )
    figure.suptitle(
        "Backbone representations — shared absolute scale + "
        "Student within-level relative contrast",
        fontsize=15,
    )
    figure.savefig(output_dir / "01_backbone_raw_energy.png", dpi=170)
    plt.close(figure)

    figure, axes = plt.subplots(4, 6, figsize=(22, 15), constrained_layout=True)
    last_energy = None
    last_share = None
    for row, level in enumerate(LEVELS):
        names = (
            ("expert_p", "Expert P"),
            ("expert_s", "Expert S"),
            ("student_p", "Student P"),
            ("student_s", "Student S"),
        )
        for column, (name, title) in enumerate(names):
            last_energy = _heat(
                axes[row, column],
                branch_maps[level][name],
                title=f"res{level}: {title}" if column == 0 else title,
                vmax=branch_vmax,
            )
        last_share = _share(
            axes[row, 4],
            branch_maps[level]["expert_s_share"],
            title="Expert S share",
        )
        _share(
            axes[row, 5],
            branch_maps[level]["student_s_share"],
            title="Student S share",
        )
    figure.colorbar(
        last_energy,
        ax=axes[:, :4],
        shrink=0.68,
        label="Channel-RMS energy",
    )
    figure.colorbar(
        last_share,
        ax=axes[:, 4:],
        shrink=0.68,
        label="S / (P + S)",
    )
    figure.suptitle(
        "Decomposed expert/student representations — shared absolute scale",
        fontsize=15,
    )
    figure.savefig(output_dir / "02_disentangled_energy.png", dpi=170)
    plt.close(figure)

    figure, axes = plt.subplots(4, 4, figsize=(15, 15), constrained_layout=True)
    last_energy = None
    last_share = None
    for row, level in enumerate(LEVELS):
        _draw_gt(axes[row, 0], image, gt, label_names)
        axes[row, 0].set_title(DECODER_LABELS[level], fontsize=9)
        last_energy = _heat(
            axes[row, 1],
            decoder_maps[level]["p"],
            title="P decoder RMS",
            vmax=decoder_vmax,
        )
        _heat(
            axes[row, 2],
            decoder_maps[level]["s"],
            title="S decoder RMS",
            vmax=decoder_vmax,
        )
        last_share = _share(
            axes[row, 3],
            decoder_maps[level]["s_share"],
            title="Decoder S share",
        )
    figure.colorbar(
        last_energy,
        ax=axes[:, 1:3],
        shrink=0.70,
        label="Channel-RMS energy",
    )
    figure.colorbar(
        last_share,
        ax=axes[:, 3],
        shrink=0.70,
        label="S / (P + S)",
    )
    figure.suptitle("Pixel-decoder P/S representations", fontsize=15)
    figure.savefig(output_dir / "03_pixel_decoder_energy.png", dpi=170)
    plt.close(figure)

    np.savez_compressed(
        output_dir / "representation_maps.npz",
        gt=gt,
        **{
            f"raw_res{level}_{name}": value
            for level, level_maps in raw_maps.items()
            for name, value in level_maps.items()
        },
        **{
            f"raw_res{level}_student_relative": value
            for level, value in relative_maps.items()
        },
        **{
            f"branch_res{level}_{name}": value
            for level, level_maps in branch_maps.items()
            for name, value in level_maps.items()
        },
        **{
            f"decoder_res{level}_{name}": value
            for level, level_maps in decoder_maps.items()
            for name, value in level_maps.items()
        },
    )
    return scales


def _plot_token_layout(
    path: Path,
    rgb: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    title: str,
    received_visibility: np.ndarray | None = None,
    pca_info: dict | None = None,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    if received_visibility is None:
        labels = ("Expert tokens", "Transported teacher", "Student tokens")
        shown = rgb
    else:
        alpha = np.clip(received_visibility, 0.0, 1.0)[..., None]
        neutral = np.full_like(rgb[1], 0.72)
        shown = (
            rgb[0],
            rgb[1] * alpha + neutral * (1.0 - alpha),
            rgb[2] * alpha + neutral * (1.0 - alpha),
        )
        labels = (
            "Expert S tokens",
            "Barycentric teacher\nvisibility = received mass",
            "Student S tokens\nvisibility = received mass",
        )
    for axis, value, label in zip(axes, shown, labels):
        axis.imshow(value)
        axis.set_title(label)
        axis.axis("off")
    if received_visibility is not None:
        scalar = ScalarMappable(norm=Normalize(0.0, 1.0), cmap="Greys")
        scalar.set_array([])
        figure.colorbar(
            scalar,
            ax=axes[1:],
            shrink=0.62,
            label="Received-mass visibility",
        )
    if pca_info is not None:
        explained = pca_info["explained_variance_ratio"]
        title = (
            f"{title}\nweighted PCA explained variance: "
            f"{100 * explained[0]:.1f}% / {100 * explained[1]:.1f}% / "
            f"{100 * explained[2]:.1f}%"
        )
    figure.suptitle(title, fontsize=14)
    figure.savefig(path, dpi=190)
    plt.close(figure)


def _plot_p_ot(
    path: Path,
    *,
    level: int,
    image: np.ndarray,
    gt: np.ndarray,
    label_names: dict[int, str],
    grid: tuple[int, int],
    base_tokens: torch.Tensor,
    expert_tokens: torch.Tensor,
    transport: torch.Tensor,
    mass_a: torch.Tensor,
    mass_b: torch.Tensor,
    teacher_tokens: torch.Tensor,
    components: dict[str, torch.Tensor],
    total_cost: torch.Tensor,
) -> dict[str, np.ndarray]:
    student_energy = _token_image(_token_rms(base_tokens)[0], grid)
    expert_energy = _token_image(_token_rms(expert_tokens)[0], grid)
    teacher_energy = _token_image(_token_rms(teacher_tokens)[0], grid)
    student_norm = F.normalize(base_tokens.float(), dim=-1, eps=1e-8)
    teacher_norm = F.normalize(teacher_tokens.float(), dim=-1, eps=1e-8)
    residual = (
        1.0 - (student_norm * teacher_norm).sum(dim=-1)
    )[0].detach().cpu().numpy().reshape(grid)
    received = transport.sum(dim=-1)
    mass_a_map = _mass_image(mass_a, grid)
    mass_b_map = _mass_image(mass_b, grid)
    received_map = _mass_image(received, grid)
    aggregated = _aggregate_transport(transport, grid, grid)
    log_density = _transport_log_density(aggregated)
    log_conditional = _transport_log_conditional(aggregated)
    component_maps = {
        name: _token_image(_row_expected(transport, value), grid)
        for name, value in components.items()
    }
    component_maps["total"] = _token_image(
        _row_expected(transport, total_cost), grid
    )

    energy_vmax = _robust_max((student_energy, expert_energy, teacher_energy))
    mass_vmax = _robust_max((mass_a_map, mass_b_map, received_map))
    cost_vmax = _robust_max(component_maps.values())
    residual_vmax = _robust_max((residual,))

    figure, axes = plt.subplots(4, 4, figsize=(17, 16), constrained_layout=True)
    _draw_gt(axes[0, 0], image, gt, label_names, legend=True)
    axes[0, 0].set_title(f"res{level} P: CT + GT")
    energy_image = _heat(
        axes[0, 1],
        student_energy,
        title="Student P token RMS",
        vmax=energy_vmax,
    )
    _heat(
        axes[0, 2],
        expert_energy,
        title="Expert P token RMS",
        vmax=energy_vmax,
    )
    _heat(
        axes[0, 3],
        teacher_energy,
        title="Transported teacher RMS",
        vmax=energy_vmax,
    )

    mass_image = _heat(
        axes[1, 0], mass_a_map, title="Student demand a × N", vmax=mass_vmax
    )
    _heat(axes[1, 1], mass_b_map, title="Expert supply b × N", vmax=mass_vmax)
    _heat(
        axes[1, 2],
        received_map,
        title="Student received mass × N",
        vmax=mass_vmax,
    )
    residual_image = _heat(
        axes[1, 3],
        residual,
        title="1 − cos(student, teacher)",
        vmax=residual_vmax,
        cmap="inferno",
    )

    cost_image = None
    cost_names = ("feature", "coordinate", "semantic", "total")
    for column, name in enumerate(cost_names):
        value = component_maps.get(name, np.zeros(grid, dtype=np.float32))
        cost_image = _heat(
            axes[2, column],
            value,
            title=f"Expected {name} cost",
            vmax=cost_vmax,
            cmap="viridis",
        )

    matrix_image = axes[3, 0].imshow(
        log_density, cmap="magma", vmin=-3.0, vmax=2.0, aspect="auto"
    )
    axes[3, 0].set_title("Aggregated log₁₀ transport density")
    axes[3, 0].set_xlabel("Expert spatial token")
    axes[3, 0].set_ylabel("Student spatial token")
    axes[3, 1].imshow(
        log_conditional, cmap="coolwarm", vmin=-2.0, vmax=2.0, aspect="auto"
    )
    axes[3, 1].set_title("Row-conditional routing vs uniform")
    axes[3, 1].set_xlabel("Expert spatial token")
    axes[3, 1].set_ylabel("Student spatial token")
    _route_field(
        axes[3, 2],
        image,
        gt,
        label_names,
        transport,
        grid,
        grid,
        received,
        title="Top received routes: student → expert source",
    )
    axes[3, 3].axis("off")
    summary = (
        f"transport total = {transport.sum().item():.4f}\n"
        f"row L1 error = {(received - mass_a).abs().sum().item():.4e}\n"
        f"col L1 error = "
        f"{(transport.sum(dim=-2) - mass_b).abs().sum().item():.4e}\n"
        f"mean residual = {residual.mean():.4f}"
    )
    axes[3, 3].text(
        0.05,
        0.92,
        summary,
        va="top",
        family="monospace",
        fontsize=12,
        transform=axes[3, 3].transAxes,
    )
    figure.colorbar(energy_image, ax=axes[0, 1:], shrink=0.65, label="Token RMS")
    figure.colorbar(mass_image, ax=axes[1, :3], shrink=0.65, label="Uniform = 1")
    figure.colorbar(
        residual_image, ax=axes[1, 3], shrink=0.65, label="Cosine residual"
    )
    figure.colorbar(cost_image, ax=axes[2, :], shrink=0.65, label="Expected cost")
    figure.colorbar(
        matrix_image, ax=axes[3, :2], shrink=0.65, label="log₁₀ relative density"
    )
    figure.suptitle(
        f"res{level} balanced P-OT — mass, routing, and dynamic teacher",
        fontsize=15,
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return {
        "student_energy": student_energy,
        "expert_energy": expert_energy,
        "teacher_energy": teacher_energy,
        "residual": residual,
        "demand_density": mass_a_map,
        "supply_density": mass_b_map,
        "received_density": received_map,
        "transport_log_density": log_density,
        "transport_log_conditional": log_conditional,
        **{f"expected_cost_{name}": value for name, value in component_maps.items()},
    }


def _plot_s_ot(
    path: Path,
    *,
    level: int,
    image: np.ndarray,
    gt: np.ndarray,
    label_names: dict[int, str],
    grid: tuple[int, int],
    base_tokens: torch.Tensor,
    expert_tokens: torch.Tensor,
    transport: torch.Tensor,
    mass: dict[str, torch.Tensor],
    output: dict[str, torch.Tensor],
    teacher_tokens: torch.Tensor,
) -> dict[str, np.ndarray]:
    student_energy = _token_image(_token_rms(base_tokens)[0], grid)
    expert_energy = _token_image(_token_rms(expert_tokens)[0], grid)
    teacher_energy = _token_image(_token_rms(teacher_tokens)[0], grid)
    received_signal_tokens = torch.bmm(transport.float(), expert_tokens.float())
    received_signal_energy = _token_image(
        _token_rms(received_signal_tokens)[0], grid
    )
    student_norm = F.normalize(base_tokens.float(), dim=-1, eps=1e-8)
    teacher_norm = F.normalize(teacher_tokens.float(), dim=-1, eps=1e-8)
    residual = (
        1.0 - (student_norm * teacher_norm).sum(dim=-1)
    )[0].detach().cpu().numpy().reshape(grid)

    a_map = _mass_image(mass["a"], grid)
    b_map = _mass_image(mass["b"], grid)
    received_map = _mass_image(output["received"], grid)
    transported_map = _mass_image(output["transported"], grid)
    rejected_map = _mass_image(output["rejected"], grid)
    rejection_ratio = (
        output["rejected"] / mass["b"].clamp_min(1e-8)
    )[0].clamp(0.0, 1.0).detach().cpu().numpy().reshape(grid)
    difficulty = mass["difficulty"][0].detach().cpu().numpy().reshape(grid)
    gain = mass["gain"][0].detach().cpu().numpy().reshape(grid)
    gain_mode = str(mass.get("gain_mode", "hard_positive"))
    base_specificity = (
        mass["base_specificity"][0].detach().cpu().numpy().reshape(grid)
    )
    expert_specificity = (
        mass["expert_specificity"][0].detach().cpu().numpy().reshape(grid)
    )
    aggregated = _aggregate_transport(transport, grid, grid)
    log_density = _transport_log_density(aggregated)
    log_conditional = _transport_log_conditional(aggregated)

    energy_vmax = _robust_max((student_energy, expert_energy))
    received_signal_vmax = _robust_max((received_signal_energy,))
    mass_vmax = _robust_max(
        (a_map, b_map, received_map, transported_map, rejected_map)
    )
    difficulty_vmax = _robust_max((difficulty,))
    task_vmax = (
        _robust_max((difficulty, gain))
        if gain_mode == "hard_positive"
        else difficulty_vmax
    )
    specificity_vmax = _robust_max((base_specificity, expert_specificity))
    residual_vmax = _robust_max((residual,))

    figure, axes = plt.subplots(4, 4, figsize=(17, 16), constrained_layout=True)
    _draw_gt(axes[0, 0], image, gt, label_names, legend=True)
    axes[0, 0].set_title(f"res{level} S: CT + GT")
    task_image = _heat(
        axes[0, 1], difficulty, title="Student difficulty", vmax=task_vmax
    )
    if gain_mode == "smooth_advantage":
        gain_image = _heat(
            axes[0, 2],
            gain,
            title="Smooth expert advantage (1 = neutral)",
            vmin=0.0,
            vmax=2.0,
            cmap="coolwarm",
        )
    else:
        gain_image = _heat(
            axes[0, 2], gain, title="Positive expert gain", vmax=task_vmax
        )
    specificity_image = _heat(
        axes[0, 3],
        base_specificity,
        title="Student S specificity",
        vmax=specificity_vmax,
        cmap="viridis",
    )

    mass_image = _heat(
        axes[1, 0], a_map, title="Student demand a × N", vmax=mass_vmax
    )
    _heat(axes[1, 1], b_map, title="Expert supply b × N", vmax=mass_vmax)
    _heat(
        axes[1, 2],
        transported_map,
        title="Expert transported × N",
        vmax=mass_vmax,
    )
    rejection_image = _share(
        axes[1, 3], rejection_ratio, title="Expert rejection ratio"
    )

    energy_image = _heat(
        axes[2, 0],
        student_energy,
        title="Student S token RMS",
        vmax=energy_vmax,
    )
    _heat(
        axes[2, 1],
        expert_energy,
        title="Expert S token RMS",
        vmax=energy_vmax,
    )
    third_energy_image = _heat(
        axes[2, 2],
        received_signal_energy,
        title="Received expert signal RMS",
        vmax=received_signal_vmax,
    )
    residual_image = _heat(
        axes[2, 3],
        residual,
        title="1 − cos(student, teacher)",
        vmax=residual_vmax,
        cmap="inferno",
    )

    matrix_image = axes[3, 0].imshow(
        log_density, cmap="magma", vmin=-3.0, vmax=2.0, aspect="auto"
    )
    axes[3, 0].set_title("Aggregated log₁₀ transport density")
    axes[3, 0].set_xlabel("Expert spatial token")
    axes[3, 0].set_ylabel("Student spatial token")
    axes[3, 1].imshow(
        log_conditional, cmap="coolwarm", vmin=-2.0, vmax=2.0, aspect="auto"
    )
    axes[3, 1].set_title("Row-conditional routing vs uniform")
    axes[3, 1].set_xlabel("Expert spatial token")
    axes[3, 1].set_ylabel("Student spatial token")
    _route_field(
        axes[3, 2],
        image,
        gt,
        label_names,
        transport,
        grid,
        grid,
        output["received"],
        title="Top received routes: student → expert source",
    )
    axes[3, 3].axis("off")
    summary = (
        f"transport total = {transport.sum().item():.4f}\n"
        f"accept ratio = {output['accept_ratio'].mean().item():.4f}\n"
        f"rejected total = {output['rejected'].sum().item():.4f}\n"
        f"received total = {output['received'].sum().item():.4f}\n"
        f"mean residual = {residual.mean():.4f}"
    )
    axes[3, 3].text(
        0.05,
        0.92,
        summary,
        va="top",
        family="monospace",
        fontsize=12,
        transform=axes[3, 3].transAxes,
    )
    if gain_mode == "smooth_advantage":
        figure.colorbar(
            task_image,
            ax=axes[0, 1],
            shrink=0.65,
            label="Prompt-mean BCE",
        )
        figure.colorbar(
            gain_image,
            ax=axes[0, 2],
            shrink=0.65,
            label="<1 expert worse | >1 expert better",
        )
    else:
        figure.colorbar(
            task_image,
            ax=axes[0, 1:3],
            shrink=0.65,
            label="BCE-derived score",
        )
    figure.colorbar(
        specificity_image,
        ax=axes[0, 3],
        shrink=0.65,
        label="S / (P + S)",
    )
    figure.colorbar(mass_image, ax=axes[1, :3], shrink=0.65, label="Uniform = 1")
    figure.colorbar(
        rejection_image, ax=axes[1, 3], shrink=0.65, label="Rejected / supply"
    )
    figure.colorbar(
        energy_image,
        ax=axes[2, :2],
        shrink=0.65,
        label="Student / expert token RMS",
    )
    figure.colorbar(
        third_energy_image,
        ax=axes[2, 2],
        shrink=0.65,
        label="RMS of Uᵢ = Σⱼ πᵢⱼEⱼ",
    )
    figure.colorbar(
        residual_image, ax=axes[2, 3], shrink=0.65, label="Cosine residual"
    )
    figure.colorbar(
        matrix_image, ax=axes[3, :2], shrink=0.65, label="log₁₀ relative density"
    )
    figure.suptitle(
        f"res{level} unbalanced S-OT — demand, rejection, routing, and teacher",
        fontsize=15,
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return {
        "student_energy": student_energy,
        "expert_energy": expert_energy,
        "received_signal_energy": received_signal_energy,
        "barycentric_teacher_energy": teacher_energy,
        "residual": residual,
        "difficulty": difficulty,
        "gain": gain,
        "student_specificity": base_specificity,
        "expert_specificity": expert_specificity,
        "demand_density": a_map,
        "supply_density": b_map,
        "received_density": received_map,
        "transported_density": transported_map,
        "rejected_density": rejected_map,
        "rejection_ratio": rejection_ratio,
        "transport_log_density": log_density,
        "transport_log_conditional": log_conditional,
    }


def _plot_decision(
    output_dir: Path,
    *,
    image: np.ndarray,
    gt: np.ndarray,
    label_names: dict[int, str],
    decoder_maps: dict[int, dict[str, np.ndarray]],
    decoder_features: dict[int, dict[str, torch.Tensor]],
    route: dict[str, torch.Tensor],
    class_ids: Sequence[int],
    logits: torch.Tensor,
    z_index: int,
    output_hw: tuple[int, int],
    decoder_vmax: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    mean = route["mean"].detach().cpu().numpy()
    concentration = route["concentration"].detach().cpu().numpy()
    uncertainty = (
        route["alpha"]
        * route["beta"]
        / (
            (route["alpha"] + route["beta"]).square()
            * (route["alpha"] + route["beta"] + 1.0)
        )
    ).sqrt().detach().cpu().numpy()

    figure, axes = plt.subplots(4, 4, figsize=(15, 15), constrained_layout=True)
    last_energy = None
    last_share = None
    for row, level in enumerate(LEVELS):
        _draw_gt(axes[row, 0], image, gt, label_names)
        axes[row, 0].set_title(DECODER_LABELS[level], fontsize=9)
        last_energy = _heat(
            axes[row, 1],
            decoder_maps[level]["p"],
            title="P decoder RMS",
            vmax=decoder_vmax,
        )
        _heat(
            axes[row, 2],
            decoder_maps[level]["s"],
            title="S decoder RMS",
            vmax=decoder_vmax,
        )
        last_share = _share(
            axes[row, 3],
            decoder_maps[level]["s_share"],
            title="S share",
        )
    figure.colorbar(
        last_energy, ax=axes[:, 1:3], shrink=0.70, label="Channel-RMS energy"
    )
    figure.colorbar(last_share, ax=axes[:, 3], shrink=0.70, label="S / (P + S)")
    figure.suptitle("Decision input: pixel-decoder P/S branches", fontsize=15)
    figure.savefig(output_dir / "01_decoder_branches.png", dpi=180)
    plt.close(figure)

    class_names = [label_names.get(cid, str(cid)) for cid in class_ids]
    figure, axes = plt.subplots(
        len(class_names),
        3,
        figsize=(12, 3.6 * len(class_names)),
        constrained_layout=True,
        squeeze=False,
    )
    for row, class_name in enumerate(class_names):
        panels = (
            (mean[row], "P gate mean", 0.0, 1.0, "viridis"),
            (
                concentration[row],
                "Beta concentration",
                None,
                None,
                "magma",
            ),
            (
                uncertainty[row],
                "Gate standard deviation",
                None,
                None,
                "magma",
            ),
        )
        for column, (value, title, vmin, vmax, cmap) in enumerate(panels):
            plot = axes[row, column].imshow(
                value,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
            axes[row, column].set_title(f"{class_name}: {title}", fontsize=9)
            axes[row, column].axis("off")
            figure.colorbar(plot, ax=axes[row, column], shrink=0.72)
    figure.suptitle(
        "Prompt-conditioned finest spatial Beta router", fontsize=15
    )
    figure.savefig(output_dir / "02_router_heatmaps.png", dpi=190)
    plt.close(figure)

    fused_maps: dict[int, dict[int, np.ndarray]] = {}
    fixed_maps: dict[int, dict[int, np.ndarray]] = {}
    delta_maps: list[np.ndarray] = []
    for prompt_index, class_id in enumerate(class_ids):
        fused_maps[class_id] = {}
        fixed_maps[class_id] = {}
        for level in LEVELS:
            p = decoder_features[level]["p"]
            s = decoder_features[level]["s"]
            gate_value = F.interpolate(
                route["mean"][prompt_index][None, None],
                size=p.shape[-2:],
                mode="area",
            )
            fused = gate_value * p + (1.0 - gate_value) * s
            fixed = 0.5 * p + 0.5 * s
            fused_map = _rms_4d(fused, z_index, output_hw)
            fixed_map = _rms_4d(fixed, z_index, output_hw)
            fused_maps[class_id][level] = fused_map
            fixed_maps[class_id][level] = fixed_map
            delta_maps.append(fused_map - fixed_map)
    delta_limit = _symmetric_limit(delta_maps)

    figure, axes = plt.subplots(
        len(class_ids),
        5,
        figsize=(17, 3.1 * len(class_ids)),
        constrained_layout=True,
    )
    last = None
    for row, class_id in enumerate(class_ids):
        _draw_gt(
            axes[row, 0],
            image,
            gt,
            label_names,
            class_id=class_id,
        )
        axes[row, 0].set_title(
            f"{label_names.get(class_id, class_id)} GT"
            if row == 0
            else label_names.get(class_id, str(class_id))
        )
        for column, level in enumerate(LEVELS, start=1):
            last = _heat(
                axes[row, column],
                fused_maps[class_id][level],
                title=f"res{level}" if row == 0 else f"P gate={mean[row, column-1]:.3f}",
                vmax=decoder_vmax,
            )
    figure.colorbar(
        last,
        ax=axes[:, 1:],
        shrink=0.60,
        label="Fused channel-RMS energy",
    )
    figure.suptitle(
        "All prompt-gated fused representations — one shared absolute scale",
        fontsize=15,
    )
    figure.savefig(output_dir / "03_all_class_fusion.png", dpi=180)
    plt.close(figure)

    per_class_dir = output_dir / "per_class"
    per_class_dir.mkdir(parents=True, exist_ok=True)
    probabilities = torch.sigmoid(logits[0, :, z_index]).detach().cpu().numpy()
    saved = {}
    for prompt_index, class_id in enumerate(class_ids):
        class_name = label_names.get(class_id, f"class_{class_id}")
        safe_name = "".join(
            character if character.isalnum() else "_"
            for character in class_name
        ).strip("_")
        figure, axes = plt.subplots(4, 5, figsize=(18, 15), constrained_layout=True)
        delta_image = None
        energy_image = None
        for row, level in enumerate(LEVELS):
            _draw_gt(
                axes[row, 0],
                image,
                gt,
                label_names,
                class_id=class_id,
            )
            axes[row, 0].set_title(DECODER_LABELS[level], fontsize=9)
            energy_image = _heat(
                axes[row, 1],
                decoder_maps[level]["p"],
                title="P RMS",
                vmax=decoder_vmax,
            )
            _heat(
                axes[row, 2],
                decoder_maps[level]["s"],
                title="S RMS",
                vmax=decoder_vmax,
            )
            _heat(
                axes[row, 3],
                fused_maps[class_id][level],
                title=f"Fused, P gate={mean[prompt_index, row]:.3f}",
                vmax=decoder_vmax,
            )
            delta = fused_maps[class_id][level] - fixed_maps[class_id][level]
            norm = TwoSlopeNorm(vmin=-delta_limit, vcenter=0.0, vmax=delta_limit)
            delta_image = axes[row, 4].imshow(delta, cmap="coolwarm", norm=norm)
            axes[row, 4].set_title("ΔE vs fixed gate=0.5", fontsize=10)
            axes[row, 4].axis("off")
        figure.colorbar(
            energy_image,
            ax=axes[:, 1:4],
            shrink=0.68,
            label="Channel-RMS energy",
        )
        figure.colorbar(
            delta_image,
            ax=axes[:, 4],
            shrink=0.68,
            label="Fused RMS difference",
        )
        figure.suptitle(
            f"{class_name}: P/S decoder branches and prompt-gated fusion",
            fontsize=15,
        )
        detail_path = per_class_dir / f"class{class_id:02d}_{safe_name}_fusion.png"
        figure.savefig(detail_path, dpi=180)
        plt.close(figure)

        probability = probabilities[prompt_index]
        prediction = probability >= 0.5
        target = gt == class_id
        figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
        _draw_gt(
            axes[0], image, gt, label_names, class_id=class_id
        )
        axes[0].set_title(f"GT: {class_name}")
        prob_image = axes[1].imshow(probability, cmap="viridis", vmin=0.0, vmax=1.0)
        axes[1].set_title("Final class probability")
        axes[1].axis("off")
        lo, hi = _ct_limits(image)
        axes[2].imshow(image, cmap="gray", vmin=lo, vmax=hi)
        if target.any():
            axes[2].contour(target, levels=[0.5], colors=["#00ff3b"], linewidths=1.5)
        if prediction.any():
            axes[2].contour(
                prediction, levels=[0.5], colors=["#ff2d2d"], linewidths=1.1
            )
        axes[2].set_title("GT green / prediction red")
        axes[2].axis("off")
        figure.colorbar(prob_image, ax=axes[1], shrink=0.75)
        output_path = per_class_dir / f"class{class_id:02d}_{safe_name}_output.png"
        figure.savefig(output_path, dpi=190)
        plt.close(figure)
        saved[f"class{class_id:02d}_{safe_name}"] = {
            "fusion": str(detail_path),
            "output": str(output_path),
        }

    np.savez_compressed(
        output_dir / "decision_maps.npz",
        gt=gt,
        gate_mean=mean,
        gate_concentration=concentration,
        gate_uncertainty=uncertainty,
        probabilities=probabilities,
        **{
            f"class{class_id:02d}_res{level}_fused": fused_maps[class_id][level]
            for class_id in class_ids
            for level in LEVELS
        },
        **{
            f"class{class_id:02d}_res{level}_delta_fixed05": (
                fused_maps[class_id][level] - fixed_maps[class_id][level]
            )
            for class_id in class_ids
            for level in LEVELS
        },
    )
    _save_json(
        output_dir / "decision_summary.json",
        {
            "scale_order": [f"res{level}" for level in LEVELS],
            "class_ids": list(class_ids),
            "class_names": class_names,
            "gate_mean": mean.tolist(),
            "gate_concentration": concentration.tolist(),
            "gate_uncertainty": uncertainty.tolist(),
            "decoder_shared_vmax": decoder_vmax,
            "router_effect_symmetric_limit": delta_limit,
            "per_class_files": saved,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--case_id", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--slice_index", type=int, default=None)
    parser.add_argument(
        "--selection",
        choices=("largest_foreground", "all_classes"),
        default="largest_foreground",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--block_z", type=int, default=6)
    parser.add_argument(
        "--shared_color_scales",
        required=True,
        help=(
            "JSON with fixed raw/decomposed/pixel_decoder vmax values and "
            "per-level Student relative P1/P99"
        ),
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Root under which <case>_z<slice> is created",
    )
    args = parser.parse_args()

    cfg = TrainConfig(device=args.device, block_z=args.block_z, num_workers=0)
    cfg.resolve_paths()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    checkpoint_cfg = checkpoint.get("config", {})
    coordinate_weight = float(
        checkpoint_cfg.get("ot_coordinate_weight", cfg.ot_coordinate_weight)
    )
    coordinate_radius = float(
        checkpoint_cfg.get("ot_coordinate_radius", 0.0)
    )
    s_gain_mode = str(
        checkpoint_cfg.get("s_gain_mode", "hard_positive")
    )
    s_gain_temperature = float(
        checkpoint_cfg.get("s_gain_temperature", cfg.s_gain_temperature)
    )
    remove_res5_expert_branch_norm = bool(
        checkpoint_cfg.get("remove_res5_expert_branch_norm", False)
    )
    images_dir = cfg.imagesTr_dir if args.split == "val" else cfg.imagesTs_dir
    labels_dir = cfg.labelsTr_dir if args.split == "val" else cfg.labelsTs_dir
    with open(cfg.dataset_json_path, encoding="utf-8") as handle:
        dataset_json = json.load(handle)
    ending = dataset_json.get("file_ending", ".nii.gz")
    label_names = {
        int(class_id): str(name)
        for name, class_id in dataset_json.get("labels", {}).items()
        if int(class_id) > 0
    }
    image_files = find_raw_image_files(images_dir, args.case_id, ending)
    if not image_files:
        raise FileNotFoundError(args.case_id)
    label_path = os.path.join(labels_dir, args.case_id + ending)
    raw = np.asarray(sitk.GetArrayFromImage(sitk.ReadImage(image_files[0])))
    gt_volume = np.asarray(sitk.GetArrayFromImage(sitk.ReadImage(label_path)))
    center = _select_slice(gt_volume, args.selection, args.slice_index)

    dataset = FullSliceBlockDataset(
        [args.case_id],
        nnunet_preproc_dir=cfg.nnunet_preproc_dir,
        images_dir=images_dir,
        labels_dir=labels_dir,
        file_ending=ending,
        image_size=cfg.image_size,
        block_z=cfg.block_z,
        norm_mode=cfg.norm_mode,
        window_level=cfg.window_level,
        window_width=cfg.window_width,
        low_percentile=cfg.low_percentile,
        high_percentile=cfg.high_percentile,
        raw_cache_cases=1,
        require_no_crop=cfg.require_no_crop,
        biomedparse_modality=cfg.biomedparse_modality,
    )
    record_index = next(
        index
        for index, (_case_id, z_start, valid_count) in enumerate(dataset.records)
        if z_start <= center < z_start + valid_count
    )
    item = dataset[record_index]
    center_local = center - int(item["z_start"])
    bp = item["biomedparse_image"].unsqueeze(0)
    nn_input = (
        item["nnunet_image"]
        .unsqueeze(0)
        .permute(0, 2, 1, 3, 4)
        .contiguous()
    )
    gt_block = item["gt"].unsqueeze(0)
    valid_z = item["valid_z"].unsqueeze(0)

    model_nnunet, model_biomedparse = load_frozen_backbones(
        cfg.nnunet_model_dir, cfg.fold, device
    )
    prompts, prompt_to_class_id = build_text_prompts_for_dataset(
        dataset_name=cfg.dataset_name
    )
    prompt_features = build_prompt_features(model_biomedparse, prompts, device)
    modules = build_fusion_modules(
        model_nnunet,
        model_biomedparse,
        len(prompts),
        device,
        text_dim=int(prompt_features["class_emb"].shape[-1]),
        route_prior_p_mean=cfg.route_prior_p_mean,
        route_prior_concentration=cfg.route_prior_concentration,
        route_spatial_basis_grid_size=cfg.route_spatial_basis_grid_size,
        route_spatial_basis_sigma=cfg.route_spatial_basis_sigma,
        ot_feature_weight=cfg.ot_feature_weight,
        ot_coordinate_weight=coordinate_weight,
        ot_coordinate_radius=coordinate_radius,
        p_ot_semantic_weight=cfg.p_ot_semantic_weight,
        s_gain_mode=s_gain_mode,
        s_gain_temperature=s_gain_temperature,
        p_ot_epsilon=cfg.p_ot_epsilon,
        s_ot_epsilon=cfg.s_ot_epsilon,
        s_ot_rho_base=cfg.s_ot_rho_base,
        s_ot_rho_expert=cfg.s_ot_rho_expert,
        ot_sinkhorn_iterations=cfg.ot_sinkhorn_iterations,
        ot_max_grid_size=cfg.ot_max_grid_size,
        remove_res5_expert_branch_norm=remove_res5_expert_branch_norm,
    )
    required = [
        *(f"ae_enc{level}_to_res{level}" for level in LEVELS),
        *(f"dis_b_res{level}" for level in LEVELS),
        "beta_router",
    ]
    missing = [name for name in required if name not in checkpoint]
    if missing:
        raise KeyError(f"Full fusion checkpoint is missing {missing}")
    for name, module in modules.items():
        if name in checkpoint:
            module.load_state_dict(checkpoint[name])
        module.eval()

    output_hw = tuple(int(value) for value in gt_volume.shape[-2:])
    image = raw[center]
    gt = gt_volume[center]
    raw_maps: dict[int, dict[str, np.ndarray]] = {}
    branch_maps: dict[int, dict[str, np.ndarray]] = {}
    decoder_maps: dict[int, dict[str, np.ndarray]] = {}
    features: dict[str, torch.Tensor] = {}

    with torch.no_grad():
        expert_raw, _deepest, expert_logits = extract_nnunet_features(
            model_nnunet,
            nn_input.to(device),
            device,
            return_logits=True,
        )
        embeds, student_raw = extract_biomedparse_backbone_features_2p5d(
            model_biomedparse,
            bp.to(device),
            device,
            res_names=("res2", "res3", "res4", "res5"),
        )
        for level in LEVELS:
            student = student_raw[f"res{level}"]
            expert_native = expert_raw[f"enc{level}"]
            expert_aligned = expert_native
            if expert_aligned.shape[-3:] != student.shape[-3:]:
                expert_aligned = F.interpolate(
                    expert_aligned,
                    size=student.shape[-3:],
                    mode="trilinear",
                    align_corners=False,
                )
            expert_p, expert_s, _p_rec, _s_rec = modules[
                f"ae_enc{level}_to_res{level}"
            ](expert_aligned)
            student_p, student_s = modules[f"dis_b_res{level}"](student)
            features[f"Zn{level}_p"] = expert_p
            features[f"Zn{level}_s"] = expert_s
            features[f"Zb{level}_p"] = student_p
            features[f"Zb{level}_s"] = student_s

            raw_maps[level] = {
                "expert": _rms_5d(expert_native, center_local, output_hw),
                "student": _rms_5d(student, center_local, output_hw),
            }
            level_branch = {
                "expert_p": _rms_5d(expert_p, center_local, output_hw),
                "expert_s": _rms_5d(expert_s, center_local, output_hw),
                "student_p": _rms_5d(student_p, center_local, output_hw),
                "student_s": _rms_5d(student_s, center_local, output_hw),
            }
            level_branch["expert_s_share"] = level_branch["expert_s"] / (
                level_branch["expert_p"] + level_branch["expert_s"] + 1e-8
            )
            level_branch["student_s_share"] = level_branch["student_s"] / (
                level_branch["student_p"] + level_branch["student_s"] + 1e-8
            )
            branch_maps[level] = level_branch

        embeds_base = dict(embeds)
        for level in LEVELS:
            embeds_base.pop(f"res{level}", None)
        mask_p, multi_p = _run_pixel_decoder(
            model_biomedparse,
            embeds_base,
            features,
            "p",
            B=1,
            Dm=bp.shape[1],
        )
        mask_s, multi_s = _run_pixel_decoder(
            model_biomedparse,
            embeds_base,
            features,
            "s",
            B=1,
            Dm=bp.shape[1],
        )
        decoder_features = {
            2: {"p": mask_p, "s": mask_s},
            3: {"p": multi_p[2], "s": multi_s[2]},
            4: {"p": multi_p[1], "s": multi_s[1]},
            5: {"p": multi_p[0], "s": multi_s[0]},
        }
        for level in LEVELS:
            p_map = _rms_4d(
                decoder_features[level]["p"], center_local, output_hw
            )
            s_map = _rms_4d(
                decoder_features[level]["s"], center_local, output_hw
            )
            decoder_maps[level] = {
                "p": p_map,
                "s": s_map,
                "s_share": s_map / (p_map + s_map + 1e-8),
            }
        route = modules["beta_router"](
            prompt_features["class_emb"].detach(),
            spatial_size=mask_p.shape[-2:],
            batch_size=1,
            sample=False,
        )
        all_prompt_logits = _predict_all_prompt_logits(
            sem_seg_head=model_biomedparse.sem_seg_head,
            mask_features_p=mask_p,
            mask_features_s=mask_s,
            ms_p=multi_p,
            ms_s=multi_s,
            gate=route["gate"],
            prompt_features=prompt_features,
            B=1,
            Z=bp.shape[1],
            P=len(prompts),
            output_shape=(bp.shape[1], *output_hw),
        )
        class_ids_tensor = torch.tensor(
            [
                prompt_to_class_id[prompt_index]
                for prompt_index in range(len(prompts))
            ],
            device=device,
            dtype=gt_block.dtype,
        )
        base_error, expert_error = _compute_detached_pixel_error_maps(
            all_prompt_logits,
            expert_logits,
            gt_block.to(device),
            valid_z.to(device),
            class_ids_tensor,
        )

    case_root = (
        Path(args.output_root).resolve()
        / f"{args.case_id}_z{center:04d}"
    )
    representation_dir = case_root / "representation"
    ot_dir = case_root / "ot"
    decision_dir = case_root / "decision"
    shared_scales_path = (
        Path(args.shared_color_scales).resolve()
        if args.shared_color_scales is not None
        else None
    )
    shared_scales = None
    if shared_scales_path is not None:
        with shared_scales_path.open(encoding="utf-8") as handle:
            shared_scales = json.load(handle)
    scales = _plot_representation(
        representation_dir,
        image,
        gt,
        label_names,
        raw_maps,
        branch_maps,
        decoder_maps,
        scales_override=shared_scales,
    )

    class_ids = [int(value) for value in class_ids_tensor.tolist()]
    gt_slice = gt_block[:, center_local].to(device)
    semantic = torch.stack(
        [(gt_slice == class_id).float() for class_id in class_ids], dim=1
    )
    ot_module = modules["ot_distillation"]
    ot_dir.mkdir(parents=True, exist_ok=True)
    ot_summary = {
        "case_id": args.case_id,
        "z": center,
        "energy_definition": "sqrt(mean(channel^2))",
        "transport_matrix_display": (
            "block-summed to at most 16x16 spatial grids; matrix values shown "
            "relative to a uniform plan"
        ),
        "levels": {},
    }
    ot_npz: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for level in LEVELS:
            p_base = _flatten_feature_slice(features[f"Zb{level}_p"], center_local)
            s_base = _flatten_feature_slice(features[f"Zb{level}_s"], center_local)
            p_expert = _flatten_feature_slice(
                features[f"Zn{level}_p"], center_local
            )
            s_expert = _flatten_feature_slice(
                features[f"Zn{level}_s"], center_local
            )
            grid = ot_module._target_size(p_base)

            p_mass = ot_module.structure_mass(
                gt_slice,
                p_base,
                p_expert,
                class_ids=class_ids,
                target_size=grid,
            )
            p_cost = ot_module.p_cost(
                p_base,
                p_expert,
                target_size=grid,
                base_semantic=semantic,
                expert_semantic=semantic,
            )
            p_transport = ot_module.balanced(
                p_mass["a"], p_mass["b"], p_cost["cost"]
            )
            p_teacher = ot_module.projector(
                p_transport["transport"], p_cost["expert_tokens"]
            )
            p_components = _feature_cost_components(
                p_cost["base_tokens"],
                p_cost["expert_tokens"],
                grid,
                semantic,
                feature_weight=cfg.ot_feature_weight,
                coordinate_weight=coordinate_weight,
                coordinate_radius=coordinate_radius,
                semantic_weight=cfg.p_ot_semantic_weight,
            )
            p_maps = _plot_p_ot(
                ot_dir / f"res{level}_p_transport.png",
                level=level,
                image=image,
                gt=gt,
                label_names=label_names,
                grid=grid,
                base_tokens=p_cost["base_tokens"],
                expert_tokens=p_cost["expert_tokens"],
                transport=p_transport["transport"],
                mass_a=p_mass["a"],
                mass_b=p_mass["b"],
                teacher_tokens=p_teacher["teacher"],
                components=p_components,
                total_cost=p_cost["cost"],
            )
            p_rgb = _shared_pca_rgb(
                p_cost["expert_tokens"],
                p_teacher["teacher"],
                p_cost["base_tokens"],
                grid,
            )
            _plot_token_layout(
                ot_dir / f"res{level}_p_token_layout.png",
                p_rgb,
                title=f"res{level} P: shared PCA-RGB token arrangement",
            )

            s_mass = ot_module.residual_mass(
                p_base,
                s_base,
                p_expert,
                s_expert,
                base_error=base_error[:, center_local],
                expert_error=expert_error[:, center_local],
                target_size=grid,
            )
            s_cost = ot_module.s_cost(
                s_base,
                s_expert,
                target_size=grid,
            )
            s_transport = ot_module.unbalanced(
                s_mass["a"], s_mass["b"], s_cost["cost"]
            )
            s_teacher = ot_module.projector(
                s_transport["transport"], s_cost["expert_tokens"]
            )
            s_maps = _plot_s_ot(
                ot_dir / f"res{level}_s_transport.png",
                level=level,
                image=image,
                gt=gt,
                label_names=label_names,
                grid=grid,
                base_tokens=s_cost["base_tokens"],
                expert_tokens=s_cost["expert_tokens"],
                transport=s_transport["transport"],
                mass=s_mass,
                output=s_transport,
                teacher_tokens=s_teacher["teacher"],
            )
            s_rgb, s_visibility, s_pca_info = _received_weighted_pca_rgb(
                s_cost["expert_tokens"],
                s_teacher["teacher"],
                s_cost["base_tokens"],
                s_transport["received"],
                grid,
            )
            _plot_token_layout(
                ot_dir / f"res{level}_s_token_layout.png",
                s_rgb,
                title=f"res{level} S: shared PCA-RGB token arrangement",
                received_visibility=s_visibility,
                pca_info=s_pca_info,
            )
            ot_summary["levels"][f"res{level}"] = {
                "grid": list(grid),
                "p": {
                    "transport_total": float(
                        p_transport["transport"].sum().item()
                    ),
                    "row_l1_error": float(
                        p_transport["row_error"].mean().item()
                    ),
                    "col_l1_error": float(
                        p_transport["col_error"].mean().item()
                    ),
                    "transport_cost": float(
                        p_transport["cost"].mean().item()
                    ),
                    "mean_student_teacher_residual": float(
                        p_maps["residual"].mean()
                    ),
                },
                "s": {
                    "transport_total": float(
                        s_transport["transport"].sum().item()
                    ),
                    "accept_ratio": float(
                        s_transport["accept_ratio"].mean().item()
                    ),
                    "rejected_total": float(
                        s_transport["rejected"].sum().item()
                    ),
                    "transport_cost": float(
                        s_transport["cost"].mean().item()
                    ),
                    "mean_student_teacher_residual": float(
                        s_maps["residual"].mean()
                    ),
                    "received_signal_definition": "U_i = sum_j pi_ij E_j",
                    "token_layout": s_pca_info,
                },
            }
            for name, value in p_maps.items():
                ot_npz[f"res{level}_p_{name}"] = value
            for name, value in s_maps.items():
                ot_npz[f"res{level}_s_{name}"] = value
    _save_json(ot_dir / "ot_summary.json", ot_summary)
    np.savez_compressed(ot_dir / "ot_derived_maps.npz", gt=gt, **ot_npz)

    _plot_decision(
        decision_dir,
        image=image,
        gt=gt,
        label_names=label_names,
        decoder_maps=decoder_maps,
        decoder_features=decoder_features,
        route=route,
        class_ids=class_ids,
        logits=all_prompt_logits,
        z_index=center_local,
        output_hw=output_hw,
        decoder_vmax=scales["pixel_decoder"]["vmax"],
    )

    checkpoint_path = Path(args.checkpoint).resolve()
    manifest = {
        "schema": "oodka-mechanism-visualization-v3",
        "case_id": args.case_id,
        "split": args.split,
        "z": center,
        "selection": args.selection,
        "shared_color_scales": (
            str(shared_scales_path) if shared_scales_path is not None else None
        ),
        "labels_present": sorted(
            int(value) for value in np.unique(gt) if int(value) > 0
        ),
        "git_commit": _git_commit(),
        "generator": str(Path(__file__).resolve()),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "block_z": args.block_z,
        "block_z_start": int(item["z_start"]),
        "center_local": center_local,
        "color_scales": scales,
        "refinements": [
            "Student raw RMS relative contrast with per-level cross-case P1/P99",
            "S-UOT displays true received expert signal U_i = sum_j pi_ij E_j",
            "S token layout uses received-mass-weighted PCA and visibility",
        ],
        "directories": {
            "representation": str(representation_dir),
            "ot": str(ot_dir),
            "decision": str(decision_dir),
        },
    }
    _save_json(case_root / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    print(f"saved={case_root}")


if __name__ == "__main__":
    main()
