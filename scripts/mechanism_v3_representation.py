"""Representation and token-layout visualizations for mechanism-v3."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/oodka_mechanism_v3_mpl")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
import torch
import torch.nn.functional as F

from oodka.models.ot.cost import _coordinates
from scripts.mechanism_v3_common import (
    DECODER_LABELS,
    LEVELS,
    _draw_gt,
    _heat,
    _robust_max,
    _save_json,
    _share,
)


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
    relative_maps: dict[int, np.ndarray] = {}
    scales["student_relative"] = {}
    for level in LEVELS:
        if scales_override is not None and "student_relative" in scales_override:
            window = scales_override["student_relative"][f"res{level}"]
            p1 = float(window["p1"])
            p99 = float(window["p99"])
            scope = str(window.get("scope", "shared override"))
        else:
            student = raw_maps[level]["student"]
            p1, p99 = (float(value) for value in np.percentile(student, (1, 99)))
            if p99 <= p1:
                p99 = p1 + 1e-8
            scope = "single-pass fallback"
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
            "scope": scope,
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
