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


def _transport_direction_diagnostics(
    *,
    student_tokens: torch.Tensor,
    expert_tokens: torch.Tensor,
    transport: torch.Tensor,
    projector,
    grid: tuple[int, int],
) -> dict:
    """Return the exact forward/reverse cosine-KD objects and diagnostics."""
    forward = projector(transport, expert_tokens)
    reverse = projector(transport.transpose(1, 2), student_tokens)
    student_norm = F.normalize(student_tokens.float(), dim=-1, eps=1e-8)
    expert_norm = F.normalize(expert_tokens.float(), dim=-1, eps=1e-8)
    forward_norm = F.normalize(forward["teacher"].float(), dim=-1, eps=1e-8)
    reverse_norm = F.normalize(reverse["teacher"].float(), dim=-1, eps=1e-8)
    forward_cos = (student_norm * forward_norm).sum(dim=-1)
    reverse_cos = (expert_norm * reverse_norm).sum(dim=-1)
    same_position_cos = (student_norm * expert_norm).sum(dim=-1)

    received = transport.sum(dim=-1).float()
    sent = transport.sum(dim=-2).float()
    conditional = transport.float() / received.unsqueeze(-1).clamp_min(1e-12)
    entropy = -(conditional.clamp_min(1e-12) * conditional.clamp_min(1e-12).log()).sum(
        dim=-1
    )
    entropy = entropy / max(float(np.log(transport.shape[-1])), 1e-8)
    positive = received[received > 0]
    mass_scale = (
        torch.quantile(positive, 0.99).clamp_min(1e-12)
        if positive.numel()
        else received.new_tensor(1.0)
    )
    mass_visibility = (received / mass_scale).clamp(0.0, 1.0)
    confidence = mass_visibility * (1.0 - entropy).clamp(0.0, 1.0)

    def weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> float:
        return float(
            ((value * weight).sum() / weight.sum().clamp_min(1e-8)).item()
        )

    def summarize(value: torch.Tensor, weight: torch.Tensor) -> dict:
        flat = value[0].detach().float().cpu()
        return {
            "weighted_mean": weighted_mean(value, weight),
            "p10": float(torch.quantile(flat, 0.10).item()),
            "median": float(torch.quantile(flat, 0.50).item()),
            "p90": float(torch.quantile(flat, 0.90).item()),
            "fraction_ge_0p8": float((flat >= 0.8).float().mean().item()),
            "fraction_ge_0p9": float((flat >= 0.9).float().mean().item()),
        }

    pca = _normalized_joint_pca_rgb(
        (student_tokens, expert_tokens, forward["teacher"], reverse["teacher"]),
        (grid, grid, grid, grid),
    )
    return {
        "pca": pca,
        "forward_cos": forward_cos[0].detach().cpu().numpy().reshape(grid),
        "reverse_cos": reverse_cos[0].detach().cpu().numpy().reshape(grid),
        "same_position_cos": same_position_cos[0].detach().cpu().numpy().reshape(grid),
        "received_mass": received[0].detach().cpu().numpy().reshape(grid),
        "normalized_entropy": entropy[0].detach().cpu().numpy().reshape(grid),
        "confidence": confidence[0].detach().cpu().numpy().reshape(grid),
        "summary": {
            "forward_cosine": summarize(forward_cos, received),
            "reverse_cosine": summarize(reverse_cos, sent),
            "same_position_cosine": summarize(
                same_position_cos, torch.ones_like(received)
            ),
            "mean_normalized_entropy": weighted_mean(entropy, received),
            "mean_confidence": weighted_mean(confidence, received),
            "received_mass_p99": float(mass_scale.item()),
        },
    }


def _plot_kd_direction_alignment(
    output_dir: Path,
    *,
    level: int,
    branch_diagnostics: dict[str, dict],
) -> None:
    """Plot exact OT-cosine agreement and shared normalized-token PCA-RGB."""
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 7, figsize=(25, 8), constrained_layout=True)
    labels = (
        "Student normalized\ntoken PCA",
        "Raw Expert normalized\ntoken PCA",
        "Transported Expert PCA\n$T:E\\rightarrow B$",
        "Forward cosine\n$\\cos(B,\\widetilde E)$",
        "Transported Student PCA\n$T^\\top:B\\rightarrow E$",
        "Reverse cosine\n$\\cos(E,\\widetilde B)$",
        "Transport confidence\nreceived mass × (1−entropy)",
    )
    for row, branch in enumerate(("p", "s")):
        diag = branch_diagnostics[branch]
        values = (
            diag["pca"][0],
            diag["pca"][1],
            diag["pca"][2],
            diag["forward_cos"],
            diag["pca"][3],
            diag["reverse_cos"],
            diag["confidence"],
        )
        for col, (axis, value, label) in enumerate(zip(axes[row], values, labels)):
            if col in (3, 5):
                shown = axis.imshow(value, cmap="magma", vmin=0.0, vmax=1.0)
            elif col == 6:
                shown = axis.imshow(value, cmap="viridis", vmin=0.0, vmax=1.0)
            else:
                shown = axis.imshow(value)
            axis.set_title(label, fontsize=10)
            axis.axis("off")
        axes[row, 0].set_ylabel(f"{branch.upper()} branch", fontsize=13)
        summary = diag["summary"]
        axes[row, 3].text(
            0.02,
            0.02,
            f"weighted mean={summary['forward_cosine']['weighted_mean']:.3f}",
            transform=axes[row, 3].transAxes,
            color="white",
            fontsize=9,
            bbox={"facecolor": "black", "alpha": 0.65, "pad": 2},
        )
        axes[row, 5].text(
            0.02,
            0.02,
            f"weighted mean={summary['reverse_cosine']['weighted_mean']:.3f}",
            transform=axes[row, 5].transAxes,
            color="white",
            fontsize=9,
            bbox={"facecolor": "black", "alpha": 0.65, "pad": 2},
        )
    cosine_scalar = ScalarMappable(norm=Normalize(0.0, 1.0), cmap="magma")
    cosine_scalar.set_array([])
    figure.colorbar(cosine_scalar, ax=axes[:, (3, 5)], shrink=0.55, label="Cosine similarity")
    confidence_scalar = ScalarMappable(norm=Normalize(0.0, 1.0), cmap="viridis")
    confidence_scalar.set_array([])
    figure.colorbar(confidence_scalar, ax=axes[:, 6], shrink=0.55, label="Transport confidence")
    figure.suptitle(
        f"res{level}: bidirectional OT-KD channel-direction agreement",
        fontsize=16,
    )
    figure.savefig(output_dir / f"res{level}_kd_direction_alignment.png", dpi=190)
    plt.close(figure)

    pca_figure, pca_axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
    pca_labels = (
        "Student normalized tokens",
        "Raw Expert normalized tokens",
        "Expert transported to Student",
        "Student transported to Expert",
    )
    for row, branch in enumerate(("p", "s")):
        for axis, value, label in zip(
            pca_axes[row], branch_diagnostics[branch]["pca"], pca_labels
        ):
            axis.imshow(value)
            axis.set_title(label, fontsize=10)
            axis.axis("off")
        pca_axes[row, 0].set_ylabel(f"{branch.upper()} branch", fontsize=13)
    pca_figure.suptitle(
        f"res{level}: shared PCA-RGB of L2-normalized channel tokens",
        fontsize=15,
    )
    pca_figure.savefig(output_dir / f"res{level}_shared_pca_rgb.png", dpi=190)
    plt.close(pca_figure)

    standalone_dir = output_dir / "standalone"
    standalone_dir.mkdir(parents=True, exist_ok=True)

    def locally_stretch_rgb(value: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        stretched = np.empty_like(value)
        channel_stats = []
        for channel in range(3):
            plane = value[..., channel]
            lo, hi = np.quantile(plane, (0.01, 0.99))
            stretched[..., channel] = np.clip(
                (plane - lo) / max(float(hi - lo), 1e-8), 0.0, 1.0
            )
            channel_stats.append(
                {
                    "p01": float(lo),
                    "p99": float(hi),
                    "range": float(hi - lo),
                }
            )
        return stretched, channel_stats

    def save_rgb(
        path: Path,
        value: np.ndarray,
        *,
        title: str,
    ) -> None:
        stretched, stats = locally_stretch_rgb(value)
        figure, axis = plt.subplots(figsize=(7, 7), constrained_layout=True)
        axis.imshow(stretched)
        axis.set_title(
            f"{title}\nlocal per-channel P1–P99 RGB stretch",
            fontsize=14,
        )
        axis.axis("off")
        ranges = " / ".join(f"{item['range']:.3g}" for item in stats)
        figure.text(
            0.5,
            0.01,
            f"Original shared-PCA RGB P1–P99 ranges (R/G/B): {ranges}. "
            "Locally stretched colors are not cross-panel comparable.",
            ha="center",
            fontsize=9,
        )
        figure.savefig(path, dpi=220)
        plt.close(figure)

    def save_cosine(
        path: Path,
        value: np.ndarray,
        *,
        title: str,
    ) -> None:
        finite = value[np.isfinite(value)]
        lo, hi = np.quantile(finite, (0.01, 0.99))
        if hi - lo < 1e-6:
            lo, hi = float(finite.min()), float(finite.max())
        if hi - lo < 1e-6:
            lo, hi = lo - 5e-7, hi + 5e-7
        figure, axis = plt.subplots(figsize=(7.8, 7), constrained_layout=True)
        shown = axis.imshow(value, cmap="magma", vmin=float(lo), vmax=float(hi))
        axis.set_title(
            f"{title}\nlocal P1–P99 cosine scale [{lo:.4f}, {hi:.4f}]",
            fontsize=14,
        )
        axis.axis("off")
        figure.colorbar(shown, ax=axis, shrink=0.78, label="Cosine similarity")
        figure.text(
            0.5,
            0.01,
            f"min={finite.min():.4f}, median={np.median(finite):.4f}, "
            f"mean={finite.mean():.4f}, max={finite.max():.4f}",
            ha="center",
            fontsize=10,
        )
        figure.savefig(path, dpi=220)
        plt.close(figure)

    for branch in ("p", "s"):
        diag = branch_diagnostics[branch]
        prefix = standalone_dir / f"res{level}_{branch}"
        save_rgb(
            Path(f"{prefix}_transported_expert_pca.png"),
            diag["pca"][2],
            title=f"res{level} {branch.upper()}: Transported Expert PCA  T:E→B",
        )
        save_cosine(
            Path(f"{prefix}_forward_cosine.png"),
            diag["forward_cos"],
            title=f"res{level} {branch.upper()}: Forward cosine  cos(B,Ẽ)",
        )
        save_rgb(
            Path(f"{prefix}_transported_student_pca.png"),
            diag["pca"][3],
            title=f"res{level} {branch.upper()}: Transported Student PCA  Tᵀ:B→E",
        )
        save_cosine(
            Path(f"{prefix}_reverse_cosine.png"),
            diag["reverse_cos"],
            title=f"res{level} {branch.upper()}: Reverse cosine  cos(E,B̃)",
        )


def _morton_order(height: int, width: int) -> np.ndarray:
    """Return raster indices ordered by a 2D Morton/Z-order curve."""
    entries = []
    bits = max(height, width).bit_length()
    for y in range(height):
        for x in range(width):
            code = 0
            for bit in range(bits):
                code |= ((x >> bit) & 1) << (2 * bit)
                code |= ((y >> bit) & 1) << (2 * bit + 1)
            entries.append((code, y * width + x))
    return np.asarray([index for _code, index in sorted(entries)], dtype=np.int64)


def _conditional_transport(transport: torch.Tensor) -> np.ndarray:
    value = transport[0].detach().float().cpu().numpy()
    return value / np.maximum(value.sum(axis=1, keepdims=True), 1e-12)


def _transport_entropy(conditional: np.ndarray) -> np.ndarray:
    count = conditional.shape[1]
    return -(
        conditional * np.log(np.maximum(conditional, 1e-12))
    ).sum(axis=1) / max(float(np.log(count)), 1e-8)


def _select_transport_queries(
    labels: np.ndarray,
    demand: np.ndarray,
    *,
    max_queries: int = 8,
) -> tuple[list[int], list[str]]:
    """Select semantic centroids, then spatially separated hard-demand tokens."""
    height, width = labels.shape
    selected: list[int] = []
    names: list[str] = []
    for class_id in sorted(int(v) for v in np.unique(labels) if int(v) > 0):
        positions = np.argwhere(labels == class_id)
        centroid = positions.mean(axis=0)
        y, x = positions[np.square(positions - centroid).sum(axis=1).argmin()]
        selected.append(int(y * width + x))
        names.append(f"class {class_id} centroid")
        if len(selected) >= max_queries:
            return selected, names

    ranked = np.argsort(demand.reshape(-1))[::-1]
    for index in ranked:
        y, x = divmod(int(index), width)
        if any(
            (y - divmod(existing, width)[0]) ** 2
            + (x - divmod(existing, width)[1]) ** 2
            < 16
            for existing in selected
        ):
            continue
        selected.append(int(index))
        names.append(f"high-demand {len(selected)}")
        if len(selected) >= max_queries:
            break
    return selected, names


def _plot_semantic_sorted_transport(
    path: Path,
    *,
    level: int,
    labels: np.ndarray,
    label_names: dict[int, str],
    transports: dict[str, torch.Tensor],
) -> None:
    height, width = labels.shape
    morton = _morton_order(height, width)
    flat_labels = labels.reshape(-1)
    order = np.asarray(
        sorted(morton.tolist(), key=lambda index: int(flat_labels[index])),
        dtype=np.int64,
    )
    ordered_labels = flat_labels[order]
    segments = []
    start = 0
    for class_id in np.unique(ordered_labels):
        end = start + int((ordered_labels == class_id).sum())
        segments.append((int(class_id), start, end))
        start = end

    figure, axes = plt.subplots(1, 2, figsize=(17, 8), constrained_layout=True)
    shown = None
    for axis, branch in zip(axes, ("p", "s")):
        conditional = _conditional_transport(transports[branch])
        conditional = conditional[np.ix_(order, order)]
        log_relative = np.log10(
            conditional * conditional.shape[1] + 1e-6
        )
        shown = axis.imshow(
            log_relative,
            cmap="coolwarm",
            vmin=-3.0,
            vmax=2.0,
            interpolation="nearest",
            rasterized=True,
        )
        centers = []
        tick_labels = []
        for class_id, seg_start, seg_end in segments:
            axis.axhline(seg_start - 0.5, color="black", lw=0.45, alpha=0.7)
            axis.axvline(seg_start - 0.5, color="black", lw=0.45, alpha=0.7)
            centers.append((seg_start + seg_end - 1) / 2)
            tick_labels.append(label_names.get(class_id, "background" if class_id == 0 else str(class_id)))
        axis.set_xticks(centers, tick_labels, rotation=45, ha="right", fontsize=8)
        axis.set_yticks(centers, tick_labels, fontsize=8)
        axis.set_xlabel("Expert/source tokens grouped by anatomy")
        axis.set_ylabel("Student/query tokens grouped by anatomy")
        axis.set_title(f"res{level} {branch.upper()}: row-conditional coupling")
    figure.colorbar(
        shown,
        ax=axes,
        shrink=0.78,
        label=r"$\log_{10}[N_E\,P(\mathrm{expert}\mid\mathrm{student})]$",
    )
    figure.suptitle(
        "Semantic-grouped OT matrices; within each group tokens follow Morton/Z-order",
        fontsize=15,
    )
    figure.savefig(path, dpi=210)
    plt.close(figure)


def _plot_query_transport_atlas(
    path: Path,
    *,
    level: int,
    image: np.ndarray,
    gt: np.ndarray,
    label_names: dict[int, str],
    grid: tuple[int, int],
    transports: dict[str, torch.Tensor],
    demand: np.ndarray,
) -> None:
    labels = F.interpolate(
        torch.from_numpy(gt)[None, None].float(), size=grid, mode="nearest"
    )[0, 0].numpy().astype(np.int16)
    queries, query_names = _select_transport_queries(labels, demand)
    conditionals = {
        branch: _conditional_transport(transport)
        for branch, transport in transports.items()
    }
    entropies = {
        branch: _transport_entropy(value)
        for branch, value in conditionals.items()
    }

    figure = plt.figure(figsize=(16, 19), constrained_layout=True)
    spec = figure.add_gridspec(5, 4, height_ratios=(1.25, 1, 1, 1, 1))
    overview = figure.add_subplot(spec[0, :2])
    _draw_gt(overview, image, gt, label_names, legend=True)
    overview.set_title(f"res{level}: shared Student query locations")
    h_img, w_img = image.shape
    for number, index in enumerate(queries, start=1):
        y, x = divmod(index, grid[1])
        px = (x + 0.5) / grid[1] * w_img
        py = (y + 0.5) / grid[0] * h_img
        overview.scatter(px, py, s=90, c="white", edgecolors="black", linewidths=1.2)
        overview.text(px, py, str(number), ha="center", va="center", fontsize=9, weight="bold")
    legend_axis = figure.add_subplot(spec[0, 2:])
    legend_axis.axis("off")
    legend_axis.text(
        0.0,
        1.0,
        "\n".join(f"q{i}: {name}" for i, name in enumerate(query_names, start=1)),
        va="top",
        fontsize=12,
    )

    shown = None
    for branch_row, branch in enumerate(("p", "s")):
        for local_index, query in enumerate(queries):
            row = 1 + branch_row * 2 + local_index // 4
            col = local_index % 4
            axis = figure.add_subplot(spec[row, col])
            conditional = conditionals[branch][query]
            log_relative = np.log10(conditional * conditional.size + 1e-6).reshape(grid)
            shown = axis.imshow(log_relative, cmap="coolwarm", vmin=-3.0, vmax=2.0)
            qy, qx = divmod(query, grid[1])
            axis.scatter(qx, qy, marker="x", s=50, c="black", linewidths=1.6)
            entropy = entropies[branch][query]
            effective = float(conditional.size ** entropy)
            top8 = float(np.partition(conditional, -8)[-8:].sum())
            axis.set_title(
                f"{branch.upper()} q{local_index + 1}: H={entropy:.2f}, "
                f"N_eff={effective:.0f}, top8={top8:.2f}",
                fontsize=10,
            )
            axis.axis("off")
    figure.colorbar(
        shown,
        ax=figure.axes,
        shrink=0.35,
        label=r"$\log_{10}[N_E\,P(\mathrm{source}\mid q)]$",
    )
    figure.suptitle(
        f"res{level}: query-conditioned 2D transport atlas (same queries for P and S)",
        fontsize=17,
    )
    figure.savefig(path, dpi=210)
    plt.close(figure)


def _plot_p_query_transport_atlas(
    path: Path,
    *,
    level: int,
    image: np.ndarray,
    gt: np.ndarray,
    label_names: dict[int, str],
    grid: tuple[int, int],
    transport: torch.Tensor,
    demand: torch.Tensor,
) -> None:
    """P-only atlas: preserve the clean spatial retrieval story."""
    labels = F.interpolate(
        torch.from_numpy(gt)[None, None].float(), size=grid, mode="nearest"
    )[0, 0].numpy().astype(np.int16)
    demand_np = demand[0].detach().float().cpu().numpy().reshape(grid)
    queries, query_names = _select_transport_queries(labels, demand_np)
    conditional = _conditional_transport(transport)
    entropy = _transport_entropy(conditional)

    figure = plt.figure(figsize=(16, 10.5), constrained_layout=True)
    spec = figure.add_gridspec(3, 4, height_ratios=(1.20, 1, 1))
    overview = figure.add_subplot(spec[0, :2])
    _draw_gt(overview, image, gt, label_names, legend=True)
    overview.set_title(f"res{level} P-OT: Student query locations")
    h_img, w_img = image.shape
    for number, index in enumerate(queries, start=1):
        y, x = divmod(index, grid[1])
        px = (x + 0.5) / grid[1] * w_img
        py = (y + 0.5) / grid[0] * h_img
        overview.scatter(
            px, py, s=90, c="white", edgecolors="black", linewidths=1.2
        )
        overview.text(
            px, py, str(number), ha="center", va="center", fontsize=9,
            weight="bold",
        )
    legend_axis = figure.add_subplot(spec[0, 2:])
    legend_axis.axis("off")
    legend_axis.text(
        0.0,
        1.0,
        "\n".join(
            f"q{i}: {name}" for i, name in enumerate(query_names, start=1)
        ),
        va="top",
        fontsize=12,
    )

    shown = None
    for local_index, query in enumerate(queries):
        axis = figure.add_subplot(spec[1 + local_index // 4, local_index % 4])
        query_conditional = conditional[query]
        log_relative = np.log10(
            query_conditional * query_conditional.size + 1e-6
        ).reshape(grid)
        shown = axis.imshow(
            log_relative, cmap="coolwarm", vmin=-3.0, vmax=2.0
        )
        qy, qx = divmod(query, grid[1])
        axis.scatter(qx, qy, marker="x", s=50, c="black", linewidths=1.6)
        query_entropy = entropy[query]
        effective = float(query_conditional.size ** query_entropy)
        top8 = float(np.partition(query_conditional, -8)[-8:].sum())
        axis.set_title(
            f"q{local_index + 1}: H={query_entropy:.2f}, "
            f"N_eff={effective:.0f}, top8={top8:.2f}",
            fontsize=10,
        )
        axis.axis("off")
    figure.colorbar(
        shown,
        ax=figure.axes,
        shrink=0.45,
        label=r"$\log_{10}[N_E\,P(\mathrm{expert}\mid q)]$",
    )
    figure.suptitle(
        f"res{level} P-OT: query-conditioned spatial expert retrieval",
        fontsize=17,
    )
    figure.savefig(path, dpi=210)
    plt.close(figure)


def _select_s_uot_queries(
    labels: np.ndarray,
    demand: np.ndarray,
    source_acceptance: np.ndarray,
    entropy: np.ndarray,
    *,
    max_queries: int = 6,
) -> tuple[list[int], list[str]]:
    """Select anatomy anchors plus accepted and rejected S-UOT examples."""
    height, width = labels.shape
    selected: list[int] = []
    names: list[str] = []
    for class_id in sorted(int(v) for v in np.unique(labels) if int(v) > 0):
        positions = np.argwhere(labels == class_id)
        centroid = positions.mean(axis=0)
        y, x = positions[np.square(positions - centroid).sum(axis=1).argmin()]
        selected.append(int(y * width + x))
        names.append(f"class {class_id} centroid")
        if len(selected) >= max_queries:
            return selected, names

    demand_flat = demand.reshape(-1)
    acceptance_flat = source_acceptance.reshape(-1)
    entropy_flat = entropy.reshape(-1)
    positive = demand_flat > np.quantile(demand_flat[demand_flat > 0], 0.25)
    foreground = labels.reshape(-1) > 0
    eligible = positive & foreground
    if not eligible.any():
        eligible = positive
    candidate_scores = (
        acceptance_flat * (1.0 - entropy_flat),
        (1.0 - acceptance_flat) * np.sqrt(np.maximum(demand_flat, 0.0)),
    )
    candidate_names = ("high-accept / concentrated", "high-rejection")
    for scores, name in zip(candidate_scores, candidate_names):
        ranked = np.argsort(np.where(eligible, scores, -np.inf))[::-1]
        for index in ranked:
            if not np.isfinite(scores[index]):
                continue
            y, x = divmod(int(index), width)
            if any(
                (y - divmod(existing, width)[0]) ** 2
                + (x - divmod(existing, width)[1]) ** 2
                < 9
                for existing in selected
            ):
                continue
            selected.append(int(index))
            names.append(name)
            break
        if len(selected) >= max_queries:
            break
    return selected, names


def _plot_s_uot_rejection_atlas(
    path: Path,
    *,
    level: int,
    image: np.ndarray,
    gt: np.ndarray,
    label_names: dict[int, str],
    grid: tuple[int, int],
    transport: torch.Tensor,
    demand: torch.Tensor,
    supply: torch.Tensor,
    output: dict[str, torch.Tensor],
) -> None:
    """S-only view separating selection/rejection from accepted routing."""
    labels = F.interpolate(
        torch.from_numpy(gt)[None, None].float(), size=grid, mode="nearest"
    )[0, 0].numpy().astype(np.int16)
    plan = transport[0].detach().float().cpu().numpy()
    demand_np = demand[0].detach().float().cpu().numpy()
    supply_np = supply[0].detach().float().cpu().numpy()
    received_np = output["received"][0].detach().float().cpu().numpy()
    transported_np = output["transported"][0].detach().float().cpu().numpy()
    row_unused = output.get(
        "row_unused", (demand - output["received"]).clamp_min(0.0)
    )[0].detach().float().cpu().numpy()
    rejected_np = output["rejected"][0].detach().float().cpu().numpy()

    source_acceptance = np.clip(
        received_np / np.maximum(demand_np, 1e-12), 0.0, 1.0
    )
    source_rejection = np.clip(
        row_unused / np.maximum(demand_np, 1e-12), 0.0, 1.0
    )
    expert_rejection = np.clip(
        rejected_np / np.maximum(supply_np, 1e-12), 0.0, 1.0
    )
    conditional = plan / np.maximum(plan.sum(axis=1, keepdims=True), 1e-12)
    entropy = _transport_entropy(conditional)
    queries, query_names = _select_s_uot_queries(
        labels,
        demand_np.reshape(grid),
        source_acceptance.reshape(grid),
        entropy,
    )
    target_prior = supply_np / max(float(supply_np.sum()), 1e-12)
    target_count = plan.shape[1]

    figure = plt.figure(figsize=(19, 11), constrained_layout=True)
    spec = figure.add_gridspec(3, 6, height_ratios=(1.12, 1, 1))
    overview = figure.add_subplot(spec[0, :2])
    _draw_gt(overview, image, gt, label_names, legend=True)
    overview.set_title(f"res{level} S-UOT: selected Student queries")
    h_img, w_img = image.shape
    for number, index in enumerate(queries, start=1):
        y, x = divmod(index, grid[1])
        px = (x + 0.5) / grid[1] * w_img
        py = (y + 0.5) / grid[0] * h_img
        overview.scatter(
            px, py, s=90, c="white", edgecolors="black", linewidths=1.2
        )
        overview.text(
            px, py, str(number), ha="center", va="center", fontsize=9,
            weight="bold",
        )
    legend_axis = figure.add_subplot(spec[0, 2])
    legend_axis.axis("off")
    legend_axis.text(
        0.0,
        1.0,
        "\n".join(
            f"q{i}: {name}" for i, name in enumerate(query_names, start=1)
        ),
        va="top",
        fontsize=10,
    )
    ratio_axes = [figure.add_subplot(spec[0, column]) for column in (3, 4, 5)]
    ratio_image = _share(
        ratio_axes[0],
        source_acceptance.reshape(grid),
        title="Student accepted demand",
    )
    _share(
        ratio_axes[1],
        source_rejection.reshape(grid),
        title="Student rejected / unused demand",
    )
    _share(
        ratio_axes[2],
        expert_rejection.reshape(grid),
        title="Expert rejected / unused supply",
    )

    accepted_image = None
    lift_image = None
    accepted_axes = []
    lift_axes = []
    for local_index, query in enumerate(queries):
        qy, qx = divmod(query, grid[1])
        accepted_axis = figure.add_subplot(spec[1, local_index])
        accepted_axes.append(accepted_axis)
        accepted_relative = np.log10(
            target_count * plan[query] / max(float(demand_np[query]), 1e-12)
            + 1e-6
        ).reshape(grid)
        accepted_image = accepted_axis.imshow(
            accepted_relative, cmap="coolwarm", vmin=-3.0, vmax=2.0
        )
        accepted_axis.scatter(
            qx, qy, marker="x", s=45, c="black", linewidths=1.5
        )
        accepted_axis.set_title(
            f"q{local_index + 1}: accepted={source_acceptance[query]:.2f}, "
            f"H={entropy[query]:.2f}",
            fontsize=9,
        )
        accepted_axis.axis("off")

        lift_axis = figure.add_subplot(spec[2, local_index])
        lift_axes.append(lift_axis)
        prior_lift = np.log10(
            (conditional[query] + 1e-12) / (target_prior + 1e-12)
        ).reshape(grid)
        lift_image = lift_axis.imshow(
            prior_lift, cmap="coolwarm", vmin=-2.0, vmax=2.0
        )
        lift_axis.scatter(
            qx, qy, marker="x", s=45, c="black", linewidths=1.5
        )
        lift_axis.set_title(
            f"q{local_index + 1}: routing lift vs supply prior", fontsize=9
        )
        lift_axis.axis("off")

    figure.colorbar(
        ratio_image,
        ax=ratio_axes,
        shrink=0.55,
        label="Fraction of local mass",
    )
    figure.colorbar(
        accepted_image,
        ax=accepted_axes,
        shrink=0.52,
        label=r"$\log_{10}[N_E\,\pi_{ij}/a_i]$ (rejection retained)",
    )
    figure.colorbar(
        lift_image,
        ax=lift_axes,
        shrink=0.52,
        label=r"$\log_{10}[P(j\mid i)/(b_j/\sum b)]$",
    )
    figure.suptitle(
        f"res{level} S-UOT: rejection first, accepted routing second",
        fontsize=17,
    )
    figure.savefig(path, dpi=210)
    plt.close(figure)


def _plot_barycentric_flow(
    path: Path,
    *,
    level: int,
    image: np.ndarray,
    gt: np.ndarray,
    label_names: dict[int, str],
    grid: tuple[int, int],
    transports: dict[str, torch.Tensor],
    masses: dict[str, torch.Tensor],
) -> None:
    coordinates = _coordinates(
        *grid, device=torch.device("cpu"), dtype=torch.float32
    ).numpy()
    h_img, w_img = image.shape
    figure, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
    flow_data = {}
    all_magnitudes = []
    for branch in ("p", "s"):
        conditional = _conditional_transport(transports[branch])
        expected = conditional @ coordinates
        delta = expected - coordinates
        magnitude = np.linalg.norm(delta, axis=1)
        entropy = _transport_entropy(conditional)
        demand = masses[branch][0].detach().float().cpu().numpy()
        positive = demand[demand > 0]
        mass_scale = np.quantile(positive, 0.99) if positive.size else 1.0
        score = np.clip(demand / max(float(mass_scale), 1e-12), 0, 1) * (1 - entropy)
        flow_data[branch] = (expected, delta, magnitude, score)
        all_magnitudes.append(magnitude)
    magnitude_max = max(float(np.quantile(np.concatenate(all_magnitudes), 0.99)), 1e-6)
    color_norm = Normalize(0.0, magnitude_max)

    for axis, branch in zip(axes, ("p", "s")):
        _draw_gt(axis, image, gt, label_names, legend=(branch == "p"))
        expected, delta, magnitude, score = flow_data[branch]
        candidates = np.argsort(score)[::-1][:72]
        for index in candidates:
            if score[index] <= 0:
                continue
            y0, x0 = coordinates[index]
            y1, x1 = expected[index]
            start = ((x0 + 1) * 0.5 * w_img, (y0 + 1) * 0.5 * h_img)
            end = ((x1 + 1) * 0.5 * w_img, (y1 + 1) * 0.5 * h_img)
            color = plt.cm.plasma(color_norm(magnitude[index]))
            arrow = FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=7,
                linewidth=0.5 + 1.8 * score[index],
                color=color,
                alpha=0.25 + 0.7 * score[index],
            )
            axis.add_patch(arrow)
        axis.set_title(
            f"res{level} {branch.upper()}: barycentric flow\n"
            "top 72 demand×concentration tokens",
            fontsize=13,
        )
    scalar = ScalarMappable(norm=color_norm, cmap="plasma")
    scalar.set_array([])
    figure.colorbar(
        scalar,
        ax=axes,
        shrink=0.72,
        label="Expected displacement in normalized coordinates",
    )
    figure.savefig(path, dpi=210)
    plt.close(figure)


def _plot_spatial_transport_suite(
    output_dir: Path,
    *,
    level: int,
    image: np.ndarray,
    gt: np.ndarray,
    label_names: dict[int, str],
    grid: tuple[int, int],
    p_transport: torch.Tensor,
    s_transport: torch.Tensor,
    p_mass: torch.Tensor,
    s_mass_a: torch.Tensor,
    s_mass_b: torch.Tensor,
    s_output: dict[str, torch.Tensor],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    transports = {"p": p_transport, "s": s_transport}
    masses = {"p": p_mass, "s": s_mass_a}
    labels = F.interpolate(
        torch.from_numpy(gt)[None, None].float(), size=grid, mode="nearest"
    )[0, 0].numpy().astype(np.int16)
    combined_demand = np.maximum(
        p_mass[0].detach().float().cpu().numpy(),
        s_mass_a[0].detach().float().cpu().numpy(),
    ).reshape(grid)
    _plot_semantic_sorted_transport(
        output_dir / f"res{level}_ps_semantic_sorted_transport.png",
        level=level,
        labels=labels,
        label_names=label_names,
        transports=transports,
    )
    _plot_query_transport_atlas(
        output_dir / f"res{level}_ps_query_transport_atlas.png",
        level=level,
        image=image,
        gt=gt,
        label_names=label_names,
        grid=grid,
        transports=transports,
        demand=combined_demand,
    )
    _plot_p_query_transport_atlas(
        output_dir / f"res{level}_p_query_transport_atlas.png",
        level=level,
        image=image,
        gt=gt,
        label_names=label_names,
        grid=grid,
        transport=p_transport,
        demand=p_mass,
    )
    _plot_s_uot_rejection_atlas(
        output_dir / f"res{level}_s_uot_rejection_atlas.png",
        level=level,
        image=image,
        gt=gt,
        label_names=label_names,
        grid=grid,
        transport=s_transport,
        demand=s_mass_a,
        supply=s_mass_b,
        output=s_output,
    )
    _plot_barycentric_flow(
        output_dir / f"res{level}_ps_barycentric_flow.png",
        level=level,
        image=image,
        gt=gt,
        label_names=label_names,
        grid=grid,
        transports=transports,
        masses=masses,
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
    overused = output.get(
        "overused", (output["transported"] - mass["b"]).clamp_min(0.0)
    )
    overused_map = _mass_image(overused, grid)
    rejection_ratio = (
        output["rejected"] / mass["b"].clamp_min(1e-8)
    )[0].clamp(0.0, 1.0).detach().cpu().numpy().reshape(grid)
    overuse_ratio = (
        overused / mass["b"].clamp_min(1e-8)
    )[0].detach().cpu().numpy().reshape(grid)
    usage_ratio = (
        output["transported"] / mass["b"].clamp_min(1e-8)
    )[0].detach().cpu().numpy().reshape(grid)
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
        axes[1, 3], rejection_ratio, title="Expert under-supply / rejection ratio"
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
        f"overused total = {overused.sum().item():.4f}\n"
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

    marginal_figure, marginal_axes = plt.subplots(
        1, 5, figsize=(19, 4), constrained_layout=True
    )
    marginal_values = (
        (b_map, "Supply $b_j$ × N", "magma", 0.0, mass_vmax),
        (transported_map, "Used $m_j$ × N", "magma", 0.0, mass_vmax),
        (rejection_ratio, "Underfill $[b_j-m_j]_+/b_j$", "viridis", 0.0, 1.0),
        (np.log2(1.0 + overuse_ratio), "Overuse $\\log_2(1+[m_j-b_j]_+/b_j)$", "inferno", 0.0, None),
        (np.log2(np.maximum(usage_ratio, 1e-8)), "Usage $\\log_2(m_j/b_j)$", "coolwarm", -4.0, 4.0),
    )
    for axis, (value, title, cmap, vmin, vmax) in zip(
        marginal_axes, marginal_values
    ):
        shown = axis.imshow(value, cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(title, fontsize=10)
        axis.axis("off")
        marginal_figure.colorbar(shown, ax=axis, shrink=0.72)
    marginal_figure.suptitle(
        f"res{level} S transport marginal audit: underfill and overuse are distinct",
        fontsize=14,
    )
    marginal_figure.savefig(
        path.with_name(f"{path.stem}_marginals.png"), dpi=190
    )
    plt.close(marginal_figure)
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
        "overused_density": overused_map,
        "rejection_ratio": rejection_ratio,
        "overuse_ratio": overuse_ratio,
        "usage_ratio": usage_ratio,
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
                title=(
                    f"res{level}"
                    if row == 0
                    else f"mean P gate={float(mean[row].mean()):.3f}"
                ),
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
                title=f"Fused, mean P gate={float(mean[prompt_index].mean()):.3f}",
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




__all__ = [name for name in globals() if not name.startswith("__")]
