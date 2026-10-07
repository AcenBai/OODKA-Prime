"""KD-direction diagnostics and alignment figures for mechanism-v3."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/oodka_mechanism_v3_mpl")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
import torch
import torch.nn.functional as F

from scripts.mechanism_v3_common import _normalized_joint_pca_rgb


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
