"""P/S optimal-transport visualizations for mechanism-v3."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/oodka_mechanism_v3_mpl")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from scripts.mechanism_v3_common import (
    _aggregate_transport,
    _draw_gt,
    _heat,
    _mass_image,
    _robust_max,
    _row_expected,
    _share,
    _token_image,
    _token_rms,
    _transport_log_conditional,
    _transport_log_density,
)
from scripts.mechanism_v3_representation import _route_field


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
