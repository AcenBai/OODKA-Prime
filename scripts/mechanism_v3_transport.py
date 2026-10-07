"""Transport plan and spatial-flow visualizations for mechanism-v3."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/oodka_mechanism_v3_mpl")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch
import numpy as np
import torch
import torch.nn.functional as F

from oodka.models.ot.cost import _coordinates
from scripts.mechanism_v3_common import (
    _draw_gt,
    _share,
)


from scripts.mechanism_v3_transport_alignment import (
    _plot_kd_direction_alignment,
    _transport_direction_diagnostics,
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
