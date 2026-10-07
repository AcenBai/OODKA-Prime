"""Router and final-decision visualizations for mechanism-v3."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/oodka_mechanism_v3_mpl")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import torch
import torch.nn.functional as F

from scripts.mechanism_v3_common import (
    DECODER_LABELS,
    LEVELS,
    _ct_limits,
    _draw_gt,
    _heat,
    _rms_4d,
    _save_json,
    _share,
    _symmetric_limit,
)


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
