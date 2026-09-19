#!/usr/bin/env python3
"""Quantify and visualize MRI GV Beta-router spatial/prompt collapse."""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from oodka.models.beta_router import PromptBetaRouter
from oodka.models.prompts import WHS_MRI_GV_LOCALIZATION_PROMPTS, WHS_MRI_GV_REFINEMENT_PROMPTS
from oodka.train.model_builder import build_prompt_features, load_frozen_biomedparse


ANCHOR_NAMES = ["LV", "RV", "LA", "RA", "Myo", "GV"]
REFINE_NAMES = ["LV", "RV", "LA", "RA", "Myo", "AO", "PA"]


def _router(checkpoint, device):
    cfg = checkpoint.get("config", {})
    module = PromptBetaRouter(
        text_dim=512,
        prior_p_mean=float(cfg.get("route_prior_p_mean", 0.7)),
        prior_concentration=float(cfg.get("route_prior_concentration", 10.0)),
        basis_grid_size=int(cfg.get("route_spatial_basis_grid_size", 8)),
        basis_sigma=float(cfg.get("route_spatial_basis_sigma", 0.0)),
    ).to(device).eval()
    module.load_state_dict(checkpoint["beta_router"])
    return module


def _affine_difference(maps):
    maps = maps[:, None]
    transforms = {
        "horizontal_flip": torch.tensor([[[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]),
        "vertical_flip": torch.tensor([[[1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]]),
        "translate_x_5pct": torch.tensor([[[1.0, 0.0, 0.10], [0.0, 1.0, 0.0]]]),
        "scale_0p95": torch.tensor([[[1.0 / 0.95, 0.0, 0.0], [0.0, 1.0 / 0.95, 0.0]]]),
    }
    angle = np.deg2rad(10.0)
    transforms["rotate_10deg"] = torch.tensor([[[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0]]], dtype=torch.float32)
    output = {}
    for name, theta in transforms.items():
        theta = theta.to(device=maps.device, dtype=maps.dtype).expand(maps.shape[0], -1, -1)
        grid = F.affine_grid(theta, maps.shape, align_corners=False)
        warped = F.grid_sample(maps, grid, mode="bilinear", padding_mode="border", align_corners=False)
        output[name] = {
            "mean_absolute_gate_mismatch_if_gate_is_not_warped": float((maps - warped).abs().mean()),
            "max_absolute_gate_mismatch": float((maps - warped).abs().max()),
        }
    return output


def _metrics(maps, concentration, names, prior):
    flat = maps.flatten(1)
    centered = flat - flat.mean(dim=1, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    energy = singular.square()
    effective_rank = float((energy.sum().square() / energy.square().sum().clamp_min(1e-12)).cpu())
    normalized = F.normalize(centered, dim=1)
    correlation = normalized @ normalized.T
    offdiag = correlation[~torch.eye(len(names), dtype=torch.bool, device=correlation.device)]
    per_prompt = []
    for index, name in enumerate(names):
        gate = maps[index]
        per_prompt.append({
            "prompt": name,
            "mean": float(gate.mean()), "spatial_std": float(gate.std()),
            "min": float(gate.min()), "max": float(gate.max()),
            "range": float(gate.max() - gate.min()),
            "mean_minus_prior": float(gate.mean() - prior),
            "saturation_fraction_lt_0p1_or_gt_0p9": float(((gate < 0.1) | (gate > 0.9)).float().mean()),
            "concentration_mean": float(concentration[index].mean()),
            "concentration_std": float(concentration[index].std()),
        })
    return {
        "per_prompt": per_prompt,
        "between_prompt_std_mean": float(maps.std(dim=0).mean()),
        "between_prompt_mean_range": float(maps.mean(dim=(1, 2)).max() - maps.mean(dim=(1, 2)).min()),
        "spatial_std_mean": float(maps.std(dim=(1, 2)).mean()),
        "offdiagonal_spatial_correlation_mean": float(offdiag.mean()) if offdiag.numel() else 1.0,
        "effective_rank_of_centered_prompt_maps": effective_rank,
        "affine_mismatch": _affine_difference(maps),
    }


def _plot(rows, output_path):
    n_checkpoints = len(rows)
    fig, axes = plt.subplots(2 * n_checkpoints, 7, figsize=(20, 5.3 * n_checkpoints), constrained_layout=True)
    if n_checkpoints == 1:
        axes = np.asarray(axes).reshape(2, 7)
    for checkpoint_index, row in enumerate(rows):
        for branch_index, (branch, names) in enumerate((("anchor", ANCHOR_NAMES), ("refinement", REFINE_NAMES))):
            maps = row[f"{branch}_maps"]
            axis_row = axes[2 * checkpoint_index + branch_index]
            for index, ax in enumerate(axis_row):
                if index >= len(names):
                    ax.axis("off"); continue
                image = ax.imshow(maps[index], cmap="viridis", vmin=0, vmax=1)
                prompt = row[branch]["per_prompt"][index]
                ax.set_title(f"{names[index]}  mean={prompt['mean']:.3f}\nstd={prompt['spatial_std']:.4f} range={prompt['range']:.3f}")
                ax.axis("off")
            axis_row[0].set_ylabel(f"{row['label']}\n{branch}", fontsize=12)
    fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.012, pad=0.01, label="P-route Beta mean")
    fig.suptitle("MRI GV prompt-specific spatial router maps", fontsize=16)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--label", action="append", default=[])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--height", type=int, default=80)
    parser.add_argument("--width", type=int, default=80)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    model = load_frozen_biomedparse(device)
    anchor_features = build_prompt_features(model, WHS_MRI_GV_LOCALIZATION_PROMPTS, device)
    refine_features = build_prompt_features(model, WHS_MRI_GV_REFINEMENT_PROMPTS, device)
    output_rows, plot_rows = [], []
    for index, path in enumerate(args.checkpoint):
        checkpoint = torch.load(path, map_location=device)
        router = _router(checkpoint, device)
        prior = float(checkpoint.get("config", {}).get("route_prior_p_mean", 0.7))
        branch_outputs = {}
        with torch.no_grad():
            for branch, features, names in (
                ("anchor", anchor_features, ANCHOR_NAMES),
                ("refinement", refine_features, REFINE_NAMES),
            ):
                routed = router(features["class_emb"].detach(), spatial_size=(args.height, args.width), batch_size=1, sample=False)
                maps = routed["mean"].detach()
                branch_outputs[branch] = _metrics(maps, routed["concentration"], names, prior)
                branch_outputs[f"{branch}_maps"] = maps.cpu().numpy()
        label = args.label[index] if index < len(args.label) else os.path.basename(path)
        output_rows.append({"label": label, "checkpoint": os.path.abspath(path), "prior_p_mean": prior, "anchor": branch_outputs["anchor"], "refinement": branch_outputs["refinement"]})
        plot_rows.append({"label": label, **branch_outputs})
    with open(os.path.join(args.output_dir, "router_collapse_metrics.json"), "w") as handle:
        json.dump({"checkpoints": output_rows}, handle, indent=2)
    _plot(plot_rows, os.path.join(args.output_dir, "router_gate_maps.png"))
    print(json.dumps({"output_dir": os.path.abspath(args.output_dir), "checkpoints": output_rows}, indent=2))


if __name__ == "__main__":
    main()
