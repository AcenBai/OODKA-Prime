"""Shared command-line and configuration helpers for training entry points."""

from __future__ import annotations

import argparse
import os
import subprocess
from typing import Any, Dict

from ..config import TrainConfig


def add_common_training_arguments(
    parser: argparse.ArgumentParser,
    *,
    device_default: str,
    n_epochs_default: int,
    batch_size_default: int,
    image_size_default: int,
    num_workers_default: int,
    output_required: bool,
    raw_cache_cases_default: int | None = None,
) -> None:
    """Add runtime arguments shared by the standard and ROI trainers."""
    parser.add_argument("--device", default=device_default)
    parser.add_argument("--n_epochs", type=int, default=n_epochs_default)
    parser.add_argument("--batch_size", type=int, default=batch_size_default)
    parser.add_argument("--image_size", type=int, default=image_size_default)
    parser.add_argument("--num_workers", type=int, default=num_workers_default)
    if raw_cache_cases_default is not None:
        parser.add_argument(
            "--raw_cache_cases", type=int, default=raw_cache_cases_default
        )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output_dir", required=output_required, default=None)
    parser.add_argument("--val_every_epochs", type=int, default=5)
    parser.add_argument("--train_case_limit", type=int, default=0)
    parser.add_argument("--val_case_limit", type=int, default=0)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_val_batches", type=int, default=0)
    parser.add_argument("--no_amp", action="store_true")


def add_fusion_training_arguments(parser: argparse.ArgumentParser) -> None:
    """Expose every expert-alignment choice as an explicit soft switch."""
    parser.add_argument(
        "--expert_adapter_variant",
        choices=("legacy", "direct_shared"),
        default="legacy",
    )
    parser.add_argument(
        "--expert_ortho_weight",
        type=float,
        default=1.0,
        help="Multiplier on only the Expert-side P/S orthogonality term.",
    )
    parser.add_argument(
        "--s_transport_mode",
        choices=("unbalanced", "capacity_partial"),
        default="capacity_partial",
    )
    parser.add_argument("--s_partial_mass_fraction", type=float, default=0.5)
    relative_group = parser.add_mutually_exclusive_group()
    relative_group.add_argument(
        "--relative_kd",
        dest="relative_kd",
        action="store_true",
        help="Enable reverse student-to-expert KD (default).",
    )
    relative_group.add_argument(
        "--no_relative_kd",
        "--no-relative-kd",
        dest="relative_kd",
        action="store_false",
        help="Disable reverse student-to-expert KD.",
    )
    parser.set_defaults(relative_kd=True)
    parser.add_argument("--relative_kd_expert_weight", type=float, default=1.0)
    parser.add_argument(
        "--relative_kd_branches", choices=("both", "p", "s"), default="both"
    )
    parser.add_argument("--relative_kd_rms_weight", type=float, default=0.0)


def add_augmentation_switch(
    parser: argparse.ArgumentParser, *, default: bool | None
) -> None:
    """Add symmetric augmentation switches without hiding the default."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--augment", dest="augment", action="store_true")
    group.add_argument(
        "--no_augment", "--no-augment", dest="augment", action="store_false"
    )
    parser.set_defaults(augment=default)


def common_train_config_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    """Translate shared CLI arguments into ``TrainConfig`` keyword values."""
    names = (
        "n_epochs",
        "batch_size",
        "image_size",
        "num_workers",
        "raw_cache_cases",
        "lr",
        "device",
        "output_dir",
        "val_every_epochs",
        "train_case_limit",
        "val_case_limit",
        "max_train_batches",
        "max_val_batches",
    )
    values = {
        name: getattr(args, name)
        for name in names
        if hasattr(args, name) and getattr(args, name) is not None
    }
    values["amp"] = not args.no_amp
    return values


def fusion_train_config_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    """Translate alignment switches into ``TrainConfig`` keyword values."""
    return {
        "expert_adapter_variant": args.expert_adapter_variant,
        "expert_ortho_weight": args.expert_ortho_weight,
        "s_transport_mode": args.s_transport_mode,
        "s_partial_mass_fraction": args.s_partial_mass_fraction,
        "relative_kd": args.relative_kd,
        "relative_kd_expert_weight": args.relative_kd_expert_weight,
        "relative_kd_branches": args.relative_kd_branches,
        "relative_kd_rms_weight": args.relative_kd_rms_weight,
    }


def fusion_builder_kwargs(cfg: TrainConfig) -> Dict[str, Any]:
    """Return the complete alignment configuration for module construction."""
    return {
        "route_prior_p_mean": cfg.route_prior_p_mean,
        "route_prior_concentration": cfg.route_prior_concentration,
        "route_spatial_basis_grid_size": cfg.route_spatial_basis_grid_size,
        "route_spatial_basis_sigma": cfg.route_spatial_basis_sigma,
        "ot_feature_weight": cfg.ot_feature_weight,
        "ot_coordinate_weight": cfg.ot_coordinate_weight,
        "ot_coordinate_radius": cfg.ot_coordinate_radius,
        "p_ot_semantic_weight": cfg.p_ot_semantic_weight,
        "s_gain_mode": cfg.s_gain_mode,
        "s_gain_temperature": cfg.s_gain_temperature,
        "p_ot_epsilon": cfg.p_ot_epsilon,
        "s_ot_epsilon": cfg.s_ot_epsilon,
        "s_ot_rho_base": cfg.s_ot_rho_base,
        "s_ot_rho_expert": cfg.s_ot_rho_expert,
        "ot_sinkhorn_iterations": cfg.ot_sinkhorn_iterations,
        "ot_max_grid_size": cfg.ot_max_grid_size,
        "s_transport_mode": cfg.s_transport_mode,
        "s_partial_mass_fraction": cfg.s_partial_mass_fraction,
        "relative_kd": cfg.relative_kd,
        "relative_kd_expert_weight": cfg.relative_kd_expert_weight,
        "relative_kd_rms_weight": cfg.relative_kd_rms_weight,
        "relative_kd_branches": cfg.relative_kd_branches,
        "expert_adapter_variant": cfg.expert_adapter_variant,
        "remove_res5_expert_branch_norm": cfg.remove_res5_expert_branch_norm,
    }


def record_source_metadata(cfg: TrainConfig, repo_dir: str) -> None:
    """Record the exact Git source state without making Git mandatory."""
    repo_dir = os.path.abspath(repo_dir)
    try:
        cfg.source_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, text=True
        ).strip()
        cfg.source_branch = subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=repo_dir, text=True
        ).strip()
        cfg.source_tracked_dirty = subprocess.run(
            ["git", "diff", "--quiet"], cwd=repo_dir, check=False
        ).returncode != 0
    except (OSError, subprocess.CalledProcessError):
        pass
