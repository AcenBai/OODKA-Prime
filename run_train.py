#!/usr/bin/env python3
"""Entry point: OODKA training."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from oodka.config import TrainConfig, ensure_nnunet_on_path, ensure_biomedparse_on_path
from oodka.models.prompts import build_text_prompts_for_dataset
from oodka.train.cli import (
    add_common_training_arguments,
    add_fusion_training_arguments,
    common_train_config_kwargs,
    fusion_builder_kwargs,
    fusion_train_config_kwargs,
    record_source_metadata,
)
from oodka.train.model_builder import load_frozen_backbones, build_fusion_modules, build_prompt_features
from oodka.train.engine import OODKATrainer


def main():
    parser = argparse.ArgumentParser(description="OODKA training")
    add_common_training_arguments(
        parser,
        device_default="cuda:0",
        n_epochs_default=100,
        batch_size_default=1,
        image_size_default=512,
        num_workers_default=2,
        output_required=False,
        raw_cache_cases_default=2,
    )
    add_fusion_training_arguments(parser)
    parser.add_argument("--dataset_name", type=str, default="Dataset009_CT_OOD")
    parser.add_argument("--nnunet_trainer_tag", type=str, default="nnUNetTrainer_500epochs__nnUNetPlans__2d")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--block_z", type=int, default=4,
                        help="Number Z of consecutive slices per block")
    parser.add_argument("--norm_mode", type=str, default="ct", choices=("ct", "mri"))
    parser.add_argument(
        "--biomedparse_preproc_dir",
        type=str,
        default="",
        help=(
            "Offline BiomedParse npz/pkl store aligned to nnUNet geometry; "
            "required for cropped/resampled MRI datasets"
        ),
    )
    parser.add_argument("--low_percentile", type=float, default=1.0)
    parser.add_argument("--high_percentile", type=float, default=99.0)
    parser.add_argument(
        "--lr_schedule",
        choices=("constant", "cosine"),
        default="constant",
    )
    parser.add_argument("--lr_warmup_epochs", type=int, default=0)
    parser.add_argument("--min_lr_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--w_seg", type=float, default=3.0)
    parser.add_argument("--w_ae", type=float, default=0.2)
    parser.add_argument("--w_ort", type=float, default=0.3)
    parser.add_argument("--w_route", type=float, default=1e-3)
    parser.add_argument("--route_warmup_epochs", type=int, default=5)
    parser.add_argument("--w_p_ot", type=float, default=0.1)
    parser.add_argument("--w_s_ot", type=float, default=0.1)
    parser.add_argument("--p_ot_start_epoch", type=int, default=2)
    parser.add_argument("--s_ot_start_epoch", type=int, default=3)
    parser.add_argument("--ot_warmup_epochs", type=int, default=5)
    parser.add_argument("--ot_max_grid_size", type=int, default=32)
    parser.add_argument("--resume_checkpoint", type=str, default="")
    args = parser.parse_args()

    shared_config = common_train_config_kwargs(args)
    shared_config.update(fusion_train_config_kwargs(args))
    cfg = TrainConfig(
        dataset_name=args.dataset_name,
        nnunet_trainer_tag=args.nnunet_trainer_tag,
        fold=args.fold,
        block_z=args.block_z,
        norm_mode=args.norm_mode,
        biomedparse_preproc_dir=args.biomedparse_preproc_dir,
        use_aligned_biomedparse_preprocessing=bool(
            args.biomedparse_preproc_dir
        ),
        low_percentile=args.low_percentile,
        high_percentile=args.high_percentile,
        lr_schedule=args.lr_schedule,
        lr_warmup_epochs=args.lr_warmup_epochs,
        min_lr_ratio=args.min_lr_ratio,
        seed=args.seed,
        w_seg=args.w_seg,
        w_ae=args.w_ae,
        w_ort=args.w_ort,
        w_route=args.w_route,
        route_warmup_epochs=args.route_warmup_epochs,
        w_p_ot=args.w_p_ot,
        w_s_ot=args.w_s_ot,
        p_ot_start_epoch=args.p_ot_start_epoch,
        s_ot_start_epoch=args.s_ot_start_epoch,
        ot_warmup_epochs=args.ot_warmup_epochs,
        ot_max_grid_size=args.ot_max_grid_size,
        resume_checkpoint=args.resume_checkpoint,
        **shared_config,
    )
    cfg.resolve_paths()
    record_source_metadata(cfg, os.path.dirname(os.path.abspath(__file__)))
    if cfg.resume_checkpoint:
        # Resume means exact continuation. Checkpoints before this refinement
        # used the original coordinate cost, hard-positive gain, and res5
        # branch-output normalization.
        resume_state = torch.load(cfg.resume_checkpoint, map_location="cpu")
        resume_cfg = resume_state.get("config", {})
        cfg.ot_coordinate_weight = float(
            resume_cfg.get("ot_coordinate_weight", cfg.ot_coordinate_weight)
        )
        cfg.ot_coordinate_radius = float(
            resume_cfg.get("ot_coordinate_radius", 0.0)
        )
        cfg.s_gain_mode = str(
            resume_cfg.get("s_gain_mode", "hard_positive")
        )
        cfg.s_gain_temperature = float(
            resume_cfg.get(
                "s_gain_temperature", cfg.s_gain_temperature
            )
        )
        cfg.remove_res5_expert_branch_norm = bool(
            resume_cfg.get("remove_res5_expert_branch_norm", False)
        )
        cfg.expert_adapter_variant = str(
            resume_cfg.get("expert_adapter_variant", "legacy")
        )
        cfg.relative_kd = bool(resume_cfg.get("relative_kd", False))
        cfg.relative_kd_expert_weight = float(
            resume_cfg.get("relative_kd_expert_weight", 1.0)
        )
        cfg.expert_ortho_weight = float(
            resume_cfg.get("expert_ortho_weight", 1.0)
        )
        cfg.relative_kd_branches = str(
            resume_cfg.get("relative_kd_branches", "both")
        )
        cfg.relative_kd_rms_weight = float(
            resume_cfg.get("relative_kd_rms_weight", 0.0)
        )
        cfg.s_transport_mode = str(
            resume_cfg.get("s_transport_mode", "unbalanced")
        )
        cfg.s_partial_mass_fraction = float(
            resume_cfg.get("s_partial_mass_fraction", 0.5)
        )
        if "route_prior_p_mean" not in resume_cfg:
            raise ValueError(
                "This branch uses a spatial Beta router and cannot resume a "
                "legacy scalar-router checkpoint. Start a new experiment."
            )
        cfg.route_prior_p_mean = float(
            resume_cfg.get("route_prior_p_mean", cfg.route_prior_p_mean)
        )
        cfg.route_prior_concentration = float(
            resume_cfg.get(
                "route_prior_concentration",
                cfg.route_prior_concentration,
            )
        )
        cfg.route_spatial_basis_grid_size = int(
            resume_cfg.get(
                "route_spatial_basis_grid_size",
                cfg.route_spatial_basis_grid_size,
            )
        )
        cfg.route_spatial_basis_sigma = float(
            resume_cfg.get(
                "route_spatial_basis_sigma",
                cfg.route_spatial_basis_sigma,
            )
        )
    if cfg.dataset_name == "Dataset011_MYO_LGE_BC_OOD":
        if cfg.norm_mode != "mri":
            raise ValueError(
                "Dataset011_MYO_LGE_BC_OOD requires --norm_mode mri"
            )
        if not cfg.use_aligned_biomedparse_preprocessing:
            raise ValueError(
                "Dataset011_MYO_LGE_BC_OOD is cropped by nnUNet and requires "
                "--biomedparse_preproc_dir with aligned npz/pkl files"
            )
    if cfg.lr_warmup_epochs < 0:
        raise ValueError("--lr_warmup_epochs must be non-negative")
    if not 0.0 <= cfg.min_lr_ratio <= 1.0:
        raise ValueError("--min_lr_ratio must be in [0,1]")
    device = torch.device(cfg.device)

    print("=" * 60)
    print("OODKA Training")
    print("=" * 60)
    print(f"Dataset: {cfg.dataset_name}")
    print(f"Device:  {cfg.device}")
    print(f"Output:  {cfg.output_dir}")
    print()

    print("Loading frozen backbones...")
    model_nnunet, model_biomedparse = load_frozen_backbones(cfg.nnunet_model_dir, cfg.fold, device)

    text_prompts, prompt_to_class_id = build_text_prompts_for_dataset(dataset_name=cfg.dataset_name)
    P = len(text_prompts)
    print(f"Prompts: {P} classes")

    print("Building prompt features...")
    prompt_features = build_prompt_features(model_biomedparse, text_prompts, device)

    print("Building fusion modules...")
    text_dim = int(prompt_features["class_emb"].shape[-1])
    fusion_modules = build_fusion_modules(
        model_nnunet,
        model_biomedparse,
        P,
        device,
        text_dim=text_dim,
        **fusion_builder_kwargs(cfg),
    )
    n_params = sum(p.numel() for m in fusion_modules.values() for p in m.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    trainer = OODKATrainer(
        cfg=cfg,
        model_nnunet=model_nnunet,
        model_biomedparse=model_biomedparse,
        fusion_modules=fusion_modules,
        prompt_features=prompt_features,
        prompt_to_class_id=prompt_to_class_id,
        P=P,
    )
    trainer.train()


if __name__ == "__main__":
    main()
