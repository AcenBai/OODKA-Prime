#!/usr/bin/env python3
"""Train shared OODKA with full-image anatomy and predicted-MYO ROI refinement."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from oodka.config import TrainConfig
from oodka.models.prompts import (
    MYOPS_LGE_ROI_ANATOMY_GROUPS,
    MYOPS_LGE_ROI_ANATOMY_PROMPTS,
    MYOPS_LGE_ROI_REFINEMENT_GROUPS,
    MYOPS_LGE_ROI_REFINEMENT_PROMPTS,
    MYOPS_LGE_ROI_V2_REFINEMENT_GROUPS,
    MYOPS_LGE_ROI_V2_REFINEMENT_PROMPTS,
)
from oodka.train.lge_roi_engine import LGEROIMixedTrainer
from oodka.train.model_builder import (
    build_fusion_modules,
    build_prompt_features,
    load_frozen_backbones,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--roi_threshold", type=float, default=0.3)
    parser.add_argument("--roi_expand", type=float, default=1.25)
    parser.add_argument("--roi_fallback", choices=("full", "center"), default="full")
    parser.add_argument("--roi_refresh_every", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_every_epochs", type=int, default=5)
    parser.add_argument("--train_case_limit", type=int, default=0)
    parser.add_argument("--val_case_limit", type=int, default=0)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_val_batches", type=int, default=0)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument(
        "--v2",
        action="store_true",
        help="ROI predicts LV/RV/normal/scar-edema with hard spatial switching.",
    )
    parser.add_argument("--no_augment", action="store_true")
    parser.add_argument("--test_best_on_improvement", action="store_true")
    parser.add_argument("--best_test_device", default="cuda:2")
    args = parser.parse_args()

    cfg = TrainConfig(
        dataset_name="Dataset011_MYO_LGE_BC_OOD",
        fold=0,
        block_z=1,
        batch_size=args.batch_size,
        image_size=args.image_size,
        norm_mode="mri",
        pseudo_rgb_mode="center_repeat",
        biomedparse_preproc_dir=(
            "/data4/baihexiang/SegMan/Distangler3/distangler3_output/"
            "biomedparse_preprocessed_Dataset011_MYO_LGE_BC_OOD"
        ),
        use_aligned_biomedparse_preprocessing=True,
        n_epochs=args.n_epochs,
        num_workers=args.num_workers,
        raw_cache_cases=max(2, min(8, args.batch_size)),
        lr=args.lr,
        lr_schedule="cosine",
        lr_warmup_epochs=0,
        min_lr_ratio=0.05,
        w_seg=1.0,
        w_ae=0.2,
        w_ort=0.3,
        w_route=1e-3,
        w_p_ot=0.1,
        w_s_ot=0.1,
        p_ot_start_epoch=2,
        s_ot_start_epoch=3,
        ot_warmup_epochs=5,
        lge_roi_two_pass=True,
        roi_warmup_epochs=args.warmup_epochs,
        roi_threshold=args.roi_threshold,
        roi_expand=args.roi_expand,
        roi_fallback=args.roi_fallback,
        roi_refresh_every=args.roi_refresh_every,
        lambda_anchor=1.0,
        lambda_refine=1.0,
        roi_v2_hard_switch=args.v2,
        roi_visibility_min_coverage=0.01,
        roi_jitter_center_fraction=(
            0.03 if args.v2 and not args.no_augment else 0.0
        ),
        roi_jitter_scale_min=1.0,
        roi_jitter_scale_max=(
            1.15 if args.v2 and not args.no_augment else 1.0
        ),
        lge_augment=args.v2 and not args.no_augment,
        augment_rotation_degrees=10.0,
        augment_scale_min=0.95,
        augment_scale_max=1.05,
        augment_translation_fraction=0.05,
        augment_horizontal_flip_probability=0.5,
        augment_vertical_flip_probability=0.2,
        augment_intensity_probability=0.8,
        best_test_on_improvement=args.test_best_on_improvement,
        best_test_device=args.best_test_device,
        best_test_batch_size=args.batch_size,
        amp=not args.no_amp,
        device=args.device,
        output_dir=args.output_dir,
        val_every_epochs=args.val_every_epochs,
        train_case_limit=args.train_case_limit,
        val_case_limit=args.val_case_limit,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
    )
    cfg.resolve_paths()
    cfg.source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__), text=True
    ).strip()
    cfg.source_branch = subprocess.check_output(
        ["git", "branch", "--show-current"],
        cwd=os.path.dirname(__file__), text=True,
    ).strip()
    cfg.source_tracked_dirty = subprocess.run(
        ["git", "diff", "--quiet"], cwd=os.path.dirname(__file__)
    ).returncode != 0

    device = torch.device(cfg.device)
    print(f"Loading frozen backbones on {device} ...")
    model_nnunet, model_biomedparse = load_frozen_backbones(
        cfg.nnunet_model_dir, cfg.fold, device
    )
    anatomy_features = build_prompt_features(
        model_biomedparse, MYOPS_LGE_ROI_ANATOMY_PROMPTS, device
    )
    refinement_prompts = (
        MYOPS_LGE_ROI_V2_REFINEMENT_PROMPTS
        if args.v2 else MYOPS_LGE_ROI_REFINEMENT_PROMPTS
    )
    refinement_groups = (
        MYOPS_LGE_ROI_V2_REFINEMENT_GROUPS
        if args.v2 else MYOPS_LGE_ROI_REFINEMENT_GROUPS
    )
    refinement_features = build_prompt_features(
        model_biomedparse, refinement_prompts, device
    )
    fusion_modules = build_fusion_modules(
        model_nnunet,
        model_biomedparse,
        len(MYOPS_LGE_ROI_ANATOMY_GROUPS),
        device,
        text_dim=int(anatomy_features["class_emb"].shape[-1]),
        route_prior_p_mean=cfg.route_prior_p_mean,
        route_prior_concentration=cfg.route_prior_concentration,
        route_spatial_basis_grid_size=cfg.route_spatial_basis_grid_size,
        route_spatial_basis_sigma=cfg.route_spatial_basis_sigma,
        ot_feature_weight=cfg.ot_feature_weight,
        ot_coordinate_weight=cfg.ot_coordinate_weight,
        ot_coordinate_radius=cfg.ot_coordinate_radius,
        p_ot_semantic_weight=cfg.p_ot_semantic_weight,
        s_gain_mode=cfg.s_gain_mode,
        s_gain_temperature=cfg.s_gain_temperature,
        p_ot_epsilon=cfg.p_ot_epsilon,
        s_ot_epsilon=cfg.s_ot_epsilon,
        s_ot_rho_base=cfg.s_ot_rho_base,
        s_ot_rho_expert=cfg.s_ot_rho_expert,
        ot_sinkhorn_iterations=cfg.ot_sinkhorn_iterations,
        ot_max_grid_size=cfg.ot_max_grid_size,
        remove_res5_expert_branch_norm=cfg.remove_res5_expert_branch_norm,
    )
    prompt_texts = {
        "anatomy": MYOPS_LGE_ROI_ANATOMY_PROMPTS,
        "refinement": refinement_prompts,
    }
    trainer = LGEROIMixedTrainer(
        cfg=cfg,
        model_nnunet=model_nnunet,
        model_biomedparse=model_biomedparse,
        fusion_modules=fusion_modules,
        anatomy_prompt_features=anatomy_features,
        refinement_prompt_features=refinement_features,
        anatomy_groups=MYOPS_LGE_ROI_ANATOMY_GROUPS,
        refinement_groups=refinement_groups,
        prompt_texts=prompt_texts,
    )
    trainer.train()


if __name__ == "__main__":
    main()
