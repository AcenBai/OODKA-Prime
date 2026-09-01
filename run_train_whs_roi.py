#!/usr/bin/env python3
"""Train shared OODKA with whole-heart localization and seven-class ROI refinement."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from oodka.config import TrainConfig
from oodka.models.prompts import (
    WHS_CT_ROI_LOCALIZATION_PROMPTS,
    WHS_CT_ROI_REFINEMENT_PROMPTS,
    WHS_CT_GV_LOCALIZATION_PROMPTS,
    WHS_CT_GV_REFINEMENT_PROMPTS,
    WHS_MRI_ROI_LOCALIZATION_PROMPTS,
    WHS_MRI_ROI_REFINEMENT_PROMPTS,
    WHS_MRI_GV_LOCALIZATION_PROMPTS,
    WHS_MRI_GV_REFINEMENT_PROMPTS,
    WHS_GV_LOCALIZATION_GROUPS,
    WHS_GV_OUTSIDE_PROMPT_MAPPING,
    WHS_GV_REFINEMENT_GROUPS,
    WHS_ROI_LOCALIZATION_GROUPS,
    WHS_ROI_REFINEMENT_GROUPS,
)
from oodka.train.lge_roi_engine import LGEROIMixedTrainer
from oodka.train.model_builder import (
    build_fusion_modules,
    build_prompt_features,
    load_frozen_backbones,
)


DATASET_SPECS = {
    "Dataset009_CT_OOD": {
        "modality": "ct",
        "default_image_size": 512,
        "localization_prompts": WHS_CT_ROI_LOCALIZATION_PROMPTS,
        "refinement_prompts": WHS_CT_ROI_REFINEMENT_PROMPTS,
        "gv_localization_prompts": WHS_CT_GV_LOCALIZATION_PROMPTS,
        "gv_refinement_prompts": WHS_CT_GV_REFINEMENT_PROMPTS,
    },
    "Dataset010_WHS_MRI_OOD": {
        "modality": "mri",
        "default_image_size": 320,
        "localization_prompts": WHS_MRI_ROI_LOCALIZATION_PROMPTS,
        "refinement_prompts": WHS_MRI_ROI_REFINEMENT_PROMPTS,
        "gv_localization_prompts": WHS_MRI_GV_LOCALIZATION_PROMPTS,
        "gv_refinement_prompts": WHS_MRI_GV_REFINEMENT_PROMPTS,
    },
}


def _source_metadata(cfg: TrainConfig) -> None:
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        cfg.source_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, text=True
        ).strip()
        cfg.source_branch = subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=repo_dir, text=True
        ).strip()
        cfg.source_tracked_dirty = (
            subprocess.run(
                ["git", "diff", "--quiet"], cwd=repo_dir, check=False
            ).returncode
            != 0
        )
    except (OSError, subprocess.CalledProcessError):
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", required=True, choices=tuple(DATASET_SPECS))
    parser.add_argument(
        "--roi_strategy",
        choices=("whole_heart", "great_vessel"),
        default="whole_heart",
    )
    parser.add_argument(
        "--selective_children_only",
        action="store_true",
        help=(
            "For great-vessel ROI training, supervise only the GV-union "
            "localizer and the AO/PA child prompts."
        ),
    )
    parser.add_argument("--device", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--block_z", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--raw_cache_cases", type=int, default=4)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--roi_threshold", type=float, default=0.3)
    parser.add_argument("--roi_expand", type=float, default=1.25)
    parser.add_argument("--roi_fallback", choices=("full", "center"), default="full")
    parser.add_argument("--roi_refresh_every", type=int, default=0)
    parser.add_argument(
        "--pseudo_rgb_mode",
        choices=("adjacent", "center_repeat"),
        default="center_repeat",
    )
    parser.add_argument(
        "--roi_prompt_loss_reduction",
        choices=("sum", "mean", "prompt_mean"),
        default="sum",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_every_epochs", type=int, default=5)
    parser.add_argument("--train_case_limit", type=int, default=0)
    parser.add_argument("--val_case_limit", type=int, default=0)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_val_batches", type=int, default=0)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--no_augment", action="store_true")
    parser.add_argument("--biomedparse_preproc_dir", default="")
    parser.add_argument(
        "--init_checkpoint",
        default="",
        help="Optional ROI checkpoint used to initialize fusion modules only.",
    )
    args = parser.parse_args()
    if args.selective_children_only and args.roi_strategy != "great_vessel":
        raise ValueError(
            "--selective_children_only requires --roi_strategy great_vessel"
        )

    spec = DATASET_SPECS[args.dataset_name]
    image_size = args.image_size or int(spec["default_image_size"])
    aligned_dir = args.biomedparse_preproc_dir or os.path.join(
        "/data4/baihexiang/SegMan/Distangler3/distangler3_output",
        f"biomedparse_preprocessed_{args.dataset_name}",
    )
    if not os.path.isdir(aligned_dir):
        raise FileNotFoundError(aligned_dir)

    cfg = TrainConfig(
        dataset_name=args.dataset_name,
        fold=0,
        block_z=args.block_z,
        batch_size=args.batch_size,
        image_size=image_size,
        norm_mode=str(spec["modality"]),
        pseudo_rgb_mode=args.pseudo_rgb_mode,
        biomedparse_preproc_dir=aligned_dir,
        use_aligned_biomedparse_preprocessing=True,
        n_epochs=args.n_epochs,
        num_workers=args.num_workers,
        raw_cache_cases=args.raw_cache_cases,
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
        roi_prompt_loss_reduction=args.roi_prompt_loss_reduction,
        roi_v2_hard_switch=True,
        roi_visibility_min_coverage=0.01,
        roi_jitter_center_fraction=0.0 if args.no_augment else 0.03,
        roi_jitter_scale_min=1.0,
        roi_jitter_scale_max=1.0 if args.no_augment else 1.15,
        lge_augment=not args.no_augment,
        augment_rotation_degrees=10.0,
        augment_scale_min=0.95,
        augment_scale_max=1.05,
        augment_translation_fraction=0.05,
        augment_horizontal_flip_probability=0.5,
        augment_vertical_flip_probability=0.2,
        augment_intensity_probability=0.8,
        amp=not args.no_amp,
        device=args.device,
        output_dir=args.output_dir,
        val_every_epochs=args.val_every_epochs,
        train_case_limit=args.train_case_limit,
        val_case_limit=args.val_case_limit,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        resume_checkpoint=args.init_checkpoint,
    )
    cfg.resolve_paths()
    _source_metadata(cfg)

    device = torch.device(cfg.device)
    print(f"Loading task teacher and BiomedParse on {device} ...")
    model_nnunet, model_biomedparse = load_frozen_backbones(
        cfg.nnunet_model_dir, cfg.fold, device
    )
    great_vessel = args.roi_strategy == "great_vessel"
    if args.selective_children_only:
        gv_localization = spec["gv_localization_prompts"]
        gv_refinement = spec["gv_refinement_prompts"]
        localization_prompts = {"1": gv_localization["6"]}
        refinement_prompts = {
            "1": gv_refinement["6"],
            "2": gv_refinement["7"],
        }
        localization_groups = ((6, 7),)
        refinement_groups = ((6,), (7,))
    else:
        localization_prompts = spec[
            "gv_localization_prompts" if great_vessel else "localization_prompts"
        ]
        refinement_prompts = spec[
            "gv_refinement_prompts" if great_vessel else "refinement_prompts"
        ]
        localization_groups = (
            WHS_GV_LOCALIZATION_GROUPS if great_vessel else WHS_ROI_LOCALIZATION_GROUPS
        )
        refinement_groups = (
            WHS_GV_REFINEMENT_GROUPS if great_vessel else WHS_ROI_REFINEMENT_GROUPS
        )
    localization_features = build_prompt_features(
        model_biomedparse, localization_prompts, device
    )
    refinement_features = build_prompt_features(
        model_biomedparse, refinement_prompts, device
    )
    fusion_modules = build_fusion_modules(
        model_nnunet,
        model_biomedparse,
        len(localization_groups),
        device,
        text_dim=int(localization_features["class_emb"].shape[-1]),
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
    if args.init_checkpoint:
        initialization = torch.load(args.init_checkpoint, map_location=device)
        missing_modules = [
            name for name in fusion_modules if name not in initialization
        ]
        if missing_modules:
            raise KeyError(
                f"Initialization checkpoint misses modules: {missing_modules}"
            )
        for name, module in fusion_modules.items():
            module.load_state_dict(initialization[name], strict=True)
        print(f"Initialized fusion modules from {args.init_checkpoint}")
    trainer = LGEROIMixedTrainer(
        cfg=cfg,
        model_nnunet=model_nnunet,
        model_biomedparse=model_biomedparse,
        fusion_modules=fusion_modules,
        anatomy_prompt_features=localization_features,
        refinement_prompt_features=refinement_features,
        anatomy_groups=localization_groups,
        refinement_groups=refinement_groups,
        prompt_texts={
            "localization": localization_prompts,
            "refinement": refinement_prompts,
        },
        roi_prompt_index=(
            0 if args.selective_children_only else 5 if great_vessel else 0
        ),
        roi_source_labels=(6, 7) if great_vessel else tuple(range(1, 8)),
        refinement_only_output=args.selective_children_only or not great_vessel,
        outside_prompt_mapping=(
            ()
            if args.selective_children_only
            else WHS_GV_OUTSIDE_PROMPT_MAPPING if great_vessel else ((0, 0), (1, 1))
        ),
        experiment_name=(
            f"WHS-{str(spec['modality']).upper()}-GV-SELECTIVE"
            if args.selective_children_only
            else (
                f"WHS-{str(spec['modality']).upper()}-GV"
                if great_vessel
                else f"WHS-{str(spec['modality']).upper()}"
            )
        ),
        checkpoint_format=(
            "oodka_whs_gv_selective_v1"
            if args.selective_children_only
            else "oodka_whs_gv_roi_v1" if great_vessel else "oodka_whs_roi_v1"
        ),
        checkpoint_prefix=(
            f"fusion_whs_{spec['modality']}_gv_selective"
            if args.selective_children_only
            else (
                f"fusion_whs_{spec['modality']}_gv_roi"
                if great_vessel
                else f"fusion_whs_{spec['modality']}_roi"
            )
        ),
    )
    trainer.train()


if __name__ == "__main__":
    main()
