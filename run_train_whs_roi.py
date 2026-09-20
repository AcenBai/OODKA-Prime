#!/usr/bin/env python3
"""Train shared OODKA with whole-heart localization and seven-class ROI refinement."""

from __future__ import annotations

import argparse
import os
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
from oodka.train.cli import (
    add_augmentation_switch,
    add_common_training_arguments,
    add_fusion_training_arguments,
    common_train_config_kwargs,
    fusion_builder_kwargs,
    fusion_train_config_kwargs,
    record_source_metadata,
)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_training_arguments(
        parser,
        device_default="cuda:0",
        n_epochs_default=100,
        batch_size_default=1,
        image_size_default=0,
        num_workers_default=4,
        output_required=True,
        raw_cache_cases_default=4,
    )
    add_fusion_training_arguments(parser)
    add_augmentation_switch(parser, default=True)
    parser.add_argument("--dataset_name", required=True, choices=tuple(DATASET_SPECS))
    parser.add_argument(
        "--roi_strategy",
        choices=("whole_heart", "great_vessel"),
        default="whole_heart",
    )
    parser.add_argument("--block_z", type=int, default=1)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--roi_threshold", type=float, default=0.3)
    parser.add_argument("--roi_expand", type=float, default=1.25)
    parser.add_argument("--roi_fallback", choices=("full", "center"), default="full")
    parser.add_argument("--roi_refresh_every", type=int, default=0)
    parser.add_argument(
        "--roi_train_source",
        choices=("predicted", "ground_truth", "full"),
        default="predicted",
        help=(
            "Use predicted, GT-oracle, or full-image ROIs throughout "
            "train/validation."
        ),
    )
    parser.add_argument(
        "--roi_transform",
        choices=("resize", "pad", "letterbox"),
        default="resize",
        help="Map an ROI crop to the fixed model canvas.",
    )
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
    parser.add_argument("--biomedparse_preproc_dir", default="")
    args = parser.parse_args()

    spec = DATASET_SPECS[args.dataset_name]
    image_size = args.image_size or int(spec["default_image_size"])
    aligned_dir = args.biomedparse_preproc_dir or os.path.join(
        "/data4/baihexiang/SegMan/Distangler3/distangler3_output",
        f"biomedparse_preprocessed_{args.dataset_name}",
    )
    if not os.path.isdir(aligned_dir):
        raise FileNotFoundError(aligned_dir)

    shared_config = common_train_config_kwargs(args)
    shared_config.update(fusion_train_config_kwargs(args))
    shared_config["image_size"] = image_size
    cfg = TrainConfig(
        dataset_name=args.dataset_name,
        fold=0,
        block_z=args.block_z,
        norm_mode=str(spec["modality"]),
        pseudo_rgb_mode=args.pseudo_rgb_mode,
        biomedparse_preproc_dir=aligned_dir,
        use_aligned_biomedparse_preprocessing=True,
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
        roi_train_source=args.roi_train_source,
        roi_transform=args.roi_transform,
        lambda_anchor=1.0,
        lambda_refine=1.0,
        roi_prompt_loss_reduction=args.roi_prompt_loss_reduction,
        roi_v2_hard_switch=True,
        roi_visibility_min_coverage=0.01,
        roi_jitter_center_fraction=0.03 if args.augment else 0.0,
        roi_jitter_scale_min=1.0,
        roi_jitter_scale_max=1.15 if args.augment else 1.0,
        lge_augment=args.augment,
        augment_rotation_degrees=10.0,
        augment_scale_min=0.95,
        augment_scale_max=1.05,
        augment_translation_fraction=0.05,
        augment_horizontal_flip_probability=0.5,
        augment_vertical_flip_probability=0.2,
        augment_intensity_probability=0.8,
        **shared_config,
    )
    cfg.resolve_paths()
    record_source_metadata(cfg, os.path.dirname(os.path.abspath(__file__)))

    device = torch.device(cfg.device)
    print(f"Loading task teacher and BiomedParse on {device} ...")
    model_nnunet, model_biomedparse = load_frozen_backbones(
        cfg.nnunet_model_dir, cfg.fold, device
    )
    great_vessel = args.roi_strategy == "great_vessel"
    localization_prompts = spec[
        "gv_localization_prompts" if great_vessel else "localization_prompts"
    ]
    refinement_prompts = spec[
        "gv_refinement_prompts" if great_vessel else "refinement_prompts"
    ]
    localization_groups = (
        WHS_GV_LOCALIZATION_GROUPS
        if great_vessel else WHS_ROI_LOCALIZATION_GROUPS
    )
    refinement_groups = (
        WHS_GV_REFINEMENT_GROUPS
        if great_vessel else WHS_ROI_REFINEMENT_GROUPS
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
        **fusion_builder_kwargs(cfg),
    )
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
        roi_prompt_index=5 if great_vessel else 0,
        roi_source_labels=(6, 7) if great_vessel else tuple(range(1, 8)),
        refinement_only_output=not great_vessel,
        outside_prompt_mapping=(
            WHS_GV_OUTSIDE_PROMPT_MAPPING
            if great_vessel else ()
        ),
        experiment_name=(
            f"WHS-{str(spec['modality']).upper()}-GV"
            if great_vessel else f"WHS-{str(spec['modality']).upper()}"
        ),
        checkpoint_format=(
            "oodka_whs_gv_roi_v1" if great_vessel else "oodka_whs_roi_v1"
        ),
        checkpoint_prefix=(
            f"fusion_whs_{spec['modality']}_gv_roi"
            if great_vessel else f"fusion_whs_{spec['modality']}_roi"
        ),
    )
    trainer.train()


if __name__ == "__main__":
    main()
