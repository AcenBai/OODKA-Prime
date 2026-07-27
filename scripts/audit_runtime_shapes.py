#!/usr/bin/env python3
"""Record real Dataset009 backbone/decomposer/pixel-decoder tensor shapes."""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
from torch.utils.data._utils.collate import default_collate

from oodka.config import TrainConfig
from oodka.data.slice_dataset import FullSliceBlockDataset
from oodka.models.biomedparse_helpers import parse_pixel_decoder_out
from oodka.models.feature_extraction import (
    extract_biomedparse_backbone_features_2p5d,
    extract_nnunet_features,
)
from oodka.models.prompts import build_text_prompts_for_dataset
from oodka.train.model_builder import (
    build_fusion_modules,
    build_prompt_features,
    load_frozen_backbones,
)


def _shape(value):
    return list(value.shape) if torch.is_tensor(value) else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--block_z", type=int, default=1)
    parser.add_argument(
        "--output",
        default="outputs/oodka_ot_experiments/audit/runtime_shapes.json",
    )
    args = parser.parse_args()

    cfg = TrainConfig(device=args.device, block_z=args.block_z, num_workers=0)
    cfg.resolve_paths()
    with open(cfg.dataset_json_path, encoding="utf-8") as file_handle:
        dataset_json = json.load(file_handle)
    with open(cfg.splits_final_json, encoding="utf-8") as file_handle:
        split = json.load(file_handle)[cfg.fold]
    case_id = split["train"][0]
    dataset = FullSliceBlockDataset(
        [case_id],
        nnunet_preproc_dir=cfg.nnunet_preproc_dir,
        images_dir=cfg.imagesTr_dir,
        labels_dir=cfg.labelsTr_dir,
        file_ending=dataset_json.get("file_ending", ".nii.gz"),
        image_size=cfg.image_size,
        block_z=cfg.block_z,
        norm_mode=cfg.norm_mode,
        raw_cache_cases=1,
    )
    batch = default_collate([dataset[0]])
    device = torch.device(args.device)
    model_nnunet, model_biomedparse = load_frozen_backbones(
        cfg.nnunet_model_dir, cfg.fold, device
    )
    prompts, _mapping = build_text_prompts_for_dataset(dataset_name=cfg.dataset_name)
    prompt_features = build_prompt_features(model_biomedparse, prompts, device)
    modules = build_fusion_modules(
        model_nnunet,
        model_biomedparse,
        len(prompts),
        device,
        text_dim=int(prompt_features["class_emb"].shape[-1]),
    )

    nn_images = batch["nnunet_image"].to(device)
    bp_images = batch["biomedparse_image"].to(device)
    nn_blocks = nn_images.permute(0, 2, 1, 3, 4).contiguous()
    with torch.no_grad():
        enc, deepest = extract_nnunet_features(model_nnunet, nn_blocks, device)
        bp_embeds, bp_5d = extract_biomedparse_backbone_features_2p5d(
            model_biomedparse, bp_images, device
        )
        flat_nn = nn_images.reshape(
            nn_images.shape[0] * nn_images.shape[1],
            nn_images.shape[2],
            nn_images.shape[3],
            nn_images.shape[4],
        )
        expert_logits = model_nnunet(flat_nn)
        if isinstance(expert_logits, (list, tuple)):
            expert_logits = expert_logits[0]

        aligned = {}
        decomposed = {}
        for level in [2, 3, 4, 5]:
            expert = enc[f"enc{level}"]
            base = bp_5d[f"res{level}"]
            if expert.shape[-3:] != base.shape[-3:]:
                expert = torch.nn.functional.interpolate(
                    expert,
                    size=base.shape[-3:],
                    mode="trilinear",
                    align_corners=False,
                )
            p_e, s_e, _rp, _rs = modules[f"ae_enc{level}_to_res{level}"](expert)
            p_b, s_b = modules[f"dis_b_res{level}"](base)
            aligned[f"expert_res{level}"] = _shape(expert)
            decomposed[f"P_E_res{level}"] = _shape(p_e)
            decomposed[f"S_E_res{level}"] = _shape(s_e)
            decomposed[f"P_B_res{level}"] = _shape(p_b)
            decomposed[f"S_B_res{level}"] = _shape(s_b)

        decoder_shapes = {}
        for branch in ["P", "S"]:
            injected = dict(bp_embeds)
            for level in [2, 3, 4, 5]:
                feature = (
                    modules[f"dis_b_res{level}"](bp_5d[f"res{level}"])[0]
                    if branch == "P"
                    else modules[f"dis_b_res{level}"](bp_5d[f"res{level}"])[1]
                )
                injected[f"res{level}"] = (
                    feature.permute(0, 2, 1, 3, 4)
                    .reshape(-1, feature.shape[1], feature.shape[3], feature.shape[4])
                )
            mask, multi_scale = parse_pixel_decoder_out(
                model_biomedparse.sem_seg_head.pixel_decoder.forward_features(injected)
            )
            decoder_shapes[branch] = {
                "mask": _shape(mask),
                "multi_scale": [_shape(feature) for feature in multi_scale],
            }

    report = {
        "dataset": cfg.dataset_name,
        "case_id": case_id,
        "input": {
            "nnunet": _shape(nn_images),
            "biomedparse": _shape(bp_images),
            "gt": _shape(batch["gt"]),
        },
        "nnunet_encoder": {name: _shape(value) for name, value in enc.items()},
        "nnunet_deepest": _shape(deepest),
        "nnunet_logits": _shape(expert_logits),
        "biomedparse_backbone_flat": {
            name: _shape(value) for name, value in bp_embeds.items()
        },
        "biomedparse_backbone_5d": {
            name: _shape(value) for name, value in bp_5d.items()
        },
        "aligned_expert": aligned,
        "decomposed": decomposed,
        "pixel_decoder": decoder_shapes,
        "prompt_features": {
            name: _shape(value)
            for name, value in prompt_features.items()
            if torch.is_tensor(value)
        },
    }
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file_handle:
        json.dump(report, file_handle, indent=2)
    print(json.dumps(report, indent=2))
    print(f"saved={output_path}")


if __name__ == "__main__":
    main()
