#!/usr/bin/env python3
"""Measure UOT acceptance under deliberately incompatible expert features."""

from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/oodka_uot_mpl")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib.pyplot as plt
import torch
from torch.utils.data._utils.collate import default_collate

from oodka.config import TrainConfig
from oodka.data.slice_dataset import FullSliceBlockDataset
from oodka.models.prompts import build_text_prompts_for_dataset
from oodka.train.forward import forward_one_batch
from oodka.train.model_builder import (
    build_fusion_modules,
    build_prompt_features,
    load_frozen_backbones,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--block_z", type=int, default=6)
    parser.add_argument("--case_id", default="heart_1004")
    parser.add_argument(
        "--output",
        default="outputs/oodka_ot_experiments/interpretability/uot_stress.json",
    )
    args = parser.parse_args()

    cfg = TrainConfig(device=args.device, block_z=args.block_z, num_workers=0)
    cfg.resolve_paths()
    with open(cfg.dataset_json_path, encoding="utf-8") as file_handle:
        dataset_json = json.load(file_handle)
    dataset = FullSliceBlockDataset(
        [args.case_id],
        nnunet_preproc_dir=cfg.nnunet_preproc_dir,
        images_dir=cfg.imagesTr_dir,
        labels_dir=cfg.labelsTr_dir,
        file_ending=dataset_json.get("file_ending", ".nii.gz"),
        image_size=cfg.image_size,
        block_z=cfg.block_z,
        norm_mode=cfg.norm_mode,
        raw_cache_cases=1,
    )
    selected = None
    for index in range(len(dataset)):
        item = dataset[index]
        if (item["gt"] > 0).any():
            selected = item
            break
    if selected is None:
        raise RuntimeError(f"No foreground block found for {args.case_id}")
    batch = default_collate([selected])

    device = torch.device(args.device)
    model_nnunet, model_biomedparse = load_frozen_backbones(
        cfg.nnunet_model_dir, cfg.fold, device
    )
    prompts, mapping = build_text_prompts_for_dataset(dataset_name=cfg.dataset_name)
    prompt_features = build_prompt_features(model_biomedparse, prompts, device)
    modules = build_fusion_modules(
        model_nnunet,
        model_biomedparse,
        len(prompts),
        device,
        text_dim=int(prompt_features["class_emb"].shape[-1]),
    )
    checkpoint = torch.load(args.checkpoint, map_location=device)
    for name, module in modules.items():
        if name in checkpoint:
            module.load_state_dict(checkpoint[name])
        module.eval()

    results = {}
    modes = [
        None,
        "spatial_shift",
        "channel_reverse",
        "s_cost_offset_0p25",
        "s_cost_offset_0p5",
        "s_cost_offset_1p0",
        "s_cost_offset_2p0",
    ]
    with torch.no_grad():
        for mode in modes:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _loss, logs = forward_one_batch(
                    batch_data=batch,
                    block_shape=[cfg.block_z, cfg.image_size, cfg.image_size],
                    prompt_features=prompt_features,
                    P=len(prompts),
                    prompt_to_class_id=mapping,
                    w_seg=0.0,
                    w_ae=0.0,
                    w_ort=0.0,
                    w_route=0.0,
                    w_p_ot=1.0,
                    w_s_ot=1.0,
                    model_nnunet=model_nnunet,
                    model_biomedparse=model_biomedparse,
                    fusion_modules=modules,
                    device=device,
                    route_sample=False,
                    ot_expert_perturbation=mode,
                )
            name = mode or "clean"
            results[name] = {
                f"res{level}": {
                    "s_accept_ratio": logs[f"ot_res{level}_s_accept_ratio"],
                    "s_rejected": logs[f"ot_res{level}_s_rejected"],
                    "s_cost": logs[f"ot_res{level}_s_cost"],
                    "p_cost": logs[f"ot_res{level}_p_cost"],
                    "s_cost_offset": logs[f"ot_res{level}_s_cost_offset"],
                }
                for level in [2, 3, 4, 5]
            }
    controlled = [
        (0.0, "clean"),
        (0.25, "s_cost_offset_0p25"),
        (0.5, "s_cost_offset_0p5"),
        (1.0, "s_cost_offset_1p0"),
        (2.0, "s_cost_offset_2p0"),
    ]
    monotonic_by_level = {}
    acceptance_by_level = {}
    for level in [2, 3, 4, 5]:
        values = [
            results[name][f"res{level}"]["s_accept_ratio"]
            for _offset, name in controlled
        ]
        acceptance_by_level[f"res{level}"] = values
        monotonic_by_level[f"res{level}"] = all(
            right < left for left, right in zip(values, values[1:])
        )
    results["_controlled_cost_validation"] = {
        "offsets": [offset for offset, _name in controlled],
        "acceptance_by_level": acceptance_by_level,
        "strictly_decreasing_by_level": monotonic_by_level,
        "all_levels_pass": all(monotonic_by_level.values()),
    }
    if not results["_controlled_cost_validation"]["all_levels_pass"]:
        raise AssertionError(
            "UOT acceptance did not decrease at every level under controlled cost shifts"
        )

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file_handle:
        json.dump(results, file_handle, indent=2)
    offsets = results["_controlled_cost_validation"]["offsets"]
    for level in [2, 3, 4, 5]:
        plt.plot(
            offsets,
            acceptance_by_level[f"res{level}"],
            marker="o",
            label=f"res{level}",
        )
    plt.xlabel("Controlled additive S-OT cost penalty")
    plt.ylabel("UOT accepted expert mass ratio")
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    figure_path = os.path.splitext(output_path)[0] + ".png"
    plt.savefig(figure_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(json.dumps(results, indent=2))
    print(f"saved={output_path}")
    print(f"saved={figure_path}")


if __name__ == "__main__":
    main()
