#!/usr/bin/env python3
"""Compare the production all-prompt predictor with a serial reference."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate

from oodka.config import TrainConfig
from oodka.data import FullSliceBlockDataset
from oodka.models.biomedparse_helpers import (
    parse_pixel_decoder_out,
    run_biomedparse_predictor_override,
    select_best_mask_from_queries,
)
from oodka.models.feature_extraction import (
    extract_biomedparse_backbone_features_2p5d,
    extract_nnunet_features,
)
from oodka.models.prompts import build_text_prompts_for_dataset
from oodka.train.forward import (
    _compute_segmentation_loss_and_metrics,
    _predict_all_prompt_logits,
)
from oodka.train.model_builder import (
    build_fusion_modules,
    build_prompt_features,
    load_frozen_backbones,
)


def _slice_prompt(prompt_features: dict, prompt_index: int) -> dict:
    """Test-only reconstruction of the previous single-prompt path."""
    sliced = prompt_features.copy()
    grounding = prompt_features["grounding_tokens"]
    sliced["grounding_tokens"] = grounding[:, prompt_index : prompt_index + 1]
    if torch.is_tensor(prompt_features.get("class_emb")):
        sliced["class_emb"] = prompt_features["class_emb"][
            prompt_index : prompt_index + 1
        ]
    if torch.is_tensor(prompt_features.get("num_prompts")):
        value = prompt_features["num_prompts"]
        sliced["num_prompts"] = torch.ones(1, device=value.device, dtype=value.dtype)
    return sliced


def _serial_segmentation_loss(logits, gt, valid_z, class_ids):
    """Previous prompt-loop loss retained only as a validation oracle."""
    total = torch.tensor(0.0, device=logits.device)
    weight_total = torch.tensor(0.0, device=logits.device)
    valid = (gt != -1).float() * valid_z[:, :, None, None].float()
    valid_mask = valid > 0.5
    for prompt_index, class_id in enumerate(class_ids):
        class_logits = logits[:, prompt_index]
        target = (gt == class_id).float()
        valid_voxels = valid_mask.flatten(1).sum(1).float()
        foreground = ((target > 0.5) & valid_mask).flatten(1).sum(1).float()
        empty = foreground < valid_voxels * 0.0005

        bce = F.binary_cross_entropy_with_logits(
            class_logits, target, reduction="none"
        )
        denominator = valid.sum(dim=(1, 2, 3)).clamp_min(1.0)
        bce = (bce * valid).sum(dim=(1, 2, 3)) / denominator
        probabilities = torch.sigmoid(class_logits) * valid
        target = target * valid
        intersection = (probabilities * target).sum(dim=(1, 2, 3))
        union = probabilities.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice_loss = 1.0 - (2.0 * intersection + 1e-6) / (union + 1e-6)
        pair_loss = torch.where(empty, bce, bce + dice_loss)
        weights = torch.where(
            empty, torch.full_like(pair_loss, 0.5), torch.ones_like(pair_loss)
        )
        total = total + (weights * pair_loss).sum()
        weight_total = weight_total + weights.sum()
    return total / weight_total.clamp_min(1e-6)


@torch.no_grad()
def _serial_prompt_logits(
    *,
    sem_seg_head,
    mask_features_p,
    mask_features_s,
    ms_p,
    ms_s,
    tau_mask,
    tau_ms_list,
    prompt_features,
    B,
    Z,
    P,
    output_shape,
):
    """Previous P-call implementation retained only as a validation oracle."""
    N = B * Z
    logits_per_prompt = []
    for prompt_index in range(P):
        tau = tau_mask[:, prompt_index]
        channels = tau.shape[-1]
        tau_2d = tau[:, None].expand(B, Z, channels).reshape(N, channels, 1, 1)
        mask_features = tau_2d * mask_features_p + (1.0 - tau_2d) * mask_features_s

        multi_scale_features = []
        for private, shared, tau_level in zip(ms_p, ms_s, tau_ms_list):
            tau_prompt = tau_level[:, prompt_index]
            channels = tau_prompt.shape[-1]
            tau_level_2d = (
                tau_prompt[:, None]
                .expand(B, Z, channels)
                .reshape(N, channels, 1, 1)
            )
            multi_scale_features.append(
                tau_level_2d * private + (1.0 - tau_level_2d) * shared
            )

        prediction = run_biomedparse_predictor_override(
            sem_seg_head,
            multi_scale_features,
            mask_features,
            _slice_prompt(prompt_features, prompt_index),
        )
        logits = select_best_mask_from_queries(
            prediction["pred_gmasks"], prediction.get("object_existence")
        )
        logits_per_prompt.append(
            F.interpolate(
                logits.reshape(B, Z, logits.shape[-2], logits.shape[-1]).unsqueeze(1),
                size=output_shape,
                mode="trilinear",
                align_corners=False,
            ).squeeze(1)
        )
    return torch.stack(logits_per_prompt, dim=1)


def _load_decoded_features(cfg, batch, device):
    model_nnunet, model_biomedparse = load_frozen_backbones(
        cfg.nnunet_model_dir, cfg.fold, device
    )
    text_prompts, prompt_to_class_id = build_text_prompts_for_dataset(
        dataset_name=cfg.dataset_name
    )
    prompt_features = build_prompt_features(model_biomedparse, text_prompts, device)
    P = len(text_prompts)
    fusion_modules = build_fusion_modules(
        model_nnunet, model_biomedparse, P, device
    )
    checkpoint = torch.load(
        os.path.join(cfg.output_dir, "fusion_disentangle_best.pth"),
        map_location=device,
    )
    for name, module in fusion_modules.items():
        module.load_state_dict(checkpoint[name])
        module.eval()

    nn_images = batch["nnunet_image"].to(device)
    bp_images = batch["biomedparse_image"].to(device)
    valid_z = batch["valid_z"].to(device)
    B, Z = nn_images.shape[:2]
    _enc, deepest = extract_nnunet_features(
        model_nnunet, nn_images.permute(0, 2, 1, 3, 4).contiguous(), device
    )
    image_embeds, res3d = extract_biomedparse_backbone_features_2p5d(
        model_biomedparse,
        bp_images,
        device,
        res_names=("res2", "res3", "res4", "res5"),
    )
    for name in ["res2", "res3", "res4", "res5"]:
        image_embeds.pop(name, None)

    disentangled = {}
    for level in [2, 3, 4, 5]:
        private, shared = fusion_modules[f"dis_b_res{level}"](res3d[f"res{level}"])
        disentangled[f"Zb{level}_p"] = private
        disentangled[f"Zb{level}_s"] = shared

    N = B * Z

    def decode(branch):
        injected = dict(image_embeds)
        for level in [2, 3, 4, 5]:
            feature = disentangled[f"Zb{level}_{branch}"]
            injected[f"res{level}"] = (
                feature.permute(0, 2, 1, 3, 4)
                .reshape(N, feature.shape[1], feature.shape[3], feature.shape[4])
                .contiguous()
            )
        return parse_pixel_decoder_out(
            model_biomedparse.sem_seg_head.pixel_decoder.forward_features(injected)
        )

    mask_p, ms_p = decode("p")
    mask_s, ms_s = decode("s")
    mu, _ = fusion_modules["class_query_pooler"](deepest, valid_z=valid_z)
    tau = fusion_modules["gate_net"](mu)
    return {
        "model_biomedparse": model_biomedparse,
        "prompt_features": prompt_features,
        "prompt_to_class_id": prompt_to_class_id,
        "P": P,
        "B": B,
        "Z": Z,
        "mask_p": mask_p,
        "mask_s": mask_s,
        "ms_p": ms_p,
        "ms_s": ms_s,
        "tau_mask": tau["mask"],
        "tau_ms": tau["ms"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--block_z", type=int, default=2)
    parser.add_argument("--block_index", type=int, default=44)
    args = parser.parse_args()

    cfg = TrainConfig(device=args.device, block_z=args.block_z, num_workers=0)
    cfg.resolve_paths()
    dataset_json = json.load(open(cfg.dataset_json_path, encoding="utf-8"))
    split = json.load(open(cfg.splits_final_json, encoding="utf-8"))
    case_id = split[cfg.fold]["train"][0]
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
    batch = default_collate(
        [
            dataset[min(args.block_index + offset, len(dataset) - 1)]
            for offset in range(args.batch_size)
        ]
    )
    device = torch.device(cfg.device)
    values = _load_decoded_features(cfg, batch, device)
    predictor = values["model_biomedparse"].sem_seg_head.predictor
    previous_boltzmann = predictor.boltzmann_sampling["do_boltzmann"]
    predictor.boltzmann_sampling["do_boltzmann"] = False

    common = dict(
        sem_seg_head=values["model_biomedparse"].sem_seg_head,
        mask_features_p=values["mask_p"],
        mask_features_s=values["mask_s"],
        ms_p=values["ms_p"],
        ms_s=values["ms_s"],
        tau_mask=values["tau_mask"],
        tau_ms_list=values["tau_ms"],
        prompt_features=values["prompt_features"],
        B=values["B"],
        Z=values["Z"],
        P=values["P"],
        output_shape=(cfg.block_z, cfg.image_size, cfg.image_size),
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    serial = _serial_prompt_logits(**common)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    serial_seconds = time.perf_counter() - start

    start = time.perf_counter()
    parallel = _predict_all_prompt_logits(**common)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    parallel_seconds = time.perf_counter() - start
    predictor.boltzmann_sampling["do_boltzmann"] = previous_boltzmann

    difference = (serial - parallel).abs()
    print(
        f"serial={tuple(serial.shape)} parallel={tuple(parallel.shape)} "
        f"max_abs={difference.max().item():.8g} mean_abs={difference.mean().item():.8g}"
    )
    print(
        f"serial_seconds={serial_seconds:.4f} "
        f"parallel_seconds={parallel_seconds:.4f}"
    )
    torch.testing.assert_close(serial, parallel, rtol=2e-4, atol=2e-4)

    class_ids = torch.tensor(
        [values["prompt_to_class_id"][index] for index in range(values["P"])],
        device=device,
        dtype=batch["gt"].dtype,
    )
    serial_loss = _serial_segmentation_loss(
        serial, batch["gt"].to(device), batch["valid_z"].to(device), class_ids
    )
    parallel_loss = _compute_segmentation_loss_and_metrics(
        parallel, batch["gt"].to(device), batch["valid_z"].to(device), class_ids
    )[0]
    torch.testing.assert_close(serial_loss, parallel_loss, rtol=2e-5, atol=2e-5)
    print(
        f"serial_loss={serial_loss.item():.8f} "
        f"parallel_loss={parallel_loss.item():.8f}"
    )
    print("serial/parallel prompt equivalence: OK")


if __name__ == "__main__":
    main()
