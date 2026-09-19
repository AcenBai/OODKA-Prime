#!/usr/bin/env python3
"""Two-column soft-OT token views: original vs barycentric transport.

Student view keeps the student grid: original B | projector(π, E).
Expert view keeps the expert grid: original E | projector(πᵀ, B).
P and S are written as separate figures. Each figure stacks PCA-RGB and RMS
for res2–res5.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/oodka_ot_before_after_mpl")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F

from oodka.config import TrainConfig
from oodka.data.slice_dataset import FullSliceBlockDataset
from oodka.models.feature_extraction import (
    extract_biomedparse_backbone_features_2p5d,
    extract_nnunet_features,
)
from oodka.models.prompts import build_text_prompts_for_dataset
from oodka.train.forward import (
    _compute_detached_pixel_error_maps,
    _predict_all_prompt_logits,
    _run_pixel_decoder,
)
from oodka.train.model_builder import (
    build_fusion_modules,
    build_prompt_features,
    load_frozen_backbones,
)
from oodka.utils.io_utils import find_raw_image_files

from visualize_mechanism_v3 import (
    LEVELS,
    _flatten_feature_slice,
    _normalized_joint_pca_rgb,
    _select_slice,
    _token_image,
    _token_rms,
)


def _rms_pair(
    original: torch.Tensor,
    transported: torch.Tensor,
    grid: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, float]:
    left = _token_image(_token_rms(original)[0], grid)
    right = _token_image(_token_rms(transported)[0], grid)
    vmax = max(float(np.percentile(np.concatenate([left.ravel(), right.ravel()]), 99.5)), 1e-8)
    return left, right, vmax


def _top_mass_mask(mass: torch.Tensor, grid: tuple[int, int], keep_fraction: float) -> np.ndarray:
    values = mass[0].detach().float().cpu().numpy().reshape(grid)
    if values.size == 0:
        return np.zeros(grid, dtype=bool)
    threshold = float(np.quantile(values, 1.0 - keep_fraction))
    kept = values >= max(threshold, 1e-12)
    if not kept.any():
        kept = values >= float(values.max())
    return kept


def _masked_pca_pair(
    original: torch.Tensor,
    transported: torch.Tensor,
    keep: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    grid = keep.shape
    gray = np.full((*grid, 3), 0.72, dtype=np.float32)
    if not keep.any():
        return gray.copy(), gray.copy()
    index = keep.reshape(-1)
    pca0, pca1 = _normalized_joint_pca_rgb(
        (original[:, index], transported[:, index]),
        ((int(index.sum()), 1), (int(index.sum()), 1)),
    )
    left = gray.copy()
    right = gray.copy()
    left[keep] = pca0.reshape(-1, 3)
    right[keep] = pca1.reshape(-1, 3)
    return left, right


def _masked_rms_pair(
    original: torch.Tensor,
    transported: torch.Tensor,
    keep: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    left = _token_image(_token_rms(original)[0], keep.shape)
    right = _token_image(_token_rms(transported)[0], keep.shape)
    if keep.any():
        vmax = max(float(np.percentile(np.concatenate([left[keep], right[keep]]), 99.5)), 1e-8)
    else:
        vmax = 1.0
    left = np.where(keep, left, np.nan)
    right = np.where(keep, right, np.nan)
    return left, right, vmax


def _plot_view(
    path: Path,
    *,
    title: str,
    col0: str,
    col1: str,
    rows: list[dict],
) -> None:
    figure, axes = plt.subplots(
        len(rows) * 2,
        2,
        figsize=(8.6, 2.05 * len(rows) * 2 + 0.8),
        constrained_layout=True,
    )
    rms_cmap = plt.get_cmap("magma").copy()
    rms_cmap.set_bad((0.72, 0.72, 0.72, 1.0))
    last_rms = None
    for index, row in enumerate(rows):
        pca_row = index * 2
        rms_row = index * 2 + 1
        axes[pca_row, 0].imshow(row["pca0"], interpolation="nearest")
        axes[pca_row, 1].imshow(row["pca1"], interpolation="nearest")
        last_rms = axes[rms_row, 0].imshow(
            row["rms0"],
            cmap=rms_cmap,
            vmin=0.0,
            vmax=row["rms_vmax"],
            interpolation="nearest",
        )
        axes[rms_row, 1].imshow(
            row["rms1"],
            cmap=rms_cmap,
            vmin=0.0,
            vmax=row["rms_vmax"],
            interpolation="nearest",
        )
        kept = row.get("n_keep")
        total = row.get("n_total")
        suffix = f"  {kept}/{total}" if kept is not None and total is not None else ""
        axes[pca_row, 0].set_ylabel(f"{row['level']} PCA{suffix}", fontsize=8)
        axes[rms_row, 0].set_ylabel(f"{row['level']} RMS", fontsize=9)
        if index == 0:
            axes[pca_row, 0].set_title(col0, fontsize=11)
            axes[pca_row, 1].set_title(col1, fontsize=11)
        for axis in (axes[pca_row, 0], axes[pca_row, 1], axes[rms_row, 0], axes[rms_row, 1]):
            axis.set_xticks([])
            axis.set_yticks([])
    figure.colorbar(last_rms, ax=axes[1::2, :], shrink=0.72, label="Token RMS")
    figure.suptitle(title, fontsize=13)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--case_id", default="heart_1004")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--slice_index", type=int, default=79)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--block_z", type=int, default=4)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--s_keep_fraction",
        type=float,
        default=0.2,
        help="Keep this top fraction of S tokens by received mass.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    saved = checkpoint.get("config", {})
    cfg = TrainConfig(
        device=args.device,
        block_z=args.block_z,
        num_workers=0,
        dataset_name=str(saved.get("dataset_name", "Dataset009_CT_OOD")),
        image_size=int(saved.get("image_size", 512)),
        norm_mode=str(saved.get("norm_mode", "ct")),
        pseudo_rgb_mode=str(saved.get("pseudo_rgb_mode", "adjacent")),
        low_percentile=float(saved.get("low_percentile", 1.0)),
        high_percentile=float(saved.get("high_percentile", 99.0)),
        require_no_crop=bool(saved.get("require_no_crop", True)),
        biomedparse_modality=int(saved.get("biomedparse_modality", 0)),
    )
    cfg.resolve_paths()
    if saved.get("biomedparse_preproc_dir"):
        cfg.biomedparse_preproc_dir = str(saved["biomedparse_preproc_dir"])

    images_dir = cfg.imagesTr_dir if args.split == "val" else cfg.imagesTs_dir
    labels_dir = cfg.labelsTr_dir if args.split == "val" else cfg.labelsTs_dir
    with open(cfg.dataset_json_path, encoding="utf-8") as handle:
        dataset_json = json.load(handle)
    ending = dataset_json.get("file_ending", ".nii.gz")
    image_files = find_raw_image_files(images_dir, args.case_id, ending)
    if not image_files:
        raise FileNotFoundError(args.case_id)
    gt_volume = np.asarray(
        sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(labels_dir, args.case_id + ending)))
    )
    center = _select_slice(gt_volume, "all_classes", args.slice_index)

    dataset = FullSliceBlockDataset(
        [args.case_id],
        nnunet_preproc_dir=cfg.nnunet_preproc_dir,
        images_dir=images_dir,
        labels_dir=labels_dir,
        file_ending=ending,
        image_size=cfg.image_size,
        block_z=cfg.block_z,
        norm_mode=cfg.norm_mode,
        window_level=cfg.window_level,
        window_width=cfg.window_width,
        low_percentile=cfg.low_percentile,
        high_percentile=cfg.high_percentile,
        raw_cache_cases=1,
        require_no_crop=cfg.require_no_crop,
        biomedparse_modality=cfg.biomedparse_modality,
        biomedparse_preproc_dir=cfg.biomedparse_preproc_dir,
        pseudo_rgb_mode=cfg.pseudo_rgb_mode,
    )
    record_index = next(
        index
        for index, (_case_id, z_start, valid_count) in enumerate(dataset.records)
        if z_start <= center < z_start + valid_count
    )
    item = dataset[record_index]
    center_local = center - int(item["z_start"])
    bp = item["biomedparse_image"].unsqueeze(0)
    nn_input = item["nnunet_image"].unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()
    gt_block = item["gt"].unsqueeze(0)
    valid_z = item["valid_z"].unsqueeze(0)

    model_nnunet, model_biomedparse = load_frozen_backbones(
        cfg.nnunet_model_dir, cfg.fold, device
    )
    prompts, prompt_to_class_id = build_text_prompts_for_dataset(
        dataset_name=cfg.dataset_name
    )
    prompt_features = build_prompt_features(model_biomedparse, prompts, device)
    modules = build_fusion_modules(
        model_nnunet,
        model_biomedparse,
        len(prompts),
        device,
        text_dim=int(prompt_features["class_emb"].shape[-1]),
        route_prior_p_mean=float(saved.get("route_prior_p_mean", cfg.route_prior_p_mean)),
        route_prior_concentration=float(
            saved.get("route_prior_concentration", cfg.route_prior_concentration)
        ),
        route_spatial_basis_grid_size=int(
            saved.get("route_spatial_basis_grid_size", cfg.route_spatial_basis_grid_size)
        ),
        route_spatial_basis_sigma=float(
            saved.get("route_spatial_basis_sigma", cfg.route_spatial_basis_sigma)
        ),
        ot_feature_weight=float(saved.get("ot_feature_weight", cfg.ot_feature_weight)),
        ot_coordinate_weight=float(saved.get("ot_coordinate_weight", cfg.ot_coordinate_weight)),
        ot_coordinate_radius=float(saved.get("ot_coordinate_radius", 0.0)),
        p_ot_semantic_weight=float(saved.get("p_ot_semantic_weight", cfg.p_ot_semantic_weight)),
        s_gain_mode=str(saved.get("s_gain_mode", "hard_positive")),
        s_gain_temperature=float(saved.get("s_gain_temperature", cfg.s_gain_temperature)),
        p_ot_epsilon=float(saved.get("p_ot_epsilon", cfg.p_ot_epsilon)),
        s_ot_epsilon=float(saved.get("s_ot_epsilon", cfg.s_ot_epsilon)),
        s_ot_rho_base=float(saved.get("s_ot_rho_base", cfg.s_ot_rho_base)),
        s_ot_rho_expert=float(saved.get("s_ot_rho_expert", cfg.s_ot_rho_expert)),
        ot_sinkhorn_iterations=int(saved.get("ot_sinkhorn_iterations", cfg.ot_sinkhorn_iterations)),
        ot_max_grid_size=int(saved.get("ot_max_grid_size", cfg.ot_max_grid_size)),
        s_transport_mode=str(saved.get("s_transport_mode", "unbalanced")),
        s_partial_mass_fraction=float(saved.get("s_partial_mass_fraction", 0.5)),
        expert_adapter_variant=str(saved.get("expert_adapter_variant", "legacy")),
        remove_res5_expert_branch_norm=bool(
            saved.get("remove_res5_expert_branch_norm", False)
        ),
    )
    for name, module in modules.items():
        if name in checkpoint:
            module.load_state_dict(checkpoint[name])
        module.eval()

    output_hw = tuple(int(value) for value in gt_volume.shape[-2:])
    features: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        expert_raw, _deepest, expert_logits = extract_nnunet_features(
            model_nnunet, nn_input.to(device), device, return_logits=True
        )
        embeds, student_raw = extract_biomedparse_backbone_features_2p5d(
            model_biomedparse,
            bp.to(device),
            device,
            res_names=("res2", "res3", "res4", "res5"),
        )
        for level in LEVELS:
            student = student_raw[f"res{level}"]
            expert_native = expert_raw[f"enc{level}"]
            expert_aligned = expert_native
            if expert_aligned.shape[-3:] != student.shape[-3:]:
                expert_aligned = F.interpolate(
                    expert_aligned,
                    size=student.shape[-3:],
                    mode="trilinear",
                    align_corners=False,
                )
            expert_outputs = modules[f"ae_enc{level}_to_res{level}"](expert_aligned)
            if len(expert_outputs) == 4:
                expert_p, expert_s, _p_rec, _s_rec = expert_outputs
            elif len(expert_outputs) == 3:
                expert_p, expert_s, _reconstruction = expert_outputs
            else:
                raise RuntimeError(f"Unexpected Expert adapter outputs: {len(expert_outputs)}")
            student_p, student_s = modules[f"dis_b_res{level}"](student)
            features[f"Zn{level}_p"] = expert_p
            features[f"Zn{level}_s"] = expert_s
            features[f"Zb{level}_p"] = student_p
            features[f"Zb{level}_s"] = student_s

        embeds_base = dict(embeds)
        for level in LEVELS:
            embeds_base.pop(f"res{level}", None)
        mask_p, multi_p = _run_pixel_decoder(
            model_biomedparse, embeds_base, features, "p", B=1, Dm=bp.shape[1]
        )
        mask_s, multi_s = _run_pixel_decoder(
            model_biomedparse, embeds_base, features, "s", B=1, Dm=bp.shape[1]
        )
        route = modules["beta_router"](
            prompt_features["class_emb"].detach(),
            spatial_size=mask_p.shape[-2:],
            batch_size=1,
            sample=False,
        )
        all_prompt_logits = _predict_all_prompt_logits(
            sem_seg_head=model_biomedparse.sem_seg_head,
            mask_features_p=mask_p,
            mask_features_s=mask_s,
            ms_p=multi_p,
            ms_s=multi_s,
            gate=route["gate"],
            prompt_features=prompt_features,
            B=1,
            Z=bp.shape[1],
            P=len(prompts),
            output_shape=(bp.shape[1], *output_hw),
        )
        class_ids_tensor = torch.tensor(
            [prompt_to_class_id[index] for index in range(len(prompts))],
            device=device,
            dtype=gt_block.dtype,
        )
        base_error, expert_error = _compute_detached_pixel_error_maps(
            all_prompt_logits,
            expert_logits,
            gt_block.to(device),
            valid_z.to(device),
            class_ids_tensor,
            expert_class_groups=None,
        )

        class_ids = [int(value) for value in class_ids_tensor.tolist()]
        gt_slice = gt_block[:, center_local].to(device)
        semantic = torch.stack(
            [(gt_slice == class_id).float() for class_id in class_ids], dim=1
        )
        ot_module = modules["ot_distillation"]
        panels = {
            "student_p": [],
            "student_s": [],
            "expert_p": [],
            "expert_s": [],
            "student_s_topmass": [],
            "expert_s_topmass": [],
        }
        for level in LEVELS:
            p_base = _flatten_feature_slice(features[f"Zb{level}_p"], center_local)
            s_base = _flatten_feature_slice(features[f"Zb{level}_s"], center_local)
            p_expert = _flatten_feature_slice(features[f"Zn{level}_p"], center_local)
            s_expert = _flatten_feature_slice(features[f"Zn{level}_s"], center_local)
            grid = ot_module._target_size(p_base)
            p_mass = ot_module.structure_mass(
                gt_slice, p_base, p_expert, class_ids=class_ids, target_size=grid
            )
            p_cost = ot_module.p_cost(
                p_base,
                p_expert,
                target_size=grid,
                base_semantic=semantic,
                expert_semantic=semantic,
            )
            p_transport = ot_module.balanced(p_mass["a"], p_mass["b"], p_cost["cost"])
            p_forward = ot_module.projector(
                p_transport["transport"], p_cost["expert_tokens"]
            )
            p_reverse = ot_module.projector(
                p_transport["transport"].transpose(1, 2), p_cost["base_tokens"]
            )

            s_mass = ot_module.residual_mass(
                p_base,
                s_base,
                p_expert,
                s_expert,
                base_error=base_error[:, center_local],
                expert_error=expert_error[:, center_local],
                target_size=grid,
            )
            s_cost = ot_module.s_cost(s_base, s_expert, target_size=grid)
            s_transport = ot_module.unbalanced(s_mass["a"], s_mass["b"], s_cost["cost"])
            s_forward = ot_module.projector(
                s_transport["transport"], s_cost["expert_tokens"]
            )
            s_reverse = ot_module.projector(
                s_transport["transport"].transpose(1, 2), s_cost["base_tokens"]
            )

            specs = (
                ("student_p", p_cost["base_tokens"], p_forward["teacher"], grid, None),
                ("student_s", s_cost["base_tokens"], s_forward["teacher"], grid, s_forward["received"]),
                ("expert_p", p_cost["expert_tokens"], p_reverse["teacher"], grid, None),
                ("expert_s", s_cost["expert_tokens"], s_reverse["teacher"], grid, s_reverse["received"]),
            )
            for key, original, transported, token_grid, mass in specs:
                pca0, pca1 = _normalized_joint_pca_rgb(
                    (original, transported),
                    (token_grid, token_grid),
                )
                rms0, rms1, vmax = _rms_pair(original, transported, token_grid)
                panels[key].append(
                    {
                        "level": f"res{level}",
                        "pca0": pca0,
                        "pca1": pca1,
                        "rms0": rms0,
                        "rms1": rms1,
                        "rms_vmax": vmax,
                    }
                )
                if mass is None:
                    continue
                keep = _top_mass_mask(mass, token_grid, args.s_keep_fraction)
                pca0_m, pca1_m = _masked_pca_pair(original, transported, keep)
                rms0_m, rms1_m, vmax_m = _masked_rms_pair(original, transported, keep)
                panels[f"{key}_topmass"].append(
                    {
                        "level": f"res{level}",
                        "pca0": pca0_m,
                        "pca1": pca1_m,
                        "rms0": rms0_m,
                        "rms1": rms1_m,
                        "rms_vmax": vmax_m,
                        "n_keep": int(keep.sum()),
                        "n_total": int(keep.size),
                    }
                )

    out = Path(args.output_dir)
    _plot_view(
        out / "student_p.png",
        title="Student view · P  ·  original B  |  soft T:E→B",
        col0="Original Student P",
        col1="Soft-transported Expert P",
        rows=panels["student_p"],
    )
    _plot_view(
        out / "student_s.png",
        title="Student view · S  ·  original B  |  soft T:E→B",
        col0="Original Student S",
        col1="Soft-transported Expert S",
        rows=panels["student_s"],
    )
    _plot_view(
        out / "expert_p.png",
        title="Expert view · P  ·  original E  |  soft Tᵀ:B→E",
        col0="Original Expert P",
        col1="Soft-transported Student P",
        rows=panels["expert_p"],
    )
    _plot_view(
        out / "expert_s.png",
        title="Expert view · S  ·  original E  |  soft Tᵀ:B→E",
        col0="Original Expert S",
        col1="Soft-transported Student S",
        rows=panels["expert_s"],
    )
    keep_pct = int(round(args.s_keep_fraction * 100))
    _plot_view(
        out / "student_s_top20mass.png",
        title=f"Student view · S  ·  top {keep_pct}% received mass",
        col0="Original Student S",
        col1="Soft-transported Expert S",
        rows=panels["student_s_topmass"],
    )
    _plot_view(
        out / "expert_s_top20mass.png",
        title=f"Expert view · S  ·  top {keep_pct}% received mass",
        col0="Original Expert S",
        col1="Soft-transported Student S",
        rows=panels["expert_s_topmass"],
    )
    (out / "README.json").write_text(
        json.dumps(
            {
                "case_id": args.case_id,
                "z": center,
                "transport": "barycentric projector(pi, .) and projector(pi^T, .)",
                "pca": "shared L2-normalized PCA-RGB on the two columns of each row",
                "rms": "channel RMS, shared vmax per level",
                "s_keep_fraction": args.s_keep_fraction,
                "s_topmass": (
                    "S tokens below the received-mass quantile are grayed out; "
                    "PCA/RMS vmax are fit on the kept tokens only"
                ),
                "files": [
                    "student_p.png",
                    "student_s.png",
                    "expert_p.png",
                    "expert_s.png",
                    "student_s_top20mass.png",
                    "expert_s_top20mass.png",
                ],
            },
            indent=2,
        )
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
