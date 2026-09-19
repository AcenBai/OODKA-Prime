#!/usr/bin/env python3
"""CLI orchestration for OODKA mechanism-v3 visualization."""

from __future__ import annotations

try:
    from . import mechanism_v3_plots as _plots
except ImportError:  # Direct execution: python scripts/visualize_mechanism_v3.py
    import mechanism_v3_plots as _plots

# Re-export plotting helpers for existing analysis scripts that import this file.
globals().update({name: getattr(_plots, name) for name in _plots.__all__})

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--case_id", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--slice_index", type=int, default=None)
    parser.add_argument(
        "--selection",
        choices=("largest_foreground", "all_classes"),
        default="largest_foreground",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--block_z", type=int, default=6)
    parser.add_argument(
        "--shared_color_scales",
        default=None,
        help=(
            "JSON with fixed raw/decomposed/pixel_decoder vmax values and "
            "per-level Student relative P1/P99"
        ),
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Root under which <case>_z<slice> is created",
    )
    parser.add_argument(
        "--lge_roi_pass",
        choices=("none", "pass1", "pass2"),
        default="none",
        help=(
            "Reproduce the LGE ROI-v2 full-image anatomy pass or predicted-ROI "
            "refinement pass while retaining the complete mechanism-v3 dump."
        ),
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    checkpoint_cfg = checkpoint.get("config", {})
    cfg = TrainConfig(
        device=args.device,
        block_z=args.block_z,
        num_workers=0,
        dataset_name=str(checkpoint_cfg.get("dataset_name", "Dataset009_CT_OOD")),
        image_size=int(checkpoint_cfg.get("image_size", 512)),
        norm_mode=str(checkpoint_cfg.get("norm_mode", "ct")),
        pseudo_rgb_mode=str(checkpoint_cfg.get("pseudo_rgb_mode", "adjacent")),
        low_percentile=float(checkpoint_cfg.get("low_percentile", 1.0)),
        high_percentile=float(checkpoint_cfg.get("high_percentile", 99.0)),
        require_no_crop=bool(checkpoint_cfg.get("require_no_crop", True)),
        biomedparse_modality=int(checkpoint_cfg.get("biomedparse_modality", 0)),
    )
    cfg.resolve_paths()
    if checkpoint_cfg.get("biomedparse_preproc_dir"):
        cfg.biomedparse_preproc_dir = str(checkpoint_cfg["biomedparse_preproc_dir"])
    coordinate_weight = float(
        checkpoint_cfg.get("ot_coordinate_weight", cfg.ot_coordinate_weight)
    )
    coordinate_radius = float(
        checkpoint_cfg.get("ot_coordinate_radius", 0.0)
    )
    s_gain_mode = str(
        checkpoint_cfg.get("s_gain_mode", "hard_positive")
    )
    s_gain_temperature = float(
        checkpoint_cfg.get("s_gain_temperature", cfg.s_gain_temperature)
    )
    s_transport_mode = str(
        checkpoint_cfg.get("s_transport_mode", "unbalanced")
    )
    s_partial_mass_fraction = float(
        checkpoint_cfg.get("s_partial_mass_fraction", 0.5)
    )
    expert_adapter_variant = str(
        checkpoint_cfg.get("expert_adapter_variant", "legacy")
    )
    remove_res5_expert_branch_norm = bool(
        checkpoint_cfg.get("remove_res5_expert_branch_norm", False)
    )
    images_dir = cfg.imagesTr_dir if args.split == "val" else cfg.imagesTs_dir
    labels_dir = cfg.labelsTr_dir if args.split == "val" else cfg.labelsTs_dir
    with open(cfg.dataset_json_path, encoding="utf-8") as handle:
        dataset_json = json.load(handle)
    ending = dataset_json.get("file_ending", ".nii.gz")
    label_names = {
        int(class_id): str(name)
        for name, class_id in dataset_json.get("labels", {}).items()
        if int(class_id) > 0
    }
    image_files = find_raw_image_files(images_dir, args.case_id, ending)
    if not image_files:
        raise FileNotFoundError(args.case_id)
    label_path = os.path.join(labels_dir, args.case_id + ending)
    if args.lge_roi_pass != "none":
        aligned_path = Path(cfg.biomedparse_preproc_dir) / f"{args.case_id}.npz"
        with np.load(aligned_path) as aligned:
            raw = np.asarray(
                aligned["data"][cfg.biomedparse_modality], dtype=np.float32
            )
            gt_volume = np.asarray(aligned["seg"][0], dtype=np.int16)
    else:
        raw = np.asarray(sitk.GetArrayFromImage(sitk.ReadImage(image_files[0])))
        gt_volume = np.asarray(sitk.GetArrayFromImage(sitk.ReadImage(label_path)))
    center = _select_slice(gt_volume, args.selection, args.slice_index)

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
    nn_input = (
        item["nnunet_image"]
        .unsqueeze(0)
        .permute(0, 2, 1, 3, 4)
        .contiguous()
    )
    gt_block = item["gt"].unsqueeze(0)
    valid_z = item["valid_z"].unsqueeze(0)

    model_nnunet, model_biomedparse = load_frozen_backbones(
        cfg.nnunet_model_dir, cfg.fold, device
    )
    expert_class_groups = None
    if args.lge_roi_pass == "pass1":
        prompts = MYOPS_LGE_ROI_ANATOMY_PROMPTS
        expert_class_groups = ((3,), (5,), (1, 2, 4))
        prompt_to_class_id = {index: index + 1 for index in range(len(prompts))}
        label_names = {1: "LV", 2: "RV", 3: "total_myo"}
    elif args.lge_roi_pass == "pass2":
        prompts = MYOPS_LGE_ROI_V2_REFINEMENT_PROMPTS
        expert_class_groups = ((3,), (5,), (4,), (1, 2))
        prompt_to_class_id = {index: index + 1 for index in range(len(prompts))}
        label_names = {1: "LV", 2: "RV", 3: "normal_myo", 4: "scar_edema"}
    else:
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
        route_prior_p_mean=cfg.route_prior_p_mean,
        route_prior_concentration=cfg.route_prior_concentration,
        route_spatial_basis_grid_size=cfg.route_spatial_basis_grid_size,
        route_spatial_basis_sigma=cfg.route_spatial_basis_sigma,
        ot_feature_weight=cfg.ot_feature_weight,
        ot_coordinate_weight=coordinate_weight,
        ot_coordinate_radius=coordinate_radius,
        p_ot_semantic_weight=cfg.p_ot_semantic_weight,
        s_gain_mode=s_gain_mode,
        s_gain_temperature=s_gain_temperature,
        p_ot_epsilon=cfg.p_ot_epsilon,
        s_ot_epsilon=cfg.s_ot_epsilon,
        s_ot_rho_base=cfg.s_ot_rho_base,
        s_ot_rho_expert=cfg.s_ot_rho_expert,
        ot_sinkhorn_iterations=cfg.ot_sinkhorn_iterations,
        ot_max_grid_size=cfg.ot_max_grid_size,
        s_transport_mode=s_transport_mode,
        s_partial_mass_fraction=s_partial_mass_fraction,
        expert_adapter_variant=expert_adapter_variant,
        remove_res5_expert_branch_norm=remove_res5_expert_branch_norm,
    )
    required = [
        *(f"ae_enc{level}_to_res{level}" for level in LEVELS),
        *(f"dis_b_res{level}" for level in LEVELS),
        "beta_router",
    ]
    missing = [name for name in required if name not in checkpoint]
    if missing:
        raise KeyError(f"Full fusion checkpoint is missing {missing}")
    for name, module in modules.items():
        if name in checkpoint:
            module.load_state_dict(checkpoint[name])
        module.eval()

    roi_metadata = None
    if args.lge_roi_pass == "pass2":
        anatomy_prompt_features = build_prompt_features(
            model_biomedparse, MYOPS_LGE_ROI_ANATOMY_PROMPTS, device
        )
        with torch.no_grad():
            anatomy_embeds, anatomy_student = (
                extract_biomedparse_backbone_features_2p5d(
                    model_biomedparse,
                    bp.to(device),
                    device,
                    res_names=("res2", "res3", "res4", "res5"),
                )
            )
            anatomy_features = {}
            for level in LEVELS:
                p_value, s_value = modules[f"dis_b_res{level}"](
                    anatomy_student[f"res{level}"]
                )
                anatomy_features[f"Zb{level}_p"] = p_value
                anatomy_features[f"Zb{level}_s"] = s_value
            anatomy_base = dict(anatomy_embeds)
            for level in LEVELS:
                anatomy_base.pop(f"res{level}", None)
            anatomy_p, anatomy_multi_p = _run_pixel_decoder(
                model_biomedparse, anatomy_base, anatomy_features, "p", B=1, Dm=1
            )
            anatomy_s, anatomy_multi_s = _run_pixel_decoder(
                model_biomedparse, anatomy_base, anatomy_features, "s", B=1, Dm=1
            )
            anatomy_route = modules["beta_router"](
                anatomy_prompt_features["class_emb"].detach(),
                spatial_size=anatomy_p.shape[-2:],
                batch_size=1,
                sample=False,
            )
            anatomy_logits = _predict_all_prompt_logits(
                sem_seg_head=model_biomedparse.sem_seg_head,
                mask_features_p=anatomy_p,
                mask_features_s=anatomy_s,
                ms_p=anatomy_multi_p,
                ms_s=anatomy_multi_s,
                gate=anatomy_route["gate"],
                prompt_features=anatomy_prompt_features,
                B=1,
                Z=1,
                P=len(MYOPS_LGE_ROI_ANATOMY_PROMPTS),
                output_shape=(1, *gt_block.shape[-2:]),
            )
        roi_generator = ROIGenerator(
            threshold=float(checkpoint_cfg.get("roi_threshold", 0.3)),
            expand=float(checkpoint_cfg.get("roi_expand", 1.25)),
            fallback=str(checkpoint_cfg.get("roi_fallback", "full")),
        )
        roi = roi_generator.from_probability(
            torch.sigmoid(anatomy_logits[0, 2, 0]).detach()
        )
        cropped = crop_and_resize_batch(
            {
                "nnunet_image": item["nnunet_image"].unsqueeze(0),
                "biomedparse_image": item["biomedparse_image"].unsqueeze(0),
                "gt": item["gt"].unsqueeze(0),
            },
            [roi],
        )
        bp = cropped["biomedparse_image"]
        nn_input = cropped["nnunet_image"].permute(0, 2, 1, 3, 4).contiguous()
        gt_block = cropped["gt"]
        valid_z = torch.ones((1, 1), dtype=torch.bool)
        image = bp[0, 0, 0].cpu().numpy()
        center_local = 0
        roi_metadata = {
            "x0": roi.x0, "y0": roi.y0, "x1": roi.x1, "y1": roi.y1,
            "width": roi.width, "height": roi.height,
            "fallback": roi.fallback,
        }

    if expert_class_groups is not None:
        gt_block = remap_grouped_labels(gt_block, expert_class_groups)
        gt = gt_block[0, center_local].cpu().numpy()

    if args.lge_roi_pass != "none":
        output_hw = tuple(int(value) for value in gt_block.shape[-2:])
        image = bp[0, center_local, 0].cpu().numpy()
        gt = gt_block[0, center_local].cpu().numpy()
    else:
        output_hw = tuple(int(value) for value in gt_volume.shape[-2:])
        image = raw[center]
        gt = gt_volume[center]
    raw_maps: dict[int, dict[str, np.ndarray]] = {}
    branch_maps: dict[int, dict[str, np.ndarray]] = {}
    decoder_maps: dict[int, dict[str, np.ndarray]] = {}
    features: dict[str, torch.Tensor] = {}

    with torch.no_grad():
        expert_raw, _deepest, expert_logits = extract_nnunet_features(
            model_nnunet,
            nn_input.to(device),
            device,
            return_logits=True,
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
            expert_outputs = modules[f"ae_enc{level}_to_res{level}"](
                expert_aligned
            )
            if len(expert_outputs) == 4:
                expert_p, expert_s, _p_rec, _s_rec = expert_outputs
            elif len(expert_outputs) == 3:
                expert_p, expert_s, _reconstruction = expert_outputs
            else:
                raise RuntimeError(
                    "Unexpected Expert adapter output count: "
                    f"{len(expert_outputs)}"
                )
            student_p, student_s = modules[f"dis_b_res{level}"](student)
            features[f"Zn{level}_p"] = expert_p
            features[f"Zn{level}_s"] = expert_s
            features[f"Zb{level}_p"] = student_p
            features[f"Zb{level}_s"] = student_s

            raw_maps[level] = {
                "expert": _rms_5d(expert_native, center_local, output_hw),
                "student": _rms_5d(student, center_local, output_hw),
            }
            level_branch = {
                "expert_p": _rms_5d(expert_p, center_local, output_hw),
                "expert_s": _rms_5d(expert_s, center_local, output_hw),
                "student_p": _rms_5d(student_p, center_local, output_hw),
                "student_s": _rms_5d(student_s, center_local, output_hw),
            }
            level_branch["expert_s_share"] = level_branch["expert_s"] / (
                level_branch["expert_p"] + level_branch["expert_s"] + 1e-8
            )
            level_branch["student_s_share"] = level_branch["student_s"] / (
                level_branch["student_p"] + level_branch["student_s"] + 1e-8
            )
            branch_maps[level] = level_branch

        embeds_base = dict(embeds)
        for level in LEVELS:
            embeds_base.pop(f"res{level}", None)
        mask_p, multi_p = _run_pixel_decoder(
            model_biomedparse,
            embeds_base,
            features,
            "p",
            B=1,
            Dm=bp.shape[1],
        )
        mask_s, multi_s = _run_pixel_decoder(
            model_biomedparse,
            embeds_base,
            features,
            "s",
            B=1,
            Dm=bp.shape[1],
        )
        decoder_features = {
            2: {"p": mask_p, "s": mask_s},
            3: {"p": multi_p[2], "s": multi_s[2]},
            4: {"p": multi_p[1], "s": multi_s[1]},
            5: {"p": multi_p[0], "s": multi_s[0]},
        }
        for level in LEVELS:
            p_map = _rms_4d(
                decoder_features[level]["p"], center_local, output_hw
            )
            s_map = _rms_4d(
                decoder_features[level]["s"], center_local, output_hw
            )
            decoder_maps[level] = {
                "p": p_map,
                "s": s_map,
                "s_share": s_map / (p_map + s_map + 1e-8),
            }
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
            [
                prompt_to_class_id[prompt_index]
                for prompt_index in range(len(prompts))
            ],
            device=device,
            dtype=gt_block.dtype,
        )
        base_error, expert_error = _compute_detached_pixel_error_maps(
            all_prompt_logits,
            expert_logits,
            gt_block.to(device),
            valid_z.to(device),
            class_ids_tensor,
            expert_class_groups=expert_class_groups,
        )

    case_root = Path(args.output_root).resolve() / f"{args.case_id}_z{center:04d}"
    representation_dir = case_root / "representation"
    ot_dir = case_root / "ot"
    decision_dir = case_root / "decision"
    shared_scales_path = (
        Path(args.shared_color_scales).resolve()
        if args.shared_color_scales is not None
        else None
    )
    shared_scales = None
    if shared_scales_path is not None:
        with shared_scales_path.open(encoding="utf-8") as handle:
            shared_scales = json.load(handle)
    scales = _plot_representation(
        representation_dir,
        image,
        gt,
        label_names,
        raw_maps,
        branch_maps,
        decoder_maps,
        scales_override=shared_scales,
    )

    class_ids = [int(value) for value in class_ids_tensor.tolist()]
    gt_slice = gt_block[:, center_local].to(device)
    semantic = torch.stack(
        [(gt_slice == class_id).float() for class_id in class_ids], dim=1
    )
    ot_module = modules["ot_distillation"]
    ot_dir.mkdir(parents=True, exist_ok=True)
    kd_alignment_dir = ot_dir / "kd_alignment"
    ot_summary = {
        "case_id": args.case_id,
        "z": center,
        "energy_definition": "sqrt(mean(channel^2))",
        "transport_matrix_display": (
            "block-summed to at most 16x16 spatial grids; matrix values shown "
            "relative to a uniform plan"
        ),
        "levels": {},
    }
    ot_npz: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for level in LEVELS:
            p_base = _flatten_feature_slice(features[f"Zb{level}_p"], center_local)
            s_base = _flatten_feature_slice(features[f"Zb{level}_s"], center_local)
            p_expert = _flatten_feature_slice(
                features[f"Zn{level}_p"], center_local
            )
            s_expert = _flatten_feature_slice(
                features[f"Zn{level}_s"], center_local
            )
            grid = ot_module._target_size(p_base)

            p_mass = ot_module.structure_mass(
                gt_slice,
                p_base,
                p_expert,
                class_ids=class_ids,
                target_size=grid,
            )
            p_cost = ot_module.p_cost(
                p_base,
                p_expert,
                target_size=grid,
                base_semantic=semantic,
                expert_semantic=semantic,
            )
            p_transport = ot_module.balanced(
                p_mass["a"], p_mass["b"], p_cost["cost"]
            )
            p_teacher = ot_module.projector(
                p_transport["transport"], p_cost["expert_tokens"]
            )
            p_components = _feature_cost_components(
                p_cost["base_tokens"],
                p_cost["expert_tokens"],
                grid,
                semantic,
                feature_weight=cfg.ot_feature_weight,
                coordinate_weight=coordinate_weight,
                coordinate_radius=coordinate_radius,
                semantic_weight=cfg.p_ot_semantic_weight,
            )
            p_maps = _plot_p_ot(
                ot_dir / f"res{level}_p_transport.png",
                level=level,
                image=image,
                gt=gt,
                label_names=label_names,
                grid=grid,
                base_tokens=p_cost["base_tokens"],
                expert_tokens=p_cost["expert_tokens"],
                transport=p_transport["transport"],
                mass_a=p_mass["a"],
                mass_b=p_mass["b"],
                teacher_tokens=p_teacher["teacher"],
                components=p_components,
                total_cost=p_cost["cost"],
            )
            p_rgb = _shared_pca_rgb(
                p_cost["expert_tokens"],
                p_teacher["teacher"],
                p_cost["base_tokens"],
                grid,
            )
            _plot_token_layout(
                ot_dir / f"res{level}_p_token_layout.png",
                p_rgb,
                title=f"res{level} P: shared PCA-RGB token arrangement",
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
            s_cost = ot_module.s_cost(
                s_base,
                s_expert,
                target_size=grid,
            )
            s_transport = ot_module.unbalanced(
                s_mass["a"], s_mass["b"], s_cost["cost"]
            )
            s_teacher = ot_module.projector(
                s_transport["transport"], s_cost["expert_tokens"]
            )
            if level == 2:
                _plot_spatial_transport_suite(
                    ot_dir / "spatial_transport",
                    level=level,
                    image=image,
                    gt=gt,
                    label_names=label_names,
                    grid=grid,
                    p_transport=p_transport["transport"],
                    s_transport=s_transport["transport"],
                    p_mass=p_mass["a"],
                    s_mass_a=s_mass["a"],
                    s_mass_b=s_mass["b"],
                    s_output=s_transport,
                )
            branch_diagnostics = {
                "p": _transport_direction_diagnostics(
                    student_tokens=p_cost["base_tokens"],
                    expert_tokens=p_cost["expert_tokens"],
                    transport=p_transport["transport"],
                    projector=ot_module.projector,
                    grid=grid,
                ),
                "s": _transport_direction_diagnostics(
                    student_tokens=s_cost["base_tokens"],
                    expert_tokens=s_cost["expert_tokens"],
                    transport=s_transport["transport"],
                    projector=ot_module.projector,
                    grid=grid,
                ),
            }
            _plot_kd_direction_alignment(
                kd_alignment_dir,
                level=level,
                branch_diagnostics=branch_diagnostics,
            )
            s_maps = _plot_s_ot(
                ot_dir / f"res{level}_s_transport.png",
                level=level,
                image=image,
                gt=gt,
                label_names=label_names,
                grid=grid,
                base_tokens=s_cost["base_tokens"],
                expert_tokens=s_cost["expert_tokens"],
                transport=s_transport["transport"],
                mass=s_mass,
                output=s_transport,
                teacher_tokens=s_teacher["teacher"],
            )
            s_rgb, s_visibility, s_pca_info = _received_weighted_pca_rgb(
                s_cost["expert_tokens"],
                s_teacher["teacher"],
                s_cost["base_tokens"],
                s_transport["received"],
                grid,
            )
            _plot_token_layout(
                ot_dir / f"res{level}_s_token_layout.png",
                s_rgb,
                title=f"res{level} S: shared PCA-RGB token arrangement",
                received_visibility=s_visibility,
                pca_info=s_pca_info,
            )
            ot_summary["levels"][f"res{level}"] = {
                "grid": list(grid),
                "p": {
                    "transport_total": float(
                        p_transport["transport"].sum().item()
                    ),
                    "row_l1_error": float(
                        p_transport["row_error"].mean().item()
                    ),
                    "col_l1_error": float(
                        p_transport["col_error"].mean().item()
                    ),
                    "transport_cost": float(
                        p_transport["cost"].mean().item()
                    ),
                    "mean_student_teacher_residual": float(
                        p_maps["residual"].mean()
                    ),
                },
                "s": {
                    "transport_total": float(
                        s_transport["transport"].sum().item()
                    ),
                    "accept_ratio": float(
                        s_transport["accept_ratio"].mean().item()
                    ),
                    "rejected_total": float(
                        s_transport["rejected"].sum().item()
                    ),
                    "overused_total": float(
                        s_transport.get(
                            "overused",
                            (
                                s_transport["transported"] - s_mass["b"]
                            ).clamp_min(0.0),
                        ).sum().item()
                    ),
                    "transport_mode": s_transport_mode,
                    "partial_mass_fraction": s_partial_mass_fraction,
                    "transport_cost": float(
                        s_transport["cost"].mean().item()
                    ),
                    "mean_student_teacher_residual": float(
                        s_maps["residual"].mean()
                    ),
                    "received_signal_definition": "U_i = sum_j pi_ij E_j",
                    "token_layout": s_pca_info,
                },
                "kd_direction_alignment": {
                    branch: diagnostics["summary"]
                    for branch, diagnostics in branch_diagnostics.items()
                },
            }
            for branch, diagnostics in branch_diagnostics.items():
                for name in (
                    "forward_cos",
                    "reverse_cos",
                    "same_position_cos",
                    "received_mass",
                    "normalized_entropy",
                    "confidence",
                ):
                    ot_npz[f"res{level}_{branch}_kd_{name}"] = diagnostics[name]
            for name, value in p_maps.items():
                ot_npz[f"res{level}_p_{name}"] = value
            for name, value in s_maps.items():
                ot_npz[f"res{level}_s_{name}"] = value
    _save_json(ot_dir / "ot_summary.json", ot_summary)
    np.savez_compressed(ot_dir / "ot_derived_maps.npz", gt=gt, **ot_npz)

    _plot_decision(
        decision_dir,
        image=image,
        gt=gt,
        label_names=label_names,
        decoder_maps=decoder_maps,
        decoder_features=decoder_features,
        route=route,
        class_ids=class_ids,
        logits=all_prompt_logits,
        z_index=center_local,
        output_hw=output_hw,
        decoder_vmax=scales["pixel_decoder"]["vmax"],
    )

    checkpoint_path = Path(args.checkpoint).resolve()
    manifest = {
        "schema": "oodka-mechanism-visualization-v3",
        "case_id": args.case_id,
        "split": args.split,
        "z": center,
        "selection": args.selection,
        "shared_color_scales": (
            str(shared_scales_path) if shared_scales_path is not None else None
        ),
        "labels_present": sorted(
            int(value) for value in np.unique(gt) if int(value) > 0
        ),
        "git_commit": _git_commit(),
        "generator": str(Path(__file__).resolve()),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "block_z": args.block_z,
        "dataset_name": cfg.dataset_name,
        "norm_mode": cfg.norm_mode,
        "pseudo_rgb_mode": cfg.pseudo_rgb_mode,
        "input_geometry": (
            "predicted total_myo ROI resized to model canvas"
            if args.lge_roi_pass == "pass2"
            else "aligned full-slice"
            if args.lge_roi_pass == "pass1"
            else "raw full-slice"
        ),
        "lge_roi_pass": args.lge_roi_pass,
        "predicted_roi": roi_metadata,
        "block_z_start": int(item["z_start"]),
        "center_local": center_local,
        "color_scales": scales,
        "refinements": [
            "Student raw RMS relative contrast with per-level cross-case P1/P99",
            "S-UOT displays true received expert signal U_i = sum_j pi_ij E_j",
            "S token layout uses received-mass-weighted PCA and visibility",
        ],
        "directories": {
            "representation": str(representation_dir),
            "ot": str(ot_dir),
            "decision": str(decision_dir),
        },
    }
    _save_json(case_root / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    print(f"saved={case_root}")


if __name__ == "__main__":
    main()
