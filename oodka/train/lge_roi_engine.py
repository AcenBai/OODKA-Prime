"""Mixed full-image anatomy and predicted-ROI pathology training for LGE."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List, Sequence

import numpy as np
import torch
from tqdm import tqdm

from ..data.lge_roi import (
    ROICache,
    ROICoordinates,
    ROIGenerator,
    augment_lge_batch,
    crop_and_resize_batch,
    hard_switch_foreground_logits,
    jitter_roi,
    remap_grouped_labels,
    restore_roi_logits,
    roi_diagnostics,
    roi_prompt_visibility,
)
from ..models.prompts import (
    MYOPS_LGE_ROI_FINAL_NAMES,
)
from ..utils.io_utils import maybe_mkdir_p
from .engine import OODKATrainer, learning_rate_scale, load_fold_cases, set_seed
from .forward import forward_one_batch, predict_block_logits_per_class


def _compact_mapping(count: int) -> Dict[int, int]:
    return {index: index + 1 for index in range(count)}


def _case_dice_from_counts(counts: dict, class_count: int) -> tuple[float, dict]:
    per_class = {}
    all_values = []
    for class_id in range(1, class_count + 1):
        values = []
        for case_counts in counts.values():
            gt_count = case_counts["gt"][class_id]
            if gt_count <= 0:
                continue
            denominator = case_counts["pred"][class_id] + gt_count
            value = (
                2.0 * case_counts["intersection"][class_id] / denominator
                if denominator > 0 else 0.0
            )
            values.append(value)
        per_class[class_id] = float(np.mean(values)) if values else None
        all_values.extend(values)
    return (float(np.mean(all_values)) if all_values else 0.0), per_class


def _update_case_counts(
    counts: dict,
    case_ids: Sequence[str],
    prediction: torch.Tensor,
    target: torch.Tensor,
    class_count: int,
) -> None:
    prediction = prediction.detach().cpu()
    target = target.detach().cpu()
    for index, case_id in enumerate(case_ids):
        entry = counts.setdefault(
            str(case_id),
            {
                "intersection": np.zeros(class_count + 1, dtype=np.float64),
                "pred": np.zeros(class_count + 1, dtype=np.float64),
                "gt": np.zeros(class_count + 1, dtype=np.float64),
            },
        )
        valid = target[index] >= 0
        for class_id in range(1, class_count + 1):
            pred_mask = (prediction[index] == class_id) & valid
            gt_mask = (target[index] == class_id) & valid
            entry["intersection"][class_id] += float((pred_mask & gt_mask).sum())
            entry["pred"][class_id] += float(pred_mask.sum())
            entry["gt"][class_id] += float(gt_mask.sum())


class LGEROIMixedTrainer(OODKATrainer):
    """One shared model, two forwards, one joint optimizer update."""

    def __init__(
        self,
        *,
        anatomy_prompt_features: dict,
        refinement_prompt_features: dict,
        anatomy_groups: Sequence[Sequence[int]],
        refinement_groups: Sequence[Sequence[int]],
        prompt_texts: dict,
        **kwargs,
    ) -> None:
        anatomy_count = len(anatomy_groups)
        super().__init__(
            prompt_features=anatomy_prompt_features,
            prompt_to_class_id=_compact_mapping(anatomy_count),
            P=anatomy_count,
            **kwargs,
        )
        if self.cfg.block_z != 1:
            raise ValueError("LGE ROI mixed training requires block_z=1")
        self.anatomy_prompt_features = anatomy_prompt_features
        self.refinement_prompt_features = refinement_prompt_features
        self.anatomy_groups = tuple(tuple(int(v) for v in g) for g in anatomy_groups)
        self.refinement_groups = tuple(
            tuple(int(v) for v in g) for g in refinement_groups
        )
        self.prompt_texts = prompt_texts
        self.roi_generator = ROIGenerator(
            threshold=self.cfg.roi_threshold,
            expand=self.cfg.roi_expand,
            fallback=self.cfg.roi_fallback,
        )
        self.roi_cache: ROICache | None = None
        self.history = []

    def _branch_forward(
        self,
        batch_data: dict,
        *,
        prompt_features: dict,
        groups: Sequence[Sequence[int]],
        regularizer_scale: float,
        w_route: float,
        w_p_ot: float,
        w_s_ot: float,
    ):
        branch_data = dict(batch_data)
        branch_data["gt"] = remap_grouped_labels(batch_data["gt"], groups)
        return forward_one_batch(
            batch_data=branch_data,
            block_shape=self.block_shape,
            prompt_features=prompt_features,
            P=len(groups),
            prompt_to_class_id=_compact_mapping(len(groups)),
            w_seg=1.0,
            w_ae=self.cfg.w_ae * regularizer_scale,
            w_ort=self.cfg.w_ort * regularizer_scale,
            model_nnunet=self.model_nnunet,
            model_biomedparse=self.model_biomedparse,
            fusion_modules=self.fusion_modules,
            device=self.device,
            w_route=w_route * regularizer_scale,
            w_p_ot=w_p_ot * regularizer_scale,
            w_s_ot=w_s_ot * regularizer_scale,
            expert_class_groups=groups,
            prompt_loss_reduction="sum",
            return_logits=True,
        )

    def _online_rois(self, anatomy_logits: torch.Tensor) -> List[ROICoordinates]:
        probabilities = torch.sigmoid(anatomy_logits[:, 2, 0].detach())
        return [self.roi_generator.from_probability(value) for value in probabilities]

    def _cached_rois(self, batch_data: dict) -> List[ROICoordinates]:
        if self.roi_cache is None:
            raise RuntimeError("ROI cache has not been generated")
        starts = batch_data["z_start"]
        if torch.is_tensor(starts):
            starts = starts.tolist()
        return [
            self.roi_cache.get(case_id, int(z_index))
            for case_id, z_index in zip(batch_data["case_id"], starts)
        ]

    @staticmethod
    def _roi_quality(
        original_gt: torch.Tensor,
        rois: Sequence[ROICoordinates],
    ) -> tuple[float, float]:
        recalls = []
        area_fractions = []
        height, width = original_gt.shape[-2:]
        for index, roi in enumerate(rois):
            total_myo = (
                (original_gt[index, 0] == 1)
                | (original_gt[index, 0] == 2)
                | (original_gt[index, 0] == 4)
            )
            total = int(total_myo.sum())
            inside = int(
                total_myo[roi.y0 : roi.y1, roi.x0 : roi.x1].sum()
            )
            recalls.append(inside / total if total > 0 else 1.0)
            area_fractions.append(
                roi.width * roi.height / float(height * width)
            )
        return float(np.mean(recalls)), float(np.mean(area_fractions))

    def _run_loader(
        self,
        loader,
        *,
        train: bool,
        epoch: int,
        mixed: bool,
        w_route: float,
        w_p_ot: float,
        w_s_ot: float,
    ) -> dict:
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self._set_fusion_mode(train)
        meter_keys = (
            "loss_total", "loss_anchor", "loss_refine",
            "anchor_seg", "refine_seg", "loss_ae", "loss_ortho",
            "loss_route", "loss_p_ot", "loss_s_ot", "roi_gt_recall",
            "roi_area_fraction", "roi_fallback_rate",
            "roi_lv_loss_valid_rate", "roi_rv_loss_valid_rate",
            "anchor_sigmoid_dice", "refine_sigmoid_dice",
        )
        meter = {key: 0.0 for key in meter_keys}
        n_batches = 0
        case_counts = {}
        grad_context = nullcontext() if train else torch.no_grad()
        label = "Train" if train else "Val"

        with grad_context:
            for batch_data in tqdm(
                loader, desc=f"[{label} {epoch:03d}]", leave=False
            ):
                max_batches = (
                    self.cfg.max_train_batches if train
                    else self.cfg.max_val_batches
                )
                if max_batches > 0 and n_batches >= max_batches:
                    break
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self.amp_dtype,
                    enabled=self.amp_enabled,
                ):
                    regularizer_scale = 0.5 if mixed else 1.0
                    anchor_batch = batch_data
                    if train and self.cfg.lge_augment:
                        anchor_batch = augment_lge_batch(
                            batch_data,
                            rotation_degrees=self.cfg.augment_rotation_degrees,
                            scale_min=self.cfg.augment_scale_min,
                            scale_max=self.cfg.augment_scale_max,
                            translation_fraction=self.cfg.augment_translation_fraction,
                            horizontal_flip_probability=self.cfg.augment_horizontal_flip_probability,
                            vertical_flip_probability=self.cfg.augment_vertical_flip_probability,
                            intensity_probability=self.cfg.augment_intensity_probability,
                        )
                    anchor_loss, anchor_logs = self._branch_forward(
                        anchor_batch,
                        prompt_features=self.anatomy_prompt_features,
                        groups=self.anatomy_groups,
                        regularizer_scale=regularizer_scale,
                        w_route=w_route,
                        w_p_ot=w_p_ot,
                        w_s_ot=w_s_ot,
                    )
                    anatomy_logits = anchor_logs.pop("_logits")
                    refine_loss = anchor_loss * 0.0
                    refine_logs = {}
                    rois = []
                    if mixed:
                        rois = (
                            self._cached_rois(batch_data)
                            if train else self._online_rois(anatomy_logits)
                        )
                        if train and self.cfg.roi_v2_hard_switch:
                            rois = [
                                jitter_roi(
                                    roi,
                                    batch_data["gt"].shape[-2:],
                                    center_fraction=self.cfg.roi_jitter_center_fraction,
                                    scale_min=self.cfg.roi_jitter_scale_min,
                                    scale_max=self.cfg.roi_jitter_scale_max,
                                )
                                for roi in rois
                            ]
                        roi_batch = crop_and_resize_batch(batch_data, rois)
                        if self.cfg.roi_v2_hard_switch:
                            prompt_valid = roi_prompt_visibility(
                                batch_data["gt"],
                                rois,
                                self.refinement_groups,
                                min_coverage=self.cfg.roi_visibility_min_coverage,
                            )
                            roi_batch["prompt_valid"] = prompt_valid
                            meter["roi_lv_loss_valid_rate"] += float(
                                prompt_valid[:, 0].float().mean()
                            )
                            meter["roi_rv_loss_valid_rate"] += float(
                                prompt_valid[:, 1].float().mean()
                            )
                        if train and self.cfg.lge_augment:
                            roi_batch = augment_lge_batch(
                                roi_batch,
                                rotation_degrees=self.cfg.augment_rotation_degrees,
                                scale_min=self.cfg.augment_scale_min,
                                scale_max=self.cfg.augment_scale_max,
                                translation_fraction=self.cfg.augment_translation_fraction,
                                horizontal_flip_probability=self.cfg.augment_horizontal_flip_probability,
                                vertical_flip_probability=self.cfg.augment_vertical_flip_probability,
                                intensity_probability=self.cfg.augment_intensity_probability,
                            )
                        refine_loss, refine_logs = self._branch_forward(
                            roi_batch,
                            prompt_features=self.refinement_prompt_features,
                            groups=self.refinement_groups,
                            regularizer_scale=regularizer_scale,
                            w_route=w_route,
                            w_p_ot=w_p_ot,
                            w_s_ot=w_s_ot,
                        )
                        refinement_logits = refine_logs.pop("_logits")
                        total_loss = (
                            self.cfg.lambda_anchor * anchor_loss
                            + self.cfg.lambda_refine * refine_loss
                        )
                    else:
                        total_loss = self.cfg.lambda_anchor * anchor_loss

                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(total_loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.trainable_params, max_norm=5.0
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                # Full and ROI branches receive independent random affine transforms
                # during training. Their logits therefore cannot be restored into one
                # coordinate system for a meaningful hard-switch training metric.
                # Validation is unaugmented and remains the exact inference pipeline.
                score_hard_switch = not (train and self.cfg.lge_augment)
                if mixed and score_hard_switch:
                    restored = restore_roi_logits(
                        refinement_logits,
                        rois,
                        batch_data["gt"].shape[-2:],
                    )
                    background = torch.zeros_like(anatomy_logits[:, :1])
                    if self.cfg.roi_v2_hard_switch:
                        foreground = hard_switch_foreground_logits(
                            anatomy_logits, restored, rois
                        )
                        final_scores = torch.cat([background, foreground], dim=1)
                    else:
                        final_scores = torch.cat(
                            [
                                background,
                                anatomy_logits[:, 0:1],
                                anatomy_logits[:, 1:2],
                                restored[:, 0:1],
                                restored[:, 1:2],
                            ],
                            dim=1,
                        )
                    prediction = final_scores.argmax(dim=1)
                    target = remap_grouped_labels(
                        batch_data["gt"],
                        ((3,), (5,), (4,), (1, 2)),
                    )
                    _update_case_counts(
                        case_counts,
                        batch_data["case_id"],
                        prediction,
                        target,
                        4,
                    )
                else:
                    if not mixed:
                        background = torch.zeros_like(anatomy_logits[:, :1])
                        prediction = torch.cat(
                            [background, anatomy_logits], dim=1
                        ).argmax(dim=1)
                        target = remap_grouped_labels(
                            anchor_batch["gt"], self.anatomy_groups
                        )
                        _update_case_counts(
                            case_counts,
                            batch_data["case_id"],
                            prediction,
                            target,
                            3,
                        )

                if mixed:
                    roi_recall, roi_area = self._roi_quality(
                        batch_data["gt"], rois
                    )
                    meter["roi_gt_recall"] += roi_recall
                    meter["roi_area_fraction"] += roi_area
                    meter["roi_fallback_rate"] += float(
                        np.mean([roi.fallback for roi in rois])
                    )

                meter["loss_total"] += float(total_loss.detach())
                meter["loss_anchor"] += float(anchor_loss.detach())
                meter["loss_refine"] += float(refine_loss.detach())
                meter["anchor_seg"] += anchor_logs.get("loss_seg", 0.0)
                meter["refine_seg"] += refine_logs.get("loss_seg", 0.0)
                meter["anchor_sigmoid_dice"] += anchor_logs.get(
                    "dice_mean", 0.0
                )
                meter["refine_sigmoid_dice"] += refine_logs.get(
                    "dice_mean", 0.0
                )
                for name in (
                    "loss_ae", "loss_ortho", "loss_route",
                    "loss_p_ot", "loss_s_ot",
                ):
                    meter[name] += anchor_logs.get(name, 0.0)
                    meter[name] += refine_logs.get(name, 0.0)
                n_batches += 1

        for key in meter:
            meter[key] /= max(1, n_batches)
        class_count = 4 if mixed else 3
        macro, per_class = _case_dice_from_counts(case_counts, class_count)
        meter["exclusive_macro_dice"] = macro
        meter["exclusive_dice_per_class"] = per_class
        if mixed and train and self.cfg.lge_augment:
            meter["mode"] = "mixed_augmented_branches"
        else:
            meter["mode"] = "mixed" if mixed else "warmup_anatomy"
        meter["peak_cuda_memory_gib"] = (
            float(torch.cuda.max_memory_allocated(self.device) / 1024**3)
            if self.device.type == "cuda"
            else 0.0
        )
        return meter

    @torch.no_grad()
    def generate_roi_cache(self, loader, *, epoch: int) -> tuple[ROICache, dict]:
        self._set_fusion_mode(False)
        cache = ROICache()
        all_rois = []
        gt_recalls = []
        threshold_intersection = 0.0
        threshold_predicted = 0.0
        threshold_target = 0.0
        probability_inside = []
        probability_outside = []
        for batch_data in tqdm(loader, desc=f"[ROI cache e{epoch:03d}]", leave=False):
            bp_images = batch_data["biomedparse_image"].to(self.device)
            valid_z = batch_data["valid_z"].to(self.device)
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.amp_enabled,
            ):
                logits = predict_block_logits_per_class(
                    biomedparse_images=bp_images,
                    valid_z=valid_z,
                    output_size=(self.cfg.image_size, self.cfg.image_size),
                    prompt_features=self.anatomy_prompt_features,
                    P=len(self.anatomy_groups),
                    model_biomedparse=self.model_biomedparse,
                    fusion_modules=self.fusion_modules,
                    device=self.device,
                )
            rois = self._online_rois(logits)
            probabilities = torch.sigmoid(logits[:, 2, 0]).cpu()
            targets = (
                (batch_data["gt"][:, 0] == 1)
                | (batch_data["gt"][:, 0] == 2)
                | (batch_data["gt"][:, 0] == 4)
            )
            predictions = probabilities >= self.cfg.roi_threshold
            threshold_intersection += float((predictions & targets).sum())
            threshold_predicted += float(predictions.sum())
            threshold_target += float(targets.sum())
            if targets.any():
                probability_inside.append(float(probabilities[targets].mean()))
            if (~targets).any():
                probability_outside.append(float(probabilities[~targets].mean()))
            starts = batch_data["z_start"]
            if torch.is_tensor(starts):
                starts = starts.tolist()
            for case_id, z_index, roi in zip(
                batch_data["case_id"], starts, rois
            ):
                cache.set(case_id, int(z_index), roi)
            recall, _ = self._roi_quality(batch_data["gt"], rois)
            gt_recalls.append(recall)
            all_rois.extend(rois)
        diagnostics = roi_diagnostics(
            all_rois, (self.cfg.image_size, self.cfg.image_size)
        )
        diagnostics["total_myo_gt_recall_mean"] = float(np.mean(gt_recalls))
        diagnostics["threshold_mask_dice"] = (
            2.0 * threshold_intersection
            / max(1.0, threshold_predicted + threshold_target)
        )
        diagnostics["threshold_mask_precision"] = (
            threshold_intersection / max(1.0, threshold_predicted)
        )
        diagnostics["threshold_mask_recall"] = (
            threshold_intersection / max(1.0, threshold_target)
        )
        diagnostics["probability_inside_gt_mean"] = float(
            np.mean(probability_inside) if probability_inside else 0.0
        )
        diagnostics["probability_outside_gt_mean"] = float(
            np.mean(probability_outside) if probability_outside else 0.0
        )
        diagnostics.update(
            epoch=epoch,
            threshold=self.cfg.roi_threshold,
            expand=self.cfg.roi_expand,
            fallback=self.cfg.roi_fallback,
        )
        cache_path = os.path.join(self.cfg.output_dir, "roi_cache_train.json")
        cache.save(cache_path, metadata=diagnostics)
        with open(
            os.path.join(self.cfg.output_dir, "roi_cache_summary.json"),
            "w", encoding="utf-8",
        ) as handle:
            json.dump(diagnostics, handle, indent=2)
        return cache, diagnostics

    def _save_roi_checkpoint(self, epoch: int, best: bool = False) -> None:
        state = {
            "epoch": epoch,
            "best_val_dice": self.best_val_dice,
            "config": asdict(self.cfg),
            "format": (
                "oodka_lge_roi_v2"
                if self.cfg.roi_v2_hard_switch
                else "oodka_lge_roi_v1"
            ),
            "prompt_texts": self.prompt_texts,
            "anatomy_groups": self.anatomy_groups,
            "refinement_groups": self.refinement_groups,
        }
        for name, module in self.fusion_modules.items():
            state[name] = module.state_dict()
        state["optimizer"] = self.optimizer.state_dict()
        state["scaler"] = self.scaler.state_dict()
        prefix = "fusion_lge_roi_v2" if self.cfg.roi_v2_hard_switch else "fusion_lge_roi"
        filename = f"{prefix}_best.pth" if best else f"{prefix}_epoch{epoch:03d}.pth"
        torch.save(state, os.path.join(self.cfg.output_dir, filename))

    def _evaluate_best_on_test(self, epoch: int) -> dict:
        """Run the current validation-best checkpoint on test for diagnostics."""
        prefix = (
            "fusion_lge_roi_v2"
            if self.cfg.roi_v2_hard_switch else "fusion_lge_roi"
        )
        checkpoint = os.path.join(self.cfg.output_dir, f"{prefix}_best.pth")
        out_dir = os.path.join(
            self.cfg.output_dir, "test_by_val_best", f"epoch{epoch:03d}"
        )
        maybe_mkdir_p(out_dir)
        repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        command = [
            sys.executable,
            os.path.join(repo_dir, "run_eval_lge_roi.py"),
            "--checkpoint", checkpoint,
            "--split", "test",
            "--device", self.cfg.best_test_device,
            "--batch_size", str(self.cfg.best_test_batch_size),
            "--decision", "auto",
            "--out_dir", out_dir,
        ]
        with open(
            os.path.join(out_dir, "evaluation.log"), "w", encoding="utf-8"
        ) as handle:
            subprocess.run(
                command,
                cwd=repo_dir,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=True,
            )
        with open(
            os.path.join(out_dir, "summary.json"), encoding="utf-8"
        ) as handle:
            return json.load(handle)

    def train(self) -> None:
        cfg = self.cfg
        set_seed(cfg.seed)
        maybe_mkdir_p(cfg.output_dir)
        with open(
            os.path.join(cfg.output_dir, "resolved_config.json"),
            "w", encoding="utf-8",
        ) as handle:
            json.dump(asdict(cfg), handle, indent=2)
        with open(
            os.path.join(cfg.output_dir, "prompts.json"),
            "w", encoding="utf-8",
        ) as handle:
            json.dump(self.prompt_texts, handle, indent=2)
        log_dir = os.path.join(cfg.output_dir, "logs")
        maybe_mkdir_p(log_dir)
        log_handle = open(
            os.path.join(
                log_dir,
                f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            ),
            "w", encoding="utf-8",
        )

        def log(message: str) -> None:
            print(message)
            log_handle.write(message + "\n")
            log_handle.flush()

        train_ids, val_ids = load_fold_cases(cfg.splits_final_json, cfg.fold)
        if cfg.train_case_limit:
            train_ids = train_ids[: cfg.train_case_limit]
        if cfg.val_case_limit:
            val_ids = val_ids[: cfg.val_case_limit]
        with open(cfg.dataset_json_path, encoding="utf-8") as handle:
            file_ending = json.load(handle).get("file_ending", ".nii.gz")
        train_dataset = self._make_dataset(train_ids, file_ending)
        val_dataset = self._make_dataset(val_ids, file_ending)
        train_loader, train_sampler = self._make_loader(train_dataset, shuffle=True)
        cache_loader, _ = self._make_loader(train_dataset, shuffle=False)
        val_loader, val_sampler = self._make_loader(val_dataset, shuffle=False)
        log(
            f"LGE ROI {'v2' if cfg.roi_v2_hard_switch else 'v1'}: "
            f"train={len(train_ids)} cases/{len(train_dataset)} slices, "
            f"val={len(val_ids)} cases/{len(val_dataset)} slices"
        )
        log(
            f"B={cfg.batch_size}, Z=1, image={cfg.image_size}, "
            f"warmup={cfg.roi_warmup_epochs}, threshold={cfg.roi_threshold}, "
            f"expand={cfg.roi_expand}, pseudoRGB={cfg.pseudo_rgb_mode}, "
            f"augment={cfg.lge_augment}"
        )

        for epoch in range(self.start_epoch, cfg.n_epochs + 1):
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)
            mixed = epoch > cfg.roi_warmup_epochs
            if mixed and (
                self.roi_cache is None
                or (
                    cfg.roi_refresh_every > 0
                    and (epoch - cfg.roi_warmup_epochs - 1)
                    % cfg.roi_refresh_every
                    == 0
                )
            ):
                self.roi_cache, cache_stats = self.generate_roi_cache(
                    cache_loader, epoch=epoch - 1
                )
                log(f"ROI cache: {json.dumps(cache_stats, sort_keys=True)}")

            lr_scale = learning_rate_scale(
                epoch,
                n_epochs=cfg.n_epochs,
                schedule=cfg.lr_schedule,
                warmup_epochs=cfg.lr_warmup_epochs,
                min_lr_ratio=cfg.min_lr_ratio,
            )
            current_lr = cfg.lr * lr_scale
            for group in self.optimizer.param_groups:
                group["lr"] = current_lr
            w_route = cfg.w_route * min(
                1.0, epoch / max(1, cfg.route_warmup_epochs)
            )

            def scheduled(target: float, start: int) -> float:
                if epoch < start:
                    return 0.0
                return target * min(
                    1.0,
                    (epoch - start + 1) / max(1, cfg.ot_warmup_epochs),
                )

            w_p_ot = scheduled(cfg.w_p_ot, cfg.p_ot_start_epoch)
            w_s_ot = scheduled(cfg.w_s_ot, cfg.s_ot_start_epoch)
            train_metrics = self._run_loader(
                train_loader,
                train=True,
                epoch=epoch,
                mixed=mixed,
                w_route=w_route,
                w_p_ot=w_p_ot,
                w_s_ot=w_s_ot,
            )
            log(
                f"[Epoch {epoch:03d}] train {train_metrics['mode']}: "
                f"loss={train_metrics['loss_total']:.4f} "
                f"exclusive={train_metrics['exclusive_macro_dice']:.4f} "
                f"roiRecall={train_metrics['roi_gt_recall']:.4f} "
                f"peakMem={train_metrics['peak_cuda_memory_gib']:.2f}GiB "
                f"lr={current_lr:.3g}"
            )
            val_metrics = None
            test_best_metrics = None
            if epoch % cfg.val_every_epochs == 0 or epoch == cfg.n_epochs:
                val_metrics = self._run_loader(
                    val_loader,
                    train=False,
                    epoch=epoch,
                    mixed=mixed,
                    w_route=w_route,
                    w_p_ot=w_p_ot,
                    w_s_ot=w_s_ot,
                )
                log(
                    f"[Epoch {epoch:03d}] val {val_metrics['mode']}: "
                    f"loss={val_metrics['loss_total']:.4f} "
                    f"exclusive={val_metrics['exclusive_macro_dice']:.4f} "
                    f"perClass={val_metrics['exclusive_dice_per_class']} "
                    f"roiRecall={val_metrics['roi_gt_recall']:.4f} "
                    f"roiArea={val_metrics['roi_area_fraction']:.4f} "
                    f"fallback={val_metrics['roi_fallback_rate']:.4f} "
                    f"peakMem={val_metrics['peak_cuda_memory_gib']:.2f}GiB"
                )
                if mixed and val_metrics["exclusive_macro_dice"] > self.best_val_dice:
                    self.best_val_dice = val_metrics["exclusive_macro_dice"]
                    self._save_roi_checkpoint(epoch, best=True)
                    log(f"  -> new best exclusive argmax={self.best_val_dice:.4f}")
                    if cfg.best_test_on_improvement:
                        log(
                            "  -> evaluating validation-best checkpoint on test "
                            f"using {cfg.best_test_device}"
                        )
                        test_best_metrics = self._evaluate_best_on_test(epoch)
                        log(
                            "  -> diagnostic test macro="
                            f"{test_best_metrics['mean_dice_gt_present']:.4f}"
                        )
            record = {
                "epoch": epoch,
                "lr": current_lr,
                "train": train_metrics,
                "val": val_metrics,
                "test_best_diagnostic": test_best_metrics,
            }
            self.history.append(record)
            with open(
                os.path.join(cfg.output_dir, "history.json"),
                "w", encoding="utf-8",
            ) as handle:
                json.dump(self.history, handle, indent=2)

        self._save_roi_checkpoint(cfg.n_epochs, best=False)
        log(f"Training complete. Best exclusive val Dice={self.best_val_dice:.4f}")
        log_handle.close()
