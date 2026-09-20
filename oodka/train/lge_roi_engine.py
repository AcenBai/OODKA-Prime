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
    oracle_block_rois,
    remap_grouped_labels,
    restore_roi_logits,
    roi_diagnostics,
    roi_prompt_visibility,
)
from ..utils.io_utils import maybe_mkdir_p
from .engine import OODKATrainer, learning_rate_scale, load_fold_cases, set_seed
from .forward import forward_one_batch, predict_block_logits_per_class
from .lge_roi_lifecycle import LGEROILifecycleMixin


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


class LGEROIMixedTrainer(LGEROILifecycleMixin, OODKATrainer):
    """One shared model, two forwards, one joint optimizer update."""

    def __init__(
        self,
        *,
        anatomy_prompt_features: dict,
        refinement_prompt_features: dict,
        anatomy_groups: Sequence[Sequence[int]],
        refinement_groups: Sequence[Sequence[int]],
        prompt_texts: dict,
        roi_prompt_index: int = 2,
        roi_source_labels: Sequence[int] = (1, 2, 4),
        refinement_only_output: bool = False,
        outside_prompt_mapping: Sequence[tuple[int, int]] = ((0, 0), (1, 1)),
        experiment_name: str = "LGE",
        checkpoint_format: str = "",
        checkpoint_prefix: str = "",
        **kwargs,
    ) -> None:
        anatomy_count = len(anatomy_groups)
        super().__init__(
            prompt_features=anatomy_prompt_features,
            prompt_to_class_id=_compact_mapping(anatomy_count),
            P=anatomy_count,
            **kwargs,
        )
        if self.cfg.roi_prompt_loss_reduction not in {
            "mean", "sum", "prompt_mean"
        }:
            raise ValueError(
                "roi_prompt_loss_reduction must be mean, sum, or prompt_mean"
            )
        self.anatomy_prompt_features = anatomy_prompt_features
        self.refinement_prompt_features = refinement_prompt_features
        self.anatomy_groups = tuple(tuple(int(v) for v in g) for g in anatomy_groups)
        self.refinement_groups = tuple(
            tuple(int(v) for v in g) for g in refinement_groups
        )
        self.prompt_texts = prompt_texts
        self.roi_prompt_index = int(roi_prompt_index)
        if not 0 <= self.roi_prompt_index < len(self.anatomy_groups):
            raise ValueError(
                f"roi_prompt_index={self.roi_prompt_index} is invalid for "
                f"{len(self.anatomy_groups)} localization prompts"
            )
        self.roi_source_labels = tuple(int(value) for value in roi_source_labels)
        if not self.roi_source_labels:
            raise ValueError("roi_source_labels must not be empty")
        self.refinement_only_output = bool(refinement_only_output)
        self.outside_prompt_mapping = tuple(
            (int(anatomy_index), int(output_index))
            for anatomy_index, output_index in outside_prompt_mapping
        )
        for anatomy_index, output_index in self.outside_prompt_mapping:
            if not 0 <= anatomy_index < len(self.anatomy_groups):
                raise ValueError(
                    f"Invalid outside anatomy prompt index {anatomy_index}"
                )
            if not 0 <= output_index < len(self.refinement_groups):
                raise ValueError(
                    f"Invalid outside refinement index {output_index}"
                )
        self.experiment_name = str(experiment_name)
        self.checkpoint_format = str(checkpoint_format)
        self.checkpoint_prefix = str(checkpoint_prefix)
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
            expert_ortho_weight=self.cfg.expert_ortho_weight,
            w_route=w_route * regularizer_scale,
            w_p_ot=w_p_ot * regularizer_scale,
            w_s_ot=w_s_ot * regularizer_scale,
            expert_class_groups=groups,
            prompt_loss_reduction=self.cfg.roi_prompt_loss_reduction,
            return_logits=True,
        )

    def _online_rois(
        self,
        anatomy_logits: torch.Tensor,
        valid_z: torch.Tensor | None = None,
    ) -> List[ROICoordinates]:
        probabilities = torch.sigmoid(
            anatomy_logits[:, self.roi_prompt_index].detach()
        )
        if valid_z is not None:
            probabilities = probabilities.masked_fill(
                ~valid_z.to(probabilities.device)[:, :, None, None], 0.0
            )
        # A single XY box is shared by the whole contiguous Z block. Taking
        # the maximum along Z is equivalent to bounding the thresholded 3-D
        # foreground cuboid over its complete block extent.
        block_probability = probabilities.amax(dim=1)
        return [
            self.roi_generator.from_probability(value)
            for value in block_probability
        ]

    def _roi_target_mask(self, labels: torch.Tensor) -> torch.Tensor:
        target = torch.zeros_like(labels, dtype=torch.bool)
        for source_id in self.roi_source_labels:
            target |= labels == source_id
        return target

    def _ground_truth_rois(
        self,
        labels: torch.Tensor,
        valid_z: torch.Tensor | None = None,
    ) -> List[ROICoordinates]:
        """Build the exact block-coherent oracle ROI used for ceiling runs."""
        return oracle_block_rois(
            labels,
            self.roi_source_labels,
            self.roi_generator,
            valid_z,
        )

    def _cached_rois(self, batch_data: dict) -> List[ROICoordinates]:
        if self.roi_cache is None:
            raise RuntimeError("ROI cache has not been generated")
        starts = batch_data["z_start"]
        if torch.is_tensor(starts):
            starts = starts.tolist()
        return [
            self.roi_cache.get(case_id, int(z_start))
            for case_id, z_start in zip(batch_data["case_id"], starts)
        ]

    def _roi_quality(
        self,
        original_gt: torch.Tensor,
        rois: Sequence[ROICoordinates],
        valid_z: torch.Tensor | None = None,
    ) -> tuple[float, float]:
        recalls = []
        area_fractions = []
        height, width = original_gt.shape[-2:]
        batch_size, block_z = original_gt.shape[:2]
        if len(rois) != batch_size:
            raise ValueError("ROI count must equal B")
        for batch_index, roi in enumerate(rois):
            roi_target = self._roi_target_mask(original_gt[batch_index])
            if valid_z is not None:
                roi_target &= valid_z[batch_index, :, None, None].to(
                    roi_target.device
                )
            total = int(roi_target.sum())
            inside = int(
                roi_target[:, roi.y0 : roi.y1, roi.x0 : roi.x1].sum()
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
                        if self.cfg.roi_train_source == "ground_truth":
                            rois = self._ground_truth_rois(
                                batch_data["gt"], batch_data["valid_z"]
                            )
                        elif self.cfg.roi_train_source == "full":
                            height, width = batch_data["gt"].shape[-2:]
                            rois = [
                                ROICoordinates(0, 0, width, height)
                                for _ in range(batch_data["gt"].shape[0])
                            ]
                        elif self.cfg.roi_train_source == "predicted":
                            rois = (
                                self._cached_rois(batch_data)
                                if train else self._online_rois(
                                    anatomy_logits, batch_data["valid_z"]
                                )
                            )
                        else:
                            raise ValueError(
                                "roi_train_source must be predicted, "
                                "ground_truth, or full; got "
                                f"{self.cfg.roi_train_source!r}"
                            )
                        if (
                            train
                            and self.cfg.roi_v2_hard_switch
                            and self.cfg.roi_train_source == "predicted"
                        ):
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
                        roi_batch = crop_and_resize_batch(
                            batch_data,
                            rois,
                            transform=self.cfg.roi_transform,
                        )
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
                        transform=self.cfg.roi_transform,
                    )
                    background = torch.zeros_like(anatomy_logits[:, :1])
                    if self.refinement_only_output:
                        final_scores = torch.cat([background, restored], dim=1)
                    elif self.cfg.roi_v2_hard_switch:
                        foreground = hard_switch_foreground_logits(
                            anatomy_logits,
                            restored,
                            rois,
                            self.outside_prompt_mapping,
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
                    final_groups = (
                        self.refinement_groups
                        if self.refinement_only_output
                        or self.cfg.roi_v2_hard_switch
                        else ((3,), (5,), (4,), (1, 2))
                    )
                    target = remap_grouped_labels(
                        batch_data["gt"], final_groups
                    )
                    _update_case_counts(
                        case_counts,
                        batch_data["case_id"],
                        prediction,
                        target,
                        len(final_groups),
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
                            len(self.anatomy_groups),
                        )

                if mixed:
                    roi_recall, roi_area = self._roi_quality(
                        batch_data["gt"], rois, batch_data["valid_z"]
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
        class_count = (
            len(self.refinement_groups)
            if mixed
            and (self.refinement_only_output or self.cfg.roi_v2_hard_switch)
            else (4 if mixed else len(self.anatomy_groups))
        )
        macro, per_class = _case_dice_from_counts(case_counts, class_count)
        meter["exclusive_macro_dice"] = macro
        meter["exclusive_dice_per_class"] = per_class
        if self.cfg.lge_flat_four_prompt:
            meter["mode"] = "flat_full_image"
        elif mixed and train and self.cfg.lge_augment:
            meter["mode"] = "mixed_augmented_branches"
        else:
            meter["mode"] = "mixed" if mixed else "warmup_anatomy"
        meter["peak_cuda_memory_gib"] = (
            float(torch.cuda.max_memory_allocated(self.device) / 1024**3)
            if self.device.type == "cuda"
            else 0.0
        )
        return meter
