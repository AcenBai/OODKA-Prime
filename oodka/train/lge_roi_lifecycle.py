"""ROI cache, checkpoint, evaluation, and epoch lifecycle helpers."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

from ..data.lge_roi import ROICache, roi_diagnostics
from ..utils.io_utils import maybe_mkdir_p
from .engine import learning_rate_scale, load_fold_cases, set_seed
from .forward import predict_block_logits_per_class


class LGEROILifecycleMixin:
    """Lifecycle mixin used by the mixed two-pass ROI trainer."""

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
            rois = self._online_rois(logits, batch_data["valid_z"])
            probabilities = torch.sigmoid(
                logits[:, self.roi_prompt_index]
            ).cpu()
            targets = self._roi_target_mask(batch_data["gt"])
            valid_z_cpu = batch_data["valid_z"].bool().cpu()
            valid_pixels = valid_z_cpu[:, :, None, None].expand_as(targets)
            predictions = probabilities >= self.cfg.roi_threshold
            threshold_intersection += float(
                (predictions & targets & valid_pixels).sum()
            )
            threshold_predicted += float((predictions & valid_pixels).sum())
            threshold_target += float((targets & valid_pixels).sum())
            inside_mask = targets & valid_pixels
            outside_mask = (~targets) & valid_pixels
            if inside_mask.any():
                probability_inside.append(
                    float(probabilities[inside_mask].mean())
                )
            if outside_mask.any():
                probability_outside.append(
                    float(probabilities[outside_mask].mean())
                )
            starts = batch_data["z_start"]
            if torch.is_tensor(starts):
                starts = starts.tolist()
            for case_id, z_start, roi in zip(
                batch_data["case_id"], starts, rois
            ):
                cache.set(case_id, int(z_start), roi)
                all_rois.append(roi)
            recall, _ = self._roi_quality(
                batch_data["gt"], rois, batch_data["valid_z"]
            )
            gt_recalls.append(recall)
        diagnostics = roi_diagnostics(
            all_rois,
            (self.cfg.image_size, self.cfg.image_size),
            transform=self.cfg.roi_transform,
        )
        diagnostics["roi_target_gt_recall_mean"] = float(np.mean(gt_recalls))
        if self.experiment_name == "LGE":
            diagnostics["total_myo_gt_recall_mean"] = diagnostics[
                "roi_target_gt_recall_mean"
            ]
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
        default_format = (
            "oodka_lge_flat_v1"
            if self.cfg.lge_flat_four_prompt
            else (
                "oodka_lge_roi_v3_split5"
                if self.cfg.lge_split_pathology
                else "oodka_lge_roi_v2"
                if self.cfg.roi_v2_hard_switch
                else "oodka_lge_roi_v1"
            )
        )
        state = {
            "epoch": epoch,
            "best_val_dice": self.best_val_dice,
            "config": asdict(self.cfg),
            "format": self.checkpoint_format or default_format,
            "prompt_texts": self.prompt_texts,
            "anatomy_groups": self.anatomy_groups,
            "refinement_groups": self.refinement_groups,
            "roi_prompt_index": self.roi_prompt_index,
            "roi_source_labels": self.roi_source_labels,
            "refinement_only_output": self.refinement_only_output,
            "outside_prompt_mapping": self.outside_prompt_mapping,
        }
        for name, module in self.fusion_modules.items():
            state[name] = module.state_dict()
        state["optimizer"] = self.optimizer.state_dict()
        state["scaler"] = self.scaler.state_dict()
        if self.checkpoint_prefix:
            prefix = self.checkpoint_prefix
        elif self.cfg.lge_flat_four_prompt:
            prefix = "fusion_lge_flat"
        else:
            prefix = (
                "fusion_lge_roi_v3_split5"
                if self.cfg.lge_split_pathology
                else "fusion_lge_roi_v2"
                if self.cfg.roi_v2_hard_switch else "fusion_lge_roi"
            )
        filename = f"{prefix}_best.pth" if best else f"{prefix}_epoch{epoch:03d}.pth"
        torch.save(state, os.path.join(self.cfg.output_dir, filename))

    def _evaluate_best_on_test(self, epoch: int) -> dict:
        """Run the current validation-best checkpoint on test for diagnostics."""
        if self.checkpoint_prefix:
            prefix = self.checkpoint_prefix
        elif self.cfg.lge_flat_four_prompt:
            prefix = "fusion_lge_flat"
        else:
            prefix = (
                "fusion_lge_roi_v3_split5"
                if self.cfg.lge_split_pathology
                else "fusion_lge_roi_v2"
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
            "--roi_source", self.cfg.roi_train_source,
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
            f"{self.experiment_name} "
            f"{'flat-4' if cfg.lge_flat_four_prompt else ('ROI v3 split-5' if cfg.lge_split_pathology else ('ROI v2' if cfg.roi_v2_hard_switch else 'ROI v1'))}: "
            f"train={len(train_ids)} cases/{len(train_dataset)} slices, "
            f"val={len(val_ids)} cases/{len(val_dataset)} slices"
        )
        log(
            f"B={cfg.batch_size}, Z={cfg.block_z}, image={cfg.image_size}, "
            f"warmup={cfg.roi_warmup_epochs}, threshold={cfg.roi_threshold}, "
            f"expand={cfg.roi_expand}, pseudoRGB={cfg.pseudo_rgb_mode}, "
            f"roiSource={cfg.roi_train_source}, "
            f"roiTransform={cfg.roi_transform}, "
            f"augment={cfg.lge_augment}, "
            f"promptReduction={cfg.roi_prompt_loss_reduction}"
        )

        for epoch in range(self.start_epoch, cfg.n_epochs + 1):
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)
            mixed = (
                not cfg.lge_flat_four_prompt
                and epoch > cfg.roi_warmup_epochs
            )
            if mixed and cfg.roi_train_source == "predicted" and (
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
                if (
                    (mixed or cfg.lge_flat_four_prompt)
                    and val_metrics["exclusive_macro_dice"] > self.best_val_dice
                ):
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
