"""Training loop for contiguous full-slice-block OODKA."""

from __future__ import annotations

import json
import os
import random
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import TrainConfig
from ..data.slice_dataset import FullSliceBlockDataset, CaseBlockBatchSampler
from ..utils.io_utils import maybe_mkdir_p
from ..utils.visualization import plot_training_curves
from .forward import forward_one_batch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_fold_cases(splits_path: str, fold: int) -> Tuple[List[str], List[str]]:
    with open(splits_path, encoding="utf-8") as f:
        splits = json.load(f)
    if not isinstance(splits, list) or not 0 <= fold < len(splits):
        raise ValueError(f"Invalid fold={fold} for {splits_path}")
    return list(splits[fold]["train"]), list(splits[fold]["val"])


class OODKATrainer:
    """Train 3D fusion adapters over batches of independent 2.5D Z blocks."""

    def __init__(
        self,
        cfg: TrainConfig,
        model_nnunet: nn.Module,
        model_biomedparse: nn.Module,
        fusion_modules: Dict[str, nn.Module],
        prompt_features: dict,
        prompt_to_class_id: Dict[int, int],
        P: int,
    ):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model_nnunet = model_nnunet
        self.model_biomedparse = model_biomedparse
        self.fusion_modules = fusion_modules
        self.prompt_features = prompt_features
        self.prompt_to_class_id = prompt_to_class_id
        self.P = P
        self.block_shape = [cfg.block_z, cfg.image_size, cfg.image_size]

        trainable_params = []
        for module in fusion_modules.values():
            trainable_params.extend(
                parameter for parameter in module.parameters()
                if parameter.requires_grad
            )
        self.trainable_params = trainable_params
        self.optimizer = torch.optim.AdamW(
            trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self.amp_enabled = bool(cfg.amp and self.device.type == "cuda")
        self.amp_dtype = (
            torch.bfloat16 if cfg.amp_dtype == "bfloat16" else torch.float16
        )
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.amp_enabled and self.amp_dtype == torch.float16
        )
        self.history = {
            "epochs": [],
            "train_loss_total": [], "train_loss_seg": [],
            "train_loss_ae": [], "train_loss_ortho": [],
            "train_loss_route": [],
            "train_loss_p_ot": [], "train_loss_s_ot": [],
            "val_loss_total": [], "val_loss_seg": [],
            "val_loss_ae": [], "val_loss_ortho": [],
            "val_loss_route": [],
            "val_loss_p_ot": [], "val_loss_s_ot": [],
            "train_dice_mean": [], "val_dice_mean": [],
            "train_dice_per_class": [], "val_dice_per_class": [],
            "train_gate_per_class": [],
        }
        self.best_val_dice = -float("inf")
        self.start_epoch = 1
        if cfg.resume_checkpoint:
            self._load_checkpoint(cfg.resume_checkpoint)

    def _set_fusion_mode(self, train: bool) -> None:
        for module in self.fusion_modules.values():
            module.train(train)

    def _make_dataset(self, case_ids: List[str], file_ending: str) -> FullSliceBlockDataset:
        cfg = self.cfg
        return FullSliceBlockDataset(
            case_ids,
            nnunet_preproc_dir=cfg.nnunet_preproc_dir,
            images_dir=cfg.imagesTr_dir,
            labels_dir=cfg.labelsTr_dir,
            file_ending=file_ending,
            image_size=cfg.image_size,
            block_z=cfg.block_z,
            norm_mode=cfg.norm_mode,
            window_level=cfg.window_level,
            window_width=cfg.window_width,
            low_percentile=cfg.low_percentile,
            high_percentile=cfg.high_percentile,
            raw_cache_cases=cfg.raw_cache_cases,
            require_no_crop=cfg.require_no_crop,
            biomedparse_modality=cfg.biomedparse_modality,
        )

    def _make_loader(
        self,
        dataset: FullSliceBlockDataset,
        *,
        shuffle: bool,
    ) -> Tuple[DataLoader, CaseBlockBatchSampler]:
        cfg = self.cfg
        sampler = CaseBlockBatchSampler(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            drop_last=False,
            seed=cfg.seed,
        )
        kwargs = dict(
            dataset=dataset,
            batch_sampler=sampler,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )
        if cfg.num_workers > 0:
            kwargs["persistent_workers"] = True
        return DataLoader(**kwargs), sampler

    def _run_loader(
        self,
        loader: DataLoader,
        *,
        train: bool,
        epoch: int,
        w_route: float,
        w_p_ot: float,
        w_s_ot: float,
    ) -> Tuple[dict, float, dict]:
        cfg = self.cfg
        self._set_fusion_mode(train)
        meter = {
            key: 0.0
            for key in [
                "loss_total", "loss_seg", "loss_ae", "loss_ortho",
                "loss_route",
                "loss_p_ot", "loss_s_ot",
            ]
        }
        ot_keys = [
            f"ot_res{level}_{name}"
            for level in [2, 3, 4, 5]
            for name in [
                "p_cost", "p_row_error", "p_col_error", "p_entropy",
                "s_cost", "s_received", "s_transported", "s_rejected",
                "s_accept_ratio", "s_entropy", "s_gain",
            ]
        ]
        meter.update({key: 0.0 for key in ot_keys})
        dice_pc_sum = np.zeros(self.P, dtype=np.float64)
        dice_pc_cnt = np.zeros(self.P, dtype=np.int64)
        n_batches = 0
        grad_context = nullcontext() if train else torch.no_grad()
        label = "Train" if train else "Val"

        with grad_context:
            for batch_data in tqdm(loader, desc=f"[{label} {epoch:03d}]", leave=False):
                max_batches = (
                    cfg.max_train_batches if train else cfg.max_val_batches
                )
                if max_batches > 0 and n_batches >= max_batches:
                    break
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self.amp_dtype,
                    enabled=self.amp_enabled,
                ):
                    total_loss, logs = forward_one_batch(
                        batch_data=batch_data,
                        block_shape=self.block_shape,
                        prompt_features=self.prompt_features,
                        P=self.P,
                        prompt_to_class_id=self.prompt_to_class_id,
                        w_seg=cfg.w_seg,
                        w_ae=cfg.w_ae,
                        w_ort=cfg.w_ort,
                        model_nnunet=self.model_nnunet,
                        model_biomedparse=self.model_biomedparse,
                        fusion_modules=self.fusion_modules,
                        device=self.device,
                        w_route=w_route,
                        w_p_ot=w_p_ot,
                        w_s_ot=w_s_ot,
                    )
                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(total_loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.trainable_params, max_norm=5.0
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                for key in meter:
                    meter[key] += logs.get(key, 0.0)
                for pi in range(self.P):
                    value = logs.get("dice_per_class", {}).get(pi)
                    if value is not None:
                        dice_pc_sum[pi] += value
                        dice_pc_cnt[pi] += 1
                n_batches += 1

        for key in meter:
            meter[key] /= max(n_batches, 1)
        dice_per_class = {
            pi: float(dice_pc_sum[pi] / dice_pc_cnt[pi])
            if dice_pc_cnt[pi] > 0 else None
            for pi in range(self.P)
        }
        present = [value for value in dice_per_class.values() if value is not None]
        dice_mean = float(np.mean(present)) if present else 0.0
        return meter, dice_mean, dice_per_class

    def train(self) -> None:
        cfg = self.cfg
        set_seed(cfg.seed)
        maybe_mkdir_p(cfg.output_dir)
        with open(
            os.path.join(cfg.output_dir, "resolved_config.json"),
            "w",
            encoding="utf-8",
        ) as config_handle:
            json.dump(asdict(cfg), config_handle, indent=2)
        log_dir = os.path.join(cfg.output_dir, "logs")
        maybe_mkdir_p(log_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_fh = open(
            os.path.join(log_dir, f"training_{timestamp}.log"),
            "w",
            encoding="utf-8",
        )

        def log(message: str) -> None:
            print(message)
            log_fh.write(message + "\n")
            log_fh.flush()

        train_ids, val_ids = load_fold_cases(cfg.splits_final_json, cfg.fold)
        if cfg.train_case_limit > 0:
            train_ids = train_ids[: cfg.train_case_limit]
        if cfg.val_case_limit > 0:
            val_ids = val_ids[: cfg.val_case_limit]
        with open(cfg.dataset_json_path, encoding="utf-8") as f:
            file_ending = json.load(f).get("file_ending", ".nii.gz")
        train_dataset = self._make_dataset(train_ids, file_ending)
        val_dataset = self._make_dataset(val_ids, file_ending)
        train_loader, train_sampler = self._make_loader(train_dataset, shuffle=True)
        val_loader, val_sampler = self._make_loader(val_dataset, shuffle=False)

        log(f"Train: {len(train_ids)} cases, {len(train_dataset)} blocks, "
            f"{train_dataset.total_real_slices} real slices")
        log(f"Val: {len(val_ids)} cases, {len(val_dataset)} blocks, "
            f"{val_dataset.total_real_slices} real slices")
        log(f"Batch contract: B={cfg.batch_size}, Z={cfg.block_z}, "
            f"C_nn=dataset, C_bp=3, H=W={cfg.image_size}")
        log(f"Output: {cfg.output_dir}")

        for epoch in range(self.start_epoch, cfg.n_epochs + 1):
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)
            if cfg.route_warmup_epochs > 0 and epoch <= cfg.route_warmup_epochs:
                w_route = cfg.w_route * (epoch / cfg.route_warmup_epochs)
            else:
                w_route = cfg.w_route

            def scheduled_weight(target: float, start: int) -> float:
                if epoch < start:
                    return 0.0
                if cfg.ot_warmup_epochs <= 0:
                    return target
                progress = min(
                    1.0, (epoch - start + 1) / cfg.ot_warmup_epochs
                )
                return target * progress

            w_p_ot = scheduled_weight(cfg.w_p_ot, cfg.p_ot_start_epoch)
            w_s_ot = scheduled_weight(cfg.w_s_ot, cfg.s_ot_start_epoch)

            train_meter, train_dice, train_pc = self._run_loader(
                train_loader,
                train=True,
                epoch=epoch,
                w_route=w_route,
                w_p_ot=w_p_ot,
                w_s_ot=w_s_ot,
            )
            log(f"[Epoch {epoch:03d}] Train: "
                f"loss={train_meter['loss_total']:.4f} dice={train_dice:.4f} "
                f"wP={w_p_ot:.4g} wS={w_s_ot:.4g}")

            val_meter = {key: 0.0 for key in train_meter}
            val_dice = 0.0
            val_pc = {pi: None for pi in range(self.P)}
            if epoch % cfg.val_every_epochs == 0 or epoch == cfg.n_epochs:
                val_meter, val_dice, val_pc = self._run_loader(
                    val_loader,
                    train=False,
                    epoch=epoch,
                    w_route=w_route,
                    w_p_ot=w_p_ot,
                    w_s_ot=w_s_ot,
                )
                log(f"[Epoch {epoch:03d}] Val: "
                    f"loss={val_meter['loss_total']:.4f} dice={val_dice:.4f}")
                if w_p_ot > 0:
                    log(
                        "  OT: "
                        f"P={val_meter['loss_p_ot']:.4f} "
                        f"S={val_meter['loss_s_ot']:.4f} "
                        f"res4_row={val_meter['ot_res4_p_row_error']:.3e} "
                        f"res4_col={val_meter['ot_res4_p_col_error']:.3e} "
                        f"res4_accept={val_meter['ot_res4_s_accept_ratio']:.4f}"
                    )
                for pi in range(self.P):
                    log(f"  class {self.prompt_to_class_id[pi]}: {val_pc[pi]}")
                if val_dice > self.best_val_dice:
                    self.best_val_dice = val_dice
                    self._save_checkpoint(epoch, best=True)
                    log(f"  -> New best model saved (dice={val_dice:.4f})")

            self.history["epochs"].append(epoch)
            for key in [
                "loss_total", "loss_seg", "loss_ae", "loss_ortho",
                "loss_route",
                "loss_p_ot", "loss_s_ot",
            ]:
                self.history[f"train_{key}"].append(train_meter[key])
                self.history[f"val_{key}"].append(val_meter[key])
            self.history["train_dice_mean"].append(train_dice)
            self.history["val_dice_mean"].append(val_dice)
            self.history["train_dice_per_class"].append(train_pc)
            self.history["val_dice_per_class"].append(val_pc)
            plot_training_curves(self.history, cfg.output_dir, epoch, self.P)

        self._save_checkpoint(cfg.n_epochs, best=False)
        log(f"Training complete. Best val dice: {self.best_val_dice:.4f}")
        log_fh.close()

    def _save_checkpoint(self, epoch: int, best: bool = False) -> None:
        state = {
            "epoch": epoch,
            "best_val_dice": self.best_val_dice,
            "config": asdict(self.cfg),
        }
        for name, module in self.fusion_modules.items():
            state[name] = module.state_dict()
        state["optimizer"] = self.optimizer.state_dict()
        state["scaler"] = self.scaler.state_dict()
        filename = "fusion_disentangle_best.pth" if best else f"checkpoint_epoch{epoch:03d}.pth"
        torch.save(state, os.path.join(self.cfg.output_dir, filename))
        deploy_state = {
            "epoch": epoch,
            "best_val_dice": self.best_val_dice,
            "format": "oodka_student_v1",
        }
        for name, module in self.fusion_modules.items():
            if name.startswith("dis_b_") or name == "beta_router":
                deploy_state[name] = module.state_dict()
        deploy_name = (
            "student_deploy_best.pth"
            if best
            else f"student_deploy_epoch{epoch:03d}.pth"
        )
        torch.save(deploy_state, os.path.join(self.cfg.output_dir, deploy_name))

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        for name, module in self.fusion_modules.items():
            if name in checkpoint:
                module.load_state_dict(checkpoint[name])
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])
        self.best_val_dice = float(
            checkpoint.get("best_val_dice", self.best_val_dice)
        )
        self.start_epoch = int(checkpoint.get("epoch", 0)) + 1
