"""Training loop orchestrator."""

from __future__ import annotations

import os
import sys
import json
import random
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import TrainConfig, ensure_nnunet_on_path
from ..data.datasets import PatchDataset, custom_collate_fn
from ..utils.patch_sampling import PatchSampler, load_fold_cases_from_splits_final
from ..utils.metrics import dice_ignore_minus_one
from ..utils.visualization import plot_training_curves
from ..utils.io_utils import maybe_mkdir_p
from .forward import forward_one_batch, predict_patch_logits_per_class


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class OODKATrainer:
    """End-to-end OODKA training loop."""

    def __init__(
        self,
        cfg: TrainConfig,
        model_nnunet: nn.Module,
        model_biomedparse: nn.Module,
        fusion_modules: Dict[str, nn.Module],
        prompt_features: dict,
        class_emb: torch.Tensor,
        prompt_to_class_id: Dict[int, int],
        P: int,
    ):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model_nnunet = model_nnunet
        self.model_biomedparse = model_biomedparse
        self.fusion_modules = fusion_modules
        self.prompt_features = prompt_features
        self.class_emb = class_emb
        self.prompt_to_class_id = prompt_to_class_id
        self.P = P
        self.patch_size = self._resolve_patch_size()

        # Optimizer: only trainable fusion modules
        trainable_params = []
        for m in fusion_modules.values():
            trainable_params.extend(m.parameters())
        self.optimizer = torch.optim.AdamW(
            trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay
        )

        # History
        self.history = {
            "epochs": [],
            "train_loss_total": [], "train_loss_seg": [],
            "train_loss_ae": [], "train_loss_ortho": [], "train_loss_ka": [],
            "val_loss_total": [], "val_loss_seg": [],
            "val_loss_ae": [], "val_loss_ortho": [], "val_loss_ka": [],
            "train_dice_mean": [], "val_dice_mean": [],
            "train_dice_per_class": [], "val_dice_per_class": [],
            "train_tau_per_class": [],
        }
        self.best_val_dice = -float("inf")

    def _resolve_patch_size(self) -> List[int]:
        ensure_nnunet_on_path()
        from batchgenerators.utilities.file_and_folder_operations import load_json
        from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

        cfg = self.cfg
        plans = load_json(os.path.join(cfg.nnunet_model_dir, "plans.json"))
        pm = PlansManager(plans)
        cm = pm.get_configuration(cfg.nnunet_configuration)
        yx = list(cm.patch_size)
        return [cfg.block_z] + yx

    def _set_fusion_mode(self, train: bool):
        for m in self.fusion_modules.values():
            if train:
                m.train()
            else:
                m.eval()

    def train(self):
        cfg = self.cfg
        set_seed(cfg.seed)

        maybe_mkdir_p(cfg.output_dir)
        log_dir = os.path.join(cfg.output_dir, "logs")
        maybe_mkdir_p(log_dir)

        # Setup logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"training_{timestamp}.log")
        log_fh = open(log_file, "w", encoding="utf-8")

        def _log(msg):
            print(msg)
            log_fh.write(msg + "\n")
            log_fh.flush()

        train_ids, val_ids = load_fold_cases_from_splits_final(
            cfg.splits_final_json, cfg.fold
        )
        _log(f"Train: {len(train_ids)} cases | Val: {len(val_ids)} cases")
        _log(f"Patch size: {self.patch_size}")
        _log(f"Output: {cfg.output_dir}")

        sampler = PatchSampler(
            nnunet_preproc_dir=cfg.nnunet_preproc_dir,
            biomedparse_preproc_dir=cfg.biomedparse_preproc_train,
            plans_path=cfg.plans_path,
            dataset_json_path=cfg.dataset_json_path,
            patch_size=self.patch_size,
        )

        for epoch in range(1, cfg.n_epochs + 1):
            # P-reg schedule
            if cfg.p_reg_warmup_epochs > 0 and epoch <= cfg.p_reg_warmup_epochs:
                w_p_reg = cfg.w_p_reg * (epoch / cfg.p_reg_warmup_epochs)
            elif cfg.p_reg_decay_epochs > 0 and epoch > cfg.n_epochs - cfg.p_reg_decay_epochs:
                decay_start = cfg.n_epochs - cfg.p_reg_decay_epochs
                w_p_reg = cfg.w_p_reg * (1.0 - (epoch - decay_start) / cfg.p_reg_decay_epochs)
            else:
                w_p_reg = cfg.w_p_reg

            # Sample patches
            _log(f"\n[Epoch {epoch:03d}] Sampling patches...")
            train_df = sampler.sample_train_patches(
                case_ids=train_ids,
                num_epoch_cycles=cfg.num_epoch_cycles,
                seed=None,
            )

            train_dataset = PatchDataset(
                train_df, cfg.nnunet_preproc_dir,
                cfg.biomedparse_preproc_train, self.patch_size
            )
            train_loader = DataLoader(
                train_dataset, batch_size=cfg.batch_size,
                shuffle=True, num_workers=4, pin_memory=True,
                drop_last=True, collate_fn=custom_collate_fn
            )

            # Train
            self._set_fusion_mode(train=True)
            meter = {k: 0.0 for k in ["loss_total", "loss_seg", "loss_ae", "loss_ortho", "loss_ka"]}
            dice_pc_sum = np.zeros(self.P, dtype=np.float64)
            dice_pc_cnt = np.zeros(self.P, dtype=np.int64)
            n_batches = 0

            for batch_data in tqdm(train_loader, desc=f"[Train {epoch:03d}]", leave=False):
                total_loss, logs = forward_one_batch(
                    batch_data=batch_data,
                    patch_size=self.patch_size,
                    prompt_features=self.prompt_features,
                    class_emb=self.class_emb,
                    P=self.P,
                    prompt_to_class_id=self.prompt_to_class_id,
                    w_seg=cfg.w_seg, w_ae=cfg.w_ae,
                    w_ort=cfg.w_ort, w_ka=cfg.w_ka,
                    model_nnunet=self.model_nnunet,
                    model_biomedparse=self.model_biomedparse,
                    fusion_modules=self.fusion_modules,
                    device=self.device,
                    w_p_reg=w_p_reg,
                    train=True,
                )

                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                all_params = []
                for m in self.fusion_modules.values():
                    all_params.extend(m.parameters())
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=5.0)
                self.optimizer.step()

                for k in meter:
                    meter[k] += logs.get(k, 0.0)
                for pi in range(self.P):
                    d = logs.get("dice_per_class", {}).get(pi)
                    if d is not None:
                        dice_pc_sum[pi] += d
                        dice_pc_cnt[pi] += 1
                n_batches += 1

            # Average
            for k in meter:
                meter[k] /= max(n_batches, 1)
            train_dice_pc = {
                i: float(dice_pc_sum[i] / dice_pc_cnt[i]) if dice_pc_cnt[i] > 0 else None
                for i in range(self.P)
            }
            train_dice_mean = float(np.mean([v for v in train_dice_pc.values() if v is not None])) \
                if any(v is not None for v in train_dice_pc.values()) else 0.0

            _log(f"[Epoch {epoch:03d}] Train: loss={meter['loss_total']:.4f} dice={train_dice_mean:.4f}")

            # Validation
            val_dice_mean = 0.0
            val_dice_pc = {i: None for i in range(self.P)}
            if epoch % cfg.val_every_epochs == 0 or epoch == cfg.n_epochs:
                val_dice_mean, val_dice_pc_list = self._validate_full_cases(
                    val_ids, epoch
                )
                val_dice_pc = {i: val_dice_pc_list[i] for i in range(self.P)}
                _log(f"[Epoch {epoch:03d}] Val: mean_dice={val_dice_mean:.4f}")
                for i in range(self.P):
                    _log(f"  class {self.prompt_to_class_id[i]}: {val_dice_pc_list[i]:.4f}")

                if val_dice_mean > self.best_val_dice:
                    self.best_val_dice = val_dice_mean
                    self._save_checkpoint(epoch, best=True)
                    _log(f"  -> New best model saved (dice={val_dice_mean:.4f})")

            # Update history
            self.history["epochs"].append(epoch)
            for k in ["loss_total", "loss_seg", "loss_ae", "loss_ortho", "loss_ka"]:
                self.history[f"train_{k}"].append(meter[k])
                self.history[f"val_{k}"].append(0.0)
            self.history["train_dice_mean"].append(train_dice_mean)
            self.history["val_dice_mean"].append(val_dice_mean)
            self.history["train_dice_per_class"].append(train_dice_pc)
            self.history["val_dice_per_class"].append(val_dice_pc)

            plot_training_curves(self.history, cfg.output_dir, epoch, self.P)

        log_fh.close()
        _log(f"\nTraining complete. Best val dice: {self.best_val_dice:.4f}")

    @torch.no_grad()
    def _validate_full_cases(
        self, val_ids: List[str], epoch: int
    ) -> Tuple[float, List[float]]:
        ensure_nnunet_on_path()
        from nnunetv2.inference.sliding_window_prediction import compute_steps_for_sliding_window, compute_gaussian
        from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class

        cfg = self.cfg
        self._set_fusion_mode(train=False)
        class_ids = [int(self.prompt_to_class_id[i]) for i in range(self.P)]
        patch_size_zyx = tuple(int(x) for x in self.patch_size)
        gaussian = compute_gaussian(
            patch_size_zyx, sigma_scale=1.0 / 8, value_scaling_factor=10, device=torch.device("cpu")
        ).cpu().numpy().astype(np.float32)

        ds_cls = infer_dataset_class(cfg.nnunet_preproc_dir)
        nn_ds = ds_cls(cfg.nnunet_preproc_dir)

        dice_means = []
        per_class_lists = [[] for _ in range(self.P)]

        for case_id in tqdm(val_ids, desc=f"[Val@{epoch:03d}]", leave=False):
            nn_data, nn_seg, _prev, _props = nn_ds.load_case(case_id)
            nn_data = np.asarray(nn_data, dtype=np.float32)
            gt = np.asarray(nn_seg, dtype=np.int16)[0]
            Z, Y, X = gt.shape

            bp_npz = os.path.join(cfg.biomedparse_preproc_val, f"{case_id}.npz")
            bp_data = np.load(bp_npz)["data"].astype(np.float32)

            # Pad to min patch size
            def _pad(arr, pad_val):
                z, y, x = arr.shape[-3:]
                pz = max(0, patch_size_zyx[0] - z)
                py = max(0, patch_size_zyx[1] - y)
                px = max(0, patch_size_zyx[2] - x)
                p0 = ((pz // 2, pz - pz // 2), (py // 2, py - py // 2), (px // 2, px - px // 2))
                if arr.ndim == 4:
                    return np.pad(arr, ((0, 0),) + tuple(p0), constant_values=pad_val), p0
                return np.pad(arr, p0, constant_values=pad_val), p0

            nn_data_p, pad = _pad(nn_data, 0.0)
            bp_data_p, _ = _pad(bp_data, 0.0)
            gt_p, _ = _pad(gt, -1)

            Zp, Yp, Xp = gt_p.shape
            steps = compute_steps_for_sliding_window((Zp, Yp, Xp), patch_size_zyx, float(cfg.tile_step_size))
            slicers = [
                (slice(z0, z0 + patch_size_zyx[0]), slice(y0, y0 + patch_size_zyx[1]), slice(x0, x0 + patch_size_zyx[2]))
                for z0 in steps[0] for y0 in steps[1] for x0 in steps[2]
            ]

            logits_sum = np.zeros((self.P, Zp, Yp, Xp), dtype=np.float32)
            wsum = np.zeros((Zp, Yp, Xp), dtype=np.float32)
            is_multi = bp_data_p.ndim == 4

            for slz, sly, slx in slicers:
                nn_patch = nn_data_p[:, slz, sly, slx]
                if is_multi:
                    bp_patch_t = torch.from_numpy(bp_data_p[:, slz, sly, slx][None])
                else:
                    bp_patch_t = torch.from_numpy(bp_data_p[slz, sly, slx][None])
                nn_patch_t = torch.from_numpy(nn_patch[None])

                patch_logits = predict_patch_logits_per_class(
                    nnunet_patch_5d=nn_patch_t,
                    biomedparse_patch_4d=bp_patch_t,
                    patch_size=self.patch_size,
                    prompt_features=self.prompt_features,
                    P=self.P,
                    model_nnunet=self.model_nnunet,
                    model_biomedparse=self.model_biomedparse,
                    fusion_modules=self.fusion_modules,
                    device=self.device,
                ).cpu().numpy().astype(np.float32)

                logits_sum[:, slz, sly, slx] += patch_logits * gaussian[None]
                wsum[slz, sly, slx] += gaussian

            fused = logits_sum / np.clip(wsum[None], 1e-6, None)
            # Crop padding
            pz0 = pad[0][0]; py0 = pad[1][0]; px0 = pad[2][0]
            fused = fused[:, pz0:pz0 + Z, py0:py0 + Y, px0:px0 + X]

            full = np.concatenate([np.zeros((1, Z, Y, X), dtype=np.float32), fused], axis=0)
            pred_compact = np.argmax(full, axis=0).astype(np.int16)

            pred_orig = np.zeros_like(gt, dtype=np.int16)
            for pi in range(self.P):
                pred_orig[pred_compact == (pi + 1)] = int(self.prompt_to_class_id[pi])

            dpc, md, _ = dice_ignore_minus_one(pred_orig, gt, class_ids)
            dice_means.append(md)
            for pi in range(self.P):
                c = class_ids[pi]
                if dpc.get(c) is not None:
                    per_class_lists[pi].append(dpc[c])

        mean_overall = float(np.mean(dice_means)) if dice_means else 0.0
        per_class_mean = [
            float(np.mean(per_class_lists[pi])) if per_class_lists[pi] else float("nan")
            for pi in range(self.P)
        ]
        return mean_overall, per_class_mean

    def _save_checkpoint(self, epoch: int, best: bool = False):
        state = {"epoch": epoch, "best_val_dice": self.best_val_dice}
        for name, m in self.fusion_modules.items():
            state[name] = m.state_dict()
        state["optimizer"] = self.optimizer.state_dict()

        fname = "fusion_disentangle_best.pth" if best else f"checkpoint_epoch{epoch:03d}.pth"
        path = os.path.join(self.cfg.output_dir, fname)
        torch.save(state, path)
