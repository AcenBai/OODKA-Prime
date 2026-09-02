#!/usr/bin/env python3
"""Train a lightweight protected AO/PA correction gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from oodka.eval.correction_gate import CorrectionGate, FEATURE_NAMES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hidden_dims", default="32,16")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--positive_weight", type=float, default=1.0)
    parser.add_argument("--negative_weight", type=float, default=1.0)
    parser.add_argument("--focal_gamma", type=float, default=1.0)
    parser.add_argument("--candidate_min_probability", type=float, default=0.5)
    parser.add_argument("--gv_core_threshold", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    hidden_dims = tuple(
        int(value.strip()) for value in args.hidden_dims.split(",") if value.strip()
    )
    if not hidden_dims:
        raise ValueError("hidden_dims must not be empty")

    with np.load(args.dataset) as payload:
        features = torch.from_numpy(payload["features"].astype(np.float32))
        targets = torch.from_numpy(payload["targets"].astype(np.float32))
        feature_names = tuple(str(value) for value in payload["feature_names"])
        case_ids = tuple(str(value) for value in payload["case_ids"])
    if feature_names != FEATURE_NAMES:
        raise ValueError("Correction feature schema mismatch")
    dataset = TensorDataset(features, targets)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )
    model = CorrectionGate(
        input_dim=len(feature_names),
        hidden_dims=hidden_dims,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.05
    )
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        correct = 0
        true_positive = 0
        predicted_positive = 0
        target_positive = 0
        count = 0
        for batch_features, batch_targets in loader:
            batch_features = batch_features.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)
            logits = model(batch_features)
            bce = F.binary_cross_entropy_with_logits(
                logits, batch_targets, reduction="none"
            )
            probabilities = torch.sigmoid(logits)
            pt = torch.where(batch_targets > 0.5, probabilities, 1.0 - probabilities)
            class_weight = torch.where(
                batch_targets > 0.5,
                bce.new_tensor(args.positive_weight),
                bce.new_tensor(args.negative_weight),
            )
            loss = (class_weight * (1.0 - pt).pow(args.focal_gamma) * bce).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            prediction = probabilities >= 0.5
            target_bool = batch_targets > 0.5
            batch_count = len(batch_targets)
            loss_sum += float(loss.detach()) * batch_count
            correct += int((prediction == target_bool).sum())
            true_positive += int((prediction & target_bool).sum())
            predicted_positive += int(prediction.sum())
            target_positive += int(target_bool.sum())
            count += batch_count
        scheduler.step()
        record = {
            "epoch": epoch,
            "loss": loss_sum / max(1, count),
            "accuracy": correct / max(1, count),
            "precision": true_positive / max(1, predicted_positive),
            "recall": true_positive / max(1, target_positive),
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(record)
        print(json.dumps(record))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "oodka_correction_gate_v1",
            "model": model.state_dict(),
            "feature_names": feature_names,
            "hidden_dims": hidden_dims,
            "dropout": args.dropout,
            "candidate_min_probability": args.candidate_min_probability,
            "gv_core_threshold": args.gv_core_threshold,
            "training_case_ids": case_ids,
            "config": vars(args)
            | {"dataset": str(args.dataset), "output": str(args.output)},
            "history": history,
        },
        args.output,
    )


if __name__ == "__main__":
    main()
