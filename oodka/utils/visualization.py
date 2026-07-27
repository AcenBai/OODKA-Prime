"""Training curve visualization."""

from __future__ import annotations

import os
from typing import Dict

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_training_curves(
    history: Dict,
    output_dir: str,
    epoch: int,
    P: int,
):
    log_dir = os.path.join(output_dir, "plots")
    os.makedirs(log_dir, exist_ok=True)
    epochs = history["epochs"]

    # 1. Loss curves
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for ax, (train_key, val_key, title) in zip(
        axes.flat,
        [
            ("train_loss_total", "val_loss_total", "Total Loss"),
            ("train_loss_seg", "val_loss_seg", "Segmentation Loss"),
            ("train_loss_ae", "val_loss_ae", "Autoencoder Loss"),
            ("train_loss_ortho", "val_loss_ortho", "P/S Separation Loss"),
        ],
    ):
        ax.plot(epochs, history[train_key], "b-", label="Train", linewidth=2)
        ax.plot(epochs, history[val_key], "r-", label="Val", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "loss_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Per-class dice
    n_rows = (P + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(15, 4 * n_rows))
    axes = np.atleast_1d(axes).flatten()
    for p_idx in range(P):
        train_d = [np.nan if h.get(p_idx) is None else h[p_idx] for h in history["train_dice_per_class"]]
        val_d = [np.nan if h.get(p_idx) is None else h[p_idx] for h in history["val_dice_per_class"]]
        axes[p_idx].plot(epochs, train_d, "b-o", label="Train", markersize=4)
        axes[p_idx].plot(epochs, val_d, "r-s", label="Val", markersize=4)
        axes[p_idx].set_title(f"Class {p_idx} Dice")
        axes[p_idx].set_ylim([0, 1])
        axes[p_idx].legend()
        axes[p_idx].grid(True, alpha=0.3)
    for i in range(P, len(axes)):
        axes[i].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "dice_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 3. Mean dice
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, history["train_dice_mean"], "b-o", label="Train", markersize=5)
    ax.plot(epochs, history["val_dice_mean"], "r-s", label="Val", markersize=5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Dice")
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "mean_dice.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 4. OT and route losses
    fig, ax = plt.subplots(figsize=(10, 6))
    for key, label in [
        ("train_loss_p_ot", "Train P-OT"),
        ("train_loss_s_ot", "Train S-UOT"),
        ("train_loss_route", "Train Route KL"),
    ]:
        ax.plot(epochs, history[key], label=label, linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_title(f"OT / Route Losses (Epoch {epoch})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "ot_route_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()
