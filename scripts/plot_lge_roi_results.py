#!/usr/bin/env python3
"""Plot the compact diagnostics produced by the LGE two-pass experiment."""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_dir", required=True)
    args = parser.parse_args()

    history_path = os.path.join(args.experiment_dir, "history.json")
    with open(history_path, encoding="utf-8") as handle:
        history = json.load(handle)
    plots_dir = os.path.join(args.experiment_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    axes[0, 0].plot(epochs, [row["train"]["loss_total"] for row in history], label="train")
    val_rows = [row for row in history if row["val"] is not None]
    axes[0, 0].plot(
        [row["epoch"] for row in val_rows],
        [row["val"]["loss_total"] for row in val_rows],
        label="val",
    )
    axes[0, 0].set_title("Joint loss")
    axes[0, 0].legend()

    axes[0, 1].plot(
        [row["epoch"] for row in val_rows],
        [row["val"]["exclusive_macro_dice"] for row in val_rows],
        marker="o", label="validation",
    )
    test_rows = [
        row for row in history if row.get("test_best_diagnostic") is not None
    ]
    axes[0, 1].plot(
        [row["epoch"] for row in test_rows],
        [
            row["test_best_diagnostic"]["mean_dice_gt_present"]
            for row in test_rows
        ],
        marker="x",
        label="diagnostic test",
    )
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_title("Validation exclusive argmax Dice")
    axes[0, 1].legend()

    axes[0, 2].plot(
        epochs,
        [row["train"]["anchor_sigmoid_dice"] for row in history],
        label="Pass1 anchor",
    )
    axes[0, 2].plot(
        epochs,
        [row["train"]["refine_sigmoid_dice"] for row in history],
        label="Pass2 ROI",
    )
    axes[0, 2].set_ylim(0, 1)
    axes[0, 2].set_title("Train independent-sigmoid Dice")
    axes[0, 2].legend()

    mixed_val = [row for row in val_rows if row["val"]["mode"] == "mixed"]
    for class_id, label in ((1, "LV"), (2, "RV"), (3, "normal_myo"), (4, "scar_edema")):
        axes[1, 0].plot(
            [row["epoch"] for row in mixed_val],
            [row["val"]["exclusive_dice_per_class"].get(str(class_id), 0.0) for row in mixed_val],
            marker=".",
            label=label,
        )
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_title("Validation Dice by final class")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(
        [row["epoch"] for row in mixed_val],
        [row["val"]["roi_gt_recall"] for row in mixed_val],
        label="GT myocardium recall",
    )
    axes[1, 1].plot(
        [row["epoch"] for row in mixed_val],
        [row["val"]["roi_area_fraction"] for row in mixed_val],
        label="ROI area fraction",
    )
    axes[1, 1].plot(
        [row["epoch"] for row in mixed_val],
        [row["val"]["roi_fallback_rate"] for row in mixed_val],
        label="fallback rate",
    )
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_title("ROI behavior")
    axes[1, 1].legend(fontsize=8)

    axes[1, 2].plot(
        [row["epoch"] for row in mixed_val],
        [row["val"]["roi_lv_loss_valid_rate"] for row in mixed_val],
        label="LV valid",
    )
    axes[1, 2].plot(
        [row["epoch"] for row in mixed_val],
        [row["val"]["roi_rv_loss_valid_rate"] for row in mixed_val],
        label="RV valid",
    )
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].set_title("ROI cavity supervision visibility")
    axes[1, 2].legend(fontsize=8)

    for axis in axes.flat:
        axis.grid(alpha=0.25)
        axis.set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "lge_roi_training_summary.png"), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
