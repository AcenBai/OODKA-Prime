#!/usr/bin/env python3
"""Export prompt-wise Beta routing statistics for seen and paraphrased prompts."""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch

from oodka.models.beta_router import PromptBetaRouter
from oodka.models.prompts import WHS_CT_PROMPTS
from oodka.train.model_builder import build_prompt_features, load_frozen_biomedparse


PARAPHRASED_PROMPTS = {
    "1": "left ventricular blood pool on a cardiac CT scan",
    "2": "right ventricle cavity visible in CT imaging",
    "3": "left atrial chamber on contrast-enhanced cardiac CT",
    "4": "right atrial blood cavity in a CT volume",
    "5": "muscular wall of the left ventricle on CT",
    "6": "the proximal ascending aorta on thoracic CT",
    "7": "main pulmonary arterial trunk in cardiac CT",
}

LEVEL_NAMES = ("res2", "res3", "res4", "res5")


def _statistics(router, model, prompts, device):
    encoded = build_prompt_features(model, prompts, device)
    with torch.no_grad():
        output = router(encoded["class_emb"].detach(), batch_size=1, sample=False)
    rows = []
    for index, key in enumerate(sorted(prompts, key=int)):
        rows.append(
            {
                "class_id": int(key),
                "prompt": prompts[key],
                "levels": {
                    level: {
                        "alpha": float(output["alpha"][index, level_index].item()),
                        "beta": float(output["beta"][index, level_index].item()),
                        "gate_mean": float(output["mean"][index, level_index].item()),
                        "concentration": float(
                            output["concentration"][index, level_index].item()
                        ),
                    }
                    for level_index, level in enumerate(LEVEL_NAMES)
                },
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument(
        "--output",
        default="outputs/oodka_ot_experiments/interpretability/beta_router.json",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if "beta_router" not in checkpoint:
        raise KeyError("checkpoint does not contain beta_router")
    router = PromptBetaRouter(text_dim=512).to(device).eval()
    router.load_state_dict(checkpoint["beta_router"])
    model = load_frozen_biomedparse(device)

    seen = _statistics(router, model, WHS_CT_PROMPTS, device)
    paraphrased = _statistics(router, model, PARAPHRASED_PROMPTS, device)
    mean_abs_shift = sum(
        abs(
            left["levels"][level]["gate_mean"]
            - right["levels"][level]["gate_mean"]
        )
        for left, right in zip(seen, paraphrased)
        for level in LEVEL_NAMES
    ) / (len(seen) * len(LEVEL_NAMES))
    report = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "seen": seen,
        "paraphrased": paraphrased,
        "mean_absolute_gate_shift": mean_abs_shift,
    }
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file_handle:
        json.dump(report, file_handle, indent=2)
    print(json.dumps(report, indent=2))
    print(f"saved={output_path}")


if __name__ == "__main__":
    main()
