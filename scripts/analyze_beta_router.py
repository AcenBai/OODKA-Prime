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

def _statistics(router, model, prompts, device, spatial_size):
    encoded = build_prompt_features(model, prompts, device)
    with torch.no_grad():
        output = router(
            encoded["class_emb"].detach(),
            spatial_size=spatial_size,
            batch_size=1,
            sample=False,
        )
    rows = []
    for index, key in enumerate(sorted(prompts, key=int)):
        gate = output["mean"][index]
        rows.append(
            {
                "class_id": int(key),
                "prompt": prompts[key],
                "spatial_gate": {
                    "height": int(gate.shape[0]),
                    "width": int(gate.shape[1]),
                    "mean": float(gate.mean().item()),
                    "std": float(gate.std().item()),
                    "min": float(gate.min().item()),
                    "max": float(gate.max().item()),
                    "concentration_mean": float(
                        output["concentration"][index].mean().item()
                    ),
                },
            }
        )
    return rows, output["mean"].detach().cpu()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument(
        "--output",
        default="outputs/oodka_ot_experiments/interpretability/beta_router.json",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if "beta_router" not in checkpoint:
        raise KeyError("checkpoint does not contain beta_router")
    checkpoint_cfg = checkpoint.get("config", {})
    router = PromptBetaRouter(
        text_dim=512,
        prior_p_mean=float(
            checkpoint_cfg.get("route_prior_p_mean", 0.7)
        ),
        prior_concentration=float(
            checkpoint_cfg.get("route_prior_concentration", 10.0)
        ),
        basis_grid_size=int(
            checkpoint_cfg.get("route_spatial_basis_grid_size", 8)
        ),
        basis_sigma=float(
            checkpoint_cfg.get("route_spatial_basis_sigma", 0.0)
        ),
    ).to(device).eval()
    router.load_state_dict(checkpoint["beta_router"])
    model = load_frozen_biomedparse(device)

    spatial_size = (args.height, args.width)
    seen, seen_maps = _statistics(
        router, model, WHS_CT_PROMPTS, device, spatial_size
    )
    paraphrased, paraphrased_maps = _statistics(
        router, model, PARAPHRASED_PROMPTS, device, spatial_size
    )
    mean_abs_shift = float((seen_maps - paraphrased_maps).abs().mean().item())
    report = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "spatial_size": list(spatial_size),
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
