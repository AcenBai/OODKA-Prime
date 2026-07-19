"""BiomedParse predictor interaction helpers."""

from __future__ import annotations

from typing import List, Tuple

import torch


def parse_pixel_decoder_out(pd_out) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Extract mask_features and multi_scale_features from pixel_decoder output."""
    if isinstance(pd_out, (tuple, list)) and len(pd_out) >= 1:
        mask_features = pd_out[0]
        multi_scale = None
        for item in pd_out[1:]:
            if isinstance(item, (list, tuple)) and len(item) > 0 and torch.is_tensor(item[0]):
                multi_scale = list(item)
                break
        if multi_scale is None:
            raise RuntimeError("pixel_decoder did not return multi_scale_features")
        return mask_features, multi_scale
    raise RuntimeError("Unexpected pixel_decoder output type")


def expand_prompt_features_for_blocks(
    prompt_features: dict,
    *,
    B: int,
    Z: int,
    P: int,
) -> dict:
    """Align prompt embeddings with visual samples flattened as ``[B,Z,P]``."""
    visual_count = int(B) * int(Z)
    pair_count = visual_count * int(P)
    expanded = prompt_features.copy()

    grounding = prompt_features.get("grounding_tokens")
    if not torch.is_tensor(grounding) or grounding.ndim != 3:
        raise ValueError(
            "grounding_tokens must be a tensor shaped [L,P,D], "
            f"got {type(grounding).__name__}"
        )
    if grounding.shape[1] != P:
        raise ValueError(
            f"grounding_tokens prompt count={grounding.shape[1]} != P={P}"
        )
    expanded["grounding_tokens"] = (
        grounding[:, None, :, :]
        .expand(-1, visual_count, -1, -1)
        .reshape(grounding.shape[0], pair_count, grounding.shape[2])
        .contiguous()
    )

    class_emb = prompt_features.get("class_emb")
    if torch.is_tensor(class_emb):
        if class_emb.ndim != 2 or class_emb.shape[0] != P:
            raise ValueError(f"class_emb must be [P,D] with P={P}, got {class_emb.shape}")
        expanded["class_emb"] = (
            class_emb[None]
            .expand(visual_count, -1, -1)
            .reshape(pair_count, class_emb.shape[1])
            .contiguous()
        )

    num_prompts = prompt_features.get("num_prompts")
    if torch.is_tensor(num_prompts):
        expanded["num_prompts"] = torch.ones(
            pair_count,
            device=num_prompts.device,
            dtype=num_prompts.dtype,
        )
    return expanded


def select_best_mask_from_queries(
    pred_gmasks: torch.Tensor,
    object_existence: torch.Tensor = None,
) -> torch.Tensor:
    """Aggregate masks from multiple queries (mean pooling)."""
    return pred_gmasks.mean(dim=1)


def run_biomedparse_predictor_override(
    sem_seg_head, multi_scale_features, mask_features, prompt_features,
) -> dict:
    """Call the official predictor with already injected and aligned features."""
    visual_batch = (
        multi_scale_features[0].shape[0]
        if multi_scale_features
        else mask_features.shape[0]
    )
    if mask_features.shape[0] != visual_batch or any(
        feature.shape[0] != visual_batch for feature in multi_scale_features
    ):
        raise ValueError("All predictor visual features must share the same batch size")
    pf = prompt_features.copy()

    if "grounding_tokens" in pf and torch.is_tensor(pf["grounding_tokens"]):
        gt = pf["grounding_tokens"]
        if gt.ndim != 3:
            raise ValueError(f"grounding_tokens must be [L,N,D], got {gt.shape}")
        if gt.shape[1] == 1 and visual_batch > 1:
            gt = gt.expand(-1, visual_batch, -1)
        elif gt.shape[1] != visual_batch:
            raise ValueError(
                f"grounding token batch={gt.shape[1]} != visual batch={visual_batch}"
            )
        pf["grounding_tokens"] = gt

    if "class_emb" in pf and torch.is_tensor(pf["class_emb"]):
        ce = pf["class_emb"]
        if ce.ndim != 2:
            raise ValueError(f"class_emb must be [N,D], got {ce.shape}")
        if ce.shape[0] == 1 and visual_batch > 1:
            ce = ce.expand(visual_batch, -1)
        elif ce.shape[0] != visual_batch:
            raise ValueError(
                f"class embedding batch={ce.shape[0]} != visual batch={visual_batch}"
            )
        pf["class_emb"] = ce

    if hasattr(sem_seg_head, "predictor"):
        return sem_seg_head.predictor(multi_scale_features, mask_features, mask=None, extra=pf)
    raise RuntimeError("Could not call BiomedParse predictor")
