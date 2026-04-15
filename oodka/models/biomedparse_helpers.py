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


def slice_prompt_features(prompt_features: dict, idx: int) -> dict:
    """Extract single-prompt features from a multi-prompt batch."""
    pf = {}
    for k, v in prompt_features.items():
        if not torch.is_tensor(v):
            pf[k] = v
            continue
        if k == "grounding_tokens":
            if v.ndim == 3 and v.shape[1] > 1:
                pf[k] = v[:, idx:idx + 1, :]
            elif v.ndim == 3 and v.shape[0] > 1:
                pf[k] = v[idx:idx + 1]
            else:
                pf[k] = v
        elif k == "class_emb":
            pf[k] = v[idx:idx + 1, :]
        elif k == "num_prompts":
            pf[k] = torch.tensor([1], device=v.device, dtype=v.dtype)
        else:
            pf[k] = v

    if "text" in prompt_features and "text" not in pf:
        text_val = prompt_features["text"]
        pf["text"] = [text_val[idx]] if isinstance(text_val, list) and len(text_val) > 1 else text_val
    return pf


def select_best_mask_from_queries(
    pred_gmasks: torch.Tensor,
    object_existence: torch.Tensor = None,
) -> torch.Tensor:
    """Aggregate masks from multiple queries (mean pooling)."""
    return pred_gmasks.mean(dim=1)


def run_biomedparse_predictor_override(
    sem_seg_head, multi_scale_features, mask_features, prompt_features,
) -> dict:
    """Call BiomedParse predictor with injected features."""
    N = multi_scale_features[0].shape[0] if multi_scale_features else mask_features.shape[0]
    pf = prompt_features.copy()

    if "grounding_tokens" in pf and torch.is_tensor(pf["grounding_tokens"]):
        gt = pf["grounding_tokens"]
        if gt.ndim == 3:
            pf["grounding_tokens"] = gt.repeat(1, N, 1)

    if "class_emb" in pf and torch.is_tensor(pf["class_emb"]):
        ce = pf["class_emb"]
        if ce.ndim == 2:
            pf["class_emb"] = ce.repeat(N, 1)

    if hasattr(sem_seg_head, "predictor"):
        return sem_seg_head.predictor(multi_scale_features, mask_features, mask=None, extra=pf)
    raise RuntimeError("Could not call BiomedParse predictor")
