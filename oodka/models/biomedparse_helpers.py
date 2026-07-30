"""BiomedParse predictor interaction helpers."""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F


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


def gates_for_biomedparse_predictor(
    gate: torch.Tensor, *, B: int, P: int
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Map [res2,res3,res4,res5] gates to mask and coarse-to-fine inputs."""
    if gate.shape != (B, P, 4):
        raise ValueError(f"gate must be [B,P,4]=[{B},{P},4], got {gate.shape}")
    return gate[:, :, 0], [gate[:, :, i] for i in (3, 2, 1)]


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


def aggregate_initial_query_masks(
    initial_masks: torch.Tensor,
    *,
    topk: int,
) -> torch.Tensor:
    """Create one detached soft spatial proposal from query-wise mask logits.

    The strongest ``topk`` query probabilities are averaged independently at
    every spatial position. This preserves a soft union of complementary query
    hypotheses without letting a single query dominate everywhere.
    """
    if initial_masks.ndim != 4:
        raise ValueError(
            f"initial_masks must be [N,Q,H,W], got {initial_masks.shape}"
        )
    query_count = int(initial_masks.shape[1])
    if query_count <= 0:
        raise ValueError("initial_masks must contain at least one query")
    k = min(max(int(topk), 1), query_count)
    probabilities = initial_masks.detach().float().sigmoid()
    return probabilities.topk(k, dim=1).values.mean(dim=1, keepdim=True)


def inject_query_guided_residual(
    p_feature: torch.Tensor,
    s_feature: torch.Tensor,
    proposal: torch.Tensor,
    *,
    s_floor: float,
) -> torch.Tensor:
    """Return ``P + [floor + (1-floor) R] * S`` at one feature scale."""
    if p_feature.shape != s_feature.shape or p_feature.ndim != 4:
        raise ValueError(
            "P/S features must share [N,C,H,W], got "
            f"{p_feature.shape} and {s_feature.shape}"
        )
    if proposal.ndim != 4 or proposal.shape[:2] != (
        p_feature.shape[0],
        1,
    ):
        raise ValueError(
            f"proposal must be [N,1,H,W] with N={p_feature.shape[0]}, "
            f"got {proposal.shape}"
        )
    floor = float(s_floor)
    if not 0.0 <= floor <= 1.0:
        raise ValueError(f"s_floor must be in [0,1], got {floor}")
    resized = F.interpolate(
        proposal.to(dtype=p_feature.dtype),
        size=p_feature.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    residual_gate = floor + (1.0 - floor) * resized
    return p_feature + residual_gate * s_feature


def run_query_guided_s_predictor(
    sem_seg_head,
    multi_scale_p: List[torch.Tensor],
    multi_scale_s: List[torch.Tensor],
    mask_features_p: torch.Tensor,
    mask_features_s: torch.Tensor,
    prompt_features: dict,
    *,
    proposal_topk: int = 4,
    s_floor: float = 0.2,
    run_object_existence: bool = False,
) -> dict:
    """Run frozen BoltzFormer once with P-controlled S residual injection.

    This reproduces the official decoder loop without modifying BiomedParse
    source code. The official prompt-query pre-attention and initial mask head
    first query ``mask_features_p``. Their detached initial masks form a single
    proposal that gates every S feature scale. The original decoder layers then
    refine queries using ``P + gated S`` memories.
    """
    if not hasattr(sem_seg_head, "predictor"):
        raise RuntimeError("sem_seg_head does not expose a predictor")
    predictor = sem_seg_head.predictor
    if len(multi_scale_p) != predictor.num_feature_levels:
        raise ValueError(
            f"Expected {predictor.num_feature_levels} P scales, "
            f"got {len(multi_scale_p)}"
        )
    if len(multi_scale_s) != len(multi_scale_p):
        raise ValueError("P/S multi-scale feature counts must match")
    if mask_features_p.shape != mask_features_s.shape:
        raise ValueError("P/S mask features must share shape")
    visual_batch = int(mask_features_p.shape[0])
    for p_feature, s_feature in zip(multi_scale_p, multi_scale_s):
        if p_feature.shape != s_feature.shape:
            raise ValueError("P/S features must share every scale shape")
        if p_feature.shape[0] != visual_batch:
            raise ValueError("All visual features must share batch size")

    grounding_tokens = prompt_features.get("grounding_tokens")
    if (
        not torch.is_tensor(grounding_tokens)
        or grounding_tokens.ndim != 3
        or grounding_tokens.shape[1] != visual_batch
    ):
        raise ValueError(
            "grounding_tokens must be [L,N,D] aligned with the visual batch"
        )

    query_embed = predictor.query_embed_.weight.unsqueeze(1).repeat(
        1, visual_batch, 1
    )
    output = predictor.query_feat_.weight.unsqueeze(1).repeat(
        1, visual_batch, 1
    )
    text_output = grounding_tokens
    text_embed = text_output.detach().clone()

    if predictor.pre_self_attention:
        combined_output = torch.cat([output, text_output], dim=0)
        combined_embed = torch.cat([query_embed, text_embed], dim=0)
        combined_output, _ = predictor.initial_self_attention_layer(
            combined_output,
            tgt_mask=None,
            tgt_key_padding_mask=None,
            query_pos=combined_embed,
        )
        output = combined_output[: predictor.num_queries]
        text_output = combined_output[predictor.num_queries :]
        output = predictor.initial_ffn_layer(output)

    initial_masks, attention_mask = predictor.forward_prediction_heads(
        output,
        mask_features_p,
        attn_mask_target_size=multi_scale_p[0].shape[-2:],
    )
    proposal = aggregate_initial_query_masks(
        initial_masks,
        topk=proposal_topk,
    )
    fused_mask_features = inject_query_guided_residual(
        mask_features_p,
        mask_features_s,
        proposal,
        s_floor=s_floor,
    )
    fused_multi_scale = [
        inject_query_guided_residual(
            p_feature,
            s_feature,
            proposal,
            s_floor=s_floor,
        )
        for p_feature, s_feature in zip(multi_scale_p, multi_scale_s)
    ]

    src = []
    pos = []
    size_list = []
    for level_index, feature in enumerate(fused_multi_scale):
        size_list.append(feature.shape[-2:])
        pos_embed = (
            predictor.pe_layer(feature, None)
            .flatten(2)
            .permute(2, 0, 1)
        )
        projected = (
            predictor.input_proj[level_index](feature).flatten(2)
            + predictor.level_embed.weight[level_index][None, :, None]
        )
        pos.append(pos_embed)
        src.append(projected.permute(2, 0, 1))

    # OODKA deliberately does not use the domain-mismatched official
    # object-existence classifier. It can still be enabled for controlled
    # compatibility checks without affecting the mask-query path.
    output_classifier = bool(
        run_object_existence
        and getattr(predictor, "output_classifier", False)
    )
    cls_output = None
    if output_classifier:
        cls_output = predictor.classifier_query.weight.unsqueeze(1).repeat(
            1, visual_batch, 1
        )

    predictions_mask = [initial_masks]
    for layer_index in range(predictor.num_layers):
        level_index = layer_index % predictor.num_feature_levels
        fully_masked = attention_mask.sum(-1) == attention_mask.shape[-1]
        attention_mask[fully_masked] = False
        output, _ = predictor.transformer_cross_attention_layers[layer_index](
            output,
            src[level_index],
            memory_mask=attention_mask,
            memory_key_padding_mask=None,
            pos=pos[level_index],
            query_pos=query_embed,
        )

        combined_output = torch.cat([output, text_output], dim=0)
        combined_embed = torch.cat([query_embed, text_embed], dim=0)
        combined_output, _ = predictor.transformer_self_attention_layers[
            layer_index
        ](
            combined_output,
            tgt_mask=None,
            tgt_key_padding_mask=None,
            query_pos=combined_embed,
        )
        output = combined_output[: predictor.num_queries]
        text_output = combined_output[predictor.num_queries :]
        output = predictor.transformer_ffn_layers[layer_index](output)

        outputs_mask, attention_mask = predictor.forward_prediction_heads(
            output,
            fused_mask_features,
            attn_mask_target_size=size_list[
                (layer_index + 1) % predictor.num_feature_levels
            ],
            layer_id=layer_index,
        )
        predictions_mask.append(outputs_mask)

        if output_classifier:
            cls_output, _ = predictor.classifier_cross_attention_layers[
                layer_index
            ](
                cls_output,
                combined_output.detach(),
                memory_mask=None,
                memory_key_padding_mask=None,
                pos=combined_embed.detach(),
                query_pos=None,
            )
            cls_output = predictor.classifier_ffn_layers[layer_index](
                cls_output
            )

    result = {
        "pred_gmasks": predictions_mask[-1],
        "initial_p_gmasks": initial_masks,
        "proposal_map": proposal,
    }
    if output_classifier:
        result["object_existence"] = predictor.classifier(cls_output[0])
    return result
