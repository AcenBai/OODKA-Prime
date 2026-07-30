import torch
import torch.nn as nn

from oodka.config import ensure_biomedparse_on_path
from oodka.models.biomedparse_helpers import (
    aggregate_initial_query_masks,
    inject_query_guided_residual,
    run_query_guided_s_predictor,
)


def test_aggregate_initial_query_masks_is_detached_topk_soft_union():
    probabilities = torch.tensor(
        [[[[0.1, 0.8]], [[0.7, 0.4]], [[0.5, 0.9]]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    logits = torch.logit(probabilities)

    proposal = aggregate_initial_query_masks(logits, topk=2)

    expected = torch.tensor([[[[(0.7 + 0.5) / 2, (0.9 + 0.8) / 2]]]])
    torch.testing.assert_close(proposal, expected)
    assert proposal.shape == (1, 1, 1, 2)
    assert not proposal.requires_grad


def test_query_guided_residual_preserves_s_floor_and_full_s_path():
    p_feature = torch.full((1, 2, 1, 2), 2.0, requires_grad=True)
    s_feature = torch.full((1, 2, 1, 2), 3.0, requires_grad=True)
    proposal = torch.tensor([[[[0.0, 1.0]]]])

    fused = inject_query_guided_residual(
        p_feature,
        s_feature,
        proposal,
        s_floor=0.2,
    )

    expected = torch.tensor([[[[2.6, 5.0]], [[2.6, 5.0]]]])
    torch.testing.assert_close(fused, expected)
    fused.sum().backward()
    torch.testing.assert_close(p_feature.grad, torch.ones_like(p_feature))
    expected_s_grad = torch.tensor(
        [[[[0.2, 1.0]], [[0.2, 1.0]]]]
    )
    torch.testing.assert_close(s_feature.grad, expected_s_grad)


def test_query_guided_residual_resizes_proposal_and_validates_floor():
    p_feature = torch.zeros(2, 3, 4, 4)
    s_feature = torch.ones_like(p_feature)
    proposal = torch.ones(2, 1, 2, 2)

    fused = inject_query_guided_residual(
        p_feature,
        s_feature,
        proposal,
        s_floor=0.2,
    )

    torch.testing.assert_close(fused, torch.ones_like(fused))
    try:
        inject_query_guided_residual(
            p_feature,
            s_feature,
            proposal,
            s_floor=1.1,
        )
    except ValueError as error:
        assert "s_floor" in str(error)
    else:
        raise AssertionError("An invalid S floor must raise ValueError")


class _PassAttention(nn.Module):
    def forward(self, target, *_args, **_kwargs):
        return target, None


class _ZeroPosition(nn.Module):
    def forward(self, feature, _mask):
        return torch.zeros_like(feature)


class _FakePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_feature_levels = 3
        self.num_queries = 2
        self.num_heads = 1
        self.num_layers = 1
        self.pre_self_attention = False
        self.query_embed_ = nn.Embedding(2, 2)
        self.query_feat_ = nn.Embedding(2, 2)
        with torch.no_grad():
            self.query_embed_.weight.zero_()
            self.query_feat_.weight.copy_(torch.eye(2))
        self.pe_layer = _ZeroPosition()
        self.input_proj = nn.ModuleList([nn.Identity() for _ in range(3)])
        self.level_embed = nn.Embedding(3, 2)
        with torch.no_grad():
            self.level_embed.weight.zero_()
        self.transformer_cross_attention_layers = nn.ModuleList(
            [_PassAttention()]
        )
        self.transformer_self_attention_layers = nn.ModuleList(
            [_PassAttention()]
        )
        self.transformer_ffn_layers = nn.ModuleList([nn.Identity()])

    def forward_prediction_heads(
        self,
        output,
        mask_features,
        attn_mask_target_size,
        layer_id=-1,
    ):
        del layer_id
        mask_embed = output.transpose(0, 1)
        masks = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
        target_tokens = int(attn_mask_target_size[0] * attn_mask_target_size[1])
        attention_mask = torch.zeros(
            mask_features.shape[0] * self.num_heads,
            self.num_queries,
            target_tokens,
            dtype=torch.bool,
            device=mask_features.device,
        )
        return masks, attention_mask


class _FakeHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.predictor = _FakePredictor()


def test_full_query_guided_wrapper_uses_p_initial_mask_and_gated_s_final_mask():
    head = _FakeHead()
    mask_p = torch.tensor(
        [[[[2.0, -2.0]], [[-1.0, 1.0]]]]
    )
    mask_s = torch.ones_like(mask_p)
    multi_scale_p = [torch.zeros(1, 2, 1, 2) for _ in range(3)]
    multi_scale_s = [torch.ones(1, 2, 1, 2) for _ in range(3)]
    prompt_features = {"grounding_tokens": torch.zeros(1, 1, 2)}

    result = run_query_guided_s_predictor(
        head,
        multi_scale_p,
        multi_scale_s,
        mask_p,
        mask_s,
        prompt_features,
        proposal_topk=1,
        s_floor=0.2,
    )

    torch.testing.assert_close(result["initial_p_gmasks"], mask_p)
    proposal = torch.maximum(mask_p[:, :1].sigmoid(), mask_p[:, 1:].sigmoid())
    torch.testing.assert_close(result["proposal_map"], proposal)
    gate = 0.2 + 0.8 * proposal
    expected_fused_mask = mask_p + gate * mask_s
    torch.testing.assert_close(result["pred_gmasks"], expected_fused_mask)


def test_wrapper_matches_official_mask_decoder_when_s_is_zero():
    ensure_biomedparse_on_path()
    from src.model.transformer_decoder.boltzformer_cls_decoder import (
        BoltzFormerTextDecoder,
    )

    torch.manual_seed(7)
    decoder = BoltzFormerTextDecoder(
        language_encoder=None,
        in_channels=8,
        hidden_dim=8,
        dim_proj=8,
        num_queries=4,
        nheads=2,
        dim_feedforward=16,
        dec_layers=3,
        mask_dim=8,
        pre_self_attention=True,
        output_classifier=True,
        boltzmann_sampling={
            "mask_threshold": 0.5,
            "do_boltzmann": False,
            "sample_ratio": 0.1,
            "base_temp": 1.0,
        },
    ).eval()
    head = nn.Module()
    head.predictor = decoder
    multi_scale_p = [
        torch.randn(2, 8, 4, 4),
        torch.randn(2, 8, 8, 8),
        torch.randn(2, 8, 16, 16),
    ]
    multi_scale_s = [torch.zeros_like(feature) for feature in multi_scale_p]
    mask_p = torch.randn(2, 8, 16, 16)
    mask_s = torch.zeros_like(mask_p)
    prompt_features = {"grounding_tokens": torch.randn(1, 2, 8)}

    with torch.no_grad():
        official = decoder(
            multi_scale_p,
            mask_p,
            mask=None,
            extra=prompt_features,
        )
        wrapped = run_query_guided_s_predictor(
            head,
            multi_scale_p,
            multi_scale_s,
            mask_p,
            mask_s,
            prompt_features,
            proposal_topk=2,
            s_floor=0.2,
            run_object_existence=True,
        )

    torch.testing.assert_close(
        wrapped["pred_gmasks"],
        official["pred_gmasks"],
    )
