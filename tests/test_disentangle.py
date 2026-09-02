import torch
import torch.nn as nn

from oodka.models.disentangle import DirectSharedDecoderExpertAdapter
from oodka.train import model_builder


def test_direct_shared_expert_adapter_is_exactly_three_bias_free_convs():
    module = DirectSharedDecoderExpertAdapter(c_in=4, c_out=6)
    children = list(module.modules())[1:]
    assert len(children) == 3
    assert all(isinstance(child, nn.Conv3d) for child in children)
    assert all(child.kernel_size == (1, 1, 1) for child in children)
    assert all(child.bias is None for child in children)


def test_direct_shared_expert_adapter_decodes_sum_and_backpropagates_both_branches():
    torch.manual_seed(7)
    module = DirectSharedDecoderExpertAdapter(c_in=4, c_out=6)
    feature = torch.randn(2, 4, 3, 5, 5)
    p_feature, s_feature, reconstruction = module(feature)

    assert p_feature.shape == s_feature.shape == (2, 6, 3, 5, 5)
    assert reconstruction.shape == feature.shape
    torch.testing.assert_close(
        reconstruction,
        module.decoder(p_feature + s_feature),
    )

    reconstruction.square().mean().backward()
    assert module.proj_p.weight.grad is not None
    assert module.proj_s.weight.grad is not None
    assert module.decoder.weight.grad is not None


def test_builder_selects_direct_shared_expert_adapter(monkeypatch):
    monkeypatch.setattr(
        model_builder,
        "_detect_biomedparse_res_channels",
        lambda _model, _device: {f"res{i}": 6 for i in range(2, 6)},
    )
    monkeypatch.setattr(
        model_builder,
        "_detect_nnunet_enc_channels",
        lambda _model, _device: {f"enc{i}": 4 for i in range(2, 6)},
    )

    modules = model_builder.build_fusion_modules(
        nn.Identity(),
        nn.Identity(),
        P=2,
        device=torch.device("cpu"),
        expert_adapter_variant="direct_shared",
    )

    for level in range(2, 6):
        assert isinstance(
            modules[f"ae_enc{level}_to_res{level}"],
            DirectSharedDecoderExpertAdapter,
        )
