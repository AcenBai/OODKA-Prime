import torch
import torch.nn as nn

from oodka.models.disentangle import DirectSharedDecoderExpertAdapter
from oodka.train import model_builder
from oodka.train.forward import _compute_reconstruction_separation_losses


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


def test_expert_orthogonality_can_be_disabled_without_disabling_student_term():
    torch.manual_seed(11)
    features = {}
    for level in range(2, 6):
        shape = (1, 4, 2, 3, 3)
        features[f"Z_n{level}"] = torch.randn(shape)
        features[f"Zn{level}_rec"] = torch.randn(shape)
        features[f"Zb_res{level}"] = torch.randn(shape)
        for prefix in ("Zn", "Zb"):
            features[f"{prefix}{level}_p"] = torch.randn(shape)
            features[f"{prefix}{level}_s"] = torch.randn(shape)
    valid_z = torch.ones(1, 2, dtype=torch.bool)

    _ae, combined, student, expert = _compute_reconstruction_separation_losses(
        features,
        valid_z,
        expert_ortho_weight=0.0,
    )
    torch.testing.assert_close(combined, student)
    assert student.item() > 0.0
    assert expert.item() > 0.0
