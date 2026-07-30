import torch

from oodka.train.forward import _fuse_all_prompt_features


def test_spatial_prompt_gate_is_a_pixelwise_convex_combination():
    batch_size, slices, prompts = 2, 3, 2
    channels, height, width = 4, 5, 7
    p_feature = torch.full(
        (batch_size * slices, channels, height, width), 2.0
    )
    s_feature = torch.full_like(p_feature, 10.0)
    gate = torch.empty(batch_size, prompts, height, width)
    gate[:, 0] = 1.0
    gate[:, 1] = 0.25

    fused = _fuse_all_prompt_features(
        p_feature,
        s_feature,
        gate,
        B=batch_size,
        Z=slices,
    ).reshape(batch_size, slices, prompts, channels, height, width)

    assert torch.equal(fused[:, :, 0], p_feature.reshape(
        batch_size, slices, channels, height, width
    ))
    assert torch.allclose(fused[:, :, 1], torch.full_like(fused[:, :, 1], 8.0))


def test_one_spatial_gate_is_shared_across_slices():
    p_feature = torch.zeros(3, 1, 2, 2)
    s_feature = torch.ones_like(p_feature)
    gate = torch.tensor([[[[0.0, 0.25], [0.5, 1.0]]]])

    fused = _fuse_all_prompt_features(
        p_feature,
        s_feature,
        gate,
        B=1,
        Z=3,
    ).reshape(1, 3, 1, 1, 2, 2)

    expected = 1.0 - gate
    assert torch.allclose(fused[0, 0, 0, 0], expected[0, 0])
    assert torch.allclose(fused[0, 1, 0, 0], expected[0, 0])
    assert torch.allclose(fused[0, 2, 0, 0], expected[0, 0])
