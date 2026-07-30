import torch

from oodka.models.beta_router import PromptBetaRouter
from oodka.models.biomedparse_helpers import gates_for_biomedparse_predictor


def test_spatial_beta_router_shapes_ranges_and_gradients():
    router = PromptBetaRouter(
        text_dim=32,
        hidden_dim=16,
        basis_grid_size=4,
    )
    router.train()
    text = torch.randn(7, 32)
    output = router(text, spatial_size=(12, 10), batch_size=3)

    assert output["alpha"].shape == (7, 12, 10)
    assert output["beta"].shape == (7, 12, 10)
    assert output["gate"].shape == (3, 7, 12, 10)
    assert torch.all(output["alpha"] > 1)
    assert torch.all(output["beta"] > 1)
    assert torch.all((output["gate"] >= 0) & (output["gate"] <= 1))
    assert torch.isfinite(output["kl"])

    (output["gate"].mean() + output["kl"]).backward()
    assert router.alpha_coeff.weight.grad is not None
    assert router.beta_coeff.weight.grad is not None
    assert torch.isfinite(router.alpha_coeff.weight.grad).all()
    assert torch.isfinite(router.beta_coeff.weight.grad).all()


def test_spatial_beta_router_eval_uses_distribution_mean():
    router = PromptBetaRouter(
        text_dim=8,
        hidden_dim=4,
        basis_grid_size=3,
    ).eval()
    output = router(
        torch.randn(2, 8),
        spatial_size=(7, 5),
        batch_size=3,
    )
    expected = output["mean"].unsqueeze(0).expand(3, -1, -1, -1)
    torch.testing.assert_close(output["gate"], expected)


def test_spatial_beta_router_starts_at_point_seven_prior_at_any_size():
    router = PromptBetaRouter(
        text_dim=8,
        hidden_dim=4,
        prior_p_mean=0.7,
        prior_concentration=10.0,
        basis_grid_size=3,
    ).eval()
    for spatial_size in ((8, 8), (16, 12)):
        output = router(
            torch.randn(3, 8),
            spatial_size=spatial_size,
            batch_size=1,
        )
        torch.testing.assert_close(
            output["mean"],
            torch.full((3, *spatial_size), 0.7),
        )
        torch.testing.assert_close(
            output["alpha"],
            torch.full((3, *spatial_size), 7.0),
        )
        torch.testing.assert_close(
            output["beta"],
            torch.full((3, *spatial_size), 3.0),
        )


def test_one_finest_gate_is_resized_to_all_predictor_levels():
    gate = torch.linspace(0.0, 1.0, 8 * 8).reshape(1, 1, 8, 8)
    mask_gate, multi_scale_gates = gates_for_biomedparse_predictor(
        gate,
        B=1,
        P=1,
        mask_size=(8, 8),
        multi_scale_sizes=((2, 2), (4, 4), (6, 6)),
    )

    torch.testing.assert_close(mask_gate, gate)
    assert [tuple(value.shape) for value in multi_scale_gates] == [
        (1, 1, 2, 2),
        (1, 1, 4, 4),
        (1, 1, 6, 6),
    ]
    for resized in multi_scale_gates:
        torch.testing.assert_close(resized.mean(), gate.mean())
