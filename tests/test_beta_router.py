import torch

from oodka.models.beta_router import PromptBetaRouter


def test_beta_router_shapes_ranges_and_gradients():
    router = PromptBetaRouter(text_dim=32, hidden_dim=16)
    router.train()
    text = torch.randn(7, 32)
    output = router(text, batch_size=3)

    assert output["alpha"].shape == (7, 1)
    assert output["beta"].shape == (7, 1)
    assert output["gate"].shape == (3, 7, 1)
    assert torch.all(output["alpha"] > 1)
    assert torch.all(output["beta"] > 1)
    assert torch.all((output["gate"] >= 0) & (output["gate"] <= 1))
    assert torch.isfinite(output["kl"])

    (output["gate"].mean() + output["kl"]).backward()
    assert router.alpha_head.weight.grad is not None
    assert router.beta_head.weight.grad is not None


def test_beta_router_eval_uses_distribution_mean():
    router = PromptBetaRouter(text_dim=8, hidden_dim=4).eval()
    output = router(torch.randn(2, 8), batch_size=3)
    expected = output["mean"].unsqueeze(0).expand(3, -1, -1)
    torch.testing.assert_close(output["gate"], expected)
