import torch

from oodka.models.beta_router import PromptBetaRouter
from oodka.models.biomedparse_helpers import gates_for_biomedparse_predictor


def test_beta_router_shapes_ranges_and_gradients():
    router = PromptBetaRouter(text_dim=32, hidden_dim=16)
    router.train()
    text = torch.randn(7, 32)
    output = router(text, batch_size=3)

    assert output["alpha"].shape == (7, 4)
    assert output["beta"].shape == (7, 4)
    assert output["gate"].shape == (3, 7, 4)
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


def test_beta_router_starts_at_scale_specific_prior():
    router = PromptBetaRouter(text_dim=8, hidden_dim=4).eval()
    output = router(torch.randn(3, 8), batch_size=1)
    expected = torch.tensor([0.5, 0.6, 0.7, 0.8]).expand(3, -1)
    torch.testing.assert_close(output["mean"], expected)
    torch.testing.assert_close(output["alpha"][0], torch.tensor([5.0, 6.0, 7.0, 8.0]))
    torch.testing.assert_close(output["beta"][0], torch.tensor([5.0, 4.0, 3.0, 2.0]))


def test_scale_gates_map_to_biomedparse_predictor_order():
    gate = torch.tensor([[[0.5, 0.6, 0.7, 0.8]]])
    mask_gate, multi_scale_gates = gates_for_biomedparse_predictor(
        gate, B=1, P=1
    )
    torch.testing.assert_close(mask_gate, torch.tensor([[0.5]]))
    torch.testing.assert_close(
        torch.stack([value.squeeze() for value in multi_scale_gates]),
        torch.tensor([0.8, 0.7, 0.6]),
    )
