import torch

from oodka.models.ot import (
    BalancedSinkhorn,
    BarycentricProjector,
    OTCostBuilder,
    MultiScaleOTDistillation,
    ResidualMassBuilder,
    StructureMassBuilder,
    UnbalancedSinkhorn,
    WeightedCosineDistillation,
)


def test_balanced_sinkhorn_nonnegative_and_marginals():
    torch.manual_seed(0)
    a = torch.rand(2, 7)
    b = torch.rand(2, 5)
    a = a / a.sum(-1, keepdim=True)
    b = b / b.sum(-1, keepdim=True)
    cost = torch.rand(2, 7, 5)
    output = BalancedSinkhorn(epsilon=0.2, iterations=100)(a, b, cost)

    transport = output["transport"]
    assert transport.shape == (2, 7, 5)
    assert torch.isfinite(transport).all()
    assert (transport >= 0).all()
    torch.testing.assert_close(transport.sum(-1), a, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(transport.sum(-2), b, atol=2e-4, rtol=2e-4)


def test_uot_rejects_high_cost_expert_token():
    a = torch.tensor([[0.5, 0.5]])
    b = torch.tensor([[0.5, 0.5]])
    cost = torch.tensor([[[0.0, 20.0], [0.0, 20.0]]])
    output = UnbalancedSinkhorn(
        epsilon=0.1, rho_base=1.0, rho_expert=0.2, iterations=100
    )(a, b, cost)

    transported = output["transported"][0]
    assert torch.isfinite(output["transport"]).all()
    assert (output["transport"] >= 0).all()
    assert transported[1] < transported[0] * 1e-3
    assert output["rejected"][0, 1] > output["rejected"][0, 0]


def test_uot_acceptance_decreases_under_controlled_global_cost_shift():
    torch.manual_seed(7)
    a = torch.rand(2, 12)
    b = torch.rand(2, 12)
    a = a / a.sum(-1, keepdim=True)
    b = b / b.sum(-1, keepdim=True)
    cost = torch.rand(2, 12, 12)
    solver = UnbalancedSinkhorn(
        epsilon=0.1, rho_base=1.0, rho_expert=0.2, iterations=100
    )

    acceptance = [
        solver(a, b, cost + offset)["accept_ratio"].mean()
        for offset in (0.0, 0.25, 0.5, 1.0, 2.0)
    ]
    assert all(
        right < left for left, right in zip(acceptance, acceptance[1:])
    )


def test_cost_barycentric_and_weighted_distillation_shapes():
    torch.manual_seed(1)
    base = torch.randn(2, 8, 10, 12, requires_grad=True)
    expert = torch.randn(2, 8, 10, 12)
    cost_output = OTCostBuilder(coordinate_weight=0.1)(
        base, expert, target_size=(4, 4)
    )
    assert cost_output["cost"].shape == (2, 16, 16)
    assert cost_output["base_tokens"].shape == (2, 16, 8)
    assert torch.isfinite(cost_output["cost"]).all()

    mass = torch.full((2, 16), 1 / 16)
    sinkhorn = BalancedSinkhorn(epsilon=0.2, iterations=80)(
        mass, mass, cost_output["cost"]
    )
    projected = BarycentricProjector()(
        sinkhorn["transport"], cost_output["expert_tokens"]
    )
    assert projected["teacher"].shape == (2, 16, 8)
    loss = WeightedCosineDistillation()(
        cost_output["base_tokens"], projected["teacher"], mass
    )
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert base.grad is not None


def test_mass_builders_handle_empty_structure_and_zero_residual():
    gt = torch.zeros(2, 32, 32, dtype=torch.long)
    feature = torch.zeros(2, 8, 8, 8, dtype=torch.float16)
    structure = StructureMassBuilder()(
        gt,
        feature,
        feature,
        class_ids=[1, 2, 3],
        target_size=(4, 4),
    )
    for mass in [structure["a"], structure["b"]]:
        assert torch.isfinite(mass).all()
        torch.testing.assert_close(mass.sum(-1), torch.ones(2))

    error = torch.zeros(2, 32, 32)
    residual = ResidualMassBuilder()(
        feature,
        feature,
        feature,
        feature,
        base_error=error,
        expert_error=error,
        target_size=(4, 4),
    )
    for mass in [residual["a"], residual["b"]]:
        assert torch.isfinite(mass).all()
        torch.testing.assert_close(mass.sum(-1), torch.ones(2))


def test_multiscale_objective_filters_invalid_z_and_backpropagates_student_only():
    torch.manual_seed(2)
    features = {}
    for level in [2, 3, 4, 5]:
        base_p = torch.randn(1, 6, 2, 4, 4, requires_grad=True)
        base_s = torch.randn(1, 6, 2, 4, 4, requires_grad=True)
        expert_p = torch.randn(1, 6, 2, 4, 4, requires_grad=True)
        expert_s = torch.randn(1, 6, 2, 4, 4, requires_grad=True)
        features[f"Zb{level}_p"] = base_p
        features[f"Zb{level}_s"] = base_s
        features[f"Zn{level}_p"] = expert_p
        features[f"Zn{level}_s"] = expert_s
    gt = torch.zeros(1, 2, 16, 16, dtype=torch.long)
    gt[:, 0, 4:12, 4:12] = 1
    error_base = torch.rand(1, 2, 16, 16)
    error_expert = error_base * 0.5
    objective = MultiScaleOTDistillation(
        grids={2: (4, 4), 3: (4, 4), 4: (4, 4), 5: (4, 4)},
        sinkhorn_iterations=20,
    )
    output = objective(
        features,
        gt=gt,
        base_error=error_base,
        expert_error=error_expert,
        valid_z=torch.tensor([[True, False]]),
        class_ids=[1],
    )
    loss = output["loss_p"] + output["loss_s"]
    assert torch.isfinite(loss)
    loss.backward()
    assert features["Zb2_p"].grad is not None
    assert features["Zb2_s"].grad is not None
    assert features["Zn2_p"].grad is None
    assert features["Zn2_s"].grad is None
