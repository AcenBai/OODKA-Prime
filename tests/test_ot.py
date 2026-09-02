import torch
import torch.nn as nn

from oodka.models.disentangle import DualBranchAutoEncoder
from oodka.models.ot import (
    BalancedSinkhorn,
    BarycentricProjector,
    OTCostBuilder,
    MultiScaleOTDistillation,
    ResidualMassBuilder,
    StructureMassBuilder,
    UnbalancedSinkhorn,
    WeightedCosineDistillation,
    WeightedLogRMSAlignment,
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


def test_log_rms_alignment_detects_scale_when_cosine_does_not():
    student = torch.tensor([[[1.0, 2.0, 3.0]]], requires_grad=True)
    teacher = student.detach() * 4.0
    weight = torch.ones(1, 1)

    cosine = WeightedCosineDistillation()(student, teacher, weight)
    rms = WeightedLogRMSAlignment()(student, teacher, weight)

    torch.testing.assert_close(cosine, torch.tensor(0.0), atol=1e-6, rtol=0.0)
    assert rms.item() > 0.0
    rms.backward()
    assert student.grad is not None
    assert student.grad.abs().sum().item() > 0.0


def test_coordinate_cost_has_a_zero_cost_radius():
    feature = torch.zeros(1, 2, 3, 3)
    output = OTCostBuilder(
        feature_weight=0.0,
        coordinate_weight=1.0,
        coordinate_radius=0.5,
    )(feature, feature, target_size=(3, 3))
    cost = output["cost"][0]

    assert cost[0, 0] == 0.0
    torch.testing.assert_close(cost[0, 1], torch.tensor(0.25))
    assert cost[0, 4] > cost[0, 1]


def test_smooth_expert_advantage_is_dense_bounded_and_signed():
    feature = torch.ones(1, 4, 2, 2)
    base_error = torch.tensor([[[0.8, 0.8], [0.2, 0.5]]])
    expert_error = torch.tensor([[[0.2, 0.8], [0.8, 0.5]]])
    output = ResidualMassBuilder(
        gain_mode="smooth_advantage",
        gain_temperature=0.5,
    )(
        feature,
        feature,
        feature,
        feature,
        base_error=base_error,
        expert_error=expert_error,
        target_size=(2, 2),
    )
    score = output["gain"].reshape(2, 2)

    assert torch.all((score > 0.0) & (score < 2.0))
    assert score[0, 0] > 1.0
    assert score[1, 0] < 1.0
    torch.testing.assert_close(score[0, 1], torch.tensor(1.0))
    torch.testing.assert_close(score[1, 1], torch.tensor(1.0))


def test_expert_branch_output_norm_can_be_removed_selectively():
    module = DualBranchAutoEncoder(
        c_in=4,
        c_mid=6,
        c_out=8,
        branch_output_norm=False,
    )
    assert isinstance(module.enc_p[1], nn.Identity)
    assert isinstance(module.enc_s[1], nn.Identity)
    assert any(isinstance(layer, nn.InstanceNorm3d) for layer in module.enc)
    assert any(isinstance(layer, nn.InstanceNorm3d) for layer in module.dec_p)
    outputs = module(torch.randn(1, 4, 2, 4, 4))
    assert [tuple(value.shape) for value in outputs] == [
        (1, 8, 2, 4, 4),
        (1, 8, 2, 4, 4),
        (1, 4, 2, 4, 4),
        (1, 4, 2, 4, 4),
    ]


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
        max_grid_size=4,
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
    for level_log in output["levels"].values():
        for key in [
            "p_mean_distance",
            "p_outside_radius",
            "s_mean_distance",
            "s_outside_radius",
            "s_expert_better_ratio",
        ]:
            assert torch.isfinite(level_log[key])
    loss.backward()
    assert features["Zb2_p"].grad is not None
    assert features["Zb2_s"].grad is not None
    assert features["Zn2_p"].grad is None
    assert features["Zn2_s"].grad is None


def test_relative_kd_reuses_detached_transport_and_backpropagates_both_sides():
    torch.manual_seed(23)
    features = {}
    for level in [2, 3, 4, 5]:
        for domain in ("Zb", "Zn"):
            for branch in ("p", "s"):
                features[f"{domain}{level}_{branch}"] = torch.randn(
                    1, 6, 1, 4, 4, requires_grad=True
                )
    gt = torch.zeros(1, 1, 16, 16, dtype=torch.long)
    gt[:, :, 4:12, 4:12] = 1
    base_error = torch.rand(1, 1, 16, 16)
    expert_error = base_error * 0.5
    objective = MultiScaleOTDistillation(
        max_grid_size=4,
        sinkhorn_iterations=20,
        relative_kd=True,
        relative_kd_expert_weight=1.0,
    )
    output = objective(
        features,
        gt=gt,
        base_error=base_error,
        expert_error=expert_error,
        valid_z=torch.ones(1, 1, dtype=torch.bool),
        class_ids=[1],
    )
    loss = output["loss_p"] + output["loss_s"]
    assert torch.isfinite(loss)
    assert output["loss_p_reverse"].item() > 0.0
    assert output["loss_s_reverse"].item() > 0.0
    loss.backward()
    for key in ("Zb2_p", "Zb2_s", "Zn2_p", "Zn2_s"):
        assert features[key].grad is not None
        assert torch.isfinite(features[key].grad).all()
        assert features[key].grad.abs().sum().item() > 0.0


def test_relative_kd_can_be_limited_to_p_branch():
    objective = MultiScaleOTDistillation(
        relative_kd=True,
        relative_kd_branches="p",
    )
    assert objective._reverse_enabled("p")
    assert not objective._reverse_enabled("s")


def test_relative_kd_branch_selector_is_validated():
    try:
        MultiScaleOTDistillation(relative_kd_branches="invalid")
    except ValueError as error:
        assert "relative_kd_branches" in str(error)
    else:
        raise AssertionError("invalid reverse-KD branch selector was accepted")


def test_multiscale_grid_caps_native_size_without_upsampling():
    objective = MultiScaleOTDistillation(max_grid_size=32)
    assert objective._target_size(torch.empty(1, 1, 128, 96)) == (32, 32)
    assert objective._target_size(torch.empty(1, 1, 32, 24)) == (32, 24)
    assert objective._target_size(torch.empty(1, 1, 16, 16)) == (16, 16)
