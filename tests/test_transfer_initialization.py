import numpy as np
import pytest
import torch
from torch import nn

from oodka.data.slice_dataset import make_biomedparse_block
from oodka.train.engine import OODKATrainer


def test_center_repeat_pseudo_rgb_copies_current_slice():
    volume = np.stack(
        [np.full((4, 4), value, dtype=np.uint8) for value in (10, 20, 30)]
    )
    block = make_biomedparse_block(
        volume,
        [1],
        4,
        pseudo_rgb_mode="center_repeat",
    )
    assert block.shape == (1, 3, 4, 4)
    assert torch.all(block[0, 0] == 20)
    assert torch.equal(block[0, 0], block[0, 1])
    assert torch.equal(block[0, 1], block[0, 2])


def test_adjacent_pseudo_rgb_keeps_neighbor_order():
    volume = np.stack(
        [np.full((4, 4), value, dtype=np.uint8) for value in (10, 20, 30)]
    )
    block = make_biomedparse_block(
        volume,
        [1],
        4,
        pseudo_rgb_mode="adjacent",
    )
    assert [int(block[0, channel, 0, 0]) for channel in range(3)] == [10, 20, 30]


def _linear(value: float) -> nn.Linear:
    module = nn.Linear(2, 2, bias=False)
    nn.init.constant_(module.weight, value)
    return module


def _checkpoint_state(value: float):
    return {
        "dis_b_res2": _linear(value).state_dict(),
        "beta_router": _linear(value).state_dict(),
        "ae_enc2_to_res2": _linear(value).state_dict(),
    }


def _trainer_shell():
    trainer = OODKATrainer.__new__(OODKATrainer)
    trainer.device = torch.device("cpu")
    trainer.fusion_modules = {
        "dis_b_res2": _linear(0.0),
        "beta_router": _linear(0.0),
        "ae_enc2_to_res2": _linear(0.0),
        "ot_distillation": nn.Identity(),
    }
    trainer.initialized_modules = []
    trainer.start_epoch = 1
    trainer.best_val_dice = -float("inf")
    return trainer


def test_student_router_transfer_leaves_expert_adapter_fresh(tmp_path):
    checkpoint_path = tmp_path / "ct.pth"
    torch.save(_checkpoint_state(3.0), checkpoint_path)
    trainer = _trainer_shell()

    trainer._load_initialization_checkpoint(
        str(checkpoint_path),
        scope="student_router",
    )

    assert trainer.initialized_modules == ["dis_b_res2", "beta_router"]
    assert torch.all(trainer.fusion_modules["dis_b_res2"].weight == 3.0)
    assert torch.all(trainer.fusion_modules["beta_router"].weight == 3.0)
    assert torch.all(trainer.fusion_modules["ae_enc2_to_res2"].weight == 0.0)
    assert trainer.start_epoch == 1
    assert trainer.best_val_dice == -float("inf")


def test_all_fusion_transfer_also_loads_expert_adapter(tmp_path):
    checkpoint_path = tmp_path / "ct.pth"
    torch.save(_checkpoint_state(4.0), checkpoint_path)
    trainer = _trainer_shell()

    trainer._load_initialization_checkpoint(
        str(checkpoint_path),
        scope="all_fusion",
    )

    assert trainer.initialized_modules == [
        "dis_b_res2",
        "beta_router",
        "ae_enc2_to_res2",
    ]
    assert torch.all(trainer.fusion_modules["ae_enc2_to_res2"].weight == 4.0)


def test_transfer_scope_is_validated(tmp_path):
    checkpoint_path = tmp_path / "ct.pth"
    torch.save(_checkpoint_state(1.0), checkpoint_path)
    trainer = _trainer_shell()
    with pytest.raises(ValueError, match="init_scope"):
        trainer._load_initialization_checkpoint(
            str(checkpoint_path),
            scope="unknown",
        )
