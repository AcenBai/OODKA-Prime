import argparse

from oodka.config import TrainConfig
from oodka.train.cli import (
    add_augmentation_switch,
    add_fusion_training_arguments,
    fusion_builder_kwargs,
    fusion_train_config_kwargs,
)


def test_new_alignment_defaults_are_relative_capacity():
    cfg = TrainConfig()
    assert cfg.relative_kd is True
    assert cfg.s_transport_mode == "capacity_partial"


def test_fusion_cli_defaults_and_soft_disable():
    parser = argparse.ArgumentParser()
    add_fusion_training_arguments(parser)

    defaults = parser.parse_args([])
    assert defaults.relative_kd is True
    assert defaults.s_transport_mode == "capacity_partial"

    legacy = parser.parse_args(
        ["--no_relative_kd", "--s_transport_mode", "unbalanced"]
    )
    values = fusion_train_config_kwargs(legacy)
    assert values["relative_kd"] is False
    assert values["s_transport_mode"] == "unbalanced"


def test_fusion_builder_receives_every_alignment_switch():
    cfg = TrainConfig(
        relative_kd=False,
        relative_kd_branches="p",
        relative_kd_rms_weight=0.5,
        s_transport_mode="unbalanced",
        expert_ortho_weight=0.0,
    )
    values = fusion_builder_kwargs(cfg)
    assert values["relative_kd"] is False
    assert values["relative_kd_branches"] == "p"
    assert values["relative_kd_rms_weight"] == 0.5
    assert values["s_transport_mode"] == "unbalanced"


def test_augmentation_switch_is_symmetric_and_can_defer_default():
    parser = argparse.ArgumentParser()
    add_augmentation_switch(parser, default=None)
    assert parser.parse_args([]).augment is None
    assert parser.parse_args(["--augment"]).augment is True
    assert parser.parse_args(["--no-augment"]).augment is False
