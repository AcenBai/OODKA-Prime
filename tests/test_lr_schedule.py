import pytest

from oodka.train.engine import learning_rate_scale


def test_linear_warmup_then_cosine_decay():
    values = [
        learning_rate_scale(
            epoch,
            n_epochs=10,
            schedule="cosine",
            warmup_epochs=2,
            min_lr_ratio=0.1,
        )
        for epoch in range(1, 11)
    ]

    assert values[:2] == [0.5, 1.0]
    assert values[2] == pytest.approx(1.0)
    assert values[-1] == pytest.approx(0.1)
    assert all(left >= right for left, right in zip(values[2:], values[3:]))


def test_constant_schedule_only_applies_warmup():
    values = [
        learning_rate_scale(
            epoch,
            n_epochs=4,
            schedule="constant",
            warmup_epochs=2,
            min_lr_ratio=0.05,
        )
        for epoch in range(1, 5)
    ]
    assert values == [0.5, 1.0, 1.0, 1.0]
