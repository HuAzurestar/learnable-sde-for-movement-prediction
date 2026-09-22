from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from experiments.pirc22.conditioners import (
    CONDITIONER_SPECS,
    ConditionerError,
    TrainingConfig,
    build_conditioner,
    fit_conditioner,
    load_conditioner,
    predict_conditioner,
)


def test_registered_capacity_families_and_exact_parameter_counts():
    assert [spec.conditioner_id for spec in CONDITIONER_SPECS] == [
        "linear",
        "mlp-small-16",
        "mlp-small-32",
        "mlp-medium-32-16",
        "mlp-medium-64-32",
    ]
    assert [spec.hidden_widths for spec in CONDITIONER_SPECS] == [
        (),
        (16,),
        (32,),
        (32, 16),
        (64, 32),
    ]
    counts = {
        spec.conditioner_id: build_conditioner(
            spec.conditioner_id, 7, seed=11
        ).trainable_parameter_count
        for spec in CONDITIONER_SPECS
    }
    assert counts == {
        "linear": 16,
        "mlp-small-16": 162,
        "mlp-small-32": 322,
        "mlp-medium-32-16": 818,
        "mlp-medium-64-32": 2658,
    }


def test_training_trace_checkpoint_and_predictions_replay_for_fixed_seed():
    rng = np.random.default_rng(71)
    train_x = rng.normal(size=(48, 3))
    validation_x = rng.normal(size=(16, 3))
    weights = np.asarray([[0.6, -0.2], [0.1, 0.7], [-0.5, 0.3]])
    train_y = train_x @ weights + np.asarray([0.2, -0.1])
    validation_y = validation_x @ weights + np.asarray([0.2, -0.1])
    config = TrainingConfig(
        learning_rate=0.02,
        weight_decay=0.0,
        batch_size=12,
        maximum_epochs=80,
        patience=12,
    )

    first = fit_conditioner(
        "mlp-small-16",
        train_x,
        train_y,
        validation_x,
        validation_y,
        seed=20260814,
        config=config,
    )
    second = fit_conditioner(
        "mlp-small-16",
        train_x,
        train_y,
        validation_x,
        validation_y,
        seed=20260814,
        config=config,
    )

    assert first.learning_curve == second.learning_curve
    assert first.checkpoint == second.checkpoint
    assert first.best_epoch <= len(first.learning_curve) <= config.maximum_epochs
    assert first.checkpoint["training_scope"] == {
        "gradient_roles": ["train"],
        "checkpoint_selection_roles": ["validation"],
    }
    replay = load_conditioner(first.checkpoint)
    assert np.array_equal(
        predict_conditioner(first.model, validation_x),
        predict_conditioner(replay, validation_x),
    )
    assert first.learning_curve[-1].train_loss < first.learning_curve[0].train_loss


def test_train_only_embedding_is_counted_trained_and_replayed():
    rng = np.random.default_rng(9)
    train_x = rng.normal(size=(36, 1))
    validation_x = rng.normal(size=(12, 1))
    train_tokens = np.arange(36, dtype=np.int64) % 12
    validation_tokens = np.arange(12, dtype=np.int64) % 12
    train_y = np.column_stack(
        (train_x[:, 0] + train_tokens / 12.0, -train_x[:, 0])
    )
    validation_y = np.column_stack(
        (validation_x[:, 0] + validation_tokens / 12.0, -validation_x[:, 0])
    )

    result = fit_conditioner(
        "linear",
        train_x,
        train_y,
        validation_x,
        validation_y,
        train_tokens=train_tokens,
        validation_tokens=validation_tokens,
        categorical_cardinality=12,
        embedding_dim=4,
        seed=22,
        config=TrainingConfig(maximum_epochs=25, patience=6, batch_size=12),
    )

    assert result.model.trainable_parameter_count == 60
    assert result.checkpoint["embedding_dim"] == 4
    replay = load_conditioner(result.checkpoint)
    assert np.array_equal(
        predict_conditioner(
            result.model, validation_x, categorical_tokens=validation_tokens
        ),
        predict_conditioner(
            replay, validation_x, categorical_tokens=validation_tokens
        ),
    )


def test_zero_width_control_and_checkpoint_tampering_are_fail_closed():
    train_x = np.empty((20, 0), dtype=float)
    validation_x = np.empty((8, 0), dtype=float)
    train_y = np.repeat([[0.5, -0.25]], len(train_x), axis=0)
    validation_y = np.repeat([[0.5, -0.25]], len(validation_x), axis=0)
    result = fit_conditioner(
        "linear",
        train_x,
        train_y,
        validation_x,
        validation_y,
        seed=5,
        config=TrainingConfig(maximum_epochs=10, patience=3, batch_size=10),
    )

    assert result.model.trainable_parameter_count == 2
    assert predict_conditioner(result.model, validation_x).shape == (8, 2)
    changed = deepcopy(result.checkpoint)
    changed["best_epoch"] = 999
    with pytest.raises(ConditionerError, match="identity mismatch"):
        load_conditioner(changed)


def test_conditioner_rejects_nonfinite_data_and_bad_tokens():
    x = np.ones((4, 1))
    y = np.ones((4, 2))
    with pytest.raises(ConditionerError, match="finite"):
        fit_conditioner(
            "linear", x * np.nan, y, x, y, seed=1
        )
    with pytest.raises(ConditionerError, match="outside the vocabulary"):
        fit_conditioner(
            "linear",
            x,
            y,
            x,
            y,
            seed=1,
            train_tokens=np.asarray([0, 1, 2, 9]),
            validation_tokens=np.asarray([0, 1, 2, 3]),
            categorical_cardinality=4,
            embedding_dim=2,
        )
