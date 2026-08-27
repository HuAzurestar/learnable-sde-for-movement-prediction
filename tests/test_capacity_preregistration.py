import json
from pathlib import Path

import pytest

from experiments.capacity_preregistration.count_parameters import (
    count_i1_parameters,
    count_module_trainable_parameters,
    count_neural_parameters,
    validate_matrix,
)
from models.segment_constant import SegmentConstantSDE


@pytest.mark.parametrize("n_modes, expected", [(1, 5), (3, 15), (10, 50), (30, 150)])
def test_i1_declared_and_runtime_counts_match(n_modes, expected):
    declared = count_i1_parameters(n_modes)
    runtime = count_module_trainable_parameters(SegmentConstantSDE(n_modes=n_modes))
    assert declared["stored_trainable_parameters"] == expected
    assert runtime == expected


@pytest.mark.parametrize(
    "hidden, expected",
    [((8, 8), 124), ((32, 32), 1252), ((64, 64), 4548), ((128, 128), 17284)],
)
def test_neural_contract_counts(hidden, expected):
    assert count_neural_parameters(hidden)["stored_trainable_parameters"] == expected


def test_neural_contract_rejects_unregistered_depth():
    with pytest.raises(ValueError, match="exactly two"):
        count_neural_parameters((8,))


def test_registered_matrix_is_internally_consistent():
    path = (
        Path(__file__).parents[1]
        / "experiments"
        / "capacity_preregistration"
        / "experiment_matrix.json"
    )
    document = json.loads(path.read_text(encoding="utf-8"))
    assert validate_matrix(document) == []
