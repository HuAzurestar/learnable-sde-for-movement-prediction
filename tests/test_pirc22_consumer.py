from __future__ import annotations

import json

import pytest

from experiments.pirc22.consumer import (
    BenchmarkConsumerError,
    COMPOSITION_DEPENDENCIES,
    build_pirc17_ablation_configs,
    load_benchmark_selection_binding,
)


def test_frozen_selection_binding_validates_and_matches_published_choice():
    binding = load_benchmark_selection_binding()

    assert binding["source_selection_identity_sha256"] == (
        "edef6e69e4ab16d36e4fd04f2313e6feb8ef97379b705eb45b18d4ffcd0f61ea"
    )
    assert binding["selected"]["candidate_key"] == (
        "R12-registered-compositions::linear"
    )
    assert binding["raw_best"]["candidate_key"] == (
        "R12-registered-compositions::mlp-small-32"
    )
    assert binding["final_eval_read_count"] == 0


def test_binding_mutation_is_rejected(tmp_path):
    binding = load_benchmark_selection_binding()
    binding["selected_configuration"]["conditioner_id"] = "mlp-small-16"
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(binding), encoding="utf-8")

    with pytest.raises(BenchmarkConsumerError, match="identity mismatch"):
        load_benchmark_selection_binding(path)


def test_pirc17_configs_cover_base_all_loo_and_required_lio_without_retuning():
    result = build_pirc17_ablation_configs()
    configurations = result["configurations"]

    assert result["configuration_count"] == 12
    assert not result["retuning_allowed"]
    assert [item["kind"] for item in configurations].count("base") == 1
    assert [item["kind"] for item in configurations].count("all_terrain") == 1
    assert [item["kind"] for item in configurations].count("leave_one_out") == 5
    assert [item["kind"] for item in configurations].count("leave_one_in") == 5
    assert {item["factor_id"] for item in configurations[2:7]} == {
        "road",
        "river",
        "worldcover",
        "surface",
        "history",
    }
    for item in configurations:
        assert item["conditioner_id"] == "linear"
        assert item["hidden_widths"] == []
        assert item["training_config"]["maximum_epochs"] == 40
        selected = set(item["variant_ids"])
        assert all(
            set(COMPOSITION_DEPENDENCIES[composition_id]) <= selected
            for composition_id in item["composition_ids"]
        )

