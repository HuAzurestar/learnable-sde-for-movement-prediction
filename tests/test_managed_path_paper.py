"""Own settled sources/targets -> independent reader and no-fallback CLI."""

from copy import deepcopy
import json

import pytest

from application.research_evidence import export_evidence
from tests.test_managed_path_admission import source, completed, METRIC
from tests.path_paper_helpers import independent_reader, paper_validate, reseal_transport


@pytest.fixture(scope="module")
def exported(source, completed):
    store, spec, _, grant, _, _ = source
    return export_evidence(store, spec["study_id"], grant)


def test_real_path_owner_export_passes_independent_reader_and_cli(exported, tmp_path):
    row = exported["cells"][0]
    assert independent_reader().validate_path_qualification(row["admission"], row) == "PASSED"
    checked = paper_validate(tmp_path, exported)
    assert checked.returncode == 0, checked.stderr
    verified = json.loads(checked.stdout)
    assert verified["source_bundle_hash"] == exported["bundle_hash"]
    assert verified["verified_path_cells"] == verified["current_passed_cells"] == verified["expected_cells"] == 1
    assert verified["current_failed_cells"] == verified["noncompleted_rows"] == 0


@pytest.mark.parametrize("fault", ["width", "sampling", "source-statistics", "source-stream", "binomial", "unknown", "grid",
    "scientific", "source-cost", "native-stop", "completion-order", "source-grant", "target-scalar", "target-statistics",
    "target-classification", "target-metric", "target-unit", "target-reference-definition"])
def test_independent_reader_refuses_resealed_saved_meanings_and_current_classification(exported, fault):
    bundle = deepcopy(exported)
    row = bundle["cells"][0]
    evidence = row["admission"]["documents"]["propagation_qualification"]
    analysis = evidence["source_result"]["forecast"]["path_qualification_analysis"]
    current = row["result"]["forecast"]["path_output_analysis"]
    if fault == "width":
        analysis["reference_width_upper"] += 1
    elif fault == "sampling":
        analysis["sampling_error"]["uncertainty_radius"] = 0.
    elif fault == "source-statistics":
        analysis["completed_statistics"]["method_state"]["statistics"][0]["n"] -= 1
    elif fault == "source-stream":
        analysis["completed_statistics"]["rng_state"]["seed"] += 1
    elif fault == "binomial":
        analysis["sampling_error"]["interval_kind"] = "exact-binomial-one-sided-95-formula"
    elif fault == "unknown":
        analysis["implementation_roundoff"] = {"value": 0, "status": "BOUNDED"}
    elif fault == "grid":
        analysis["target_certificate"]["grid"]["steps"] *= 2
    elif fault == "scientific":
        analysis["scientific_qualification"] = True
    elif fault == "source-cost":
        evidence["settlement_event"]["payload"]["charged_ms"] = 0
    elif fault == "native-stop":
        evidence["stop_event"]["payload"]["confirmation"] = "not-stopped"
    elif fault == "completion-order":
        evidence["completion_event"]["sequence"] = row["admission"]["input_evidence"][0]["sequence"]+1
    elif fault == "source-grant":
        evidence["authorization"]["consumer_study_ids"] = []
    elif fault == "target-scalar":
        row["result"]["forecast"]["functional"]["estimate"] += .001
    elif fault == "target-statistics":
        current["completed_statistics"]["method_state"]["statistics"][0]["n"] -= 1
    elif fault == "target-classification":
        current["status"] = row["result"]["forecast"]["current_output_qualification"] = "FAILED"
    elif fault == "target-metric":
        row["metrics"][METRIC] = row["result"]["metrics"][METRIC] = 100.
    elif fault == "target-unit":
        row["metric_units"][METRIC] = row["result"]["metric_units"][METRIC] = "other"
    else:
        row["result"]["forecast"]["functional"]["error_budget"]["reference"]["estimated_by"] = "other"
    reseal_transport(bundle)
    with pytest.raises(ValueError):
        independent_reader().validate_path_qualification(row["admission"], row)


@pytest.mark.parametrize("fault", ["missing-numeric", "missing-pointer", "false-current-pass"])
def test_path_cli_cannot_fall_back_to_generic_or_analytic_approval(exported, tmp_path, fault):
    bundle = deepcopy(exported)
    row = bundle["cells"][0]
    if fault in {"missing-numeric", "missing-pointer"}:
        row["admission"]["documents"].pop("propagation_qualification")
        if fault == "missing-pointer":
            row["admission"]["documents"]["package"]["payload"].pop("managed_path_qualification")
    else:
        row["result"]["forecast"]["path_output_analysis"]["sampling_error"]["uncertainty_radius"] = 0.
    reseal_transport(bundle)
    assert paper_validate(tmp_path, bundle).returncode != 0
