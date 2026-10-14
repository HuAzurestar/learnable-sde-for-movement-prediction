"""Read-only domain validation/restoration of registered formal fit records.

No data loading, fitting, forecasting, authorization or attempt reset. The
controller must separately check authority and actual artifact bytes. This
consumer validates model meaning and binding instead of trusting a PASS flag.
"""
from collections import Counter
from dataclasses import fields

import numpy as np

from experiments.nex326.model import ModelState
from .cached_method_mechanisms import restore_cached_fit as restore_serialized_method
from .direct_linear import restore_direct_dynamics
from .inference import SEEDS
from .method_training import FIT_SOURCES, VERSION as TRAINING_VERSION, FittedMethod, _digest as training_digest
from .protocol_core import digest, unpack

VERSION = "pirc17-registered-fit-result-v1"


def restore_registered_fit(record, *, work, protocol, execution, matrix, input_identity, import_bridge=None):
    """Validate a saved artifact against exact owner-supplied expected inputs.

The legacy pure method deserializer above restores only the supplied model;
it never reads its module's historical cost-probe paths or reuses old fits.
Scientific sufficiency/benefit is NOT a condition for restoring an honest fit.
"""
    from .formal_import_scope import FIT_VERSION, bridge_for
    if unpack(record).get('schema_version') == FIT_VERSION:
        bridge = bridge_for(record, protocol=protocol, execution=execution, matrix=matrix,
                            input_identity=input_identity, bridge=import_bridge)
        return bridge.restore_fit(record, work=work)
    p, e, m, scope, result = (unpack(x) for x in (protocol, execution, matrix, input_identity, record))
    contract = p["forecast_contract"]["training"]
    matches = [w for w in m["workloads"] if w["work_id"] == work["work_id"]]
    if (matches != [work] or work["kind"] not in {"method_fit", "terrain_fit"}
            or e["protocol_sha256"] != protocol["sha256"] or e["matrix_sha256"] != matrix["sha256"]
            or m["protocol_sha256"] != protocol["sha256"]
            or scope["protocol_sha256"] != protocol["sha256"] or scope["execution_sha256"] != execution["sha256"]
            or scope["method_input_sha256"] != p["dataset_inputs"]["development"]["method_input_sha256"]
            or scope["sample_counts"] != contract["method_roles"]):
        raise ValueError("saved fit expected scope differs")
    expected = dict(schema_version=VERSION, protocol_sha256=protocol["sha256"], execution_sha256=execution["sha256"],
        matrix_sha256=matrix["sha256"], input_sha256=input_identity["sha256"], approval_sha256=scope["approval_sha256"],
        work_id=work["work_id"], fit_identity=work["fit_identity"])
    if set(result) != set(expected) | {"parameter_identity", "artifact"} or any(result[k] != v for k, v in expected.items()):
        raise ValueError("saved fit scope/work/approval differs")
    artifact = result["artifact"]
    if work["kind"] == "terrain_fit":
        name = work["subject"]
        if (set(artifact) != {"family", "model", "configuration", "deterministic_training_seed", "forecast_seed_labels"}
                or artifact["family"] != "terrain" or artifact["configuration"] != name
                or work["fit_identity"] != "terrain-fit:"+name or name not in p["components"]["terrain_configurations"]
                or artifact["deterministic_training_seed"] != SEEDS[0] or artifact["forecast_seed_labels"] != list(SEEDS)):
            raise ValueError("saved terrain configuration/seed binding differs")
        model = restore_direct_dynamics(artifact["model"])
        expected_model = dict(configuration=name, seed=SEEDS[0],
            training_identity=digest({"inputs": input_identity["sha256"], "configuration": name}),
            train_transition_count=contract["outer_train_windows"], validation_transition_count=contract["validation_windows"],
            configuration_identity=p["components"]["terrain_configurations"][name]["sha256"])
        if (any(model.identity[k] != v for k, v in expected_model.items())
                or model.identity["training_policy"]["sha256"] != contract["terrain"]["training_policy_sha256"]
                or result["parameter_identity"] != model.identity["sha256"]):
            raise ValueError("saved terrain training identity/population differs")
        return model
    groups = [g for g in m["training_groups"] if "method-fit:"+g["training_components_sha256"] == work["fit_identity"]]
    if len(groups) != 1:
        raise ValueError("saved method has no unique training group")
    group = groups[0]
    if (set(artifact) != {"family", "model", "dynamics", "training", "slot_bindings", "fit_seconds"}
            or artifact["family"] != "NEX326-methods" or artifact["slot_bindings"] != group["slots"]
            or work["subject"] != group["representative_slot"]
            or type(artifact["fit_seconds"]) not in (int, float) or not np.isfinite(artifact["fit_seconds"])
            or artifact["fit_seconds"] < 0):
        raise ValueError("saved method slot bindings or fit duration differ")
    training = artifact["training"]
    components = group["training_components"]
    sources = {name: p["source_sha256"]["PSDE-SDE/"+name] for name in FIT_SOURCES}
    if (training["version"] != TRAINING_VERSION
            or training["training_identity_sha256"] != training_digest({k: v for k, v in training.items() if k != "training_identity_sha256"})
            or training["input_sha256"] != scope["method_input_sha256"] or training["source_sha256"] != sources
            or training["training_components"] != components or training["sample_counts"] != contract["method_roles"]
            or training["reference_interval_seconds"] != components["dt_seconds"]
            or training["recommended_method_history_clock_seconds"] != components["dt_seconds"]
            or training["interval_policy"] != "complete-uniform-intervals-only"
            or training["formal_training_accepted"] is not False
            or training["continuous_time_MLE_equivalence_claimed"] is not False):
        raise ValueError("saved method training identity/policy differs")
    rows = training["per_segment"]
    if (len({r["segment_id"] for r in rows}) != len(rows)
            or dict(Counter(r["role"] for r in rows)) != contract["method_roles"]
            or any(type(r["transition_count"]) is not int or r["transition_count"] < 3
                   or r["reference_interval_seconds"] != components["dt_seconds"]
                   or type(r["removed_partial_tail"]) is not bool
                   or (not 0 < r["removed_tail_seconds"] < components["dt_seconds"] if r["removed_partial_tail"]
                       else r["removed_tail_seconds"] != 0) for r in rows)
            or sum(r["transition_count"] for r in rows) > contract["methods"]["max_transitions_per_fit"]
            or training["transitions_by_role"] != {role: sum(r["transition_count"] for r in rows if r["role"] == role)
                                                   for role in contract["method_roles"]}
            or training["removed_tails_by_role"] != {role: sum(r["removed_partial_tail"] for r in rows if r["role"] == role)
                                                     for role in contract["method_roles"]}):
        raise ValueError("saved method complete role/transition accounting differs")
    data = artifact["model"]
    if (set(data) != {f.name for f in fields(ModelState)} or data["model_kind"] != components["model"]
            or data["condition_names"] != components["condition"] or data["estimator_method"] != components["estimator"]
            or data["transfer_method"] != components["transfer"] or data["finetune_method"] != components["finetune"]
            or type(data["training_sample_count"]) is not int or data["training_sample_count"] < 1
            or data["covariance_scale"] <= 0):
        raise ValueError("saved method model/component meaning differs")
    dynamics = restore_serialized_method(artifact)
    if (result["parameter_identity"] != dynamics.fit_identity
            or dynamics.reference_interval_seconds != components["dt_seconds"]):
        raise ValueError("saved method parameter identity or clock differs")
    return FittedMethod(dynamics, training, work["subject"], float(artifact["fit_seconds"]))
