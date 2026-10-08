"""Guarded, fixed-population preparation and the 16+10 actual fit consumers.

No empirical work is launched here. The retained formal worker must call these
consumers INSIDE its reserved input/fit work items. The native controller owns
deadlines, durable attempts and process closure; this module owns data/model
semantics. There is no legacy-model fallback, parameter search or public CLI.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from . import final_eval_guard as guard
from .configurations import configuration_encoder, subset_matrix, terrain_configurations
from .development import load_development, transition_rows
from .delivery_policy import training_groups
from .direct_linear import configuration_width, fit_direct_dynamics, restore_direct_dynamics, training_policy
from .features import CanonicalEncoder, EARTH_RADIUS_M
from .formal_eligibility import _bound_file, _release_sources
from .formal_fit_records import VERSION as FIT_VERSION, restore_registered_fit
from .formal_origins import RegisteredVelocityPrior, build_velocity_prior, validate_prior
from .inference import SEEDS
from .method_development import MethodDevelopment, PreparationBudget, load_method_development
from .method_training import fit_development_method
from .protocol_core import digest, envelope, publish, read_json, sha256, under, unpack

INPUT_VERSION = "pirc17-registered-training-inputs-v1"


@dataclass(frozen=True)
class RegisteredTrainingInputs:
    methods: MethodDevelopment
    terrain: dict
    encoders: dict
    prior: RegisteredVelocityPrior
    identity: dict


@dataclass(frozen=True)
class RestoredFitInputs:
    """No development training values retained or fabricated during recovery."""
    encoders: dict
    prior: RegisteredVelocityPrior
    identity: dict


def _crosscheck_development(methods, windows):
    """Both matrices must use the same original prefix/first transition."""
    if (set(windows) != {"train", "validation"} or len(methods.prefixes) != len(methods.segments)
            or any(p.assignment.sample.segment_id != s.segment_id for p, s in zip(methods.prefixes, methods.segments))):
        raise ValueError("complete aligned method segments and two development roles required")
    lookup = {p.assignment.sample.sample_id: (p, s) for p, s in zip(methods.prefixes, methods.segments)}
    if len(lookup) != len(methods.prefixes) or sorted(lookup) != methods.identity["sample_ids"]:
        raise ValueError("method development population differs from its identity")
    observed = []
    tolerance = 128*np.finfo(float).eps*EARTH_RADIUS_M
    for role, group in windows.items():
        for window in group:
            observed.append(window.sample_id)
            if window.sample_id not in lookup:
                raise ValueError("terrain window absent from method development population")
            prefix, segment = lookup[window.sample_id]
            if prefix.assignment.sample.split != role or window.role != role:
                raise ValueError("terrain/method development split differs")
            n = len(prefix.visible_epoch_ns)
            frame = prefix.condition_at.frame
            history = window.frame.from_lonlat(frame.to_lonlat(prefix.origin.history_positions_m))
            next_position = window.frame.from_lonlat(frame.to_lonlat(segment.state[n]))
            if (window.block_id != prefix.assignment.sample.independent_block_id
                    or not np.array_equal(window.origin.history_times_seconds, prefix.origin.history_times_seconds)
                    or not np.allclose(window.origin.history_positions_m, history, rtol=0, atol=tolerance)
                    or window.transition_seconds != segment.time[n]
                    or not np.allclose(window.transition_displacement_m, next_position-window.origin.position_m, rtol=0, atol=tolerance)):
                raise ValueError("method/terrain original prefix or first-future transition differs")
    if len(observed) != len(set(observed)) or set(observed) != set(lookup):
        raise ValueError("two matrices require the entire identical development population")


def _prepare(access, *, protocol, execution, eligibility_path, release, snapshot, data_root, trajectory_path):
    p = unpack(protocol)
    binding = p["dataset_inputs"]
    contract = p["forecast_contract"]["training"]
    if (access.access_kind != "final_eval_features" or access.population_sha256 is None
            or access.legacy_cohort_ack != binding["dataset_id"]
            or access.protocol_sha256 != protocol["sha256"] or access.execution_sha256 != execution["sha256"]
            or unpack(execution)["protocol_sha256"] != protocol["sha256"]):
        raise ValueError("approved population-bound snapshot access required")
    _release_sources(release, binding)
    dataset = read_json(under(release, "dataset.json"), expected_file_sha256=binding["dataset_file_sha256"])
    if dataset["source"]["trajectory"]["sha256"] != binding["source_trajectory_sha256"]:
        raise ValueError("source trajectory identity differs from protocol")
    frozen = binding["snapshot"]
    read_json(under(snapshot, "manifest.json"), expected_file_sha256=frozen["manifest_sha256"])
    read_json(under(snapshot, "feature_spec.json"), expected_file_sha256=frozen["feature_spec_file_sha256"])
    qualification = read_json(eligibility_path, expected_file_sha256=contract["eligibility_sha256"])
    eligible = qualification["eligibility"]
    rows = eligible["rows"]
    if (qualification["dataset_id"] != binding["dataset_id"] or eligible["horizon_minutes"] != 30
            or eligible["roles"] != ["train", "validation"] or any(r["split"] not in eligible["roles"] for r in rows)
            or len(rows) != contract["outer_train_windows"]+contract["validation_windows"]
            or len({r["sample_id"] for r in rows}) != len(rows)):
        raise ValueError("entire frozen task-qualified development population required")
    _bound_file(release, "condition_file_manifest.jsonl", binding["release_artifact_sha256"]["condition_file_manifest.jsonl"])
    methods = load_method_development(Path(eligibility_path), contract["eligibility_sha256"], release, snapshot,
        Path(data_root)/"cond_slices", trajectory_path, sample_ids=sorted(r["sample_id"] for r in rows),
        budget=PreparationBudget(len(rows), binding["development"]["observed_points"], 33554432, 250000,
            min(180., p["resource_contract"]["phase_caps_seconds"]["input_qualification_and_binding"])))
    if (methods.identity["sha256"] != binding["development"]["method_input_sha256"]
            or methods.identity["sample_counts"] != contract["method_roles"]
            or methods.identity["observed_points"] != binding["development"]["observed_points"]):
        raise ValueError("formal preparation differs from already qualified method inputs")
    prior = build_velocity_prior(methods, training_contract=contract,
                                 expected_input_sha256=binding["development"]["method_input_sha256"])
    # This adapter also checks held-out file hashes/Parquet metadata, hence the
    # final_eval_features guard above even though numeric rows below are ONLY
    # train/validation. No snapshot adapter is instantiated before approval.
    encoder = CanonicalEncoder.frozen_pirc22(Path(snapshot))
    windows, parents, terrain_identity = load_development(Path(eligibility_path), contract["eligibility_sha256"],
                                                          Path(release), Path(data_root), encoder)
    _crosscheck_development(methods, windows)
    if len(windows["train"]) != contract["outer_train_windows"] or len(windows["validation"]) != contract["validation_windows"]:
        raise ValueError("terrain training/diagnostic population differs")
    encoders = {name: configuration_encoder(encoder, name) for name in terrain_configurations()}
    # Transform canonical development rows once per role. Each configuration
    # gets its exact owned columns but still requires its OWN independent fit.
    full_rows = {role: transition_rows(group, encoder, encoder) for role, group in windows.items()}
    terrain = {name: {role: replace(batch, feature_matrix=subset_matrix(batch.feature_matrix, encoder.columns, selected.columns))
                        for role, batch in full_rows.items()} for name, selected in encoders.items()}
    identity = envelope({"schema_version": INPUT_VERSION, "protocol_sha256": protocol["sha256"],
        "execution_sha256": execution["sha256"], "approval_sha256": access.approval_sha256,
        "population_sha256": access.population_sha256, "access_started_sha256": access.access_started_sha256,
        "method_input_sha256": methods.identity["sha256"], "terrain_development_identity": terrain_identity,
        "terrain_training_policy_sha256": training_policy()["sha256"], "prior_evidence_sha256": prior.evidence["sha256"],
        "configuration_columns": {k: list(e.columns) for k, e in encoders.items()},
        "snapshot_parent_assets": parents, "sample_counts": dict(contract["method_roles"]),
        "outer_train_windows": contract["outer_train_windows"], "validation_windows": contract["validation_windows"],
        "final_eval_numeric_training_rows": 0, "new_predictive_model_fits": 0})
    return RegisteredTrainingInputs(methods, terrain, encoders, prior, identity)


def prepare_formal_training(*, protocol, execution, approval_path, approval_sha256, test_path, review_path,
                            journal_directory, population_path, population_sha256, final_eligibility_path,
                            development_eligibility_path, release, snapshot, data_root, trajectory_path):
    """The formal input work item calls this after final population sealing."""
    return guard.guarded_call(access_kind="final_eval_features", protocol=protocol, execution=execution,
        approval_path=approval_path, approval_sha256=approval_sha256, test_path=test_path, review_path=review_path,
        journal_directory=journal_directory, population_path=population_path, population_sha256=population_sha256,
        eligibility_path=final_eligibility_path, operation=lambda access: _prepare(access,
            protocol=protocol, execution=execution, eligibility_path=development_eligibility_path,
            release=release, snapshot=snapshot, data_root=data_root, trajectory_path=trajectory_path))


class FitConsumers:
    """Retained worker-side fit state, NOT a durable-attempt or approval ledger.

    Construct only after guarded input preparation and matrix validation. The
    owner must reserve each exact work_id before execute; the controller also
    independently verifies artifacts. In-memory attempts merely prevent an
    accidental duplicate dispatch in this same live handler.
    """
    def __init__(self, *, protocol, execution, matrix, inputs):
        if not isinstance(inputs, RegisteredTrainingInputs):
            raise ValueError("registered training inputs required")
        protocol, execution, matrix = (deepcopy(x) for x in (protocol, execution, matrix))
        inputs = replace(inputs, identity=deepcopy(inputs.identity),
                         terrain={k: dict(v) for k, v in inputs.terrain.items()}, encoders=dict(inputs.encoders))
        p, e, m, scope = (unpack(x) for x in (protocol, execution, matrix, inputs.identity))
        contract = p["forecast_contract"]["training"]
        if (scope["schema_version"] != INPUT_VERSION or scope["protocol_sha256"] != protocol["sha256"]
                or scope["execution_sha256"] != execution["sha256"] or e["protocol_sha256"] != protocol["sha256"]
                or m["protocol_sha256"] != protocol["sha256"] or e["matrix_sha256"] != matrix["sha256"]
                or scope["method_input_sha256"] != inputs.methods.identity["sha256"]
                or scope["method_input_sha256"] != p["dataset_inputs"]["development"]["method_input_sha256"]
                or digest({k: v for k, v in inputs.methods.identity.items() if k != "sha256"}) != scope["method_input_sha256"]
                or scope["sample_counts"] != contract["method_roles"]
                or inputs.methods.identity["sample_counts"] != contract["method_roles"]
                or scope["outer_train_windows"] != contract["outer_train_windows"]
                or scope["validation_windows"] != contract["validation_windows"]
                or scope["terrain_training_policy_sha256"] != contract["terrain"]["training_policy_sha256"]
                or scope["terrain_training_policy_sha256"] != training_policy()["sha256"]
                or scope["prior_evidence_sha256"] != inputs.prior.evidence["sha256"]
                or scope["final_eval_numeric_training_rows"] != 0 or scope["new_predictive_model_fits"] != 0):
            raise ValueError("fit scope/input/matrix binding differs")
        sha256(scope["approval_sha256"]); sha256(scope["population_sha256"])
        prior = validate_prior(inputs.prior)
        if (prior["input_sha256"] != scope["method_input_sha256"]
                or prior["population_identity"] != contract["population_identity"]
                or prior["outer_train_windows"] != contract["outer_train_windows"]):
            raise ValueError("prior is not derived from the registered training inputs")
        self.protocol, self.execution, self.matrix, self.inputs = protocol, execution, matrix, inputs
        self.contract = contract
        fits = [w for w in m["workloads"] if w["kind"] in {"method_fit", "terrain_fit"}]
        self.work = {w["work_id"]: w for w in fits}
        self.method_groups = {"method-fit:"+g["training_components_sha256"]: g for g in m["training_groups"]}
        configs = terrain_configurations()
        if (m["training_groups"] != contract["methods"]["groups"] or m["training_groups"] != training_groups()
                or len(fits) != 26 or len(self.work) != 26 or p["components"]["terrain_configurations"] != configs
                or set(inputs.terrain) != set(configs) or set(inputs.encoders) != set(configs)
                or scope["configuration_columns"] != {k: list(e.columns) for k, e in inputs.encoders.items()}):
            raise ValueError("all16 method and10 independent terrain fits required")
        expected = {k: ("method_fit", "method_training", g["representative_slot"], "NEX326-methods")
                    for k, g in self.method_groups.items()}
        expected.update({"terrain-fit:"+k: ("terrain_fit", "terrain_training", k, "terrain") for k in configs})
        if set(w["fit_identity"] for w in fits) != set(expected):
            raise ValueError("registered fit identities are missing or duplicated")
        for w in fits:
            if (tuple(w[k] for k in ("kind", "phase", "subject", "matrix")) != expected[w["fit_identity"]]
                    or w["work_id"] != digest({k: v for k, v in w.items() if k != "work_id"})
                    or w["generated_forecasts"] != 0 or w["scientific"] is not False
                    or any(w[k] is not None for k in ("origin_mode", "origin_rank", "seed", "repetition"))
                    or w["max_active_seconds"] != p["resource_contract"]["per_"+w["kind"]+"_seconds"]):
                raise ValueError("registered fit descriptor differs from its policy")
        for name, roles in inputs.terrain.items():
            if (set(roles) != {"train", "validation"} or 2*len(inputs.encoders[name].columns) != configuration_width(name)
                    or any(rows.feature_matrix.shape[1] != configuration_width(name) for rows in roles.values())):
                raise ValueError("configuration-owned terrain feature dimensions differ")
        self.attempted, self.models, self.receipts = set(), {}, {}

    @classmethod
    def restore_all(cls, *, bridge, encoders):
        """Recover all26 original fitted owners without raw training or fitting.

        This is a distinct, non-training initialization, not dummy development
        rows fed into __init__. It confers no worker/dispatch authority. Exact
        completed work is in attempted before a caller can execute anything.
        Unfitted canonical encoders must be restored by the guarded owner.
        """
        from .formal_import_scope import ScopeBridge
        if not isinstance(bridge, ScopeBridge):
            raise ValueError('verified partial science bridge required')
        context = bridge.new
        p, m, scope = (unpack(context[k]) for k in ('protocol', 'matrix', 'input_identity'))
        configs = terrain_configurations()
        if (p['components']['terrain_configurations'] != configs or set(encoders) != set(configs)
                or any(not isinstance(e, CanonicalEncoder) for e in encoders.values())
                or scope['configuration_columns'] != {k: list(e.columns) for k, e in encoders.items()}
                or any(2*len(encoders[k].columns) != configuration_width(k) for k in configs)
                or m['training_groups'] != p['forecast_contract']['training']['methods']['groups']
                or m['training_groups'] != training_groups()):
            raise ValueError('restored unfitted encoders/configurations/training groups differ')
        value = cls.__new__(cls)
        value.protocol, value.execution, value.matrix = (deepcopy(context[k]) for k in ('protocol', 'execution', 'matrix'))
        value.inputs = RestoredFitInputs(dict(encoders), context['prior'], deepcopy(context['input_identity']))
        value.contract = deepcopy(p['forecast_contract']['training'])
        value.import_bridge = fit_bridge = bridge.for_fits()
        value.imported_forecast_ids = frozenset(wid for wid in bridge.science['completed_sources']
            if bridge.work[wid]['kind'] in {'scientific_forecast', 'same_grid_reference'})
        fits = [w for w in m['workloads'] if w['kind'] in {'method_fit', 'terrain_fit'}]
        value.work = {w['work_id']: deepcopy(w) for w in fits}
        if (len(fits) != 26 or len(value.work) != 26
                or any(w['work_id'] not in bridge.science['completed_sources'] for w in fits)):
            raise ValueError('all26 original completed fitted owners required')
        value.method_groups = {'method-fit:'+g['training_components_sha256']: deepcopy(g) for g in m['training_groups']}
        value.attempted, value.models, value.receipts = set(value.work), {}, {}
        for work in fits:
            receipt = fit_bridge.fit_record(work)
            value.models[work['fit_identity']] = restore_registered_fit(receipt, work=work, protocol=value.protocol,
                execution=value.execution, matrix=value.matrix, input_identity=value.inputs.identity, import_bridge=fit_bridge)
            value.receipts[work['fit_identity']] = receipt
        bridge.check_references()
        return value

    def execute(self, work, *, output_directory):
        if not isinstance(work, dict) or self.work.get(work.get("work_id")) != work:
            raise ValueError("exact registered fit work item required")
        wid, fit_id = work["work_id"], work["fit_identity"]
        if wid in self.attempted or fit_id in self.models:
            raise ValueError("fit already attempted; no automatic retry or reuse across configurations")
        self.attempted.add(wid)
        scope = unpack(self.inputs.identity)
        if work["kind"] == "method_fit":
            group = self.method_groups.get(fit_id)
            if group is None or work["subject"] != group["representative_slot"] or work["phase"] != "method_training":
                raise ValueError("method fit key/representative differs")
            model = fit_development_method(self.inputs.methods, work["subject"],
                                           max_transitions=self.contract["methods"]["max_transitions_per_fit"])
            if model.training["sample_counts"] != self.contract["method_roles"]:
                raise ValueError("actual fitted method population changed")
            artifact = {"family": "NEX326-methods", "model": model.dynamics.model.to_dict(),
                "dynamics": model.dynamics.identity(), "training": model.training,
                "slot_bindings": list(group["slots"]), "fit_seconds": model.fit_seconds}
            parameter_identity = model.dynamics.fit_identity
        else:
            name = work["subject"]
            if fit_id != "terrain-fit:"+name or work["phase"] != "terrain_training":
                raise ValueError("terrain fit key differs")
            rows = self.inputs.terrain[name]
            if (len(rows["train"].elapsed_seconds) != self.contract["outer_train_windows"]
                    or len(rows["validation"].elapsed_seconds) != self.contract["validation_windows"]):
                raise ValueError("actual terrain training/diagnostic population changed")
            model = fit_direct_dynamics(rows["train"], rows["validation"], seed=SEEDS[0],
                training_identity=digest({"inputs": self.inputs.identity["sha256"], "configuration": name}), configuration=name)
            restore_direct_dynamics(model.identity)
            artifact = {"family": "terrain", "model": model.identity, "configuration": name,
                        "deterministic_training_seed": SEEDS[0], "forecast_seed_labels": list(SEEDS)}
            parameter_identity = model.identity["sha256"]
        output, record = publish(Path(output_directory), {"schema_version": FIT_VERSION,
            "protocol_sha256": self.protocol["sha256"], "execution_sha256": self.execution["sha256"],
            "matrix_sha256": self.matrix["sha256"], "input_sha256": self.inputs.identity["sha256"],
            "approval_sha256": scope["approval_sha256"], "work_id": wid, "fit_identity": fit_id,
            "parameter_identity": parameter_identity, "artifact": artifact})
        # The independent controller can call this same read-only domain
        # consumer on saved bytes; no fit or forecast is repeated to restore.
        restore_registered_fit(record, work=work, protocol=self.protocol, execution=self.execution,
                               matrix=self.matrix, input_identity=self.inputs.identity)
        self.models[fit_id] = model
        self.receipts[fit_id] = record
        return {"artifact_path": str(output), "artifact_sha256": record["sha256"],
                "fit_identity": fit_id, "parameter_identity": parameter_identity}
