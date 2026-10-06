"""Owner gate for settled affine MLMC pilots and independent production.

Only bounded arithmetic on saved certificates/statistics, never uncharged
matrix/CDF replay. Eligibility is scoped to a frozen declared synthetic law;
normal sampling intervals are estimated, not rigorous coverage guarantees.
"""

from datetime import datetime, timezone
from fractions import Fraction
import json
import math
from types import SimpleNamespace

from domain.affine_mlmc_qualification import AffineMLMCReferencePolicy
from domain.mlmc_pilot import MLMCPilotPolicy
from domain.mlmc_production import MLMCProductionPolicy
from domain.errors import DataValidationError
from inference.affine_reference import AffineReferenceCertificate, _parse_interval, algorithm_manifest
from inference.affine_qualification import _outward_float, _scaled_norm
from inference.mlmc_pilot import analyze_mlmc_pilot
from infrastructure.research_store import ResearchError, digest, encode


PAYLOAD_KEY = "managed_mlmc_qualification"
METRIC = "absolute_error_upper_vs_declared_affine_law"


def require(condition, detail):
    if not condition:
        raise ResearchError("UNQUALIFIED", "managed MLMC qualification: "+detail)


def functional_result(value):
    """Canonical saved result -> cheap statistical validator input."""
    fields = dict(value)
    fields["diagnostics"] = tuple((k, tuple(v) if type(v) is list else v) for k, v in fields["diagnostics"])
    fields["interval"] = None if fields["interval"] is None else tuple(fields["interval"])
    fields["error_budget"] = SimpleNamespace(**{k: SimpleNamespace(**v) for k, v in fields["error_budget"].items()})
    return SimpleNamespace(**fields)


def sampling_values(result, request, counts, *, pilot):
    diagnostics = dict(result.diagnostics)
    require(len(diagnostics) == len(result.diagnostics) and tuple(diagnostics["level_samples"]) == counts,
        "level counts/diagnostics differ")
    means, variances = diagnostics["level_means"], diagnostics["level_variances"]
    require(len(means) == len(variances) == len(counts)
        and all(type(v) in (int, float) and math.isfinite(v) for v in (*means, *variances))
        and all(v >= 0 for v in variances), "finite actual level moments required")
    estimate = math.fsum(means)
    se = math.sqrt(math.fsum(v/n for v, n in zip(variances, counts)))
    unresolved = request.functional == "endpoint-halfspace" and se == 0
    expected_interval = None if unresolved else (estimate-1.959963984540054*se, estimate+1.959963984540054*se)
    require(result.request_hash == request.request_hash and result.kind == "functional_estimate"
        and type(result.estimate) in (int, float) and math.isfinite(result.estimate)
        and result.estimator_id == ("coupled-euler-mlmc-pilot-v1" if pilot else "coupled-euler-mlmc-v1")
        and result.sample_count == sum(counts) and result.estimate == estimate
        and result.standard_error == (None if unresolved else se) and result.interval == expected_interval
        and result.interval_kind == ("unavailable-zero-observed-level-variance" if unresolved else "independent-level-normal-approximation-95")
        and result.status == ("UNRESOLVED_SAMPLING" if unresolved else "PILOT_ONLY" if pilot else "SUCCEEDED")
        and diagnostics["coupling"] == "coarse-increment=sum(two-fine-increments)", "actual sampling scalar/SE/interval/identity differs")
    return diagnostics


def source_analysis_values(analysis, reference, pilot, package, request, result):
    reference.validate(package, request, pilot, reference.code_hash)
    sampling_values(result, request, reference.level_samples, pilot=True)
    empirical = analyze_mlmc_pilot(request, result, pilot)
    require(encode(analysis["empirical_pilot_analysis"]) == encode(empirical), "empirical pilot recheck differs")
    require(analysis["schema_version"] == "affine-mlmc-reference-pilot-analysis-v1"
        and analysis["status"] == "NUMERICAL_READY" and analysis["qualification"] == "UNQUALIFIED"
        and analysis["scientific_qualification"] is False
        and analysis["reference_policy_hash"] == reference.policy_hash and analysis["pilot_policy_hash"] == pilot.policy_hash
        and analysis["analysis_hash"] == digest({k: v for k, v in analysis.items() if k != "analysis_hash"})
        and all(analysis[k] == reference.manifest()[k] for k in ("request_hash", "model_package_hash", "code_hash")),
        "source analysis frozen identity/readiness differs")
    finest_steps = request.steps*2**(len(reference.level_samples)-1)
    cm, fm = analysis["continuous_certificate"], analysis["finest_certificate"]
    for document, discrete in ((cm, False), (fm, True)):
        AffineReferenceCertificate(encode(document)).manifest()
        require(document["schema_version"] == ("affine-discrete-certificate-v1" if discrete else "affine-reference-certificate-v1")
            and document["scope"] == ("declared-affine-finite-grid-Gaussian-law" if discrete else "declared-affine-Gaussian-endpoint-law")
            and document["scientific_qualification"] is False and document["algorithm"] == algorithm_manifest()
            and document["status"] == "BOUNDED" and type(document["operations"]) is int
            and 0 < document["operations"] <= 200_000
            and all(document[k] == reference.manifest()[k] for k in ("request_hash", "model_package_hash", "code_hash")),
            "saved component binding differs")
    require(fm["grid"] == {"solver": "euler", "steps": finest_steps, "time_step": "exact-horizon-rational/steps",
        "noise": "independent-centered-Gaussian-increments", "roundoff_scope": "mathematical-recurrence-only"}
        and analysis["finest_grid"] == {"solver": "euler", "base_steps": request.steps,
            "level_count": len(reference.level_samples), "steps": finest_steps}
        and analysis["continuous_certificate_hash"] == digest(cm) and analysis["finest_certificate_hash"] == digest(fm),
        "actual finest grid/component hashes differ")
    cb, fb = _parse_interval(cm["functional_bounds"]), _parse_interval(fm["functional_bounds"])
    bias = _parse_interval(analysis["signed_finest_grid_bias_bounds"])
    require((bias.lo, bias.hi) == (fb.lo-cb.hi, fb.hi-cb.lo), "signed finest bias differs")
    width, absolute_bias = cb.hi-cb.lo, max(abs(bias.lo), abs(bias.hi))
    norm = max(_scaled_norm(cm, reference.state_scales), _scaled_norm(fm, reference.state_scales))
    operations = cm["operations"]+fm["operations"]+1
    failures = sorted(set(empirical["reasons"]) & {"UNRESOLVED_LEVEL_VARIANCE", "VARIANCE_DECAY_FAILED",
        "COST_GROWTH_FAILED", "SAMPLING_UNRESOLVED", "ALLOCATION_INFEASIBLE"})
    checks = {"resolved_reference": True, "reference_width": width <= Fraction(reference.maximum_reference_width),
        "finest_grid_bias": absolute_bias <= Fraction(pilot.bias_tolerance),
        "scaled_transition_growth": norm <= Fraction(reference.maximum_scaled_transition_norm),
        "arithmetic_operations": operations <= reference.maximum_operations, "empirical_sampling_and_cost": not failures}
    require(analysis["checks"] == checks and all(type(v) is bool for v in analysis["checks"].values())
        and all(checks.values()) and analysis["empirical_failures"] == failures, "actual numerical/empirical checks failed")
    expected = {"reference_width_upper": _outward_float(width), "finest_grid_bias_absolute_upper": _outward_float(absolute_bias),
        "scaled_transition_norm_upper": _outward_float(norm), "operations": operations,
        "state_scales": list(reference.state_scales), "state_scale_units": ["m", "m", "m/s", "m/s"],
        "functional_unit": "m" if request.functional == "endpoint-x" else "1",
        "observed_pilot_error_upper_vs_continuous_law": _outward_float(max(abs(Fraction(result.estimate)-cb.lo), abs(Fraction(result.estimate)-cb.hi))),
        "sampling_error": {"value": result.standard_error, "status": "ESTIMATED", "interval_kind": result.interval_kind,
            "confidence_interval": list(result.interval)},
        "sampler_roundoff": {"value": None, "status": "NOT_IDENTIFIABLE"}, "model_error": {"value": None, "status": "NOT_IDENTIFIABLE"},
        "cost_status": "OWNER_SETTLEMENT_REQUIRED", "maximum_job_seconds": reference.maximum_job_seconds,
        "production_stream": empirical["production_stream"], "requires_separate_production_registration": True,
        "automatic_execution": False, "observed_error_scope": "current scalar only; sampling and implementation effects not separated",
        "scope": "necessary-affine-finest-grid-and-empirical-pilot-checks-only"}
    require(all(encode(analysis.get(k)) == encode(v) for k, v in expected.items()), "saved numerical values/unknowns differ")
    return cb, bias, empirical


def prepare_managed_mlmc(store, spec, cell, execution_package, prereg):
    from application.propagation_execution import validate_propagation_cell, request_from_manifest
    from application.research_reuse import verified_reuse
    from experiments.pirc25.affine import code_hash
    from experiments.pirc27.mlmc_production_plugin import production_policy, PLUGIN_ID
    from experiments.pirc27.mlmc_qualification_plugin import mlmc_qualification_plugin
    try:
        package, request, config, _ = validate_propagation_cell(spec, cell)
        require(cell["plugin_id"] == PLUGIN_ID and config["method"] == "mlmc", "explicit MLMC production adapter required")
        policy = production_policy(spec, cell, package, request, config)
        pointer = execution_package["payload"][PAYLOAD_KEY]
        require(type(pointer) is dict and set(pointer) == {"schema_version", "production_policy", "source_attempt_id",
            "source_artifact_id", "source_authorization_id", "source_authorization_version"}
            and pointer["schema_version"] == "managed-affine-mlmc-qualification-v1"
            and pointer["production_policy"] == policy.manifest(), "explicit source/production pointer required")
        policies = prereg.get("mlmc_production_policies")
        require(type(policies) is list and 0 < len(policies) <= 10000 and policies.count(policy.manifest()) == 1
            and len({digest(p) for p in policies}) == len(policies) and prereg["primary_metrics"] == [METRIC],
            "production policy/metric not separately frozen")
        attempt = store.attempts()[pointer["source_attempt_id"]]
        require(attempt["state"] == "SUCCEEDED" and attempt["artifact_id"] == pointer["source_artifact_id"], "source not completed")
        run = store.manifest("run-"+attempt["run_id"])
        source_spec = store.manifest("study-"+run["study_id"])["spec"]
        sc = run["cell"]
        require(source_spec["runtime_binding"] == spec["runtime_binding"] and source_spec["code_hash"] == code_hash() == spec["code_hash"]
            and sc["plugin_id"] == "affine-mlmc-qualification-chunk" and sc["execution_role"] == "pilot"
            and sc["frozen_dynamics"] == cell["frozen_dynamics"] and sc["arm_id"] == cell["arm_id"], "source producer/law/arm/root differs")
        source_request = request_from_manifest(sc["propagation_request"])
        pilot = MLMCPilotPolicy(**sc["mlmc_pilot_policy"])
        reference = AffineMLMCReferencePolicy.from_manifest(sc["affine_mlmc_reference_policy"])
        reference.validate(package, source_request, pilot, code_hash())
        require(policy.pilot_request_hash == source_request.request_hash and policy.pilot_policy_hash == pilot.policy_hash
            and policy.reference_policy_hash == reference.policy_hash, "source policies differ")
        ignored = {"request_id", "seed", "coupling_id", "samples", "chunk_size"}
        require({k: v for k, v in sc["propagation_request"].items() if k not in ignored}
            == {k: v for k, v in cell["propagation_request"].items() if k not in ignored}
            and request.seed == pilot.production_seed and request.coupling_id == pilot.production_coupling_id,
            "production law/grid or independent stream differs")
        require({a["arm_id"]: a for a in source_spec["arms"]}[cell["arm_id"]]
            == {a["arm_id"]: a for a in spec["arms"]}[cell["arm_id"]], "original cumulative family arm differs")
        grant = store.authorization(pointer["source_authorization_id"], version=pointer["source_authorization_version"])
        require(grant["study_id"] == source_spec["study_id"] and grant["protocol_hash"] == source_spec["protocol_hash"]
            and (grant["study_id"] == spec["study_id"] or spec["study_id"] in grant.get("consumer_study_ids", []))
            and "evaluate" in grant["purposes"] and datetime.fromisoformat(grant["expires_at"]) > datetime.now(timezone.utc),
            "source grant does not authorize consumer")
        metadata = store.manifest("artifact-"+attempt["artifact_id"])
        require(metadata["size_bytes"] <= 512*1024, "source byte quota")
        content = store.read_artifact(attempt["artifact_id"], purpose="evaluate", authorization=grant)
        result = json.loads(content)
        require(content == encode(result), "canonical actual source required")
        verified_reuse(store, attempt, source_spec, sc, mlmc_qualification_plugin())
        receipt = store.manifest("admission-"+result["admission_hash"])
        blocks = [b for b in receipt["documents"]["protocol"]["blocks"] if b["block_id"] == sc["block_id"]]
        require(receipt["mode"] == "pilot" and len(blocks) == 1 and blocks[0]["split_role"] in {"train", "validation"}, "held-out pilot forbidden")
        _, _, empirical = source_analysis_values(result["forecast"]["bounded_mlmc_pilot_analysis"], reference, pilot,
            package, source_request, functional_result(result["forecast"]["functional"]))
        require(tuple(empirical["sampling_only_proposal"]) == policy.level_samples
            and sum(policy.level_samples) <= pilot.maximum_samples
            and sum(n*w for n, w in zip(policy.level_samples, dict(functional_result(result["forecast"]["functional"]).diagnostics)["level_work_per_sample"]))
                <= pilot.maximum_work_steps, "production allocation differs from bounded independent pilot proposal")
        events = store.events()
        def one(kind):
            matches = [e for e in events if e["event_kind"] == kind and e["payload"].get("attempt_id") == attempt["attempt_id"]]
            require(len(matches) == 1, "source lacks unique "+kind)
            return matches[0]
        reservation, worker, stop, settlement, admission_event = (one(k) for k in
            ("RESERVE", "WORKER_STARTED", "WORKER_TREE_STOPPED", "SETTLE", "ADMISSION"))
        completions = [e for e in events if e["event_kind"] == "ATTEMPT" and e["payload"] == attempt]
        cost = settlement["payload"]
        require(len(completions) == 1 and admission_event["payload"]["admission_hash"] == receipt["admission_hash"]
            and admission_event["sequence"] < worker["sequence"] < stop["sequence"] < settlement["sequence"] < completions[0]["sequence"]
            and type(cost["charged_ms"]) is int and 0 < cost["charged_ms"] <= cost["reserved_ms"] <= Fraction(reference.maximum_job_seconds)*1000
            and stop["payload"]["reservation_id"] == cost["reservation_id"]
            and stop["payload"]["observed_elapsed_ms"] == cost["charged_ms"]
            and stop["payload"]["confirmation"] == "native-job-or-process-group-no-running-descendants",
            "source measured native-stop settlement/order differs")
        evidence = {"schema_version": "managed-affine-mlmc-admission-evidence-v1", "production_policy": policy.manifest(),
            "pilot_policy": sc["mlmc_pilot_policy"], "reference_policy": reference.manifest(),
            "source_attempt": attempt, "source_run": run, "source_artifact": metadata, "source_result": result,
            "source_admission": receipt, "authorization": grant, "reservation_event": reservation, "worker_event": worker,
            "stop_event": stop, "settlement_event": settlement, "admission_event": admission_event, "completion_event": completions[0],
            "target_request_hash": request.request_hash, "target_cell_hash": digest(cell), "target_spec_hash": digest(spec),
            "preregistration_hash": digest(prereg)}
        evidence["evidence_hash"] = digest(evidence)
        require(len(encode(evidence)) <= 2*1024*1024, "source evidence byte quota")
        store.verify_artifact_read(attempt["artifact_id"], purpose="evaluate", authorization=grant)
        return evidence
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError, DataValidationError) as exc:
        if isinstance(exc, ResearchError):
            raise
        raise ResearchError("UNQUALIFIED", "managed MLMC evidence is missing or malformed") from exc


def qualified_mlmc_forecast(spec, cell, package, request, result, receipt):
    from application.propagation_execution import request_from_manifest
    from experiments.pirc25.affine import code_hash
    require(receipt["spec_hash"] == digest(spec) and receipt["cell_hash"] == digest(cell)
        and receipt["mode"] == "formal" and receipt["qualification"] == "qualified", "owner formal receipt required")
    evidence = receipt["documents"]["propagation_qualification"]
    require(evidence["schema_version"] == "managed-affine-mlmc-admission-evidence-v1"
        and evidence["evidence_hash"] == digest({k: v for k, v in evidence.items() if k != "evidence_hash"})
        and evidence["target_spec_hash"] == digest(spec) and evidence["target_cell_hash"] == digest(cell), "target evidence differs")
    policy = MLMCProductionPolicy.from_manifest(evidence["production_policy"])
    policy.validate(package, request, code_hash())
    require(policy.manifest() == cell["mlmc_production_policy"], "current frozen production policy differs")
    pilot = MLMCPilotPolicy(**evidence["pilot_policy"])
    reference = AffineMLMCReferencePolicy.from_manifest(evidence["reference_policy"])
    require(policy.pilot_policy_hash == pilot.policy_hash and policy.reference_policy_hash == reference.policy_hash
        and policy.pilot_request_hash == pilot.pilot_request_hash == reference.request_hash
        and reference.code_hash == policy.code_hash and request.seed == pilot.production_seed
        and request.coupling_id == pilot.production_coupling_id, "current source policy/stream linkage differs")
    source = evidence["source_result"]["forecast"]
    continuous, bias, _ = source_analysis_values(source["bounded_mlmc_pilot_analysis"], reference, pilot, package,
        request_from_manifest(evidence["source_run"]["cell"]["propagation_request"]), functional_result(source["functional"]))
    sampling = sampling_values(result, request, policy.level_samples, pilot=False)
    require(result.standard_error is not None and result.standard_error <= pilot.sampling_tolerance
        and all(v > 0 for v in sampling["level_variances"]), "production sampling target unresolved or failed")
    error = _outward_float(max(abs(Fraction(result.estimate)-continuous.lo), abs(Fraction(result.estimate)-continuous.hi)))
    components = {"reference_width_upper": _outward_float(continuous.hi-continuous.lo),
        "time_bias_absolute_upper": _outward_float(max(abs(bias.lo), abs(bias.hi))),
        "signed_time_bias_bounds": source["bounded_mlmc_pilot_analysis"]["signed_finest_grid_bias_bounds"],
        "sampling_error": {"value": result.standard_error, "status": "ESTIMATED", "interval_kind": result.interval_kind},
        "sampler_roundoff": {"value": None, "status": "NOT_IDENTIFIABLE"}, "model_error": {"value": None, "status": "NOT_IDENTIFIABLE"},
        "scope": "declared-affine-law-only;current-scalar-error-includes-sampling-and-implementation",
        "confidence_scope": "normal-approximation-not-rigorous-coverage", "policy_hash": policy.policy_hash,
        "qualification_evidence_hash": evidence["evidence_hash"], "qualification_source_attempt_id": evidence["source_attempt"]["attempt_id"],
        "stream": {"phase": 1, "seed": request.seed, "coupling_id": request.coupling_id, "level_samples": list(policy.level_samples)}}
    return error, components


def validate_formal_mlmc_result(receipt, spec, cell, result):
    from application.propagation_execution import validate_propagation_cell
    package, request, _, _ = validate_propagation_cell(spec, cell)
    functional = functional_result(result["forecast"]["functional"])
    error, components = qualified_mlmc_forecast(spec, cell, package, request, functional, receipt)
    unit = "m" if request.functional == "endpoint-x" else "1"
    require(result["metrics"] == {METRIC: error} and result["metric_units"] == {METRIC: unit}
        and result["forecast"]["model_package_hash"] == package.package_hash and result["forecast"]["request_hash"] == request.request_hash
        and result["forecast"]["qualified_error_components"] == components, "current production metrics/provenance differ")
    budget = result["forecast"]["functional"]["error_budget"]
    require(budget["reference"]["value"] == components["reference_width_upper"] and budget["reference"]["status"] == "BOUNDED"
        and budget["time_discretization"]["value"] == components["time_bias_absolute_upper"] and budget["time_discretization"]["status"] == "BOUNDED"
        and budget["sampling"]["value"] == functional.standard_error and budget["sampling"]["status"] == "ESTIMATED"
        and budget["model"]["value"] is None and budget["model"]["status"] == "NOT_IDENTIFIABLE", "separated stochastic errors differ")
