"""Exact-version workflow guard, not an OS sandbox or a human authenticator.

The authorized operator must pin the actual ACCEPT01 receipt from independently
recorded human approval. A caller-created hash/boolean/cohort ID is not that
approval. TEST01/REVIEW01 attest readiness, not final scientific acceptance.
DEV04 must route its real loaders through this guard, implement eligibility,
and enforce cumulative compute/attempt budgets separately. No real approval
receipt is created by this module or by its synthetic unit tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .delivery_policy import select_identity_metadata
from .protocol import validate_execution, validate_protocol, verify_sources
from .protocol_core import canonical, digest, publish, read_json, sha256, unpack

CHECK_VERSION = "pirc17-pre-eval-check-v1"
APPROVAL_VERSION = "pirc17-human-accept01-v1"
ELIGIBILITY_VERSION = "pirc17-final-eligibility-report-v1"
POPULATION_VERSION = "pirc17-final-population-seal-v1"
CHECK_IDS = tuple(f"HC-{i:02d}" for i in range(1, 7)) + tuple(f"EC-{i:02d}" for i in range(1, 13))
ACCESS_KINDS = ("final_eval_eligibility", "final_eval_positions", "final_eval_features", "final_eval_metrics")
LIMITS = ("historical-final-eval-exposure-retained", "finite-budget-predictive-effect-only",
          "no-global-numerical-qualification", "no-automatic-experiment-growth",
          "48h-active-compute-not-project-eta", "full-evidence-paper-and-accept02-still-required")


def _fields(value, names):
    if not isinstance(value, dict) or set(value) != set(names):
        raise ValueError("exact registered receipt fields required")


def _text(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 8192:
        raise ValueError("nonempty bounded provenance text required")
    return value


def _scoped(payload, version, protocol, execution):
    if (payload["schema_version"] != version or payload["protocol_sha256"] != protocol["sha256"]
            or payload["execution_sha256"] != execution["sha256"]):
        raise ValueError("receipt names another protocol/execution scope")


def validate_check(value, *, task, protocol, execution):
    record = unpack(value)
    _fields(record, ("schema_version", "task", "protocol_sha256", "execution_sha256", "result", "checks"))
    _scoped(record, CHECK_VERSION, protocol, execution)
    if task not in {"TEST-01", "REVIEW-01"} or record["task"] != task or record["result"] != "PASS":
        raise ValueError("both exact pre-evaluation test and review passes required")
    _fields(record["checks"], CHECK_IDS)
    for check in record["checks"].values():
        _fields(check, ("result", "evidence_sha256", "command_or_review", "actual_result"))
        if check["result"] != "PASS":
            raise ValueError("failed/unknown/unexecuted checklist item cannot authorize access")
        _text(check["command_or_review"])
        _text(check["actual_result"])
        verify_sources(check["evidence_sha256"])
    return record


def validate_approval(value, *, expected_sha256, protocol, execution, test, review):
    """expected_sha256 is the independently pinned human-decision receipt."""
    record = unpack(value, expected_sha256=sha256(expected_sha256))
    _fields(record, ("schema_version", "task", "decision", "protocol_sha256", "execution_sha256",
                    "test_sha256", "review_sha256", "human_confirmation", "acknowledged_limits", "evidence_sha256"))
    _scoped(record, APPROVAL_VERSION, protocol, execution)
    if (record["task"] != "ACCEPT-01" or record["decision"] != "CONFIRMED"
            or record["test_sha256"] != test["sha256"] or record["review_sha256"] != review["sha256"]
            or record["acknowledged_limits"] != list(LIMITS)):
        raise ValueError("exact human decision and acknowledged scope limitations required")
    provenance = record["human_confirmation"]
    _fields(provenance, ("kind", "reference", "quoted_decision", "recorded_at_utc", "recorded_by"))
    if provenance["kind"] != "user-message":
        raise ValueError("an agent-generated approval flag is not human confirmation")
    for name in ("reference", "quoted_decision", "recorded_by"):
        _text(provenance[name])
    timestamp = datetime.fromisoformat(_text(provenance["recorded_at_utc"]).replace("Z", "+00:00"))
    if timestamp.utcoffset() is None or timestamp.utcoffset().total_seconds() != 0:
        raise ValueError("explicit UTC confirmation time required")
    verify_sources(record["evidence_sha256"])
    validate_check(test, task="TEST-01", protocol=protocol, execution=execution)
    validate_check(review, task="REVIEW-01", protocol=protocol, execution=execution)
    return record


def population_contract(eligibility, *, protocol, execution):
    """Validate the full identity denominator and recompute outcome-blind choice.

This verifies the report's coverage, scope, schema and deterministic selection;
it does not itself certify that eligibility was correctly computed from raw
inputs. DEV04's source-bound eligibility implementation and TEST/REVIEW do so.
No positions, scores, errors or arbitrary caller-supplied selection are allowed.
"""
    report = unpack(eligibility)
    _fields(report, ("schema_version", "protocol_sha256", "execution_sha256", "dataset_id",
                    "eligibility_rule_sha256", "prior_performance_reads", "rows"))
    _scoped(report, ELIGIBILITY_VERSION, protocol, execution)
    sealed = unpack(protocol)
    inputs = sealed["dataset_inputs"]
    if (report["dataset_id"] != inputs["dataset_id"]
            or report["eligibility_rule_sha256"] != digest(sealed["eligibility_contract"])
            or type(report["prior_performance_reads"]) is not int or report["prior_performance_reads"] != 0
            or not isinstance(report["rows"], list)
            or len(report["rows"]) != inputs["partitions"]["counts"]["final_eval"]):
        raise ValueError("complete pre-performance final eligibility report required")
    names = ("sample_id", "segment_id", "independent_block_id")
    identities = {k: set() for k in names}
    admitted = []
    for row in report["rows"]:
        _fields(row, (*names, "split", "eligible", "reasons", "window_sha256"))
        if row["split"] != "final_eval" or type(row["eligible"]) is not bool:
            raise ValueError("final-eval identity dispositions required")
        for name in names:
            _text(row[name])
        sha256(row["sample_id"])
        if row["sample_id"] in identities["sample_id"]:
            raise ValueError("duplicate final population sample")
        for name in names:
            identities[name].add(row[name])
        reasons = row["reasons"]
        if (not isinstance(reasons, list) or any(not isinstance(r, str) or not r.strip() or len(r) > 256 for r in reasons)
                or len(set(reasons)) != len(reasons)):
            raise ValueError("explicit unique eligibility reasons required")
        if row["eligible"]:
            if reasons:
                raise ValueError("eligible row cannot hide rejection reasons")
            sha256(row["window_sha256"])
            admitted.append({k: row[k] for k in ("sample_id", "independent_block_id", "split")})
        elif not reasons or row["window_sha256"] is not None:
            raise ValueError("excluded row must retain reasons without an admitted window")
    identity = digest({k: sorted(v) for k, v in identities.items()})
    row_identity = digest(sorted([r[k] for k in names] for r in report["rows"]))
    if (identity != inputs["partitions"]["split_identity_sha256"]["final_eval"]
            or row_identity != inputs["partitions"]["split_row_identity_sha256"]["final_eval"]):
        raise ValueError("final eligibility omitted/replaced original sample/segment/block identities")
    return {"schema_version": POPULATION_VERSION, "protocol_sha256": protocol["sha256"],
        "execution_sha256": execution["sha256"], "input_binding_sha256": inputs["sha256"],
        "eligibility_sha256": eligibility["sha256"], "full_population_identity_sha256": identity,
        "full_population_row_identity_sha256": row_identity,
        "selection": select_identity_metadata(admitted), "prior_performance_reads": 0}


def validate_population(value, eligibility, *, expected_sha256, protocol, execution):
    record = unpack(value, expected_sha256=sha256(expected_sha256))
    if canonical(record) != canonical(population_contract(eligibility, protocol=protocol, execution=execution)):
        raise ValueError("population seal differs from complete frozen eligibility/selection")
    return record


@dataclass(frozen=True)
class VerifiedAccess:
    """Only supplied to the operation after every gate and durable start record."""
    access_kind: str
    legacy_cohort_ack: str
    protocol_sha256: str
    execution_sha256: str
    approval_sha256: str
    population_sha256: str | None
    access_started_sha256: str


def guarded_call(*, access_kind, protocol, execution, approval_path, approval_sha256,
                 test_path, review_path, journal_directory, operation,
                 population_path=None, population_sha256=None, eligibility_path=None):
    """Fail before operation invocation for absent/stale/wrong-version authority.

Every invocation revalidates sources and receipts. Run this at protected loader
or phase boundaries, not once per particle. Preflight and failures are charged
by DEV04's cumulative phase ledger; this journal records access, not CPU budget.
An interrupted start remains visible and must not be interpreted as zero reads.
"""
    if access_kind not in ACCESS_KINDS or not callable(operation):
        raise ValueError("registered protected operation required")
    # Fail early on absent authority, before even reading population reports.
    sha256(approval_sha256)
    if approval_path is None or test_path is None or review_path is None:
        raise ValueError("human approval and exact test/review receipt paths required")
    approval, test, review = (read_json(p) for p in (approval_path, test_path, review_path))
    sealed = validate_protocol(protocol)
    validate_execution(execution, protocol)
    validate_approval(approval, expected_sha256=approval_sha256,
                      protocol=protocol, execution=execution, test=test, review=review)
    protected_population = None
    if access_kind != "final_eval_eligibility":
        if population_path is None or population_sha256 is None or eligibility_path is None:
            raise ValueError("population must be sealed before positions/features/metrics")
        population, eligibility = read_json(population_path), read_json(eligibility_path)
        validate_population(population, eligibility, expected_sha256=population_sha256,
                            protocol=protocol, execution=execution)
        protected_population = population["sha256"]
    elif any(x is not None for x in (population_path, population_sha256, eligibility_path)):
        raise ValueError("eligibility access cannot silently consume a supplied results population")
    attempt = uuid4().hex
    started = {"schema_version": "pirc17-final-access-event-v1", "event": "started", "attempt_id": attempt,
        "access_kind": access_kind, "protocol_sha256": protocol["sha256"], "execution_sha256": execution["sha256"],
        "approval_sha256": approval_sha256, "population_sha256": protected_population,
        "at_utc": datetime.now(timezone.utc).isoformat()}
    # Exclusive durable publication must succeed BEFORE the legacy ack exists.
    _, receipt = publish(Path(journal_directory), started)
    context = VerifiedAccess(access_kind, sealed["dataset_inputs"]["dataset_id"], protocol["sha256"],
                             execution["sha256"], approval_sha256, protected_population, receipt["sha256"])
    try:
        result = operation(context)
    except BaseException as error:
        publish(Path(journal_directory), {**started, "event": "failed", "started_sha256": receipt["sha256"],
            "at_utc": datetime.now(timezone.utc).isoformat(), "exception_type": type(error).__name__,
            "possible_reads_retained": True})
        raise
    publish(Path(journal_directory), {**started, "event": "returned", "started_sha256": receipt["sha256"],
        "at_utc": datetime.now(timezone.utc).isoformat(), "possible_reads_retained": True})
    return result
