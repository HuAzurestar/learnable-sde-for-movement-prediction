"""Explicit pilot-only geometry preparation under the original reference arm."""

from dataclasses import replace

from application.research_contracts import ExecutionPlugin
from domain.errors import DataValidationError
from domain.probability_calibration import AffineHalfspaceCalibrationPolicy
from infrastructure.research_store import ResearchError
from experiments.pirc27.plugin import execution_config, propagation_command
from experiments.pirc27.qualification_plugin import qualification_plugin


PLUGIN_ID = "affine-halfspace-calibration"


def calibration_plugin():
    # Reuse conservative rational/serialization pools, not another ledger.
    # Counted reference arithmetic is not an integration grid or sampled paths.
    base = qualification_plugin()
    schema = base.registry_entry.manifest()["config_schema"]
    del schema["properties"]["qualification_policy_hash"]
    schema["required"].remove("qualification_policy_hash")
    schema["properties"]["method"]["enum"] = ["exact"]
    schema["properties"]["calibration_policy_hash"] = {"type": "string", "minLength": 64, "maxLength": 64}
    schema["properties"]["work_steps"] = {"type": "integer", "minimum": 1, "maximum": 200000}
    schema["required"] += ["calibration_policy_hash", "work_steps"]
    contract = base.registry_entry.manifest()["resource_contract"]
    contract["counts"]["paths"] = {"constant": 0}
    contract["counts"]["steps"] = {"config": ["work_steps"]}
    contract["limits"]["steps"] = 200000
    capabilities = frozenset({"exact-transition"})
    entry = replace(base.registry_entry, component_id=PLUGIN_ID, version="1.0.0",
        config_schema=schema, resource_contract=contract, capabilities=capabilities)
    return ExecutionPlugin(PLUGIN_ID, capabilities, base.state_order, base.units,
        "restart-only", propagation_command, entry)


def calibration_config(request, policy):
    from domain.propagation import PropagationRequest
    from experiments.pirc25.affine import code_hash
    if type(request) is not PropagationRequest or type(policy) is not AffineHalfspaceCalibrationPolicy:
        raise ResearchError("UNQUALIFIED", "explicit frozen probability calibration request/policy required")
    policy.manifest()  # Scalar guards before copying/hashing caller inputs.
    request.validate()
    if (policy.source_request_hash != request.request_hash
            or policy.model_package_hash != request.model_package_hash or policy.code_hash != code_hash()):
        raise ResearchError("CONTRACT_MISMATCH", "probability calibration configuration binding differs")
    return {**execution_config(request, "exact"), "calibration_policy_hash": policy.policy_hash,
        "work_steps": policy.maximum_operations}


def calibration_policy(spec, cell, package, request, config):
    from experiments.pirc25.affine import code_hash
    from infrastructure.research_admission_selection import select_admission_package
    try:
        policy = AffineHalfspaceCalibrationPolicy.from_manifest(cell["probability_calibration_policy"])
        policy.validate(package, request, code_hash())
        settings, _ = select_admission_package(spec, cell)
        arms = [arm for arm in spec["arms"] if arm["arm_id"] == request.arm_id]
        if (settings.get("mode") != "pilot" or cell.get("execution_role") != "probability-calibration"
                or len(arms) != 1 or arms[0]["method_family_id"] != "exact"
                or arms[0]["objective_id"] != "endpoint-halfspace"
                or config != calibration_config(request, policy)):
            raise ValueError("calibration pilot/reference-arm/role/config differs")
        return policy
    except (KeyError, TypeError, ValueError, DataValidationError) as exc:
        raise ResearchError("UNQUALIFIED", "frozen probability calibration pilot and original exact reference arm required") from exc
