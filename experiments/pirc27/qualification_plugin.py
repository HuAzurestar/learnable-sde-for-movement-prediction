"""Explicit pilot-only bounded analytic qualification job, not a new arm."""

from dataclasses import replace

from application.research_contracts import ExecutionPlugin
from infrastructure.research_store import ResearchError
from experiments.pirc27.plugin import propagation_command, propagation_plugin, execution_config


PLUGIN_ID = "affine-propagation-qualification"


def qualification_plugin():
    base = propagation_plugin()
    config = base.registry_entry.manifest()["config_schema"]
    config["properties"]["method"]["enum"] = ["exact", "gaussian"]
    config["properties"]["qualification_policy_hash"] = {"type": "string", "minLength": 64, "maxLength": 64}
    config["required"] += ["qualification_policy_hash"]
    contract = base.registry_entry.manifest()["resource_contract"]
    # Two pooled32MiB allowances for bounded Fraction objects/intermediates
    # and JSON certificates. Registry counts are not model/path dimensions.
    contract["counts"]["observations"] = {"constant": 256}
    for prefix in ("rational-reference", "certificate-serialization"):
        for index in range(32):
            contract["tensors"].append({"name": prefix+"-pool-"+str(index),
                "axes": ["state_dim"]*8, "item_bytes": 16})
    entry = replace(base.registry_entry, component_id=PLUGIN_ID, version="1.0.0",
        config_schema=config, resource_contract=contract)
    return ExecutionPlugin(PLUGIN_ID, base.capabilities, base.state_order, base.units,
        "restart-only", propagation_command, entry)


def qualification_config(request, method, policy):
    from experiments.pirc25.affine import code_hash
    from domain.affine_qualification import AffineQualificationPolicy
    if type(policy) is not AffineQualificationPolicy or method not in {"exact", "gaussian"}:
        raise ResearchError("UNQUALIFIED", "explicit analytic qualification method/policy required")
    # Model binding is checked against the actual frozen package at admission.
    if policy.request_hash != request.request_hash or policy.code_hash != code_hash() or policy.method != method:
        raise ResearchError("CONTRACT_MISMATCH", "qualification configuration binding differs")
    return {**execution_config(request, method), "qualification_policy_hash": policy.policy_hash}


def qualification_policy(spec, cell, package, request, config):
    from domain.affine_qualification import AffineQualificationPolicy
    from domain.errors import DataValidationError
    from experiments.pirc25.affine import code_hash
    from infrastructure.research_admission_selection import select_admission_package
    try:
        policy = AffineQualificationPolicy.from_manifest(cell["affine_qualification_policy"])
        policy.validate(package, request, config["method"], code_hash())
        settings, _ = select_admission_package(spec, cell)
        if (settings.get("mode") != "pilot" or cell.get("execution_role") != "qualification"
                or config != qualification_config(request, config["method"], policy)):
            raise ValueError("qualification mode/role/config differs")
        return policy
    except (KeyError, TypeError, ValueError, DataValidationError) as exc:
        raise ResearchError("UNQUALIFIED", "frozen analytic qualification policy and pilot role required") from exc
