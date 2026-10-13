"""Independent pilot-only MLMC reference work in the existing shared arm."""

from dataclasses import replace

from application.research_contracts import ExecutionPlugin
from domain.affine_mlmc_qualification import AffineMLMCReferencePolicy
from domain.errors import DataValidationError
from infrastructure.research_store import ResearchError
from experiments.pirc27.plugin import propagation_command, propagation_plugin, execution_config


PLUGIN_ID = "affine-mlmc-qualification-chunk"


def mlmc_qualification_recovery_plugin():
    from application.research_recovery import RecoveryPlugin
    from experiments.pirc27.plugin import propagation_resume_command
    plugin = mlmc_qualification_plugin()
    return RecoveryPlugin(plugin.plugin_id, "chunk", propagation_resume_command, plugin.registry_entry.version)


def mlmc_qualification_plugin():
    base = propagation_plugin(recovery=True)
    entry = base.registry_entry.manifest()
    config = entry["config_schema"]
    config["properties"]["method"]["enum"] = ["mlmc-pilot"]
    config["properties"]["reference_policy_hash"] = {"type": "string", "minLength": 64, "maxLength": 64}
    config["required"] += ["reference_policy_hash"]
    contract = entry["resource_contract"]
    # Same fixed64MiB Fraction/serialization pool as the analytic reference
    # producer. Counts stay physical4; no free owner-side certificate replay.
    for prefix in ("rational-reference", "certificate-serialization"):
        for index in range(32):
            contract["tensors"].append({"name": prefix+"-pool-"+str(index), "axes": ["state_dim"]*8, "item_bytes": 16})
    registered = replace(base.registry_entry, component_id=PLUGIN_ID, version="1.0.0",
        config_schema=config, resource_contract=contract)
    return ExecutionPlugin(PLUGIN_ID, base.capabilities, base.state_order, base.units,
        "chunk", propagation_command, registered)


def mlmc_qualification_config(request, reference_policy):
    from experiments.pirc25.affine import code_hash
    if (type(reference_policy) is not AffineMLMCReferencePolicy
            or reference_policy.request_hash != request.request_hash or reference_policy.code_hash != code_hash()):
        raise ResearchError("CONTRACT_MISMATCH", "bound MLMC reference configuration required")
    return {**execution_config(request, "mlmc-pilot", level_samples=reference_policy.level_samples, recovery=True),
        "reference_policy_hash": reference_policy.policy_hash}


def mlmc_reference_policy(spec, cell, package, request, config):
    from application.propagation_execution import pilot_policy
    from experiments.pirc25.affine import code_hash
    try:
        reference = AffineMLMCReferencePolicy.from_manifest(cell["affine_mlmc_reference_policy"])
        pilot = pilot_policy(spec, cell, request, config)
        reference.validate(package, request, pilot, code_hash())
        if config != mlmc_qualification_config(request, reference):
            raise ValueError("reference config differs")
        return reference
    except (KeyError, TypeError, ValueError, DataValidationError) as exc:
        raise ResearchError("UNQUALIFIED", "explicit independent MLMC pilot/reference policies required") from exc
