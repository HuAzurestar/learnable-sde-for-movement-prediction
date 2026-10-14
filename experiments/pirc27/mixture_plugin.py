"""Explicit bounded deterministic mixture adapter; never auto-qualified."""

from dataclasses import replace

from application.research_contracts import ExecutionPlugin
from domain.errors import DataValidationError
from domain.mixture import MixturePolicy
from infrastructure.research_store import ResearchError
from .plugin import propagation_command, propagation_plugin, propagation_resume_command


PLUGIN_IDS = frozenset({"affine-mixture-chunk", "synthetic-mixture-chunk"})


def mixture_plugin(*, synthetic=False):
    base = propagation_plugin(recovery=True, synthetic=synthetic)
    entry = base.registry_entry.manifest()
    config = entry["config_schema"]
    config["properties"]["method"]["enum"] = ["mixture"]
    config["properties"]["mixture_policy_hash"] = {"type": "string", "minLength": 64, "maxLength": 64}
    config["properties"]["component_cap"] = {"type": "integer", "minimum": 1, "maximum": 32}
    config["properties"]["candidate_cap"] = {"type": "integer", "minimum": 8, "maximum": 256}
    config["required"] += ["mixture_policy_hash", "component_cap", "candidate_cap"]
    contract = entry["resource_contract"]
    contract["counts"].update(paths={"constant": 1}, components={"config": ["component_cap"]},
        observations={"config": ["candidate_cap"]})
    contract["limits"].update(components=32, observations=256)
    contract["tensors"] += [
        {"name": "mixture-candidate-covariances", "axes": ["observations", "state_dim", "state_dim"], "item_bytes": 8},
        {"name": "mixture-live-covariances", "axes": ["components", "state_dim", "state_dim"], "item_bytes": 8},
        # Conservative bounded allowance for Python objects, lineage/canonical
        # checkpoint serialization and eigensolver/reduction temporaries.
        {"name": "mixture-object-state-pool", "axes": ["state_dim"]*8, "item_bytes": 16},
        {"name": "mixture-reduction-state-pool", "axes": ["state_dim"]*8, "item_bytes": 16}]
    identity = "synthetic-mixture-chunk" if synthetic else "affine-mixture-chunk"
    registered = replace(base.registry_entry, component_id=identity, version="1.0.0",
        config_schema=config, resource_contract=contract, capabilities=frozenset({"generic-rollout"}))
    return ExecutionPlugin(identity, registered.capabilities, base.state_order, base.units,
        "chunk", propagation_command, registered)


def mixture_recovery_plugin(*, synthetic=False):
    from application.research_recovery import RecoveryPlugin
    plugin = mixture_plugin(synthetic=synthetic)
    return RecoveryPlugin(plugin.plugin_id, "chunk", propagation_resume_command, "1.0.0")


def mixture_config(request, policy, *, synthetic=False):
    request.validate()
    config = {"method": "mixture", "samples": request.samples, "base_steps": request.steps,
        "steps": request.steps, "chunk_size": request.chunk_size, "level_samples": [], "proposal": [0., 0.],
        "work_steps": request.steps*policy.work_per_step, "mixture_policy_hash": policy.policy_hash,
        "component_cap": policy.component_cap, "candidate_cap": 8*policy.component_cap}
    if synthetic:
        config["workspace_rows"] = 8*policy.component_cap
    return config


def mixture_policy(spec, cell, package, request, config):
    from experiments.pirc25.affine import code_hash
    from infrastructure.research_admission_selection import select_admission_package
    try:
        policy = MixturePolicy.from_manifest(cell["mixture_policy"])
        policy.validate(package, request, code_hash())
        settings, _ = select_admission_package(spec, cell)
        arms = [arm for arm in spec["arms"] if arm["arm_id"] == request.arm_id]
        if settings.get("mode") == "formal":
            raise ValueError("no dedicated mixture qualifier")
        if (len(arms) != 1 or arms[0]["method_family_id"] != "mixture"
                or config != mixture_config(request, policy, synthetic=cell["plugin_id"].startswith("synthetic-"))):
            raise ValueError("original mixture arm/config differs")
        return policy
    except (KeyError, TypeError, ValueError, DataValidationError) as exc:
        raise ResearchError("UNQUALIFIED", "explicit frozen original-arm mixture policy required; formal mixture unqualified") from exc
