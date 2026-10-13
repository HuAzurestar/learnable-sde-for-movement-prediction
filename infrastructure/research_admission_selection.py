"""Pure bounded selection from an immutable complete cell-to-package table.

No package/grant creation, data reads, protocol/mode override or default lookup.
Older single-package studies remain explicit legacy inputs, not table fallbacks.
"""

import json
import re

from .research_disposition import declared_execution_disposition, require_executable_cell
from .research_store import ResearchError, digest, encode, identifier


MODEL_FIELDS = frozenset({"model_authorization_id", "model_authorization_version", "model_protocol_id"})


def _hash(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def cell_package_bindings(spec):
    """Return detached exact-coverage bindings, or None for the legacy schema."""
    settings = spec.get("admission") or {}
    if type(settings) is not dict:
        raise ResearchError("CONTRACT_MISMATCH", "admission settings must be an object")
    if "cell_packages" not in settings:
        return None
    if "package_hash" in settings or set(settings) & MODEL_FIELDS:
        raise ResearchError("CONTRACT_MISMATCH", "cell package table cannot coexist with default package/model bindings")
    table = settings["cell_packages"]
    if (type(table) is not dict or set(table) != {"schema_version", "bindings"}
            or table["schema_version"] != "pirc25-cell-packages-v1"
            or type(table["bindings"]) is not list or len(table["bindings"]) > 10000
            or type(spec.get("cells")) is not list or not 0 < len(spec["cells"]) <= 10000
            or any(type(cell) is not dict for cell in spec["cells"])):
        raise ResearchError("CONTRACT_MISMATCH", "invalid bounded cell package table/matrix")
    cells = {digest(cell): cell for cell in spec["cells"]}
    if len(cells) != len(spec["cells"]):
        raise ResearchError("CONTRACT_MISMATCH", "cell package table needs unique registered cells")
    expected = {key for key, cell in cells.items() if declared_execution_disposition(cell) is None}
    bindings = {}
    for binding in table["bindings"]:
        if (type(binding) is not dict or not {"cell_hash", "package_hash"} <= set(binding)
                or set(binding) - {"cell_hash", "package_hash"} - MODEL_FIELDS
                or not _hash(binding["cell_hash"]) or not _hash(binding["package_hash"])
                or binding["cell_hash"] in bindings):
            raise ResearchError("CONTRACT_MISMATCH", "invalid or duplicate cell package binding")
        for field in MODEL_FIELDS & set(binding):
            identifier(binding[field])
        if "model_authorization_version" in binding and "model_authorization_id" not in binding:
            raise ResearchError("CONTRACT_MISMATCH", "model authorization version has no bound identity")
        bindings[binding["cell_hash"]] = dict(binding)
    if set(bindings) != expected:
        raise ResearchError("CONTRACT_MISMATCH", "package bindings must cover exactly all executable registered cells")
    if len(encode(table)) > 4*1024*1024:
        raise ResearchError("TOO_LARGE", "cell package table exceeds the bounded metadata quota")
    return bindings


def admission_package_references(spec):
    """Include every declared source in conservative study visibility."""
    bindings = cell_package_bindings(spec)
    if bindings is None:
        reference = (spec.get("admission") or {}).get("package_hash")
        return (reference,) if reference else ()
    return tuple(sorted({binding["package_hash"] for binding in bindings.values()}))


def select_admission_package(spec, cell):
    """Keep common authority settings; select only package/model-source refs."""
    if type(spec.get("cells")) is not list or cell not in spec["cells"]:
        raise ResearchError("CONTRACT_MISMATCH", "admission cell is not in the registered matrix")
    require_executable_cell(cell)
    settings = spec.get("admission") or {}
    bindings = cell_package_bindings(spec)
    if bindings is None:
        return json.loads(encode(settings)), None
    key = digest(cell)
    binding = bindings[key]
    common = {name: value for name, value in settings.items() if name != "cell_packages"}
    common.update({name: value for name, value in binding.items() if name != "cell_hash"})
    selection = {"schema_version": "pirc25-cell-package-selection-v1", "cell_hash": key,
                 "package_hash": binding["package_hash"], "binding_hash": digest(binding),
                 "table_hash": digest(settings["cell_packages"])}
    return json.loads(encode(common)), selection


def verify_admission_selection(spec, cell, receipt):
    """Check the bound selection in run/resume/reuse/export; never qualify it."""
    settings, expected = select_admission_package(spec, cell)
    if receipt.get("admission_selection") != expected:
        raise ResearchError("UNQUALIFIED", "admission package selection differs from the frozen cell table")
    if expected is not None:
        package = receipt.get("documents", {}).get("package")
        if (type(package) is not dict or digest(package) != settings["package_hash"]
                or receipt.get("mode") != settings.get("mode")):
            raise ResearchError("UNQUALIFIED", "admitted package/mode differs from the frozen cell selection")
    return settings
