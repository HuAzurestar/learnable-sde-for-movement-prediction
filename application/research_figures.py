"""Validate frozen worker figures, never render or recalculate them here."""

import hashlib
import json
import math
import re
from xml.etree import ElementTree as ET

from infrastructure.research_store import ResearchError, digest, encode

NS = "http://www.w3.org/2000/svg"
MAX_FIGURE_BYTES = 2 * 1024 * 1024


def validate_figure_svg(content):
    try:
        if not isinstance(content, bytes) or len(content) > MAX_FIGURE_BYTES:
            raise ValueError("figure byte quota")
        source = content.decode("utf-8")
        if "<!" in source or "<?" in source:
            raise ValueError("XML declarations/entities are not allowed")
        root = ET.fromstring(source)
        if root.tag != f"{{{NS}}}svg":
            raise ValueError("figure must be an SVG")
        attributes = {"svg": {"role", "aria-label", "viewBox"}, "metadata": set(),
                      "text": {"x", "y", "font-size"}, "rect": {"x", "y", "width", "height", "fill"}}
        for i, node in enumerate(root.iter()):
            if i > 100_000 or not node.tag.startswith("{" + NS + "}"):
                raise ValueError("figure node/namespace quota")
            tag = node.tag.removeprefix("{" + NS + "}")
            if tag not in attributes or not set(node.attrib) <= attributes[tag]:
                raise ValueError("active or external SVG element/attribute")
            if node is not root and len(node):
                raise ValueError("nested figure nodes are not allowed")
            if tag == "rect" and node.get("fill") != "#287c9c":
                raise ValueError("unsupported figure paint")
            for key, value in node.attrib.items():
                if key in {"x", "y", "width", "height", "font-size", "viewBox"}:
                    numbers = value.split()
                    if len(numbers) != (4 if key == "viewBox" else 1) or any(
                            not math.isfinite(float(number)) or abs(float(number)) > 1e9 for number in numbers):
                        raise ValueError("invalid figure coordinates")
        metadata = root.findall("{" + NS + "}metadata")
        if len(metadata) != 1:
            raise ValueError("figure needs one provenance record")
        value = json.loads(metadata[0].text)
        encode(value)
        if not isinstance(value, dict) or value.get("schema_version") != "pirc25-figure-provenance-v1":
            raise ValueError("figure provenance schema")
        return value
    except (ValueError, TypeError, AttributeError, ET.ParseError, UnicodeError) as exc:
        raise ResearchError("CONTRACT_MISMATCH", "unsafe or incomplete frozen SVG") from exc


def validate_figure_package(aggregate, index, figures):
    try:
        if (not isinstance(index, dict) or not isinstance(figures, dict) or
                index.get("schema_version") != "pirc25-figure-index-v1" or
                index.get("aggregate_hash") != aggregate["aggregate_hash"] or
                index.get("compare_hash") != aggregate["adjudication"]["compare_hash"] or
                index.get("computation_ref") != aggregate["computation_ref"]):
            raise ValueError("figure index differs from aggregate")
        entries = index["figures"]
        if not isinstance(entries, list) or not 1 <= len(entries) <= 257:
            raise ValueError("figure count quota")
        horizons = {str(arm["comparison_dimensions"]["horizon"]) for arm in aggregate["arms"]
                    if "horizon" in arm["comparison_dimensions"]}
        names, selections = set(), set()
        for entry in entries:
            name, horizon = entry["filename"], entry["horizon"]
            if (not isinstance(name, str) or not re.fullmatch(r"figure-[0-9a-f]{64}\.svg", name) or
                    name in names or horizon in selections or (horizon is not None and horizon not in horizons) or
                    entry["kind"] != "comparison" or entry["media_type"] != "image/svg+xml"):
                raise ValueError("figure names/selectors changed")
            names.add(name)
            selections.add(horizon)
            content = figures[name].encode("utf-8")
            sha = hashlib.sha256(content).hexdigest()
            if entry["sha256"] != sha or name != "figure-" + sha + ".svg" or entry["size_bytes"] != len(content):
                raise ValueError("figure bytes differ from index")
            provenance = validate_figure_svg(content)
            selected = [arm for arm in aggregate["arms"] if horizon is None or
                        str(arm["comparison_dimensions"].get("horizon")) == horizon]
            if provenance != {"schema_version": "pirc25-figure-provenance-v1", "kind": "comparison",
                    "aggregate_hash": aggregate["aggregate_hash"], "spec_hash": aggregate["spec_hash"],
                    "protocol_hash": aggregate["protocol_hash"], "horizon": horizon,
                    "adjudication": aggregate["adjudication"], "computation_ref": aggregate["computation_ref"],
                    "cost_source": "resolve-computation-ref-after-settlement",
                    "strata": [{"stratum_id": arm["stratum_id"], "dimensions": arm["comparison_dimensions"],
                                "units": arm["metric_units"], "cost": arm["cost"]} for arm in selected]}:
                raise ValueError("figure provenance differs from frozen decision")
        if set(figures) != names or selections != {None, *horizons}:
            raise ValueError("missing or extra frozen figures")
        return digest(index)
    except (KeyError, TypeError, ValueError) as exc:
        raise ResearchError("CONTRACT_MISMATCH", "frozen figure package binding changed") from exc
