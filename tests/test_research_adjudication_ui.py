"""Execute the actual UI decision renderer against partial frozen policies."""

import json
from pathlib import Path
import subprocess

import pytest

from tests.test_research_adjudication_binding import policy


@pytest.mark.parametrize("missing", ["primary_metric", "minimum_seeds", "minimum_paired_blocks", "multiplicity", "seed_pairing"])
def test_partial_preregistration_renders_diagnostics_not_exceptions(missing):
    frozen = policy()
    frozen.pop(missing)
    value = {"adjudication": {"adjudication_spec": frozen, "status": "NEEDS_PREREGISTRATION",
        "qualification": "engineering-fixture", "compare_hash": "synthetic", "family_size": 0,
        "diagnostics": [missing], "records": []}}
    script = (Path(__file__).resolve().parents[1] / "experiments/pirc25/ui/app.js").read_text(encoding="utf-8")
    renderers = script[script.index("function adjudicationLines("):script.index("function frozenComparisonEntry(")]
    harness = """const vm = require('vm'); let output=[];
const context={document:{createElement:()=>({append:(...v)=>output.push(...v)})},
text:(_,value)=>String(value ?? 'Unavailable'),table:(headers,rows)=>JSON.stringify([headers,rows])};
"""
    harness += "vm.runInNewContext(" + json.dumps(renderers) + ",context);\n"
    harness += "const aggregate=" + json.dumps(value) + ";\n"
    harness += "output.push(...context.adjudicationLines(aggregate,null));context.adjudicationView(aggregate,null);console.log(output.join('\\n'));"
    process = subprocess.run(["node"], input=harness, capture_output=True, text=True, timeout=10)
    assert process.returncode == 0, process.stderr
    assert "NEEDS_PREREGISTRATION" in process.stdout and missing in process.stdout
    assert "undefined" not in process.stdout
