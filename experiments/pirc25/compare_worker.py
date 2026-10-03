"""Compute one frozen TSDE comparison inside the shared OS worker boundary."""

import argparse
import json
import os
from pathlib import Path
import stat
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from application.research_computation import paper_identity, comparison_plan, MAX_INPUT_BYTES, MAX_OUTPUT_BYTES
from experiments.pirc25.affine import code_hash
from infrastructure.research_store import atomic_write, digest, encode


def _input_bytes(path, *, limit=None):
    # The owner's stat preflight is not an allocation bound: a file can grow
    # after it. Check the opened object and cap the actual read before parsing.
    limit = MAX_INPUT_BYTES if limit is None else limit
    if type(limit) is not int or not 0 < limit <= MAX_INPUT_BYTES:
        raise ValueError('RESOURCE_PLAN_REJECTED input quota')
    with path.open('rb') as stream:
        information = os.fstat(stream.fileno())
        if not stat.S_ISREG(information.st_mode) or information.st_size > limit:
            raise ValueError('RESOURCE_PLAN_REJECTED input bytes')
        content = stream.read(limit + 1)
    if len(content) > limit:
        raise ValueError('RESOURCE_PLAN_REJECTED input bytes')
    return content


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("paper_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.request.stat().st_size > MAX_INPUT_BYTES or args.bundle.stat().st_size > MAX_INPUT_BYTES:
        raise ValueError("RESOURCE_PLAN_REJECTED input bytes")
    request = json.loads(_input_bytes(args.request))
    bundle = json.loads(_input_bytes(args.bundle))
    if (bundle["bundle_hash"] != request["computation_ref"]["source_bundle_hash"] or
            digest({k: v for k, v in bundle.items() if k != "bundle_hash"}) != bundle["bundle_hash"] or
            code_hash() != request["runtime_code_hash"] or paper_identity(args.paper_root) != request["paper_identity"]):
        raise ValueError("computation code/source binding changed")
    if comparison_plan(bundle, request["resource_plan"]["maximum_operations"]) != request["resource_plan"]:
        raise ValueError("RESOURCE_PLAN_REJECTED computation allocation differs from frozen source")
    sys.path.insert(0, str(args.paper_root.resolve()))
    from scripts.pirc25.comparison import compare_package
    package = compare_package(bundle, request["computation_ref"], formal=request["formal"],
                              max_operations=request["resource_plan"]["maximum_operations"])
    if code_hash() != request["runtime_code_hash"] or paper_identity(args.paper_root) != request["paper_identity"]:
        raise ValueError("computation implementation moved while running")
    result = {"schema_version": "pirc25-computation-result-v1", "request_hash": digest(request),
              "computation_ref": request["computation_ref"], **package}
    content = encode(result)
    if len(content) > MAX_OUTPUT_BYTES:
        raise ValueError("RESOURCE_PLAN_REJECTED output bytes")
    atomic_write(args.output, content)


if __name__ == "__main__":
    main()
