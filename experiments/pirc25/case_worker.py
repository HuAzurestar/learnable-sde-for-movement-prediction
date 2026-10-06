"""Render saved, already-authorized paths under the shared OS supervisor."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from application.research_case_figures import case_plan, render_case_package, MAX_OUTPUT_BYTES, MAX_SOURCE_BYTES
from experiments.pirc25.affine import code_hash
from experiments.pirc25.compare_worker import _input_bytes
from infrastructure.research_store import atomic_write, digest, encode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('request', type=Path)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    request = json.loads(_input_bytes(args.request, limit=MAX_SOURCE_BYTES))
    content = _input_bytes(args.source, limit=MAX_SOURCE_BYTES)
    if len(content) > request['resource_plan']['maximum_input_bytes']:
        raise ValueError('RESOURCE_PLAN_REJECTED case source bytes')
    source = json.loads(content)
    if (request['schema_version'] != 'pirc25-case-graph-request-v1' or
            hashlib.sha256(content).hexdigest() != request['source_artifact_id'] or
            code_hash() != request['runtime_code_hash'] or
            case_plan(source, request['resource_plan']['maximum_operations']) != request['resource_plan']):
        raise ValueError('case graph source/code/allocation binding changed')
    package = render_case_package(source, request['source_artifact_id'], request['computation_ref'],
                                  request['resource_plan']['maximum_operations'])
    if code_hash() != request['runtime_code_hash']:
        raise ValueError('case implementation moved while running')
    value = {'schema_version': 'pirc25-case-graph-result-v1', 'request_hash': digest(request),
             'computation_ref': request['computation_ref'], **package}
    content = encode(value)
    if len(content) > MAX_OUTPUT_BYTES:
        raise ValueError('RESOURCE_PLAN_REJECTED case output bytes')
    atomic_write(args.output, content)


if __name__ == '__main__':
    main()
