"""Execute registered switching studies in an existing shared runtime store."""

import argparse
import json
import os
from pathlib import Path

from application.research_budget import BudgetSpec
from application.research_recovery import SharedRecovery
from experiments.pirc25.runner import SharedRunner
from infrastructure.research_store import ResearchError, ResearchStore, digest
from .runtime import PLUGIN_ID, registries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=os.environ.get('SDE_RUNTIME_ROOT'))
    parser.add_argument('--store-id', required=True)
    parser.add_argument('--seconds', type=float, default=900)
    parser.add_argument('--stage', choices=('smoke', 'pilot', 'formal'), default='smoke')
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('run')
    run.add_argument('study_id')
    resume = commands.add_parser('resume')
    resume.add_argument('attempt_id')
    resume.add_argument('checkpoint_id')
    resume.add_argument('--authorization-id', required=True)
    resume.add_argument('--authorization-version')
    args = parser.parse_args(argv)
    if args.root is None:
        parser.error('explicit shared --root or SDE_RUNTIME_ROOT is required; no implicit new ledger')
    try:
        store = ResearchStore(Path(args.root), args.store_id)
        registry, recovery = registries()
        budget = BudgetSpec(args.seconds, category='job' if args.stage == 'formal' else args.stage)
        budget.validate()
        if args.command == 'run':
            spec = store.manifest('study-' + args.study_id)['spec']
            if any(cell.get('plugin_id') != PLUGIN_ID for cell in spec['cells']):
                raise ResearchError('CONTRACT_MISMATCH', 'study contains a different registered adapter')
            output = [SharedRunner(store, registry, recovery).run_cell(args.study_id, digest(cell), budget=budget)
                      for cell in spec['cells']]
        else:
            grant = store.authorization(args.authorization_id, version=args.authorization_version)
            output = SharedRecovery(store, registry, recovery).resume(args.attempt_id, args.checkpoint_id,
                authorization=grant, budget=budget)
        print(json.dumps(output, sort_keys=True))
        return 0
    except ResearchError as error:
        print(json.dumps({'code': error.code, 'message': str(error)}))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
