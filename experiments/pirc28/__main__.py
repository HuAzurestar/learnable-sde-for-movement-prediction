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
    qualify = commands.add_parser('qualify', help='budgeted qualification of immutable registered sources')
    qualify.add_argument('definition_hash')
    qualify.add_argument('--study-grant', action='append', required=True, metavar='STUDY=AUTHORIZATION')
    qualify.add_argument('--allow-fixture', action='store_true')
    qualify.add_argument('--parent-attempt-id')
    qualify.add_argument('--reason')
    stage = commands.add_parser('stage-definition', help='publish immutable pre-fit metadata; never creates grants or studies')
    stage.add_argument('path', type=Path)
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
        elif args.command == 'resume':
            grant = store.authorization(args.authorization_id, version=args.authorization_version)
            output = SharedRecovery(store, registry, recovery).resume(args.attempt_id, args.checkpoint_id,
                authorization=grant, budget=budget)
        elif args.command == 'qualify':
            from .computation import QualificationRunner
            from .qualification import IdentifiabilitySpec
            value = store.manifest('pirc28-identifiability-spec-'+args.definition_hash)
            if digest(value) != args.definition_hash:
                raise ResearchError('IDENTITY_CONFLICT', 'registered qualification definition differs')
            definition = IdentifiabilitySpec.from_dict(value)
            grants = {}
            for binding in args.study_grant:
                study_id, separator, authorization_id = binding.partition('=')
                if not separator or not study_id or not authorization_id or study_id in grants:
                    raise ResearchError('CONTRACT_MISMATCH', 'unique STUDY=AUTHORIZATION scopes required')
                grants[study_id] = store.authorization(authorization_id)
            output = QualificationRunner(store).run(definition, sorted(grants), grants, budget=budget,
                allow_fixture=args.allow_fixture, parent_attempt_id=args.parent_attempt_id, reason=args.reason)
        else:
            from infrastructure.research_files import opened_regular_file
            from .study import MAXIMUM_DEFINITION_BYTES, SyntheticStudyDefinition, stage_definition
            path = args.path.absolute()
            with opened_regular_file(path.parent, path, maximum_bytes=MAXIMUM_DEFINITION_BYTES) as (stream, _, _):
                definition = SyntheticStudyDefinition(stream.read(MAXIMUM_DEFINITION_BYTES+1))
            output = stage_definition(store, definition)
        print(json.dumps(output, sort_keys=True))
        return 0
    except ResearchError as error:
        print(json.dumps({'code': error.code, 'message': str(error)}))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
