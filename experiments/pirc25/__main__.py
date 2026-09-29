"""Local shared research CLI. Explicit store identity prevents silent resets."""

import argparse
import json
import os
from pathlib import Path

from application.research_budget import BudgetLedger, BudgetSpec
from infrastructure.research_index import ResearchIndex
from infrastructure.research_store import ResearchStore, ResearchError, digest
from .affine import fixture_spec
from .runner import SharedRunner


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=os.environ.get("SDE_RUNTIME_ROOT"))
    parser.add_argument("--store-id", required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init")
    fixture = commands.add_parser("fixture")
    fixture.add_argument("--study", default="affine-fixture")
    fixture.add_argument("--dimensions", type=int, choices=[1, 4], default=4)
    fixture.add_argument("--seeds", type=int, nargs="+", default=[19])
    register = commands.add_parser("register")
    register.add_argument("spec", type=Path)
    register.add_argument("--expected-hash", required=True)
    listing = commands.add_parser("list")
    listing.add_argument("--kind", default="study", choices=["study", "run", "artifact"])
    listing.add_argument("--limit", type=int, default=50)
    commands.add_parser("rebuild-index")
    show = commands.add_parser("show")
    show.add_argument("object_id")
    run = commands.add_parser("run")
    run.add_argument("study")
    run.add_argument("--seconds", type=float, default=60)
    run.add_argument("--cell")
    budget = commands.add_parser("budget")
    budget.add_argument("arm")
    recovery = commands.add_parser("recover-unknown")
    recovery.add_argument("reservation_id")
    authorization = commands.add_parser("authorize")
    authorization.add_argument("grant", type=Path)
    args = parser.parse_args(argv)
    if not args.root:
        parser.error("--root or SDE_RUNTIME_ROOT is required; no implicit store")
    try:
        store = ResearchStore(Path(args.root), args.store_id, initialize=args.command == "init")
        if args.command == "init":
            result = {"store_id": store.store_id, "schema_version": store.SCHEMA}
        elif args.command == "fixture":
            spec = fixture_spec(args.study, args.dimensions, tuple(args.seeds))
            result = store.register(spec, digest(spec))
        elif args.command == "register":
            result = store.register(json.loads(args.spec.read_text(encoding="utf-8")), args.expected_hash)
        elif args.command == "rebuild-index":
            result = {"watermark": ResearchIndex(store).rebuild()}
        elif args.command == "list":
            result = ResearchIndex(store).list(kind=args.kind, limit=args.limit)
        elif args.command == "show":
            result = store.manifest(args.object_id)
        elif args.command == "budget":
            result = BudgetLedger(store).balance(args.arm)
        elif args.command == "recover-unknown":
            result = BudgetLedger(store).recover_unknown(args.reservation_id)
        elif args.command == "authorize":
            grant = json.loads(args.grant.read_text(encoding="utf-8"))
            store.authorize(grant)
            result = {"authorization_id": grant["authorization_id"]}
        else:
            spec = store.manifest("study-" + args.study)["spec"]
            cells = [c for c in spec["cells"] if args.cell is None or digest(c) == args.cell]
            if not cells:
                raise ResearchError("MISSING_INPUT", "requested cell is not registered")
            result = [SharedRunner(store).run_cell(args.study, digest(cell),
                      budget=BudgetSpec(args.seconds, category="smoke")) for cell in cells]
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 1 if isinstance(result, list) and any(r.get("exit_code") for r in result) else 0
    except (ResearchError, OSError, ValueError) as error:
        payload = error.envelope() if isinstance(error, ResearchError) else {"code": "RUNTIME_ERROR", "retryable": False}
        print(json.dumps({"error": payload}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
