"""Local shared research CLI. Explicit store identity prevents silent resets."""

import argparse
import json
import os
from pathlib import Path

from application.research_budget import BudgetLedger, BudgetSpec
from infrastructure.research_index import ResearchIndex
from infrastructure.research_store import ResearchStore, ResearchError, digest, identifier
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
    show = commands.add_parser("show", help="registration metadata only; use authorized export/serve for results")
    show.add_argument("object_id")
    run = commands.add_parser("run")
    run.add_argument("study")
    run.add_argument("--seconds", type=float, default=60)
    run.add_argument("--cell")
    run.add_argument("--parent-attempt")
    run.add_argument("--reason")
    budget = commands.add_parser("budget")
    budget.add_argument("arm")
    recovery = commands.add_parser("recover-unknown")
    recovery.add_argument("reservation_id")
    recovery.add_argument("--stop-evidence-hash", help="hash of verified whole-tree stop evidence for launched work")
    quarantine = commands.add_parser("quarantine-tail")
    quarantine.add_argument("--reason", required=True)
    authorization = commands.add_parser("authorize")
    authorization.add_argument("grant", type=Path)
    export = commands.add_parser("export")
    export.add_argument("study")
    export.add_argument("--authorization-id", required=True)
    export.add_argument("--output", type=Path, required=True)
    evidence = commands.add_parser("import-evidence")
    evidence.add_argument("directory", type=Path)
    evidence.add_argument("--expected-hash", required=True)
    comparison = commands.add_parser("compare", help="supervised statistical comparison of a frozen aggregate")
    comparison.add_argument("study")
    comparison.add_argument("aggregate_hash")
    comparison.add_argument("--paper-root", type=Path, required=True)
    comparison.add_argument("--authorization-id", required=True)
    comparison.add_argument("--seconds", type=float, default=7200)
    comparison.add_argument("--max-operations", type=int, default=20_000_000)
    comparison.add_argument("--formal", action="store_true", default=None)
    comparison.add_argument("--parent-attempt")
    comparison.add_argument("--reason")
    comparison.add_argument("--output", type=Path)
    server = commands.add_parser("serve")
    server.add_argument("--authorization-id", required=True)
    server.add_argument("--port", type=int, default=0)
    args = parser.parse_args(argv)
    if not args.root:
        parser.error("--root or SDE_RUNTIME_ROOT is required; no implicit store")
    try:
        store = ResearchStore(Path(args.root), args.store_id, initialize=args.command == "init",
                              allow_corrupt=args.command == "quarantine-tail")
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
            # Fail closed for bundles and any future result-bearing manifest
            # kind. A past export grant is not a grant for this invocation.
            kind = identifier(args.object_id).split("-", 1)[0]
            if kind not in {"study", "run", "artifact"}:
                raise ResearchError("UNAUTHORIZED_DATA", "show is limited to registration metadata; use authorized export or serve for results")
            result = store.manifest(args.object_id)
        elif args.command == "budget":
            result = BudgetLedger(store).balance(args.arm)
        elif args.command == "recover-unknown":
            result = BudgetLedger(store).recover_unknown(args.reservation_id, stop_evidence_hash=args.stop_evidence_hash)
        elif args.command == "quarantine-tail":
            result = store.quarantine_tail(args.reason)
        elif args.command == "authorize":
            grant = json.loads(args.grant.read_text(encoding="utf-8"))
            store.authorize(grant)
            result = {"authorization_id": grant["authorization_id"]}
        elif args.command == "export":
            from application.research_evidence import export_evidence
            from infrastructure.research_store import atomic_write, encode
            grant = store.manifest("authorization-" + args.authorization_id)
            bundle = export_evidence(store, args.study, grant)
            output = args.output.resolve()
            if any((parent / ".git").exists() for parent in (output.parent, *output.parents)):
                raise ResearchError("UNAUTHORIZED_DATA", "export must stay outside Git")
            content = encode(bundle)
            if output.exists() and output.read_bytes() != content:
                raise ResearchError("IDENTITY_CONFLICT", "export exists with different content")
            output.parent.mkdir(parents=True, exist_ok=True)
            atomic_write(output, content)
            result = {"bundle_hash": bundle["bundle_hash"]}
        elif args.command == "import-evidence":
            from application.research_evidence import accept_evidence_package
            result = accept_evidence_package(store, args.directory, args.expected_hash)
        elif args.command == "compare":
            from application.research_comparison import ComparisonRunner
            result = ComparisonRunner(store, args.paper_root).run(args.study, args.aggregate_hash,
                authorization_id=args.authorization_id, budget=BudgetSpec(args.seconds),
                max_operations=args.max_operations, formal=args.formal,
                parent_attempt_id=args.parent_attempt, reason=args.reason, output=args.output)
        elif args.command == "serve":
            from .web import serve
            serve(store, args.authorization_id, args.port)
            return 0
        else:
            spec = store.manifest("study-" + args.study)["spec"]
            cells = [c for c in spec["cells"] if args.cell is None or digest(c) == args.cell]
            if not cells:
                raise ResearchError("MISSING_INPUT", "requested cell is not registered")
            result = [SharedRunner(store).run_cell(args.study, digest(cell),
                      budget=BudgetSpec(args.seconds, category="smoke"),
                      parent_attempt_id=args.parent_attempt, reason=args.reason) for cell in cells]
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        failed = (any(r.get("exit_code") for r in result) if isinstance(result, list)
                  else bool(result.get("exit_code")) if isinstance(result, dict) else False)
        return 1 if failed else 0
    except (ResearchError, OSError, ValueError) as error:
        payload = error.envelope() if isinstance(error, ResearchError) else {"code": "RUNTIME_ERROR", "retryable": False}
        print(json.dumps({"error": payload}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
