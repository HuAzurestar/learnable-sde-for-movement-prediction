"""Reproduce shared engineering in independent processes using synthetic data.

Run with --tsde-root pointing to the checked-out paper repository. All generated
data and receipts stay in a new temporary runtime root outside either Git tree.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading

from application.research_budget import BudgetLedger, BudgetSpec
from infrastructure.research_store import ResearchStore, digest, encode, atomic_write


def reproduce(tsde_root: Path, *, browser=False):
    root = Path(tempfile.mkdtemp(prefix="shared-research-reproduction-"))
    psde_root = Path(__file__).resolve().parents[2]
    base = [sys.executable, "-m", "experiments.pirc25", "--root", str(root), "--store-id", "engineering-reproduction"]
    commands = []

    def cli(*arguments):
        command = [*base, *arguments]
        result = subprocess.run(command, cwd=psde_root, capture_output=True, text=True, timeout=180)
        commands.append({"entry": "experiments.pirc25", "arguments": list(arguments), "exit_code": result.returncode})
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        return json.loads(result.stdout)

    cli("init")
    cli("fixture", "--seeds", "19", "23")
    store = ResearchStore(root, "engineering-reproduction")
    spec = store.manifest("study-affine-fixture")["spec"]
    grant = {"authorization_id": "engineering-fixture", "study_id": spec["study_id"],
             "expires_at": (datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),
             "evidence_hash": digest("explicit synthetic engineering demonstration"),
             "purposes": ["preview", "export", "resume"], "visibilities": ["synthetic"],
             "block_ids": ["synthetic-block-1"]}
    grant_path = root / "fixture-grant.json"
    atomic_write(grant_path, encode(grant))
    cli("authorize", str(grant_path))
    first = cli("run", spec["study_id"], "--cell", digest(spec["cells"][0]))[0]
    assert first["state"] == "SUCCEEDED"

    def evidence(version):
        bundle_path = root / (version + ".json")
        exported = cli("export", spec["study_id"], "--authorization-id", grant["authorization_id"], "--output", str(bundle_path))
        output = root / version
        command = [sys.executable, str(tsde_root / "scripts/pirc25/aggregate.py"), str(bundle_path),
                   "--expected-hash", exported["bundle_hash"], "--output", str(output)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        commands.append({"entry": "scripts/pirc25/aggregate.py", "version": version, "exit_code": result.returncode})
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        aggregated = json.loads(result.stdout)
        package = cli("import-evidence", str(output), "--expected-hash", aggregated["aggregate_hash"])
        return package, json.loads((output / "aggregate.json").read_bytes())

    partial_package, partial = evidence("partial")
    assert partial["successful_cell_count"] == 1 and partial["expected_cell_count"] == 2
    assert partial["arms"][0]["status"] == "incomplete"
    # Simulated interrupted work preserves an actual authoritative charge. The
    # independent CLI retry must append a new attempt, not rewrite that history.
    second_run = store.register_run(spec["study_id"], spec["cells"][1])
    parent = store.new_attempt(second_run)
    reservation = BudgetLedger(store).reserve(parent, BudgetSpec(1))
    store.transition(parent, "RUNNING")
    BudgetLedger(store).settle(reservation["reservation_id"], 123, outcome="FAILED")
    store.transition(parent, "FAILED", error_code="INJECTED_TRANSIENT_FIXTURE")
    before = BudgetLedger(store).balance("affine-4")["committed_ms"]
    retry = cli("run", spec["study_id"], "--cell", digest(spec["cells"][1]),
                "--parent-attempt", parent, "--reason", "explicit synthetic recovery drill")[0]
    assert retry["state"] == "SUCCEEDED"
    assert BudgetLedger(store).balance("affine-4")["committed_ms"] > before
    complete_package, complete = evidence("complete")
    assert complete["successful_cell_count"] == 2 and complete["arms"][0]["independent_n"] == 1
    cli("rebuild-index")
    assert cli("list")["items"][0]["manifest"]["spec_hash"] == digest(spec)
    assert cli("run", spec["study_id"])[0]["reused"]
    browser_result = "not-requested"
    if browser:
        from playwright.sync_api import sync_playwright
        from .web import make_server
        server = make_server(store, grant["authorization_id"])
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with sync_playwright() as playwright:
                browser_instance = playwright.chromium.launch(channel="msedge", headless=True)
                context = browser_instance.new_context(viewport={"width": 1280, "height": 900})
                page = context.new_page()
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(f"http://127.0.0.1:{server.server_port}/#session={server.session_token}")
                page.get_by_role("button", name="Comparisons & evidence").click()
                page.get_by_role("button", name="Compare & export").first.wait_for()
                target = page.get_by_role("row").filter(has_text=complete["aggregate_hash"])
                target.get_by_role("button", name="Compare & export").click()
                page.get_by_role("button", name="Download frozen CSV").wait_for()
                page.screenshot(path=str(root / "results-ui.png"), full_page=True)
                with page.expect_download() as downloaded:
                    page.get_by_role("button", name="Download frozen CSV").click()
                assert Path(downloaded.value.path()).read_bytes() == (root / "complete/metrics.csv").read_bytes()
                assert not errors, errors
                context.close()
                browser_instance.close()
                browser_result = "passed"
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
    refs = {name: subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
            for name, path in (("PSDE", psde_root), ("TSDE", tsde_root))}
    receipt = {"schema_version": "pirc25-engineering-reproduction-v1", "refs": refs,
               "spec_hash": digest(spec), "protocol_hash": spec["protocol_hash"], "code_hash": spec["code_hash"],
               "fixture_hash": spec["data_hash"], "partial_aggregate_hash": partial["aggregate_hash"],
               "complete_aggregate_hash": complete["aggregate_hash"], "expected_cells": 2, "successful_cells": 2,
               "independent_blocks": 1, "retained_failed_attempt": parent, "browser": browser_result,
               "scientific_qualification": "not-granted", "commands": commands}
    atomic_write(root / "reproduction.json", encode(receipt))
    return {"receipt": str(root / "reproduction.json"), "aggregate_hash": complete["aggregate_hash"], "browser": browser_result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tsde-root", required=True, type=Path)
    parser.add_argument("--browser", action="store_true")
    args = parser.parse_args()
    print(json.dumps(reproduce(args.tsde_root.resolve(), browser=args.browser)))


if __name__ == "__main__":
    main()
