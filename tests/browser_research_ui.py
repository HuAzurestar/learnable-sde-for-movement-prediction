"""Explicit synthetic browser acceptance; run with an isolated Playwright env.

python -B -m tests.browser_research_ui
No worker/scientific qualification is claimed by this display fixture.
"""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading

from application.research_budget import BudgetLedger, BudgetSpec
from application.research_evidence import export_evidence, accept_evidence_package
from experiments.pirc25.web import make_server
from infrastructure.research_store import ResearchStore, digest, encode


def fixture(root):
    store = ResearchStore(root / "runtime", "browser", initialize=True)
    spec = {"schema_version": "pirc25-contract-v1", "study_id": "browser-fixture",
        "experiment_id": "display-only", "comparison_family": "synthetic",
        **{key: digest(key) for key in ("code_hash", "data_hash", "feature_hash", "selection_hash", "protocol_hash")},
        "arms": [{"arm_id": arm, "model_family_id": arm, "method_family_id": "fixture",
                  "trainer_id": "fixture-trainer", "objective_id": "error", "budget_seconds": 86400}
                 for arm in ("baseline", "candidate")],
        "cells": [{"arm_id": arm, "block_id": "block-1", "seed": seed, "horizon": horizon,
                   "plugin_id": "fixture-predictor", "visibility": "synthetic"}
                  for arm in ("baseline", "candidate") for horizon in (1, 2) for seed in (1, 2)]}
    store.register(spec, digest(spec))
    grant = {"authorization_id": "browser", "study_id": spec["study_id"], "expires_at": "2099-01-01T00:00:00+00:00",
        "evidence_hash": digest("synthetic browser permission"), "purposes": ["preview", "export"],
        "visibilities": ["synthetic"], "block_ids": ["block-1"]}
    store.authorize(grant)
    for cell in spec["cells"]:
        run = store.register_run(spec["study_id"], cell)
        attempt = store.new_attempt(run)
        ledger = BudgetLedger(store)
        reservation = ledger.reserve(attempt, BudgetSpec(10))
        store.transition(attempt, "RUNNING")
        result = {"spec_hash": digest(spec), "cell_hash": digest(cell), "protocol_hash": spec["protocol_hash"],
            "state_order": ["x", "y"], "units": ["m", "m"], "time_unit": "s", "qualification": "fixture",
            "metrics": {"error": cell["horizon"] + (1 if cell["arm_id"] == "baseline" else 0)},
            "metric_units": {"error": "m"}, "forecast": {"horizons": [1, 2],
                "samples": [[[0, 0], [1, 2]], [[0, 1], [2, 3]]]}}
        artifact = store.artifact(encode(result), role="result", visibility="synthetic", block_ids=["block-1"], study_id=spec["study_id"])
        ledger.settle(reservation["reservation_id"], 100, outcome="SUCCEEDED")
        store.transition(attempt, "SUCCEEDED", artifact_id=artifact["artifact_id"])
    bundle = export_evidence(store, spec["study_id"], grant)
    path = root / "bundle.json"
    path.write_bytes(encode(bundle))
    aggregator = Path(__file__).resolve().parents[2] / "TSDE-SDE/scripts/pirc25/aggregate.py"
    subprocess.run([sys.executable, "-B", str(aggregator), str(path), "--expected-hash", bundle["bundle_hash"],
                    "--output", str(root / "evidence")], check=True, capture_output=True)
    aggregate = json.loads((root / "evidence/aggregate.json").read_bytes())
    package = accept_evidence_package(store, root / "evidence", aggregate["aggregate_hash"])
    return store, spec, aggregate, package


def main():
    from playwright.sync_api import sync_playwright, expect
    root = Path(tempfile.mkdtemp(prefix="pirc38-ui-acceptance-"))
    store, spec, aggregate, package = fixture(root)
    server = make_server(store, "browser")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(channel="msedge", headless=True)
            page = browser.new_page(viewport={"width": 1400, "height": 1000})
            page.set_default_timeout(6000)
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(f"http://127.0.0.1:{server.server_port}/#session={server.session_token}")
            expect(page.locator("#results tbody tr")).to_have_count(8)
            for key, value in (("model", "baseline"), ("version", digest(spec)), ("horizon", "2"),
                               ("seed", "1"), ("state", "SUCCEEDED"), ("trainer", "fixture-trainer"),
                               ("predictor", "fixture-predictor")):
                page.locator("#filter-" + key).fill(value)
            page.get_by_role("button", name="Apply filters", exact=True).click()
            expect(page.locator("#results tbody tr")).to_have_count(1)
            expect(page.locator("#results")).to_contain_text("baseline")
            expect(page.locator("#results")).to_contain_text("slot-ms")
            page.get_by_role("button", name="Inspect", exact=True).click()
            expect(page.locator("#detail-content")).to_contain_text("measured-monotonic")
            page.get_by_role("button", name="Inspect result", exact=False).click()
            page.locator("#case-horizon").select_option("1")
            expect(page.locator("#case-chart")).to_have_attribute("data-horizon", "2")
            with page.expect_download() as downloaded:
                page.get_by_role("button", name="Export case figure", exact=True).click()
            content = Path(downloaded.value.path()).read_text(encoding="utf-8")
            assert '"horizon":2' in content and '"artifact_id"' in content and digest(spec) in content
            page.get_by_role("button", name="Comparisons & evidence", exact=True).click()
            page.get_by_role("button", name="Compare & export", exact=True).click()
            page.locator("#comparison-horizon").select_option("2")
            expect(page.locator("#comparison-table tbody tr")).to_have_count(2)
            expect(page.locator("#comparison-table")).to_contain_text("3")
            with page.expect_download() as downloaded:
                page.get_by_role("button", name="Export comparison figure", exact=True).click()
            content = Path(downloaded.value.path()).read_text(encoding="utf-8")
            assert aggregate["aggregate_hash"] in content and package["aggregate_id"] in content
            assert '"horizon":2' in content
            with page.expect_download() as downloaded:
                page.get_by_role("button", name="Download frozen CSV", exact=True).click()
            assert Path(downloaded.value.path()).read_bytes() == (root / "evidence/metrics.csv").read_bytes()
            page.screenshot(path=str(root / "ui.png"), full_page=True)
            # A preview-only session can inspect the same data, but exporting a
            # client-rendered figure must still pass a fresh server export gate.
            preview = {**store.manifest("authorization-browser"), "authorization_id": "preview-only", "purposes": ["preview"]}
            store.authorize(preview)
            restricted_server = make_server(store, "preview-only")
            restricted_thread = threading.Thread(target=restricted_server.serve_forever, daemon=True)
            restricted_thread.start()
            try:
                denied_page = browser.new_page()
                downloads = []
                denied_page.on("download", lambda download: downloads.append(download))
                denied_page.goto(f"http://127.0.0.1:{restricted_server.server_port}/#session={restricted_server.session_token}")
                denied_page.get_by_role("button", name="Comparisons & evidence", exact=True).click()
                denied_page.get_by_role("button", name="Compare & export", exact=True).click()
                with denied_page.expect_response(lambda response: 'download=1' in response.url) as response:
                    denied_page.get_by_role("button", name="Export comparison figure", exact=True).click()
                assert response.value.status == 403
                expect(denied_page.locator('#error')).to_be_visible()
                assert not downloads
                denied_page.close()
            finally:
                restricted_server.shutdown()
                restricted_server.server_close()
                restricted_thread.join(timeout=5)
            assert not errors, errors
            browser.close()
        receipt = {"status": "passed", "spec_hash": digest(spec), "aggregate_hash": aggregate["aggregate_hash"],
                   "scientific_qualification": "not-granted", "root": str(root)}
        (root / "receipt.json").write_bytes(encode(receipt))
        print(json.dumps(receipt))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


if __name__ == "__main__":
    main()
