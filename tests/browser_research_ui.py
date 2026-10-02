"""Explicit synthetic browser acceptance; run with an isolated Playwright env.

python -B -m tests.browser_research_ui
No worker/scientific qualification is claimed by this display fixture.
"""

import csv
import io
import json
import platform
from pathlib import Path
import subprocess
import sys
import tempfile
import threading

from application.research_budget import BudgetLedger, BudgetSpec
from application.research_evidence import export_evidence, accept_evidence_package
from experiments.pirc25.web import make_server
from infrastructure.research_store import ResearchStore, digest, encode


def fixture(root, *, failure_states=False):
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
    for index, cell in enumerate(spec["cells"]):
        if failure_states and index == 1:
            continue
        run = store.register_run(spec["study_id"], cell)
        attempt = store.new_attempt(run)
        ledger = BudgetLedger(store)
        reservation = ledger.reserve(attempt, BudgetSpec(10))
        store.transition(attempt, "RUNNING")
        if failure_states and index in (2, 3):
            state = "FAILED" if index == 2 else "TIMEOUT"
            ledger.settle(reservation["reservation_id"], 100, outcome=state)
            store.transition(attempt, state, error_code=state)
            continue
        result = {"spec_hash": digest(spec), "cell_hash": digest(cell), "protocol_hash": spec["protocol_hash"],
            "state_order": ["x", "y"], "units": ["m", "m"], "time_unit": "s", "qualification": "fixture",
            "metrics": {"error": cell["horizon"] + (1 if cell["arm_id"] == "baseline" else 0),
                        "calibration_coverage90_fixture_v1": 0.875 if cell["arm_id"] == "baseline" else 0.75},
            "metric_units": {"error": "m", "calibration_coverage90_fixture_v1": "fraction"}, "forecast": {"horizons": [1, 2],
                "preview": {"case_selection_rule": "registered-synthetic-case", "sample_ids": ["sample-0", "sample-1"],
                            "generation_version": "ui-fixture-v1", "n_samples": 2},
                "samples": [[[0, 0], [1, 2]], [[0, 1], [2, 3]]]}}
        result["forecast"].update({
            "moments": {"estimation_kind": "sample-estimate", "mean": [0.5, 1.5]},
            "region_estimates": [{"region": "synthetic-region", "probability": 0.75,
                                  "estimator": "weighted-samples", "standard_error": 0.1, "ess": 1.6}],
            "mode_probabilities": {"labels": ["left", "right"], "probabilities": [0.25, 0.75],
                                   "role": "predictive", "K": 2},
            "uncertainty": {"kind": "synthetic-interval", "level": 0.9, "bounds": [0.1, 0.9]}})
        if cell["arm_id"] == "candidate" and cell["horizon"] == 2 and cell["seed"] == 2:
            result["forecast"]["samples"] *= 33  # 66 saved paths: preview must refuse, not truncate.
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


def check_failure_states(browser, root, expect):
    store, spec, aggregate, package = fixture(root / "states", failure_states=True)
    server = make_server(store, "browser")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        page = browser.new_page()
        page.set_default_timeout(6000)
        page.goto(f"http://127.0.0.1:{server.server_port}/#session={server.session_token}")
        expect(page.locator("#results tbody tr")).to_have_count(8)
        for state in ("MISSING", "FAILED", "TIMEOUT"):
            page.locator("#filter-state").fill(state)
            page.get_by_role("button", name="Apply filters", exact=True).click()
            expect(page.locator("#results tbody tr")).to_have_count(1)
            expect(page.locator("#results")).to_contain_text(state)
            if state != "MISSING":
                page.get_by_role("button", name="Inspect", exact=True).click()
                expect(page.locator("#detail-content")).to_contain_text(state)
        page.locator("#filter-state").fill("RUNNING")
        page.get_by_role("button", name="Apply filters", exact=True).click()
        expect(page.locator("#results")).to_contain_text("No runs match the current filters")
        page.get_by_role("button", name="Reset filters", exact=True).click()
        expect(page.locator("#results tbody tr")).to_have_count(8)
        page.get_by_role("button", name="Comparisons & evidence", exact=True).click()
        page.get_by_role("button", name="Compare & export", exact=True).click()
        expect(page.locator("#comparison-table")).to_contain_text("No complete metric")
        expect(page.locator("#detail-content")).to_contain_text("TIMEOUT")
        expect(page.locator("#detail-content")).to_contain_text("insufficient-independent-blocks")
        expect(page.locator("#comparison-status-rates")).to_contain_text("TIMEOUT")
        expect(page.locator("#comparison-status-rates")).to_contain_text("0.5")
        for arm in aggregate["arms"]:
            rates = arm["status_rates"]
            assert rates["denominator"] == 2
            assert rates["values"] == {state: count / 2 for state, count in arm["dispositions"].items()}
        # Corrupt only this disposable synthetic test artifact, never user data.
        (store.path / "artifacts" / package["aggregate_id"]).write_bytes(b"corrupt synthetic fixture")
        page.get_by_role("button", name="Compare & export", exact=True).click()
        expect(page.locator("#error")).to_be_visible()
        expect(page.locator("#error")).to_contain_text("CORRUPT_ARTIFACT")
        page.screenshot(path=str(root / "failure-states.png"), full_page=True)
        page.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def check_managed_adjudication(browser, root, expect):
    """Actual budgeted worker output, not a handcrafted browser verdict."""
    import xml.etree.ElementTree as ET
    from application.research_query import ResearchQuery
    from tests.test_research_comparison import source, compute
    managed = source.__wrapped__(root / "managed")
    store = managed[0]
    result = compute(managed)
    assert result["state"] == "SUCCEEDED"
    data = ResearchQuery(store, "viewer").comparison(result["comparison"]["aggregate_hash"])
    server = make_server(store, "viewer")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        page = browser.new_page()
        page.set_default_timeout(6000)
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(f"http://127.0.0.1:{server.server_port}/#session={server.session_token}")
        expect(page.locator("#results tbody tr")).to_have_count(8)
        page.get_by_role("button", name="Comparisons & evidence", exact=True).click()
        row = page.locator("#results tbody tr").filter(has_text=result["comparison"]["aggregate_hash"])
        row.get_by_role("button", name="Compare & export", exact=True).click()
        section = page.locator("#comparison-adjudication")
        for declared in ("GAIN", "engineering-fixture", "bonferroni", "independent-within-block", "NO_GAIN is not equivalence"):
            expect(section).to_contain_text(declared)
        verdict_row = section.locator("table").nth(1).locator("tbody tr")
        expect(verdict_row).to_have_count(1)
        expect(verdict_row.locator("td").nth(2)).to_have_text("1")
        expect(verdict_row.locator("td").nth(4)).to_have_text("1 to 1")
        expect(verdict_row.locator("td").nth(6)).to_have_text("2")
        expect(section).to_contain_text(str(data["computation_receipt"]["cost"]["charged_ms"]))
        frozen_image = page.locator('#comparison-figure img')
        expect(frozen_image).to_be_visible()
        expect(frozen_image).to_have_attribute('data-artifact-id', data['package']['figures'][0]['artifact_id'])
        page.wait_for_function("document.querySelector('#comparison-figure img')?.naturalWidth > 0")
        expect(page.locator('#comparison-figure svg')).to_have_count(0)
        decision_before = section.inner_text()
        page.locator("#comparison-horizon").select_option("all")
        assert section.inner_text() == decision_before
        with page.expect_download() as downloaded:
            page.get_by_role("button", name="Download frozen CSV", exact=True).click()
        rows = list(csv.DictReader(io.StringIO(Path(downloaded.value.path()).read_text(encoding="utf-8"))))
        assert all(json.loads(row["adjudication"]) == data["aggregate"]["adjudication"] for row in rows)
        with page.expect_download() as downloaded:
            page.get_by_role("button", name="Download computation receipt", exact=True).click()
        assert json.loads(Path(downloaded.value.path()).read_bytes()) == data["computation_receipt"]
        with page.expect_download() as downloaded:
            page.get_by_role("button", name="Export comparison figure", exact=True).click()
        content = Path(downloaded.value.path()).read_bytes()
        entry = data['package']['figures'][0]
        assert content == store.read_artifact(entry['artifact_id'], purpose='export', authorization=managed[2])
        assert downloaded.value.suggested_filename == entry['filename']
        figure = ET.fromstring(content)
        metadata = json.loads(figure.find("{http://www.w3.org/2000/svg}metadata").text)
        assert metadata["adjudication"] == data["aggregate"]["adjudication"]
        assert metadata["computation_ref"] == data["aggregate"]["computation_ref"]
        assert metadata['cost_source'] == 'resolve-computation-ref-after-settlement'
        assert "benefit 1.0 m" in "".join(figure.itertext()) or "benefit 1 m" in "".join(figure.itertext())
        assert not errors, errors
        page.screenshot(path=str(root / "managed-adjudication.png"), full_page=True)
        page.close()
        return {"aggregate_hash": data["aggregate"]["aggregate_hash"],
                "compare_hash": data["aggregate"]["adjudication"]["compare_hash"],
                "computation_receipt": data["computation_receipt"], "qualification": "engineering-fixture"}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def main():
    from playwright.sync_api import sync_playwright, expect
    root = Path(tempfile.mkdtemp(prefix="pirc38-ui-acceptance-"))
    repo_root = Path(__file__).resolve().parents[1]
    def refs():
        return {name: {"head": subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip(),
                       "status": subprocess.check_output(["git", "-C", str(path), "status", "--porcelain"], text=True).strip()}
                for name, path in (("PSDE", repo_root), ("TSDE", repo_root.parent / "TSDE-SDE"))}
    start_refs = refs()
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
            evidence_button = page.get_by_role("button", name="Open evidence", exact=False)
            for _ in range(2):
                with page.expect_response(lambda response: '/api/comparisons/' in response.url) as opened:
                    evidence_button.click()
                assert opened.value.status == 200
                # Wait for the response handler, not merely the pre-existing
                # first control; an immediate count could miss a late duplicate.
                page.wait_for_function("document.querySelector('[data-evidence-open-count]')?.dataset.evidenceOpenCount === '" + str(_ + 1) + "'")
                expect(page.locator("#comparison-horizon")).to_have_count(1)
            page.locator("#comparison-horizon").select_option("2")
            expect(page.locator("#comparison-table tbody tr")).to_have_count(4)
            page.get_by_role("button", name="Inspect result", exact=False).click()
            expect(page.locator("#preview-provenance")).to_contain_text("registered-synthetic-case")
            expect(page.locator("#preview-provenance")).to_contain_text("sample-0")
            expect(page.locator("#preview-provenance")).to_contain_text("ui-fixture-v1")
            expect(page.locator("#optional-payloads")).to_contain_text("Density: unavailable")
            expect(page.locator("#optional-payloads")).to_contain_text("Evaluation truth: unavailable")
            for saved in ("sample-estimate", "weighted-samples", "standard_error", "ess", "predictive", "synthetic-interval"):
                expect(page.locator("#optional-payloads")).to_contain_text(saved)
            expect(page.locator("#optional-payloads")).not_to_contain_text("Analytic moments")
            page.locator("#case-horizon").select_option("1")
            expect(page.locator("#case-chart")).to_have_attribute("data-horizon", "2")
            with page.expect_download() as downloaded:
                page.get_by_role("button", name="Download result manifest", exact=True).click()
            manifest = json.loads(Path(downloaded.value.path()).read_bytes())
            assert manifest["bindings"]["spec_hash"] == digest(spec)
            assert "forecast" not in manifest and "samples" not in json.dumps(manifest)
            with page.expect_download() as downloaded:
                page.get_by_role("button", name="Export case figure", exact=True).click()
            content = Path(downloaded.value.path()).read_text(encoding="utf-8")
            assert '"horizon":2' in content and '"artifact_id"' in content and digest(spec) in content
            page.locator("#filter-model").fill("candidate")
            page.locator("#filter-seed").fill("2")
            page.get_by_role("button", name="Apply filters", exact=True).click()
            expect(page.locator("#results tbody tr")).to_have_count(1)
            expect(page.locator("#results")).to_contain_text("candidate")
            page.get_by_role("button", name="Inspect", exact=True).click()
            page.get_by_role("button", name="Inspect result", exact=False).click()
            expect(page.locator("#error")).to_contain_text("TOO_LARGE")
            expect(page.locator("#error")).to_contain_text("64 trajectories")
            expect(page.locator("#case-chart")).to_have_count(0)
            page.get_by_role("button", name="Comparisons & evidence", exact=True).click()
            page.get_by_role("button", name="Compare & export", exact=True).click()
            page.locator("#comparison-horizon").select_option("2")
            expect(page.locator("#comparison-table tbody tr")).to_have_count(4)
            calibration_rows = page.locator("#comparison-table tbody tr").filter(has_text="calibration_coverage90_fixture_v1")
            expect(calibration_rows).to_have_count(2)
            expect(calibration_rows.filter(has_text="baseline")).to_contain_text("0.875")
            expect(calibration_rows.filter(has_text="candidate")).to_contain_text("0.75")
            expect(page.locator("#comparison-status-rates")).to_contain_text("registered-cells-in-arm-stratum")
            expect(page.locator("#comparison-table")).to_contain_text("3")
            expect(page.locator("#comparison-costs tbody tr")).to_have_count(2)
            expect(page.locator("#comparison-costs")).to_contain_text("200")
            assert all(arm["cost"]["charged_ms"] == 200 for arm in aggregate["arms"])
            with page.expect_download() as downloaded:
                page.get_by_role("button", name="Export comparison figure", exact=True).click()
            content = Path(downloaded.value.path()).read_text(encoding="utf-8")
            assert aggregate["aggregate_hash"] in content and package["aggregate_id"] in content
            assert '"horizon":2' in content
            with page.expect_download() as downloaded:
                page.get_by_role("button", name="Download frozen CSV", exact=True).click()
            assert Path(downloaded.value.path()).read_bytes() == (root / "evidence/metrics.csv").read_bytes()
            csv_rows = list(csv.DictReader(io.StringIO(Path(downloaded.value.path()).read_text(encoding="utf-8"))))
            for arm in aggregate["arms"]:
                row = next(row for row in csv_rows if row["arm_id"] == arm["arm_id"] and row["stratum_id"] == arm["stratum_id"] and row["metric"] == "calibration_coverage90_fixture_v1")
                assert float(row["value"]) == arm["metrics"]["calibration_coverage90_fixture_v1"]
                assert row["unit"] == "fraction"
                assert json.loads(row["status_rates"]) == arm["status_rates"]
            with page.expect_download() as downloaded:
                page.get_by_role("button", name="Download aggregate manifest", exact=True).click()
            manifest = json.loads(Path(downloaded.value.path()).read_bytes())
            assert manifest["artifact"]["artifact_id"] == package["aggregate_id"]
            assert manifest["bindings"]["aggregate_hash"] == aggregate["aggregate_hash"]
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
            check_failure_states(browser, root, expect)
            managed_adjudication = check_managed_adjudication(browser, root, expect)
            browser_version = browser.version
            browser.close()
        end_refs = refs()
        assert start_refs == end_refs, "Source refs changed during browser acceptance"
        receipt = {"status": "passed", "spec_hash": digest(spec), "aggregate_hash": aggregate["aggregate_hash"],
                   "managed_adjudication": managed_adjudication,
                   "scientific_qualification": "not-granted", "root": str(root), "repositories": end_refs,
                   "python": platform.python_version(), "browser": browser_version}
        (root / "receipt.json").write_bytes(encode(receipt))
        print(json.dumps(receipt))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


if __name__ == "__main__":
    main()
