"""Real Edge refresh through an actual audit-fsync failure; no route mocks."""
import json
import os
from pathlib import Path
import subprocess

import pytest
from playwright.sync_api import expect, sync_playwright

from application.research_data import EvaluationExposureLedger
from infrastructure.research_store import encode
from tests.research_audit_fixtures import audit_fault, payload_opens
from tests.test_research_data_access_view import access_service


def refs():
    return {name: {"head": subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip(),
                   "status": subprocess.check_output(["git", "-C", str(path), "status", "--porcelain"], text=True).strip()}
            for name, path in {"runtime": Path.cwd(), "paper": Path.cwd().parent / "TSDE-SDE"}.items()}


def main():
    root = Path(os.environ["PIRC_AUDIT_BROWSER_ROOT"])
    root.mkdir()
    receipt = {"repositories": refs(), "fixture": "explicit synthetic; filesystem-call injection, not exhausted disk",
               "scientific_qualification": "not-granted", "cases": []}
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(channel="msedge", headless=True)
            receipt["browser"] = browser.version
            for exposed in (False, True):
                case_root = root / ("exposed" if exposed else "unread")
                case_root.mkdir()
                with access_service(case_root) as (store, _, grant, _, server):
                    if exposed:
                        assert EvaluationExposureLedger(store).read("inputs", "fixture-1", purpose="evaluate",
                            authorization_id=grant["authorization_id"], data_root=case_root) == b"explicit synthetic plugin input"
                    original_reads = [e for e in store.events() if e["event_kind"].startswith("READ_")]
                    page = browser.new_page(viewport={"width": 1600, "height": 1000})
                    page.set_default_timeout(6000)
                    responses = []
                    page.on("response", lambda response: responses.append(response) if "/api/data-access" in response.url else None)
                    with page.expect_response(lambda r: "/api/runs?" in r.url) as initial_response:
                        page.goto(f"http://127.0.0.1:{server.server_port}/#session={server.session_token}")
                    assert initial_response.value.status == 200
                    page.get_by_role("button", name="Data access & exposure", exact=True).click()
                    table = page.locator("#data-access-table")
                    expect(table.locator("tbody tr")).to_have_count(2)
                    expect(table).to_contain_text("EXPOSED" if exposed else "NO_RECORDED_EXPOSURE")
                    changes = pytest.MonkeyPatch()
                    with payload_opens([case_root / "plugin-input.bin"]) as opened, audit_fault(store, "DISCLOSURE_ALLOWED", changes) as fired:
                        with page.expect_response(lambda r: "/api/data-access?" in r.url) as refresh_response:
                            page.get_by_role("button", name="Refresh", exact=True).click()
                        expect(page.locator("#error")).to_contain_text("EXPOSURE_AUDIT_UNAVAILABLE")
                        expect(table).to_have_count(0)
                    observation = {"fired": fired, "opened": opened,
                                   "responses": [{"url": r.url, "status": r.status} for r in responses]}
                    (case_root / "observation.json").write_bytes(encode(observation))
                    page.screenshot(path=str(case_root / "audit-unavailable.png"), full_page=True)
                    assert fired and not opened and refresh_response.value.status == 503, observation
                    body = refresh_response.value.json()
                    assert body["error"]["code"] == "EXPOSURE_AUDIT_UNAVAILABLE" and body["error"]["trace_id"]
                    assert str(case_root) not in json.dumps(body) and "PRIVATE" not in page.locator("#error").inner_text()
                    assert [e for e in store.events() if e["event_kind"].startswith("READ_")] == original_reads
                    page.screenshot(path=str(case_root / "audit-unavailable.png"), full_page=True)
                    # An explicit user refresh after fault removal, not automatic
                    # retry/permission caching. Original exposure must reappear.
                    page.get_by_role("button", name="Refresh", exact=True).click()
                    expect(page.locator("#data-access-table tbody tr")).to_have_count(2)
                    expect(page.locator("#data-access-table")).to_contain_text("EXPOSED" if exposed else "NO_RECORDED_EXPOSURE")
                    assert [e for e in store.events() if e["event_kind"].startswith("READ_")] == original_reads
                    receipt["cases"].append({"exposed": exposed, "fired": fired, "status": 503,
                                             "error": body["error"], "original_reads": original_reads})
                    page.close()
            browser.close()
        assert refs() == receipt["repositories"], "source refs changed during browser check"
        receipt["status"] = "passed"
    except Exception as exc:
        receipt.update(status="failed", error=str(exc))
        raise
    finally:
        (root / "receipt.json").write_bytes(encode(receipt))
    print(json.dumps({"status": receipt["status"], "browser": receipt["browser"], "root": str(root)}))


if __name__ == "__main__":
    main()
