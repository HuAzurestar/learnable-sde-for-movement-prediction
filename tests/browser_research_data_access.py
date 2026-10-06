"""Real local Edge purpose/scope/exposure UI, actual reads, no route mocks."""
import json
import os
from pathlib import Path

from playwright.sync_api import sync_playwright, expect

from application.research_data import EvaluationExposureLedger
from infrastructure.research_store import encode
from tests.test_research_data_access_view import access_service


def main():
    root = Path(os.environ["PIRC_DATA_ACCESS_BROWSER_ROOT"])
    root.mkdir()
    receipt = {"fixture": "actual synthetic metadata and provider read; no protected data/scientific qualification"}
    try:
        with access_service(root) as (store, spec, grant, preview, server):
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(channel="msedge", headless=True)
                receipt["browser"] = browser.version
                page = browser.new_page(viewport={"width": 1600, "height": 1000})
                page.set_default_timeout(6000)
                page.goto(f"http://127.0.0.1:{server.server_port}/#session={server.session_token}")
                page.get_by_role("button", name="Data access & exposure", exact=True).click()
                table = page.locator("#data-access-table")
                expect(table.locator("tbody tr")).to_have_count(2)
                receipt["rendered_headers"] = table.locator("th").all_inner_texts()
                receipt["table_accessibility"] = table.aria_snapshot()
                page.screenshot(path=str(root / "data-access-headers.png"), full_page=True)
                for name in ("Purpose", "Authorization", "Exposure"):
                    expect(table.get_by_role("columnheader", name=name, exact=True)).to_have_count(1)
                expect(table).to_contain_text("final-eval")
                expect(table).to_contain_text("SCOPE_CHECK_PASSED")
                expect(table).to_contain_text("NO_RECORDED_EXPOSURE")
                expect(page.locator("#data-access-context")).to_contain_text("not proof of a blind test")
                assert not any(e["event_kind"].startswith("READ_") for e in store.events())
                page.screenshot(path=str(root / "data-access-unread.png"), full_page=True)
                assert EvaluationExposureLedger(store).read("inputs", "fixture-1", purpose="evaluate",
                    authorization_id=grant["authorization_id"], data_root=root) == b"explicit synthetic plugin input"
                reads = [e for e in store.events() if e["event_kind"].startswith("READ_")]
                page.get_by_role("button", name="Refresh", exact=True).click()
                expect(table).to_contain_text("EXPOSED")
                expect(table).to_contain_text(reads[0]["hash"])
                assert "NO_RECORDED_EXPOSURE" not in table.inner_text()
                assert str(root) not in page.locator("#results").inner_text()
                assert [e for e in store.events() if e["event_kind"].startswith("READ_")] == reads
                page.screenshot(path=str(root / "data-access-exposed.png"), full_page=True)
                # Physical immutable-grant corruption is a hard integrity refusal,
                # not a legitimate versioned revocation. Failed refresh clears rows.
                path = store.path / "manifests" / ("authorization-" + preview["authorization_id"] + ".json")
                path.write_bytes(encode({**preview, "purposes": []}))
                page.get_by_role("button", name="Refresh", exact=True).click()
                expect(page.locator("#error")).to_contain_text("CORRUPT_ARTIFACT")
                expect(table).to_have_count(0)
                page.screenshot(path=str(root / "data-access-denied.png"), full_page=True)
                receipt.update(status="passed", study_id=spec["study_id"], original_reads=reads)
                denied_root = root / "scope-denied"
                denied_root.mkdir()
                with access_service(denied_root, version="v1", grant_change={"test_authorization": False}) as (denied_store, _, _, _, denied_server):
                    denied_page = browser.new_page(viewport={"width": 1600, "height": 1000})
                    denied_page.set_default_timeout(6000)
                    denied_page.goto(f"http://127.0.0.1:{denied_server.server_port}/#session={denied_server.session_token}")
                    denied_page.get_by_role("button", name="Data access & exposure", exact=True).click()
                    denied_table = denied_page.locator("#data-access-table")
                    expect(denied_table.locator("tbody tr")).to_have_count(2)
                    expect(denied_table).to_contain_text("SCOPE_CHECK_DENIED")
                    expect(denied_table).to_contain_text("UNAUTHORIZED_DATA")
                    expect(denied_table).to_contain_text("NO_RECORDED_EXPOSURE")
                    assert not any(e["event_kind"].startswith("READ_") for e in denied_store.events())
                    denied_page.screenshot(path=str(root / "data-access-scope-denied.png"), full_page=True)
                    receipt["test_scope_denial"] = "actual versioned grant lacks test authorization; metadata preview separately permitted"
                    denied_page.close()
                browser.close()
    except Exception as exc:
        receipt.update(status="failed", error=str(exc))
        raise
    finally:
        (root / "browser-receipt.json").write_bytes(encode(receipt))
    print(json.dumps({"status": receipt["status"], "browser": receipt["browser"]}))


if __name__ == "__main__":
    main()
