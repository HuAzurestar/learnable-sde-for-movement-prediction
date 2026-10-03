"""Real Edge rendering of actual recorded synthetic refusals; no route mocks."""
import json
import os
from pathlib import Path
import threading

from playwright.sync_api import sync_playwright, expect

from experiments.pirc25.web import make_server
from infrastructure.research_store import encode
from tests.research_upstream_view_fixtures import recorded_view


def main():
    root = Path(os.environ["PIRC_UPSTREAM_BROWSER_ROOT"])
    root.mkdir()
    store, spec, grant = recorded_view(root)
    server = make_server(store, grant["authorization_id"])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    receipt = {"study_id": spec["study_id"], "fixture": "actual synthetic pre-read refusal; not real upstream acceptance"}
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(channel="msedge", headless=True)
            receipt["browser"] = browser.version
            page = browser.new_page()
            page.set_default_timeout(6000)
            page.goto(f"http://127.0.0.1:{server.server_port}/#session={server.session_token}")
            page.get_by_role("button", name="Upstream inputs", exact=True).click()
            expect(page.locator("#upstream-table tbody tr")).to_have_count(2)
            expect(page.locator("#upstream-table")).to_contain_text("MISSING_ARTIFACT")
            expect(page.locator("#upstream-table")).to_contain_text("NOT_CHECKED")
            expect(page.locator("#upstream-context")).to_contain_text(spec["study_id"])
            expect(page.locator("#upstream-context")).to_contain_text("not data permission")
            assert str(root) not in page.locator("#results").inner_text()
            page.screenshot(path=str(root / "upstream-view.png"), full_page=True)
            page.get_by_role("button", name="Refresh", exact=True).click()
            expect(page.locator("#upstream-table tbody tr")).to_have_count(2)
            assert not any(event["event_kind"] in {"READ_STARTED", "READ_COMPLETED", "WORKER_STARTED"} for event in store.events())
            grant_path = store.path / "manifests" / ("authorization-" + grant["authorization_id"] + ".json")
            grant_path.write_bytes(encode({**grant, "purposes": []}))
            page.get_by_role("button", name="Refresh", exact=True).click()
            # Physically changing immutable grant content breaks its recorded
            # hash; this is actual integrity denial, not a versioned revocation.
            expect(page.locator("#error")).to_contain_text("CORRUPT_ARTIFACT")
            expect(page.locator("#upstream-table")).to_have_count(0)
            page.screenshot(path=str(root / "upstream-denied.png"), full_page=True)
            receipt["status"] = "passed"
            receipt["upstream_events"] = [event for event in store.events() if event["event_kind"].startswith("UPSTREAM_")]
            browser.close()
    except Exception as exc:
        receipt.update(status="failed", error=str(exc))
        raise
    finally:
        (root / "browser-receipt.json").write_bytes(encode(receipt))
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    print(json.dumps({"status": receipt["status"], "browser": receipt["browser"]}))


if __name__ == "__main__":
    main()
