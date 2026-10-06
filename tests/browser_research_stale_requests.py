"""Real Edge/server stale loads: forward original promises, responses and I/O."""
import json
import os
from pathlib import Path
import threading

import pytest
from playwright.sync_api import expect, sync_playwright

from application.research_query import ResearchQuery
from infrastructure.research_store import encode
from tests.browser_research_audit_unavailable import refs
from tests.research_audit_fixtures import audit_fault, payload_opens
from tests.test_research_data_access_view import access_service


def check(browser, root, phase, observations):
    started, released = threading.Event(), threading.Event()
    with access_service(root) as (store, _, _, _, server):
        page = browser.new_page(viewport={'width': 1600, 'height': 1000})
        page.set_default_timeout(6000)
        responses = []
        page.on('response', lambda response: responses.append(response)
                if '/api/' in response.url else None)
        try:
            with page.expect_response(lambda response: '/api/runs?' in response.url) as initial:
                page.goto(f'http://127.0.0.1:{server.server_port}/#session={server.session_token}')
            assert initial.value.status == 200
            # Observe the SAME original promise from real user button calls.
            # Awaiting allSettled later also lets their already-attached
            # original catch(showError) finish. No response/result is replaced.
            page.evaluate('''() => {
                const original = load;
                globalThis.__observedLoads = [];
                load = (...args) => {
                    const operation = original(...args);
                    globalThis.__observedLoads.push(operation);
                    return operation;
                };
            }''')
            responses.clear()
            original = ResearchQuery.list
            delayed_kind = 'run' if phase == 'obsolete-error' else 'study'

            def delayed(query, kind, *args, **kwargs):
                if kind == delayed_kind and not started.is_set():
                    started.set()
                    assert released.wait(6), 'original obsolete query was not released'
                return original(query, kind, *args, **kwargs)

            changes = pytest.MonkeyPatch()
            try:
                changes.setattr(ResearchQuery, 'list', delayed)
                page.get_by_role('button', name='Refresh', exact=True).click()
                assert started.wait(6), 'actual original server request did not arrive'
                with page.expect_response(lambda response: '/api/data-access?' in response.url) as current:
                    page.get_by_role('button', name='Data access & exposure', exact=True).click()
                assert current.value.status == 200
                expect(page.locator('#data-access-table tbody tr')).to_have_count(2)
                original_reads = [event for event in store.events()
                                  if event['event_kind'].startswith('READ_')]
                fired, body = [], None
                with payload_opens([root / 'plugin-input.bin']) as opened:
                    if phase == 'obsolete-error':
                        # Current data-access is already complete. Only the
                        # delayed original run request can consume this real
                        # event staging fsync fault and actual server 503.
                        with audit_fault(store, 'DISCLOSURE_ALLOWED', changes) as fired:
                            with page.expect_response(lambda response: '/api/runs?' in response.url) as obsolete:
                                released.set()
                            page.evaluate('Promise.allSettled(globalThis.__observedLoads)')
                            assert obsolete.value.status == 503 and fired
                            body = obsolete.value.json()
                            assert body['error']['code'] == 'EXPOSURE_AUDIT_UNAVAILABLE'
                    else:
                        released.set()
                        page.evaluate('Promise.allSettled(globalThis.__observedLoads)')
                observation = {'phase': phase, 'fired': list(fired), 'payload_opens': list(opened),
                    'responses': [{'url': response.url, 'status': response.status} for response in responses],
                    'obsolete_error': body,
                    'page': page.evaluate('''() => ({view, loadSequence,
                        errorHidden: el('error').hidden, errorText: el('error').textContent,
                        summary: el('summary').textContent,
                        currentRows: document.querySelectorAll('#data-access-table tbody tr').length})''')}
                observations.append(observation)
                (root / 'observation.json').write_bytes(encode(observation))
                page.screenshot(path=str(root / 'stale-request.png'), full_page=True)
                assert not opened
                assert [event for event in store.events() if event['event_kind'].startswith('READ_')] == original_reads
                expect(page.locator('#data-access-table tbody tr')).to_have_count(2)
                expect(page.locator('#error')).to_be_hidden()
                expect(page.locator('#summary')).to_have_text('Purpose / authorization / exposure · metadata only')
                if phase == 'obsolete-continuation':
                    assert not [response for response in responses if '/api/runs?' in response.url], observation
            finally:
                released.set()
                changes.undo()
        finally:
            page.close()


def main():
    root = Path(os.environ['PIRC_STALE_BROWSER_ROOT'])
    root.mkdir()
    receipt = {'repositories': refs(), 'fixture': 'explicit synthetic; real server/request gates and event fsync',
        'scientific_qualification': 'not-granted', 'page_timeout_ms': 6000,
        'expect_timeout_ms': 5000, 'cases': [], 'failures': []}
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(channel='msedge', headless=True)
            receipt['browser'] = browser.version
            for phase in ('obsolete-error', 'obsolete-continuation'):
                case_root = root / phase
                case_root.mkdir()
                try:
                    check(browser, case_root, phase, receipt['cases'])
                except Exception as error:
                    receipt['failures'].append({'phase': phase, 'error': str(error)})
            browser.close()
        assert refs() == receipt['repositories'], 'source refs changed during stale-request checks'
        receipt['status'] = 'failed' if receipt['failures'] else 'passed'
        assert not receipt['failures'], receipt['failures']
    except Exception:
        receipt['status'] = 'failed'
        raise
    finally:
        (root / 'receipt.json').write_bytes(encode(receipt))
    print(json.dumps({'status': receipt['status'], 'browser': receipt['browser'], 'root': str(root)}))


if __name__ == '__main__':
    main()
