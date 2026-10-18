"""Real-lock engineering controls, never production fitting or SCI authority.

Use disposable admitted metadata and the original 15-second writer timeout.
The observer is an independent thread using a real store append, not a mocked
lock, a relaxed supervisor limit or a numerical result. No shared store touched.
"""

import threading
import time

from experiments.pirc28.source_preflight import validate_scientific_source
from infrastructure.research_store import ResearchError
from tests.test_pirc28_source_preflight import admitted_sources


def test_preflight_budget_handoff_allows_real_owner_heartbeat(admitted_sources):
    store, _, _, sources = admitted_sources
    study, cell, attempt = sources[0]
    finished = threading.Event()
    observations = {}
    writer = None
    polls = 0

    def owner_heartbeat():
        started = time.monotonic()
        try:
            # A separate thread has no right to reuse the reader's thread-local
            # snapshot/OS-lock ownership. Default lock timeout remains 15s.
            store.append('HEARTBEAT', {'attempt_id': attempt['attempt_id'],
                                      'monotonic_elapsed_ms': 0})
            observations['state'] = 'APPENDED'
        except ResearchError as error:
            observations.update(state='REFUSED', error_code=error.code)
        finally:
            observations['elapsed_seconds'] = time.monotonic()-started
            finished.set()

    def budget_handoff():
        nonlocal polls, writer
        polls += 1
        if polls == 2:
            observations['reader_scope_owned_at_handoff'] = store._read_snapshot() is not None
            writer = threading.Thread(target=owner_heartbeat)
            writer.start()
            # This simulates a reader spending time at a cancellation/budget
            # boundary. It does not change the lock's timeout or grant rights.
            assert finished.wait(20), 'original owner heartbeat did not finish'

    try:
        result = validate_scientific_source(store, study, cell, attempt,
                                            check_budget=budget_handoff)
    finally:
        if writer is not None:
            writer.join(25)
            assert not writer.is_alive(), 'retain a terminal observer, never detach it'
    assert polls >= 2
    assert result['qualified_identifiability'] is False
    assert observations.get('state') == 'APPENDED', observations
    assert observations['reader_scope_owned_at_handoff'] is False
    assert not any(event['event_kind'] == 'WORKER_STARTED' for event in store.events())
