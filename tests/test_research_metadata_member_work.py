"""Count forwarded actual data-descriptor queries, not cached file facts.

CRT-internal queries are outside this Python-visible work counter. Every
original native name/stamp/directory check and production JSON parse still runs.
"""
import os
from collections import Counter

import pytest

from infrastructure.research_store import ResearchStore, digest, encode


@pytest.mark.parametrize('count', [8, 32])
def test_actual_prefix_member_stamps_do_not_duplicate_descriptor_queries(tmp_path, monkeypatch, count):
    store = ResearchStore(tmp_path, 'member-work', initialize=True)
    for position in range(count):
        store.append('MEMBER_WORK', {'position': position})
    paths = sorted((store.path / 'events').glob('*.json')) + [store.path / 'head.json']
    names = {}
    for path in paths:
        information = path.stat()
        names[(information.st_dev, information.st_ino)] = path.name
    assert len(names) == len(paths) == count + 1
    original_fstat, queries = os.fstat, Counter()

    def fstat(descriptor):
        information = original_fstat(descriptor)
        identity = information.st_dev, information.st_ino
        if identity in names:
            queries[names[identity]] += 1
        return information

    monkeypatch.setattr(os, 'fstat', fstat)
    actual = store.events()
    # No observation stat is added to the actual production read. Events/head
    # are all consumed, with original sequence/hash/type verification unchanged.
    observed = {'event_count': len(actual), 'data_fstat_queries': dict(queries),
                'platform': os.name, 'expected_per_member': 6 if os.name == 'nt' else 3}
    (tmp_path / 'member-work-observed.json').write_bytes(encode(observed))
    assert len(actual) == count
    assert [event['payload'] for event in actual] == [{'position': value} for value in range(count)]
    assert all(event['hash'] == digest({key: value for key, value in event.items() if key != 'hash'})
               for event in actual)
    assert set(queries) == set(names.values()), observed
    assert all(value == observed['expected_per_member'] for value in queries.values()), observed
