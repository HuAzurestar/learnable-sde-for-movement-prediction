"""Pure identity metadata fixtures; no trajectories, fitting or SDE rollouts."""
import hashlib

import pytest

from experiments.pirc17.partition_description import audit_rows, metadata_functions
from experiments.pirc17.protocol_core import digest


BUILDER = '''
def _split_for_block(block_id, seed):
    value = int.from_bytes(hashlib.sha256(f'{seed}:{block_id}'.encode()).digest()[:8], 'big') / 2**64
    return 'train' if value < .7 else 'validation' if value < .85 else 'final_eval'
def forbidden_builder():
    raise AssertionError('not imported or called')
'''
ADAPTER = '''
ADAPT_SEED = 20260912
ADAPT_FRACTION = .20
def _rank(block_id):
    return hashlib.sha256(f'{ADAPT_SEED}:{block_id}'.encode()).hexdigest()
def _roles(samples, final_eval_unlocked):
    blocks = sorted({s.independent_block_id for s in samples if s.split == 'train'}, key=_rank)
    count = max(1, min(len(blocks)-1, round((1-ADAPT_FRACTION)*len(blocks))))
    fit = set(blocks[:count])
    return {s.segment_id: ('train' if s.independent_block_id in fit else 'adapt')
            if s.split == 'train' else 'validation' for s in samples if s.split != 'final_eval'}
def forbidden_fit():
    raise AssertionError('not imported or called')
'''


def fixture():
    splitter, roles = metadata_functions(BUILDER, ADAPTER)
    assignments, samples = [], []
    for i in range(40):
        block = hashlib.sha256(f'artificial-{i}'.encode()).hexdigest()
        role = splitter(block, 20260912)
        assignment = dict(file_id=f'fixture-file-{i}', independent_block_id=block, split=role)
        identity = dict(data_version='software-fixture', file_id=assignment['file_id'],
                        segment_id=f'fixture-segment-{i}', history_start=0, history_end=3,
                        target_start=4, target_end=7)
        assignments.append(assignment)
        samples.append(dict(identity, sample_id=digest(identity), independent_block_id=block, split=role))
    return assignments, samples, splitter, roles


def run(assignments, samples, splitter, roles):
    return audit_rows(assignments, samples, seed=20260912,
                      split_function=splitter, role_function=roles)


def test_complete_metadata_counts_and_zero_intersections():
    public, partition, segments, roles = run(*fixture())
    assert sum(partition['counts'].values()) == len(segments) == 40
    assert all(n == 0 for fields in public['release_intersections'].values() for n in fields.values())
    assert all(n == 0 for fields in public['global_development_role_intersections'].values() for n in fields.values())
    assert len(roles) == partition['counts']['train'] + partition['counts']['validation']
    assert sum(v['independent_block_id'] for k, v in public['global_development_roles'].items()
               if k in ('train', 'adapt')) == public['release_counts']['train']['sampled_recording_hash_blocks']


def test_public_projection_contains_no_recording_ids():
    assignments, samples, splitter, roles = fixture()
    public, *_ = run(assignments, samples, splitter, roles)
    text = str(public)
    for row in samples:
        assert row['file_id'] not in text
        assert row['segment_id'] not in text
        assert row['independent_block_id'] not in text


def test_duplicate_assignment_rejected():
    a, s, f, r = fixture()
    with pytest.raises(ValueError, match='duplicate file'):
        run(a + [a[0]], s, f, r)


def test_assignment_seed_rule_and_sample_join_rejected():
    a, s, f, r = fixture()
    a[0]['split'] = 'train' if a[0]['split'] != 'train' else 'validation'
    with pytest.raises(ValueError, match='assignment'):
        run(a, s, f, r)
    a, s, f, r = fixture()
    s[0]['file_id'] = 'unregistered'
    with pytest.raises(ValueError, match='join'):
        run(a, s, f, r)


@pytest.mark.parametrize('field,value', [('history_end', 4), ('target_start', 3),
                                      ('history_start', 1), ('target_end', 2)])
def test_midpoint_or_history_target_overlap_rejected(field, value):
    a, s, f, r = fixture()
    s[0][field] = value
    with pytest.raises(ValueError, match='bounds'):
        run(a, s, f, r)


def test_changed_window_identity_and_duplicate_segment_rejected():
    a, s, f, r = fixture()
    s[0]['sample_id'] = '0' * 64
    with pytest.raises(ValueError, match='sample identity'):
        run(a, s, f, r)
    a, s, f, r = fixture()
    with pytest.raises(ValueError, match='unique midpoint'):
        run(a, s + [s[0]], f, r)


def test_only_hash_functions_are_extracted():
    with pytest.raises(ValueError, match='complete source'):
        metadata_functions(BUILDER.replace('_split_for_block', 'missing'), ADAPTER)
    with pytest.raises(ValueError, match='adaptation'):
        metadata_functions(BUILDER, ADAPTER.replace('ADAPT_FRACTION = .20', 'ADAPT_FRACTION = .30'))
