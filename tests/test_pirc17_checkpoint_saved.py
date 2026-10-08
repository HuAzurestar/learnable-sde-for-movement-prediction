"""Local scientific pointer checks, never a startup-wide history replay."""
from copy import deepcopy

import pytest

from experiments.pirc17.checkpoint_saved import source_leaf
from experiments.pirc17.checkpoint_state import atomic_json
from experiments.pirc17.protocol_core import envelope, file_hash


def chain(tmp_path):
    leaf = envelope(dict(schema_version='original', work_id='w', status='success', arrays={'sha256':'original-arrays'},
        parameter_identity='original-fit', execution_sha256='original-execution'))
    path = tmp_path/'leaf.json'
    atomic_json(path, leaf)
    ref = dict(path=str(path), content_sha256=leaf['sha256'], file_sha256=file_hash(path))
    wrapped = envelope(dict(leaf['payload'], schema_version='import', execution_sha256='new-execution', source_artifact=ref))
    return leaf,path,wrapped


def test_transport_changes_preserve_original_scientific_values(tmp_path):
    leaf,path,wrapped = chain(tmp_path)
    assert source_leaf(wrapped,tmp_path/'import.json',scientific=True) == (leaf,path)


@pytest.mark.parametrize('field,value', [('status','failed'),('arrays',{'sha256':'different'}),('parameter_identity','refit')])
def test_source_wrapper_cannot_change_science(tmp_path,field,value):
    leaf,path,wrapped = chain(tmp_path)
    payload = deepcopy(wrapped['payload']);payload[field] = value
    with pytest.raises(ValueError,match='scientific'):
        source_leaf(envelope(payload),tmp_path/'import.json',scientific=True)


def test_source_bytes_checked_only_when_that_item_is_consumed(tmp_path):
    leaf,path,wrapped = chain(tmp_path)
    atomic_json(path,envelope(dict(leaf['payload'],status='failed')))
    with pytest.raises(ValueError,match='hash'):
        source_leaf(wrapped,tmp_path/'import.json',scientific=True)
