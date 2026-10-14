from types import SimpleNamespace

import pytest

from experiments.pirc17.pilot_selection import select_windows


def window(sample,block,role="validation"):
    # Deliberately no position, label or score attributes: selection cannot read them.
    return SimpleNamespace(sample_id=sample,block_id=block,role=role)


def test_independent_blocks_are_deterministic_not_duplicate_windows():
    rows=[window("c","block2"),window("b","block1"),window("a","block1")]
    assert [w.sample_id for w in select_windows(rows,count=2)]==["a","b"]
    selected=select_windows(rows,count=2,policy="lexical_independent_blocks")
    assert [w.sample_id for w in selected]==["a","c"]
    assert selected==select_windows(list(reversed(rows)),count=2,policy="lexical_independent_blocks")


@pytest.mark.parametrize("rows,count,policy",[
    ([window("a","b","final_eval")],1,"lexical_origins"),
    ([window("a","b","train")],1,"lexical_origins"),
    ([window("a","b"),window("a","c")],1,"lexical_origins"),
    ([window("a","b"),window("c","b")],2,"lexical_independent_blocks"),
    ([window("a","b")],0,"lexical_origins"),
    ([window("a","b")],1,"best_scores"),
])
def test_invalid_selection_fails_closed(rows,count,policy):
    with pytest.raises(ValueError):
        select_windows(rows,count=count,policy=policy)
