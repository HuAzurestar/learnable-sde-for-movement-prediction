"""Merge controls for independently scoped PIRC-27/28 execution authorities."""

import pytest

from application.research_contracts import CapabilityRegistry
from experiments.pirc25.runner import SharedRunner
from infrastructure.research_store import ResearchError


@pytest.mark.parametrize('restart,transition', [({}, {}), ({'receipt': 'restart'}, {'receipt': 'transition'})])
def test_mixed_source_authorities_refused_before_any_store_access(restart, transition):
    class NoStoreAccess:
        def __getattribute__(self, name):
            raise AssertionError('Mixed authorities must not read, fund, register or launch')

    runner = SharedRunner(NoStoreAccess(), CapabilityRegistry())
    with pytest.raises(ResearchError, match='CONTRACT_MISMATCH.*mutually exclusive'):
        runner.run_cell('unused-study', 'unused-cell', restart_adjudication=restart,
                        source_transition_ref=transition)
