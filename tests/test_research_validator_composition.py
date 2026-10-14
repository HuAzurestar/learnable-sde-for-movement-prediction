"""Combined upstream/PIRC-28 plugin bindings; engineering-only controls."""

from dataclasses import replace

import pytest

from application.research_admission import plugin_binding
from application.research_contracts import CapabilityRegistry
from experiments.pirc28.runtime import switching_plugin
from infrastructure.research_store import ResearchError


def metadata_validator(*args, **kwargs):
    return None


def test_upstream_hooks_and_pirc28_failure_handler_are_all_bound():
    original = switching_plugin()
    combined = replace(original, pre_read_validator=metadata_validator,
                       checkpoint_validator=metadata_validator,
                       result_validator=metadata_validator, validator_store_context=True)
    CapabilityRegistry().register(combined)
    binding = plugin_binding(combined)
    assert combined.failure_validator is original.failure_validator is not None
    assert binding != plugin_binding(original)
    for name in ("pre_read_validator", "checkpoint_validator", "result_validator", "failure_validator"):
        assert plugin_binding(replace(combined, **{name: None})) != binding
    assert plugin_binding(replace(combined, validator_store_context=False)) != binding


@pytest.mark.parametrize("name", ["pre_read_validator", "checkpoint_validator", "result_validator", "failure_validator"])
def test_each_hook_still_requires_callable_registration(name):
    with pytest.raises(ResearchError, match="execution plugin"):
        CapabilityRegistry().register(replace(switching_plugin(), **{name: object()}))
