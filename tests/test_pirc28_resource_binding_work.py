"""Recipe work is independent of replicate count; bindings remain detached."""

from copy import deepcopy

import pytest

from experiments.pirc28 import runtime
from models.switching import SwitchingError
from tests.test_pirc28_runtime import recipe


def test_resource_derivation_once_per_k_per_recipe_call(monkeypatch):
    calls = []
    original = runtime.recipe_resources

    def derive(value, k):
        calls.append(k)
        return original(value, k)

    monkeypatch.setattr(runtime, 'recipe_resources', derive)
    value = recipe()
    cells = [{'fit_K': k} for _ in range(16) for k in (1, 2)]
    result = runtime.bind_recipe_execution({'switching_recipe': value, 'cells': cells})
    assert calls == [1, 2]
    expected = deepcopy(result['cells'][2]['execution'])
    result['cells'][0]['execution']['config']['paths'] = 999
    assert result['cells'][2]['execution'] == expected

    changed = deepcopy(value)
    changed['train_paths'] += 1
    fresh = runtime.bind_recipe_execution({'switching_recipe': changed, 'cells': [{'fit_K': 1}]})
    assert calls == [1, 2, 1]
    assert fresh['cells'][0]['execution']['inputs'] != expected['inputs']


@pytest.mark.parametrize('invalid', [True, 1.0, '1', {}, None])
def test_cached_k_cannot_hide_invalid_cell_type(invalid):
    with pytest.raises(SwitchingError) as error:
        runtime.bind_recipe_execution({'switching_recipe': recipe(),
                                       'cells': [{'fit_K': 1}, {'fit_K': invalid}]})
    assert error.value.code == 'CONTRACT_MISMATCH'
