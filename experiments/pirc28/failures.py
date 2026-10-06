"""Bounded numerical disclosure, never a downgrade of execution authority."""

import numpy as np

from models.switching import SwitchingError

NUMERICAL_CODES = frozenset({'INFERENCE_INAPPLICABLE', 'NONFINITE', 'NUMERICAL_FAILURE'})
NUMERICAL_EXCEPTIONS = (SwitchingError, np.linalg.LinAlgError, FloatingPointError, OverflowError)
EXCEPTION_TYPES = frozenset({'SwitchingError', 'LinAlgError', 'FloatingPointError', 'OverflowError'})


def numerical_diagnostic(error, stage, initialization_id):
    if isinstance(error, SwitchingError):
        if error.code not in NUMERICAL_CODES:
            raise error
        code = error.code
    elif isinstance(error, (np.linalg.LinAlgError, FloatingPointError, OverflowError)):
        code = 'NUMERICAL_FAILURE'
    else:
        raise error
    # Do not copy arbitrary exception messages, arrays or source paths.
    return {'schema_version': 'pirc28-initialization-failure-v1', 'state': 'FAILED',
        'initialization_id': initialization_id, 'stage': stage, 'error_code': code,
        'exception_type': 'SwitchingError' if isinstance(error, SwitchingError) else type(error).__name__}


def valid_numerical_diagnostic(value, stage, initialization_id):
    return (type(value) is dict and value.get('schema_version') == 'pirc28-initialization-failure-v1'
        and value.get('state') == 'FAILED' and value.get('initialization_id') == initialization_id
        and value.get('stage') == stage and type(value.get('error_code')) is str
        and value['error_code'] in NUMERICAL_CODES and type(value.get('exception_type')) is str
        and value['exception_type'] in EXCEPTION_TYPES)


class NumericalCellFailure(SwitchingError):
    """Observation-only failure receipt; not a successful result or metric."""
    def __init__(self, stage, initializations, selection=None, *, error_code='INFERENCE_INAPPLICABLE', cause=None):
        super().__init__(error_code, 'registered switching cell is numerically inapplicable at '+stage)
        self.diagnostic = {'schema_version': 'pirc28-numerical-cell-failure-v1',
            'stage': stage, 'error_code': self.code, 'initializations': initializations,
            'selection': selection, 'cause': cause}
