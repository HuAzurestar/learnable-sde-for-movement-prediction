"""Additive-noise specialization of Kidger et al., arXiv:2105.13493, Alg. 1.

The auxiliary state is part of the numerical method, not a fifth model state.
This is not a multiplicative-noise Itô solver or an A-stability certificate.
"""

import numpy as np


def reversible_step(state, auxiliary, drift, noise, dt, drift_at_auxiliary=None):
    """Advance the extended pair; reverse with -dt and the same negated noise.

    Keeping drift at the auxiliary state needs only one new drift evaluation
    per step, after initialization. Callers validate the frozen dynamics and
    reject nonfinite extended states. Noise is the already projected L @ dW.
    """
    first = drift(auxiliary) if drift_at_auxiliary is None else drift_at_auxiliary
    next_auxiliary = 2*state - auxiliary + dt*first + noise
    second = drift(next_auxiliary)
    return state + 0.5*dt*(first+second) + noise, next_auxiliary, second


def affine_extended_scheme(A, b, L, dt):
    """Exact algebraic Gaussian recurrence for [physical, auxiliary] states.

    Initial states are identical, with fully correlated initial covariance.
    Only the physical marginal is an endpoint prediction.
    """
    identity = np.eye(4)
    F = np.block([[identity+dt*A, 0.5*dt**2*(A@A)],
                  [2*identity, dt*A-identity]])
    offset = np.concatenate((dt*b+0.5*dt**2*(A@b), dt*b))
    noise = np.vstack(((identity+0.5*dt*A)@L, L))
    return F, offset, noise
