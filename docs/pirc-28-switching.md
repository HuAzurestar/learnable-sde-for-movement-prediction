# PIRC-28 switching research

The first-round model has K=1 or K=2, state `[x,y,vx,vy]` in SI units, and a
time-homogeneous continuous-time Markov chain. Each mode has affine acceleration
and constant velocity diffusion. Labels `state-1/state-2` have no behavior names.

`models.switching.SwitchingSDEModel` follows the existing `SDEModel` drift and
diffusion interfaces with an explicit `ModelContext.regime`. It intentionally
does not implement the joint `ExactTransitionProvider` capability. The
`conditional_affine_transition(mode, dt)` kernel is exact for a fixed-mode
segment and reuses the existing matrix-exponential covariance implementation.
The CTMC endpoint matrix alone is not an exact switching motion likelihood.

`experiments.pirc28.synthetic.ScenarioSpec` freezes the model, irregular observation
schedule, observation matrix/noise, mode-conditioned initial state distribution,
path count, seed, label map and jump quota. `generate_scenario(spec,
check_budget=...)` samples exponential dwell times and all jumps, propagating
continuous motion separately in each segment. The caller's poll must connect to
shared supervisor control in production. This module creates no budget ledger,
runtime root or implicit study authorization.

The returned `ObservationBatch` contains only times, observations, H/R and scenario
identity. Mode/state truth, jump paths and dwell statistics are separate
`SyntheticTruth` evaluator inputs. Role-specific PCG64 streams keep observation
noise from changing the latent path. Per-path seed identities support replay;
the current adapter declares `restart-only`, not unimplemented exact recovery.
Expected-jump preflight, actual jump limits, finite checks and conservative tensor
quotas refuse incomplete paths. A refused path is not a successful truncated one.

The scientific design's conditional state-path formulation is motivated by
[Michelot & Blackwell (2019)](https://doi.org/10.1111/2041-210X.13154).
Our generator and planned approximate filter are separate implementations; no
claim is made that the paper validates our inference candidate.

Run the numerical qualification fixtures with:

```console
python -m pytest tests/test_pirc28_synthetic.py -q
```

These fixtures establish generator semantics only. Identifiability, fitted
parameter recovery, predictive benefit and real-data eligibility require the
separate registered protocol, inference qualification and shared runtime gate.
Research execution must use the same explicit external runtime root/store identity
as PIRC-26/27. Do not initialize a fresh per-worktree store to refresh arm budgets.
Private or reserved data is never an input to these generator fixtures.
