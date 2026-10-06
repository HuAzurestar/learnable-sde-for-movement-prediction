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

## Unlabeled inference and forecasting

`inference.switching.filter_switching` propagates every source/destination mode
pair on substeps with `max(-qii) * dt <= 0.05`, updates observations and moment
matches all incoming Gaussians. It retains one Gaussian per mode. The returned
objective is explicitly the approximate split-step observation likelihood.
`smooth_switching` adds moment-based backward diagnostics. Filtering posterior
hashes include only their observed prefix; smoothing is refused as a forecast
origin. K=1 reduces to the ordinary affine Kalman filter at observation times.

`estimation.switching.fit_switching` uses these moments for a bounded generalized
EM proposal and line search. Conditional CTMC bridge moments include repeat
jumps, including paths that return to their initial mode. Accepted objective
values, rejected candidates, all initializations, occupancy, collapse and stopping
reason remain available. Initialization selection uses a separate selection batch.
No approximate objective is relabeled as the exact switching likelihood.
Numerically identical mode dynamics are flagged even when occupancy is balanced;
the tolerance is frozen in the inference specification. This flag is a numerical
diagnostic, not a replacement for a registered scientific identifiability gate.

`forecast_switching` first samples a mode and then that mode's conditional state,
preserving their correlation. Future paths are event-driven; mode marginals,
at-least-one-jump probability and different-endpoint-mode probability are reported
separately. A mean trajectory is a summary, not smooth switching dynamics.

## Shared execution and recovery

`experiments.pirc28.runtime` declares the `pirc28-switching` versioned CPU adapter,
its resource plan and numerical-tolerance recovery plugin. The worker refuses
execution without shared supervisor control. It validates the frozen recipe,
source tree, store identity, resource binding and owner-created checkpoint bytes.
No new ledger or mutable arm budget is implemented by PIRC-28.
Resource counts separately bound live fit histories and all test forecast paths;
completed training histories are released after recording their diagnostics.

Numerical recovery restores accepted EM iteration/model/objective identities.
Generation, selection and forecast role streams are deterministically recomputed;
their work remains charged to the original arm in a new linked attempt. This is
not exact mid-segment Brownian recovery. A changed recipe, seed, observations,
initialization, objective or model family refuses recovery. Shared arm closure
cannot be reopened from a checkpoint.

The CLI runs only an already registered study in an existing external store:

```console
python -m experiments.pirc28 --root C:/absolute/shared/runtime --store-id ORIGINAL_ID --stage smoke --seconds 900 run REGISTERED_STUDY
python -m experiments.pirc28 --root C:/absolute/shared/runtime --store-id ORIGINAL_ID --stage pilot --seconds 1800 resume ATTEMPT_ID CHECKPOINT_ID --authorization-id EXISTING_GRANT
```

These are placeholders, not a configured production root or a grant. Registration,
frozen scientific thresholds, qualification and source/data access evidence must
precede a research run. The CLI has no `init` command or unbudgeted fit route.
Production synthetic admission binds the canonical recipe's physical content hash
in a `synthetic-pirc28` source protocol. It does not permit reading a real data
source through this synthetic adapter. Formal metric names and definitions must
match the independently frozen comparison plan. A tiny fixture source protocol is
accepted only with explicit fixture admission and produces fixture qualification.

The frozen recipe separates generated train/selection/test streams and matches
K=1/K=2 test blocks and seeds. Truth is only used by the synthetic evaluator.
Mode Brier diagnostics are permutation-invariant; K=1 does not fabricate a
two-mode score. Results retain all initialization diagnostics and shared attempt
lineage. The synthetic pipeline does not claim identifiability qualification.

```console
python -m pytest tests/test_pirc28_synthetic.py tests/test_pirc28_inference.py tests/test_pirc28_runtime.py -q
```
