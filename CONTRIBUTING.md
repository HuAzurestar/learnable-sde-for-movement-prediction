# Contributing

Contributions should preserve numerical behavior, make dependency ownership
explicit, and remain reproducible without private data.

## Development setup

```bash
python -m venv .venv
# activate the environment for your shell
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
python -m pytest -q
```

The complete characterization suite takes several minutes on CPU. Fast
architecture and application checks can be run with:

```bash
python -m pytest tests/test_architecture_contracts.py tests/test_application.py -q
```

## Change workflow

1. Create or triage one actionable public GitHub Issue and select one primary
   type: `feature`, `bug`, `documentation`, `refactor`, `performance`, `test`,
   `build`, `ci`, or `maintenance`.
2. Work from protected `main` on a short-lived branch named
   `<prefix>/<issue>-<summary>` (for example, `feature/142-inference-route` or
   `ci/203-python-matrix`). Use `hotfix/<issue>-<summary>` only for an urgent
   bug from a deployed release, and `release/<version>` only for a documented
   stabilization window.
3. Keep one conceptual change per branch and commit. Ordinary commit and PR
   titles use `#<issue> <type>(optional-scope): imperative summary`; the type
   must agree with the branch prefix. `chore: repository bootstrap ...` is the
   sole documented no-Issue exception.
4. Add success-path and failure-path tests for a new component.
5. Run the canonical checks below and `python scripts/check_public_release.py`.
6. Open a focused PR to `main`; include `Refs: #<issue>` (or `Closes: #<issue>`
   when merging should close it), validation results, and compatibility,
   numerical, data, and reproducibility effects in the
   pull request.

`main` must remain green and receive reviewed PRs only. Do not force-push or
delete it. Prefer squash merge after the PR title and checks have been
validated; delete the merged topic branch. See `GIT_WORKFLOW.md` for sync,
conflict recovery, hotfix, and release procedures.

## Canonical checks

```bash
python -m compileall -q application cli data domain estimation evaluation experiments inference infrastructure models transfer
python -m pytest -q
python scripts/check_public_release.py
```

New models, estimators, inference engines, and data sources implement their
corresponding interface and are registered only at the composition root. CLI
modules parse input and delegate; they do not contain training or inference
algorithms. See `DESIGN.md` for the dependency rules.

## Commit messages

Use the Issue-first Conventional Commits shape:

```text
#<issue> <type>(optional-scope): imperative summary
```

Supported types:

- `feat`: a new user-visible capability;
- `fix`: a correctness or reliability fix;
- `refactor`: a structural change intended to preserve behavior;
- `test`: tests or test fixtures;
- `docs`: documentation only;
- `build`: packaging, dependency, or CI changes;
- `chore`: repository maintenance.

Examples:

```text
#142 feat(inference): add Euler-Maruyama engine
#87 fix(data): reject non-monotonic segment timestamps
#203 docs: document the local dataset contract
```

Use a `BREAKING CHANGE:` footer when a public configuration key, import path,
checkpoint format, or CLI contract changes. Include `Refs: #123`; use
`Closes: #123` only when landing on `main` should close the public Issue. Do
not include private ticket identifiers or workstation paths.

## Data and artifacts

Never commit raw trajectories, row-level predictions, checkpoints, local logs,
or archives containing them. A minimal bug report must use synthetic data. See
`DATA.md` and `SECURITY.md` before attaching any artifact to an issue.

## Pull-request acceptance

A pull request is ready when:

- tests pass on supported Python versions;
- new behavior has an explicit interface and owner;
- errors fail fast rather than silently selecting another implementation;
- random behavior uses an explicit generator;
- documentation distinguishes implemented, experimental, and unavailable work;
- the public release scan reports no sensitive material.
