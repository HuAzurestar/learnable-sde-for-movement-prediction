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

1. Open or reference a public GitHub issue for behavior-changing work.
2. Keep one conceptual change per branch.
3. Add success-path and failure-path tests for a new component.
4. Run the test suite and relevant numerical smoke workflow.
5. Run `python scripts/check_public_release.py`.
6. Describe compatibility, numerical, data, and reproducibility effects in the
   pull request.

New models, estimators, inference engines, and data sources implement their
corresponding interface and are registered only at the composition root. CLI
modules parse input and delegate; they do not contain training or inference
algorithms. See `DESIGN.md` for the dependency rules.

## Commit messages

Use the Conventional Commits shape:

```text
<type>(optional-scope): imperative summary
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
feat(inference): add Euler-Maruyama engine
fix(data): reject non-monotonic segment timestamps
docs: document the local dataset contract
```

Use a `BREAKING CHANGE:` footer when a public configuration key, import path,
checkpoint format, or CLI contract changes. Reference public issues with
`Closes #123`; do not include private ticket identifiers or workstation paths.

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
