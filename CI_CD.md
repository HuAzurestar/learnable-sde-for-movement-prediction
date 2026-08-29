# CI/CD guide

This guide explains the repository workflows. A green check is evidence only
for the commit and event shown by GitHub Actions; this document does not claim
the outcome of any particular remote run.

## When workflows run

`CI` runs for pushes to `main`, pull requests targeting `main`, and manual
`workflow_dispatch` runs. It has read-only repository permission.

`Release` can also be started manually to reproduce a build, but it publishes
only from a `v*` tag ref. The publish job alone receives `contents: write`;
ordinary CI and non-tag manual runs cannot create a release.

## CI jobs

| Job shown in GitHub | What it verifies | Why it exists |
| --- | --- | --- |
| `policy / traceability and public boundary` | On pull requests, the title and branch satisfy the traceability convention. On every trigger, `scripts/check_public_release.py` rejects files or identifiers that violate the public-release boundary. | Keeps each change reviewable and stops private material from entering a public build. |
| `python / compile and tests (3.10)` | Installs the test extra, byte-compiles the public Python packages, then runs `pytest`. | Verifies the supported baseline interpreter and catches syntax or behavioural regressions. |
| `python / compile and tests (3.12)` | Performs the same install, compilation, and test suite on Python 3.12. | Detects compatibility regressions on the newer supported interpreter. |
| `required` (shown by GitHub as `CI / required`) | Runs even if an upstream job fails and succeeds only when both `policy` and the complete Python matrix succeed. | Gives branch protection one unambiguous required check instead of requiring readers to interpret several job states. |

The matrix is reported to the final `required` job as one `python` result.
Consequently, `required` fails whenever either Python version fails, is cancelled, or
is skipped rather than successfully completing.

The workflow now renders the aggregate as `CI / required` rather than the
redundant `CI / ci / required`. GitHub branch protection/ruleset configuration
is external to this repository: maintainers should update any required-check
entry to the new displayed name after this workflow change lands.

## Release outputs

For a `v*` tag, `Release` re-runs the public-release scan, installs and tests
the source tree, creates the wheel and source distribution, checks them with
Twine, and installs the built wheel in a fresh virtual environment for a CLI
smoke test. The build output is retained as a short-lived Actions artifact.

After the build succeeds, the tag-only publish job writes `SHA256SUMS` beside
the wheel and source distribution and attaches all three files to the GitHub
Release for that immutable tag. Verify a downloaded artifact locally with
`sha256sum -c SHA256SUMS` from the directory containing the release assets.

## Data boundary

CI uses the deterministic synthetic workflow and public source tree. Raw or
derived trajectory data, identifiers, coordinates, timestamps, predictions,
and trained checkpoints are local-only inputs and must not be committed or
published as CI artifacts or Release assets. See [DATA.md](DATA.md) for the
data contract.
