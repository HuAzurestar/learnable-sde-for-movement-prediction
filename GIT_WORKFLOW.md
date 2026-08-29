# Git collaboration policy

`main` is the protected, always-green integration branch. Work starts from an
open GitHub Issue, uses a short-lived branch, lands through a reviewed PR, and
deletes the topic branch after merge.

## Traceability

Use `feature`, `fix`, `docs`, `refactor`, `perf`, `test`, `build`, `ci`, or
`chore` as matching branch-prefix and commit types. Branches are
`<prefix>/<issue>-<summary>` and PR/commit titles are
`#<issue> <type>(optional-scope): imperative summary`. The documented
repository-bootstrap exception is `chore: repository bootstrap ...`; it must
explain why no Issue exists.

## Daily and recovery workflow

```bash
git switch main
git pull --ff-only origin main
git switch -c feature/142-inference-route
# make one logical change; run the canonical checks in CONTRIBUTING.md
git add <paths>
git diff --cached
git commit
git push -u origin feature/142-inference-route
```

Refresh a private topic branch with `git fetch origin` then
`git rebase origin/main`; never rebase or force-push `main`. Resolve conflicts
deliberately, run the affected checks, and use `git rebase --abort` when the
intended result is uncertain. Correct published work with `git revert`, not a
history rewrite. Release tags are annotated and made only from a verified
`main` commit; no deployment workflow is configured for this repository.

## Data and artifact boundary

Only synthetic fixtures and reviewed aggregate material may enter Git. Raw or
transformed trajectories, row-level outputs, checkpoints, local mounts, logs,
credentials, and archives containing them are prohibited. CI runs the public
release scan; protected-branch settings must require the stable `ci / required`
check and a review before merge.
