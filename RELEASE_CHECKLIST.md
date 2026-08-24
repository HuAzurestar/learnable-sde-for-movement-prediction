# Public release checklist

Use this checklist when preparing a tagged public release.

## Required decisions

- [x] Confirm the public repository name:
      `learnable-sde-for-movement-prediction`.
- [ ] Select a software license, add its SPDX identifier to `CITATION.cff`, and
      add the corresponding `LICENSE` file.
- [x] Confirm the public author or organization name used by release metadata.
- [x] Configure Git commits with the intended public name and a GitHub noreply
      email if personal email disclosure is not desired.

## Content gate

- [x] Run `python scripts/check_public_release.py`.
- [x] Run `python -m pytest -q`.
- [x] Run the synthetic train/checkpoint/predict CLI workflow.
- [x] Confirm that no raw data, derived row-level results, checkpoint, ZIP, or
      local log is tracked.
- [ ] Review every PDF or image added after this checklist for metadata and
      publication rights.

## Git gate

- [x] Publish from a clean public branch or a new repository with no reachable
      internal history.
- [ ] Inspect `git log --format=fuller` for author metadata.
- [ ] Inspect `git ls-files` and `git count-objects -vH` before pushing.
- [ ] Enable branch protection, required CI, Dependabot, and private
      vulnerability reporting on GitHub.

Do not add private development branches or tags to the public remote.
