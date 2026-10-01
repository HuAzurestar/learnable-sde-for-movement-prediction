# Shared engineering validation

Integration evidence for PR #22, using synthetic fixtures only:

- Runtime checkpoint: `3dbeaaf445ddac4f8bb835a0fa797b43aa2ab37b`.
- Paper-side checkpoint: `5d2659a012cd41cbda450df165dc58d2a5f56fc8` (related PR #13).
- Local Python 3.10: 322 tests passed; compilation and public release scan passed.
- GitHub Linux Python 3.10 and 3.12: both test jobs passed on the runtime checkpoint.
- Paper repository: 31 local Python tests passed; both local PDF/reference checks
  and all PR CI jobs passed on the paper-side checkpoint.
- Independent CLI reproduction passed with Playwright 1.55 and headless Edge,
  including retained failure/retry cost, full matrix export, independent-block
  counts, evidence round-trip, index rebuild, successful-cell reuse, comparison
  rendering and exact-byte CSV download.

The complete aggregate was
`f108ac7ddc0960c526587d8342a4448bd41649c6ad520d86cd181f9796dfe211`.
Receipts and screenshots remain outside Git. The reproduction command in the
runtime guide generates a fresh version; run/attempt identities are intentionally
new rather than copies of this receipt. Two synthetic seeds still represent only
one independent block, not scientific replication or qualification.

The original Playwright 1.45 browser teardown failed despite passing page/CSV
assertions; that attempt is not counted as a successful reproduction. The
successful rerun used an isolated browser environment. The initial runtime PR
policy job captured the old draft title and failed; publication of this note
triggers verification with the issue-bound title. Neither these observations nor
local tests waive required CI or review. PR acceptance remains pending until
the required reviews, checks and both merges are recorded.
