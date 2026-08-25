# GeoLife ingestion correction and claim boundary

This note supersedes the immutable first-upload report/comment wherever their
split segment counts appear. The authoritative 70/15/15 file-ID split contains:

| Partition | Files | Segments | Points | Users |
|---|---:|---:|---:|---:|
| train | 10,332 | **126,837** | 4,728,924 | 165 |
| validation | 2,214 | **26,589** | 995,266 | 137 |
| evaluation | 2,214 | **27,925** | 1,061,017 | 139 |

The earlier values `12,683 / 2,659 / 2,793` omitted one digit. The corrected
segment counts sum to 181,351 and agree with the split registry, build
statistics, and independently recomputed table counts.

## What the partition gate proves

The gate proves only that the three **file-ID** partitions are pairwise
disjoint and exhaustive, and that the GeoLife and OSM file-ID namespaces have
intersection zero. It does not establish user-independent partitions or
content-level deduplication.

The registered snapshot has these user overlaps:

- train/validation: 136;
- train/evaluation: 136;
- validation/evaluation: 122;
- all three partitions: 121.

Accordingly, this split cannot support a claim of user-independent
generalization. Cross-source duplicate people or trajectories have not been
content-deduplicated and remain an explicit boundary.

## Quality, duration, and release gates

The audit independently verifies all 14 column names and dtypes, requires an
explicit build `status` whose only allowlisted success value is `pass`, and
checks `n_failed == 0`, `schema_identical is true`, row/file/segment counts,
byte size, and SHA-256. Missing, unknown, false, failed, or inconsistent evidence
makes the technical gate fail and the CLI exit nonzero.

Duration claims retain both definitions:

- cleaned `geolife_leg` with a 60-second gap: 0 segments at or above 6 h;
- walk-filtered tiers with a 300-second gap: 40 segments/22 users at or above
  6 h, 0 at or above 12 h, maximum 11.65 h.

Source/licence registration and independent legal release review are separate.
The current source record is registered, but independent legal approval is not
recorded. Therefore a technical pass still yields an overall `provisional`
result and must not be described as release approval.
