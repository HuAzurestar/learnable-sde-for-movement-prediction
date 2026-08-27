# OSM distance fields

This pipeline turns real OSM ways into three projected GeoTIFF channels:
`road_dist`, `water_dist`, and `building_dist`. Values are metres. A fourth
`osm_coverage` raster is authoritative and prevents an uncovered coordinate
from being interpreted as a long observed distance.

## Algorithm and contracts

The direct baseline would compute the nearest geometry for every trajectory
point, costing roughly `O(points × features)`. The implemented path rasterizes
the selected features once and applies SciPy's exact Euclidean distance
transform to the binary grid. For a grid with `N` pixels and `F` input
geometries, the dominant work is `O(N + F)` per channel after PBF parsing.

The core grid is expanded by `max_distance_m` before rasterization. The distance
transform is computed on this halo and cropped back to the requested grid, so a
tile edge cannot create a false nearest feature within the declared cap.
Distances at the cap are valid lower bounds (`has_osm=1`); missing coverage is
encoded as GeoTIFF nodata `-9999`, `osm_coverage=0`, and slice-level NaN.

Feature rules are frozen in `scripts/build_osm_distance_fields.py` and copied
into every manifest:

- roads: active `highway=*` ways, including paths/footways/tracks; proposed,
  construction, abandoned, razed, raceway, platform, corridor, elevator, and
  planned values are excluded;
- water: `waterway` river/stream/canal/drain/ditch/tidal_channel plus closed-way
  `natural=water`, `water=*`, and `landuse=reservoir` areas;
- buildings: active `building=*` ways; no/construction/proposed/planned/
  abandoned/demolished/razed values are excluded.

Version 1 intentionally declares one limitation: relation-only multipolygons
are not assembled. Closed ways are included. A future relation-aware release
must use a new manifest schema version and be compared against v1.

## Reproducible environment

Python 3.10+ is required. Install the project and the optional geospatial set:

```powershell
python -m venv .venv-osm
.\.venv-osm\Scripts\python.exe -m pip install -e '.[osm,test]'
```

For large PBFs, install osmium-tool and prefilter a target plus a halo wider
than the distance cap. The GB smoke used osmium-tool 1.19.1:

```powershell
osmium extract `
  --bbox=-0.20,51.45,-0.05,51.57 `
  --strategy=simple --set-bounds `
  --output gb_westminster_extract.osm.pbf `
  united_kingdom-latest.osm.pbf
```

`simple` is safe here only because the requested core is strictly farther from
the extract boundary than the 2 km distance cap. Country/province production
jobs should instead use reviewed coverage polygons and spatial tiling.

## Build examples

Zhejiang/Hangzhou verified interior smoke (50 m, 5 km cap):

```powershell
python scripts/build_osm_distance_fields.py `
  --pbf E:\Worktable\NEX-317_data_sources\map_data\zhejiang-latest.osm.pbf `
  --output-dir artifacts\nex380\zhejiang_hangzhou_smoke `
  --region zhejiang_hangzhou_smoke `
  --bounds 120.05 30.15 120.25 30.35 `
  --crs EPSG:32651 --resolution-m 50 --max-distance-m 5000 `
  --location-index flex_mem --assume-bounds-covered `
  --coverage-note 'Verified interior Hangzhou smoke window.' `
  --source-url https://download.openstreetmap.fr/extracts/asia/china/zhejiang-latest.osm.pbf `
  --source-version 2026-08-13T00:51:06Z `
  --expected-sha256 A30017225E70490409736810962BC634D44DB875CA023F6A0C745F923122E15E
```

GB/Westminster is the same command with `EPSG:27700`; pass the prefiltered PBF
as `--pbf` and the original 2.45 GB file through `--parent-pbf`,
`--parent-source-url`, and `--parent-expected-sha256`. This preserves both
checksums and the exact derivation note.

For arbitrary or full administrative coverage, omit
`--assume-bounds-covered` and provide a reviewed WGS84 Polygon/MultiPolygon via
`--coverage-geojson`. The command refuses ambiguous coverage by default.

## Condition-slice integration

Never mutate the existing condition tree in place:

```powershell
python scripts/augment_cond_slices_osm.py `
  --input-root E:\Worktable\NEX-317_data_sources\cond_slices `
  --output-root E:\Worktable\NEX-317_data_sources\cond_slices_osm_v1 `
  --manifest path\to\gb\manifest.json `
  --manifest path\to\zhejiang\manifest.json
```

The adapter preserves `dem_elev`, `dem_slope`, `landcover`, and `has_map`, then
adds the three distance columns and `has_osm`. `data.source.CONDITION_SPECS`
exposes these as condition kind `osm`; rows with `has_osm=0` are excluded.

## Implementation sources

- Pyosmium documents location indexes and recommends file-backed indexes when
  memory is constrained: https://docs.osmcode.org/osmium/latest/osmium-index-types.html
- Rasterio's `rasterize` burns GeoJSON-like geometry into a declared affine
  grid: https://rasterio.readthedocs.io/en/stable/api/rasterio.features.html
- SciPy documents `distance_transform_edt` as an exact Euclidean distance
  transform with physical sampling: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.distance_transform_edt.html
- OSM tagging references: https://wiki.openstreetmap.org/wiki/Key:building and
  https://wiki.openstreetmap.org/wiki/Key:waterway
