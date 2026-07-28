# SpatialForge

SpatialForge is a standalone spatial mapping and localization engine. It will
turn calibrated indoor scans into metric, semantic, localizable maps while
remaining independent of navigation products such as VoiceGIS.

This repository currently implements these narrow foundations:

- a versioned, folder-backed `ScanSession` (`.vgsession`) contract;
- validation for calibration, timestamps, file references, depth scale, IMU
  samples, and rigid camera poses;
- deterministic offline replay around RGB observations;
- an extracted TUM RGB-D folder importer with known-pose support;
- calibrated, known-pose RGB-D back-projection to a deterministic colored PLY;
- fixed-bounds projective TSDF integration as a deterministic CPU reference;
- deterministic, depth-derived world-aligned TSDF volume bounds;
- deterministic zero-crossing surface-point extraction from the TSDF; and
- deterministic six-tetrahedron reference triangle meshing.

Optimized fusion, robust outlier filtering, production meshing, normals, SLAM,
map packages, mobile capture, and the visual inspector are deliberately not
implemented yet.

## Set up

Python 3.11 or newer is required.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
```

## Validate and replay a session

```powershell
.\.venv\Scripts\python.exe -m spatialforge scan validate `
  tests/fixtures/minimal.vgsession
.\.venv\Scripts\python.exe -m spatialforge scan replay `
  tests/fixtures/minimal.vgsession
```

Import the committed tiny TUM-layout fixture:

```powershell
.\.venv\Scripts\python.exe -m spatialforge scan import-tum `
  tests/fixtures/tum/rgbd_dataset_freiburg1_tiny `
  outputs/tum-tiny.vgsession
.\.venv\Scripts\python.exe -m spatialforge scan validate `
  outputs/tum-tiny.vgsession
```

## Build a known-pose point cloud

```powershell
.\.venv\Scripts\python.exe -m spatialforge reconstruct point-cloud `
  tests/fixtures/minimal.vgsession `
  outputs/minimal.ply
```

## Check current progress yourself

First, run the complete automated test suite:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The last line should be `OK`.

Then run the small numerical TSDF proof:

```powershell
.\.venv\Scripts\python.exe -m spatialforge reconstruct tsdf `
  tests/fixtures/minimal.vgsession `
  outputs/progress.sftsdf `
  --origin 0 -0.25 -0.25 `
  --dimensions 4 1 1 `
  --voxel-size-m 0.5 `
  --truncation-m 0.5
```

The important output is:

```text
voxels: total=4 observed=3 fused=3
voxel_updates: 6 max_weight=2
```

`max_weight=2` proves that both frames contributed to the same voxels. Inspect
the exact signed distances with:

```powershell
Get-Content outputs/progress.sftsdf
```

The three observed TSDF values should be `1.0`, `0.5`, and `-0.5`, each with
weight `2`. The sign change brackets the known plane at world `X=1.0 m`.

Extract that zero crossing:

```powershell
.\.venv\Scripts\python.exe -m spatialforge reconstruct surface-points `
  outputs/progress.sftsdf `
  outputs/progress-surface.ply
```

Expected:

```text
crossings: x=1 y=0 z=0
points: exact_zero=0 crossing=1 total=1
```

The final line of `outputs/progress-surface.ply` should be:

```text
1.000000000 0.000000000 0.000000000
```

The one-dimensional TSDF above proves the zero crossing but cannot contain a
triangle. Automatically infer a padded, world-aligned 3D volume from the known
depth and poses:

```powershell
.\.venv\Scripts\python.exe -m spatialforge reconstruct tsdf-auto `
  tests/fixtures/minimal.vgsession `
  outputs/progress-auto.sftsdf `
  --voxel-size-m 0.5 `
  --truncation-m 0.5
```

Expected:

```text
bounds_depth: valid=8 invalid=0
volume: origin=(0.500000000, -1.000000000, -1.000000000) dimensions=(2, 4, 4) voxels=32
integration: observed=8 fused=8 updates=16 max_weight=2
output_sha256: e61803737cdd68b209459fb644cc2f67f18e0420d306273316808e27d2e89994
```

The inferred artifact uses the same `.sftsdf` contract. Mesh it directly:

```powershell
.\.venv\Scripts\python.exe -m spatialforge reconstruct triangle-mesh `
  outputs/progress-auto.sftsdf `
  outputs/progress-auto-mesh.ply
```

Expected:

```text
cells: total=9 eligible=1 active=1
skipped_cells: unknown=8 exact_zero=0
mesh: vertices=9 triangles=8 boundary_edges=8
```

All mesh vertices lie on `X=1.0 m`; its eight triangles cover a
`0.5 m x 0.5 m` square and face the positive/free-space side. The exact
meshing contract is documented in `docs/triangle-mesh.md`.

For a visual check, open `outputs/minimal.ply` from the point-cloud command in
a PLY viewer. You can also open `outputs/progress-auto-mesh.ply` in a viewer
that supports PLY faces. Commands refuse to overwrite outputs, so delete an
old diagnostic or choose a new filename before rerunning it.

The format and coordinate conventions are documented in
[`docs/scan-session-v0.md`](docs/scan-session-v0.md). TUM-specific conversion
rules are in [`docs/tum-import.md`](docs/tum-import.md), and the current
reconstruction steps are in
[`docs/known-pose-point-cloud.md`](docs/known-pose-point-cloud.md) and
[`docs/reference-tsdf.md`](docs/reference-tsdf.md). Automatic volume selection
is in [`docs/automatic-tsdf-bounds.md`](docs/automatic-tsdf-bounds.md), surface
extraction is in [`docs/surface-points.md`](docs/surface-points.md), reference
triangle meshing is in [`docs/triangle-mesh.md`](docs/triangle-mesh.md), and
overall status is in [`docs/roadmap.md`](docs/roadmap.md).
