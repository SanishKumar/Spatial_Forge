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
- fixed-bounds TSDF integration with sparse in-memory accumulator state, dense
  traversal, and exact dense-reference parity;
- deterministic known-pose depth planning of candidate 8 x 8 x 8 voxel blocks
  around observed surfaces;
- strict immutable loading of `.sftplan` diagnostics and read-only verification
  of their current ScanSession replay binding;
- deterministic, depth-derived world-aligned TSDF volume bounds;
- deterministic zero-crossing surface-point extraction from the TSDF; and
- deterministic six-tetrahedron reference triangle meshing.

Sparse traversal and full-sequence optimization, robust outlier filtering,
production meshing, normals, SLAM, map packages, mobile capture, and the visual
inspector are deliberately not implemented yet.

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

Use those exact bounds with the first sparse-storage checkpoint:

```powershell
.\.venv\Scripts\python.exe -m spatialforge reconstruct tsdf-sparse `
  tests/fixtures/minimal.vgsession `
  outputs/progress-sparse.sftsdf `
  --origin 0.5 -1 -1 `
  --dimensions 2 4 4 `
  --voxel-size-m 0.5 `
  --truncation-m 0.5
```

Expected:

```text
voxels: total=32 observed=8 fused=8
voxel_updates: 16 max_weight=2
storage: sparse accumulator_entries=8
output_sha256: e61803737cdd68b209459fb644cc2f67f18e0420d306273316808e27d2e89994
```

The equal digest confirms byte parity for this fixture. The sparse command
stores sums and weights only for updated voxels, but deliberately retains the
bounded dense traversal in this checkpoint.

Plan candidate blocks for the eventual block-backed reconstruction path:

```powershell
.\.venv\Scripts\python.exe -m spatialforge reconstruct tsdf-block-plan `
  tests/fixtures/minimal.vgsession `
  outputs/progress-blocks.sftplan `
  --voxel-size-m 0.125 `
  --truncation-m 0.5
```

Expected:

```text
depth_samples: valid=8 invalid=0
grid: voxel_size_m=0.125000000 block_resolution=8 block_extent_m=1.000000000
candidate_blocks: surface=4 active=8 halo=4 voxel_slots=4096
block_bounds: min=(0, -1, -1) max=(1, 0, 0)
output_sha256: 372c7c5d49eff1a30317ceb8b67cb3c40683f1d9d2a6179a049772f763d6f79d
```

This `.sftplan` is a deterministic surface-neighborhood plan only. Fusion does
not consume it yet, and it deliberately does not plan the dense reference
backend's full camera-to-surface free-space updates.

Strictly load that artifact and check it against the current session replay:

```powershell
.\.venv\Scripts\python.exe -m spatialforge reconstruct tsdf-block-plan-verify `
  outputs/progress-blocks.sftplan `
  tests/fixtures/minimal.vgsession
```

The important status lines are:

```text
artifact: valid
session_replay: matched
geometry_recomputed: no
```

`artifact: valid` means the strict loader accepted the schema, types, ordering,
limits, and cross-field invariants. `session_replay: matched` means the
artifact's session ID and replay digest still match the current sensor inputs
and replay-derived frame metadata. `geometry_recomputed: no` is equally
important: verification does not decode depth again, regenerate block
coordinates, allocate TSDF blocks, or fuse any values.

The plan SHA-256 identifies the exact artifact bytes. It does not authenticate
the artifact or prove that its candidate geometry came from trusted planner
code. This verification command is read-only and writes no output file.

Both TSDF artifacts use the same `.sftsdf` contract. Mesh the sparse result
directly:

```powershell
.\.venv\Scripts\python.exe -m spatialforge reconstruct triangle-mesh `
  outputs/progress-sparse.sftsdf `
  outputs/progress-sparse-mesh.ply
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
a PLY viewer. You can also open `outputs/progress-sparse-mesh.ply` in a viewer
that supports PLY faces. Commands refuse to overwrite outputs, so delete an
old diagnostic or choose a new filename before rerunning it.

The format and coordinate conventions are documented in
[`docs/scan-session-v0.md`](docs/scan-session-v0.md). TUM-specific conversion
rules are in [`docs/tum-import.md`](docs/tum-import.md), and the current
reconstruction steps are in
[`docs/known-pose-point-cloud.md`](docs/known-pose-point-cloud.md) and
[`docs/reference-tsdf.md`](docs/reference-tsdf.md). Automatic volume selection
is in [`docs/automatic-tsdf-bounds.md`](docs/automatic-tsdf-bounds.md), sparse
accumulation is in [`docs/sparse-tsdf.md`](docs/sparse-tsdf.md), surface
TSDF block planning is in
[`docs/tsdf-block-plan.md`](docs/tsdf-block-plan.md),
surface extraction is in [`docs/surface-points.md`](docs/surface-points.md),
reference triangle meshing is in
[`docs/triangle-mesh.md`](docs/triangle-mesh.md), and overall status is in
[`docs/roadmap.md`](docs/roadmap.md).
