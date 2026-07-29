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
- deterministic allocation of replay-matched candidate blocks into temporary,
  zeroed float64-sum and uint32-weight buffers;
- deterministic signed global-voxel addressing into planned block rows and
  local `(z, y, x)` array positions, without allocating missing blocks;
- read-only evaluation of one replay-selected observation at one planned
  voxel, returning an immutable projective TSDF sum/weight delta when exact
  depth and pose exist, or a skip diagnostic, without applying it;
- deterministic, depth-derived world-aligned TSDF volume bounds;
- deterministic zero-crossing surface-point extraction from the TSDF; and
- deterministic six-tetrahedron reference triangle meshing.

Applying block contributions, camera-to-surface free-space coverage, sparse
traversal and full-sequence optimization, robust outlier filtering, production
meshing, normals, SLAM, map packages, mobile capture, and the visual inspector
are deliberately not implemented yet.

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

Allocate the verified candidate coordinates as empty in-memory TSDF block
storage:

```powershell
.\.venv\Scripts\python.exe -m spatialforge reconstruct tsdf-block-allocate `
  outputs/progress-blocks.sftplan `
  tests/fixtures/minimal.vgsession
```

Expected:

```text
artifact: valid
session_replay: matched
depth_decoded: no
geometry_recomputed: no
fusion_performed: no
artifact_written: no
allocation: blocks=8 resolution=8 voxel_slots=4096
layout: shape=(8, 8, 8, 8) axes=block-z-y-x x_fastest=yes
dtypes: tsdf_sums=float64 weights=uint32
zero_state: nonzero_sums=0 nonzero_weights=0 unknown_voxels=4096
payload_bytes: tsdf_sums=32768 weights=16384 total=49152
block_rows: first=(0, -1, -1) last=(1, 0, 0)
```

This command strict-loads and replay-verifies the plan, allocates the buffers,
reports their zero state, and discards them when the process exits. Replay
verification hashes sensor payload bytes but does not decode depth pixels.
The reference allocator caps numeric array payload at `64 MiB`; this does not
include Python, NumPy-header, allocator, or process-memory overhead. There is no
TSDF update, free-space decision, fusion, authentication, or output artifact in
this checkpoint.

Resolve signed global voxel indices in that temporary storage:

```powershell
.\.venv\Scripts\python.exe -m spatialforge reconstruct tsdf-block-address `
  outputs/progress-blocks.sftplan `
  tests/fixtures/minimal.vgsession `
  --voxel 7 -1 -1 `
  --voxel 8 0 0 `
  --voxel -1 0 0
```

Expected:

```text
depth_decoded: no
geometry_recomputed: no
fusion_performed: no
storage_mutated: no
addressing_created_blocks: no
artifact_written: no
allocation: blocks=8 voxel_slots=4096
queries: requested=3 resolved=2 unplanned=1
voxel[0]: status=planned global=(7, -1, -1) block=(0, -1, -1) local=(7, 7, 7) row=0 array=(0, 7, 7, 7) local_flat=511 storage_flat=511
voxel[1]: status=planned global=(8, 0, 0) block=(1, 0, 0) local=(0, 0, 0) row=7 array=(7, 0, 0, 0) local_flat=0 storage_flat=3584
voxel[2]: status=unplanned global=(-1, 0, 0)
```

The valid third query belongs to a block that is absent from the plan, so the
API returns `None` and the CLI reports `unplanned`. It does not insert a block.
Address resolution only computes indices; the zeroed sum and weight arrays are
unchanged and no output file is written.

Evaluate one resolved voxel against one selected known-pose depth observation:

```powershell
.\.venv\Scripts\python.exe -m spatialforge reconstruct tsdf-block-contribution `
  outputs/progress-blocks.sftplan `
  tests/fixtures/minimal.vgsession `
  --observation-sequence 0 `
  --voxel 8 -1 -1
```

Expected:

```text
TSDF BLOCK CONTRIBUTION CHECK scan-synthetic-0001
artifact: valid
session_replay: matched
observation_sequence: 0
voxel: global=(8, -1, -1) block=(1, -1, -1) local=(0, 7, 7) row=1 array=(1, 7, 7, 0) storage_flat=1016
world_xyz_m: (1.062500000, -0.062500000, -0.062500000)
camera_xyz_m: (0.062500000, 0.062500000, 1.062500000)
projected_uv: (0.617647059, 0.617647059)
pixel_uv: (1, 1)
depth_decoded: yes
measured_depth_m: 1.000000000
signed_distance_m: -0.062500000
evaluation: contributes
proposed_delta: tsdf_sum=-0.125000000 weight=1
contributions_applied: 0
fusion_performed: no
storage_mutated: no
missing_blocks_created: no
artifact_written: no
```

The voxel center is transformed into camera space, projected with the aligned
RGB intrinsics, and sampled at the nearest depth pixel. The measured depth of
`1.0 m` minus camera-space depth `1.0625 m` gives `-0.0625 m`; division by the
plan's `0.5 m` truncation gives the proposed TSDF sum delta `-0.125`. A proposed
weight of one describes what a later fusion operation could apply. This
checkpoint applies zero contributions: it evaluates no other observation or
voxel, runs no frame- or block-wide contribution loop, leaves both storage
arrays zero, and writes no artifact.

This single-voxel rule retains the reference TSDF sign and truncation
conventions, but it does not decide which camera-to-surface free-space blocks
should exist. Applying deltas, block/frame traversal, free-space coverage,
fusion, and persistence remain separate later checkpoints. The exact result
and skip contract is documented in
[`docs/tsdf-voxel-contribution.md`](docs/tsdf-voxel-contribution.md).

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
empty block allocation is in
[`docs/tsdf-block-storage.md`](docs/tsdf-block-storage.md),
signed voxel addressing is in
[`docs/tsdf-voxel-addressing.md`](docs/tsdf-voxel-addressing.md),
single-observation voxel evaluation is in
[`docs/tsdf-voxel-contribution.md`](docs/tsdf-voxel-contribution.md),
surface extraction is in [`docs/surface-points.md`](docs/surface-points.md),
reference triangle meshing is in
[`docs/triangle-mesh.md`](docs/triangle-mesh.md), and overall status is in
[`docs/roadmap.md`](docs/roadmap.md).
