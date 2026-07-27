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
  and
- deterministic zero-crossing surface-point extraction from the TSDF.

Automatic TSDF volume sizing, optimized fusion, triangle meshing, SLAM, map
packages, mobile capture, and the visual inspector are deliberately not
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

For a visual check, open `outputs/minimal.ply` from the point-cloud command in
a PLY viewer. Commands refuse to overwrite outputs, so delete an old diagnostic
or choose a new filename before rerunning it.

The format and coordinate conventions are documented in
[`docs/scan-session-v0.md`](docs/scan-session-v0.md). TUM-specific conversion
rules are in [`docs/tum-import.md`](docs/tum-import.md), and the current
reconstruction steps are in
[`docs/known-pose-point-cloud.md`](docs/known-pose-point-cloud.md) and
[`docs/reference-tsdf.md`](docs/reference-tsdf.md). Surface extraction is in
[`docs/surface-points.md`](docs/surface-points.md), and overall status is in
[`docs/roadmap.md`](docs/roadmap.md).
