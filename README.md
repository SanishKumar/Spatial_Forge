# SpatialForge

SpatialForge is a standalone spatial mapping and localization engine. It will
turn calibrated indoor scans into metric, semantic, localizable maps while
remaining independent of navigation products such as VoiceGIS.

This repository currently implements the first two narrow foundations:

- a versioned, folder-backed `ScanSession` (`.vgsession`) contract;
- validation for calibration, timestamps, file references, depth scale, IMU
  samples, and rigid camera poses;
- deterministic offline replay around RGB observations; and
- an extracted TUM RGB-D folder importer with known-pose support.

Reconstruction, SLAM, map packages, mobile capture, and the visual inspector
are deliberately not implemented yet.

## Run the first milestone

Python 3.11 or newer is required. There are no third-party runtime or test
dependencies.

```powershell
python -m spatialforge scan validate tests/fixtures/minimal.vgsession
python -m spatialforge scan replay tests/fixtures/minimal.vgsession
python -m unittest discover -s tests -v
```

Import the committed tiny TUM-layout fixture:

```powershell
python -m spatialforge scan import-tum `
  tests/fixtures/tum/rgbd_dataset_freiburg1_tiny `
  outputs/tum-tiny.vgsession
python -m spatialforge scan validate outputs/tum-tiny.vgsession
```

The format and coordinate conventions are documented in
[`docs/scan-session-v0.md`](docs/scan-session-v0.md). TUM-specific conversion
rules are in [`docs/tum-import.md`](docs/tum-import.md).

