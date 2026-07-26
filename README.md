# SpatialForge

SpatialForge is a standalone spatial mapping and localization engine. It will
turn calibrated indoor scans into metric, semantic, localizable maps while
remaining independent of navigation products such as VoiceGIS.

This repository currently implements only the first foundation:

- a versioned, folder-backed `ScanSession` (`.vgsession`) contract;
- validation for calibration, timestamps, file references, depth scale, IMU
  samples, and rigid camera poses;
- deterministic offline replay around RGB observations; and
- a tiny synthetic RGB-D + IMU + known-pose fixture.

Reconstruction, SLAM, map packages, mobile capture, and the visual inspector
are deliberately not part of this milestone.

## Run the first milestone

Python 3.11 or newer is required. There are no third-party runtime or test
dependencies.

```powershell
python -m spatialforge scan validate tests/fixtures/minimal.vgsession
python -m spatialforge scan replay tests/fixtures/minimal.vgsession
python -m unittest discover -s tests -v
```

The format and coordinate conventions are documented in
[`docs/scan-session-v0.md`](docs/scan-session-v0.md).

