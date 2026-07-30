# SpatialForge roadmap

This is the current implementation status against the original architecture.
The project is intentionally advancing through small, testable checkpoints.

## Working now

- `ScanSession v0.1` folder contract with calibrated RGB, depth, IMU, and known
  pose streams.
- Strict validation of timestamps, paths, depth scale, camera models, rigid
  transforms, and coordinate conventions.
- Deterministic replay, exact timestamp association, and SHA-256 input digests.
- Deterministic TUM RGB-D folder import with known-pose conversion.
- Known-pose RGB-D back-projection to colored ASCII PLY.
- Fixed-bounds projective TSDF integration with deterministic diagnostics.
- Fixed-bounds sparse in-memory TSDF accumulation with dense traversal,
  byte-identical dense-reference results, and the unchanged artifact contract.
- Deterministic planning of zero-anchored 8 x 8 x 8 candidate voxel blocks around
  known-pose depth surfaces, with a separate replay-bound diagnostic artifact.
- Strict immutable in-memory loading of `.sftplan` diagnostics and read-only
  verification of their session IDs, replay digests, and replay-derived frame
  metadata.
- Replay-matched allocation of canonical candidate coordinates into temporary
  zeroed `8 x 8 x 8` float64-sum and uint32-weight block buffers.
- Deterministic signed global-voxel addressing into canonical block rows,
  local coordinates, and flat storage offsets, with read-only sparse misses.
- Read-only evaluation of one known-pose depth observation at one planned
  voxel, including world-center transformation, aligned pinhole projection,
  nearest-depth sampling, signed-distance/truncation classification, and an
  immutable weight-one contribution or skip diagnostic.
- Plan- and replay-bound application of one accepted immutable contribution
  to exactly one addressed temporary TSDF block slot, with uint32 overflow
  preflight, replay-bracketed rollback, and a frozen before/after receipt.
- Known-pose depth AABB inference with truncation padding, outward global-grid
  snapping, and the same bounded TSDF integrator.
- TSDF exact-zero and sign-changing-edge extraction to XYZ surface-point PLY.
- Fully observed TSDF-cell extraction to a deterministic, indexed triangle PLY
  using a fixed six-tetrahedron reference split.
- CLI error handling, overwrite protection, synthetic fixtures, documentation,
  and automated numerical regression tests.

## Remaining in known-pose reconstruction

These complete the architecture's first geometric proof before pose estimation
or SLAM:

1. traverse contribution evaluation and application across replay-selected
   observations and planned block addresses, decide and plan
   camera-to-surface free-space coverage, prevent duplicate observation
   application, and report complete block-fusion diagnostics;
2. culled, scalable sparse traversal, larger-volume artifacts and consumers,
   and an optimized backend suitable for full sequences;
3. robust depth/pose outlier filtering and configurable production bounds;
4. full TUM/ARKitScenes sample runs and geometry accuracy reports;
5. production mesh refinement: exact-zero cells, normals, connected-component
   and quality validation, and an optimized extraction backend;
6. gravity/floor alignment, floor and wall candidates, and openings; and
7. a top-down/3D Inspector view.

## Later major milestones

1. Define and export `SpatialMapPackage`, including quality, uncertainty, and
   provenance.
2. Visual keyframes, features, matching, robust pose estimation, and trajectory
   evaluation against known poses.
3. IMU preintegration, visual-inertial optimization, factor graph, loop
   closure, and place recognition.
4. Structural mapping: floors, walls, openings, rooms, corridors, walkable
   regions, and coverage.
5. Semantic mapping: objects, doors, stairs, lifts, signs, OCR, confidence, and
   multi-view provenance.
6. Visitor localization: retrieval, local matching, 2D/3D correspondences,
   PnP, pose verification, tracking, and relocalization.
7. Multi-session and multi-floor alignment, connector handling, and change
   detection.
8. Mobile scan capture, active coverage guidance, benchmarks, and the eventual
   VoiceGIS import adapter.

SLAM, semantics, and localization should not begin until the known-pose
reconstruction path is numerically and visually trustworthy.

## How to verify progress

Run:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The suite must finish with `OK`.

Then follow the point-cloud, fixed/automatic/sparse TSDF, candidate-block,
strict plan verification, empty block allocation, signed voxel addressing,
single-observation voxel contribution, single-slot contribution application,
surface-point, and triangle-mesh commands in the repository README. Their
deterministic hashes, inferred bounds, dense/sparse parity, planned block
coordinates, replay binding, zero-state storage layout, signed address
round-trips, proposed and applied contribution deltas, before/after receipts,
fusion weights, signed values, exact coordinates, topology, winding, and
boundary counts provide checks independent of visual appearance.
