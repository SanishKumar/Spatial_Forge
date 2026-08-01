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
- Deterministic traversal of all replay observations selected by the plan for
  one addressed voxel, with evaluate-all-then-apply ordering, stable skip
  accounting, an empty-target duplicate guard, and traversal-wide rollback
  after caught application-phase failures.
- Plan- and replay-bound construction of a frozen in-memory
  `TsdfReplayDepthContext` for every selected observation, decoding each ready
  aligned-depth frame exactly once into immutable C-contiguous float64 metric
  storage. It is consumed by separate scalar context evaluation and
  application APIs plus context-backed one-voxel and one-selected-block
  traversals; the deliberately redundant session-backed one-voxel traversal
  remains available as a reference path.
- Plan-bound read-only evaluation of one selected replay/depth-context
  observation at one planned voxel, using the copied pose and immutable metric
  depth without evaluation-time replay hashing or depth decoding and returning
  the existing frozen contribution/skip contract without mutation.
- Context-bound guarded application of one accepted prepared contribution to
  exactly one addressed temporary slot, preserving plan/context/storage
  provenance, uint32 and accumulator preflight, exact scalar-write checks, an
  immutable receipt, and caught-failure rollback without application-time
  session replay, source I/O, or depth access.
- Context-backed traversal of every prepared plan-selected observation for one
  addressed voxel, preserving evaluate-all-before-apply ordering, canonical
  sequence order, stable skips, the empty-target duplicate guard, sequential
  float64 accumulation, and whole-target caught-failure rollback without
  traversal-time replay, source I/O, or depth decoding.
- Context-backed traversal of all 512 canonical X-fastest addresses in one
  caller-selected planned block, retaining every per-voxel receipt, requiring
  an empty selected row before the first child, and restoring that complete
  row after a caught later-child failure without targeting a second block.
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

1. traverse the plan's existing canonical block rows with one shared
   replay/depth context while preserving deterministic block/address order,
   per-block and per-voxel transcripts, and bounded failure handling;
2. separately decide and plan camera-to-surface free-space coverage, replace
   the empty-target guard with an explicit cross-call
   observation/idempotency policy, and define complete fusion diagnostics;
3. culled, scalable sparse traversal, larger-volume artifacts and consumers,
   and an optimized backend suitable for full sequences;
4. robust depth/pose outlier filtering and configurable production bounds;
5. full TUM/ARKitScenes sample runs and geometry accuracy reports;
6. production mesh refinement: exact-zero cells, normals, connected-component
   and quality validation, and an optimized extraction backend;
7. gravity/floor alignment, floor and wall candidates, and openings; and
8. a top-down/3D Inspector view.

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
single-voxel selected-observation traversal, immutable replay/depth context,
context-backed single-observation contribution, context-bound single-slot
application, context-backed single-voxel traversal, context-backed
single-selected-block traversal, surface-point, and triangle-mesh commands in
the repository
README. Their
deterministic hashes, inferred bounds, dense/sparse parity, planned block
coordinates, replay binding, zero-state storage layout, signed address
round-trips, proposed and applied contribution deltas, before/after receipts,
selected/evaluated/applied traversal counts, canonical 512-address ordering,
selected-block isolation, immutable metric-depth layout, fusion weights,
signed values, exact coordinates, topology, winding, and boundary counts
provide checks independent of visual appearance.
