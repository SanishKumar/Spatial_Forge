# SpatialForge roadmap

This is the current implementation status against the original architecture.
The project is intentionally advancing through small, testable checkpoints.

For a shorter read: [`status.md`](status.md) covers where things stand and
what the open risks are; [`working-method.md`](working-method.md) covers how a
checkpoint is added and the invariants everything is pinned to.

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
  traversals plus traversal of the complete existing plan block set; the
  deliberately redundant session-backed one-voxel traversal remains available
  as a reference path.
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
- Context-backed traversal of every existing canonical active-block row in the
  plan, retaining complete nested block/voxel/observation receipts, enforcing a
  262,144-outcome diagnostic cap, and restoring the preflight-proven all-zero
  planned storage after a caught later-block failure.
- Context-backed read-only tracing of the pixel-center camera-to-measured-
  surface rays for exactly one plan-selected prepared observation, retaining
  row-major thin-DDA paths, deduplicating them in canonical block order, and
  partitioning existing versus unplanned coordinates without expanding the
  plan, allocating storage, or fusing voxels.
- Context-backed aggregation of those one-observation ray transcripts across
  the plan's complete canonical selected-observation tuple, re-deriving the
  combined canonical coverage union, its existing/unplanned partition, and
  per-block observation-support counts from the retained children, under a
  preflighted and accumulated 262,144-outcome cap, still without approving
  coverage, expanding the plan, allocating storage, or fusing voxels.
- Conservative nearest-pixel footprint coverage for one prepared pixel,
  covering the half-open sampling square's whole apex-to-measured-depth wedge
  with a six-plane block superset that provably contains every wedge point's
  owning block and its own re-derived centreline path, partitioned into
  existing and unplanned coordinates without a per-voxel, occlusion, or
  culling rule and without expanding the plan or fusing voxels.
- Per-voxel sampling classification for one signed global voxel centre and one
  prepared observation, separating observed free space from the surface band
  and from occluded space, accepting unplanned voxels without storage, proved
  to agree with the reference contribution evaluator's projection and accept
  rule and to lie inside its own pixel's conservative footprint coverage.
- Cross-view resolution of one voxel over the complete selected-observation
  tuple, ranking surface-band evidence above free space above occlusion above
  absence so that occlusion and missing input never become free space, and
  reproducing the fusing traversal's exact weight, float64 sum, and applied
  observation sequence while explaining which evidence produced them.
- Conservative footprint coverage unioned across every pixel of one prepared
  observation in canonical row-major order, retaining each pixel's transcript,
  re-deriving the canonical union and its existing/unplanned partition, and
  proved to contain the independent centreline ray union, without approving
  coverage, expanding the plan, allocating storage, or fusing voxels.
- The same conservative footprint coverage unioned across the plan's complete
  canonical selected-observation tuple, retaining every observation
  transcript, re-deriving the canonical union, its existing/unplanned
  partition and per-block observation support, and proved to contain the
  independent all-observation centreline survey union.
- Block-wide cross-view resolution of all 512 voxels of one signed block,
  planned or not, separating surface, carvable free space, occlusion and
  absence, and proved voxel-by-voxel to reproduce the fusing block traversal's
  exact weights and float64 sums across every active block.
- Cross-view resolution swept across the whole surveyed coverage domain,
  binding the conservative footprint survey's provenance to the verdict
  rule, producing the whole-scan carvable free-space set split by plan
  membership, and reproducing the fusing plan traversal's exact weight
  total, observed-voxel count and maximum weight.
- Read-only plan expansion proposal approving every covered block that
  holds at least one observed voxel, pruning grazed-but-unobserved
  coverage, merging the approved set with the source plan without ever
  removing a planned block, and reporting the proposed block and
  voxel-slot deltas without serializing a plan.
- Serialization of that approved set as a new `.sftplan` carrying the
  conservative-nearest-pixel-footprint free-space rule, the source plan
  digest and the approval rule, never overwriting or modifying its
  source, and proved to strict-load, replay-verify and allocate storage.
- Resumable, idempotent block-row fusion tracked by an in-memory ledger,
  replacing the blanket empty-storage guard with a per-row one, skipping
  already-fused rows, restoring only the rows a failed pass touched, and
  proved byte-identical to the one-shot traversal at every chunk size.
- Frame-major fusion at (row, observation) granularity, where each row
  records a canonical-order prefix of the selection so a row can hold a
  partial set of frames and absorb the rest later, restoring exact
  pre-pass bytes on failure and staying byte-identical to the one-shot
  traversal at every pair chunk size.
- Vectorised read-only evaluation of all 512 voxels of one planned block
  against one prepared observation in a single float64 NumPy pass, returning
  immutable per-voxel status, sum-delta and weight-delta arrays in canonical
  local-flat order, proved bit-identical to the scalar contribution evaluator
  on every fixture block and on off-grid room-scan blocks, and matching the
  fusing plan traversal's accepted total, without applying a contribution,
  touching storage, or performing replay, hashing, source I/O or depth
  decoding.
- Field-driven fusion of one planned block, applying one vectorised field per
  selected observation to the block's storage row in canonical observation
  order with two array additions each, under an empty-row precondition,
  uint32 overflow preflight, weight-envelope and finiteness checks,
  byte-verified writes and whole-row rollback, with a receipt that replays its
  own retained fields, proved byte-identical to the voxel-by-voxel block
  traversal on every fixture block and on room-scan blocks, and to the
  one-shot plan traversal across a whole fused plan.
- First real-sensor validation: 99 frames of TUM `rgbd_dataset_freiburg1_xyz`
  fused at 30 mm voxels and scored against 98 held-out frames the fusion never
  saw, giving a 9.3 mm median and 20.4 mm rms trilinear surface error with
  99.8% of held-out depth landing in observed voxels, plus a committed
  reproducible report tool.
- First non-degenerate validation: a seeded 20-frame 64x48 room scan with
  off-grid surfaces and 4 mm depth noise, recovering the known walls to
  sub-centimetre mean bias at 40 mm voxels and producing a 60,072-triangle
  mesh, with the one-shot traversal's scan-scale refusal pinned as a test.
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

1. vectorise the block planner, which at about 2 seconds per 640x480 frame is
   now the pipeline's dominant cost, using the existing per-pixel path as its
   reference;
2. stream or window the replay/depth context, which at 640x480 hits its
   512 MB retained-depth ceiling after roughly 208 frames and so cannot hold a
   full sequence;
3. drive the block and observation ledgers from field fusion, so resumable,
   idempotent and partial-frame fusion runs at the vector path's rate too and
   a plan-wide entry point exists;
4. persist a block-backed TSDF artifact together with its fusion ledger, so
   resumption survives process exit, and connect normalization plus sparse
   surface/mesh consumers;
5. add culled, scalable sparse traversal suitable for full sequences;
6. refine production meshing with exact-zero cells, normals,
   connected-component and quality validation, and optimized extraction, which
   real data already requires: the fixed six-tetrahedron split produces
   non-manifold vertices on the TUM volume and the mesher correctly refuses it;
7. add confidence and sensor-dependent weighting, robust depth/pose outlier
   filtering, the visibility/culling policy, and configurable production
   bounds;
8. run further real datasets such as ARKitScenes and publish accuracy reports;
9. add gravity/floor alignment, floor and wall candidates, and openings; and
10. add a top-down/3D Inspector view.

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
single-selected-block traversal, context-backed existing-plan traversal,
one-observation context-backed block-ray tracing, complete selected-observation
block-ray survey, conservative one-pixel footprint coverage, per-voxel
sampling classification, cross-view voxel and block resolution,
one-observation and whole-scan footprint coverage, whole-scan carvable
free space, plan-expansion proposal, expanded-plan writing, resumable
ledgered fusion, frame-major observation fusion, vectorised block
contribution evaluation, field-driven block fusion,
surface-point, and triangle-mesh commands in the repository
README. Their
deterministic hashes, inferred bounds, dense/sparse parity, planned block
coordinates, replay binding, zero-state storage layout, signed address
round-trips, proposed and applied contribution deltas, before/after receipts,
selected/evaluated/applied traversal counts, canonical 512-address ordering,
selected-block isolation, complete plan-row ordering, bounded retained-outcome
workload, whole-storage rollback scope, immutable metric-depth layout,
row-major pixel outcomes, exact thin-DDA tie handling, canonical covered-block
partitions, per-block observation support, fusion weights, signed values,
exact coordinates, topology,
winding, and boundary counts provide checks independent of visual appearance.
