# One-observation camera-to-surface block rays

This checkpoint traces the pixel-center rays of exactly one plan-selected,
prepared observation through the TSDF block grid:

```powershell
.\.venv\Scripts\python.exe -m spatialforge reconstruct tsdf-block-context-observation-rays `
  outputs\progress-blocks.sftplan `
  tests\fixtures\minimal.vgsession `
  --observation-sequence 0
```

It is a read-only coverage diagnostic. For every positive finite depth pixel,
it traces the closed centerline segment from the copied camera origin to that
pixel's measured surface point. It reports which block coordinates the
centerlines visit, which already exist in the source plan, and which are
unplanned.

This is deliberately narrower than free-space planning. A pixel-center line
does not conservatively cover the whole solid angle represented by a nearest-
pixel depth sample, and visiting a block does not prove that every voxel in
that block is free. The operation does not expand the `.sftplan`, create or
allocate blocks, mutate TSDF storage, aggregate multiple observations, fuse a
voxel, or write an artifact.

## Public API and immutable records

The public API is:

```python
receipt = trace_tsdf_observation_block_rays_from_context(
    plan,
    context,
    observation_sequence,
)
```

`plan` must be a strict-loaded `TsdfBlockPlan`. `context` must be its matching
immutable `TsdfReplayDepthContext`, and `observation_sequence` must be one of
the context's complete canonical frame-stride selection. The caller cannot
provide a depth frame, pose, camera, block subset, or alternative grid.

The frozen `TsdfObservationBlockRayTraceReceipt` stores:

```text
source_plan_digest_sha256
replay_digest_sha256
frame_stride
total_observations
source_plan_block_indices
observation_sequence
observation_status
block_resolution
block_extent_m
camera_origin_world_m
image_size
ray_receipts
covered_block_indices
existing_plan_block_indices
unplanned_block_indices
```

It derives pixel, traversed-ray, invalid-depth, block-visit, duplicate-visit,
maximum-blocks-per-ray, retained-outcome, surface-endpoint-block, nonterminal-
block, and prepared-depth-access counts.

The retained frame stride and total-observation count make the selected-
sequence invariant independently checkable by the frozen receipt. The retained
canonical source-plan block tuple likewise lets receipt construction validate
the exact existing/unplanned membership partition instead of trusting a
caller-supplied classification.

A ready observation retains one frozen `TsdfObservationBlockRayReceipt` for
every image pixel in row-major order: increasing `u`, then increasing `v`.
Each child stores:

```text
pixel_uv
status
camera_origin_world_m
block_extent_m
measured_depth_m
surface_world_m
surface_block_index
block_indices
```

Every child retains the parent's camera origin and block extent, validates its
own DDA path from those values, and must match that shared geometry when the
parent validates the row-major child tuple. A traversed receipt also retains
the positive finite measured depth, world-space surface endpoint, endpoint
block, and complete ordered block path. A `depth-invalid` receipt has no
invented measured depth, endpoint, surface block, or block path.

Prepared observations missing depth or pose return a successful parent
receipt with no pixel receipts or covered blocks. A `missing-depth`
observation can retain its copied camera origin because its pose exists.
Pose-missing observations cannot retain an origin. Missing input is therefore
diagnosed without inventing rays or free space. For these zero-ray receipts,
the CLI reports `centerline_ray_coverage_computed: no`; the `yes` value in the
fixture proof below is specific to its ready observation.

## Pixel-center geometry

For one positive finite metric depth `z`, the pixel center `(u, v)` is
back-projected with the aligned pinhole calibration:

```text
x_camera = (u - cx) * z / fx
y_camera = (v - cy) * z / fy
z_camera = z
```

The prepared observation's copied `T_world_camera` transforms that point to
the measured world-space surface endpoint. The camera origin is the transform
translation:

```text
(T_world_camera[0,3], T_world_camera[1,3], T_world_camera[2,3])
```

The traced parameter interval is the closed segment:

```text
p(t) = camera_origin + t * (surface_point - camera_origin),  0 <= t <= 1
```

The start-owned camera block and endpoint-owned surface block are both
retained. The trace stops at the measured surface. It does not extend through
the object, add the TSDF truncation distance, or infer any block behind an
invalid or absent measurement.

## Exact thin-DDA block rule

The block grid comes entirely from the source plan. It is anchored at world
zero, uses the plan's `8 * voxel_size_m` block extent, and assigns coordinates
to lower-inclusive, upper-exclusive blocks.

Each closed segment uses a thin three-dimensional DDA:

1. Retain the block that owns the exact camera origin.
2. Compute the next block-boundary parameter independently for X, Y, and Z.
3. Select the exact smallest binary64 parameter without an epsilon.
4. If multiple axes have exactly equal parameters, advance all tied axes at
   once and retain only the diagonally entered block.
5. Continue until the endpoint-owned block has been retained.

This simultaneous tie rule excludes blocks that the zero-width centerline
only touches along a side or corner at one parameter value. The algorithm is
therefore **not a geometric supercover**. It is deterministic thin centerline
coverage. A future conservative nearest-pixel coverage rule would have to
cover the pixel footprint or viewing cone and prove which voxel centers can
sample that measurement.

Per-ray paths remain in traversal order. Their union is deduplicated and
sorted in the plan's canonical X-fastest, then Y, then Z order. The receipt
partitions that exact union into:

```text
existing_plan_block_indices = covered blocks in plan.active_blocks
unplanned_block_indices     = covered blocks absent from plan.active_blocks
```

The partition is information only. Unplanned coordinates are never inserted.

`nonterminal_block_indices` is the union of every block occurring before the
final position of at least one ray path. The name is deliberately structural:
one coordinate can be nonterminal for one ray and the surface endpoint of
another. It does not certify every point or voxel in a whole block as visible
free space. Similarly, a surface-endpoint block can contain both free-side and
behind-surface voxels.

## Provenance and source-I/O boundary

The API validates plan, context, camera, selection, block geometry, and digest
provenance before tracing. It accepts no `ScanSession` and performs no replay,
source hashing, filesystem I/O, or depth decoding. A ready trace necessarily
reads the immutable metric depth already retained by the context.

The CLI performs earlier setup: it strict-loads the plan, replay-verifies the
session while constructing the context, and decodes each ready selected depth
frame once. The later trace uses that completed construction-time snapshot.
Rebuild the context when current folder freshness is required.

The source plan is used only for provenance, grid geometry, observation
selection, and the existing/unplanned partition. Its
`activation.free_space_rule` remains `not-planned`, its active tuple remains
unchanged, and no replacement plan is serialized.

## Bounded diagnostic workload

The retained outcome count is:

```text
pixel receipts + traversed block visits
```

`MAX_TSDF_OBSERVATION_BLOCK_RAY_OUTCOMES` is `262,144`. The implementation
preflights the image pixel count and, before each valid ray, adds a
deterministic upper bound on that ray's visits. If the accumulated bound would
exceed the cap, the diagnostic is rejected before tracing that ray. Exact tied
crossings can make the eventual path shorter than the bound, so this is a
conservative admission check, not a claim that every bounded visit is
required. The implementation also retains the existing 100,000 unique-block
reference limit. A limit failure returns no partial receipt; there is no
storage or artifact to roll back.

For `P` image pixels and `V` retained per-ray block visits, runtime and the
diagnostic transcript are `O(P + V)`. These caps do not make this a scalable,
streaming, parallel, GPU, or production coverage planner.

## Exact fixture proof

The committed observation sequence zero has a camera origin at `(0, 0, 0)`
and four positive finite depth pixels. Their row-major measured endpoints are:

```text
pixel (0, 0): (1.0,  0.25,  0.25)
pixel (1, 0): (1.0, -0.25,  0.25)
pixel (0, 1): (1.0,  0.25, -0.25)
pixel (1, 1): (1.0, -0.25, -0.25)
```

The four paths contain `2`, `3`, `3`, and `3` blocks. Run the command above
after creating `outputs\progress-blocks.sftplan` with the README's block-plan
command. The key output is:

```text
observation: sequence=0 status=ready
camera_origin_world_m: (0.000000000, 0.000000000, 0.000000000)
image: width=2 height=2
pixel_outcomes: total=4 traversed=4 depth_invalid=0
ray_block_visits: total=11 unique=8 duplicate=3 maximum_per_ray=3
coverage_blocks: total=8 nonterminal=4 surface_endpoint=4
coverage_partition: existing_plan=8 unplanned=0
coverage_order: canonical-x-fastest
first_covered_block: (0, -1, -1)
last_covered_block: (1, 0, 0)
trace_session_replay: no
trace_source_io: no
trace_depth_decoding: no
trace_prepared_depth_access: yes
trace_workload: retained_outcomes=15 maximum=262144
coverage_scope: one-prepared-observation-only
visibility_rule: positive-finite-depth-stops-at-measured-surface
block_traversal_rule: closed-half-open-grid-thin-dda-simultaneous-exact-ties
ray_traversal_performed: yes
centerline_ray_coverage_computed: yes
conservative_nearest_pixel_free_space_coverage_proven: no
multiple_observation_coverage_computed: no
plan_expanded: no
missing_blocks_created: no
storage_allocated: no
storage_mutated: no
full_fusion_performed: no
artifact_written: no
```

The 15 outcomes are four pixel receipts plus eleven block visits. Deduplication
leaves eight coordinates. All eight happen to be present in this tiny plan's
surface/truncation band, so `unplanned=0`; this fixture does **not** demonstrate
plan expansion. Focused tests use longer rays to prove that absent coordinates
are reported as unplanned without modifying the source plan.

## Precisely deferred next phases

These ordered one-observation receipts are now aggregated across the context's
complete selected-observation tuple by
[`tsdf-plan-block-ray-survey.md`](tsdf-plan-block-ray-survey.md). That phase
still keeps coverage computation separate from choosing and publishing an
expanded plan.

The conservative nearest-pixel coverage this file says it does not provide is
now defined for one pixel in
[`tsdf-pixel-footprint-coverage.md`](tsdf-pixel-footprint-coverage.md), and
unioned across a whole observation in
[`tsdf-observation-footprint.md`](tsdf-observation-footprint.md). Those rules
cover each pixel's whole sampling wedge and are tested to contain the
centreline coverage traced here.

Later checkpoints remain for:

1. the companion per-voxel sampling proof and any frustum, visibility,
   occlusion, or culling policy;
2. combining approved coverage coordinates with the surface/truncation plan,
   allocating missing blocks, and preserving per-observation provenance;
3. explicit idempotency and resumable/nonempty fusion over that chosen domain;
4. complete fusion diagnostics, persistent block-backed TSDF artifacts,
   normalization, and sparse surface/mesh consumers;
5. scalable streaming, optimized CPU, parallel, or GPU execution; and
6. robust filtering, real-dataset accuracy, production meshing, structural
   mapping, Inspector work, pose estimation, SLAM, semantics, localization,
   and `SpatialMapPackage` export.
