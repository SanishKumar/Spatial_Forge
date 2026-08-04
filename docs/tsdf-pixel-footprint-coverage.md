# Conservative nearest-pixel footprint coverage

This checkpoint answers a question the centerline ray checkpoints could not:
which blocks can contain voxel centres that one depth pixel's measurement is
allowed to constrain?

```powershell
.\.venv\Scripts\python.exe -m spatialforge reconstruct tsdf-block-context-pixel-footprint `
  outputs\progress-blocks.sftplan `
  tests\fixtures\minimal.vgsession `
  --observation-sequence 0 `
  --pixel 1 1
```

It is a read-only coverage diagnostic for exactly one pixel of exactly one
plan-selected prepared observation. It covers that pixel's whole sampling
wedge, not just its centreline, and reports which blocks already exist in the
source plan and which are unplanned. It does not expand the `.sftplan`, create
or allocate blocks, mutate TSDF storage, fuse a voxel, or write an artifact.

## Why the centerline was not enough

[`tsdf-observation-block-rays.md`](tsdf-observation-block-rays.md) traces a
zero-width line through each pixel centre. That line is a deterministic
discovery aid, but it is not a coverage proof: a pixel represents an angular
footprint, and a voxel centre well away from the centreline can still be the
one that samples this pixel.

This checkpoint replaces the line with the region the sampling rule actually
implies, and the receipt proves the relationship by re-deriving the centreline
path and requiring coverage to contain it. On the committed fixture, three of
the four pixels cover strictly more blocks than their centreline — 8 versus 3
for pixel `(1, 1)`.

## The exact sampling rule this covers

The TSDF evaluator samples depth with nearest-pixel rounding:

```python
pixel_u = math.floor(projected_u + 0.5)
pixel_v = math.floor(projected_v + 0.5)
```

So pixel `(u, v)` owns exactly the half-open image square:

```text
[u - 0.5, u + 0.5) x [v - 0.5, v + 0.5)
```

The set of world points that pixel `(u, v)` can sample, for a measurement that
stops at the measured surface, is therefore the **sampling wedge**:

```text
W = { p : 0 < z_camera(p) <= measured_depth,
          projection of p lies in that half-open square }
```

`W` is the convex pyramid with apex at the copied camera origin and base the
four image-square corners back-projected at `z_camera = measured_depth`. The
receipt retains those four world corners in the canonical order
`(u-, v-)`, `(u+, v-)`, `(u+, v+)`, `(u-, v+)`.

Because back-projection at a fixed camera depth is affine in `(u, v)`, the
centroid of the four corners is exactly the pixel-centre point the ray
checkpoint uses as its segment endpoint. The receipt uses that identity to
re-derive its own centreline instead of storing an unrelated path.

## The coverage rule and what "conservative" means

The wedge is bounded by six outward-oriented planes: four side planes through
the apex and one image-square edge each, one far cap at the measured depth,
and one near cap at the apex. The near cap matters: four planes through a
common apex describe a *double* cone, so without it the mirror wedge behind
the camera would be admitted.

Candidate blocks are the axis-aligned block range spanned by the apex and the
four corners. A candidate is rejected only when the whole closed block cell
lies strictly outside one plane. Everything else is retained.

This is deliberately a **superset**, and the direction of the error is the
part that matters:

- **No false negatives.** If any point of `W` lies in a block's half-open
  cell, that point satisfies every plane, so the block is never rejected.
- **Possible false positives.** Testing planes one at a time can retain a
  block that lies outside the wedge near an edge or corner but not outside any
  single plane. Blocks that touch the wedge only on a zero-volume boundary are
  also retained, because block cells are half-open and a boundary point is
  genuinely owned by one of them.

For allocation this is the safe direction: a conservative rule may allocate a
block that turns out to carry no contribution, but it never silently drops one
that could.

The tests check the no-false-negative direction directly rather than only
pinning numbers. A deterministic lattice of points inside the wedge — nine
depth steps by nine by nine image offsets across the half-open square — is
back-projected, and every sampled point's owning block must appear in the
covered tuple, at both 1.0 m and 3.0 m, for all four fixture pixels.

## Public API and immutable record

```python
receipt = evaluate_tsdf_pixel_footprint_coverage_from_context(
    plan,
    context,
    observation_sequence,
    pixel_uv,
)
```

`plan` must be a strict-loaded `TsdfBlockPlan`, `context` its matching
immutable `TsdfReplayDepthContext`, `observation_sequence` one of the
context's canonical frame-stride selection, and `pixel_uv` a pixel inside the
calibrated image. The caller cannot supply a depth frame, pose, camera, wedge,
block subset, or alternative grid.

The frozen `TsdfPixelFootprintCoverageReceipt` stores:

```text
source_plan_digest_sha256      observation_status
replay_digest_sha256           pixel_uv
frame_stride                   status
total_observations             block_resolution
source_plan_block_indices      block_extent_m
observation_sequence           image_size
camera_origin_world_m          candidate_min_block_index
measured_depth_m               candidate_max_block_index
footprint_corners_world_m      covered_block_indices
centerline_block_indices       existing_plan_block_indices
                               unplanned_block_indices
```

Candidate, covered, rejected, centreline, footprint-only, widening, and
prepared-depth-access counts are derived on access.

Construction re-derives the geometry from the retained apex, corners, and
block extent: the candidate range, the covered tuple, and the centreline path
must all match, coverage must contain the centreline, and the
existing/unplanned split must reproduce the intersection with and difference
from the retained source-plan tuple.

Two retained values are deliberately *not* re-derivable, matching the existing
ray receipt's contract. `measured_depth_m` is a metre measurement, and the
receipt keeps no intrinsics or pose, so only its sign and finiteness are
checked; the coverage proof rests on the apex and corners instead. Reversing
the corner tuple describes the identical wedge and is accepted for the same
reason.

## Statuses

`covered` is the only status that retains geometry. `depth-invalid` is
returned for a ready observation whose pixel has a non-positive or non-finite
depth — no wedge is invented and no free space is claimed. `missing-depth`,
`missing-pose`, and `missing-depth-and-pose` mirror the prepared observation
status; a pose-bearing observation may still retain its copied camera origin,
and a pose-missing one may not. Every non-`covered` status retains no depth,
corners, centreline, candidate range, or coverage at all.

## Provenance and source-I/O boundary

The API validates plan, context, camera, selection, pixel bounds, block
geometry, and digest provenance before computing anything. It accepts no
`ScanSession` and performs no replay, source hashing, filesystem I/O, or depth
decoding; a covered evaluation reads only the immutable metric depth already
retained by the context. The source plan supplies provenance, grid geometry,
observation selection, and the existing/unplanned partition; its
`activation.free_space_rule` remains `not-planned` and its active tuple is
unchanged.

## Bounded diagnostic workload

`MAX_TSDF_PIXEL_FOOTPRINT_CANDIDATE_BLOCKS` is `262,144`. The candidate block
count is computed from the apex/corner range and rejected before any plane
test if it exceeds the cap, so a distant measurement on a fine voxel grid
fails fast instead of enumerating. The existing 100,000 unique-block reference
limit also applies to the covered tuple. A limit failure returns no partial
receipt; nothing is allocated or mutated, so there is no rollback to perform.

Runtime is `O(candidate blocks)` per pixel. This is a single-pixel reference
rule, not a scalable, streaming, parallel, or GPU coverage planner.

## Exact fixture proof

Observation zero has its camera at the world origin and four pixels measuring
1.0 m, with a 1.0 m block extent. For pixel `(1, 1)`:

```text
pixel: uv=(1, 1) image=2x2
footprint_status: covered
measured_depth_m: 1.000000000
sampling_rule: nearest-pixel-half-open-unit-square
wedge_rule: apex-to-measured-depth-convex-pyramid
coverage_rule: conservative-plane-superset-of-half-open-cells
candidate_blocks: total=8 min=(0, -1, -1) max=(1, 0, 0)
coverage_blocks: covered=8 rejected=0
centerline_blocks: total=3 footprint_only=5
centerline_contained_in_coverage: yes
widens_centerline_coverage: yes
coverage_partition: existing_plan=8 unplanned=0
conservative_nearest_pixel_footprint_coverage_computed: yes
per_voxel_sampling_proof_computed: no
occlusion_rule_defined: no
visibility_culling_rule_defined: no
multi_pixel_coverage_computed: no
multi_observation_coverage_computed: no
plan_expanded: no
storage_allocated: no
full_fusion_performed: no
artifact_written: no
```

Every fixture coordinate lands exactly on a block boundary, so this run
rejects no candidate. That makes it a poor test of the plane rule on its own,
so a focused test measures the same pixel at 3.0 m: 36 candidates, 31 covered,
5 rejected — the five near-apex blocks whose `y` or `z` band the wedge cannot
reach — with 13 unplanned coordinates reported while the plan's active tuple
stays byte-identical.

## What this still does not prove

Covering a block does not mean every voxel in it is free space, and this
checkpoint deliberately stops short of the rest of the free-space rule:

- there is no per-voxel proof yet — that is the natural companion rule, and it
  is what actually decides a voxel's contribution;
- there is no occlusion policy: one pixel's wedge says nothing about a surface
  another pixel or another view places inside it;
- there is no frustum, visibility, or culling policy across pixels or views;
- invalid and absent depth remain unknown, never free; and
- nothing here approves coverage or expands a plan.

## Precisely deferred next phases

1. define the companion per-voxel sampling proof, then the occlusion,
   visibility, and culling semantics across pixels and views;
2. aggregate footprint coverage across a whole observation and then the
   complete selected-observation tuple, alongside the existing centreline
   survey in [`tsdf-plan-block-ray-survey.md`](tsdf-plan-block-ray-survey.md);
3. combine approved coverage with the surface/truncation plan, allocate
   missing blocks, and preserve per-observation provenance;
4. explicit idempotency and resumable/nonempty fusion over that chosen domain;
5. complete fusion diagnostics, persistent block-backed TSDF artifacts,
   normalization, and sparse surface/mesh consumers;
6. scalable streaming, optimized CPU, parallel, or GPU execution; and
7. robust filtering, real-dataset accuracy, production meshing, structural
   mapping, Inspector work, pose estimation, SLAM, semantics, localization,
   and `SpatialMapPackage` export.
