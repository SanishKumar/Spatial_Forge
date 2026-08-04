# Multi-observation camera-to-surface block-ray survey

This checkpoint aggregates the existing one-observation centerline ray trace
across the block plan's complete selected-observation tuple:

```powershell
.\.venv\Scripts\python.exe -m spatialforge reconstruct tsdf-block-context-plan-rays `
  outputs\progress-blocks.sftplan `
  tests\fixtures\minimal.vgsession
```

It is a read-only coverage diagnostic. For every plan-selected prepared
observation, in canonical frame-stride order, it delegates to
`trace_tsdf_observation_block_rays_from_context` and retains that observation's
complete frozen transcript. It then reports the combined block coverage, its
existing/unplanned partition, and how many distinct observations cover each
coordinate.

Aggregation is deliberately separated from acting on the aggregate. The
operation does not expand the `.sftplan`, create or allocate blocks, mutate
TSDF storage, fuse a voxel, approve coverage, or write an artifact. See
[`tsdf-observation-block-rays.md`](tsdf-observation-block-rays.md) for the
per-observation geometry, thin-DDA tie rule, and its non-goals; every one of
those limits still applies to each aggregated child.

## Public API and immutable record

The public API is:

```python
receipt = survey_tsdf_plan_block_rays_from_context(plan, context)
```

`plan` must be a strict-loaded `TsdfBlockPlan` and `context` must be its
matching immutable `TsdfReplayDepthContext`. Unlike the one-observation trace,
this API takes no sequence argument: the survey always covers the complete
canonical selection and cannot be pointed at a caller-chosen subset, block
subset, depth frame, pose, or alternative grid.

The frozen `TsdfPlanBlockRaySurveyReceipt` stores:

```text
source_plan_digest_sha256
replay_digest_sha256
frame_stride
total_observations
selected_observation_sequences
source_plan_block_indices
block_resolution
block_extent_m
image_size
observation_receipts
covered_block_indices
existing_plan_block_indices
unplanned_block_indices
```

Every aggregate is derived from `observation_receipts` on access rather than
stored a second time, so a count cannot drift from the transcripts it
summarises. The receipt derives observation, traced-observation, pixel,
traversed-ray, invalid-depth, block-visit, duplicate-visit, maximum-blocks-per-
ray, retained-outcome, surface-endpoint-block, nonterminal-block, per-block
observation-support, multi-observation-block, and prepared-depth-access
counts, plus the `TsdfReplayDepthStatus` histogram.

## Validated composition

Construction re-derives, rather than trusts, the relationship between the
parent and its children:

1. `selected_observation_sequences` must equal
   `range(0, total_observations, frame_stride)` exactly.
2. There must be exactly one child receipt per selected sequence, in that
   canonical order, each reporting its own matching `observation_sequence`.
3. Each child's plan digest, replay digest, frame stride, total observations,
   source-plan block tuple, block resolution, block extent, and image size
   must equal the parent's.
4. `covered_block_indices` must equal the canonically ordered union of the
   children's covered tuples.
5. The existing/unplanned split must be exactly that union intersected with,
   and subtracted from, the retained source-plan block tuple. The two sides
   must be disjoint and must together reproduce the union.

Because each child already validates its own DDA paths, endpoint blocks, and
plan membership, the survey inherits those proofs instead of recomputing
geometry.

## Per-block observation support

`covered_block_observation_counts` pairs every covered coordinate, in
canonical order, with the number of distinct selected observations whose rays
cover it. Each child's covered tuple is already deduplicated, so a block that
several pixels of one observation traverse still counts once for that
observation.

`multi_observation_block_indices` is the subset with a count above one, and
`maximum_block_observation_count` is the largest count.

This is structural support, not evidence quality. A high count means several
prepared observations' centerlines entered a block. It does not weigh
observations, prove visibility, prove that any particular voxel inside the
block is free space, or by itself justify allocating that block. Choosing a
support threshold — if one is used at all — belongs to the separate expansion
checkpoint.

## Provenance and source-I/O boundary

The survey validates plan, context, camera, selection, block geometry, and
digest provenance before tracing anything. It accepts no `ScanSession` and
performs no replay, source hashing, filesystem I/O, or depth decoding; each
ready child necessarily reads the immutable metric depth already retained by
the context.

The CLI performs the earlier setup exactly once: it strict-loads the plan,
replay-verifies the session while constructing the context, and decodes each
ready selected depth frame once. Every observation in the survey then reuses
that one completed construction-time snapshot — the whole point of aggregating
against a context rather than re-reading the session per observation. Rebuild
the context when current folder freshness is required.

The source plan is used only for provenance, grid geometry, observation
selection, and the existing/unplanned partition. Its
`activation.free_space_rule` remains `not-planned`, its active tuple remains
unchanged, and no replacement plan is serialized.

## Bounded diagnostic workload

The retained outcome count is the sum of the children's own retained counts:

```text
sum over selected observations of (pixel receipts + traversed block visits)
```

`MAX_TSDF_PLAN_BLOCK_RAY_SURVEY_OUTCOMES` is `262,144`. The survey enforces it
twice:

1. Before tracing anything, it rejects when the pixel-receipt component alone,
   `selected observations * image pixels`, would exceed the cap. This is a
   conservative admission check on that component, not a prediction of the
   total.
2. After each child, it rejects as soon as the exact accumulated total exceeds
   the cap.

Each child additionally enforces its own identical per-observation cap and the
100,000 unique-block reference limit; the survey re-checks that block limit
against the growing union. A limit failure returns no partial receipt. Nothing
was allocated or mutated, so there is no rollback to perform — unlike the
storage-mutating plan traversal, atomicity here is a property of the operation
rather than a restore step.

For `N` selected observations, `P` image pixels, and `V` retained block visits,
runtime and the transcript are `O(N * P + V)`. Every child transcript is
retained in memory, so the cap bounds transcript size as well as time. These
caps do not make this a scalable, streaming, parallel, GPU, or production
coverage planner.

## Exact fixture proof

The committed fixture selects both observations at `frame_stride=1`. Each is
ready, has four positive finite depth pixels, and covers the same eight
blocks, so the union equals either observation's coverage while every
coordinate gains support two. The key output is:

```text
observations: selected=2 traced=2 ready=2 missing_depth=0 missing_pose=0 missing_depth_and_pose=0
observation_order: canonical-frame-stride sequences=0..1
first_observation: sequence=0 status=ready covered_blocks=8
last_observation: sequence=1 status=ready covered_blocks=8
image: width=2 height=2
pixel_outcomes: total=8 traversed=8 depth_invalid=0
ray_block_visits: total=22 unique=8 duplicate=14 maximum_per_ray=3
coverage_blocks: total=8 nonterminal=4 surface_endpoint=4
coverage_partition: existing_plan=8 unplanned=0
coverage_support: multi_observation=8 maximum_observations=2
coverage_order: canonical-x-fastest
first_covered_block: (0, -1, -1)
last_covered_block: (1, 0, 0)
survey_session_replay: no
survey_depth_decoding: no
survey_prepared_depth_access: yes
survey_workload: retained_outcomes=30 maximum=262144
coverage_scope: all-plan-selected-observations
multiple_observation_coverage_computed: yes
all_selected_observations_surveyed: yes
conservative_nearest_pixel_free_space_coverage_proven: no
coverage_approved_for_expansion: no
plan_expanded: no
missing_blocks_created: no
storage_allocated: no
storage_mutated: no
full_fusion_performed: no
artifact_written: no
```

The 30 outcomes are two 15-outcome children. The 22 visits deduplicate to
eight coordinates, all already present in this tiny plan.

Because that fixture's two observations are geometrically identical, it does
**not** demonstrate that aggregation widens coverage. Focused tests cover the
cases it cannot:

- one observation measuring at `1.0 m` and the other at `3.0 m` produces a
  16-block union strictly larger than the near observation's eight, with
  support two on the shared near blocks and one on the far blocks, and an
  aggregate `maximum_per_ray` of five taken from the farther child;
- both observations measuring at `3.0 m` produce eight existing and eight
  unplanned coordinates, and the source plan's active tuple is byte-identical
  afterwards;
- a missing-depth, missing-pose, or missing-both first observation contributes
  a zero-ray child and no coverage, while the surviving ready observation
  still reports its eight blocks with support one; and
- `frame_stride=2` surveys exactly sequence zero.

## Precisely deferred next phases

Coverage is now computed over the complete selection, but nothing consumes it.
The next checkpoint should define a conservative nearest-pixel or voxel-center
coverage proof, together with the frustum, visibility, occlusion, and culling
semantics that a real free-space rule needs. Only after that should approved
coordinates become a canonical expanded fusion domain.

Later checkpoints remain for:

1. combining approved coverage with the surface/truncation plan, allocating
   missing blocks, and preserving per-observation provenance;
2. explicit idempotency and resumable/nonempty fusion over that chosen domain;
3. complete fusion diagnostics, persistent block-backed TSDF artifacts,
   normalization, and sparse surface/mesh consumers;
4. scalable streaming, optimized CPU, parallel, or GPU execution; and
5. robust filtering, real-dataset accuracy, production meshing, structural
   mapping, Inspector work, pose estimation, SLAM, semantics, localization,
   and `SpatialMapPackage` export.
