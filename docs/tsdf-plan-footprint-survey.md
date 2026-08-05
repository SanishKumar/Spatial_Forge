# Whole-scan conservative footprint coverage

[`tsdf-observation-footprint.md`](tsdf-observation-footprint.md) unions the
nearest-pixel wedge rule across one observation. This checkpoint completes the
coverage ladder by unioning it across the block plan's complete canonical
selected-observation tuple:

```powershell
.\.venv\Scripts\python.exe -m spatialforge reconstruct tsdf-block-context-plan-footprint `
  outputs\progress-blocks.sftplan `
  tests\fixtures\minimal.vgsession
```

It is the conservative counterpart of the centreline
[`tsdf-plan-block-ray-survey.md`](tsdf-plan-block-ray-survey.md): same
selection, same canonical order, same existing/unplanned partition, but every
pixel contributes its whole sampling wedge rather than a zero-width line.

This is the last read-only coverage step. It produces the complete candidate
domain but does not approve it: nothing here expands the `.sftplan`, allocates
TSDF storage, carves free space, fuses voxels, or writes an artifact.

## Public API and immutable record

```python
receipt = survey_tsdf_plan_pixel_footprints_from_context(plan, context)
```

There is no observation argument. The survey always covers the complete
canonical frame-stride selection and cannot be pointed at a subset.

The frozen `TsdfPlanFootprintSurveyReceipt` retains provenance, the selected
sequence tuple, the source-plan block tuple, grid geometry, image size, one
`TsdfObservationFootprintReceipt` per selected observation, and the canonical
covered/existing/unplanned partition. Every count — observation, pixel,
candidate, rejection, visit, duplicate, centreline, footprint-only, per-block
observation support — is derived from those children on access, so no summary
can drift from what it summarises.

Construction re-derives the composition rather than trusting it: the selected
tuple must equal `range(0, total, stride)`, there must be exactly one child per
sequence in canonical order, every child's provenance and geometry must match
the parent's, the covered tuple must equal the canonically ordered union of
the children's coverage, the existing/unplanned split must reproduce
intersection with and difference from the retained source-plan tuple, and the
union must contain the children's centreline paths.

## Proof obligation: containing the centreline survey

The all-observation centreline survey is an independent implementation over
the same selection, so it makes a good adversary. The tests require, at 1.0 m
and at 3.0 m:

```text
ray_survey.covered_block_indices == receipt.centerline_block_indices
ray_survey.covered_block_indices  ⊆ receipt.covered_block_indices
```

The equality checks that unioning the children's re-derived centrelines
reproduces the centreline survey's own union exactly; the subset relation is
the conservativeness claim at whole-scan scope.

## What the fixture does and does not show

```text
observations: selected=2 surveyed=2 ready=2 missing_depth=0 missing_pose=0 missing_depth_and_pose=0
pixel_outcomes: total=8 covered=8 depth_invalid=0
candidate_blocks: total=36 rejected=0
pixel_block_visits: total=36 unique=8 duplicate=28 maximum_per_pixel=8
coverage_blocks: total=8 centerline=8 footprint_only=0
widens_centerline_coverage: no
coverage_partition: existing_plan=8 unplanned=0
coverage_support: multi_observation=8 maximum_observations=2
```

As at observation scope, `widens_centerline_coverage: no`. The fixture's two
2x2 observations have only eight centrelines between them and those already
reach all eight blocks, so the union cannot grow. The fixture proves the
aggregation and the containment, not the widening.

A focused test measures both observations at 3.0 m, where the difference is
decisive:

```text
candidate_blocks: total=200 rejected=48
pixel_block_visits: total=152 unique=52 duplicate=100 maximum_per_pixel=31
coverage_blocks: total=52 centerline=16 footprint_only=36
coverage_partition: existing_plan=32 unplanned=20
coverage_support: multi_observation=40 maximum_observations=2
```

52 covered coordinates against the centrelines' 16. A thin-ray plan expansion
would have missed 36 blocks that a nearest-pixel measurement can genuinely
sample. 20 of the 52 are absent from the source plan, and its active tuple is
byte-identical afterwards.

Another test removes the first pose record: that observation contributes a
zero-coverage child, `surveyed=1`, and the surviving observation still reports
the same eight blocks at support one. Missing input reduces evidence without
inventing coverage.

## Per-block observation support

`covered_block_observation_counts` pairs every covered coordinate with the
number of distinct selected observations covering it; each child's coverage is
already deduplicated, so many pixels of one observation still count once.
`multi_observation_block_indices` and `maximum_block_observation_count` expose
the subset seen more than once and the largest count.

As with the ray survey, this is structural support. It counts observations,
not confidence, and does not by itself justify allocating a block. Any
threshold belongs to the expansion checkpoint.

## Provenance, boundary, and bounded workload

The API validates plan, context, camera, selection, block geometry, and digest
provenance before covering anything. It accepts no `ScanSession` and performs
no replay, source hashing, filesystem I/O, or depth decoding; every child
reads only the immutable metric depth the one prepared context already holds —
the point of aggregating against a context rather than re-reading the session
per observation.

`MAX_TSDF_PLAN_FOOTPRINT_CANDIDATE_BLOCKS` is `262,144`, accumulated across
observations and checked after each one, on top of each child's identical
per-observation and per-pixel caps and the 100,000 unique-block reference
limit. A limit failure returns no partial receipt; nothing is allocated or
mutated, so there is no rollback to perform. Runtime is the sum of the
per-pixel candidate counts across the selection.

## What this still does not do

The complete conservative candidate domain now exists, but nothing consumes
it:

- no voxel-level verdict is applied over these coordinates, so there is still
  no carvable free-space set — [`tsdf-voxel-cross-view.md`](tsdf-voxel-cross-view.md)
  resolves one voxel at a time and nothing yet sweeps it across a domain;
- covering a block does not mean every voxel in it is free space;
- there is no occlusion, visibility, or culling policy across views; and
- no coverage is approved, no plan is expanded, and no block is allocated.

## Precisely deferred next phases

The cross-view verdict is now applied across this domain by
[`tsdf-domain-cross-view.md`](tsdf-domain-cross-view.md), which turns this
coverage into the whole-scan carvable free-space set.

1. combine approved coverage with the surface/truncation plan, allocate
   missing blocks, and preserve per-observation provenance;
3. explicit idempotency and resumable/nonempty fusion over that domain;
4. complete fusion diagnostics, persistent block-backed TSDF artifacts,
   normalization, and sparse surface/mesh consumers;
5. confidence and sensor-dependent weighting, outlier rejection, and the
   visibility/culling policy;
6. scalable streaming, optimized CPU, parallel, or GPU execution; and
7. real-dataset accuracy, production meshing, structural mapping, Inspector
   work, pose estimation, SLAM, semantics, localization, and
   `SpatialMapPackage` export.
