# One-observation conservative footprint coverage

[`tsdf-pixel-footprint-coverage.md`](tsdf-pixel-footprint-coverage.md) covers
the sampling wedge of one pixel. This checkpoint unions that rule across every
pixel of one prepared observation:

```powershell
.\.venv\Scripts\python.exe -m spatialforge reconstruct tsdf-block-context-observation-footprint `
  outputs\progress-blocks.sftplan `
  tests\fixtures\minimal.vgsession `
  --observation-sequence 0
```

It is the conservative counterpart of the centreline
[`tsdf-observation-block-rays.md`](tsdf-observation-block-rays.md): same
observation, same canonical row-major pixel order, same existing/unplanned
partition, but each pixel contributes its whole wedge rather than a
zero-width line. It does not expand the `.sftplan`, allocate TSDF storage,
carve free space, fuse voxels, or write an artifact.

## Public API and immutable record

```python
receipt = survey_tsdf_observation_pixel_footprints_from_context(
    plan,
    context,
    observation_sequence,
)
```

The frozen `TsdfObservationFootprintReceipt` retains provenance, the
observation sequence and prepared status, the block grid, image size, camera
origin, one `TsdfPixelFootprintCoverageReceipt` per pixel in row-major order,
and the canonical covered/existing/unplanned partition. Pixel, candidate,
rejection, visit, duplicate, centreline, footprint-only, and per-block pixel
support counts are all derived from those children on access.

Construction re-derives the composition rather than trusting it: a ready
observation must retain exactly one receipt per pixel in canonical row-major
order, every child's provenance, sequence, status, grid, image size, and
camera origin must match the parent's, the covered tuple must equal the
canonically ordered union of the children's coverage, the existing/unplanned
split must reproduce intersection with and difference from the retained
source-plan tuple, and the union must contain the children's centreline paths.

Observations missing depth or pose return a successful receipt with no pixel
receipts and no coverage. A `missing-depth` observation may retain its copied
camera origin because its pose exists; a pose-missing one may not.

## Proof obligation: containing the centreline union

The centreline ray trace is an independent implementation over the same
observation, so it makes a good adversary. The tests require, at 1.0 m and at
3.0 m:

```text
trace.covered_block_indices == receipt.centerline_block_indices
trace.covered_block_indices  ⊆ receipt.covered_block_indices
```

The first equality checks that unioning the children's re-derived centrelines
reproduces the ray checkpoint's own union exactly; the second is the
conservativeness claim at observation scope.

## What the fixture does and does not show

```text
pixel_outcomes: total=4 covered=4 depth_invalid=0
candidate_blocks: total=18 rejected=0
pixel_block_visits: total=18 unique=8 duplicate=10 maximum_per_pixel=8
coverage_blocks: total=8 centerline=8 footprint_only=0
widens_centerline_coverage: no
coverage_partition: existing_plan=8 unplanned=0
coverage_support: maximum_pixels_per_block=4
```

Note `widens_centerline_coverage: no`. Per pixel the wedge is strictly wider
than its centreline — pixel `(1, 1)` covers 8 blocks against its centreline's
3 — but this 2x2 image has only four centrelines and between them they already
reach all eight blocks, so the observation-level union does not grow. The
fixture proves the aggregation runs and the containment holds; it does **not**
demonstrate the widening.

A focused test measures the same observation at 3.0 m, where the difference is
unmistakable:

```text
candidate_blocks: total=100 rejected=16
pixel_block_visits: total=84 unique=52 duplicate=32 maximum_per_pixel=31
coverage_blocks: total=52 centerline=16 footprint_only=36
coverage_partition: existing_plan=32 unplanned=20
```

52 covered blocks against the centrelines' 16 — the thin trace would have
missed 36 coordinates that a nearest-pixel measurement can genuinely sample.
20 of the 52 are absent from the source plan, and the plan's active tuple is
byte-identical afterwards.

Another test blanks one pixel's depth: that pixel yields a `depth-invalid`
child with no candidates and no coverage, the observation reports
`covered=3 depth_invalid=1`, and the surviving three pixels still union to the
same eight blocks. Invalid depth reduces evidence; it never invents free
space.

## Per-block pixel support

`covered_block_pixel_counts` pairs every covered coordinate with the number of
pixels of this observation whose wedge covers it, and
`maximum_block_pixel_count` reports the largest. Like the ray survey's
observation support, this is structural: it counts wedges, not confidence, and
does not by itself justify allocating a block.

## Provenance, boundary, and bounded workload

The API validates plan, context, camera, selection, block geometry, and digest
provenance before covering anything. It accepts no `ScanSession` and performs
no replay, source hashing, filesystem I/O, or depth decoding; each ready child
reads only the immutable metric depth the context already holds. The source
plan supplies provenance, grid geometry, selection, and the partition only.

`MAX_TSDF_OBSERVATION_FOOTPRINT_CANDIDATE_BLOCKS` is `262,144`, accumulated
across pixels and checked after each one, on top of each child's own identical
per-pixel cap and the 100,000 unique-block reference limit. A limit failure
returns no partial receipt; nothing is allocated or mutated, so there is no
rollback to perform. Runtime is the sum of the per-pixel candidate counts.

## What this still does not do

- One observation only. The complete selected-observation tuple is not
  aggregated, so there is still no whole-scan conservative coverage.
- Nothing applies the cross-view voxel verdict from
  [`tsdf-voxel-cross-view.md`](tsdf-voxel-cross-view.md) over these
  coordinates, so no carvable free-space set exists yet.
- Covering a block does not mean every voxel in it is free space, there is no
  occlusion, visibility, or culling policy, and no coverage is approved.

## Precisely deferred next phases

1. aggregate this coverage across the complete selected-observation tuple;
2. apply the cross-view verdict across the voxels of the covered domain to
   produce a carvable free-space set;
3. combine approved coverage with the surface/truncation plan, allocate
   missing blocks, and preserve per-observation provenance;
4. explicit idempotency and resumable/nonempty fusion over that domain;
5. complete fusion diagnostics, persistent block-backed TSDF artifacts,
   normalization, and sparse surface/mesh consumers;
6. confidence and sensor-dependent weighting, outlier rejection, and the
   visibility/culling policy;
7. scalable streaming, optimized CPU, parallel, or GPU execution; and
8. real-dataset accuracy, production meshing, structural mapping, Inspector
   work, pose estimation, SLAM, semantics, localization, and
   `SpatialMapPackage` export.
