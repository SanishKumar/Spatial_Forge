# Block-wide cross-view verdicts

[`tsdf-voxel-cross-view.md`](tsdf-voxel-cross-view.md) resolves one voxel
against every selected observation. This checkpoint sweeps that rule across a
whole block, producing the project's first **carvable free-space set**:

```powershell
.\.venv\Scripts\python.exe -m spatialforge reconstruct tsdf-block-context-block-cross-view `
  outputs\progress-blocks.sftplan `
  tests\fixtures\minimal.vgsession `
  --block 0 0 0
```

It resolves all 512 voxels of exactly one signed block, planned or not, in
canonical x-fastest local-flat order. It does not carve, expand the plan,
allocate TSDF storage, fuse voxels, or write an artifact.

## What it produces that storage cannot

A fused block row holds 512 sums and weights. It cannot distinguish a voxel
whose weight came from surface-band measurements from one whose weight came
from free-space observations, and it cannot distinguish an untouched slot that
was occluded from one that nothing ever looked at.

This receipt separates all four:

```text
surface      some observation placed the voxel in its band
free-space   seen empty, never banded  -> carvable
occluded     blocked in every observation that reached it
unseen       no observation sampled it at all
```

`carvable_free_space_voxel_indices` is the canonical tuple of global voxel
indices whose verdict is `free-space` — the set a carving step would be
entitled to mark empty. On the committed fixture, block `(0, 0, 0)` yields six
such voxels; the surface block `(1, -1, -1)` yields none, because every voxel
there that any observation sees is inside a measured band.

## Proof obligation: reproducing the fused block row

Sweeping a rule across 512 voxels is only trustworthy if it still agrees with
what fusion writes. For every one of the plan's eight active blocks, the tests
require, voxel by voxel in the same canonical order:

```text
receipt.voxel_receipts[i].local_index_xyz    == traversal.voxel_receipts[i].address.local_index_xyz
receipt.voxel_receipts[i].reference_weight   == traversal.voxel_receipts[i].weight_after
receipt.voxel_receipts[i].reference_tsdf_sum == traversal.voxel_receipts[i].tsdf_sum_after
```

against `traverse_tsdf_block_voxels_from_context`, which actually mutates
storage. The comparison is exact, not approximate, and it covers all four
verdicts across the eight blocks.

## Public API and immutable record

```python
receipt = classify_tsdf_block_voxels_across_observations_from_context(
    plan,
    context,
    block_index_xyz,
)
```

There is no observation argument and no voxel argument: the sweep always
covers all 512 voxels against the complete canonical frame-stride selection.
The block need not belong to the plan's active tuple — `planned_block` is
reported alongside the result, which matters because the surveyed coverage
domain includes unplanned coordinates.

The frozen `TsdfBlockCrossViewReceipt` retains provenance, the selected
sequence tuple, the block index, `planned_block`, grid geometry, image size,
and one `TsdfVoxelCrossViewReceipt` per voxel. Verdict counts, the carvable
tuple, reference weight and sum totals, and the maximum voxel weight are all
derived from those children on access.

Construction re-derives the composition: exactly 512 children, each one's
local index equal to the x-fastest decomposition of its position, each global
index equal to `compose(block_index, local_index)`, and every child's
provenance, selection, `planned_block`, grid, and image size matching the
parent's.

## Exact fixture proof

```text
block (1, -1, -1)  planned    surface=102 free_space=0   occluded=198 unseen=212
                   carvable 0, weight total 204, sum total -117.0

block (0, 0, 0)    planned    surface=38  free_space=6   occluded=0   unseen=468
                   carvable 6, weight total 88,  sum total 41.0

block (7, 0, 0)    unplanned  surface=0   free_space=0   occluded=512 unseen=0
                   carvable 0, weight total 0
```

The three cases are deliberately different in kind. `(1, -1, -1)` sits on the
measured surface, so its observed voxels are all band and nothing is carvable.
`(0, 0, 0)` sits between the camera and that surface, so it contributes the
first carvable voxels. `(7, 0, 0)` lies behind the surface and is unplanned:
every voxel is occluded, nothing is carvable, and the plan is untouched.

A further test removes the first pose record. Every voxel's first observation
becomes `missing-pose`, the maximum voxel weight drops to one, and surface
verdicts survive on the strength of the remaining observation — reduced
evidence, not inverted verdicts.

## Provenance, boundary, and bounded workload

The API validates plan, context, camera, selection, block index, geometry, and
digest provenance before sweeping. It accepts no `ScanSession` and performs no
replay, source hashing, filesystem I/O, or depth decoding; every child reads
only the immutable metric depth the context already holds. It never consults
or mutates TSDF storage — storage need not exist — and leaves the plan
unchanged.

`MAX_TSDF_BLOCK_CROSS_VIEW_OUTCOMES` is `262,144`, preflighted as
`512 * selected observations` before any voxel is resolved, so an oversized
selection is rejected up front rather than part-way through.

## What this still does not do

- One block. There is no sweep across the surveyed multi-block domain, so the
  carvable set is per-block, not whole-scan.
- Nothing carves. The set is reported, not applied: no plan is expanded, no
  block allocated, no storage written.
- The precedence rule remains unweighted — no confidence, distance,
  incidence-angle or sensor-dependent weighting and no outlier rejection.
- There is no visibility or culling policy, and a voxel that is band in one
  view and free space in another is reported as `surface` with both counts
  visible, not flagged as a conflict.

## Precisely deferred next phases

This block rule is now swept across the whole surveyed coverage domain by
[`tsdf-domain-cross-view.md`](tsdf-domain-cross-view.md), producing the
whole-scan carvable set.

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
