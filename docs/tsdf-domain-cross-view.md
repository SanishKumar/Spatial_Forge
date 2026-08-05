# Whole-scan carvable free space

This checkpoint joins the project's two independent ladders for the first
time. The coverage ladder ends at
[`tsdf-plan-footprint-survey.md`](tsdf-plan-footprint-survey.md), which says
*which blocks a measurement could sample*. The verdict ladder ends at
[`tsdf-block-cross-view.md`](tsdf-block-cross-view.md), which says *what one
block's voxels actually are*. Running the second over the domain produced by
the first yields the whole-scan carvable free-space set:

```powershell
.\.venv\Scripts\python.exe -m spatialforge reconstruct tsdf-block-context-domain-cross-view `
  outputs\progress-blocks.sftplan `
  tests\fixtures\minimal.vgsession
```

It is still read-only. Nothing here carves, approves coverage, expands the
plan, allocates storage, fuses voxels, or writes an artifact.

## Public API and immutable record

```python
coverage = survey_tsdf_plan_pixel_footprints_from_context(plan, context)
receipt = sweep_tsdf_coverage_domain_cross_view_from_context(
    plan,
    context,
    coverage,
)
```

The domain is not a caller-supplied block list. It is taken from a
`TsdfPlanFootprintSurveyReceipt`, whose plan digest, replay digest, selected
sequence tuple, and source-plan block tuple must all match the plan being
swept. A survey built from a different plan is rejected rather than silently
mixed, so the domain always carries the provenance of the coverage that
produced it.

The frozen `TsdfCoverageDomainCrossViewReceipt` retains provenance, the
selected sequence tuple, the canonical domain block tuple with its
existing/unplanned split, and one `TsdfBlockCrossViewReceipt` per domain
block. All verdict counts, the carvable set, the planned/unplanned carvable
split, and the reference totals are derived from those children on access.

Construction re-derives the composition: one child per domain block in
canonical order, each child's `planned_block` agreeing with the retained
partition, and every child's provenance, selection, and block resolution
matching the parent's.

## Proof obligation: reproducing the fused plan

The sweep now touches thousands of voxels, so it is anchored to the fusing
path one more time. On the committed fixture the surveyed domain happens to
equal the plan's active tuple, which makes the comparison exact:

```text
receipt.reference_weight_total == plan_traversal.weight_delta
receipt.observed_voxel_count   == plan_traversal.observed_voxel_count
receipt.maximum_voxel_weight   == plan_traversal.maximum_weight_after
```

against `traverse_tsdf_plan_blocks_from_context`, which actually writes to
storage. Those come out as `1168`, `584` and `2` — the same 1,168 contributions
over 584 slots the plan traversal has reported since it was written.

## The number that motivates the next checkpoint

The fixture's own domain is entirely inside the plan, so it carves 24 voxels
and reports `unplanned=0`. That is not the interesting case.

A focused test measures both observations at 3.0 m with `frame_stride=2`:

```text
coverage_domain: blocks=52 existing_plan=32 unplanned=20
domain_voxels: total=26624
voxel_verdicts: surface=4656 free_space=2680 occluded=3608 unseen=15680
carvable_free_space_voxels: total=2680 in_plan=1304 unplanned=1376
carvable_blocks: total=24 unplanned=8
```

**More than half the carvable free space — 1,376 of 2,680 voxels, in 8 blocks —
lies outside the current plan.** The surface/truncation planner never had a
reason to allocate those blocks, and until the conservative coverage survey
existed there was no deterministic way to discover them. That gap is precisely
what the plan-expansion checkpoint has to close, and it is now measured rather
than assumed.

## Verdict semantics, unchanged

The per-voxel rule is inherited exactly from
[`tsdf-voxel-cross-view.md`](tsdf-voxel-cross-view.md): surface outranks free
space, free space outranks occlusion, and occlusion or missing input never
becomes free space. `carvable_free_space_voxel_indices` therefore contains
only voxels seen empty by some observation and placed in a band by none.

The receipt splits that set by plan membership because the two halves have
different consequences: in-plan carvable voxels could be fused today, while
unplanned ones require allocating a block first.

## Provenance, boundary, and bounded workload

The API validates plan, context, camera, selection, coverage provenance, and
block geometry before sweeping. It accepts no `ScanSession` and performs no
replay, source hashing, filesystem I/O, or depth decoding; every descendant
reads only the immutable metric depth the one prepared context already holds.
It never consults or mutates TSDF storage — storage need not exist — and
leaves the plan unchanged.

`MAX_TSDF_COVERAGE_DOMAIN_CROSS_VIEW_OUTCOMES` is `262,144`, preflighted as
`domain blocks * 512 * selected observations` before any block is resolved, so
an oversized domain is rejected up front rather than part-way through. The
fixture's own sweep is 8,192 outcomes; the far case is 26,624.

This retains every voxel's full transcript, so memory and time both scale with
the domain. It is a reference sweep, not a streaming, parallel, or GPU path.

## What this still does not do

- Nothing carves. The set is reported, not applied.
- No plan is expanded and no block allocated, so the 1,376 unplanned carvable
  voxels remain unreachable by fusion.
- The precedence rule stays unweighted: no confidence, distance,
  incidence-angle or sensor-dependent weighting, and no outlier rejection.
- There is no visibility or culling policy, and no conflict flagging when a
  voxel is band in one view and free space in another.

## Precisely deferred next phases

The approval and merge step now exists as a read-only proposal in
[`tsdf-plan-expansion.md`](tsdf-plan-expansion.md), which turns this carvable
set into a concrete expanded block set.

1. serialize that proposed block set as a new `.sftplan`, carrying the
   approval rule, source plan digest, and per-block provenance — the first
   checkpoint that writes a plan;
2. explicit idempotency and resumable/nonempty fusion over that domain;
3. complete fusion diagnostics, persistent block-backed TSDF artifacts,
   normalization, and sparse surface/mesh consumers;
4. confidence and sensor-dependent weighting, outlier rejection, and the
   visibility/culling policy;
5. scalable streaming, optimized CPU, parallel, or GPU execution; and
6. real-dataset accuracy, production meshing, structural mapping, Inspector
   work, pose estimation, SLAM, semantics, localization, and
   `SpatialMapPackage` export.
