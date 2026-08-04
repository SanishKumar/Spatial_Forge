# Cross-view voxel verdict

[`tsdf-voxel-sampling.md`](tsdf-voxel-sampling.md) classifies one voxel against
*one* observation. Several observations routinely disagree about the same
voxel, and that disagreement is not noise — it is the information free-space
carving depends on. This checkpoint defines how the disagreement resolves:

```powershell
.\.venv\Scripts\python.exe -m spatialforge reconstruct tsdf-block-context-voxel-cross-view `
  outputs\progress-blocks.sftplan `
  tests\fixtures\minimal.vgsession `
  --voxel 8 -1 -1
```

It combines every plan-selected observation's verdict for exactly one signed
global voxel centre. It does not carve free space, expand the plan, allocate
TSDF storage, fuse a value, or write an artifact.

## The rule

Each observation contributes one per-voxel status. The combined verdict is
decided by evidence specificity, highest first:

```text
surface      at least one observation places the voxel in its surface band
free-space   no band evidence, but at least one observation sees it as empty
occluded     no positive evidence, but at least one observation is blocked
unseen       no observation sampled it at all
```

The reasoning behind the order:

- **Surface band outranks free space.** Band evidence localises a measured
  surface at this voxel; free-space evidence only asserts emptiness somewhere
  in front of a surface. A voxel that any view places on a surface must not be
  carved away because another view, looking along a grazing angle from
  further back, saw empty space there.
- **Free space outranks occlusion.** Free space is positive evidence of
  emptiness. Occlusion is the *absence* of evidence: it says only that this
  particular camera could not see past a nearer surface.
- **Occlusion and every missing-input status never become free space.** This
  is the invariant the tests pin hardest. `unobserved-occluded`,
  `unobserved-behind-camera`, `unobserved-outside-image`,
  `unobserved-depth-invalid`, `unobserved-nonfinite`, and the three
  missing-input statuses all carry no positive evidence. A voxel seen only
  through those stays `occluded` or `unseen` and is never carvable.

`carvable_free_space` is true only for the `free-space` verdict: seen empty by
at least one observation and placed in a surface band by none.

## Proof obligation: reproducing the fused result

A verdict rule that disagreed with what fusion actually writes would be worse
than useless. This receipt derives, from the same per-observation receipts:

```text
reference_weight     number of observations the evaluator would accept
reference_tsdf_sum   their truncated values summed in canonical order
reference_tsdf_value reference_tsdf_sum / reference_weight, or None
```

For every planned voxel in the fixture, those must equal the `weight_after`
and `tsdf_sum_after` that `traverse_tsdf_voxel_observations_from_context`
actually accumulates into storage, and
`contributing_observation_sequences` must equal the traversal's applied update
sequence. The summation deliberately runs in canonical observation order so
the float64 result is bit-identical to the traversal's sequential
accumulation, not merely close.

The test asserts exact equality. Note that the fixture's two poses differ by
0.05 m, so the two per-observation values differ slightly and their sum is
`-0.24999999999999978`, not `-0.25`; the CLI's nine-decimal formatting rounds
it. The fixture-level test therefore compares to twelve places and leaves the
bit-exact claim to the traversal comparison, where it belongs.

So this checkpoint adds an explanation storage cannot give. A fused slot holds
a sum and a weight; it cannot say whether weight two meant two band
observations, two free-space observations, or one of each — nor whether a
weight of zero meant occlusion or nothing looking there at all.

## Public API and immutable record

```python
receipt = classify_tsdf_voxel_across_observations_from_context(
    plan,
    context,
    global_index_xyz,
)
```

There is no observation argument: the verdict always covers the complete
canonical frame-stride selection and cannot be pointed at a subset. The voxel
need not belong to the plan's active blocks, and `planned_block` is reported
alongside the verdict.

The frozen `TsdfVoxelCrossViewReceipt` retains provenance, the selected
sequence tuple, the address decomposition, `planned_block`, grid geometry,
image size, world centre, the verdict, and one child
`TsdfVoxelSamplingReceipt` per selected observation. Counts, reference totals,
contributing and wedge sequence tuples, and the status histogram are derived
on access, so no summary can drift from its children.

Construction re-derives the composition rather than trusting it: the selected
tuple must equal `range(0, total, stride)`, there must be exactly one child
per sequence in canonical order, every child's provenance, address, grid, and
world centre must match the parent's, and the verdict must equal the
recombination of the children's own statuses.

## Provenance and source-I/O boundary

The API validates plan, context, camera, selection, voxel range, block
geometry, and digest provenance before classifying. It accepts no
`ScanSession` and performs no replay, source hashing, filesystem I/O, or depth
decoding; each child reads only the immutable metric depth already retained by
the context. It never consults or mutates TSDF storage — storage need not
exist — and leaves the plan untouched.
`MAX_TSDF_VOXEL_CROSS_VIEW_OBSERVATIONS` is `262,144`.

## Exact fixture proof

```text
voxel (8, -1, -1)  planned    surface_band=2 free_space=0 occluded=0 unseen=0
                   verdict surface, weight 2, value -0.125
                   contributing (0, 1), wedge (), carvable no

voxel (2, 0, 0)    planned    surface_band=0 free_space=2 occluded=0 unseen=0
                   verdict free-space, weight 2, value +1.0
                   contributing (0, 1), wedge (0, 1), carvable yes

voxel (60, 0, 0)   unplanned  surface_band=0 free_space=0 occluded=2 unseen=0
                   verdict occluded, weight 0, value none
                   contributing (), carvable no
```

A focused test also removes the first pose record, so observation zero becomes
`missing-pose`. The surface voxel's verdict stays `surface` on the strength of
observation one alone, with weight one and contributing sequence `(1,)` — a
missing input reduces evidence without inverting a verdict.

## What this still does not do

- It resolves one voxel. There is no traversal across voxels or blocks, so
  nothing yet produces a carvable-free-space set.
- The precedence rule is deliberately unweighted. There is no confidence,
  distance, incidence-angle, or sensor-dependent weighting, and no outlier
  rejection: one band observation out of fifty decides `surface`.
- It is not a visibility or culling policy, and it does not detect moving
  objects or pose error — a voxel that is band in one view and free space in
  another is reported as `surface` with both counts visible, not flagged.
- Nothing here carves, approves coverage, expands a plan, allocates storage,
  or fuses a value.

## Precisely deferred next phases

1. aggregate conservative footprint coverage across one observation and then
   the complete selected-observation tuple, then apply this verdict across
   voxels to produce a carvable set;
2. combine approved coverage with the surface/truncation plan, allocate
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
