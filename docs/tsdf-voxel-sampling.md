# Per-voxel sampling classification

This checkpoint is the companion to
[`tsdf-pixel-footprint-coverage.md`](tsdf-pixel-footprint-coverage.md). The
footprint rule answers a block-level question conservatively — *which blocks
could contain voxel centres this pixel samples?* This one answers the exact
voxel-level question:

```powershell
.\.venv\Scripts\python.exe -m spatialforge reconstruct tsdf-block-context-voxel-sampling `
  outputs\progress-blocks.sftplan `
  tests\fixtures\minimal.vgsession `
  --observation-sequence 0 `
  --voxel 2 0 0
```

It classifies exactly one signed global voxel centre against exactly one
plan-selected prepared observation. It does not expand the plan, allocate TSDF
storage, carve free space, fuse a voxel, or write an artifact.

## The semantic gap this closes

The reference evaluator in
[`tsdf-voxel-contribution.md`](tsdf-voxel-contribution.md) accepts every voxel
whose signed distance is at or above `-truncation`, clipping the stored value
to `[-1, 1]`. That single `contributes` status covers two physically different
situations:

```text
signed_distance >  +truncation   voxel is well in front of the surface
                                 -> observed empty space, value clipped to +1
|signed_distance| <= truncation  voxel is inside the surface band
                                 -> the measurement's actual gradient
```

Free-space planning needs those separated: the first is what justifies
carving, the second is what defines the surface. The evaluator also conflates
nothing on the far side — `behind-truncation` is already distinct — but it can
only be asked about voxels that already exist in allocated storage, which is
exactly the wrong constraint when deciding *whether* to allocate.

This API takes a raw signed global voxel index instead of a located storage
address, so it can classify voxels in blocks the plan has never seen, and it
reports `planned_block` alongside the classification.

## Statuses

```text
observed-free-space          signed_distance >  +truncation
observed-surface-band        |signed_distance| <= truncation
unobserved-occluded          signed_distance <  -truncation
unobserved-behind-camera     camera z <= 0
unobserved-outside-image     projection outside the calibrated image
unobserved-depth-invalid     sampled pixel has no positive finite depth
unobserved-nonfinite         a nonfinite camera point, projection, or distance
missing-depth
missing-pose
missing-depth-and-pose
```

Both truncation boundaries belong to the surface band: the rules are strict
`>` and `<`, so a signed distance of exactly `+truncation` or `-truncation` is
band, never free space or occluded. Invalid and absent depth remain
*unobserved*; they never become free space.

## Proof obligation 1: agreement with the reference evaluator

This must not become a second, subtly different projection path. For every
planned voxel in the fixture, at both selected observations, the classifier's
world centre, camera point, projection, sampled pixel, measured depth, and
signed distance are all required to equal the existing context evaluator's,
its status must map onto the evaluator's status, and its truncated value must
equal the evaluator's `tsdf_sum_delta` whenever the evaluator contributes:

```text
contributes         <-> observed-free-space or observed-surface-band
behind-truncation   <-> unobserved-occluded
camera-z-nonpositive<-> unobserved-behind-camera
projection-outside-image <-> unobserved-outside-image
depth-invalid       <-> unobserved-depth-invalid
```

The `contributes_to_reference_tsdf` property states that mapping explicitly,
so a future change to either rule breaks a test rather than drifting.

## Proof obligation 2: containment in the pixel footprint

The footprint rule claims to be conservative. This checkpoint tests that claim
against the exact per-voxel rule rather than only against sampled lattice
points: for every planned voxel whose centre lies inside its pixel's closed
sampling wedge — that is, `0 < camera_z <= measured_depth` — the voxel's
containing block must appear in that pixel's footprint coverage.

`inside_sampling_wedge` is the predicate that links the two checkpoints. Note
what it deliberately excludes: a surface-band voxel slightly *behind* the
measured surface has `camera_z > measured_depth`, so it is observed but not
inside the wedge, and the footprint rule makes no claim about it. The wedge
stops at the measurement.

## Public API and immutable record

```python
receipt = classify_tsdf_voxel_sampling_from_context(
    plan,
    context,
    observation_sequence,
    global_index_xyz,
)
```

The frozen `TsdfVoxelSamplingReceipt` retains provenance, the observation
sequence and prepared status, the global/block/local address decomposition,
`planned_block`, the grid geometry, the image size, the status, the world
centre, and the camera point, projection, sampled pixel, measured depth,
signed distance, and truncated value that the reached stage justifies.

Construction re-derives nearly everything it stores:

- the block and local indices must equal the decomposition of the global
  index, and the world centre must equal that index scaled by the voxel size;
- the sampled pixel must equal `floor(projected + 0.5)` of its own retained
  projection, and must lie inside the retained image size;
- the signed distance must equal `measured_depth - camera_z` exactly;
- the status must equal the classification of that signed distance against the
  retained truncation, and must be consistent with the prepared observation
  status; and
- the truncated value must equal the clipped ratio.

Each status also pins exactly which later fields may be present, so an
early-exit receipt cannot carry invented downstream geometry.

## Provenance and source-I/O boundary

The API validates plan, context, camera, selection, voxel index range, block
geometry, and digest provenance before classifying. It accepts no
`ScanSession` and performs no replay, source hashing, filesystem I/O, or depth
decoding; a ready classification reads only the immutable metric depth already
retained by the context. It never consults or mutates TSDF storage — it does
not need storage to exist at all — and leaves the plan untouched.

## Exact fixture proof

With `voxel_size_m=0.125` and `truncation_m=0.5`, observation zero:

```text
voxel (2, 0, 0)    block (0, 0, 0)   planned    camera z 0.3125
                   signed +0.6875    value +1.0 observed-free-space
                   inside_sampling_wedge: yes

voxel (8, -1, -1)  block (1, -1, -1) planned    camera z 1.0625
                   signed -0.0625    value -0.125
                   observed-surface-band, inside_sampling_wedge: no

voxel (60, 0, 0)   block (7, 0, 0)   unplanned  camera z 7.5625
                   signed -6.5625    value -1.0 unobserved-occluded
```

The middle case reproduces the `tsdf_sum=-0.125` contribution that the
single-voxel checkpoints have used since
[`tsdf-voxel-contribution.md`](tsdf-voxel-contribution.md), now labelled as
surface band rather than an undifferentiated `contributes`. The third shows an
unplanned block classified without the plan being touched.

## What this still does not do

- It classifies one voxel against one observation. There is no aggregation
  across observations and no rule for combining disagreeing views.
- `unobserved-occluded` records that *this* observation cannot see the voxel.
  It is not a cross-view occlusion policy, and it does not decide what a voxel
  hidden in one view but visible in another should become.
- There is no visibility or culling policy, and no frustum traversal.
- Nothing here carves free space, approves coverage, expands a plan, allocates
  storage, or fuses a value.

## Precisely deferred next phases

The cross-view rule combining several observations' verdicts for one voxel now
exists in [`tsdf-voxel-cross-view.md`](tsdf-voxel-cross-view.md), which also
proves that its combined totals reproduce the fusing traversal exactly.

1. aggregate conservative footprint coverage across one observation and then
   the complete selected-observation tuple, then apply the cross-view verdict
   across voxels to produce a carvable free-space set;
3. combine approved coverage with the surface/truncation plan, allocate
   missing blocks, and preserve per-observation provenance;
4. explicit idempotency and resumable/nonempty fusion over that domain;
5. complete fusion diagnostics, persistent block-backed TSDF artifacts,
   normalization, and sparse surface/mesh consumers;
6. scalable streaming, optimized CPU, parallel, or GPU execution; and
7. robust filtering, real-dataset accuracy, production meshing, structural
   mapping, Inspector work, pose estimation, SLAM, semantics, localization,
   and `SpatialMapPackage` export.
