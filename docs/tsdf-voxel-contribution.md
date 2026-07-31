# Read-only TSDF voxel contribution

This checkpoint evaluates one already planned voxel against one replay-selected
observation. When exact depth and known pose are both present, it performs one
projective TSDF evaluation:

```powershell
.\.venv\Scripts\python.exe -m spatialforge reconstruct tsdf-block-contribution `
  outputs/progress-blocks.sftplan `
  tests/fixtures/minimal.vgsession `
  --observation-sequence 0 `
  --voxel 8 -1 -1
```

The result describes the TSDF sum and weight deltas that the separate
single-slot update primitive can apply. This command applies neither delta. It
evaluates no other voxel or observation, leaves the temporary block buffers
unchanged, and writes no artifact.

## Public API

```python
contribution = evaluate_tsdf_voxel_contribution(
    storage,
    address,
    session,
    observation_sequence,
)
```

The inputs are:

- replay-matched `TsdfBlockStorage` allocated from a strict-loaded
  `TsdfBlockPlan`;
- a frozen `TsdfVoxelAddress` that resolves back to that exact storage;
- the current loaded `ScanSession`; and
- a zero-based replay observation sequence selected by the plan's frame
  stride.

The function returns a frozen `TsdfVoxelContribution`. In addition to its
address, observation, numerical result, and diagnostics, every contribution
records:

```text
source_plan_digest_sha256
replay_digest_sha256
```

The source-plan digest identifies the exact loaded `.sftplan` bytes. The replay
digest identifies the session-input snapshot used during evaluation. A later
one-slot update compares both fields with its destination storage and current
session before mutation. Matching SHA-256 values establish deterministic byte
identity; they are not signatures, authentication, or proof that the inputs
are trustworthy.

The address is an input because planning and signed voxel addressing are
already separate checkpoints. A valid global coordinate in an unplanned block
remains an addressing miss; the CLI rejects it instead of creating a block or
treating it as a measurement skip.

The evaluator replay-checks the current session against the source plan before
sampling and again after evaluation. A session mismatch, stale replay digest,
out-of-range or unselected observation sequence, address/storage mismatch, or
invalid reconstruction contract raises `TsdfError`.

## Prepared-context public API

The separate context-aware evaluator has the same one-observation, one-voxel,
read-only result boundary:

```python
contribution = evaluate_tsdf_voxel_contribution_from_context(
    storage,
    address,
    context,
    observation_sequence,
)
```

Its `context` must be a frozen `TsdfReplayDepthContext` whose session ID,
source-plan digest, replay digest, frame selection, depth-sample counts, and
paired-pixel capacity match the destination storage's strict-loaded plan. The
address must still re-resolve exactly, and the requested sequence must be one
of the context's canonical plan-selected observations.

The evaluator takes no `ScanSession`. For a ready record, it uses the copied
`T_world_camera`, a fresh read-only view of the immutable metric depth bytes,
and the context camera. Missing-input context statuses map directly to the
existing contribution skip statuses. The function returns the same frozen
`TsdfVoxelContribution` type and applies the same projective rule below.

Context construction already replay-hashed the source inputs, decoded each
ready selected depth frame once, and copied the poses. This scalar evaluation
does not replay or hash the session, reopen a depth file, decode depth, mutate
storage, or iterate another observation or voxel. Its provenance is the
context's construction-time plan and replay binding, not a new current-folder
freshness check.

## Projective contribution rule

For signed global voxel index `(gx, gy, gz)` and plan voxel edge length `s`,
the zero-anchored voxel center is:

```text
world_xyz_m = ((gx + 0.5) * s,
               (gy + 0.5) * s,
               (gz + 0.5) * s)
```

The evaluator applies the inverse of the observation's `T_world_camera` to
obtain `(x, y, z)` in camera coordinates. A finite point with `z > 0` is
projected through the aligned RGB pinhole calibration:

For a complete observation, the aligned depth frame is decoded and scaled
once before voxel rejection, preserving the reference integrator's malformed
image behavior. Evaluation still samples only the one projected pixel.

```text
u = fx * x / z + cx
v = fy * y / z + cy
```

The continuous projection must lie in:

```text
-0.5 <= u < width  - 0.5
-0.5 <= v < height - 0.5
```

Nearest-pixel sampling uses:

```text
pixel_u = floor(u + 0.5)
pixel_v = floor(v + 0.5)
```

After converting the sampled depth to metres, the signed distance and proposed
delta are:

```text
signed_distance_m = measured_depth_m - z
tsdf_sum_delta = clip(signed_distance_m / truncation_m, -1, 1)
weight_delta = 1
```

A positive value is on the camera/free-space side of the measured surface. A
negative value is behind the surface. Finite positive depth contributes while
the signed distance is at least `-truncation_m`; a voxel farther behind the
surface is skipped. Positive distances may clamp to `+1`, matching the
fixed-bounds reference TSDF.

This defines the numerical result for one already planned voxel only. It does
not trace a ray or decide whether additional camera-to-surface free-space
blocks should be planned.

## Immutable result and skip statuses

Every result records the input address, observation sequence, world center,
status, and `weight_delta`. Finite diagnostics from successfully completed
stages are also retained: camera point, continuous projection, nearest pixel,
whether depth was decoded, measured depth, signed distance, and proposed TSDF
sum delta.

An accepted result has:

```text
status = contributes
tsdf_sum_delta in [-1, 1]
weight_delta = 1
```

Skipped results have no TSDF sum delta and a zero weight delta. Their stable
statuses are:

| Status | Meaning |
|---|---|
| `missing-depth` | The selected observation has no exact depth sample. |
| `missing-pose` | The selected observation has no exact known pose. |
| `missing-depth-and-pose` | The selected observation has neither exact input. |
| `camera-point-nonfinite` | Transforming the voxel produced a non-finite camera point. |
| `camera-z-nonpositive` | The voxel is on or behind the camera plane. |
| `projection-nonfinite` | Pinhole projection produced a non-finite coordinate. |
| `projection-outside-image` | The continuous projection or rounded pixel is outside the image. |
| `depth-invalid` | The sampled metric depth is zero, negative, or non-finite. |
| `signed-distance-nonfinite` | Depth minus camera-space Z is non-finite. |
| `behind-truncation` | The voxel is more than one truncation distance behind the surface. |

The projection and pixel are outputs, not caller-supplied inputs. This keeps
nearest-depth sampling identical to the deterministic dense reference rule.

## Exact fixture proof

The committed fixture plan uses:

```text
voxel_size_m = 0.125
truncation_m = 0.5
```

Global voxel `(8, -1, -1)` resolves to block `(1, -1, -1)`, local coordinate
`(0, 7, 7)`, block row `1`, array index `(1, 7, 7, 0)`, and storage flat index
`1016`.

The relevant command output is:

```text
TSDF BLOCK CONTRIBUTION CHECK scan-synthetic-0001
artifact: valid
session_replay: matched
observation_sequence: 0
voxel: global=(8, -1, -1) block=(1, -1, -1) local=(0, 7, 7) row=1 array=(1, 7, 7, 0) storage_flat=1016
world_xyz_m: (1.062500000, -0.062500000, -0.062500000)
camera_xyz_m: (0.062500000, 0.062500000, 1.062500000)
projected_uv: (0.617647059, 0.617647059)
pixel_uv: (1, 1)
depth_decoded: yes
measured_depth_m: 1.000000000
signed_distance_m: -0.062500000
evaluation: contributes
proposed_delta: tsdf_sum=-0.125000000 weight=1
contributions_applied: 0
fusion_performed: no
storage_mutated: no
missing_blocks_created: no
artifact_written: no
```

The voxel center transforms to camera depth `1.0625 m`. The nearest projected
pixel contains `1.0 m`, so:

```text
signed distance = 1.0 - 1.0625 = -0.0625 m
TSDF sum delta  = -0.0625 / 0.5 = -0.125
weight delta    = 1
applied deltas  = 0
```

The final boundary lines are as important as the numerical result. The
temporary storage still has zero sums and weights, no missing block was
created, no fusion ran, and no output artifact was written.

## Exact prepared-context fixture proof

Evaluate the same observation and voxel from the immutable replay/depth
context:

```powershell
.\.venv\Scripts\python.exe -m spatialforge reconstruct tsdf-block-context-contribution `
  outputs\progress-blocks.sftplan `
  tests\fixtures\minimal.vgsession `
  --observation-sequence 0 `
  --voxel 8 -1 -1
```

The exact output is:

```text
TSDF BLOCK CONTEXT CONTRIBUTION CHECK scan-synthetic-0001
artifact: valid
session_replay: matched
context_selection: frame_stride=1 total=2 selected=2
context_immutable: yes
depth_source: replay-depth-context
observation_sequence: 0
voxel: global=(8, -1, -1) block=(1, -1, -1) local=(0, 7, 7) row=1 array=(1, 7, 7, 0) storage_flat=1016
world_xyz_m: (1.062500000, -0.062500000, -0.062500000)
camera_xyz_m: (0.062500000, 0.062500000, 1.062500000)
projected_uv: (0.617647059, 0.617647059)
pixel_uv: (1, 1)
depth_decoded: yes
measured_depth_m: 1.000000000
signed_distance_m: -0.062500000
evaluation: contributes
proposed_delta: tsdf_sum=-0.125000000 weight=1
evaluation_replay_hashing: no
evaluation_depth_decoding: no
contributions_applied: 0
storage_mutated: no
voxel_observation_traversal_performed: no
voxel_address_traversal_performed: no
fusion_block_traversal_performed: no
full_fusion_performed: no
missing_blocks_created: no
artifact_written: no
context_persisted: no
plan_sha256: 372c7c5d49eff1a30317ceb8b67cb3c40683f1d9d2a6179a049772f763d6f79d
replay_digest_sha256: dc001ae0ca01004a21ad227b12d57a7350bbc23904040453ca73fd955ef050b8
```

The command performs source I/O before this evaluation stage: it builds the
full selected-observation context and allocates replay-matched temporary
storage. `depth_decoded: yes` describes the already prepared record.
`evaluation_replay_hashing: no` and `evaluation_depth_decoding: no` describe
only the subsequent scalar evaluation. Its camera point, projection, pixel,
measurement, signed distance, status, and proposed delta exactly match the
session-backed proof above.

The remaining boundary lines prove that this checkpoint still evaluates only
sequence zero at one address. It applies no contribution, mutates no storage,
does not invoke selected-observation or address traversal, creates no block,
and writes or persists no artifact.

## Mutation consumers

An accepted contribution can now be applied to one destination slot:

```python
receipt = apply_tsdf_voxel_contribution(
    storage,
    contribution,
    session,
)
```

The updater checks the contribution's plan and replay provenance, re-resolves
its address, replay-checks the current session around the mutation, and returns
an immutable before/after receipt. It does not make this evaluator mutating.
It is unchanged by the prepared-context evaluator and still requires a current
`ScanSession`; this checkpoint does not add context-bound mutation.
The exact update, rollback, and repeated-application rules are documented in
[`tsdf-voxel-update.md`](tsdf-voxel-update.md).

The selected-observation traversal still calls the session-backed evaluator
for one fixed address and every replay observation selected by the plan:

```python
traversal = traverse_tsdf_voxel_observations(
    storage,
    address,
    session,
)
```

It evaluates every selected sequence before applying any accepted result, then
uses the separate updater in canonical observation order. Skips remain
immutable diagnostics and are not passed to the updater. The traversal is a
consumer of the session-backed read-only evaluator; it does not call
`evaluate_tsdf_voxel_contribution_from_context`. See
[`tsdf-voxel-traversal.md`](tsdf-voxel-traversal.md).

## Explicitly deferred

- directly iterating additional observations inside this evaluator; the
  separate traversal covers all plan-selected observations for one address,
  while other voxels, blocks, frusta, and rays remain deferred;
- applying proposed deltas within this read-only evaluator; the separate
  update primitive applies one accepted delta only;
- context-bound guarded application and rewiring selected-observation
  traversal to use the prepared-context evaluator;
- normalization, sensor-dependent weighting, an observation ledger, or a
  block-wide multi-observation fusion loop;
- deciding or planning full camera-to-surface free-space block coverage;
- dynamic block insertion, eviction, streaming, or persistent block storage;
- block-backed `.sftsdf` output and sparse-aware surface or mesh consumers;
- color, confidence, normals, robust depth/pose outlier filtering, visibility,
  and occlusion;
- optimized CPU, parallel, Open3D, GPU, adaptive-resolution, submap, or
  full-sequence implementations; and
- floor/wall/opening extraction, Inspector work, pose estimation, SLAM,
  semantics, localization, and `SpatialMapPackage` export.
