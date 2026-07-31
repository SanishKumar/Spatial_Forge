# Immutable selected-observation replay/depth context

This checkpoint materializes the complete canonical observation selection
owned by one strict-loaded TSDF block plan into a frozen in-memory
`TsdfReplayDepthContext`:

```python
context = build_tsdf_replay_depth_context(plan, session)
```

The builder replay-verifies the current `ScanSession` before and after
preparation. Each selected observation with exact depth and known pose is
recorded as ready, its pose is copied, and its aligned depth payload is decoded
exactly once into immutable C-contiguous float64 metric storage.

The result is a self-contained construction-time measurement snapshot for
scalar evaluation and a future traversal checkpoint. It is not a live view of
the `.vgsession` folder. The builder itself does not allocate or mutate TSDF
storage, evaluate a voxel, traverse voxel addresses or blocks, trace a frustum
or ray, plan free space, perform fusion, or write an artifact.

## Public API and immutable records

The public builder is:

```python
context = build_tsdf_replay_depth_context(
    plan,
    session,
)
```

Its inputs are a strict-loaded `TsdfBlockPlan` and the current validated
`ScanSession`. Observation sequences cannot be supplied or reordered by the
caller. The builder derives the complete canonical tuple selected by the
plan's frame stride:

```text
0, frame_stride, 2 * frame_stride, ... < total_observations
```

The frozen `TsdfReplayDepthContext` records:

```text
session_id
source_plan_digest_sha256
replay_digest_sha256
frame_stride
total_observations
selected_observation_sequences
camera
depth_scale_m
observations
valid_depth_samples
invalid_depth_samples
```

Each frozen `TsdfReplayDepthObservation` records its sequence, a stable
`TsdfReplayDepthStatus`, an optional copied `T_world_camera`, and an optional
metric `depth_m` frame. The context exposes derived selected, ready, decoded-
frame, sample, payload-byte, and nonzero per-status counts.

The stable observation statuses are:

| Status | Snapshot contents |
| --- | --- |
| `ready` | Copied pose and decoded metric depth frame. |
| `missing-depth` | Available pose is preserved; depth is `None`. |
| `missing-pose` | Both fields are `None`; the otherwise-present depth is deliberately not decoded. |
| `missing-depth-and-pose` | Both fields are `None`. |

Missing inputs are successful diagnostic records. A decode failure for a ready
observation aborts the complete build and returns no partial context.

## Bytes-backed metric depth

For each ready observation, raw depth samples are converted in canonical image
order to metres in a mutable float64 working frame. The builder then copies
that frame into immutable `bytes`. Each `depth_m` access creates a fresh NumPy
view from those bytes in the retained canonical image shape, so reshaping one
retrieved view cannot alter the record or later views.

Every public depth frame is therefore:

- an exact base `numpy.ndarray` with shape `(camera.height, camera.width)`;
- native float64 and C-contiguous;
- non-owning and non-writeable; and
- backed by immutable bytes, so both element writes and ordinary
  `setflags(write=True)` attempts fail.

Zero, negative, and non-finite metric samples remain present in the immutable
frame. They are counted as invalid measurements rather than treated as a
context-build failure. This preserves the later projective evaluator's ability
to produce its existing `depth-invalid` diagnostic.

## Plan validation and replay bracket

The builder performs these bounded stages:

1. Validate input types, plan identity and digest fields, the plan/session ID,
   and the aligned pinhole reconstruction contract.
2. Replay the session once and require its digest to equal the plan's replay
   digest.
3. Derive the complete selected sequence tuple and verify the plan's total,
   selected, paired, missing-depth, and missing-pose counters. Missing-depth
   and missing-pose counts may overlap for one observation.
4. Require the plan's valid plus invalid sample counts to equal the paired
   observation count times the calibrated image dimensions.
5. Preflight the retained numeric payload cap before the first depth decode.
6. Visit selected observations once in canonical sequence order, snapshot
   ready poses, decode each ready depth frame once, and count its actual valid
   positive finite and invalid metric samples.
7. Require those actual sample counts to equal the plan's recorded counts.
8. Replay the session once more and require the ending digest to equal both
   the starting replay digest and the plan replay digest.
9. Construct the frozen context only after every check succeeds.

If replay-visible sensor inputs differ at the ending check, the build is
rejected. Validation, decoding, or replay errors create no context artifact
and cannot mutate TSDF storage because this API accepts no storage object.

## Snapshot lifetime and source freshness

After a successful ending replay check, the context is a self-contained
measurement snapshot. Later edits to the session folder cannot change its
copied poses or immutable depth bytes. Its replay digest proves
construction-time provenance; it does not claim that the folder will remain
unchanged or monitor it after construction.

Rebuild the context when an operation must use current folder contents. A
future outer fusion transaction may instead replay-check once before and after
consuming a context if it requires a current-source liveness guarantee. That
consumer contract is deliberately not defined by this standalone checkpoint.

## Memory boundary

`MAX_TSDF_REPLAY_DEPTH_CONTEXT_BYTES` caps retained numeric depth payload at
512 MiB. The builder preflights:

```text
ready_observations * camera.width * camera.height * 8 <= 512 MiB
```

before decoding the first frame. The context also retains `O(selected
observations)` immutable record and pose metadata.

The 512 MiB limit is not a whole-process peak-memory bound. During construction
of one ready frame, memory additionally includes the decoder's raw result, one
mutable float64 working frame, and the immutable byte copy that becomes the
retained view. Lazy decoding, paging, eviction, and streaming are deferred.

## Exact two-observation fixture proof

Run:

```powershell
.\.venv\Scripts\python.exe -W error -m spatialforge reconstruct tsdf-block-replay-context `
  outputs\progress-blocks.sftplan `
  tests\fixtures\minimal.vgsession
```

The relevant output is:

```text
TSDF BLOCK REPLAY DEPTH CONTEXT CHECK scan-synthetic-0001
artifact: valid
session_replay: matched
selection: frame_stride=1 total=2 selected=2
observation[0]: sequence=0 status=ready depth_decoded=yes
observation[1]: sequence=1 status=ready depth_decoded=yes
status_counts: ready=2
ready_observations: 2
depth_frames_decoded: 2
depth_layout: shape=(2, 2) dtype=float64 samples=8 payload_bytes=64
context_immutable: yes
tsdf_storage_allocated: no
voxel_evaluation_performed: no
voxel_observation_traversal_performed: no
voxel_address_traversal_performed: no
fusion_block_traversal_performed: no
ray_traversal_performed: no
full_fusion_performed: no
artifact_written: no
context_persisted: no
plan_sha256: 372c7c5d49eff1a30317ceb8b67cb3c40683f1d9d2a6179a049772f763d6f79d
replay_digest_sha256: dc001ae0ca01004a21ad227b12d57a7350bbc23904040453ca73fd955ef050b8
```

The committed plan selects observation sequences `(0, 1)`. Both have exact
depth and known pose. Their immutable `2 x 2` float64 metric frames contain
`1.0` and `0.9500000000000001`, respectively. That is two decoded frames,
eight samples, and 64 retained numeric payload bytes.

The final boundary labels are as important as the depth values. This command
does not allocate TSDF storage, evaluate or update a voxel, traverse selected
observations for a voxel, visit another address or block, perform fusion, or
write or persist an artifact. The context exists only for the life of the
process.

## First scalar consumer; not yet consumed by traversal

One prepared observation can now be evaluated at one planned voxel without
source I/O:

```python
contribution = evaluate_tsdf_voxel_contribution_from_context(
    storage,
    address,
    context,
    observation_sequence,
)
```

The evaluator validates the context against the destination plan, reads the
selected copied pose and immutable metric depth, and returns the existing
frozen `TsdfVoxelContribution` or skip status. It performs no evaluation-time
replay hashing or depth decoding and does not mutate storage or iterate another
observation or voxel. Context building remains the earlier I/O stage: it must
replay-check the session and decode ready frames before this evaluator runs.

`traverse_tsdf_voxel_observations` does not accept or use this context-aware
primitive yet. It still calls the session-backed scalar evaluator and updater,
which repeatedly replay and hash session inputs, and depth is still decoded
through the per-observation evaluation path for that traversal.

The next checkpoint is context-bound guarded application of one accepted
contribution to one addressed slot, preserving the current provenance,
overflow, byte-verification, and rollback guarantees without replaying the
session for every update. Only after that should one-voxel traversal be rewired
to the context-aware scalar primitives while preserving its transcript,
canonical sequential float64 application, duplicate guard, and traversal-wide
rollback.

## Explicitly deferred

- consuming the context in guarded scalar application or single-voxel
  selected-observation traversal;
- iterating planned voxel addresses, blocks, frusta, or camera rays;
- culling, visibility, occlusion, and camera-to-surface free-space planning;
- TSDF storage allocation or mutation, complete block fusion, normalization,
  color, confidence, normals, or topology updates;
- a persistent/resumable observation ledger, nonempty-target continuation,
  cross-call idempotency, and multi-slot batch transactions;
- context serialization, persistent or crash-atomic checkpoints, block-backed
  `.sftsdf` output, and sparse-aware surface or mesh consumers;
- lazy depth decoding, paging, eviction, streaming, parallel construction,
  optimized CPU, Open3D, GPU, adaptive resolution, and submaps;
- perpetual folder freshness, file watching, thread safety, concurrent source
  mutation, and lock-protected consumers;
- robust depth/pose outlier filtering, sensor-dependent weighting, and
  configurable production bounds; and
- floor/wall/opening extraction, Inspector work, pose estimation, SLAM,
  semantics, localization, and `SpatialMapPackage` export.
