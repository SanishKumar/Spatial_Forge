# Context-bound one-slot TSDF voxel update

This checkpoint evaluates one prepared replay/depth-context observation at one
planned voxel and applies its accepted contribution to exactly one temporary
accumulator slot:

```powershell
.\.venv\Scripts\python.exe -m spatialforge reconstruct tsdf-block-context-contribution-apply `
  outputs\progress-blocks.sftplan `
  tests\fixtures\minimal.vgsession `
  --observation-sequence 0 `
  --voxel 8 -1 -1
```

It is the construction-time-provenance counterpart to the existing
session-backed scalar updater. It does not traverse another observation,
voxel, block, frustum, or ray. The temporary storage and context are discarded
when the command exits, and no artifact is written.

## Public API and frozen receipt

```python
receipt = apply_tsdf_voxel_contribution_from_context(
    storage,
    contribution,
    context,
)
```

The inputs are:

- canonical temporary `TsdfBlockStorage` allocated from a strict-loaded plan;
- one accepted frozen `TsdfVoxelContribution`; and
- the frozen `TsdfReplayDepthContext` whose construction-time provenance
  produced and supports that contribution.

Skipped contributions are rejected before mutation. A successful call returns
the same frozen `TsdfVoxelUpdateReceipt` used by the session-backed updater:

```text
contribution
tsdf_sum_before
weight_before
tsdf_sum_after
weight_after
```

The receipt records one exact before/delta/after accumulator transition. It
does not claim that another observation or voxel was processed.

## Context and destination preflight

Before either scalar is written, the updater requires:

- an accepted finite TSDF sum delta in `[-1, 1]` with weight delta one;
- context, contribution, destination storage, and plan source digests that
  agree exactly;
- matching replay digests and session identities;
- context selection and depth-sample counters consistent with the plan;
- a contribution observation sequence selected by the context whose record is
  `ready`;
- a contribution address that re-resolves exactly in the destination storage;
- base NumPy sum and weight arrays with canonical shape, dtypes, C order,
  writable flags, payload size, and non-overlapping memory;
- a finite current sum and uint32 current weight;
- canonical positive `0.0` whenever the current weight is zero;
- a current nonzero-weight sum inside its `[-weight, +weight]` envelope; and
- a finite resulting sum inside the new weight envelope.

Weight `4,294,967,295` is rejected before mutation. The updater neither wraps
nor saturates uint32 weights. Only the contribution address's exact
`array_index_bzyx` slot can be changed.

The updater validates the context observation's status and identity but does
not retrieve its metric depth payload. The numerical delta is already frozen
inside the accepted contribution.

## Construction-time freshness; no scalar source I/O

The context builder replay-checks the current session before and after copying
poses and decoding ready selected depth frames. After a successful build, its
digests and immutable measurement bytes describe that construction-time
snapshot. Later edits to the session folder cannot mutate the context.

The context-bound updater accepts no `ScanSession`. During scalar application
it therefore performs:

- no session replay;
- no replay hashing;
- no session-file or other source I/O; and
- no metric-depth access or decoding.

This removes the session-backed updater's per-contribution replay cost, but it
also changes the freshness guarantee. The context-bound updater cannot detect
session-folder changes after context construction. A caller that requires
current-source liveness must rebuild or externally replay-bracket the larger
operation. The context and SHA-256 digests identify source bytes; they do not
authenticate or establish trust in them.

## Exact writes and rollback

The updater saves both target scalars and their exact bytes, records the
storage layout identity, computes the float64 sum and uint32 weight transition,
and constructs the receipt before writing. It then writes and byte-verifies
exactly that target sum and weight while requiring the array objects and layout
to remain unchanged.

After the first scalar write, a caught write or post-write layout/value
verification failure triggers restoration and byte verification of both prior
target scalars. If restoration fails, the error explicitly warns that storage
may be inconsistent.

There is deliberately no post-write session replay in this variant, so a
source change cannot trigger scalar rollback. This remains bounded in-process
exception rollback under exclusive access. It is not crash-atomic,
process-interruption-safe, lock-protected, thread-safe, safe for concurrent
writers, or a persistent transaction.

## Exact fixture proof

The command at the top of this document prints:

```text
TSDF BLOCK CONTEXT CONTRIBUTION APPLY CHECK scan-synthetic-0001
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
slot_before: tsdf_sum=0.000000000 weight=0
applied_delta: tsdf_sum=-0.125000000 weight=1
slot_after: tsdf_sum=-0.125000000 weight=1
storage_before: nonzero_sums=0 nonzero_weights=0 unknown_voxels=4096
storage_after: nonzero_sums=1 nonzero_weights=1 unknown_voxels=4095
context_provenance: matched
application_source_freshness: construction-time-context
application_session_replay: no
application_replay_hashing: no
application_source_io: no
application_depth_access: no
contributions_evaluated: 1
contributions_applied: 1
storage_slots_updated: 1
voxel_observation_traversal_performed: no
voxel_address_traversal_performed: no
fusion_block_traversal_performed: no
ray_traversal_performed: no
full_fusion_performed: no
missing_blocks_created: no
artifact_written: no
storage_persisted: no
context_persisted: no
plan_sha256: 372c7c5d49eff1a30317ceb8b67cb3c40683f1d9d2a6179a049772f763d6f79d
replay_digest_sha256: dc001ae0ca01004a21ad227b12d57a7350bbc23904040453ca73fd955ef050b8
```

The command setup still performs source work. Allocation replay-verifies the
loaded plan, and context construction replay-brackets the snapshot and decodes
both ready `2 x 2` depth frames exactly once. `session_replay: matched` and
`depth_decoded: yes` describe those setup/evaluation inputs, not work performed
inside the scalar application.

The prepared evaluation proposes `(-0.125, 1)`, and the updater changes the one
addressed slot from `(0.0, 0)` to `(-0.125, 1)`. One of 4,096 slots becomes
observed; the other 4,095 remain unknown. Weight one means one observation, not
multi-view or block-wide fusion.

## Repeated application and traversal consumer

Like the session-backed scalar updater, this API has no observation ledger or
idempotency token. Reapplying the same accepted contribution to the same live
storage accumulates the same delta again when all preconditions still hold.

The existing `apply_tsdf_voxel_contribution` remains session-backed and still
replay-checks around each mutation. The existing session-backed
`traverse_tsdf_voxel_observations` also remains available as a live-source
reference path.

Its context-backed sibling consumes this updater together with the prepared
evaluator:

```python
traversal = traverse_tsdf_voxel_observations_from_context(
    storage,
    address,
    context,
)
```

It evaluates the complete context-selected transcript before mutation, then
passes accepted contributions to this updater in canonical sequence order.
Skips remain diagnostics. The traversal performs no replay, source I/O, or
depth decoding; evaluation reads prepared metric depth, while application
does not access depth. The canonical empty-target guard and whole-target
caught-failure rollback belong to the traversal layer.

## Explicitly deferred

- directly traversing addresses inside this scalar updater; the separate
  [`traverse_tsdf_block_voxels_from_context`](tsdf-context-block-traversal.md)
  consumer now covers all 512 addresses of one selected planned block, while
  any second block, frustum, or camera ray remains deferred;
- camera-to-surface free-space planning, visibility, culling, and occlusion;
- a persistent observation ledger, cross-call idempotency, nonempty-target
  resume, batching, and multi-slot transactions;
- persistent or crash-atomic checkpoints, block-backed `.sftsdf` output, and
  sparse-aware surface or mesh consumers;
- dynamic insertion, eviction, paging, streaming, parallel writers, optimized
  CPU, Open3D, GPU, adaptive resolution, and submaps;
- normalization, color, confidence, normals, topology, robust weighting, and
  depth/pose outlier filtering; and
- floor/wall/opening extraction, Inspector work, pose estimation, SLAM,
  semantics, localization, and `SpatialMapPackage` export.
