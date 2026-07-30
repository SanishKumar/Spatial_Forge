# One-slot TSDF voxel update

This checkpoint evaluates one replay-selected observation at one planned voxel
and applies its accepted contribution to exactly one temporary accumulator
slot:

```powershell
.\.venv\Scripts\python.exe -m spatialforge reconstruct tsdf-block-contribution-apply `
  outputs/progress-blocks.sftplan `
  tests/fixtures/minimal.vgsession `
  --observation-sequence 0 `
  --voxel 8 -1 -1
```

It is the scalar mutation primitive that a later block-fusion loop can call.
It does not evaluate contributions for other observations or perform
contribution/fusion traversal over other voxels, blocks, frusta, or rays. The
storage exists only in this process, is discarded when the command exits, and
is not serialized.

## Public API and receipt

```python
receipt = apply_tsdf_voxel_contribution(
    storage,
    contribution,
    session,
)
```

The inputs are:

- canonical temporary `TsdfBlockStorage` allocated from a strict-loaded plan;
- one frozen `TsdfVoxelContribution` whose status is `contributes`; and
- the current loaded `ScanSession`.

Skipped contribution statuses are not silent no-ops. The updater rejects them
before mutation, and no receipt is returned.

A successful call returns a frozen `TsdfVoxelUpdateReceipt` containing:

```text
contribution
tsdf_sum_before
weight_before
tsdf_sum_after
weight_after
```

The retained contribution supplies the immutable address, observation
sequence, source plan and replay digests, TSDF sum delta, weight delta, and
projective diagnostics. A successful receipt therefore records one complete
before/delta/after transition without copying those source fields into a
second mutable structure.

## Destination and accumulator preflight

Before either scalar is written, the updater requires:

- an accepted finite TSDF sum delta in `[-1, 1]` with weight delta one;
- a contribution source-plan digest equal to the destination plan artifact
  digest;
- a contribution replay digest equal to the destination plan replay digest;
- a destination plan session ID equal to the loaded session ID;
- a contribution address that re-resolves exactly in the destination storage;
- base NumPy sum and weight arrays with the canonical shape, dtypes, C order,
  writable flags, and payload size, where the two arrays do not overlap each
  other;
- a finite current sum and a current weight inside uint32;
- canonical positive `0.0` whenever the current weight is zero;
- a current nonzero-weight sum inside its `[-weight, +weight]` envelope; and
- a finite resulting sum inside the new weight envelope.

If the current weight is `4,294,967,295`, applying weight one fails before
mutation. The updater neither wraps nor saturates uint32 weights.

Only `contribution.address.array_index_bzyx` is updated. The block tuple, array
objects, array layout, all other slots, and allocation size remain unchanged.

## Replay bracket and rollback

The contribution carries:

```text
source_plan_digest_sha256
replay_digest_sha256
```

The updater first compares those values with the destination plan, then
replays the current session immediately before reading and writing the target.
After writing the sum and weight, it verifies the array layout and scalar
values and replays the session again.

Under exclusive single-thread access, a caught write-verification failure,
post-write replay error, or changed replay digest triggers restoration of both
target scalars from their saved values. If that restoration itself fails, the
updater raises an explicit error that storage may be inconsistent.

This is bounded in-process exception rollback, not a persistent transaction.
It is not crash-atomic, process-interruption-safe, lock-protected, thread-safe,
or safe for concurrent writers. SHA-256 matching identifies bytes but does not
authenticate or establish trust in the plan or session.

## Exact fixture proof

The committed fixture contribution resolves to:

```text
global=(8, -1, -1)
block=(1, -1, -1)
local=(0, 7, 7)
row=1
array=(1, 7, 7, 0)
storage_flat=1016
```

Its previously verified projective delta is TSDF sum `-0.125` and weight one.
Run the command at the top of this document. The relevant output is:

```text
TSDF BLOCK CONTRIBUTION APPLY CHECK scan-synthetic-0001
artifact: valid
session_replay: matched
observation_sequence: 0
voxel: global=(8, -1, -1) block=(1, -1, -1) local=(0, 7, 7) row=1 array=(1, 7, 7, 0) storage_flat=1016
evaluation: contributes
slot_before: tsdf_sum=0.000000000 weight=0
applied_delta: tsdf_sum=-0.125000000 weight=1
slot_after: tsdf_sum=-0.125000000 weight=1
storage_before: nonzero_sums=0 nonzero_weights=0 unknown_voxels=4096
storage_after: nonzero_sums=1 nonzero_weights=1 unknown_voxels=4095
contributions_evaluated: 1
contributions_applied: 1
storage_slots_updated: 1
fusion_block_traversal_performed: no
ray_traversal_performed: no
full_fusion_performed: no
missing_blocks_created: no
artifact_written: no
storage_persisted: no
```

The receipt returned by this successful call records:

```text
before = (tsdf_sum 0.0,    weight 0)
delta  = (tsdf_sum -0.125, weight 1)
after  = (tsdf_sum -0.125, weight 1)
```

Exactly one unknown slot becomes observed, so the nonzero sum and weight counts
change from zero to one and the unknown count changes from 4,096 to 4,095.
Weight one means one observation; it must not be reported as a multi-view fused
voxel.

## Repeated application

The updater has no observation ledger, duplicate key, or idempotency token.
Each successful call applies the accepted delta again. Applying this same
fixture contribution a second time to the same live storage would produce:

```text
before = (tsdf_sum -0.125, weight 1)
delta  = (tsdf_sum -0.125, weight 1)
after  = (tsdf_sum -0.250, weight 2)
```

That behavior is deterministic accumulation, not proof that two independent
sensor observations were fused. A future traversal layer must decide ordering
and prevent accidental duplicate application.

## Explicitly deferred

- iterating replay-selected observations or planned voxel/block ranges;
- duplicate detection, idempotency, ordering, batching, and per-observation
  provenance ledgers;
- camera-to-surface free-space block planning, frustum/ray traversal,
  visibility, and occlusion;
- normalization into final TSDF values, sensor-dependent weighting, weight
  caps below uint32, and robust depth/pose outlier filtering;
- dynamic block insertion, eviction, streaming, adaptive resolution, submaps,
  parallel writers, Open3D, GPU, and optimized full-sequence execution;
- persistent or crash-atomic transactions, checkpoints, a block-backed
  `.sftsdf`, and sparse-aware surface or mesh consumers;
- full block fusion, coverage or geometry-quality claims, color, confidence,
  normals, and topology updates; and
- floor/wall/opening extraction, Inspector work, pose estimation, SLAM,
  semantics, localization, and `SpatialMapPackage` export.
