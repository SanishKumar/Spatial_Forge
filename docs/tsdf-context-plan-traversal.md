# Existing-plan block-set context traversal

This checkpoint traverses every block row that already exists in one loaded
TSDF block plan. It reuses one immutable `TsdfReplayDepthContext` across the
complete planned storage:

```powershell
.\.venv\Scripts\python.exe -m spatialforge reconstruct tsdf-block-context-plan-traverse `
  outputs\progress-blocks.sftplan `
  tests\fixtures\minimal.vgsession
```

There is no block selector. The strict-loaded plan owns the exact target tuple,
and the operation must visit every existing active row once in canonical order.
It does not recompute or expand that tuple, create a missing block, plan
camera-to-surface free space, or persist the temporary accumulator.

## Public API and frozen receipt

The public API is:

```python
receipt = traverse_tsdf_plan_blocks_from_context(
    storage,
    context,
)
```

`storage` must be canonical `TsdfBlockStorage` allocated from a strict-loaded
plan. Its block coordinates must equal `plan.active_blocks` exactly. `context`
must be the matching immutable selected-observation replay/depth snapshot. The
API accepts neither a `ScanSession` nor a caller-supplied block subset.

A successful call returns a frozen `TsdfPlanTraversalReceipt` with these stored
fields:

```text
source_plan_digest_sha256
replay_digest_sha256
block_resolution
frame_stride
total_observations
selected_observation_sequences
block_indices
planned_voxel_slots
block_receipts
```

`block_indices` is the complete canonical active-block tuple.
`planned_voxel_slots` must equal `len(block_indices) * 512`.
`block_receipts` contains exactly one `TsdfBlockTraversalReceipt` for every
coordinate and row. Each block receipt retains its 512 complete one-voxel
receipts, and each voxel receipt retains every selected contribution or skip
outcome.

The receipt derives block, voxel, observation-traversal, evaluated, applied,
skipped, observed, unknown, nonzero-sum, updated-slot, total-weight,
maximum-weight, prepared-depth-access, and stable status counts from that
nested transcript. It deliberately exposes no cross-block floating TSDF-sum
aggregate, which would add an unnecessary reassociation and rounding contract.

## Complete nested order

The caller cannot omit, duplicate, select, or reorder work. Execution order is:

```text
for block_row, block_index in enumerate(plan.active_blocks):
    # block order: X fastest, then Y, then Z
    for local_flat in 0..511:                  # local X, then Y, then Z
        for sequence in selected_observations: # ascending plan stride
```

The outer tuple must equal both `storage.block_indices` and
`plan.active_blocks`. Every child block receipt must match its exact coordinate,
zero-based row, plan/replay digests, resolution, and observation selection.
Each child then enforces the existing contiguous address and per-voxel receipt
contracts.

For the fixture, row zero begins at storage flat position `0`; row seven ends
at `4095`. The operation covers all 4,096 planned voxel slots exactly once.

## Workload preflight

This diagnostic path retains every contribution outcome. Before inspecting or
mutating accumulator values it computes:

```text
retained_outcomes = planned_voxel_slots * selected_observations
```

The public reference maximum is:

```text
MAX_TSDF_PLAN_TRAVERSAL_OUTCOMES = 262,144
```

A larger request is rejected before the first block traversal. The error tells
the caller to increase the plan frame stride, use a smaller plan, or wait for a
future scalable fusion path. This cap bounds the nested diagnostic transcript;
it is independent of the storage allocator's 64 MiB numeric-payload cap and the
context's 512 MiB retained-depth cap. None of those values is a whole-process
peak-memory guarantee.

## Whole-storage preflight and composition

After the workload check, the operation validates the storage layout,
plan/context provenance, canonical selection, and complete block-row tuple. It
then requires every byte in both accumulator arrays to be zero before invoking
the first child. Raw byte inspection rejects negative zero, nonzero weights,
NaNs, infinities, and any partly populated row.

The operation calls `traverse_tsdf_block_voxels_from_context` exactly once for
each canonical row and passes the identical context object every time. Each
block child calls the existing one-voxel traversal exactly once per address.
Consequently:

- evaluate-all-before-apply remains a **per-voxel** guarantee;
- earlier voxels and blocks may already be updated while a later voxel is
  evaluated;
- the parent does not precompute the complete plan transcript before its first
  write; and
- all numerical accumulation remains the existing sequential float64 behavior
  inside each voxel.

After the final child, the parent reconstructs the expected values from the
nested receipts one row at a time and verifies exact storage bytes before it
constructs the frozen plan receipt.

## Allocation-free whole-storage rollback

The successful preflight proves that the complete starting numeric state is
canonical all-zero bytes. The plan traversal therefore does not allocate a
second whole-storage snapshot. It records storage identity and, after any
caught child, postcondition, expected-byte, or receipt-construction failure,
restores the known starting state by filling the existing float64 and uint32
arrays with canonical zero.

Rollback then revalidates the array layout, plan identity, block tuple, and
all-zero bytes. This parent rollback is allocation-free and covers every
planned row, including rows successfully completed before a later failure.
Individual block children retain their own selected-row rollback guards.

If restoration or verification itself fails, an explicit error warns that
storage may be inconsistent. These guarantees assume exclusive in-process
access. They are not crash atomicity, process-interruption safety, locking,
thread safety, a persistent transaction, or protection from concurrent or
hostile out-of-contract writers.

## Empty-storage duplicate guard

The complete all-zero precondition is a coarse cross-call guard:

- after any accepted contribution, a second call on the same live storage is
  rejected before the first child;
- a partly populated or noncanonical storage is rejected in full; and
- if every outcome across every row is skipped, storage remains empty and a
  later call is allowed because no accumulator update was duplicated.

Accumulator sums and weights do not identify contributing observations. This
is not an observation ledger, idempotency token, resume protocol, incremental
fusion contract, or defense against out-of-band resets.

## Execution cost

For `B` existing blocks and `S` selected observations:

```text
voxel receipts       = B * 512
contribution outcomes = B * 512 * S
runtime               = O(B * 512 * S)
retained transcript   = O(B * 512 * S)
```

The parent adds no full-storage rollback copy. Each child uses the existing
6,144-byte selected-row rollback snapshot and bounded per-row expected buffers.
The final exact-byte check also constructs one expected 8 x 8 x 8 row at a
time. This is still a deterministic diagnostic CPU path with nested Python
receipts and repeated scalar validation, not a scalable, vectorized, parallel,
GPU, streaming, or production fusion backend.

## Exact fixture proof

The committed fixture has eight existing active blocks: four surface blocks at
`X=1` and four truncation-halo blocks at `X=0`. Its live command prints:

```text
TSDF BLOCK CONTEXT PLAN TRAVERSAL CHECK scan-synthetic-0001
artifact: valid
session_replay: matched
context_selection: frame_stride=1 total=2 selected=2
context_immutable: yes
depth_source: replay-depth-context
plan_blocks: active=8 surface=4 halo=4 resolution=8 voxel_slots=4096
block_order: plan-canonical-x-fastest rows=0..7
voxel_order: block-row-then-local-flat-x-fastest local_flat=0..511
first_block: index=(0, -1, -1) row=0 storage_flat_range=0..511
last_block: index=(1, 0, 0) row=7 storage_flat_range=3584..4095
first_voxel: global=(0, -8, -8) local=(0, 0, 0) array=(0, 0, 0, 0) storage_flat=0
last_voxel: global=(15, 7, 7) local=(7, 7, 7) array=(7, 7, 7, 7) storage_flat=4095
selection: frame_stride=1 total=2 selected=2
storage_before: nonzero_sums=0 nonzero_weights=0 unknown_voxels=4096
storage_after: nonzero_sums=584 nonzero_weights=584 unknown_voxels=3512
status_counts: contributes=1168 projection-outside-image=5440 behind-truncation=1584
plan_weight_sum_after: 1168
plan_max_weight_after: 2
context_provenance: matched
traversal_source_freshness: construction-time-context
traversal_session_replay: no
traversal_replay_hashing: no
traversal_source_io: no
traversal_depth_decoding: no
traversal_prepared_depth_access: yes
traversal_workload: retained_outcomes=8192 maximum=262144
blocks_traversed: 8
block_transcripts_retained: 8
voxel_addresses_traversed: 4096
voxel_transcripts_retained: 4096
voxel_observation_traversals: 4096
contributions_evaluated: 8192
contributions_applied: 1168
contributions_skipped: 7024
storage_slots_updated: 584
voxel_observation_traversal_performed: yes
voxel_address_traversal_performed: yes
fusion_block_traversal_performed: yes
fusion_block_traversal_scope: existing-plan-block-set-only
multiple_block_traversal_performed: yes
planned_block_set_traversal_performed: yes
all_existing_plan_blocks_traversed: yes
unplanned_blocks_visited: 0
free_space_coverage_planned: no
ray_traversal_performed: no
full_fusion_performed: no
missing_blocks_created: no
caught_failure_rollback_scope: complete-planned-storage
artifact_written: no
storage_persisted: no
context_persisted: no
plan_sha256: 372c7c5d49eff1a30317ceb8b67cb3c40683f1d9d2a6179a049772f763d6f79d
replay_digest_sha256: dc001ae0ca01004a21ad227b12d57a7350bbc23904040453ca73fd955ef050b8
```

The 8,192 outcomes are `8 blocks * 512 voxels * 2 observations`. Exactly
1,168 weight-one contributions are applied to 584 slots; 7,024 outcomes are
stable skips. The final integer weight sum is 1,168 and the maximum voxel
weight is two. The remaining 3,512 slots stay unknown.

`fusion_block_traversal_performed: yes` and
`planned_block_set_traversal_performed: yes` mean the contribution path covered
the complete block tuple already stored in this artifact. The scope line is
equally important: no unplanned block was visited or created.

## Context and source-I/O boundary

The CLI performs setup before calling the plan traversal. Allocation
replay-verifies the plan, and context construction replay-brackets the session
inputs and decodes each ready selected frame once. The later traversal accepts
only the completed storage and context.

During that traversal there is no session replay, source hashing, source-file
I/O, or depth decoding. Ready voxel evaluations read immutable metric depth
already retained by the context. Source freshness is therefore the context's
completed construction-time snapshot. Rebuild it, or run an explicit live
replay check outside the traversal, when current folder freshness is required.

## Why this is not full fusion

The `.sftplan` active tuple is an outward-conservative surface band. It contains
surface blocks and truncation-neighborhood halo blocks, but it deliberately
does not contain the dense reference path's complete camera-to-surface
free-space rays. Traversing every existing row proves complete execution over
that artifact; it does not prove that the artifact covers the desired full
fusion domain.

This checkpoint also has no visibility, occlusion, or culling rule; observation
ledger; nonempty resume policy; persistent TSDF artifact; stored normalization;
color, confidence, normals, or topology integration; or block-backed surface
and mesh consumer. `full_fusion_performed: no` is therefore intentional.

## Precisely deferred next phases

The next phase should separately define deterministic camera-to-surface
free-space activation and the associated visibility/culling semantics. It
should not simultaneously add persistence or optimization.

After that, separate checkpoints remain for:

1. explicit observation provenance, idempotency, and resumable/nonempty fusion;
2. complete fusion diagnostics over the chosen spatial domain;
3. a persistent block-backed TSDF artifact, normalization contract, and
   sparse surface/mesh consumers;
4. scalable traversal, bounded streaming, optimized CPU, parallelism, or GPU;
5. robust depth/pose filtering, production bounds, real-dataset accuracy, and
   production meshing; and
6. structural extraction, Inspector work, pose estimation, SLAM, semantics,
   localization, and `SpatialMapPackage` export.
