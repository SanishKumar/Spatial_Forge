# Single selected-block context traversal

This checkpoint traverses every voxel address in exactly one caller-selected,
already planned `8 x 8 x 8` TSDF block. Every voxel is evaluated against every
observation selected by the block plan through one shared immutable
`TsdfReplayDepthContext`:

```powershell
.\.venv\Scripts\python.exe -m spatialforge reconstruct tsdf-block-context-block-traverse `
  outputs\progress-blocks.sftplan `
  tests\fixtures\minimal.vgsession `
  --block 1 -1 -1
```

The command allocates temporary storage, builds the context, traverses the 512
addresses of the selected block, reports the result, and discards the storage.
It does not visit a second block, plan camera-to-surface free space, perform
plan-wide fusion, or write an artifact.

## Public API and frozen receipt

The public API is:

```python
receipt = traverse_tsdf_block_voxels_from_context(
    storage,
    block_index_xyz,
    context,
)
```

This API is also the exact child primitive used by
`traverse_tsdf_plan_blocks_from_context`. The parent derives the complete
active-block tuple and calls this operation once per canonical row; it does not
broaden this direct API's one-selected-block input contract. See
[`tsdf-context-plan-traversal.md`](tsdf-context-plan-traversal.md).

Its inputs are:

- canonical temporary `TsdfBlockStorage` allocated from a strict-loaded plan;
- one signed block coordinate already present in that storage; and
- a matching immutable `TsdfReplayDepthContext`.

The API accepts no `ScanSession` and no collection of block coordinates. A
valid signed coordinate absent from storage is an error for this operation;
the traversal never allocates or inserts a missing block.

A successful call returns a frozen `TsdfBlockTraversalReceipt`. It identifies
the selected block row and plan/context provenance and retains exactly 512
`TsdfVoxelTraversalReceipt` children. Those child receipts are the complete
per-voxel transcripts: each contains every plan-selected observation outcome,
the contributing update subsequence, and its exact accumulator transition.
Aggregate evaluated, applied, skipped, status, and updated-slot counts are
derived from the retained children rather than maintained as a separate
mutable counter.

The stored receipt fields are:

```text
block_index_xyz
block_row
source_plan_digest_sha256
replay_digest_sha256
block_resolution
frame_stride
total_observations
selected_observation_sequences
voxel_receipts
```

Derived properties expose voxel/address/traversal counts, observed and unknown
voxel counts, nonzero sums, evaluated/applied/skipped contribution counts,
updated slots, integer weight delta, maximum final weight, prepared-depth
access, and stable nonzero status counts. There is deliberately no
cross-voxel floating TSDF-sum aggregate: introducing one would add an
unnecessary reassociation and rounding contract.

## Canonical address order

The block resolution is fixed at eight, so the traversal derives every local
address from this complete range:

```text
local_flat = 0, 1, 2, ... 511
```

For each flat position:

```text
local_x = local_flat % 8
local_y = (local_flat // 8) % 8
local_z = local_flat // 64
```

This is X-fastest order: X changes in the inner loop, then Y, then Z. Each
local coordinate is composed with the selected signed block coordinate and
resolved through the existing address contract. The resolved address must
round-trip to the same block row, local coordinate, array index, and storage
flat position. The caller cannot omit, duplicate, or reorder addresses.

For fixture block `(1, -1, -1)`, canonical row `1` begins and ends at:

```text
first: local=(0, 0, 0) global=(8, -8, -8) array=(1, 0, 0, 0) storage_flat=512
last:  local=(7, 7, 7) global=(15, -1, -1) array=(1, 7, 7, 7) storage_flat=1023
```

No address from another block is a target of the call.

## Whole-block preflight and per-voxel execution

Before starting the first child traversal, the block operation validates the
storage, context, provenance, block coordinate, canonical writable array
layout, and all 512 selected-row accumulator slots. Every selected sum must be
canonical positive `+0.0` and every selected weight must be zero.

The whole-row empty check is deliberately stronger than waiting for each child
to discover its own target state. A partially populated block is rejected
before the first evaluation or write, so this checkpoint does not silently
resume an incomplete traversal.

After preflight, the operation invokes
`traverse_tsdf_voxel_observations_from_context` once for each canonical
address. Each child:

1. evaluates every context-selected observation before changing its own slot;
2. retains accepted and skipped outcomes in canonical observation order;
3. applies accepted contributions sequentially in float64; and
4. verifies its exact final accumulator state.

The evaluate-all-before-apply guarantee remains **per voxel**. The block
operation is intentionally voxel-major: earlier voxel slots may already be
updated while a later voxel's observations are being evaluated. It does not
precompute all 512 per-voxel transcripts before making the first block write.

## Execution cost

Runtime and retained transcript memory are both
`O(512 * selected_observations)`. The receipt keeps 512 complete child
receipts, including every selected observation outcome and accepted update
receipt. The 6,144-byte rollback snapshot is small, but it is not the dominant
whole-operation memory cost.

The context's 512 MiB retained-depth limit and the storage's 64 MiB numeric
limit are independent component limits, not a whole-process peak-memory bound
once the complete diagnostic transcript is included. The existing-plan parent
composes this receipt-heavy CPU reference path; neither layer is the optimized
representation for scalable or full fusion.

## Selected-block rollback

The operation snapshots the selected row's float64 sums and uint32 weights
before the first child traversal. If a later child, final validation, or
receipt construction raises a caught failure, it attempts to restore all 512
selected sums and weights to their exact traversal-start bytes and verifies
the restoration. Thus an earlier successful child is not intentionally left
committed after a later child fails.

If restoration itself fails, an explicit error warns that storage may be
inconsistent. This is bounded in-process rollback for one 6,144-byte numeric
block payload under exclusive access. It is not crash atomicity, process-
interruption safety, a lock, thread safety, a persistent transaction, or a
general multi-block transaction.

Only the selected row is a mutation target. Focused isolation tests compare
all other block rows before and after both success and injected failure paths.

## Empty-block duplicate guard

The canonical whole-block empty precondition is a coarse cross-call guard:

- if at least one selected voxel receives a contribution, a second call on
  the same live block is rejected because the block is no longer empty;
- if every observation is skipped for all 512 voxels, the block remains empty
  and a later call is allowed because no accumulator update was duplicated;
  and
- a manually or partly populated selected block is rejected in full.

Sums and weights do not identify which observation sequences produced them.
This guard is therefore not an observation ledger, idempotency token, resume
protocol, or proof against out-of-band array resets.

## Exact fixture proof

The committed fixture plan selects two observations and contains block
`(1, -1, -1)` at canonical row `1`. Traversing its complete local-flat range
prints this stable diagnostic transcript:

```text
TSDF BLOCK CONTEXT BLOCK TRAVERSAL CHECK scan-synthetic-0001
artifact: valid
session_replay: matched
context_selection: frame_stride=1 total=2 selected=2
context_immutable: yes
depth_source: replay-depth-context
block: index=(1, -1, -1) row=1 resolution=8 voxel_slots=512
storage_flat_range: 512..1023
address_order: local-flat-x-fastest local_flat=0..511
first_voxel: global=(8, -8, -8) local=(0, 0, 0) array=(1, 0, 0, 0) storage_flat=512
last_voxel: global=(15, -1, -1) local=(7, 7, 7) array=(1, 7, 7, 7) storage_flat=1023
selection: frame_stride=1 total=2 selected=2
block_before: nonzero_sums=0 nonzero_weights=0 unknown_voxels=512
block_after: nonzero_sums=102 nonzero_weights=102 unknown_voxels=410
storage_before: nonzero_sums=0 nonzero_weights=0 unknown_voxels=4096
storage_after: nonzero_sums=102 nonzero_weights=102 unknown_voxels=3994
status_counts: contributes=204 projection-outside-image=424 behind-truncation=396
block_weight_sum_after: 204
block_max_weight_after: 2
context_provenance: matched
traversal_source_freshness: construction-time-context
traversal_session_replay: no
traversal_replay_hashing: no
traversal_source_io: no
traversal_depth_decoding: no
traversal_prepared_depth_access: yes
voxel_addresses_traversed: 512
voxel_transcripts_retained: 512
voxel_observation_traversals: 512
contributions_evaluated: 1024
contributions_applied: 204
contributions_skipped: 820
storage_slots_updated: 102
blocks_traversed: 1
additional_blocks_visited: 0
voxel_observation_traversal_performed: yes
voxel_address_traversal_performed: yes
selected_block_traversal_performed: yes
fusion_block_traversal_performed: yes
fusion_block_traversal_scope: selected-planned-block-only
multiple_block_traversal_performed: no
planned_block_set_traversal_performed: no
free_space_coverage_planned: no
ray_traversal_performed: no
full_fusion_performed: no
missing_blocks_created: no
caught_failure_rollback_scope: selected-block
artifact_written: no
storage_persisted: no
context_persisted: no
plan_sha256: 372c7c5d49eff1a30317ceb8b67cb3c40683f1d9d2a6179a049772f763d6f79d
replay_digest_sha256: dc001ae0ca01004a21ad227b12d57a7350bbc23904040453ca73fd955ef050b8
```

The 1,024 evaluations are `512 addresses * 2 selected observations`. Exactly
102 voxel slots accept both observations, producing 204 applied weight-one
contributions. Its total integer weight is `204` and its maximum voxel weight
is `2`. Every other block row remains byte-identical to its initial zero state.

Weight two proves two distinct selected observations accumulated at each of
those 102 fixture voxels. It does not prove visibility-aware coverage,
plan-wide fusion, or a complete reconstructed volume. Normalized TSDF values
are derived from sum divided by weight; this checkpoint does not store a
separate normalized field.

`fusion_block_traversal_performed: yes` means all 512 slots in the selected
planned block were processed by the TSDF contribution path. The immediately
following scope and boundary lines restrict that statement to one existing
block: no second plan row, missing block, ray, or free-space coverage is
visited. It therefore does not contradict `full_fusion_performed: no`.

## Context and source-I/O boundary

The CLI performs setup before calling the block traversal: allocation
replay-verifies the loaded plan, and context construction replay-brackets the
session inputs and decodes each ready selected depth frame once. The later
block traversal accepts only the completed context.

During that traversal:

- replay and source hashing are not repeated;
- source RGB/depth files are not opened or read;
- depth frames are not decoded again;
- ready evaluations do read immutable prepared metric depth; and
- source freshness remains the context's construction-time snapshot.

Changes to the session folder after context construction are not detected by
this API. Rebuild the context, or run an explicit live replay check outside
the traversal, when current source-folder freshness is required.

## What this checkpoint is not

The selected `.sftplan` block is one candidate surface-neighborhood/halo block.
Enumerating its 512 existing slots does not design which additional
camera-to-surface blocks should exist. There is no ray or frustum traversal,
visibility or occlusion test, culling rule, dynamic block insertion, color,
confidence, normal, or topology integration.

The existing-plan parent now traverses every canonical block row with the same
shared context while retaining this complete child contract. That proves
execution over the artifact's current surface-band tuple, not free-space-aware
full fusion.

A separate diagnostic can now trace the thin camera-to-measured-surface block
rays for exactly one prepared observation. It reports existing and unplanned
coordinates without changing this one-block operation or the artifact. Those
receipts are now also aggregated across the complete selected tuple;
conservative nearest-pixel coverage, plan expansion, and fusion remain later
work. See [`tsdf-observation-block-rays.md`](tsdf-observation-block-rays.md)
and [`tsdf-plan-block-ray-survey.md`](tsdf-plan-block-ray-survey.md).

## Explicitly deferred

- visiting any second block in this call or accepting a block collection;
- dynamic block creation, multi-observation block-ray aggregation,
  conservative nearest-pixel free-space planning, frustum traversal, culling,
  visibility, and occlusion;
- a persistent/resumable observation ledger, nonempty-block continuation, and
  general cross-call idempotency;
- full fusion, normalization storage, color, confidence, normals, topology,
  and geometry-quality claims;
- persistent or crash-atomic checkpoints, block-backed `.sftsdf` output, and
  sparse-aware surface or mesh consumers;
- scalable full-sequence execution, parallel writers, optimized CPU, Open3D,
  GPU, paging, eviction, streaming, adaptive resolution, and submaps;
- robust depth/pose filtering, sensor-dependent weighting, and production
  bounds; and
- structure, Inspector work, pose estimation, SLAM, semantics, localization,
  and `SpatialMapPackage` export.
