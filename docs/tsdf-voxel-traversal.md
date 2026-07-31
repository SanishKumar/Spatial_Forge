# Single-voxel selected-observation traversal

This checkpoint traverses every replay observation selected by a loaded TSDF
block plan for exactly one already planned voxel:

```powershell
.\.venv\Scripts\python.exe -m spatialforge reconstruct tsdf-block-voxel-traverse `
  outputs/progress-blocks.sftplan `
  tests/fixtures/minimal.vgsession `
  --voxel 8 -1 -1
```

It completes the observation-traversal axis for one address. It does not
traverse another voxel or block, trace a camera ray or frustum, decide
free-space block coverage, create missing blocks, or persist the temporary
accumulator.

## Public API and frozen receipt

```python
receipt = traverse_tsdf_voxel_observations(
    storage,
    address,
    session,
)
```

The inputs are:

- canonical temporary `TsdfBlockStorage` allocated from a strict-loaded plan;
- one `TsdfVoxelAddress` that resolves exactly in that storage; and
- the current loaded `ScanSession`.

The caller cannot supply observation sequences or change their order. The
traversal derives the complete tuple selected by the plan's frame stride:

```text
0, frame_stride, 2 * frame_stride, ... < total_observations
```

For the same reason, the CLI has no `--observation-sequence` option. The loaded
plan owns selection; `--voxel` selects only the single target address.

A successful call returns a frozen `TsdfVoxelTraversalReceipt` containing:

```text
address
source_plan_digest_sha256
replay_digest_sha256
frame_stride
total_observations
selected_observation_sequences
contributions
update_receipts
tsdf_sum_before
weight_before
tsdf_sum_after
weight_after
```

There is one immutable `TsdfVoxelContribution` for every selected sequence,
including stable skip results. `update_receipts` contains only the contributing
subsequence. Its before/after transitions form an exact ordered chain, and the
receipt exposes evaluated, applied, skipped, nonzero per-status, and updated-
slot counts derived from that transcript.

The final sum is the result of sequential float64 application in observation
order. It is not computed by reassociating all deltas, and the receipt stores
the accumulator sum and weight rather than a normalized TSDF value.

## Evaluate all, then apply

The traversal performs these bounded stages:

1. Validate the storage, address, plan/session identity, replay binding, and
   writable canonical array layout.
2. Require the addressed target to contain canonical positive `+0.0` and
   uint32 weight zero.
3. Derive the complete ascending sequence tuple selected by the plan.
4. Evaluate every sequence read-only with
   `evaluate_tsdf_voxel_contribution`, preserving its result order.
5. Validate the complete transcript and accepted-count capacity before the
   first write.
6. Recheck that evaluation did not change the target or storage layout.
7. Apply accepted contributions in the same order with
   `apply_tsdf_voxel_contribution`; skipped results are retained but never
   passed to the updater.
8. Replay-check once more, verify the original address and storage layout plus
   the exact expected final target bytes, and then construct the frozen
   traversal receipt inside the rollback guard.

Missing depth, missing pose, invalid sampled depth, out-of-view projection,
and the evaluator's other documented skip statuses are successful evaluated
outcomes. They increase the skipped count and do not abort traversal. Invalid
provenance, malformed storage, replay changes, depth-decoding errors, or other
contract failures raise an error.

Evaluating the complete transcript before mutation means an evaluation-stage
failure leaves the target untouched. It also separates the deterministic
measurement decision from ordered accumulation.

## Empty-target duplicate guard

One invocation derives each selected sequence once and evaluates it once. The
canonical empty-target precondition then provides a deliberately coarse guard
against applying a second successful traversal to the same live slot:

- after at least one accepted contribution, the target weight is nonzero, so
  another traversal is rejected before evaluation;
- if every selected observation was skipped, the target remains empty and a
  later traversal is allowed because no accumulator update is duplicated; and
- the lower-level scalar updater remains repeatable when called directly.

An accumulator sum and weight cannot identify which observations produced it.
The empty-target check is therefore not a per-observation ledger, an
idempotency token, or support for resuming a partly accumulated target. A
future block-fusion layer needs explicit provenance if it must distinguish an
intentional existing value from duplicate work. Out-of-band array resets are
also outside this guard's model.

## Traversal-wide rollback

The traversal saves the initial target state and wraps the complete application
phase. After a caught apply failure, final replay mismatch, storage-layout
change, or receipt-construction failure, it attempts to restore both target
scalars to their traversal-start values and verify the restoration. Earlier
successful contributions are not deliberately left applied when a later
contribution fails.

If restoration itself fails, an explicit error warns that storage may be
inconsistent. This is bounded in-process exception rollback under exclusive
access. It is not crash-atomic, process-interruption-safe, lock-protected,
thread-safe, safe for concurrent writers, or a persistent transaction.

## Exact two-observation fixture proof

The committed plan has `frame_stride=1`, `total_observations=2`, and therefore
selects sequences `(0, 1)`. Global voxel `(8, -1, -1)` resolves to array index
`(1, 7, 7, 0)` and storage flat index `1016`.

The relevant CLI output is:

```text
TSDF BLOCK VOXEL TRAVERSAL CHECK scan-synthetic-0001
artifact: valid
session_replay: matched
voxel: global=(8, -1, -1) block=(1, -1, -1) local=(0, 7, 7) row=1 array=(1, 7, 7, 0) storage_flat=1016
selection: frame_stride=1 total=2 selected=2
slot_before: tsdf_sum=0.000000000 weight=0
observation[0]: sequence=0 status=contributes delta_sum=-0.125000000 delta_weight=1
observation[1]: sequence=1 status=contributes delta_sum=-0.125000000 delta_weight=1
status_counts: contributes=2
accumulated_delta: tsdf_sum=-0.250000000 weight=2
slot_after: tsdf_sum=-0.250000000 weight=2
storage_before: nonzero_sums=0 nonzero_weights=0 unknown_voxels=4096
storage_after: nonzero_sums=1 nonzero_weights=1 unknown_voxels=4095
contributions_evaluated: 2
contributions_applied: 2
contributions_skipped: 0
duplicate_observation_applications: 0
storage_slots_updated: 1
voxel_observation_traversal_performed: yes
voxel_address_traversal_performed: no
fusion_block_traversal_performed: no
ray_traversal_performed: no
full_fusion_performed: no
missing_blocks_created: no
artifact_written: no
storage_persisted: no
plan_sha256: 372c7c5d49eff1a30317ceb8b67cb3c40683f1d9d2a6179a049772f763d6f79d
replay_digest_sha256: dc001ae0ca01004a21ad227b12d57a7350bbc23904040453ca73fd955ef050b8
```

`voxel_observation_traversal_performed: yes` is deliberately narrower than
voxel-address or block traversal. The corresponding `no` labels state that
this command stayed at one caller-supplied address; the focused isolation
tests verify that boundary independently.

Both observations have exact depth and known pose. Their ordered accepted
contributions are:

```text
sequence 0: tsdf_sum_delta = -0.125
            weight_delta   = 1
sequence 1: tsdf_sum_delta = -0.12499999999999978
            weight_delta   = 1
```

Sequential float64 application produces:

```text
before = (tsdf_sum  0.0,                 weight 0)
after  = (tsdf_sum -0.24999999999999978, weight 2)
```

Nine-decimal CLI formatting displays the final sum as `-0.250000000`. The
derived normalized value is approximately `-0.125`, but normalization is not
stored or performed by this checkpoint.

The proof should report two selected and evaluated observations, two accepted
and applied contributions, zero skips, zero duplicate applications, and one
updated storage slot. The other 4,095 planned slots remain unknown. Weight two
on this target proves that two distinct selected observation sequences were
accumulated for this one voxel; it does not prove block-wide or full-volume
fusion.

The CLI always allocates fresh empty storage, performs this traversal, reports
the in-memory result, and discards the storage when the process exits. Running
the command twice therefore repeats the same fresh-storage proof. It does not
demonstrate that calling the API twice on the same live storage is idempotent.

## Reference implementation cost

This checkpoint composes the existing strict evaluator and guarded scalar
updater. Those primitives replay-check their inputs around their own work, so
one traversal repeatedly replays and hashes session inputs for each selected
observation. Depth is also decoded through the existing single-observation
path. This intentionally redundant behavior preserves the established safety
contracts, but it is a diagnostic CPU reference rather than a scalable fusion
loop.

Before traversal expands to many voxel addresses, it needs a shared verified
replay/depth context, culling, and an execution design that does not repeatedly
rehash and decode the same inputs.

## Explicitly deferred

- iterating any other planned voxel, block, frustum, or camera ray;
- camera-to-surface free-space block planning, visibility, and occlusion;
- dynamic block insertion, eviction, streaming, adaptive resolution, and
  submaps;
- a persistent/resumable observation ledger, nonempty-target continuation,
  and general cross-call idempotency;
- multi-slot batch transactions, persistent or crash-atomic checkpoints,
  block-backed `.sftsdf` output, and sparse-aware surface or mesh consumers;
- normalized TSDF output, sensor-dependent or robust weighting, application
  weight caps below uint32, and depth/pose outlier filtering;
- scalable full-sequence execution, parallel writers, optimized CPU, Open3D,
  or GPU fusion;
- complete block fusion, coverage, color, confidence, normals, topology, or
  geometry-quality claims; and
- floor/wall/opening extraction, Inspector work, pose estimation, SLAM,
  semantics, localization, and `SpatialMapPackage` export.
