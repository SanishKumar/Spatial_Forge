# Vectorised block contribution evaluation

Every fusing path in this project ultimately calls
`evaluate_tsdf_voxel_contribution_from_context`, which answers one voxel
against one observation and builds a fully self-validating receipt for the
answer. That is the reference definition of the projective rule and it is
worth every cycle it costs — as a *definition*. As the hot loop it is the
reason the sparse block architecture runs about 17,000 voxel-observations per
second while the dense integrator it is meant to replace does equivalent work
in about a second.

This checkpoint answers the same question for all 512 voxels of one planned
block in a single NumPy pass.

```powershell
.\.venv\Scripts\python.exe -m spatialforge reconstruct tsdf-block-context-block-contributions `
  outputs\progress-blocks.sftplan `
  tests\fixtures\minimal.vgsession `
  --block 1 -1 -1 `
  --observation-sequence 0
```

## The contract

`evaluate_tsdf_block_contributions_from_context(storage, block, context,
observation_sequence)` returns a frozen `TsdfBlockContributionField` holding
three immutable, bytes-backed arrays in canonical local-flat order — x
fastest, the same order `TsdfBlockTraversalReceipt.voxel_receipts` uses:

```text
status_codes      uint8[512]    index into TSDF_CONTRIBUTION_STATUS_ORDER
tsdf_sum_deltas   float64[512]  the accepted delta, or +0.0 when skipped
weight_deltas     uint32[512]   1 when accepted, 0 when skipped
```

That order is not a coincidence: a storage row is a C-contiguous
`(8, 8, 8)` block indexed `(z, y, x)`, so its flat layout *is* local-flat
order. `storage.tsdf_sums[row].reshape(512)` lines up with these arrays
element for element, which is what the next checkpoint needs to apply them.

Provenance travels as `source_plan_digest_sha256` + `replay_digest_sha256`,
and `observation_status` records how the frame was prepared, so an unprepared
frame's uniform skip is distinguishable from a geometric one.

## Bit-identical, not close

The pinning rule for this checkpoint is stricter than usual. The vector path
must reproduce the scalar path's `status`, its integer `weight_delta`, and a
`tsdf_sum_delta` with the **same float64 bit pattern** — compared as
`float.hex()`, so `-0.0` and `0.0` are not interchangeable.

Meeting that is a matter of writing the arithmetic in the scalar path's exact
order:

- the world-to-camera product is written out term by term rather than as a
  matrix product, because a dot product is free to reassociate or fuse the
  multiply-add and that changes the last bits;
- `fx * x / z + cx` stays left-associated;
- the pixel rule stays `floor(u + 0.5)` after the half-open `[-0.5, w - 0.5)`
  bound, and the redundant-looking integer bound after it is kept, because a
  `u` one ulp below `w - 0.5` can round up to exactly `w` when `0.5` is
  added;
- the delta stays `clip(signed / truncation, -1, 1)`.

Voxels that fail an early gate still take part in the later arithmetic — that
is what makes it one vector pass — so their nonsense values are silenced with
`np.errstate` and never read. Pixel indices for dead voxels are pinned to
`(0, 0)` so the depth gather stays in bounds, and their samples are discarded
immediately.

## The status ladder

Statuses are assigned in the scalar evaluator's early-return order, first
failure wins:

```text
camera-point-nonfinite -> camera-z-nonpositive -> projection-nonfinite
  -> projection-outside-image -> depth-invalid
  -> signed-distance-nonfinite -> behind-truncation -> contributes
```

`missing-depth`, `missing-pose` and `missing-depth-and-pose` come from the
context's preparation status and apply uniformly to all 512 voxels.

`signed-distance-nonfinite` is unreachable in both paths: by the time it is
tested, the measured depth is finite and positive and the camera z is finite
and positive, so their difference cannot overflow. It is kept because the
scalar path keeps it, and a test pins that it stays unreached even at
`1e308` depths.

## What it is pinned against

| claim | pinned against |
|---|---|
| per-voxel status, weight, float64 bits | the scalar context evaluator, on every block and observation of `minimal.vgsession` |
| the same, on off-grid noisy data | the scalar context evaluator, on sampled room-fixture blocks and frames |
| the accept rule in aggregate | `traverse_tsdf_plan_blocks_from_context`: 8,192 evaluated, 1,168 accepted |
| the accept rule at scale | the room scan's recorded fusion run: 3,594,240 evaluated, 1,127,112 accepted |
| statuses no fixture reaches | the scalar `_evaluate_metric_observation`, on hostile poses and cameras |

The last row is the honest exception. `minimal.vgsession` and the room scan
between them only ever produce `contributes`, `projection-outside-image` and
`behind-truncation`. Reaching a non-finite camera point or a non-positive
camera z needs a pose no plausible scan produces, so those rungs are compared
against the scalar evaluator's own metric-frame entry point instead of through
a session.

## What it costs

Measured on the room fixture, 351 blocks at 40 mm voxels over 20 frames:

```text
scalar, per voxel        ~26,000 voxel-observations/sec
vectorised, per block   ~2,500,000 voxel-observations/sec
whole plan               3,594,240 voxel-observations in 1.45 s
```

The same sweep produces exactly the 1,127,112 accepted contributions that the
ledgered fusion run produced in 210 seconds. Evaluation is no longer what
makes that number expensive.

## Non-goals

This evaluates. It does not:

- **apply anything.** No slot is written, no weight is incremented, and no
  overflow preflight or rollback is involved. Storage is read only to resolve
  the block row, and tests assert its bytes are unchanged.
- **traverse more than one block or more than one observation.** Sweeping and
  accumulating across frames stays with the existing traversal and fusion
  paths, which own accumulation order and therefore the last bits.
- **replace the scalar path.** The scalar evaluator remains the definition and
  the reference, exactly as the dense integrator remains the reference for the
  sparse one.
- **accept unplanned blocks.** The block must resolve in the destination
  storage; an unplanned block is refused by name.
- **do any I/O.** No replay, no hashing, no source reads, no depth decoding —
  it reads the immutable metric depth the context already holds.

Wiring the field into fusion, so a real scan actually fuses at this speed, is
the next checkpoint.
