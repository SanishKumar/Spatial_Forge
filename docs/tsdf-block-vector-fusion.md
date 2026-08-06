# Field-driven block fusion

[`tsdf-block-contributions.md`](tsdf-block-contributions.md) made evaluation
fast and proved it bit-identical to the scalar path, but it applied nothing.
Fusion still walked 512 voxels per block and issued one guarded scalar write
per accepted contribution, so a real scan still cost minutes. This checkpoint
closes that: one block is fused from one vector field per observation.

```powershell
.\.venv\Scripts\python.exe -m spatialforge reconstruct tsdf-block-context-block-fuse `
  outputs\progress-blocks.sftplan `
  tests\fixtures\minimal.vgsession `
  --block 1 -1 -1
```

## What it does

`fuse_tsdf_block_from_vector_fields(storage, block, context)` evaluates each
selected observation as a whole-block field, then applies the fields to the
block's storage row in canonical observation order — two array additions per
field, rather than 512 evaluations and up to 512 guarded writes.

A storage row is a C-contiguous `(8, 8, 8)` block indexed `(z, y, x)`, so its
flat layout *is* canonical local-flat order. The fields need no rearranging;
they line up with the row element for element.

## Why the order still gives identical bytes

Per-voxel accumulation is a chain of float64 additions, and addition is
commutative but **not** associative, so anything that reorders it changes the
last bits. Two properties keep that chain intact:

- **One contribution per voxel per field.** A single observation can accept a
  voxel at most once, so applying fields in canonical observation order
  performs exactly the scalar path's sequence of additions per voxel.
- **Skipped voxels carry `+0.0`.** `x + 0.0` is bit-preserving for every
  value the accumulator can hold. The one exception, `-0.0 + 0.0 = +0.0`,
  cannot arise: the accumulator starts at `+0.0`, and IEEE round-to-nearest
  produces `-0.0` from a sum only when both operands are `-0.0`.

So no masking is needed; the dead voxels add zero and the arithmetic is the
same arithmetic.

## The receipt re-derives its own result

`TsdfBlockVectorFusionReceipt` retains the starting sums and weights, the
ending sums and weights, and every field it applied. Its `__post_init__`
replays the retained fields onto the retained starting state and requires the
recorded ending state **byte for byte**. It also re-checks the weight envelope
(`|sum| <= weight`) and finiteness.

This is a genuine re-derivation rather than a restatement, and unlike the
evaluation receipt it is cheap — replaying 512-element additions costs
nothing next to producing them.

## What it is pinned against

| claim | pinned against |
|---|---|
| one fused block's bytes | `traverse_tsdf_block_voxels_from_context`, every active fixture block |
| every count on the receipt | the same traversal's `applied`, `skipped`, `slots_updated`, `max_weight`, `status_counts` |
| a whole fused plan's bytes | `traverse_tsdf_plan_blocks_from_context`, all 4,096 slots |
| the same on off-grid noisy data | the scalar block traversal, on sampled room-fixture blocks |
| the accept rule at scale | the room scan's recorded fusion run: 1,127,112 contributions, 81,292 observed voxels |

The whole-plan comparison is the strongest one: fusing the fixture block by
block through fields and fusing it in one scalar traversal produce identical
`tsdf_sums` and `weights` buffers, not merely identical totals.

## What it costs

The room scan — 351 blocks, 40 mm voxels, 20 frames, 3,594,240
voxel-observations:

```text
ledgered scalar fusion, recorded    ~210 s
field-driven block fusion             2.0 s
```

Same 1,127,112 contributions, same 81,292 observed voxels. The one-shot plan
traversal still refuses this scan on its 262,144-outcome cap, so fusing block
by block through fields is currently the fastest route that produces it at
all.

## Guarantees kept

- **Empty-row precondition.** The selected block must be canonically empty —
  weights zero and sums positive zero — exactly as the scalar block traversal
  requires. Re-fusing a fused block is refused, not silently doubled.
- **Overflow preflight.** The summed weight deltas are checked against the
  uint32 maximum before anything is written.
- **Envelope and finiteness checks** before the write, not after.
- **Byte-verified writes.** After writing, the row is re-read and compared to
  the expected bytes, and storage identity is re-checked.
- **Whole-row rollback.** A caught failure restores the row's exact starting
  bytes and verifies the restoration. Bounded in-process rollback under
  exclusive access — not crash atomicity, thread safety or persistence.
- **No I/O.** No replay, hashing, source reads or depth decoding.

## Non-goals

- **No ledger.** This is not resumable and not idempotent. It fuses one empty
  block completely or fails. Wiring fields into `TsdfFusionLedger` and
  `TsdfObservationLedger`, so resumption and partial-frame absorption run at
  this speed too, is the next checkpoint.
- **No plan-wide entry point.** Callers loop over blocks themselves; there is
  no cap accounting, no ordering guarantee across blocks beyond what the
  caller imposes, and no partial-plan receipt.
- **No second block.** One block per call, and tests assert every other row
  stays untouched.
- **The scalar traversal stays.** It remains the definition of a fused block
  and the reference this path is measured against.
