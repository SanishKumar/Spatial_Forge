# Frame-major fusion with an observation ledger

[`tsdf-plan-fusion.md`](tsdf-plan-fusion.md) made fusion resumable, but its
ledger is per block: a row is either fully fused for the complete
selected-observation tuple or not at all. A capture does not arrive that way.
It arrives one frame at a time, each frame touching many rows.

```powershell
.\.venv\Scripts\python.exe -m spatialforge reconstruct tsdf-block-context-observation-fuse `
  outputs\progress-blocks.sftplan `
  tests\fixtures\minimal.vgsession `
  --pair-limit 5
```

This checkpoint tracks fusion at `(row, observation)` granularity, so a row can
hold a *partial* prefix of the selection and absorb the rest later.

## Why the ledger stores a prefix length, not a set

The obvious design is a set of absorbed observation sequences per row. The
prefix design is deliberately narrower, and the reason is numerical.

Per-voxel accumulation is a chain of float64 additions. Addition is
commutative, so `a + b == b + a` exactly, but it is **not associative**:
`(a + b) + c` and `a + (b + c)` can differ in the last bit. If a row could
absorb observation 2 before observation 0, its voxels would accumulate in a
different order than the one-shot traversal and the result could drift.

So each row records how many observations it has taken, and a row that has
taken `k` has taken exactly sequences `0..k-1`. Absorption in canonical order
is enforced, and that is what lets the checkpoint promise bit-identical
results at any chunk size.

## Frame-major walk

Pending pairs are walked observation-outer, row-inner:

```text
observation 0 -> row 0, row 1, ... row 7
observation 1 -> row 0, row 1, ... row 7
```

That is the order a streaming capture produces: one frame applied across every
row it touches, then the next frame. Because observation positions are visited
in increasing order, a row only reaches position `k` once it already holds
`0..k-1`, so the canonical-order rule holds without extra bookkeeping.

`pair_limit` bounds how many pairs one pass absorbs, which is what makes the
work chunkable. On the fixture, 8 rows by 2 observations is 16 pairs, so
`--pair-limit 8` is exactly "apply frame 0 everywhere, then frame 1".

## Proof obligation: chunking must not perturb the sum

Every pair limit must reproduce the one-shot traversal **byte for byte**:

```text
pair_limit 1     -> 16 passes
pair_limit 5     ->  4 passes
pair_limit 8     ->  2 passes
no limit         ->  1 pass
```

All four leave storage byte-identical to
`traverse_tsdf_plan_blocks_from_context` on fresh storage, and each applies
1,168 contributions in total — the same figure every other checkpoint reports.
The test asserts byte equality, not closeness, because the whole point of the
prefix rule is that the last bits agree.

## Rollback restores bytes, not zeros

The block-level ledger could roll back by zeroing, because any row it touched
started empty. That is no longer true here: a failing pass may write rows that
already hold earlier observations. Zeroing them would destroy committed work.

So a pass snapshots each row's exact bytes before first writing to it, and on a
caught failure restores those bytes. Earlier observations survive, the caller's
pre-pass ledger still describes storage accurately, and the pass can be
retried. A test absorbs frame 0 across all rows, injects a failure into the
frame-1 pass, asserts storage is byte-identical to the frame-0 state, then
resumes to completion.

The untouched-row guard is correspondingly narrower than the block version: it
only requires rows the ledger records as having absorbed *nothing* to still be
empty. Rows with a partial prefix legitimately hold data.

## What this still does not do

- **The ledger is still in memory.** Resumption works within a process, not
  across runs. Persisting it is the next checkpoint and is what makes a long
  capture genuinely restartable.
- Absorption order is fixed to canonical. Out-of-order absorption is rejected
  rather than supported, for the numerical reason above; supporting it would
  mean giving up bit-exactness or reworking accumulation.
- Storage is temporary and discarded when the command exits.
- This is exclusive-access, in-process resumption: not crash atomicity, not
  thread safety, not a concurrent writer protocol.
- The per-pair path evaluates all 512 voxels of a row per observation via the
  scalar primitives. It is deliberately the same arithmetic as the reference
  traversal, not a faster one.

## Precisely deferred next phases

1. persist the ledger alongside a block-backed TSDF artifact, so resumption
   survives process exit;
2. complete fusion diagnostics, normalization, and sparse surface/mesh
   consumers over the fused volume;
3. evidence thresholds, confidence and sensor-dependent weighting, outlier
   rejection, and the visibility/culling policy;
4. scalable streaming, optimized CPU, parallel, or GPU execution; and
5. real-dataset accuracy, production meshing, structural mapping, Inspector
   work, pose estimation, SLAM, semantics, localization, and
   `SpatialMapPackage` export.
