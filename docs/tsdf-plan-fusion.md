# Resumable, ledgered plan fusion

Fusion until now was strictly one-shot. `traverse_tsdf_plan_blocks_from_context`
demanded canonical all-zero storage, fused every planned row, and rolled the
*whole* volume back on failure. That is a fine reference path and it stays, but
it cannot resume, cannot be run in bounded chunks, and cannot be safely
repeated.

```powershell
.\.venv\Scripts\python.exe -m spatialforge reconstruct tsdf-block-context-plan-fuse `
  outputs\progress-blocks.sftplan `
  tests\fixtures\minimal.vgsession `
  --block-limit 3
```

## The ledger replaces the empty-storage guard

The blanket "storage must be empty" rule is replaced by a weaker, more useful
invariant:

> Storage must be **consistent with its ledger**: every planned row the ledger
> does not claim must still be untouched.

Rows the ledger *does* claim may hold anything, because fusion put it there.
A nonempty row the ledger does not claim means storage was written outside
this path, and resuming on top of it could silently double-count — so it is
rejected outright rather than fused into.

```python
ledger = begin_tsdf_fusion_ledger(plan)
while not ledger.is_complete:
    receipt = fuse_tsdf_plan_blocks_from_context(
        storage, context, ledger, block_limit=3,
    )
    ledger = receipt.ledger_after
```

`TsdfFusionLedger` is frozen and carries the plan digest, replay digest, the
full canonical plan row tuple, and the subset already fused. It validates that
the fused subset really is a subset of the plan and that both tuples are
canonically ordered.

## Idempotency

Rows the ledger already claims are **skipped, not re-fused**. Running a
completed pass again fuses zero rows, applies zero contributions, returns an
empty receipt tuple, and leaves storage byte-identical. The CLI proves this on
every run by deliberately issuing one extra pass after completion and
reporting `idempotent_repeat: yes`.

The receipt also enforces monotonicity: `ledger_before.fused ⊆ ledger_after.fused`,
so a pass can never un-fuse a row, and the retained block receipts must equal
exactly the ledger delta in canonical order.

## Proof obligation: resuming must equal not resuming

The property that matters is that chunking changes nothing:

```text
fuse in chunks of 1  ->  8 passes
fuse in chunks of 3  ->  3 passes
fuse in one pass     ->  1 pass
```

All three must leave storage **byte-identical** to what the one-shot
`traverse_tsdf_plan_blocks_from_context` produces on fresh storage, and their
per-pass `weight_delta` values must sum to that traversal's total. On the
fixture that is `380 + 496 + 292 = 1168`, the same 1,168 contributions over 584
slots every other checkpoint reports.

## Rollback scope is now per pass, not per volume

The one-shot traversal restores *all* planned storage on a caught failure,
which is correct when it owns the whole volume. A resumable pass must not do
that — it would destroy work committed by earlier passes.

So a failed pass zeroes only the rows **it** fused. Earlier rows survive, and
because the caller still holds the pre-pass ledger, storage and ledger remain
consistent and the pass can simply be retried. A test fuses three rows, injects
a failure into the next pass, asserts storage is byte-identical to the
three-row state, and then resumes to completion.

If that restore itself fails, the error says so explicitly rather than leaving
a silently inconsistent volume. That path is real: an early draft of this
checkpoint used `TSDF_SUM_DTYPE(0.0)` to zero rows, which raises because the
constant is a dtype *instance* and not callable. The guard caught it and
reported a failed rollback instead of quietly corrupting storage.

## What this still does not do

- **The ledger is per block, not per observation.** A row is either fully
  fused for the complete selected-observation tuple or not at all. Adding new
  observations to an already-fused row is not supported; that needs an
  observation-level ledger and is the natural next refinement.
- **The ledger is in memory only.** Nothing is persisted, so resumption works
  within a process, not across runs. Persisting it belongs with the persistent
  block-backed TSDF artifact.
- Storage itself is still temporary and discarded when the command exits.
- This is exclusive-access, in-process resumption: not crash atomicity, not
  thread safety, and not a concurrent writer protocol.

## Precisely deferred next phases

1. an observation-level ledger, so a row can absorb newly selected
   observations without being re-fused from scratch;
2. persist the ledger alongside a block-backed TSDF artifact, making
   resumption survive process exit;
3. complete fusion diagnostics, normalization, and sparse surface/mesh
   consumers over the fused volume;
4. evidence thresholds, confidence and sensor-dependent weighting, outlier
   rejection, and the visibility/culling policy;
5. scalable streaming, optimized CPU, parallel, or GPU execution; and
6. real-dataset accuracy, production meshing, structural mapping, Inspector
   work, pose estimation, SLAM, semantics, localization, and
   `SpatialMapPackage` export.
