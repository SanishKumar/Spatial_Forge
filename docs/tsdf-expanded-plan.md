# Writing an expanded block plan

This is the first checkpoint in the project that produces a new plan artifact.
Everything up to here computed coverage, verdicts and a proposal in memory and
wrote nothing:

```powershell
.\.venv\Scripts\python.exe -m spatialforge reconstruct tsdf-block-plan-expand `
  outputs\progress-blocks.sftplan `
  tests\fixtures\minimal.vgsession `
  outputs\progress-blocks-expanded.sftplan
```

The command strict-loads the source plan, prepares one replay/depth context,
surveys the conservative footprint coverage, resolves every covered voxel,
approves the evidence-bearing blocks, and serializes the merged block set as a
new `.sftplan`.

## What is written, and what is carried through

Only four things change relative to the source plan:

```text
active_blocks              the merged, canonically ordered expanded set
planning counts            active/halo/planned_voxel_slots, min/max block index
activation.free_space_rule conservative-nearest-pixel-footprint
expansion                  new provenance object
```

Everything else — session id, replay digest, voxel size, block resolution and
extent, truncation, frame stride, every observation and depth-sample count,
and the whole `surface_blocks` tuple — is carried through unchanged, and a
test asserts that field by field.

`surface_blocks` deliberately stays as it was. The surface band did not move;
expansion only adds free-space blocks around it. The loader's existing rule
that every surface block must also be active still holds, because expansion
only ever grows the active set.

## The new provenance object

```json
"expansion": {
  "source_plan_sha256": "<digest of the plan this was expanded from>",
  "approval_rule": "covered-block-with-at-least-one-observed-voxel",
  "added_blocks": 8
}
```

An expanded plan is not interchangeable with a surface/truncation plan: its
active set now contains blocks that hold no measured surface at all. Recording
where it came from and by which rule keeps that difference auditable rather
than implicit. `free_space_rule` moves from `not-planned` to
`conservative-nearest-pixel-footprint` for the same reason — a plan that
contains free-space blocks must not keep claiming it plans no free space.

Both are validated on load. An unknown `free_space_rule`, a malformed
`source_plan_sha256`, an altered `approval_rule`, an unknown field inside
`expansion`, or a negative `added_blocks` are all rejected.

## Contract changes this required

The plan format is strict, so accommodating an expanded plan meant three
deliberate, narrow changes:

1. `free_space_rule` is now checked against a permitted set rather than pinned
   to one exact string. `not-planned` remains valid and is still what
   `plan_tsdf_blocks` writes.
2. `expansion` is now an accepted optional top-level object. Plans without it
   load exactly as before and report `expanded_from_plan_sha256` as `None`.
3. The pre-parse structural guard allowed at most four JSON objects — root,
   grid, activation, planning. It now allows five, for the optional expansion
   object. The corresponding test derives its count from the constant so the
   two cannot drift apart again.

`TsdfBlockPlan` gained `free_space_rule` and `expanded_from_plan_sha256` so
consumers can tell the two kinds of plan apart. The schema version is
unchanged: this is an additive, backward-compatible extension, and every plan
written before it still loads.

## Safety around writing

- The output must end in `.sftplan` and must not already exist.
- Writing the expanded plan over its own source is rejected explicitly, in
  addition to the existing must-not-exist rule.
- The write goes through the same atomic temp-file-then-replace path as the
  original planner, so a failure leaves no partial artifact — a test injects a
  write failure and asserts the output directory is left completely empty.
- The source plan is byte-identical afterwards, asserted in both the API and
  CLI tests.
- Encoding matches the existing artifact conventions exactly: ASCII, sorted
  keys, two-space indent, trailing newline. Writing the same proposal twice
  produces byte-identical files.

## Round-trip proof

The expanded plan is not merely written; it is proved usable. The tests write
it, then:

```text
load_tsdf_block_plan(output)                -> strict load succeeds
verify_tsdf_block_plan_replay(plan, session)-> replay binding still holds
allocate_empty_tsdf_blocks(plan, session)   -> 20,480 voxel slots
```

So the expanded plan is a first-class plan: it loads under the same strict
rules, still verifies against the session it was derived from, and allocates
storage for its larger block set. Loading it against a *different* session is
still rejected.

On the far-depth fixture the source plan holds 32 blocks and 16,384 voxel
slots; the expanded plan holds 40 blocks and 20,480 slots, having approved 40
of 52 covered blocks and added the 8 that were missing.

## What this still does not do

- Nothing fuses into the added blocks. Allocation works, but the fusion
  traversal still requires canonical empty storage and has no observation
  ledger, so a resumable or incremental pass over the expanded domain is the
  next checkpoint.
- The approval rule remains unweighted and unthresholded.
- The expanded plan records *how many* blocks were added and by which rule,
  but not *which observations* justified each one. Per-block observation
  provenance is deferred.
- Expansion is not iterative: expanding an already-expanded plan is untested
  and its `expansion` object would record only the immediate source.

## Precisely deferred next phases

1. replace the empty-storage guard with explicit observation provenance,
   idempotency, and resumable/nonempty fusion over the expanded domain;
2. complete fusion diagnostics, persistent block-backed TSDF artifacts,
   normalization, and sparse surface/mesh consumers;
3. evidence thresholds, confidence and sensor-dependent weighting, outlier
   rejection, and the visibility/culling policy;
4. scalable streaming, optimized CPU, parallel, or GPU execution; and
5. real-dataset accuracy, production meshing, structural mapping, Inspector
   work, pose estimation, SLAM, semantics, localization, and
   `SpatialMapPackage` export.
