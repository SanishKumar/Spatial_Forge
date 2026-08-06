# Plan expansion proposal

Every checkpoint so far has been strictly read-only, and this one still is —
but it is the first to answer the question the whole coverage stack was built
for: **which blocks should the plan actually hold?**

```powershell
.\.venv\Scripts\python.exe -m spatialforge reconstruct tsdf-block-context-plan-expansion `
  outputs\progress-blocks.sftplan `
  tests\fixtures\minimal.vgsession
```

It proposes the canonical expanded block set. It deliberately does **not**
serialize a plan, so the approval policy stays reviewable and testable
independently of the code that writes an artifact; writing the proposal out is
a separate step, in
[`tsdf-expanded-plan.md`](tsdf-expanded-plan.md).

## The approval rule

The conservative footprint survey is, by construction, a superset: it retains
any block a nearest-pixel wedge could touch, including blocks the wedge only
grazes. Allocating all of them would waste memory on coordinates no
measurement actually constrains.

The rule is therefore the narrowest one that uses the evidence already
computed:

```text
approve a covered block  <->  it contains at least one observed voxel
```

where *observed* means the block-wide cross-view sweep gave that voxel a
`surface` or `free-space` verdict. A block whose voxels are all `occluded` or
`unseen` carries no positive evidence and is rejected.

This prunes real waste. On the far-depth fixture the survey covers 52 blocks
and the rule rejects 12 of them — every one a block the wedge grazes but where
no voxel centre is ever actually sampled.

The rule is deliberately minimal, and the CLI says so:
`evidence_threshold_applied: no`. There is no support threshold ("at least N
observations"), no confidence or sensor weighting, and no outlier rejection.
One observed voxel is enough. Those refinements are later checkpoints; making
the policy explicit and separately testable first is the point.

## The merge, and what it never does

```text
expanded = source plan blocks  ∪  approved coverage blocks
```

The proposal **never removes a source plan block**, even if the coverage
domain found no evidence in it. `removed_block_count` is a property that
always returns zero and the receipt validates `source ⊆ expanded`. Expansion
is additive by construction: a surface/truncation block that this particular
selection could not corroborate is not evidence of absence, and silently
dropping planned geometry would be a far worse failure than retaining a block
that turns out empty.

## Public API and immutable record

```python
coverage = survey_tsdf_plan_pixel_footprints_from_context(plan, context)
domain = sweep_tsdf_coverage_domain_cross_view_from_context(
    plan,
    context,
    coverage,
)
proposal = propose_tsdf_plan_expansion_from_domain(plan, domain)
```

The domain must come from a `TsdfCoverageDomainCrossViewReceipt` whose plan
digest, replay digest, frame stride, total observations and block resolution
all match the plan being expanded; a domain resolved against a different plan
is rejected rather than silently merged.

The frozen `TsdfPlanExpansionProposal` retains provenance, the source plan
tuple, the domain tuple, the approved/rejected split and the expanded tuple.
Added blocks, retained and removed counts, and voxel-slot totals are derived
on access.

Construction re-derives the composition: every tuple must be canonical and
strictly ordered, approved and rejected must partition the domain exactly, the
expanded tuple must equal the canonical union of source and approved, the
source must be a subset of the expanded set, and the result must stay within
the 100,000-block reference limit.

## Exact fixture proof

The committed fixture's plan already covers its own domain, so it proposes no
change at all:

```text
source_plan: blocks=8 voxel_slots=4096
coverage_domain: blocks=8 approved=8 rejected=0
proposed_plan: blocks=8 voxel_slots=4096
proposed_delta: added=0 retained=8 removed=0 added_voxel_slots=0
expands_plan: no
```

That is a useful null result — the expansion rule does not invent work — but
it demonstrates nothing about expansion. A focused test measures both
observations at 3.0 m with `frame_stride=2`:

```text
source_plan: blocks=32 voxel_slots=16384
coverage_domain: blocks=52 approved=40 rejected=12
proposed_plan: blocks=40 voxel_slots=20480
proposed_delta: added=8 retained=32 removed=0 added_voxel_slots=4096
expands_plan: yes
```

Eight blocks of free space between the camera and the measured surface enter
the plan; twelve grazed-but-unobserved blocks are pruned; all thirty-two
original blocks are retained. That is the gap the whole-scan carvable set
measured in the previous checkpoint, now resolved into a concrete block set.

## Provenance, boundary, and cost

The proposal itself performs no replay, hashing, filesystem I/O or depth
decoding — it consumes already-resolved receipts and does set arithmetic. The
CLI does the earlier work: strict plan load, one context, one coverage survey,
one domain sweep. Its cost is therefore the domain sweep's cost, bounded by
that checkpoint's 262,144-outcome cap.

## What this still does not do

- No plan is written. `plan_written: no` and `source_plan_mutated: no` are
  literal: the `.sftplan` on disk is untouched and no new one is produced.
- No storage is allocated for the added blocks, so nothing can fuse into them
  yet.
- The approval rule is unweighted and unthresholded, as described above.
- There is still no visibility or culling policy, and no per-observation
  provenance is recorded for the added blocks — a future expanded artifact
  will need to say *why* each block was added.

## Precisely deferred next phases

The proposal is now serialized by
[`tsdf-expanded-plan.md`](tsdf-expanded-plan.md), which writes it as a new
`.sftplan` carrying the approval rule and source plan digest.

1. explicit idempotency and resumable/nonempty fusion over that expanded
   domain;
2. complete fusion diagnostics, persistent block-backed TSDF artifacts,
   normalization, and sparse surface/mesh consumers;
3. evidence thresholds, confidence and sensor-dependent weighting, outlier
   rejection, and the visibility/culling policy;
4. scalable streaming, optimized CPU, parallel, or GPU execution; and
5. real-dataset accuracy, production meshing, structural mapping, Inspector
   work, pose estimation, SLAM, semantics, localization, and
   `SpatialMapPackage` export.
