# Where the project stands

A snapshot of current state, open risks and immediate next steps. For the
full checkpoint list see [`roadmap.md`](roadmap.md); for how work is added see
[`working-method.md`](working-method.md).

## What SpatialForge is for

Turn calibrated indoor scans into **metric, semantic, localizable maps**, so
that someone entering that building later can be positioned inside it. TSDF
fusion, block planning and free-space carving are infrastructure. The product
is a map you can navigate and localize in. It stays independent of any
navigation product that consumes it.

That framing matters when prioritising: the geometry engine is a means, and
work that does not eventually serve "scan a space, localize in it" is not on
the critical path.

## Current state

The known-pose reconstruction path works end to end and is validated on
non-degenerate data.

**Reference geometry**: session format, validation, deterministic replay with
SHA-256 digests, TUM import, known-pose point clouds, dense and sparse TSDF
with byte-identical parity, automatic bounds, surface points, triangle mesh.

**Sparse block architecture**: block planning, strict `.sftplan` loading and
replay verification, allocation, signed voxel addressing, contribution
evaluation and guarded application, one-voxel/one-block/whole-plan traversal,
and an immutable replay/depth context that decodes each depth frame once.

**Free-space reasoning**, two independent ladders that meet:

```text
coverage   pixel wedge -> one observation -> whole selection
verdict    one voxel -> across observations -> across a block -> across the domain
```

Together they produce the whole-scan carvable free-space set, an expansion
proposal, and a written expanded `.sftplan`.

**Fusion**: resumable and idempotent, at block granularity and at
`(row, observation)` granularity, both byte-identical to the one-shot
traversal at any chunk size.

394 tests, `OK` under `-W error`, about 65 seconds.

## Open risks, most important first

1. **Performance.** The block path runs about 17,000 voxel-observations per
   second because it loops in Python; the dense integrator does the equivalent
   work vectorised in about a second. The sparse architecture intended to be
   the scalable one is roughly two orders of magnitude slower than the dense
   reference it replaces. A small room at 5 cm voxels is a minute; at 2 cm it
   is hours; a building floor is days. **This gates everything else.**
2. **No real sensor data yet.** The room fixture removes the old fixture's
   degeneracy but is still synthetic: clean gaussian noise, exact poses, no
   motion blur, rolling shutter, reflective surfaces or missing returns. The
   TUM importer exists and has never been pointed at a full sequence.
3. **Poses are ground truth.** There is no pose estimation, so the engine
   cannot take a phone capture. This is the largest single gap to the product
   idea and an entirely separate discipline.
4. **The one-shot traversal does not scale** and is now explicitly a
   tiny-fixture reference path, pinned as such by a test.
5. **Ledgers are in memory.** Fusion resumes within a process, not across
   runs.

## Two different finish lines

**Showable** — a real scan in, a rendered reconstruction out. Close: the
pipeline already produces a 60,072-triangle mesh from the room fixture. What
stands in the way is performance at useful voxel sizes and an actual TUM run.

**MVP of the idea** — someone scans a space, someone else localizes in it.
Far: needs pose estimation, SLAM, structure, semantics, localization and
`SpatialMapPackage`. Most of the original architecture is untouched.

## Immediate next steps

1. **Vectorise the block evaluator**, with the existing scalar path as its
   reference, exactly as the dense integrator is the reference for the sparse
   one. This is the critical path.
2. Persist a block-backed TSDF artifact together with its fusion ledger, and
   connect normalization plus sparse surface/mesh consumers.
3. Run a full TUM RGB-D sequence against its ground-truth trajectory and
   publish accuracy. Requires downloading a sequence, roughly 460 MB.

## Reproducing the current results

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -W error -m unittest discover -s tests -p "test_*.py"
```

The room-scan numbers, the scale limit and the performance measurements are in
[`real-scale-validation.md`](real-scale-validation.md). Every diagnostic
command and its expected output is in the repository README.
