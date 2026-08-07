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

The known-pose reconstruction path works end to end and is validated on real
sensor data: 99 frames of TUM `rgbd_dataset_freiburg1_xyz` fused at 30 mm
voxels score a 9.3 mm median surface error against 98 held-out frames the
fusion never saw, with 99.8% of held-out depth landing in observed voxels.
See [`tum-validation.md`](tum-validation.md).

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

**Vector path**: one block against one observation evaluated in a single
float64 NumPy pass, bit-identical to the scalar evaluator; and whole-block
fusion from those fields, byte-identical to the scalar traversal. The room
scan fuses in 2 seconds against the scalar path's 210.

434 tests, `OK` under `-W error`, about 80 seconds.

## Open risks, most important first

1. **The planner is the new bottleneck.** Fusion went from 210 s to 2 s on the
   room scan, and the TUM run showed where the cost moved: block planning is a
   pure-Python per-pixel loop costing about 2 seconds per 640x480 frame — 201 s
   of a 246 s pipeline. Fusion is now a fifth of planning.
2. **The depth context cannot hold a full sequence.** It retains float64 metric
   depth for every selected frame against a 512 MB ceiling, so 640x480 caps out
   near 208 frames. The TUM run used 99 of 792 for that reason. Streaming or
   windowing is a requirement, not a tuning knob.
3. **Meshing fails on real data.** The fixed six-tetrahedron split produces
   non-manifold vertices on the TUM volume and the mesher correctly refuses.
   Surface-point extraction on the same volume succeeds, so the volume is sound
   and the triangulation is not.
4. **Poses are ground truth.** There is no pose estimation, so the engine
   cannot take a phone capture. This is the largest single gap to the product
   idea and an entirely separate discipline. The TUM result measures geometric
   self-consistency across viewpoints, not the ability to recover poses.
5. **One real sequence, one scene type.** freiburg1_xyz is a desk at close
   range with good texture and lighting — not a corridor, a lobby, a glass door
   or a building.
6. **The ledgered path still runs voxel by voxel**, so incremental capture is
   slow even though a full rebuild is fast, and ledgers remain in memory:
   fusion resumes within a process, not across runs.
7. **The one-shot traversal does not scale** and is now explicitly a
   tiny-fixture reference path, pinned as such by a test.

## Two different finish lines

**Showable** — a real scan in, a rendered reconstruction out. The numbers are
there: real TUM data reconstructs to 9.3 mm median against held-out frames.
What is missing is the picture. The mesher refuses the real volume, and there
is no committed renderer, so the result currently exists only as a table.

**MVP of the idea** — someone scans a space, someone else localizes in it.
Far: needs pose estimation, SLAM, structure, semantics, localization and
`SpatialMapPackage`. Most of the original architecture is untouched.

## Immediate next steps

1. **Vectorise the block planner** against its existing per-pixel path, the
   same way the evaluator was done. It is now the dominant pipeline cost.
2. Stream or window the replay/depth context so a full sequence fits.
3. Fix meshing on real data: connected-component and quality validation, so
   the TUM volume produces a surface rather than a refusal.

## Reproducing the current results

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -W error -m unittest discover -s tests -p "test_*.py"
```

The real-sensor result, its method and everything it does not prove are in
[`tum-validation.md`](tum-validation.md). The room-scan numbers, the scale
limit and the performance measurements are in
[`real-scale-validation.md`](real-scale-validation.md). Every diagnostic
command and its expected output is in the repository README.
