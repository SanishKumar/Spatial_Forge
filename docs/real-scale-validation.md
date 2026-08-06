# Real-scale validation

Every checkpoint before this one was validated against `minimal.vgsession`: a
2x2 image, two frames, a camera at the exact world origin, and surfaces on
exact voxel and block boundaries. That fixture is ideal for pinning exact
numbers and worthless for showing the geometry survives a real sensor. This
page records the first run that removes those degeneracies.

## The room fixture

`tests/room_fixture.py` generates a scan from a fixed seed rather than
committing 1.2 MB of images. It breaks each degeneracy on purpose:

```text
image             64 x 48, not 2 x 2
frames            20, with the camera translating and yawing along an arc
camera origin     never (0, 0, 0)
surfaces          2.537 / 1.313 / -1.229 / -1.117 / 1.409 m
                  none of them a multiple of the 0.04 m voxel or 0.32 m block
depth             millimetre-quantised with 4 mm gaussian noise
scene             room shell plus an interior box, also off-grid
```

Generation takes about 350 ms, so the fixture is built once per test run and
shared.

## What held

**Geometry is recovered correctly.** Inferred bounds bracket every true plane,
and surface points sit within a fraction of a voxel of the walls that drew
them:

```text
plane          points   mean bias    rms      max
far wall x      4144     -5.1 mm    17.6 mm   85.0 mm
left wall y     1650     -2.6 mm    11.4 mm   53.0 mm
right wall y    2098     +5.0 mm    19.1 mm   89.0 mm
floor z          357    +12.8 mm    27.8 mm   68.7 mm
ceiling z        163    -42.1 mm    50.1 mm   69.0 mm
```

With 40 mm voxels and 4 mm depth noise, sub-centimetre mean bias on the three
well-observed walls is the pipeline working. The floor and especially the
ceiling are worse and have far fewer points, which is expected rather than a
defect: the camera looks horizontally, so those surfaces are only ever seen at
a grazing angle.

**The full chain runs end to end on this data**: validate, replay, point
cloud (61,440 points), block plan (351 blocks), dense TSDF, surface points
(7,978), triangle mesh (30,573 vertices, 60,072 triangles). The mesh is a
recognisable room with the interior box cut into it.

## What broke, and what it means

**The one-shot plan traversal refuses a real scan.** A 20-frame 64x48 scan
plans 351 blocks, which is `351 x 512 x 20 = 3,594,240` retained contribution
outcomes against a 262,144 cap:

```text
TSDF context plan traversal requires 3594240 retained contribution outcomes;
reference maximum is 262144.
```

This is not a bug — the cap exists deliberately and the message names the fix
— but it does settle the traversal's status. It is a reference path for the
tiny fixture, not a scan-scale one. A regression test now pins that behaviour
so the limit stays visible instead of being rediscovered later.

**The resumable ledgered fusion is what makes the same plan tractable.** Each
bounded pass stays under the cap, so the identical 351-block plan fuses
completely in 15 passes:

```text
fusion_passes: count=15 block_limit=25
ledger: fused=351 pending=0 complete=yes
contributions_applied: 1127112
elapsed: 210s
```

That checkpoint was built for incremental capture; it turns out to also be the
only route to fusing a real scan at all.

## The performance wall

The two fusion paths are not remotely comparable:

```text
dense integrator    226,845 voxels, 2,008,493 updates      ~1 s
block fusion        3,594,240 voxel-observations         ~210 s
```

The dense path is numpy-vectorised. The block path evaluates voxel by voxel in
Python at roughly 17,000 evaluations per second. Extrapolating that rate:

```text
small room, 5 cm voxels, 60 frames      ~1 minute      workable
small room, 2 cm voxels, 300 frames     ~2 hours       not workable
building floor, 5 cm, 2000 frames       ~2 days        not workable
```

So the sparse block architecture — which is the whole point of the scalable
path — is currently about two orders of magnitude slower than the dense
reference it is meant to replace. Vectorising the evaluator is now the
critical path for anything beyond a single coarse room, and the existing
scalar path becomes its reference, exactly as the dense integrator is the
reference for the sparse one.

**Since this run**, the evaluator half of that gap is closed. A vectorised
block evaluator sweeps the same 3,594,240 voxel-observations in 1.45 seconds
and accepts exactly the same 1,127,112 contributions, bit for bit
([`tsdf-block-contributions.md`](tsdf-block-contributions.md)). The 210
seconds above were dominated by evaluation, so what remains of the wall is
application: fusion still writes one guarded slot at a time. Wiring the
vector field into fusion is the next checkpoint.

## What this run does not prove

- The data is synthetic. It has clean gaussian depth noise, exact poses, no
  motion blur, no rolling shutter, no reflective or transparent surfaces, and
  no missing returns. It removes the *degeneracy* of the old fixture, not the
  difference between simulation and a real sensor.
- Poses are still ground truth. Nothing here exercises pose error, which is
  what a real capture would bring and what the engine currently has no defence
  against.
- 64x48 is still small. Real depth cameras are 640x480 or larger, which is a
  hundred times the pixels per frame.

Running an actual TUM RGB-D sequence remains the next validation step, and the
importer for it already exists.
