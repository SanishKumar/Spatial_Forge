# Real-sensor validation: TUM freiburg1_xyz

Everything else in the test suite runs on data SpatialForge generated itself:
clean noise, exact poses, nothing missing. This page records what happens on
a real depth camera, and it is the only result here that should be quoted.

The sequence is `rgbd_dataset_freiburg1_xyz` from the
[TUM RGB-D benchmark](https://cvg.cit.tum.de/data/datasets/rgbd-dataset): a
Kinect-class sensor moved around a desk, with motion-capture poses.

## One reconstruction, one chain

There is one volume. It is planned, fused, written to disk, scored, meshed and
rendered, and each step records the digest of the thing it was given:

| Artifact | SHA-256 |
|---|---|
| Scan (replay digest) | `aed9fc15…2f49e7` |
| Block plan `.sftplan` | `f4af0566…daaace3f` |
| Fused volume `.sftvol` | `027ed602…07c16855` |
| Mesh `.ply` | `57b0184d…8b012f4d` |

The volume names its plan and its scan. The mesh names its volume. The report
tool refuses a mesh whose header names any other volume, and it refuses a
scan whose replay digest is not the one the volume recorded. Full values, the
commit, and the working-tree state are in
[`../results/tum-freiburg1-xyz-15mm.json`](../results/tum-freiburg1-xyz-15mm.json).

This matters because an earlier version of this page got it wrong. Its
number came from one volume, its mesher failure from a second, and its
picture from a third, at three voxel sizes, and it described them as one. The
chain above is what makes that mistake fail loudly instead of reading well.

## What was run

```text
source                 798 RGB, 798 depth, 3000 pose records
imported               792 associated RGB-D pairs, 790 with poses
camera                 640 x 480, fx = fy = 525.0
selected / fused       396 / 395   (frame_stride 2; one has no pose)
voxel / truncation     15 mm / 45 mm
planned blocks         5,401   (3,288 surface, 2,113 halo)
planned voxel slots    2,765,312
observed after fusion  1,492,894   (54%)
contributions          66,200,108 applied of 1,095,063,552 evaluated
```

The commands are in the [README](../README.md#run-it-on-real-data).

## How it is measured

TUM ships a ground-truth *trajectory*, not a ground-truth surface, so there
is nothing to compare a reconstruction against directly. What is available is
cross-validation.

The volume is fused from frames `0, 2, 4, …`. Frames `1, 3, 5, …` are held
out: none of them contributes a voxel. Each held-out depth pixel is
back-projected with that frame's own pose, giving a point where a camera
measured a surface. A correct TSDF reads zero at such a point, so the
interpolated value there, scaled by the truncation, is a signed residual in
metres.

It is a **held-out TSDF residual**. It is not geometric accuracy and should
not be quoted as one. It uses ground-truth poses throughout, so it says
nothing about pose estimation; and it compares the volume with depth, not
with a surveyed surface, so a reconstruction that was wrong in the same way
from every viewpoint would still score well.

The tool refuses a held-out offset that is a multiple of the stride, so the
evaluation set cannot silently become the fused set.

## Results

```text
held-out frames                  395   (never fused)
held-out depth samples     5,757,844   (every 4th pixel)
inside observed voxels     5,755,186   (99.95%)
```

| | mean signed | median abs | rms | p95 abs | within 1 voxel |
|---|---|---|---|---|---|
| nearest voxel | +3.2 mm | 8.5 mm | 14.4 mm | 30.6 mm | 73.7% |
| trilinear | +3.3 mm | 7.5 mm | 13.4 mm | 28.8 mm | 77.0% |

A **7.5 mm median residual** against frames the volume never saw, at 15 mm
voxels, is about what this sensor's noise supports at desk range.

The **+3.3 mm signed bias** is small and systematic: held-out surfaces read
slightly positive, so the fused surface sits marginally further from the
camera than a held-out frame measures it. That is consistent with projective
distance being averaged along the viewing ray, and it has barely moved with
voxel size, which suggests it is a property of the rule rather than of the
grid. It is recorded, not corrected.

**99.95% coverage** says block planning is not missing surfaces: almost every
held-out surface point falls inside a voxel that fusion actually observed.

### The earlier 30 mm run, reproduced

The first version of this result fused 99 frames at 30 mm and reported a
9.3 mm median. It is kept as
[`../results/tum-freiburg1-xyz-30mm.json`](../results/tum-freiburg1-xyz-30mm.json),
regenerated through today's pipeline — a vectorised planner, streaming
fusion, a volume written to disk and read back. The plan file has the same
digest as the one published then, fusion applies the same 3,052,349
contributions, and every statistic matches to the last digit of the float:

| | then | now |
|---|---|---|
| plan digest | `46a21d10…fe1ae2383` | `46a21d10…fe1ae2383` |
| contributions applied | 3,052,349 | 3,052,349 |
| median residual | 9.288872313054917 mm | 9.288872313054917 mm |
| planning | 201 s | 13 s |
| fusion | 33 s | 10 s |

Three stages were rewritten in between. Nothing moved, which is the whole
point of pinning each rewrite to the path it replaced.

## Cost

One laptop CPU core, NumPy, no GPU:

```text
import                          4 s
plan     396 frames            44 s
fuse     1.1e9 voxel-obs      106 s     (about 10 million per second)
mesh     535,486 triangles    2.5 s
score    395 held-out frames   12 s
render   36 frames, coloured   37 s
```

Peak retained depth during fusion is 2.5 MB — one frame — however long the
sequence is. Fusing all 790 posed frames at 30 mm takes 65 s.

## What the first run broke, and what became of each

**The planner was the bottleneck**: a per-pixel Python loop, 2 s per frame,
201 s of a 242 s run. It is now vectorised and writes byte-identical plans.

**The sequence did not fit.** Depth for every selected frame was held at
once against a 512 MB ceiling, which is about 208 frames at this resolution,
so 99 of 792 were used. Fusion now streams one frame at a time.

**The mesher refused real data**: a 50 mm dense volume of this scene gave 10
non-manifold vertices and the reference mesher rejects any. The sparse mesher
uses the same split and meets the same kind of vertex — 31 in this volume —
and gives each triangle fan its own copy. See
[`sparse-meshing.md`](sparse-meshing.md).

**The measured volume could not be saved**, so it could not be shown. It is
now a file, and the mesh and the render come from it.

## What still does not hold

- **Nothing about pose estimation.** The engine has none.
- **No absolute accuracy.** That needs a dataset with a ground-truth surface.
- **One sequence, one kind of scene.** A well-lit, textured desk at close
  range. Not a corridor, a glass door, a dark room or a building.
- **10 mm is refused.** At 10 mm this scene needs 12,503 blocks; accumulator
  storage is capped at 64 MB, which is 10,922. The limit is a constant, not a
  design boundary, but it is the limit today.
- **Half the frames, by design.** Stride 2 is what leaves frames to hold out.
  A volume fused from every frame cannot be scored this way at all.
- **The free-space path was not run.** Coverage, cross-view resolution and
  plan expansion carry a 262,144-outcome cap that one 640x480 frame exceeds.
- **Mesh filters are choices.** The published mesh treats voxels seen fewer
  than three times as unknown and drops fragments under 200 triangles: 751
  fragments, 12,004 triangles. The volume and its score are unaffected; the
  unfiltered mesh is one flag away.
