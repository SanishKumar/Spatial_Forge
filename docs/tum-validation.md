# Real-sensor validation: TUM freiburg1_xyz

Everything else in the test suite runs on data SpatialForge generated itself:
clean noise, exact poses, nothing missing. This page records what happens on
a real depth camera. Its companion,
[`icl-nuim-validation.md`](icl-nuim-validation.md), measures the same engine
against a surface that is known.

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

What this residual reads on a noisy sensor was measured afterwards, on
ICL-NUIM, where the truth is known. With a Kinect's noise simulated on
that sequence the held-out residual is 4.5 mm at the median while the mesh is
0.86 mm from the true surface
([details](icl-nuim-validation.md#what-it-does-to-the-held-out-residual)).
Held-out frames carry the sensor's noise in full, so the residual mostly
reads that. And where every frame is wrong the same way it reads too
little: on the dataset's own noisy sequence, whose depth is offset towards
the camera, the residual is 7.0 mm and the true error 9.3 mm
([details](icl-nuim-validation.md#the-datasets-own-noisy-sequence)).
The figures below say the volume agrees with this Kinect to within the
Kinect's noise. They are not an estimate of how far the surface is from
the truth, in either direction.

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
camera than a held-out frame measures it. It does not move with voxel size.
An earlier version of this page took that to mean it was a property of the
fusion rule. ICL-NUIM says otherwise: the same rule on noise-free depth
leaves a bias of 0.2 to 0.5 mm, a tenth of this. So most of it arrives with
the data, in the sensor's noise or its calibration, and it is recorded, not
corrected.

**99.95% coverage** says block planning is not missing surfaces: almost every
held-out surface point falls inside a voxel that fusion actually observed.

### At 10 mm

The first version of this page had to say that 10 mm was refused: the scene
needs 12,503 blocks and accumulator storage stopped at 10,922. Storage now
holds whatever the planner will plan, now 250,000 blocks, and the result is
[`../results/tum-freiburg1-xyz-10mm.json`](../results/tum-freiburg1-xyz-10mm.json):

| | 15 mm | 10 mm |
|---|---|---|
| planned blocks | 5,401 | 12,503 |
| observed voxels | 1,492,894 | 3,869,699 |
| mesh triangles | 535,486 | 1,269,564 |
| median residual | 7.5 mm | 6.6 mm |
| rms | 13.4 mm | 10.7 mm |
| p95 | 28.8 mm | 22.1 mm |
| mean signed | +3.3 mm | +3.3 mm |
| held-out depth inside observed voxels | 99.95% | 99.93% |

Two and a half times the voxels buy an eighth off the median. A finer grid
stops helping once what is left is not the grid's, and that is where this
is: on noise-free depth with exact poses the same engine's held-out median
is 0.4 mm ([ICL-NUIM](icl-nuim-validation.md)). Nearly all of these 6.6 mm
come with the data: a real sensor, its calibration, and measured poses.

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
fuse     1.1e9 voxel-obs       79 s     (about 14 million per second)
mesh     535,486 triangles    2.5 s
score    395 held-out frames   12 s
render   36 frames, coloured   37 s
```

Peak retained depth during fusion is 2.5 MB — one frame — however long the
sequence is. Fusing all 790 posed frames at 30 mm takes 42 s.

Fusion was 106 s when this page was first written. It now skips
blocks a frame cannot see, a whole block at a time, and produces the
same volume, digest for digest. At 10 mm the same steps take 147 s to
fuse 2.5 billion voxel-observations and 5 s to mesh 1.3 million
triangles.

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
- **No absolute accuracy on this sensor.** That needs a surveyed real scene.
  The engine's own error against a known surface is measured on synthetic
  depth in [`icl-nuim-validation.md`](icl-nuim-validation.md).
- **One sequence, one kind of scene.** A well-lit, textured desk at close
  range. Not a corridor, a glass door, a dark room or a building.
- **Half the frames, by design.** Stride 2 is what leaves frames to hold out.
  A volume fused from every frame cannot be scored this way at all.
- **The free-space path was not run.** Coverage, cross-view resolution and
  plan expansion carry a 262,144-outcome cap that one 640x480 frame exceeds.
- **Mesh filters are choices.** The published mesh treats voxels seen fewer
  than three times as unknown and drops fragments under 200 triangles: 751
  fragments, 12,004 triangles. The volume and its score are unaffected; the
  unfiltered mesh is one flag away.
