# Real-sensor validation: TUM RGB-D

Everything else in the test suite runs on data SpatialForge generated itself:
clean noise, exact poses, nothing missing. This page records what happens on
a real depth camera. Its companion,
[`icl-nuim-validation.md`](icl-nuim-validation.md), measures the same engine
against a surface that is known.

The sequence is `rgbd_dataset_freiburg1_xyz` from the
[TUM RGB-D benchmark](https://cvg.cit.tum.de/data/datasets/rgbd-dataset): a
Kinect-class sensor moved around a desk, with motion-capture poses. The
same sensor carried round [a whole room](#a-whole-room) is further down.

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

## A whole room

`freiburg1_xyz` is a desk seen from one side. `freiburg1_room` is the same
sensor carried once round the whole office.

<p align="center">
  <img src="assets/tum-room.png" width="80%" alt="An office reconstructed from the TUM RGB-D freiburg1_room sequence, seen from above and coloured from the scan: desks with monitors, keyboards, a laptop and papers on them, two office chairs, one with a teddy bear sitting in it, shelves against the far wall, and a wooden floor with gaps where no frame looked">
</p>

<p align="center">
  <em>The office at 15 mm, 3.5 million triangles, from 676 frames. Seen
  from above with surfaces facing away left out, and cropped to the
  room.<br>The gaps in the floor are real: the camera went round once,
  looking at the desks and the walls.</em>
</p>

```text
source                 1,362 RGB, 1,360 depth, 4,887 pose records
imported               1,352 associated RGB-D pairs, all with poses
selected / fused       676 / 676   (frame_stride 2)
voxel / truncation     15 mm / 45 mm
planned blocks         41,318   (27,079 surface, 14,239 halo)
planned voxel slots    21,154,816
observed after fusion  12,205,959   (58%)
contributions          222,263,250 applied of 14,300,655,616 evaluated
mesh                   1,836,470 vertices, 3,502,836 triangles
peak memory            0.34 GiB to fuse, 2.03 GiB to mesh
```

It was imported level, with `--up z`, which the map further down needs.
Against the 676 frames that were not fused:

```text
held-out depth samples     9,888,272   (every 4th pixel)
inside observed voxels     9,872,334   (99.84%)
```

| | mean signed | median abs | rms | p95 abs | within 1 voxel |
|---|---|---|---|---|---|
| nearest voxel | +6.4 mm | 11.3 mm | 17.5 mm | 35.8 mm | 61.5% |
| trilinear | +6.5 mm | 10.5 mm | 16.7 mm | 34.6 mm | 64.7% |

A **10.5 mm median residual** where the desk had 7.5 mm, and it is still
agreement between views and not accuracy.

Most of the difference is distance. Filed by the depth each held-out
sample was measured at:

| Range | Desk: share of samples | Median | Room: share of samples | Median |
|---|---|---|---|---|
| under 1 m | 52.5% | 6.1 mm | 19.5% | 6.4 mm |
| 1 to 1.5 m | 33.0% | 8.7 mm | 36.8% | 10.5 mm |
| 1.5 to 2 m | 6.5% | 10.6 mm | 23.1% | 12.0 mm |
| 2 to 2.5 m | 3.5% | 12.7 mm | 11.8% | 14.3 mm |
| 2.5 to 3 m | 2.6% | 14.8 mm | 5.6% | 14.9 mm |
| 3 to 4 m | 1.7% | 13.8 mm | 2.4% | 14.8 mm |
| beyond 4 m | none | | 0.7% | 27.2 mm |

Within a metre the two scans read alike, and half the desk's samples are
that close where a fifth of the room's are. The residual climbs with range
in both, as a Kinect's depth noise does. Between one and two and a half
metres the room reads about 1.5 mm more than the desk at the same range,
and that part is not explained here: the room was scanned in one turn
round it lasting 45 seconds, and nothing in this measurement separates
the sensor from the poses. (Measured once from the two volumes with a
scratch script. The report does not file its samples by range.)

The manifest is
[`../results/tum-freiburg1-room-15mm.json`](../results/tum-freiburg1-room-15mm.json).

### The grid's orientation is not in the answer

The importer normally anchors a session to its first camera. Here the
first frame looks 41 degrees below the horizon, so that session's grid is
tipped 41 degrees against the room. The room was reconstructed on both:

| | As the camera was held | Level |
|---|---|---|
| planned blocks | 43,089 | 41,318 |
| observed voxels | 12,600,224 | 12,205,959 |
| mesh triangles | 3,495,826 | 3,502,836 |
| held-out samples in observed voxels | 9,872,288 | 9,872,334 |
| median residual | 10.49 mm | 10.49 mm |
| rms | 16.73 mm | 16.73 mm |
| p95 | 34.61 mm | 34.62 mm |
| mean signed | +6.54 mm | +6.53 mm |

Two grids with no voxel in common, and the residual agrees to a hundredth
of a millimetre. The level one needs 4% fewer blocks, because walls and
floor run along its axes instead of across them. The tipped run is kept as
[`../results/tum-freiburg1-room-15mm-as-held.json`](../results/tum-freiburg1-room-15mm-as-held.json).

### Where there is room in it

Expanding the plan into observed free space adds 28,097 blocks and changes
no triangle of the mesh. Between the desk tops and the camera the expanded
volume holds 16.3 m² as free where the surface plan's holds 3.8 m². Over
the height of someone standing it holds 3.5 m², because this scan never
looked at the floor in the middle of the room. The map, the check that no
camera was behind a surface, and what the map cannot say are in
[`free-space-expansion.md`](free-space-expansion.md#a-whole-room).

The picture at the head of this section is drawn by

```bash
python tools/render_mesh.py datasets/fr1room-level-15mm.ply \
  datasets/fr1room-render --session datasets/freiburg1-room-level.vgsession \
  --cull-back-faces --azimuth 125 --elevation 62 --frames 1 --size 1400 \
  --colour-stride 4
```

and cropped. The mesh also holds fragments well outside the room, from
readings through its openings, and the renderer frames all of them.

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
- **Two sequences, one sensor, one office.** A well-lit, textured desk at
  close range and the room it stands in. Not a corridor, a glass door, a
  dark room or a building.
- **Half the frames, by design.** Stride 2 is what leaves frames to hold out.
  A volume fused from every frame cannot be scored this way at all.
- **Free space came later, by another path.** The staged path this page
  first named carries a 262,144-outcome cap that one 640x480 frame
  exceeds, and remains as a reference for a fixture.
  [`free-space-expansion.md`](free-space-expansion.md) expands both of
  these scans with one that takes real frames.
- **Mesh filters are choices.** The published mesh treats voxels seen fewer
  than three times as unknown and drops fragments under 200 triangles: 751
  fragments, 12,004 triangles. The volume and its score are unaffected; the
  unfiltered mesh is one flag away.
