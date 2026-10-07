# Absolute accuracy: ICL-NUIM living room

The [TUM result](tum-validation.md) is agreement between views: a volume
fused from half the frames, checked against depth from the other half. It
cannot say how far the surface is from where the surface really is, because
TUM does not publish where the surface really is.

[ICL-NUIM](https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html) does. Its
living room is a synthetic scene, rendered from a model that ships with the
dataset. This page measures the reconstructed mesh against that model.

<p align="center">
  <img src="assets/icl-room.png" width="49%" alt="The ICL-NUIM living room reconstructed at 10 mm and coloured from the scan, seen from above with the near walls removed">
  <img src="assets/icl-room-error.png" width="49%" alt="The same mesh coloured by distance to the ground-truth model: blue almost everywhere, with thin red lines along the edges of furniture, picture frames and the door">
</p>

<p align="center">
  <em>Left: the room at 10 mm voxels, 7.2 million triangles, coloured from
  the scan. Right: the same mesh<br>coloured by its distance to the
  ground-truth surface, on a scale that saturates at 5 mm; 95% of it is
  under 2.4 mm.</em>
</p>

## What this isolates

The sequence is `living_room_traj2` ("lr kt2"), the noise-free variant: 880
posed frames at 640x480. The camera poses are the dataset's own and the
depth is exactly what the renderer computed.

So nothing here is a sensor and nothing is a tracker. What is left is the
engine: the voxel grid, the fusion rule, and the mesher. This is the error
SpatialForge adds to perfect input, and it is a floor under any real
result, not a forecast of one.

## Three things the dataset does not tell you

**It is left-handed.** The published intrinsics have `fy = -480`: the scene
was rendered by POV-Ray, whose camera y axis points up. Dropping the sign
and keeping the poses raises no error; it reconstructs the mirror image of
every frame, placed by poses that belong to the unmirrored one.
[`tools/prepare_icl_nuim.py`](../tools/prepare_icl_nuim.py) converts the
poses instead, `R' = F R S` and `t' = F t`, which for a unit quaternion is a
permutation of its components and one negated coordinate. The importer
refuses a negative focal length outright, so the shortcut is not available
by accident.

**What the files mean.** Depth is planar z, not distance along the ray, and
pose `k` belongs to image `k`, which leaves frame 0 without a pose. The
first is not written down with the data, and the second has to be taken
on trust from a trajectory with one row fewer than there are images. So
both were decided against the model:

| Reading of the files | Raw depth within 2 cm of the model | Median distance to its surface |
|---|---|---|
| planar depth, pose `k` with image `k` | 100% | 0.43 mm |
| planar depth, pose `k` with image `k-1` | 94% | 3.7 mm |
| ray distance, pose `k` with image `k` | 16% | 8.7 mm |

Each reading was given its own best-fit alignment first, so the table is
not an artefact of aligning with the first one. The second and third rows
were measured once, with a scratch script, while working this out; the
first is what the report tool reproduces.

**Where the model is.** The trajectory and the model are in different
frames and the transform between them is not published. The dataset's own
evaluation tool, [SurfReg](https://github.com/mp3guy/SurfReg), starts from
a fixed guess per trajectory and runs ICP on the reconstruction being
evaluated.

Aligning the thing you are measuring lets the alignment absorb part of its
error. Here the transform is one rigid motion fitted by point-to-plane
registration of the **raw depth images** to the model, never of the mesh:

```text
fitted to            93,574 depth samples from 44 frames
rotation             1.1758 degrees
translation          (-0.74539, +1.29978, -0.78788) m
last update          1.8e-9 m
```

The mesh is then judged in a frame it had no part in choosing. The test
suite checks the property directly: a mesh with every vertex moved 25 mm
must score worse, which is exactly what a fit to the mesh would undo.

## How it is measured

For every vertex of the mesh, the nearest of the model's 9,982,296 oriented
points is found, exactly. Two distances are reported.

**To that point.** This is the statistic SurfReg reports. It is an upper
bound on the distance to the true surface: the model's points lie on the
surface, but only every few millimetres, so a vertex sitting exactly on it
is still a few millimetres from the nearest one.

**To the tangent plane through that point**, along the model's normal. This
takes the sampling out, and is the figure to read.

Both have a floor, and it is measured rather than assumed. Depth from 44
frames the fit did not use is scored the same way. That is what a perfect
reconstruction would get under this alignment and this model.

The nearest-neighbour search is a uniform grid that is exact by
construction: a candidate is accepted only if it is no farther than one
cell, in which case nothing nearer can lie outside the 27 cells searched;
otherwise the query is retried on a grid twice as coarse. It is tested
against comparing every pair, to the index and to the last bit of the
distance.

## Results

440 frames fused (every second one), mesh filtered as on the TUM page:
voxels seen fewer than three times are unknown, fragments under 200
triangles are dropped.

Distance from each mesh vertex to the model's tangent plane:

| Voxel | Vertices | Median | Mean | RMS | p95 | p99 |
|---|---|---|---|---|---|---|
| 20 mm | 911,173 | 0.48 mm | 1.37 mm | 4.15 mm | 4.45 mm | 22.5 mm |
| 15 mm | 1,597,055 | 0.45 mm | 1.02 mm | 2.87 mm | 3.20 mm | 13.4 mm |
| 10 mm | 3,656,989 | **0.45 mm** | **0.77 mm** | **1.69 mm** | 2.34 mm | 6.5 mm |
| raw depth (floor) | 93,549 samples | 0.43 mm | 0.57 mm | 0.83 mm | 1.56 mm | 3.1 mm |

Distance to the nearest model point, the SurfReg statistic:

| Voxel | Mean | Median | Within 5 mm | Within 10 mm |
|---|---|---|---|---|
| 20 mm | 4.10 mm | 3.42 mm | 77.1% | 97.2% |
| 15 mm | 3.79 mm | 3.34 mm | 78.9% | 98.3% |
| 10 mm | 3.58 mm | 3.28 mm | 80.4% | 99.2% |
| raw depth (floor) | 3.42 mm | 3.21 mm | 82.3% | 99.9% |

No vertex of any of the three meshes is farther than 160 mm from the model;
the worst is 75 mm at 20 mm voxels and 35 mm at 10 mm.

Three readings.

**The median is the floor.** Half the surface is within 0.45 mm of the
model's planes, and raw depth, rendered from the same scene, scores
0.43 mm. At the median there is nothing left to measure: the reconstruction
is as close as this model and this alignment can tell.

**The mean is the tail.** As the voxel halves, RMS falls from 4.2 mm to
1.7 mm and p99 from 22.5 mm to 6.5 mm, while the median barely moves. The
error is not spread over the surface. It is concentrated, and the picture
at the top of this page shows where: along silhouette edges, where a thin
structure or an occlusion boundary is narrower than the truncation band,
and the surface either rounds off or carries a fringe past the edge.
Walls, floor and table tops stay at the low end of the scale.

**There is a small bias towards the camera.** The mean signed distance is
+0.33 mm at 10 mm, with the model's normals pointing into free space. Raw
depth shows +0.29 mm by itself, so most of it is already there before
anything is fused.

## The held-out residual, checked

The TUM page can only report a held-out TSDF residual and has to say that
it is not accuracy. Here both exist for the same volume, so one can be held
against the other:

| Voxel | Held-out residual, rms | Distance to truth, rms | Held-out median | Truth median |
|---|---|---|---|---|
| 20 mm | 3.63 mm | 4.15 mm | 0.40 mm | 0.48 mm |
| 15 mm | 2.52 mm | 2.87 mm | 0.38 mm | 0.45 mm |
| 10 mm | 1.50 mm | 1.69 mm | 0.36 mm | 0.45 mm |

The residual runs at 87 to 89% of the true error at every voxel size. They
are different measurements, of a volume in one case and a mesh in the
other, and one scene does not make a law. But on the one dataset where it
can be checked, the number TUM is limited to moves with the real one and
slightly understates it.

It also puts the TUM figure in context. The same engine that is 0.4 mm from
held-out depth here is 6.6 mm from it on the Kinect sequence (medians,
10 mm voxels both). The difference comes with the data: a real sensor,
its calibration, and measured poses.

## One chain per volume

Each volume is planned, fused, written, meshed and scored, and each step
records the digest of what it was given. For the 10 mm reconstruction:

| Artifact | SHA-256 |
|---|---|
| Scan (replay digest) | `c52207e9…86574593` |
| Block plan `.sftplan` | `876c6bcd…e762c6db` |
| Fused volume `.sftvol` | `e54492db…c7554788` |
| Mesh `.ply` | `017449f0…0cb358cf` |
| Ground-truth model | `414c0579…2be4279c` |

The full values, the fitted transform, the commit and the state of the
working tree are in
[`../results/icl-nuim-lr-kt2-10mm-surface.json`](../results/icl-nuim-lr-kt2-10mm-surface.json),
with the held-out report for the same volume beside it and the same pair
for 15 mm and 20 mm.

## What was run

```text
source                 881 RGB-D frames, 880 poses
imported               880 frames, all with poses (frame 0, which has none, is left out)
camera                 640 x 480, fx = 481.2, fy = 480.0
selected / fused       440 / 440   (frame_stride 2)

                         20 mm         15 mm          10 mm
truncation               60 mm         45 mm          30 mm
planned blocks           7,655        14,059         27,965
observed voxels      2,520,109     4,238,805      8,890,488
voxel-observations   1.72e9        3.17e9         6.30e9
  of which applied   188,387,632   309,488,634    657,800,997
mesh triangles       1,796,756     3,153,732      7,237,866
```

## Cost

One laptop CPU core, NumPy, no GPU, at 10 mm:

```text
plan      440 frames                      83 s
fuse      6.3e9 voxel-observations       211 s
mesh      7,237,866 triangles             42 s
align     93,574 depth samples            46 s
measure   3,656,989 vertices              67 s
```

Of those 6.3 billion voxel-observations, 81% are of a voxel behind the
camera or outside the image: a camera inside a room has most of the
room behind it or beside it. Most of them are settled a whole block at
a time, without being evaluated and without changing a byte of the
result; see
[`vectorised-fusion.md`](vectorised-fusion.md#blocks-a-frame-cannot-see).
At 20 mm fusion takes 54 s with that test and 257 s without it.

## Reproduce it

Download `living_room_traj2_frei_png.tar.gz` and `living-room.ply.tar.gz`
from the dataset page and extract them.

```bash
# 1. Convert the left-handed sequence to a right-handed TUM folder
python tools/prepare_icl_nuim.py \
  datasets/icl/living_room_traj2 datasets/icl/icl-nuim-living-room-kt2
```

```bash
# 2. Import it with the dataset's camera
python -m spatialforge scan import-tum \
  datasets/icl/icl-nuim-living-room-kt2 datasets/icl-kt2.vgsession \
  --fx 481.2 --fy 480 --cx 319.5 --cy 239.5
```

```bash
# 3. Plan, fuse and mesh
python -m spatialforge reconstruct tsdf-block-plan \
  datasets/icl-kt2.vgsession datasets/icl-kt2-10mm.sftplan \
  --voxel-size-m 0.01 --truncation-m 0.03 --frame-stride 2
```

```bash
python -m spatialforge reconstruct tsdf-block-volume \
  datasets/icl-kt2-10mm.sftplan datasets/icl-kt2.vgsession \
  datasets/icl-kt2-10mm.sftvol
```

```bash
python -m spatialforge reconstruct tsdf-block-volume-mesh \
  datasets/icl-kt2-10mm.sftvol datasets/icl-kt2-10mm.ply \
  --min-weight 3 --min-component-triangles 200
```

```bash
# 4. Measure it against the model
python tools/surface_accuracy_report.py \
  datasets/icl-kt2.vgsession datasets/icl-kt2-10mm.sftvol \
  datasets/icl-kt2-10mm.ply datasets/icl/model/living-room.ply \
  --initial-translation -0.75 1.30 -0.79 \
  --errors-out datasets/icl-kt2-10mm-errors.npy
```

```bash
# 5. Draw where the error is
python tools/render_mesh.py datasets/icl-kt2-10mm.ply datasets/icl-error \
  --vertex-errors datasets/icl-kt2-10mm-errors.npy --error-scale-mm 5 \
  --cull-back-faces --azimuth 200 --elevation 35
```

The initial translation is where the trajectory's origin sits in the
model's frame, to the nearest few centimetres. It came from a coarse search
(the floor height from a histogram, the rest from correlating top-down
occupancy) and the fit does not depend on it. Started 8 cm away in two
opposite directions, it returns the same transform to fifteen significant
digits and the same statistics.

## What this does not show

- **Sensor noise.** The depth is exact. ICL-NUIM also publishes this
  sequence with simulated Kinect noise; that variant was not run.
- **Pose error.** Poses are ground truth. The engine has no tracker.
- **Completeness.** This is accuracy: how close the reconstructed surface
  is to the true one. It says nothing about surface that was never
  reconstructed, and this trajectory leaves holes: the black patches in
  the pictures above are floor and furniture no frame looked at.
- **Comparison with published numbers.** Figures quoted for SLAM systems on
  ICL-NUIM include tracking drift and are usually on the noisy sequences
  after aligning the reconstruction to the model. These are not the same
  experiment and should not be set beside them.
- **Another scene.** One room, one trajectory.
- **Mesh filters are choices.** Dropping fragments under 200 triangles
  removes 12,030 of 7.2 million triangles at 10 mm. The volume is
  unaffected.
