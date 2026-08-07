# First real-sensor validation: TUM freiburg1_xyz

Every result before this one came from data SpatialForge generated itself.
[`real-scale-validation.md`](real-scale-validation.md) removed the committed
fixture's degeneracies but kept its synthetic nature: clean gaussian noise,
exact poses, no motion blur, no reflective surfaces, no missing returns. This
page records the first run against a real depth camera.

## What was run

`rgbd_dataset_freiburg1_xyz` from the TUM RGB-D benchmark — a Kinect-class
sensor waved around a desk, with motion-capture ground-truth poses.

```text
source                 798 RGB, 798 depth, 3000 pose records
imported               792 associated RGB-D pairs, 790 with poses
camera                 640 x 480, fx = fy = 525.0
fused frames           99  (frame_stride 8)
voxel / truncation     30 mm / 90 mm
planned blocks         1241 active, 698 surface
planned voxel slots    635,392
observed after fusion  255,995 slots (40.3%)
contributions applied  3,052,349 of 62,903,808 evaluated
```

Reproduce with:

```powershell
.\.venv\Scripts\python.exe -m spatialforge scan import-tum `
  datasets\rgbd_dataset_freiburg1_xyz datasets\freiburg1-xyz.vgsession
.\.venv\Scripts\python.exe -m spatialforge reconstruct tsdf-block-plan `
  datasets\freiburg1-xyz.vgsession datasets\fr1xyz.sftplan `
  --voxel-size-m 0.03 --truncation-m 0.09 --frame-stride 8
.\.venv\Scripts\python.exe tools\tum_reconstruction_report.py `
  datasets\freiburg1-xyz.vgsession datasets\fr1xyz.sftplan
```

## How accuracy is measured without a ground-truth mesh

TUM ships a ground-truth *trajectory*, not a ground-truth surface, so there is
nothing to compare a reconstruction against directly. The available check is
cross-validation:

> Fuse from one set of frames. Take depth measured by frames the fusion never
> saw, back-project it with its own ground-truth pose, and ask what the
> reconstruction says at those points. A correct TSDF reads zero on a real
> surface, so the interpolated value scaled by the truncation is a signed
> surface error in metres.

The plan fuses observations `0, 8, 16, …`; the report evaluates against
observations `4, 12, 20, …` — 98 frames, none of which contributed a single
voxel. The tool refuses an offset that is a multiple of the stride, so the
evaluation set cannot silently become the training set.

## Results

```text
held-out frames                 98  (never fused)
held-out depth samples   1,428,048  (every 4th pixel)
landing in observed voxels 1,425,233  (99.8%)
```

| | mean signed | median abs | rms | p95 abs | within 1 voxel |
|---|---|---|---|---|---|
| nearest voxel | +3.5 mm | 12.4 mm | 23.8 mm | 52.8 mm | 83.6% |
| trilinear | +3.8 mm | 9.3 mm | 20.4 mm | 45.6 mm | 87.3% |

At 30 mm voxels, a **9.3 mm median error against frames the reconstruction
never saw** is the pipeline working. It is close to what this sensor's own
noise can support at desk range, and the nearest-voxel column shows how much
of the remainder is grid quantisation rather than reconstruction error.

The **+3.8 mm signed bias** is small but systematic: held-out surfaces read
slightly positive, meaning the fused surface sits marginally further from the
camera than the held-out frame measures it. Consistent with projective TSDF
averaging pulling the level set along the viewing ray. Worth revisiting when
confidence weighting arrives; not worth acting on now.

**99.8% coverage** says the plan's block selection is not missing surfaces —
almost every held-out surface point falls inside a voxel the fusion actually
observed.

## What broke

This is why real data is worth running.

**The block planner is now the bottleneck.** It projects every depth pixel in
a pure-Python loop and then walks that pixel's candidate block span. At
640x480 it runs about 2 seconds per frame:

```text
import                     4 s
block plan (99 frames)   201 s      <-- dominates
allocate                   1 s
context build              7 s
vector fusion             33 s
```

Fusion, which was the wall two checkpoints ago, is now a fifth of the cost of
planning. Vectorising the planner is the obvious next performance checkpoint,
and the evaluator gives it a template.

**The retained-depth cap limits sequences to about 208 frames at this
resolution.** `TsdfReplayDepthContext` holds float64 metric depth for every
selected frame against a 512 MB ceiling; 640x480 costs 2.36 MB per frame, so
the full 792-frame sequence would need 1.9 GB. The 99-frame run retains
243 MB. Streaming or windowing the context is a real requirement for full
sequences, not a tuning knob.

**The dense reference path cannot reach 30 mm on this scene.** Its
1,000,000-voxel reference cap allows 50 mm here — the inferred bounds at 30 mm
would need about 2.75 million voxels against the sparse path's 635,392 planned
slots. The sparse architecture earning its keep is exactly the point, but it
means dense/sparse parity can no longer be checked at the working voxel size
on real data.

**The mesher refuses real data.**

```text
TRIANGLE MESH FAILED
- mesh construction produced 10 non-manifold triangle vertices
```

The fixed six-tetrahedron split produces a clean 60,072-triangle mesh on the
synthetic room and a non-manifold one here. The validation is correct to
refuse, and this is not a regression — it is the first time the mesher has
seen noisy real geometry. Surface-point extraction succeeds on the same
volume (10,235 crossing points), so the volume itself is sound; it is the
triangulation that needs the connected-component and quality work already
listed on the roadmap.

## What this does not prove

- **Nothing about pose estimation.** Held-out frames carry the same
  motion-capture ground truth, taken as exact. This measures geometric
  self-consistency across viewpoints, not the ability to recover poses — the
  engine still has none.
- **Not absolute accuracy.** Without a ground-truth surface, a reconstruction
  that is consistently wrong in the same way across all viewpoints would
  score well here. Cross-validation bounds random error, not shared bias.
- **One sequence, one scene type.** freiburg1_xyz is a desk at close range
  with good texture and lighting. It is not a corridor, a lobby, a glass
  door, or a dark room, and it is not a building.
- **A fifth of the frames.** Stride 8 was chosen to fit the retained-depth
  cap, not because 99 frames is enough.

## Status of this checkpoint

No engine behaviour changed, so nothing new is pinned in the test suite; the
434 tests are unchanged. What is committed is the reproducible report tool in
`tools/tum_reconstruction_report.py` and this page. The dataset is not
committed — it is 448 MB, and `datasets/` is gitignored.
