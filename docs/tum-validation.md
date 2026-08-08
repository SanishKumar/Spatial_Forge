# First real-sensor validation: TUM freiburg1_xyz

Every result before this one came from data SpatialForge generated itself.
The larger synthetic-room test removed the committed fixture's degeneracies
but kept clean gaussian noise, exact poses, no motion blur, no reflective
surfaces, and no missing returns. This page records the first run against a
real depth camera.

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

## Three different reconstructions live on this page

This is the part that is easiest to get wrong when quoting these numbers.
The same TUM sequence was reconstructed three times at three voxel sizes,
by two different code paths, for three different purposes:

| Artifact | Path | Voxel / truncation | Persisted? |
|---|---|---|---|
| The 9.3 mm residual and 99.8% coverage | sparse block fusion | **30 mm / 90 mm** | no — in memory only |
| The mesher refusal, 10,235 surface points | dense reference TSDF | **50 mm / 150 mm** | `.sftsdf` |
| The rendered GIF and still, 14,625 points | dense reference TSDF | **42 mm / 126 mm** | `.sftsdf` |

They are not the same volume and their numbers are not interchangeable.

The headline result is the sparse 30 mm one, and it **cannot be rendered or
meshed at all**: block-backed volumes have no persistence format and no
surface or mesh consumer yet. Everything visual on this page therefore comes
from the dense reference path at a coarser voxel size, because that is the
only path that currently produces a file the extractors can read.

So the correct sentence is:

> The sparse 30 mm known-pose reconstruction produced a 9.3 mm median
> held-out TSDF residual. The animation is a separate 42 mm dense
> point-cloud reconstruction of the same sequence.

Not "fused at 30 mm, and that is what the animation shows."

The machine-readable provenance for the 30 mm run — commit, parameters,
environment, timings, metrics and input digests — is in
[`../results/tum-freiburg1-xyz.json`](../results/tum-freiburg1-xyz.json),
written by the report tool itself so it cannot drift from what was run.

## What the 9.3 mm actually measures

It is a **held-out TSDF residual** — also fair to call it a cross-view
consistency residual. It is *not* "geometric accuracy" and not distance to a
surveyed surface, and it should never be quoted as either.

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

At 30 mm voxels, a **9.3 mm median residual against frames the reconstruction
never saw** is the pipeline working. It is close to what this sensor's own
noise can support at desk range, and the nearest-voxel column shows how much
of the remainder is grid quantisation rather than reconstruction error.

Two things this number is not. It uses ground-truth poses throughout, so it
says nothing about pose estimation. And it compares the volume against depth
rather than against a surveyed surface, so it bounds disagreement between
viewpoints, not absolute correctness.

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

**The mesher refuses real data.** This was the 50 mm dense volume, not the
30 mm sparse one — the sparse volume cannot be meshed at all yet, so it has
never been offered to the mesher.

```text
TRIANGLE MESH FAILED
- mesh construction produced 10 non-manifold triangle vertices
```

The fixed six-tetrahedron split produces a clean 60,072-triangle mesh on the
synthetic room and a non-manifold one here. The validation is correct to
refuse, and this is not a regression — it is the first time the mesher has
seen noisy real geometry. Surface-point extraction succeeds on that same
50 mm volume (10,235 crossing points), so the volume itself is sound; it is
the triangulation that needs connected-component and quality work.

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
- **Nothing about the free-space path.** The coverage, cross-view and
  plan-expansion ladder was not run on this sequence and cannot be: its
  262,144 retained-outcome cap is exceeded by a single 640x480 frame's
  307,200 pixels, and the whole-domain workload here is orders of magnitude
  beyond it. That path remains a bounded fixture-scale diagnostic; the
  fusion numbers above do not exercise it.
- **The 30 mm volume does not survive the process.** It is fused in memory
  by the report tool and discarded. There is no block-backed artifact
  format yet, which is also why it cannot be rendered or meshed.

## Status of this checkpoint

No engine behaviour changed, so nothing new is pinned about fusion; the
existing tests are unchanged. What is committed is the reproducible report
tool in `tools/tum_reconstruction_report.py`, its fixture-scale tests, the
result manifest, and this page. The dataset is not committed — it is 448 MB,
and `datasets/` is gitignored.
