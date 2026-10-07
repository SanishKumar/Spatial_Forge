# SpatialForge

[![tests](https://github.com/SanishKumar/Spatial_Forge/actions/workflows/tests.yml/badge.svg)](https://github.com/SanishKumar/Spatial_Forge/actions/workflows/tests.yml)

Deterministic 3D reconstruction from calibrated RGB-D scans, built for indoor
mapping. Turns a folder of depth frames and known camera poses into a sparse
metric TSDF volume and a triangle mesh — reproducibly, bit for bit, with every
fast path proven identical to a slower one that is easier to trust.

<p align="center">
  <img src="docs/assets/icl-room-and-error.gif" width="100%" alt="Two views of a reconstructed living room turning slowly together. On the left it is coloured from the scan: a sofa, a coffee table, pictures on the walls, a lamp. On the right the same mesh is coloured by its distance to the ground-truth surface: blue almost everywhere, with thin red lines along the edges of the furniture.">
</p>

<p align="center">
  <em>A living room reconstructed at 10 mm from 440 depth frames: 7.2
  million triangles. Left, coloured from the scan.<br>Right, the same mesh
  coloured by its measured distance from the ground-truth surface, on a
  scale that stops at 5 mm.<br>Half of it is within 0.45 mm. The scene is
  ICL-NUIM's, synthetic and with exact poses; a real sensor is further
  down.</em>
</p>

| | |
|---|---|
| Against a known surface | **0.45 mm** median, 0.77 mm mean distance to ground truth at 10 mm voxels ([ICL-NUIM](#against-a-known-surface), synthetic depth, exact poses) |
| On a real depth camera | **7.5 mm** median residual against frames never fused, at 15 mm voxels ([TUM RGB-D](#results-on-real-data)) |
| Reproducible | the committed scan reconstructs to the same bytes on Linux, macOS and Windows, under Python 3.11 and 3.14, checked on every push |
| Cost | one CPU core and NumPy: 6.3 billion voxel-observations fused in 211 s |

---

## Why this exists

Most reconstruction libraries optimise for speed and quality. This one
optimises for **being able to prove the output is right**.

That constraint came from the use case: it feeds an indoor navigation system,
where a wrong wall means a person walks into it. So every stage carries its
provenance, every operation is reproducible byte for byte, and every fast path
is checked against a slower, simpler one.

- **Deterministic.** The same scan produces the same bytes. Every artifact
  records the digest of what it was made from and refuses to load against
  anything else.
- **Every optimisation has a reference.** The vectorised planner writes the
  same plan file as the per-pixel one. Vectorised fusion leaves the same
  accumulator bytes as the per-voxel one. The sparse mesher emits the same
  vertices and triangles as the dense one. Not "within tolerance" — identical.
- **Failures are loud.** Operations preflight before writing, roll back on
  error, and refuse rather than guess.

None of the underlying algorithms are new; see
[where this sits](#where-this-sits). What is unusual is the standard of
evidence.

## Results on real data

[TUM RGB-D](https://cvg.cit.tum.de/data/datasets/rgbd-dataset)
`freiburg1_xyz`: a real Kinect-class sensor with motion-capture poses. Every
second frame is fused; the frames in between are held out and never
contribute a voxel.

| | |
|---|---|
| Voxel size / truncation | 15 mm / 45 mm |
| Frames fused | 395 |
| Voxel-observations evaluated | 1,095,063,552 |
| Observed voxels | 1,492,894 in 5,401 sparse blocks |
| Mesh | 279,979 vertices, 535,486 triangles |
| Held-out frames | 395, none of which contributed |
| Held-out depth samples | 5,757,844 |
| Landing inside observed voxels | **99.95%** |
| Median held-out TSDF residual | **7.5 mm** |
| RMS / p95 | 13.4 mm / 28.8 mm |

**Read that number carefully.** A correct TSDF reads zero exactly where a
camera measures a surface, so this is how far the volume disagrees with depth
from viewpoints it never saw. It is not distance to a surveyed surface, and it
uses ground-truth poses throughout, so it says nothing about pose estimation.
It bounds disagreement between views, not absolute correctness.

<p align="center">
  <img src="docs/assets/tum-mesh.gif" width="49%" alt="A desk reconstructed from the TUM RGB-D freiburg1_xyz sequence, coloured from the scan and rotating slightly">
  <img src="docs/assets/tum-mesh-geometry.png" width="49%" alt="The same mesh shaded without colour">
</p>

<p align="center">
  <em>The desk at 15 mm, 535,486 triangles, from 395 frames of a real
  depth camera. Left: each vertex coloured<br>only by RGB frames that
  actually see it. Right: the geometry alone. This is the measured
  volume: the numbers<br>above and this mesh come from the same file,
  tied together by digest.</em>
</p>

On one laptop CPU core, with NumPy and no GPU:

| Stage | Time |
|---|---|
| Plan sparse blocks (396 frames) | 44 s |
| Fuse 1.1 billion voxel-observations | 79 s |
| Extract the mesh | 2.5 s |
| Score against 395 held-out frames | 12 s |

The whole chain is recorded in
[`results/tum-freiburg1-xyz-15mm.json`](results/tum-freiburg1-xyz-15mm.json):
the scan, plan, volume and mesh digests, the commit that produced them, and
whether the working tree was clean. The method, the earlier 30 mm run it
supersedes, and everything this does not prove are in
[`docs/tum-validation.md`](docs/tum-validation.md).

## Against a known surface

The TUM number is agreement between views. It cannot say how far the
surface is from the truth, because TUM does not publish the truth.
[ICL-NUIM](https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html) does: a
synthetic living room, rendered from a model that ships with the dataset.

<p align="center">
  <img src="docs/assets/icl-room-geometry.png" width="49%" alt="The ICL-NUIM living room reconstructed at 10 mm, shaded without colour and seen from above with the near walls removed: sofa cushions, picture frames and a door panel are all resolved">
  <img src="docs/assets/icl-room-error.png" width="49%" alt="The same mesh coloured by distance to the ground-truth model: blue almost everywhere, finely speckled with lighter points, and thin red lines along the edges of furniture, picture frames and the door">
</p>

<p align="center">
  <em>Left: the geometry alone at 10 mm voxels, 7.2 million triangles.
  Right: the same mesh coloured by its<br>distance to the ground-truth
  surface. The scale saturates at 5 mm; 95% of the surface is under
  2.4 mm.</em>
</p>

Distance from every mesh vertex to the true surface, sequence `lr kt2`, 440
frames fused:

| Voxel | Median | Mean | RMS | p95 | Within 10 mm |
|---|---|---|---|---|---|
| 20 mm | 0.48 mm | 1.37 mm | 4.15 mm | 4.45 mm | 97.2% |
| 15 mm | 0.45 mm | 1.02 mm | 2.87 mm | 3.20 mm | 98.3% |
| 10 mm | **0.45 mm** | **0.77 mm** | **1.69 mm** | 2.34 mm | 99.2% |
| raw depth, the floor | 0.43 mm | 0.57 mm | 0.83 mm | 1.56 mm | 99.9% |

The last row is the dataset's own depth images measured the same way: what
a perfect reconstruction would score. At the median the mesh is already
there. What separates them is the tail, and the picture shows where the
tail lives: on silhouette edges narrower than the truncation band, not on
walls.

**Read this one carefully too.** Poses are the dataset's and the depth has
no sensor noise, so this is the error the engine adds to perfect input: a
floor under any real result, not a forecast of one. It is not comparable
with numbers published for SLAM systems on this dataset, which include
tracking drift.

Three things make the figure worth having:

- **The alignment is not fitted to the answer.** The dataset publishes its
  trajectory and its model in different frames. Its own tool aligns the
  reconstruction to the model with ICP, which lets the alignment absorb
  error. Here one rigid motion is fitted to the raw depth images, and the
  mesh is judged in a frame it had no part in choosing. A test moves a mesh
  25 mm and requires the score to get worse.
- **The handedness is converted, not ignored.** ICL-NUIM publishes
  `fy = -480`. Dropping the sign reconstructs a mirror-image room without a
  single error. The importer refuses it; a converter rewrites the poses.
- **It checks the other number.** On this dataset the held-out residual and
  the true error exist for the same volume: 1.50 mm against 1.69 mm RMS at
  10 mm, 3.63 against 4.15 at 20 mm. The cross-view figure TUM is limited to
  runs at 87 to 89% of the real one here.

Method, the three undocumented conventions of the dataset, and everything
this does not show: [`docs/icl-nuim-validation.md`](docs/icl-nuim-validation.md).

## The same bytes everywhere

Reproducibility here is a tested property, not an intention. The committed
fixture reconstructs to a volume and a mesh whose SHA-256 digests are written
literally into the test suite, and CI runs that suite on Linux, macOS and
Windows under Python 3.11 and 3.14. They only pass if all six agree.

You can check it on your own machine in three commands — see below.

## Install

Python 3.11+ with NumPy and Pillow. No GPU, no compiler, no build step.

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e .
```

## Quickstart

Reconstruct the committed test fixture end to end:

```bash
python -m spatialforge reconstruct tsdf-block-plan \
  tests/fixtures/minimal.vgsession outputs/demo.sftplan \
  --voxel-size-m 0.125 --truncation-m 0.5
```

```bash
python -m spatialforge reconstruct tsdf-block-volume \
  outputs/demo.sftplan tests/fixtures/minimal.vgsession outputs/demo.sftvol
```

```bash
python -m spatialforge reconstruct tsdf-block-volume-mesh \
  outputs/demo.sftvol outputs/demo.ply
```

Each prints the digest of what it wrote. Whatever machine you are on, the
volume and the mesh should be:

```text
output_sha256: 29f9f427b43c2a9e8ec416dad9a3ba1572044c716407ef2b30989daed9c2adc3
output_sha256: 8c019fe39e181ef1855c8c229422528cf4218ca5b4998af533ae5e00d4e5f190
```

Every command prints `key: value` diagnostics, one fact per line, including
explicit negatives so you can see what it did *not* do.

## Run it on real data

Download and extract a
[TUM RGB-D sequence](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download),
then:

```bash
# 1. Import it into the session format
python -m spatialforge scan import-tum \
  datasets/rgbd_dataset_freiburg1_xyz datasets/fr1xyz.vgsession
```

```bash
# 2. Plan which voxel blocks the depth actually reaches
python -m spatialforge reconstruct tsdf-block-plan \
  datasets/fr1xyz.vgsession datasets/fr1xyz.sftplan \
  --voxel-size-m 0.015 --truncation-m 0.045 --frame-stride 2
```

```bash
# 3. Fuse, one frame at a time, into a persisted sparse volume
python -m spatialforge reconstruct tsdf-block-volume \
  datasets/fr1xyz.sftplan datasets/fr1xyz.vgsession datasets/fr1xyz.sftvol
```

```bash
# 4. Mesh the volume
python -m spatialforge reconstruct tsdf-block-volume-mesh \
  datasets/fr1xyz.sftvol datasets/fr1xyz.ply \
  --min-weight 3 --min-component-triangles 200
```

```bash
# 5. Score the volume against the frames it never saw
python tools/tum_reconstruction_report.py \
  datasets/fr1xyz.vgsession datasets/fr1xyz.sftvol --mesh datasets/fr1xyz.ply
```

```bash
# 6. Render it
python tools/render_mesh.py datasets/fr1xyz.ply datasets/fr1xyz-render \
  --session datasets/fr1xyz.vgsession --azimuth 190 --elevation 12 --sweep 28
```

Step 5 refuses a mesh that was not extracted from the volume it is scoring,
so the picture cannot quietly be of something else.

## How it works

```mermaid
flowchart LR
    A[".vgsession<br/>RGB-D + poses"] --> B["replay<br/>SHA-256 digest"]
    B --> C["block plan<br/>.sftplan"]
    C --> D["streaming fusion<br/>one frame in memory"]
    D --> E["sparse volume<br/>.sftvol"]
    E --> F["triangle mesh<br/>.ply"]
    E --> G["held-out<br/>residual"]
    F --> H["render"]
    F --> I["distance to<br/>ground truth"]
```

**Session and replay.** A `.vgsession` is a folder of RGB, depth, IMU and pose
streams with a manifest. Replay associates them by exact timestamp and hashes
every input file into one digest that identifies the scan.

**Block planning.** Rather than allocating a dense grid over the whole scene,
the planner selects only the 8×8×8 voxel blocks that surfaces reach, plus a
truncation halo. On the TUM scene that is 2.8 million voxel slots where a
dense grid over the same bounds would need about 38 million.

**Streaming fusion.** Each frame is decoded once, every planned voxel is
projected, sampled and classified against it in NumPy, the result is added,
and the frame is dropped. Memory is one depth image however long the scan is.

**Persisted volume.** The fused sums and weights are written as a `.sftvol`:
a canonical JSON header and fixed little-endian arrays, with a digest per
array. The loader re-derives every count the header claims from the payload
and refuses a file where they disagree.

**Meshing.** Every fully observed cell is split into six tetrahedra and
triangulated, vectorised over all cells at once.

## Three problems worth describing

### Making it fast without changing the answer

The sparse path started about 100× slower than the dense one it was meant to
replace, because it looped in Python. Vectorising it took a small room scan
from **210 s to under a second**.

The constraint was that the fast path had to be **byte-identical** to the slow
one, and that is harder than it sounds, because floating-point addition is not
associative:

```text
(a + b) + c  ≠  a + (b + c)      in the last bits
```

Any reordering silently changes the reconstruction. So the world-to-camera
product is written out term by term instead of as a matrix multiply, which is
free to reassociate or fuse its multiply-add; `fx * x / z + cx` stays
left-associated; and voxels that fail an early gate still ride through the
later arithmetic, their garbage silenced and their pixel indices pinned in
bounds, because that is what makes it one pass.

### Fusing a long scan without holding it

The first real-data run could use 99 of 792 frames: decoded depth for every
frame was held at once and ran out of room.

Fusion never needed the frames together. A voxel's value is a sum over
observations in order, so frames can be consumed one after another. Walking
frames in the outer loop instead of blocks changes *which voxel is visited
when*, never *the order in which one voxel receives its contributions* — and
that order is the only thing that fixes the last bits. So the streaming path
is byte-identical to the block-by-block one by construction, and all 790
posed frames now fuse in 42 s at 30 mm.

### Meshing data that is not tidy

The reference mesher refused the first real volume outright: non-manifold
vertices.

Real scans have ragged observed regions. Around a grid edge, the cells on two
opposite sides can be observed while the cells between them are not, so two
separate fans of triangles meet at a single vertex — a pinch. That is not an
error in the data. Each fan now gets its own copy of the vertex, which is the
standard repair, and the *reference mesher's own* topology check is then run
on the result to confirm it is a surface the reference would have accepted.
The TUM volume has 31 of them in 535,486 triangles.

Details: [`docs/vectorised-fusion.md`](docs/vectorised-fusion.md),
[`docs/sparse-volume-format.md`](docs/sparse-volume-format.md) and
[`docs/sparse-meshing.md`](docs/sparse-meshing.md).

## What it does not do

Stated plainly, because the gaps matter more than the features:

- **No pose estimation.** Poses must be supplied. There is no SLAM, no visual
  odometry, no bundle adjustment. It cannot take a raw phone video.
- **No relocalization.** It builds maps; it does not position anyone inside
  one.
- **No semantics.** It produces geometry, not rooms, doors or accessibility.
- **No absolute accuracy on a real sensor.** The ground-truth comparison is
  on synthetic, noise-free depth. On real data there is only the held-out
  residual, which is agreement between views: a reconstruction wrong the
  same way from every viewpoint would still score well. Closing that needs a
  surveyed real scene.
- **Not real time.** Fourteen to thirty million voxel-observations per
  second on one CPU core, depending on how much of the scene each frame
  can see. A GPU system does this live; this one takes minutes.
- **A size ceiling.** A plan is limited to 100,000 blocks: 51 million
  voxels, 614 MB of accumulators held in memory. A scene that needs more
  is refused with a message saying so.
- **Fusion does not resume across runs.** A volume is written once, complete.
  The resumable fusion ledgers exist but live in memory.
- **The free-space path is fixture-scale.** Coverage, cross-view resolution
  and plan expansion carry a 262,144-outcome cap that a single 640×480 frame
  exceeds. They are bounded diagnostics, not part of the real-data pipeline.

## Where this sits

Truncated signed distance fusion is
[Curless and Levoy, 1996](https://graphics.stanford.edu/papers/volrange/);
doing it live on a depth camera is
[KinectFusion, 2011](https://doi.org/10.1109/ISMAR.2011.6092378);
storing it in sparse hashed blocks is
[Nießner et al., 2013](https://niessnerlab.org/projects/niessner2013hashing.html).
[InfiniTAM](https://arxiv.org/abs/1410.0925),
[voxblox](https://arxiv.org/abs/1611.03631),
[VDBFusion](https://pmc.ncbi.nlm.nih.gov/articles/PMC8838740/),
[nvblox](https://arxiv.org/abs/2311.00626) and Open3D are mature, much faster
implementations. Nothing here is a new capability, and if you need a
reconstruction, use one of those.

What those systems do not try to be is bit-reproducible. A real-time GPU
pipeline is built to tolerate small inconsistencies in exchange for speed —
InfiniTAM, for instance, allocates blocks with non-atomic writes and accepts
that a hash collision within a frame is simply corrected in the next. That is
the right trade for a robot. This project makes the opposite one: slower, and
able to say exactly which bytes a given scan must produce.

More in [`docs/related-work.md`](docs/related-work.md).

## Layout

```text
spatialforge/     library and CLI
  replay.py               deterministic session replay and digests
  tsdf_block_plan.py      sparse block planning
  tsdf_stream_fusion.py   frame-at-a-time fusion
  tsdf_block_volume.py    the .sftvol format
  tsdf_block_mesh.py      sparse meshing
  tsdf.py, mesh.py        dense reference integrator and mesher
  tum_importer.py         TUM RGB-D -> session format
tools/            reproducible scoring and rendering scripts
tests/            49 test modules
results/          generated result manifests, one per published run
docs/             format specs, algorithm notes, validation reports
```

## Development

```bash
.venv/Scripts/python.exe -W error -m unittest discover -s tests -p "test_*.py"
```

Warnings are errors. Tests assert that operations leave the filesystem alone
and patch functions an operation must not call, then assert they were never
reached.

## Status

A research-grade known-pose reconstruction pipeline, complete from scan to
mesh, validated on real sensor data and measured against a ground-truth
surface. Pose estimation, localization and semantic mapping are outside it.
Built as the reconstruction backend for an indoor navigation project.
