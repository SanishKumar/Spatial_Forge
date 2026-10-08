# Changelog

What changed between tagged versions. Numbers quoted here are the ones in
`results/`; each manifest there records the commit it was produced from.

## 0.4.1 (2026-10-08)

0.4.0, with a test that passes on every platform.

- A new test put a point exactly on the edge between two viewing-angle
  bands. Which side it falls depends on how the platform rounds an
  arccosine, and three of the six CI jobs disagreed with the other three.
  The test no longer asserts it, and the report says so about its bands.
- The last viewing-angle band is open above, so rounding cannot leave a
  point seen exactly edge-on outside every band.
- No published number changes.

## 0.4.0 (2026-10-08)

Sensor noise and completeness: the two things the 0.3.0 accuracy figure
left out.

Measured

- **Simulated Kinect noise.** The ICL-NUIM paper's depth-noise model, from
  a seed, applied to the clean `lr kt2` sequence. Depth frames a median
  3.16 mm from the truth fuse to a mesh a median 0.86 mm from it at 10 mm
  voxels (0.77 mm at 15 mm, 0.81 mm at 20 mm).
- **The dataset's own noisy sequence.** Its files sit half a disparity
  level towards the camera: 3 mm at 1.2 m, 17 mm at 3.5 m. Fusion averages
  noise and cannot average an offset, so the mesh is a median 9.27 mm from
  the truth at 15 mm voxels, about where the frames are.
- **The held-out residual is not an error bar.** At 15 mm voxels it reads
  0.8 times the true error on exact depth, 6.2 times on unbiased noise and
  0.8 times on the offset sequence. No factor converts one to the other.
- **Completeness.** Of the ground-truth surface that at least three fused
  frames saw, 93.3% is within 5 mm of the 10 mm mesh and 98.4% within
  20 mm; 1.1% has no mesh within 30 mm. Simulated noise leaves that almost
  where it was: 91.4%, 98.6% and 0.9%.
- **What is missing was only glanced at.** A tenth of the seen surface was
  never viewed within 70.5° of head-on, and it holds 96% of what the 10 mm
  mesh lacks. That is the angle at which a truncation band of three voxels
  reaches less than one voxel behind a surface. Of the rest, one point in
  two thousand is missing.

Added

- `tools/surface_completeness_report.py`: of the true surface the fused
  frames saw, how much is in the mesh. Sight is ruled per frame against
  the measured depth, so a point behind the sofa does not count as seen.
  The result is also split by the most head-on view of each point.
- `tools/_triangles.py`: exact distance from a point to a triangle mesh.
- `tools/simulate_kinect_noise.py` and `tools/depth_noise_report.py`.
- `tools/surface_accuracy_report.py --alignment`: judge a scan in the frame
  an earlier report fitted, accepted only for the same model and the same
  trajectory.
- `tools/render_mesh.py`: the scan's colour and its error side by side
  from one camera.

Changed

- Block storage accepts every plan the planner will write: 100,000 blocks,
  614 MB. It used to stop at 43,690.
- Expanded block plans are schema version `0.2.0`, and the loader refuses a
  plan whose version, free-space rule and provenance disagree. An expanded
  plan written by 0.3.0 has to be expanded again. Surface plans, and every
  published digest, are unchanged.
- A fusion receipt whose counts, pairs or digests could not have come from
  a real pass is refused when it is constructed.

## 0.3.0 (2026-10-07)

Absolute accuracy, against a surface that is known.

- ICL-NUIM `lr kt2`, exact depth and poses: mesh vertices a median 0.45 mm
  and a mean 0.77 mm from the ground-truth model at 10 mm voxels. Raw depth
  measured the same way scores 0.43 mm.
- `tools/surface_accuracy_report.py`. The model and the trajectory are
  published in different frames; the rigid motion between them is fitted to
  raw depth, never to the mesh being scored.
- `tools/prepare_icl_nuim.py` converts the dataset's left-handed poses. The
  TUM importer takes a camera other than the Kinect's and refuses a
  negative focal length instead of building a mirrored room.
- Streaming fusion skips blocks a frame cannot see. Volumes and receipts are
  byte-identical to the unculled path.
- The accumulator ceiling is raised and fusion no longer holds every voxel
  centre, which makes TUM `freiburg1_xyz` at 10 mm possible: 6.6 mm median
  held-out residual.
- Error maps: `tools/render_mesh.py --vertex-errors`.

## 0.2.0 (2026-10-06)

Known-pose reconstruction, complete from scan to mesh.

- A vectorised block planner, streaming fusion that holds one frame in
  memory, and a sparse mesher, each byte-identical to the slower reference
  it replaces.
- `.sftvol`, a persisted volume whose loader re-derives the header's counts
  from the payload.
- A mesh renderer with no dependencies beyond NumPy and Pillow.
- TUM `freiburg1_xyz` at 15 mm: 7.5 mm median held-out residual.
- CI on Linux, macOS and Windows under Python 3.11 and 3.14, with the
  fixture's volume and mesh digests pinned.

## 0.1.1 and 0.1 (2026-08-08)

The known-pose research baseline: the scan-session format and its replay
digest, the dense reference TSDF, block plans, block-by-block fusion with
ledgers, and the first TUM validation.
