# Changelog

What changed between tagged versions. Numbers quoted here are the ones in
`results/`; each manifest there records the commit it was produced from.

## 0.6.0 (2026-10-09)

Larger volumes, and free space on a real sensor.

Measured

- **Free space on the real Kinect desk.** TUM `freiburg1_xyz` at 15 mm,
  expanded from 5,401 blocks to 8,692. The mesh is the same mesh and the
  held-out residual the same to a thousandth of a millimetre. Of 790
  camera positions the voxel is observed free for 476, unseen for 314 and
  behind a surface for none.
- **The living room's free space at 10 mm.** 122,688 blocks, more than a
  plan could hold before. Its map agrees with the 20 mm one, 14.25 m² free
  against 14.33, and all 880 camera positions are in observed free space.
- **Free space can add triangles.** At 20 mm and on the desk the expanded
  volume's mesh is identical. At 10 mm it has 118 more triangles in 7.2
  million, 4.5 cm² along one vertical edge, and none fewer. 0.5.0 said free
  space adds no surface; that was two cases, not a rule.
- **The dataset's noisy sequence at 10 mm.** 126,967 blocks, refused until
  now. A median 8.46 mm from the truth and 8.86 mm on the camera's side of
  it: finer voxels do not remove the offset.

Changed

- A plan may hold 250,000 blocks, up from 100,000: 1.5 GB of accumulators.
- Fusing, writing and loading a volume no longer copy it. Writing took
  twice the payload on top of the storage and now takes under a tenth of
  it. Loading a two-megabyte volume allocated 616 MB, because a read of "up
  to the maximum" reserves the maximum; it now allocates the file. Fusing
  the largest volume here, 0.70 GiB of accumulators, peaks at 0.90 GiB
  where it peaked at 1.30, and writes the same bytes.
- Plan expansion settles blocks hidden behind a frame's surfaces from the
  depth image instead of evaluating their voxels. The real desk's plan
  expands in 148 s where it took 1,607 s, to the same plan.
- The mesher classifies cells a run of blocks at a time: about 10 kB of
  memory a block where it was 16. The meshes are the same files.

## 0.5.0 (2026-10-08)

Three things 0.4 listed as not done: the glancing-angle loss is tested
instead of only predicted, an interrupted fusion can be continued, and
free space is planned at the scale of a real scan.

Measured

- **The glancing-angle prediction, tested.** The room reconstructed again
  with truncation bands of four, six and eight voxels. The loss moves to
  the angle each width predicts: between 80° and 85° the 20 mm mesh loses
  36% of the seen surface with a band of three voxels and 1.3% with eight.
  It is paid for in the tail of the accuracy figures: RMS error goes from
  4.2 mm to 16.3 mm while the median stays at half a millimetre. The tail
  follows the truncation in millimetres; the loss follows the band in
  voxels.
- **Free space on the living room.** The 20 mm plan expanded from 7,655
  blocks to 17,394. The mesh from the expanded volume is the surface
  plan's mesh byte for byte. Between 0.28 m and 0.86 m above the floor the
  expanded volume holds 14.33 m² as observed free where the surface plan's
  holds 2.05 m², and the voxel at each of the 880 camera positions is one
  other frames looked through.
- **An interrupted fusion, continued.** The room's 20 mm fusion was killed
  80 frames in and the command run again. The volume it wrote has the
  digest of the published one.

Added

- `reconstruct tsdf-block-volume --checkpoint PATH`: progress is saved as
  fusion goes, and running the same command after an interruption
  continues from the last save to the same bytes. `--checkpoint-every` and
  `--stop-after` control it.
- `.sftckpt`, a fusion saved part way, and the staged fusion it rests on:
  `advance_tsdf_plan_streaming`, `finish_tsdf_plan_streaming`,
  `TsdfStreamFusionProgress`.
- `propose_tsdf_plan_expansion_streaming`: every block holding a voxel some
  frame observed, found by running the fusion evaluator over a box of
  candidate blocks whose size is derived.
- `tools/free_space_map.py`: free, occupied and unknown for every floor
  column of a volume over a height band, with the camera path.

Changed

- `reconstruct tsdf-block-plan-expand` uses the new path and writes plans
  whose `free_space_rule` is `every-block-with-an-observed-voxel`. The
  earlier rule is still read, and remains as a fixture-scale reference.
  The two are not the same set of blocks, and a plan records which made
  it.

Corrected

- The README and the ICL-NUIM page said a weight that falls with the
  viewing angle would bring back surface lost at glancing angles. It would
  not. Both now say what does: a wider band, at a measured cost.

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
