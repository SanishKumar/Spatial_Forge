# Deterministic TSDF candidate-block planning

This checkpoint plans a bounded set of candidate voxel blocks from known-pose
depth without fusing TSDF values:

```powershell
.\.venv\Scripts\python.exe -m spatialforge reconstruct tsdf-block-plan `
  tests/fixtures/minimal.vgsession `
  outputs/progress-blocks.sftplan `
  --voxel-size-m 0.125 `
  --truncation-m 0.5
```

The result is a separate `.sftplan` diagnostic. Current TSDF integrators,
surface extraction, and triangle meshing do not consume it.

## Grid contract

- Blocks contain exactly `8 x 8 x 8` voxels in this version.
- Both voxel and block grids are anchored at world `(0, 0, 0)`.
- A block has signed index `(bx, by, bz)` and world extent
  `8 * voxel_size_m` along every axis.
- Block bounds are lower-inclusive and upper-exclusive.
- Coordinates are unique and sorted X-fastest, then Y, then Z.
- Index arithmetic uses binary64 values and mathematical floor/ceiling without
  an epsilon. Multiplying an index back by the block extent corrects any bound
  that rounded inward. Signed negative and rounded boundaries are covered by
  regression tests.

For each positive finite depth pixel, the planner uses the same calibrated
back-projection and `T_world_camera` transform as automatic bounds. Missing
depth or pose is counted and skipped; neither is invented or interpolated.

## Activation rule

Each surface point belongs to one surface block. Candidate active blocks form
an outward-conservative cover of the point's half-open, axis-aligned
neighborhood:

```text
[point - truncation, point + truncation)
```

The three axis ranges form an L-infinity cube. Candidate coordinates are
deduplicated across all selected frames. `halo_blocks` is the number of active
blocks that do not directly contain a surface sample.

The planner first applies binary64 floor/ceiling, multiplies the resulting
indices back by the block extent, and expands any endpoint that rounded inward.
It never shrinks the range. At an exactly representable physical boundary whose
division rounded across that boundary, this may retain one extra neighboring
block per endpoint and axis. This small overcoverage is intentional: missing a
candidate surface block is less safe than planning an extra one.

Here, `active` means selected by the planner only. No block is allocated for
TSDF values or fused during this command.

This is deliberately a surface-band plan. The dense reference TSDF also
updates positive free space from the camera toward the measured surface; those
ray blocks are not represented here. A future block-backed fusion phase must
define that semantic choice explicitly.

## Exact fixture proof

At `0.125 m` voxels, each block is `1 m` wide. With `0.5 m` truncation, the
fixture reports:

```text
frames: total=2 selected=2 paired=2
depth_samples: valid=8 invalid=0
candidate_blocks: surface=4 active=8 halo=4 voxel_slots=4096
block_bounds: min=(0, -1, -1) max=(1, 0, 0)
output_sha256: 372c7c5d49eff1a30317ceb8b67cb3c40683f1d9d2a6179a049772f763d6f79d
```

The four surface blocks lie at `X=1`; the truncation neighborhood adds the four
matching blocks at `X=0`.

## Artifact and safety

The artifact schema is `spatialforge.tsdf-block-plan` version `0.1.0`. It
records:

- the session and replay SHA-256 digest;
- grid and activation conventions;
- frame, missing-data, and depth-sample counts;
- surface, active, and halo block counts;
- candidate voxel-slot capacity and signed block bounds; and
- canonical surface and active block index lists.

It contains no timestamp or filesystem path. Planning rechecks the replay
digest before publication, permits at most 100,000 unique active blocks,
requires signed 32-bit block coordinates, and refuses to overwrite either an
existing or race-created target.

## Explicitly deferred

- consuming the plan during TSDF fusion;
- camera-to-surface free-space or frustum/ray block activation;
- configurable block resolution and per-block observation provenance;
- an immutable snapshot spanning every input-file read;
- larger `.sftsdf` volumes and sparse-aware surface or mesh traversal;
- performance or full-sequence scalability claims;
- Open3D, GPU, parallel, adaptive-resolution, or submap backends;
- outlier filtering, confidence, color, and sensor-dependent weighting; and
- production meshing, structure, pose estimation, SLAM, and localization.

This use of blocks belongs to dense reconstruction. It is unrelated to the
sparse visual landmarks and localization map planned later.
