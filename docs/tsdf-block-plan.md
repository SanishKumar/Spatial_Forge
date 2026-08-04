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

The result is a separate `.sftplan` diagnostic. The dense/sparse reference TSDF
artifacts, surface extraction, and triangle meshing do not consume it.
Temporary block allocation and context-backed traversal diagnostics consume its
canonical active-block coordinates in memory.

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
ray blocks are not represented here. Traversing every existing active row does
not change this coverage boundary. A complete block-backed fusion phase must
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

## Strict loading and replay-bound verification

The `.sftplan` loader creates a frozen in-memory snapshot after strictly
checking:

- the exact schema version, required fields, and absence of unknown or duplicate
  JSON keys;
- ASCII JSON types, finite numeric values, fixed grid and activation
  conventions, and the 100,000-block limit;
- positive and internally consistent frame, depth-sample, and block counters;
- signed 32-bit block coordinates in strict X-fastest order, with no
  duplicates;
- the surface-block subset, halo count, planned voxel-slot count, and
  componentwise block bounds.

Loading does not modify or make the file itself immutable. The returned object
is the immutable snapshot.

Check that snapshot against a current ScanSession replay with:

```powershell
.\.venv\Scripts\python.exe -m spatialforge reconstruct tsdf-block-plan-verify `
  outputs/progress-blocks.sftplan `
  tests/fixtures/minimal.vgsession
```

A successful check reports:

```text
artifact: valid
session_replay: matched
geometry_recomputed: no
```

The verifier compares the session ID, replay SHA-256, and replay-derived
observation, selection, pairing, and missing-stream counts. The artifact
SHA-256 identifies the exact plan bytes; the replay SHA-256 binds the plan to
the current calibrated session inputs.

This boundary is deliberately narrow. The verifier does not decode depth,
recompute candidate coordinates, prove that the artifact was produced by
trusted planner code, authenticate or sign it, allocate blocks, or fuse TSDF
values. It is read-only and creates no new artifact.

## Empty-storage consumer

The first consumer remains narrower than fusion:

```powershell
.\.venv\Scripts\python.exe -m spatialforge reconstruct tsdf-block-allocate `
  outputs/progress-blocks.sftplan `
  tests/fixtures/minimal.vgsession
```

It strict-loads the plan, replay-verifies the current session, and allocates one
zeroed `8 x 8 x 8` numeric block row for every canonical active coordinate.
It does not decode depth, regenerate the plan, update a voxel, decide
camera-to-surface free space, or write an artifact. The temporary storage
contract is documented in
[`tsdf-block-storage.md`](tsdf-block-storage.md).

## Existing-plan traversal consumer

The complete active-block tuple now has a bounded in-memory traversal consumer:

```powershell
.\.venv\Scripts\python.exe -m spatialforge reconstruct tsdf-block-context-plan-traverse `
  outputs\progress-blocks.sftplan `
  tests\fixtures\minimal.vgsession
```

It requires the allocated row tuple to equal `active_blocks`, then visits every
row in the artifact's canonical X-fastest order with one prepared replay/depth
context. It neither selects a subset nor creates a coordinate. The retained
`planned_voxel_slots * selected_observations` diagnostic workload is capped at
262,144 outcomes.

For the fixture this means all eight active rows: four direct surface blocks
and four truncation-halo blocks. This is complete traversal of the artifact,
not proof that the artifact is a complete fusion domain. Free-space activation
and visibility/culling remain undefined by this artifact. The exact execution
and rollback contract is documented in
[`tsdf-context-plan-traversal.md`](tsdf-context-plan-traversal.md).

## One-observation block-ray diagnostic

One prepared observation can now report camera-to-measured-surface centerline
block paths separately from the artifact's active tuple:

```powershell
.\.venv\Scripts\python.exe -m spatialforge reconstruct tsdf-block-context-observation-rays `
  outputs\progress-blocks.sftplan `
  tests\fixtures\minimal.vgsession `
  --observation-sequence 0
```

The source plan supplies the grid, selection, and provenance. Positive finite
depth pixels produce closed thin-DDA paths; invalid depth produces no ray. The
canonical covered-block union is partitioned into coordinates already present
in `active_blocks` and coordinates absent from it.

This is a read-only comparison. It does not change the artifact's
`free_space_rule: not-planned`, append an active coordinate, publish a revised
plan, allocate storage, or fuse voxels. The thin pixel-center paths are not a
geometric supercover or proof of conservative nearest-pixel free-space
coverage. See
[`tsdf-observation-block-rays.md`](tsdf-observation-block-rays.md).

## Explicitly deferred

- expanding the surface-band plan into a complete fusion domain;
- aggregating one-observation ray receipts across the complete selected tuple;
- conservative nearest-pixel or voxel-center coverage, frustum activation,
  visibility, occlusion, and culling;
- configurable block resolution and per-block observation provenance;
- an immutable snapshot spanning every input-file read;
- recomputing candidate geometry during verification;
- artifact signatures, authentication, schema migration, or canonical
  rewriting;
- persistent block-backed fusion artifacts and their replay/provenance binding;
- larger `.sftsdf` volumes and sparse-aware surface or mesh traversal;
- performance or full-sequence scalability claims;
- Open3D, GPU, parallel, adaptive-resolution, or submap backends;
- outlier filtering, confidence, color, and sensor-dependent weighting; and
- production meshing, structure, pose estimation, SLAM, and localization.

This use of blocks belongs to dense reconstruction. It is unrelated to the
sparse visual landmarks and localization map planned later.
