# Fixed-bounds reference TSDF

This milestone adds one part of dense reconstruction: deterministic projective
TSDF integration for aligned RGB-D frames with known camera poses. It is a
small CPU numerical reference, not the optimized reconstruction backend.

The fixed command accepts explicit bounds. A separate deterministic
`tsdf-auto` step can derive bounds from known-pose depth before invoking this
same integrator. Surface-point and triangle-mesh steps consume either output.

## Run the exact fixture proof

```powershell
.\.venv\Scripts\python.exe -m spatialforge reconstruct tsdf `
  tests/fixtures/minimal.vgsession `
  outputs/progress.sftsdf `
  --origin 0 -0.25 -0.25 `
  --dimensions 4 1 1 `
  --voxel-size-m 0.5 `
  --truncation-m 0.5
```

The output is a human-readable JSON diagnostic. It refuses to overwrite an
existing file.

The committed fixture observes the same world plane from two camera poses. A
successful run reports:

```text
voxels: total=4 observed=3 fused=3
voxel_updates: 6 max_weight=2
```

The three observed voxel centers lie at world `X` positions `0.25`, `0.75`,
and `1.25 m`. Their TSDF values are `1.0`, `0.5`, and `-0.5`, all with weight
`2`. The fourth center, at `1.75 m`, is too far behind the observed surface and
remains unknown. Positive and negative values therefore bracket the known
plane at `X=1.0 m`.

## Volume definition

The caller supplies:

- `--origin X Y Z`: the minimum world-space corner in metres;
- `--dimensions NX NY NZ`: positive voxel counts;
- `--voxel-size-m`: the edge length of every voxel; and
- `--truncation-m`: the trusted signed-distance band, at least one voxel wide.

Voxel `(ix, iy, iz)` is evaluated at its center:

```text
p_world = origin + (index + 0.5) * voxel_size
```

The flattened order is X-fastest, then Y, then Z. This reference implementation
allows at most 1,000,000 voxels so an accidental command cannot allocate an
unbounded dense volume.

## Integration rule

For every selected observation, each voxel center is transformed from world
space into the camera:

```text
p_camera = inverse(T_world_camera) * p_world
```

The center is projected with the RGB pinhole intrinsics because depth is
required to be aligned to RGB. Depth uses nearest-pixel sampling. For measured
optical-axis depth `d` and voxel camera depth `z`:

```text
sdf = d - z
tsdf_observation = clamp(sdf / truncation, -1, 1)
```

A voxel farther than the truncation distance behind the visible surface is not
updated. Every valid frame contributes uniform weight `1`, and accumulation
uses float64 values in replay order.

- positive TSDF: camera/free-space side;
- zero: measured surface;
- negative TSDF: just behind the surface; and
- weight zero: unknown, regardless of any default value.

Observations with missing depth or pose are counted and skipped. Poses are
never invented or interpolated.

## Diagnostic artifact

The `.sftsdf` JSON records:

- the session and replay digest;
- volume origin, dimensions, resolution, truncation, and index conventions;
- integrated/skipped frame counts;
- observed, unknown, and multiply fused voxel counts;
- total voxel updates and maximum weight; and
- observed voxel indices, TSDF values, and weights.

The file is deterministic and includes no timestamp or machine-specific path.
It is a reference diagnostic format, not a promised long-term storage format.

## Explicitly deferred

- robust automatic-bound outlier handling and sparse voxel blocks;
- color fusion and sensor-dependent weighting;
- Open3D, GPU, or other optimized backends;
- production meshing, exact-zero-cell handling, and optimized surface
  reconstruction;
- smoothing, normals, floor/wall detection, and visualization; and
- pose estimation, tracking, loop closure, and SLAM.
