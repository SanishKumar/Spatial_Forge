# TSDF zero-crossing surface points

This milestone converts a validated `.sftsdf` reference volume into
deterministic world-space surface points:

```powershell
.\.venv\Scripts\python.exe -m spatialforge reconstruct surface-points `
  outputs/progress.sftsdf `
  outputs/progress-surface.ply
```

It is the narrow surface step between TSDF integration and a future triangle
mesh. The output is an ASCII PLY containing XYZ points only.

## Extraction rule

Only observed voxels from the TSDF artifact participate.

1. Exact TSDF-zero voxel centers are emitted once in X-fastest order.
2. Remaining edges are traversed along `+X`, then `+Y`, then `+Z`.
3. Both edge endpoints must be observed.
4. Both values must be nonzero and have strictly opposite signs.
5. The crossing is linearly interpolated:

```text
t = tsdf_a / (tsdf_a - tsdf_b)
p = center_a + t * (center_b - center_a)
```

An edge touching an exact-zero voxel emits no additional point. An unknown
voxel is never treated as zero, and the extractor never bridges an unknown
gap. Voxel weights establish that a value is observed, but do not move the
interpolated point in this milestone.

Output order, nine-decimal coordinate formatting, LF line endings, comments,
and hashes are deterministic. Existing outputs are never overwritten.

## Exact fixture proof

The reference TSDF command documented in the README produces observed voxel
centers at world `X=0.25`, `0.75`, and `1.25 m`, with TSDF values `1.0`, `0.5`,
and `-0.5`.

The sign-changing edge between `0.75` and `1.25 m` yields:

```text
1.000000000 0.000000000 0.000000000
```

A successful extraction reports:

```text
voxels: total=4 observed=3
observed_edges: 2
crossings: x=1 y=0 z=0
points: exact_zero=0 crossing=1 total=1
```

## Input validation

The extractor validates the reference schema and version, coordinate/index
conventions, dimensions and volume extent, ordered unique voxel indices, TSDF
range, positive weights, replay digest, and all integration counts. Malformed
or inconsistent artifacts fail without a partial PLY.

## Explicitly deferred

- triangle faces and marching cubes;
- normals, colors, smoothing, and deduplication;
- connected components and watertightness;
- optimized or sparse TSDF storage;
- floor, wall, opening, and room extraction; and
- visualization and map-package export.
