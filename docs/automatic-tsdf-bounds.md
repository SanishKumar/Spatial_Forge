# Deterministic automatic TSDF bounds

This milestone removes the need to hand-calculate a dense reference volume:

```powershell
.\.venv\Scripts\python.exe -m spatialforge reconstruct tsdf-auto `
  tests/fixtures/minimal.vgsession `
  outputs/progress-auto.sftsdf `
  --voxel-size-m 0.5 `
  --truncation-m 0.5
```

It derives only the volume origin and dimensions. Fusion still uses the same
fixed-bounds CPU reference and produces the unchanged `.sftsdf` schema.

## Bound rule

The prepass uses every selected RGB observation that has both exact depth and
an exact known pose. Every positive finite depth pixel is back-projected with
the aligned pinhole calibration and transformed by `T_world_camera`. There is
no pixel subsampling, pose interpolation, or fabricated identity pose.

The world-space surface minimum and maximum are padded by exactly the TSDF
truncation distance. Each axis is then snapped outward to a grid anchored at
world zero:

```text
lower_index = floor((surface_min - truncation) / voxel_size)
upper_index = ceil((surface_max + truncation) / voxel_size)
origin = lower_index * voxel_size
dimension = upper_index - lower_index
```

Binary64 arithmetic is used directly without an epsilon. Negative zero is
normalized. Invalid depth samples are counted and excluded; a valid depth that
transforms to a non-finite world coordinate fails actionably.

The inferred dense volume must contain at most 1,000,000 voxels. SpatialForge
never silently increases resolution, discards an outlier, or allocates beyond
that cap. The fusion replay must also reproduce the bounds-prepass replay
digest; inputs changed between those checkpoints fail before the artifact is
published.

## Exact fixture proof

The committed two-frame fixture produces eight surface points with:

```text
surface minimum: (1.0, -0.25, -0.25)
surface maximum: (1.0,  0.25,  0.25)
```

At `0.5 m` voxels and `0.5 m` truncation, the result is:

```text
origin:     (0.5, -1.0, -1.0)
upper:      (1.5,  1.0,  1.0)
dimensions: (2, 4, 4)
voxels:     32
```

The resulting reference TSDF reports eight observed and fused voxels, sixteen
updates, maximum weight two, and this deterministic SHA-256:

```text
e61803737cdd68b209459fb644cc2f67f18e0420d306273316808e27d2e89994
```

It is byte-identical to manually running `reconstruct tsdf` with the inferred
origin and dimensions.

## Explicitly deferred

- configurable padding or automatic voxel/truncation selection;
- depth-range clipping, confidence masks, and statistical outlier rejection;
- oriented, PCA, frustum, camera-trajectory, or room-aware bounds;
- sparse blocks, adaptive resolution, submaps, and GPU integration;
- immutable input snapshots during each individual replay/decode pass;
- mesh refinements, structural extraction, semantics, and visualization; and
- pose estimation, tracking, loop closure, and SLAM.
