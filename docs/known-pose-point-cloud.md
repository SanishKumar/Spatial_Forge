# Known-pose RGB-D point cloud

This milestone performs one geometric reconstruction step: it turns exact,
calibrated RGB-D observations with known camera poses into a colored ASCII PLY.
It does not fuse observations into a surface.

```powershell
.\.venv\Scripts\python.exe -m spatialforge reconstruct point-cloud `
  path/to/input.vgsession `
  outputs/scan.ply
```

The command refuses to overwrite an existing output.

## Required input

The input must be a valid `ScanSession v0.1` with:

- RGB and depth streams;
- depth declared as `aligned_to: "rgb"`;
- a positive `depth_scale_m`;
- a pinhole RGB calibration with `distortion.model: "none"`; and
- at least one RGB observation with exact timestamp associations for both depth
  and `T_world_camera`.

RGB and depth images must match the calibrated RGB width and height. Pillow
decodes the image payloads, including RGB images and 16-bit integer depth PNGs.
Raw depth value zero is invalid and is omitted.

Observations with missing depth or pose are counted and skipped. SpatialForge
does not synthesize an identity pose or interpolate an association.

## Geometry

For each sampled pixel `(u, v)` with raw depth `d`:

```text
z = d * depth_scale_m
x = (u - cx) * z / fx
y = (v - cy) * z / fy

p_world = T_world_camera * [x, y, z, 1]
```

The PLY therefore uses the session-world axes: `+X` forward, `+Y` left, and
`+Z` up, in metres. RGB values come from the aligned pixel.

Vertices are written deterministically in replay-frame order and then
row-major pixel order. Coordinates use nine decimal places, line endings are
LF, and the header records the session ID, replay digest, strides, and
integrated frame count. The CLI also reports the output SHA-256 digest.

## Optional sampling

Two positive integer options reduce work while preserving deterministic order:

```powershell
--frame-stride 2
--pixel-stride 4
```

Both are zero-based: a frame stride selects observation sequences `0, 2, 4,
...`; a pixel stride samples rows and columns `0, 4, 8, ...`.

## Explicitly deferred

- lens undistortion and unregistered RGB/depth alignment;
- pose estimation, interpolation, tracking, and SLAM;
- voxel filtering, deduplication, outlier removal, and normals;
- TSDF fusion in this command (the bounded reference step is documented
  separately);
- surface reconstruction and meshing; and
- semantic structure, map packages, and visualization.
