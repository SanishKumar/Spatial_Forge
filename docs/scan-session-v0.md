# ScanSession v0.1

Status: first working contract. Backward compatibility is not promised until
the format reaches v1.

## Purpose and scope

A `ScanSession` is the deterministic input boundary for SpatialForge. It
records calibrated sensor observations so the same input can be validated and
replayed repeatedly while mapping algorithms change.

Version 0.1 supports one canonical representation: a directory whose name ends
in `.vgsession`. Archives, live devices, TUM/ARKitScenes adapters, and network
resources are out of scope.

```text
example.vgsession/
    manifest.json
    calibration/
        cameras.json
    streams/
        rgb.jsonl
        depth.jsonl       optional
        imu.jsonl         optional
        poses.jsonl       optional
    data/
        rgb/...
        depth/...
```

All paths stored in JSON are POSIX-style, relative to the session root, and
must remain inside that root.

## Time

Every sample has an integer `timestamp_ns`.

```json
{
  "timebase": {
    "clock": "monotonic",
    "unit": "nanoseconds",
    "epoch": "session_start"
  }
}
```

Timestamps are elapsed nanoseconds from session start, not wall-clock time.
They must be non-negative and strictly increasing within each stream. Wall
clock creation time is separately recorded as `created_at_utc`.

## Coordinates and transforms

All coordinate systems are right-handed.

- World/rig axes: `+X` forward, `+Y` left, `+Z` up.
- Camera axes: OpenCV convention, `+X` right, `+Y` down, `+Z` forward.
- Distance unit: metres.
- Pose convention: `T_world_camera`.

`T_world_camera` maps a homogeneous point expressed in camera coordinates into
session-world coordinates:

```text
p_world = T_world_camera * p_camera
```

Transforms are row-major 4-by-4 rigid transforms encoded as 16 finite numbers.
The rotation must be orthonormal with determinant `+1`, and the final row must
be `[0, 0, 0, 1]`.

The manifest must state these conventions exactly:

```json
{
  "coordinate_system": {
    "handedness": "right",
    "world_axes": {
      "x": "forward",
      "y": "left",
      "z": "up"
    },
    "camera_axes": {
      "x": "right",
      "y": "down",
      "z": "forward"
    },
    "distance_unit": "metres",
    "pose": "T_world_camera"
  }
}
```

## Manifest

`manifest.json` contains:

| Field | Requirement |
| --- | --- |
| `schema` | Must be `spatialforge.scan-session` |
| `schema_version` | Must be `0.1.0` |
| `session_id` | Stable lowercase identifier |
| `created_at_utc` | ISO 8601 UTC timestamp |
| `timebase` | The canonical timebase above |
| `coordinate_system` | The canonical coordinates above |
| `calibration` | Relative path to the camera calibration JSON |
| `streams` | Stream definitions indexed by canonical stream name |

The `rgb` stream is required. `depth`, `imu`, and `pose` are optional.

```json
{
  "streams": {
    "rgb": {
      "kind": "rgb",
      "index": "streams/rgb.jsonl",
      "calibration_id": "camera-rgb"
    },
    "depth": {
      "kind": "depth",
      "index": "streams/depth.jsonl",
      "calibration_id": "camera-depth",
      "depth_scale_m": 0.001,
      "aligned_to": "rgb"
    },
    "imu": {
      "kind": "imu",
      "index": "streams/imu.jsonl"
    },
    "pose": {
      "kind": "pose",
      "index": "streams/poses.jsonl",
      "transform": "T_world_camera"
    }
  }
}
```

Raw depth value `d` becomes metres with
`depth_metres = d * depth_scale_m`. The scale must be finite and positive.

## Camera calibration

The calibration file contains a unique entry for every camera referenced by a
stream. Version 0.1 supports `pinhole` cameras with `none` or
`opencv-radtan` distortion.

```json
{
  "schema": "spatialforge.camera-calibration",
  "schema_version": "0.1.0",
  "cameras": [
    {
      "id": "camera-rgb",
      "model": "pinhole",
      "width": 640,
      "height": 480,
      "intrinsics": {
        "fx": 525.0,
        "fy": 525.0,
        "cx": 319.5,
        "cy": 239.5
      },
      "distortion": {
        "model": "none",
        "coefficients": []
      },
      "T_rig_camera": [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
      ]
    }
  ]
}
```

`T_rig_camera` maps camera coordinates into the capture rig coordinates and is
validated as a rigid transform.

## Stream records

Each index is UTF-8 JSON Lines with one object per line. Blank lines are
ignored. Every record requires a unique lowercase `id` and a
`timestamp_ns`.

RGB and depth records reference a file:

```json
{"id":"rgb-000000","timestamp_ns":0,"path":"data/rgb/000000.ppm"}
```

IMU values use SI units:

```json
{
  "id": "imu-000000",
  "timestamp_ns": 0,
  "accelerometer_m_s2": [0.0, 0.0, 9.80665],
  "gyroscope_rad_s": [0.0, 0.0, 0.0]
}
```

IMU vectors are already expressed in the rig frame (`+X` forward, `+Y` left,
`+Z` up). Raw device-axis samples must be transformed by a future capture
adapter before producing this canonical format.

Known poses contain the canonical transform:

```json
{
  "id": "pose-000000",
  "timestamp_ns": 0,
  "T_world_camera": [0,0,1,0, -1,0,0,0, 0,-1,0,0, 0,0,0,1]
}
```

## Replay semantics

RGB is the observation clock. For each RGB timestamp, replay:

1. associates depth with the exact same timestamp, if present;
2. associates a known pose with the exact same timestamp, if present; and
3. includes IMU samples in `(previous_rgb_timestamp, rgb_timestamp]`. The
   first observation includes all IMU samples up to its timestamp.

RGB records are replayed in timestamp order. A SHA-256 digest is computed from
the complete validated input: calibration, stream definitions, every stream
record (including records not associated to an RGB frame), referenced RGB and
depth file bytes, and the canonical observation sequence. The same valid
session must produce the same sequence and digest on every run; changing sensor
bytes or calibration changes the digest.

Missing optional associations are reported, not invented or interpolated.
