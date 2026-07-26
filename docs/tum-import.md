# TUM RGB-D importer

This step converts one already-extracted TUM RGB-D benchmark sequence into the
canonical folder-backed `ScanSession v0.1` format.

```powershell
python -m spatialforge scan import-tum `
  path/to/rgbd_dataset_freiburg1_xyz `
  outputs/freiburg1-xyz.vgsession
```

The importer refuses to overwrite an existing output. It builds into a
temporary sibling directory, validates the complete session, and only then
renames it to the requested output.

## Supported input

```text
rgbd_dataset_.../
    rgb.txt
    depth.txt
    groundtruth.txt    optional
    rgb/
    depth/
```

Downloading archives, extracting TGZ files, reading ROS bags, and importing
accelerometer data are not part of this step.

The official TUM format stores RGB and depth timestamps independently. RGB is
640 by 480, depth is a registered 16-bit image, and raw depth is converted with
`metres = raw / 5000`. A raw zero is missing depth. See the official
[TUM file-format documentation](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats).

The text export does not provide complete accelerometer and gyroscope data, so
the importer deliberately emits no IMU stream.

## Timestamp association

The importer follows the TUM benchmark association rule:

```text
abs(rgb_timestamp - depth_timestamp) < 0.02 seconds
```

All candidate pairs are sorted by absolute difference and then greedily
accepted one-to-one. The 20 ms limit is strict; a pair exactly 20 ms apart is
not accepted. This follows the benchmark's
[official association tool](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/tools).

SpatialForge parses the source decimals into exact integer nanoseconds before
matching. TUM's legacy Python script parses binary floating-point seconds, so a
pair written as exactly 20 ms can occasionally fall just below the threshold
through floating-point rounding. SpatialForge deliberately applies the stated
threshold exactly while retaining the same global greedy matching strategy.

Only matched RGB-D pairs enter the output. This guarantees that every replayed
RGB observation has depth. Matched depth and pose records receive their RGB
frame's canonical timestamp, while retaining:

```json
{
  "source_timestamp_ns": 1305031102001000000,
  "association_delta_ns": 1000000
}
```

Canonical timestamps are integer nanoseconds from the first imported RGB
frame. Source timestamps remain Unix-epoch nanoseconds for provenance.

## Calibration

TUM's RGB and depth images are already registered pixel-for-pixel. The importer
uses the benchmark's recommended OpenNI/default projection:

```text
width  = 640
height = 480
fx     = 525.0
fy     = 525.0
cx     = 319.5
cy     = 239.5
distortion = none
depth_scale_m = 0.0002
```

Device-specific depth correction is not applied again because TUM already
pre-scaled the released depth images.

## Pose conversion

When `groundtruth.txt` exists, rows have:

```text
timestamp tx ty tz qx qy qz qw
```

The quaternion is normalized before conversion. TUM's pose maps the RGB optical
camera into its motion-capture world:

```text
P_i = T_tum_world_camera_i
```

The motion-capture world is source-specific, while SpatialForge declares a
forward/left/up session frame. The first associated pose `P_0` therefore anchors
the imported session:

```text
C = T_rig_camera

    0  0  1  0
   -1  0  0  0
    0 -1  0  0
    0  0  0  1

T_session_world_camera_i = C * inverse(P_0) * P_i
```

This makes the first known rig pose the session origin while keeping emitted
camera poses in the documented OpenCV camera-axis convention. Poses are not
interpolated. Missing or gapped ground truth remains missing.

## Explicitly deferred

- dataset downloads and archive extraction;
- real PNG decoding or image preprocessing;
- IMU import;
- point-cloud generation;
- Open3D, TSDF fusion, meshing, or reconstruction;
- visual tracking and SLAM.
