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

Other datasets publish in the TUM layout with a different camera. For those,
the projection can be given explicitly; the defaults are the values above,
so an import without these options is byte-for-byte what it always was:

```powershell
python -m spatialforge scan import-tum SOURCE OUTPUT.vgsession `
  --fx 481.2 --fy 480 --cx 319.5 --cy 239.5
```

A focal length that is zero, negative or not finite is refused. A negative
one is not a typo to be corrected by taking its absolute value: it is how
some renderers describe a left-handed camera, and flipping the sign of one
number without converting the poses would import a mirrored room. ICL-NUIM
is published that way; [`icl-nuim-validation.md`](icl-nuim-validation.md)
describes the conversion.

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

### A level session

That frame's "up" is the first camera's own. A camera held looking down at
a desk takes the whole room with it: in `freiburg1_room` the first frame
looks 41 degrees below the horizon, and the session's z axis is 41 degrees
off the room's. Reconstruction does not mind, since nothing in fusion knows
which way is down. Anything that reads the volume as a floor plan does: a
[free-space map](free-space-expansion.md) is a column of voxels over a
height band, and a tilted grid has no such columns.

`--up` names the axis of the dataset's frame that points up and asks for a
level session instead:

```powershell
python -m spatialforge scan import-tum SOURCE OUTPUT.vgsession --up z
```

```text
z_session  = the named axis
origin     = the first posed camera
x_session  = the way that camera faces, with its climb or dive taken out
y_session  = z cross x
```

The TUM benchmark's motion-capture frame has z up, which the depth itself
confirms for `freiburg1_room`: the direction most of its surfaces face is
0.9 degrees from that axis, with the floor at 0.00 m and the desk tops at
0.75 m. `--down y` is the same for a frame whose y points at the floor.

The motion between frames is the same in either session; only the frame
they are written in differs, so the two reconstruct the same room on
differently turned grids. When the first camera happens to be level
already, the two sessions are the same files. A first camera that looks
straight along the up axis faces no way along the floor, and the import is
refused, as is a level session of a sequence with no poses. Without `--up`
or `--down` an import is byte for byte what it always was.

## Explicitly deferred

- dataset downloads and archive extraction;
- image preprocessing;
- IMU import;
- Open3D, TSDF fusion, meshing, or reconstruction;
- visual tracking and SLAM.
