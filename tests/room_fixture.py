"""A deterministic room scan that is deliberately not degenerate.

The committed `minimal.vgsession` is a 2x2 image whose surfaces land exactly
on voxel and block boundaries, whose camera sits at the world origin, and
whose depths are exact binary fractions. That makes it perfect for pinning
exact expectations and useless for showing that the geometry survives real
sensor data.

This fixture breaks every one of those degeneracies:

  - a real image size rather than 2x2;
  - room surfaces at metre values that are never a multiple of the voxel or
    block extent;
  - camera poses that translate and rotate, never at the exact origin;
  - millimetre-quantised depth with gaussian noise;
  - many frames rather than two.

It is generated from a fixed seed rather than committed, because the images
are about 1.2 MB while the generator is a few kilobytes and reproduces them
byte for byte.
"""

from __future__ import annotations

import atexit
import json
import math
import random
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parent

WIDTH, HEIGHT = 64, 48
FX = FY = 48.0
CX, CY = 31.5, 23.5
FRAMES = 20
DEPTH_SCALE_M = 0.001
NOISE_SIGMA_M = 0.004
SEED = 20260806

# Ground truth. None of these is a multiple of 0.04 (voxel) or 0.32 (block).
FAR_WALL_X = 2.537
LEFT_WALL_Y = 1.313
RIGHT_WALL_Y = -1.229
FLOOR_Z = -1.117
CEILING_Z = 1.409
BOX = (1.271, 1.783, -0.451, 0.219, FLOOR_Z, -0.337)


@dataclass(frozen=True, slots=True)
class RoomTruth:
    """The planes the generator drew, for accuracy assertions."""

    far_wall_x: float = FAR_WALL_X
    left_wall_y: float = LEFT_WALL_Y
    right_wall_y: float = RIGHT_WALL_Y
    floor_z: float = FLOOR_Z
    ceiling_z: float = CEILING_Z
    noise_sigma_m: float = NOISE_SIGMA_M
    frames: int = FRAMES
    width: int = WIDTH
    height: int = HEIGHT


def _ray_depth(ox, oy, oz, dx, dy, dz) -> float:
    """Nearest positive hit against the room shell and the interior box."""

    def plane(origin, direction, value):
        if abs(direction) < 1e-12:
            return math.inf
        t = (value - origin) / direction
        return t if t > 1e-6 else math.inf

    best = min(
        plane(ox, dx, FAR_WALL_X),
        plane(oy, dy, LEFT_WALL_Y),
        plane(oy, dy, RIGHT_WALL_Y),
        plane(oz, dz, FLOOR_Z),
        plane(oz, dz, CEILING_Z),
    )

    x0, x1, y0, y1, z0, z1 = BOX
    tmin, tmax = 1e-6, best
    for origin, direction, lo, hi in (
        (ox, dx, x0, x1),
        (oy, dy, y0, y1),
        (oz, dz, z0, z1),
    ):
        if abs(direction) < 1e-12:
            if origin < lo or origin > hi:
                tmin, tmax = 1.0, -1.0
                break
            continue
        near, far = (lo - origin) / direction, (hi - origin) / direction
        if near > far:
            near, far = far, near
        tmin, tmax = max(tmin, near), min(tmax, far)
    if tmin <= tmax and tmin < best:
        best = tmin
    return best


def _pose_for(frame: int) -> tuple[float, ...]:
    """A short arc: translating and yawing, never at the world origin."""

    u = frame / max(FRAMES - 1, 1)
    origin = (
        -0.083 + 0.117 * u,
        -0.271 + 0.463 * u,
        0.059 + 0.088 * math.sin(u * math.pi),
    )
    yaw = math.radians(-7.3 + 14.6 * u)
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    forward = (cos_yaw, sin_yaw, 0.0)
    right = (sin_yaw, -cos_yaw, 0.0)
    down = (0.0, 0.0, -1.0)
    return (
        right[0], down[0], forward[0], origin[0],
        right[1], down[1], forward[1], origin[1],
        right[2], down[2], forward[2], origin[2],
        0.0, 0.0, 0.0, 1.0,
    )


def _write_session(root: Path) -> None:
    (root / "data" / "depth").mkdir(parents=True)
    (root / "data" / "rgb").mkdir(parents=True)
    (root / "streams").mkdir()
    (root / "calibration").mkdir()

    rng = random.Random(SEED)
    indexes: dict[str, list[str]] = {
        "rgb": [], "depth": [], "poses": [], "imu": [],
    }

    for frame in range(FRAMES):
        timestamp = frame * 33_333_333
        pose = _pose_for(frame)
        ox, oy, oz = pose[3], pose[7], pose[11]
        axis_x = (pose[0], pose[4], pose[8])
        axis_y = (pose[1], pose[5], pose[9])
        axis_z = (pose[2], pose[6], pose[10])

        samples: list[int] = []
        for v in range(HEIGHT):
            for u in range(WIDTH):
                cx = (u - CX) / FX
                cy = (v - CY) / FY
                dx = axis_x[0] * cx + axis_y[0] * cy + axis_z[0]
                dy = axis_x[1] * cx + axis_y[1] * cy + axis_z[1]
                dz = axis_x[2] * cx + axis_y[2] * cy + axis_z[2]
                norm = math.sqrt(dx * dx + dy * dy + dz * dz)
                distance = _ray_depth(
                    ox, oy, oz, dx / norm, dy / norm, dz / norm
                )
                if not math.isfinite(distance):
                    samples.append(0)
                    continue
                # Metric depth is along the camera z axis, not along the ray.
                along_z = (
                    dx * axis_z[0] + dy * axis_z[1] + dz * axis_z[2]
                ) / norm
                metres = distance * along_z + rng.gauss(0.0, NOISE_SIGMA_M)
                raw = int(round(metres / DEPTH_SCALE_M))
                samples.append(raw if 0 < raw < 65535 else 0)

        name = f"{frame:06d}"
        rows = [
            " ".join(str(samples[v * WIDTH + u]) for u in range(WIDTH))
            for v in range(HEIGHT)
        ]
        (root / "data" / "depth" / f"{name}.pgm").write_text(
            f"P2\n{WIDTH} {HEIGHT}\n65535\n" + "\n".join(rows) + "\n",
            encoding="ascii",
        )
        grey = [
            str(min(255, 40 + (sample // 20) % 200)) for sample in samples
        ]
        rgb_rows = [
            " ".join(
                f"{grey[v * WIDTH + u]} {grey[v * WIDTH + u]} "
                f"{grey[v * WIDTH + u]}"
                for u in range(WIDTH)
            )
            for v in range(HEIGHT)
        ]
        (root / "data" / "rgb" / f"{name}.ppm").write_text(
            f"P3\n{WIDTH} {HEIGHT}\n255\n" + "\n".join(rgb_rows) + "\n",
            encoding="ascii",
        )

        indexes["rgb"].append(json.dumps({
            "id": f"rgb-{name}", "timestamp_ns": timestamp,
            "path": f"data/rgb/{name}.ppm"}))
        indexes["depth"].append(json.dumps({
            "id": f"depth-{name}", "timestamp_ns": timestamp,
            "path": f"data/depth/{name}.pgm"}))
        indexes["poses"].append(json.dumps({
            "id": f"pose-{name}", "timestamp_ns": timestamp,
            "T_world_camera": list(pose)}))
        indexes["imu"].append(json.dumps({
            "id": f"imu-{name}", "timestamp_ns": timestamp,
            "accelerometer_m_s2": [0.0, 0.0, 9.80665],
            "gyroscope_rad_s": [0.0, 0.0, 0.01]}))

    for name, lines in indexes.items():
        (root / "streams" / f"{name}.jsonl").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

    camera = {
        "id": "camera-rgb", "model": "pinhole",
        "width": WIDTH, "height": HEIGHT,
        "intrinsics": {"fx": FX, "fy": FY, "cx": CX, "cy": CY},
        "distortion": {"model": "none", "coefficients": []},
        "T_rig_camera": [0, 0, 1, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 1],
    }
    (root / "calibration" / "cameras.json").write_text(
        json.dumps({
            "schema": "spatialforge.camera-calibration",
            "schema_version": "0.1.0",
            "cameras": [camera, dict(camera, id="camera-depth")],
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    (root / "manifest.json").write_text(
        json.dumps({
            "schema": "spatialforge.scan-session",
            "schema_version": "0.1.0",
            "session_id": "scan-room-0001",
            "created_at_utc": "2026-08-06T00:00:00Z",
            "timebase": {
                "clock": "monotonic", "unit": "nanoseconds",
                "epoch": "session_start"},
            "coordinate_system": {
                "handedness": "right",
                "world_axes": {"x": "forward", "y": "left", "z": "up"},
                "camera_axes": {"x": "right", "y": "down", "z": "forward"},
                "distance_unit": "metres", "pose": "T_world_camera"},
            "calibration": "calibration/cameras.json",
            "streams": {
                "rgb": {
                    "kind": "rgb", "index": "streams/rgb.jsonl",
                    "calibration_id": "camera-rgb"},
                "depth": {
                    "kind": "depth", "index": "streams/depth.jsonl",
                    "calibration_id": "camera-depth",
                    "depth_scale_m": DEPTH_SCALE_M, "aligned_to": "rgb"},
                "imu": {"kind": "imu", "index": "streams/imu.jsonl"},
                "pose": {
                    "kind": "pose", "index": "streams/poses.jsonl",
                    "transform": "T_world_camera"}},
        }, indent=2) + "\n",
        encoding="utf-8",
    )


_ROOT: Path | None = None


def room_session() -> Path:
    """Return the generated room session, building it once per run."""

    global _ROOT
    if _ROOT is None:
        holder = Path(tempfile.mkdtemp(dir=TESTS_ROOT))
        _ROOT = holder / "room.vgsession"
        _write_session(_ROOT)
        atexit.register(shutil.rmtree, holder, True)
    return _ROOT


def room_truth() -> RoomTruth:
    return RoomTruth()
