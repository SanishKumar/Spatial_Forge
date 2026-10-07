"""Deterministic importer for extracted TUM RGB-D benchmark sequences."""

from __future__ import annotations

import bisect
import json
import math
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

from .errors import SessionValidationError, TumImportError
from .session_loader import load_scan_session

NANOSECONDS_PER_SECOND = Decimal(1_000_000_000)
MAX_ASSOCIATION_DIFFERENCE_NS = 20_000_000
DEPTH_SCALE_M = 1.0 / 5000.0

# Maps OpenCV optical camera axes (right, down, forward) into the canonical
# SpatialForge rig axes (forward, left, up).
T_RIG_CAMERA = (
    0.0,
    0.0,
    1.0,
    0.0,
    -1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    -1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
)


@dataclass(frozen=True, slots=True)
class TumCameraIntrinsics:
    """Pinhole parameters, in pixels, for a 640 by 480 registered image."""

    fx: float
    fy: float
    cx: float
    cy: float


# The projection the TUM benchmark recommends for its Kinect sequences.
TUM_DEFAULT_INTRINSICS = TumCameraIntrinsics(
    fx=525.0,
    fy=525.0,
    cx=319.5,
    cy=239.5,
)


@dataclass(frozen=True, slots=True)
class TumImportReport:
    source: Path
    output: Path
    session_id: str
    source_rgb_count: int
    source_depth_count: int
    source_pose_count: int
    matched_rgbd_count: int
    matched_pose_count: int

    @property
    def unmatched_rgb_count(self) -> int:
        return self.source_rgb_count - self.matched_rgbd_count

    @property
    def unmatched_depth_count(self) -> int:
        return self.source_depth_count - self.matched_rgbd_count

    @property
    def unmatched_pose_count(self) -> int:
        return self.source_pose_count - self.matched_pose_count


@dataclass(frozen=True, slots=True)
class _TimedPath:
    timestamp_ns: int
    reference: str
    path: Path


@dataclass(frozen=True, slots=True)
class _TumPose:
    timestamp_ns: int
    translation_m: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]
    transform: tuple[float, ...]


def import_tum_dataset(
    source: str | Path,
    output: str | Path,
    *,
    intrinsics: TumCameraIntrinsics = TUM_DEFAULT_INTRINSICS,
) -> TumImportReport:
    """Convert one extracted TUM folder into a validated ScanSession.

    ``intrinsics`` is for sequences published in the TUM layout by a
    different camera; the default is the benchmark's own.
    """

    source_root = Path(source)
    output_root = Path(output)
    _validate_intrinsics(intrinsics)

    if not source_root.exists():
        raise TumImportError(f"source directory does not exist: {source_root}")
    if not source_root.is_dir():
        raise TumImportError(f"source must be a directory: {source_root}")
    if output_root.suffix != ".vgsession":
        raise TumImportError("output directory name must end in .vgsession")

    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise TumImportError(f"output already exists: {output_root}")

    rgb_samples = _parse_timed_paths(source_root, "rgb.txt")
    depth_samples = _parse_timed_paths(source_root, "depth.txt")
    pose_path = source_root / "groundtruth.txt"
    poses = _parse_poses(pose_path) if pose_path.is_file() else []

    rgbd_matches = _associate_timestamps(
        [sample.timestamp_ns for sample in rgb_samples],
        [sample.timestamp_ns for sample in depth_samples],
    )
    if not rgbd_matches:
        raise TumImportError(
            "rgb.txt and depth.txt have no unique pairs within 20 ms"
        )

    matched_pairs = sorted(rgbd_matches.items())
    matched_rgb_timestamps = [
        rgb_samples[rgb_index].timestamp_ns
        for rgb_index, _ in matched_pairs
    ]
    pose_matches = _associate_timestamps(
        matched_rgb_timestamps,
        [pose.timestamp_ns for pose in poses],
    )

    session_id = _session_id(source_root.name)
    report = TumImportReport(
        source=source_root,
        output=output_root,
        session_id=session_id,
        source_rgb_count=len(rgb_samples),
        source_depth_count=len(depth_samples),
        source_pose_count=len(poses),
        matched_rgbd_count=len(matched_pairs),
        matched_pose_count=len(pose_matches),
    )

    try:
        output_root.parent.mkdir(parents=True, exist_ok=True)
        temporary_root = Path(
            tempfile.mkdtemp(
                prefix=f".{output_root.stem}-",
                suffix=".vgsession",
                dir=output_root.parent,
            )
        )
    except OSError as error:
        raise TumImportError(
            f"cannot create output staging directory: {error}"
        ) from error

    try:
        _write_session(
            temporary_root,
            report,
            rgb_samples,
            depth_samples,
            poses,
            matched_pairs,
            pose_matches,
            intrinsics,
        )
        try:
            load_scan_session(temporary_root)
        except SessionValidationError as error:
            details = "\n".join(f"- {problem}" for problem in error.errors)
            raise TumImportError(
                f"generated session failed validation:\n{details}"
            ) from error
        if output_root.exists():
            raise TumImportError(
                f"output appeared while importing; refusing to overwrite: "
                f"{output_root}"
            )
        temporary_root.rename(output_root)
    except TumImportError:
        raise
    except OSError as error:
        raise TumImportError(f"cannot build output session: {error}") from error
    finally:
        if temporary_root.exists():
            shutil.rmtree(temporary_root, ignore_errors=True)

    return report


def _validate_intrinsics(intrinsics: TumCameraIntrinsics) -> None:
    for name in ("fx", "fy", "cx", "cy"):
        value = getattr(intrinsics, name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise TumImportError(f"intrinsics {name} must be a finite number")
    for name in ("fx", "fy"):
        if getattr(intrinsics, name) <= 0.0:
            # A negative focal length is how some renderers publish a
            # left-handed camera. Accepting it would mirror the scene
            # silently; the poses have to be converted along with it.
            raise TumImportError(
                f"intrinsics {name} must be positive; a sequence published "
                "with a negative focal length uses a mirrored camera frame "
                "and must be converted before import"
            )


def _parse_timed_paths(root: Path, filename: str) -> list[_TimedPath]:
    path = root / filename
    lines = _read_lines(path, filename)
    samples: list[_TimedPath] = []
    previous_timestamp: int | None = None

    for line_number, line in lines:
        fields = line.split()
        if len(fields) != 2:
            raise TumImportError(
                f"{filename}:{line_number}: expected "
                "'timestamp relative/path'"
            )

        timestamp = _timestamp_ns(
            fields[0], f"{filename}:{line_number}: timestamp"
        )
        if previous_timestamp is not None and timestamp <= previous_timestamp:
            raise TumImportError(
                f"{filename}:{line_number}: timestamps must be "
                "strictly increasing"
            )
        previous_timestamp = timestamp

        reference, resolved = _resolve_source_file(
            root,
            fields[1],
            f"{filename}:{line_number}: path",
        )
        samples.append(
            _TimedPath(
                timestamp_ns=timestamp,
                reference=reference,
                path=resolved,
            )
        )

    if not samples:
        raise TumImportError(f"{filename}: expected at least one data row")
    return samples


def _parse_poses(path: Path) -> list[_TumPose]:
    lines = _read_lines(path, "groundtruth.txt")
    poses: list[_TumPose] = []
    previous_timestamp: int | None = None

    for line_number, line in lines:
        fields = line.split()
        label = f"groundtruth.txt:{line_number}"
        if len(fields) != 8:
            raise TumImportError(
                f"{label}: expected 'timestamp tx ty tz qx qy qz qw'"
            )

        timestamp = _timestamp_ns(fields[0], f"{label}: timestamp")
        if previous_timestamp is not None and timestamp <= previous_timestamp:
            raise TumImportError(
                f"{label}: timestamps must be strictly increasing"
            )
        previous_timestamp = timestamp

        translation = tuple(
            _finite_float(value, f"{label}: translation[{index}]")
            for index, value in enumerate(fields[1:4])
        )
        quaternion = tuple(
            _finite_float(value, f"{label}: quaternion[{index}]")
            for index, value in enumerate(fields[4:8])
        )
        transform = _pose_transform(translation, quaternion, label)
        poses.append(
            _TumPose(
                timestamp_ns=timestamp,
                translation_m=translation,  # type: ignore[arg-type]
                quaternion_xyzw=quaternion,  # type: ignore[arg-type]
                transform=transform,
            )
        )

    return poses


def _read_lines(path: Path, label: str) -> list[tuple[int, str]]:
    if not path.is_file():
        raise TumImportError(f"required file does not exist: {label}")
    try:
        raw_lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise TumImportError(f"cannot read {label}: {error}") from error

    lines: list[tuple[int, str]] = []
    for line_number, raw_line in enumerate(raw_lines, start=1):
        line = raw_line.split("#", 1)[0].strip()
        if line:
            lines.append((line_number, line))
    return lines


def _timestamp_ns(value: str, label: str) -> int:
    try:
        timestamp = Decimal(value)
    except InvalidOperation as error:
        raise TumImportError(f"{label}: expected decimal Unix seconds") from error
    if not timestamp.is_finite() or timestamp < 0:
        raise TumImportError(
            f"{label}: expected non-negative finite Unix seconds"
        )

    nanoseconds = int(
        (timestamp * NANOSECONDS_PER_SECOND).to_integral_value(
            rounding=ROUND_HALF_EVEN
        )
    )
    if nanoseconds > 2**63 - 1:
        raise TumImportError(f"{label}: timestamp exceeds signed 64-bit range")
    return nanoseconds


def _finite_float(value: str, label: str) -> float:
    try:
        parsed = Decimal(value)
    except InvalidOperation as error:
        raise TumImportError(f"{label}: expected a finite number") from error
    if not parsed.is_finite():
        raise TumImportError(f"{label}: expected a finite number")
    try:
        converted = float(parsed)
    except (OverflowError, ValueError) as error:
        raise TumImportError(f"{label}: expected a finite number") from error
    if not math.isfinite(converted):
        raise TumImportError(f"{label}: expected a finite number")
    return converted


def _resolve_source_file(
    root: Path,
    reference: str,
    label: str,
) -> tuple[str, Path]:
    if "\\" in reference:
        raise TumImportError(f"{label}: use POSIX '/' path separators")
    pure_path = PurePosixPath(reference)
    if pure_path.is_absolute() or ".." in pure_path.parts:
        raise TumImportError(
            f"{label}: path must be relative and remain inside the dataset"
        )

    resolved = root.joinpath(*pure_path.parts).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise TumImportError(
            f"{label}: resolved path escapes the dataset directory"
        ) from error
    if not resolved.is_file():
        raise TumImportError(f"{label}: referenced file does not exist")
    return pure_path.as_posix(), resolved


def _associate_timestamps(
    primary: Sequence[int],
    secondary: Sequence[int],
    max_difference_ns: int = MAX_ASSOCIATION_DIFFERENCE_NS,
) -> dict[int, int]:
    """Apply TUM's global greedy association using exact integer nanoseconds."""

    candidates: list[tuple[int, int, int, int, int]] = []
    for primary_index, primary_timestamp in enumerate(primary):
        lower = bisect.bisect_right(
            secondary, primary_timestamp - max_difference_ns
        )
        upper = bisect.bisect_left(
            secondary, primary_timestamp + max_difference_ns
        )
        for secondary_index in range(lower, upper):
            secondary_timestamp = secondary[secondary_index]
            difference = abs(primary_timestamp - secondary_timestamp)
            candidates.append(
                (
                    difference,
                    primary_timestamp,
                    secondary_timestamp,
                    primary_index,
                    secondary_index,
                )
            )

    candidates.sort()
    used_primary: set[int] = set()
    used_secondary: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _, _, _, primary_index, secondary_index in candidates:
        if (
            primary_index in used_primary
            or secondary_index in used_secondary
        ):
            continue
        used_primary.add(primary_index)
        used_secondary.add(secondary_index)
        matches.append((primary_index, secondary_index))

    return dict(sorted(matches))


def _pose_transform(
    translation: Sequence[float],
    quaternion: Sequence[float],
    label: str,
) -> tuple[float, ...]:
    x, y, z, w = quaternion
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12:
        raise TumImportError(f"{label}: quaternion norm must be positive")
    x, y, z, w = (component / norm for component in (x, y, z, w))

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    xw, yw, zw = x * w, y * w, z * w
    tx, ty, tz = translation

    return tuple(
        _clean_float(value)
        for value in (
            1.0 - 2.0 * (yy + zz),
            2.0 * (xy - zw),
            2.0 * (xz + yw),
            tx,
            2.0 * (xy + zw),
            1.0 - 2.0 * (xx + zz),
            2.0 * (yz - xw),
            ty,
            2.0 * (xz - yw),
            2.0 * (yz + xw),
            1.0 - 2.0 * (xx + yy),
            tz,
            0.0,
            0.0,
            0.0,
            1.0,
        )
    )


def _write_session(
    root: Path,
    report: TumImportReport,
    rgb_samples: Sequence[_TimedPath],
    depth_samples: Sequence[_TimedPath],
    poses: Sequence[_TumPose],
    matched_pairs: Sequence[tuple[int, int]],
    pose_matches: dict[int, int],
    intrinsics: TumCameraIntrinsics,
) -> None:
    (root / "calibration").mkdir(parents=True)
    (root / "streams").mkdir()
    (root / "data" / "rgb").mkdir(parents=True)
    (root / "data" / "depth").mkdir()

    first_timestamp = rgb_samples[matched_pairs[0][0]].timestamp_ns
    rgb_records: list[dict[str, Any]] = []
    depth_records: list[dict[str, Any]] = []
    matched_rgb_timestamps: list[int] = []

    for output_index, (rgb_index, depth_index) in enumerate(matched_pairs):
        rgb = rgb_samples[rgb_index]
        depth = depth_samples[depth_index]
        canonical_timestamp = rgb.timestamp_ns - first_timestamp
        matched_rgb_timestamps.append(rgb.timestamp_ns)

        rgb_reference = _copy_sensor_file(
            rgb.path, root / "data" / "rgb", output_index
        )
        depth_reference = _copy_sensor_file(
            depth.path, root / "data" / "depth", output_index
        )
        rgb_records.append(
            {
                "id": f"rgb-{output_index:06d}",
                "timestamp_ns": canonical_timestamp,
                "path": f"data/rgb/{rgb_reference}",
                "source_timestamp_ns": rgb.timestamp_ns,
                "source_path": rgb.reference,
            }
        )
        depth_records.append(
            {
                "id": f"depth-{output_index:06d}",
                "timestamp_ns": canonical_timestamp,
                "path": f"data/depth/{depth_reference}",
                "source_timestamp_ns": depth.timestamp_ns,
                "source_path": depth.reference,
                "association_delta_ns": (
                    depth.timestamp_ns - rgb.timestamp_ns
                ),
            }
        )

    pose_records = _normalized_pose_records(
        poses,
        pose_matches,
        matched_rgb_timestamps,
        first_timestamp,
    )

    streams: dict[str, dict[str, Any]] = {
        "rgb": {
            "kind": "rgb",
            "index": "streams/rgb.jsonl",
            "calibration_id": "camera-rgb",
        },
        "depth": {
            "kind": "depth",
            "index": "streams/depth.jsonl",
            "calibration_id": "camera-rgb",
            "depth_scale_m": DEPTH_SCALE_M,
            "aligned_to": "rgb",
        },
    }
    if pose_records:
        streams["pose"] = {
            "kind": "pose",
            "index": "streams/poses.jsonl",
            "transform": "T_world_camera",
        }

    manifest = {
        "schema": "spatialforge.scan-session",
        "schema_version": "0.1.0",
        "session_id": report.session_id,
        "created_at_utc": _utc_timestamp(first_timestamp),
        "timebase": {
            "clock": "monotonic",
            "unit": "nanoseconds",
            "epoch": "session_start",
        },
        "coordinate_system": {
            "handedness": "right",
            "world_axes": {
                "x": "forward",
                "y": "left",
                "z": "up",
            },
            "camera_axes": {
                "x": "right",
                "y": "down",
                "z": "forward",
            },
            "distance_unit": "metres",
            "pose": "T_world_camera",
        },
        "calibration": "calibration/cameras.json",
        "streams": streams,
    }
    calibration = {
        "schema": "spatialforge.camera-calibration",
        "schema_version": "0.1.0",
        "cameras": [
            {
                "id": "camera-rgb",
                "model": "pinhole",
                "width": 640,
                "height": 480,
                "intrinsics": {
                    "fx": float(intrinsics.fx),
                    "fy": float(intrinsics.fy),
                    "cx": float(intrinsics.cx),
                    "cy": float(intrinsics.cy),
                },
                "distortion": {
                    "model": "none",
                    "coefficients": [],
                },
                "T_rig_camera": list(T_RIG_CAMERA),
            }
        ],
    }

    _write_json(root / "manifest.json", manifest)
    _write_json(root / "calibration" / "cameras.json", calibration)
    _write_jsonl(root / "streams" / "rgb.jsonl", rgb_records)
    _write_jsonl(root / "streams" / "depth.jsonl", depth_records)
    if pose_records:
        _write_jsonl(root / "streams" / "poses.jsonl", pose_records)


def _normalized_pose_records(
    poses: Sequence[_TumPose],
    matches: dict[int, int],
    rgb_timestamps: Sequence[int],
    first_rgb_timestamp: int,
) -> list[dict[str, Any]]:
    if not matches:
        return []

    first_output_index = min(matches)
    anchor_pose = poses[matches[first_output_index]].transform
    session_from_tum = _multiply_transform(
        T_RIG_CAMERA, _invert_rigid_transform(anchor_pose)
    )

    records: list[dict[str, Any]] = []
    for output_index, pose_index in sorted(matches.items()):
        pose = poses[pose_index]
        rgb_timestamp = rgb_timestamps[output_index]
        records.append(
            {
                "id": f"pose-{output_index:06d}",
                "timestamp_ns": rgb_timestamp - first_rgb_timestamp,
                "T_world_camera": list(
                    _multiply_transform(session_from_tum, pose.transform)
                ),
                "source_timestamp_ns": pose.timestamp_ns,
                "association_delta_ns": (
                    pose.timestamp_ns - rgb_timestamp
                ),
                "source_translation_m": list(pose.translation_m),
                "source_quaternion_xyzw": list(pose.quaternion_xyzw),
            }
        )
    return records


def _multiply_transform(
    left: Sequence[float],
    right: Sequence[float],
) -> tuple[float, ...]:
    return tuple(
        _clean_float(
            sum(left[row * 4 + axis] * right[axis * 4 + column] for axis in range(4))
        )
        for row in range(4)
        for column in range(4)
    )


def _invert_rigid_transform(transform: Sequence[float]) -> tuple[float, ...]:
    rotation_transpose = (
        transform[0],
        transform[4],
        transform[8],
        transform[1],
        transform[5],
        transform[9],
        transform[2],
        transform[6],
        transform[10],
    )
    tx, ty, tz = transform[3], transform[7], transform[11]
    inverse_translation = (
        -(rotation_transpose[0] * tx + rotation_transpose[1] * ty + rotation_transpose[2] * tz),
        -(rotation_transpose[3] * tx + rotation_transpose[4] * ty + rotation_transpose[5] * tz),
        -(rotation_transpose[6] * tx + rotation_transpose[7] * ty + rotation_transpose[8] * tz),
    )
    return tuple(
        _clean_float(value)
        for value in (
            rotation_transpose[0],
            rotation_transpose[1],
            rotation_transpose[2],
            inverse_translation[0],
            rotation_transpose[3],
            rotation_transpose[4],
            rotation_transpose[5],
            inverse_translation[1],
            rotation_transpose[6],
            rotation_transpose[7],
            rotation_transpose[8],
            inverse_translation[2],
            0.0,
            0.0,
            0.0,
            1.0,
        )
    )


def _copy_sensor_file(source: Path, destination: Path, index: int) -> str:
    suffix = source.suffix.lower() or ".bin"
    if not re.fullmatch(r"\.[a-z0-9]{1,10}", suffix):
        suffix = ".bin"
    filename = f"{index:06d}{suffix}"
    shutil.copyfile(source, destination / filename)
    return filename


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: Sequence[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(
                value,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
            for value in values
        ),
        encoding="utf-8",
    )


def _utc_timestamp(timestamp_ns: int) -> str:
    seconds, nanoseconds = divmod(timestamp_ns, 1_000_000_000)
    try:
        value = datetime.fromtimestamp(seconds, tz=timezone.utc).replace(
            microsecond=nanoseconds // 1000
        )
    except (OSError, OverflowError, ValueError) as error:
        raise TumImportError(
            "first imported RGB timestamp cannot be represented as UTC"
        ) from error
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _session_id(source_name: str) -> str:
    normalized = re.sub(r"[^a-z0-9._-]+", "-", source_name.lower()).strip(
        ".-_"
    )
    if not normalized:
        normalized = "dataset"
    if not normalized.startswith("tum-"):
        normalized = f"tum-{normalized}"
    return normalized[:128]


def _clean_float(value: float) -> float:
    for exact in (-1.0, 0.0, 1.0):
        if abs(value - exact) < 1e-15:
            return exact
    return float(value)
