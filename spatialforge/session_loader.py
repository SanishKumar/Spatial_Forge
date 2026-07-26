"""Load and validate the canonical folder-backed ScanSession v0.1 format."""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

from .errors import SessionValidationError
from .model import (
    CameraCalibration,
    ScanSession,
    StreamDefinition,
    StreamSample,
)

SCAN_SCHEMA = "spatialforge.scan-session"
CALIBRATION_SCHEMA = "spatialforge.camera-calibration"
SCHEMA_VERSION = "0.1.0"
STREAM_ORDER = ("rgb", "depth", "imu", "pose")

_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_TIMEBASE = {
    "clock": "monotonic",
    "unit": "nanoseconds",
    "epoch": "session_start",
}
_COORDINATE_SYSTEM = {
    "handedness": "right",
    "world_axes": {"x": "forward", "y": "left", "z": "up"},
    "camera_axes": {"x": "right", "y": "down", "z": "forward"},
    "distance_unit": "metres",
    "pose": "T_world_camera",
}
_IDENTITY = (
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
)


def load_scan_session(path: str | Path) -> ScanSession:
    """Load a session or raise one error containing all discovered problems."""

    root = Path(path)
    errors: list[str] = []

    if not root.exists():
        raise SessionValidationError([f"session: directory does not exist: {root}"])
    if not root.is_dir():
        raise SessionValidationError([f"session: expected a directory: {root}"])
    if root.suffix != ".vgsession":
        errors.append("session: directory name must end in .vgsession")

    root = root.resolve()
    manifest = _read_json_object(root / "manifest.json", "manifest", errors)
    if manifest is None:
        raise SessionValidationError(errors)

    schema = _required_string(manifest, "schema", "manifest", errors)
    if schema and schema != SCAN_SCHEMA:
        errors.append(
            f"manifest.schema: expected {SCAN_SCHEMA!r}, received {schema!r}"
        )

    schema_version = _required_string(
        manifest, "schema_version", "manifest", errors
    )
    if schema_version and schema_version != SCHEMA_VERSION:
        errors.append(
            "manifest.schema_version: "
            f"expected {SCHEMA_VERSION!r}, received {schema_version!r}"
        )

    session_id = _required_identifier(
        manifest, "session_id", "manifest", errors
    )
    created_at_utc = _required_string(
        manifest, "created_at_utc", "manifest", errors
    )
    if created_at_utc:
        _validate_utc_timestamp(created_at_utc, "manifest.created_at_utc", errors)

    _validate_exact_object(
        manifest.get("timebase"),
        _TIMEBASE,
        "manifest.timebase",
        errors,
    )
    _validate_exact_object(
        manifest.get("coordinate_system"),
        _COORDINATE_SYSTEM,
        "manifest.coordinate_system",
        errors,
    )

    calibration_ref = _required_string(
        manifest, "calibration", "manifest", errors
    )
    calibration_path = _resolve_session_file(
        root, calibration_ref, "manifest.calibration", errors
    )
    calibrations = (
        _load_calibrations(calibration_path, errors)
        if calibration_path is not None
        else {}
    )

    stream_definitions, streams = _load_streams(
        manifest.get("streams"), root, calibrations, errors
    )

    if errors:
        raise SessionValidationError(errors)

    return ScanSession(
        root=root,
        session_id=session_id,
        schema_version=schema_version,
        created_at_utc=created_at_utc,
        calibrations=MappingProxyType(dict(calibrations)),
        stream_definitions=MappingProxyType(dict(stream_definitions)),
        streams=MappingProxyType(dict(streams)),
    )


def _load_calibrations(
    path: Path,
    errors: list[str],
) -> dict[str, CameraCalibration]:
    document = _read_json_object(path, "calibration", errors)
    if document is None:
        return {}

    schema = _required_string(document, "schema", "calibration", errors)
    if schema and schema != CALIBRATION_SCHEMA:
        errors.append(
            f"calibration.schema: expected {CALIBRATION_SCHEMA!r}, "
            f"received {schema!r}"
        )

    version = _required_string(
        document, "schema_version", "calibration", errors
    )
    if version and version != SCHEMA_VERSION:
        errors.append(
            "calibration.schema_version: "
            f"expected {SCHEMA_VERSION!r}, received {version!r}"
        )

    raw_cameras = document.get("cameras")
    if not isinstance(raw_cameras, list) or not raw_cameras:
        errors.append("calibration.cameras: expected a non-empty array")
        return {}

    cameras: dict[str, CameraCalibration] = {}
    for index, raw_camera in enumerate(raw_cameras):
        label = f"calibration.cameras[{index}]"
        if not isinstance(raw_camera, dict):
            errors.append(f"{label}: expected an object")
            continue

        camera_id = _required_identifier(raw_camera, "id", label, errors)
        if camera_id in cameras:
            errors.append(f"{label}.id: duplicate camera id {camera_id!r}")

        model = _required_string(raw_camera, "model", label, errors)
        if model and model != "pinhole":
            errors.append(f"{label}.model: only 'pinhole' is supported in v0.1")

        width = _required_positive_integer(
            raw_camera, "width", label, errors
        )
        height = _required_positive_integer(
            raw_camera, "height", label, errors
        )

        raw_intrinsics = raw_camera.get("intrinsics")
        if not isinstance(raw_intrinsics, dict):
            errors.append(f"{label}.intrinsics: expected an object")
            raw_intrinsics = {}
        fx = _required_number(raw_intrinsics, "fx", f"{label}.intrinsics", errors)
        fy = _required_number(raw_intrinsics, "fy", f"{label}.intrinsics", errors)
        cx = _required_number(raw_intrinsics, "cx", f"{label}.intrinsics", errors)
        cy = _required_number(raw_intrinsics, "cy", f"{label}.intrinsics", errors)

        if fx <= 0:
            errors.append(f"{label}.intrinsics.fx: must be positive")
        if fy <= 0:
            errors.append(f"{label}.intrinsics.fy: must be positive")
        if width > 0 and not 0 <= cx < width:
            errors.append(
                f"{label}.intrinsics.cx: must be within image width [0, {width})"
            )
        if height > 0 and not 0 <= cy < height:
            errors.append(
                f"{label}.intrinsics.cy: must be within image height [0, {height})"
            )

        distortion_model, distortion_coefficients = _load_distortion(
            raw_camera.get("distortion"), label, errors
        )
        t_rig_camera = _validate_rigid_transform(
            raw_camera.get("T_rig_camera"),
            f"{label}.T_rig_camera",
            errors,
        )

        if camera_id and camera_id not in cameras:
            cameras[camera_id] = CameraCalibration(
                id=camera_id,
                model=model,
                width=width,
                height=height,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                distortion_model=distortion_model,
                distortion_coefficients=distortion_coefficients,
                t_rig_camera=t_rig_camera,
            )

    return cameras


def _load_distortion(
    raw_distortion: Any,
    camera_label: str,
    errors: list[str],
) -> tuple[str, tuple[float, ...]]:
    label = f"{camera_label}.distortion"
    if not isinstance(raw_distortion, dict):
        errors.append(f"{label}: expected an object")
        return "", ()

    model = _required_string(raw_distortion, "model", label, errors)
    raw_coefficients = raw_distortion.get("coefficients")
    if not isinstance(raw_coefficients, list):
        errors.append(f"{label}.coefficients: expected an array")
        return model, ()

    coefficients: list[float] = []
    for index, coefficient in enumerate(raw_coefficients):
        value = _as_finite_number(
            coefficient, f"{label}.coefficients[{index}]", errors
        )
        coefficients.append(value)

    if model == "none" and coefficients:
        errors.append(f"{label}.coefficients: must be empty for model 'none'")
    elif model == "opencv-radtan" and len(coefficients) != 5:
        errors.append(
            f"{label}.coefficients: model 'opencv-radtan' requires 5 values"
        )
    elif model not in {"none", "opencv-radtan"}:
        errors.append(
            f"{label}.model: expected 'none' or 'opencv-radtan', "
            f"received {model!r}"
        )

    return model, tuple(coefficients)


def _load_streams(
    raw_streams: Any,
    root: Path,
    calibrations: dict[str, CameraCalibration],
    errors: list[str],
) -> tuple[
    dict[str, StreamDefinition],
    dict[str, tuple[StreamSample, ...]],
]:
    if not isinstance(raw_streams, dict):
        errors.append("manifest.streams: expected an object")
        return {}, {}

    unknown_streams = sorted(set(raw_streams) - set(STREAM_ORDER))
    for name in unknown_streams:
        errors.append(
            f"manifest.streams.{name}: unsupported stream name in v0.1"
        )
    if "rgb" not in raw_streams:
        errors.append("manifest.streams.rgb: required stream is missing")

    definitions: dict[str, StreamDefinition] = {}
    streams: dict[str, tuple[StreamSample, ...]] = {}

    for name in STREAM_ORDER:
        if name not in raw_streams:
            continue

        raw_definition = raw_streams[name]
        label = f"manifest.streams.{name}"
        if not isinstance(raw_definition, dict):
            errors.append(f"{label}: expected an object")
            continue

        kind = _required_string(raw_definition, "kind", label, errors)
        if kind and kind != name:
            errors.append(
                f"{label}.kind: expected {name!r}, received {kind!r}"
            )

        index_ref = _required_string(raw_definition, "index", label, errors)
        index_path = _resolve_session_file(
            root, index_ref, f"{label}.index", errors
        )

        calibration_id: str | None = None
        depth_scale_m: float | None = None
        aligned_to: str | None = None
        transform: str | None = None

        if name in {"rgb", "depth"}:
            calibration_id = _required_identifier(
                raw_definition, "calibration_id", label, errors
            )
            if calibration_id and calibration_id not in calibrations:
                errors.append(
                    f"{label}.calibration_id: unknown camera "
                    f"{calibration_id!r}"
                )

        if name == "depth":
            depth_scale_m = _required_number(
                raw_definition, "depth_scale_m", label, errors
            )
            if depth_scale_m <= 0:
                errors.append(f"{label}.depth_scale_m: must be positive")

            raw_aligned_to = raw_definition.get("aligned_to")
            if raw_aligned_to is not None:
                if not isinstance(raw_aligned_to, str):
                    errors.append(f"{label}.aligned_to: expected a string")
                elif raw_aligned_to != "rgb":
                    errors.append(
                        f"{label}.aligned_to: only 'rgb' is supported in v0.1"
                    )
                else:
                    aligned_to = raw_aligned_to

        if name == "pose":
            transform = _required_string(
                raw_definition, "transform", label, errors
            )
            if transform and transform != "T_world_camera":
                errors.append(
                    f"{label}.transform: expected 'T_world_camera', "
                    f"received {transform!r}"
                )

        definitions[name] = StreamDefinition(
            name=name,
            kind=kind,
            index=index_ref,
            calibration_id=calibration_id,
            depth_scale_m=depth_scale_m,
            aligned_to=aligned_to,
            transform=transform,
        )

        if index_path is not None:
            records = _load_stream_records(name, index_path, root, errors)
            if not records:
                errors.append(f"{label}.index: stream must contain a sample")
            streams[name] = tuple(records)

    return definitions, streams


def _load_stream_records(
    stream_name: str,
    index_path: Path,
    root: Path,
    errors: list[str],
) -> list[StreamSample]:
    try:
        lines = index_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        errors.append(f"streams.{stream_name}: cannot read index: {error}")
        return []

    records: list[StreamSample] = []
    seen_ids: set[str] = set()
    previous_timestamp: int | None = None

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue

        label = f"streams.{stream_name}[line {line_number}]"
        try:
            raw_record = json.loads(line)
        except json.JSONDecodeError as error:
            errors.append(
                f"{label}: invalid JSON at column {error.colno}: {error.msg}"
            )
            continue
        if not isinstance(raw_record, dict):
            errors.append(f"{label}: expected an object")
            continue

        sample_id = _required_identifier(raw_record, "id", label, errors)
        if sample_id in seen_ids:
            errors.append(f"{label}.id: duplicate id {sample_id!r}")
        elif sample_id:
            seen_ids.add(sample_id)

        timestamp = _required_timestamp(raw_record, label, errors)
        if (
            previous_timestamp is not None
            and timestamp <= previous_timestamp
        ):
            errors.append(
                f"{label}.timestamp_ns: must be strictly increasing; "
                f"previous value was {previous_timestamp}"
            )
        previous_timestamp = timestamp

        if stream_name in {"rgb", "depth"}:
            file_ref = _required_string(raw_record, "path", label, errors)
            _resolve_session_file(root, file_ref, f"{label}.path", errors)
        elif stream_name == "imu":
            _validate_vector3(
                raw_record.get("accelerometer_m_s2"),
                f"{label}.accelerometer_m_s2",
                errors,
            )
            _validate_vector3(
                raw_record.get("gyroscope_rad_s"),
                f"{label}.gyroscope_rad_s",
                errors,
            )
        elif stream_name == "pose":
            _validate_rigid_transform(
                raw_record.get("T_world_camera"),
                f"{label}.T_world_camera",
                errors,
            )

        records.append(
            StreamSample(
                stream=stream_name,
                id=sample_id,
                timestamp_ns=timestamp,
                data=_freeze_json(raw_record),
            )
        )

    return records


def _read_json_object(
    path: Path,
    label: str,
    errors: list[str],
) -> dict[str, Any] | None:
    if not path.is_file():
        errors.append(f"{label}: file does not exist: {path}")
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        errors.append(
            f"{label}: invalid JSON at line {error.lineno}, "
            f"column {error.colno}: {error.msg}"
        )
        return None
    except (OSError, UnicodeError) as error:
        errors.append(f"{label}: cannot read file: {error}")
        return None
    if not isinstance(value, dict):
        errors.append(f"{label}: expected a JSON object")
        return None
    return value


def _resolve_session_file(
    root: Path,
    reference: str,
    label: str,
    errors: list[str],
) -> Path | None:
    if not reference:
        return None
    if "\\" in reference:
        errors.append(f"{label}: paths must use POSIX '/' separators")
        return None

    pure_path = PurePosixPath(reference)
    if pure_path.is_absolute() or ".." in pure_path.parts:
        errors.append(f"{label}: path must be relative and remain in the session")
        return None

    candidate = (root / Path(*pure_path.parts)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        errors.append(f"{label}: resolved path escapes the session directory")
        return None

    if not candidate.is_file():
        errors.append(f"{label}: referenced file does not exist: {reference}")
        return None
    return candidate


def _required_string(
    document: dict[str, Any],
    key: str,
    label: str,
    errors: list[str],
) -> str:
    value = document.get(key)
    if not isinstance(value, str) or not value:
        errors.append(f"{label}.{key}: expected a non-empty string")
        return ""
    return value


def _required_identifier(
    document: dict[str, Any],
    key: str,
    label: str,
    errors: list[str],
) -> str:
    value = _required_string(document, key, label, errors)
    if value and not _IDENTIFIER.fullmatch(value):
        errors.append(
            f"{label}.{key}: expected lowercase letters, digits, '.', '_' or '-'"
        )
    return value


def _required_positive_integer(
    document: dict[str, Any],
    key: str,
    label: str,
    errors: list[str],
) -> int:
    value = document.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        errors.append(f"{label}.{key}: expected a positive integer")
        return 0
    return value


def _required_timestamp(
    document: dict[str, Any],
    label: str,
    errors: list[str],
) -> int:
    value = document.get("timestamp_ns")
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value > 2**63 - 1
    ):
        errors.append(
            f"{label}.timestamp_ns: expected an integer in [0, 2^63-1]"
        )
        return 0
    return value


def _required_number(
    document: dict[str, Any],
    key: str,
    label: str,
    errors: list[str],
) -> float:
    return _as_finite_number(document.get(key), f"{label}.{key}", errors)


def _as_finite_number(
    value: Any,
    label: str,
    errors: list[str],
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append(f"{label}: expected a finite number")
        return 0.0
    try:
        converted = float(value)
    except (OverflowError, ValueError):
        errors.append(f"{label}: expected a finite number")
        return 0.0
    if not math.isfinite(converted):
        errors.append(f"{label}: expected a finite number")
        return 0.0
    return converted


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_json(child) for key, child in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(child) for child in value)
    return value


def _validate_vector3(
    value: Any,
    label: str,
    errors: list[str],
) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        errors.append(f"{label}: expected an array of 3 finite numbers")
        return (0.0, 0.0, 0.0)
    values = tuple(
        _as_finite_number(component, f"{label}[{index}]", errors)
        for index, component in enumerate(value)
    )
    return values  # type: ignore[return-value]


def _validate_rigid_transform(
    value: Any,
    label: str,
    errors: list[str],
) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != 16:
        errors.append(f"{label}: expected a row-major array of 16 numbers")
        return _IDENTITY

    matrix = tuple(
        _as_finite_number(component, f"{label}[{index}]", errors)
        for index, component in enumerate(value)
    )
    tolerance = 1e-5

    expected_last_row = (0.0, 0.0, 0.0, 1.0)
    if any(
        abs(actual - expected) > tolerance
        for actual, expected in zip(matrix[12:16], expected_last_row)
    ):
        errors.append(f"{label}: final row must be [0, 0, 0, 1]")

    rotation = (
        (matrix[0], matrix[1], matrix[2]),
        (matrix[4], matrix[5], matrix[6]),
        (matrix[8], matrix[9], matrix[10]),
    )
    rotation_is_orthonormal = True
    for row_index in range(3):
        for column_index in range(3):
            dot = sum(
                rotation[row_index][axis] * rotation[column_index][axis]
                for axis in range(3)
            )
            expected = 1.0 if row_index == column_index else 0.0
            if abs(dot - expected) > tolerance:
                rotation_is_orthonormal = False
                break
        if not rotation_is_orthonormal:
            break
    if not rotation_is_orthonormal:
        errors.append(f"{label}: rotation must be orthonormal")

    determinant = (
        rotation[0][0]
        * (
            rotation[1][1] * rotation[2][2]
            - rotation[1][2] * rotation[2][1]
        )
        - rotation[0][1]
        * (
            rotation[1][0] * rotation[2][2]
            - rotation[1][2] * rotation[2][0]
        )
        + rotation[0][2]
        * (
            rotation[1][0] * rotation[2][1]
            - rotation[1][1] * rotation[2][0]
        )
    )
    if abs(determinant - 1.0) > tolerance:
        errors.append(
            f"{label}: rotation determinant must be +1, got {determinant:.6g}"
        )

    return matrix


def _validate_utc_timestamp(
    value: str,
    label: str,
    errors: list[str],
) -> None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        errors.append(f"{label}: expected an ISO 8601 UTC timestamp")
        return
    if (
        not value.endswith("Z")
        or parsed.tzinfo is None
        or parsed.utcoffset() != timezone.utc.utcoffset(parsed)
    ):
        errors.append(f"{label}: timestamp must use the UTC 'Z' suffix")


def _validate_exact_object(
    actual: Any,
    expected: dict[str, Any],
    label: str,
    errors: list[str],
) -> None:
    if not isinstance(actual, dict):
        errors.append(f"{label}: expected an object")
        return
    _compare_contract_values(actual, expected, label, errors)


def _compare_contract_values(
    actual: dict[str, Any],
    expected: dict[str, Any],
    label: str,
    errors: list[str],
) -> None:
    for key, expected_value in expected.items():
        current_label = f"{label}.{key}"
        if key not in actual:
            errors.append(f"{current_label}: required field is missing")
            continue
        actual_value = actual[key]
        if isinstance(expected_value, dict):
            if not isinstance(actual_value, dict):
                errors.append(f"{current_label}: expected an object")
            else:
                _compare_contract_values(
                    actual_value, expected_value, current_label, errors
                )
        elif actual_value != expected_value:
            errors.append(
                f"{current_label}: expected {expected_value!r}, "
                f"received {actual_value!r}"
            )
