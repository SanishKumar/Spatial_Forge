"""Deterministic zero-crossing surface-point extraction from reference TSDF."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .errors import SurfaceExtractionError
from .tsdf import (
    MAX_REFERENCE_VOXELS,
    REFERENCE_TSDF_SCHEMA,
    REFERENCE_TSDF_SCHEMA_VERSION,
)

_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_INDEX_ORDER = "x-fastest-then-y-then-z"
_TSDF_SIGN = "positive-free-space-negative-behind-surface"
_UNKNOWN_RULE = "weight-zero"
_AXES = (
    ("x", (1, 0, 0)),
    ("y", (0, 1, 0)),
    ("z", (0, 0, 1)),
)


@dataclass(frozen=True, slots=True)
class _TsdfVoxel:
    index: tuple[int, int, int]
    tsdf: float
    weight: int


@dataclass(frozen=True, slots=True)
class _ReferenceTsdf:
    path: Path
    digest_sha256: str
    session_id: str
    replay_digest_sha256: str
    origin_world_m: tuple[float, float, float]
    dimensions: tuple[int, int, int]
    voxel_size_m: float
    total_voxels: int
    voxels: tuple[_TsdfVoxel, ...]


@dataclass(frozen=True, slots=True)
class SurfacePointReport:
    session_id: str
    input: Path
    output: Path
    total_voxels: int
    observed_voxels: int
    observed_edges: int
    exact_zero_points: int
    crossing_x_points: int
    crossing_y_points: int
    crossing_z_points: int
    points_written: int
    source_tsdf_digest_sha256: str
    output_digest_sha256: str


def extract_surface_points(
    input: str | Path,
    output: str | Path,
) -> SurfacePointReport:
    """Extract exact-zero centers and strict TSDF sign-changing edge points."""

    volume = _load_reference_tsdf(input)
    output_path = Path(output).resolve()
    if output_path.suffix.lower() != ".ply":
        raise SurfaceExtractionError("output filename must end in .ply")
    if output_path.exists():
        raise SurfaceExtractionError(f"output already exists: {output_path}")

    voxel_by_index = {voxel.index: voxel for voxel in volume.voxels}
    (
        observed_edges,
        exact_zero_points,
        crossing_counts,
    ) = _count_surface_points(volume, voxel_by_index)
    points_written = exact_zero_points + sum(crossing_counts)
    if points_written == 0:
        raise SurfaceExtractionError(
            "TSDF volume contains no exact-zero voxel or strict sign-changing edge"
        )

    output_digest = _write_surface_ply(
        volume,
        voxel_by_index,
        output_path,
        points_written,
    )
    return SurfacePointReport(
        session_id=volume.session_id,
        input=volume.path,
        output=output_path,
        total_voxels=volume.total_voxels,
        observed_voxels=len(volume.voxels),
        observed_edges=observed_edges,
        exact_zero_points=exact_zero_points,
        crossing_x_points=crossing_counts[0],
        crossing_y_points=crossing_counts[1],
        crossing_z_points=crossing_counts[2],
        points_written=points_written,
        source_tsdf_digest_sha256=volume.digest_sha256,
        output_digest_sha256=output_digest,
    )


def _load_reference_tsdf(path: str | Path) -> _ReferenceTsdf:
    input_path = Path(path).resolve()
    if input_path.suffix.lower() != ".sftsdf":
        raise SurfaceExtractionError("input filename must end in .sftsdf")
    if not input_path.is_file():
        raise SurfaceExtractionError(f"TSDF input file does not exist: {input_path}")

    try:
        encoded = input_path.read_bytes()
    except OSError as error:
        raise SurfaceExtractionError(f"cannot read TSDF input: {error}") from error
    digest = hashlib.sha256(encoded).hexdigest()
    try:
        text = encoded.decode("utf-8")
    except UnicodeError as error:
        raise SurfaceExtractionError("TSDF input must be UTF-8 JSON") from error

    try:
        document = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except _DuplicateJsonKey as error:
        raise SurfaceExtractionError(
            f"TSDF input contains duplicate JSON key {error.key!r}"
        ) from error
    except RecursionError as error:
        raise SurfaceExtractionError(
            "TSDF input JSON is nested too deeply"
        ) from error
    except (json.JSONDecodeError, ValueError) as error:
        raise SurfaceExtractionError(f"TSDF input is invalid JSON: {error}") from error

    root = _require_object(document, "TSDF")
    _reject_unknown_fields(
        root,
        {
            "schema",
            "schema_version",
            "session_id",
            "replay_digest_sha256",
            "volume",
            "integration",
            "voxels",
        },
        "TSDF",
    )
    schema = _require_string(root, "schema", "TSDF")
    if schema != REFERENCE_TSDF_SCHEMA:
        raise SurfaceExtractionError(
            f"TSDF.schema: expected {REFERENCE_TSDF_SCHEMA!r}, received {schema!r}"
        )
    version = _require_string(root, "schema_version", "TSDF")
    if version != REFERENCE_TSDF_SCHEMA_VERSION:
        raise SurfaceExtractionError(
            "TSDF.schema_version: "
            f"expected {REFERENCE_TSDF_SCHEMA_VERSION!r}, received {version!r}"
        )
    session_id = _require_string(root, "session_id", "TSDF")
    if not _IDENTIFIER.fullmatch(session_id):
        raise SurfaceExtractionError("TSDF.session_id: invalid identifier")
    replay_digest = _require_string(root, "replay_digest_sha256", "TSDF")
    if not _SHA256.fullmatch(replay_digest):
        raise SurfaceExtractionError(
            "TSDF.replay_digest_sha256: expected 64 lowercase hexadecimal characters"
        )

    raw_volume = _require_object(
        _require_field(root, "volume", "TSDF"),
        "TSDF.volume",
    )
    _reject_unknown_fields(
        raw_volume,
        {
            "origin_world_m",
            "dimensions_xyz",
            "voxel_size_m",
            "truncation_m",
            "index_order",
            "tsdf_sign",
            "unknown_rule",
        },
        "TSDF.volume",
    )
    origin = _require_number_triplet(
        _require_field(raw_volume, "origin_world_m", "TSDF.volume"),
        "TSDF.volume.origin_world_m",
    )
    dimensions = _require_integer_triplet(
        _require_field(raw_volume, "dimensions_xyz", "TSDF.volume"),
        "TSDF.volume.dimensions_xyz",
    )
    voxel_size = _require_positive_number(
        _require_field(raw_volume, "voxel_size_m", "TSDF.volume"),
        "TSDF.volume.voxel_size_m",
    )
    truncation = _require_positive_number(
        _require_field(raw_volume, "truncation_m", "TSDF.volume"),
        "TSDF.volume.truncation_m",
    )
    if truncation < voxel_size:
        raise SurfaceExtractionError(
            "TSDF.volume.truncation_m: must be at least voxel_size_m"
        )
    _require_exact_string(
        raw_volume,
        "index_order",
        _INDEX_ORDER,
        "TSDF.volume",
    )
    _require_exact_string(
        raw_volume,
        "tsdf_sign",
        _TSDF_SIGN,
        "TSDF.volume",
    )
    _require_exact_string(
        raw_volume,
        "unknown_rule",
        _UNKNOWN_RULE,
        "TSDF.volume",
    )

    total_voxels = math.prod(dimensions)
    if total_voxels > MAX_REFERENCE_VOXELS:
        raise SurfaceExtractionError(
            f"TSDF volume has {total_voxels} voxels; "
            f"maximum supported is {MAX_REFERENCE_VOXELS}"
        )
    _validate_volume_extent(origin, dimensions, voxel_size)

    raw_voxels = _require_field(root, "voxels", "TSDF")
    if not isinstance(raw_voxels, list):
        raise SurfaceExtractionError("TSDF.voxels: expected an array")
    voxels = _load_voxels(raw_voxels, dimensions)
    if not voxels:
        raise SurfaceExtractionError("TSDF.voxels: expected at least one voxel")

    raw_integration = _require_object(
        _require_field(root, "integration", "TSDF"),
        "TSDF.integration",
    )
    _validate_integration_metadata(
        raw_integration,
        total_voxels,
        voxels,
    )
    return _ReferenceTsdf(
        path=input_path,
        digest_sha256=digest,
        session_id=session_id,
        replay_digest_sha256=replay_digest,
        origin_world_m=origin,
        dimensions=dimensions,
        voxel_size_m=voxel_size,
        total_voxels=total_voxels,
        voxels=voxels,
    )


def _load_voxels(
    raw_voxels: list[Any],
    dimensions: tuple[int, int, int],
) -> tuple[_TsdfVoxel, ...]:
    nx, ny, nz = dimensions
    voxels: list[_TsdfVoxel] = []
    previous_flat_index = -1
    for position, raw_voxel in enumerate(raw_voxels):
        label = f"TSDF.voxels[{position}]"
        voxel = _require_object(raw_voxel, label)
        _reject_unknown_fields(voxel, {"index", "tsdf", "weight"}, label)
        index = _require_integer_triplet(
            _require_field(voxel, "index", label),
            f"{label}.index",
            allow_zero=True,
        )
        if not (
            index[0] < nx
            and index[1] < ny
            and index[2] < nz
        ):
            raise SurfaceExtractionError(f"{label}.index: outside volume dimensions")
        flat_index = (index[2] * ny + index[1]) * nx + index[0]
        if flat_index <= previous_flat_index:
            raise SurfaceExtractionError(
                "TSDF.voxels: indices must be unique and strictly "
                "x-fastest ordered"
            )
        previous_flat_index = flat_index
        tsdf = _require_finite_number(
            _require_field(voxel, "tsdf", label),
            f"{label}.tsdf",
        )
        if not -1.0 <= tsdf <= 1.0:
            raise SurfaceExtractionError(f"{label}.tsdf: expected value in [-1, 1]")
        weight = _require_positive_integer(
            _require_field(voxel, "weight", label),
            f"{label}.weight",
        )
        voxels.append(
            _TsdfVoxel(
                index=index,
                tsdf=tsdf,
                weight=weight,
            )
        )
    return tuple(voxels)


def _validate_integration_metadata(
    integration: dict[str, Any],
    total_voxels: int,
    voxels: tuple[_TsdfVoxel, ...],
) -> None:
    _require_positive_integer(
        _require_field(integration, "frame_stride", "TSDF.integration"),
        "TSDF.integration.frame_stride",
    )

    counter_names = (
        "total_observations",
        "selected_observations",
        "integrated_frames",
        "skipped_missing_depth",
        "skipped_missing_pose",
        "invalid_depth_pixels",
        "total_voxels",
        "observed_voxels",
        "unknown_voxels",
        "fused_voxels",
        "voxel_updates",
        "max_weight",
    )
    counters = {
        name: _require_nonnegative_integer(
            _require_field(integration, name, "TSDF.integration"),
            f"TSDF.integration.{name}",
        )
        for name in counter_names
    }
    _reject_unknown_fields(
        integration,
        {"frame_stride", *counter_names},
        "TSDF.integration",
    )
    for name in (
        "total_observations",
        "selected_observations",
        "integrated_frames",
    ):
        if counters[name] < 1:
            raise SurfaceExtractionError(
                f"TSDF.integration.{name}: expected a positive integer"
            )
    if counters["selected_observations"] > counters["total_observations"]:
        raise SurfaceExtractionError(
            "TSDF.integration: selected observations exceed total observations"
        )
    if counters["integrated_frames"] > counters["selected_observations"]:
        raise SurfaceExtractionError(
            "TSDF.integration: integrated frames exceed selected observations"
        )
    frame_stride = integration["frame_stride"]
    expected_selected = (
        (counters["total_observations"] - 1) // frame_stride
    ) + 1
    if counters["selected_observations"] != expected_selected:
        raise SurfaceExtractionError(
            "TSDF.integration.selected_observations: "
            f"expected {expected_selected} for frame_stride {frame_stride}, "
            f"received {counters['selected_observations']}"
        )

    missing_depth = counters["skipped_missing_depth"]
    missing_pose = counters["skipped_missing_pose"]
    skipped_frames = (
        counters["selected_observations"] - counters["integrated_frames"]
    )
    if not (
        max(missing_depth, missing_pose)
        <= skipped_frames
        <= missing_depth + missing_pose
    ):
        raise SurfaceExtractionError(
            "TSDF.integration: missing-depth/pose counts are inconsistent "
            "with integrated frames"
        )

    expected = {
        "total_voxels": total_voxels,
        "observed_voxels": len(voxels),
        "unknown_voxels": total_voxels - len(voxels),
        "fused_voxels": sum(voxel.weight > 1 for voxel in voxels),
        "voxel_updates": sum(voxel.weight for voxel in voxels),
        "max_weight": max(voxel.weight for voxel in voxels),
    }
    for name, expected_value in expected.items():
        if counters[name] != expected_value:
            raise SurfaceExtractionError(
                f"TSDF.integration.{name}: expected {expected_value}, "
                f"received {counters[name]}"
            )
    if expected["max_weight"] > counters["integrated_frames"]:
        raise SurfaceExtractionError(
            "TSDF.integration.max_weight: cannot exceed integrated_frames"
        )


def _count_surface_points(
    volume: _ReferenceTsdf,
    voxel_by_index: dict[tuple[int, int, int], _TsdfVoxel],
) -> tuple[int, int, tuple[int, int, int]]:
    exact_zero_points = sum(voxel.tsdf == 0.0 for voxel in volume.voxels)
    observed_edges = 0
    crossing_counts: list[int] = []
    for _, direction in _AXES:
        axis_crossings = 0
        for voxel in volume.voxels:
            neighbor = voxel_by_index.get(_offset_index(voxel.index, direction))
            if neighbor is None:
                continue
            observed_edges += 1
            if _strict_sign_change(voxel.tsdf, neighbor.tsdf):
                axis_crossings += 1
        crossing_counts.append(axis_crossings)
    return (
        observed_edges,
        exact_zero_points,
        (crossing_counts[0], crossing_counts[1], crossing_counts[2]),
    )


def _iter_surface_points(
    volume: _ReferenceTsdf,
    voxel_by_index: dict[tuple[int, int, int], _TsdfVoxel],
) -> Iterable[tuple[float, float, float]]:
    for voxel in volume.voxels:
        if voxel.tsdf == 0.0:
            yield _voxel_center(volume, voxel.index)

    for _, direction in _AXES:
        for voxel in volume.voxels:
            neighbor = voxel_by_index.get(_offset_index(voxel.index, direction))
            if neighbor is None:
                continue
            if _strict_sign_change(voxel.tsdf, neighbor.tsdf):
                yield _interpolate_crossing(volume, voxel, neighbor)


def _interpolate_crossing(
    volume: _ReferenceTsdf,
    first: _TsdfVoxel,
    second: _TsdfVoxel,
) -> tuple[float, float, float]:
    alpha = first.tsdf / (first.tsdf - second.tsdf)
    first_center = _voxel_center(volume, first.index)
    second_center = _voxel_center(volume, second.index)
    point = tuple(
        first_value + alpha * (second_value - first_value)
        for first_value, second_value in zip(
            first_center,
            second_center,
            strict=True,
        )
    )
    if not all(math.isfinite(value) for value in point):
        raise SurfaceExtractionError(
            "surface interpolation produced a non-finite coordinate"
        )
    return point  # type: ignore[return-value]


def _voxel_center(
    volume: _ReferenceTsdf,
    index: tuple[int, int, int],
) -> tuple[float, float, float]:
    center = tuple(
        axis_origin + (axis_index + 0.5) * volume.voxel_size_m
        for axis_origin, axis_index in zip(
            volume.origin_world_m,
            index,
            strict=True,
        )
    )
    return center  # type: ignore[return-value]


def _offset_index(
    index: tuple[int, int, int],
    direction: tuple[int, int, int],
) -> tuple[int, int, int]:
    return (
        index[0] + direction[0],
        index[1] + direction[1],
        index[2] + direction[2],
    )


def _strict_sign_change(first: float, second: float) -> bool:
    return (first < 0.0 < second) or (second < 0.0 < first)


def _write_surface_ply(
    volume: _ReferenceTsdf,
    voxel_by_index: dict[tuple[int, int, int], _TsdfVoxel],
    output_path: Path,
    points_written: int,
) -> str:
    temporary_path: Path | None = None
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output_path.stem}-",
            suffix=".ply",
            dir=output_path.parent,
        )
        temporary_path = Path(temporary_name)
        header = "\n".join(
            [
                "ply",
                "format ascii 1.0",
                f"comment spatialforge_session {volume.session_id}",
                "comment spatialforge_replay_sha256 "
                f"{volume.replay_digest_sha256}",
                "comment spatialforge_source_tsdf_sha256 "
                f"{volume.digest_sha256}",
                "comment coordinates metres world_x_forward world_y_left "
                "world_z_up",
                "comment spatialforge_surface_rule "
                "exact_zero_centers_then_strict_edges_x_y_z",
                f"element vertex {points_written}",
                "property double x",
                "property double y",
                "property double z",
                "end_header",
                "",
            ]
        ).encode("ascii")
        with open(descriptor, "wb", closefd=True) as output_file:
            output_file.write(header)
            lines: list[bytes] = []
            for point in _iter_surface_points(volume, voxel_by_index):
                lines.append(
                    (
                        f"{_format_coordinate(point[0])} "
                        f"{_format_coordinate(point[1])} "
                        f"{_format_coordinate(point[2])}\n"
                    ).encode("ascii")
                )
                if len(lines) >= 8192:
                    output_file.writelines(lines)
                    lines.clear()
            if lines:
                output_file.writelines(lines)

        output_digest = _sha256_file(temporary_path)
        try:
            os.link(temporary_path, output_path)
        except FileExistsError as error:
            raise SurfaceExtractionError(
                "output appeared while extracting; refusing to overwrite: "
                f"{output_path}"
            ) from error
        except OSError as error:
            raise SurfaceExtractionError(
                f"cannot publish surface output without overwriting: {error}"
            ) from error
        return output_digest
    except SurfaceExtractionError:
        raise
    except OSError as error:
        raise SurfaceExtractionError(f"cannot write surface output: {error}") from error
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as input_file:
            while chunk := input_file.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise SurfaceExtractionError(
            f"cannot hash surface output: {error}"
        ) from error
    return digest.hexdigest()


def _format_coordinate(value: float) -> str:
    if abs(value) < 0.5e-9:
        value = 0.0
    return f"{value:.9f}"


def _require_field(
    document: dict[str, Any],
    key: str,
    label: str,
) -> Any:
    if key not in document:
        raise SurfaceExtractionError(f"{label}.{key}: required field is missing")
    return document[key]


def _require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SurfaceExtractionError(f"{label}: expected an object")
    return value


def _reject_unknown_fields(
    document: dict[str, Any],
    expected: set[str],
    label: str,
) -> None:
    unknown = sorted(set(document) - expected)
    if unknown:
        raise SurfaceExtractionError(
            f"{label}: unexpected field {unknown[0]!r}"
        )


def _require_string(
    document: dict[str, Any],
    key: str,
    label: str,
) -> str:
    value = _require_field(document, key, label)
    if not isinstance(value, str) or not value:
        raise SurfaceExtractionError(f"{label}.{key}: expected a non-empty string")
    return value


def _require_exact_string(
    document: dict[str, Any],
    key: str,
    expected: str,
    label: str,
) -> None:
    value = _require_string(document, key, label)
    if value != expected:
        raise SurfaceExtractionError(
            f"{label}.{key}: expected {expected!r}, received {value!r}"
        )


def _require_number_triplet(
    value: Any,
    label: str,
) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise SurfaceExtractionError(f"{label}: expected 3 finite numbers")
    converted = tuple(
        _require_finite_number(component, f"{label}[{index}]")
        for index, component in enumerate(value)
    )
    return converted  # type: ignore[return-value]


def _require_integer_triplet(
    value: Any,
    label: str,
    *,
    allow_zero: bool = False,
) -> tuple[int, int, int]:
    if not isinstance(value, list) or len(value) != 3:
        requirement = "non-negative" if allow_zero else "positive"
        raise SurfaceExtractionError(
            f"{label}: expected 3 {requirement} integers"
        )
    converted: list[int] = []
    for index, component in enumerate(value):
        if isinstance(component, bool) or not isinstance(component, int):
            raise SurfaceExtractionError(
                f"{label}[{index}]: expected an integer"
            )
        if component < 0 or (component == 0 and not allow_zero):
            requirement = "non-negative" if allow_zero else "positive"
            raise SurfaceExtractionError(
                f"{label}[{index}]: expected a {requirement} integer"
            )
        converted.append(component)
    return (converted[0], converted[1], converted[2])


def _require_finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SurfaceExtractionError(f"{label}: expected a finite number")
    try:
        converted = float(value)
    except (OverflowError, ValueError) as error:
        raise SurfaceExtractionError(f"{label}: expected a finite number") from error
    if not math.isfinite(converted):
        raise SurfaceExtractionError(f"{label}: expected a finite number")
    return converted


def _require_positive_number(value: Any, label: str) -> float:
    converted = _require_finite_number(value, label)
    if converted <= 0:
        raise SurfaceExtractionError(f"{label}: expected a positive number")
    return converted


def _require_nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SurfaceExtractionError(f"{label}: expected a non-negative integer")
    return value


def _require_positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SurfaceExtractionError(f"{label}: expected a positive integer")
    return value


def _validate_volume_extent(
    origin: tuple[float, float, float],
    dimensions: tuple[int, int, int],
    voxel_size_m: float,
) -> None:
    for axis_origin, axis_dimension in zip(origin, dimensions, strict=True):
        extent = axis_origin + axis_dimension * voxel_size_m
        if not math.isfinite(extent):
            raise SurfaceExtractionError("TSDF.volume: extent must remain finite")


class _DuplicateJsonKey(ValueError):
    def __init__(self, key: str) -> None:
        self.key = key
        super().__init__(key)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKey(key)
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-standard constant {value!r} is not allowed")
