"""Deterministic fixed-bounds reference TSDF integration."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .errors import PointCloudError, TsdfError
from .model import CameraCalibration, Observation, ScanSession
from .point_cloud import (
    _read_depth,
    _sample_path,
    _validate_reconstruction_contract,
)
from .replay import replay_session

REFERENCE_TSDF_SCHEMA = "spatialforge.reference-tsdf"
REFERENCE_TSDF_SCHEMA_VERSION = "0.1.0"
MAX_REFERENCE_VOXELS = 1_000_000
_INTEGRATION_CHUNK_VOXELS = 131_072


@dataclass(frozen=True, slots=True)
class TsdfReport:
    session_id: str
    output: Path
    total_observations: int
    selected_observations: int
    integrated_frames: int
    skipped_missing_depth: int
    skipped_missing_pose: int
    invalid_depth_pixels: int
    total_voxels: int
    observed_voxels: int
    fused_voxels: int
    voxel_updates: int
    max_weight: int
    replay_digest_sha256: str
    output_digest_sha256: str


def integrate_tsdf(
    session: ScanSession,
    output: str | Path,
    *,
    origin_world_m: Sequence[float],
    dimensions: Sequence[int],
    voxel_size_m: float,
    truncation_m: float,
    frame_stride: int = 1,
) -> TsdfReport:
    """Integrate known-pose depth into a bounded projective TSDF volume."""

    origin = _validate_origin(origin_world_m)
    volume_dimensions = _validate_dimensions(dimensions)
    voxel_size = _validate_positive_number(voxel_size_m, "voxel_size_m")
    truncation = _validate_positive_number(truncation_m, "truncation_m")
    if truncation < voxel_size:
        raise TsdfError("truncation_m must be greater than or equal to voxel_size_m")
    _validate_positive_integer(frame_stride, "frame_stride")
    total_voxels = math.prod(volume_dimensions)
    if total_voxels > MAX_REFERENCE_VOXELS:
        raise TsdfError(
            f"reference TSDF volume has {total_voxels} voxels; "
            f"maximum is {MAX_REFERENCE_VOXELS}"
        )
    _validate_finite_volume_extent(origin, volume_dimensions, voxel_size)

    output_path = Path(output).resolve()
    if output_path.suffix.lower() != ".sftsdf":
        raise TsdfError("output filename must end in .sftsdf")
    if output_path.exists():
        raise TsdfError(f"output already exists: {output_path}")

    try:
        camera, depth_scale_m = _validate_reconstruction_contract(session)
    except PointCloudError as error:
        raise TsdfError(str(error)) from error

    replay = replay_session(session)
    selected = tuple(
        observation
        for observation in replay.observations
        if observation.sequence % frame_stride == 0
    )
    tsdf_sums = np.zeros(total_voxels, dtype=np.float64)
    weights = np.zeros(total_voxels, dtype=np.uint32)

    integrated_frames = 0
    skipped_missing_depth = 0
    skipped_missing_pose = 0
    invalid_depth_pixels = 0
    voxel_updates = 0

    for observation in selected:
        missing_depth = observation.depth is None
        missing_pose = observation.pose is None
        if missing_depth:
            skipped_missing_depth += 1
        if missing_pose:
            skipped_missing_pose += 1
        if missing_depth or missing_pose:
            continue

        frame_invalid, frame_updates = _integrate_observation(
            session,
            observation,
            camera,
            depth_scale_m,
            origin,
            volume_dimensions,
            voxel_size,
            truncation,
            tsdf_sums,
            weights,
        )
        integrated_frames += 1
        invalid_depth_pixels += frame_invalid
        voxel_updates += frame_updates

    if integrated_frames == 0:
        raise TsdfError(
            "no selected RGB observation has both exact depth and pose"
        )

    observed_indices = np.flatnonzero(weights)
    observed_voxels = int(observed_indices.size)
    if observed_voxels == 0:
        raise TsdfError("selected frames do not observe any voxel in the volume")

    observed_weights = weights[observed_indices]
    observed_tsdf = tsdf_sums[observed_indices] / observed_weights
    if not np.all(np.isfinite(observed_tsdf)):
        raise TsdfError("TSDF integration produced a non-finite value")
    if np.any(observed_tsdf < -1.0) or np.any(observed_tsdf > 1.0):
        raise TsdfError("TSDF integration produced a value outside [-1, 1]")

    fused_voxels = int(np.count_nonzero(observed_weights > 1))
    max_weight = int(np.max(observed_weights))
    document = _build_diagnostic_document(
        session,
        replay.digest_sha256,
        origin,
        volume_dimensions,
        voxel_size,
        truncation,
        frame_stride,
        len(replay.observations),
        len(selected),
        integrated_frames,
        skipped_missing_depth,
        skipped_missing_pose,
        invalid_depth_pixels,
        total_voxels,
        observed_indices,
        observed_tsdf,
        observed_weights,
        voxel_updates,
        fused_voxels,
        max_weight,
    )
    encoded = (
        json.dumps(
            document,
            ensure_ascii=True,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")
    output_digest = hashlib.sha256(encoded).hexdigest()
    _write_output_without_overwrite(output_path, encoded)

    return TsdfReport(
        session_id=session.session_id,
        output=output_path,
        total_observations=len(replay.observations),
        selected_observations=len(selected),
        integrated_frames=integrated_frames,
        skipped_missing_depth=skipped_missing_depth,
        skipped_missing_pose=skipped_missing_pose,
        invalid_depth_pixels=invalid_depth_pixels,
        total_voxels=total_voxels,
        observed_voxels=observed_voxels,
        fused_voxels=fused_voxels,
        voxel_updates=voxel_updates,
        max_weight=max_weight,
        replay_digest_sha256=replay.digest_sha256,
        output_digest_sha256=output_digest,
    )


def _integrate_observation(
    session: ScanSession,
    observation: Observation,
    camera: CameraCalibration,
    depth_scale_m: float,
    origin: tuple[float, float, float],
    dimensions: tuple[int, int, int],
    voxel_size_m: float,
    truncation_m: float,
    tsdf_sums: np.ndarray,
    weights: np.ndarray,
) -> tuple[int, int]:
    if observation.depth is None or observation.pose is None:
        raise AssertionError("caller must filter incomplete observations")

    try:
        depth_path = _sample_path(session, observation.depth.data, "depth")
        depth_values = np.asarray(
            _read_depth(depth_path, camera.width, camera.height),
            dtype=np.float64,
        ).reshape((camera.height, camera.width))
    except PointCloudError as error:
        raise TsdfError(str(error)) from error

    with np.errstate(over="ignore", invalid="ignore"):
        depth_metres = depth_values * depth_scale_m
    valid_depth = (depth_metres > 0) & np.isfinite(depth_metres)
    invalid_depth_pixels = int(np.count_nonzero(~valid_depth))
    transform = tuple(observation.pose.data["T_world_camera"])
    updates = 0
    for start in range(0, tsdf_sums.size, _INTEGRATION_CHUNK_VOXELS):
        stop = min(start + _INTEGRATION_CHUNK_VOXELS, tsdf_sums.size)
        updates += _integrate_voxel_chunk(
            start,
            stop,
            depth_metres,
            camera,
            transform,
            origin,
            dimensions,
            voxel_size_m,
            truncation_m,
            tsdf_sums,
            weights,
        )
    return invalid_depth_pixels, updates


def _integrate_voxel_chunk(
    start: int,
    stop: int,
    depth_metres: np.ndarray,
    camera: CameraCalibration,
    transform: Sequence[float],
    origin: tuple[float, float, float],
    dimensions: tuple[int, int, int],
    voxel_size_m: float,
    truncation_m: float,
    tsdf_sums: np.ndarray,
    weights: np.ndarray,
) -> int:
    nx, ny, _ = dimensions
    flat_indices = np.arange(start, stop, dtype=np.int64)
    x_indices = flat_indices % nx
    yz_indices = flat_indices // nx
    y_indices = yz_indices % ny
    z_indices = yz_indices // ny

    x_world = origin[0] + (x_indices + 0.5) * voxel_size_m
    y_world = origin[1] + (y_indices + 0.5) * voxel_size_m
    z_world = origin[2] + (z_indices + 0.5) * voxel_size_m
    delta_x = x_world - transform[3]
    delta_y = y_world - transform[7]
    delta_z = z_world - transform[11]

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        x_camera = (
            transform[0] * delta_x
            + transform[4] * delta_y
            + transform[8] * delta_z
        )
        y_camera = (
            transform[1] * delta_x
            + transform[5] * delta_y
            + transform[9] * delta_z
        )
        z_camera = (
            transform[2] * delta_x
            + transform[6] * delta_y
            + transform[10] * delta_z
        )

    candidate_offsets = np.flatnonzero(
        (z_camera > 0)
        & np.isfinite(x_camera)
        & np.isfinite(y_camera)
        & np.isfinite(z_camera)
    )
    if candidate_offsets.size == 0:
        return 0

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        projected_u = (
            camera.fx
            * x_camera[candidate_offsets]
            / z_camera[candidate_offsets]
            + camera.cx
        )
        projected_v = (
            camera.fy
            * y_camera[candidate_offsets]
            / z_camera[candidate_offsets]
            + camera.cy
        )
    finite_projection = np.isfinite(projected_u) & np.isfinite(projected_v)
    candidate_offsets = candidate_offsets[finite_projection]
    projected_u = projected_u[finite_projection]
    projected_v = projected_v[finite_projection]
    if candidate_offsets.size == 0:
        return 0

    projection_in_range = (
        (projected_u >= -0.5)
        & (projected_u < camera.width - 0.5)
        & (projected_v >= -0.5)
        & (projected_v < camera.height - 0.5)
    )
    candidate_offsets = candidate_offsets[projection_in_range]
    projected_u = projected_u[projection_in_range]
    projected_v = projected_v[projection_in_range]
    if candidate_offsets.size == 0:
        return 0

    pixel_u = np.floor(projected_u + 0.5).astype(np.int64)
    pixel_v = np.floor(projected_v + 0.5).astype(np.int64)
    inside_image = (
        (pixel_u >= 0)
        & (pixel_u < camera.width)
        & (pixel_v >= 0)
        & (pixel_v < camera.height)
    )
    candidate_offsets = candidate_offsets[inside_image]
    pixel_u = pixel_u[inside_image]
    pixel_v = pixel_v[inside_image]
    if candidate_offsets.size == 0:
        return 0

    measured_depth = depth_metres[pixel_v, pixel_u]
    signed_distance = measured_depth - z_camera[candidate_offsets]
    valid_measurement = (
        (measured_depth > 0)
        & np.isfinite(measured_depth)
        & np.isfinite(signed_distance)
        & (signed_distance >= -truncation_m)
    )
    candidate_offsets = candidate_offsets[valid_measurement]
    signed_distance = signed_distance[valid_measurement]
    if candidate_offsets.size == 0:
        return 0

    observed_tsdf = np.clip(
        signed_distance / truncation_m,
        -1.0,
        1.0,
    )
    observed_flat_indices = flat_indices[candidate_offsets]
    tsdf_sums[observed_flat_indices] += observed_tsdf
    weights[observed_flat_indices] += 1
    return int(observed_flat_indices.size)


def _build_diagnostic_document(
    session: ScanSession,
    replay_digest: str,
    origin: tuple[float, float, float],
    dimensions: tuple[int, int, int],
    voxel_size_m: float,
    truncation_m: float,
    frame_stride: int,
    total_observations: int,
    selected_observations: int,
    integrated_frames: int,
    skipped_missing_depth: int,
    skipped_missing_pose: int,
    invalid_depth_pixels: int,
    total_voxels: int,
    observed_indices: np.ndarray,
    observed_tsdf: np.ndarray,
    observed_weights: np.ndarray,
    voxel_updates: int,
    fused_voxels: int,
    max_weight: int,
) -> dict[str, object]:
    nx, ny, _ = dimensions
    voxels: list[dict[str, object]] = []
    for flat_index, tsdf_value, weight in zip(
        observed_indices.tolist(),
        observed_tsdf.tolist(),
        observed_weights.tolist(),
        strict=True,
    ):
        x_index = flat_index % nx
        yz_index = flat_index // nx
        y_index = yz_index % ny
        z_index = yz_index // ny
        voxels.append(
            {
                "index": [x_index, y_index, z_index],
                "tsdf": _canonical_float(tsdf_value),
                "weight": int(weight),
            }
        )

    return {
        "schema": REFERENCE_TSDF_SCHEMA,
        "schema_version": REFERENCE_TSDF_SCHEMA_VERSION,
        "session_id": session.session_id,
        "replay_digest_sha256": replay_digest,
        "volume": {
            "origin_world_m": [_canonical_float(value) for value in origin],
            "dimensions_xyz": list(dimensions),
            "voxel_size_m": _canonical_float(voxel_size_m),
            "truncation_m": _canonical_float(truncation_m),
            "index_order": "x-fastest-then-y-then-z",
            "tsdf_sign": "positive-free-space-negative-behind-surface",
            "unknown_rule": "weight-zero",
        },
        "integration": {
            "frame_stride": frame_stride,
            "total_observations": total_observations,
            "selected_observations": selected_observations,
            "integrated_frames": integrated_frames,
            "skipped_missing_depth": skipped_missing_depth,
            "skipped_missing_pose": skipped_missing_pose,
            "invalid_depth_pixels": invalid_depth_pixels,
            "total_voxels": total_voxels,
            "observed_voxels": len(voxels),
            "unknown_voxels": total_voxels - len(voxels),
            "fused_voxels": fused_voxels,
            "voxel_updates": voxel_updates,
            "max_weight": max_weight,
        },
        "voxels": voxels,
    }


def _validate_origin(value: Sequence[float]) -> tuple[float, float, float]:
    try:
        has_three_components = len(value) == 3
    except TypeError:
        has_three_components = False
    if isinstance(value, (str, bytes)) or not has_three_components:
        raise TsdfError("origin_world_m must contain exactly 3 finite numbers")
    converted: list[float] = []
    for component in value:
        if isinstance(component, bool) or not isinstance(component, (int, float)):
            raise TsdfError("origin_world_m must contain exactly 3 finite numbers")
        try:
            number = float(component)
        except (OverflowError, ValueError) as error:
            raise TsdfError(
                "origin_world_m must contain exactly 3 finite numbers"
            ) from error
        if not math.isfinite(number):
            raise TsdfError("origin_world_m must contain exactly 3 finite numbers")
        converted.append(number)
    return (converted[0], converted[1], converted[2])


def _validate_dimensions(value: Sequence[int]) -> tuple[int, int, int]:
    try:
        has_three_components = len(value) == 3
    except TypeError:
        has_three_components = False
    if isinstance(value, (str, bytes)) or not has_three_components:
        raise TsdfError("dimensions must contain exactly 3 positive integers")
    converted: list[int] = []
    for component in value:
        if isinstance(component, bool) or not isinstance(component, int):
            raise TsdfError("dimensions must contain exactly 3 positive integers")
        if component < 1:
            raise TsdfError("dimensions must contain exactly 3 positive integers")
        converted.append(component)
    return (converted[0], converted[1], converted[2])


def _validate_positive_number(value: float, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TsdfError(f"{label} must be a finite positive number")
    try:
        converted = float(value)
    except (OverflowError, ValueError) as error:
        raise TsdfError(f"{label} must be a finite positive number") from error
    if not math.isfinite(converted) or converted <= 0:
        raise TsdfError(f"{label} must be a finite positive number")
    return converted


def _validate_positive_integer(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise TsdfError(f"{label} must be a positive integer")


def _validate_finite_volume_extent(
    origin: tuple[float, float, float],
    dimensions: tuple[int, int, int],
    voxel_size_m: float,
) -> None:
    for axis_origin, axis_dimension in zip(origin, dimensions, strict=True):
        extent = axis_origin + axis_dimension * voxel_size_m
        if not math.isfinite(extent):
            raise TsdfError("volume extent must remain finite")


def _canonical_float(value: float) -> float:
    if abs(value) < 0.5e-9:
        value = 0.0
    return float(f"{value:.9f}")


def _write_output_without_overwrite(output_path: Path, encoded: bytes) -> None:
    temporary_path: Path | None = None
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output_path.stem}-",
            suffix=".sftsdf",
            dir=output_path.parent,
        )
        temporary_path = Path(temporary_name)
        with open(descriptor, "wb", closefd=True) as output_file:
            output_file.write(encoded)
        try:
            os.link(temporary_path, output_path)
        except FileExistsError as error:
            raise TsdfError(
                "output appeared while integrating; refusing to overwrite: "
                f"{output_path}"
            ) from error
        except OSError as error:
            raise TsdfError(
                f"cannot publish TSDF output without overwriting: {error}"
            ) from error
    except TsdfError:
        raise
    except OSError as error:
        raise TsdfError(f"cannot write TSDF output: {error}") from error
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
