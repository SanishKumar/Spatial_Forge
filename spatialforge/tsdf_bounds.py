"""Deterministic world-aligned TSDF bounds from known-pose depth."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .errors import PointCloudError, TsdfError
from .model import CameraCalibration, Observation, ScanSession
from .point_cloud import (
    _read_depth,
    _sample_path,
    _transform_point,
    _validate_reconstruction_contract,
)
from .replay import replay_session
from .tsdf import (
    MAX_REFERENCE_VOXELS,
    _validate_dimensions,
    _validate_finite_volume_extent,
    _validate_origin,
    _validate_positive_integer,
    _validate_positive_number,
)


@dataclass(frozen=True, slots=True)
class TsdfBoundsReport:
    session_id: str
    replay_digest_sha256: str
    surface_min_world_m: tuple[float, float, float]
    surface_max_world_m: tuple[float, float, float]
    origin_world_m: tuple[float, float, float]
    upper_world_m: tuple[float, float, float]
    dimensions_xyz: tuple[int, int, int]
    voxel_size_m: float
    padding_m: float
    total_voxels: int
    total_observations: int
    selected_observations: int
    paired_observations: int
    skipped_missing_depth: int
    skipped_missing_pose: int
    valid_depth_points: int
    invalid_depth_samples: int


def infer_tsdf_bounds(
    session: ScanSession,
    *,
    voxel_size_m: float,
    truncation_m: float,
    frame_stride: int = 1,
) -> TsdfBoundsReport:
    """Infer an outward-snapped dense volume from known-pose depth points."""

    voxel_size = _validate_positive_number(voxel_size_m, "voxel_size_m")
    truncation = _validate_positive_number(truncation_m, "truncation_m")
    if truncation < voxel_size:
        raise TsdfError("truncation_m must be greater than or equal to voxel_size_m")
    _validate_positive_integer(frame_stride, "frame_stride")

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

    surface_min = [math.inf, math.inf, math.inf]
    surface_max = [-math.inf, -math.inf, -math.inf]
    paired_observations = 0
    skipped_missing_depth = 0
    skipped_missing_pose = 0
    valid_depth_points = 0
    invalid_depth_samples = 0

    for observation in selected:
        missing_depth = observation.depth is None
        missing_pose = observation.pose is None
        if missing_depth:
            skipped_missing_depth += 1
        if missing_pose:
            skipped_missing_pose += 1
        if missing_depth or missing_pose:
            continue

        paired_observations += 1
        (
            frame_min,
            frame_max,
            frame_valid,
            frame_invalid,
        ) = _observation_surface_bounds(
            session,
            observation,
            camera,
            depth_scale_m,
        )
        valid_depth_points += frame_valid
        invalid_depth_samples += frame_invalid
        if frame_valid == 0:
            continue
        for axis in range(3):
            surface_min[axis] = min(surface_min[axis], frame_min[axis])
            surface_max[axis] = max(surface_max[axis], frame_max[axis])

    if paired_observations == 0:
        raise TsdfError(
            "no selected RGB observation has both exact depth and pose "
            "for automatic bounds"
        )
    if valid_depth_points == 0:
        raise TsdfError(
            "selected frames contain no positive finite depth samples "
            "for automatic bounds"
        )

    surface_minimum = (surface_min[0], surface_min[1], surface_min[2])
    surface_maximum = (surface_max[0], surface_max[1], surface_max[2])
    origin, upper, dimensions = _snap_outward(
        surface_minimum,
        surface_maximum,
        voxel_size,
        truncation,
    )
    total_voxels = math.prod(dimensions)
    if total_voxels > MAX_REFERENCE_VOXELS:
        raise TsdfError(
            "automatic TSDF bounds require "
            f"{total_voxels} voxels with dimensions {dimensions}; "
            f"maximum is {MAX_REFERENCE_VOXELS}. Increase --voxel-size-m "
            "or clean the depth/pose input."
        )

    return TsdfBoundsReport(
        session_id=session.session_id,
        replay_digest_sha256=replay.digest_sha256,
        surface_min_world_m=surface_minimum,
        surface_max_world_m=surface_maximum,
        origin_world_m=origin,
        upper_world_m=upper,
        dimensions_xyz=dimensions,
        voxel_size_m=voxel_size,
        padding_m=truncation,
        total_voxels=total_voxels,
        total_observations=len(replay.observations),
        selected_observations=len(selected),
        paired_observations=paired_observations,
        skipped_missing_depth=skipped_missing_depth,
        skipped_missing_pose=skipped_missing_pose,
        valid_depth_points=valid_depth_points,
        invalid_depth_samples=invalid_depth_samples,
    )


def _observation_surface_bounds(
    session: ScanSession,
    observation: Observation,
    camera: CameraCalibration,
    depth_scale_m: float,
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    int,
    int,
]:
    if observation.depth is None or observation.pose is None:
        raise AssertionError("caller must filter incomplete observations")

    try:
        depth_path = _sample_path(session, observation.depth.data, "depth")
        depth_pixels = _read_depth(
            depth_path,
            camera.width,
            camera.height,
        )
    except PointCloudError as error:
        raise TsdfError(str(error)) from error

    transform = tuple(observation.pose.data["T_world_camera"])
    minimum = [math.inf, math.inf, math.inf]
    maximum = [-math.inf, -math.inf, -math.inf]
    valid_points = 0
    invalid_samples = 0

    for v in range(camera.height):
        row_offset = v * camera.width
        for u in range(camera.width):
            raw_depth = depth_pixels[row_offset + u]
            if raw_depth <= 0:
                invalid_samples += 1
                continue
            z_camera = raw_depth * depth_scale_m
            if not math.isfinite(z_camera) or z_camera <= 0.0:
                invalid_samples += 1
                continue

            x_camera = (u - camera.cx) * z_camera / camera.fx
            y_camera = (v - camera.cy) * z_camera / camera.fy
            world = _transform_point(
                transform,
                x_camera,
                y_camera,
                z_camera,
            )
            if not all(math.isfinite(value) for value in world):
                raise TsdfError(
                    "automatic bounds projection produced a non-finite "
                    f"coordinate for RGB sample {observation.rgb.id!r} "
                    f"at pixel ({u}, {v})"
                )
            for axis, value in enumerate(world):
                minimum[axis] = min(minimum[axis], value)
                maximum[axis] = max(maximum[axis], value)
            valid_points += 1

    return (
        (minimum[0], minimum[1], minimum[2]),
        (maximum[0], maximum[1], maximum[2]),
        valid_points,
        invalid_samples,
    )


def _snap_outward(
    surface_min: tuple[float, float, float],
    surface_max: tuple[float, float, float],
    voxel_size_m: float,
    padding_m: float,
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[int, int, int],
]:
    lower_indices: list[int] = []
    upper_indices: list[int] = []
    padded_minimum: list[float] = []
    padded_maximum: list[float] = []
    for minimum, maximum in zip(surface_min, surface_max, strict=True):
        padded_min = minimum - padding_m
        padded_max = maximum + padding_m
        lower_ratio = padded_min / voxel_size_m
        upper_ratio = padded_max / voxel_size_m
        if not all(
            math.isfinite(value)
            for value in (
                padded_min,
                padded_max,
                lower_ratio,
                upper_ratio,
            )
        ):
            raise TsdfError("automatic TSDF bounds must remain finite")
        lower_indices.append(math.floor(lower_ratio))
        upper_indices.append(math.ceil(upper_ratio))
        padded_minimum.append(padded_min)
        padded_maximum.append(padded_max)

    origin_values = [
        lower_index * voxel_size_m for lower_index in lower_indices
    ]
    upper_values = [
        upper_index * voxel_size_m for upper_index in upper_indices
    ]
    for axis in range(3):
        if origin_values[axis] > padded_minimum[axis]:
            lower_indices[axis] -= 1
            origin_values[axis] = lower_indices[axis] * voxel_size_m
        if upper_values[axis] < padded_maximum[axis]:
            upper_indices[axis] += 1
            upper_values[axis] = upper_indices[axis] * voxel_size_m

    origin = tuple(
        0.0 if value == 0.0 else value for value in origin_values
    )
    dimensions = tuple(
        upper - lower
        for lower, upper in zip(
            lower_indices,
            upper_indices,
            strict=True,
        )
    )
    _validate_origin(origin)
    _validate_dimensions(dimensions)
    _validate_finite_volume_extent(origin, dimensions, voxel_size_m)
    upper = tuple(
        axis_origin + axis_dimension * voxel_size_m
        for axis_origin, axis_dimension in zip(
            origin,
            dimensions,
            strict=True,
        )
    )
    if not all(math.isfinite(value) for value in upper):
        raise TsdfError("automatic TSDF bounds must remain finite")
    if any(
        lower > padded_lower or upper_value < padded_upper
        for lower, upper_value, padded_lower, padded_upper in zip(
            origin,
            upper,
            padded_minimum,
            padded_maximum,
            strict=True,
        )
    ):
        raise TsdfError("automatic TSDF bounds could not be snapped outward")
    return (
        (origin[0], origin[1], origin[2]),
        (upper[0], upper[1], upper[2]),
        (dimensions[0], dimensions[1], dimensions[2]),
    )
