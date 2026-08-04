"""Conservative nearest-pixel footprint block coverage for one pixel."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from .errors import TsdfError
from .point_cloud import _transform_point
from .tsdf_block_plan import (
    MAX_PLANNED_BLOCKS,
    TSDF_BLOCK_RESOLUTION,
    _containing_block_index,
    _ordered_blocks,
    _validate_block_index,
)
from .tsdf_block_plan_loader import TsdfBlockPlan
from .tsdf_observation_block_rays import (
    _is_finite_number,
    _is_sha256,
    _trace_closed_block_segment,
    _validate_block_index_xyz,
    _validate_image_size,
    _validate_pixel_uv,
    _validate_point,
    _validate_trace_plan,
)
from .tsdf_replay_depth_context import (
    TsdfReplayDepthContext,
    TsdfReplayDepthStatus,
)
from .tsdf_voxel_contribution import _validate_contribution_context

_BlockIndex = tuple[int, int, int]
_Point3 = tuple[float, float, float]
_Plane = tuple[_Point3, _Point3]

MAX_TSDF_PIXEL_FOOTPRINT_CANDIDATE_BLOCKS = 262_144

_FOOTPRINT_CORNER_OFFSETS = (
    (-0.5, -0.5),
    (0.5, -0.5),
    (0.5, 0.5),
    (-0.5, 0.5),
)


class TsdfPixelFootprintStatus(StrEnum):
    """Stable outcomes for one prepared pixel's footprint coverage."""

    COVERED = "covered"
    DEPTH_INVALID = "depth-invalid"
    MISSING_DEPTH = "missing-depth"
    MISSING_POSE = "missing-pose"
    MISSING_DEPTH_AND_POSE = "missing-depth-and-pose"


@dataclass(frozen=True, slots=True)
class TsdfPixelFootprintCoverageReceipt:
    """Immutable conservative coverage transcript for one prepared pixel."""

    source_plan_digest_sha256: str
    replay_digest_sha256: str
    frame_stride: int
    total_observations: int
    source_plan_block_indices: tuple[_BlockIndex, ...]
    observation_sequence: int
    observation_status: TsdfReplayDepthStatus
    pixel_uv: tuple[int, int]
    status: TsdfPixelFootprintStatus
    block_resolution: int
    block_extent_m: float
    image_size: tuple[int, int]
    camera_origin_world_m: _Point3 | None
    measured_depth_m: float | None
    footprint_corners_world_m: tuple[_Point3, ...]
    centerline_block_indices: tuple[_BlockIndex, ...]
    candidate_min_block_index: _BlockIndex | None
    candidate_max_block_index: _BlockIndex | None
    covered_block_indices: tuple[_BlockIndex, ...]
    existing_plan_block_indices: tuple[_BlockIndex, ...]
    unplanned_block_indices: tuple[_BlockIndex, ...]

    def __post_init__(self) -> None:
        if not _is_sha256(self.source_plan_digest_sha256):
            raise TsdfError(
                "TSDF pixel footprint source plan digest is invalid"
            )
        if not _is_sha256(self.replay_digest_sha256):
            raise TsdfError("TSDF pixel footprint replay digest is invalid")
        if (
            isinstance(self.frame_stride, bool)
            or not isinstance(self.frame_stride, int)
            or self.frame_stride < 1
        ):
            raise TsdfError(
                "TSDF pixel footprint frame stride must be positive"
            )
        if (
            isinstance(self.total_observations, bool)
            or not isinstance(self.total_observations, int)
            or self.total_observations < 1
        ):
            raise TsdfError(
                "TSDF pixel footprint total observations must be positive"
            )
        _validate_canonical_footprint_blocks(
            self.source_plan_block_indices,
            "source plan blocks",
        )
        if (
            not self.source_plan_block_indices
            or len(self.source_plan_block_indices) > MAX_PLANNED_BLOCKS
        ):
            raise TsdfError(
                "TSDF pixel footprint source plan blocks are invalid"
            )
        if (
            isinstance(self.observation_sequence, bool)
            or not isinstance(self.observation_sequence, int)
            or self.observation_sequence < 0
        ):
            raise TsdfError("TSDF pixel footprint sequence is invalid")
        if self.observation_sequence >= self.total_observations:
            raise TsdfError(
                "TSDF pixel footprint sequence is outside its total "
                "observation range"
            )
        if self.observation_sequence % self.frame_stride != 0:
            raise TsdfError(
                "TSDF pixel footprint sequence is not selected by its frame "
                "stride"
            )
        if not isinstance(self.observation_status, TsdfReplayDepthStatus):
            raise TsdfError(
                "TSDF pixel footprint prepared status is invalid"
            )
        if not isinstance(self.status, TsdfPixelFootprintStatus):
            raise TsdfError("TSDF pixel footprint status is invalid")
        _validate_pixel_uv(self.pixel_uv)
        _validate_image_size(self.image_size)
        if (
            self.pixel_uv[0] >= self.image_size[0]
            or self.pixel_uv[1] >= self.image_size[1]
        ):
            raise TsdfError(
                "TSDF pixel footprint pixel is outside its camera image"
            )
        if (
            isinstance(self.block_resolution, bool)
            or not isinstance(self.block_resolution, int)
            or self.block_resolution != TSDF_BLOCK_RESOLUTION
        ):
            raise TsdfError(
                "TSDF pixel footprint requires block resolution "
                f"{TSDF_BLOCK_RESOLUTION}"
            )
        if (
            not _is_finite_number(self.block_extent_m)
            or self.block_extent_m <= 0.0
        ):
            raise TsdfError(
                "TSDF pixel footprint block extent must be finite and "
                "positive"
            )
        _validate_expected_status(self.status, self.observation_status)

        pose_available = self.observation_status in (
            TsdfReplayDepthStatus.READY,
            TsdfReplayDepthStatus.MISSING_DEPTH,
        )
        if pose_available:
            _validate_point(self.camera_origin_world_m, "camera origin")
        elif self.camera_origin_world_m is not None:
            raise TsdfError(
                "pose-missing TSDF pixel footprint cannot retain a camera "
                "origin"
            )

        if self.status is not TsdfPixelFootprintStatus.COVERED:
            if (
                self.measured_depth_m is not None
                or self.footprint_corners_world_m
                or self.centerline_block_indices
                or self.candidate_min_block_index is not None
                or self.candidate_max_block_index is not None
                or self.covered_block_indices
                or self.existing_plan_block_indices
                or self.unplanned_block_indices
            ):
                raise TsdfError(
                    "uncovered TSDF pixel footprint cannot retain geometry"
                )
            return

        if (
            not _is_finite_number(self.measured_depth_m)
            or self.measured_depth_m <= 0.0
        ):
            raise TsdfError(
                "covered TSDF pixel footprint depth must be finite and "
                "positive"
            )
        if (
            not isinstance(self.footprint_corners_world_m, tuple)
            or len(self.footprint_corners_world_m) != 4
        ):
            raise TsdfError(
                "covered TSDF pixel footprint requires four world corners"
            )
        for corner in self.footprint_corners_world_m:
            _validate_point(corner, "footprint corner")
        if self.camera_origin_world_m is None:
            raise AssertionError("covered footprint must retain an apex")

        expected_covered, expected_min, expected_max = _cover_footprint_blocks(
            self.camera_origin_world_m,
            self.footprint_corners_world_m,
            self.block_extent_m,
        )
        if (
            self.candidate_min_block_index != expected_min
            or self.candidate_max_block_index != expected_max
        ):
            raise TsdfError(
                "TSDF pixel footprint candidate range does not match its "
                "world footprint"
            )
        if self.covered_block_indices != expected_covered:
            raise TsdfError(
                "TSDF pixel footprint covered blocks do not match its world "
                "footprint"
            )
        expected_centerline = _trace_closed_block_segment(
            self.camera_origin_world_m,
            _centroid(self.footprint_corners_world_m),
            self.block_extent_m,
        )
        if self.centerline_block_indices != expected_centerline:
            raise TsdfError(
                "TSDF pixel footprint centerline does not match its world "
                "footprint centre"
            )
        if not set(self.centerline_block_indices) <= set(
            self.covered_block_indices
        ):
            raise TsdfError(
                "TSDF pixel footprint coverage does not contain its own "
                "centerline path"
            )

        _validate_canonical_footprint_blocks(
            self.existing_plan_block_indices,
            "existing-plan blocks",
        )
        _validate_canonical_footprint_blocks(
            self.unplanned_block_indices,
            "unplanned blocks",
        )
        covered = set(self.covered_block_indices)
        existing = set(self.existing_plan_block_indices)
        unplanned = set(self.unplanned_block_indices)
        if existing & unplanned or existing | unplanned != covered:
            raise TsdfError(
                "TSDF pixel footprint plan partition is inconsistent"
            )
        source_plan = set(self.source_plan_block_indices)
        if (
            self.existing_plan_block_indices
            != _ordered_blocks(covered & source_plan)
            or self.unplanned_block_indices
            != _ordered_blocks(covered - source_plan)
        ):
            raise TsdfError(
                "TSDF pixel footprint plan membership is inconsistent"
            )
        if len(self.covered_block_indices) > MAX_PLANNED_BLOCKS:
            raise TsdfError(
                "TSDF pixel footprint exceeds the unique-block limit"
            )
        if (
            self.candidate_block_count
            > MAX_TSDF_PIXEL_FOOTPRINT_CANDIDATE_BLOCKS
        ):
            raise TsdfError(
                "TSDF pixel footprint exceeds the candidate block limit"
            )

    @property
    def covered(self) -> bool:
        return self.status is TsdfPixelFootprintStatus.COVERED

    @property
    def candidate_block_count(self) -> int:
        if (
            self.candidate_min_block_index is None
            or self.candidate_max_block_index is None
        ):
            return 0
        count = 1
        for axis in range(3):
            count *= (
                self.candidate_max_block_index[axis]
                - self.candidate_min_block_index[axis]
                + 1
            )
        return count

    @property
    def covered_block_count(self) -> int:
        return len(self.covered_block_indices)

    @property
    def rejected_candidate_count(self) -> int:
        return self.candidate_block_count - self.covered_block_count

    @property
    def centerline_block_count(self) -> int:
        return len(self.centerline_block_indices)

    @property
    def footprint_only_block_indices(self) -> tuple[_BlockIndex, ...]:
        return _ordered_blocks(
            set(self.covered_block_indices)
            - set(self.centerline_block_indices)
        )

    @property
    def widens_centerline_coverage(self) -> bool:
        return bool(self.footprint_only_block_indices)

    @property
    def prepared_depth_accessed(self) -> bool:
        return self.observation_status is TsdfReplayDepthStatus.READY


def evaluate_tsdf_pixel_footprint_coverage_from_context(
    plan: TsdfBlockPlan,
    context: TsdfReplayDepthContext,
    observation_sequence: int,
    pixel_uv: tuple[int, int],
) -> TsdfPixelFootprintCoverageReceipt:
    """Cover one prepared pixel's sampling wedge without I/O or mutation."""

    if not isinstance(plan, TsdfBlockPlan):
        raise TsdfError(
            "TSDF pixel footprint coverage requires a loaded TsdfBlockPlan"
        )
    if not isinstance(context, TsdfReplayDepthContext):
        raise TsdfError(
            "TSDF pixel footprint coverage requires a prepared "
            "TsdfReplayDepthContext"
        )
    if isinstance(observation_sequence, bool) or not isinstance(
        observation_sequence,
        int,
    ):
        raise TsdfError("observation_sequence: expected an integer")
    if observation_sequence < 0:
        raise TsdfError(
            "observation_sequence: expected a non-negative integer"
        )
    _validate_pixel_uv(pixel_uv)

    try:
        _validate_trace_plan(plan)
        _validate_contribution_context(plan, context)
        observation = _select_observation(context, observation_sequence)
        camera = context.camera
        image_size = (camera.width, camera.height)
        if (
            pixel_uv[0] >= image_size[0]
            or pixel_uv[1] >= image_size[1]
        ):
            raise TsdfError(
                f"pixel_uv: outside the camera image {image_size[0]}x"
                f"{image_size[1]}"
            )
        transform = observation.t_world_camera
        camera_origin = _camera_origin(transform)
        if observation.status is not TsdfReplayDepthStatus.READY:
            return _uncovered(
                plan,
                observation_sequence,
                observation.status,
                pixel_uv,
                image_size,
                camera_origin,
                _status_for_missing_input(observation.status),
            )

        depth_m = observation.depth_m
        if transform is None or depth_m is None or camera_origin is None:
            raise TsdfError(
                "ready TSDF replay/depth observation is incomplete"
            )
        measured_depth_m = float(depth_m[pixel_uv[1], pixel_uv[0]])
        if measured_depth_m <= 0.0 or not math.isfinite(measured_depth_m):
            return _uncovered(
                plan,
                observation_sequence,
                observation.status,
                pixel_uv,
                image_size,
                camera_origin,
                TsdfPixelFootprintStatus.DEPTH_INVALID,
            )

        corners = _footprint_corners_world_m(
            camera,
            transform,
            pixel_uv,
            measured_depth_m,
        )
        candidate_count = _candidate_block_count(
            camera_origin,
            corners,
            plan.block_extent_m,
        )
        if candidate_count > MAX_TSDF_PIXEL_FOOTPRINT_CANDIDATE_BLOCKS:
            raise TsdfError(
                "TSDF pixel footprint coverage requires "
                f"{candidate_count} candidate blocks; reference maximum is "
                f"{MAX_TSDF_PIXEL_FOOTPRINT_CANDIDATE_BLOCKS}. Use a coarser "
                "voxel size, a nearer measurement, or a future scalable "
                "coverage path"
            )
        covered, candidate_min, candidate_max = _cover_footprint_blocks(
            camera_origin,
            corners,
            plan.block_extent_m,
        )
        if len(covered) > MAX_PLANNED_BLOCKS:
            raise TsdfError(
                "TSDF pixel footprint coverage exceeds the "
                f"{MAX_PLANNED_BLOCKS}-block reference limit. Use a coarser "
                "voxel size or a future scalable coverage path"
            )
        centerline = _trace_closed_block_segment(
            camera_origin,
            _centroid(corners),
            plan.block_extent_m,
        )
        active = set(plan.active_blocks)
        covered_set = set(covered)
        return TsdfPixelFootprintCoverageReceipt(
            source_plan_digest_sha256=plan.artifact_digest_sha256,
            replay_digest_sha256=plan.replay_digest_sha256,
            frame_stride=plan.frame_stride,
            total_observations=plan.total_observations,
            source_plan_block_indices=plan.active_blocks,
            observation_sequence=observation_sequence,
            observation_status=observation.status,
            pixel_uv=pixel_uv,
            status=TsdfPixelFootprintStatus.COVERED,
            block_resolution=plan.block_resolution,
            block_extent_m=plan.block_extent_m,
            image_size=image_size,
            camera_origin_world_m=camera_origin,
            measured_depth_m=measured_depth_m,
            footprint_corners_world_m=corners,
            centerline_block_indices=centerline,
            candidate_min_block_index=candidate_min,
            candidate_max_block_index=candidate_max,
            covered_block_indices=covered,
            existing_plan_block_indices=_ordered_blocks(covered_set & active),
            unplanned_block_indices=_ordered_blocks(covered_set - active),
        )
    except TsdfError:
        raise
    except Exception as error:
        raise TsdfError(
            f"cannot cover prepared TSDF pixel footprint: {error}"
        ) from error


def _uncovered(
    plan: TsdfBlockPlan,
    observation_sequence: int,
    observation_status: TsdfReplayDepthStatus,
    pixel_uv: tuple[int, int],
    image_size: tuple[int, int],
    camera_origin: _Point3 | None,
    status: TsdfPixelFootprintStatus,
) -> TsdfPixelFootprintCoverageReceipt:
    return TsdfPixelFootprintCoverageReceipt(
        source_plan_digest_sha256=plan.artifact_digest_sha256,
        replay_digest_sha256=plan.replay_digest_sha256,
        frame_stride=plan.frame_stride,
        total_observations=plan.total_observations,
        source_plan_block_indices=plan.active_blocks,
        observation_sequence=observation_sequence,
        observation_status=observation_status,
        pixel_uv=pixel_uv,
        status=status,
        block_resolution=plan.block_resolution,
        block_extent_m=plan.block_extent_m,
        image_size=image_size,
        camera_origin_world_m=camera_origin,
        measured_depth_m=None,
        footprint_corners_world_m=(),
        centerline_block_indices=(),
        candidate_min_block_index=None,
        candidate_max_block_index=None,
        covered_block_indices=(),
        existing_plan_block_indices=(),
        unplanned_block_indices=(),
    )


def _status_for_missing_input(
    observation_status: TsdfReplayDepthStatus,
) -> TsdfPixelFootprintStatus:
    if observation_status is TsdfReplayDepthStatus.MISSING_DEPTH:
        return TsdfPixelFootprintStatus.MISSING_DEPTH
    if observation_status is TsdfReplayDepthStatus.MISSING_POSE:
        return TsdfPixelFootprintStatus.MISSING_POSE
    if observation_status is TsdfReplayDepthStatus.MISSING_DEPTH_AND_POSE:
        return TsdfPixelFootprintStatus.MISSING_DEPTH_AND_POSE
    raise TsdfError(
        "TSDF pixel footprint cannot classify a ready observation as missing"
    )


def _validate_expected_status(
    status: TsdfPixelFootprintStatus,
    observation_status: TsdfReplayDepthStatus,
) -> None:
    if observation_status is TsdfReplayDepthStatus.READY:
        allowed = (
            TsdfPixelFootprintStatus.COVERED,
            TsdfPixelFootprintStatus.DEPTH_INVALID,
        )
    else:
        allowed = (_status_for_missing_input(observation_status),)
    if status not in allowed:
        raise TsdfError(
            "TSDF pixel footprint status does not match its prepared "
            "observation status"
        )


def _select_observation(
    context: TsdfReplayDepthContext,
    observation_sequence: int,
):
    if observation_sequence >= context.total_observations:
        raise TsdfError(
            "observation_sequence: outside context range "
            f"[0, {context.total_observations - 1}]"
        )
    if observation_sequence % context.frame_stride != 0:
        raise TsdfError(
            "observation_sequence: not selected by the block plan's "
            f"frame_stride={context.frame_stride}"
        )
    observation = context.observations[
        observation_sequence // context.frame_stride
    ]
    if observation.observation_sequence != observation_sequence:
        raise TsdfError(
            "TSDF replay/depth context observation lookup is inconsistent"
        )
    return observation


def _camera_origin(transform: tuple[float, ...] | None) -> _Point3 | None:
    if transform is None:
        return None
    origin = (transform[3], transform[7], transform[11])
    _validate_point(origin, "camera origin")
    return origin


def _footprint_corners_world_m(
    camera,
    transform: tuple[float, ...],
    pixel_uv: tuple[int, int],
    measured_depth_m: float,
) -> tuple[_Point3, _Point3, _Point3, _Point3]:
    corners: list[_Point3] = []
    for offset_u, offset_v in _FOOTPRINT_CORNER_OFFSETS:
        x_camera = (
            (pixel_uv[0] + offset_u - camera.cx)
            * measured_depth_m
            / camera.fx
        )
        y_camera = (
            (pixel_uv[1] + offset_v - camera.cy)
            * measured_depth_m
            / camera.fy
        )
        corner = _transform_point(
            transform,
            x_camera,
            y_camera,
            measured_depth_m,
        )
        _validate_point(corner, "footprint corner")
        corners.append(corner)
    return tuple(corners)  # type: ignore[return-value]


def _centroid(points: tuple[_Point3, ...]) -> _Point3:
    count = float(len(points))
    centroid = tuple(
        sum(point[axis] for point in points) / count for axis in range(3)
    )
    _validate_point(centroid, "footprint centre")
    return centroid  # type: ignore[return-value]


def _candidate_block_range(
    apex: _Point3,
    corners: tuple[_Point3, ...],
    block_extent_m: float,
) -> tuple[_BlockIndex, _BlockIndex]:
    vertices = (apex, *corners)
    minimum: list[int] = []
    maximum: list[int] = []
    for axis in range(3):
        values = [vertex[axis] for vertex in vertices]
        low = _containing_block_index(min(values), block_extent_m)
        high = _containing_block_index(max(values), block_extent_m)
        _validate_block_index(low)
        _validate_block_index(high)
        minimum.append(low)
        maximum.append(high)
    return tuple(minimum), tuple(maximum)  # type: ignore[return-value]


def _candidate_block_count(
    apex: _Point3,
    corners: tuple[_Point3, ...],
    block_extent_m: float,
) -> int:
    minimum, maximum = _candidate_block_range(apex, corners, block_extent_m)
    count = 1
    for axis in range(3):
        count *= maximum[axis] - minimum[axis] + 1
    return count


def _cover_footprint_blocks(
    apex: _Point3,
    corners: tuple[_Point3, ...],
    block_extent_m: float,
) -> tuple[tuple[_BlockIndex, ...], _BlockIndex, _BlockIndex]:
    """Cover a nearest-pixel sampling wedge with a conservative superset."""

    _validate_point(apex, "footprint apex")
    if not isinstance(corners, tuple) or len(corners) != 4:
        raise TsdfError(
            "TSDF pixel footprint requires exactly four world corners"
        )
    for corner in corners:
        _validate_point(corner, "footprint corner")
    if not _is_finite_number(block_extent_m) or block_extent_m <= 0.0:
        raise TsdfError(
            "TSDF pixel footprint block extent must be finite and positive"
        )

    minimum, maximum = _candidate_block_range(apex, corners, block_extent_m)
    planes = _footprint_planes(apex, corners)
    covered: list[_BlockIndex] = []
    for block_z in range(minimum[2], maximum[2] + 1):
        for block_y in range(minimum[1], maximum[1] + 1):
            for block_x in range(minimum[0], maximum[0] + 1):
                block_index = (block_x, block_y, block_z)
                if _block_outside_any_plane(
                    block_index,
                    block_extent_m,
                    planes,
                ):
                    continue
                covered.append(block_index)
    return tuple(covered), minimum, maximum


def _footprint_planes(
    apex: _Point3,
    corners: tuple[_Point3, ...],
) -> tuple[_Plane, ...]:
    """Build outward-oriented bounding planes for the sampling wedge."""

    interior = _centroid((apex, *corners))
    candidates: list[tuple[_Point3, _Point3]] = []
    for index in range(4):
        candidates.append(
            (
                apex,
                _cross(
                    _difference(corners[index], apex),
                    _difference(corners[(index + 1) % 4], apex),
                ),
            )
        )
    candidates.append(
        (
            corners[0],
            _cross(
                _difference(corners[1], corners[0]),
                _difference(corners[2], corners[0]),
            ),
        )
    )
    candidates.append(
        (apex, _difference(apex, _centroid(corners))),
    )

    planes: list[_Plane] = []
    for point, normal in candidates:
        interior_side = _dot(normal, _difference(interior, point))
        if not math.isfinite(interior_side) or interior_side == 0.0:
            continue
        if interior_side > 0.0:
            normal = (-normal[0], -normal[1], -normal[2])
        planes.append((point, normal))
    if not planes:
        raise TsdfError(
            "TSDF pixel footprint wedge is degenerate and cannot be bounded"
        )
    return tuple(planes)


def _block_outside_any_plane(
    block_index: _BlockIndex,
    block_extent_m: float,
    planes: tuple[_Plane, ...],
) -> bool:
    low = tuple(component * block_extent_m for component in block_index)
    high = tuple(value + block_extent_m for value in low)
    for point, normal in planes:
        nearest: _Point3 = tuple(  # type: ignore[assignment]
            low[axis] if normal[axis] >= 0.0 else high[axis]
            for axis in range(3)
        )
        signed = _dot(normal, _difference(nearest, point))
        if not math.isfinite(signed):
            raise TsdfError(
                "TSDF pixel footprint plane distance must remain finite"
            )
        if signed > 0.0:
            return True
    return False


def _difference(left: _Point3, right: _Point3) -> _Point3:
    return (
        left[0] - right[0],
        left[1] - right[1],
        left[2] - right[2],
    )


def _cross(left: _Point3, right: _Point3) -> _Point3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _dot(left: _Point3, right: _Point3) -> float:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _validate_canonical_footprint_blocks(value: object, label: str) -> None:
    if not isinstance(value, tuple):
        raise TsdfError(f"TSDF pixel footprint {label} must be a tuple")
    previous_key: tuple[int, int, int] | None = None
    for block_index in value:
        _validate_block_index_xyz(block_index, label)
        key = (block_index[2], block_index[1], block_index[0])
        if previous_key is not None and key <= previous_key:
            raise TsdfError(
                f"TSDF pixel footprint {label} must be unique and strictly "
                "x-fastest ordered"
            )
        previous_key = key
