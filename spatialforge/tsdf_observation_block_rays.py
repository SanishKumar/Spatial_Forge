"""Read-only block-ray tracing for one prepared TSDF observation."""

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
from .tsdf_replay_depth_context import (
    TsdfReplayDepthContext,
    TsdfReplayDepthStatus,
)
from .tsdf_voxel_contribution import (
    _validate_contribution_context,
    _validate_contribution_plan,
)

_BlockIndex = tuple[int, int, int]
_Point3 = tuple[float, float, float]

MAX_TSDF_OBSERVATION_BLOCK_RAY_OUTCOMES = 262_144


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


class TsdfObservationBlockRayStatus(StrEnum):
    """Stable outcomes for one prepared depth pixel."""

    TRAVERSED = "traversed"
    DEPTH_INVALID = "depth-invalid"


@dataclass(frozen=True, slots=True)
class TsdfObservationBlockRayReceipt:
    """Immutable outcome for one pixel-center camera-to-surface ray."""

    pixel_uv: tuple[int, int]
    status: TsdfObservationBlockRayStatus
    camera_origin_world_m: _Point3
    block_extent_m: float
    measured_depth_m: float | None
    surface_world_m: _Point3 | None
    surface_block_index: _BlockIndex | None
    block_indices: tuple[_BlockIndex, ...]

    def __post_init__(self) -> None:
        _validate_pixel_uv(self.pixel_uv)
        if not isinstance(self.status, TsdfObservationBlockRayStatus):
            raise TsdfError("TSDF observation block ray status is invalid")
        _validate_point(self.camera_origin_world_m, "camera origin")
        if (
            not _is_finite_number(self.block_extent_m)
            or self.block_extent_m <= 0.0
        ):
            raise TsdfError(
                "TSDF observation block ray extent must be finite and "
                "positive"
            )
        if not isinstance(self.block_indices, tuple):
            raise TsdfError(
                "TSDF observation block ray indices must be a tuple"
            )

        if self.status is TsdfObservationBlockRayStatus.DEPTH_INVALID:
            if (
                self.measured_depth_m is not None
                or self.surface_world_m is not None
                or self.surface_block_index is not None
                or self.block_indices
            ):
                raise TsdfError(
                    "invalid-depth TSDF observation ray cannot retain "
                    "geometry"
                )
            return

        if (
            not _is_finite_number(self.measured_depth_m)
            or self.measured_depth_m <= 0.0
        ):
            raise TsdfError(
                "traversed TSDF observation ray depth must be finite and "
                "positive"
            )
        _validate_point(self.surface_world_m, "surface world point")
        _validate_block_index_xyz(
            self.surface_block_index,
            "surface block index",
        )
        if not self.block_indices:
            raise TsdfError(
                "traversed TSDF observation ray requires block indices"
            )
        for block_index in self.block_indices:
            _validate_block_index_xyz(block_index, "block index")
        if self.block_indices[-1] != self.surface_block_index:
            raise TsdfError(
                "TSDF observation ray surface block must terminate its path"
            )
        if self.surface_world_m is None:
            raise AssertionError("traversed ray must retain a surface point")
        expected_path = _trace_closed_block_segment(
            self.camera_origin_world_m,
            self.surface_world_m,
            self.block_extent_m,
        )
        if self.block_indices != expected_path:
            raise TsdfError(
                "TSDF observation ray path does not match its closed "
                "half-open-grid segment"
            )

    @property
    def traversed(self) -> bool:
        return self.status is TsdfObservationBlockRayStatus.TRAVERSED


@dataclass(frozen=True, slots=True)
class TsdfObservationBlockRayTraceReceipt:
    """Immutable transcript for one prepared observation's block rays."""

    source_plan_digest_sha256: str
    replay_digest_sha256: str
    frame_stride: int
    total_observations: int
    source_plan_block_indices: tuple[_BlockIndex, ...]
    observation_sequence: int
    observation_status: TsdfReplayDepthStatus
    block_resolution: int
    block_extent_m: float
    camera_origin_world_m: _Point3 | None
    image_size: tuple[int, int]
    ray_receipts: tuple[TsdfObservationBlockRayReceipt, ...]
    covered_block_indices: tuple[_BlockIndex, ...]
    existing_plan_block_indices: tuple[_BlockIndex, ...]
    unplanned_block_indices: tuple[_BlockIndex, ...]

    def __post_init__(self) -> None:
        if not _is_sha256(self.source_plan_digest_sha256):
            raise TsdfError(
                "TSDF observation ray trace source plan digest is invalid"
            )
        if not _is_sha256(self.replay_digest_sha256):
            raise TsdfError(
                "TSDF observation ray trace replay digest is invalid"
            )
        if (
            isinstance(self.frame_stride, bool)
            or not isinstance(self.frame_stride, int)
            or self.frame_stride < 1
        ):
            raise TsdfError(
                "TSDF observation ray trace frame stride must be positive"
            )
        if (
            isinstance(self.total_observations, bool)
            or not isinstance(self.total_observations, int)
            or self.total_observations < 1
        ):
            raise TsdfError(
                "TSDF observation ray trace total observations must be "
                "positive"
            )
        _validate_canonical_block_tuple(
            self.source_plan_block_indices,
            "source plan blocks",
        )
        if (
            not self.source_plan_block_indices
            or len(self.source_plan_block_indices) > MAX_PLANNED_BLOCKS
        ):
            raise TsdfError(
                "TSDF observation ray trace source plan blocks are invalid"
            )
        if (
            isinstance(self.observation_sequence, bool)
            or not isinstance(self.observation_sequence, int)
            or self.observation_sequence < 0
        ):
            raise TsdfError(
                "TSDF observation ray trace sequence is invalid"
            )
        if self.observation_sequence >= self.total_observations:
            raise TsdfError(
                "TSDF observation ray trace sequence is outside its total "
                "observation range"
            )
        if self.observation_sequence % self.frame_stride != 0:
            raise TsdfError(
                "TSDF observation ray trace sequence is not selected by "
                "its frame stride"
            )
        if not isinstance(self.observation_status, TsdfReplayDepthStatus):
            raise TsdfError(
                "TSDF observation ray trace prepared status is invalid"
            )
        if (
            isinstance(self.block_resolution, bool)
            or not isinstance(self.block_resolution, int)
            or self.block_resolution != TSDF_BLOCK_RESOLUTION
        ):
            raise TsdfError(
                "TSDF observation ray trace requires block resolution "
                f"{TSDF_BLOCK_RESOLUTION}"
            )
        if (
            not _is_finite_number(self.block_extent_m)
            or self.block_extent_m <= 0.0
        ):
            raise TsdfError(
                "TSDF observation ray trace block extent must be finite "
                "and positive"
            )
        _validate_image_size(self.image_size)
        if not isinstance(self.ray_receipts, tuple):
            raise TsdfError(
                "TSDF observation ray trace pixel receipts must be a tuple"
            )

        pose_available = self.observation_status in (
            TsdfReplayDepthStatus.READY,
            TsdfReplayDepthStatus.MISSING_DEPTH,
        )
        if pose_available:
            _validate_point(self.camera_origin_world_m, "camera origin")
        elif self.camera_origin_world_m is not None:
            raise TsdfError(
                "pose-missing TSDF observation ray trace cannot retain a "
                "camera origin"
            )

        if self.observation_status is TsdfReplayDepthStatus.READY:
            expected_pixels = self.image_size[0] * self.image_size[1]
            if len(self.ray_receipts) != expected_pixels:
                raise TsdfError(
                    "ready TSDF observation ray trace requires exactly one "
                    "receipt per camera pixel"
                )
            if self.camera_origin_world_m is None:
                raise AssertionError("ready trace must retain camera origin")
            for pixel_index, ray_receipt in enumerate(self.ray_receipts):
                if not isinstance(
                    ray_receipt,
                    TsdfObservationBlockRayReceipt,
                ):
                    raise TsdfError(
                        "TSDF observation ray trace contains an invalid "
                        "pixel receipt"
                    )
                expected_uv = (
                    pixel_index % self.image_size[0],
                    pixel_index // self.image_size[0],
                )
                if ray_receipt.pixel_uv != expected_uv:
                    raise TsdfError(
                        "TSDF observation ray receipts must be in canonical "
                        "row-major pixel order"
                    )
                if (
                    ray_receipt.camera_origin_world_m
                    != self.camera_origin_world_m
                    or ray_receipt.block_extent_m != self.block_extent_m
                ):
                    raise TsdfError(
                        "TSDF observation ray geometry does not match its "
                        "parent trace"
                    )
        elif self.ray_receipts:
            raise TsdfError(
                "input-missing TSDF observation ray trace cannot retain "
                "pixel receipts"
            )

        expected_covered = _ordered_blocks(
            {
                block_index
                for ray_receipt in self.ray_receipts
                for block_index in ray_receipt.block_indices
            }
        )
        if self.covered_block_indices != expected_covered:
            raise TsdfError(
                "TSDF observation ray trace covered blocks do not match "
                "its ray paths"
            )
        _validate_canonical_block_tuple(
            self.existing_plan_block_indices,
            "existing-plan blocks",
        )
        _validate_canonical_block_tuple(
            self.unplanned_block_indices,
            "unplanned blocks",
        )
        existing = set(self.existing_plan_block_indices)
        unplanned = set(self.unplanned_block_indices)
        if existing & unplanned or existing | unplanned != set(
            self.covered_block_indices
        ):
            raise TsdfError(
                "TSDF observation ray trace plan partition is inconsistent"
            )
        source_plan = set(self.source_plan_block_indices)
        if (
            self.existing_plan_block_indices
            != _ordered_blocks(set(self.covered_block_indices) & source_plan)
            or self.unplanned_block_indices
            != _ordered_blocks(set(self.covered_block_indices) - source_plan)
        ):
            raise TsdfError(
                "TSDF observation ray trace plan membership is inconsistent"
            )
        if len(self.covered_block_indices) > MAX_PLANNED_BLOCKS:
            raise TsdfError(
                "TSDF observation ray trace exceeds the unique-block limit"
            )
        if (
            self.retained_outcome_count
            > MAX_TSDF_OBSERVATION_BLOCK_RAY_OUTCOMES
        ):
            raise TsdfError(
                "TSDF observation ray trace exceeds the retained outcome "
                "limit"
            )

    @property
    def pixel_count(self) -> int:
        return len(self.ray_receipts)

    @property
    def traversed_ray_count(self) -> int:
        return sum(ray.traversed for ray in self.ray_receipts)

    @property
    def invalid_depth_count(self) -> int:
        return self.pixel_count - self.traversed_ray_count

    @property
    def block_visit_count(self) -> int:
        return sum(len(ray.block_indices) for ray in self.ray_receipts)

    @property
    def duplicate_block_visit_count(self) -> int:
        return self.block_visit_count - len(self.covered_block_indices)

    @property
    def retained_outcome_count(self) -> int:
        return self.pixel_count + self.block_visit_count

    @property
    def maximum_blocks_per_ray(self) -> int:
        return max(
            (len(ray.block_indices) for ray in self.ray_receipts),
            default=0,
        )

    @property
    def surface_block_indices(self) -> tuple[_BlockIndex, ...]:
        return _ordered_blocks(
            {
                ray.surface_block_index
                for ray in self.ray_receipts
                if ray.surface_block_index is not None
            }
        )

    @property
    def nonterminal_block_indices(self) -> tuple[_BlockIndex, ...]:
        return _ordered_blocks(
            {
                block_index
                for ray in self.ray_receipts
                for block_index in ray.block_indices[:-1]
            }
        )

    @property
    def prepared_depth_accessed(self) -> bool:
        return self.observation_status is TsdfReplayDepthStatus.READY

    @property
    def selected_observation_sequences(self) -> tuple[int, ...]:
        return tuple(range(0, self.total_observations, self.frame_stride))


def trace_tsdf_observation_block_rays_from_context(
    plan: TsdfBlockPlan,
    context: TsdfReplayDepthContext,
    observation_sequence: int,
) -> TsdfObservationBlockRayTraceReceipt:
    """Trace one selected prepared observation without I/O or mutation."""

    if not isinstance(plan, TsdfBlockPlan):
        raise TsdfError(
            "TSDF observation block ray tracing requires a loaded "
            "TsdfBlockPlan"
        )
    if not isinstance(context, TsdfReplayDepthContext):
        raise TsdfError(
            "TSDF observation block ray tracing requires a prepared "
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

    try:
        _validate_trace_plan(plan)
        _validate_contribution_context(plan, context)
        observation = _select_observation(context, observation_sequence)
        camera_origin = _camera_origin(observation.t_world_camera)
        image_size = (context.camera.width, context.camera.height)
        if observation.status is not TsdfReplayDepthStatus.READY:
            return TsdfObservationBlockRayTraceReceipt(
                source_plan_digest_sha256=plan.artifact_digest_sha256,
                replay_digest_sha256=plan.replay_digest_sha256,
                frame_stride=plan.frame_stride,
                total_observations=plan.total_observations,
                source_plan_block_indices=plan.active_blocks,
                observation_sequence=observation_sequence,
                observation_status=observation.status,
                block_resolution=plan.block_resolution,
                block_extent_m=plan.block_extent_m,
                camera_origin_world_m=camera_origin,
                image_size=image_size,
                ray_receipts=(),
                covered_block_indices=(),
                existing_plan_block_indices=(),
                unplanned_block_indices=(),
            )

        transform = observation.t_world_camera
        depth_m = observation.depth_m
        if transform is None or depth_m is None or camera_origin is None:
            raise TsdfError(
                "ready TSDF replay/depth observation is incomplete"
            )
        pixel_count = context.camera.width * context.camera.height
        if pixel_count > MAX_TSDF_OBSERVATION_BLOCK_RAY_OUTCOMES:
            raise TsdfError(
                "TSDF observation block ray trace requires at least "
                f"{pixel_count} retained pixel outcomes; reference maximum "
                f"is {MAX_TSDF_OBSERVATION_BLOCK_RAY_OUTCOMES}. Use a "
                "smaller diagnostic input or a future scalable coverage path"
            )

        ray_receipts: list[TsdfObservationBlockRayReceipt] = []
        covered_blocks: set[_BlockIndex] = set()
        retained_outcomes = pixel_count
        for pixel_v in range(context.camera.height):
            for pixel_u in range(context.camera.width):
                measured_depth_m = float(depth_m[pixel_v, pixel_u])
                if (
                    measured_depth_m <= 0.0
                    or not math.isfinite(measured_depth_m)
                ):
                    ray_receipts.append(
                        TsdfObservationBlockRayReceipt(
                            pixel_uv=(pixel_u, pixel_v),
                            status=TsdfObservationBlockRayStatus.DEPTH_INVALID,
                            camera_origin_world_m=camera_origin,
                            block_extent_m=plan.block_extent_m,
                            measured_depth_m=None,
                            surface_world_m=None,
                            surface_block_index=None,
                            block_indices=(),
                        )
                    )
                    continue

                surface_world_m = _surface_world_m(
                    context,
                    transform,
                    pixel_u,
                    pixel_v,
                    measured_depth_m,
                )
                maximum_visits = _maximum_segment_block_visits(
                    camera_origin,
                    surface_world_m,
                    plan.block_extent_m,
                )
                if (
                    retained_outcomes + maximum_visits
                    > MAX_TSDF_OBSERVATION_BLOCK_RAY_OUTCOMES
                ):
                    raise TsdfError(
                        "TSDF observation block ray trace conservative "
                        "retained-outcome bound reaches "
                        f"{retained_outcomes + maximum_visits}; reference "
                        "maximum is "
                        f"{MAX_TSDF_OBSERVATION_BLOCK_RAY_OUTCOMES}. Use a "
                        "smaller diagnostic input, a coarser voxel size, or "
                        "a future scalable coverage path"
                    )
                block_indices = _trace_closed_block_segment(
                    camera_origin,
                    surface_world_m,
                    plan.block_extent_m,
                )
                retained_outcomes += len(block_indices)
                covered_blocks.update(block_indices)
                if len(covered_blocks) > MAX_PLANNED_BLOCKS:
                    raise TsdfError(
                        "TSDF observation block ray trace exceeds the "
                        f"{MAX_PLANNED_BLOCKS}-block reference limit. Use a "
                        "coarser voxel size or a future scalable coverage path"
                    )
                ray_receipts.append(
                    TsdfObservationBlockRayReceipt(
                        pixel_uv=(pixel_u, pixel_v),
                        status=TsdfObservationBlockRayStatus.TRAVERSED,
                        camera_origin_world_m=camera_origin,
                        block_extent_m=plan.block_extent_m,
                        measured_depth_m=measured_depth_m,
                        surface_world_m=surface_world_m,
                        surface_block_index=block_indices[-1],
                        block_indices=block_indices,
                    )
                )

        covered = _ordered_blocks(covered_blocks)
        active = set(plan.active_blocks)
        existing = _ordered_blocks(covered_blocks & active)
        unplanned = _ordered_blocks(covered_blocks - active)
        return TsdfObservationBlockRayTraceReceipt(
            source_plan_digest_sha256=plan.artifact_digest_sha256,
            replay_digest_sha256=plan.replay_digest_sha256,
            frame_stride=plan.frame_stride,
            total_observations=plan.total_observations,
            source_plan_block_indices=plan.active_blocks,
            observation_sequence=observation_sequence,
            observation_status=observation.status,
            block_resolution=plan.block_resolution,
            block_extent_m=plan.block_extent_m,
            camera_origin_world_m=camera_origin,
            image_size=image_size,
            ray_receipts=tuple(ray_receipts),
            covered_block_indices=covered,
            existing_plan_block_indices=existing,
            unplanned_block_indices=unplanned,
        )
    except TsdfError:
        raise
    except Exception as error:
        raise TsdfError(
            f"cannot trace prepared TSDF observation block rays: {error}"
        ) from error


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


def _surface_world_m(
    context: TsdfReplayDepthContext,
    transform: tuple[float, ...],
    pixel_u: int,
    pixel_v: int,
    measured_depth_m: float,
) -> _Point3:
    camera = context.camera
    x_camera = (pixel_u - camera.cx) * measured_depth_m / camera.fx
    y_camera = (pixel_v - camera.cy) * measured_depth_m / camera.fy
    surface_world_m = _transform_point(
        transform,
        x_camera,
        y_camera,
        measured_depth_m,
    )
    _validate_point(surface_world_m, "surface world point")
    return surface_world_m


def _maximum_segment_block_visits(
    start_world_m: _Point3,
    end_world_m: _Point3,
    block_extent_m: float,
) -> int:
    start_index = tuple(
        _containing_block_index(component, block_extent_m)
        for component in start_world_m
    )
    end_index = tuple(
        _containing_block_index(component, block_extent_m)
        for component in end_world_m
    )
    return 1 + sum(
        abs(end_index[axis] - start_index[axis])
        for axis in range(3)
    )


def _trace_closed_block_segment(
    start_world_m: _Point3,
    end_world_m: _Point3,
    block_extent_m: float,
) -> tuple[_BlockIndex, ...]:
    """Trace a closed segment through a half-open grid using thin DDA."""

    _validate_point(start_world_m, "segment start")
    _validate_point(end_world_m, "segment end")
    if not _is_finite_number(block_extent_m) or block_extent_m <= 0.0:
        raise TsdfError(
            "TSDF observation block ray extent must be finite and positive"
        )

    current = [
        _containing_block_index(component, block_extent_m)
        for component in start_world_m
    ]
    target = tuple(
        _containing_block_index(component, block_extent_m)
        for component in end_world_m
    )
    path: list[_BlockIndex] = [tuple(current)]  # type: ignore[list-item]
    if tuple(current) == target:
        return tuple(path)

    direction = tuple(
        end_world_m[axis] - start_world_m[axis]
        for axis in range(3)
    )
    if any(not math.isfinite(component) for component in direction):
        raise TsdfError(
            "TSDF observation block ray direction must remain finite"
        )
    step = tuple(
        1 if component > 0.0 else -1 if component < 0.0 else 0
        for component in direction
    )
    maximum_visits = 1 + sum(
        abs(target[axis] - current[axis]) for axis in range(3)
    )

    def next_parameter(axis: int) -> float:
        if step[axis] == 0:
            return math.inf
        boundary_index = current[axis] + (1 if step[axis] > 0 else 0)
        boundary = boundary_index * block_extent_m
        parameter = (
            boundary - start_world_m[axis]
        ) / direction[axis]
        if not math.isfinite(parameter):
            raise TsdfError(
                "TSDF observation block ray boundary must remain finite"
            )
        return parameter

    parameters = [next_parameter(axis) for axis in range(3)]
    while tuple(current) != target:
        next_crossing = min(parameters)
        if not math.isfinite(next_crossing):
            raise TsdfError(
                "TSDF observation block ray traversal cannot progress"
            )
        tied_axes = tuple(
            axis
            for axis, parameter in enumerate(parameters)
            if parameter == next_crossing
        )
        if not tied_axes:
            raise TsdfError(
                "TSDF observation block ray traversal cannot select an axis"
            )
        previous_distance = sum(
            abs(target[axis] - current[axis]) for axis in range(3)
        )
        for axis in tied_axes:
            current[axis] += step[axis]
            _validate_block_index(current[axis])
        current_distance = sum(
            abs(target[axis] - current[axis]) for axis in range(3)
        )
        if current_distance >= previous_distance:
            raise TsdfError(
                "TSDF observation block ray traversal overshot its endpoint"
            )
        path.append(tuple(current))  # type: ignore[arg-type]
        if len(path) > maximum_visits:
            raise TsdfError(
                "TSDF observation block ray traversal exceeded its "
                "deterministic visit bound"
            )
        for axis in tied_axes:
            parameters[axis] = next_parameter(axis)
    return tuple(path)


def _validate_trace_plan(plan: TsdfBlockPlan) -> None:
    _validate_contribution_plan(plan)
    if plan.block_resolution != TSDF_BLOCK_RESOLUTION:
        raise TsdfError(
            "TSDF observation block ray trace requires block resolution "
            f"{TSDF_BLOCK_RESOLUTION}"
        )
    if (
        not isinstance(plan.active_blocks, tuple)
        or not plan.active_blocks
        or len(plan.active_blocks) > MAX_PLANNED_BLOCKS
    ):
        raise TsdfError(
            "TSDF observation block ray trace source plan blocks are invalid"
        )
    _validate_canonical_block_tuple(plan.active_blocks, "source plan blocks")
    expected_slots = len(plan.active_blocks) * plan.block_resolution**3
    if plan.planned_voxel_slots != expected_slots:
        raise TsdfError(
            "TSDF observation block ray trace source plan voxel slots are "
            "inconsistent"
        )


def _validate_pixel_uv(value: object) -> None:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(
            isinstance(component, bool)
            or not isinstance(component, int)
            or component < 0
            for component in value
        )
    ):
        raise TsdfError(
            "TSDF observation block ray pixel must contain two nonnegative "
            "integers"
        )


def _validate_image_size(value: object) -> None:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(
            isinstance(component, bool)
            or not isinstance(component, int)
            or component < 1
            for component in value
        )
    ):
        raise TsdfError(
            "TSDF observation ray trace image size must contain two positive "
            "integers"
        )


def _validate_point(value: object, label: str) -> None:
    if (
        not isinstance(value, tuple)
        or len(value) != 3
        or any(not _is_finite_number(component) for component in value)
    ):
        raise TsdfError(
            f"TSDF observation block ray {label} must contain three finite "
            "values"
        )


def _validate_block_index_xyz(value: object, label: str) -> None:
    if (
        not isinstance(value, tuple)
        or len(value) != 3
        or any(
            isinstance(component, bool) or not isinstance(component, int)
            for component in value
        )
    ):
        raise TsdfError(
            f"TSDF observation block ray {label} must contain three integers"
        )
    for component in value:
        _validate_block_index(component)


def _validate_canonical_block_tuple(value: object, label: str) -> None:
    if not isinstance(value, tuple):
        raise TsdfError(
            f"TSDF observation ray trace {label} must be a tuple"
        )
    previous_key: tuple[int, int, int] | None = None
    for block_index in value:
        _validate_block_index_xyz(block_index, label)
        key = (block_index[2], block_index[1], block_index[0])
        if previous_key is not None and key <= previous_key:
            raise TsdfError(
                f"TSDF observation ray trace {label} must be unique and "
                "strictly x-fastest ordered"
            )
        previous_key = key
