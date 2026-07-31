"""Read-only evaluation of one planned voxel against one depth observation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from .errors import PointCloudError, TsdfError
from .model import CameraCalibration, ScanSession
from .point_cloud import (
    _read_depth,
    _sample_path,
    _validate_reconstruction_contract,
)
from .replay import replay_session
from .tsdf_block_plan_loader import TsdfBlockPlan
from .tsdf_block_storage import TsdfBlockStorage
from .tsdf_replay_depth_context import (
    TsdfReplayDepthContext,
    TsdfReplayDepthStatus,
)
from .tsdf_voxel_address import TsdfVoxelAddress, locate_tsdf_voxel

_Point2 = tuple[float, float]
_Point3 = tuple[float, float, float]
_Pixel2 = tuple[int, int]


def _is_finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


class TsdfContributionStatus(StrEnum):
    """Stable result categories for one read-only projective evaluation."""

    CONTRIBUTES = "contributes"
    MISSING_DEPTH = "missing-depth"
    MISSING_POSE = "missing-pose"
    MISSING_DEPTH_AND_POSE = "missing-depth-and-pose"
    CAMERA_POINT_NONFINITE = "camera-point-nonfinite"
    CAMERA_Z_NONPOSITIVE = "camera-z-nonpositive"
    PROJECTION_NONFINITE = "projection-nonfinite"
    PROJECTION_OUTSIDE_IMAGE = "projection-outside-image"
    DEPTH_INVALID = "depth-invalid"
    SIGNED_DISTANCE_NONFINITE = "signed-distance-nonfinite"
    BEHIND_TRUNCATION = "behind-truncation"


@dataclass(frozen=True, slots=True)
class TsdfVoxelContribution:
    """Immutable contribution or skip diagnostic for one voxel and frame."""

    address: TsdfVoxelAddress
    observation_sequence: int
    source_plan_digest_sha256: str
    replay_digest_sha256: str
    status: TsdfContributionStatus
    world_xyz_m: _Point3
    camera_xyz_m: _Point3 | None = None
    projected_uv: _Point2 | None = None
    pixel_uv: _Pixel2 | None = None
    depth_decoded: bool = False
    measured_depth_m: float | None = None
    signed_distance_m: float | None = None
    tsdf_sum_delta: float | None = None
    weight_delta: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.address, TsdfVoxelAddress):
            raise TsdfError(
                "TSDF voxel contribution requires a TsdfVoxelAddress"
            )
        if (
            isinstance(self.observation_sequence, bool)
            or not isinstance(self.observation_sequence, int)
            or self.observation_sequence < 0
        ):
            raise TsdfError(
                "TSDF voxel contribution observation sequence must be "
                "a non-negative integer"
            )
        if not isinstance(self.status, TsdfContributionStatus):
            raise TsdfError(
                "TSDF voxel contribution status is not recognized"
            )
        if not _is_sha256(self.source_plan_digest_sha256):
            raise TsdfError(
                "TSDF voxel contribution source plan digest is invalid"
            )
        if not _is_sha256(self.replay_digest_sha256):
            raise TsdfError(
                "TSDF voxel contribution source replay digest is invalid"
            )
        _validate_result_point(self.world_xyz_m, "world_xyz_m")
        if self.camera_xyz_m is not None:
            _validate_result_point(self.camera_xyz_m, "camera_xyz_m")
        if self.projected_uv is not None:
            _validate_result_point(self.projected_uv, "projected_uv", size=2)
        if self.pixel_uv is not None and (
            not isinstance(self.pixel_uv, tuple)
            or len(self.pixel_uv) != 2
            or any(
                isinstance(component, bool)
                or not isinstance(component, int)
                or component < 0
                for component in self.pixel_uv
            )
        ):
            raise TsdfError(
                "TSDF voxel contribution pixel_uv must contain two "
                "non-negative integers"
            )
        if not isinstance(self.depth_decoded, bool):
            raise TsdfError(
                "TSDF voxel contribution depth_decoded must be boolean"
            )
        for value, label in (
            (self.measured_depth_m, "measured_depth_m"),
            (self.signed_distance_m, "signed_distance_m"),
            (self.tsdf_sum_delta, "tsdf_sum_delta"),
        ):
            if value is not None and not _is_finite_number(value):
                raise TsdfError(
                    f"TSDF voxel contribution {label} must be finite"
                )
        if (
            isinstance(self.weight_delta, bool)
            or not isinstance(self.weight_delta, int)
            or self.weight_delta not in (0, 1)
        ):
            raise TsdfError(
                "TSDF voxel contribution weight_delta must be 0 or 1"
            )
        if self.status is TsdfContributionStatus.CONTRIBUTES:
            if (
                not self.depth_decoded
                or self.camera_xyz_m is None
                or self.projected_uv is None
                or self.pixel_uv is None
                or self.measured_depth_m is None
                or self.measured_depth_m <= 0.0
                or self.signed_distance_m is None
                or self.tsdf_sum_delta is None
                or not -1.0 <= self.tsdf_sum_delta <= 1.0
                or self.weight_delta != 1
            ):
                raise TsdfError(
                    "contributing TSDF voxel result is internally "
                    "inconsistent"
                )
        elif self.tsdf_sum_delta is not None or self.weight_delta != 0:
            raise TsdfError(
                "skipped TSDF voxel result cannot carry a contribution"
            )

    @property
    def contributes(self) -> bool:
        return self.status is TsdfContributionStatus.CONTRIBUTES


def evaluate_tsdf_voxel_contribution(
    storage: TsdfBlockStorage,
    address: TsdfVoxelAddress,
    session: ScanSession,
    observation_sequence: int,
) -> TsdfVoxelContribution:
    """Evaluate one replay-selected observation without applying its result."""

    if not isinstance(storage, TsdfBlockStorage):
        raise TsdfError(
            "TSDF contribution evaluation requires allocated "
            "TsdfBlockStorage"
        )
    if not isinstance(address, TsdfVoxelAddress):
        raise TsdfError(
            "TSDF contribution evaluation requires a TsdfVoxelAddress"
        )
    if not isinstance(session, ScanSession):
        raise TsdfError(
            "TSDF contribution evaluation requires a loaded ScanSession"
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

    resolved_address = locate_tsdf_voxel(
        storage,
        address.global_index_xyz,
    )
    if resolved_address != address:
        raise TsdfError(
            "TSDF voxel address does not match the allocated block storage"
        )

    plan = storage.source_plan
    _validate_contribution_plan(plan)
    if plan.session_id != session.session_id:
        raise TsdfError(
            "TSDF block plan session_id does not match the loaded session: "
            f"{plan.session_id!r} != {session.session_id!r}"
        )
    starting_replay = replay_session(session)
    if starting_replay.digest_sha256 != plan.replay_digest_sha256:
        raise TsdfError(
            "TSDF block plan replay digest does not match current session "
            "inputs; regenerate the plan"
        )
    if observation_sequence >= len(starting_replay.observations):
        raise TsdfError(
            "observation_sequence: outside replay range "
            f"[0, {len(starting_replay.observations) - 1}]"
        )
    if observation_sequence % plan.frame_stride != 0:
        raise TsdfError(
            "observation_sequence: not selected by the block plan's "
            f"frame_stride={plan.frame_stride}"
        )

    try:
        camera, depth_scale_m = _validate_reconstruction_contract(session)
    except PointCloudError as error:
        raise TsdfError(str(error)) from error

    observation = starting_replay.observations[observation_sequence]
    world_xyz_m = _voxel_center_world_m(
        address.global_index_xyz,
        plan.voxel_size_m,
    )
    if observation.depth is None and observation.pose is None:
        result = _skipped(
            address,
            observation_sequence,
            plan.artifact_digest_sha256,
            plan.replay_digest_sha256,
            TsdfContributionStatus.MISSING_DEPTH_AND_POSE,
            world_xyz_m,
        )
    elif observation.depth is None:
        result = _skipped(
            address,
            observation_sequence,
            plan.artifact_digest_sha256,
            plan.replay_digest_sha256,
            TsdfContributionStatus.MISSING_DEPTH,
            world_xyz_m,
        )
    elif observation.pose is None:
        result = _skipped(
            address,
            observation_sequence,
            plan.artifact_digest_sha256,
            plan.replay_digest_sha256,
            TsdfContributionStatus.MISSING_POSE,
            world_xyz_m,
        )
    else:
        transform = tuple(observation.pose.data["T_world_camera"])
        result = _evaluate_complete_observation(
            address,
            observation_sequence,
            plan.artifact_digest_sha256,
            plan.replay_digest_sha256,
            plan.truncation_m,
            session,
            observation.depth.data,
            camera,
            depth_scale_m,
            transform,
            world_xyz_m,
        )

    ending_replay = replay_session(session)
    if (
        ending_replay.digest_sha256 != starting_replay.digest_sha256
        or ending_replay.digest_sha256 != plan.replay_digest_sha256
    ):
        raise TsdfError(
            "session inputs changed while evaluating a TSDF contribution; "
            "rerun the command"
        )
    return result


def evaluate_tsdf_voxel_contribution_from_context(
    storage: TsdfBlockStorage,
    address: TsdfVoxelAddress,
    context: TsdfReplayDepthContext,
    observation_sequence: int,
) -> TsdfVoxelContribution:
    """Evaluate one prepared observation without replay or depth I/O."""

    if not isinstance(storage, TsdfBlockStorage):
        raise TsdfError(
            "TSDF context contribution evaluation requires allocated "
            "TsdfBlockStorage"
        )
    if not isinstance(address, TsdfVoxelAddress):
        raise TsdfError(
            "TSDF context contribution evaluation requires a "
            "TsdfVoxelAddress"
        )
    if not isinstance(context, TsdfReplayDepthContext):
        raise TsdfError(
            "TSDF context contribution evaluation requires a prepared "
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

    resolved_address = locate_tsdf_voxel(
        storage,
        address.global_index_xyz,
    )
    if resolved_address != address:
        raise TsdfError(
            "TSDF voxel address does not match the allocated block storage"
        )

    plan = storage.source_plan
    _validate_contribution_plan(plan)
    _validate_contribution_context(plan, context)
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
    world_xyz_m = _voxel_center_world_m(
        address.global_index_xyz,
        plan.voxel_size_m,
    )
    if observation.status is TsdfReplayDepthStatus.MISSING_DEPTH_AND_POSE:
        return _skipped(
            address,
            observation_sequence,
            plan.artifact_digest_sha256,
            plan.replay_digest_sha256,
            TsdfContributionStatus.MISSING_DEPTH_AND_POSE,
            world_xyz_m,
        )
    if observation.status is TsdfReplayDepthStatus.MISSING_DEPTH:
        return _skipped(
            address,
            observation_sequence,
            plan.artifact_digest_sha256,
            plan.replay_digest_sha256,
            TsdfContributionStatus.MISSING_DEPTH,
            world_xyz_m,
        )
    if observation.status is TsdfReplayDepthStatus.MISSING_POSE:
        return _skipped(
            address,
            observation_sequence,
            plan.artifact_digest_sha256,
            plan.replay_digest_sha256,
            TsdfContributionStatus.MISSING_POSE,
            world_xyz_m,
        )

    transform = observation.t_world_camera
    depth_m = observation.depth_m
    if transform is None or depth_m is None:
        raise TsdfError(
            "ready TSDF replay/depth observation is incomplete"
        )
    return _evaluate_metric_observation(
        address,
        observation_sequence,
        plan.artifact_digest_sha256,
        plan.replay_digest_sha256,
        plan.truncation_m,
        context.camera,
        transform,
        depth_m,
        world_xyz_m,
    )


def _evaluate_complete_observation(
    address: TsdfVoxelAddress,
    observation_sequence: int,
    source_plan_digest_sha256: str,
    replay_digest_sha256: str,
    truncation_m: float,
    session: ScanSession,
    depth_sample: object,
    camera: CameraCalibration,
    depth_scale_m: float,
    transform: tuple[object, ...],
    world_xyz_m: _Point3,
) -> TsdfVoxelContribution:
    try:
        depth_path = _sample_path(session, depth_sample, "depth")
        depth_values = np.asarray(
            _read_depth(depth_path, camera.width, camera.height),
            dtype=np.float64,
        ).reshape((camera.height, camera.width))
    except PointCloudError as error:
        raise TsdfError(str(error)) from error

    with np.errstate(over="ignore", invalid="ignore"):
        depth_metres = depth_values * depth_scale_m

    return _evaluate_metric_observation(
        address,
        observation_sequence,
        source_plan_digest_sha256,
        replay_digest_sha256,
        truncation_m,
        camera,
        transform,
        depth_metres,
        world_xyz_m,
    )


def _evaluate_metric_observation(
    address: TsdfVoxelAddress,
    observation_sequence: int,
    source_plan_digest_sha256: str,
    replay_digest_sha256: str,
    truncation_m: float,
    camera: CameraCalibration,
    transform: tuple[object, ...],
    depth_metres: np.ndarray,
    world_xyz_m: _Point3,
) -> TsdfVoxelContribution:
    """Evaluate one already-decoded metric depth frame."""

    camera_xyz_m = _world_to_camera(transform, world_xyz_m)
    if camera_xyz_m is None:
        return _skipped(
            address,
            observation_sequence,
            source_plan_digest_sha256,
            replay_digest_sha256,
            TsdfContributionStatus.CAMERA_POINT_NONFINITE,
            world_xyz_m,
            depth_decoded=True,
        )

    x_camera, y_camera, z_camera = camera_xyz_m
    if z_camera <= 0.0:
        return _skipped(
            address,
            observation_sequence,
            source_plan_digest_sha256,
            replay_digest_sha256,
            TsdfContributionStatus.CAMERA_Z_NONPOSITIVE,
            world_xyz_m,
            camera_xyz_m=camera_xyz_m,
            depth_decoded=True,
        )

    projected_u = camera.fx * x_camera / z_camera + camera.cx
    projected_v = camera.fy * y_camera / z_camera + camera.cy
    if not math.isfinite(projected_u) or not math.isfinite(projected_v):
        return _skipped(
            address,
            observation_sequence,
            source_plan_digest_sha256,
            replay_digest_sha256,
            TsdfContributionStatus.PROJECTION_NONFINITE,
            world_xyz_m,
            camera_xyz_m=camera_xyz_m,
            depth_decoded=True,
        )
    projected_uv = (projected_u, projected_v)
    if not (
        -0.5 <= projected_u < camera.width - 0.5
        and -0.5 <= projected_v < camera.height - 0.5
    ):
        return _skipped(
            address,
            observation_sequence,
            source_plan_digest_sha256,
            replay_digest_sha256,
            TsdfContributionStatus.PROJECTION_OUTSIDE_IMAGE,
            world_xyz_m,
            camera_xyz_m=camera_xyz_m,
            projected_uv=projected_uv,
            depth_decoded=True,
        )

    pixel_u = math.floor(projected_u + 0.5)
    pixel_v = math.floor(projected_v + 0.5)
    pixel_uv = (pixel_u, pixel_v)
    if not (
        0 <= pixel_u < camera.width
        and 0 <= pixel_v < camera.height
    ):
        return _skipped(
            address,
            observation_sequence,
            source_plan_digest_sha256,
            replay_digest_sha256,
            TsdfContributionStatus.PROJECTION_OUTSIDE_IMAGE,
            world_xyz_m,
            camera_xyz_m=camera_xyz_m,
            projected_uv=projected_uv,
            depth_decoded=True,
        )

    measured_depth_m = float(depth_metres[pixel_v, pixel_u])
    if measured_depth_m <= 0.0 or not math.isfinite(measured_depth_m):
        return _skipped(
            address,
            observation_sequence,
            source_plan_digest_sha256,
            replay_digest_sha256,
            TsdfContributionStatus.DEPTH_INVALID,
            world_xyz_m,
            camera_xyz_m=camera_xyz_m,
            projected_uv=projected_uv,
            pixel_uv=pixel_uv,
            depth_decoded=True,
        )

    signed_distance_m = measured_depth_m - z_camera
    if not math.isfinite(signed_distance_m):
        return _skipped(
            address,
            observation_sequence,
            source_plan_digest_sha256,
            replay_digest_sha256,
            TsdfContributionStatus.SIGNED_DISTANCE_NONFINITE,
            world_xyz_m,
            camera_xyz_m=camera_xyz_m,
            projected_uv=projected_uv,
            pixel_uv=pixel_uv,
            depth_decoded=True,
            measured_depth_m=measured_depth_m,
        )

    if signed_distance_m < -truncation_m:
        return _skipped(
            address,
            observation_sequence,
            source_plan_digest_sha256,
            replay_digest_sha256,
            TsdfContributionStatus.BEHIND_TRUNCATION,
            world_xyz_m,
            camera_xyz_m=camera_xyz_m,
            projected_uv=projected_uv,
            pixel_uv=pixel_uv,
            depth_decoded=True,
            measured_depth_m=measured_depth_m,
            signed_distance_m=signed_distance_m,
        )

    with np.errstate(over="ignore", invalid="ignore"):
        tsdf_sum_delta = float(
            np.clip(
                np.float64(signed_distance_m) / np.float64(truncation_m),
                -1.0,
                1.0,
            )
        )
    if not math.isfinite(tsdf_sum_delta):
        raise TsdfError("TSDF contribution must remain finite")
    return TsdfVoxelContribution(
        address=address,
        observation_sequence=observation_sequence,
        source_plan_digest_sha256=source_plan_digest_sha256,
        replay_digest_sha256=replay_digest_sha256,
        status=TsdfContributionStatus.CONTRIBUTES,
        world_xyz_m=world_xyz_m,
        camera_xyz_m=camera_xyz_m,
        projected_uv=projected_uv,
        pixel_uv=pixel_uv,
        depth_decoded=True,
        measured_depth_m=measured_depth_m,
        signed_distance_m=signed_distance_m,
        tsdf_sum_delta=tsdf_sum_delta,
        weight_delta=1,
    )


def _voxel_center_world_m(
    global_index_xyz: tuple[int, int, int],
    voxel_size_m: float,
) -> _Point3:
    try:
        world_xyz_m = tuple(
            (component + 0.5) * voxel_size_m
            for component in global_index_xyz
        )
    except OverflowError as error:
        raise TsdfError(
            "TSDF voxel center must remain finite in world coordinates"
        ) from error
    if not all(math.isfinite(component) for component in world_xyz_m):
        raise TsdfError(
            "TSDF voxel center must remain finite in world coordinates"
        )
    return world_xyz_m  # type: ignore[return-value]


def _validate_contribution_plan(plan: TsdfBlockPlan) -> None:
    if not isinstance(plan, TsdfBlockPlan):
        raise TsdfError(
            "TSDF contribution evaluation requires a loaded TsdfBlockPlan"
        )
    for value, label in (
        (plan.voxel_size_m, "voxel_size_m"),
        (plan.truncation_m, "truncation_m"),
        (plan.block_extent_m, "block_extent_m"),
    ):
        if (
            not _is_finite_number(value)
            or value <= 0.0
        ):
            raise TsdfError(
                f"TSDF block plan {label} must be finite and positive"
            )
    if plan.truncation_m < plan.voxel_size_m:
        raise TsdfError(
            "TSDF block plan truncation_m must be greater than or equal "
            "to voxel_size_m"
        )
    if (
        isinstance(plan.frame_stride, bool)
        or not isinstance(plan.frame_stride, int)
        or plan.frame_stride < 1
    ):
        raise TsdfError(
            "TSDF block plan frame_stride must be a positive integer"
        )
    for value, label in (
        (plan.total_observations, "total_observations"),
        (plan.selected_observations, "selected_observations"),
        (plan.paired_observations, "paired_observations"),
        (plan.valid_depth_points, "valid_depth_points"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise TsdfError(
                f"TSDF block plan {label} must be a positive integer"
            )
    for value, label in (
        (plan.skipped_missing_depth, "skipped_missing_depth"),
        (plan.skipped_missing_pose, "skipped_missing_pose"),
        (plan.invalid_depth_samples, "invalid_depth_samples"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise TsdfError(
                f"TSDF block plan {label} must be a nonnegative integer"
            )
    expected_selected = (
        (plan.total_observations - 1) // plan.frame_stride
    ) + 1
    if plan.selected_observations != expected_selected:
        raise TsdfError(
            "TSDF block plan selected observations do not match its stride"
        )
    if plan.paired_observations > plan.selected_observations:
        raise TsdfError(
            "TSDF block plan paired observations exceed selection"
        )
    skipped = plan.selected_observations - plan.paired_observations
    if not (
        max(plan.skipped_missing_depth, plan.skipped_missing_pose)
        <= skipped
        <= plan.skipped_missing_depth + plan.skipped_missing_pose
    ):
        raise TsdfError(
            "TSDF block plan missing-input counts are inconsistent"
        )
    expected_extent = plan.voxel_size_m * plan.block_resolution
    if (
        not math.isfinite(expected_extent)
        or plan.block_extent_m != expected_extent
    ):
        raise TsdfError(
            "TSDF block plan block_extent_m does not match its voxel grid"
        )


def _validate_contribution_context(
    plan: TsdfBlockPlan,
    context: TsdfReplayDepthContext,
) -> None:
    if context.source_plan_digest_sha256 != plan.artifact_digest_sha256:
        raise TsdfError(
            "TSDF replay/depth context source plan digest does not match "
            "block storage"
        )
    if context.replay_digest_sha256 != plan.replay_digest_sha256:
        raise TsdfError(
            "TSDF replay/depth context replay digest does not match block "
            "storage"
        )
    if context.session_id != plan.session_id:
        raise TsdfError(
            "TSDF replay/depth context session_id does not match block "
            "storage"
        )

    if (
        context.frame_stride != plan.frame_stride
        or context.total_observations != plan.total_observations
        or context.selected_observation_count != plan.selected_observations
    ):
        raise TsdfError(
            "TSDF replay/depth context selection does not match block plan"
        )
    if (
        context.valid_depth_samples != plan.valid_depth_points
        or context.invalid_depth_samples != plan.invalid_depth_samples
    ):
        raise TsdfError(
            "TSDF replay/depth context sample counts do not match block plan"
        )
    expected_samples = (
        plan.paired_observations
        * context.camera.width
        * context.camera.height
    )
    if (
        plan.valid_depth_points + plan.invalid_depth_samples
        != expected_samples
    ):
        raise TsdfError(
            "TSDF block plan depth samples do not match context camera"
        )


def _validate_result_point(
    value: object,
    label: str,
    *,
    size: int = 3,
) -> None:
    if (
        not isinstance(value, tuple)
        or len(value) != size
        or any(
            not _is_finite_number(component)
            for component in value
        )
    ):
        raise TsdfError(
            f"TSDF voxel contribution {label} must contain "
            f"{size} finite numeric components"
        )


def _world_to_camera(
    transform: tuple[object, ...],
    world_xyz_m: _Point3,
) -> _Point3 | None:
    if len(transform) != 16 or any(
        not _is_finite_number(component)
        for component in transform
    ):
        raise TsdfError(
            "T_world_camera must contain 16 finite numeric components"
        )
    x_world, y_world, z_world = world_xyz_m
    delta_x = x_world - float(transform[3])
    delta_y = y_world - float(transform[7])
    delta_z = z_world - float(transform[11])
    x_camera = (
        float(transform[0]) * delta_x
        + float(transform[4]) * delta_y
        + float(transform[8]) * delta_z
    )
    y_camera = (
        float(transform[1]) * delta_x
        + float(transform[5]) * delta_y
        + float(transform[9]) * delta_z
    )
    z_camera = (
        float(transform[2]) * delta_x
        + float(transform[6]) * delta_y
        + float(transform[10]) * delta_z
    )
    if not all(
        math.isfinite(component)
        for component in (x_camera, y_camera, z_camera)
    ):
        return None
    return (x_camera, y_camera, z_camera)


def _skipped(
    address: TsdfVoxelAddress,
    observation_sequence: int,
    source_plan_digest_sha256: str,
    replay_digest_sha256: str,
    status: TsdfContributionStatus,
    world_xyz_m: _Point3,
    *,
    camera_xyz_m: _Point3 | None = None,
    projected_uv: _Point2 | None = None,
    pixel_uv: _Pixel2 | None = None,
    depth_decoded: bool = False,
    measured_depth_m: float | None = None,
    signed_distance_m: float | None = None,
) -> TsdfVoxelContribution:
    return TsdfVoxelContribution(
        address=address,
        observation_sequence=observation_sequence,
        source_plan_digest_sha256=source_plan_digest_sha256,
        replay_digest_sha256=replay_digest_sha256,
        status=status,
        world_xyz_m=world_xyz_m,
        camera_xyz_m=camera_xyz_m,
        projected_uv=projected_uv,
        pixel_uv=pixel_uv,
        depth_decoded=depth_decoded,
        measured_depth_m=measured_depth_m,
        signed_distance_m=signed_distance_m,
    )
