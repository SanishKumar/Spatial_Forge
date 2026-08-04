"""Per-voxel sampling classification for one prepared TSDF observation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from .errors import TsdfError
from .tsdf_block_plan import TSDF_BLOCK_RESOLUTION
from .tsdf_block_plan_loader import TsdfBlockPlan
from .tsdf_observation_block_rays import (
    _is_finite_number,
    _is_sha256,
    _validate_block_index_xyz,
    _validate_image_size,
    _validate_point,
    _validate_trace_plan,
)
from .tsdf_replay_depth_context import (
    TsdfReplayDepthContext,
    TsdfReplayDepthStatus,
)
from .tsdf_voxel_address import _split_tsdf_global_voxel_index
from .tsdf_voxel_contribution import (
    _validate_contribution_context,
    _voxel_center_world_m,
    _world_to_camera,
)

_Index3 = tuple[int, int, int]
_Point2 = tuple[float, float]
_Point3 = tuple[float, float, float]


class TsdfVoxelSamplingStatus(StrEnum):
    """Stable per-voxel sampling outcomes for one prepared observation."""

    OBSERVED_FREE_SPACE = "observed-free-space"
    OBSERVED_SURFACE_BAND = "observed-surface-band"
    UNOBSERVED_OCCLUDED = "unobserved-occluded"
    UNOBSERVED_BEHIND_CAMERA = "unobserved-behind-camera"
    UNOBSERVED_OUTSIDE_IMAGE = "unobserved-outside-image"
    UNOBSERVED_DEPTH_INVALID = "unobserved-depth-invalid"
    UNOBSERVED_NONFINITE = "unobserved-nonfinite"
    MISSING_DEPTH = "missing-depth"
    MISSING_POSE = "missing-pose"
    MISSING_DEPTH_AND_POSE = "missing-depth-and-pose"


_MISSING_INPUT_STATUSES = {
    TsdfReplayDepthStatus.MISSING_DEPTH:
        TsdfVoxelSamplingStatus.MISSING_DEPTH,
    TsdfReplayDepthStatus.MISSING_POSE:
        TsdfVoxelSamplingStatus.MISSING_POSE,
    TsdfReplayDepthStatus.MISSING_DEPTH_AND_POSE:
        TsdfVoxelSamplingStatus.MISSING_DEPTH_AND_POSE,
}

_SAMPLED_STATUSES = (
    TsdfVoxelSamplingStatus.OBSERVED_FREE_SPACE,
    TsdfVoxelSamplingStatus.OBSERVED_SURFACE_BAND,
    TsdfVoxelSamplingStatus.UNOBSERVED_OCCLUDED,
)


@dataclass(frozen=True, slots=True)
class TsdfVoxelSamplingReceipt:
    """Immutable sampling classification for one voxel and observation."""

    source_plan_digest_sha256: str
    replay_digest_sha256: str
    frame_stride: int
    total_observations: int
    observation_sequence: int
    observation_status: TsdfReplayDepthStatus
    global_index_xyz: _Index3
    block_index_xyz: _Index3
    local_index_xyz: _Index3
    planned_block: bool
    voxel_size_m: float
    truncation_m: float
    block_resolution: int
    image_size: tuple[int, int]
    status: TsdfVoxelSamplingStatus
    world_xyz_m: _Point3
    camera_xyz_m: _Point3 | None
    projected_uv: _Point2 | None
    pixel_uv: tuple[int, int] | None
    measured_depth_m: float | None
    signed_distance_m: float | None
    truncated_tsdf_value: float | None

    def __post_init__(self) -> None:
        if not _is_sha256(self.source_plan_digest_sha256):
            raise TsdfError(
                "TSDF voxel sampling source plan digest is invalid"
            )
        if not _is_sha256(self.replay_digest_sha256):
            raise TsdfError("TSDF voxel sampling replay digest is invalid")
        for value, label in (
            (self.frame_stride, "frame stride"),
            (self.total_observations, "total observations"),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
            ):
                raise TsdfError(
                    f"TSDF voxel sampling {label} must be positive"
                )
        if (
            isinstance(self.observation_sequence, bool)
            or not isinstance(self.observation_sequence, int)
            or self.observation_sequence < 0
        ):
            raise TsdfError("TSDF voxel sampling sequence is invalid")
        if self.observation_sequence >= self.total_observations:
            raise TsdfError(
                "TSDF voxel sampling sequence is outside its total "
                "observation range"
            )
        if self.observation_sequence % self.frame_stride != 0:
            raise TsdfError(
                "TSDF voxel sampling sequence is not selected by its frame "
                "stride"
            )
        if not isinstance(self.observation_status, TsdfReplayDepthStatus):
            raise TsdfError(
                "TSDF voxel sampling prepared status is invalid"
            )
        if not isinstance(self.status, TsdfVoxelSamplingStatus):
            raise TsdfError("TSDF voxel sampling status is invalid")
        if isinstance(self.planned_block, np.bool_) or not isinstance(
            self.planned_block,
            bool,
        ):
            raise TsdfError(
                "TSDF voxel sampling planned_block must be a bool"
            )
        if (
            isinstance(self.block_resolution, bool)
            or not isinstance(self.block_resolution, int)
            or self.block_resolution != TSDF_BLOCK_RESOLUTION
        ):
            raise TsdfError(
                "TSDF voxel sampling requires block resolution "
                f"{TSDF_BLOCK_RESOLUTION}"
            )
        for value, label in (
            (self.voxel_size_m, "voxel size"),
            (self.truncation_m, "truncation"),
        ):
            if not _is_finite_number(value) or value <= 0.0:
                raise TsdfError(
                    f"TSDF voxel sampling {label} must be finite and positive"
                )
        _validate_image_size(self.image_size)

        expected_global, expected_block, expected_local = (
            _split_tsdf_global_voxel_index(self.global_index_xyz)
        )
        if (
            self.global_index_xyz != expected_global
            or self.block_index_xyz != expected_block
            or self.local_index_xyz != expected_local
        ):
            raise TsdfError(
                "TSDF voxel sampling address decomposition is inconsistent"
            )
        _validate_block_index_xyz(self.block_index_xyz, "block index")
        expected_world = _voxel_center_world_m(
            self.global_index_xyz,
            self.voxel_size_m,
        )
        _validate_point(self.world_xyz_m, "voxel centre")
        if self.world_xyz_m != expected_world:
            raise TsdfError(
                "TSDF voxel sampling world centre does not match its global "
                "index and voxel size"
            )
        _validate_expected_sampling_status(
            self.status,
            self.observation_status,
        )
        self._validate_retained_stage()

    def _validate_retained_stage(self) -> None:
        camera_retained = self.status not in (
            TsdfVoxelSamplingStatus.MISSING_POSE,
            TsdfVoxelSamplingStatus.MISSING_DEPTH_AND_POSE,
            TsdfVoxelSamplingStatus.MISSING_DEPTH,
        )
        if not camera_retained:
            if (
                self.camera_xyz_m is not None
                or self.projected_uv is not None
                or self.pixel_uv is not None
                or self.measured_depth_m is not None
                or self.signed_distance_m is not None
                or self.truncated_tsdf_value is not None
            ):
                raise TsdfError(
                    "input-missing TSDF voxel sampling cannot retain "
                    "projection geometry"
                )
            return

        if self.status is TsdfVoxelSamplingStatus.UNOBSERVED_NONFINITE:
            return
        _validate_point(self.camera_xyz_m, "camera point")
        if self.camera_xyz_m is None:
            raise AssertionError("sampled voxel must retain a camera point")
        camera_z = self.camera_xyz_m[2]

        if self.status is TsdfVoxelSamplingStatus.UNOBSERVED_BEHIND_CAMERA:
            if camera_z > 0.0:
                raise TsdfError(
                    "behind-camera TSDF voxel sampling requires a "
                    "nonpositive camera depth"
                )
            if (
                self.projected_uv is not None
                or self.pixel_uv is not None
                or self.measured_depth_m is not None
                or self.signed_distance_m is not None
                or self.truncated_tsdf_value is not None
            ):
                raise TsdfError(
                    "behind-camera TSDF voxel sampling cannot retain a "
                    "projection"
                )
            return
        if camera_z <= 0.0:
            raise TsdfError(
                "projected TSDF voxel sampling requires a positive camera "
                "depth"
            )
        if (
            not isinstance(self.projected_uv, tuple)
            or len(self.projected_uv) != 2
            or any(
                not _is_finite_number(component)
                for component in self.projected_uv
            )
        ):
            raise TsdfError(
                "projected TSDF voxel sampling requires two finite image "
                "coordinates"
            )

        if self.status is TsdfVoxelSamplingStatus.UNOBSERVED_OUTSIDE_IMAGE:
            if (
                self.pixel_uv is not None
                or self.measured_depth_m is not None
                or self.signed_distance_m is not None
                or self.truncated_tsdf_value is not None
            ):
                raise TsdfError(
                    "outside-image TSDF voxel sampling cannot retain a "
                    "sampled pixel"
                )
            if _pixel_inside_image(self.projected_uv, self.image_size):
                raise TsdfError(
                    "outside-image TSDF voxel sampling requires a projection "
                    "outside its camera image"
                )
            return

        expected_pixel = _nearest_pixel(self.projected_uv)
        if self.pixel_uv != expected_pixel:
            raise TsdfError(
                "TSDF voxel sampling pixel does not match its own nearest-"
                "pixel projection"
            )
        if not _pixel_inside_image(self.projected_uv, self.image_size):
            raise TsdfError(
                "sampled TSDF voxel sampling requires a projection inside "
                "its camera image"
            )

        if self.status is TsdfVoxelSamplingStatus.UNOBSERVED_DEPTH_INVALID:
            if (
                self.measured_depth_m is not None
                or self.signed_distance_m is not None
                or self.truncated_tsdf_value is not None
            ):
                raise TsdfError(
                    "depth-invalid TSDF voxel sampling cannot retain a "
                    "measurement"
                )
            return

        if (
            not _is_finite_number(self.measured_depth_m)
            or self.measured_depth_m <= 0.0
        ):
            raise TsdfError(
                "sampled TSDF voxel sampling depth must be finite and "
                "positive"
            )
        if self.measured_depth_m is None:
            raise AssertionError("sampled voxel must retain a depth")
        expected_signed = self.measured_depth_m - camera_z
        if (
            not _is_finite_number(self.signed_distance_m)
            or self.signed_distance_m != expected_signed
        ):
            raise TsdfError(
                "TSDF voxel sampling signed distance does not match its own "
                "measurement and camera depth"
            )
        if self.signed_distance_m is None:
            raise AssertionError("sampled voxel must retain a distance")
        expected_status = _classify_signed_distance(
            self.signed_distance_m,
            self.truncation_m,
        )
        if self.status is not expected_status:
            raise TsdfError(
                "TSDF voxel sampling status does not match its own signed "
                "distance and truncation"
            )
        expected_value = _truncated_tsdf_value(
            self.signed_distance_m,
            self.truncation_m,
        )
        if self.truncated_tsdf_value != expected_value:
            raise TsdfError(
                "TSDF voxel sampling truncated value does not match its own "
                "signed distance and truncation"
            )

    @property
    def sampled(self) -> bool:
        return self.status in _SAMPLED_STATUSES

    @property
    def observed(self) -> bool:
        return self.status in (
            TsdfVoxelSamplingStatus.OBSERVED_FREE_SPACE,
            TsdfVoxelSamplingStatus.OBSERVED_SURFACE_BAND,
        )

    @property
    def inside_sampling_wedge(self) -> bool:
        """Report whether the centre lies in its pixel's closed wedge."""

        if (
            self.camera_xyz_m is None
            or self.measured_depth_m is None
            or self.pixel_uv is None
        ):
            return False
        camera_z = self.camera_xyz_m[2]
        return 0.0 < camera_z <= self.measured_depth_m

    @property
    def contributes_to_reference_tsdf(self) -> bool:
        """Report agreement with the existing evaluator's accept rule."""

        return self.status in (
            TsdfVoxelSamplingStatus.OBSERVED_FREE_SPACE,
            TsdfVoxelSamplingStatus.OBSERVED_SURFACE_BAND,
        )

    @property
    def prepared_depth_accessed(self) -> bool:
        return self.observation_status is TsdfReplayDepthStatus.READY


def classify_tsdf_voxel_sampling_from_context(
    plan: TsdfBlockPlan,
    context: TsdfReplayDepthContext,
    observation_sequence: int,
    global_index_xyz: _Index3,
) -> TsdfVoxelSamplingReceipt:
    """Classify how one observation samples one voxel centre.

    The voxel does not need to belong to the plan's active blocks.
    """

    if not isinstance(plan, TsdfBlockPlan):
        raise TsdfError(
            "TSDF voxel sampling requires a loaded TsdfBlockPlan"
        )
    if not isinstance(context, TsdfReplayDepthContext):
        raise TsdfError(
            "TSDF voxel sampling requires a prepared TsdfReplayDepthContext"
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
        global_index, block_index, local_index = (
            _split_tsdf_global_voxel_index(global_index_xyz)
        )
        camera = context.camera
        image_size = (camera.width, camera.height)
        world_xyz_m = _voxel_center_world_m(global_index, plan.voxel_size_m)
        planned_block = block_index in set(plan.active_blocks)

        def build(
            status: TsdfVoxelSamplingStatus,
            *,
            camera_xyz_m: _Point3 | None = None,
            projected_uv: _Point2 | None = None,
            pixel_uv: tuple[int, int] | None = None,
            measured_depth_m: float | None = None,
            signed_distance_m: float | None = None,
            truncated_tsdf_value: float | None = None,
        ) -> TsdfVoxelSamplingReceipt:
            return TsdfVoxelSamplingReceipt(
                source_plan_digest_sha256=plan.artifact_digest_sha256,
                replay_digest_sha256=plan.replay_digest_sha256,
                frame_stride=plan.frame_stride,
                total_observations=plan.total_observations,
                observation_sequence=observation_sequence,
                observation_status=observation.status,
                global_index_xyz=global_index,
                block_index_xyz=block_index,
                local_index_xyz=local_index,
                planned_block=planned_block,
                voxel_size_m=plan.voxel_size_m,
                truncation_m=plan.truncation_m,
                block_resolution=plan.block_resolution,
                image_size=image_size,
                status=status,
                world_xyz_m=world_xyz_m,
                camera_xyz_m=camera_xyz_m,
                projected_uv=projected_uv,
                pixel_uv=pixel_uv,
                measured_depth_m=measured_depth_m,
                signed_distance_m=signed_distance_m,
                truncated_tsdf_value=truncated_tsdf_value,
            )

        if observation.status is not TsdfReplayDepthStatus.READY:
            return build(_MISSING_INPUT_STATUSES[observation.status])

        transform = observation.t_world_camera
        depth_m = observation.depth_m
        if transform is None or depth_m is None:
            raise TsdfError(
                "ready TSDF replay/depth observation is incomplete"
            )

        camera_xyz_m = _world_to_camera(transform, world_xyz_m)
        if camera_xyz_m is None:
            return build(TsdfVoxelSamplingStatus.UNOBSERVED_NONFINITE)
        if camera_xyz_m[2] <= 0.0:
            return build(
                TsdfVoxelSamplingStatus.UNOBSERVED_BEHIND_CAMERA,
                camera_xyz_m=camera_xyz_m,
            )

        projected_u = camera.fx * camera_xyz_m[0] / camera_xyz_m[2] + camera.cx
        projected_v = camera.fy * camera_xyz_m[1] / camera_xyz_m[2] + camera.cy
        if not math.isfinite(projected_u) or not math.isfinite(projected_v):
            return build(
                TsdfVoxelSamplingStatus.UNOBSERVED_NONFINITE,
                camera_xyz_m=camera_xyz_m,
            )
        projected_uv = (projected_u, projected_v)
        if not _pixel_inside_image(projected_uv, image_size):
            return build(
                TsdfVoxelSamplingStatus.UNOBSERVED_OUTSIDE_IMAGE,
                camera_xyz_m=camera_xyz_m,
                projected_uv=projected_uv,
            )

        pixel_uv = _nearest_pixel(projected_uv)
        measured_depth_m = float(depth_m[pixel_uv[1], pixel_uv[0]])
        if measured_depth_m <= 0.0 or not math.isfinite(measured_depth_m):
            return build(
                TsdfVoxelSamplingStatus.UNOBSERVED_DEPTH_INVALID,
                camera_xyz_m=camera_xyz_m,
                projected_uv=projected_uv,
                pixel_uv=pixel_uv,
            )

        signed_distance_m = measured_depth_m - camera_xyz_m[2]
        if not math.isfinite(signed_distance_m):
            return build(
                TsdfVoxelSamplingStatus.UNOBSERVED_NONFINITE,
                camera_xyz_m=camera_xyz_m,
            )
        return build(
            _classify_signed_distance(signed_distance_m, plan.truncation_m),
            camera_xyz_m=camera_xyz_m,
            projected_uv=projected_uv,
            pixel_uv=pixel_uv,
            measured_depth_m=measured_depth_m,
            signed_distance_m=signed_distance_m,
            truncated_tsdf_value=_truncated_tsdf_value(
                signed_distance_m,
                plan.truncation_m,
            ),
        )
    except TsdfError:
        raise
    except Exception as error:
        raise TsdfError(
            f"cannot classify prepared TSDF voxel sampling: {error}"
        ) from error


def _classify_signed_distance(
    signed_distance_m: float,
    truncation_m: float,
) -> TsdfVoxelSamplingStatus:
    """Split the evaluator's accept rule into free space and surface band."""

    if signed_distance_m < -truncation_m:
        return TsdfVoxelSamplingStatus.UNOBSERVED_OCCLUDED
    if signed_distance_m > truncation_m:
        return TsdfVoxelSamplingStatus.OBSERVED_FREE_SPACE
    return TsdfVoxelSamplingStatus.OBSERVED_SURFACE_BAND


def _truncated_tsdf_value(
    signed_distance_m: float,
    truncation_m: float,
) -> float:
    with np.errstate(over="ignore", invalid="ignore"):
        value = float(
            np.clip(
                np.float64(signed_distance_m) / np.float64(truncation_m),
                -1.0,
                1.0,
            )
        )
    if not math.isfinite(value):
        raise TsdfError("TSDF voxel sampling value must remain finite")
    return value


def _nearest_pixel(projected_uv: _Point2) -> tuple[int, int]:
    return (
        math.floor(projected_uv[0] + 0.5),
        math.floor(projected_uv[1] + 0.5),
    )


def _pixel_inside_image(
    projected_uv: _Point2,
    image_size: tuple[int, int],
) -> bool:
    if not (
        -0.5 <= projected_uv[0] < image_size[0] - 0.5
        and -0.5 <= projected_uv[1] < image_size[1] - 0.5
    ):
        return False
    pixel_u, pixel_v = _nearest_pixel(projected_uv)
    return 0 <= pixel_u < image_size[0] and 0 <= pixel_v < image_size[1]


def _validate_expected_sampling_status(
    status: TsdfVoxelSamplingStatus,
    observation_status: TsdfReplayDepthStatus,
) -> None:
    if observation_status is TsdfReplayDepthStatus.READY:
        if status in _MISSING_INPUT_STATUSES.values():
            raise TsdfError(
                "ready TSDF voxel sampling cannot report a missing input"
            )
        return
    if status is not _MISSING_INPUT_STATUSES[observation_status]:
        raise TsdfError(
            "TSDF voxel sampling status does not match its prepared "
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
