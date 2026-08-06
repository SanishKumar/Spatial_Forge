"""Vectorised evaluation of one planned TSDF block against one observation.

The scalar evaluator in :mod:`tsdf_voxel_contribution` answers one voxel and
one observation at a time and builds a full validating receipt for each. That
is the reference definition of the projective rule, and it is also why the
block path runs about two orders of magnitude slower than the dense
integrator it is meant to replace.

This module answers the same question for all 512 voxels of one block in a
single NumPy pass. It stays inside float64 and performs every arithmetic step
in the scalar path's exact order, so the two paths agree bit for bit rather
than approximately. Nothing here applies a contribution or touches the
accumulator buffers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .errors import TsdfError
from .model import CameraCalibration
from .tsdf_block_plan import TSDF_BLOCK_RESOLUTION
from .tsdf_block_storage import TSDF_BLOCK_VOXELS, TsdfBlockStorage
from .tsdf_replay_depth_context import (
    TsdfReplayDepthContext,
    TsdfReplayDepthStatus,
    _immutable_bytes_base,
)
from .tsdf_voxel_address import (
    compose_tsdf_global_voxel_index,
    locate_tsdf_voxel,
)
from .tsdf_voxel_contribution import (
    TsdfContributionStatus,
    _is_finite_number,
    _validate_contribution_context,
    _validate_contribution_plan,
)

TSDF_CONTRIBUTION_STATUS_ORDER = tuple(TsdfContributionStatus)
TSDF_CONTRIBUTION_STATUS_CODE_DTYPE = np.dtype(np.uint8)
TSDF_CONTRIBUTION_SUM_DELTA_DTYPE = np.dtype(np.float64)
TSDF_CONTRIBUTION_WEIGHT_DELTA_DTYPE = np.dtype(np.uint32)

_BlockIndex = tuple[int, int, int]

_STATUS_CODE = {
    status: index
    for index, status in enumerate(TSDF_CONTRIBUTION_STATUS_ORDER)
}
_PREPARATION_SKIP_STATUS = {
    TsdfReplayDepthStatus.MISSING_DEPTH: (
        TsdfContributionStatus.MISSING_DEPTH
    ),
    TsdfReplayDepthStatus.MISSING_POSE: (
        TsdfContributionStatus.MISSING_POSE
    ),
    TsdfReplayDepthStatus.MISSING_DEPTH_AND_POSE: (
        TsdfContributionStatus.MISSING_DEPTH_AND_POSE
    ),
}

# Canonical local-flat order, x fastest, matching ``local_flat_index`` and
# therefore also matching a storage row's own C-ordered (z, y, x) layout.
_LOCAL_FLAT = np.arange(TSDF_BLOCK_VOXELS, dtype=np.int64)
_LOCAL_X = _LOCAL_FLAT % TSDF_BLOCK_RESOLUTION
_LOCAL_Y = (_LOCAL_FLAT // TSDF_BLOCK_RESOLUTION) % TSDF_BLOCK_RESOLUTION
_LOCAL_Z = _LOCAL_FLAT // (TSDF_BLOCK_RESOLUTION * TSDF_BLOCK_RESOLUTION)
for _axis in (_LOCAL_FLAT, _LOCAL_X, _LOCAL_Y, _LOCAL_Z):
    _axis.setflags(write=False)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True, slots=True, eq=False)
class TsdfBlockContributionField:
    """Immutable per-voxel evaluation of one block against one observation."""

    block_index_xyz: _BlockIndex
    block_row: int
    observation_sequence: int
    source_plan_digest_sha256: str
    replay_digest_sha256: str
    block_resolution: int
    observation_status: TsdfReplayDepthStatus
    status_codes: np.ndarray
    tsdf_sum_deltas: np.ndarray
    weight_deltas: np.ndarray

    def __post_init__(self) -> None:
        compose_tsdf_global_voxel_index(self.block_index_xyz, (0, 0, 0))
        for value, label in (
            (self.block_row, "block row"),
            (self.observation_sequence, "observation sequence"),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise TsdfError(
                    f"TSDF block contribution field {label} must be a "
                    "non-negative integer"
                )
        if not _is_sha256(self.source_plan_digest_sha256):
            raise TsdfError(
                "TSDF block contribution field source plan digest is invalid"
            )
        if not _is_sha256(self.replay_digest_sha256):
            raise TsdfError(
                "TSDF block contribution field replay digest is invalid"
            )
        if (
            isinstance(self.block_resolution, bool)
            or not isinstance(self.block_resolution, int)
            or self.block_resolution != TSDF_BLOCK_RESOLUTION
        ):
            raise TsdfError(
                "TSDF block contribution field requires block resolution "
                f"{TSDF_BLOCK_RESOLUTION}"
            )
        if not isinstance(self.observation_status, TsdfReplayDepthStatus):
            raise TsdfError(
                "TSDF block contribution field observation status is not "
                "recognized"
            )

        _validate_frozen_array(
            self.status_codes,
            TSDF_CONTRIBUTION_STATUS_CODE_DTYPE,
            "status codes",
        )
        _validate_frozen_array(
            self.tsdf_sum_deltas,
            TSDF_CONTRIBUTION_SUM_DELTA_DTYPE,
            "sum deltas",
        )
        _validate_frozen_array(
            self.weight_deltas,
            TSDF_CONTRIBUTION_WEIGHT_DELTA_DTYPE,
            "weight deltas",
        )
        if int(self.status_codes.max()) >= len(
            TSDF_CONTRIBUTION_STATUS_ORDER
        ):
            raise TsdfError(
                "TSDF block contribution field status code is not recognized"
            )

        contributing = self.status_codes == _STATUS_CODE[
            TsdfContributionStatus.CONTRIBUTES
        ]
        if not np.array_equal(
            self.weight_deltas,
            contributing.astype(TSDF_CONTRIBUTION_WEIGHT_DELTA_DTYPE),
        ):
            raise TsdfError(
                "TSDF block contribution field weight deltas do not match "
                "its contributing voxels"
            )
        accepted = self.tsdf_sum_deltas[contributing]
        if not np.all(np.isfinite(accepted)) or np.any(
            np.abs(accepted) > 1.0
        ):
            raise TsdfError(
                "contributing TSDF block contribution field values must be "
                "finite and within [-1, 1]"
            )
        skipped = self.tsdf_sum_deltas[~contributing]
        if np.any(skipped != 0.0) or np.any(np.signbit(skipped)):
            raise TsdfError(
                "skipped TSDF block contribution field voxels cannot carry "
                "a contribution"
            )

        if self.observation_status is not TsdfReplayDepthStatus.READY:
            expected_code = _STATUS_CODE[
                _PREPARATION_SKIP_STATUS[self.observation_status]
            ]
            if int(self.status_codes.min()) != expected_code or int(
                self.status_codes.max()
            ) != expected_code:
                raise TsdfError(
                    "unprepared TSDF block contribution field must skip "
                    "every voxel with its preparation status"
                )

    @property
    def voxel_count(self) -> int:
        return int(self.status_codes.size)

    @property
    def evaluated_count(self) -> int:
        return self.voxel_count

    @property
    def contributing_count(self) -> int:
        return int(np.count_nonzero(self.weight_deltas))

    @property
    def skipped_count(self) -> int:
        return self.evaluated_count - self.contributing_count

    @property
    def weight_delta_total(self) -> int:
        return int(self.weight_deltas.sum(dtype=np.uint64))

    @property
    def voxel_statuses(self) -> tuple[TsdfContributionStatus, ...]:
        """Materialise the per-voxel statuses in canonical local-flat order."""

        return tuple(
            TSDF_CONTRIBUTION_STATUS_ORDER[code]
            for code in self.status_codes.tolist()
        )

    @property
    def status_counts(
        self,
    ) -> tuple[tuple[TsdfContributionStatus, int], ...]:
        return tuple(
            (status, count)
            for index, status in enumerate(TSDF_CONTRIBUTION_STATUS_ORDER)
            if (
                count := int(
                    np.count_nonzero(self.status_codes == index)
                )
            )
        )


def evaluate_tsdf_block_contributions_from_context(
    storage: TsdfBlockStorage,
    block_index_xyz: _BlockIndex,
    context: TsdfReplayDepthContext,
    observation_sequence: int,
) -> TsdfBlockContributionField:
    """Evaluate all 512 voxels of one planned block in one vector pass."""

    if not isinstance(storage, TsdfBlockStorage):
        raise TsdfError(
            "TSDF block contribution evaluation requires allocated "
            "TsdfBlockStorage"
        )
    compose_tsdf_global_voxel_index(block_index_xyz, (0, 0, 0))
    if not isinstance(context, TsdfReplayDepthContext):
        raise TsdfError(
            "TSDF block contribution evaluation requires a prepared "
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

    first_address = locate_tsdf_voxel(
        storage,
        compose_tsdf_global_voxel_index(block_index_xyz, (0, 0, 0)),
    )
    if first_address is None:
        raise TsdfError(
            f"TSDF block {block_index_xyz} is not planned in destination "
            "storage"
        )
    block_row = first_address.block_row

    observation = context.observations[
        observation_sequence // context.frame_stride
    ]
    if observation.observation_sequence != observation_sequence:
        raise TsdfError(
            "TSDF replay/depth context observation lookup is inconsistent"
        )

    world_xyz_m = _block_voxel_centres_world_m(
        block_index_xyz,
        plan.voxel_size_m,
    )
    if observation.status is TsdfReplayDepthStatus.READY:
        transform = observation.t_world_camera
        depth_m = observation.depth_m
        if transform is None or depth_m is None:
            raise TsdfError(
                "ready TSDF replay/depth observation is incomplete"
            )
        status_codes, sum_deltas, weight_deltas = _evaluate_ready_block(
            context.camera,
            transform,
            depth_m,
            plan.truncation_m,
            world_xyz_m,
        )
    else:
        status_codes = np.full(
            TSDF_BLOCK_VOXELS,
            _STATUS_CODE[_PREPARATION_SKIP_STATUS[observation.status]],
            dtype=TSDF_CONTRIBUTION_STATUS_CODE_DTYPE,
        )
        sum_deltas = np.zeros(
            TSDF_BLOCK_VOXELS,
            dtype=TSDF_CONTRIBUTION_SUM_DELTA_DTYPE,
        )
        weight_deltas = np.zeros(
            TSDF_BLOCK_VOXELS,
            dtype=TSDF_CONTRIBUTION_WEIGHT_DELTA_DTYPE,
        )

    return TsdfBlockContributionField(
        block_index_xyz=block_index_xyz,
        block_row=block_row,
        observation_sequence=observation_sequence,
        source_plan_digest_sha256=plan.artifact_digest_sha256,
        replay_digest_sha256=plan.replay_digest_sha256,
        block_resolution=plan.block_resolution,
        observation_status=observation.status,
        status_codes=_freeze(status_codes),
        tsdf_sum_deltas=_freeze(sum_deltas),
        weight_deltas=_freeze(weight_deltas),
    )


def _evaluate_ready_block(
    camera: CameraCalibration,
    transform: tuple[float, ...],
    depth_m: np.ndarray,
    truncation_m: float,
    world_xyz_m: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project, sample and classify every voxel of one prepared frame.

    Every step below is the scalar evaluator's step in the scalar
    evaluator's order, so the float64 results are bit-identical rather than
    merely close. In particular the world-to-camera product is written out
    term by term instead of as a matrix product, because a fused or
    reassociated dot product would change the last bits.
    """

    if len(transform) != 16 or any(
        not _is_finite_number(component)
        for component in transform
    ):
        raise TsdfError(
            "T_world_camera must contain 16 finite numeric components"
        )
    matrix = [float(component) for component in transform]
    world_x, world_y, world_z = world_xyz_m

    status_codes = np.full(
        TSDF_BLOCK_VOXELS,
        _STATUS_CODE[TsdfContributionStatus.CONTRIBUTES],
        dtype=TSDF_CONTRIBUTION_STATUS_CODE_DTYPE,
    )
    alive = np.ones(TSDF_BLOCK_VOXELS, dtype=bool)

    def classify(
        failed: np.ndarray,
        status: TsdfContributionStatus,
    ) -> None:
        """Assign ``status`` to still-unclassified voxels that failed."""

        hit = alive & failed
        status_codes[hit] = _STATUS_CODE[status]
        np.logical_and(alive, ~hit, out=alive)

    # Voxels that fail an early gate still take part in the later arithmetic;
    # their results are nonsense but are never read, so the usual overflow,
    # invalid and divide-by-zero reports are silenced rather than avoided.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        delta_x = world_x - matrix[3]
        delta_y = world_y - matrix[7]
        delta_z = world_z - matrix[11]
        camera_x = (
            matrix[0] * delta_x
            + matrix[4] * delta_y
            + matrix[8] * delta_z
        )
        camera_y = (
            matrix[1] * delta_x
            + matrix[5] * delta_y
            + matrix[9] * delta_z
        )
        camera_z = (
            matrix[2] * delta_x
            + matrix[6] * delta_y
            + matrix[10] * delta_z
        )
        classify(
            ~(
                np.isfinite(camera_x)
                & np.isfinite(camera_y)
                & np.isfinite(camera_z)
            ),
            TsdfContributionStatus.CAMERA_POINT_NONFINITE,
        )
        classify(
            camera_z <= 0.0,
            TsdfContributionStatus.CAMERA_Z_NONPOSITIVE,
        )

        projected_u = camera.fx * camera_x / camera_z + camera.cx
        projected_v = camera.fy * camera_y / camera_z + camera.cy
        classify(
            ~(np.isfinite(projected_u) & np.isfinite(projected_v)),
            TsdfContributionStatus.PROJECTION_NONFINITE,
        )
        classify(
            ~(
                (projected_u >= -0.5)
                & (projected_u < camera.width - 0.5)
                & (projected_v >= -0.5)
                & (projected_v < camera.height - 0.5)
            ),
            TsdfContributionStatus.PROJECTION_OUTSIDE_IMAGE,
        )

        pixel_u = np.floor(projected_u + 0.5)
        pixel_v = np.floor(projected_v + 0.5)
        classify(
            ~(
                (pixel_u >= 0.0)
                & (pixel_u < camera.width)
                & (pixel_v >= 0.0)
                & (pixel_v < camera.height)
            ),
            TsdfContributionStatus.PROJECTION_OUTSIDE_IMAGE,
        )

        # Only addressable pixels survive to here; the rest are pinned to
        # (0, 0) so the gather stays in bounds, and their samples are
        # discarded immediately afterwards.
        column = np.where(alive, pixel_u, 0.0).astype(np.int64)
        row = np.where(alive, pixel_v, 0.0).astype(np.int64)
        measured_depth_m = np.where(alive, depth_m[row, column], 0.0)
        classify(
            ~(
                np.isfinite(measured_depth_m)
                & (measured_depth_m > 0.0)
            ),
            TsdfContributionStatus.DEPTH_INVALID,
        )

        signed_distance_m = measured_depth_m - camera_z
        classify(
            ~np.isfinite(signed_distance_m),
            TsdfContributionStatus.SIGNED_DISTANCE_NONFINITE,
        )
        classify(
            signed_distance_m < -truncation_m,
            TsdfContributionStatus.BEHIND_TRUNCATION,
        )

        sum_deltas = np.where(
            alive,
            np.clip(signed_distance_m / truncation_m, -1.0, 1.0),
            0.0,
        )

    if not np.all(np.isfinite(sum_deltas)):
        raise TsdfError("TSDF contribution must remain finite")
    weight_deltas = alive.astype(TSDF_CONTRIBUTION_WEIGHT_DELTA_DTYPE)
    return status_codes, sum_deltas, weight_deltas


def _block_voxel_centres_world_m(
    block_index_xyz: _BlockIndex,
    voxel_size_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Voxel-centre world coordinates in canonical local-flat order."""

    block_x, block_y, block_z = block_index_xyz
    centres = tuple(
        (
            (
                local + block * TSDF_BLOCK_RESOLUTION
            ).astype(np.float64)
            + 0.5
        )
        * voxel_size_m
        for local, block in (
            (_LOCAL_X, block_x),
            (_LOCAL_Y, block_y),
            (_LOCAL_Z, block_z),
        )
    )
    if not all(np.all(np.isfinite(axis)) for axis in centres):
        raise TsdfError(
            "TSDF voxel center must remain finite in world coordinates"
        )
    return centres  # type: ignore[return-value]


def _freeze(array: np.ndarray) -> np.ndarray:
    """Return a bytes-backed, non-writeable copy of one result array."""

    payload = array.tobytes(order="C")
    return np.frombuffer(payload, dtype=array.dtype).reshape(array.shape)


def _validate_frozen_array(
    array: object,
    dtype: np.dtype,
    label: str,
) -> None:
    if type(array) is not np.ndarray:
        raise TsdfError(
            f"TSDF block contribution field {label} must be a base NumPy "
            "array"
        )
    if array.shape != (TSDF_BLOCK_VOXELS,) or array.dtype != dtype:
        raise TsdfError(
            f"TSDF block contribution field {label} must be a "
            f"{TSDF_BLOCK_VOXELS}-element {dtype.name} array"
        )
    if (
        not array.flags.c_contiguous
        or array.flags.owndata
        or array.flags.writeable
        or (payload := _immutable_bytes_base(array)) is None
        or len(payload) != array.nbytes
    ):
        raise TsdfError(
            f"TSDF block contribution field {label} must be immutable "
            "C-contiguous bytes-backed storage"
        )
