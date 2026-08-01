"""Rollback-guarded application of one evaluated TSDF voxel contribution."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .errors import SessionReplayError, TsdfError
from .model import ScanSession
from .replay import replay_session
from .tsdf_block_plan import TSDF_BLOCK_RESOLUTION
from .tsdf_block_plan_loader import TsdfBlockPlan
from .tsdf_block_storage import (
    TSDF_BLOCK_STORAGE_BYTES_PER_VOXEL,
    TSDF_BLOCK_VOXELS,
    TsdfBlockStorage,
    _validate_storage_arrays,
)
from .tsdf_replay_depth_context import (
    TsdfReplayDepthContext,
    TsdfReplayDepthStatus,
)
from .tsdf_voxel_address import locate_tsdf_voxel
from .tsdf_voxel_contribution import (
    TsdfContributionStatus,
    TsdfVoxelContribution,
    _validate_contribution_context,
    _validate_contribution_plan,
)

MAX_TSDF_VOXEL_WEIGHT = int(np.iinfo(np.uint32).max)


def _is_finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


@dataclass(frozen=True, slots=True)
class TsdfVoxelUpdateReceipt:
    """Immutable evidence of one successful ephemeral accumulator update."""

    contribution: TsdfVoxelContribution
    tsdf_sum_before: float
    weight_before: int
    tsdf_sum_after: float
    weight_after: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.contribution, TsdfVoxelContribution)
            or self.contribution.status
            is not TsdfContributionStatus.CONTRIBUTES
            or self.contribution.tsdf_sum_delta is None
            or self.contribution.weight_delta != 1
        ):
            raise TsdfError(
                "TSDF voxel update receipt requires one accepted "
                "contribution"
            )
        for value, label in (
            (self.tsdf_sum_before, "tsdf_sum_before"),
            (self.tsdf_sum_after, "tsdf_sum_after"),
        ):
            if not _is_finite_number(value):
                raise TsdfError(
                    f"TSDF voxel update receipt {label} must be finite"
                )
        if (
            isinstance(self.weight_before, bool)
            or not isinstance(self.weight_before, int)
            or self.weight_before < 0
            or self.weight_before >= MAX_TSDF_VOXEL_WEIGHT
        ):
            raise TsdfError(
                "TSDF voxel update receipt prior weight is invalid"
            )
        if (
            isinstance(self.weight_after, bool)
            or not isinstance(self.weight_after, int)
            or self.weight_after != self.weight_before + 1
        ):
            raise TsdfError(
                "TSDF voxel update receipt weight transition is invalid"
            )
        if self.weight_before == 0:
            if (
                self.tsdf_sum_before != 0.0
                or bool(np.signbit(self.tsdf_sum_before))
            ):
                raise TsdfError(
                    "TSDF voxel update receipt prior unknown state must "
                    "have canonical positive zero sum"
                )
        elif abs(self.tsdf_sum_before) > self.weight_before:
            raise TsdfError(
                "TSDF voxel update receipt prior sum exceeds its weight "
                "envelope"
            )
        if abs(self.tsdf_sum_after) > self.weight_after:
            raise TsdfError(
                "TSDF voxel update receipt resulting sum exceeds its "
                "weight envelope"
            )
        expected_sum = float(
            np.float64(self.tsdf_sum_before)
            + np.float64(self.contribution.tsdf_sum_delta)
        )
        if self.tsdf_sum_after != expected_sum:
            raise TsdfError(
                "TSDF voxel update receipt sum transition is invalid"
            )


def apply_tsdf_voxel_contribution(
    storage: TsdfBlockStorage,
    contribution: TsdfVoxelContribution,
    session: ScanSession,
) -> TsdfVoxelUpdateReceipt:
    """Apply one accepted contribution to exactly one temporary voxel slot."""

    if not isinstance(storage, TsdfBlockStorage):
        raise TsdfError(
            "TSDF voxel update requires allocated TsdfBlockStorage"
        )
    if not isinstance(contribution, TsdfVoxelContribution):
        raise TsdfError(
            "TSDF voxel update requires a TsdfVoxelContribution"
        )
    if not isinstance(session, ScanSession):
        raise TsdfError(
            "TSDF voxel update requires a loaded ScanSession"
        )
    _validate_accepted_contribution(contribution)
    plan = _validate_contribution_storage_provenance(storage, contribution)
    if plan.session_id != session.session_id:
        raise TsdfError(
            "TSDF block plan session_id does not match the loaded session"
        )
    _validate_update_destination(storage, contribution)

    starting_replay = replay_session(session)
    if starting_replay.digest_sha256 != contribution.replay_digest_sha256:
        raise TsdfError(
            "TSDF contribution replay digest does not match current session "
            "inputs"
        )

    def validate_ending_replay() -> None:
        ending_replay = replay_session(session)
        if (
            ending_replay.digest_sha256
            != contribution.replay_digest_sha256
            or ending_replay.digest_sha256
            != starting_replay.digest_sha256
        ):
            raise TsdfError(
                "session inputs changed while applying a TSDF contribution"
            )

    return _apply_tsdf_voxel_contribution_core(
        storage,
        contribution,
        postwrite_validation=validate_ending_replay,
        unexpected_error_prefix="cannot apply TSDF voxel contribution",
    )


def apply_tsdf_voxel_contribution_from_context(
    storage: TsdfBlockStorage,
    contribution: TsdfVoxelContribution,
    context: TsdfReplayDepthContext,
) -> TsdfVoxelUpdateReceipt:
    """Apply one accepted contribution using construction-time provenance."""

    if not isinstance(storage, TsdfBlockStorage):
        raise TsdfError(
            "TSDF context voxel update requires allocated TsdfBlockStorage"
        )
    if not isinstance(contribution, TsdfVoxelContribution):
        raise TsdfError(
            "TSDF context voxel update requires a TsdfVoxelContribution"
        )
    if not isinstance(context, TsdfReplayDepthContext):
        raise TsdfError(
            "TSDF context voxel update requires a TsdfReplayDepthContext"
        )
    _validate_accepted_contribution(contribution)

    plan = storage.source_plan
    _validate_contribution_plan(plan)
    _validate_contribution_context(plan, context)
    _validate_contribution_storage_provenance(storage, contribution)
    if (
        contribution.source_plan_digest_sha256
        != context.source_plan_digest_sha256
    ):
        raise TsdfError(
            "TSDF contribution source plan does not match replay/depth "
            "context"
        )
    if contribution.replay_digest_sha256 != context.replay_digest_sha256:
        raise TsdfError(
            "TSDF contribution replay digest does not match replay/depth "
            "context"
        )
    _validate_context_contribution_observation(context, contribution)
    _validate_update_destination(storage, contribution)
    return _apply_tsdf_voxel_contribution_core(
        storage,
        contribution,
        postwrite_validation=None,
        unexpected_error_prefix=(
            "cannot apply context-bound TSDF voxel contribution"
        ),
    )


def _validate_accepted_contribution(
    contribution: TsdfVoxelContribution,
) -> None:
    if (
        contribution.status is not TsdfContributionStatus.CONTRIBUTES
        or not contribution.contributes
        or contribution.tsdf_sum_delta is None
        or not math.isfinite(contribution.tsdf_sum_delta)
        or not -1.0 <= contribution.tsdf_sum_delta <= 1.0
        or contribution.weight_delta != 1
    ):
        raise TsdfError(
            "TSDF voxel update requires an accepted finite contribution"
        )


def _validate_contribution_storage_provenance(
    storage: TsdfBlockStorage,
    contribution: TsdfVoxelContribution,
) -> TsdfBlockPlan:
    plan = storage.source_plan
    if contribution.source_plan_digest_sha256 != plan.artifact_digest_sha256:
        raise TsdfError(
            "TSDF contribution source plan does not match destination "
            "storage"
        )
    if contribution.replay_digest_sha256 != plan.replay_digest_sha256:
        raise TsdfError(
            "TSDF contribution replay digest does not match destination "
            "storage"
        )
    return plan


def _validate_context_contribution_observation(
    context: TsdfReplayDepthContext,
    contribution: TsdfVoxelContribution,
) -> None:
    sequence = contribution.observation_sequence
    if sequence >= context.total_observations:
        raise TsdfError(
            "TSDF contribution observation is outside replay/depth context"
        )
    if sequence % context.frame_stride != 0:
        raise TsdfError(
            "TSDF contribution observation is not selected by context"
        )
    observation = context.observations[sequence // context.frame_stride]
    if observation.observation_sequence != sequence:
        raise TsdfError(
            "TSDF replay/depth context observation lookup is inconsistent"
        )
    if observation.status is not TsdfReplayDepthStatus.READY:
        raise TsdfError(
            "TSDF context voxel update requires a ready observation"
        )


def _validate_update_destination(
    storage: TsdfBlockStorage,
    contribution: TsdfVoxelContribution,
) -> None:
    _validate_update_storage(storage)
    resolved_address = locate_tsdf_voxel(
        storage,
        contribution.address.global_index_xyz,
    )
    if resolved_address != contribution.address:
        raise TsdfError(
            "TSDF contribution address does not match destination storage"
        )


def _apply_tsdf_voxel_contribution_core(
    storage: TsdfBlockStorage,
    contribution: TsdfVoxelContribution,
    *,
    postwrite_validation: Callable[[], None] | None,
    unexpected_error_prefix: str,
) -> TsdfVoxelUpdateReceipt:
    array_index = contribution.address.array_index_bzyx
    stored_sum = storage.tsdf_sums[array_index]
    stored_weight = storage.weights[array_index]
    tsdf_sum_before = float(stored_sum)
    weight_before = int(stored_weight)
    _validate_target_prestate(
        stored_sum,
        stored_weight,
        tsdf_sum_before,
        weight_before,
    )
    if weight_before == MAX_TSDF_VOXEL_WEIGHT:
        raise TsdfError(
            "TSDF voxel weight cannot exceed uint32 maximum"
        )

    with np.errstate(over="ignore", invalid="ignore"):
        stored_sum_after = (
            np.float64(stored_sum)
            + np.float64(contribution.tsdf_sum_delta)
        )
    tsdf_sum_after = float(stored_sum_after)
    weight_after = weight_before + 1
    if (
        not math.isfinite(tsdf_sum_after)
        or abs(tsdf_sum_after) > weight_after
    ):
        raise TsdfError(
            "TSDF voxel sum update would produce invalid accumulator state"
        )

    receipt = TsdfVoxelUpdateReceipt(
        contribution=contribution,
        tsdf_sum_before=tsdf_sum_before,
        weight_before=weight_before,
        tsdf_sum_after=tsdf_sum_after,
        weight_after=weight_after,
    )
    layout_before = _storage_layout_identity(storage)
    stored_sum_bytes = stored_sum.tobytes()
    stored_weight_bytes = stored_weight.tobytes()
    expected_sum_bytes = stored_sum_after.tobytes()
    expected_weight_bytes = np.uint32(weight_after).tobytes()
    wrote_target = False
    try:
        storage.tsdf_sums[array_index] = stored_sum_after
        wrote_target = True
        storage.weights[array_index] = np.uint32(weight_after)
        if (
            _storage_layout_identity(storage) != layout_before
            or storage.tsdf_sums[array_index].tobytes()
            != expected_sum_bytes
            or storage.weights[array_index].tobytes()
            != expected_weight_bytes
        ):
            raise TsdfError(
                "TSDF voxel update could not verify its scalar writes"
            )
        if postwrite_validation is not None:
            postwrite_validation()
    except Exception as error:
        if wrote_target:
            _restore_target_or_fail(
                storage,
                array_index,
                stored_sum,
                stored_weight,
                layout_before,
                stored_sum_bytes,
                stored_weight_bytes,
            )
        if isinstance(error, (TsdfError, SessionReplayError)):
            raise
        raise TsdfError(
            f"{unexpected_error_prefix}: {error}"
        ) from error
    return receipt


def _validate_update_storage(storage: TsdfBlockStorage) -> None:
    expected_shape = (
        storage.block_count,
        TSDF_BLOCK_RESOLUTION,
        TSDF_BLOCK_RESOLUTION,
        TSDF_BLOCK_RESOLUTION,
    )
    expected_payload_bytes = (
        storage.block_count
        * TSDF_BLOCK_VOXELS
        * TSDF_BLOCK_STORAGE_BYTES_PER_VOXEL
    )
    _validate_storage_arrays(
        storage.tsdf_sums,
        storage.weights,
        expected_shape,
        expected_payload_bytes,
    )
    if (
        type(storage.tsdf_sums) is not np.ndarray
        or type(storage.weights) is not np.ndarray
    ):
        raise TsdfError(
            "TSDF voxel update requires base NumPy storage arrays"
        )
    if (
        not storage.tsdf_sums.flags.writeable
        or not storage.weights.flags.writeable
    ):
        raise TsdfError(
            "TSDF voxel update requires writable sum and weight arrays"
        )
    if np.shares_memory(storage.tsdf_sums, storage.weights):
        raise TsdfError(
            "TSDF voxel update requires non-overlapping storage arrays"
        )


def _validate_target_prestate(
    stored_sum: np.float64,
    stored_weight: np.uint32,
    tsdf_sum_before: float,
    weight_before: int,
) -> None:
    if not math.isfinite(tsdf_sum_before):
        raise TsdfError("TSDF voxel accumulator sum must be finite")
    if weight_before == 0:
        if (
            tsdf_sum_before != 0.0
            or bool(np.signbit(stored_sum))
        ):
            raise TsdfError(
                "unknown TSDF voxel must have canonical positive zero sum"
            )
        return
    if abs(tsdf_sum_before) > weight_before:
        raise TsdfError(
            "TSDF voxel accumulator sum exceeds its weight envelope"
        )
    if int(stored_weight) != weight_before:
        raise TsdfError("TSDF voxel accumulator weight is inconsistent")


def _storage_layout_identity(
    storage: TsdfBlockStorage,
) -> tuple[object, ...]:
    return (
        id(storage.tsdf_sums),
        id(storage.weights),
        storage.tsdf_sums.shape,
        storage.weights.shape,
        storage.tsdf_sums.strides,
        storage.weights.strides,
        storage.tsdf_sums.dtype.str,
        storage.weights.dtype.str,
        bool(storage.tsdf_sums.flags.c_contiguous),
        bool(storage.weights.flags.c_contiguous),
        bool(storage.tsdf_sums.flags.writeable),
        bool(storage.weights.flags.writeable),
        id(storage.tsdf_sums.base),
        id(storage.weights.base),
        int(storage.tsdf_sums.ctypes.data),
        int(storage.weights.ctypes.data),
    )


def _restore_target_or_fail(
    storage: TsdfBlockStorage,
    array_index: tuple[int, int, int, int],
    stored_sum: np.float64,
    stored_weight: np.uint32,
    layout_before: tuple[object, ...],
    stored_sum_bytes: bytes,
    stored_weight_bytes: bytes,
) -> None:
    try:
        storage.tsdf_sums[array_index] = stored_sum
        storage.weights[array_index] = stored_weight
        if (
            _storage_layout_identity(storage) != layout_before
            or storage.tsdf_sums[array_index].tobytes()
            != stored_sum_bytes
            or storage.weights[array_index].tobytes()
            != stored_weight_bytes
        ):
            raise ValueError(
                "restored scalar bytes or storage layout do not match"
            )
    except Exception as rollback_error:
        raise TsdfError(
            "TSDF voxel update failed and rollback failed; storage may be "
            "inconsistent"
        ) from rollback_error
