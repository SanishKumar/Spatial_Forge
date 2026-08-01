"""Context-backed traversal of every existing planned TSDF block row."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .errors import SessionReplayError, TsdfError
from .tsdf_block_plan import TSDF_BLOCK_RESOLUTION
from .tsdf_block_storage import (
    TSDF_BLOCK_VOXELS,
    TSDF_SUM_DTYPE,
    TSDF_WEIGHT_DTYPE,
    TsdfBlockStorage,
)
from .tsdf_block_traversal import (
    TsdfBlockTraversalReceipt,
    traverse_tsdf_block_voxels_from_context,
)
from .tsdf_replay_depth_context import TsdfReplayDepthContext
from .tsdf_voxel_address import compose_tsdf_global_voxel_index
from .tsdf_voxel_contribution import (
    TsdfContributionStatus,
    _validate_contribution_context,
    _validate_contribution_plan,
)
from .tsdf_voxel_update import (
    _storage_layout_identity,
    _validate_update_storage,
)

_BlockIndex = tuple[int, int, int]

MAX_TSDF_PLAN_TRAVERSAL_OUTCOMES = 262_144


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True, slots=True)
class TsdfPlanTraversalReceipt:
    """Immutable transcript for every existing planned TSDF block row."""

    source_plan_digest_sha256: str
    replay_digest_sha256: str
    block_resolution: int
    frame_stride: int
    total_observations: int
    selected_observation_sequences: tuple[int, ...]
    block_indices: tuple[_BlockIndex, ...]
    planned_voxel_slots: int
    block_receipts: tuple[TsdfBlockTraversalReceipt, ...]

    def __post_init__(self) -> None:
        if not _is_sha256(self.source_plan_digest_sha256):
            raise TsdfError(
                "TSDF plan traversal source plan digest is invalid"
            )
        if not _is_sha256(self.replay_digest_sha256):
            raise TsdfError(
                "TSDF plan traversal replay digest is invalid"
            )
        if (
            isinstance(self.block_resolution, bool)
            or not isinstance(self.block_resolution, int)
            or self.block_resolution != TSDF_BLOCK_RESOLUTION
        ):
            raise TsdfError(
                "TSDF plan traversal receipt requires block resolution "
                f"{TSDF_BLOCK_RESOLUTION}"
            )
        if (
            isinstance(self.frame_stride, bool)
            or not isinstance(self.frame_stride, int)
            or self.frame_stride < 1
        ):
            raise TsdfError(
                "TSDF plan traversal frame stride must be positive"
            )
        if (
            isinstance(self.total_observations, bool)
            or not isinstance(self.total_observations, int)
            or self.total_observations < 1
        ):
            raise TsdfError(
                "TSDF plan traversal total observations must be positive"
            )
        expected_sequences = tuple(
            range(0, self.total_observations, self.frame_stride)
        )
        if (
            not isinstance(self.selected_observation_sequences, tuple)
            or self.selected_observation_sequences != expected_sequences
        ):
            raise TsdfError(
                "TSDF plan traversal observations are not the complete "
                "canonical stride selection"
            )
        if not isinstance(self.block_indices, tuple) or not self.block_indices:
            raise TsdfError(
                "TSDF plan traversal requires at least one planned block"
            )
        expected_voxel_slots = len(self.block_indices) * TSDF_BLOCK_VOXELS
        if (
            isinstance(self.planned_voxel_slots, bool)
            or not isinstance(self.planned_voxel_slots, int)
            or self.planned_voxel_slots != expected_voxel_slots
        ):
            raise TsdfError(
                "TSDF plan traversal planned voxel slots do not match its "
                "block rows"
            )
        if (
            not isinstance(self.block_receipts, tuple)
            or len(self.block_receipts) != len(self.block_indices)
        ):
            raise TsdfError(
                "TSDF plan traversal requires exactly one receipt for "
                "every planned block"
            )

        previous_key: tuple[int, int, int] | None = None
        for block_row, block_index_xyz in enumerate(self.block_indices):
            compose_tsdf_global_voxel_index(
                block_index_xyz,
                (0, 0, 0),
            )
            key = (
                block_index_xyz[2],
                block_index_xyz[1],
                block_index_xyz[0],
            )
            if previous_key is not None and key <= previous_key:
                raise TsdfError(
                    "TSDF plan traversal blocks must be unique and "
                    "strictly x-fastest ordered"
                )
            previous_key = key

            receipt = self.block_receipts[block_row]
            if not isinstance(receipt, TsdfBlockTraversalReceipt):
                raise TsdfError(
                    "TSDF plan traversal contains an invalid block receipt"
                )
            if (
                receipt.block_index_xyz != block_index_xyz
                or receipt.block_row != block_row
                or receipt.source_plan_digest_sha256
                != self.source_plan_digest_sha256
                or receipt.replay_digest_sha256
                != self.replay_digest_sha256
                or receipt.block_resolution != self.block_resolution
                or receipt.frame_stride != self.frame_stride
                or receipt.total_observations != self.total_observations
                or receipt.selected_observation_sequences
                != expected_sequences
            ):
                raise TsdfError(
                    "TSDF plan traversal block receipt order is "
                    "inconsistent"
                )
        expected_outcomes = expected_voxel_slots * len(expected_sequences)
        if self.evaluated_count != expected_outcomes:
            raise TsdfError(
                "TSDF plan traversal evaluated outcomes are incomplete"
            )

    @property
    def block_count(self) -> int:
        return len(self.block_receipts)

    @property
    def voxel_count(self) -> int:
        return sum(receipt.voxel_count for receipt in self.block_receipts)

    @property
    def voxel_address_count(self) -> int:
        return self.voxel_count

    @property
    def voxel_observation_traversal_count(self) -> int:
        return sum(
            receipt.voxel_observation_traversal_count
            for receipt in self.block_receipts
        )

    @property
    def evaluated_count(self) -> int:
        return sum(
            receipt.evaluated_count for receipt in self.block_receipts
        )

    @property
    def applied_count(self) -> int:
        return sum(
            receipt.applied_count for receipt in self.block_receipts
        )

    @property
    def skipped_count(self) -> int:
        return self.evaluated_count - self.applied_count

    @property
    def storage_slots_updated(self) -> int:
        return sum(
            receipt.storage_slots_updated
            for receipt in self.block_receipts
        )

    @property
    def observed_voxel_count(self) -> int:
        return sum(
            receipt.observed_voxel_count
            for receipt in self.block_receipts
        )

    @property
    def unknown_voxel_count(self) -> int:
        return self.voxel_address_count - self.observed_voxel_count

    @property
    def nonzero_sum_count(self) -> int:
        return sum(
            receipt.nonzero_sum_count for receipt in self.block_receipts
        )

    @property
    def weight_delta(self) -> int:
        return sum(
            receipt.weight_delta for receipt in self.block_receipts
        )

    @property
    def maximum_weight_after(self) -> int:
        return max(
            receipt.maximum_weight_after
            for receipt in self.block_receipts
        )

    @property
    def prepared_depth_accessed(self) -> bool:
        return any(
            receipt.prepared_depth_accessed
            for receipt in self.block_receipts
        )

    @property
    def status_counts(
        self,
    ) -> tuple[tuple[TsdfContributionStatus, int], ...]:
        return tuple(
            (status, count)
            for status in TsdfContributionStatus
            if (
                count := sum(
                    child_count
                    for receipt in self.block_receipts
                    for child_status, child_count in receipt.status_counts
                    if child_status is status
                )
            )
        )


def traverse_tsdf_plan_blocks_from_context(
    storage: TsdfBlockStorage,
    context: TsdfReplayDepthContext,
) -> TsdfPlanTraversalReceipt:
    """Traverse every existing planned block row in canonical order."""

    if not isinstance(storage, TsdfBlockStorage):
        raise TsdfError(
            "TSDF context plan traversal requires allocated "
            "TsdfBlockStorage"
        )
    if not isinstance(context, TsdfReplayDepthContext):
        raise TsdfError(
            "TSDF context plan traversal requires a prepared "
            "TsdfReplayDepthContext"
        )

    _validate_update_storage(storage)
    plan = storage.source_plan
    _validate_contribution_plan(plan)
    _validate_contribution_context(plan, context)
    block_indices = _validate_plan_storage_rows(storage)
    selected_sequences = tuple(
        range(0, plan.total_observations, plan.frame_stride)
    )
    if (
        context.selected_observation_sequences != selected_sequences
        or len(context.observations) != len(selected_sequences)
    ):
        raise TsdfError(
            "TSDF replay/depth context observations are not the complete "
            "canonical plan traversal selection"
        )
    outcome_count = storage.voxel_slots * len(selected_sequences)
    if outcome_count > MAX_TSDF_PLAN_TRAVERSAL_OUTCOMES:
        raise TsdfError(
            "TSDF context plan traversal requires "
            f"{outcome_count} retained contribution outcomes; reference "
            f"maximum is {MAX_TSDF_PLAN_TRAVERSAL_OUTCOMES}. Use a larger "
            "frame stride, a smaller plan, or a future scalable fusion path"
        )

    identity_before = _plan_storage_identity(storage)
    _require_empty_plan_storage(storage)
    _require_plan_storage_identity(
        storage,
        identity_before,
        require_empty=True,
        problem="changed during empty-storage preflight",
    )

    block_receipts: list[TsdfBlockTraversalReceipt] = []
    try:
        for block_row, block_index_xyz in enumerate(block_indices):
            receipt = traverse_tsdf_block_voxels_from_context(
                storage,
                block_index_xyz,
                context,
            )
            if (
                not isinstance(receipt, TsdfBlockTraversalReceipt)
                or receipt.block_index_xyz != block_index_xyz
                or receipt.block_row != block_row
            ):
                raise TsdfError(
                    "TSDF context plan traversal child block scope is "
                    "inconsistent"
                )
            block_receipts.append(receipt)

        _require_storage_matches_receipts(
            storage,
            identity_before,
            tuple(block_receipts),
        )
        result = TsdfPlanTraversalReceipt(
            source_plan_digest_sha256=plan.artifact_digest_sha256,
            replay_digest_sha256=plan.replay_digest_sha256,
            block_resolution=plan.block_resolution,
            frame_stride=plan.frame_stride,
            total_observations=plan.total_observations,
            selected_observation_sequences=selected_sequences,
            block_indices=block_indices,
            planned_voxel_slots=storage.voxel_slots,
            block_receipts=tuple(block_receipts),
        )
    except Exception as error:
        _restore_empty_plan_storage_or_fail(
            storage,
            identity_before,
        )
        if isinstance(error, (TsdfError, SessionReplayError)):
            raise
        raise TsdfError(
            f"cannot apply context-bound TSDF plan traversal: {error}"
        ) from error
    return result


def _validate_plan_storage_rows(
    storage: TsdfBlockStorage,
) -> tuple[_BlockIndex, ...]:
    plan = storage.source_plan
    block_indices = storage.block_indices
    if (
        not _is_sha256(plan.artifact_digest_sha256)
        or not _is_sha256(plan.replay_digest_sha256)
    ):
        raise TsdfError(
            "TSDF context plan traversal source provenance is invalid"
        )
    if not isinstance(block_indices, tuple) or not block_indices:
        raise TsdfError(
            "TSDF context plan traversal requires at least one block row"
        )
    if block_indices != plan.active_blocks:
        raise TsdfError(
            "TSDF context plan traversal rows must match the source plan"
        )
    if (
        plan.block_resolution != TSDF_BLOCK_RESOLUTION
        or plan.planned_voxel_slots
        != len(block_indices) * TSDF_BLOCK_VOXELS
    ):
        raise TsdfError(
            "TSDF context plan traversal storage geometry is inconsistent"
        )

    previous_key: tuple[int, int, int] | None = None
    for block_index_xyz in block_indices:
        compose_tsdf_global_voxel_index(block_index_xyz, (0, 0, 0))
        key = (
            block_index_xyz[2],
            block_index_xyz[1],
            block_index_xyz[0],
        )
        if previous_key is not None and key <= previous_key:
            raise TsdfError(
                "TSDF context plan traversal rows must be unique and "
                "strictly x-fastest ordered"
            )
        previous_key = key
    return block_indices


def _plan_storage_identity(storage: TsdfBlockStorage) -> tuple[object, ...]:
    return (
        id(storage.source_plan),
        storage.block_indices,
        *_storage_layout_identity(storage),
    )


def _storage_has_nonzero_bytes(storage: TsdfBlockStorage) -> bool:
    return bool(
        np.any(storage.tsdf_sums.view(np.uint8))
        or np.any(storage.weights.view(np.uint8))
    )


def _require_empty_plan_storage(storage: TsdfBlockStorage) -> None:
    try:
        is_nonempty = _storage_has_nonzero_bytes(storage)
    except (MemoryError, TypeError, ValueError) as error:
        raise TsdfError(
            "cannot inspect TSDF plan storage before traversal"
        ) from error
    if is_nonempty:
        raise TsdfError(
            "TSDF context plan traversal requires canonical empty storage"
        )


def _require_plan_storage_identity(
    storage: TsdfBlockStorage,
    identity_before: tuple[object, ...],
    *,
    require_empty: bool,
    problem: str,
) -> None:
    _validate_update_storage(storage)
    if _plan_storage_identity(storage) != identity_before:
        raise TsdfError(f"TSDF context plan traversal {problem}")
    if require_empty:
        _require_empty_plan_storage(storage)


def _require_storage_matches_receipts(
    storage: TsdfBlockStorage,
    identity_before: tuple[object, ...],
    block_receipts: tuple[TsdfBlockTraversalReceipt, ...],
) -> None:
    _require_plan_storage_identity(
        storage,
        identity_before,
        require_empty=False,
        problem="changed storage identity during traversal",
    )
    expected_shape = (
        TSDF_BLOCK_RESOLUTION,
        TSDF_BLOCK_RESOLUTION,
        TSDF_BLOCK_RESOLUTION,
    )
    for block_row, receipt in enumerate(block_receipts):
        expected_sums = np.empty(expected_shape, dtype=TSDF_SUM_DTYPE)
        expected_weights = np.empty(expected_shape, dtype=TSDF_WEIGHT_DTYPE)
        expected_sums.fill(np.float64(0.0))
        expected_weights.fill(np.uint32(0))
        for voxel_receipt in receipt.voxel_receipts:
            local_x, local_y, local_z = (
                voxel_receipt.address.local_index_xyz
            )
            array_index_zyx = (local_z, local_y, local_x)
            expected_sums[array_index_zyx] = np.float64(
                voxel_receipt.tsdf_sum_after
            )
            expected_weights[array_index_zyx] = np.uint32(
                voxel_receipt.weight_after
            )
        if (
            storage.tsdf_sums[block_row].tobytes()
            != expected_sums.tobytes()
            or storage.weights[block_row].tobytes()
            != expected_weights.tobytes()
        ):
            raise TsdfError(
                "TSDF context plan traversal storage does not match its "
                "block receipts"
            )


def _restore_empty_plan_storage_or_fail(
    storage: TsdfBlockStorage,
    identity_before: tuple[object, ...],
) -> None:
    try:
        storage.tsdf_sums.fill(np.float64(0.0))
        storage.weights.fill(np.uint32(0))
        _validate_update_storage(storage)
        if (
            _plan_storage_identity(storage) != identity_before
            or _storage_has_nonzero_bytes(storage)
        ):
            raise ValueError(
                "restored storage bytes or identity do not match"
            )
    except Exception as rollback_error:
        raise TsdfError(
            "TSDF plan traversal failed and rollback failed; storage may "
            "be inconsistent"
        ) from rollback_error
