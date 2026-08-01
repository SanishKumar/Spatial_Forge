"""Context-backed traversal of one explicitly selected planned TSDF block."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .errors import SessionReplayError, TsdfError
from .tsdf_block_plan import TSDF_BLOCK_RESOLUTION
from .tsdf_block_storage import (
    TSDF_BLOCK_VOXELS,
    TsdfBlockStorage,
)
from .tsdf_replay_depth_context import TsdfReplayDepthContext
from .tsdf_voxel_address import (
    TsdfVoxelAddress,
    compose_tsdf_global_voxel_index,
    locate_tsdf_voxel,
)
from .tsdf_voxel_contribution import (
    TsdfContributionStatus,
    _validate_contribution_context,
    _validate_contribution_plan,
)
from .tsdf_voxel_traversal import (
    TsdfVoxelTraversalReceipt,
    traverse_tsdf_voxel_observations_from_context,
)
from .tsdf_voxel_update import (
    _storage_layout_identity,
    _validate_target_prestate,
    _validate_update_storage,
)

_BlockIndex = tuple[int, int, int]


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True, slots=True)
class TsdfBlockTraversalReceipt:
    """Immutable transcript for all voxels in one selected planned block."""

    block_index_xyz: _BlockIndex
    block_row: int
    source_plan_digest_sha256: str
    replay_digest_sha256: str
    block_resolution: int
    frame_stride: int
    total_observations: int
    selected_observation_sequences: tuple[int, ...]
    voxel_receipts: tuple[TsdfVoxelTraversalReceipt, ...]

    def __post_init__(self) -> None:
        compose_tsdf_global_voxel_index(
            self.block_index_xyz,
            (0, 0, 0),
        )
        if (
            isinstance(self.block_row, bool)
            or not isinstance(self.block_row, int)
            or self.block_row < 0
        ):
            raise TsdfError(
                "TSDF block traversal receipt block row must be nonnegative"
            )
        if not _is_sha256(self.source_plan_digest_sha256):
            raise TsdfError(
                "TSDF block traversal source plan digest is invalid"
            )
        if not _is_sha256(self.replay_digest_sha256):
            raise TsdfError(
                "TSDF block traversal replay digest is invalid"
            )
        if (
            isinstance(self.block_resolution, bool)
            or not isinstance(self.block_resolution, int)
            or self.block_resolution != TSDF_BLOCK_RESOLUTION
        ):
            raise TsdfError(
                "TSDF block traversal receipt requires block resolution "
                f"{TSDF_BLOCK_RESOLUTION}"
            )
        if (
            isinstance(self.frame_stride, bool)
            or not isinstance(self.frame_stride, int)
            or self.frame_stride < 1
        ):
            raise TsdfError(
                "TSDF block traversal frame stride must be positive"
            )
        if (
            isinstance(self.total_observations, bool)
            or not isinstance(self.total_observations, int)
            or self.total_observations < 1
        ):
            raise TsdfError(
                "TSDF block traversal total observations must be positive"
            )
        expected_sequences = tuple(
            range(0, self.total_observations, self.frame_stride)
        )
        if (
            not isinstance(self.selected_observation_sequences, tuple)
            or self.selected_observation_sequences != expected_sequences
        ):
            raise TsdfError(
                "TSDF block traversal observations are not the complete "
                "canonical stride selection"
            )
        if (
            not isinstance(self.voxel_receipts, tuple)
            or len(self.voxel_receipts) != TSDF_BLOCK_VOXELS
        ):
            raise TsdfError(
                "TSDF block traversal requires exactly "
                f"{TSDF_BLOCK_VOXELS} voxel receipts"
            )

        for local_flat_index, receipt in enumerate(self.voxel_receipts):
            if not isinstance(receipt, TsdfVoxelTraversalReceipt):
                raise TsdfError(
                    "TSDF block traversal contains an invalid voxel receipt"
                )
            local_index_xyz = _local_index_from_flat(local_flat_index)
            global_index_xyz = compose_tsdf_global_voxel_index(
                self.block_index_xyz,
                local_index_xyz,
            )
            address = receipt.address
            if (
                address.global_index_xyz != global_index_xyz
                or address.block_index_xyz != self.block_index_xyz
                or address.local_index_xyz != local_index_xyz
                or address.block_row != self.block_row
                or address.local_flat_index != local_flat_index
                or address.storage_flat_index
                != self.block_row * TSDF_BLOCK_VOXELS + local_flat_index
                or receipt.source_plan_digest_sha256
                != self.source_plan_digest_sha256
                or receipt.replay_digest_sha256
                != self.replay_digest_sha256
                or receipt.frame_stride != self.frame_stride
                or receipt.total_observations != self.total_observations
                or receipt.selected_observation_sequences
                != expected_sequences
            ):
                raise TsdfError(
                    "TSDF block traversal voxel receipt order is "
                    "inconsistent"
                )
            if (
                receipt.tsdf_sum_before != 0.0
                or bool(np.signbit(receipt.tsdf_sum_before))
                or receipt.weight_before != 0
            ):
                raise TsdfError(
                    "TSDF block traversal voxel receipts must start empty"
                )

    @property
    def voxel_count(self) -> int:
        return len(self.voxel_receipts)

    @property
    def voxel_address_count(self) -> int:
        return self.voxel_count

    @property
    def voxel_observation_traversal_count(self) -> int:
        return len(self.voxel_receipts)

    @property
    def evaluated_count(self) -> int:
        return sum(receipt.evaluated_count for receipt in self.voxel_receipts)

    @property
    def applied_count(self) -> int:
        return sum(receipt.applied_count for receipt in self.voxel_receipts)

    @property
    def skipped_count(self) -> int:
        return self.evaluated_count - self.applied_count

    @property
    def storage_slots_updated(self) -> int:
        return sum(
            receipt.storage_slots_updated
            for receipt in self.voxel_receipts
        )

    @property
    def observed_voxel_count(self) -> int:
        return sum(
            receipt.weight_after > 0
            for receipt in self.voxel_receipts
        )

    @property
    def unknown_voxel_count(self) -> int:
        return self.voxel_address_count - self.observed_voxel_count

    @property
    def nonzero_sum_count(self) -> int:
        return sum(
            receipt.tsdf_sum_after != 0.0
            for receipt in self.voxel_receipts
        )

    @property
    def weight_delta(self) -> int:
        return sum(receipt.weight_delta for receipt in self.voxel_receipts)

    @property
    def maximum_weight_after(self) -> int:
        return max(
            receipt.weight_after
            for receipt in self.voxel_receipts
        )

    @property
    def prepared_depth_accessed(self) -> bool:
        return any(
            contribution.depth_decoded
            for receipt in self.voxel_receipts
            for contribution in receipt.contributions
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
                    contribution.status is status
                    for receipt in self.voxel_receipts
                    for contribution in receipt.contributions
                )
            )
        )


def traverse_tsdf_block_voxels_from_context(
    storage: TsdfBlockStorage,
    block_index_xyz: _BlockIndex,
    context: TsdfReplayDepthContext,
) -> TsdfBlockTraversalReceipt:
    """Traverse every canonical voxel in one selected planned block."""

    if not isinstance(storage, TsdfBlockStorage):
        raise TsdfError(
            "TSDF context block traversal requires allocated "
            "TsdfBlockStorage"
        )
    compose_tsdf_global_voxel_index(block_index_xyz, (0, 0, 0))
    if not isinstance(context, TsdfReplayDepthContext):
        raise TsdfError(
            "TSDF context block traversal requires a prepared "
            "TsdfReplayDepthContext"
        )

    _validate_update_storage(storage)
    plan = storage.source_plan
    _validate_contribution_plan(plan)
    _validate_contribution_context(plan, context)
    selected_sequences = tuple(
        range(0, plan.total_observations, plan.frame_stride)
    )
    if (
        context.selected_observation_sequences != selected_sequences
        or len(context.observations) != len(selected_sequences)
    ):
        raise TsdfError(
            "TSDF replay/depth context observations are not the complete "
            "canonical block traversal selection"
        )

    addresses = _resolve_block_addresses(storage, block_index_xyz)
    block_row = addresses[0].block_row
    identity_before = _block_storage_identity(storage)
    try:
        stored_sums = storage.tsdf_sums[block_row].copy(order="C")
        stored_weights = storage.weights[block_row].copy(order="C")
    except (MemoryError, ValueError) as error:
        raise TsdfError(
            "cannot snapshot the selected TSDF block for traversal"
        ) from error
    stored_sum_bytes = stored_sums.tobytes()
    stored_weight_bytes = stored_weights.tobytes()
    _require_empty_selected_block(
        addresses,
        stored_sums,
        stored_weights,
    )
    _require_selected_block_bytes(
        storage,
        block_index_xyz,
        block_row,
        identity_before,
        stored_sum_bytes,
        stored_weight_bytes,
        "changed during empty-block preflight",
    )

    voxel_receipts: list[TsdfVoxelTraversalReceipt] = []
    try:
        for address in addresses:
            voxel_receipts.append(
                traverse_tsdf_voxel_observations_from_context(
                    storage,
                    address,
                    context,
                )
            )

        expected_sums = stored_sums.copy(order="C")
        expected_weights = stored_weights.copy(order="C")
        for receipt in voxel_receipts:
            local_x, local_y, local_z = receipt.address.local_index_xyz
            array_index_zyx = (local_z, local_y, local_x)
            expected_sums[array_index_zyx] = np.float64(
                receipt.tsdf_sum_after
            )
            expected_weights[array_index_zyx] = np.uint32(
                receipt.weight_after
            )
        _require_selected_block_bytes(
            storage,
            block_index_xyz,
            block_row,
            identity_before,
            expected_sums.tobytes(),
            expected_weights.tobytes(),
            "does not match its voxel traversal receipts",
        )
        result = TsdfBlockTraversalReceipt(
            block_index_xyz=block_index_xyz,
            block_row=block_row,
            source_plan_digest_sha256=plan.artifact_digest_sha256,
            replay_digest_sha256=plan.replay_digest_sha256,
            block_resolution=plan.block_resolution,
            frame_stride=plan.frame_stride,
            total_observations=plan.total_observations,
            selected_observation_sequences=selected_sequences,
            voxel_receipts=tuple(voxel_receipts),
        )
    except Exception as error:
        _restore_selected_block_or_fail(
            storage,
            block_row,
            identity_before,
            stored_sums,
            stored_weights,
            stored_sum_bytes,
            stored_weight_bytes,
        )
        if isinstance(error, (TsdfError, SessionReplayError)):
            raise
        raise TsdfError(
            f"cannot apply context-bound TSDF block traversal: {error}"
        ) from error
    return result


def _local_index_from_flat(local_flat_index: int) -> _BlockIndex:
    local_x = local_flat_index % TSDF_BLOCK_RESOLUTION
    local_y = (
        local_flat_index // TSDF_BLOCK_RESOLUTION
    ) % TSDF_BLOCK_RESOLUTION
    local_z = local_flat_index // (
        TSDF_BLOCK_RESOLUTION * TSDF_BLOCK_RESOLUTION
    )
    return (local_x, local_y, local_z)


def _resolve_block_addresses(
    storage: TsdfBlockStorage,
    block_index_xyz: _BlockIndex,
) -> tuple[TsdfVoxelAddress, ...]:
    addresses: list[TsdfVoxelAddress] = []
    block_row: int | None = None
    for local_flat_index in range(TSDF_BLOCK_VOXELS):
        local_index_xyz = _local_index_from_flat(local_flat_index)
        global_index_xyz = compose_tsdf_global_voxel_index(
            block_index_xyz,
            local_index_xyz,
        )
        address = locate_tsdf_voxel(storage, global_index_xyz)
        if address is None:
            if local_flat_index == 0:
                raise TsdfError(
                    f"TSDF block {block_index_xyz} is not planned in "
                    "destination storage"
                )
            raise TsdfError(
                "TSDF block traversal address resolution is incomplete"
            )
        if block_row is None:
            block_row = address.block_row
        if (
            address.block_index_xyz != block_index_xyz
            or address.local_index_xyz != local_index_xyz
            or address.local_flat_index != local_flat_index
            or address.block_row != block_row
            or address.storage_flat_index
            != block_row * TSDF_BLOCK_VOXELS + local_flat_index
        ):
            raise TsdfError(
                "TSDF block traversal address order is inconsistent"
            )
        addresses.append(address)
    return tuple(addresses)


def _require_empty_selected_block(
    addresses: tuple[TsdfVoxelAddress, ...],
    stored_sums: np.ndarray,
    stored_weights: np.ndarray,
) -> None:
    for address in addresses:
        local_x, local_y, local_z = address.local_index_xyz
        array_index_zyx = (local_z, local_y, local_x)
        stored_sum = stored_sums[array_index_zyx]
        stored_weight = stored_weights[array_index_zyx]
        try:
            _validate_target_prestate(
                stored_sum,
                stored_weight,
                float(stored_sum),
                int(stored_weight),
            )
        except TsdfError as error:
            raise TsdfError(
                "TSDF context block traversal local voxel "
                f"{address.local_index_xyz}: {error}"
            ) from error
        if int(stored_weight) != 0:
            raise TsdfError(
                "TSDF context block traversal requires a canonical empty "
                "selected block"
            )


def _block_storage_identity(storage: TsdfBlockStorage) -> tuple[object, ...]:
    return (
        id(storage.source_plan),
        storage.block_indices,
        *_storage_layout_identity(storage),
    )


def _require_selected_block_bytes(
    storage: TsdfBlockStorage,
    block_index_xyz: _BlockIndex,
    block_row: int,
    identity_before: tuple[object, ...],
    expected_sum_bytes: bytes,
    expected_weight_bytes: bytes,
    problem: str,
) -> None:
    _validate_update_storage(storage)
    if (
        _block_storage_identity(storage) != identity_before
        or block_row >= storage.block_count
        or storage.block_indices[block_row] != block_index_xyz
        or storage.tsdf_sums[block_row].tobytes()
        != expected_sum_bytes
        or storage.weights[block_row].tobytes()
        != expected_weight_bytes
    ):
        raise TsdfError(f"TSDF context block traversal {problem}")


def _restore_selected_block_or_fail(
    storage: TsdfBlockStorage,
    block_row: int,
    identity_before: tuple[object, ...],
    stored_sums: np.ndarray,
    stored_weights: np.ndarray,
    stored_sum_bytes: bytes,
    stored_weight_bytes: bytes,
) -> None:
    try:
        storage.tsdf_sums[block_row] = stored_sums
        storage.weights[block_row] = stored_weights
        if (
            _block_storage_identity(storage) != identity_before
            or storage.tsdf_sums[block_row].tobytes()
            != stored_sum_bytes
            or storage.weights[block_row].tobytes()
            != stored_weight_bytes
        ):
            raise ValueError(
                "restored block bytes or storage identity do not match"
            )
    except Exception as rollback_error:
        raise TsdfError(
            "TSDF block traversal failed and rollback failed; storage may "
            "be inconsistent"
        ) from rollback_error
