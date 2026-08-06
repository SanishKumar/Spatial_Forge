"""Whole-block fusion driven by vectorised per-observation fields.

`traverse_tsdf_block_voxels_from_context` fuses one block by walking its 512
voxels, evaluating every selected observation for each and applying accepted
contributions one guarded scalar write at a time. It is the reference for what
a fused block contains, and it is slow for exactly the reason it is
trustworthy.

This module fuses the same block from the vectorised fields in
:mod:`tsdf_block_contributions`: one field per selected observation, applied
to the block's storage row in canonical observation order with two array
additions each. The result must be byte-identical to the traversal's, which is
what makes the substitution legitimate rather than merely plausible.

Per-voxel accumulation order is preserved because each field holds at most one
contribution per voxel, so applying fields in canonical observation order
performs exactly the scalar path's sequence of float64 additions. Skipped
voxels carry `+0.0`, and `x + 0.0` is bit-preserving for every value the
accumulator can hold: it starts at `+0.0` and IEEE round-to-nearest never
produces `-0.0` from a sum unless both operands are `-0.0`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .errors import SessionReplayError, TsdfError
from .tsdf_block_contributions import (
    TSDF_CONTRIBUTION_SUM_DELTA_DTYPE,
    TSDF_CONTRIBUTION_WEIGHT_DELTA_DTYPE,
    TsdfBlockContributionField,
    _freeze,
    _validate_frozen_array,
    evaluate_tsdf_block_contributions_from_context,
)
from .tsdf_block_plan import TSDF_BLOCK_RESOLUTION
from .tsdf_block_storage import (
    TSDF_BLOCK_VOXELS,
    TSDF_SUM_DTYPE,
    TSDF_WEIGHT_DTYPE,
    TsdfBlockStorage,
)
from .tsdf_block_traversal import _block_storage_identity
from .tsdf_replay_depth_context import TsdfReplayDepthContext
from .tsdf_voxel_address import (
    compose_tsdf_global_voxel_index,
    locate_tsdf_voxel,
)
from .tsdf_voxel_contribution import (
    TsdfContributionStatus,
    _validate_contribution_context,
    _validate_contribution_plan,
)
from .tsdf_voxel_update import (
    MAX_TSDF_VOXEL_WEIGHT,
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


@dataclass(frozen=True, slots=True, eq=False)
class TsdfBlockVectorFusionReceipt:
    """Immutable evidence of one whole-block field-driven fusion."""

    block_index_xyz: _BlockIndex
    block_row: int
    source_plan_digest_sha256: str
    replay_digest_sha256: str
    block_resolution: int
    frame_stride: int
    total_observations: int
    selected_observation_sequences: tuple[int, ...]
    fields: tuple[TsdfBlockContributionField, ...]
    tsdf_sums_before: np.ndarray
    weights_before: np.ndarray
    tsdf_sums_after: np.ndarray
    weights_after: np.ndarray

    def __post_init__(self) -> None:
        compose_tsdf_global_voxel_index(self.block_index_xyz, (0, 0, 0))
        if (
            isinstance(self.block_row, bool)
            or not isinstance(self.block_row, int)
            or self.block_row < 0
        ):
            raise TsdfError(
                "TSDF block vector fusion block row must be nonnegative"
            )
        if not _is_sha256(self.source_plan_digest_sha256):
            raise TsdfError(
                "TSDF block vector fusion source plan digest is invalid"
            )
        if not _is_sha256(self.replay_digest_sha256):
            raise TsdfError(
                "TSDF block vector fusion replay digest is invalid"
            )
        if (
            isinstance(self.block_resolution, bool)
            or not isinstance(self.block_resolution, int)
            or self.block_resolution != TSDF_BLOCK_RESOLUTION
        ):
            raise TsdfError(
                "TSDF block vector fusion requires block resolution "
                f"{TSDF_BLOCK_RESOLUTION}"
            )
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
                    f"TSDF block vector fusion {label} must be positive"
                )
        expected_sequences = tuple(
            range(0, self.total_observations, self.frame_stride)
        )
        if (
            not isinstance(self.selected_observation_sequences, tuple)
            or self.selected_observation_sequences != expected_sequences
        ):
            raise TsdfError(
                "TSDF block vector fusion observations are not the complete "
                "canonical stride selection"
            )
        if (
            not isinstance(self.fields, tuple)
            or len(self.fields) != len(expected_sequences)
        ):
            raise TsdfError(
                "TSDF block vector fusion requires one field per selected "
                "observation"
            )
        for sequence, field in zip(
            expected_sequences,
            self.fields,
            strict=True,
        ):
            if (
                not isinstance(field, TsdfBlockContributionField)
                or field.observation_sequence != sequence
                or field.block_index_xyz != self.block_index_xyz
                or field.block_row != self.block_row
                or field.source_plan_digest_sha256
                != self.source_plan_digest_sha256
                or field.replay_digest_sha256 != self.replay_digest_sha256
            ):
                raise TsdfError(
                    "TSDF block vector fusion field order or provenance is "
                    "inconsistent"
                )

        for array, dtype, label in (
            (self.tsdf_sums_before, TSDF_SUM_DTYPE, "sums before"),
            (self.weights_before, TSDF_WEIGHT_DTYPE, "weights before"),
            (self.tsdf_sums_after, TSDF_SUM_DTYPE, "sums after"),
            (self.weights_after, TSDF_WEIGHT_DTYPE, "weights after"),
        ):
            _validate_frozen_array(array, dtype, label)

        # Re-derive rather than trust: replay the retained fields onto the
        # retained starting state and require the recorded result exactly.
        derived_sums = self.tsdf_sums_before.astype(
            TSDF_CONTRIBUTION_SUM_DELTA_DTYPE
        )
        derived_weights = self.weights_before.astype(np.uint64)
        with np.errstate(over="ignore", invalid="ignore"):
            for field in self.fields:
                derived_sums += field.tsdf_sum_deltas
                derived_weights += field.weight_deltas
        if derived_weights.max() > MAX_TSDF_VOXEL_WEIGHT:
            raise TsdfError(
                "TSDF block vector fusion weight exceeds the uint32 maximum"
            )
        if (
            derived_sums.tobytes() != self.tsdf_sums_after.tobytes()
            or derived_weights.astype(TSDF_WEIGHT_DTYPE).tobytes()
            != self.weights_after.tobytes()
        ):
            raise TsdfError(
                "TSDF block vector fusion result does not match replaying "
                "its own retained fields"
            )
        if not np.all(np.isfinite(self.tsdf_sums_after)):
            raise TsdfError(
                "TSDF block vector fusion sums must remain finite"
            )
        if np.any(
            np.abs(self.tsdf_sums_after)
            > self.weights_after.astype(TSDF_CONTRIBUTION_SUM_DELTA_DTYPE)
        ):
            raise TsdfError(
                "TSDF block vector fusion sum exceeds its weight envelope"
            )

    @property
    def voxel_count(self) -> int:
        return TSDF_BLOCK_VOXELS

    @property
    def evaluated_count(self) -> int:
        return sum(field.evaluated_count for field in self.fields)

    @property
    def applied_count(self) -> int:
        return sum(field.contributing_count for field in self.fields)

    @property
    def skipped_count(self) -> int:
        return self.evaluated_count - self.applied_count

    @property
    def storage_slots_updated(self) -> int:
        return int(
            np.count_nonzero(self.weights_after != self.weights_before)
        )

    @property
    def observed_voxel_count(self) -> int:
        return int(np.count_nonzero(self.weights_after))

    @property
    def unknown_voxel_count(self) -> int:
        return self.voxel_count - self.observed_voxel_count

    @property
    def nonzero_sum_count(self) -> int:
        return int(np.count_nonzero(self.tsdf_sums_after))

    @property
    def weight_delta(self) -> int:
        return int(
            self.weights_after.astype(np.uint64).sum()
            - self.weights_before.astype(np.uint64).sum()
        )

    @property
    def maximum_weight_after(self) -> int:
        return int(self.weights_after.max())

    @property
    def status_counts(
        self,
    ) -> tuple[tuple[TsdfContributionStatus, int], ...]:
        totals: dict[TsdfContributionStatus, int] = {}
        for field in self.fields:
            for status, count in field.status_counts:
                totals[status] = totals.get(status, 0) + count
        return tuple(
            (status, totals[status])
            for status in TsdfContributionStatus
            if status in totals
        )


def fuse_tsdf_block_from_vector_fields(
    storage: TsdfBlockStorage,
    block_index_xyz: _BlockIndex,
    context: TsdfReplayDepthContext,
) -> TsdfBlockVectorFusionReceipt:
    """Fuse one planned block from one vector field per observation."""

    if not isinstance(storage, TsdfBlockStorage):
        raise TsdfError(
            "TSDF block vector fusion requires allocated TsdfBlockStorage"
        )
    compose_tsdf_global_voxel_index(block_index_xyz, (0, 0, 0))
    if not isinstance(context, TsdfReplayDepthContext):
        raise TsdfError(
            "TSDF block vector fusion requires a prepared "
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
            "canonical block fusion selection"
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

    identity_before = _block_storage_identity(storage)
    try:
        stored_sums = storage.tsdf_sums[block_row].copy(order="C")
        stored_weights = storage.weights[block_row].copy(order="C")
    except (MemoryError, ValueError) as error:
        raise TsdfError(
            "cannot snapshot the selected TSDF block for vector fusion"
        ) from error
    stored_sum_bytes = stored_sums.tobytes()
    stored_weight_bytes = stored_weights.tobytes()
    sums_before = stored_sums.reshape(TSDF_BLOCK_VOXELS)
    weights_before = stored_weights.reshape(TSDF_BLOCK_VOXELS)
    _require_empty_row(sums_before, weights_before)

    try:
        fields = tuple(
            evaluate_tsdf_block_contributions_from_context(
                storage,
                block_index_xyz,
                context,
                sequence,
            )
            for sequence in selected_sequences
        )
        _require_block_row_bytes(
            storage,
            block_index_xyz,
            block_row,
            identity_before,
            stored_sum_bytes,
            stored_weight_bytes,
            "changed while evaluating its observation fields",
        )

        sums_after = sums_before.astype(TSDF_CONTRIBUTION_SUM_DELTA_DTYPE)
        weights_after = weights_before.astype(np.uint64)
        pending_weight = np.zeros(TSDF_BLOCK_VOXELS, dtype=np.uint64)
        for field in fields:
            pending_weight += field.weight_deltas
        if int((weights_after + pending_weight).max()) > MAX_TSDF_VOXEL_WEIGHT:
            raise TsdfError(
                "TSDF block vector fusion would exceed the uint32 voxel "
                "weight maximum"
            )

        with np.errstate(over="ignore", invalid="ignore"):
            for field in fields:
                sums_after += field.tsdf_sum_deltas
                weights_after += field.weight_deltas
        if not np.all(np.isfinite(sums_after)):
            raise TsdfError(
                "TSDF block vector fusion would produce a non-finite sum"
            )
        if np.any(
            np.abs(sums_after)
            > weights_after.astype(TSDF_CONTRIBUTION_SUM_DELTA_DTYPE)
        ):
            raise TsdfError(
                "TSDF block vector fusion would exceed the weight envelope"
            )
        applied_weights = weights_after.astype(
            TSDF_CONTRIBUTION_WEIGHT_DELTA_DTYPE
        )

        expected_sum_bytes = sums_after.tobytes()
        expected_weight_bytes = applied_weights.tobytes()
        storage.tsdf_sums[block_row] = sums_after.reshape(
            stored_sums.shape
        )
        storage.weights[block_row] = applied_weights.reshape(
            stored_weights.shape
        )
        _require_block_row_bytes(
            storage,
            block_index_xyz,
            block_row,
            identity_before,
            expected_sum_bytes,
            expected_weight_bytes,
            "could not verify its vector writes",
        )

        result = TsdfBlockVectorFusionReceipt(
            block_index_xyz=block_index_xyz,
            block_row=block_row,
            source_plan_digest_sha256=plan.artifact_digest_sha256,
            replay_digest_sha256=plan.replay_digest_sha256,
            block_resolution=plan.block_resolution,
            frame_stride=plan.frame_stride,
            total_observations=plan.total_observations,
            selected_observation_sequences=selected_sequences,
            fields=fields,
            tsdf_sums_before=_freeze(sums_before),
            weights_before=_freeze(weights_before),
            tsdf_sums_after=_freeze(sums_after),
            weights_after=_freeze(applied_weights),
        )
    except Exception as error:
        _restore_block_row_or_fail(
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
            f"cannot apply TSDF block vector fusion: {error}"
        ) from error
    return result


def _require_empty_row(
    sums_before: np.ndarray,
    weights_before: np.ndarray,
) -> None:
    for local_flat_index in range(TSDF_BLOCK_VOXELS):
        stored_sum = sums_before[local_flat_index]
        stored_weight = weights_before[local_flat_index]
        try:
            _validate_target_prestate(
                stored_sum,
                stored_weight,
                float(stored_sum),
                int(stored_weight),
            )
        except TsdfError as error:
            raise TsdfError(
                "TSDF block vector fusion local flat "
                f"{local_flat_index}: {error}"
            ) from error
        if int(stored_weight) != 0:
            raise TsdfError(
                "TSDF block vector fusion requires a canonical empty "
                "selected block"
            )


def _require_block_row_bytes(
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
        or storage.tsdf_sums[block_row].tobytes() != expected_sum_bytes
        or storage.weights[block_row].tobytes() != expected_weight_bytes
    ):
        raise TsdfError(f"TSDF block vector fusion {problem}")


def _restore_block_row_or_fail(
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
            or storage.tsdf_sums[block_row].tobytes() != stored_sum_bytes
            or storage.weights[block_row].tobytes() != stored_weight_bytes
        ):
            raise ValueError(
                "restored block bytes or storage identity do not match"
            )
    except Exception as rollback_error:
        raise TsdfError(
            "TSDF block vector fusion failed and rollback failed; storage "
            "may be inconsistent"
        ) from rollback_error
