"""Resumable, idempotent fusion of planned TSDF block rows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .errors import SessionReplayError, TsdfError
from .tsdf_block_plan import _ordered_blocks
from .tsdf_block_plan_loader import TsdfBlockPlan
from .tsdf_block_storage import (
    TSDF_BLOCK_VOXELS,
    TsdfBlockStorage,
)
from .tsdf_block_traversal import (
    TsdfBlockTraversalReceipt,
    traverse_tsdf_block_voxels_from_context,
)
from .tsdf_observation_block_rays import _is_sha256, _validate_block_index_xyz
from .tsdf_plan_traversal import (
    MAX_TSDF_PLAN_TRAVERSAL_OUTCOMES,
    _plan_storage_identity,
    _validate_plan_storage_rows,
)
from .tsdf_replay_depth_context import TsdfReplayDepthContext
from .tsdf_voxel_contribution import (
    TsdfContributionStatus,
    _validate_contribution_context,
    _validate_contribution_plan,
)
from .tsdf_voxel_update import _validate_update_storage

_Index3 = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class TsdfFusionLedger:
    """Immutable record of which planned block rows have been fused.

    The ledger is what replaces the blanket empty-storage guard. Storage is
    no longer required to be empty, only to be *consistent with its ledger*:
    every row the ledger does not claim must still be untouched.
    """

    source_plan_digest_sha256: str
    replay_digest_sha256: str
    plan_block_indices: tuple[_Index3, ...]
    fused_block_indices: tuple[_Index3, ...]

    def __post_init__(self) -> None:
        if not _is_sha256(self.source_plan_digest_sha256):
            raise TsdfError("TSDF fusion ledger source plan digest is invalid")
        if not _is_sha256(self.replay_digest_sha256):
            raise TsdfError("TSDF fusion ledger replay digest is invalid")
        _validate_canonical_ledger_blocks(
            self.plan_block_indices,
            "plan blocks",
        )
        _validate_canonical_ledger_blocks(
            self.fused_block_indices,
            "fused blocks",
        )
        if not self.plan_block_indices:
            raise TsdfError("TSDF fusion ledger requires at least one row")
        if not set(self.fused_block_indices) <= set(self.plan_block_indices):
            raise TsdfError(
                "TSDF fusion ledger fused rows must belong to the plan"
            )

    @property
    def plan_block_count(self) -> int:
        return len(self.plan_block_indices)

    @property
    def fused_block_count(self) -> int:
        return len(self.fused_block_indices)

    @property
    def pending_block_indices(self) -> tuple[_Index3, ...]:
        fused = set(self.fused_block_indices)
        return tuple(
            block_index
            for block_index in self.plan_block_indices
            if block_index not in fused
        )

    @property
    def pending_block_count(self) -> int:
        return self.plan_block_count - self.fused_block_count

    @property
    def is_complete(self) -> bool:
        return self.pending_block_count == 0


@dataclass(frozen=True, slots=True)
class TsdfPlanFusionReceipt:
    """Immutable transcript for one resumable fusion pass."""

    source_plan_digest_sha256: str
    replay_digest_sha256: str
    selected_observation_sequences: tuple[int, ...]
    ledger_before: TsdfFusionLedger
    ledger_after: TsdfFusionLedger
    block_receipts: tuple[TsdfBlockTraversalReceipt, ...]

    def __post_init__(self) -> None:
        for ledger, label in (
            (self.ledger_before, "before"),
            (self.ledger_after, "after"),
        ):
            if not isinstance(ledger, TsdfFusionLedger):
                raise TsdfError(
                    f"TSDF fusion receipt {label} ledger is invalid"
                )
        if (
            self.ledger_before.source_plan_digest_sha256
            != self.source_plan_digest_sha256
            or self.ledger_after.source_plan_digest_sha256
            != self.source_plan_digest_sha256
            or self.ledger_before.replay_digest_sha256
            != self.replay_digest_sha256
            or self.ledger_after.replay_digest_sha256
            != self.replay_digest_sha256
            or self.ledger_before.plan_block_indices
            != self.ledger_after.plan_block_indices
        ):
            raise TsdfError(
                "TSDF fusion receipt ledger provenance is inconsistent"
            )
        before = set(self.ledger_before.fused_block_indices)
        after = set(self.ledger_after.fused_block_indices)
        if not before <= after:
            raise TsdfError(
                "TSDF fusion must never un-fuse a previously fused row"
            )
        if not isinstance(self.block_receipts, tuple):
            raise TsdfError("TSDF fusion receipt rows must be a tuple")
        fused_now = _ordered_blocks(after - before)
        if (
            tuple(
                receipt.block_index_xyz for receipt in self.block_receipts
            )
            != fused_now
        ):
            raise TsdfError(
                "TSDF fusion receipt rows must equal the ledger delta in "
                "canonical order"
            )
        for receipt in self.block_receipts:
            if not isinstance(receipt, TsdfBlockTraversalReceipt):
                raise TsdfError(
                    "TSDF fusion receipt contains an invalid block receipt"
                )
            if (
                receipt.source_plan_digest_sha256
                != self.source_plan_digest_sha256
                or receipt.replay_digest_sha256 != self.replay_digest_sha256
                or receipt.selected_observation_sequences
                != self.selected_observation_sequences
            ):
                raise TsdfError(
                    "TSDF fusion receipt block provenance is inconsistent"
                )

    @property
    def blocks_fused_now(self) -> int:
        return len(self.block_receipts)

    @property
    def blocks_already_fused(self) -> int:
        return self.ledger_before.fused_block_count

    @property
    def blocks_pending(self) -> int:
        return self.ledger_after.pending_block_count

    @property
    def is_complete(self) -> bool:
        return self.ledger_after.is_complete

    @property
    def evaluated_count(self) -> int:
        return sum(
            receipt.evaluated_count for receipt in self.block_receipts
        )

    @property
    def applied_count(self) -> int:
        return sum(receipt.applied_count for receipt in self.block_receipts)

    @property
    def weight_delta(self) -> int:
        return sum(receipt.weight_delta for receipt in self.block_receipts)

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


def begin_tsdf_fusion_ledger(plan: TsdfBlockPlan) -> TsdfFusionLedger:
    """Open an empty ledger for one plan: nothing fused yet."""

    if not isinstance(plan, TsdfBlockPlan):
        raise TsdfError("TSDF fusion ledger requires a loaded TsdfBlockPlan")
    _validate_contribution_plan(plan)
    return TsdfFusionLedger(
        source_plan_digest_sha256=plan.artifact_digest_sha256,
        replay_digest_sha256=plan.replay_digest_sha256,
        plan_block_indices=plan.active_blocks,
        fused_block_indices=(),
    )


def fuse_tsdf_plan_blocks_from_context(
    storage: TsdfBlockStorage,
    context: TsdfReplayDepthContext,
    ledger: TsdfFusionLedger,
    *,
    block_limit: int | None = None,
) -> TsdfPlanFusionReceipt:
    """Fuse pending planned rows, resuming from an existing ledger.

    Rows the ledger already claims are skipped rather than re-fused, so
    repeating a pass is a no-op. At most ``block_limit`` pending rows are
    fused, which is what makes the work resumable in bounded chunks.
    """

    if not isinstance(storage, TsdfBlockStorage):
        raise TsdfError(
            "TSDF plan fusion requires allocated TsdfBlockStorage"
        )
    if not isinstance(context, TsdfReplayDepthContext):
        raise TsdfError(
            "TSDF plan fusion requires a prepared TsdfReplayDepthContext"
        )
    if not isinstance(ledger, TsdfFusionLedger):
        raise TsdfError("TSDF plan fusion requires a TsdfFusionLedger")
    if block_limit is not None and (
        isinstance(block_limit, bool)
        or not isinstance(block_limit, int)
        or block_limit < 1
    ):
        raise TsdfError("block_limit: expected a positive integer or None")

    _validate_update_storage(storage)
    plan = storage.source_plan
    _validate_contribution_plan(plan)
    _validate_contribution_context(plan, context)
    block_indices = _validate_plan_storage_rows(storage)
    if (
        ledger.source_plan_digest_sha256 != plan.artifact_digest_sha256
        or ledger.replay_digest_sha256 != plan.replay_digest_sha256
        or ledger.plan_block_indices != block_indices
    ):
        raise TsdfError(
            "TSDF fusion ledger provenance does not match the block storage"
        )

    selected_sequences = tuple(
        range(0, plan.total_observations, plan.frame_stride)
    )
    if (
        context.selected_observation_sequences != selected_sequences
        or len(context.observations) != len(selected_sequences)
    ):
        raise TsdfError(
            "TSDF replay/depth context observations are not the complete "
            "canonical fusion selection"
        )

    identity_before = _plan_storage_identity(storage)
    _require_storage_matches_ledger(storage, block_indices, ledger)

    pending = ledger.pending_block_indices
    if block_limit is not None:
        pending = pending[:block_limit]
    outcome_count = (
        len(pending) * TSDF_BLOCK_VOXELS * len(selected_sequences)
    )
    if outcome_count > MAX_TSDF_PLAN_TRAVERSAL_OUTCOMES:
        raise TsdfError(
            "TSDF plan fusion pass requires "
            f"{outcome_count} retained contribution outcomes; reference "
            f"maximum is {MAX_TSDF_PLAN_TRAVERSAL_OUTCOMES}. Use a smaller "
            "block_limit, a larger frame stride, or a future scalable "
            "fusion path"
        )

    block_receipts: list[TsdfBlockTraversalReceipt] = []
    fused_rows: list[int] = []
    try:
        for block_index in pending:
            block_row = block_indices.index(block_index)
            receipt = traverse_tsdf_block_voxels_from_context(
                storage,
                block_index,
                context,
            )
            if (
                not isinstance(receipt, TsdfBlockTraversalReceipt)
                or receipt.block_index_xyz != block_index
                or receipt.block_row != block_row
            ):
                raise TsdfError(
                    "TSDF plan fusion child block scope is inconsistent"
                )
            fused_rows.append(block_row)
            block_receipts.append(receipt)

        ledger_after = TsdfFusionLedger(
            source_plan_digest_sha256=ledger.source_plan_digest_sha256,
            replay_digest_sha256=ledger.replay_digest_sha256,
            plan_block_indices=ledger.plan_block_indices,
            fused_block_indices=_ordered_blocks(
                set(ledger.fused_block_indices) | set(pending)
            ),
        )
        result = TsdfPlanFusionReceipt(
            source_plan_digest_sha256=plan.artifact_digest_sha256,
            replay_digest_sha256=plan.replay_digest_sha256,
            selected_observation_sequences=selected_sequences,
            ledger_before=ledger,
            ledger_after=ledger_after,
            block_receipts=tuple(block_receipts),
        )
    except Exception as error:
        _restore_rows_or_fail(storage, identity_before, fused_rows)
        if isinstance(error, (TsdfError, SessionReplayError)):
            raise
        raise TsdfError(
            f"cannot fuse TSDF plan blocks: {error}"
        ) from error
    return result


def _row_is_empty(storage: TsdfBlockStorage, block_row: int) -> bool:
    return not bool(
        np.any(storage.tsdf_sums[block_row].view(np.uint8))
        or np.any(storage.weights[block_row].view(np.uint8))
    )


def _require_storage_matches_ledger(
    storage: TsdfBlockStorage,
    block_indices: tuple[_Index3, ...],
    ledger: TsdfFusionLedger,
) -> None:
    """Reject storage whose untouched rows disagree with the ledger.

    A row the ledger does not claim must still be all-zero. Anything else
    means storage was written outside this fusion path, and resuming on top
    of it could silently double-count.
    """

    fused = set(ledger.fused_block_indices)
    try:
        for block_row, block_index in enumerate(block_indices):
            if block_index in fused:
                continue
            if not _row_is_empty(storage, block_row):
                raise TsdfError(
                    "TSDF plan fusion found a nonempty row that its ledger "
                    f"does not claim: {block_index}"
                )
    except TsdfError:
        raise
    except (MemoryError, TypeError, ValueError) as error:
        raise TsdfError(
            "cannot inspect TSDF plan storage against its fusion ledger"
        ) from error


def _restore_rows_or_fail(
    storage: TsdfBlockStorage,
    identity_before: tuple[object, ...],
    block_rows: list[int],
) -> None:
    """Zero only the rows this pass fused, leaving earlier work intact."""

    try:
        for block_row in block_rows:
            storage.tsdf_sums[block_row].fill(np.float64(0.0))
            storage.weights[block_row].fill(np.uint32(0))
        _validate_update_storage(storage)
        if _plan_storage_identity(storage) != identity_before:
            raise ValueError("restored storage identity does not match")
        for block_row in block_rows:
            if not _row_is_empty(storage, block_row):
                raise ValueError("restored row is not empty")
    except Exception as rollback_error:
        raise TsdfError(
            "TSDF plan fusion failed and rollback failed; storage may be "
            "inconsistent"
        ) from rollback_error


def _validate_canonical_ledger_blocks(value: object, label: str) -> None:
    if not isinstance(value, tuple):
        raise TsdfError(f"TSDF fusion ledger {label} must be a tuple")
    previous_key: tuple[int, int, int] | None = None
    for block_index in value:
        _validate_block_index_xyz(block_index, label)
        key = (block_index[2], block_index[1], block_index[0])
        if previous_key is not None and key <= previous_key:
            raise TsdfError(
                f"TSDF fusion ledger {label} must be unique and strictly "
                "x-fastest ordered"
            )
        previous_key = key
