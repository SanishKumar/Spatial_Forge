"""Frame-major fusion tracked by a per-block observation ledger."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .errors import SessionReplayError, TsdfError
from .tsdf_block_plan import TSDF_BLOCK_RESOLUTION
from .tsdf_block_plan_loader import TsdfBlockPlan
from .tsdf_block_storage import TSDF_BLOCK_VOXELS, TsdfBlockStorage
from .tsdf_observation_block_rays import _is_sha256, _validate_block_index_xyz
from .tsdf_plan_traversal import (
    MAX_TSDF_PLAN_TRAVERSAL_OUTCOMES,
    _plan_storage_identity,
    _validate_plan_storage_rows,
)
from .tsdf_replay_depth_context import TsdfReplayDepthContext
from .tsdf_voxel_address import (
    compose_tsdf_global_voxel_index,
    locate_tsdf_voxel,
)
from .tsdf_voxel_contribution import (
    TsdfContributionStatus,
    _validate_contribution_context,
    _validate_contribution_plan,
    evaluate_tsdf_voxel_contribution_from_context,
)
from .tsdf_voxel_update import (
    _validate_update_storage,
    apply_tsdf_voxel_contribution_from_context,
)

_Index3 = tuple[int, int, int]


def _local_index_for_flat(local_flat_index: int) -> _Index3:
    return (
        local_flat_index % TSDF_BLOCK_RESOLUTION,
        (local_flat_index // TSDF_BLOCK_RESOLUTION) % TSDF_BLOCK_RESOLUTION,
        local_flat_index // (TSDF_BLOCK_RESOLUTION**2),
    )


@dataclass(frozen=True, slots=True)
class TsdfObservationLedger:
    """Per-block record of how many observations a row has absorbed.

    Each entry is a *prefix length*, not a set. Observations are absorbed in
    canonical order, so a row that has absorbed ``k`` has absorbed exactly
    sequences ``0..k-1`` of the selection. That restriction is what keeps
    float64 accumulation bit-identical to the one-shot traversal: addition is
    commutative but not associative, so per-voxel sums must accumulate in the
    same order however the work is chunked.
    """

    source_plan_digest_sha256: str
    replay_digest_sha256: str
    plan_block_indices: tuple[_Index3, ...]
    selected_observation_sequences: tuple[int, ...]
    absorbed_counts: tuple[int, ...]

    def __post_init__(self) -> None:
        if not _is_sha256(self.source_plan_digest_sha256):
            raise TsdfError(
                "TSDF observation ledger source plan digest is invalid"
            )
        if not _is_sha256(self.replay_digest_sha256):
            raise TsdfError(
                "TSDF observation ledger replay digest is invalid"
            )
        if (
            not isinstance(self.plan_block_indices, tuple)
            or not self.plan_block_indices
        ):
            raise TsdfError(
                "TSDF observation ledger requires at least one plan row"
            )
        previous_key: tuple[int, int, int] | None = None
        for block_index in self.plan_block_indices:
            _validate_block_index_xyz(block_index, "plan block")
            key = (block_index[2], block_index[1], block_index[0])
            if previous_key is not None and key <= previous_key:
                raise TsdfError(
                    "TSDF observation ledger plan rows must be unique and "
                    "strictly x-fastest ordered"
                )
            previous_key = key
        if (
            not isinstance(self.selected_observation_sequences, tuple)
            or not self.selected_observation_sequences
        ):
            raise TsdfError(
                "TSDF observation ledger requires a nonempty selection"
            )
        if (
            not isinstance(self.absorbed_counts, tuple)
            or len(self.absorbed_counts) != len(self.plan_block_indices)
        ):
            raise TsdfError(
                "TSDF observation ledger requires one count per plan row"
            )
        limit = len(self.selected_observation_sequences)
        for count in self.absorbed_counts:
            if (
                isinstance(count, bool)
                or not isinstance(count, int)
                or count < 0
                or count > limit
            ):
                raise TsdfError(
                    "TSDF observation ledger counts must lie in "
                    f"[0, {limit}]"
                )

    @property
    def plan_block_count(self) -> int:
        return len(self.plan_block_indices)

    @property
    def observation_count(self) -> int:
        return len(self.selected_observation_sequences)

    @property
    def absorbed_pair_count(self) -> int:
        return sum(self.absorbed_counts)

    @property
    def total_pair_count(self) -> int:
        return self.plan_block_count * self.observation_count

    @property
    def pending_pair_count(self) -> int:
        return self.total_pair_count - self.absorbed_pair_count

    @property
    def is_complete(self) -> bool:
        return self.pending_pair_count == 0

    @property
    def untouched_block_indices(self) -> tuple[_Index3, ...]:
        return tuple(
            block_index
            for block_index, count in zip(
                self.plan_block_indices,
                self.absorbed_counts,
            )
            if count == 0
        )

    def absorbed_for(self, block_index: _Index3) -> int:
        try:
            position = self.plan_block_indices.index(block_index)
        except ValueError as error:
            raise TsdfError(
                f"TSDF observation ledger has no row {block_index}"
            ) from error
        return self.absorbed_counts[position]


@dataclass(frozen=True, slots=True)
class TsdfObservationFusionReceipt:
    """Immutable transcript for one frame-major fusion pass."""

    source_plan_digest_sha256: str
    replay_digest_sha256: str
    ledger_before: TsdfObservationLedger
    ledger_after: TsdfObservationLedger
    fused_pairs: tuple[tuple[_Index3, int], ...]
    evaluated_count: int
    applied_count: int
    weight_delta: int
    status_counts: tuple[tuple[TsdfContributionStatus, int], ...]

    def __post_init__(self) -> None:
        for ledger, label in (
            (self.ledger_before, "before"),
            (self.ledger_after, "after"),
        ):
            if not isinstance(ledger, TsdfObservationLedger):
                raise TsdfError(
                    f"TSDF observation fusion {label} ledger is invalid"
                )
        if (
            self.ledger_before.source_plan_digest_sha256
            != self.source_plan_digest_sha256
            or self.ledger_after.source_plan_digest_sha256
            != self.source_plan_digest_sha256
            or self.ledger_before.plan_block_indices
            != self.ledger_after.plan_block_indices
            or self.ledger_before.selected_observation_sequences
            != self.ledger_after.selected_observation_sequences
        ):
            raise TsdfError(
                "TSDF observation fusion ledger provenance is inconsistent"
            )
        for before, after in zip(
            self.ledger_before.absorbed_counts,
            self.ledger_after.absorbed_counts,
        ):
            if after < before:
                raise TsdfError(
                    "TSDF observation fusion must never un-absorb an "
                    "observation"
                )
        expected_pairs = (
            self.ledger_after.absorbed_pair_count
            - self.ledger_before.absorbed_pair_count
        )
        if not isinstance(self.fused_pairs, tuple):
            raise TsdfError("TSDF observation fusion pairs must be a tuple")
        if len(self.fused_pairs) != expected_pairs:
            raise TsdfError(
                "TSDF observation fusion pairs must equal the ledger delta"
            )
        for value in (
            self.evaluated_count,
            self.applied_count,
            self.weight_delta,
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise TsdfError(
                    "TSDF observation fusion counts must be nonnegative "
                    "integers"
                )
        if self.applied_count != self.weight_delta:
            raise TsdfError(
                "TSDF observation fusion applies exactly one weight per "
                "accepted contribution"
            )
        if self.evaluated_count != len(self.fused_pairs) * TSDF_BLOCK_VOXELS:
            raise TsdfError(
                "TSDF observation fusion must evaluate every voxel of every "
                "fused pair"
            )

    @property
    def pairs_fused_now(self) -> int:
        return len(self.fused_pairs)

    @property
    def pairs_pending(self) -> int:
        return self.ledger_after.pending_pair_count

    @property
    def is_complete(self) -> bool:
        return self.ledger_after.is_complete

    @property
    def skipped_count(self) -> int:
        return self.evaluated_count - self.applied_count


def begin_tsdf_observation_ledger(
    plan: TsdfBlockPlan,
    context: TsdfReplayDepthContext,
) -> TsdfObservationLedger:
    """Open a ledger with every planned row holding zero observations."""

    if not isinstance(plan, TsdfBlockPlan):
        raise TsdfError(
            "TSDF observation ledger requires a loaded TsdfBlockPlan"
        )
    if not isinstance(context, TsdfReplayDepthContext):
        raise TsdfError(
            "TSDF observation ledger requires a prepared "
            "TsdfReplayDepthContext"
        )
    _validate_contribution_plan(plan)
    _validate_contribution_context(plan, context)
    selected = tuple(range(0, plan.total_observations, plan.frame_stride))
    return TsdfObservationLedger(
        source_plan_digest_sha256=plan.artifact_digest_sha256,
        replay_digest_sha256=plan.replay_digest_sha256,
        plan_block_indices=plan.active_blocks,
        selected_observation_sequences=selected,
        absorbed_counts=(0,) * len(plan.active_blocks),
    )


def fuse_tsdf_plan_observations_from_context(
    storage: TsdfBlockStorage,
    context: TsdfReplayDepthContext,
    ledger: TsdfObservationLedger,
    *,
    pair_limit: int | None = None,
) -> TsdfObservationFusionReceipt:
    """Absorb pending (row, observation) pairs in frame-major order.

    Pairs are walked observation-outer, row-inner, which is the order a
    streaming capture would produce them in: one frame applied across every
    row it touches, then the next frame. A pair is skipped when its row has
    already absorbed that observation, so repeating a pass is a no-op.
    """

    if not isinstance(storage, TsdfBlockStorage):
        raise TsdfError(
            "TSDF observation fusion requires allocated TsdfBlockStorage"
        )
    if not isinstance(context, TsdfReplayDepthContext):
        raise TsdfError(
            "TSDF observation fusion requires a prepared "
            "TsdfReplayDepthContext"
        )
    if not isinstance(ledger, TsdfObservationLedger):
        raise TsdfError(
            "TSDF observation fusion requires a TsdfObservationLedger"
        )
    if pair_limit is not None and (
        isinstance(pair_limit, bool)
        or not isinstance(pair_limit, int)
        or pair_limit < 1
    ):
        raise TsdfError("pair_limit: expected a positive integer or None")

    _validate_update_storage(storage)
    plan = storage.source_plan
    _validate_contribution_plan(plan)
    _validate_contribution_context(plan, context)
    block_indices = _validate_plan_storage_rows(storage)
    selected = tuple(range(0, plan.total_observations, plan.frame_stride))
    if (
        ledger.source_plan_digest_sha256 != plan.artifact_digest_sha256
        or ledger.replay_digest_sha256 != plan.replay_digest_sha256
        or ledger.plan_block_indices != block_indices
        or ledger.selected_observation_sequences != selected
    ):
        raise TsdfError(
            "TSDF observation ledger provenance does not match the block "
            "storage"
        )
    if (
        context.selected_observation_sequences != selected
        or len(context.observations) != len(selected)
    ):
        raise TsdfError(
            "TSDF replay/depth context observations are not the complete "
            "canonical fusion selection"
        )

    _require_storage_matches_observation_ledger(
        storage,
        block_indices,
        ledger,
    )

    pending: list[tuple[int, int]] = []
    for observation_position, _ in enumerate(selected):
        for block_row, _ in enumerate(block_indices):
            if ledger.absorbed_counts[block_row] <= observation_position:
                pending.append((block_row, observation_position))
    # Walking observation positions in increasing order means a row only
    # reaches position k once it has absorbed 0..k-1, so the per-voxel
    # accumulation order stays canonical without extra filtering.
    if pair_limit is not None:
        pending = pending[:pair_limit]

    outcome_count = len(pending) * TSDF_BLOCK_VOXELS
    if outcome_count > MAX_TSDF_PLAN_TRAVERSAL_OUTCOMES:
        raise TsdfError(
            "TSDF observation fusion pass requires "
            f"{outcome_count} retained contribution outcomes; reference "
            f"maximum is {MAX_TSDF_PLAN_TRAVERSAL_OUTCOMES}. Use a smaller "
            "pair_limit or a future scalable fusion path"
        )

    identity_before = _plan_storage_identity(storage)
    counts = list(ledger.absorbed_counts)
    fused_pairs: list[tuple[_Index3, int]] = []
    touched_rows: dict[int, bytes] = {}
    evaluated = 0
    applied = 0
    statuses: dict[TsdfContributionStatus, int] = {}
    try:
        for block_row, observation_position in pending:
            if counts[block_row] != observation_position:
                raise TsdfError(
                    "TSDF observation fusion must absorb observations in "
                    "canonical order per row"
                )
            if block_row not in touched_rows:
                touched_rows[block_row] = (
                    storage.tsdf_sums[block_row].tobytes()
                    + storage.weights[block_row].tobytes()
                )
            block_index = block_indices[block_row]
            observation_sequence = selected[observation_position]
            for local_flat_index in range(TSDF_BLOCK_VOXELS):
                address = locate_tsdf_voxel(
                    storage,
                    compose_tsdf_global_voxel_index(
                        block_index,
                        _local_index_for_flat(local_flat_index),
                    ),
                )
                if address is None or address.block_row != block_row:
                    raise TsdfError(
                        "TSDF observation fusion address lookup is "
                        "inconsistent"
                    )
                contribution = evaluate_tsdf_voxel_contribution_from_context(
                    storage,
                    address,
                    context,
                    observation_sequence,
                )
                evaluated += 1
                statuses[contribution.status] = (
                    statuses.get(contribution.status, 0) + 1
                )
                if contribution.status is TsdfContributionStatus.CONTRIBUTES:
                    apply_tsdf_voxel_contribution_from_context(
                        storage,
                        contribution,
                        context,
                    )
                    applied += 1
            counts[block_row] = observation_position + 1
            fused_pairs.append((block_index, observation_sequence))

        ledger_after = TsdfObservationLedger(
            source_plan_digest_sha256=ledger.source_plan_digest_sha256,
            replay_digest_sha256=ledger.replay_digest_sha256,
            plan_block_indices=ledger.plan_block_indices,
            selected_observation_sequences=(
                ledger.selected_observation_sequences
            ),
            absorbed_counts=tuple(counts),
        )
        result = TsdfObservationFusionReceipt(
            source_plan_digest_sha256=plan.artifact_digest_sha256,
            replay_digest_sha256=plan.replay_digest_sha256,
            ledger_before=ledger,
            ledger_after=ledger_after,
            fused_pairs=tuple(fused_pairs),
            evaluated_count=evaluated,
            applied_count=applied,
            weight_delta=applied,
            status_counts=tuple(
                (status, statuses[status])
                for status in TsdfContributionStatus
                if status in statuses
            ),
        )
    except Exception as error:
        _restore_touched_rows_or_fail(
            storage,
            identity_before,
            touched_rows,
        )
        if isinstance(error, (TsdfError, SessionReplayError)):
            raise
        raise TsdfError(
            f"cannot fuse TSDF plan observations: {error}"
        ) from error
    return result


def _require_storage_matches_observation_ledger(
    storage: TsdfBlockStorage,
    block_indices: tuple[_Index3, ...],
    ledger: TsdfObservationLedger,
) -> None:
    """Reject storage whose never-touched rows disagree with the ledger."""

    try:
        for block_row, block_index in enumerate(block_indices):
            if ledger.absorbed_counts[block_row]:
                continue
            if bool(
                np.any(storage.tsdf_sums[block_row].view(np.uint8))
                or np.any(storage.weights[block_row].view(np.uint8))
            ):
                raise TsdfError(
                    "TSDF observation fusion found a nonempty row its ledger "
                    f"records as untouched: {block_index}"
                )
    except TsdfError:
        raise
    except (MemoryError, TypeError, ValueError) as error:
        raise TsdfError(
            "cannot inspect TSDF plan storage against its observation ledger"
        ) from error


def _restore_touched_rows_or_fail(
    storage: TsdfBlockStorage,
    identity_before: tuple[object, ...],
    touched_rows: dict[int, bytes],
) -> None:
    """Restore the exact pre-pass bytes of every row this pass wrote."""

    try:
        sum_bytes = storage.tsdf_sums[0].nbytes
        for block_row, snapshot in touched_rows.items():
            sums = np.frombuffer(
                snapshot[:sum_bytes],
                dtype=storage.tsdf_sums.dtype,
            ).reshape(storage.tsdf_sums[block_row].shape)
            weights = np.frombuffer(
                snapshot[sum_bytes:],
                dtype=storage.weights.dtype,
            ).reshape(storage.weights[block_row].shape)
            storage.tsdf_sums[block_row] = sums
            storage.weights[block_row] = weights
        _validate_update_storage(storage)
        if _plan_storage_identity(storage) != identity_before:
            raise ValueError("restored storage identity does not match")
        for block_row, snapshot in touched_rows.items():
            current = (
                storage.tsdf_sums[block_row].tobytes()
                + storage.weights[block_row].tobytes()
            )
            if current != snapshot:
                raise ValueError("restored row bytes do not match")
    except Exception as rollback_error:
        raise TsdfError(
            "TSDF observation fusion failed and rollback failed; storage may "
            "be inconsistent"
        ) from rollback_error
