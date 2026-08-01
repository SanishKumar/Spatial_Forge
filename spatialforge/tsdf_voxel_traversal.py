"""One-shot selected-observation traversal for one planned TSDF voxel."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .errors import SessionReplayError, TsdfError
from .model import ScanSession
from .replay import replay_session
from .tsdf_block_storage import TsdfBlockStorage
from .tsdf_replay_depth_context import TsdfReplayDepthContext
from .tsdf_voxel_address import TsdfVoxelAddress, locate_tsdf_voxel
from .tsdf_voxel_contribution import (
    TsdfContributionStatus,
    TsdfVoxelContribution,
    _validate_contribution_context,
    _validate_contribution_plan,
    evaluate_tsdf_voxel_contribution,
    evaluate_tsdf_voxel_contribution_from_context,
)
from .tsdf_voxel_update import (
    MAX_TSDF_VOXEL_WEIGHT,
    TsdfVoxelUpdateReceipt,
    _restore_target_or_fail,
    _storage_layout_identity,
    _validate_target_prestate,
    _validate_update_storage,
    apply_tsdf_voxel_contribution,
    apply_tsdf_voxel_contribution_from_context,
)


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
    except OverflowError:
        return False


@dataclass(frozen=True, slots=True)
class TsdfVoxelTraversalReceipt:
    """Immutable transcript for one successful one-voxel traversal."""

    address: TsdfVoxelAddress
    source_plan_digest_sha256: str
    replay_digest_sha256: str
    frame_stride: int
    total_observations: int
    selected_observation_sequences: tuple[int, ...]
    contributions: tuple[TsdfVoxelContribution, ...]
    update_receipts: tuple[TsdfVoxelUpdateReceipt, ...]
    tsdf_sum_before: float
    weight_before: int
    tsdf_sum_after: float
    weight_after: int

    def __post_init__(self) -> None:
        if not isinstance(self.address, TsdfVoxelAddress):
            raise TsdfError(
                "TSDF voxel traversal receipt requires a voxel address"
            )
        if not _is_sha256(self.source_plan_digest_sha256):
            raise TsdfError(
                "TSDF voxel traversal source plan digest is invalid"
            )
        if not _is_sha256(self.replay_digest_sha256):
            raise TsdfError(
                "TSDF voxel traversal replay digest is invalid"
            )
        if (
            isinstance(self.frame_stride, bool)
            or not isinstance(self.frame_stride, int)
            or self.frame_stride < 1
        ):
            raise TsdfError(
                "TSDF voxel traversal frame stride must be positive"
            )
        if (
            isinstance(self.total_observations, bool)
            or not isinstance(self.total_observations, int)
            or self.total_observations < 1
        ):
            raise TsdfError(
                "TSDF voxel traversal total observations must be positive"
            )
        if (
            not isinstance(self.selected_observation_sequences, tuple)
            or not self.selected_observation_sequences
            or any(
                isinstance(sequence, bool)
                or not isinstance(sequence, int)
                or sequence < 0
                for sequence in self.selected_observation_sequences
            )
        ):
            raise TsdfError(
                "TSDF voxel traversal requires selected observations"
            )
        expected_sequences = tuple(
            range(0, self.total_observations, self.frame_stride)
        )
        if self.selected_observation_sequences != expected_sequences:
            raise TsdfError(
                "TSDF voxel traversal observation sequences are not the "
                "complete canonical stride order"
            )
        if (
            not isinstance(self.contributions, tuple)
            or len(self.contributions)
            != len(self.selected_observation_sequences)
        ):
            raise TsdfError(
                "TSDF voxel traversal contributions do not match the "
                "selected observations"
            )
        for sequence, contribution in zip(
            self.selected_observation_sequences,
            self.contributions,
            strict=True,
        ):
            if not isinstance(contribution, TsdfVoxelContribution):
                raise TsdfError(
                    "TSDF voxel traversal contains an invalid contribution"
                )
            if (
                contribution.observation_sequence != sequence
                or contribution.address != self.address
                or contribution.source_plan_digest_sha256
                != self.source_plan_digest_sha256
                or contribution.replay_digest_sha256
                != self.replay_digest_sha256
            ):
                raise TsdfError(
                    "TSDF voxel traversal contribution transcript is "
                    "inconsistent"
                )
        accepted = tuple(
            contribution
            for contribution in self.contributions
            if contribution.contributes
        )
        if (
            not isinstance(self.update_receipts, tuple)
            or len(self.update_receipts) != len(accepted)
        ):
            raise TsdfError(
                "TSDF voxel traversal updates do not match accepted "
                "contributions"
            )
        if (
            not _is_finite_number(self.tsdf_sum_before)
            or self.tsdf_sum_before != 0.0
            or bool(np.signbit(self.tsdf_sum_before))
            or isinstance(self.weight_before, bool)
            or not isinstance(self.weight_before, int)
            or self.weight_before != 0
        ):
            raise TsdfError(
                "TSDF voxel traversal receipt requires a canonical empty "
                "prior state"
            )

        running_sum = np.float64(self.tsdf_sum_before)
        running_weight = self.weight_before
        for contribution, receipt in zip(
            accepted,
            self.update_receipts,
            strict=True,
        ):
            if (
                not isinstance(receipt, TsdfVoxelUpdateReceipt)
                or receipt.contribution != contribution
                or receipt.tsdf_sum_before != float(running_sum)
                or receipt.weight_before != running_weight
            ):
                raise TsdfError(
                    "TSDF voxel traversal update chain is inconsistent"
                )
            running_sum = np.float64(receipt.tsdf_sum_after)
            running_weight = receipt.weight_after

        if (
            not _is_finite_number(self.tsdf_sum_after)
            or isinstance(self.weight_after, bool)
            or not isinstance(self.weight_after, int)
            or self.tsdf_sum_after != float(running_sum)
            or self.weight_after != running_weight
            or self.weight_after != len(self.update_receipts)
        ):
            raise TsdfError(
                "TSDF voxel traversal final accumulator state is "
                "inconsistent"
            )
        if (
            self.weight_after == 0
            and bool(np.signbit(self.tsdf_sum_after))
        ):
            raise TsdfError(
                "empty TSDF voxel traversal result must retain positive zero"
            )

    @property
    def evaluated_count(self) -> int:
        return len(self.contributions)

    @property
    def applied_count(self) -> int:
        return len(self.update_receipts)

    @property
    def skipped_count(self) -> int:
        return self.evaluated_count - self.applied_count

    @property
    def storage_slots_updated(self) -> int:
        return int(self.applied_count > 0)

    @property
    def tsdf_sum_delta(self) -> float:
        return float(
            np.float64(self.tsdf_sum_after)
            - np.float64(self.tsdf_sum_before)
        )

    @property
    def weight_delta(self) -> int:
        return self.weight_after - self.weight_before

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
                    for contribution in self.contributions
                )
            )
        )


def traverse_tsdf_voxel_observations(
    storage: TsdfBlockStorage,
    address: TsdfVoxelAddress,
    session: ScanSession,
) -> TsdfVoxelTraversalReceipt:
    """Evaluate and apply all plan-selected observations to one empty slot."""

    if not isinstance(storage, TsdfBlockStorage):
        raise TsdfError(
            "TSDF voxel traversal requires allocated TsdfBlockStorage"
        )
    if not isinstance(address, TsdfVoxelAddress):
        raise TsdfError(
            "TSDF voxel traversal requires a TsdfVoxelAddress"
        )
    if not isinstance(session, ScanSession):
        raise TsdfError(
            "TSDF voxel traversal requires a loaded ScanSession"
        )

    _validate_update_storage(storage)
    resolved_address = locate_tsdf_voxel(
        storage,
        address.global_index_xyz,
    )
    if resolved_address != address:
        raise TsdfError(
            "TSDF voxel traversal address does not match destination storage"
        )

    plan = storage.source_plan
    _validate_traversal_plan(plan)
    if plan.session_id != session.session_id:
        raise TsdfError(
            "TSDF block plan session_id does not match the loaded session"
        )
    starting_replay = replay_session(session)
    if starting_replay.digest_sha256 != plan.replay_digest_sha256:
        raise TsdfError(
            "TSDF block plan replay digest does not match current session "
            "inputs; regenerate the plan"
        )
    if len(starting_replay.observations) != plan.total_observations:
        raise TsdfError(
            "TSDF block plan total observations do not match current replay"
        )
    selected_sequences = tuple(
        range(0, plan.total_observations, plan.frame_stride)
    )
    if len(selected_sequences) != plan.selected_observations:
        raise TsdfError(
            "TSDF block plan selected observations do not match its stride"
        )

    array_index = address.array_index_bzyx
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
    if weight_before != 0:
        raise TsdfError(
            "TSDF voxel traversal requires a canonical empty target slot"
        )
    layout_before = _storage_layout_identity(storage)
    stored_sum_bytes = stored_sum.tobytes()
    stored_weight_bytes = stored_weight.tobytes()

    contributions = tuple(
        evaluate_tsdf_voxel_contribution(
            storage,
            address,
            session,
            sequence,
        )
        for sequence in selected_sequences
    )
    evaluation_replay = replay_session(session)
    if (
        evaluation_replay.digest_sha256
        != starting_replay.digest_sha256
        or evaluation_replay.digest_sha256
        != plan.replay_digest_sha256
    ):
        raise TsdfError(
            "session inputs changed while evaluating a TSDF voxel traversal"
        )
    _validate_contribution_transcript(
        contributions,
        selected_sequences,
        address,
        plan.artifact_digest_sha256,
        plan.replay_digest_sha256,
    )
    _require_unchanged_empty_target(
        storage,
        address,
        layout_before,
        stored_sum_bytes,
        stored_weight_bytes,
    )

    accepted = tuple(
        contribution
        for contribution in contributions
        if contribution.contributes
    )
    if len(accepted) > MAX_TSDF_VOXEL_WEIGHT:
        raise TsdfError(
            "TSDF voxel traversal accepted contribution count exceeds "
            "uint32 capacity"
        )

    update_receipts: list[TsdfVoxelUpdateReceipt] = []
    try:
        for contribution in accepted:
            update_receipts.append(
                apply_tsdf_voxel_contribution(
                    storage,
                    contribution,
                    session,
                )
            )
        ending_replay = replay_session(session)
        if (
            ending_replay.digest_sha256
            != starting_replay.digest_sha256
            or ending_replay.digest_sha256
            != plan.replay_digest_sha256
        ):
            raise TsdfError(
                "session inputs changed while applying a TSDF voxel "
                "traversal"
            )
        _require_expected_final_target(
            storage,
            address,
            layout_before,
            update_receipts,
            stored_sum_bytes,
            stored_weight_bytes,
        )
        receipt = TsdfVoxelTraversalReceipt(
            address=address,
            source_plan_digest_sha256=plan.artifact_digest_sha256,
            replay_digest_sha256=plan.replay_digest_sha256,
            frame_stride=plan.frame_stride,
            total_observations=plan.total_observations,
            selected_observation_sequences=selected_sequences,
            contributions=contributions,
            update_receipts=tuple(update_receipts),
            tsdf_sum_before=tsdf_sum_before,
            weight_before=weight_before,
            tsdf_sum_after=float(storage.tsdf_sums[array_index]),
            weight_after=int(storage.weights[array_index]),
        )
    except Exception as error:
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
            f"cannot apply TSDF voxel traversal: {error}"
        ) from error
    return receipt


def traverse_tsdf_voxel_observations_from_context(
    storage: TsdfBlockStorage,
    address: TsdfVoxelAddress,
    context: TsdfReplayDepthContext,
) -> TsdfVoxelTraversalReceipt:
    """Evaluate and apply one context's selected observations to one slot."""

    if not isinstance(storage, TsdfBlockStorage):
        raise TsdfError(
            "TSDF context voxel traversal requires allocated "
            "TsdfBlockStorage"
        )
    if not isinstance(address, TsdfVoxelAddress):
        raise TsdfError(
            "TSDF context voxel traversal requires a TsdfVoxelAddress"
        )
    if not isinstance(context, TsdfReplayDepthContext):
        raise TsdfError(
            "TSDF context voxel traversal requires a prepared "
            "TsdfReplayDepthContext"
        )

    _validate_update_storage(storage)
    resolved_address = locate_tsdf_voxel(
        storage,
        address.global_index_xyz,
    )
    if resolved_address != address:
        raise TsdfError(
            "TSDF context voxel traversal address does not match "
            "destination storage"
        )

    plan = storage.source_plan
    _validate_contribution_plan(plan)
    _validate_contribution_context(plan, context)
    _validate_traversal_plan(plan)
    selected_sequences = tuple(
        range(0, plan.total_observations, plan.frame_stride)
    )
    if (
        context.selected_observation_sequences != selected_sequences
        or len(context.observations) != len(selected_sequences)
    ):
        raise TsdfError(
            "TSDF replay/depth context observations are not the complete "
            "canonical traversal selection"
        )

    array_index = address.array_index_bzyx
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
    if weight_before != 0:
        raise TsdfError(
            "TSDF context voxel traversal requires a canonical empty "
            "target slot"
        )
    layout_before = _storage_layout_identity(storage)
    stored_sum_bytes = stored_sum.tobytes()
    stored_weight_bytes = stored_weight.tobytes()

    contributions = tuple(
        evaluate_tsdf_voxel_contribution_from_context(
            storage,
            address,
            context,
            sequence,
        )
        for sequence in selected_sequences
    )
    _validate_contribution_transcript(
        contributions,
        selected_sequences,
        address,
        plan.artifact_digest_sha256,
        plan.replay_digest_sha256,
    )
    _require_unchanged_empty_target(
        storage,
        address,
        layout_before,
        stored_sum_bytes,
        stored_weight_bytes,
    )

    accepted = tuple(
        contribution
        for contribution in contributions
        if contribution.contributes
    )
    if len(accepted) > MAX_TSDF_VOXEL_WEIGHT:
        raise TsdfError(
            "TSDF context voxel traversal accepted contribution count "
            "exceeds uint32 capacity"
        )

    update_receipts: list[TsdfVoxelUpdateReceipt] = []
    try:
        for contribution in accepted:
            update_receipts.append(
                apply_tsdf_voxel_contribution_from_context(
                    storage,
                    contribution,
                    context,
                )
            )
        _require_expected_final_target(
            storage,
            address,
            layout_before,
            update_receipts,
            stored_sum_bytes,
            stored_weight_bytes,
        )
        receipt = TsdfVoxelTraversalReceipt(
            address=address,
            source_plan_digest_sha256=plan.artifact_digest_sha256,
            replay_digest_sha256=plan.replay_digest_sha256,
            frame_stride=plan.frame_stride,
            total_observations=plan.total_observations,
            selected_observation_sequences=selected_sequences,
            contributions=contributions,
            update_receipts=tuple(update_receipts),
            tsdf_sum_before=tsdf_sum_before,
            weight_before=weight_before,
            tsdf_sum_after=float(storage.tsdf_sums[array_index]),
            weight_after=int(storage.weights[array_index]),
        )
    except Exception as error:
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
            f"cannot apply context-bound TSDF voxel traversal: {error}"
        ) from error
    return receipt


def _validate_traversal_plan(plan: object) -> None:
    for value, label in (
        (getattr(plan, "frame_stride", None), "frame_stride"),
        (getattr(plan, "total_observations", None), "total_observations"),
        (
            getattr(plan, "selected_observations", None),
            "selected_observations",
        ),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise TsdfError(
                f"TSDF block plan {label} must be a positive integer"
            )


def _validate_contribution_transcript(
    contributions: tuple[TsdfVoxelContribution, ...],
    selected_sequences: tuple[int, ...],
    address: TsdfVoxelAddress,
    source_plan_digest_sha256: str,
    replay_digest_sha256: str,
) -> None:
    if len(contributions) != len(selected_sequences):
        raise TsdfError(
            "TSDF voxel traversal contribution count is inconsistent"
        )
    for sequence, contribution in zip(
        selected_sequences,
        contributions,
        strict=True,
    ):
        if (
            not isinstance(contribution, TsdfVoxelContribution)
            or contribution.observation_sequence != sequence
            or contribution.address != address
            or contribution.source_plan_digest_sha256
            != source_plan_digest_sha256
            or contribution.replay_digest_sha256
            != replay_digest_sha256
        ):
            raise TsdfError(
                "TSDF voxel traversal contribution transcript is "
                "inconsistent"
            )


def _require_unchanged_empty_target(
    storage: TsdfBlockStorage,
    address: TsdfVoxelAddress,
    layout_before: tuple[object, ...],
    stored_sum_bytes: bytes,
    stored_weight_bytes: bytes,
) -> None:
    _validate_update_storage(storage)
    if (
        locate_tsdf_voxel(storage, address.global_index_xyz) != address
        or _storage_layout_identity(storage) != layout_before
        or storage.tsdf_sums[address.array_index_bzyx].tobytes()
        != stored_sum_bytes
        or storage.weights[address.array_index_bzyx].tobytes()
        != stored_weight_bytes
    ):
        raise TsdfError(
            "TSDF voxel traversal target changed during read-only evaluation"
        )


def _require_expected_final_target(
    storage: TsdfBlockStorage,
    address: TsdfVoxelAddress,
    layout_before: tuple[object, ...],
    update_receipts: list[TsdfVoxelUpdateReceipt],
    stored_sum_bytes: bytes,
    stored_weight_bytes: bytes,
) -> None:
    _validate_update_storage(storage)
    if update_receipts:
        expected_sum_bytes = np.float64(
            update_receipts[-1].tsdf_sum_after
        ).tobytes()
        expected_weight_bytes = np.uint32(
            update_receipts[-1].weight_after
        ).tobytes()
    else:
        expected_sum_bytes = stored_sum_bytes
        expected_weight_bytes = stored_weight_bytes
    if (
        locate_tsdf_voxel(storage, address.global_index_xyz) != address
        or _storage_layout_identity(storage) != layout_before
        or storage.tsdf_sums[address.array_index_bzyx].tobytes()
        != expected_sum_bytes
        or storage.weights[address.array_index_bzyx].tobytes()
        != expected_weight_bytes
    ):
        raise TsdfError(
            "TSDF voxel traversal final storage state changed before "
            "receipt construction"
        )
