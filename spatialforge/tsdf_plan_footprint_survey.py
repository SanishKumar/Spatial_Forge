"""Conservative footprint coverage across every plan-selected observation."""

from __future__ import annotations

from dataclasses import dataclass

from .errors import TsdfError
from .tsdf_block_plan import (
    MAX_PLANNED_BLOCKS,
    TSDF_BLOCK_RESOLUTION,
    _ordered_blocks,
)
from .tsdf_block_plan_loader import TsdfBlockPlan
from .tsdf_observation_block_rays import (
    _is_finite_number,
    _is_sha256,
    _validate_block_index_xyz,
    _validate_image_size,
    _validate_trace_plan,
)
from .tsdf_observation_footprint import (
    TsdfObservationFootprintReceipt,
    survey_tsdf_observation_pixel_footprints_from_context,
)
from .tsdf_replay_depth_context import (
    TsdfReplayDepthContext,
    TsdfReplayDepthStatus,
)
from .tsdf_voxel_contribution import _validate_contribution_context

_BlockIndex = tuple[int, int, int]

MAX_TSDF_PLAN_FOOTPRINT_CANDIDATE_BLOCKS = 262_144


@dataclass(frozen=True, slots=True)
class TsdfPlanFootprintSurveyReceipt:
    """Immutable footprint transcript for every selected observation."""

    source_plan_digest_sha256: str
    replay_digest_sha256: str
    frame_stride: int
    total_observations: int
    selected_observation_sequences: tuple[int, ...]
    source_plan_block_indices: tuple[_BlockIndex, ...]
    block_resolution: int
    block_extent_m: float
    image_size: tuple[int, int]
    observation_receipts: tuple[TsdfObservationFootprintReceipt, ...]
    covered_block_indices: tuple[_BlockIndex, ...]
    existing_plan_block_indices: tuple[_BlockIndex, ...]
    unplanned_block_indices: tuple[_BlockIndex, ...]

    def __post_init__(self) -> None:
        if not _is_sha256(self.source_plan_digest_sha256):
            raise TsdfError(
                "TSDF plan footprint survey source plan digest is invalid"
            )
        if not _is_sha256(self.replay_digest_sha256):
            raise TsdfError(
                "TSDF plan footprint survey replay digest is invalid"
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
                    f"TSDF plan footprint survey {label} must be positive"
                )
        expected_sequences = tuple(
            range(0, self.total_observations, self.frame_stride)
        )
        if (
            not isinstance(self.selected_observation_sequences, tuple)
            or self.selected_observation_sequences != expected_sequences
        ):
            raise TsdfError(
                "TSDF plan footprint survey observations are not the "
                "complete canonical stride selection"
            )
        _validate_canonical_survey_blocks(
            self.source_plan_block_indices,
            "source plan blocks",
        )
        if (
            not self.source_plan_block_indices
            or len(self.source_plan_block_indices) > MAX_PLANNED_BLOCKS
        ):
            raise TsdfError(
                "TSDF plan footprint survey source plan blocks are invalid"
            )
        if (
            isinstance(self.block_resolution, bool)
            or not isinstance(self.block_resolution, int)
            or self.block_resolution != TSDF_BLOCK_RESOLUTION
        ):
            raise TsdfError(
                "TSDF plan footprint survey requires block resolution "
                f"{TSDF_BLOCK_RESOLUTION}"
            )
        if (
            not _is_finite_number(self.block_extent_m)
            or self.block_extent_m <= 0.0
        ):
            raise TsdfError(
                "TSDF plan footprint survey block extent must be finite and "
                "positive"
            )
        _validate_image_size(self.image_size)
        if (
            not isinstance(self.observation_receipts, tuple)
            or len(self.observation_receipts) != len(expected_sequences)
        ):
            raise TsdfError(
                "TSDF plan footprint survey requires exactly one observation "
                "receipt for every selected observation"
            )

        for position, receipt in enumerate(self.observation_receipts):
            if not isinstance(receipt, TsdfObservationFootprintReceipt):
                raise TsdfError(
                    "TSDF plan footprint survey contains an invalid "
                    "observation receipt"
                )
            if receipt.observation_sequence != expected_sequences[position]:
                raise TsdfError(
                    "TSDF plan footprint survey observation receipts must "
                    "follow the canonical selected-sequence order"
                )
            if (
                receipt.source_plan_digest_sha256
                != self.source_plan_digest_sha256
                or receipt.replay_digest_sha256 != self.replay_digest_sha256
                or receipt.frame_stride != self.frame_stride
                or receipt.total_observations != self.total_observations
                or receipt.source_plan_block_indices
                != self.source_plan_block_indices
                or receipt.block_resolution != self.block_resolution
                or receipt.block_extent_m != self.block_extent_m
                or receipt.image_size != self.image_size
            ):
                raise TsdfError(
                    "TSDF plan footprint survey observation receipt "
                    "provenance is inconsistent"
                )

        expected_covered = _ordered_blocks(
            {
                block_index
                for receipt in self.observation_receipts
                for block_index in receipt.covered_block_indices
            }
        )
        if self.covered_block_indices != expected_covered:
            raise TsdfError(
                "TSDF plan footprint survey covered blocks do not match its "
                "observation receipts"
            )
        _validate_canonical_survey_blocks(
            self.existing_plan_block_indices,
            "existing-plan blocks",
        )
        _validate_canonical_survey_blocks(
            self.unplanned_block_indices,
            "unplanned blocks",
        )
        covered = set(self.covered_block_indices)
        existing = set(self.existing_plan_block_indices)
        unplanned = set(self.unplanned_block_indices)
        if existing & unplanned or existing | unplanned != covered:
            raise TsdfError(
                "TSDF plan footprint survey plan partition is inconsistent"
            )
        source_plan = set(self.source_plan_block_indices)
        if (
            self.existing_plan_block_indices
            != _ordered_blocks(covered & source_plan)
            or self.unplanned_block_indices
            != _ordered_blocks(covered - source_plan)
        ):
            raise TsdfError(
                "TSDF plan footprint survey plan membership is inconsistent"
            )
        if not set(self.centerline_block_indices) <= covered:
            raise TsdfError(
                "TSDF plan footprint survey coverage does not contain its "
                "own centreline paths"
            )
        if len(self.covered_block_indices) > MAX_PLANNED_BLOCKS:
            raise TsdfError(
                "TSDF plan footprint survey exceeds the unique-block limit"
            )
        if (
            self.candidate_block_count
            > MAX_TSDF_PLAN_FOOTPRINT_CANDIDATE_BLOCKS
        ):
            raise TsdfError(
                "TSDF plan footprint survey exceeds the candidate block limit"
            )

    @property
    def observation_count(self) -> int:
        return len(self.observation_receipts)

    @property
    def surveyed_observation_count(self) -> int:
        return sum(
            receipt.prepared_depth_accessed
            for receipt in self.observation_receipts
        )

    @property
    def pixel_count(self) -> int:
        return sum(
            receipt.pixel_count for receipt in self.observation_receipts
        )

    @property
    def covered_pixel_count(self) -> int:
        return sum(
            receipt.covered_pixel_count
            for receipt in self.observation_receipts
        )

    @property
    def depth_invalid_count(self) -> int:
        return sum(
            receipt.depth_invalid_count
            for receipt in self.observation_receipts
        )

    @property
    def candidate_block_count(self) -> int:
        return sum(
            receipt.candidate_block_count
            for receipt in self.observation_receipts
        )

    @property
    def rejected_candidate_count(self) -> int:
        return sum(
            receipt.rejected_candidate_count
            for receipt in self.observation_receipts
        )

    @property
    def block_visit_count(self) -> int:
        return sum(
            receipt.block_visit_count
            for receipt in self.observation_receipts
        )

    @property
    def duplicate_block_visit_count(self) -> int:
        return self.block_visit_count - len(self.covered_block_indices)

    @property
    def maximum_blocks_per_pixel(self) -> int:
        return max(
            (
                receipt.maximum_blocks_per_pixel
                for receipt in self.observation_receipts
            ),
            default=0,
        )

    @property
    def centerline_block_indices(self) -> tuple[_BlockIndex, ...]:
        return _ordered_blocks(
            {
                block_index
                for receipt in self.observation_receipts
                for block_index in receipt.centerline_block_indices
            }
        )

    @property
    def footprint_only_block_indices(self) -> tuple[_BlockIndex, ...]:
        return _ordered_blocks(
            set(self.covered_block_indices)
            - set(self.centerline_block_indices)
        )

    @property
    def widens_centerline_coverage(self) -> bool:
        return bool(self.footprint_only_block_indices)

    @property
    def covered_block_observation_counts(
        self,
    ) -> tuple[tuple[_BlockIndex, int], ...]:
        counts: dict[_BlockIndex, int] = {}
        for receipt in self.observation_receipts:
            for block_index in receipt.covered_block_indices:
                counts[block_index] = counts.get(block_index, 0) + 1
        return tuple(
            (block_index, counts[block_index])
            for block_index in self.covered_block_indices
        )

    @property
    def multi_observation_block_indices(self) -> tuple[_BlockIndex, ...]:
        return tuple(
            block_index
            for block_index, count in self.covered_block_observation_counts
            if count > 1
        )

    @property
    def maximum_block_observation_count(self) -> int:
        return max(
            (count for _, count in self.covered_block_observation_counts),
            default=0,
        )

    @property
    def prepared_depth_accessed(self) -> bool:
        return any(
            receipt.prepared_depth_accessed
            for receipt in self.observation_receipts
        )

    @property
    def observation_status_counts(
        self,
    ) -> tuple[tuple[TsdfReplayDepthStatus, int], ...]:
        return tuple(
            (status, count)
            for status in TsdfReplayDepthStatus
            if (
                count := sum(
                    receipt.observation_status is status
                    for receipt in self.observation_receipts
                )
            )
        )


def survey_tsdf_plan_pixel_footprints_from_context(
    plan: TsdfBlockPlan,
    context: TsdfReplayDepthContext,
) -> TsdfPlanFootprintSurveyReceipt:
    """Cover every selected observation's pixel wedges without mutation."""

    if not isinstance(plan, TsdfBlockPlan):
        raise TsdfError(
            "TSDF plan footprint survey requires a loaded TsdfBlockPlan"
        )
    if not isinstance(context, TsdfReplayDepthContext):
        raise TsdfError(
            "TSDF plan footprint survey requires a prepared "
            "TsdfReplayDepthContext"
        )

    try:
        _validate_trace_plan(plan)
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
                "canonical plan footprint survey selection"
            )

        observation_receipts: list[TsdfObservationFootprintReceipt] = []
        covered_blocks: set[_BlockIndex] = set()
        candidate_blocks = 0
        for position, observation_sequence in enumerate(selected_sequences):
            receipt = survey_tsdf_observation_pixel_footprints_from_context(
                plan,
                context,
                observation_sequence,
            )
            if (
                not isinstance(receipt, TsdfObservationFootprintReceipt)
                or receipt.observation_sequence != observation_sequence
                or len(observation_receipts) != position
            ):
                raise TsdfError(
                    "TSDF plan footprint survey child observation scope is "
                    "inconsistent"
                )
            candidate_blocks += receipt.candidate_block_count
            if candidate_blocks > MAX_TSDF_PLAN_FOOTPRINT_CANDIDATE_BLOCKS:
                raise TsdfError(
                    "TSDF plan footprint survey candidate blocks reach "
                    f"{candidate_blocks}; reference maximum is "
                    f"{MAX_TSDF_PLAN_FOOTPRINT_CANDIDATE_BLOCKS}. Use a "
                    "larger frame stride, a coarser voxel size, or a future "
                    "scalable coverage path"
                )
            covered_blocks.update(receipt.covered_block_indices)
            if len(covered_blocks) > MAX_PLANNED_BLOCKS:
                raise TsdfError(
                    "TSDF plan footprint survey exceeds the "
                    f"{MAX_PLANNED_BLOCKS}-block reference limit. Use a "
                    "coarser voxel size or a future scalable coverage path"
                )
            observation_receipts.append(receipt)

        active = set(plan.active_blocks)
        return TsdfPlanFootprintSurveyReceipt(
            source_plan_digest_sha256=plan.artifact_digest_sha256,
            replay_digest_sha256=plan.replay_digest_sha256,
            frame_stride=plan.frame_stride,
            total_observations=plan.total_observations,
            selected_observation_sequences=selected_sequences,
            source_plan_block_indices=plan.active_blocks,
            block_resolution=plan.block_resolution,
            block_extent_m=plan.block_extent_m,
            image_size=(context.camera.width, context.camera.height),
            observation_receipts=tuple(observation_receipts),
            covered_block_indices=_ordered_blocks(covered_blocks),
            existing_plan_block_indices=_ordered_blocks(
                covered_blocks & active
            ),
            unplanned_block_indices=_ordered_blocks(covered_blocks - active),
        )
    except TsdfError:
        raise
    except Exception as error:
        raise TsdfError(
            f"cannot survey prepared TSDF plan footprints: {error}"
        ) from error


def _validate_canonical_survey_blocks(value: object, label: str) -> None:
    if not isinstance(value, tuple):
        raise TsdfError(
            f"TSDF plan footprint survey {label} must be a tuple"
        )
    previous_key: tuple[int, int, int] | None = None
    for block_index in value:
        _validate_block_index_xyz(block_index, label)
        key = (block_index[2], block_index[1], block_index[0])
        if previous_key is not None and key <= previous_key:
            raise TsdfError(
                f"TSDF plan footprint survey {label} must be unique and "
                "strictly x-fastest ordered"
            )
        previous_key = key
