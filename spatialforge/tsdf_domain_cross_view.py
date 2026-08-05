"""Cross-view verdicts across a surveyed conservative coverage domain."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .errors import TsdfError
from .tsdf_block_cross_view import (
    TsdfBlockCrossViewReceipt,
    classify_tsdf_block_voxels_across_observations_from_context,
)
from .tsdf_block_plan import (
    MAX_PLANNED_BLOCKS,
    TSDF_BLOCK_RESOLUTION,
    _ordered_blocks,
)
from .tsdf_block_plan_loader import TsdfBlockPlan
from .tsdf_block_storage import TSDF_BLOCK_VOXELS
from .tsdf_observation_block_rays import (
    _is_sha256,
    _validate_block_index_xyz,
    _validate_trace_plan,
)
from .tsdf_plan_footprint_survey import TsdfPlanFootprintSurveyReceipt
from .tsdf_replay_depth_context import TsdfReplayDepthContext
from .tsdf_voxel_contribution import _validate_contribution_context
from .tsdf_voxel_cross_view import TsdfVoxelCrossViewVerdict

_Index3 = tuple[int, int, int]

MAX_TSDF_COVERAGE_DOMAIN_CROSS_VIEW_OUTCOMES = 262_144


@dataclass(frozen=True, slots=True)
class TsdfCoverageDomainCrossViewReceipt:
    """Immutable cross-view transcript for a surveyed coverage domain."""

    source_plan_digest_sha256: str
    replay_digest_sha256: str
    frame_stride: int
    total_observations: int
    selected_observation_sequences: tuple[int, ...]
    domain_block_indices: tuple[_Index3, ...]
    existing_plan_block_indices: tuple[_Index3, ...]
    unplanned_block_indices: tuple[_Index3, ...]
    block_resolution: int
    block_receipts: tuple[TsdfBlockCrossViewReceipt, ...]

    def __post_init__(self) -> None:
        if not _is_sha256(self.source_plan_digest_sha256):
            raise TsdfError(
                "TSDF domain cross-view source plan digest is invalid"
            )
        if not _is_sha256(self.replay_digest_sha256):
            raise TsdfError("TSDF domain cross-view replay digest is invalid")
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
                    f"TSDF domain cross-view {label} must be positive"
                )
        expected_sequences = tuple(
            range(0, self.total_observations, self.frame_stride)
        )
        if (
            not isinstance(self.selected_observation_sequences, tuple)
            or self.selected_observation_sequences != expected_sequences
        ):
            raise TsdfError(
                "TSDF domain cross-view observations are not the complete "
                "canonical stride selection"
            )
        if (
            isinstance(self.block_resolution, bool)
            or not isinstance(self.block_resolution, int)
            or self.block_resolution != TSDF_BLOCK_RESOLUTION
        ):
            raise TsdfError(
                "TSDF domain cross-view requires block resolution "
                f"{TSDF_BLOCK_RESOLUTION}"
            )
        _validate_canonical_domain_blocks(
            self.domain_block_indices,
            "domain blocks",
        )
        if (
            not self.domain_block_indices
            or len(self.domain_block_indices) > MAX_PLANNED_BLOCKS
        ):
            raise TsdfError(
                "TSDF domain cross-view requires at least one domain block"
            )
        _validate_canonical_domain_blocks(
            self.existing_plan_block_indices,
            "existing-plan blocks",
        )
        _validate_canonical_domain_blocks(
            self.unplanned_block_indices,
            "unplanned blocks",
        )
        domain = set(self.domain_block_indices)
        existing = set(self.existing_plan_block_indices)
        unplanned = set(self.unplanned_block_indices)
        if existing & unplanned or existing | unplanned != domain:
            raise TsdfError(
                "TSDF domain cross-view plan partition is inconsistent"
            )
        if (
            not isinstance(self.block_receipts, tuple)
            or len(self.block_receipts) != len(self.domain_block_indices)
        ):
            raise TsdfError(
                "TSDF domain cross-view requires exactly one block receipt "
                "for every domain block"
            )

        for position, receipt in enumerate(self.block_receipts):
            if not isinstance(receipt, TsdfBlockCrossViewReceipt):
                raise TsdfError(
                    "TSDF domain cross-view contains an invalid block receipt"
                )
            block_index = self.domain_block_indices[position]
            if receipt.block_index_xyz != block_index:
                raise TsdfError(
                    "TSDF domain cross-view block receipts must follow the "
                    "canonical domain order"
                )
            if receipt.planned_block != (block_index in existing):
                raise TsdfError(
                    "TSDF domain cross-view block receipt plan membership is "
                    "inconsistent"
                )
            if (
                receipt.source_plan_digest_sha256
                != self.source_plan_digest_sha256
                or receipt.replay_digest_sha256 != self.replay_digest_sha256
                or receipt.frame_stride != self.frame_stride
                or receipt.total_observations != self.total_observations
                or receipt.selected_observation_sequences
                != self.selected_observation_sequences
                or receipt.block_resolution != self.block_resolution
            ):
                raise TsdfError(
                    "TSDF domain cross-view block receipt provenance is "
                    "inconsistent"
                )
        if (
            self.retained_outcome_count
            > MAX_TSDF_COVERAGE_DOMAIN_CROSS_VIEW_OUTCOMES
        ):
            raise TsdfError(
                "TSDF domain cross-view exceeds the retained outcome limit"
            )

    @property
    def block_count(self) -> int:
        return len(self.block_receipts)

    @property
    def voxel_count(self) -> int:
        return sum(receipt.voxel_count for receipt in self.block_receipts)

    @property
    def observation_count(self) -> int:
        return len(self.selected_observation_sequences)

    @property
    def retained_outcome_count(self) -> int:
        return self.voxel_count * self.observation_count

    def _count(self, verdict: TsdfVoxelCrossViewVerdict) -> int:
        return sum(
            sum(voxel.verdict is verdict for voxel in receipt.voxel_receipts)
            for receipt in self.block_receipts
        )

    @property
    def surface_voxel_count(self) -> int:
        return self._count(TsdfVoxelCrossViewVerdict.SURFACE)

    @property
    def free_space_voxel_count(self) -> int:
        return self._count(TsdfVoxelCrossViewVerdict.FREE_SPACE)

    @property
    def occluded_voxel_count(self) -> int:
        return self._count(TsdfVoxelCrossViewVerdict.OCCLUDED)

    @property
    def unseen_voxel_count(self) -> int:
        return self._count(TsdfVoxelCrossViewVerdict.UNSEEN)

    @property
    def observed_voxel_count(self) -> int:
        return self.surface_voxel_count + self.free_space_voxel_count

    @property
    def carvable_free_space_voxel_indices(self) -> tuple[_Index3, ...]:
        return tuple(
            voxel_index
            for receipt in self.block_receipts
            for voxel_index in receipt.carvable_free_space_voxel_indices
        )

    @property
    def carvable_free_space_voxel_count(self) -> int:
        return sum(
            receipt.carvable_free_space_voxel_count
            for receipt in self.block_receipts
        )

    @property
    def carvable_blocks(self) -> tuple[_Index3, ...]:
        return tuple(
            receipt.block_index_xyz
            for receipt in self.block_receipts
            if receipt.carves_free_space
        )

    @property
    def planned_carvable_voxel_count(self) -> int:
        return sum(
            receipt.carvable_free_space_voxel_count
            for receipt in self.block_receipts
            if receipt.planned_block
        )

    @property
    def unplanned_carvable_voxel_count(self) -> int:
        return (
            self.carvable_free_space_voxel_count
            - self.planned_carvable_voxel_count
        )

    @property
    def unplanned_carvable_blocks(self) -> tuple[_Index3, ...]:
        return tuple(
            receipt.block_index_xyz
            for receipt in self.block_receipts
            if receipt.carves_free_space and not receipt.planned_block
        )

    @property
    def reference_weight_total(self) -> int:
        return sum(
            receipt.reference_weight_total
            for receipt in self.block_receipts
        )

    @property
    def reference_tsdf_sum_total(self) -> float:
        total = 0.0
        for receipt in self.block_receipts:
            total += receipt.reference_tsdf_sum_total
        if not math.isfinite(total):
            raise TsdfError(
                "TSDF domain cross-view sum total must remain finite"
            )
        return total

    @property
    def planned_reference_weight_total(self) -> int:
        return sum(
            receipt.reference_weight_total
            for receipt in self.block_receipts
            if receipt.planned_block
        )

    @property
    def planned_observed_voxel_count(self) -> int:
        return sum(
            receipt.observed_voxel_count
            for receipt in self.block_receipts
            if receipt.planned_block
        )

    @property
    def maximum_voxel_weight(self) -> int:
        return max(
            (receipt.maximum_voxel_weight for receipt in self.block_receipts),
            default=0,
        )

    @property
    def verdict_counts(
        self,
    ) -> tuple[tuple[TsdfVoxelCrossViewVerdict, int], ...]:
        return tuple(
            (verdict, count)
            for verdict in TsdfVoxelCrossViewVerdict
            if (count := self._count(verdict))
        )


def sweep_tsdf_coverage_domain_cross_view_from_context(
    plan: TsdfBlockPlan,
    context: TsdfReplayDepthContext,
    coverage: TsdfPlanFootprintSurveyReceipt,
) -> TsdfCoverageDomainCrossViewReceipt:
    """Resolve every voxel of a surveyed coverage domain, read-only."""

    if not isinstance(plan, TsdfBlockPlan):
        raise TsdfError(
            "TSDF domain cross-view requires a loaded TsdfBlockPlan"
        )
    if not isinstance(context, TsdfReplayDepthContext):
        raise TsdfError(
            "TSDF domain cross-view requires a prepared "
            "TsdfReplayDepthContext"
        )
    if not isinstance(coverage, TsdfPlanFootprintSurveyReceipt):
        raise TsdfError(
            "TSDF domain cross-view requires a surveyed "
            "TsdfPlanFootprintSurveyReceipt"
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
                "canonical domain cross-view selection"
            )
        if (
            coverage.source_plan_digest_sha256 != plan.artifact_digest_sha256
            or coverage.replay_digest_sha256 != plan.replay_digest_sha256
        ):
            raise TsdfError(
                "TSDF coverage survey provenance does not match the block "
                "plan"
            )
        if (
            coverage.selected_observation_sequences != selected_sequences
            or coverage.source_plan_block_indices != plan.active_blocks
        ):
            raise TsdfError(
                "TSDF coverage survey selection does not match the block plan"
            )

        domain_blocks = coverage.covered_block_indices
        if not domain_blocks:
            raise TsdfError(
                "TSDF domain cross-view requires a nonempty surveyed coverage "
                "domain"
            )
        outcome_count = (
            len(domain_blocks) * TSDF_BLOCK_VOXELS * len(selected_sequences)
        )
        if outcome_count > MAX_TSDF_COVERAGE_DOMAIN_CROSS_VIEW_OUTCOMES:
            raise TsdfError(
                "TSDF domain cross-view requires "
                f"{outcome_count} retained outcomes; reference maximum is "
                f"{MAX_TSDF_COVERAGE_DOMAIN_CROSS_VIEW_OUTCOMES}. Use a "
                "larger frame stride, a coarser voxel size, or a future "
                "scalable fusion path"
            )

        block_receipts: list[TsdfBlockCrossViewReceipt] = []
        for position, block_index in enumerate(domain_blocks):
            receipt = (
                classify_tsdf_block_voxels_across_observations_from_context(
                    plan,
                    context,
                    block_index,
                )
            )
            if (
                not isinstance(receipt, TsdfBlockCrossViewReceipt)
                or receipt.block_index_xyz != block_index
                or len(block_receipts) != position
            ):
                raise TsdfError(
                    "TSDF domain cross-view child block scope is inconsistent"
                )
            block_receipts.append(receipt)

        return TsdfCoverageDomainCrossViewReceipt(
            source_plan_digest_sha256=plan.artifact_digest_sha256,
            replay_digest_sha256=plan.replay_digest_sha256,
            frame_stride=plan.frame_stride,
            total_observations=plan.total_observations,
            selected_observation_sequences=selected_sequences,
            domain_block_indices=domain_blocks,
            existing_plan_block_indices=coverage.existing_plan_block_indices,
            unplanned_block_indices=coverage.unplanned_block_indices,
            block_resolution=plan.block_resolution,
            block_receipts=tuple(block_receipts),
        )
    except TsdfError:
        raise
    except Exception as error:
        raise TsdfError(
            f"cannot sweep prepared TSDF coverage domain: {error}"
        ) from error


def _validate_canonical_domain_blocks(value: object, label: str) -> None:
    if not isinstance(value, tuple):
        raise TsdfError(f"TSDF domain cross-view {label} must be a tuple")
    previous_key: tuple[int, int, int] | None = None
    for block_index in value:
        _validate_block_index_xyz(block_index, label)
        key = (block_index[2], block_index[1], block_index[0])
        if previous_key is not None and key <= previous_key:
            raise TsdfError(
                f"TSDF domain cross-view {label} must be unique and strictly "
                "x-fastest ordered"
            )
        previous_key = key
    if value and _ordered_blocks(set(value)) != value:
        raise TsdfError(
            f"TSDF domain cross-view {label} must be canonically ordered"
        )
