"""Cross-view verdicts for every voxel of one TSDF block."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .errors import TsdfError
from .tsdf_block_plan import TSDF_BLOCK_RESOLUTION
from .tsdf_block_plan_loader import TsdfBlockPlan
from .tsdf_block_storage import TSDF_BLOCK_VOXELS
from .tsdf_observation_block_rays import (
    _is_finite_number,
    _is_sha256,
    _validate_block_index_xyz,
    _validate_image_size,
    _validate_trace_plan,
)
from .tsdf_replay_depth_context import TsdfReplayDepthContext
from .tsdf_voxel_address import compose_tsdf_global_voxel_index
from .tsdf_voxel_contribution import _validate_contribution_context
from .tsdf_voxel_cross_view import (
    TsdfVoxelCrossViewReceipt,
    TsdfVoxelCrossViewVerdict,
    classify_tsdf_voxel_across_observations_from_context,
)

_Index3 = tuple[int, int, int]

MAX_TSDF_BLOCK_CROSS_VIEW_OUTCOMES = 262_144


def _local_index_for_flat(local_flat_index: int) -> _Index3:
    """Decompose a canonical x-fastest local flat index."""

    local_x = local_flat_index % TSDF_BLOCK_RESOLUTION
    local_y = (
        local_flat_index // TSDF_BLOCK_RESOLUTION
    ) % TSDF_BLOCK_RESOLUTION
    local_z = local_flat_index // (TSDF_BLOCK_RESOLUTION**2)
    return (local_x, local_y, local_z)


@dataclass(frozen=True, slots=True)
class TsdfBlockCrossViewReceipt:
    """Immutable cross-view transcript for one block's whole voxel row."""

    source_plan_digest_sha256: str
    replay_digest_sha256: str
    frame_stride: int
    total_observations: int
    selected_observation_sequences: tuple[int, ...]
    block_index_xyz: _Index3
    planned_block: bool
    block_resolution: int
    voxel_size_m: float
    truncation_m: float
    image_size: tuple[int, int]
    voxel_receipts: tuple[TsdfVoxelCrossViewReceipt, ...]

    def __post_init__(self) -> None:
        if not _is_sha256(self.source_plan_digest_sha256):
            raise TsdfError(
                "TSDF block cross-view source plan digest is invalid"
            )
        if not _is_sha256(self.replay_digest_sha256):
            raise TsdfError("TSDF block cross-view replay digest is invalid")
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
                    f"TSDF block cross-view {label} must be positive"
                )
        expected_sequences = tuple(
            range(0, self.total_observations, self.frame_stride)
        )
        if (
            not isinstance(self.selected_observation_sequences, tuple)
            or self.selected_observation_sequences != expected_sequences
        ):
            raise TsdfError(
                "TSDF block cross-view observations are not the complete "
                "canonical stride selection"
            )
        _validate_block_index_xyz(self.block_index_xyz, "block index")
        if not isinstance(self.planned_block, bool):
            raise TsdfError(
                "TSDF block cross-view planned_block must be a bool"
            )
        if (
            isinstance(self.block_resolution, bool)
            or not isinstance(self.block_resolution, int)
            or self.block_resolution != TSDF_BLOCK_RESOLUTION
        ):
            raise TsdfError(
                "TSDF block cross-view requires block resolution "
                f"{TSDF_BLOCK_RESOLUTION}"
            )
        for value, label in (
            (self.voxel_size_m, "voxel size"),
            (self.truncation_m, "truncation"),
        ):
            if not _is_finite_number(value) or value <= 0.0:
                raise TsdfError(
                    f"TSDF block cross-view {label} must be finite and "
                    "positive"
                )
        _validate_image_size(self.image_size)
        if (
            not isinstance(self.voxel_receipts, tuple)
            or len(self.voxel_receipts) != TSDF_BLOCK_VOXELS
        ):
            raise TsdfError(
                "TSDF block cross-view requires exactly "
                f"{TSDF_BLOCK_VOXELS} canonical voxel receipts"
            )

        for local_flat_index, receipt in enumerate(self.voxel_receipts):
            if not isinstance(receipt, TsdfVoxelCrossViewReceipt):
                raise TsdfError(
                    "TSDF block cross-view contains an invalid voxel receipt"
                )
            expected_local = _local_index_for_flat(local_flat_index)
            expected_global = compose_tsdf_global_voxel_index(
                self.block_index_xyz,
                expected_local,
            )
            if (
                receipt.local_index_xyz != expected_local
                or receipt.global_index_xyz != expected_global
                or receipt.block_index_xyz != self.block_index_xyz
            ):
                raise TsdfError(
                    "TSDF block cross-view voxel receipts must be in "
                    "canonical x-fastest local-flat order"
                )
            if (
                receipt.source_plan_digest_sha256
                != self.source_plan_digest_sha256
                or receipt.replay_digest_sha256 != self.replay_digest_sha256
                or receipt.frame_stride != self.frame_stride
                or receipt.total_observations != self.total_observations
                or receipt.selected_observation_sequences
                != self.selected_observation_sequences
                or receipt.planned_block != self.planned_block
                or receipt.block_resolution != self.block_resolution
                or receipt.voxel_size_m != self.voxel_size_m
                or receipt.truncation_m != self.truncation_m
                or receipt.image_size != self.image_size
            ):
                raise TsdfError(
                    "TSDF block cross-view voxel receipt scope is "
                    "inconsistent"
                )
        if self.retained_outcome_count > MAX_TSDF_BLOCK_CROSS_VIEW_OUTCOMES:
            raise TsdfError(
                "TSDF block cross-view exceeds the retained outcome limit"
            )

    @property
    def voxel_count(self) -> int:
        return len(self.voxel_receipts)

    @property
    def observation_count(self) -> int:
        return len(self.selected_observation_sequences)

    @property
    def retained_outcome_count(self) -> int:
        return self.voxel_count * self.observation_count

    def _count(self, verdict: TsdfVoxelCrossViewVerdict) -> int:
        return sum(
            receipt.verdict is verdict for receipt in self.voxel_receipts
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
        """Report the block's pure free-space voxels in canonical order."""

        return tuple(
            receipt.global_index_xyz
            for receipt in self.voxel_receipts
            if receipt.carvable_free_space
        )

    @property
    def carvable_free_space_voxel_count(self) -> int:
        return len(self.carvable_free_space_voxel_indices)

    @property
    def carves_free_space(self) -> bool:
        return bool(self.carvable_free_space_voxel_count)

    @property
    def reference_weight_total(self) -> int:
        return sum(
            receipt.reference_weight for receipt in self.voxel_receipts
        )

    @property
    def reference_tsdf_sum_total(self) -> float:
        total = 0.0
        for receipt in self.voxel_receipts:
            total += receipt.reference_tsdf_sum
        if not math.isfinite(total):
            raise TsdfError(
                "TSDF block cross-view sum total must remain finite"
            )
        return total

    @property
    def maximum_voxel_weight(self) -> int:
        return max(
            (receipt.reference_weight for receipt in self.voxel_receipts),
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

    @property
    def prepared_depth_accessed(self) -> bool:
        return any(
            child.prepared_depth_accessed
            for receipt in self.voxel_receipts
            for child in receipt.observation_receipts
        )


def classify_tsdf_block_voxels_across_observations_from_context(
    plan: TsdfBlockPlan,
    context: TsdfReplayDepthContext,
    block_index_xyz: _Index3,
) -> TsdfBlockCrossViewReceipt:
    """Resolve every voxel of one block against all selected observations."""

    if not isinstance(plan, TsdfBlockPlan):
        raise TsdfError(
            "TSDF block cross-view requires a loaded TsdfBlockPlan"
        )
    if not isinstance(context, TsdfReplayDepthContext):
        raise TsdfError(
            "TSDF block cross-view requires a prepared "
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
                "canonical block cross-view selection"
            )
        outcome_count = TSDF_BLOCK_VOXELS * len(selected_sequences)
        if outcome_count > MAX_TSDF_BLOCK_CROSS_VIEW_OUTCOMES:
            raise TsdfError(
                "TSDF block cross-view requires "
                f"{outcome_count} retained outcomes; reference maximum is "
                f"{MAX_TSDF_BLOCK_CROSS_VIEW_OUTCOMES}. Use a larger frame "
                "stride or a future scalable fusion path"
            )
        _validate_block_index_xyz(block_index_xyz, "block index")
        block_index = (
            block_index_xyz[0],
            block_index_xyz[1],
            block_index_xyz[2],
        )

        voxel_receipts: list[TsdfVoxelCrossViewReceipt] = []
        for local_flat_index in range(TSDF_BLOCK_VOXELS):
            local_index = _local_index_for_flat(local_flat_index)
            global_index = compose_tsdf_global_voxel_index(
                block_index,
                local_index,
            )
            receipt = classify_tsdf_voxel_across_observations_from_context(
                plan,
                context,
                global_index,
            )
            if (
                not isinstance(receipt, TsdfVoxelCrossViewReceipt)
                or receipt.global_index_xyz != global_index
                or receipt.block_index_xyz != block_index
                or len(voxel_receipts) != local_flat_index
            ):
                raise TsdfError(
                    "TSDF block cross-view child voxel scope is inconsistent"
                )
            voxel_receipts.append(receipt)

        return TsdfBlockCrossViewReceipt(
            source_plan_digest_sha256=plan.artifact_digest_sha256,
            replay_digest_sha256=plan.replay_digest_sha256,
            frame_stride=plan.frame_stride,
            total_observations=plan.total_observations,
            selected_observation_sequences=selected_sequences,
            block_index_xyz=block_index,
            planned_block=block_index in set(plan.active_blocks),
            block_resolution=plan.block_resolution,
            voxel_size_m=plan.voxel_size_m,
            truncation_m=plan.truncation_m,
            image_size=(context.camera.width, context.camera.height),
            voxel_receipts=tuple(voxel_receipts),
        )
    except TsdfError:
        raise
    except Exception as error:
        raise TsdfError(
            f"cannot classify prepared TSDF block across observations: "
            f"{error}"
        ) from error
