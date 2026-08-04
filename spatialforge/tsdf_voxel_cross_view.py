"""Cross-view sampling verdict for one voxel over selected observations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from .errors import TsdfError
from .tsdf_block_plan import TSDF_BLOCK_RESOLUTION
from .tsdf_block_plan_loader import TsdfBlockPlan
from .tsdf_observation_block_rays import (
    _is_finite_number,
    _is_sha256,
    _validate_image_size,
    _validate_point,
    _validate_trace_plan,
)
from .tsdf_replay_depth_context import TsdfReplayDepthContext
from .tsdf_voxel_address import _split_tsdf_global_voxel_index
from .tsdf_voxel_contribution import (
    _validate_contribution_context,
    _voxel_center_world_m,
)
from .tsdf_voxel_sampling import (
    TsdfVoxelSamplingReceipt,
    TsdfVoxelSamplingStatus,
    classify_tsdf_voxel_sampling_from_context,
)

_Index3 = tuple[int, int, int]
_Point3 = tuple[float, float, float]

MAX_TSDF_VOXEL_CROSS_VIEW_OBSERVATIONS = 262_144


class TsdfVoxelCrossViewVerdict(StrEnum):
    """Stable cross-view outcomes ordered by evidence specificity."""

    SURFACE = "surface"
    FREE_SPACE = "free-space"
    OCCLUDED = "occluded"
    UNSEEN = "unseen"


@dataclass(frozen=True, slots=True)
class TsdfVoxelCrossViewReceipt:
    """Immutable cross-view transcript for one voxel over all observations."""

    source_plan_digest_sha256: str
    replay_digest_sha256: str
    frame_stride: int
    total_observations: int
    selected_observation_sequences: tuple[int, ...]
    global_index_xyz: _Index3
    block_index_xyz: _Index3
    local_index_xyz: _Index3
    planned_block: bool
    voxel_size_m: float
    truncation_m: float
    block_resolution: int
    image_size: tuple[int, int]
    world_xyz_m: _Point3
    verdict: TsdfVoxelCrossViewVerdict
    observation_receipts: tuple[TsdfVoxelSamplingReceipt, ...]

    def __post_init__(self) -> None:
        if not _is_sha256(self.source_plan_digest_sha256):
            raise TsdfError(
                "TSDF voxel cross-view source plan digest is invalid"
            )
        if not _is_sha256(self.replay_digest_sha256):
            raise TsdfError("TSDF voxel cross-view replay digest is invalid")
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
                    f"TSDF voxel cross-view {label} must be positive"
                )
        expected_sequences = tuple(
            range(0, self.total_observations, self.frame_stride)
        )
        if (
            not isinstance(self.selected_observation_sequences, tuple)
            or self.selected_observation_sequences != expected_sequences
        ):
            raise TsdfError(
                "TSDF voxel cross-view observations are not the complete "
                "canonical stride selection"
            )
        if not isinstance(self.planned_block, bool):
            raise TsdfError(
                "TSDF voxel cross-view planned_block must be a bool"
            )
        if (
            isinstance(self.block_resolution, bool)
            or not isinstance(self.block_resolution, int)
            or self.block_resolution != TSDF_BLOCK_RESOLUTION
        ):
            raise TsdfError(
                "TSDF voxel cross-view requires block resolution "
                f"{TSDF_BLOCK_RESOLUTION}"
            )
        for value, label in (
            (self.voxel_size_m, "voxel size"),
            (self.truncation_m, "truncation"),
        ):
            if not _is_finite_number(value) or value <= 0.0:
                raise TsdfError(
                    f"TSDF voxel cross-view {label} must be finite and "
                    "positive"
                )
        _validate_image_size(self.image_size)
        if not isinstance(self.verdict, TsdfVoxelCrossViewVerdict):
            raise TsdfError("TSDF voxel cross-view verdict is invalid")

        expected_global, expected_block, expected_local = (
            _split_tsdf_global_voxel_index(self.global_index_xyz)
        )
        if (
            self.global_index_xyz != expected_global
            or self.block_index_xyz != expected_block
            or self.local_index_xyz != expected_local
        ):
            raise TsdfError(
                "TSDF voxel cross-view address decomposition is inconsistent"
            )
        _validate_point(self.world_xyz_m, "voxel centre")
        if self.world_xyz_m != _voxel_center_world_m(
            self.global_index_xyz,
            self.voxel_size_m,
        ):
            raise TsdfError(
                "TSDF voxel cross-view world centre does not match its "
                "global index and voxel size"
            )
        if (
            not isinstance(self.observation_receipts, tuple)
            or len(self.observation_receipts) != len(expected_sequences)
        ):
            raise TsdfError(
                "TSDF voxel cross-view requires exactly one sampling receipt "
                "for every selected observation"
            )
        if (
            len(self.observation_receipts)
            > MAX_TSDF_VOXEL_CROSS_VIEW_OBSERVATIONS
        ):
            raise TsdfError(
                "TSDF voxel cross-view exceeds the retained observation limit"
            )

        for position, receipt in enumerate(self.observation_receipts):
            if not isinstance(receipt, TsdfVoxelSamplingReceipt):
                raise TsdfError(
                    "TSDF voxel cross-view contains an invalid sampling "
                    "receipt"
                )
            if receipt.observation_sequence != expected_sequences[position]:
                raise TsdfError(
                    "TSDF voxel cross-view sampling receipts must follow the "
                    "canonical selected-sequence order"
                )
            if (
                receipt.source_plan_digest_sha256
                != self.source_plan_digest_sha256
                or receipt.replay_digest_sha256 != self.replay_digest_sha256
                or receipt.frame_stride != self.frame_stride
                or receipt.total_observations != self.total_observations
                or receipt.global_index_xyz != self.global_index_xyz
                or receipt.block_index_xyz != self.block_index_xyz
                or receipt.local_index_xyz != self.local_index_xyz
                or receipt.planned_block != self.planned_block
                or receipt.voxel_size_m != self.voxel_size_m
                or receipt.truncation_m != self.truncation_m
                or receipt.block_resolution != self.block_resolution
                or receipt.image_size != self.image_size
                or receipt.world_xyz_m != self.world_xyz_m
            ):
                raise TsdfError(
                    "TSDF voxel cross-view sampling receipt scope is "
                    "inconsistent"
                )

        expected_verdict = _combine_sampling_statuses(
            tuple(
                receipt.status for receipt in self.observation_receipts
            )
        )
        if self.verdict is not expected_verdict:
            raise TsdfError(
                "TSDF voxel cross-view verdict does not match its own "
                "sampling receipts"
            )

    @property
    def observation_count(self) -> int:
        return len(self.observation_receipts)

    @property
    def surface_band_count(self) -> int:
        return self._count(TsdfVoxelSamplingStatus.OBSERVED_SURFACE_BAND)

    @property
    def free_space_count(self) -> int:
        return self._count(TsdfVoxelSamplingStatus.OBSERVED_FREE_SPACE)

    @property
    def occluded_count(self) -> int:
        return self._count(TsdfVoxelSamplingStatus.UNOBSERVED_OCCLUDED)

    @property
    def unseen_count(self) -> int:
        return (
            self.observation_count
            - self.surface_band_count
            - self.free_space_count
            - self.occluded_count
        )

    @property
    def observed_count(self) -> int:
        return self.surface_band_count + self.free_space_count

    @property
    def reference_weight(self) -> int:
        """Report the weight the reference evaluator would accumulate."""

        return self.observed_count

    @property
    def reference_tsdf_sum(self) -> float:
        """Sum accepted truncated values in canonical observation order."""

        total = 0.0
        for receipt in self.observation_receipts:
            if receipt.contributes_to_reference_tsdf:
                if receipt.truncated_tsdf_value is None:
                    raise TsdfError(
                        "accepted TSDF voxel sampling must retain a value"
                    )
                total += receipt.truncated_tsdf_value
        if not math.isfinite(total):
            raise TsdfError("TSDF voxel cross-view sum must remain finite")
        return total

    @property
    def reference_tsdf_value(self) -> float | None:
        if self.reference_weight == 0:
            return None
        value = self.reference_tsdf_sum / self.reference_weight
        if not math.isfinite(value):
            raise TsdfError("TSDF voxel cross-view value must remain finite")
        return value

    @property
    def contributing_observation_sequences(self) -> tuple[int, ...]:
        return tuple(
            receipt.observation_sequence
            for receipt in self.observation_receipts
            if receipt.contributes_to_reference_tsdf
        )

    @property
    def wedge_observation_sequences(self) -> tuple[int, ...]:
        return tuple(
            receipt.observation_sequence
            for receipt in self.observation_receipts
            if receipt.inside_sampling_wedge
        )

    @property
    def observed(self) -> bool:
        return self.verdict in (
            TsdfVoxelCrossViewVerdict.SURFACE,
            TsdfVoxelCrossViewVerdict.FREE_SPACE,
        )

    @property
    def carvable_free_space(self) -> bool:
        """Report pure free space: seen empty and never in a surface band."""

        return self.verdict is TsdfVoxelCrossViewVerdict.FREE_SPACE

    @property
    def status_counts(
        self,
    ) -> tuple[tuple[TsdfVoxelSamplingStatus, int], ...]:
        return tuple(
            (status, count)
            for status in TsdfVoxelSamplingStatus
            if (count := self._count(status))
        )

    def _count(self, status: TsdfVoxelSamplingStatus) -> int:
        return sum(
            receipt.status is status
            for receipt in self.observation_receipts
        )


def classify_tsdf_voxel_across_observations_from_context(
    plan: TsdfBlockPlan,
    context: TsdfReplayDepthContext,
    global_index_xyz: _Index3,
) -> TsdfVoxelCrossViewReceipt:
    """Combine every selected observation's verdict for one voxel centre."""

    if not isinstance(plan, TsdfBlockPlan):
        raise TsdfError(
            "TSDF voxel cross-view requires a loaded TsdfBlockPlan"
        )
    if not isinstance(context, TsdfReplayDepthContext):
        raise TsdfError(
            "TSDF voxel cross-view requires a prepared "
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
                "canonical cross-view selection"
            )
        if len(selected_sequences) > MAX_TSDF_VOXEL_CROSS_VIEW_OBSERVATIONS:
            raise TsdfError(
                "TSDF voxel cross-view requires "
                f"{len(selected_sequences)} retained observations; reference "
                f"maximum is {MAX_TSDF_VOXEL_CROSS_VIEW_OBSERVATIONS}. Use a "
                "larger frame stride or a future scalable fusion path"
            )
        global_index, block_index, local_index = (
            _split_tsdf_global_voxel_index(global_index_xyz)
        )

        observation_receipts: list[TsdfVoxelSamplingReceipt] = []
        for position, observation_sequence in enumerate(selected_sequences):
            receipt = classify_tsdf_voxel_sampling_from_context(
                plan,
                context,
                observation_sequence,
                global_index,
            )
            if (
                not isinstance(receipt, TsdfVoxelSamplingReceipt)
                or receipt.observation_sequence != observation_sequence
                or receipt.global_index_xyz != global_index
                or len(observation_receipts) != position
            ):
                raise TsdfError(
                    "TSDF voxel cross-view child sampling scope is "
                    "inconsistent"
                )
            observation_receipts.append(receipt)

        statuses = tuple(
            receipt.status for receipt in observation_receipts
        )
        return TsdfVoxelCrossViewReceipt(
            source_plan_digest_sha256=plan.artifact_digest_sha256,
            replay_digest_sha256=plan.replay_digest_sha256,
            frame_stride=plan.frame_stride,
            total_observations=plan.total_observations,
            selected_observation_sequences=selected_sequences,
            global_index_xyz=global_index,
            block_index_xyz=block_index,
            local_index_xyz=local_index,
            planned_block=block_index in set(plan.active_blocks),
            voxel_size_m=plan.voxel_size_m,
            truncation_m=plan.truncation_m,
            block_resolution=plan.block_resolution,
            image_size=(context.camera.width, context.camera.height),
            world_xyz_m=_voxel_center_world_m(
                global_index,
                plan.voxel_size_m,
            ),
            verdict=_combine_sampling_statuses(statuses),
            observation_receipts=tuple(observation_receipts),
        )
    except TsdfError:
        raise
    except Exception as error:
        raise TsdfError(
            f"cannot classify prepared TSDF voxel across observations: "
            f"{error}"
        ) from error


def _combine_sampling_statuses(
    statuses: tuple[TsdfVoxelSamplingStatus, ...],
) -> TsdfVoxelCrossViewVerdict:
    """Resolve per-observation statuses by evidence specificity.

    Surface-band evidence localises a measured surface, so it outranks
    free-space evidence. Free space is positive evidence of emptiness, so it
    outranks occlusion. Occlusion and every missing-input status carry no
    positive evidence at all and never become free space.
    """

    for status in statuses:
        if not isinstance(status, TsdfVoxelSamplingStatus):
            raise TsdfError(
                "TSDF voxel cross-view sampling status is invalid"
            )
    if TsdfVoxelSamplingStatus.OBSERVED_SURFACE_BAND in statuses:
        return TsdfVoxelCrossViewVerdict.SURFACE
    if TsdfVoxelSamplingStatus.OBSERVED_FREE_SPACE in statuses:
        return TsdfVoxelCrossViewVerdict.FREE_SPACE
    if TsdfVoxelSamplingStatus.UNOBSERVED_OCCLUDED in statuses:
        return TsdfVoxelCrossViewVerdict.OCCLUDED
    return TsdfVoxelCrossViewVerdict.UNSEEN
