"""Read-only expansion proposal from a resolved coverage domain."""

from __future__ import annotations

from dataclasses import dataclass

from .errors import TsdfError
from .tsdf_block_plan import (
    MAX_PLANNED_BLOCKS,
    TSDF_BLOCK_RESOLUTION,
    _ordered_blocks,
)
from .tsdf_block_plan_loader import TsdfBlockPlan
from .tsdf_block_storage import TSDF_BLOCK_VOXELS
from .tsdf_domain_cross_view import TsdfCoverageDomainCrossViewReceipt
from .tsdf_observation_block_rays import (
    _is_sha256,
    _validate_block_index_xyz,
    _validate_trace_plan,
)

_Index3 = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class TsdfPlanExpansionProposal:
    """Immutable proposal for the blocks an expanded plan would hold."""

    source_plan_digest_sha256: str
    replay_digest_sha256: str
    frame_stride: int
    total_observations: int
    block_resolution: int
    source_plan_block_indices: tuple[_Index3, ...]
    domain_block_indices: tuple[_Index3, ...]
    approved_block_indices: tuple[_Index3, ...]
    rejected_block_indices: tuple[_Index3, ...]
    expanded_block_indices: tuple[_Index3, ...]

    def __post_init__(self) -> None:
        if not _is_sha256(self.source_plan_digest_sha256):
            raise TsdfError(
                "TSDF plan expansion source plan digest is invalid"
            )
        if not _is_sha256(self.replay_digest_sha256):
            raise TsdfError("TSDF plan expansion replay digest is invalid")
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
                    f"TSDF plan expansion {label} must be positive"
                )
        if (
            isinstance(self.block_resolution, bool)
            or not isinstance(self.block_resolution, int)
            or self.block_resolution != TSDF_BLOCK_RESOLUTION
        ):
            raise TsdfError(
                "TSDF plan expansion requires block resolution "
                f"{TSDF_BLOCK_RESOLUTION}"
            )
        for value, label in (
            (self.source_plan_block_indices, "source plan blocks"),
            (self.domain_block_indices, "domain blocks"),
            (self.approved_block_indices, "approved blocks"),
            (self.rejected_block_indices, "rejected blocks"),
            (self.expanded_block_indices, "expanded blocks"),
        ):
            _validate_canonical_expansion_blocks(value, label)
        if not self.source_plan_block_indices:
            raise TsdfError(
                "TSDF plan expansion requires a nonempty source plan"
            )
        if not self.domain_block_indices:
            raise TsdfError(
                "TSDF plan expansion requires a nonempty coverage domain"
            )

        domain = set(self.domain_block_indices)
        approved = set(self.approved_block_indices)
        rejected = set(self.rejected_block_indices)
        source = set(self.source_plan_block_indices)
        expanded = set(self.expanded_block_indices)
        if approved & rejected or approved | rejected != domain:
            raise TsdfError(
                "TSDF plan expansion approval partition is inconsistent"
            )
        if self.expanded_block_indices != _ordered_blocks(source | approved):
            raise TsdfError(
                "TSDF plan expansion blocks are not the canonical union of "
                "the source plan and its approved coverage"
            )
        if not source <= expanded:
            raise TsdfError(
                "TSDF plan expansion must retain every source plan block"
            )
        if len(self.expanded_block_indices) > MAX_PLANNED_BLOCKS:
            raise TsdfError(
                "TSDF plan expansion exceeds the "
                f"{MAX_PLANNED_BLOCKS}-block reference limit"
            )

    @property
    def source_block_count(self) -> int:
        return len(self.source_plan_block_indices)

    @property
    def domain_block_count(self) -> int:
        return len(self.domain_block_indices)

    @property
    def approved_block_count(self) -> int:
        return len(self.approved_block_indices)

    @property
    def rejected_block_count(self) -> int:
        return len(self.rejected_block_indices)

    @property
    def expanded_block_count(self) -> int:
        return len(self.expanded_block_indices)

    @property
    def added_block_indices(self) -> tuple[_Index3, ...]:
        return _ordered_blocks(
            set(self.expanded_block_indices)
            - set(self.source_plan_block_indices)
        )

    @property
    def added_block_count(self) -> int:
        return len(self.added_block_indices)

    @property
    def retained_block_count(self) -> int:
        return self.source_block_count

    @property
    def removed_block_count(self) -> int:
        """Report removals, which this proposal never performs."""

        return 0

    @property
    def source_voxel_slots(self) -> int:
        return self.source_block_count * TSDF_BLOCK_VOXELS

    @property
    def expanded_voxel_slots(self) -> int:
        return self.expanded_block_count * TSDF_BLOCK_VOXELS

    @property
    def added_voxel_slots(self) -> int:
        return self.added_block_count * TSDF_BLOCK_VOXELS

    @property
    def expands_plan(self) -> bool:
        return bool(self.added_block_count)


def propose_tsdf_plan_expansion_from_domain(
    plan: TsdfBlockPlan,
    domain: TsdfCoverageDomainCrossViewReceipt,
) -> TsdfPlanExpansionProposal:
    """Approve evidence-bearing coverage and merge it with the source plan."""

    if not isinstance(plan, TsdfBlockPlan):
        raise TsdfError(
            "TSDF plan expansion requires a loaded TsdfBlockPlan"
        )
    if not isinstance(domain, TsdfCoverageDomainCrossViewReceipt):
        raise TsdfError(
            "TSDF plan expansion requires a resolved "
            "TsdfCoverageDomainCrossViewReceipt"
        )

    try:
        _validate_trace_plan(plan)
        if (
            domain.source_plan_digest_sha256 != plan.artifact_digest_sha256
            or domain.replay_digest_sha256 != plan.replay_digest_sha256
        ):
            raise TsdfError(
                "TSDF coverage domain provenance does not match the block "
                "plan"
            )
        if (
            domain.frame_stride != plan.frame_stride
            or domain.total_observations != plan.total_observations
            or domain.block_resolution != plan.block_resolution
        ):
            raise TsdfError(
                "TSDF coverage domain selection does not match the block plan"
            )

        approved: list[_Index3] = []
        rejected: list[_Index3] = []
        for receipt in domain.block_receipts:
            if receipt.observed_voxel_count > 0:
                approved.append(receipt.block_index_xyz)
            else:
                rejected.append(receipt.block_index_xyz)

        source = set(plan.active_blocks)
        expanded = _ordered_blocks(source | set(approved))
        if len(expanded) > MAX_PLANNED_BLOCKS:
            raise TsdfError(
                "TSDF plan expansion would exceed the "
                f"{MAX_PLANNED_BLOCKS}-block reference limit. Use a coarser "
                "voxel size or a future scalable planning path"
            )
        return TsdfPlanExpansionProposal(
            source_plan_digest_sha256=plan.artifact_digest_sha256,
            replay_digest_sha256=plan.replay_digest_sha256,
            frame_stride=plan.frame_stride,
            total_observations=plan.total_observations,
            block_resolution=plan.block_resolution,
            source_plan_block_indices=plan.active_blocks,
            domain_block_indices=domain.domain_block_indices,
            approved_block_indices=_ordered_blocks(set(approved)),
            rejected_block_indices=_ordered_blocks(set(rejected)),
            expanded_block_indices=expanded,
        )
    except TsdfError:
        raise
    except Exception as error:
        raise TsdfError(
            f"cannot propose TSDF plan expansion: {error}"
        ) from error


def _validate_canonical_expansion_blocks(value: object, label: str) -> None:
    if not isinstance(value, tuple):
        raise TsdfError(f"TSDF plan expansion {label} must be a tuple")
    previous_key: tuple[int, int, int] | None = None
    for block_index in value:
        _validate_block_index_xyz(block_index, label)
        key = (block_index[2], block_index[1], block_index[0])
        if previous_key is not None and key <= previous_key:
            raise TsdfError(
                f"TSDF plan expansion {label} must be unique and strictly "
                "x-fastest ordered"
            )
        previous_key = key
