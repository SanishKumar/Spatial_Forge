"""Conservative footprint coverage for every pixel of one observation."""

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
    _validate_point,
    _validate_trace_plan,
)
from .tsdf_pixel_footprint_coverage import (
    TsdfPixelFootprintCoverageReceipt,
    TsdfPixelFootprintStatus,
    evaluate_tsdf_pixel_footprint_coverage_from_context,
)
from .tsdf_replay_depth_context import (
    TsdfReplayDepthContext,
    TsdfReplayDepthStatus,
)
from .tsdf_voxel_contribution import _validate_contribution_context

_BlockIndex = tuple[int, int, int]
_Point3 = tuple[float, float, float]

MAX_TSDF_OBSERVATION_FOOTPRINT_CANDIDATE_BLOCKS = 262_144


@dataclass(frozen=True, slots=True)
class TsdfObservationFootprintReceipt:
    """Immutable footprint transcript for one prepared observation."""

    source_plan_digest_sha256: str
    replay_digest_sha256: str
    frame_stride: int
    total_observations: int
    source_plan_block_indices: tuple[_BlockIndex, ...]
    observation_sequence: int
    observation_status: TsdfReplayDepthStatus
    block_resolution: int
    block_extent_m: float
    image_size: tuple[int, int]
    camera_origin_world_m: _Point3 | None
    pixel_receipts: tuple[TsdfPixelFootprintCoverageReceipt, ...]
    covered_block_indices: tuple[_BlockIndex, ...]
    existing_plan_block_indices: tuple[_BlockIndex, ...]
    unplanned_block_indices: tuple[_BlockIndex, ...]

    def __post_init__(self) -> None:
        if not _is_sha256(self.source_plan_digest_sha256):
            raise TsdfError(
                "TSDF observation footprint source plan digest is invalid"
            )
        if not _is_sha256(self.replay_digest_sha256):
            raise TsdfError(
                "TSDF observation footprint replay digest is invalid"
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
                    f"TSDF observation footprint {label} must be positive"
                )
        _validate_canonical_footprint_blocks(
            self.source_plan_block_indices,
            "source plan blocks",
        )
        if (
            not self.source_plan_block_indices
            or len(self.source_plan_block_indices) > MAX_PLANNED_BLOCKS
        ):
            raise TsdfError(
                "TSDF observation footprint source plan blocks are invalid"
            )
        if (
            isinstance(self.observation_sequence, bool)
            or not isinstance(self.observation_sequence, int)
            or self.observation_sequence < 0
        ):
            raise TsdfError(
                "TSDF observation footprint sequence is invalid"
            )
        if self.observation_sequence >= self.total_observations:
            raise TsdfError(
                "TSDF observation footprint sequence is outside its total "
                "observation range"
            )
        if self.observation_sequence % self.frame_stride != 0:
            raise TsdfError(
                "TSDF observation footprint sequence is not selected by its "
                "frame stride"
            )
        if not isinstance(self.observation_status, TsdfReplayDepthStatus):
            raise TsdfError(
                "TSDF observation footprint prepared status is invalid"
            )
        if (
            isinstance(self.block_resolution, bool)
            or not isinstance(self.block_resolution, int)
            or self.block_resolution != TSDF_BLOCK_RESOLUTION
        ):
            raise TsdfError(
                "TSDF observation footprint requires block resolution "
                f"{TSDF_BLOCK_RESOLUTION}"
            )
        if (
            not _is_finite_number(self.block_extent_m)
            or self.block_extent_m <= 0.0
        ):
            raise TsdfError(
                "TSDF observation footprint block extent must be finite and "
                "positive"
            )
        _validate_image_size(self.image_size)
        if not isinstance(self.pixel_receipts, tuple):
            raise TsdfError(
                "TSDF observation footprint pixel receipts must be a tuple"
            )

        pose_available = self.observation_status in (
            TsdfReplayDepthStatus.READY,
            TsdfReplayDepthStatus.MISSING_DEPTH,
        )
        if pose_available:
            _validate_point(self.camera_origin_world_m, "camera origin")
        elif self.camera_origin_world_m is not None:
            raise TsdfError(
                "pose-missing TSDF observation footprint cannot retain a "
                "camera origin"
            )

        if self.observation_status is TsdfReplayDepthStatus.READY:
            expected_pixels = self.image_size[0] * self.image_size[1]
            if len(self.pixel_receipts) != expected_pixels:
                raise TsdfError(
                    "ready TSDF observation footprint requires exactly one "
                    "receipt per camera pixel"
                )
            for pixel_index, receipt in enumerate(self.pixel_receipts):
                if not isinstance(
                    receipt,
                    TsdfPixelFootprintCoverageReceipt,
                ):
                    raise TsdfError(
                        "TSDF observation footprint contains an invalid "
                        "pixel receipt"
                    )
                expected_uv = (
                    pixel_index % self.image_size[0],
                    pixel_index // self.image_size[0],
                )
                if receipt.pixel_uv != expected_uv:
                    raise TsdfError(
                        "TSDF observation footprint receipts must be in "
                        "canonical row-major pixel order"
                    )
                if (
                    receipt.source_plan_digest_sha256
                    != self.source_plan_digest_sha256
                    or receipt.replay_digest_sha256
                    != self.replay_digest_sha256
                    or receipt.frame_stride != self.frame_stride
                    or receipt.total_observations != self.total_observations
                    or receipt.source_plan_block_indices
                    != self.source_plan_block_indices
                    or receipt.observation_sequence
                    != self.observation_sequence
                    or receipt.observation_status != self.observation_status
                    or receipt.block_resolution != self.block_resolution
                    or receipt.block_extent_m != self.block_extent_m
                    or receipt.image_size != self.image_size
                    or receipt.camera_origin_world_m
                    != self.camera_origin_world_m
                ):
                    raise TsdfError(
                        "TSDF observation footprint pixel receipt scope is "
                        "inconsistent"
                    )
        elif self.pixel_receipts:
            raise TsdfError(
                "input-missing TSDF observation footprint cannot retain "
                "pixel receipts"
            )

        expected_covered = _ordered_blocks(
            {
                block_index
                for receipt in self.pixel_receipts
                for block_index in receipt.covered_block_indices
            }
        )
        if self.covered_block_indices != expected_covered:
            raise TsdfError(
                "TSDF observation footprint covered blocks do not match its "
                "pixel receipts"
            )
        _validate_canonical_footprint_blocks(
            self.existing_plan_block_indices,
            "existing-plan blocks",
        )
        _validate_canonical_footprint_blocks(
            self.unplanned_block_indices,
            "unplanned blocks",
        )
        covered = set(self.covered_block_indices)
        existing = set(self.existing_plan_block_indices)
        unplanned = set(self.unplanned_block_indices)
        if existing & unplanned or existing | unplanned != covered:
            raise TsdfError(
                "TSDF observation footprint plan partition is inconsistent"
            )
        source_plan = set(self.source_plan_block_indices)
        if (
            self.existing_plan_block_indices
            != _ordered_blocks(covered & source_plan)
            or self.unplanned_block_indices
            != _ordered_blocks(covered - source_plan)
        ):
            raise TsdfError(
                "TSDF observation footprint plan membership is inconsistent"
            )
        if not set(self.centerline_block_indices) <= covered:
            raise TsdfError(
                "TSDF observation footprint coverage does not contain its "
                "own centreline paths"
            )
        if len(self.covered_block_indices) > MAX_PLANNED_BLOCKS:
            raise TsdfError(
                "TSDF observation footprint exceeds the unique-block limit"
            )
        if (
            self.candidate_block_count
            > MAX_TSDF_OBSERVATION_FOOTPRINT_CANDIDATE_BLOCKS
        ):
            raise TsdfError(
                "TSDF observation footprint exceeds the candidate block limit"
            )

    @property
    def pixel_count(self) -> int:
        return len(self.pixel_receipts)

    @property
    def covered_pixel_count(self) -> int:
        return sum(receipt.covered for receipt in self.pixel_receipts)

    @property
    def depth_invalid_count(self) -> int:
        return sum(
            receipt.status is TsdfPixelFootprintStatus.DEPTH_INVALID
            for receipt in self.pixel_receipts
        )

    @property
    def candidate_block_count(self) -> int:
        return sum(
            receipt.candidate_block_count
            for receipt in self.pixel_receipts
        )

    @property
    def rejected_candidate_count(self) -> int:
        return sum(
            receipt.rejected_candidate_count
            for receipt in self.pixel_receipts
        )

    @property
    def block_visit_count(self) -> int:
        return sum(
            receipt.covered_block_count
            for receipt in self.pixel_receipts
        )

    @property
    def duplicate_block_visit_count(self) -> int:
        return self.block_visit_count - len(self.covered_block_indices)

    @property
    def maximum_blocks_per_pixel(self) -> int:
        return max(
            (
                receipt.covered_block_count
                for receipt in self.pixel_receipts
            ),
            default=0,
        )

    @property
    def centerline_block_indices(self) -> tuple[_BlockIndex, ...]:
        return _ordered_blocks(
            {
                block_index
                for receipt in self.pixel_receipts
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
    def covered_block_pixel_counts(
        self,
    ) -> tuple[tuple[_BlockIndex, int], ...]:
        counts: dict[_BlockIndex, int] = {}
        for receipt in self.pixel_receipts:
            for block_index in receipt.covered_block_indices:
                counts[block_index] = counts.get(block_index, 0) + 1
        return tuple(
            (block_index, counts[block_index])
            for block_index in self.covered_block_indices
        )

    @property
    def maximum_block_pixel_count(self) -> int:
        return max(
            (count for _, count in self.covered_block_pixel_counts),
            default=0,
        )

    @property
    def prepared_depth_accessed(self) -> bool:
        return self.observation_status is TsdfReplayDepthStatus.READY


def survey_tsdf_observation_pixel_footprints_from_context(
    plan: TsdfBlockPlan,
    context: TsdfReplayDepthContext,
    observation_sequence: int,
) -> TsdfObservationFootprintReceipt:
    """Cover every pixel's sampling wedge for one prepared observation."""

    if not isinstance(plan, TsdfBlockPlan):
        raise TsdfError(
            "TSDF observation footprint requires a loaded TsdfBlockPlan"
        )
    if not isinstance(context, TsdfReplayDepthContext):
        raise TsdfError(
            "TSDF observation footprint requires a prepared "
            "TsdfReplayDepthContext"
        )
    if isinstance(observation_sequence, bool) or not isinstance(
        observation_sequence,
        int,
    ):
        raise TsdfError("observation_sequence: expected an integer")
    if observation_sequence < 0:
        raise TsdfError(
            "observation_sequence: expected a non-negative integer"
        )

    try:
        _validate_trace_plan(plan)
        _validate_contribution_context(plan, context)
        observation = _select_observation(context, observation_sequence)
        image_size = (context.camera.width, context.camera.height)
        camera_origin = _camera_origin(observation.t_world_camera)
        if observation.status is not TsdfReplayDepthStatus.READY:
            return TsdfObservationFootprintReceipt(
                source_plan_digest_sha256=plan.artifact_digest_sha256,
                replay_digest_sha256=plan.replay_digest_sha256,
                frame_stride=plan.frame_stride,
                total_observations=plan.total_observations,
                source_plan_block_indices=plan.active_blocks,
                observation_sequence=observation_sequence,
                observation_status=observation.status,
                block_resolution=plan.block_resolution,
                block_extent_m=plan.block_extent_m,
                image_size=image_size,
                camera_origin_world_m=camera_origin,
                pixel_receipts=(),
                covered_block_indices=(),
                existing_plan_block_indices=(),
                unplanned_block_indices=(),
            )

        pixel_receipts: list[TsdfPixelFootprintCoverageReceipt] = []
        covered_blocks: set[_BlockIndex] = set()
        candidate_blocks = 0
        for pixel_v in range(image_size[1]):
            for pixel_u in range(image_size[0]):
                receipt = evaluate_tsdf_pixel_footprint_coverage_from_context(
                    plan,
                    context,
                    observation_sequence,
                    (pixel_u, pixel_v),
                )
                if (
                    not isinstance(
                        receipt,
                        TsdfPixelFootprintCoverageReceipt,
                    )
                    or receipt.pixel_uv != (pixel_u, pixel_v)
                    or receipt.observation_sequence != observation_sequence
                ):
                    raise TsdfError(
                        "TSDF observation footprint child pixel scope is "
                        "inconsistent"
                    )
                candidate_blocks += receipt.candidate_block_count
                if (
                    candidate_blocks
                    > MAX_TSDF_OBSERVATION_FOOTPRINT_CANDIDATE_BLOCKS
                ):
                    raise TsdfError(
                        "TSDF observation footprint candidate blocks reach "
                        f"{candidate_blocks}; reference maximum is "
                        f"{MAX_TSDF_OBSERVATION_FOOTPRINT_CANDIDATE_BLOCKS}. "
                        "Use a coarser voxel size, a smaller image, or a "
                        "future scalable coverage path"
                    )
                covered_blocks.update(receipt.covered_block_indices)
                if len(covered_blocks) > MAX_PLANNED_BLOCKS:
                    raise TsdfError(
                        "TSDF observation footprint exceeds the "
                        f"{MAX_PLANNED_BLOCKS}-block reference limit. Use a "
                        "coarser voxel size or a future scalable coverage "
                        "path"
                    )
                pixel_receipts.append(receipt)

        active = set(plan.active_blocks)
        return TsdfObservationFootprintReceipt(
            source_plan_digest_sha256=plan.artifact_digest_sha256,
            replay_digest_sha256=plan.replay_digest_sha256,
            frame_stride=plan.frame_stride,
            total_observations=plan.total_observations,
            source_plan_block_indices=plan.active_blocks,
            observation_sequence=observation_sequence,
            observation_status=observation.status,
            block_resolution=plan.block_resolution,
            block_extent_m=plan.block_extent_m,
            image_size=image_size,
            camera_origin_world_m=camera_origin,
            pixel_receipts=tuple(pixel_receipts),
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
            f"cannot survey prepared TSDF observation footprints: {error}"
        ) from error


def _select_observation(
    context: TsdfReplayDepthContext,
    observation_sequence: int,
):
    if observation_sequence >= context.total_observations:
        raise TsdfError(
            "observation_sequence: outside context range "
            f"[0, {context.total_observations - 1}]"
        )
    if observation_sequence % context.frame_stride != 0:
        raise TsdfError(
            "observation_sequence: not selected by the block plan's "
            f"frame_stride={context.frame_stride}"
        )
    observation = context.observations[
        observation_sequence // context.frame_stride
    ]
    if observation.observation_sequence != observation_sequence:
        raise TsdfError(
            "TSDF replay/depth context observation lookup is inconsistent"
        )
    return observation


def _camera_origin(transform: tuple[float, ...] | None) -> _Point3 | None:
    if transform is None:
        return None
    origin = (transform[3], transform[7], transform[11])
    _validate_point(origin, "camera origin")
    return origin


def _validate_canonical_footprint_blocks(value: object, label: str) -> None:
    if not isinstance(value, tuple):
        raise TsdfError(
            f"TSDF observation footprint {label} must be a tuple"
        )
    previous_key: tuple[int, int, int] | None = None
    for block_index in value:
        _validate_block_index_xyz(block_index, label)
        key = (block_index[2], block_index[1], block_index[0])
        if previous_key is not None and key <= previous_key:
            raise TsdfError(
                f"TSDF observation footprint {label} must be unique and "
                "strictly x-fastest ordered"
            )
        previous_key = key
