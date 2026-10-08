"""Whole-plan fusion that holds one decoded depth frame at a time.

Every earlier fusion path reads its observations from a
``TsdfReplayDepthContext``, which decodes each selected frame once and keeps
all of them. That is what removed file I/O from the hot path, and it is also
a ceiling: float64 depth for every selected frame, held at once, runs out of
room after about two hundred 640x480 frames.

Fusion does not need the frames together. Each voxel's value is a sum over
observations in canonical order, so the frames can be consumed one after
another: decode a frame, evaluate every planned voxel against it, add the
result, discard the frame. Memory no longer depends on sequence length.

The result is byte-identical to the block-by-block paths, and not by
coincidence. Per-voxel accumulation order is what determines the last bits,
and it is the same here: observation order, one addition per observation,
``+0.0`` where a voxel is skipped. Walking frames in the outer loop instead
of blocks changes which voxel is visited when, never the order in which one
voxel receives its contributions. The evaluation itself is the single shared
``_evaluate_ready_voxels``.

Most of that evaluation is spent on voxels a frame cannot see. In a room,
four in five voxel-observations end as "behind the camera" or "projects
outside the image". Those two verdicts can be reached for a whole block at
once, because each is a half-space in camera coordinates and a block's
voxel centres lie inside the box spanned by its eight extreme centres: if
all eight corners are behind the camera, or all eight are on the outer
side of one edge of the image, so is every centre between them.

Such a block is not evaluated. Its 512 voxels are counted under the
status every one of them would have received, and nothing is added to its
accumulators, which is what the evaluation would have added: ``+0.0`` and
zero weight. The fused bytes and the receipt are therefore unchanged, and
the tests hold the block verdict to the per-voxel one directly.

A pass can also be taken in stages. Frames are consumed in order and a
voxel's sum is the same additions whichever stage makes them, so stopping
after any frame and continuing later arrives at the same bytes. What has
to be carried between stages is small: how many of the selected
observations are behind it, the counts they produced, and the digests of
the accumulators those counts describe. ``TsdfStreamFusionProgress`` is
that record, and a stage refuses storage it does not describe.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .errors import PointCloudError, SessionReplayError, TsdfError
from .model import ScanSession
from .point_cloud import (
    _read_depth_array,
    _sample_path,
    _validate_reconstruction_contract,
)
from .replay import replay_session
from .tsdf_block_contributions import (
    _LOCAL_X,
    _LOCAL_Y,
    _LOCAL_Z,
    _PREPARATION_SKIP_STATUS,
    _STATUS_CODE,
    TSDF_CONTRIBUTION_STATUS_ORDER,
    _evaluate_ready_voxels,
)
from .tsdf_block_plan import TSDF_BLOCK_RESOLUTION
from .tsdf_block_storage import TSDF_BLOCK_VOXELS, TsdfBlockStorage
from .tsdf_replay_depth_context import (
    TsdfReplayDepthStatus,
    _classify_observation,
    _is_rigid_transform,
    _prepared_transform,
    _validate_context_camera,
    _validate_context_plan,
    _validate_plan_associations,
)
from .tsdf_voxel_contribution import (
    TsdfContributionStatus,
    _validate_contribution_plan,
)
from .tsdf_voxel_update import (
    MAX_TSDF_VOXEL_WEIGHT,
    _validate_update_storage,
)

# Blocks evaluated per NumPy pass. Small enough that the dozen or so
# temporaries of one pass stay cache-resident, which is worth more than the
# Python overhead it costs; the fused bytes do not depend on it.
STREAM_FUSION_CHUNK_BLOCKS = 256

# Verdicts of the whole-block visibility test.
BLOCK_EVALUATE = 0
BLOCK_BEHIND_CAMERA = 1
BLOCK_OUTSIDE_IMAGE = 2

# A whole-block verdict is a claim about 512 voxels that are evaluated with
# rounding the block test does not reproduce, so it is only given with room
# to spare: this far in depth, relative to how far away the block is, and
# this many pixels beyond the edge of the image. For a camera within the
# bounds below, rounding is orders of magnitude smaller than either.
_CULL_RELATIVE_DEPTH_SLACK = 1e-9
_CULL_PIXEL_SLACK = 0.5
_CULL_MAX_FOCAL_PIXELS = 1.0e4
_CULL_MAX_IMAGE_PIXELS = 1.0e5

# Set to False to evaluate every voxel of every block; the result must be
# the same, and a test fuses both ways to hold it to that.
STREAM_FUSION_CULLS_BLOCKS = True


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True, slots=True)
class TsdfStreamFusionReceipt:
    """Immutable account of one whole-plan streaming fusion."""

    session_id: str
    source_plan_digest_sha256: str
    replay_digest_sha256: str
    frame_stride: int
    total_observations: int
    selected_observations: int
    fused_observations: int
    skipped_missing_depth: int
    skipped_missing_pose: int
    block_count: int
    voxel_slots: int
    status_counts: tuple[tuple[TsdfContributionStatus, int], ...]
    observed_voxel_count: int
    maximum_weight: int
    valid_depth_samples: int
    invalid_depth_samples: int
    peak_retained_depth_bytes: int
    tsdf_sums_sha256: str
    weights_sha256: str

    def __post_init__(self) -> None:
        for digest, label in (
            (self.source_plan_digest_sha256, "source plan digest"),
            (self.replay_digest_sha256, "replay digest"),
            (self.tsdf_sums_sha256, "sum payload digest"),
            (self.weights_sha256, "weight payload digest"),
        ):
            if not _is_sha256(digest):
                raise TsdfError(f"TSDF stream fusion {label} is invalid")
        for value, label in (
            (self.frame_stride, "frame stride"),
            (self.total_observations, "total observations"),
            (self.selected_observations, "selected observations"),
            (self.block_count, "block count"),
            (self.voxel_slots, "voxel slots"),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
            ):
                raise TsdfError(
                    f"TSDF stream fusion {label} must be a positive integer"
                )
        for value, label in (
            (self.fused_observations, "fused observations"),
            (self.skipped_missing_depth, "missing-depth count"),
            (self.skipped_missing_pose, "missing-pose count"),
            (self.observed_voxel_count, "observed voxel count"),
            (self.maximum_weight, "maximum weight"),
            (self.valid_depth_samples, "valid depth samples"),
            (self.invalid_depth_samples, "invalid depth samples"),
            (self.peak_retained_depth_bytes, "retained depth bytes"),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise TsdfError(
                    f"TSDF stream fusion {label} must be a nonnegative "
                    "integer"
                )
        if self.voxel_slots != self.block_count * TSDF_BLOCK_VOXELS:
            raise TsdfError(
                "TSDF stream fusion voxel slots do not match its blocks"
            )
        expected_selected = (
            (self.total_observations - 1) // self.frame_stride
        ) + 1
        if self.selected_observations != expected_selected:
            raise TsdfError(
                "TSDF stream fusion selection does not match its stride"
            )
        skipped = self.selected_observations - self.fused_observations
        if not (
            max(self.skipped_missing_depth, self.skipped_missing_pose)
            <= skipped
            <= self.skipped_missing_depth + self.skipped_missing_pose
        ):
            raise TsdfError(
                "TSDF stream fusion skipped-frame counts are inconsistent"
            )

        if not isinstance(self.status_counts, tuple) or any(
            not isinstance(entry, tuple)
            or len(entry) != 2
            or not isinstance(entry[0], TsdfContributionStatus)
            or isinstance(entry[1], bool)
            or not isinstance(entry[1], int)
            or entry[1] < 1
            for entry in self.status_counts
        ):
            raise TsdfError("TSDF stream fusion status counts are invalid")
        order = [
            _STATUS_CODE[status] for status, _ in self.status_counts
        ]
        if order != sorted(set(order)):
            raise TsdfError(
                "TSDF stream fusion status counts are not in canonical order"
            )
        if self.evaluated_count != (
            self.voxel_slots * self.selected_observations
        ):
            raise TsdfError(
                "TSDF stream fusion status counts do not cover every "
                "voxel-observation"
            )
        if (
            self.observed_voxel_count > self.voxel_slots
            or self.maximum_weight > self.fused_observations
            or self.applied_count
            > self.observed_voxel_count * max(self.maximum_weight, 1)
            or self.applied_count < self.observed_voxel_count
            or (self.observed_voxel_count == 0) != (self.maximum_weight == 0)
        ):
            raise TsdfError(
                "TSDF stream fusion weight summary is inconsistent"
            )

    @property
    def evaluated_count(self) -> int:
        return sum(count for _, count in self.status_counts)

    @property
    def applied_count(self) -> int:
        return sum(
            count
            for status, count in self.status_counts
            if status is TsdfContributionStatus.CONTRIBUTES
        )

    @property
    def skipped_count(self) -> int:
        return self.evaluated_count - self.applied_count

    @property
    def unknown_voxel_count(self) -> int:
        return self.voxel_slots - self.observed_voxel_count


@dataclass(frozen=True, slots=True)
class TsdfStreamFusionProgress:
    """How far a staged fusion has got, bound to the bytes it has produced.

    ``processed_observations`` is a prefix of the plan's selection: that
    many observations, in order, have been fused or skipped for a missing
    input. The two payload digests are those of the storage at that point.
    """

    session_id: str
    source_plan_digest_sha256: str
    replay_digest_sha256: str
    frame_stride: int
    total_observations: int
    selected_observations: int
    processed_observations: int
    fused_observations: int
    skipped_missing_depth: int
    skipped_missing_pose: int
    block_count: int
    voxel_slots: int
    status_counts: tuple[tuple[TsdfContributionStatus, int], ...]
    valid_depth_samples: int
    invalid_depth_samples: int
    tsdf_sums_sha256: str
    weights_sha256: str

    def __post_init__(self) -> None:
        for digest, label in (
            (self.source_plan_digest_sha256, "source plan digest"),
            (self.replay_digest_sha256, "replay digest"),
            (self.tsdf_sums_sha256, "sum payload digest"),
            (self.weights_sha256, "weight payload digest"),
        ):
            if not _is_sha256(digest):
                raise TsdfError(
                    f"TSDF stream fusion progress {label} is invalid"
                )
        if not isinstance(self.session_id, str) or not self.session_id:
            raise TsdfError("TSDF stream fusion progress session is invalid")
        for value, label, minimum in (
            (self.frame_stride, "frame stride", 1),
            (self.total_observations, "total observations", 1),
            (self.selected_observations, "selected observations", 1),
            (self.block_count, "block count", 1),
            (self.voxel_slots, "voxel slots", 1),
            (self.processed_observations, "processed observations", 0),
            (self.fused_observations, "fused observations", 0),
            (self.skipped_missing_depth, "missing-depth count", 0),
            (self.skipped_missing_pose, "missing-pose count", 0),
            (self.valid_depth_samples, "valid depth samples", 0),
            (self.invalid_depth_samples, "invalid depth samples", 0),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < minimum
            ):
                raise TsdfError(
                    f"TSDF stream fusion progress {label} must be an "
                    f"integer of at least {minimum}"
                )
        if self.voxel_slots != self.block_count * TSDF_BLOCK_VOXELS:
            raise TsdfError(
                "TSDF stream fusion progress voxel slots do not match its "
                "blocks"
            )
        expected_selected = (
            (self.total_observations - 1) // self.frame_stride
        ) + 1
        if self.selected_observations != expected_selected:
            raise TsdfError(
                "TSDF stream fusion progress selection does not match its "
                "stride"
            )
        if not (
            self.fused_observations
            <= self.processed_observations
            <= self.selected_observations
        ):
            raise TsdfError(
                "TSDF stream fusion progress has processed more "
                "observations than it selected, or fused more than it "
                "processed"
            )
        skipped = self.processed_observations - self.fused_observations
        if not (
            max(self.skipped_missing_depth, self.skipped_missing_pose)
            <= skipped
            <= self.skipped_missing_depth + self.skipped_missing_pose
        ):
            raise TsdfError(
                "TSDF stream fusion progress skipped-frame counts are "
                "inconsistent"
            )
        if not isinstance(self.status_counts, tuple) or any(
            not isinstance(entry, tuple)
            or len(entry) != 2
            or not isinstance(entry[0], TsdfContributionStatus)
            or isinstance(entry[1], bool)
            or not isinstance(entry[1], int)
            or entry[1] < 1
            for entry in self.status_counts
        ):
            raise TsdfError(
                "TSDF stream fusion progress status counts are invalid"
            )
        order = [_STATUS_CODE[status] for status, _ in self.status_counts]
        if order != sorted(set(order)):
            raise TsdfError(
                "TSDF stream fusion progress status counts are not in "
                "canonical order"
            )
        if self.evaluated_count != (
            self.voxel_slots * self.processed_observations
        ):
            raise TsdfError(
                "TSDF stream fusion progress status counts do not cover "
                "every voxel-observation processed"
            )
        if self.fused_observations == 0 and (
            self.applied_count
            or self.valid_depth_samples
            or self.invalid_depth_samples
        ):
            raise TsdfError(
                "TSDF stream fusion progress reports depth from frames it "
                "did not fuse"
            )

    @property
    def evaluated_count(self) -> int:
        return sum(count for _, count in self.status_counts)

    @property
    def applied_count(self) -> int:
        return sum(
            count
            for status, count in self.status_counts
            if status is TsdfContributionStatus.CONTRIBUTES
        )

    @property
    def remaining_observations(self) -> int:
        return self.selected_observations - self.processed_observations

    @property
    def is_complete(self) -> bool:
        return self.remaining_observations == 0


def fuse_tsdf_plan_streaming(
    storage: TsdfBlockStorage,
    session: ScanSession,
) -> TsdfStreamFusionReceipt:
    """Fuse every selected observation into empty planned storage."""

    _, receipt = _fuse_stage(storage, session, None, None)
    if receipt is None:
        raise TsdfError("TSDF stream fusion did not reach its last frame")
    return receipt


def advance_tsdf_plan_streaming(
    storage: TsdfBlockStorage,
    session: ScanSession,
    progress: TsdfStreamFusionProgress | None = None,
    *,
    observations: int | None = None,
) -> TsdfStreamFusionProgress:
    """Fuse the next selected observations and say how far that got.

    ``progress`` is what an earlier stage returned for this storage, or
    ``None`` for storage that is still empty. ``observations`` limits the
    stage to that many of the plan's selected observations, counted whether
    a frame is fused or skipped for a missing input; ``None`` runs to the
    end.

    Stages add up to a single pass exactly. Each voxel still receives its
    contributions in observation order, one addition per observation, so
    where a pass is cut changes nothing about the sums.

    A stage that fails leaves the storage cleared. Its earlier state cannot
    be restored from here, and cleared storage is at least unmistakable:
    the progress that described it no longer matches, so nothing can
    continue from a half-applied frame.
    """

    advanced, _ = _fuse_stage(storage, session, progress, observations)
    return advanced


def finish_tsdf_plan_streaming(
    storage: TsdfBlockStorage,
    session: ScanSession,
    progress: TsdfStreamFusionProgress,
) -> TsdfStreamFusionReceipt:
    """The receipt of a single pass, for stages that have reached the end."""

    if not isinstance(storage, TsdfBlockStorage):
        raise TsdfError(
            "TSDF stream fusion requires allocated TsdfBlockStorage"
        )
    if not isinstance(session, ScanSession):
        raise TsdfError("TSDF stream fusion requires a loaded ScanSession")
    if not isinstance(progress, TsdfStreamFusionProgress):
        raise TsdfError(
            "TSDF stream fusion requires a TsdfStreamFusionProgress"
        )
    _validate_update_storage(storage)
    plan = storage.source_plan
    if plan.session_id != session.session_id:
        raise TsdfError(
            "TSDF block plan session_id does not match the loaded session"
        )
    _require_progress_describes(progress, storage)
    if not progress.is_complete:
        raise TsdfError(
            "TSDF stream fusion is not complete: "
            f"{progress.processed_observations} of "
            f"{progress.selected_observations} selected observations "
            "processed"
        )
    try:
        camera, _ = _validate_reconstruction_contract(session)
    except PointCloudError as error:
        raise TsdfError(str(error)) from error
    if replay_session(session).digest_sha256 != plan.replay_digest_sha256:
        raise TsdfError(
            "TSDF block plan replay digest does not match current session "
            "inputs; regenerate the plan"
        )
    if (
        progress.valid_depth_samples != plan.valid_depth_points
        or progress.invalid_depth_samples != plan.invalid_depth_samples
    ):
        raise TsdfError(
            "TSDF stream fusion depth sample counts do not match the plan"
        )
    return _receipt_for(progress, storage, camera)


def _require_progress_describes(
    progress: TsdfStreamFusionProgress,
    storage: TsdfBlockStorage,
) -> None:
    """Refuse progress that belongs to another plan or to other bytes."""

    plan = storage.source_plan
    if (
        progress.session_id != plan.session_id
        or progress.source_plan_digest_sha256 != plan.artifact_digest_sha256
        or progress.replay_digest_sha256 != plan.replay_digest_sha256
        or progress.frame_stride != plan.frame_stride
        or progress.total_observations != plan.total_observations
        or progress.selected_observations != plan.selected_observations
        or progress.block_count != storage.block_count
        or progress.voxel_slots != storage.voxel_slots
    ):
        raise TsdfError(
            "TSDF stream fusion progress does not describe this plan"
        )
    if (
        storage_payload_sha256(storage.tsdf_sums) != progress.tsdf_sums_sha256
        or storage_payload_sha256(storage.weights) != progress.weights_sha256
    ):
        raise TsdfError(
            "TSDF block storage does not hold the bytes this progress "
            "describes"
        )


def _receipt_for(
    progress: TsdfStreamFusionProgress,
    storage: TsdfBlockStorage,
    camera,
) -> TsdfStreamFusionReceipt:
    frame_bytes = camera.width * camera.height * np.dtype(np.float64).itemsize
    return TsdfStreamFusionReceipt(
        session_id=progress.session_id,
        source_plan_digest_sha256=progress.source_plan_digest_sha256,
        replay_digest_sha256=progress.replay_digest_sha256,
        frame_stride=progress.frame_stride,
        total_observations=progress.total_observations,
        selected_observations=progress.selected_observations,
        fused_observations=progress.fused_observations,
        skipped_missing_depth=progress.skipped_missing_depth,
        skipped_missing_pose=progress.skipped_missing_pose,
        block_count=progress.block_count,
        voxel_slots=progress.voxel_slots,
        status_counts=progress.status_counts,
        observed_voxel_count=int(np.count_nonzero(storage.weights)),
        maximum_weight=int(storage.weights.max()),
        valid_depth_samples=progress.valid_depth_samples,
        invalid_depth_samples=progress.invalid_depth_samples,
        peak_retained_depth_bytes=(
            frame_bytes if progress.fused_observations else 0
        ),
        tsdf_sums_sha256=progress.tsdf_sums_sha256,
        weights_sha256=progress.weights_sha256,
    )


def _fuse_stage(
    storage: TsdfBlockStorage,
    session: ScanSession,
    progress: TsdfStreamFusionProgress | None,
    observations: int | None,
) -> tuple[TsdfStreamFusionProgress, TsdfStreamFusionReceipt | None]:
    """One stage of a pass; the receipt comes with the stage that ends it."""

    if not isinstance(storage, TsdfBlockStorage):
        raise TsdfError(
            "TSDF stream fusion requires allocated TsdfBlockStorage"
        )
    if not isinstance(session, ScanSession):
        raise TsdfError("TSDF stream fusion requires a loaded ScanSession")
    if progress is not None and not isinstance(
        progress, TsdfStreamFusionProgress
    ):
        raise TsdfError(
            "TSDF stream fusion requires a TsdfStreamFusionProgress"
        )
    if observations is not None and (
        isinstance(observations, bool)
        or not isinstance(observations, int)
        or observations < 1
    ):
        raise TsdfError(
            "TSDF stream fusion stage length must be a positive integer"
        )
    _validate_update_storage(storage)
    plan = storage.source_plan
    _validate_contribution_plan(plan)
    _validate_context_plan(plan)
    if plan.session_id != session.session_id:
        raise TsdfError(
            "TSDF block plan session_id does not match the loaded session"
        )
    if plan.selected_observations > MAX_TSDF_VOXEL_WEIGHT:
        raise TsdfError(
            "TSDF stream fusion would exceed the uint32 voxel weight maximum"
        )
    try:
        camera, depth_scale_m = _validate_reconstruction_contract(session)
    except PointCloudError as error:
        raise TsdfError(str(error)) from error
    _validate_context_camera(camera)

    status_totals = np.zeros(
        len(TSDF_CONTRIBUTION_STATUS_ORDER),
        dtype=np.int64,
    )
    if progress is None:
        if (
            np.count_nonzero(storage.weights)
            or np.count_nonzero(storage.tsdf_sums)
            or bool(np.signbit(storage.tsdf_sums).any())
        ):
            raise TsdfError(
                "TSDF stream fusion requires canonical empty planned storage"
            )
        first_position = 0
        fused_observations = 0
        skipped_missing_depth = 0
        skipped_missing_pose = 0
        valid_depth_samples = 0
        invalid_depth_samples = 0
    else:
        _require_progress_describes(progress, storage)
        if progress.is_complete:
            raise TsdfError(
                "TSDF stream fusion has no observation left to fuse"
            )
        first_position = progress.processed_observations
        fused_observations = progress.fused_observations
        skipped_missing_depth = progress.skipped_missing_depth
        skipped_missing_pose = progress.skipped_missing_pose
        valid_depth_samples = progress.valid_depth_samples
        invalid_depth_samples = progress.invalid_depth_samples
        for status, count in progress.status_counts:
            status_totals[_STATUS_CODE[status]] = count

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
    selected = tuple(
        starting_replay.observations[sequence]
        for sequence in selected_sequences
    )
    _validate_plan_associations(plan, selected, camera)
    stop_position = (
        len(selected)
        if observations is None
        else min(len(selected), first_position + observations)
    )

    blocks = np.asarray(storage.block_indices, dtype=np.int64)
    # Checked for the whole plan before anything is accumulated, so an
    # unrepresentable grid is refused rather than discovered mid-fusion.
    for first_block in range(0, len(blocks), STREAM_FUSION_CHUNK_BLOCKS):
        _chunk_voxel_centres_world_m(
            blocks[first_block:first_block + STREAM_FUSION_CHUNK_BLOCKS],
            plan.voxel_size_m,
        )
    voxel_slots = storage.voxel_slots
    sums = storage.tsdf_sums.reshape(-1)
    weights = storage.weights.reshape(-1)
    block_sums = storage.tsdf_sums.reshape((len(blocks), TSDF_BLOCK_VOXELS))
    block_weights = storage.weights.reshape(
        (len(blocks), TSDF_BLOCK_VOXELS)
    )
    if not (
        np.shares_memory(sums, storage.tsdf_sums)
        and np.shares_memory(weights, storage.weights)
        and np.shares_memory(block_sums, storage.tsdf_sums)
        and np.shares_memory(block_weights, storage.weights)
    ):
        raise TsdfError(
            "TSDF stream fusion could not address storage in place"
        )

    try:
        for observation in selected[first_position:stop_position]:
            status = _classify_observation(observation)
            if status is not TsdfReplayDepthStatus.READY:
                skipped_missing_depth += int(observation.depth is None)
                skipped_missing_pose += int(observation.pose is None)
                status_totals[
                    _STATUS_CODE[_PREPARATION_SKIP_STATUS[status]]
                ] += voxel_slots
                continue

            transform = _prepared_transform(observation, status)
            if transform is None or not _is_rigid_transform(transform):
                raise TsdfError(
                    "TSDF stream fusion requires a rigid T_world_camera for "
                    f"observation {observation.sequence}"
                )
            depth_m = _decode_metric_depth(
                session,
                observation,
                camera.width,
                camera.height,
                depth_scale_m,
            )
            frame_valid = int(
                np.count_nonzero(np.isfinite(depth_m) & (depth_m > 0.0))
            )
            valid_depth_samples += frame_valid
            invalid_depth_samples += int(depth_m.size) - frame_valid

            if STREAM_FUSION_CULLS_BLOCKS:
                verdicts = _classify_blocks(
                    camera, transform, blocks, plan.voxel_size_m
                )
                for verdict, settled in (
                    (
                        BLOCK_BEHIND_CAMERA,
                        TsdfContributionStatus.CAMERA_Z_NONPOSITIVE,
                    ),
                    (
                        BLOCK_OUTSIDE_IMAGE,
                        TsdfContributionStatus.PROJECTION_OUTSIDE_IMAGE,
                    ),
                ):
                    status_totals[_STATUS_CODE[settled]] += (
                        TSDF_BLOCK_VOXELS
                        * int(np.count_nonzero(verdicts == verdict))
                    )
                visible = np.flatnonzero(verdicts == BLOCK_EVALUATE)
            else:
                visible = np.arange(len(blocks), dtype=np.int64)

            for first in range(0, len(visible), STREAM_FUSION_CHUNK_BLOCKS):
                rows = visible[first:first + STREAM_FUSION_CHUNK_BLOCKS]
                codes, sum_deltas, weight_deltas = _evaluate_ready_voxels(
                    camera,
                    transform,
                    depth_m,
                    plan.truncation_m,
                    _chunk_voxel_centres_world_m(
                        blocks[rows], plan.voxel_size_m
                    ),
                )
                # Rows are distinct, so this gathers, adds and scatters
                # each voxel exactly once: the same single addition per
                # observation as a slice would make.
                block_sums[rows] += sum_deltas.reshape(
                    (len(rows), TSDF_BLOCK_VOXELS)
                )
                block_weights[rows] += weight_deltas.reshape(
                    (len(rows), TSDF_BLOCK_VOXELS)
                )
                status_totals += np.bincount(
                    codes,
                    minlength=len(TSDF_CONTRIBUTION_STATUS_ORDER),
                )
            fused_observations += 1

        complete = stop_position == len(selected)
        if complete and (
            valid_depth_samples != plan.valid_depth_points
            or invalid_depth_samples != plan.invalid_depth_samples
        ):
            raise TsdfError(
                "TSDF stream fusion depth sample counts do not match the "
                "plan"
            )
        ending_replay = replay_session(session)
        if ending_replay.digest_sha256 != starting_replay.digest_sha256:
            raise TsdfError(
                "session inputs changed during TSDF stream fusion; rerun "
                "the command"
            )
        if not np.all(np.isfinite(sums)):
            raise TsdfError(
                "TSDF stream fusion produced a non-finite sum"
            )
        if np.any(np.abs(sums) > weights):
            raise TsdfError(
                "TSDF stream fusion sum exceeds its weight envelope"
            )

        advanced = TsdfStreamFusionProgress(
            session_id=session.session_id,
            source_plan_digest_sha256=plan.artifact_digest_sha256,
            replay_digest_sha256=plan.replay_digest_sha256,
            frame_stride=plan.frame_stride,
            total_observations=plan.total_observations,
            selected_observations=plan.selected_observations,
            processed_observations=stop_position,
            fused_observations=fused_observations,
            skipped_missing_depth=skipped_missing_depth,
            skipped_missing_pose=skipped_missing_pose,
            block_count=storage.block_count,
            voxel_slots=voxel_slots,
            status_counts=tuple(
                (status, int(status_totals[index]))
                for index, status in enumerate(TSDF_CONTRIBUTION_STATUS_ORDER)
                if status_totals[index]
            ),
            valid_depth_samples=valid_depth_samples,
            invalid_depth_samples=invalid_depth_samples,
            tsdf_sums_sha256=storage_payload_sha256(storage.tsdf_sums),
            weights_sha256=storage_payload_sha256(storage.weights),
        )
        receipt = (
            _receipt_for(advanced, storage, camera) if complete else None
        )
    except Exception as error:
        # From empty storage this restores it exactly. From an earlier
        # stage it does not, and cannot: that state is gone. Cleared
        # storage no longer matches the progress that described it, so
        # nothing can continue from a frame that was half applied.
        storage.tsdf_sums.fill(0.0)
        storage.weights.fill(0)
        if isinstance(error, (TsdfError, SessionReplayError)):
            raise
        if isinstance(error, PointCloudError):
            raise TsdfError(str(error)) from error
        raise TsdfError(
            f"cannot complete TSDF stream fusion: {error}"
        ) from error
    return advanced, receipt


def storage_payload_sha256(array: np.ndarray) -> str:
    """Digest one storage array as little-endian C-ordered bytes."""

    canonical = array.astype(array.dtype.newbyteorder("<"), copy=False)
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _decode_metric_depth(
    session: ScanSession,
    observation: object,
    width: int,
    height: int,
    depth_scale_m: float,
) -> np.ndarray:
    """Decode one frame to float64 metres, exactly as the context does."""

    depth = getattr(observation, "depth", None)
    if depth is None:
        raise AssertionError("ready observation must have depth")
    depth_path = _sample_path(session, depth.data, "depth")
    depth_m = _read_depth_array(depth_path, width, height).astype(
        np.float64
    ).reshape((height, width))
    with np.errstate(over="ignore", invalid="ignore"):
        np.multiply(depth_m, depth_scale_m, out=depth_m)
    return depth_m


def _classify_blocks(
    camera,
    transform: tuple[float, ...],
    blocks: np.ndarray,
    voxel_size_m: float,
) -> np.ndarray:
    """Blocks none of whose voxels one frame can possibly see.

    Returns one verdict per block. ``BLOCK_BEHIND_CAMERA`` and
    ``BLOCK_OUTSIDE_IMAGE`` promise that the per-voxel evaluator would
    give all 512 voxels of the block that one status; ``BLOCK_EVALUATE``
    promises nothing and is always safe.

    A block's voxel centres span, on each axis, from local index 0 to
    local index 7, computed here by the same expression the evaluator's
    centres use, so every centre lies in the box these eight corners
    span. Camera depth is linear in position, and so is each image-edge
    test once multiplied through by a positive depth: ``u < -0.5`` is
    ``fx * x + (cx + 0.5) * z < 0``. A linear function that has one sign
    at all eight corners has it throughout the box.
    """

    verdicts = np.zeros(len(blocks), dtype=np.int8)
    if not (
        0.0 < abs(camera.fx) <= _CULL_MAX_FOCAL_PIXELS
        and 0.0 < abs(camera.fy) <= _CULL_MAX_FOCAL_PIXELS
        and camera.width <= _CULL_MAX_IMAGE_PIXELS
        and camera.height <= _CULL_MAX_IMAGE_PIXELS
        and abs(camera.cx) <= _CULL_MAX_IMAGE_PIXELS
        and abs(camera.cy) <= _CULL_MAX_IMAGE_PIXELS
    ):
        # Outside these bounds the slack below is not known to cover the
        # evaluator's rounding, so no block is settled without it.
        return verdicts

    matrix = [float(component) for component in transform]
    base = blocks * TSDF_BLOCK_RESOLUTION
    corners = np.empty((len(blocks), 8, 3))
    # A grid too large to represent overflows here and is caught by
    # the finiteness test below, which settles nothing for it.
    with np.errstate(over="ignore", invalid="ignore"):
        low = (base.astype(np.float64) + 0.5) * voxel_size_m
        high = (
            (base + (TSDF_BLOCK_RESOLUTION - 1)).astype(np.float64) + 0.5
        ) * voxel_size_m
        for corner in range(8):
            for axis in range(3):
                corners[:, corner, axis] = (
                    high if (corner >> axis) & 1 else low
                )[:, axis]

        delta_x = corners[:, :, 0] - matrix[3]
        delta_y = corners[:, :, 1] - matrix[7]
        delta_z = corners[:, :, 2] - matrix[11]
        camera_x = (
            matrix[0] * delta_x + matrix[4] * delta_y + matrix[8] * delta_z
        )
        camera_y = (
            matrix[1] * delta_x + matrix[5] * delta_y + matrix[9] * delta_z
        )
        camera_z = (
            matrix[2] * delta_x + matrix[6] * delta_y + matrix[10] * delta_z
        )
        reach = np.maximum(
            np.maximum(np.abs(delta_x), np.abs(delta_y)), np.abs(delta_z)
        ).max(axis=1)
        slack = _CULL_RELATIVE_DEPTH_SLACK * (1.0 + reach)
        usable = (
            np.isfinite(reach)
            & np.isfinite(camera_x).all(axis=1)
            & np.isfinite(camera_y).all(axis=1)
            & np.isfinite(camera_z).all(axis=1)
        )
        behind = usable & (camera_z.max(axis=1) < -slack)
        in_front = usable & (camera_z.min(axis=1) > slack)

        left = (
            camera.fx * camera_x
            + (camera.cx + 0.5 + _CULL_PIXEL_SLACK) * camera_z
        )
        right = (
            camera.fx * camera_x
            + (camera.cx - (camera.width - 0.5) - _CULL_PIXEL_SLACK)
            * camera_z
        )
        top = (
            camera.fy * camera_y
            + (camera.cy + 0.5 + _CULL_PIXEL_SLACK) * camera_z
        )
        bottom = (
            camera.fy * camera_y
            + (camera.cy - (camera.height - 0.5) - _CULL_PIXEL_SLACK)
            * camera_z
        )
        outside = in_front & (
            (left.max(axis=1) < 0.0)
            | (right.min(axis=1) > 0.0)
            | (top.max(axis=1) < 0.0)
            | (bottom.min(axis=1) > 0.0)
        )
    verdicts[behind] = BLOCK_BEHIND_CAMERA
    verdicts[outside] = BLOCK_OUTSIDE_IMAGE
    return verdicts


def _chunk_voxel_centres_world_m(
    blocks: np.ndarray,
    voxel_size_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World centres of every voxel of some blocks, in storage order.

    Storage order is block row, then canonical local-flat, which is exactly
    the flattened layout of the ``(blocks, z, y, x)`` accumulator arrays. The
    arithmetic matches the per-block form term for term.

    Computed for a run of blocks at a time and thrown away, rather than
    once for the plan: three float64 per voxel is twice the size of the
    accumulators themselves, and holding it would make that, not the
    volume, the memory fusion needs.
    """

    centres = tuple(
        (
            (
                blocks[:, axis, None] * TSDF_BLOCK_RESOLUTION + local[None, :]
            ).astype(np.float64)
            + 0.5
        ).reshape(-1)
        * voxel_size_m
        for axis, local in enumerate((_LOCAL_X, _LOCAL_Y, _LOCAL_Z))
    )
    if not all(bool(np.all(np.isfinite(axis))) for axis in centres):
        raise TsdfError(
            "TSDF voxel center must remain finite in world coordinates"
        )
    return centres  # type: ignore[return-value]
