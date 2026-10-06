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

# Voxels evaluated per NumPy pass. Small enough that the dozen or so
# temporaries of one pass stay cache-resident, which is worth more than the
# Python overhead it costs; the fused bytes do not depend on it.
STREAM_FUSION_CHUNK_VOXELS = 256 * TSDF_BLOCK_VOXELS


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


def fuse_tsdf_plan_streaming(
    storage: TsdfBlockStorage,
    session: ScanSession,
) -> TsdfStreamFusionReceipt:
    """Fuse every selected observation into empty planned storage."""

    if not isinstance(storage, TsdfBlockStorage):
        raise TsdfError(
            "TSDF stream fusion requires allocated TsdfBlockStorage"
        )
    if not isinstance(session, ScanSession):
        raise TsdfError("TSDF stream fusion requires a loaded ScanSession")
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

    if (
        np.count_nonzero(storage.weights)
        or np.count_nonzero(storage.tsdf_sums)
        or bool(np.signbit(storage.tsdf_sums).any())
    ):
        raise TsdfError(
            "TSDF stream fusion requires canonical empty planned storage"
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
    selected = tuple(
        starting_replay.observations[sequence]
        for sequence in selected_sequences
    )
    _validate_plan_associations(plan, selected, camera)

    world_xyz_m = _plan_voxel_centres_world_m(
        storage.block_indices,
        plan.voxel_size_m,
    )
    voxel_slots = storage.voxel_slots
    sums = storage.tsdf_sums.reshape(-1)
    weights = storage.weights.reshape(-1)
    if not (
        np.shares_memory(sums, storage.tsdf_sums)
        and np.shares_memory(weights, storage.weights)
    ):
        raise TsdfError(
            "TSDF stream fusion could not address storage in place"
        )

    status_totals = np.zeros(
        len(TSDF_CONTRIBUTION_STATUS_ORDER),
        dtype=np.int64,
    )
    fused_observations = 0
    skipped_missing_depth = 0
    skipped_missing_pose = 0
    valid_depth_samples = 0
    invalid_depth_samples = 0
    frame_bytes = camera.width * camera.height * np.dtype(np.float64).itemsize

    try:
        for observation in selected:
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

            for start in range(0, voxel_slots, STREAM_FUSION_CHUNK_VOXELS):
                stop = min(start + STREAM_FUSION_CHUNK_VOXELS, voxel_slots)
                codes, sum_deltas, weight_deltas = _evaluate_ready_voxels(
                    camera,
                    transform,
                    depth_m,
                    plan.truncation_m,
                    (
                        world_xyz_m[0][start:stop],
                        world_xyz_m[1][start:stop],
                        world_xyz_m[2][start:stop],
                    ),
                )
                sums[start:stop] += sum_deltas
                weights[start:stop] += weight_deltas
                status_totals += np.bincount(
                    codes,
                    minlength=len(TSDF_CONTRIBUTION_STATUS_ORDER),
                )
            fused_observations += 1

        if (
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

        receipt = TsdfStreamFusionReceipt(
            session_id=session.session_id,
            source_plan_digest_sha256=plan.artifact_digest_sha256,
            replay_digest_sha256=plan.replay_digest_sha256,
            frame_stride=plan.frame_stride,
            total_observations=plan.total_observations,
            selected_observations=plan.selected_observations,
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
            observed_voxel_count=int(np.count_nonzero(weights)),
            maximum_weight=int(weights.max()),
            valid_depth_samples=valid_depth_samples,
            invalid_depth_samples=invalid_depth_samples,
            peak_retained_depth_bytes=(
                frame_bytes if fused_observations else 0
            ),
            tsdf_sums_sha256=storage_payload_sha256(storage.tsdf_sums),
            weights_sha256=storage_payload_sha256(storage.weights),
        )
    except Exception as error:
        # Storage was required to be canonically empty, so restoring it is
        # exact: there is no earlier state a partial pass could have lost.
        storage.tsdf_sums.fill(0.0)
        storage.weights.fill(0)
        if isinstance(error, (TsdfError, SessionReplayError)):
            raise
        if isinstance(error, PointCloudError):
            raise TsdfError(str(error)) from error
        raise TsdfError(
            f"cannot complete TSDF stream fusion: {error}"
        ) from error
    return receipt


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


def _plan_voxel_centres_world_m(
    block_indices: tuple[tuple[int, int, int], ...],
    voxel_size_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World centres of every planned voxel, in storage order.

    Storage order is block row, then canonical local-flat, which is exactly
    the flattened layout of the ``(blocks, z, y, x)`` accumulator arrays. The
    arithmetic matches the per-block form term for term.
    """

    blocks = np.asarray(block_indices, dtype=np.int64)
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
