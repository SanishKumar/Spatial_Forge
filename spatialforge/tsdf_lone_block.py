"""Fuse one block of a volume by itself.

A fused volume is tens of thousands of blocks, and the only way to find out
whether one of them is what the scan gives has been to fuse all of them
again. That is not necessary. A voxel's accumulators are the sum, in frame
order, of one contribution from each frame, and a contribution reads one
pixel of that frame's depth. Nothing in it depends on another voxel, or on
which other blocks happened to be planned.

So a block is a function of the scan and three numbers: the voxel size,
the truncation and the frame stride. This module evaluates that function
for a block on its own. No plan is consulted and no storage is allocated,
and the block need not belong to any volume.

Most frames can be ruled out before their depth is opened. Fusion already
settles a whole block as behind the camera or outside the image from its
eight corners and the frame's pose, and promises that every voxel of such a
block would have been given that verdict one at a time. The same verdicts
are used here, so a block is fused from the frames that are left, and a
frame that can reach none of the blocks asked for is never decoded.

Checking a volume is then a comparison. Fuse some of its blocks alone and
require the bytes the volume holds. One block that differs shows that the
volume is not the fusion of that scan, without fusing the rest.

This says a volume follows from a scan. It does not say the scan is true,
that its poses are right, or that the surface is accurate.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, NamedTuple, Sequence

import numpy as np

from .errors import PointCloudError, SessionReplayError, TsdfError
from .model import ScanSession
from .point_cloud import _validate_reconstruction_contract
from .replay import replay_session
from .tsdf import _validate_positive_integer, _validate_positive_number
from .tsdf_block_contributions import _evaluate_ready_voxels
from .tsdf_block_plan import (
    MAX_BLOCK_INDEX,
    MIN_BLOCK_INDEX,
    TSDF_BLOCK_RESOLUTION,
    _ordered_blocks,
)
from .tsdf_block_storage import (
    TSDF_BLOCK_VOXELS,
    TSDF_SUM_DTYPE,
    TSDF_WEIGHT_DTYPE,
)
from .tsdf_block_volume import TsdfBlockVolume
from .tsdf_replay_depth_context import (
    TsdfReplayDepthStatus,
    _classify_observation,
    _is_rigid_transform,
    _prepared_transform,
    _validate_context_camera,
)
from .tsdf_stream_fusion import (
    BLOCK_BEHIND_CAMERA,
    BLOCK_EVALUATE,
    BLOCK_OUTSIDE_IMAGE,
    STREAM_FUSION_CHUNK_BLOCKS,
    _chunk_voxel_centres_world_m,
    _classify_blocks,
    _decode_metric_depth,
    _is_sha256,
    storage_payload_sha256,
)
from .tsdf_voxel_update import MAX_TSDF_VOXEL_WEIGHT

# Blocks fused alone in one call. Each is 6,144 bytes of accumulators, so
# this is a limit on patience, not on memory: past it, fuse the volume.
MAX_TSDF_LONE_BLOCKS = 4096

_BlockIndex = tuple[int, int, int]
_BLOCK_SHAPE = (
    TSDF_BLOCK_RESOLUTION,
    TSDF_BLOCK_RESOLUTION,
    TSDF_BLOCK_RESOLUTION,
)
_SAMPLE_DOMAIN = "spatialforge.tsdf-block-sample"


class TsdfBlockCheckStatus(StrEnum):
    """What a volume holds where a block was fused alone."""

    IDENTICAL = "identical"
    DIFFERENT = "different"
    ABSENT = "absent"


@dataclass(frozen=True, slots=True, eq=False)
class TsdfLoneBlock:
    """One block's accumulators, fused from a scan with no other block.

    ``tsdf_sums`` and ``weights`` are laid out as a volume lays out a
    block: ``[z, y, x]``, immutable. The counts say what it took. Of the
    frames that had both depth and a pose, some were ruled out from the
    pose alone, the rest were evaluated, and some of those gave the block
    at least one contribution.
    """

    session_id: str
    replay_digest_sha256: str
    voxel_size_m: float
    truncation_m: float
    frame_stride: int
    block_index: _BlockIndex
    selected_observations: int
    ready_observations: int
    ruled_out_behind_camera: int
    ruled_out_outside_image: int
    evaluated_observations: int
    contributing_observations: int
    tsdf_sums: np.ndarray
    weights: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str) or not self.session_id:
            raise TsdfError("TSDF lone block session_id must be nonempty")
        if not _is_sha256(self.replay_digest_sha256):
            raise TsdfError("TSDF lone block replay digest is invalid")
        _validate_block_index(self.block_index)
        for array, dtype, label in (
            (self.tsdf_sums, TSDF_SUM_DTYPE, "sums"),
            (self.weights, TSDF_WEIGHT_DTYPE, "weights"),
        ):
            if (
                type(array) is not np.ndarray
                or array.shape != _BLOCK_SHAPE
                or array.dtype != dtype
                or array.flags.writeable
            ):
                raise TsdfError(
                    f"TSDF lone block {label} must be an immutable "
                    f"{dtype.name} array of shape {_BLOCK_SHAPE}"
                )
        counts = (
            self.selected_observations,
            self.ready_observations,
            self.ruled_out_behind_camera,
            self.ruled_out_outside_image,
            self.evaluated_observations,
            self.contributing_observations,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in counts
        ):
            raise TsdfError(
                "TSDF lone block frame counts must be nonnegative integers"
            )
        if (
            self.selected_observations < 1
            or self.ready_observations > self.selected_observations
            or self.ruled_out_behind_camera
            + self.ruled_out_outside_image
            + self.evaluated_observations
            != self.ready_observations
            or self.contributing_observations > self.evaluated_observations
        ):
            raise TsdfError(
                "TSDF lone block frame counts do not account for every "
                "ready frame"
            )
        # What the accumulators themselves can confirm. A voxel is given
        # at most one contribution a frame, and only by a frame that
        # gave the block something.
        heaviest = int(self.weights.max())
        if heaviest > self.contributing_observations or (
            heaviest == 0
        ) != (self.contributing_observations == 0):
            raise TsdfError(
                "TSDF lone block weights do not match its contributing "
                "frames"
            )
        if not bool(np.all(np.isfinite(self.tsdf_sums))):
            raise TsdfError("TSDF lone block contains a non-finite sum")
        if bool(np.any(np.abs(self.tsdf_sums) > self.weights)):
            raise TsdfError(
                "TSDF lone block sum exceeds its weight envelope"
            )
        if bool(np.any(np.signbit(self.tsdf_sums[self.weights == 0]))):
            raise TsdfError(
                "TSDF lone block unobserved voxel is not canonical zero"
            )

    @property
    def contributions_applied(self) -> int:
        return int(self.weights.sum(dtype=np.uint64))

    @property
    def observed_voxel_count(self) -> int:
        return int(np.count_nonzero(self.weights))

    @property
    def maximum_weight(self) -> int:
        return int(self.weights.max())

    @property
    def ruled_out_observations(self) -> int:
        return self.ruled_out_behind_camera + self.ruled_out_outside_image

    @property
    def tsdf_sums_sha256(self) -> str:
        return storage_payload_sha256(self.tsdf_sums)

    @property
    def weights_sha256(self) -> str:
        return storage_payload_sha256(self.weights)


@dataclass(frozen=True, slots=True, eq=False)
class TsdfVolumeBlockCheck:
    """One block of a volume, held against the same block fused alone.

    ``differing_voxels`` counts voxels whose sum or weight is not, bit for
    bit, what the scan gives. A block the volume does not hold is absent,
    which is unknown space and not a difference; ``lone`` still says what
    the scan would have put there.
    """

    volume_digest_sha256: str
    status: TsdfBlockCheckStatus
    differing_voxels: int
    lone: TsdfLoneBlock

    def __post_init__(self) -> None:
        if not _is_sha256(self.volume_digest_sha256):
            raise TsdfError("TSDF block check volume digest is invalid")
        if not isinstance(self.status, TsdfBlockCheckStatus):
            raise TsdfError("TSDF block check status is invalid")
        if not isinstance(self.lone, TsdfLoneBlock):
            raise TsdfError("TSDF block check requires a TsdfLoneBlock")
        if (
            isinstance(self.differing_voxels, bool)
            or not isinstance(self.differing_voxels, int)
            or not 0 <= self.differing_voxels <= TSDF_BLOCK_VOXELS
            or (self.status is TsdfBlockCheckStatus.DIFFERENT)
            != (self.differing_voxels > 0)
        ):
            raise TsdfError(
                "TSDF block check differing voxels do not match its status"
            )

    @property
    def block_index(self) -> _BlockIndex:
        return self.lone.block_index


def fuse_tsdf_block_alone(
    session: ScanSession,
    block_index: _BlockIndex,
    *,
    voxel_size_m: float,
    truncation_m: float,
    frame_stride: int = 1,
    rule_out_by_pose: bool = True,
) -> TsdfLoneBlock:
    """What fusing ``session`` puts in one block, and nothing else."""

    return fuse_tsdf_blocks_alone(
        session,
        (block_index,),
        voxel_size_m=voxel_size_m,
        truncation_m=truncation_m,
        frame_stride=frame_stride,
        rule_out_by_pose=rule_out_by_pose,
    )[0]


def fuse_tsdf_blocks_alone(
    session: ScanSession,
    block_indices: Sequence[_BlockIndex],
    *,
    voxel_size_m: float,
    truncation_m: float,
    frame_stride: int = 1,
    rule_out_by_pose: bool = True,
) -> tuple[TsdfLoneBlock, ...]:
    """Each of several blocks as it would be fused alone, in one pass.

    The blocks share the reading of the frames and nothing else: what one
    of them comes to does not depend on which others were asked for. They
    are returned in the order given.

    ``rule_out_by_pose=False`` evaluates every ready frame against every
    block. The accumulators are the same; it exists so that they can be
    shown to be.
    """

    return _fuse_blocks_alone(
        session,
        block_indices,
        voxel_size_m=voxel_size_m,
        truncation_m=truncation_m,
        frame_stride=frame_stride,
        rule_out_by_pose=rule_out_by_pose,
        expected_replay_digest=None,
        expected_total_observations=None,
    )


def check_tsdf_volume_blocks(
    volume: TsdfBlockVolume,
    session: ScanSession,
    block_indices: Sequence[_BlockIndex],
    *,
    rule_out_by_pose: bool = True,
) -> tuple[TsdfVolumeBlockCheck, ...]:
    """Hold blocks of a volume against the same blocks fused alone.

    The volume's own voxel size, truncation and stride are used, and the
    session must be the scan the volume records, by replay digest. The
    volume is read and never written.
    """

    _validate_check_inputs(volume, session)
    return _check_against_volume(
        volume,
        _fuse_blocks_alone(
            session,
            block_indices,
            voxel_size_m=volume.voxel_size_m,
            truncation_m=volume.truncation_m,
            frame_stride=volume.frame_stride,
            rule_out_by_pose=rule_out_by_pose,
            expected_replay_digest=volume.replay_digest_sha256,
            expected_total_observations=volume.total_observations,
        ),
    )


def _validate_check_inputs(volume: object, session: object) -> None:
    if not isinstance(volume, TsdfBlockVolume):
        raise TsdfError(
            "TSDF block check requires a loaded TsdfBlockVolume"
        )
    if not isinstance(session, ScanSession):
        raise TsdfError("TSDF block check requires a loaded ScanSession")
    if volume.block_resolution != TSDF_BLOCK_RESOLUTION:
        raise TsdfError(
            "TSDF block check requires block resolution "
            f"{TSDF_BLOCK_RESOLUTION}"
        )
    if volume.session_id != session.session_id:
        raise TsdfError(
            f"volume was fused from session {volume.session_id!r}, not "
            f"{session.session_id!r}"
        )


def _check_against_volume(
    volume: TsdfBlockVolume,
    lone_blocks: Sequence[TsdfLoneBlock],
) -> tuple[TsdfVolumeBlockCheck, ...]:
    """Hold blocks fused alone, by whatever path, against a volume."""

    rows = {block: row for row, block in enumerate(volume.block_indices)}
    checks = []
    for lone in lone_blocks:
        row = rows.get(lone.block_index)
        if row is None:
            status = TsdfBlockCheckStatus.ABSENT
            differing = 0
        else:
            # Compared as bits: a negative zero or a different NaN is a
            # different volume, whatever arithmetic would say of it.
            differing = int(
                np.count_nonzero(
                    (
                        volume.tsdf_sums[row].view(np.uint64)
                        != lone.tsdf_sums.view(np.uint64)
                    )
                    | (volume.weights[row] != lone.weights)
                )
            )
            status = (
                TsdfBlockCheckStatus.DIFFERENT
                if differing
                else TsdfBlockCheckStatus.IDENTICAL
            )
        checks.append(
            TsdfVolumeBlockCheck(
                volume_digest_sha256=volume.artifact_digest_sha256,
                status=status,
                differing_voxels=differing,
                lone=lone,
            )
        )
    return tuple(checks)


def sample_tsdf_volume_blocks(
    volume: TsdfBlockVolume,
    count: int,
    *,
    seed: int = 0,
) -> tuple[_BlockIndex, ...]:
    """``count`` distinct blocks of a volume, chosen by hash.

    A block's place in the draw is the SHA-256 of the seed and its index,
    so the same seed picks the same blocks on every machine and under
    every interpreter, and a block's chance does not depend on where the
    volume happens to list it. Returned in canonical block order.
    """

    if not isinstance(volume, TsdfBlockVolume):
        raise TsdfError(
            "TSDF block sampling requires a loaded TsdfBlockVolume"
        )
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise TsdfError("TSDF block sample count must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise TsdfError("TSDF block sample seed must be a nonnegative integer")
    if count > volume.block_count:
        raise TsdfError(
            f"cannot sample {count} blocks of a volume that holds "
            f"{volume.block_count}"
        )
    if count > MAX_TSDF_LONE_BLOCKS:
        raise TsdfError(
            f"cannot sample {count} blocks; at most {MAX_TSDF_LONE_BLOCKS} "
            "are fused alone in one call"
        )

    def place(block: _BlockIndex) -> bytes:
        return hashlib.sha256(
            f"{_SAMPLE_DOMAIN}/{seed}/{block[0]},{block[1]},{block[2]}"
            .encode("ascii")
        ).digest()

    return _ordered_blocks(sorted(volume.block_indices, key=place)[:count])


def _validate_block_index(block_index: object) -> _BlockIndex:
    if (
        not isinstance(block_index, tuple)
        or len(block_index) != 3
        or any(
            isinstance(component, bool) or not isinstance(component, int)
            for component in block_index
        )
        or any(
            not MIN_BLOCK_INDEX <= component <= MAX_BLOCK_INDEX
            for component in block_index
        )
    ):
        raise TsdfError(
            "TSDF block index must be three integers within the planning "
            "range"
        )
    return block_index  # type: ignore[return-value]


class _LoneFusionSetup(NamedTuple):
    """What fusing blocks alone needs before it opens a frame."""

    requested: tuple[_BlockIndex, ...]
    blocks: np.ndarray
    voxel_size_m: float
    truncation_m: float
    camera: Any
    depth_scale_m: float
    replay: Any
    selected: tuple[Any, ...]


def _prepare_lone_fusion(
    session: ScanSession,
    block_indices: Sequence[_BlockIndex],
    *,
    voxel_size_m: float,
    truncation_m: float,
    frame_stride: int,
    expected_replay_digest: str | None,
    expected_total_observations: int | None,
) -> _LoneFusionSetup:
    """Validate the request and select the frames, as fusion selects them."""

    if not isinstance(session, ScanSession):
        raise TsdfError("TSDF lone block fusion requires a loaded ScanSession")
    if isinstance(block_indices, (str, bytes)) or not isinstance(
        block_indices, Sequence
    ):
        raise TsdfError(
            "TSDF lone block fusion requires a sequence of block indices"
        )
    requested = tuple(_validate_block_index(index) for index in block_indices)
    if not requested:
        raise TsdfError("TSDF lone block fusion requires at least one block")
    if len(set(requested)) != len(requested):
        raise TsdfError("TSDF lone block fusion was given a block twice")
    if len(requested) > MAX_TSDF_LONE_BLOCKS:
        raise TsdfError(
            f"TSDF lone block fusion was given {len(requested)} blocks; "
            f"the maximum is {MAX_TSDF_LONE_BLOCKS}. Fuse the volume"
        )
    voxel_size = _validate_positive_number(voxel_size_m, "voxel_size_m")
    truncation = _validate_positive_number(truncation_m, "truncation_m")
    if truncation < voxel_size:
        raise TsdfError(
            "truncation_m must be greater than or equal to voxel_size_m"
        )
    _validate_positive_integer(frame_stride, "frame_stride")
    try:
        camera, depth_scale_m = _validate_reconstruction_contract(session)
    except PointCloudError as error:
        raise TsdfError(str(error)) from error
    _validate_context_camera(camera)

    blocks = np.asarray(requested, dtype=np.int64).reshape((-1, 3))
    # Before any frame is opened, so an unrepresentable grid is refused
    # rather than discovered at the first frame that sees it.
    for first in range(0, len(blocks), STREAM_FUSION_CHUNK_BLOCKS):
        _chunk_voxel_centres_world_m(
            blocks[first:first + STREAM_FUSION_CHUNK_BLOCKS], voxel_size
        )

    starting_replay = replay_session(session)
    if (
        expected_replay_digest is not None
        and starting_replay.digest_sha256 != expected_replay_digest
    ):
        raise TsdfError(
            "the session's replay digest does not match the one recorded "
            "in the volume; this is not the scan the volume was fused from"
        )
    total_observations = len(starting_replay.observations)
    if (
        expected_total_observations is not None
        and total_observations != expected_total_observations
    ):
        raise TsdfError(
            "the session's observation count does not match the one "
            "recorded in the volume"
        )
    selected = tuple(
        starting_replay.observations[sequence]
        for sequence in range(0, total_observations, frame_stride)
    )
    if not selected:
        raise TsdfError("TSDF lone block fusion found no frame to select")
    if len(selected) > MAX_TSDF_VOXEL_WEIGHT:
        raise TsdfError(
            "TSDF lone block fusion would exceed the uint32 voxel weight "
            "maximum"
        )
    return _LoneFusionSetup(
        requested=requested,
        blocks=blocks,
        voxel_size_m=voxel_size,
        truncation_m=truncation,
        camera=camera,
        depth_scale_m=depth_scale_m,
        replay=starting_replay,
        selected=selected,
    )


def _fuse_blocks_alone(
    session: ScanSession,
    block_indices: Sequence[_BlockIndex],
    *,
    voxel_size_m: float,
    truncation_m: float,
    frame_stride: int,
    rule_out_by_pose: bool,
    expected_replay_digest: str | None,
    expected_total_observations: int | None,
) -> tuple[TsdfLoneBlock, ...]:
    if not isinstance(rule_out_by_pose, bool):
        raise TsdfError("rule_out_by_pose must be true or false")
    setup = _prepare_lone_fusion(
        session,
        block_indices,
        voxel_size_m=voxel_size_m,
        truncation_m=truncation_m,
        frame_stride=frame_stride,
        expected_replay_digest=expected_replay_digest,
        expected_total_observations=expected_total_observations,
    )
    requested = setup.requested
    blocks = setup.blocks
    voxel_size = setup.voxel_size_m
    truncation = setup.truncation_m
    camera = setup.camera
    depth_scale_m = setup.depth_scale_m
    starting_replay = setup.replay
    selected = setup.selected

    sums = np.zeros((len(blocks), TSDF_BLOCK_VOXELS), dtype=TSDF_SUM_DTYPE)
    weights = np.zeros(
        (len(blocks), TSDF_BLOCK_VOXELS), dtype=TSDF_WEIGHT_DTYPE
    )
    behind = np.zeros(len(blocks), dtype=np.int64)
    outside = np.zeros(len(blocks), dtype=np.int64)
    evaluated = np.zeros(len(blocks), dtype=np.int64)
    contributing = np.zeros(len(blocks), dtype=np.int64)
    ready = 0
    every_block = np.arange(len(blocks), dtype=np.int64)

    try:
        for observation in selected:
            status = _classify_observation(observation)
            if status is not TsdfReplayDepthStatus.READY:
                continue
            transform = _prepared_transform(observation, status)
            if transform is None or not _is_rigid_transform(transform):
                raise TsdfError(
                    "TSDF lone block fusion requires a rigid "
                    f"T_world_camera for observation {observation.sequence}"
                )
            ready += 1
            if rule_out_by_pose:
                verdicts = _classify_blocks(
                    camera, transform, blocks, voxel_size
                )
                behind += verdicts == BLOCK_BEHIND_CAMERA
                outside += verdicts == BLOCK_OUTSIDE_IMAGE
                visible = np.flatnonzero(verdicts == BLOCK_EVALUATE)
            else:
                visible = every_block
            if not len(visible):
                # No block asked for can be reached from this pose, so
                # its depth is not opened.
                continue
            evaluated[visible] += 1
            depth_m = _decode_metric_depth(
                session,
                observation,
                camera.width,
                camera.height,
                depth_scale_m,
            )
            for first in range(0, len(visible), STREAM_FUSION_CHUNK_BLOCKS):
                rows = visible[first:first + STREAM_FUSION_CHUNK_BLOCKS]
                _, sum_deltas, weight_deltas = _evaluate_ready_voxels(
                    camera,
                    transform,
                    depth_m,
                    truncation,
                    _chunk_voxel_centres_world_m(blocks[rows], voxel_size),
                )
                given = weight_deltas.reshape(
                    (len(rows), TSDF_BLOCK_VOXELS)
                )
                # Rows are distinct, so each voxel is gathered, added
                # to and scattered once: the single addition a frame
                # that fusion makes.
                sums[rows] += sum_deltas.reshape(
                    (len(rows), TSDF_BLOCK_VOXELS)
                )
                weights[rows] += given
                contributing[rows] += given.any(axis=1)

        if replay_session(session).digest_sha256 != (
            starting_replay.digest_sha256
        ):
            raise TsdfError(
                "session inputs changed during TSDF lone block fusion; "
                "rerun the command"
            )
    except (TsdfError, SessionReplayError):
        raise
    except PointCloudError as error:
        raise TsdfError(str(error)) from error
    except Exception as error:
        raise TsdfError(
            f"cannot complete TSDF lone block fusion: {error}"
        ) from error

    return tuple(
        _lone_block(
            session_id=session.session_id,
            replay_digest_sha256=starting_replay.digest_sha256,
            voxel_size_m=voxel_size,
            truncation_m=truncation,
            frame_stride=frame_stride,
            block_index=block,
            selected_observations=len(selected),
            ready_observations=ready,
            ruled_out_behind_camera=int(behind[row]),
            ruled_out_outside_image=int(outside[row]),
            evaluated_observations=int(evaluated[row]),
            contributing_observations=int(contributing[row]),
            flat_sums=sums[row],
            flat_weights=weights[row],
        )
        for row, block in enumerate(requested)
    )


def _lone_block(
    *,
    flat_sums: np.ndarray,
    flat_weights: np.ndarray,
    **described: Any,
) -> TsdfLoneBlock:
    """A block's 512 accumulators as an immutable ``[z, y, x]`` receipt."""

    block_sums = flat_sums.reshape(_BLOCK_SHAPE).copy()
    block_weights = flat_weights.reshape(_BLOCK_SHAPE).copy()
    block_sums.setflags(write=False)
    block_weights.setflags(write=False)
    return TsdfLoneBlock(
        tsdf_sums=block_sums,
        weights=block_weights,
        **described,
    )
