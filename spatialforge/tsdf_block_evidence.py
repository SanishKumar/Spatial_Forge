"""Fuse one block from the pieces of the scan it reads, and no more.

Fusing a block alone still opens every frame that can reach it, and a frame
is 300,000 depth samples of which the block reads a few hundred: each of
its 512 voxels reads one pixel. So what a block is a function of is small.
It is the camera, the pose of every selected frame, and a scatter of depth
pixels.

This module gathers exactly that and calls it the block's evidence. Depth
is held in tiles, squares of pixels cut on a fixed grid, because a tile is
something a hash can later be given to; a tile is kept if the block reads
any pixel of it. Then it fuses the block from the evidence alone. That
second half touches no session and opens no file.

Which pixels a block reads is not worked out twice. The voxel evaluator is
run with a reader that notes the pixels it is asked for, and the same
evaluator is run again over the tiles with a reader that refuses a pixel it
does not hold. A block fused from evidence is therefore either the block
fused alone, to the byte, or an error naming the pixel that was missing. It
is never a block fused from a guess at what the missing depth was.

The poses of all selected frames are part of the evidence, including the
frames that contribute nothing, because that they contribute nothing is
decided from their pose and has to be decidable by whoever checks. Nothing
here says the evidence is the scan's. That is what a commitment is for.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .errors import PointCloudError, SessionReplayError, TsdfError
from .model import ScanSession
from .replay import replay_session
from .tsdf_block_contributions import _evaluate_ready_voxels_reading
from .tsdf_block_storage import (
    TSDF_BLOCK_VOXELS,
    TSDF_SUM_DTYPE,
    TSDF_WEIGHT_DTYPE,
)
from .tsdf_block_volume import TsdfBlockVolume
from .tsdf_lone_block import (
    TsdfLoneBlock,
    TsdfVolumeBlockCheck,
    _check_against_volume,
    _lone_block,
    _prepare_lone_fusion,
    _validate_block_index,
    _validate_check_inputs,
)
from .tsdf_replay_depth_context import (
    TsdfReplayDepthStatus,
    _classify_observation,
    _is_rigid_transform,
    _prepared_transform,
)
from .tsdf_stream_fusion import (
    BLOCK_BEHIND_CAMERA,
    BLOCK_EVALUATE,
    BLOCK_OUTSIDE_IMAGE,
    _chunk_voxel_centres_world_m,
    _classify_blocks,
    _decode_raw_depth,
    _is_sha256,
    _metric_depth,
)
from .tsdf_voxel_update import MAX_TSDF_VOXEL_WEIGHT

# Side of a depth tile, in pixels, unless asked otherwise.
TSDF_DEPTH_TILE_PIXELS = 16
MAX_TSDF_DEPTH_TILE_PIXELS = 1024
# Depth is held as the scan holds it: unsigned 16-bit samples.
TSDF_DEPTH_SAMPLE_DTYPE = np.dtype(np.uint16)

_BlockIndex = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class TsdfPinholeCamera:
    """The six numbers of the camera that fusion reads."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in (self.width, self.height)
        ):
            raise TsdfError("TSDF evidence camera dimensions are invalid")
        if (
            not _finite(self.fx)
            or self.fx <= 0
            or not _finite(self.fy)
            or self.fy <= 0
            or not _finite(self.cx)
            or not 0 <= self.cx < self.width
            or not _finite(self.cy)
            or not 0 <= self.cy < self.height
        ):
            raise TsdfError("TSDF evidence camera intrinsics are invalid")


@dataclass(frozen=True, slots=True, eq=False)
class TsdfDepthTiles:
    """Some tiles of one frame's depth, as raw samples.

    The image is cut into squares of ``tile_pixels`` from its top left
    corner; tiles on the right and bottom edges are as much smaller as the
    image requires. ``tiles`` holds ``(tile_row, tile_column, samples)`` in
    increasing order, each ``samples`` an immutable uint16 array of exactly
    that tile's shape.
    """

    width: int
    height: int
    tile_pixels: int
    tiles: tuple[tuple[int, int, np.ndarray], ...]

    def __post_init__(self) -> None:
        _validate_tile_grid(self.width, self.height, self.tile_pixels)
        if not isinstance(self.tiles, tuple) or not self.tiles:
            raise TsdfError("TSDF depth tiles must hold at least one tile")
        previous = None
        for entry in self.tiles:
            if not isinstance(entry, tuple) or len(entry) != 3:
                raise TsdfError(
                    "a TSDF depth tile is a row, a column and its samples"
                )
            tile_row, tile_column, samples = entry
            if any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in (tile_row, tile_column)
            ) or not (
                0 <= tile_row < self.tile_rows
                and 0 <= tile_column < self.tile_columns
            ):
                raise TsdfError("TSDF depth tile lies outside the image")
            if previous is not None and (tile_row, tile_column) <= previous:
                raise TsdfError(
                    "TSDF depth tiles must be in increasing order"
                )
            previous = (tile_row, tile_column)
            side = self.tile_pixels
            shape = (
                min(side, self.height - tile_row * side),
                min(side, self.width - tile_column * side),
            )
            if (
                type(samples) is not np.ndarray
                or samples.dtype != TSDF_DEPTH_SAMPLE_DTYPE
                or samples.shape != shape
                or samples.flags.writeable
            ):
                raise TsdfError(
                    "TSDF depth tile samples must be an immutable uint16 "
                    f"array of shape {shape}"
                )

    @property
    def tile_rows(self) -> int:
        return -(-self.height // self.tile_pixels)

    @property
    def tile_columns(self) -> int:
        return -(-self.width // self.tile_pixels)

    @property
    def tile_count(self) -> int:
        return len(self.tiles)

    @property
    def sample_count(self) -> int:
        return sum(samples.size for _, _, samples in self.tiles)

    @property
    def byte_count(self) -> int:
        return self.sample_count * TSDF_DEPTH_SAMPLE_DTYPE.itemsize

    def read_depth_m(
        self,
        row: np.ndarray,
        column: np.ndarray,
        alive: np.ndarray,
        depth_scale_m: float,
        *,
        sequence: int,
    ) -> np.ndarray:
        """Metric depth at the pixels that are read, or a refusal.

        The gather the voxel evaluator asks for. Entries where ``alive``
        is not set are returned as zero and stand for nothing.
        """

        depth_m = np.zeros(len(row), dtype=np.float64)
        wanted = np.flatnonzero(alive)
        if not len(wanted):
            return depth_m
        rows = row[wanted]
        columns = column[wanted]
        tile_row = rows // self.tile_pixels
        tile_column = columns // self.tile_pixels
        key = tile_row * self.tile_columns + tile_column
        held = {
            held_row * self.tile_columns + held_column: samples
            for held_row, held_column, samples in self.tiles
        }
        raw = np.zeros(len(wanted), dtype=TSDF_DEPTH_SAMPLE_DTYPE)
        for tile in np.unique(key):
            chosen = key == tile
            samples = held.get(int(tile))
            if samples is None:
                first = int(np.flatnonzero(chosen)[0])
                raise TsdfError(_missing_pixel(sequence, rows, columns, first))
            raw[chosen] = samples[
                rows[chosen] - tile_row[chosen] * self.tile_pixels,
                columns[chosen] - tile_column[chosen] * self.tile_pixels,
            ]
        depth_m[wanted] = _metric_depth(raw, depth_scale_m)
        return depth_m


@dataclass(frozen=True, slots=True, eq=False)
class TsdfBlockFrameEvidence:
    """One selected frame, as much of it as a block is a function of.

    Always its place in the scan and, if it has one, its pose. Whether it
    has depth at all. And the tiles of that depth the block reads, if it
    reads any: a frame ruled out by its pose, or one in which no voxel of
    the block lands on the image, has none.
    """

    sequence: int
    transform: tuple[float, ...] | None
    has_depth: bool
    tiles: TsdfDepthTiles | None

    def __post_init__(self) -> None:
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence < 0
        ):
            raise TsdfError("TSDF frame evidence sequence is invalid")
        if self.transform is not None and (
            not isinstance(self.transform, tuple)
            or len(self.transform) != 16
            or any(not _finite(value) for value in self.transform)
        ):
            raise TsdfError(
                "TSDF frame evidence pose must be 16 finite numbers or "
                "absent"
            )
        if not isinstance(self.has_depth, bool):
            raise TsdfError("TSDF frame evidence has_depth must be a flag")
        if self.tiles is not None:
            if not isinstance(self.tiles, TsdfDepthTiles):
                raise TsdfError(
                    "TSDF frame evidence tiles must be TsdfDepthTiles"
                )
            if not self.is_ready:
                raise TsdfError(
                    "a frame with no depth or no pose is not fused, and "
                    "cannot carry depth tiles"
                )

    @property
    def is_ready(self) -> bool:
        return self.has_depth and self.transform is not None


@dataclass(frozen=True, slots=True, eq=False)
class TsdfBlockEvidence:
    """Everything one block is a function of.

    ``frames`` is every selected frame in the order fusion takes them:
    sequences ``0, stride, 2 * stride`` and so on to the end of the scan.
    """

    session_id: str
    replay_digest_sha256: str
    camera: TsdfPinholeCamera
    depth_scale_m: float
    voxel_size_m: float
    truncation_m: float
    frame_stride: int
    total_observations: int
    block_index: _BlockIndex
    tile_pixels: int
    frames: tuple[TsdfBlockFrameEvidence, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str) or not self.session_id:
            raise TsdfError("TSDF block evidence session_id must be nonempty")
        if not _is_sha256(self.replay_digest_sha256):
            raise TsdfError("TSDF block evidence replay digest is invalid")
        if not isinstance(self.camera, TsdfPinholeCamera):
            raise TsdfError(
                "TSDF block evidence camera must be a TsdfPinholeCamera"
            )
        for value, label in (
            (self.depth_scale_m, "depth_scale_m"),
            (self.voxel_size_m, "voxel_size_m"),
            (self.truncation_m, "truncation_m"),
        ):
            if not _finite(value) or value <= 0:
                raise TsdfError(
                    f"TSDF block evidence {label} must be finite and "
                    "positive"
                )
        if self.truncation_m < self.voxel_size_m:
            raise TsdfError(
                "TSDF block evidence truncation_m must be greater than or "
                "equal to voxel_size_m"
            )
        for value, label in (
            (self.frame_stride, "frame_stride"),
            (self.total_observations, "total_observations"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or (
                value < 1
            ):
                raise TsdfError(
                    f"TSDF block evidence {label} must be a positive integer"
                )
        _validate_block_index(self.block_index)
        _validate_tile_grid(
            self.camera.width, self.camera.height, self.tile_pixels
        )
        if not isinstance(self.frames, tuple) or any(
            not isinstance(frame, TsdfBlockFrameEvidence)
            for frame in self.frames
        ):
            raise TsdfError(
                "TSDF block evidence frames must be TsdfBlockFrameEvidence"
            )
        sequences = tuple(
            range(0, self.total_observations, self.frame_stride)
        )
        if tuple(frame.sequence for frame in self.frames) != sequences:
            raise TsdfError(
                "TSDF block evidence must hold every selected frame, in "
                "order"
            )
        if len(self.frames) > MAX_TSDF_VOXEL_WEIGHT:
            raise TsdfError(
                "TSDF block evidence would exceed the uint32 voxel weight "
                "maximum"
            )
        for frame in self.frames:
            if frame.tiles is not None and (
                frame.tiles.width != self.camera.width
                or frame.tiles.height != self.camera.height
                or frame.tiles.tile_pixels != self.tile_pixels
            ):
                raise TsdfError(
                    "TSDF block evidence tiles are not cut from this "
                    "camera's image on this tile grid"
                )

    @property
    def selected_observations(self) -> int:
        return len(self.frames)

    @property
    def frames_with_tiles(self) -> int:
        return sum(frame.tiles is not None for frame in self.frames)

    @property
    def depth_tile_count(self) -> int:
        return sum(
            frame.tiles.tile_count
            for frame in self.frames
            if frame.tiles is not None
        )

    @property
    def depth_sample_count(self) -> int:
        return sum(
            frame.tiles.sample_count
            for frame in self.frames
            if frame.tiles is not None
        )

    @property
    def depth_byte_count(self) -> int:
        return self.depth_sample_count * TSDF_DEPTH_SAMPLE_DTYPE.itemsize

    @property
    def scan_depth_byte_count(self) -> int:
        """Depth the selected frames hold in all, in the same bytes."""

        return (
            sum(frame.has_depth for frame in self.frames)
            * self.camera.width
            * self.camera.height
            * TSDF_DEPTH_SAMPLE_DTYPE.itemsize
        )


def gather_tsdf_block_evidence(
    session: ScanSession,
    block_index: _BlockIndex,
    *,
    voxel_size_m: float,
    truncation_m: float,
    frame_stride: int = 1,
    tile_pixels: int = TSDF_DEPTH_TILE_PIXELS,
) -> TsdfBlockEvidence:
    """Everything of ``session`` that one block is a function of."""

    return gather_tsdf_blocks_evidence(
        session,
        (block_index,),
        voxel_size_m=voxel_size_m,
        truncation_m=truncation_m,
        frame_stride=frame_stride,
        tile_pixels=tile_pixels,
    )[0]


def gather_tsdf_blocks_evidence(
    session: ScanSession,
    block_indices: Sequence[_BlockIndex],
    *,
    voxel_size_m: float,
    truncation_m: float,
    frame_stride: int = 1,
    tile_pixels: int = TSDF_DEPTH_TILE_PIXELS,
) -> tuple[TsdfBlockEvidence, ...]:
    """Evidence for each of several blocks, reading each frame once.

    Each block gets its own, holding the tiles that block reads and no
    tile kept for another. They are returned in the order given.
    """

    return _gather_blocks_evidence(
        session,
        block_indices,
        voxel_size_m=voxel_size_m,
        truncation_m=truncation_m,
        frame_stride=frame_stride,
        tile_pixels=tile_pixels,
        expected_replay_digest=None,
        expected_total_observations=None,
    )


def fuse_tsdf_block_from_evidence(
    evidence: TsdfBlockEvidence,
) -> TsdfLoneBlock:
    """The block its evidence determines. No session, no file.

    Frames are ruled out from their poses here, not taken on trust from
    which of them came with tiles, and every depth pixel the evaluator
    asks for must be held. A pixel that is not held is an error naming
    it; nothing is assumed about depth that was not given.
    """

    if not isinstance(evidence, TsdfBlockEvidence):
        raise TsdfError(
            "TSDF block fusion from evidence requires a TsdfBlockEvidence"
        )
    camera = evidence.camera
    blocks = np.asarray([evidence.block_index], dtype=np.int64)
    centres = _chunk_voxel_centres_world_m(blocks, evidence.voxel_size_m)
    sums = np.zeros(TSDF_BLOCK_VOXELS, dtype=TSDF_SUM_DTYPE)
    weights = np.zeros(TSDF_BLOCK_VOXELS, dtype=TSDF_WEIGHT_DTYPE)
    ready = behind = outside = evaluated = contributing = 0

    for frame in evidence.frames:
        if not frame.is_ready:
            continue
        if not _is_rigid_transform(frame.transform):
            raise TsdfError(
                "TSDF block fusion from evidence requires a rigid "
                f"T_world_camera for observation {frame.sequence}"
            )
        ready += 1
        verdict = int(
            _classify_blocks(
                camera, frame.transform, blocks, evidence.voxel_size_m
            )[0]
        )
        if verdict == BLOCK_BEHIND_CAMERA:
            behind += 1
            continue
        if verdict == BLOCK_OUTSIDE_IMAGE:
            outside += 1
            continue
        if verdict != BLOCK_EVALUATE:
            raise AssertionError("unknown whole-block verdict")
        evaluated += 1
        _, sum_deltas, weight_deltas = _evaluate_ready_voxels_reading(
            camera,
            frame.transform,
            _tile_reader(frame, evidence.depth_scale_m),
            evidence.truncation_m,
            centres,
        )
        sums += sum_deltas
        weights += weight_deltas
        contributing += int(weight_deltas.any())

    return _lone_block(
        session_id=evidence.session_id,
        replay_digest_sha256=evidence.replay_digest_sha256,
        voxel_size_m=evidence.voxel_size_m,
        truncation_m=evidence.truncation_m,
        frame_stride=evidence.frame_stride,
        block_index=evidence.block_index,
        selected_observations=evidence.selected_observations,
        ready_observations=ready,
        ruled_out_behind_camera=behind,
        ruled_out_outside_image=outside,
        evaluated_observations=evaluated,
        contributing_observations=contributing,
        flat_sums=sums,
        flat_weights=weights,
    )


def check_tsdf_volume_blocks_from_tiles(
    volume: TsdfBlockVolume,
    session: ScanSession,
    block_indices: Sequence[_BlockIndex],
    *,
    tile_pixels: int = TSDF_DEPTH_TILE_PIXELS,
) -> tuple[tuple[TsdfVolumeBlockCheck, TsdfBlockEvidence], ...]:
    """Hold blocks of a volume against the same blocks fused from tiles.

    Each block's evidence is gathered from the session and the block is
    then fused from that evidence and nothing else. Returned with the
    evidence, which says how much of the scan the check took.
    """

    _validate_check_inputs(volume, session)
    gathered = _gather_blocks_evidence(
        session,
        block_indices,
        voxel_size_m=volume.voxel_size_m,
        truncation_m=volume.truncation_m,
        frame_stride=volume.frame_stride,
        tile_pixels=tile_pixels,
        expected_replay_digest=volume.replay_digest_sha256,
        expected_total_observations=volume.total_observations,
    )
    checks = _check_against_volume(
        volume,
        tuple(
            fuse_tsdf_block_from_evidence(evidence) for evidence in gathered
        ),
    )
    return tuple(zip(checks, gathered, strict=True))


def _finite(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def _validate_tile_grid(
    width: object,
    height: object,
    tile_pixels: object,
) -> None:
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in (width, height)
    ):
        raise TsdfError("TSDF depth tile image dimensions are invalid")
    if (
        isinstance(tile_pixels, bool)
        or not isinstance(tile_pixels, int)
        or not 1 <= tile_pixels <= MAX_TSDF_DEPTH_TILE_PIXELS
    ):
        raise TsdfError(
            "TSDF depth tile size must be an integer from 1 to "
            f"{MAX_TSDF_DEPTH_TILE_PIXELS} pixels"
        )


def _missing_pixel(
    sequence: int,
    rows: np.ndarray,
    columns: np.ndarray,
    position: int,
) -> str:
    return (
        f"observation {sequence}: the block reads the depth pixel at row "
        f"{int(rows[position])}, column {int(columns[position])}, and the "
        "evidence holds no tile for it"
    )


def _tile_reader(frame: TsdfBlockFrameEvidence, depth_scale_m: float):
    """The gather the evaluator asks for, over one frame's tiles."""

    def read(row: np.ndarray, column: np.ndarray, alive: np.ndarray):
        if frame.tiles is None:
            wanted = np.flatnonzero(alive)
            if len(wanted):
                raise TsdfError(
                    _missing_pixel(
                        frame.sequence, row[wanted], column[wanted], 0
                    )
                )
            return np.zeros(len(row), dtype=np.float64)
        return frame.tiles.read_depth_m(
            row, column, alive, depth_scale_m, sequence=frame.sequence
        )

    return read


def _cut_tiles(
    raw: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
    tile_pixels: int,
) -> TsdfDepthTiles | None:
    """The tiles of one frame that hold the pixels read, or none."""

    if not len(rows):
        return None
    height, width = raw.shape
    tile_columns = -(-width // tile_pixels)
    keys = np.unique(
        (rows // tile_pixels) * tile_columns + columns // tile_pixels
    )
    tiles = []
    for key in keys.tolist():
        tile_row, tile_column = divmod(key, tile_columns)
        samples = raw[
            tile_row * tile_pixels:(tile_row + 1) * tile_pixels,
            tile_column * tile_pixels:(tile_column + 1) * tile_pixels,
        ].astype(TSDF_DEPTH_SAMPLE_DTYPE)
        samples.setflags(write=False)
        tiles.append((tile_row, tile_column, samples))
    return TsdfDepthTiles(
        width=width,
        height=height,
        tile_pixels=tile_pixels,
        tiles=tuple(tiles),
    )


def _gather_blocks_evidence(
    session: ScanSession,
    block_indices: Sequence[_BlockIndex],
    *,
    voxel_size_m: float,
    truncation_m: float,
    frame_stride: int,
    tile_pixels: int,
    expected_replay_digest: str | None,
    expected_total_observations: int | None,
) -> tuple[TsdfBlockEvidence, ...]:
    setup = _prepare_lone_fusion(
        session,
        block_indices,
        voxel_size_m=voxel_size_m,
        truncation_m=truncation_m,
        frame_stride=frame_stride,
        expected_replay_digest=expected_replay_digest,
        expected_total_observations=expected_total_observations,
    )
    camera = TsdfPinholeCamera(
        width=setup.camera.width,
        height=setup.camera.height,
        fx=float(setup.camera.fx),
        fy=float(setup.camera.fy),
        cx=float(setup.camera.cx),
        cy=float(setup.camera.cy),
    )
    _validate_tile_grid(camera.width, camera.height, tile_pixels)
    blocks = setup.blocks
    frames: list[list[TsdfBlockFrameEvidence]] = [[] for _ in blocks]

    try:
        for observation in setup.selected:
            status = _classify_observation(observation)
            has_depth = getattr(observation, "depth", None) is not None
            transform = _prepared_transform(observation, status)
            tiles: list[TsdfDepthTiles | None] = [None] * len(blocks)
            if status is TsdfReplayDepthStatus.READY:
                if transform is None or not _is_rigid_transform(transform):
                    raise TsdfError(
                        "TSDF block evidence requires a rigid "
                        "T_world_camera for observation "
                        f"{observation.sequence}"
                    )
                visible = np.flatnonzero(
                    _classify_blocks(
                        camera, transform, blocks, setup.voxel_size_m
                    )
                    == BLOCK_EVALUATE
                )
                if len(visible):
                    raw = _decode_raw_depth(
                        session, observation, camera.width, camera.height
                    )
                    if int(raw.min()) < 0 or int(raw.max()) > np.iinfo(
                        TSDF_DEPTH_SAMPLE_DTYPE
                    ).max:
                        raise TsdfError(
                            f"observation {observation.sequence} holds a "
                            "depth sample that does not fit 16 bits"
                        )
                    depth_m = _metric_depth(raw, setup.depth_scale_m)
                    for row in visible.tolist():
                        tiles[row] = _cut_tiles(
                            raw,
                            *_pixels_read(
                                camera,
                                transform,
                                depth_m,
                                setup.truncation_m,
                                blocks[row:row + 1],
                                setup.voxel_size_m,
                            ),
                            tile_pixels,
                        )
            for row in range(len(blocks)):
                frames[row].append(
                    TsdfBlockFrameEvidence(
                        sequence=observation.sequence,
                        transform=transform,
                        has_depth=has_depth,
                        tiles=tiles[row],
                    )
                )

        if replay_session(session).digest_sha256 != (
            setup.replay.digest_sha256
        ):
            raise TsdfError(
                "session inputs changed while TSDF block evidence was "
                "gathered; rerun the command"
            )
    except (TsdfError, SessionReplayError):
        raise
    except PointCloudError as error:
        raise TsdfError(str(error)) from error
    except Exception as error:
        raise TsdfError(
            f"cannot gather TSDF block evidence: {error}"
        ) from error

    return tuple(
        TsdfBlockEvidence(
            session_id=session.session_id,
            replay_digest_sha256=setup.replay.digest_sha256,
            camera=camera,
            depth_scale_m=float(setup.depth_scale_m),
            voxel_size_m=setup.voxel_size_m,
            truncation_m=setup.truncation_m,
            frame_stride=frame_stride,
            total_observations=len(setup.replay.observations),
            block_index=block,
            tile_pixels=tile_pixels,
            frames=tuple(frames[row]),
        )
        for row, block in enumerate(setup.requested)
    )


def _pixels_read(
    camera: TsdfPinholeCamera,
    transform: tuple[float, ...],
    depth_m: np.ndarray,
    truncation_m: float,
    block: np.ndarray,
    voxel_size_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """The depth pixels the evaluator reads for one block and one frame.

    Found by running the evaluator and noting what it asks for, so that
    there is one account of which pixel a voxel reads, not two.
    """

    asked: list[tuple[np.ndarray, np.ndarray]] = []

    def read(row: np.ndarray, column: np.ndarray, alive: np.ndarray):
        asked.append((row[alive], column[alive]))
        return depth_m[row, column]

    _evaluate_ready_voxels_reading(
        camera,
        transform,
        read,
        truncation_m,
        _chunk_voxel_centres_world_m(block, voxel_size_m),
    )
    if len(asked) != 1:
        raise AssertionError("the evaluator reads depth exactly once")
    return asked[0]
