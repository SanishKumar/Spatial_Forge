"""Several blocks of a map, held with what a check of them needs.

A block's proof carries every frame's leaf and the tiles that block reads.
Proofs of many blocks under one statement would repeat the leaves in every
one, and repeat a tile in each proof of a block that reads it. Held
together, the leaves are carried once, a tile once, and each frame's tiles
are tied to the frame by one proof shared between them.

Two things hold blocks this way: a route's certificate, whose blocks are
the ones the route touches, and an audit, whose blocks are drawn by lot.
What they share is here. Neither says anything a block's proof does not:
every block is tied to the map's root at its own row, every tile to its
own frame's depth root, and every block is fused again from the tiles and
compared.

Nothing here is public. It is the half of ``tsdf_route_certificate`` and
``tsdf_map_audit`` that is the same.
"""

from __future__ import annotations

import struct
from typing import Any, NamedTuple, Sequence

import numpy as np

from .errors import TsdfError
from .merkle import (
    MerkleTree,
    merkle_leaf_hash,
    merkle_path_length,
    merkle_root,
    merkle_root_from_path,
)
from .model import ScanSession
from .tsdf_block_evidence import (
    TsdfBlockFrameEvidence,
    _fuse_blocks_from_frames,
    _gather_pooled_frames,
    _validate_tile_grid,
)
from .tsdf_block_plan import TSDF_BLOCK_RESOLUTION
from .tsdf_block_proof import (
    TsdfProofFrameTiles,
    _decoded_tile_frames,
    _encoded_tile_frames,
    _immutable,
    _places,
    _pose_fusion_would_refuse,
    _Reader,
    _refuse_another_map,
    _tiles_not_under_depth_roots,
    _tiles_of_a_frame_not_fused,
    _validate_path,
)
from .tsdf_block_storage import TSDF_BLOCK_VOXELS
from .tsdf_block_volume import TsdfBlockVolume
from .tsdf_commitment import (
    _BLOCK_TAG,
    TSDF_FRAME_LEAF_BYTES,
    TsdfFrameCommitment,
    TsdfMapStatement,
    TsdfVolumeCommitment,
    _block_leaf_data,
    _commit_tsdf_scan,
    _frame_from_leaf_data,
    _statement,
    _statement_differences,
    tsdf_depth_tile_leaf_hash,
    tsdf_volume_merkle_tree,
)
from .tsdf_lone_block import _validate_block_index
from .tsdf_stream_fusion import _is_sha256

_BLOCK_SHAPE = (
    TSDF_BLOCK_RESOLUTION,
    TSDF_BLOCK_RESOLUTION,
    TSDF_BLOCK_RESOLUTION,
)
_SUM_DTYPE = np.dtype("<f8")
_WEIGHT_DTYPE = np.dtype("<u4")
_SUM_BYTES = TSDF_BLOCK_VOXELS * _SUM_DTYPE.itemsize
_BLOCK_LEAF_BYTES = (
    len(_BLOCK_TAG) + 12 + _SUM_BYTES + TSDF_BLOCK_VOXELS * 4
)
_ROW = struct.Struct("<Q")

_BlockIndex = tuple[int, int, int]


class _HeldBlocks(NamedTuple):
    """The fields a holder of blocks has, as they are made or read."""

    frames: tuple[TsdfFrameCommitment, ...]
    block_rows: tuple[int, ...]
    block_indices: tuple[_BlockIndex, ...]
    tsdf_sums: np.ndarray
    weights: np.ndarray
    block_paths: tuple[tuple[bytes, ...], ...]
    tile_frames: tuple[TsdfProofFrameTiles, ...]


class _HeldBlocksOutcome(NamedTuple):
    """What the checks every holder shares came to.

    ``failed_check`` is absent if everything is under the statement's
    roots. Then ``refuted`` gives each block that is not what the scan
    gives, with how many of its voxels differ.
    """

    failed_check: str | None
    detail: str | None
    refuted: tuple[tuple[_BlockIndex, int], ...]
    ready_observations: int


def _validate_held_blocks(
    holder: Any,
    what: str,
    most_blocks: int,
    *,
    in_block_order: bool,
) -> None:
    """Refuse a holder whose fields are not shaped as blocks are held.

    Blocks are distinct and in order: by index in the canonical block
    order, or by row of the map, as the holder keeps them.
    """

    if not _is_sha256(holder.statement_sha256):
        raise TsdfError(f"{what} statement digest is invalid")
    _validate_tile_grid(holder.width, holder.height, holder.tile_pixels)
    if (
        not isinstance(holder.frames, tuple)
        or not holder.frames
        or any(
            not isinstance(frame, TsdfFrameCommitment)
            for frame in holder.frames
        )
    ):
        raise TsdfError(f"a {what} needs at least one TsdfFrameCommitment")
    if tuple(frame.sequence for frame in holder.frames) != tuple(
        range(len(holder.frames))
    ):
        raise TsdfError(
            f"a {what} must hold every frame, in replay order"
        )
    if (
        isinstance(holder.block_count, bool)
        or not isinstance(holder.block_count, int)
        or holder.block_count < 1
    ):
        raise TsdfError(f"{what} block count must be a positive integer")
    held = (
        len(holder.block_indices)
        if isinstance(holder.block_indices, tuple)
        else 0
    )
    if not 1 <= held <= most_blocks:
        raise TsdfError(f"a {what} holds from 1 to {most_blocks} blocks")
    for block in holder.block_indices:
        _validate_block_index(block)
    if (
        not isinstance(holder.block_rows, tuple)
        or len(holder.block_rows) != held
        or any(
            isinstance(row, bool)
            or not isinstance(row, int)
            or not 0 <= row < holder.block_count
            for row in holder.block_rows
        )
    ):
        raise TsdfError(f"{what} needs a row of the map for each block")
    if in_block_order:
        order = [block[::-1] for block in holder.block_indices]
        if any(later <= earlier for earlier, later in zip(order, order[1:])):
            raise TsdfError(
                f"{what} blocks must be distinct and in canonical order"
            )
    elif any(
        later <= earlier
        for earlier, later in zip(holder.block_rows, holder.block_rows[1:])
    ):
        raise TsdfError(
            f"{what} blocks must be distinct and in order of their rows"
        )
    for array, dtype, label in (
        (holder.tsdf_sums, _SUM_DTYPE, "sums"),
        (holder.weights, _WEIGHT_DTYPE, "weights"),
    ):
        if (
            type(array) is not np.ndarray
            or array.shape != (held, *_BLOCK_SHAPE)
            or array.dtype != dtype
            or array.flags.writeable
        ):
            raise TsdfError(
                f"{what} {label} must be an immutable {dtype.name} array "
                f"of shape {(held, *_BLOCK_SHAPE)}"
            )
    if (
        not isinstance(holder.block_paths, tuple)
        or len(holder.block_paths) != held
    ):
        raise TsdfError(f"{what} needs a path for each block")
    for row, path in zip(holder.block_rows, holder.block_paths, strict=True):
        _validate_path(
            path, merkle_path_length(row, holder.block_count), "block path"
        )
    if not isinstance(holder.tile_frames, tuple) or any(
        not isinstance(tiles, TsdfProofFrameTiles)
        for tiles in holder.tile_frames
    ):
        raise TsdfError(f"{what} tiles must be TsdfProofFrameTiles")
    last = -1
    for tiles in holder.tile_frames:
        if tiles.sequence <= last:
            raise TsdfError(
                f"{what} tiles must be in increasing frame order"
            )
        last = tiles.sequence
        if tiles.sequence >= len(holder.frames):
            raise TsdfError(
                f"{what} holds tiles of a frame it does not hold"
            )
        if (
            tiles.tiles.width != holder.width
            or tiles.tiles.height != holder.height
            or tiles.tiles.tile_pixels != holder.tile_pixels
        ):
            raise TsdfError(f"{what} tiles are not cut on its tile grid")


def _prove_held_blocks(
    statement: TsdfMapStatement,
    volume: TsdfBlockVolume,
    session: ScanSession,
    block_indices: Sequence[_BlockIndex],
    changed: str,
) -> _HeldBlocks:
    """What a check of these blocks of a volume needs, under a statement.

    Every block must be one the volume holds. The statement is
    recomputed from the volume and the scan, the map's half before the
    scan is read. ``changed`` is what to say if the scan is found to
    have been rewritten between the two readings of it.
    """

    rows = {block: row for row, block in enumerate(volume.block_indices)}
    block_rows = tuple(rows[block] for block in block_indices)
    tree = tsdf_volume_merkle_tree(volume)
    committed = TsdfVolumeCommitment(
        session_id=volume.session_id,
        voxel_size_m=float(volume.voxel_size_m),
        truncation_m=float(volume.truncation_m),
        frame_stride=volume.frame_stride,
        block_count=volume.block_count,
        root_sha256=tree.root.hex(),
    )
    _refuse_another_map(statement, committed)

    gathered = _gather_pooled_frames(
        session,
        block_indices,
        voxel_size_m=volume.voxel_size_m,
        truncation_m=volume.truncation_m,
        frame_stride=volume.frame_stride,
        tile_pixels=statement.tile_pixels,
    )
    # As in a block's proof: each tile's leaf from the first reading of
    # the scan is held to the tree of the second.
    leaves: dict[int, dict[int, bytes]] = {}
    for frame in gathered:
        if frame.tiles is None:
            continue
        leaves[frame.sequence] = {
            place: tsdf_depth_tile_leaf_hash(samples)
            for place, (_, _, samples) in zip(
                _places(frame.tiles), frame.tiles.tiles, strict=True
            )
        }
    shared: dict[int, tuple[bytes, ...]] = {}

    def keep_proofs(sequence: int, frame_tree: MerkleTree) -> None:
        of_frame = leaves.get(sequence)
        if of_frame is None:
            return
        for place, leaf in of_frame.items():
            if frame_tree.leaf(place) != leaf:
                raise TsdfError(changed)
        shared[sequence] = frame_tree.shared_proof(sorted(of_frame))

    scan = _commit_tsdf_scan(session, statement.tile_pixels, keep_proofs)
    if len(shared) != len(leaves):
        raise TsdfError(changed)
    differences = _statement_differences(
        statement, _statement(scan, committed)
    )
    if differences:
        raise TsdfError(
            "the statement is not a statement of this volume and this "
            "scan: " + ", ".join(differences)
        )
    return _HeldBlocks(
        frames=scan.frames,
        block_rows=block_rows,
        block_indices=tuple(block_indices),
        tsdf_sums=_immutable(volume.tsdf_sums[list(block_rows)], _SUM_DTYPE),
        weights=_immutable(volume.weights[list(block_rows)], _WEIGHT_DTYPE),
        block_paths=tuple(tree.path(row) for row in block_rows),
        tile_frames=tuple(
            TsdfProofFrameTiles(
                sequence=frame.sequence,
                tiles=frame.tiles,
                shared_proof=shared[frame.sequence],
            )
            for frame in gathered
            if frame.tiles is not None
        ),
    )


def _check_held_blocks(
    statement: TsdfMapStatement,
    holder: Any,
    *,
    noun: str,
    readers: str,
) -> _HeldBlocksOutcome:
    """The checks every holder shares, from the scan's root to fusion.

    ``noun`` is what the holder is called and ``readers`` what its blocks
    are, for the details: a certificate and a block of the route, an
    audit and a block drawn.
    """

    def failed(check: str, detail: str) -> _HeldBlocksOutcome:
        return _HeldBlocksOutcome(check, detail, (), 0)

    selected = holder.frames[::statement.frame_stride]
    held = len(holder.block_indices)

    # scan root: the frames' leaves are the scan's.
    scan_root = merkle_root([frame.leaf_hash() for frame in holder.frames])
    if scan_root.hex() != statement.scan_root_sha256:
        return failed(
            "scan root", f"the frames' leaves lead to {scan_root.hex()}"
        )

    # map root: every block is the map's, at its own row.
    for place, (row, block, path) in enumerate(
        zip(
            holder.block_rows,
            holder.block_indices,
            holder.block_paths,
            strict=True,
        )
    ):
        map_root = merkle_root_from_path(
            merkle_leaf_hash(
                _block_leaf_data(
                    block, holder.tsdf_sums[place], holder.weights[place]
                )
            ),
            row,
            holder.block_count,
            path,
        )
        if map_root.hex() != statement.map_root_sha256:
            return failed(
                "map root",
                f"block {block} and its path lead to {map_root.hex()}",
            )

    # depth roots: each frame's tiles are that frame's, at their places.
    wrong = _tiles_not_under_depth_roots(holder.frames, holder.tile_frames)
    if wrong is not None:
        return failed(
            "depth roots",
            wrong.replace("the proof holds", f"the {noun} holds"),
        )

    # poses: every pose fusion would read is one it could read.
    wrong = _pose_fusion_would_refuse(selected)
    if wrong is not None:
        return failed("poses", wrong)

    # pixels: the tiles hold every pixel any block reads, and no tile
    # none of them reads a pixel of.
    wrong = _tiles_of_a_frame_not_fused(selected, holder.tile_frames)
    if wrong is not None:
        return failed(
            "pixels", wrong.replace("the proof holds", f"the {noun} holds")
        )
    by_frame = {tiles.sequence: tiles for tiles in holder.tile_frames}
    tile_columns = -(-holder.width // holder.tile_pixels)
    tile_rows = -(-holder.height // holder.tile_pixels)
    read: dict[int, np.ndarray] = {}

    def note(sequence: int, rows: np.ndarray, columns: np.ndarray) -> None:
        grid = read.get(sequence)
        if grid is None:
            grid = read[sequence] = np.zeros(
                (tile_rows, tile_columns), dtype=np.bool_
            )
        grid[rows // holder.tile_pixels, columns // holder.tile_pixels] = True

    try:
        fused = _fuse_blocks_from_frames(
            statement.camera,
            statement.depth_scale_m,
            statement.voxel_size_m,
            statement.truncation_m,
            holder.block_indices,
            [
                TsdfBlockFrameEvidence(
                    sequence=frame.sequence,
                    transform=frame.transform,
                    has_depth=frame.depth_root is not None,
                    tiles=(
                        by_frame[frame.sequence].tiles
                        if frame.sequence in by_frame
                        else None
                    ),
                )
                for frame in selected
            ],
            asked=note,
        )
    except TsdfError as error:
        return failed("pixels", str(error))
    for tiles in holder.tile_frames:
        grid = read.get(tiles.sequence)
        for tile_row, tile_column, _ in tiles.tiles.tiles:
            if grid is None or not grid[tile_row, tile_column]:
                return failed(
                    "pixels",
                    f"observation {tiles.sequence}: the {noun} holds the "
                    f"tile at row {tile_row}, column {tile_column}, and no "
                    f"{readers} reads a pixel of it",
                )

    # fusion: every block fused from the tiles is the block held.
    # Compared as bits.
    differing = np.count_nonzero(
        (
            fused.sums.view(np.uint64)
            != holder.tsdf_sums.reshape((held, -1)).view(np.uint64)
        )
        | (fused.weights != holder.weights.reshape((held, -1))),
        axis=1,
    )
    return _HeldBlocksOutcome(
        None,
        None,
        tuple(
            (holder.block_indices[place], int(differing[place]))
            for place in np.flatnonzero(differing).tolist()
        ),
        fused.ready_observations,
    )


def _encoded_held_blocks(holder: Any) -> list[bytes]:
    """Frames, blocks and tiles as a holder's bytes carry them, in pieces."""

    pieces = [frame.leaf_data() for frame in holder.frames]
    for place, (row, block, path) in enumerate(
        zip(
            holder.block_rows,
            holder.block_indices,
            holder.block_paths,
            strict=True,
        )
    ):
        pieces.append(_ROW.pack(row))
        pieces.append(
            _block_leaf_data(
                block, holder.tsdf_sums[place], holder.weights[place]
            )
        )
        pieces.extend(path)
    pieces.extend(_encoded_tile_frames(holder.tile_frames))
    return pieces


def _decoded_held_blocks(
    reader: _Reader,
    *,
    frames: int,
    of: int,
    held: int,
    tile_frames: int,
    width: int,
    height: int,
    tile_pixels: int,
    most_blocks: int,
) -> _HeldBlocks:
    """Frames, blocks and tiles, read from where the reader stands."""

    read_frames = tuple(
        _frame_from_leaf_data(reader.take(TSDF_FRAME_LEAF_BYTES, "a frame"))
        for _ in range(reader.count(frames, TSDF_FRAME_LEAF_BYTES))
    )
    held = reader.count(held, _BLOCK_LEAF_BYTES)
    if held > most_blocks:
        raise TsdfError(
            f"a {reader.name} holds from 1 to {most_blocks} blocks"
        )
    block_rows = []
    block_indices = []
    block_paths = []
    sums = np.empty((held, *_BLOCK_SHAPE), dtype=_SUM_DTYPE)
    weights = np.empty((held, *_BLOCK_SHAPE), dtype=_WEIGHT_DTYPE)
    for place in range(held):
        (row,) = _ROW.unpack(reader.take(_ROW.size, "a block's row"))
        if row >= of:
            raise TsdfError(
                f"{reader.name} row lies outside the blocks it counts"
            )
        leaf = reader.take(_BLOCK_LEAF_BYTES, "a block")
        if not leaf.startswith(_BLOCK_TAG):
            raise TsdfError(
                f"{reader.name} block does not start with its tag"
            )
        block_rows.append(row)
        block_indices.append(struct.unpack_from("<3i", leaf, len(_BLOCK_TAG)))
        sums[place] = np.frombuffer(
            leaf,
            dtype=_SUM_DTYPE,
            count=TSDF_BLOCK_VOXELS,
            offset=len(_BLOCK_TAG) + 12,
        ).reshape(_BLOCK_SHAPE)
        weights[place] = np.frombuffer(
            leaf,
            dtype=_WEIGHT_DTYPE,
            count=TSDF_BLOCK_VOXELS,
            offset=len(_BLOCK_TAG) + 12 + _SUM_BYTES,
        ).reshape(_BLOCK_SHAPE)
        block_paths.append(
            reader.hashes(merkle_path_length(row, of), "block path")
        )
    sums.setflags(write=False)
    weights.setflags(write=False)
    _validate_tile_grid(width, height, tile_pixels)
    return _HeldBlocks(
        frames=read_frames,
        block_rows=tuple(block_rows),
        block_indices=tuple(block_indices),
        tsdf_sums=sums,
        weights=weights,
        block_paths=tuple(block_paths),
        tile_frames=_decoded_tile_frames(
            reader, tile_frames, width, height, tile_pixels
        ),
    )
