"""A block's proof, and a check that reads nothing else.

A statement names two roots. A proof is what lets someone who holds only
the statement decide, for one block, whether the map's block follows from
the scan. It carries:

- every frame's leaf, which is every pose and every depth root, and from
  which the scan's root is recomputed;
- the block as the map holds it, with its path to the map's root;
- the tiles of depth the block reads, each with its path to its own
  frame's depth root.

The check ties each of those to the statement and then fuses the block
again from the tiles. It reads the statement and the proof. It opens no
session, no volume and no file.

There are three things it can say, and they are kept apart.

``verified``
    Everything in the proof is under the statement's roots, and the block
    fused from it is the block the map holds, bit for bit.
``refuted``
    Everything in the proof is under the statement's roots, and the block
    fused from it is another block. This is a finding about the map: what
    was committed as the map does not follow from what was committed as
    the scan. The proof is the evidence, and anyone can check it.
``invalid``
    Something in the proof is not under the statement's roots, or the
    proof does not hold what fusion needs, or holds more. This is a
    finding about the proof and says nothing about the map.

The checks run in a fixed order and stop at the first that fails, so an
invalid proof names one check. The check of which pixels are needed is
made by fusing: frames are ruled out from their committed poses, and the
evaluator asks for the pixels it reads. A pixel that is not held is a
refusal, and so is a tile no pixel of which was asked for. A proof is
therefore exactly what the block needs, and a block has one proof.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .errors import TsdfError
from .merkle import (
    MERKLE_HASH_BYTES,
    MerkleTree,
    merkle_leaf_hash,
    merkle_path_length,
    merkle_root,
    merkle_root_from_path,
)
from .model import ScanSession
from .tsdf_block_evidence import (
    TSDF_DEPTH_SAMPLE_DTYPE,
    TsdfBlockFrameEvidence,
    TsdfDepthTiles,
    _fuse_block_from_frames,
    _validate_tile_grid,
    gather_tsdf_blocks_evidence,
)
from .tsdf_block_plan import TSDF_BLOCK_RESOLUTION
from .tsdf_block_storage import TSDF_BLOCK_VOXELS
from .tsdf_block_volume import (
    TsdfBlockVolume,
    _DuplicateJsonKey,
    _encode_header,
    _exact,
    _integer,
    _object,
    _reject_json_constant,
    _sha256,
    _unique_object,
)
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
    _validate_volume,
    tsdf_depth_tile_leaf_hash,
    tsdf_volume_merkle_tree,
)
from .tsdf_lone_block import _validate_block_index
from .tsdf_replay_depth_context import _is_rigid_transform
from .tsdf_stream_fusion import _is_sha256

TSDF_BLOCK_PROOF_SCHEMA = "spatialforge.tsdf-block-proof"
TSDF_BLOCK_PROOF_SCHEMA_VERSION = "0.1.0"
TSDF_BLOCK_PROOF_MAGIC = b"SFTPRF01"
MAX_TSDF_BLOCK_PROOF_HEADER_BYTES = 16 * 1024
MAX_TSDF_BLOCK_PROOF_BYTES = 1024 * 1024 * 1024

# The checks, in the order they are made.
TSDF_BLOCK_PROOF_CHECKS = (
    "statement",
    "shape",
    "scan root",
    "map root",
    "depth roots",
    "poses",
    "pixels",
    "fusion",
)

_PREAMBLE_BYTES = len(TSDF_BLOCK_PROOF_MAGIC) + 8
_BLOCK_SHAPE = (
    TSDF_BLOCK_RESOLUTION,
    TSDF_BLOCK_RESOLUTION,
    TSDF_BLOCK_RESOLUTION,
)
_SUM_DTYPE = np.dtype("<f8")
_WEIGHT_DTYPE = np.dtype("<u4")
_SUM_BYTES = TSDF_BLOCK_VOXELS * _SUM_DTYPE.itemsize
_WEIGHT_BYTES = TSDF_BLOCK_VOXELS * _WEIGHT_DTYPE.itemsize
_BLOCK_LEAF_BYTES = len(_BLOCK_TAG) + 12 + _SUM_BYTES + _WEIGHT_BYTES
_TILE_FRAME_HEAD = struct.Struct("<QI")
_TILE_HEAD = struct.Struct("<II")
_SCAN_CHANGED = (
    "session inputs changed while the block was proved; rerun the command"
)

_BlockIndex = tuple[int, int, int]


class TsdfBlockProofVerdict(StrEnum):
    """What a check of a block's proof found."""

    VERIFIED = "verified"
    REFUTED = "refuted"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True, eq=False)
class TsdfProofFrameTiles:
    """The tiles of one frame a block reads, each with its path.

    ``paths`` holds one path a tile, in the order of the tiles: the
    hashes from the tile's leaf up to the frame's depth root.
    """

    sequence: int
    tiles: TsdfDepthTiles
    paths: tuple[tuple[bytes, ...], ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or not 0 <= self.sequence < 2**63
        ):
            raise TsdfError("TSDF proof tiles name an invalid frame")
        if not isinstance(self.tiles, TsdfDepthTiles):
            raise TsdfError("TSDF proof tiles must be TsdfDepthTiles")
        if (
            not isinstance(self.paths, tuple)
            or len(self.paths) != self.tiles.tile_count
        ):
            raise TsdfError("TSDF proof tiles need one path a tile")
        total = self.tiles.tile_rows * self.tiles.tile_columns
        for (tile_row, tile_column, _), path in zip(
            self.tiles.tiles, self.paths, strict=True
        ):
            _validate_path(
                path,
                merkle_path_length(
                    tile_row * self.tiles.tile_columns + tile_column, total
                ),
                "tile",
            )

    @property
    def tile_count(self) -> int:
        return self.tiles.tile_count

    @property
    def path_hash_count(self) -> int:
        return sum(len(path) for path in self.paths)


@dataclass(frozen=True, slots=True, eq=False)
class TsdfBlockProof:
    """What a check of one block needs, and nothing it does not.

    ``frames`` is every frame of the scan as the scan's tree holds it.
    ``tile_frames`` is, for each frame the block reads a pixel of, the
    tiles holding those pixels. The image size, tile size and counts are
    kept so that a proof can be read without its statement; the check
    holds each of them to the statement.
    """

    statement_sha256: str
    width: int
    height: int
    tile_pixels: int
    frames: tuple[TsdfFrameCommitment, ...]
    block_count: int
    block_row: int
    block_index: _BlockIndex
    tsdf_sums: np.ndarray
    weights: np.ndarray
    block_path: tuple[bytes, ...]
    tile_frames: tuple[TsdfProofFrameTiles, ...]

    def __post_init__(self) -> None:
        if not _is_sha256(self.statement_sha256):
            raise TsdfError("TSDF block proof statement digest is invalid")
        _validate_tile_grid(self.width, self.height, self.tile_pixels)
        if (
            not isinstance(self.frames, tuple)
            or not self.frames
            or any(
                not isinstance(frame, TsdfFrameCommitment)
                for frame in self.frames
            )
        ):
            raise TsdfError(
                "a TSDF block proof needs at least one TsdfFrameCommitment"
            )
        if tuple(frame.sequence for frame in self.frames) != tuple(
            range(len(self.frames))
        ):
            raise TsdfError(
                "a TSDF block proof must hold every frame, in replay order"
            )
        for value, label in (
            (self.block_count, "block count"),
            (self.block_row, "block row"),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TsdfError(f"TSDF block proof {label} must be an integer")
        if self.block_count < 1 or not 0 <= self.block_row < self.block_count:
            raise TsdfError(
                "TSDF block proof row lies outside the blocks it counts"
            )
        _validate_block_index(self.block_index)
        for array, dtype, label in (
            (self.tsdf_sums, _SUM_DTYPE, "sums"),
            (self.weights, _WEIGHT_DTYPE, "weights"),
        ):
            if (
                type(array) is not np.ndarray
                or array.shape != _BLOCK_SHAPE
                or array.dtype != dtype
                or array.flags.writeable
            ):
                raise TsdfError(
                    f"TSDF block proof {label} must be an immutable "
                    f"{dtype.name} array of shape {_BLOCK_SHAPE}"
                )
        _validate_path(
            self.block_path,
            merkle_path_length(self.block_row, self.block_count),
            "block",
        )
        if not isinstance(self.tile_frames, tuple) or any(
            not isinstance(tiles, TsdfProofFrameTiles)
            for tiles in self.tile_frames
        ):
            raise TsdfError(
                "TSDF block proof tiles must be TsdfProofFrameTiles"
            )
        previous = -1
        for tiles in self.tile_frames:
            if tiles.sequence <= previous:
                raise TsdfError(
                    "TSDF block proof tiles must be in increasing frame "
                    "order"
                )
            previous = tiles.sequence
            if tiles.sequence >= len(self.frames):
                raise TsdfError(
                    "TSDF block proof holds tiles of a frame it does not "
                    "hold"
                )
            if (
                tiles.tiles.width != self.width
                or tiles.tiles.height != self.height
                or tiles.tiles.tile_pixels != self.tile_pixels
            ):
                raise TsdfError(
                    "TSDF block proof tiles are not cut on its tile grid"
                )

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def tile_count(self) -> int:
        return sum(tiles.tile_count for tiles in self.tile_frames)

    @property
    def depth_byte_count(self) -> int:
        return sum(tiles.tiles.byte_count for tiles in self.tile_frames)

    @property
    def tile_path_hash_count(self) -> int:
        return sum(tiles.path_hash_count for tiles in self.tile_frames)

    def block_leaf_data(self) -> bytes:
        """The bytes the map's tree holds this block as."""

        return _block_leaf_data(self.block_index, self.tsdf_sums, self.weights)

    def encoded(self) -> bytes:
        """The proof as the bytes of a ``.sftproof`` file."""

        header = _encode_header(
            {
                "schema": TSDF_BLOCK_PROOF_SCHEMA,
                "schema_version": TSDF_BLOCK_PROOF_SCHEMA_VERSION,
                "statement_sha256": self.statement_sha256,
                "frames": self.frame_count,
                "image": {
                    "width": self.width,
                    "height": self.height,
                    "tile_pixels": self.tile_pixels,
                },
                "block": {
                    "blocks": self.block_count,
                    "row": self.block_row,
                },
                "tile_frames": len(self.tile_frames),
            }
        )
        pieces = [
            TSDF_BLOCK_PROOF_MAGIC,
            struct.pack("<Q", len(header)),
            header,
        ]
        pieces.extend(frame.leaf_data() for frame in self.frames)
        pieces.append(self.block_leaf_data())
        pieces.extend(self.block_path)
        pieces.extend(_encoded_tile_frames(self.tile_frames))
        return b"".join(pieces)

    @property
    def digest_sha256(self) -> str:
        return hashlib.sha256(self.encoded()).hexdigest()


@dataclass(frozen=True, slots=True)
class TsdfBlockProofReport:
    """What writing a proof produced."""

    output: Path
    output_bytes: int
    output_digest_sha256: str
    proof: TsdfBlockProof


@dataclass(frozen=True, slots=True)
class TsdfBlockProofCheck:
    """One block's proof held against a statement.

    ``failed_check`` names the check that stopped it, one of
    ``TSDF_BLOCK_PROOF_CHECKS``, or is absent when the block is verified.
    A refuted block stops at ``fusion``. The counts of frames are those
    of the fusion and are zero if it was not reached.
    """

    statement_digest_sha256: str
    block_index: _BlockIndex
    verdict: TsdfBlockProofVerdict
    failed_check: str | None
    detail: str | None
    differing_voxels: int
    selected_observations: int
    ready_observations: int
    ruled_out_observations: int
    evaluated_observations: int
    contributing_observations: int

    @property
    def holds(self) -> bool:
        return self.verdict is TsdfBlockProofVerdict.VERIFIED

    @property
    def passed_checks(self) -> tuple[str, ...]:
        if self.failed_check is None:
            return TSDF_BLOCK_PROOF_CHECKS
        return TSDF_BLOCK_PROOF_CHECKS[
            :TSDF_BLOCK_PROOF_CHECKS.index(self.failed_check)
        ]


def prove_tsdf_block(
    statement: TsdfMapStatement,
    volume: TsdfBlockVolume,
    session: ScanSession,
    block_index: _BlockIndex,
) -> TsdfBlockProof:
    """The proof of one block of a volume under its statement."""

    return prove_tsdf_blocks(statement, volume, session, (block_index,))[0]


def prove_tsdf_blocks(
    statement: TsdfMapStatement,
    volume: TsdfBlockVolume,
    session: ScanSession,
    block_indices: Sequence[_BlockIndex],
) -> tuple[TsdfBlockProof, ...]:
    """A proof for each of several blocks, in the order given.

    The statement must be a statement of this volume and this scan, and
    that is recomputed here, not taken on trust. Nothing is asked of the
    volume beyond that: a proof is made of whatever block the map
    holds, and it is the check that says whether the block follows.
    """

    if not isinstance(statement, TsdfMapStatement):
        raise TsdfError("a TSDF block proof requires a TsdfMapStatement")
    _validate_volume(volume)
    if not isinstance(session, ScanSession):
        raise TsdfError("a TSDF block proof requires a loaded ScanSession")
    if isinstance(block_indices, (str, bytes)) or not isinstance(
        block_indices, Sequence
    ):
        raise TsdfError("TSDF block proofs are asked for by block index")
    requested = tuple(_validate_block_index(block) for block in block_indices)
    if not requested:
        raise TsdfError("at least one block must be asked for")
    rows = {block: row for row, block in enumerate(volume.block_indices)}
    for block in requested:
        if block not in rows:
            raise TsdfError(
                f"the volume holds no block {block}; only a block the map "
                "holds can be proved"
            )

    # The map's half of the statement first: it is cheap, and a proof
    # under a statement of another map is not worth reading a scan for.
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

    gathered = gather_tsdf_blocks_evidence(
        session,
        requested,
        voxel_size_m=volume.voxel_size_m,
        truncation_m=volume.truncation_m,
        frame_stride=volume.frame_stride,
        tile_pixels=statement.tile_pixels,
    )
    # The scan is read twice: once for the tiles, above, and once for
    # the trees their paths come from. Each tile's leaf is kept from the
    # first reading and held to the tree of the second, so that a scan
    # that changed in between is refused here and not handed on as a
    # proof that cannot hold.
    leaves: dict[int, dict[int, bytes]] = {}
    for evidence in gathered:
        for frame in evidence.frames:
            if frame.tiles is None:
                continue
            of_frame = leaves.setdefault(frame.sequence, {})
            columns = frame.tiles.tile_columns
            for tile_row, tile_column, samples in frame.tiles.tiles:
                place = tile_row * columns + tile_column
                if place not in of_frame:
                    of_frame[place] = tsdf_depth_tile_leaf_hash(samples)
    paths: dict[tuple[int, int], tuple[bytes, ...]] = {}

    def keep_paths(sequence: int, frame_tree: MerkleTree) -> None:
        for place, leaf in leaves.get(sequence, {}).items():
            if frame_tree.leaf(place) != leaf:
                raise TsdfError(_SCAN_CHANGED)
            paths[sequence, place] = frame_tree.path(place)

    scan = _commit_tsdf_scan(session, statement.tile_pixels, keep_paths)
    if len(paths) != sum(len(of_frame) for of_frame in leaves.values()):
        raise TsdfError(_SCAN_CHANGED)
    differences = _statement_differences(
        statement, _statement(scan, committed)
    )
    if differences:
        raise TsdfError(
            "the statement is not a statement of this volume and this "
            "scan: " + ", ".join(differences)
        )

    digest = statement.digest_sha256
    proofs = []
    for block, evidence in zip(requested, gathered, strict=True):
        row = rows[block]
        tile_frames = []
        for frame in evidence.frames:
            if frame.tiles is None:
                continue
            columns = frame.tiles.tile_columns
            tile_frames.append(
                TsdfProofFrameTiles(
                    sequence=frame.sequence,
                    tiles=frame.tiles,
                    paths=tuple(
                        paths[frame.sequence, tile_row * columns + tile_column]
                        for tile_row, tile_column, _ in frame.tiles.tiles
                    ),
                )
            )
        proofs.append(
            TsdfBlockProof(
                statement_sha256=digest,
                width=statement.camera.width,
                height=statement.camera.height,
                tile_pixels=statement.tile_pixels,
                frames=scan.frames,
                block_count=volume.block_count,
                block_row=row,
                block_index=block,
                tsdf_sums=_immutable(volume.tsdf_sums[row], _SUM_DTYPE),
                weights=_immutable(volume.weights[row], _WEIGHT_DTYPE),
                block_path=tree.path(row),
                tile_frames=tuple(tile_frames),
            )
        )
    return tuple(proofs)


def check_tsdf_block_proof(
    statement: TsdfMapStatement,
    proof: TsdfBlockProof,
) -> TsdfBlockProofCheck:
    """Hold one block's proof against a statement.

    Reads the two values it is given and nothing else.
    """

    if not isinstance(statement, TsdfMapStatement):
        raise TsdfError(
            "a TSDF block proof check requires a TsdfMapStatement"
        )
    if not isinstance(proof, TsdfBlockProof):
        raise TsdfError("a TSDF block proof check requires a TsdfBlockProof")
    statement_digest = statement.digest_sha256
    selected = proof.frames[::statement.frame_stride]

    def found(
        verdict: TsdfBlockProofVerdict,
        failed_check: str | None,
        detail: str | None,
        differing_voxels: int = 0,
        fused: Any = None,
    ) -> TsdfBlockProofCheck:
        return TsdfBlockProofCheck(
            statement_digest_sha256=statement_digest,
            block_index=proof.block_index,
            verdict=verdict,
            failed_check=failed_check,
            detail=detail,
            differing_voxels=differing_voxels,
            selected_observations=len(selected),
            ready_observations=0 if fused is None else (
                fused.ready_observations
            ),
            ruled_out_observations=0 if fused is None else (
                fused.ruled_out_behind_camera + fused.ruled_out_outside_image
            ),
            evaluated_observations=0 if fused is None else (
                fused.evaluated_observations
            ),
            contributing_observations=0 if fused is None else (
                fused.contributing_observations
            ),
        )

    def invalid(failed_check: str, detail: str) -> TsdfBlockProofCheck:
        return found(TsdfBlockProofVerdict.INVALID, failed_check, detail)

    # statement: the proof names the statement it was made under.
    if proof.statement_sha256 != statement_digest:
        return invalid(
            "statement",
            f"the proof was made under statement {proof.statement_sha256}",
        )

    # shape: the proof is laid out as the statement says.
    for label, held, stated in (
        ("frames", proof.frame_count, statement.frame_count),
        ("image width", proof.width, statement.camera.width),
        ("image height", proof.height, statement.camera.height),
        ("tile size", proof.tile_pixels, statement.tile_pixels),
        ("blocks", proof.block_count, statement.block_count),
    ):
        if held != stated:
            return invalid(
                "shape",
                f"the proof has {label} {held} and the statement {stated}",
            )

    # scan root: the frames' leaves are the scan's.
    scan_root = merkle_root([frame.leaf_hash() for frame in proof.frames])
    if scan_root.hex() != statement.scan_root_sha256:
        return invalid(
            "scan root",
            f"the frames' leaves lead to {scan_root.hex()}",
        )

    # map root: the block is the map's, at its own place.
    map_root = merkle_root_from_path(
        merkle_leaf_hash(proof.block_leaf_data()),
        proof.block_row,
        proof.block_count,
        proof.block_path,
    )
    if map_root.hex() != statement.map_root_sha256:
        return invalid(
            "map root",
            f"the block and its path lead to {map_root.hex()}",
        )

    # depth roots: each tile is its frame's, at its own place.
    wrong = _tiles_not_under_depth_roots(proof.frames, proof.tile_frames)
    if wrong is not None:
        return invalid("depth roots", wrong)

    # poses: every pose fusion would read is one it could read.
    wrong = _pose_fusion_would_refuse(selected)
    if wrong is not None:
        return invalid("poses", wrong)

    # pixels: the proof holds every pixel the block reads, and no tile
    # the block reads no pixel of.
    held = {tiles.sequence: tiles for tiles in proof.tile_frames}
    wrong = _tiles_of_a_frame_not_fused(selected, proof.tile_frames)
    if wrong is not None:
        return invalid("pixels", wrong)
    asked: dict[int, set[int]] = {}

    def note(sequence: int, rows: np.ndarray, columns: np.ndarray) -> None:
        asked[sequence] = set(
            (
                (rows // proof.tile_pixels)
                * -(-proof.width // proof.tile_pixels)
                + columns // proof.tile_pixels
            ).tolist()
        )

    try:
        fused = _fuse_block_from_frames(
            statement.camera,
            statement.depth_scale_m,
            statement.voxel_size_m,
            statement.truncation_m,
            proof.block_index,
            [
                TsdfBlockFrameEvidence(
                    sequence=frame.sequence,
                    transform=frame.transform,
                    has_depth=frame.depth_root is not None,
                    tiles=(
                        held[frame.sequence].tiles
                        if frame.sequence in held
                        else None
                    ),
                )
                for frame in selected
            ],
            asked=note,
        )
    except TsdfError as error:
        return invalid("pixels", str(error))
    for tiles in proof.tile_frames:
        read = asked.get(tiles.sequence, set())
        columns = tiles.tiles.tile_columns
        for tile_row, tile_column, _ in tiles.tiles.tiles:
            if tile_row * columns + tile_column not in read:
                return invalid(
                    "pixels",
                    f"observation {tiles.sequence}: the proof holds the "
                    f"tile at row {tile_row}, column {tile_column}, and "
                    "the block reads no pixel of it",
                )

    # fusion: the block fused from the proof is the block the map holds.
    # Compared as bits, as a volume's blocks are.
    differing = int(
        np.count_nonzero(
            (
                fused.flat_sums.view(np.uint64)
                != proof.tsdf_sums.reshape(-1).view(np.uint64)
            )
            | (fused.flat_weights != proof.weights.reshape(-1))
        )
    )
    if differing:
        return found(
            TsdfBlockProofVerdict.REFUTED,
            "fusion",
            f"{differing} of {TSDF_BLOCK_VOXELS} voxels are not what the "
            "scan gives",
            differing,
            fused,
        )
    return found(TsdfBlockProofVerdict.VERIFIED, None, None, 0, fused)


def write_tsdf_block_proof(
    proof: TsdfBlockProof,
    output: str | Path,
) -> TsdfBlockProofReport:
    """Write a proof as a ``.sftproof`` file. Never over an existing file."""

    if not isinstance(proof, TsdfBlockProof):
        raise TsdfError("writing requires a TsdfBlockProof")
    output_path = _validate_proof_output(output)
    encoded = proof.encoded()
    if len(encoded) > MAX_TSDF_BLOCK_PROOF_BYTES:
        raise TsdfError(
            "TSDF block proof exceeds the maximum of "
            f"{MAX_TSDF_BLOCK_PROOF_BYTES} bytes"
        )
    temporary_path: Path | None = None
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output_path.stem}-",
            suffix=".sftproof",
            dir=output_path.parent,
        )
        temporary_path = Path(temporary_name)
        with open(descriptor, "wb", closefd=True) as output_file:
            output_file.write(encoded)
        try:
            os.link(temporary_path, output_path)
        except FileExistsError as error:
            raise TsdfError(
                "output appeared while writing; refusing to overwrite: "
                f"{output_path}"
            ) from error
    except TsdfError:
        raise
    except OSError as error:
        raise TsdfError(f"cannot write TSDF block proof: {error}") from error
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
    return TsdfBlockProofReport(
        output=output_path,
        output_bytes=len(encoded),
        output_digest_sha256=hashlib.sha256(encoded).hexdigest(),
        proof=proof,
    )


def load_tsdf_block_proof(path: str | Path) -> TsdfBlockProof:
    """Read a proof, refusing anything that is not exactly one."""

    proof_path = Path(path)
    if proof_path.suffix.lower() != ".sftproof":
        raise TsdfError("proof filename must end in .sftproof")
    try:
        with proof_path.open("rb") as proof_file:
            size = os.fstat(proof_file.fileno()).st_size
            encoded = proof_file.read(
                min(size, MAX_TSDF_BLOCK_PROOF_BYTES) + 1
            )
    except OSError as error:
        raise TsdfError(f"cannot read TSDF block proof: {error}") from error
    if len(encoded) > MAX_TSDF_BLOCK_PROOF_BYTES:
        raise TsdfError(
            "TSDF block proof exceeds the maximum of "
            f"{MAX_TSDF_BLOCK_PROOF_BYTES} bytes"
        )
    return decode_tsdf_block_proof(encoded)


def decode_tsdf_block_proof(encoded: bytes) -> TsdfBlockProof:
    """The proof these bytes are, or a refusal.

    The bytes are read once from front to back, and every one of them
    is accounted for: none may be missing and none left over.
    """

    if type(encoded) is not bytes:
        raise TsdfError("a TSDF block proof is decoded from bytes")
    if len(encoded) < _PREAMBLE_BYTES or not encoded.startswith(
        TSDF_BLOCK_PROOF_MAGIC
    ):
        raise TsdfError("TSDF block proof has an unrecognized signature")
    (header_length,) = struct.unpack_from(
        "<Q", encoded, len(TSDF_BLOCK_PROOF_MAGIC)
    )
    if not 2 <= header_length <= MAX_TSDF_BLOCK_PROOF_HEADER_BYTES:
        raise TsdfError("TSDF block proof header length is out of range")
    reader = _Reader(encoded, _PREAMBLE_BYTES)
    header_bytes = reader.take(header_length, "its header")
    header = _parse_header(header_bytes)

    frames = tuple(
        _frame_from_leaf_data(reader.take(TSDF_FRAME_LEAF_BYTES, "a frame"))
        for _ in range(reader.count(header["frames"], TSDF_FRAME_LEAF_BYTES))
    )
    leaf = reader.take(_BLOCK_LEAF_BYTES, "its block")
    if not leaf.startswith(_BLOCK_TAG):
        raise TsdfError("TSDF block proof block does not start with its tag")
    block_index = struct.unpack_from("<3i", leaf, len(_BLOCK_TAG))
    sums = np.frombuffer(
        leaf,
        dtype=_SUM_DTYPE,
        count=TSDF_BLOCK_VOXELS,
        offset=len(_BLOCK_TAG) + 12,
    ).reshape(_BLOCK_SHAPE)
    weights = np.frombuffer(
        leaf,
        dtype=_WEIGHT_DTYPE,
        count=TSDF_BLOCK_VOXELS,
        offset=len(_BLOCK_TAG) + 12 + _SUM_BYTES,
    ).reshape(_BLOCK_SHAPE)
    blocks = header["block"]["blocks"]
    row = header["block"]["row"]
    if row >= blocks:
        raise TsdfError(
            "TSDF block proof row lies outside the blocks it counts"
        )
    block_path = reader.hashes(merkle_path_length(row, blocks), "block path")

    image = header["image"]
    width, height, tile_pixels = (
        image["width"], image["height"], image["tile_pixels"],
    )
    _validate_tile_grid(width, height, tile_pixels)
    tile_frames = _decoded_tile_frames(
        reader, header["tile_frames"], width, height, tile_pixels
    )
    reader.finish()

    proof = TsdfBlockProof(
        statement_sha256=header["statement_sha256"],
        width=width,
        height=height,
        tile_pixels=tile_pixels,
        frames=frames,
        block_count=blocks,
        block_row=row,
        block_index=block_index,
        tsdf_sums=sums,
        weights=weights,
        block_path=block_path,
        tile_frames=tile_frames,
    )
    if proof.encoded() != encoded:
        raise TsdfError("TSDF block proof is not in the form it is written")
    return proof


def _encoded_tile_frames(
    tile_frames: Sequence[TsdfProofFrameTiles],
) -> list[bytes]:
    """Frames' tiles as a proof's bytes hold them, in pieces."""

    pieces = []
    for tiles in tile_frames:
        pieces.append(_TILE_FRAME_HEAD.pack(tiles.sequence, tiles.tile_count))
        for (tile_row, tile_column, samples), path in zip(
            tiles.tiles.tiles, tiles.paths, strict=True
        ):
            pieces.append(_TILE_HEAD.pack(tile_row, tile_column))
            pieces.append(np.ascontiguousarray(samples, dtype="<u2").tobytes())
            pieces.extend(path)
    return pieces


def _decoded_tile_frames(
    reader: "_Reader",
    count: int,
    width: int,
    height: int,
    tile_pixels: int,
) -> tuple[TsdfProofFrameTiles, ...]:
    """``count`` frames' tiles, read from where the reader stands."""

    tile_rows = -(-height // tile_pixels)
    tile_columns = -(-width // tile_pixels)
    total = tile_rows * tile_columns
    tile_frames = []
    for _ in range(reader.count(count, _TILE_FRAME_HEAD.size)):
        sequence, tile_count = _TILE_FRAME_HEAD.unpack(
            reader.take(_TILE_FRAME_HEAD.size, "a frame's tiles")
        )
        tiles = []
        paths = []
        for _ in range(reader.count(tile_count, _TILE_HEAD.size)):
            tile_row, tile_column = _TILE_HEAD.unpack(
                reader.take(_TILE_HEAD.size, "a tile")
            )
            if not (tile_row < tile_rows and tile_column < tile_columns):
                raise TsdfError("TSDF depth tile lies outside the image")
            shape = (
                min(tile_pixels, height - tile_row * tile_pixels),
                min(tile_pixels, width - tile_column * tile_pixels),
            )
            samples = np.frombuffer(
                reader.take(2 * shape[0] * shape[1], "a tile's samples"),
                dtype="<u2",
            ).reshape(shape).astype(TSDF_DEPTH_SAMPLE_DTYPE)
            samples.setflags(write=False)
            tiles.append((tile_row, tile_column, samples))
            paths.append(
                reader.hashes(
                    merkle_path_length(
                        tile_row * tile_columns + tile_column, total
                    ),
                    "tile path",
                )
            )
        if not tiles:
            raise TsdfError(
                f"{reader.name} names a frame and holds no tile of it"
            )
        tile_frames.append(
            TsdfProofFrameTiles(
                sequence=sequence,
                tiles=TsdfDepthTiles(
                    width=width,
                    height=height,
                    tile_pixels=tile_pixels,
                    tiles=tuple(tiles),
                ),
                paths=tuple(paths),
            )
        )
    return tuple(tile_frames)


def _tiles_not_under_depth_roots(
    frames: Sequence[TsdfFrameCommitment],
    tile_frames: Sequence[TsdfProofFrameTiles],
) -> str | None:
    """What is wrong with the first tile that is not its frame's, if any."""

    for tiles in tile_frames:
        depth_root = frames[tiles.sequence].depth_root
        if depth_root is None:
            return (
                f"observation {tiles.sequence} has no depth, and the proof "
                "holds tiles of it"
            )
        columns = tiles.tiles.tile_columns
        total = tiles.tiles.tile_rows * columns
        for (tile_row, tile_column, samples), path in zip(
            tiles.tiles.tiles, tiles.paths, strict=True
        ):
            if merkle_root_from_path(
                tsdf_depth_tile_leaf_hash(samples),
                tile_row * columns + tile_column,
                total,
                path,
            ) != depth_root:
                return (
                    f"observation {tiles.sequence}: the tile at row "
                    f"{tile_row}, column {tile_column} does not lead to "
                    "the frame's depth root"
                )
    return None


def _pose_fusion_would_refuse(
    selected: Sequence[TsdfFrameCommitment],
) -> str | None:
    """What is wrong with the first pose fusion could not read, if any.

    A frame with depth is fused and needs a rigid pose; one without is
    only looked at.
    """

    for frame in selected:
        if frame.transform is None:
            continue
        if not all(math.isfinite(value) for value in frame.transform) or (
            frame.depth_root is not None
            and not _is_rigid_transform(frame.transform)
        ):
            return (
                f"observation {frame.sequence} is committed with a pose "
                "that is not a rigid transform; this scan cannot be fused"
            )
    return None


def _tiles_of_a_frame_not_fused(
    selected: Sequence[TsdfFrameCommitment],
    tile_frames: Sequence[TsdfProofFrameTiles],
) -> str | None:
    """Tiles held of a frame the stride skips, or one with no pose."""

    fused = {
        frame.sequence for frame in selected if frame.transform is not None
    }
    for tiles in tile_frames:
        if tiles.sequence not in fused:
            return (
                f"observation {tiles.sequence} is not fused, and the proof "
                "holds tiles of it"
            )
    return None


class _Reader:
    """A place in a proof's bytes, moved only forwards.

    ``name`` is what the bytes are meant to be, for its refusals.
    """

    __slots__ = ("_encoded", "_at", "name")

    def __init__(
        self,
        encoded: bytes,
        at: int,
        name: str = "TSDF block proof",
    ) -> None:
        self._encoded = encoded
        self._at = at
        self.name = name

    def take(self, length: int, what: str) -> bytes:
        end = self._at + length
        if end > len(self._encoded):
            raise TsdfError(f"{self.name} is truncated inside {what}")
        taken = self._encoded[self._at:end]
        self._at = end
        return taken

    def count(self, stated: int, least_bytes_each: int) -> int:
        """A count the header gives, refused if the bytes cannot hold it."""

        if stated * least_bytes_each > len(self._encoded) - self._at:
            raise TsdfError(
                f"{self.name} counts more than its bytes can hold"
            )
        return stated

    def hashes(self, count: int, what: str) -> tuple[bytes, ...]:
        return tuple(
            self.take(MERKLE_HASH_BYTES, f"a {what}") for _ in range(count)
        )

    def finish(self) -> None:
        if self._at != len(self._encoded):
            raise TsdfError(
                f"{self.name} holds "
                f"{len(self._encoded) - self._at} bytes after its end"
            )


def _parse_header(header_bytes: bytes) -> dict[str, Any]:
    try:
        text = header_bytes.decode("ascii")
    except UnicodeError as error:
        raise TsdfError(
            "TSDF block proof header must be ASCII JSON"
        ) from error
    if text.count("{") > 8 or "[" in text:
        raise TsdfError(
            "TSDF block proof header holds an array or is nested too deeply"
        )
    try:
        document = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except _DuplicateJsonKey as error:
        raise TsdfError(
            "TSDF block proof header contains duplicate JSON key "
            f"{error.key!r}"
        ) from error
    except (json.JSONDecodeError, ValueError, RecursionError) as error:
        raise TsdfError(
            f"TSDF block proof header is invalid JSON: {error}"
        ) from error

    root = _object(
        document,
        "proof",
        (
            "schema",
            "schema_version",
            "statement_sha256",
            "frames",
            "image",
            "block",
            "tile_frames",
        ),
    )
    _exact(root["schema"], TSDF_BLOCK_PROOF_SCHEMA, "proof.schema")
    _exact(
        root["schema_version"],
        TSDF_BLOCK_PROOF_SCHEMA_VERSION,
        "proof.schema_version",
    )
    _sha256(root["statement_sha256"], "proof.statement_sha256")
    _integer(root["frames"], "proof.frames", minimum=1)
    _integer(root["tile_frames"], "proof.tile_frames", minimum=0)
    image = _object(
        root["image"], "proof.image", ("width", "height", "tile_pixels")
    )
    for key in ("width", "height", "tile_pixels"):
        _integer(image[key], f"proof.image.{key}", minimum=1)
    block = _object(root["block"], "proof.block", ("blocks", "row"))
    _integer(block["blocks"], "proof.block.blocks", minimum=1)
    _integer(block["row"], "proof.block.row", minimum=0)
    if _encode_header(root) != header_bytes:
        raise TsdfError("TSDF block proof header is not in canonical form")
    return root


def _refuse_another_map(
    statement: TsdfMapStatement,
    committed: TsdfVolumeCommitment,
) -> None:
    differences = [
        label
        for label, stated, held in (
            ("session", statement.session_id, committed.session_id),
            ("voxel size", statement.voxel_size_m, committed.voxel_size_m),
            ("truncation", statement.truncation_m, committed.truncation_m),
            ("frame stride", statement.frame_stride, committed.frame_stride),
            ("block count", statement.block_count, committed.block_count),
            ("map root", statement.map_root_sha256, committed.root_sha256),
        )
        if stated != held
    ]
    if differences:
        raise TsdfError(
            "the statement is not a statement of this volume: "
            + ", ".join(differences)
        )


def _validate_path(path: object, length: int, what: str) -> None:
    if (
        not isinstance(path, tuple)
        or len(path) != length
        or any(
            type(step) is not bytes or len(step) != MERKLE_HASH_BYTES
            for step in path
        )
    ):
        raise TsdfError(
            f"a TSDF proof {what} path here is {length} hashes of "
            f"{MERKLE_HASH_BYTES} bytes"
        )


def _immutable(array: np.ndarray, dtype: np.dtype) -> np.ndarray:
    copy = np.array(array, dtype=dtype, order="C", copy=True)
    copy.setflags(write=False)
    return copy


def _validate_proof_output(output: str | Path) -> Path:
    output_path = Path(output).resolve()
    if output_path.suffix.lower() != ".sftproof":
        raise TsdfError("output filename must end in .sftproof")
    if output_path.exists():
        raise TsdfError(f"output already exists: {output_path}")
    return output_path
