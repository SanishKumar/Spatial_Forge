"""One hash for a scan, one for a map, and a statement that binds them.

A block can be fused from the few depth tiles it reads, but nothing in
those tiles says they are the scan's. A commitment is what says it. The
scan and the map are each reduced to one hash in such a way that a piece of
either can later be shown to belong, with a short path of other hashes, to
someone who holds only the hash.

Both are Merkle trees as RFC 6962 defines them (``merkle.py``).

**The scan.** One leaf for every frame, in replay order, whether fusion
selects it or not, so that one commitment serves volumes at any stride. A
frame's leaf holds its place, its pose if it has one, and the root of a
second tree over its depth image: the image cut into tiles on the grid
block evidence uses, each tile a leaf, as the 16-bit samples the session
decodes. Not the file's bytes. Two encodings of the same depth are the same
scan.

The frames' leaves together are small, a pose and a hash each, and a check
of any block needs every pose. So they are meant to be handed over whole:
whoever has them recomputes the scan root from them, and from then on has
each frame's pose and each frame's depth root, to which a tile is tied by
its path.

**The map.** One leaf for every block of a volume in the canonical block
order, holding the block's index and its accumulators.

**The statement** names both roots and everything else a block's bytes
depend on: the camera, the depth scale, the tile size, the voxel size, the
truncation, the stride, and the name of the fusion rule. It is a few lines
of canonical JSON, and its SHA-256 is the one value a third party needs to
hold.

A leaf of each kind starts with its own tag, so a tile can never be read as
a frame or a block.

Nothing here is a proof yet. It is what a proof will be checked against.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .errors import PointCloudError, SessionReplayError, TsdfError
from .merkle import (
    MERKLE_HASH_BYTES,
    MerkleTree,
    merkle_leaf_hash,
    merkle_root,
)
from .model import ScanSession
from .point_cloud import _validate_reconstruction_contract
from .replay import replay_session
from .tsdf_block_evidence import (
    TSDF_DEPTH_SAMPLE_DTYPE,
    TSDF_DEPTH_TILE_PIXELS,
    TsdfPinholeCamera,
    _validate_tile_grid,
)
from .tsdf_block_plan import TSDF_BLOCK_RESOLUTION
from .tsdf_block_volume import (
    _IDENTIFIER,
    TsdfBlockVolume,
    _DuplicateJsonKey,
    _encode_header,
    _exact,
    _integer,
    _object,
    _positive_float,
    _reject_json_constant,
    _sha256,
    _unique_object,
)
from .tsdf_replay_depth_context import (
    _classify_observation,
    _prepared_transform,
    _validate_context_camera,
)
from .tsdf_stream_fusion import _decode_raw_depth, _is_sha256

# The arithmetic a block's bytes come from. A change to what fusion
# computes is a new name, and a statement made under the old one is then
# a statement about a different function.
TSDF_FUSION_RULE = "projective-tsdf/unit-weight/nearest-pixel/1"
TSDF_MAP_STATEMENT_SCHEMA = "spatialforge.tsdf-map-statement"
TSDF_MAP_STATEMENT_SCHEMA_VERSION = "0.1.0"
MAX_TSDF_MAP_STATEMENT_BYTES = 16 * 1024

_TILE_TAG = b"SFTILE01"
_FRAME_TAG = b"SFFRAME1"
_BLOCK_TAG = b"SFBLOCK1"
_HASH = "sha-256"
_TREE = "rfc6962"
_SAMPLE = "uint16-le"
_NO_POSE = bytes(16 * 8)
_NO_DEPTH = bytes(MERKLE_HASH_BYTES)
# A rule's name is a path of plain words. Like a session identifier it
# holds nothing JSON gives a meaning to, so every statement that can be
# made can be written and read back.
_RULE_NAME = re.compile(r"^[a-z0-9][a-z0-9._/-]{0,127}$")


@dataclass(frozen=True, slots=True)
class TsdfFrameCommitment:
    """One frame of a scan as the scan's tree holds it.

    ``depth_root`` is the root of the frame's tile tree, or absent for a
    frame with no depth image. ``transform`` is ``T_world_camera`` as
    fusion reads it, or absent for a frame with no pose.
    """

    sequence: int
    transform: tuple[float, ...] | None
    depth_root: bytes | None

    def __post_init__(self) -> None:
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or not 0 <= self.sequence < 2**63
        ):
            raise TsdfError("TSDF frame commitment sequence is invalid")
        if self.transform is not None and (
            not isinstance(self.transform, tuple)
            or len(self.transform) != 16
            or any(type(value) is not float for value in self.transform)
        ):
            raise TsdfError(
                "TSDF frame commitment pose must be 16 floats or absent"
            )
        if self.depth_root is not None and (
            type(self.depth_root) is not bytes
            or len(self.depth_root) != MERKLE_HASH_BYTES
        ):
            raise TsdfError(
                "TSDF frame commitment depth root must be a hash or absent"
            )

    @property
    def has_depth(self) -> bool:
        return self.depth_root is not None

    def leaf_data(self) -> bytes:
        """The bytes this frame's leaf is the hash of."""

        flags = (1 if self.has_depth else 0) | (
            2 if self.transform is not None else 0
        )
        return b"".join(
            (
                _FRAME_TAG,
                struct.pack("<QB", self.sequence, flags),
                _NO_POSE
                if self.transform is None
                else struct.pack("<16d", *self.transform),
                _NO_DEPTH if self.depth_root is None else self.depth_root,
            )
        )

    def leaf_hash(self) -> bytes:
        return merkle_leaf_hash(self.leaf_data())


TSDF_FRAME_LEAF_BYTES = len(_FRAME_TAG) + 9 + len(_NO_POSE) + len(_NO_DEPTH)


def _frame_from_leaf_data(data: bytes) -> TsdfFrameCommitment:
    """The frame a leaf's bytes say, or a refusal.

    Only the bytes ``leaf_data`` writes are read back: an absent pose or
    depth root must be zeros, and no flag may be set but the two.
    """

    if type(data) is not bytes or len(data) != TSDF_FRAME_LEAF_BYTES:
        raise TsdfError(
            f"a TSDF frame leaf is {TSDF_FRAME_LEAF_BYTES} bytes"
        )
    if not data.startswith(_FRAME_TAG):
        raise TsdfError("a TSDF frame leaf starts with its tag")
    sequence, flags = struct.unpack_from("<QB", data, len(_FRAME_TAG))
    pose = data[len(_FRAME_TAG) + 9:-MERKLE_HASH_BYTES]
    depth_root = data[-MERKLE_HASH_BYTES:]
    if flags > 3 or sequence >= 2**63:
        raise TsdfError("a TSDF frame leaf holds an unknown flag or place")
    if not flags & 2 and pose != _NO_POSE:
        raise TsdfError("a TSDF frame leaf with no pose holds one")
    if not flags & 1 and depth_root != _NO_DEPTH:
        raise TsdfError("a TSDF frame leaf with no depth holds a depth root")
    frame = TsdfFrameCommitment(
        sequence=sequence,
        transform=struct.unpack("<16d", pose) if flags & 2 else None,
        depth_root=depth_root if flags & 1 else None,
    )
    if frame.leaf_data() != data:
        raise TsdfError("a TSDF frame leaf is not in the form it is written")
    return frame


@dataclass(frozen=True, slots=True, eq=False)
class TsdfScanCommitment:
    """A scan reduced to its frames' leaves, and so to one root.

    The root is not stored. It is what the frames come to, each time it
    is asked for.
    """

    session_id: str
    camera: TsdfPinholeCamera
    depth_scale_m: float
    tile_pixels: int
    frames: tuple[TsdfFrameCommitment, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str) or not self.session_id:
            raise TsdfError("TSDF scan commitment session_id must be nonempty")
        if not isinstance(self.camera, TsdfPinholeCamera):
            raise TsdfError(
                "TSDF scan commitment camera must be a TsdfPinholeCamera"
            )
        if (
            type(self.depth_scale_m) is not float
            or not np.isfinite(self.depth_scale_m)
            or self.depth_scale_m <= 0.0
        ):
            raise TsdfError(
                "TSDF scan commitment depth scale must be finite and "
                "positive"
            )
        _validate_tile_grid(
            self.camera.width, self.camera.height, self.tile_pixels
        )
        if (
            not isinstance(self.frames, tuple)
            or not self.frames
            or any(
                not isinstance(frame, TsdfFrameCommitment)
                for frame in self.frames
            )
        ):
            raise TsdfError(
                "TSDF scan commitment needs at least one "
                "TsdfFrameCommitment"
            )
        if tuple(frame.sequence for frame in self.frames) != tuple(
            range(len(self.frames))
        ):
            raise TsdfError(
                "TSDF scan commitment must hold every frame, in replay "
                "order"
            )

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def root(self) -> bytes:
        return merkle_root([frame.leaf_hash() for frame in self.frames])

    @property
    def root_sha256(self) -> str:
        return self.root.hex()

    @property
    def frame_table_bytes(self) -> int:
        """What handing over every frame's leaf costs."""

        return sum(len(frame.leaf_data()) for frame in self.frames)


@dataclass(frozen=True, slots=True)
class TsdfVolumeCommitment:
    """A volume reduced to one root, with what the root is a root of."""

    session_id: str
    voxel_size_m: float
    truncation_m: float
    frame_stride: int
    block_count: int
    root_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str) or not self.session_id:
            raise TsdfError(
                "TSDF volume commitment session_id must be nonempty"
            )
        for value, label in (
            (self.voxel_size_m, "voxel_size_m"),
            (self.truncation_m, "truncation_m"),
        ):
            if type(value) is not float or not np.isfinite(value) or (
                value <= 0.0
            ):
                raise TsdfError(
                    f"TSDF volume commitment {label} must be finite and "
                    "positive"
                )
        for value, label in (
            (self.frame_stride, "frame_stride"),
            (self.block_count, "block_count"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or (
                value < 1
            ):
                raise TsdfError(
                    f"TSDF volume commitment {label} must be a positive "
                    "integer"
                )
        if not _is_sha256(self.root_sha256):
            raise TsdfError("TSDF volume commitment root is invalid")


@dataclass(frozen=True, slots=True)
class TsdfMapStatement:
    """Everything a block's bytes depend on, and the two roots.

    ``encoded()`` is the statement as canonical JSON, and
    ``digest_sha256`` the one value that stands for all of it.
    """

    session_id: str
    rule: str
    frame_count: int
    camera: TsdfPinholeCamera
    depth_scale_m: float
    tile_pixels: int
    scan_root_sha256: str
    voxel_size_m: float
    truncation_m: float
    frame_stride: int
    block_count: int
    map_root_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str) or not _IDENTIFIER.fullmatch(
            self.session_id
        ):
            raise TsdfError(
                "TSDF map statement session_id must be a session identifier"
            )
        if not isinstance(self.rule, str) or not _RULE_NAME.fullmatch(
            self.rule
        ):
            raise TsdfError(
                "TSDF map statement rule must be named in lowercase "
                "letters, digits, '.', '_', '-' or '/'"
            )
        if not isinstance(self.camera, TsdfPinholeCamera):
            raise TsdfError(
                "TSDF map statement camera must be a TsdfPinholeCamera"
            )
        for value, label in (
            (self.depth_scale_m, "depth_scale_m"),
            (self.voxel_size_m, "voxel_size_m"),
            (self.truncation_m, "truncation_m"),
        ):
            if type(value) is not float or not np.isfinite(value) or (
                value <= 0.0
            ):
                raise TsdfError(
                    f"TSDF map statement {label} must be finite and "
                    "positive"
                )
        if self.truncation_m < self.voxel_size_m:
            raise TsdfError(
                "TSDF map statement truncation_m must be greater than or "
                "equal to voxel_size_m"
            )
        for value, label in (
            (self.frame_count, "frame_count"),
            (self.frame_stride, "frame_stride"),
            (self.block_count, "block_count"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or (
                value < 1
            ):
                raise TsdfError(
                    f"TSDF map statement {label} must be a positive integer"
                )
        _validate_tile_grid(
            self.camera.width, self.camera.height, self.tile_pixels
        )
        for value, label in (
            (self.scan_root_sha256, "scan root"),
            (self.map_root_sha256, "map root"),
        ):
            if not _is_sha256(value):
                raise TsdfError(f"TSDF map statement {label} is invalid")

    def document(self) -> dict[str, Any]:
        return {
            "schema": TSDF_MAP_STATEMENT_SCHEMA,
            "schema_version": TSDF_MAP_STATEMENT_SCHEMA_VERSION,
            "session_id": self.session_id,
            "hash": _HASH,
            "tree": _TREE,
            "rule": self.rule,
            "scan": {
                "frames": self.frame_count,
                "camera": {
                    "width": self.camera.width,
                    "height": self.camera.height,
                    "fx": float(self.camera.fx),
                    "fy": float(self.camera.fy),
                    "cx": float(self.camera.cx),
                    "cy": float(self.camera.cy),
                },
                "depth_scale_m": self.depth_scale_m,
                "depth_sample": _SAMPLE,
                "tile_pixels": self.tile_pixels,
                "root_sha256": self.scan_root_sha256,
            },
            "map": {
                "voxel_size_m": self.voxel_size_m,
                "truncation_m": self.truncation_m,
                "block_resolution": TSDF_BLOCK_RESOLUTION,
                "frame_stride": self.frame_stride,
                "blocks": self.block_count,
                "root_sha256": self.map_root_sha256,
            },
        }

    def encoded(self) -> bytes:
        return _encode_header(self.document())

    @property
    def digest_sha256(self) -> str:
        return hashlib.sha256(self.encoded()).hexdigest()


@dataclass(frozen=True, slots=True)
class TsdfMapStatementReport:
    """What writing a statement produced."""

    output: Path
    output_bytes: int
    statement: TsdfMapStatement

    @property
    def output_digest_sha256(self) -> str:
        return self.statement.digest_sha256


@dataclass(frozen=True, slots=True)
class TsdfMapStatementCheck:
    """A statement held against a volume and a scan, recomputed.

    ``differences`` names each thing that is not as the statement says,
    in a fixed order. It is empty when the statement holds.
    """

    statement_digest_sha256: str
    scan_root_sha256: str
    map_root_sha256: str
    differences: tuple[str, ...]

    @property
    def holds(self) -> bool:
        return not self.differences


def tsdf_depth_tile_leaf_hash(samples: np.ndarray) -> bytes:
    """The leaf of one tile: its raw samples, row by row."""

    if (
        type(samples) is not np.ndarray
        or samples.ndim != 2
        or samples.dtype != TSDF_DEPTH_SAMPLE_DTYPE
        or not samples.size
    ):
        raise TsdfError("a TSDF depth tile leaf is a 2-D uint16 array")
    return merkle_leaf_hash(
        _TILE_TAG + np.ascontiguousarray(samples, dtype="<u2").tobytes()
    )


def tsdf_depth_tile_tree(
    raw_depth: np.ndarray,
    tile_pixels: int,
) -> MerkleTree:
    """The tree over one frame's depth, tiles taken row by row.

    ``raw_depth`` is the frame's samples as the session decodes them.
    The leaves are in the order of ``(tile_row, tile_column)``.
    """

    if (
        type(raw_depth) is not np.ndarray
        or raw_depth.ndim != 2
        or not np.issubdtype(raw_depth.dtype, np.integer)
    ):
        raise TsdfError("TSDF depth to commit must be a 2-D integer array")
    height, width = raw_depth.shape
    _validate_tile_grid(width, height, tile_pixels)
    if raw_depth.dtype != TSDF_DEPTH_SAMPLE_DTYPE:
        if int(raw_depth.min()) < 0 or int(raw_depth.max()) > np.iinfo(
            TSDF_DEPTH_SAMPLE_DTYPE
        ).max:
            raise TsdfError(
                "TSDF depth to commit holds a sample that does not fit "
                "16 bits"
            )
    samples = np.ascontiguousarray(raw_depth, dtype="<u2")
    leaves = []
    for top in range(0, height, tile_pixels):
        band = samples[top:top + tile_pixels]
        for left in range(0, width, tile_pixels):
            leaves.append(
                merkle_leaf_hash(
                    _TILE_TAG
                    + np.ascontiguousarray(
                        band[:, left:left + tile_pixels]
                    ).tobytes()
                )
            )
    return MerkleTree(leaves)


def commit_tsdf_scan(
    session: ScanSession,
    *,
    tile_pixels: int = TSDF_DEPTH_TILE_PIXELS,
) -> TsdfScanCommitment:
    """Reduce a scan to its frames' leaves. Reads every depth image."""

    return _commit_tsdf_scan(session, tile_pixels, None)


def _commit_tsdf_scan(
    session: ScanSession,
    tile_pixels: int,
    visit: Callable[[int, MerkleTree], None] | None,
) -> TsdfScanCommitment:
    """Commit a scan, showing ``visit`` each frame's tile tree in passing.

    A tree is needed again only by whoever wants a tile's path, and is
    not kept here: a scan's trees together are as large as its depth.
    """

    if not isinstance(session, ScanSession):
        raise TsdfError("TSDF scan commitment requires a loaded ScanSession")
    try:
        camera, depth_scale_m = _validate_reconstruction_contract(session)
    except PointCloudError as error:
        raise TsdfError(str(error)) from error
    _validate_context_camera(camera)
    _validate_tile_grid(camera.width, camera.height, tile_pixels)
    try:
        replay = replay_session(session)
        frames = []
        for observation in replay.observations:
            transform = _prepared_transform(
                observation, _classify_observation(observation)
            )
            depth_root = None
            if getattr(observation, "depth", None) is not None:
                tree = tsdf_depth_tile_tree(
                    _decode_raw_depth(
                        session, observation, camera.width, camera.height
                    ),
                    tile_pixels,
                )
                depth_root = tree.root
                if visit is not None:
                    visit(observation.sequence, tree)
            frames.append(
                TsdfFrameCommitment(
                    sequence=observation.sequence,
                    transform=transform,
                    depth_root=depth_root,
                )
            )
        if replay_session(session).digest_sha256 != replay.digest_sha256:
            raise TsdfError(
                "session inputs changed while the scan was committed; "
                "rerun the command"
            )
    except (TsdfError, SessionReplayError):
        raise
    except PointCloudError as error:
        raise TsdfError(str(error)) from error
    return TsdfScanCommitment(
        session_id=session.session_id,
        camera=TsdfPinholeCamera(
            width=camera.width,
            height=camera.height,
            fx=float(camera.fx),
            fy=float(camera.fy),
            cx=float(camera.cx),
            cy=float(camera.cy),
        ),
        depth_scale_m=float(depth_scale_m),
        tile_pixels=tile_pixels,
        frames=tuple(frames),
    )


def tsdf_volume_block_leaf_hash(volume: TsdfBlockVolume, row: int) -> bytes:
    """The leaf of one block: its index and its accumulators.

    ``row`` is the block's place in the volume's own order.
    """

    _validate_volume(volume)
    if (
        isinstance(row, bool)
        or not isinstance(row, int)
        or not 0 <= row < volume.block_count
    ):
        raise TsdfError(
            f"a volume of {volume.block_count} blocks has no block at "
            f"row {row!r}"
        )
    return _block_leaf_hash(volume, row)


def _block_leaf_hash(volume: TsdfBlockVolume, row: int) -> bytes:
    return merkle_leaf_hash(
        _block_leaf_data(
            volume.block_indices[row],
            volume.tsdf_sums[row],
            volume.weights[row],
        )
    )


def _block_leaf_data(
    block_index: tuple[int, int, int],
    tsdf_sums: np.ndarray,
    weights: np.ndarray,
) -> bytes:
    """The bytes a block's leaf is the hash of."""

    return b"".join(
        (
            _BLOCK_TAG,
            struct.pack("<3i", *block_index),
            np.ascontiguousarray(tsdf_sums, dtype="<f8").tobytes(),
            np.ascontiguousarray(weights, dtype="<u4").tobytes(),
        )
    )


def tsdf_volume_merkle_tree(volume: TsdfBlockVolume) -> MerkleTree:
    """The tree over a volume's blocks, in the order the volume lists them."""

    _validate_volume(volume)
    return MerkleTree(
        [
            _block_leaf_hash(volume, row)
            for row in range(volume.block_count)
        ]
    )


def commit_tsdf_volume(volume: TsdfBlockVolume) -> TsdfVolumeCommitment:
    """Reduce a volume to one root."""

    _validate_volume(volume)
    return TsdfVolumeCommitment(
        session_id=volume.session_id,
        voxel_size_m=float(volume.voxel_size_m),
        truncation_m=float(volume.truncation_m),
        frame_stride=volume.frame_stride,
        block_count=volume.block_count,
        root_sha256=merkle_root(
            [
                _block_leaf_hash(volume, row)
                for row in range(volume.block_count)
            ]
        ).hex(),
    )


def state_tsdf_map(
    volume: TsdfBlockVolume,
    session: ScanSession,
    *,
    tile_pixels: int = TSDF_DEPTH_TILE_PIXELS,
) -> TsdfMapStatement:
    """The statement of a volume and the scan it was fused from.

    The session must be the one the volume records, by replay digest:
    a statement binds a map to its own scan and to no other.
    """

    return _state_tsdf_map(volume, session, tile_pixels)[0]


def _state_tsdf_map(
    volume: TsdfBlockVolume,
    session: ScanSession,
    tile_pixels: int,
) -> tuple[TsdfMapStatement, TsdfScanCommitment]:
    """The statement, and the scan commitment it was made from."""

    _validate_volume(volume)
    if not isinstance(session, ScanSession):
        raise TsdfError("a TSDF map statement requires a loaded ScanSession")
    if volume.session_id != session.session_id:
        raise TsdfError(
            f"volume was fused from session {volume.session_id!r}, not "
            f"{session.session_id!r}"
        )
    replay = replay_session(session)
    if replay.digest_sha256 != volume.replay_digest_sha256:
        raise TsdfError(
            "the session's replay digest does not match the one recorded "
            "in the volume; this is not the scan the volume was fused from"
        )
    scan = commit_tsdf_scan(session, tile_pixels=tile_pixels)
    if scan.frame_count != volume.total_observations:
        raise TsdfError(
            "the session's observation count does not match the one "
            "recorded in the volume"
        )
    return _statement(scan, commit_tsdf_volume(volume)), scan


def check_tsdf_map_statement(
    statement: TsdfMapStatement,
    volume: TsdfBlockVolume,
    session: ScanSession,
) -> TsdfMapStatementCheck:
    """Recompute a statement from a volume and a scan, and compare.

    This is the check that needs everything: it commits the whole scan
    and the whole volume again. It says the statement is a statement of
    these two. It does not fuse anything, so it does not say the volume
    follows from the scan.
    """

    if not isinstance(statement, TsdfMapStatement):
        raise TsdfError(
            "a TSDF map statement check requires a TsdfMapStatement"
        )
    _validate_volume(volume)
    if not isinstance(session, ScanSession):
        raise TsdfError(
            "a TSDF map statement check requires a loaded ScanSession"
        )
    scan = commit_tsdf_scan(session, tile_pixels=statement.tile_pixels)
    recomputed = _statement(scan, commit_tsdf_volume(volume))
    return TsdfMapStatementCheck(
        statement_digest_sha256=statement.digest_sha256,
        scan_root_sha256=recomputed.scan_root_sha256,
        map_root_sha256=recomputed.map_root_sha256,
        differences=_statement_differences(statement, recomputed),
    )


def _statement_differences(
    stated: TsdfMapStatement,
    found: TsdfMapStatement,
) -> tuple[str, ...]:
    """What one statement says that another does not, in a fixed order.

    The tile size is not named: it shows in the scan root it leads to.
    """

    return tuple(
        label
        for label, one, other in (
            ("session", stated.session_id, found.session_id),
            ("fusion rule", stated.rule, found.rule),
            ("frame count", stated.frame_count, found.frame_count),
            ("camera", stated.camera, found.camera),
            ("depth scale", stated.depth_scale_m, found.depth_scale_m),
            ("scan root", stated.scan_root_sha256, found.scan_root_sha256),
            ("voxel size", stated.voxel_size_m, found.voxel_size_m),
            ("truncation", stated.truncation_m, found.truncation_m),
            ("frame stride", stated.frame_stride, found.frame_stride),
            ("block count", stated.block_count, found.block_count),
            ("map root", stated.map_root_sha256, found.map_root_sha256),
        )
        if one != other
    )


def write_tsdf_map_statement(
    statement: TsdfMapStatement,
    output: str | Path,
) -> TsdfMapStatementReport:
    """Write a statement as canonical JSON. Never over an existing file."""

    if not isinstance(statement, TsdfMapStatement):
        raise TsdfError("writing requires a TsdfMapStatement")
    output_path = _validate_statement_output(output)
    encoded = statement.encoded()
    temporary_path: Path | None = None
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output_path.stem}-",
            suffix=".sftcommit",
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
        raise TsdfError(
            f"cannot write TSDF map statement: {error}"
        ) from error
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
    return TsdfMapStatementReport(
        output=output_path,
        output_bytes=len(encoded),
        statement=statement,
    )


def load_tsdf_map_statement(path: str | Path) -> TsdfMapStatement:
    """Read a statement, refusing anything that is not exactly one."""

    statement_path = Path(path)
    if statement_path.suffix.lower() != ".sftcommit":
        raise TsdfError("statement filename must end in .sftcommit")
    try:
        with statement_path.open("rb") as statement_file:
            encoded = statement_file.read(MAX_TSDF_MAP_STATEMENT_BYTES + 1)
    except OSError as error:
        raise TsdfError(f"cannot read TSDF map statement: {error}") from error
    if len(encoded) > MAX_TSDF_MAP_STATEMENT_BYTES:
        raise TsdfError(
            "TSDF map statement is larger than "
            f"{MAX_TSDF_MAP_STATEMENT_BYTES} bytes"
        )
    statement = _parse_statement(encoded)
    if statement.encoded() != encoded:
        raise TsdfError("TSDF map statement is not in canonical form")
    return statement


def _statement(
    scan: TsdfScanCommitment,
    volume: TsdfVolumeCommitment,
) -> TsdfMapStatement:
    if scan.session_id != volume.session_id:
        raise TsdfError(
            "a TSDF map statement binds a volume to its own scan"
        )
    return TsdfMapStatement(
        session_id=scan.session_id,
        rule=TSDF_FUSION_RULE,
        frame_count=scan.frame_count,
        camera=scan.camera,
        depth_scale_m=scan.depth_scale_m,
        tile_pixels=scan.tile_pixels,
        scan_root_sha256=scan.root_sha256,
        voxel_size_m=volume.voxel_size_m,
        truncation_m=volume.truncation_m,
        frame_stride=volume.frame_stride,
        block_count=volume.block_count,
        map_root_sha256=volume.root_sha256,
    )


def _validate_volume(volume: object) -> None:
    if not isinstance(volume, TsdfBlockVolume):
        raise TsdfError(
            "a TSDF volume commitment requires a loaded TsdfBlockVolume"
        )
    if volume.block_resolution != TSDF_BLOCK_RESOLUTION:
        raise TsdfError(
            "a TSDF volume commitment requires block resolution "
            f"{TSDF_BLOCK_RESOLUTION}"
        )


def _validate_statement_output(output: str | Path) -> Path:
    output_path = Path(output).resolve()
    if output_path.suffix.lower() != ".sftcommit":
        raise TsdfError("output filename must end in .sftcommit")
    if output_path.exists():
        raise TsdfError(f"output already exists: {output_path}")
    return output_path


def _parse_statement(encoded: bytes) -> TsdfMapStatement:
    try:
        text = encoded.decode("ascii")
    except UnicodeError as error:
        raise TsdfError("TSDF map statement must be ASCII JSON") from error
    if text.count("{") > 8 or "[" in text:
        raise TsdfError(
            "TSDF map statement holds an array or is nested too deeply"
        )
    try:
        document = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except _DuplicateJsonKey as error:
        raise TsdfError(
            f"TSDF map statement contains duplicate JSON key {error.key!r}"
        ) from error
    except (json.JSONDecodeError, ValueError, RecursionError) as error:
        raise TsdfError(
            f"TSDF map statement is invalid JSON: {error}"
        ) from error

    root = _object(
        document,
        "statement",
        (
            "schema",
            "schema_version",
            "session_id",
            "hash",
            "tree",
            "rule",
            "scan",
            "map",
        ),
    )
    _exact(root["schema"], TSDF_MAP_STATEMENT_SCHEMA, "statement.schema")
    _exact(
        root["schema_version"],
        TSDF_MAP_STATEMENT_SCHEMA_VERSION,
        "statement.schema_version",
    )
    _exact(root["hash"], _HASH, "statement.hash")
    _exact(root["tree"], _TREE, "statement.tree")
    for key in ("session_id", "rule"):
        if not isinstance(root[key], str) or not root[key]:
            raise TsdfError(f"statement.{key}: expected a nonempty string")
    scan = _object(
        root["scan"],
        "statement.scan",
        (
            "frames",
            "camera",
            "depth_scale_m",
            "depth_sample",
            "tile_pixels",
            "root_sha256",
        ),
    )
    camera = _object(
        scan["camera"],
        "statement.scan.camera",
        ("width", "height", "fx", "fy", "cx", "cy"),
    )
    _integer(scan["frames"], "statement.scan.frames", minimum=1)
    _integer(scan["tile_pixels"], "statement.scan.tile_pixels", minimum=1)
    _integer(camera["width"], "statement.scan.camera.width", minimum=1)
    _integer(camera["height"], "statement.scan.camera.height", minimum=1)
    for key in ("fx", "fy", "cx", "cy"):
        if type(camera[key]) is not float:
            raise TsdfError(
                f"statement.scan.camera.{key}: expected a number"
            )
    _positive_float(scan["depth_scale_m"], "statement.scan.depth_scale_m")
    _exact(scan["depth_sample"], _SAMPLE, "statement.scan.depth_sample")
    _sha256(scan["root_sha256"], "statement.scan.root_sha256")
    volume = _object(
        root["map"],
        "statement.map",
        (
            "voxel_size_m",
            "truncation_m",
            "block_resolution",
            "frame_stride",
            "blocks",
            "root_sha256",
        ),
    )
    _positive_float(volume["voxel_size_m"], "statement.map.voxel_size_m")
    _positive_float(volume["truncation_m"], "statement.map.truncation_m")
    _exact(
        volume["block_resolution"],
        TSDF_BLOCK_RESOLUTION,
        "statement.map.block_resolution",
    )
    _integer(volume["frame_stride"], "statement.map.frame_stride", minimum=1)
    _integer(volume["blocks"], "statement.map.blocks", minimum=1)
    _sha256(volume["root_sha256"], "statement.map.root_sha256")
    return TsdfMapStatement(
        session_id=root["session_id"],
        rule=root["rule"],
        frame_count=scan["frames"],
        camera=TsdfPinholeCamera(
            width=camera["width"],
            height=camera["height"],
            fx=camera["fx"],
            fy=camera["fy"],
            cx=camera["cx"],
            cy=camera["cy"],
        ),
        depth_scale_m=scan["depth_scale_m"],
        tile_pixels=scan["tile_pixels"],
        scan_root_sha256=scan["root_sha256"],
        voxel_size_m=volume["voxel_size_m"],
        truncation_m=volume["truncation_m"],
        frame_stride=volume["frame_stride"],
        block_count=volume["blocks"],
        map_root_sha256=volume["root_sha256"],
    )
