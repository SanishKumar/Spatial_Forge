"""Check one block's proof with nothing but Python's standard library.

SpatialForge proves one block of a map to someone who holds only the
map's statement: docs/tsdf-block-proof.md. The package checks such a
proof itself, and to do it fuses the block again with the evaluator that
fused the map. That is one program checking its own work.

This file is a second check. It imports nothing of the package and
nothing outside Python's standard library, so it can be read whole and
run where neither is installed:

    python checker/check_block_proof.py map.sftcommit block.sftproof

It prints what it checked, and exits 0 if the block is verified, 1 if it
is refuted or the proof is invalid, and 2 if a file cannot be read as a
statement or a proof. A packed proof is read as the proof it carries.

It was written from the documents, and from the package's evaluator for
the one thing they did not say: the order of the floating-point
operations, on which agreeing to the bit depends.
docs/stand-alone-checker.md now says that too.

It fuses differently from the package on purpose. The package first
rules frames out by the corners of the block and evaluates voxels only
for the rest; this evaluates every voxel of every frame that has depth
and a pose. The two have to agree, voxel for voxel and bit for bit.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
import sys
import zlib

RULE = "projective-tsdf/unit-weight/nearest-pixel/1"
CHECKS = (
    "statement",
    "shape",
    "scan root",
    "map root",
    "depth roots",
    "poses",
    "pixels",
    "fusion",
)

SIDE = 8
VOXELS = SIDE**3
HASH_BYTES = 32
FRAME_LEAF_BYTES = 8 + 8 + 1 + 16 * 8 + HASH_BYTES
BLOCK_LEAF_BYTES = 8 + 12 + VOXELS * 8 + VOXELS * 4
MOST_STATEMENT_BYTES = 16 * 1024
MOST_HEADER_BYTES = 16 * 1024
MOST_PROOF_BYTES = 1024**3
MOST_TILE_PIXELS = 1024
RIGID_TOLERANCE = 1e-5

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_SESSION = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")


class Unreadable(Exception):
    """A file that is not a statement, or not a proof."""


class _NotHeld(Exception):
    """A pixel the block reads that the proof does not hold."""


# RFC 6962 Merkle trees.


def leaf_hash(data: bytes) -> bytes:
    return hashlib.sha256(b"\x00" + data).digest()


def node_hash(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(b"\x01" + left + right).digest()


def split(size: int) -> int:
    """The largest power of two below ``size``, which is at least two."""

    power = 1
    while power * 2 < size:
        power *= 2
    return power


def tree_root(hashes: list[bytes]) -> bytes:
    if not hashes:
        return hashlib.sha256(b"").digest()
    if len(hashes) == 1:
        return hashes[0]
    middle = split(len(hashes))
    return node_hash(tree_root(hashes[:middle]), tree_root(hashes[middle:]))


def path_sides(index: int, size: int) -> list[bool]:
    """From the root down, whether each step's other half is to the right."""

    sides = []
    while size > 1:
        middle = split(size)
        if index < middle:
            sides.append(True)
            size = middle
        else:
            sides.append(False)
            index -= middle
            size -= middle
    return sides


def root_from_path(
    hash_: bytes, index: int, size: int, path: list[bytes]
) -> bytes:
    """The root a leaf and its path climb to. The path starts at the leaf."""

    for other_is_right, other in zip(reversed(path_sides(index, size)), path):
        hash_ = node_hash(hash_, other) if other_is_right else node_hash(
            other, hash_
        )
    return hash_


def shared_proof_length(places: list[int], size: int) -> int:
    """How many hashes the proof shared by leaves at these places holds."""

    def count(low: int, high: int, chosen: list[int]) -> int:
        if not chosen:
            return 1
        if high - low == 1:
            return 0
        middle = low + split(high - low)
        left = [place for place in chosen if place < middle]
        return count(low, middle, left) + count(
            middle, high, chosen[len(left):]
        )

    return count(0, size, places)


def root_from_shared_proof(
    hashes: list[bytes], places: list[int], size: int, proof: list[bytes]
) -> bytes:
    """The root leaves at these places and the proof they share lead to.

    The proof is the hashes of the largest subtrees that hold none of the
    leaves, left to right.
    """

    remaining = iter(proof)

    def root(low: int, high: int, chosen: list[tuple[int, bytes]]) -> bytes:
        if not chosen:
            return next(remaining)
        if high - low == 1:
            return chosen[0][1]
        middle = low + split(high - low)
        left = [leaf for leaf in chosen if leaf[0] < middle]
        left_root = root(low, middle, left)
        return node_hash(left_root, root(middle, high, chosen[len(left):]))

    return root(0, size, list(zip(places, hashes)))


# Reading the two files.


def unpacked(data: bytes) -> tuple[bytes, bool]:
    """A file's bytes as a proof's, and whether they came packed."""

    if not data.startswith(b"SFTPACK1"):
        return data, False
    if len(data) < 16:
        raise Unreadable("the packed proof ends before its length")
    (declared,) = struct.unpack_from("<Q", data, 8)
    if not 1 <= declared <= MOST_PROOF_BYTES:
        raise Unreadable(f"the packed proof declares {declared} bytes")
    inflater = zlib.decompressobj()
    try:
        carried = inflater.decompress(memoryview(data)[16:], declared)
        if not inflater.eof and inflater.decompress(
            inflater.unconsumed_tail, 1
        ):
            raise Unreadable(
                f"the packed proof inflates to more than {declared} bytes"
            )
    except zlib.error as error:
        raise Unreadable(
            f"the packed proof cannot be inflated: {error}"
        ) from error
    if not inflater.eof:
        raise Unreadable("the packed proof is cut short")
    if inflater.unused_data:
        raise Unreadable("the packed proof has bytes after its end")
    if len(carried) != declared:
        raise Unreadable(
            f"the packed proof inflates to {len(carried)} bytes and "
            f"declares {declared}"
        )
    return carried, True


def _canonical(document: dict) -> bytes:
    return (
        json.dumps(
            document,
            ensure_ascii=True,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _json(data: bytes, what: str) -> dict:
    """A JSON object read strictly, and required in canonical form."""

    try:
        text = data.decode("ascii")
    except UnicodeError as error:
        raise Unreadable(f"{what} is not ASCII") from error
    if text.count("{") > 8 or "[" in text:
        raise Unreadable(f"{what} holds an array or is nested too deeply")

    def unique(pairs: list) -> dict:
        document = {}
        for key, value in pairs:
            if key in document:
                raise Unreadable(f"{what} names {key!r} twice")
            document[key] = value
        return document

    def no_constant(name: str) -> None:
        raise Unreadable(f"{what} holds {name}")

    try:
        document = json.loads(
            text, object_pairs_hook=unique, parse_constant=no_constant
        )
    except (ValueError, RecursionError) as error:
        raise Unreadable(f"{what} is not JSON: {error}") from error
    return document


def _fields(value: object, what: str, keys: tuple[str, ...]) -> dict:
    if not isinstance(value, dict) or set(value) != set(keys):
        raise Unreadable(
            f"{what} is not an object of exactly {', '.join(sorted(keys))}"
        )
    return value


def _whole(value: object, what: str, least: int) -> int:
    if type(value) is not int or value < least:
        raise Unreadable(f"{what} is not a whole number of at least {least}")
    return value


def _finite(value: object, what: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise Unreadable(f"{what} is not a finite number")
    return value


def _positive(value: object, what: str) -> float:
    if _finite(value, what) <= 0.0:
        raise Unreadable(f"{what} is not positive")
    return value


def _digest(value: object, what: str) -> str:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise Unreadable(f"{what} is not a SHA-256 digest")
    return value


def _exactly(value: object, expected: object, what: str) -> None:
    if type(value) is not type(expected) or value != expected:
        raise Unreadable(f"{what} is not {expected!r}")


class Statement:
    """What a statement says that a check of a block reads."""

    def __init__(self, data: bytes) -> None:
        if len(data) > MOST_STATEMENT_BYTES:
            raise Unreadable("the statement is too large to be one")
        document = _json(data, "the statement")
        root = _fields(
            document,
            "the statement",
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
        _exactly(
            root["schema"], "spatialforge.tsdf-map-statement", "its schema"
        )
        _exactly(root["schema_version"], "0.1.0", "its schema version")
        _exactly(root["hash"], "sha-256", "its hash")
        _exactly(root["tree"], "rfc6962", "its tree")
        if not isinstance(root["session_id"], str) or not _SESSION.fullmatch(
            root["session_id"]
        ):
            raise Unreadable("the statement's session_id is not one")
        if root["rule"] != RULE:
            raise Unreadable(
                f"the statement is made under the rule {root['rule']!r}, "
                f"and this checker knows only {RULE!r}"
            )
        scan = _fields(
            root["scan"],
            "its scan",
            (
                "frames",
                "camera",
                "depth_scale_m",
                "depth_sample",
                "tile_pixels",
                "root_sha256",
            ),
        )
        camera = _fields(
            scan["camera"],
            "its camera",
            ("width", "height", "fx", "fy", "cx", "cy"),
        )
        volume = _fields(
            root["map"],
            "its map",
            (
                "voxel_size_m",
                "truncation_m",
                "block_resolution",
                "frame_stride",
                "blocks",
                "root_sha256",
            ),
        )
        self.frames = _whole(scan["frames"], "its frames", 1)
        self.width = _whole(camera["width"], "its image width", 1)
        self.height = _whole(camera["height"], "its image height", 1)
        self.fx = _positive(camera["fx"], "its fx")
        self.fy = _positive(camera["fy"], "its fy")
        self.cx = _finite(camera["cx"], "its cx")
        self.cy = _finite(camera["cy"], "its cy")
        if not (0.0 <= self.cx < self.width and 0.0 <= self.cy < self.height):
            raise Unreadable("the statement's principal point is not inside")
        self.depth_scale_m = _positive(
            scan["depth_scale_m"], "its depth scale"
        )
        _exactly(scan["depth_sample"], "uint16-le", "its depth sample")
        self.tile_pixels = _whole(scan["tile_pixels"], "its tile size", 1)
        if self.tile_pixels > MOST_TILE_PIXELS:
            raise Unreadable("the statement's tile size is too large")
        self.scan_root = _digest(scan["root_sha256"], "its scan root")
        self.voxel_size_m = _positive(volume["voxel_size_m"], "its voxel")
        self.truncation_m = _positive(
            volume["truncation_m"], "its truncation"
        )
        if self.truncation_m < self.voxel_size_m:
            raise Unreadable("the statement truncates within a voxel")
        _exactly(volume["block_resolution"], SIDE, "its block resolution")
        self.frame_stride = _whole(volume["frame_stride"], "its stride", 1)
        self.blocks = _whole(volume["blocks"], "its blocks", 1)
        self.map_root = _digest(volume["root_sha256"], "its map root")
        try:
            in_form = _canonical(document) == data
        except ValueError:
            in_form = False
        if not in_form:
            raise Unreadable("the statement is not in canonical form")
        self.digest = hashlib.sha256(data).hexdigest()


class Frame:
    """One frame's leaf: its place, its pose and its depth root."""

    def __init__(self, leaf: bytes) -> None:
        if not leaf.startswith(b"SFFRAME1"):
            raise Unreadable("a frame's leaf does not start with its tag")
        self.leaf = leaf
        self.sequence, flags = struct.unpack_from("<QB", leaf, 8)
        pose = leaf[17:145]
        depth_root = leaf[145:]
        if flags > 3 or self.sequence >= 2**63:
            raise Unreadable("a frame's leaf holds an unknown flag or place")
        if not flags & 2 and pose != bytes(128):
            raise Unreadable("a frame's leaf with no pose holds one")
        if not flags & 1 and depth_root != bytes(HASH_BYTES):
            raise Unreadable("a frame's leaf with no depth holds a root")
        self.pose = struct.unpack("<16d", pose) if flags & 2 else None
        self.depth_root = depth_root if flags & 1 else None


class Tiles:
    """The tiles a proof holds of one frame, and the proof they share."""

    def __init__(self, sequence: int) -> None:
        self.sequence = sequence
        self.places: list[int] = []
        self.data: list[bytes] = []
        self.samples: dict[int, tuple[int, int, tuple[int, ...]]] = {}
        self.shared_proof: list[bytes] = []


class _Bytes:
    """A place in a proof's bytes, moved only forwards."""

    def __init__(self, data: bytes, at: int) -> None:
        self.data = data
        self.at = at

    def take(self, length: int, what: str) -> bytes:
        if self.at + length > len(self.data):
            raise Unreadable(f"the proof ends inside {what}")
        piece = self.data[self.at:self.at + length]
        self.at += length
        return piece

    def room(self, count: int, each: int, what: str) -> int:
        if count * each > len(self.data) - self.at:
            raise Unreadable(f"the proof counts more {what} than it holds")
        return count


class Proof:
    """A block's proof, read from front to back with nothing left over."""

    def __init__(self, data: bytes) -> None:
        if len(data) < 16 or not data.startswith(b"SFTPRF01"):
            raise Unreadable("the file is not a block proof")
        (header_length,) = struct.unpack_from("<Q", data, 8)
        if not 2 <= header_length <= MOST_HEADER_BYTES:
            raise Unreadable("the proof's header length is out of range")
        reader = _Bytes(data, 16)
        header_bytes = reader.take(header_length, "its header")
        header = _fields(
            _json(header_bytes, "the proof's header"),
            "the proof's header",
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
        _exactly(
            header["schema"], "spatialforge.tsdf-block-proof", "its schema"
        )
        _exactly(header["schema_version"], "0.1.0", "its schema version")
        self.statement = _digest(header["statement_sha256"], "its statement")
        frame_count = _whole(header["frames"], "its frames", 1)
        tile_frame_count = _whole(header["tile_frames"], "its frames read", 0)
        image = _fields(
            header["image"], "its image", ("width", "height", "tile_pixels")
        )
        self.width = _whole(image["width"], "its image width", 1)
        self.height = _whole(image["height"], "its image height", 1)
        self.tile_pixels = _whole(image["tile_pixels"], "its tile size", 1)
        block = _fields(header["block"], "its block", ("blocks", "row"))
        self.blocks = _whole(block["blocks"], "its blocks", 1)
        self.row = _whole(block["row"], "its row", 0)
        if _canonical(header) != header_bytes:
            raise Unreadable("the proof's header is not in canonical form")

        self.frames = [
            Frame(reader.take(FRAME_LEAF_BYTES, "a frame's leaf"))
            for _ in range(
                reader.room(frame_count, FRAME_LEAF_BYTES, "frames")
            )
        ]
        self.block_leaf = reader.take(BLOCK_LEAF_BYTES, "its block")
        if not self.block_leaf.startswith(b"SFBLOCK1"):
            raise Unreadable("the proof's block does not start with its tag")
        self.index = struct.unpack_from("<3i", self.block_leaf, 8)
        self.sums = self.block_leaf[20:20 + VOXELS * 8]
        self.weights = struct.unpack_from(
            f"<{VOXELS}I", self.block_leaf, 20 + VOXELS * 8
        )
        if self.row >= self.blocks:
            raise Unreadable("the proof's row is not among its blocks")
        self.path = [
            reader.take(HASH_BYTES, "its block's path")
            for _ in path_sides(self.row, self.blocks)
        ]
        if self.tile_pixels > MOST_TILE_PIXELS:
            raise Unreadable("the proof's tile size is too large")
        self.tile_rows = -(-self.height // self.tile_pixels)
        self.tile_columns = -(-self.width // self.tile_pixels)
        tiles_in_a_frame = self.tile_rows * self.tile_columns
        self.tile_frames: list[Tiles] = []
        for _ in range(reader.room(tile_frame_count, 12, "frames read")):
            sequence, count = struct.unpack(
                "<QI", reader.take(12, "a frame's tiles")
            )
            tiles = Tiles(sequence)
            for _ in range(reader.room(count, 8, "tiles")):
                tile_row, tile_column = struct.unpack(
                    "<II", reader.take(8, "a tile")
                )
                if not (
                    tile_row < self.tile_rows
                    and tile_column < self.tile_columns
                ):
                    raise Unreadable("a tile lies outside the image")
                place = tile_row * self.tile_columns + tile_column
                if tiles.places and place <= tiles.places[-1]:
                    raise Unreadable("a frame's tiles are out of order")
                rows = min(
                    self.tile_pixels, self.height - tile_row * self.tile_pixels
                )
                columns = min(
                    self.tile_pixels,
                    self.width - tile_column * self.tile_pixels,
                )
                raw = reader.take(2 * rows * columns, "a tile's samples")
                tiles.places.append(place)
                tiles.data.append(raw)
                tiles.samples[place] = (
                    rows,
                    columns,
                    struct.unpack(f"<{rows * columns}H", raw),
                )
            if not tiles.places:
                raise Unreadable("the proof names a frame and no tile of it")
            tiles.shared_proof = [
                reader.take(HASH_BYTES, "a frame's shared proof")
                for _ in range(
                    shared_proof_length(tiles.places, tiles_in_a_frame)
                )
            ]
            self.tile_frames.append(tiles)
        if reader.at != len(data):
            raise Unreadable(
                f"the proof holds {len(data) - reader.at} bytes after its end"
            )

        if [frame.sequence for frame in self.frames] != list(
            range(frame_count)
        ):
            raise Unreadable("the proof does not hold every frame, in order")
        previous = -1
        for tiles in self.tile_frames:
            if tiles.sequence <= previous or tiles.sequence >= frame_count:
                raise Unreadable("the proof's frames read are not in order")
            previous = tiles.sequence
        self.digest = hashlib.sha256(data).hexdigest()


# The check.


class Outcome:
    """What a check found."""

    def __init__(self) -> None:
        self.verdict = "verified"
        self.failed_check: str | None = None
        self.detail: str | None = None
        self.differing_voxels = 0
        self.ready = 0
        self.read = 0
        self.contributing = 0

    def fail(self, check: str, detail: str) -> "Outcome":
        """Invalid at this check. A block that was not fused counts none."""

        self.verdict = "invalid"
        self.failed_check = check
        self.detail = detail
        self.ready = self.read = self.contributing = 0
        return self


def rigid(pose: tuple[float, ...]) -> bool:
    """Whether a pose is a rigid transform, to one part in 100,000."""

    if any(
        abs(actual - expected) > RIGID_TOLERANCE
        for actual, expected in zip(pose[12:16], (0.0, 0.0, 0.0, 1.0))
    ):
        return False
    rotation = (pose[0:3], pose[4:7], pose[8:11])
    for one in range(3):
        for other in range(3):
            dot = sum(rotation[one][axis] * rotation[other][axis]
                      for axis in range(3))
            if abs(dot - (1.0 if one == other else 0.0)) > RIGID_TOLERANCE:
                return False
    (a, b, c), (d, e, f), (g, h, i) = rotation
    determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (
        d * h - e * g
    )
    return abs(determinant - 1.0) <= RIGID_TOLERANCE


def check(statement: Statement, proof: Proof) -> Outcome:
    """Hold a proof to a statement, one check after another."""

    outcome = Outcome()
    selected = proof.frames[::statement.frame_stride]

    if proof.statement != statement.digest:
        return outcome.fail(
            "statement", f"the proof names statement {proof.statement}"
        )

    for label, held, stated in (
        ("frames", len(proof.frames), statement.frames),
        ("image width", proof.width, statement.width),
        ("image height", proof.height, statement.height),
        ("tile size", proof.tile_pixels, statement.tile_pixels),
        ("blocks", proof.blocks, statement.blocks),
    ):
        if held != stated:
            return outcome.fail(
                "shape",
                f"the proof has {label} {held}; the statement {stated}",
            )

    scan_root = tree_root([leaf_hash(frame.leaf) for frame in proof.frames])
    if scan_root.hex() != statement.scan_root:
        return outcome.fail(
            "scan root", f"the frames' leaves lead to {scan_root.hex()}"
        )

    map_root = root_from_path(
        leaf_hash(proof.block_leaf), proof.row, proof.blocks, proof.path
    )
    if map_root.hex() != statement.map_root:
        return outcome.fail(
            "map root", f"the block and its path lead to {map_root.hex()}"
        )

    tiles_in_a_frame = proof.tile_rows * proof.tile_columns
    for tiles in proof.tile_frames:
        depth_root = proof.frames[tiles.sequence].depth_root
        if depth_root is None:
            return outcome.fail(
                "depth roots",
                f"frame {tiles.sequence} has no depth, and the proof holds "
                "tiles of it",
            )
        found = root_from_shared_proof(
            [leaf_hash(b"SFTILE01" + data) for data in tiles.data],
            tiles.places,
            tiles_in_a_frame,
            tiles.shared_proof,
        )
        if found != depth_root:
            return outcome.fail(
                "depth roots",
                f"frame {tiles.sequence}'s tiles do not lead to its depth "
                "root",
            )

    for frame in selected:
        if frame.pose is None:
            continue
        if not all(math.isfinite(value) for value in frame.pose) or (
            frame.depth_root is not None and not rigid(frame.pose)
        ):
            return outcome.fail(
                "poses",
                f"frame {frame.sequence} has a pose that is not a rigid "
                "transform",
            )

    posed = {frame.sequence for frame in selected if frame.pose is not None}
    for tiles in proof.tile_frames:
        if tiles.sequence not in posed:
            return outcome.fail(
                "pixels",
                f"frame {tiles.sequence} is not fused, and the proof holds "
                "tiles of it",
            )
    try:
        sums, weights, asked = _fuse(statement, proof, selected, outcome)
    except _NotHeld as missing:
        return outcome.fail("pixels", str(missing))
    for tiles in proof.tile_frames:
        for place in tiles.places:
            if place not in asked.get(tiles.sequence, ()):
                return outcome.fail(
                    "pixels",
                    f"frame {tiles.sequence}: the proof holds tile "
                    f"{place}, and the block reads no pixel of it",
                )

    held_sums = struct.unpack(f"<{VOXELS}Q", proof.sums)
    fused_sums = struct.unpack(
        f"<{VOXELS}Q", struct.pack(f"<{VOXELS}d", *sums)
    )
    outcome.differing_voxels = sum(
        1
        for voxel in range(VOXELS)
        if held_sums[voxel] != fused_sums[voxel]
        or proof.weights[voxel] != weights[voxel]
    )
    if outcome.differing_voxels:
        outcome.verdict = "refuted"
        outcome.failed_check = "fusion"
        outcome.detail = (
            f"{outcome.differing_voxels} of {VOXELS} voxels are not what the "
            "scan gives"
        )
    return outcome


def _fuse(
    statement: Statement,
    proof: Proof,
    selected: list[Frame],
    outcome: Outcome,
) -> tuple[list[float], list[int], dict[int, set[int]]]:
    """The block fused again, frame by frame, from the tiles alone.

    Every voxel of every frame with depth and a pose is evaluated: its
    centre is taken into the camera, projected, and read at its nearest
    pixel. Each step is one IEEE operation on doubles, in the order
    docs/stand-alone-checker.md writes out.
    """

    voxel = statement.voxel_size_m
    truncation = statement.truncation_m
    width, height = statement.width, statement.height
    fx, fy, cx, cy = statement.fx, statement.fy, statement.cx, statement.cy
    tile = proof.tile_pixels
    block_x, block_y, block_z = proof.index
    centres = [
        (
            (float(flat % SIDE + block_x * SIDE) + 0.5) * voxel,
            (float(flat // SIDE % SIDE + block_y * SIDE) + 0.5) * voxel,
            (float(flat // (SIDE * SIDE) + block_z * SIDE) + 0.5) * voxel,
        )
        for flat in range(VOXELS)
    ]
    if not all(math.isfinite(axis) for centre in centres for axis in centre):
        raise _NotHeld("a voxel centre of the block is not finite")
    held = {tiles.sequence: tiles for tiles in proof.tile_frames}
    sums = [0.0] * VOXELS
    weights = [0] * VOXELS
    asked: dict[int, set[int]] = {}

    for frame in selected:
        if frame.pose is None or frame.depth_root is None:
            continue
        outcome.ready += 1
        m = frame.pose
        landed = []
        for flat, (x, y, z) in enumerate(centres):
            dx, dy, dz = x - m[3], y - m[7], z - m[11]
            camera_x = m[0] * dx + m[4] * dy + m[8] * dz
            camera_y = m[1] * dx + m[5] * dy + m[9] * dz
            camera_z = m[2] * dx + m[6] * dy + m[10] * dz
            if not (
                math.isfinite(camera_x)
                and math.isfinite(camera_y)
                and math.isfinite(camera_z)
            ) or camera_z <= 0.0:
                continue
            u = fx * camera_x / camera_z + cx
            v = fy * camera_y / camera_z + cy
            if not (math.isfinite(u) and math.isfinite(v)):
                continue
            if not (-0.5 <= u < width - 0.5 and -0.5 <= v < height - 0.5):
                continue
            column = math.floor(u + 0.5)
            row = math.floor(v + 0.5)
            if not (0 <= column < width and 0 <= row < height):
                continue
            landed.append((flat, row, column, camera_z))
        if not landed:
            continue
        places = {
            row // tile * proof.tile_columns + column // tile
            for _, row, column, _ in landed
        }
        asked[frame.sequence] = places
        tiles = held.get(frame.sequence)
        missing = sorted(
            place
            for place in places
            if tiles is None or place not in tiles.samples
        )
        if missing:
            row, column = next(
                (row, column)
                for _, row, column, _ in landed
                if tiles is None
                or row // tile * proof.tile_columns + column // tile
                == missing[0]
            )
            raise _NotHeld(
                f"frame {frame.sequence}: the block reads the depth pixel at "
                f"row {row}, column {column}, and the proof holds no tile "
                "for it"
            )
        contributed = False
        for flat, row, column, camera_z in landed:
            _, tile_width, samples = tiles.samples[
                row // tile * proof.tile_columns + column // tile
            ]
            depth = float(
                samples[(row % tile) * tile_width + column % tile]
            ) * statement.depth_scale_m
            if not (math.isfinite(depth) and depth > 0.0):
                continue
            distance = depth - camera_z
            if not math.isfinite(distance) or distance < -truncation:
                continue
            sums[flat] += max(-1.0, min(1.0, distance / truncation))
            weights[flat] += 1
            contributed = True
        outcome.read += 1
        outcome.contributing += contributed
    return sums, weights, asked


# The command.


def _read(path: str, most: int) -> bytes:
    with open(path, "rb") as file:
        data = file.read(most + 1)
    if len(data) > most:
        raise Unreadable(f"{path} is larger than {most} bytes")
    return data


def main(arguments: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if arguments is None else arguments
    if len(arguments) != 2:
        print(
            "usage: check_block_proof.py STATEMENT.sftcommit PROOF.sftproof",
            file=sys.stderr,
        )
        return 2
    statement_path, proof_path = arguments
    try:
        statement = Statement(_read(statement_path, MOST_STATEMENT_BYTES))
    except (OSError, Unreadable) as error:
        print(f"CANNOT CHECK {statement_path}", file=sys.stderr)
        print(f"- {error}", file=sys.stderr)
        return 2
    try:
        data, packed = unpacked(_read(proof_path, MOST_PROOF_BYTES))
        proof = Proof(data)
    except (OSError, Unreadable) as error:
        print(f"CANNOT CHECK {proof_path}", file=sys.stderr)
        print(f"- {error}", file=sys.stderr)
        return 2

    outcome = check(statement, proof)
    x, y, z = proof.index
    print("BLOCK PROOF CHECK, STANDARD LIBRARY ONLY")
    print(f"statement_sha256: {statement.digest}")
    print(f"proof_sha256: {proof.digest}")
    print(f"packed: {'yes' if packed else 'no'}")
    print(f"block: {x},{y},{z} row={proof.row} of={proof.blocks}")
    reached = True
    for name in CHECKS:
        if name == outcome.failed_check:
            print(f"check {name}: failed")
            reached = False
        else:
            print(f"check {name}: {'passed' if reached else 'not reached'}")
    if outcome.detail is not None:
        print(f"detail: {outcome.detail}")
    if outcome.failed_check in (None, "fusion"):
        print(
            f"fusion: ready={outcome.ready} read={outcome.read} "
            f"contributing={outcome.contributing} "
            f"differing_voxels={outcome.differing_voxels}"
        )
    print(f"verdict: {outcome.verdict}")
    return 0 if outcome.verdict == "verified" else 1


if __name__ == "__main__":
    sys.exit(main())
