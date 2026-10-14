"""Check a map's proofs with nothing but Python's standard library.

SpatialForge proves one block of a map, or a route through it, to someone
who holds only the map's statement: docs/tsdf-block-proof.md and
docs/tsdf-route-certificate.md. The package checks those itself, and to
do it fuses blocks again with the evaluator that fused the map. That is
one program checking its own work.

This file is a second check. It imports nothing of the package and
nothing outside Python's standard library, so it can be read whole and
run where neither is installed:

    python checker/check.py map.sftcommit block.sftproof
    python checker/check.py map.sftcommit way.sftroute

It tells a proof from a certificate by its first bytes, prints what it
checked, and exits 0 if the block is verified or the route is clear, 1
for any other verdict, and 2 if a file cannot be read as a statement, a
proof or a certificate. A packed file is read as what it carries.

It was written from the documents, and from the package's evaluator for
the one thing they did not say: the order of the floating-point
operations, on which agreeing to the bit depends.
docs/stand-alone-checker.md now says that too.

It fuses differently from the package on purpose. The package first
rules frames out by the corners of each block and evaluates voxels only
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
ROUTE_CHECKS = (
    "statement",
    "shape",
    "route",
    "scan root",
    "map root",
    "depth roots",
    "poses",
    "pixels",
    "fusion",
    "clearance",
)

SIDE = 8
VOXELS = SIDE**3
HASH_BYTES = 32
FRAME_LEAF_BYTES = 8 + 8 + 1 + 16 * 8 + HASH_BYTES
BLOCK_LEAF_BYTES = 8 + 12 + VOXELS * 8 + VOXELS * 4
MOST_STATEMENT_BYTES = 16 * 1024
MOST_HEADER_BYTES = 16 * 1024
MOST_PROOF_BYTES = 1024**3
MOST_CERTIFICATE_BYTES = 2 * 1024**3
MOST_TILE_PIXELS = 1024
RIGID_TOLERANCE = 1e-5

# What a route may state, and what it may come to.
MOST_VOXEL_INDEX = 2**30
MOST_ROUTE_RADIUS = 64
MOST_ROUTE_WAYPOINTS = 4096
MOST_ROUTE_CHAIN_COLUMNS = 65_536
MOST_ROUTE_SWEPT_PAIRS = 4_000_000
MOST_ROUTE_BLOCKS = 65_536
AXES = ("x", "y", "z")

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_SESSION = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_RULE = re.compile(r"^[a-z0-9][a-z0-9._/-]{0,127}$")


class Unreadable(Exception):
    """A file that is not a statement, a proof or a certificate."""


class _NotHeld(Exception):
    """A pixel a block reads that the file does not hold."""


class _NotARoute(Exception):
    """A route that cannot be worked out within its limits."""


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


# Reading the files.


def unpacked(data: bytes) -> tuple[bytes, bool]:
    """A file's bytes as a proof's, and whether they came packed."""

    if not data.startswith(b"SFTPACK1"):
        return data, False
    if len(data) < 16:
        raise Unreadable("the packed file ends before its length")
    (declared,) = struct.unpack_from("<Q", data, 8)
    if not 1 <= declared <= MOST_CERTIFICATE_BYTES:
        raise Unreadable(f"the packed file declares {declared} bytes")
    inflater = zlib.decompressobj()
    try:
        carried = inflater.decompress(memoryview(data)[16:], declared)
        if not inflater.eof and inflater.decompress(
            inflater.unconsumed_tail, 1
        ):
            raise Unreadable(
                f"the packed file inflates to more than {declared} bytes"
            )
    except zlib.error as error:
        raise Unreadable(
            f"the packed file cannot be inflated: {error}"
        ) from error
    if not inflater.eof:
        raise Unreadable("the packed file is cut short")
    if inflater.unused_data:
        raise Unreadable("the packed file has bytes after its end")
    if len(carried) != declared:
        raise Unreadable(
            f"the packed file inflates to {len(carried)} bytes and "
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
    """A JSON object read strictly."""

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
        if not isinstance(root["rule"], str) or not _RULE.fullmatch(
            root["rule"]
        ):
            raise Unreadable("the statement's rule is not the name of one")
        self.rule = root["rule"]
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


class Block:
    """One block as the map holds it, at its row, with its path."""

    def __init__(self, row: int, leaf: bytes, path: list[bytes]) -> None:
        self.row = row
        self.leaf = leaf
        self.index = struct.unpack_from("<3i", leaf, 8)
        self.sums = leaf[20:20 + VOXELS * 8]
        self.weights = struct.unpack_from(f"<{VOXELS}I", leaf, 20 + VOXELS * 8)
        self.path = path


class Tiles:
    """The tiles a file holds of one frame, and the proof they share."""

    def __init__(self, sequence: int) -> None:
        self.sequence = sequence
        self.places: list[int] = []
        self.data: list[bytes] = []
        self.samples: dict[int, tuple[int, int, tuple[int, ...]]] = {}
        self.shared_proof: list[bytes] = []


class _Bytes:
    """A place in a file's bytes, moved only forwards."""

    def __init__(self, data: bytes, at: int, name: str) -> None:
        self.data = data
        self.at = at
        self.name = name

    def take(self, length: int, what: str) -> bytes:
        if self.at + length > len(self.data):
            raise Unreadable(f"the {self.name} ends inside {what}")
        piece = self.data[self.at:self.at + length]
        self.at += length
        return piece

    def room(self, count: int, each: int, what: str) -> int:
        if count * each > len(self.data) - self.at:
            raise Unreadable(
                f"the {self.name} counts more {what} than it holds"
            )
        return count

    def finish(self) -> None:
        if self.at != len(self.data):
            raise Unreadable(
                f"the {self.name} holds {len(self.data) - self.at} bytes "
                "after its end"
            )


class _Holder:
    """What a proof and a certificate both hold: frames, blocks, tiles."""

    name = "file"

    def _start(
        self, data: bytes, magic: bytes, kind: str, keys: tuple[str, ...]
    ) -> tuple[_Bytes, dict, bytes]:
        if len(data) < 16 or not data.startswith(magic):
            raise Unreadable(f"the file is not a {kind}")
        (header_length,) = struct.unpack_from("<Q", data, 8)
        if not 2 <= header_length <= MOST_HEADER_BYTES:
            raise Unreadable(
                f"the {self.name}'s header length is out of range"
            )
        reader = _Bytes(data, 16, self.name)
        header_bytes = reader.take(header_length, "its header")
        what = f"the {self.name}'s header"
        header = _fields(_json(header_bytes, what), what, keys)
        self.statement = _digest(header["statement_sha256"], "its statement")
        return reader, header, header_bytes

    def _image(self, header: dict) -> None:
        image = _fields(
            header["image"], "its image", ("width", "height", "tile_pixels")
        )
        self.width = _whole(image["width"], "its image width", 1)
        self.height = _whole(image["height"], "its image height", 1)
        self.tile_pixels = _whole(image["tile_pixels"], "its tile size", 1)

    def _frames(self, reader: _Bytes, count: int) -> None:
        self.frames = [
            Frame(reader.take(FRAME_LEAF_BYTES, "a frame's leaf"))
            for _ in range(reader.room(count, FRAME_LEAF_BYTES, "frames"))
        ]

    def _block(self, reader: _Bytes, row: int) -> Block:
        leaf = reader.take(BLOCK_LEAF_BYTES, "a block")
        if not leaf.startswith(b"SFBLOCK1"):
            raise Unreadable(
                f"a {self.name}'s block does not start with its tag"
            )
        if row >= self.of:
            raise Unreadable(f"a {self.name}'s row is not among its blocks")
        return Block(
            row,
            leaf,
            [
                reader.take(HASH_BYTES, "a block's path")
                for _ in path_sides(row, self.of)
            ],
        )

    def _tiles(self, reader: _Bytes, count: int) -> None:
        if self.tile_pixels > MOST_TILE_PIXELS:
            raise Unreadable(f"the {self.name}'s tile size is too large")
        self.tile_rows = -(-self.height // self.tile_pixels)
        self.tile_columns = -(-self.width // self.tile_pixels)
        tiles_in_a_frame = self.tile_rows * self.tile_columns
        self.tile_frames: list[Tiles] = []
        for _ in range(reader.room(count, 12, "frames read")):
            sequence, held = struct.unpack(
                "<QI", reader.take(12, "a frame's tiles")
            )
            tiles = Tiles(sequence)
            for _ in range(reader.room(held, 8, "tiles")):
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
                raise Unreadable(
                    f"the {self.name} names a frame and no tile of it"
                )
            tiles.shared_proof = [
                reader.take(HASH_BYTES, "a frame's shared proof")
                for _ in range(
                    shared_proof_length(tiles.places, tiles_in_a_frame)
                )
            ]
            self.tile_frames.append(tiles)

    def _in_order(self, data: bytes) -> None:
        if [frame.sequence for frame in self.frames] != list(
            range(len(self.frames))
        ):
            raise Unreadable(
                f"the {self.name} does not hold every frame, in order"
            )
        previous = -1
        for tiles in self.tile_frames:
            if (
                tiles.sequence <= previous
                or tiles.sequence >= len(self.frames)
            ):
                raise Unreadable(
                    f"the {self.name}'s frames read are not in order"
                )
            previous = tiles.sequence
        self.digest = hashlib.sha256(data).hexdigest()


class Proof(_Holder):
    """A block's proof, read from front to back with nothing left over."""

    name = "proof"

    def __init__(self, data: bytes) -> None:
        reader, header, header_bytes = self._start(
            data,
            b"SFTPRF01",
            "block proof",
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
        frame_count = _whole(header["frames"], "its frames", 1)
        tile_frame_count = _whole(header["tile_frames"], "its frames read", 0)
        self._image(header)
        block = _fields(header["block"], "its block", ("blocks", "row"))
        self.of = _whole(block["blocks"], "its blocks", 1)
        row = _whole(block["row"], "its row", 0)
        if _canonical(header) != header_bytes:
            raise Unreadable("the proof's header is not in canonical form")
        self._frames(reader, frame_count)
        self.held = [self._block(reader, row)]
        self._tiles(reader, tile_frame_count)
        reader.finish()
        self._in_order(data)


class Route:
    """A corridor through a map, in whole voxels."""

    def __init__(
        self,
        up_axis: int,
        band_first: int,
        band_last: int,
        radius: int,
        min_weight: int,
        waypoints: list[tuple[int, int]],
    ) -> None:
        if any(
            abs(value) > MOST_VOXEL_INDEX for value in (band_first, band_last)
        ):
            raise Unreadable("a route's band lies too far from zero")
        if band_last < band_first:
            raise Unreadable("a route's band holds no voxel")
        if not 0 <= radius <= MOST_ROUTE_RADIUS:
            raise Unreadable("a route's radius is out of range")
        if not 1 <= min_weight < 2**32:
            raise Unreadable("a route's count of observations is out of range")
        if not 1 <= len(waypoints) <= MOST_ROUTE_WAYPOINTS:
            raise Unreadable("a route has too many waypoints")
        for place, waypoint in enumerate(waypoints):
            if any(abs(value) > MOST_VOXEL_INDEX for value in waypoint):
                raise Unreadable("a route's waypoint lies too far from zero")
            if place and waypoint == waypoints[place - 1]:
                raise Unreadable("a route's waypoint repeats the one before")
        if 1 + sum(
            max(abs(to_a - a), abs(to_b - b))
            for (a, b), (to_a, to_b) in zip(waypoints, waypoints[1:])
        ) > MOST_ROUTE_CHAIN_COLUMNS:
            raise Unreadable("a route's line runs through too many columns")
        self.up = up_axis
        self.across = tuple(axis for axis in range(3) if axis != up_axis)
        self.band_first = band_first
        self.band_last = band_last
        self.radius = radius
        self.min_weight = min_weight
        self.waypoints = waypoints

    def chain(self) -> list[tuple[int, int]]:
        """The columns the line runs through: Bresenham's, in integers."""

        columns = [self.waypoints[0]]
        for (a, b), (to_a, to_b) in zip(self.waypoints, self.waypoints[1:]):
            run_a, run_b = abs(to_a - a), abs(to_b - b)
            step_a = 1 if to_a > a else -1
            step_b = 1 if to_b > b else -1
            error = run_a - run_b
            while (a, b) != (to_a, to_b):
                doubled = 2 * error
                if doubled > -run_b:
                    error -= run_b
                    a += step_a
                if doubled < run_a:
                    error += run_a
                    b += step_b
                columns.append((a, b))
        return columns

    def columns(self) -> list[tuple[int, int]]:
        """Every column within the radius of the line, each once, in order."""

        reach = range(-self.radius, self.radius + 1)
        disc = [
            (da, db)
            for da in reach
            for db in reach
            if da * da + db * db <= self.radius * self.radius
        ]
        chain = self.chain()
        if len(chain) * len(disc) > MOST_ROUTE_SWEPT_PAIRS:
            raise _NotARoute(
                "a route this long and this wide sweeps more columns than "
                "a certificate is made for"
            )
        return sorted(
            {(a + da, b + db) for a, b in chain for da, db in disc}
        )

    def blocks(
        self, columns: list[tuple[int, int]]
    ) -> list[tuple[int, int, int]]:
        """Every block holding a voxel of the route, in z, y, x order."""

        column_blocks = sorted({(a // SIDE, b // SIDE) for a, b in columns})
        levels = range(self.band_first // SIDE, self.band_last // SIDE + 1)
        if len(column_blocks) * len(levels) > MOST_ROUTE_BLOCKS:
            raise _NotARoute(
                f"the route touches {len(column_blocks) * len(levels)} "
                f"blocks; a certificate holds at most {MOST_ROUTE_BLOCKS}"
            )
        touched = []
        for a, b in column_blocks:
            for level in levels:
                index = [0, 0, 0]
                index[self.across[0]] = a
                index[self.across[1]] = b
                index[self.up] = level
                touched.append(tuple(index))
        return sorted(touched, key=lambda index: index[::-1])


class Certificate(_Holder):
    """A route's certificate, read from front to back."""

    name = "certificate"

    def __init__(self, data: bytes) -> None:
        reader, header, header_bytes = self._start(
            data,
            b"SFTRTE01",
            "route certificate",
            (
                "schema",
                "schema_version",
                "statement_sha256",
                "frames",
                "image",
                "route",
                "blocks",
                "tile_frames",
            ),
        )
        _exactly(
            header["schema"],
            "spatialforge.tsdf-route-certificate",
            "its schema",
        )
        _exactly(header["schema_version"], "0.1.0", "its schema version")
        frame_count = _whole(header["frames"], "its frames", 1)
        tile_frame_count = _whole(header["tile_frames"], "its frames read", 0)
        self._image(header)
        route = _fields(
            header["route"],
            "its route",
            (
                "up_axis",
                "band_first",
                "band_last",
                "radius",
                "min_weight",
                "waypoints",
            ),
        )
        if route["up_axis"] not in AXES:
            raise Unreadable("the route's up axis is not x, y or z")
        band_first = _whole(
            route["band_first"], "its band", -MOST_VOXEL_INDEX
        )
        band_last = _whole(route["band_last"], "its band", -MOST_VOXEL_INDEX)
        radius = _whole(route["radius"], "its radius", 0)
        min_weight = _whole(route["min_weight"], "its observations", 1)
        waypoints = _whole(route["waypoints"], "its waypoints", 1)
        blocks = _fields(header["blocks"], "its blocks", ("of", "held"))
        self.of = _whole(blocks["of"], "its blocks", 1)
        held = _whole(blocks["held"], "the blocks it holds", 1)
        if _canonical(header) != header_bytes:
            raise Unreadable(
                "the certificate's header is not in canonical form"
            )
        self.route = Route(
            AXES.index(route["up_axis"]),
            band_first,
            band_last,
            radius,
            min_weight,
            [
                struct.unpack("<qq", reader.take(16, "a waypoint"))
                for _ in range(reader.room(waypoints, 16, "waypoints"))
            ],
        )
        self._frames(reader, frame_count)
        if reader.room(held, BLOCK_LEAF_BYTES, "blocks") > MOST_ROUTE_BLOCKS:
            raise Unreadable("the certificate holds too many blocks")
        self.held = [
            self._block(
                reader,
                struct.unpack("<Q", reader.take(8, "a block's row"))[0],
            )
            for _ in range(held)
        ]
        self._tiles(reader, tile_frame_count)
        reader.finish()
        self._in_order(data)
        order = [block.index[::-1] for block in self.held]
        if any(later <= earlier for earlier, later in zip(order, order[1:])):
            raise Unreadable(
                "the certificate's blocks are not distinct and in order"
            )


# The checks.


class Outcome:
    """What a check found."""

    def __init__(self, verdict: str) -> None:
        self.verdict = verdict
        self.failed_check: str | None = None
        self.detail: str | None = None
        self.differing_voxels = 0
        self.refuted: list[tuple[tuple[int, int, int], int]] = []
        self.clearance: tuple | None = None
        self.ready = 0
        self.read = 0
        self.contributing = 0

    def fail(self, check: str, detail: str) -> "Outcome":
        """Invalid at this check. Blocks that were not fused count none."""

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


def _the_statement(
    statement: Statement, holder: _Holder, outcome: Outcome
) -> bool:
    """The rule is the one fused here and the file names the statement."""

    if statement.rule != RULE:
        outcome.fail(
            "statement",
            f"the statement is made under the rule {statement.rule!r}, and "
            f"this check fuses by {RULE!r} alone",
        )
        return False
    if holder.statement != statement.digest:
        outcome.fail(
            "statement",
            f"the {holder.name} names statement {holder.statement}",
        )
        return False
    for label, held, stated in (
        ("frames", len(holder.frames), statement.frames),
        ("image width", holder.width, statement.width),
        ("image height", holder.height, statement.height),
        ("tile size", holder.tile_pixels, statement.tile_pixels),
        ("blocks", holder.of, statement.blocks),
    ):
        if held != stated:
            outcome.fail(
                "shape",
                f"the {holder.name} has {label} {held}; the statement "
                f"{stated}",
            )
            return False
    return True


def _held_blocks(
    statement: Statement, holder: _Holder, outcome: Outcome
) -> list[int] | None:
    """From the scan's root to fusion: what every block held is held to.

    The count of voxels differing in each block, or None if a check
    failed, which the outcome then says.
    """

    selected = holder.frames[::statement.frame_stride]
    scan_root = tree_root([leaf_hash(frame.leaf) for frame in holder.frames])
    if scan_root.hex() != statement.scan_root:
        outcome.fail(
            "scan root", f"the frames' leaves lead to {scan_root.hex()}"
        )
        return None

    for block in holder.held:
        map_root = root_from_path(
            leaf_hash(block.leaf), block.row, holder.of, block.path
        )
        if map_root.hex() != statement.map_root:
            outcome.fail(
                "map root",
                f"block {block.index} and its path lead to {map_root.hex()}",
            )
            return None

    tiles_in_a_frame = holder.tile_rows * holder.tile_columns
    for tiles in holder.tile_frames:
        depth_root = holder.frames[tiles.sequence].depth_root
        if depth_root is None:
            outcome.fail(
                "depth roots",
                f"frame {tiles.sequence} has no depth, and the "
                f"{holder.name} holds tiles of it",
            )
            return None
        found = root_from_shared_proof(
            [leaf_hash(b"SFTILE01" + data) for data in tiles.data],
            tiles.places,
            tiles_in_a_frame,
            tiles.shared_proof,
        )
        if found != depth_root:
            outcome.fail(
                "depth roots",
                f"frame {tiles.sequence}'s tiles do not lead to its depth "
                "root",
            )
            return None

    for frame in selected:
        if frame.pose is None:
            continue
        if not all(math.isfinite(value) for value in frame.pose) or (
            frame.depth_root is not None and not rigid(frame.pose)
        ):
            outcome.fail(
                "poses",
                f"frame {frame.sequence} has a pose that is not a rigid "
                "transform",
            )
            return None

    posed = {frame.sequence for frame in selected if frame.pose is not None}
    for tiles in holder.tile_frames:
        if tiles.sequence not in posed:
            outcome.fail(
                "pixels",
                f"frame {tiles.sequence} is not fused, and the "
                f"{holder.name} holds tiles of it",
            )
            return None
    try:
        fused, asked = _fuse(statement, holder, selected, outcome)
    except _NotHeld as missing:
        outcome.fail("pixels", str(missing))
        return None
    for tiles in holder.tile_frames:
        for place in tiles.places:
            if place not in asked.get(tiles.sequence, ()):
                outcome.fail(
                    "pixels",
                    f"frame {tiles.sequence}: the {holder.name} holds tile "
                    f"{place}, and no block it holds reads a pixel of it",
                )
                return None

    differing = []
    for block, (sums, weights) in zip(holder.held, fused):
        held_sums = struct.unpack(f"<{VOXELS}Q", block.sums)
        fused_sums = struct.unpack(
            f"<{VOXELS}Q", struct.pack(f"<{VOXELS}d", *sums)
        )
        differing.append(
            sum(
                1
                for voxel in range(VOXELS)
                if held_sums[voxel] != fused_sums[voxel]
                or block.weights[voxel] != weights[voxel]
            )
        )
    return differing


def check(statement: Statement, proof: Proof) -> Outcome:
    """Hold a block's proof to a statement, one check after another."""

    outcome = Outcome("verified")
    if not _the_statement(statement, proof, outcome):
        return outcome
    differing = _held_blocks(statement, proof, outcome)
    if differing is None:
        return outcome
    outcome.differing_voxels = differing[0]
    if outcome.differing_voxels:
        outcome.verdict = "refuted"
        outcome.failed_check = "fusion"
        outcome.detail = (
            f"{outcome.differing_voxels} of {VOXELS} voxels are not what the "
            "scan gives"
        )
    return outcome


def check_route(statement: Statement, certificate: Certificate) -> Outcome:
    """Hold a route's certificate to a statement, check after check."""

    outcome = Outcome("clear")
    if not _the_statement(statement, certificate, outcome):
        return outcome

    route = certificate.route
    try:
        columns = route.columns()
        touched = route.blocks(columns)
    except _NotARoute as error:
        return outcome.fail("route", str(error))
    held = [block.index for block in certificate.held]
    if touched != held:
        missing = sorted(set(touched) - set(held))
        if missing:
            return outcome.fail(
                "route",
                f"the route touches block {missing[0]}, and the "
                "certificate does not hold it",
            )
        extra = sorted(set(held) - set(touched))[0]
        return outcome.fail(
            "route",
            f"the certificate holds block {extra}, and the route does not "
            "touch it",
        )

    differing = _held_blocks(statement, certificate, outcome)
    if differing is None:
        return outcome
    outcome.refuted = [
        (block.index, count)
        for block, count in zip(certificate.held, differing)
        if count
    ]
    if outcome.refuted:
        outcome.verdict = "refuted"
        outcome.failed_check = "fusion"
        first, count = outcome.refuted[0]
        outcome.detail = (
            f"{len(outcome.refuted)} of {len(held)} blocks the route touches "
            f"are not what the scan gives, the first of them {first} in "
            f"{count} of {VOXELS} voxels"
        )
        return outcome

    outcome.clearance = _clearance(route, columns, certificate.held)
    _, voxels, unobserved, occupied, first = outcome.clearance
    if unobserved or occupied:
        outcome.verdict = "not clear"
        outcome.failed_check = "clearance"
        outcome.detail = (
            f"{unobserved} of {voxels} voxels of the route were observed "
            f"fewer than {route.min_weight} times and {occupied} are at or "
            f"behind a surface, the first of them voxel {first}"
        )
    return outcome


def _clearance(
    route: Route, columns: list[tuple[int, int]], held: list[Block]
) -> tuple:
    """What the blocks say of every voxel of the route.

    A voxel is unobserved if it was seen fewer times than the route asks,
    and occupied if it was seen that often and its sum is not above zero.
    The first of either is the least in z, then y, then x.
    """

    blocks = {
        block.index: (struct.unpack(f"<{VOXELS}d", block.sums), block.weights)
        for block in held
    }
    unobserved = occupied = 0
    first = None
    for a, b in columns:
        for level in range(route.band_first, route.band_last + 1):
            voxel = [0, 0, 0]
            voxel[route.across[0]] = a
            voxel[route.across[1]] = b
            voxel[route.up] = level
            x, y, z = voxel
            sums, weights = blocks[(x // SIDE, y // SIDE, z // SIDE)]
            flat = x % SIDE + SIDE * (y % SIDE) + SIDE * SIDE * (z % SIDE)
            if weights[flat] < route.min_weight:
                unobserved += 1
            elif sums[flat] <= 0.0:
                occupied += 1
            else:
                continue
            if first is None or (z, y, x) < first[::-1]:
                first = (x, y, z)
    voxel_count = len(columns) * (route.band_last - route.band_first + 1)
    return (len(columns), voxel_count, unobserved, occupied, first)


def _fuse(
    statement: Statement,
    holder: _Holder,
    selected: list[Frame],
    outcome: Outcome,
) -> tuple[list[tuple[list[float], list[int]]], dict[int, set[int]]]:
    """Every block held, fused again frame by frame from the tiles alone.

    Every voxel of every block is evaluated in every frame with depth and
    a pose: its centre is taken into the camera, projected, and read at
    its nearest pixel. Each step is one IEEE operation on doubles, in the
    order docs/stand-alone-checker.md writes out.
    """

    voxel = statement.voxel_size_m
    truncation = statement.truncation_m
    width, height = statement.width, statement.height
    fx, fy, cx, cy = statement.fx, statement.fy, statement.cx, statement.cy
    tile = holder.tile_pixels
    centres = []
    for block in holder.held:
        block_x, block_y, block_z = block.index
        centres.append(
            [
                (
                    (float(flat % SIDE + block_x * SIDE) + 0.5) * voxel,
                    (float(flat // SIDE % SIDE + block_y * SIDE) + 0.5)
                    * voxel,
                    (float(flat // (SIDE * SIDE) + block_z * SIDE) + 0.5)
                    * voxel,
                )
                for flat in range(VOXELS)
            ]
        )
    if not all(
        math.isfinite(axis)
        for block in centres
        for centre in block
        for axis in centre
    ):
        raise _NotHeld("a voxel centre of a block is not finite")
    held = {tiles.sequence: tiles for tiles in holder.tile_frames}
    fused = [([0.0] * VOXELS, [0] * VOXELS) for _ in holder.held]
    asked: dict[int, set[int]] = {}

    for frame in selected:
        if frame.pose is None or frame.depth_root is None:
            continue
        outcome.ready += 1
        m = frame.pose
        landed = []
        for place, block in enumerate(centres):
            for flat, (x, y, z) in enumerate(block):
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
                if not (
                    -0.5 <= u < width - 0.5 and -0.5 <= v < height - 0.5
                ):
                    continue
                column = math.floor(u + 0.5)
                row = math.floor(v + 0.5)
                if not (0 <= column < width and 0 <= row < height):
                    continue
                landed.append((place, flat, row, column, camera_z))
        if not landed:
            continue
        places = {
            row // tile * holder.tile_columns + column // tile
            for _, _, row, column, _ in landed
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
                for _, _, row, column, _ in landed
                if tiles is None
                or row // tile * holder.tile_columns + column // tile
                == missing[0]
            )
            raise _NotHeld(
                f"frame {frame.sequence}: a block reads the depth pixel at "
                f"row {row}, column {column}, and the {holder.name} holds "
                "no tile for it"
            )
        contributed = False
        for place, flat, row, column, camera_z in landed:
            _, tile_width, samples = tiles.samples[
                row // tile * holder.tile_columns + column // tile
            ]
            depth = float(
                samples[(row % tile) * tile_width + column % tile]
            ) * statement.depth_scale_m
            if not (math.isfinite(depth) and depth > 0.0):
                continue
            distance = depth - camera_z
            if not math.isfinite(distance) or distance < -truncation:
                continue
            sums, weights = fused[place]
            sums[flat] += max(-1.0, min(1.0, distance / truncation))
            weights[flat] += 1
            contributed = True
        outcome.read += 1
        outcome.contributing += contributed
    return fused, asked


# The command.


def read(data: bytes) -> tuple[_Holder, bool]:
    """A proof or a certificate, by its first bytes, and whether packed."""

    data, packed = unpacked(data)
    if data.startswith(b"SFTPRF01"):
        return Proof(data), packed
    if data.startswith(b"SFTRTE01"):
        return Certificate(data), packed
    raise Unreadable("the file is not a block proof or a route certificate")


def _read(path: str, most: int) -> bytes:
    with open(path, "rb") as file:
        data = file.read(most + 1)
    if len(data) > most:
        raise Unreadable(f"{path} is larger than {most} bytes")
    return data


def _print_checks(names: tuple[str, ...], outcome: Outcome) -> None:
    reached = True
    for name in names:
        if name == outcome.failed_check:
            print(f"check {name}: failed")
            reached = False
        else:
            print(f"check {name}: {'passed' if reached else 'not reached'}")
    if outcome.detail is not None:
        print(f"detail: {outcome.detail}")


def main(arguments: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if arguments is None else arguments
    if len(arguments) != 2:
        print(
            "usage: check.py STATEMENT.sftcommit "
            "(PROOF.sftproof | CERTIFICATE.sftroute)",
            file=sys.stderr,
        )
        return 2
    statement_path, path = arguments
    try:
        statement = Statement(_read(statement_path, MOST_STATEMENT_BYTES))
    except (OSError, Unreadable) as error:
        print(f"CANNOT CHECK {statement_path}", file=sys.stderr)
        print(f"- {error}", file=sys.stderr)
        return 2
    try:
        held, packed = read(_read(path, MOST_CERTIFICATE_BYTES))
    except (OSError, Unreadable) as error:
        print(f"CANNOT CHECK {path}", file=sys.stderr)
        print(f"- {error}", file=sys.stderr)
        return 2

    if isinstance(held, Proof):
        outcome = check(statement, held)
        x, y, z = held.held[0].index
        print("BLOCK PROOF CHECK, STANDARD LIBRARY ONLY")
        print(f"statement_sha256: {statement.digest}")
        print(f"proof_sha256: {held.digest}")
        print(f"packed: {'yes' if packed else 'no'}")
        print(f"block: {x},{y},{z} row={held.held[0].row} of={held.of}")
        _print_checks(CHECKS, outcome)
        if outcome.failed_check in (None, "fusion"):
            print(
                f"fusion: ready={outcome.ready} read={outcome.read} "
                f"contributing={outcome.contributing} "
                f"differing_voxels={outcome.differing_voxels}"
            )
        print(f"verdict: {outcome.verdict}")
        return 0 if outcome.verdict == "verified" else 1

    outcome = check_route(statement, held)
    route = held.route
    print("ROUTE CERTIFICATE CHECK, STANDARD LIBRARY ONLY")
    print(f"statement_sha256: {statement.digest}")
    print(f"certificate_sha256: {held.digest}")
    print(f"packed: {'yes' if packed else 'no'}")
    print(
        f"route: up={AXES[route.up]} band={route.band_first}.."
        f"{route.band_last} radius={route.radius} "
        f"min_weight={route.min_weight} waypoints={len(route.waypoints)}"
    )
    print(f"blocks: held={len(held.held)} of={held.of}")
    _print_checks(ROUTE_CHECKS, outcome)
    if outcome.failed_check in (None, "fusion", "clearance"):
        print(
            f"fusion: ready={outcome.ready} read={outcome.read} "
            f"contributing={outcome.contributing} "
            f"refuted_blocks={len(outcome.refuted)}"
        )
    if outcome.clearance is not None:
        columns, voxels, unobserved, occupied, _ = outcome.clearance
        print(
            f"clearance: columns={columns} voxels={voxels} "
            f"unobserved={unobserved} occupied={occupied}"
        )
    print(f"verdict: {outcome.verdict}")
    return 0 if outcome.verdict == "clear" else 1


if __name__ == "__main__":
    sys.exit(main())
