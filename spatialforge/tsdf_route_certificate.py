"""A route through a map, with what it takes to check that it is clear.

A map is asked one kind of question more than any other: can something
get from here to there. An answer taken from a map is as good as the map,
and a map taken on trust can have had an obstacle removed from it or a
corridor marked clear that nobody looked down.

A route here is stated in whole voxels, so that which voxels are meant is
not a matter of rounding:

- which of the volume's axes is up;
- a chain of columns: waypoints on the other two axes, joined by straight
  runs of adjacent columns, drawn by an integer line rule;
- a radius, in voxels: every column within it of a column of the chain;
- a band of heights, as a first and a last voxel index;
- how many times a voxel must have been observed to count as known.

The route is **clear** if every voxel of every such column in the band was
observed at least that many times and lies in front of every surface seen.
That is the rule the free-space map draws, column by column, and a test
holds the two to each other.

A certificate is the route and, for every block the route touches, what a
check of that block needs: the block with its path to the map's root, and
the depth tiles it reads with the hashes that tie them to their frames.
The frames' leaves are carried once, and so is a tile that several blocks
read. Whoever checks is given a
statement and a certificate and nothing else. They work out for themselves
which blocks the route touches, fuse every one of them again from the
tiles, and only then ask whether the route is clear in blocks they have
seen to follow from the scan.

There are four things the check can say.

``clear``
    Every block the route touches follows from the scan, and the route is
    clear in them.
``not clear``
    Every block follows from the scan, and some voxel of the route was
    not observed enough or is at or behind a surface.
``refuted``
    Everything is under the statement's roots, and some block the route
    touches is not what the scan gives. Whatever the map says of the
    route, the map does not follow from the scan there.
``invalid``
    The certificate does not hold together. It says nothing about the map
    or the route.

A block the map does not hold is space the map knows nothing of, and a
route that touches one is not clear. That cannot be certified yet: nothing
here shows a block is absent.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .errors import TsdfError
from .model import ScanSession
from .tsdf_block_plan import TSDF_BLOCK_RESOLUTION
from .tsdf_block_proof import TsdfProofFrameTiles, _Reader
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
    TsdfFrameCommitment,
    TsdfMapStatement,
    _rule_not_fused_here,
    _validate_volume,
)
from .tsdf_held_blocks import (
    _check_held_blocks,
    _decoded_held_blocks,
    _encoded_held_blocks,
    _prove_held_blocks,
    _validate_held_blocks,
)
from .tsdf_proof_pack import (
    _unpacked_if_packed,
    pack_tsdf_proof_bytes,
)

TSDF_ROUTE_CERTIFICATE_SCHEMA = "spatialforge.tsdf-route-certificate"
TSDF_ROUTE_CERTIFICATE_SCHEMA_VERSION = "0.1.0"
TSDF_ROUTE_CERTIFICATE_MAGIC = b"SFTRTE01"
MAX_TSDF_ROUTE_CERTIFICATE_HEADER_BYTES = 16 * 1024
MAX_TSDF_ROUTE_CERTIFICATE_BYTES = 2 * 1024 * 1024 * 1024
MAX_TSDF_ROUTE_WAYPOINTS = 4096
MAX_TSDF_ROUTE_RADIUS = 64
# Columns a route's line may run through. What a route costs to check is
# bounded by what it states, before any of it is worked out.
MAX_TSDF_ROUTE_CHAIN_COLUMNS = 65_536
MAX_TSDF_ROUTE_BLOCKS = 65_536
# A voxel counts as observed from this many observations, the number a
# mesh asks of one.
TSDF_ROUTE_MIN_WEIGHT = 3

# The checks, in the order they are made.
TSDF_ROUTE_CHECKS = (
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

_PREAMBLE_BYTES = len(TSDF_ROUTE_CERTIFICATE_MAGIC) + 8
_AXES = ("x", "y", "z")
# Voxel indices are kept well inside what a block index can hold.
_MAX_VOXEL_INDEX = 2**30
# Columns of a chain times columns of a disc, before they are made
# distinct.
_MAX_SWEPT_PAIRS = 4_000_000
_BLOCK_SHAPE = (
    TSDF_BLOCK_RESOLUTION,
    TSDF_BLOCK_RESOLUTION,
    TSDF_BLOCK_RESOLUTION,
)
_SUM_DTYPE = np.dtype("<f8")
_WEIGHT_DTYPE = np.dtype("<u4")
_WAYPOINT = struct.Struct("<qq")
_SCAN_CHANGED = (
    "session inputs changed while the route was certified; rerun the "
    "command"
)

_BlockIndex = tuple[int, int, int]


class TsdfRouteVerdict(StrEnum):
    """What a check of a route's certificate found."""

    CLEAR = "clear"
    NOT_CLEAR = "not clear"
    REFUTED = "refuted"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class TsdfRoute:
    """A corridor through a volume, in whole voxels.

    ``waypoints`` are columns: voxel indices on the two axes that are
    not up, in x, y, z order. ``band_first`` and ``band_last`` are voxel
    indices on the up axis, both in the band.
    """

    up_axis: int
    band_first: int
    band_last: int
    radius: int
    min_weight: int
    waypoints: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if type(self.up_axis) is not int or self.up_axis not in (0, 1, 2):
            raise TsdfError("a TSDF route's up axis is 0, 1 or 2")
        for value, label in (
            (self.band_first, "band"),
            (self.band_last, "band"),
        ):
            if type(value) is not int or abs(value) > _MAX_VOXEL_INDEX:
                raise TsdfError(
                    f"a TSDF route's {label} is two voxel indices within "
                    f"{_MAX_VOXEL_INDEX} of zero"
                )
        if self.band_last < self.band_first:
            raise TsdfError(
                "a TSDF route's band must hold at least one voxel"
            )
        if (
            type(self.radius) is not int
            or not 0 <= self.radius <= MAX_TSDF_ROUTE_RADIUS
        ):
            raise TsdfError(
                "a TSDF route's radius is a count of voxels from 0 to "
                f"{MAX_TSDF_ROUTE_RADIUS}"
            )
        if type(self.min_weight) is not int or not (
            1 <= self.min_weight < 2**32
        ):
            raise TsdfError(
                "a TSDF route's minimum weight is a positive count of "
                "observations"
            )
        if (
            not isinstance(self.waypoints, tuple)
            or not 1 <= len(self.waypoints) <= MAX_TSDF_ROUTE_WAYPOINTS
        ):
            raise TsdfError(
                "a TSDF route has from 1 to "
                f"{MAX_TSDF_ROUTE_WAYPOINTS} waypoints"
            )
        previous = None
        for waypoint in self.waypoints:
            if (
                not isinstance(waypoint, tuple)
                or len(waypoint) != 2
                or any(
                    type(value) is not int or abs(value) > _MAX_VOXEL_INDEX
                    for value in waypoint
                )
            ):
                raise TsdfError(
                    "a TSDF route's waypoint is two voxel indices within "
                    f"{_MAX_VOXEL_INDEX} of zero"
                )
            if waypoint == previous:
                raise TsdfError(
                    "a TSDF route's waypoint must differ from the one "
                    "before it"
                )
            previous = waypoint
        if 1 + sum(
            max(abs(to_a - a), abs(to_b - b))
            for (a, b), (to_a, to_b) in zip(
                self.waypoints, self.waypoints[1:]
            )
        ) > MAX_TSDF_ROUTE_CHAIN_COLUMNS:
            raise TsdfError(
                "a TSDF route's line runs through at most "
                f"{MAX_TSDF_ROUTE_CHAIN_COLUMNS} columns"
            )

    @property
    def across_axes(self) -> tuple[int, int]:
        """The two axes a column is indexed on, in x, y, z order."""

        first, second = (axis for axis in range(3) if axis != self.up_axis)
        return first, second

    @property
    def band_voxels(self) -> int:
        return self.band_last - self.band_first + 1

    def chain(self) -> np.ndarray:
        """The columns the route runs through, in order, one row each.

        Waypoints are joined by the integer line of Bresenham, taken
        from each waypoint to the next: every column is beside the one
        before it, along an axis or a diagonal, and no arithmetic in it
        rounds.
        """

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
        return np.asarray(columns, dtype=np.int64).reshape((-1, 2))

    def columns(self) -> np.ndarray:
        """Every column within the radius of the chain, each once.

        One row a column, in increasing order of the first axis and
        then the second.
        """

        reach = range(-self.radius, self.radius + 1)
        disc = np.asarray(
            [
                (da, db)
                for da in reach
                for db in reach
                if da * da + db * db <= self.radius * self.radius
            ],
            dtype=np.int64,
        )
        chain = self.chain()
        if len(chain) * len(disc) > _MAX_SWEPT_PAIRS:
            raise TsdfError(
                "a TSDF route this long and this wide sweeps more columns "
                "than a certificate is made for"
            )
        swept = (chain[:, None, :] + disc[None, :, :]).reshape((-1, 2))
        return np.unique(swept, axis=0)

    def block_indices(self) -> tuple[_BlockIndex, ...]:
        """Every block that holds a voxel of the route, in canonical order."""

        return _blocks_touched(self, self.columns())


@dataclass(frozen=True, slots=True)
class TsdfRouteClearance:
    """What some blocks say of a route.

    A voxel of the route is *unobserved* if it was seen fewer times than
    the route asks, and *occupied* if it was seen that often and lies at
    or behind a surface. The route is clear if there is neither.
    ``first_voxel_not_clear`` is the first such voxel in z, then y, then
    x, as a voxel index in x, y, z.
    """

    column_count: int
    voxel_count: int
    unobserved_voxels: int
    occupied_voxels: int
    first_voxel_not_clear: tuple[int, int, int] | None

    @property
    def is_clear(self) -> bool:
        return not (self.unobserved_voxels or self.occupied_voxels)


@dataclass(frozen=True, slots=True, eq=False)
class TsdfRouteCertificate:
    """A route, and what a check of every block it touches needs.

    The blocks are the route's, in canonical order, each at its row of
    the map with its path. ``tile_frames`` holds, for each frame any of
    them reads a pixel of, the tiles holding those pixels, each once.
    """

    statement_sha256: str
    width: int
    height: int
    tile_pixels: int
    frames: tuple[TsdfFrameCommitment, ...]
    block_count: int
    route: TsdfRoute
    block_rows: tuple[int, ...]
    block_indices: tuple[_BlockIndex, ...]
    tsdf_sums: np.ndarray
    weights: np.ndarray
    block_paths: tuple[tuple[bytes, ...], ...]
    tile_frames: tuple[TsdfProofFrameTiles, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.route, TsdfRoute):
            raise TsdfError("a TSDF route certificate needs a TsdfRoute")
        _validate_held_blocks(
            self,
            "TSDF route certificate",
            MAX_TSDF_ROUTE_BLOCKS,
            in_block_order=True,
        )

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def held_block_count(self) -> int:
        return len(self.block_indices)

    @property
    def tile_count(self) -> int:
        return sum(tiles.tile_count for tiles in self.tile_frames)

    @property
    def depth_byte_count(self) -> int:
        return sum(tiles.tiles.byte_count for tiles in self.tile_frames)

    @property
    def tile_proof_hash_count(self) -> int:
        return sum(tiles.proof_hash_count for tiles in self.tile_frames)

    @property
    def block_path_hash_count(self) -> int:
        return sum(len(path) for path in self.block_paths)

    def encoded(self) -> bytes:
        """The certificate as the bytes of a ``.sftroute`` file."""

        route = self.route
        header = _encode_header(
            {
                "schema": TSDF_ROUTE_CERTIFICATE_SCHEMA,
                "schema_version": TSDF_ROUTE_CERTIFICATE_SCHEMA_VERSION,
                "statement_sha256": self.statement_sha256,
                "frames": self.frame_count,
                "image": {
                    "width": self.width,
                    "height": self.height,
                    "tile_pixels": self.tile_pixels,
                },
                "route": {
                    "up_axis": _AXES[route.up_axis],
                    "band_first": route.band_first,
                    "band_last": route.band_last,
                    "radius": route.radius,
                    "min_weight": route.min_weight,
                    "waypoints": len(route.waypoints),
                },
                "blocks": {
                    "of": self.block_count,
                    "held": self.held_block_count,
                },
                "tile_frames": len(self.tile_frames),
            }
        )
        pieces = [
            TSDF_ROUTE_CERTIFICATE_MAGIC,
            struct.pack("<Q", len(header)),
            header,
        ]
        pieces.extend(
            _WAYPOINT.pack(*waypoint) for waypoint in route.waypoints
        )
        pieces.extend(_encoded_held_blocks(self))
        return b"".join(pieces)

    @property
    def digest_sha256(self) -> str:
        return hashlib.sha256(self.encoded()).hexdigest()


@dataclass(frozen=True, slots=True)
class TsdfRouteCertificateReport:
    """What writing a certificate produced.

    ``output_bytes`` and ``output_digest_sha256`` are the file's as it
    was written. If it was packed they are of the packed bytes, which
    need not be the same on every machine; ``unpacked_bytes`` and
    ``unpacked_digest_sha256`` are of what it carries, which are.
    """

    output: Path
    output_bytes: int
    output_digest_sha256: str
    certificate: TsdfRouteCertificate
    packed: bool
    unpacked_bytes: int
    unpacked_digest_sha256: str


@dataclass(frozen=True, slots=True)
class TsdfRouteCheck:
    """A route's certificate held against a statement.

    ``failed_check`` names the check that stopped it, one of
    ``TSDF_ROUTE_CHECKS``, or is absent when the route is clear. A
    refuted certificate stops at ``fusion`` and one that is not clear at
    ``clearance``. ``refuted_blocks`` gives each block that is not what
    the scan gives with how many of its voxels differ. ``clearance`` is
    what the blocks say of the route and is present only if every block
    was seen to follow from the scan.
    """

    statement_digest_sha256: str
    verdict: TsdfRouteVerdict
    failed_check: str | None
    detail: str | None
    block_count: int
    refuted_blocks: tuple[tuple[_BlockIndex, int], ...]
    clearance: TsdfRouteClearance | None
    selected_observations: int
    ready_observations: int

    @property
    def holds(self) -> bool:
        return self.verdict is TsdfRouteVerdict.CLEAR

    @property
    def passed_checks(self) -> tuple[str, ...]:
        if self.failed_check is None:
            return TSDF_ROUTE_CHECKS
        return TSDF_ROUTE_CHECKS[:TSDF_ROUTE_CHECKS.index(self.failed_check)]


def measure_tsdf_route(
    volume: TsdfBlockVolume,
    route: TsdfRoute,
) -> tuple[TsdfRouteClearance, tuple[_BlockIndex, ...]]:
    """What a volume says of a route, taken on trust.

    Returns the clearance and the blocks the route touches that the
    volume does not hold. A voxel of a block that is not held was never
    observed.
    """

    _validate_volume(volume)
    if not isinstance(route, TsdfRoute):
        raise TsdfError("measuring a route requires a TsdfRoute")
    columns = route.columns()
    touched = _blocks_touched(route, columns)
    rows = {block: row for row, block in enumerate(volume.block_indices)}
    sums = np.zeros((len(touched), *_BLOCK_SHAPE), dtype=_SUM_DTYPE)
    weights = np.zeros((len(touched), *_BLOCK_SHAPE), dtype=_WEIGHT_DTYPE)
    absent = []
    for place, block in enumerate(touched):
        row = rows.get(block)
        if row is None:
            absent.append(block)
            continue
        sums[place] = volume.tsdf_sums[row]
        weights[place] = volume.weights[row]
    return (
        _clearance(route, columns, touched, sums, weights),
        tuple(absent),
    )


def certify_tsdf_route(
    statement: TsdfMapStatement,
    volume: TsdfBlockVolume,
    session: ScanSession,
    route: TsdfRoute,
) -> TsdfRouteCertificate:
    """The certificate of a route through a volume, under its statement.

    The statement is recomputed from the volume and the scan first.
    Every block the route touches must be one the map holds. The
    certificate is made whether the route is clear or not: it is the
    check that says which.
    """

    if not isinstance(statement, TsdfMapStatement):
        raise TsdfError("a TSDF route certificate requires a TsdfMapStatement")
    _validate_volume(volume)
    if not isinstance(session, ScanSession):
        raise TsdfError(
            "a TSDF route certificate requires a loaded ScanSession"
        )
    if not isinstance(route, TsdfRoute):
        raise TsdfError("a TSDF route certificate requires a TsdfRoute")
    touched = route.block_indices()
    rows = {block: row for row, block in enumerate(volume.block_indices)}
    absent = [block for block in touched if block not in rows]
    if absent:
        raise TsdfError(
            f"the route touches {len(absent)} of {len(touched)} blocks "
            f"that the map does not hold, the first of them {absent[0]}; "
            "the map knows nothing of that space, so the route is not "
            "clear, and a certificate cannot yet show a block is absent"
        )

    held = _prove_held_blocks(
        statement, volume, session, touched, _SCAN_CHANGED
    )
    return TsdfRouteCertificate(
        statement_sha256=statement.digest_sha256,
        width=statement.camera.width,
        height=statement.camera.height,
        tile_pixels=statement.tile_pixels,
        frames=held.frames,
        block_count=volume.block_count,
        route=route,
        block_rows=held.block_rows,
        block_indices=held.block_indices,
        tsdf_sums=held.tsdf_sums,
        weights=held.weights,
        block_paths=held.block_paths,
        tile_frames=held.tile_frames,
    )


def check_tsdf_route_certificate(
    statement: TsdfMapStatement,
    certificate: TsdfRouteCertificate,
) -> TsdfRouteCheck:
    """Hold a route's certificate against a statement.

    Reads the two values it is given and nothing else.
    """

    if not isinstance(statement, TsdfMapStatement):
        raise TsdfError(
            "a TSDF route certificate check requires a TsdfMapStatement"
        )
    if not isinstance(certificate, TsdfRouteCertificate):
        raise TsdfError(
            "a TSDF route certificate check requires a TsdfRouteCertificate"
        )
    statement_digest = statement.digest_sha256
    selected = certificate.frames[::statement.frame_stride]
    held = certificate.held_block_count

    def found(
        verdict: TsdfRouteVerdict,
        failed_check: str | None,
        detail: str | None,
        *,
        refuted_blocks: tuple[tuple[_BlockIndex, int], ...] = (),
        clearance: TsdfRouteClearance | None = None,
        ready: int = 0,
    ) -> TsdfRouteCheck:
        return TsdfRouteCheck(
            statement_digest_sha256=statement_digest,
            verdict=verdict,
            failed_check=failed_check,
            detail=detail,
            block_count=held,
            refuted_blocks=refuted_blocks,
            clearance=clearance,
            selected_observations=len(selected),
            ready_observations=ready,
        )

    def invalid(failed_check: str, detail: str) -> TsdfRouteCheck:
        return found(TsdfRouteVerdict.INVALID, failed_check, detail)

    # statement: the statement is made under the rule fused here, and
    # the certificate names it.
    wrong = _rule_not_fused_here(statement)
    if wrong is not None:
        return invalid("statement", wrong)
    if certificate.statement_sha256 != statement_digest:
        return invalid(
            "statement",
            "the certificate was made under statement "
            f"{certificate.statement_sha256}",
        )

    # shape: the certificate is laid out as the statement says.
    for label, has, stated in (
        ("frames", certificate.frame_count, statement.frame_count),
        ("image width", certificate.width, statement.camera.width),
        ("image height", certificate.height, statement.camera.height),
        ("tile size", certificate.tile_pixels, statement.tile_pixels),
        ("blocks", certificate.block_count, statement.block_count),
    ):
        if has != stated:
            return invalid(
                "shape",
                f"the certificate has {label} {has} and the statement "
                f"{stated}",
            )

    # route: the blocks held are the blocks the route touches, worked
    # out here from the route and not taken from the certificate.
    route = certificate.route
    try:
        columns = route.columns()
        touched = _blocks_touched(route, columns)
    except TsdfError as error:
        return invalid("route", str(error))
    if touched != certificate.block_indices:
        missing = sorted(set(touched) - set(certificate.block_indices))
        if missing:
            return invalid(
                "route",
                f"the route touches block {missing[0]}, and the "
                "certificate does not hold it",
            )
        extra = sorted(set(certificate.block_indices) - set(touched))
        return invalid(
            "route",
            f"the certificate holds block {extra[0]}, and the route does "
            "not touch it",
        )

    # From the scan's root to fusion: what every holder of blocks is
    # held to.
    outcome = _check_held_blocks(
        statement,
        certificate,
        noun="certificate",
        readers="block of the route",
    )
    if outcome.failed_check is not None:
        return invalid(outcome.failed_check, outcome.detail)
    refuted = outcome.refuted
    if refuted:
        return found(
            TsdfRouteVerdict.REFUTED,
            "fusion",
            f"{len(refuted)} of {held} blocks the route touches are not "
            f"what the scan gives, the first of them {refuted[0][0]} in "
            f"{refuted[0][1]} of {TSDF_BLOCK_VOXELS} voxels",
            refuted_blocks=refuted,
            ready=outcome.ready_observations,
        )

    # clearance: in blocks now seen to follow from the scan, every
    # voxel of the route was observed and lies in front of every
    # surface seen.
    clearance = _clearance(
        route,
        columns,
        certificate.block_indices,
        certificate.tsdf_sums,
        certificate.weights,
    )
    if not clearance.is_clear:
        return found(
            TsdfRouteVerdict.NOT_CLEAR,
            "clearance",
            f"{clearance.unobserved_voxels} of {clearance.voxel_count} "
            f"voxels of the route were observed fewer than "
            f"{route.min_weight} times and {clearance.occupied_voxels} "
            "are at or behind a surface, the first of them voxel "
            f"{clearance.first_voxel_not_clear}",
            clearance=clearance,
            ready=outcome.ready_observations,
        )
    return found(
        TsdfRouteVerdict.CLEAR,
        None,
        None,
        clearance=clearance,
        ready=outcome.ready_observations,
    )


def write_tsdf_route_certificate(
    certificate: TsdfRouteCertificate,
    output: str | Path,
    *,
    packed: bool = False,
) -> TsdfRouteCertificateReport:
    """Write a certificate as a ``.sftroute`` file. Never over a file.

    ``packed`` writes the same bytes inside a zlib stream. The file is
    read back as the same certificate either way.
    """

    if not isinstance(certificate, TsdfRouteCertificate):
        raise TsdfError("writing requires a TsdfRouteCertificate")
    output_path = _validate_certificate_output(output)
    if not isinstance(packed, bool):
        raise TsdfError("packed must be true or false")
    encoded = certificate.encoded()
    if len(encoded) > MAX_TSDF_ROUTE_CERTIFICATE_BYTES:
        raise TsdfError(
            "TSDF route certificate exceeds the maximum of "
            f"{MAX_TSDF_ROUTE_CERTIFICATE_BYTES} bytes"
        )
    written = pack_tsdf_proof_bytes(encoded) if packed else encoded
    temporary_path: Path | None = None
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output_path.stem}-",
            suffix=".sftroute",
            dir=output_path.parent,
        )
        temporary_path = Path(temporary_name)
        with open(descriptor, "wb", closefd=True) as output_file:
            output_file.write(written)
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
            f"cannot write TSDF route certificate: {error}"
        ) from error
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
    return TsdfRouteCertificateReport(
        output=output_path,
        output_bytes=len(written),
        output_digest_sha256=hashlib.sha256(written).hexdigest(),
        certificate=certificate,
        packed=packed,
        unpacked_bytes=len(encoded),
        unpacked_digest_sha256=hashlib.sha256(encoded).hexdigest(),
    )


def load_tsdf_route_certificate(path: str | Path) -> TsdfRouteCertificate:
    """Read a certificate, refusing anything that is not exactly one."""

    certificate_path = Path(path)
    if certificate_path.suffix.lower() != ".sftroute":
        raise TsdfError("certificate filename must end in .sftroute")
    try:
        with certificate_path.open("rb") as certificate_file:
            size = os.fstat(certificate_file.fileno()).st_size
            encoded = certificate_file.read(
                min(size, MAX_TSDF_ROUTE_CERTIFICATE_BYTES) + 1
            )
    except OSError as error:
        raise TsdfError(
            f"cannot read TSDF route certificate: {error}"
        ) from error
    if len(encoded) > MAX_TSDF_ROUTE_CERTIFICATE_BYTES:
        raise TsdfError(
            "TSDF route certificate exceeds the maximum of "
            f"{MAX_TSDF_ROUTE_CERTIFICATE_BYTES} bytes"
        )
    return decode_tsdf_route_certificate(
        _unpacked_if_packed(encoded, MAX_TSDF_ROUTE_CERTIFICATE_BYTES)
    )


def decode_tsdf_route_certificate(encoded: bytes) -> TsdfRouteCertificate:
    """The certificate these bytes are, or a refusal.

    Read once from front to back, with every byte accounted for.
    """

    if type(encoded) is not bytes:
        raise TsdfError("a TSDF route certificate is decoded from bytes")
    if len(encoded) < _PREAMBLE_BYTES or not encoded.startswith(
        TSDF_ROUTE_CERTIFICATE_MAGIC
    ):
        raise TsdfError(
            "TSDF route certificate has an unrecognized signature"
        )
    (header_length,) = struct.unpack_from(
        "<Q", encoded, len(TSDF_ROUTE_CERTIFICATE_MAGIC)
    )
    if not 2 <= header_length <= MAX_TSDF_ROUTE_CERTIFICATE_HEADER_BYTES:
        raise TsdfError(
            "TSDF route certificate header length is out of range"
        )
    reader = _Reader(encoded, _PREAMBLE_BYTES, "TSDF route certificate")
    header = _parse_header(reader.take(header_length, "its header"))

    stated = header["route"]
    route = TsdfRoute(
        up_axis=_AXES.index(stated["up_axis"]),
        band_first=stated["band_first"],
        band_last=stated["band_last"],
        radius=stated["radius"],
        min_weight=stated["min_weight"],
        waypoints=tuple(
            _WAYPOINT.unpack(reader.take(_WAYPOINT.size, "a waypoint"))
            for _ in range(
                reader.count(stated["waypoints"], _WAYPOINT.size)
            )
        ),
    )
    image = header["image"]
    held = _decoded_held_blocks(
        reader,
        frames=header["frames"],
        of=header["blocks"]["of"],
        held=header["blocks"]["held"],
        tile_frames=header["tile_frames"],
        width=image["width"],
        height=image["height"],
        tile_pixels=image["tile_pixels"],
        most_blocks=MAX_TSDF_ROUTE_BLOCKS,
    )
    reader.finish()

    certificate = TsdfRouteCertificate(
        statement_sha256=header["statement_sha256"],
        width=image["width"],
        height=image["height"],
        tile_pixels=image["tile_pixels"],
        frames=held.frames,
        block_count=header["blocks"]["of"],
        route=route,
        block_rows=held.block_rows,
        block_indices=held.block_indices,
        tsdf_sums=held.tsdf_sums,
        weights=held.weights,
        block_paths=held.block_paths,
        tile_frames=held.tile_frames,
    )
    if certificate.encoded() != encoded:
        raise TsdfError(
            "TSDF route certificate is not in the form it is written"
        )
    return certificate


def _blocks_touched(
    route: TsdfRoute,
    columns: np.ndarray,
) -> tuple[_BlockIndex, ...]:
    """The blocks holding a voxel of these columns in the route's band."""

    across = route.across_axes
    column_blocks = np.unique(columns // TSDF_BLOCK_RESOLUTION, axis=0)
    levels = np.arange(
        route.band_first // TSDF_BLOCK_RESOLUTION,
        route.band_last // TSDF_BLOCK_RESOLUTION + 1,
        dtype=np.int64,
    )
    if len(column_blocks) * len(levels) > MAX_TSDF_ROUTE_BLOCKS:
        raise TsdfError(
            f"the route touches {len(column_blocks) * len(levels)} blocks; "
            f"a certificate holds at most {MAX_TSDF_ROUTE_BLOCKS}"
        )
    blocks = np.empty((len(column_blocks), len(levels), 3), dtype=np.int64)
    blocks[:, :, across[0]] = column_blocks[:, None, 0]
    blocks[:, :, across[1]] = column_blocks[:, None, 1]
    blocks[:, :, route.up_axis] = levels[None, :]
    blocks = blocks.reshape((-1, 3))
    order = np.lexsort((blocks[:, 0], blocks[:, 1], blocks[:, 2]))
    return tuple(
        (int(x), int(y), int(z)) for x, y, z in blocks[order].tolist()
    )


def _clearance(
    route: TsdfRoute,
    columns: np.ndarray,
    block_indices: Sequence[_BlockIndex],
    tsdf_sums: np.ndarray,
    weights: np.ndarray,
) -> TsdfRouteClearance:
    """What these blocks, which are the route's, say of the route."""

    side = TSDF_BLOCK_RESOLUTION
    across = route.across_axes
    up = route.up_axis
    column_blocks, which = np.unique(
        columns // side, axis=0, return_inverse=True
    )
    which = which.reshape(-1)
    within = columns - (columns // side) * side
    swept_columns = np.zeros((len(column_blocks), side, side), dtype=np.bool_)
    swept_columns[which, within[:, 0], within[:, 1]] = True
    place_of = {
        (int(a), int(b)): place
        for place, (a, b) in enumerate(column_blocks.tolist())
    }
    # From (first across axis, second across axis, up) to [z, y, x].
    order = (across[0], across[1], up)
    to_storage = tuple(order.index(axis) for axis in (2, 1, 0))

    swept = np.zeros((len(block_indices), side, side, side), dtype=np.bool_)
    for place, block in enumerate(block_indices):
        base = block[up] * side
        low = max(route.band_first, base) - base
        high = min(route.band_last, base + side - 1) - base
        laid = np.zeros((side, side, side), dtype=np.bool_)
        laid[:, :, low:high + 1] = swept_columns[
            place_of[block[across[0]], block[across[1]]]
        ][:, :, None]
        swept[place] = np.transpose(laid, to_storage)

    voxel_count = len(columns) * route.band_voxels
    if int(np.count_nonzero(swept)) != voxel_count:
        raise AssertionError("the blocks given are not the route's blocks")
    seen = weights >= route.min_weight
    unobserved = swept & ~seen
    occupied = swept & seen & (tsdf_sums <= 0.0)
    first = None
    places, z, y, x = np.nonzero(unobserved | occupied)
    if len(places):
        blocks = np.asarray(block_indices, dtype=np.int64)[places]
        voxels = blocks * side + np.stack([x, y, z], axis=1)
        earliest = np.lexsort((voxels[:, 0], voxels[:, 1], voxels[:, 2]))[0]
        first = tuple(int(value) for value in voxels[earliest])
    return TsdfRouteClearance(
        column_count=len(columns),
        voxel_count=voxel_count,
        unobserved_voxels=int(np.count_nonzero(unobserved)),
        occupied_voxels=int(np.count_nonzero(occupied)),
        first_voxel_not_clear=first,
    )


def _parse_header(header_bytes: bytes) -> dict[str, Any]:
    try:
        text = header_bytes.decode("ascii")
    except UnicodeError as error:
        raise TsdfError(
            "TSDF route certificate header must be ASCII JSON"
        ) from error
    if text.count("{") > 8 or "[" in text:
        raise TsdfError(
            "TSDF route certificate header holds an array or is nested too "
            "deeply"
        )
    try:
        document = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except _DuplicateJsonKey as error:
        raise TsdfError(
            "TSDF route certificate header contains duplicate JSON key "
            f"{error.key!r}"
        ) from error
    except (json.JSONDecodeError, ValueError, RecursionError) as error:
        raise TsdfError(
            f"TSDF route certificate header is invalid JSON: {error}"
        ) from error

    root = _object(
        document,
        "certificate",
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
    _exact(
        root["schema"], TSDF_ROUTE_CERTIFICATE_SCHEMA, "certificate.schema"
    )
    _exact(
        root["schema_version"],
        TSDF_ROUTE_CERTIFICATE_SCHEMA_VERSION,
        "certificate.schema_version",
    )
    _sha256(root["statement_sha256"], "certificate.statement_sha256")
    _integer(root["frames"], "certificate.frames", minimum=1)
    _integer(root["tile_frames"], "certificate.tile_frames", minimum=0)
    image = _object(
        root["image"], "certificate.image", ("width", "height", "tile_pixels")
    )
    for key in ("width", "height", "tile_pixels"):
        _integer(image[key], f"certificate.image.{key}", minimum=1)
    route = _object(
        root["route"],
        "certificate.route",
        (
            "up_axis",
            "band_first",
            "band_last",
            "radius",
            "min_weight",
            "waypoints",
        ),
    )
    if route["up_axis"] not in _AXES:
        raise TsdfError("certificate.route.up_axis: expected x, y or z")
    for key in ("band_first", "band_last"):
        _integer(
            route[key], f"certificate.route.{key}", minimum=-_MAX_VOXEL_INDEX
        )
    _integer(route["radius"], "certificate.route.radius", minimum=0)
    _integer(route["min_weight"], "certificate.route.min_weight", minimum=1)
    _integer(route["waypoints"], "certificate.route.waypoints", minimum=1)
    blocks = _object(root["blocks"], "certificate.blocks", ("of", "held"))
    _integer(blocks["of"], "certificate.blocks.of", minimum=1)
    _integer(blocks["held"], "certificate.blocks.held", minimum=1)
    if _encode_header(root) != header_bytes:
        raise TsdfError(
            "TSDF route certificate header is not in canonical form"
        )
    return root


def _validate_certificate_output(output: str | Path) -> Path:
    output_path = Path(output).resolve()
    if output_path.suffix.lower() != ".sftroute":
        raise TsdfError("output filename must end in .sftroute")
    if output_path.exists():
        raise TsdfError(f"output already exists: {output_path}")
    return output_path
