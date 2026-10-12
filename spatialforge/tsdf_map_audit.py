"""Audit a map by blocks drawn by lot.

A block's proof says one block follows from the scan. Nobody is going to
ask for every block. An audit asks for some, chosen so that whoever made
the map could not have known which.

The auditor holds the statement and picks a *seed*: any text the map's
maker could not have predicted. The seed, the statement's digest and a
count fix which rows of the map are drawn, by a rule anyone can run. The
maker answers with an audit: those blocks, with what a check of each needs,
held together as a route's certificate holds its blocks. The auditor
checks it with their own seed and count. The rows held must be exactly the
rows that seed draws, and every block is tied to the roots and fused again.

There are three things the check can say.

``passed``
    Every block drawn follows from the scan.
``refuted``
    Everything is under the statement's roots, and some block drawn is not
    what the scan gives. One is enough: the map does not follow from the
    scan.
``invalid``
    The audit does not hold together, or was not drawn with the auditor's
    seed. It says nothing about the map.

Passing says something only about how much of a map could be wrong. If
``wrong`` of a map's ``total`` blocks are not what the scan gives, a draw
of ``drawn`` distinct blocks misses all of them with probability

    C(total - wrong, drawn) / C(total, drawn)

which is worked out here in whole numbers. A draw of a few hundred blocks
finds a map that is one part in a hundred wrong nearly every time, and a
map with one block altered almost never. That is what a certificate is
for: a claim that rests on particular blocks should carry them.

The seed has to be the auditor's. A seed the maker chose, or one that
could be worked out from the statement alone, lets the maker try
statements until the draw falls where they like.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import struct
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np

from .errors import TsdfError
from .model import ScanSession
from .tsdf_block_proof import TsdfProofFrameTiles, _Reader
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
    _validate_volume,
)
from .tsdf_held_blocks import (
    _check_held_blocks,
    _decoded_held_blocks,
    _encoded_held_blocks,
    _prove_held_blocks,
    _validate_held_blocks,
)
from .tsdf_stream_fusion import _is_sha256

TSDF_MAP_AUDIT_SCHEMA = "spatialforge.tsdf-map-audit"
TSDF_MAP_AUDIT_SCHEMA_VERSION = "0.1.0"
TSDF_MAP_AUDIT_MAGIC = b"SFTAUD01"
MAX_TSDF_MAP_AUDIT_HEADER_BYTES = 16 * 1024
MAX_TSDF_MAP_AUDIT_BYTES = 2 * 1024 * 1024 * 1024
MAX_TSDF_MAP_AUDIT_BLOCKS = 65_536

# The checks, in the order they are made.
TSDF_MAP_AUDIT_CHECKS = (
    "statement",
    "shape",
    "draw",
    "scan root",
    "map root",
    "depth roots",
    "poses",
    "pixels",
    "fusion",
)

_PREAMBLE_BYTES = len(TSDF_MAP_AUDIT_MAGIC) + 8
_DRAW_TAG = b"SFAUDIT1"
_SEED = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_SCAN_CHANGED = (
    "session inputs changed while the audit was answered; rerun the "
    "command"
)

_BlockIndex = tuple[int, int, int]


class TsdfMapAuditVerdict(StrEnum):
    """What a check of an audit found."""

    PASSED = "passed"
    REFUTED = "refuted"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True, eq=False)
class TsdfMapAudit:
    """Blocks of a map drawn by a seed, with what a check of them needs.

    ``block_rows`` are the rows drawn, in increasing order, and the
    other fields of the blocks follow them.
    """

    statement_sha256: str
    width: int
    height: int
    tile_pixels: int
    frames: tuple[TsdfFrameCommitment, ...]
    block_count: int
    seed: str
    block_rows: tuple[int, ...]
    block_indices: tuple[_BlockIndex, ...]
    tsdf_sums: np.ndarray
    weights: np.ndarray
    block_paths: tuple[tuple[bytes, ...], ...]
    tile_frames: tuple[TsdfProofFrameTiles, ...]

    def __post_init__(self) -> None:
        _validate_seed(self.seed)
        _validate_held_blocks(
            self,
            "TSDF map audit",
            MAX_TSDF_MAP_AUDIT_BLOCKS,
            in_block_order=False,
        )

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def drawn_block_count(self) -> int:
        return len(self.block_rows)

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
        """The audit as the bytes of a ``.sftaudit`` file."""

        header = _encode_header(
            {
                "schema": TSDF_MAP_AUDIT_SCHEMA,
                "schema_version": TSDF_MAP_AUDIT_SCHEMA_VERSION,
                "statement_sha256": self.statement_sha256,
                "frames": self.frame_count,
                "image": {
                    "width": self.width,
                    "height": self.height,
                    "tile_pixels": self.tile_pixels,
                },
                "seed": self.seed,
                "blocks": {
                    "of": self.block_count,
                    "drawn": self.drawn_block_count,
                },
                "tile_frames": len(self.tile_frames),
            }
        )
        return b"".join(
            [
                TSDF_MAP_AUDIT_MAGIC,
                struct.pack("<Q", len(header)),
                header,
                *_encoded_held_blocks(self),
            ]
        )

    @property
    def digest_sha256(self) -> str:
        return hashlib.sha256(self.encoded()).hexdigest()


@dataclass(frozen=True, slots=True)
class TsdfMapAuditReport:
    """What writing an audit produced."""

    output: Path
    output_bytes: int
    output_digest_sha256: str
    audit: TsdfMapAudit


@dataclass(frozen=True, slots=True)
class TsdfMapAuditCheck:
    """An audit held against a statement and the auditor's own seed.

    ``failed_check`` names the check that stopped it, one of
    ``TSDF_MAP_AUDIT_CHECKS``, or is absent when the audit passed. A
    refuted audit stops at ``fusion`` and names each block that is not
    what the scan gives, with how many of its voxels differ.
    """

    statement_digest_sha256: str
    verdict: TsdfMapAuditVerdict
    failed_check: str | None
    detail: str | None
    block_count: int
    drawn_block_count: int
    refuted_blocks: tuple[tuple[_BlockIndex, int], ...]
    selected_observations: int
    ready_observations: int

    @property
    def holds(self) -> bool:
        return self.verdict is TsdfMapAuditVerdict.PASSED

    @property
    def passed_checks(self) -> tuple[str, ...]:
        if self.failed_check is None:
            return TSDF_MAP_AUDIT_CHECKS
        return TSDF_MAP_AUDIT_CHECKS[
            :TSDF_MAP_AUDIT_CHECKS.index(self.failed_check)
        ]


def draw_tsdf_audit_rows(
    statement_sha256: str,
    seed: str,
    count: int,
    block_count: int,
) -> tuple[int, ...]:
    """The rows of a map that a seed draws, in increasing order.

    Draw ``i`` is the SHA-256 of a tag, the statement's digest, the seed
    and ``i``. Its first eight bytes are read as a number, which is
    thrown away if it falls in the uneven remainder above the largest
    multiple of ``block_count``, and otherwise taken modulo it. Draws
    go on until ``count`` different rows have come up. Every row is as
    likely as every other, and the same on every machine.
    """

    if not _is_sha256(statement_sha256):
        raise TsdfError("an audit is drawn from a statement's SHA-256")
    _validate_seed(seed)
    for value, label in ((count, "count"), (block_count, "block count")):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise TsdfError(f"an audit's {label} is a positive integer")
    if count > block_count:
        raise TsdfError(
            f"an audit cannot draw {count} different blocks from a map of "
            f"{block_count}"
        )
    if count > MAX_TSDF_MAP_AUDIT_BLOCKS:
        raise TsdfError(
            f"an audit draws at most {MAX_TSDF_MAP_AUDIT_BLOCKS} blocks"
        )
    stem = b"".join(
        (
            _DRAW_TAG,
            bytes.fromhex(statement_sha256),
            struct.pack("<H", len(seed)),
            seed.encode("ascii"),
        )
    )
    even = (2**64 // block_count) * block_count
    drawn: set[int] = set()
    draw = 0
    while len(drawn) < count:
        number = int.from_bytes(
            hashlib.sha256(stem + struct.pack("<Q", draw)).digest()[:8],
            "little",
        )
        draw += 1
        if number < even:
            drawn.add(number % block_count)
    return tuple(sorted(drawn))


def tsdf_audit_wrong_blocks_caught(
    block_count: int,
    drawn: int,
    *,
    one_in: int = 100,
) -> int:
    """How few wrong blocks an audit of this size would almost surely find.

    The smallest number ``wrong`` such that a map of ``block_count``
    blocks with that many wrong passes a draw of ``drawn`` different
    blocks no more than one time in ``one_in``. Worked out exactly, in
    whole numbers.
    """

    for value, label in (
        (block_count, "block count"),
        (drawn, "count"),
        (one_in, "odds"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise TsdfError(f"an audit's {label} is a positive integer")
    if drawn > block_count:
        raise TsdfError(
            f"an audit cannot draw {drawn} different blocks from a map of "
            f"{block_count}"
        )
    every_draw = math.comb(block_count, drawn)

    def passes_too_often(wrong: int) -> bool:
        return one_in * math.comb(block_count - wrong, drawn) > every_draw

    # With more wrong blocks than are left undrawn, none can be missed.
    low, high = 1, block_count - drawn + 1
    while low < high:
        middle = (low + high) // 2
        if passes_too_often(middle):
            low = middle + 1
        else:
            high = middle
    return low


def audit_tsdf_map(
    statement: TsdfMapStatement,
    volume: TsdfBlockVolume,
    session: ScanSession,
    *,
    seed: str,
    count: int,
) -> TsdfMapAudit:
    """Answer an audit: the blocks a seed draws, with what checks them.

    The statement is recomputed from the volume and the scan first. The
    answer is made of whatever blocks the map holds at the rows drawn;
    it is the check that says whether they follow.
    """

    if not isinstance(statement, TsdfMapStatement):
        raise TsdfError("a TSDF map audit requires a TsdfMapStatement")
    _validate_volume(volume)
    if not isinstance(session, ScanSession):
        raise TsdfError("a TSDF map audit requires a loaded ScanSession")
    if statement.block_count != volume.block_count:
        raise TsdfError(
            "the statement is not a statement of this volume: block count"
        )
    rows = draw_tsdf_audit_rows(
        statement.digest_sha256, seed, count, volume.block_count
    )
    held = _prove_held_blocks(
        statement,
        volume,
        session,
        tuple(volume.block_indices[row] for row in rows),
        _SCAN_CHANGED,
    )
    return TsdfMapAudit(
        statement_sha256=statement.digest_sha256,
        width=statement.camera.width,
        height=statement.camera.height,
        tile_pixels=statement.tile_pixels,
        frames=held.frames,
        block_count=volume.block_count,
        seed=seed,
        block_rows=held.block_rows,
        block_indices=held.block_indices,
        tsdf_sums=held.tsdf_sums,
        weights=held.weights,
        block_paths=held.block_paths,
        tile_frames=held.tile_frames,
    )


def check_tsdf_map_audit(
    statement: TsdfMapStatement,
    audit: TsdfMapAudit,
    *,
    seed: str,
    count: int,
) -> TsdfMapAuditCheck:
    """Hold an audit against a statement and the auditor's seed and count.

    ``seed`` and ``count`` are the auditor's own, not read from the
    audit. Reads the values it is given and nothing else.
    """

    if not isinstance(statement, TsdfMapStatement):
        raise TsdfError("a TSDF map audit check requires a TsdfMapStatement")
    if not isinstance(audit, TsdfMapAudit):
        raise TsdfError("a TSDF map audit check requires a TsdfMapAudit")
    _validate_seed(seed)
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise TsdfError("an audit's count is a positive integer")
    statement_digest = statement.digest_sha256

    def found(
        verdict: TsdfMapAuditVerdict,
        failed_check: str | None,
        detail: str | None,
        refuted_blocks: tuple[tuple[_BlockIndex, int], ...] = (),
        ready: int = 0,
    ) -> TsdfMapAuditCheck:
        return TsdfMapAuditCheck(
            statement_digest_sha256=statement_digest,
            verdict=verdict,
            failed_check=failed_check,
            detail=detail,
            block_count=statement.block_count,
            drawn_block_count=audit.drawn_block_count,
            refuted_blocks=refuted_blocks,
            selected_observations=len(
                audit.frames[::statement.frame_stride]
            ),
            ready_observations=ready,
        )

    def invalid(failed_check: str, detail: str) -> TsdfMapAuditCheck:
        return found(TsdfMapAuditVerdict.INVALID, failed_check, detail)

    # statement: the audit names the statement it answers for.
    if audit.statement_sha256 != statement_digest:
        return invalid(
            "statement",
            f"the audit was answered under statement {audit.statement_sha256}",
        )

    # shape: the audit is laid out as the statement says.
    for label, has, stated in (
        ("frames", audit.frame_count, statement.frame_count),
        ("image width", audit.width, statement.camera.width),
        ("image height", audit.height, statement.camera.height),
        ("tile size", audit.tile_pixels, statement.tile_pixels),
        ("blocks", audit.block_count, statement.block_count),
    ):
        if has != stated:
            return invalid(
                "shape",
                f"the audit has {label} {has} and the statement {stated}",
            )

    # draw: the rows held are the rows the auditor's seed draws, worked
    # out here and not taken from the audit.
    if audit.seed != seed:
        return invalid(
            "draw",
            f"the audit was drawn with the seed {audit.seed!r}, which is "
            "not the seed it is checked with",
        )
    if count > statement.block_count:
        return invalid(
            "draw",
            f"{count} different blocks cannot be drawn from a map of "
            f"{statement.block_count}",
        )
    if count > MAX_TSDF_MAP_AUDIT_BLOCKS:
        return invalid(
            "draw",
            f"an audit draws at most {MAX_TSDF_MAP_AUDIT_BLOCKS} blocks",
        )
    drawn = draw_tsdf_audit_rows(
        statement_digest, seed, count, statement.block_count
    )
    if audit.block_rows != drawn:
        missing = sorted(set(drawn) - set(audit.block_rows))
        if missing:
            return invalid(
                "draw",
                f"the seed draws row {missing[0]} of the map, and the "
                "audit does not hold it",
            )
        extra = sorted(set(audit.block_rows) - set(drawn))
        return invalid(
            "draw",
            f"the audit holds row {extra[0]} of the map, and the seed "
            "does not draw it",
        )

    # From the scan's root to fusion: what every holder of blocks is
    # held to.
    outcome = _check_held_blocks(
        statement, audit, noun="audit", readers="block drawn"
    )
    if outcome.failed_check is not None:
        return invalid(outcome.failed_check, outcome.detail)
    refuted = outcome.refuted
    if refuted:
        return found(
            TsdfMapAuditVerdict.REFUTED,
            "fusion",
            f"{len(refuted)} of {len(drawn)} blocks drawn are not what "
            f"the scan gives, the first of them {refuted[0][0]} in "
            f"{refuted[0][1]} of 512 voxels",
            refuted,
            outcome.ready_observations,
        )
    return found(
        TsdfMapAuditVerdict.PASSED,
        None,
        None,
        (),
        outcome.ready_observations,
    )


def write_tsdf_map_audit(
    audit: TsdfMapAudit,
    output: str | Path,
) -> TsdfMapAuditReport:
    """Write an audit as a ``.sftaudit`` file. Never over a file."""

    if not isinstance(audit, TsdfMapAudit):
        raise TsdfError("writing requires a TsdfMapAudit")
    output_path = _validate_audit_output(output)
    encoded = audit.encoded()
    if len(encoded) > MAX_TSDF_MAP_AUDIT_BYTES:
        raise TsdfError(
            "TSDF map audit exceeds the maximum of "
            f"{MAX_TSDF_MAP_AUDIT_BYTES} bytes"
        )
    temporary_path: Path | None = None
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output_path.stem}-",
            suffix=".sftaudit",
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
        raise TsdfError(f"cannot write TSDF map audit: {error}") from error
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
    return TsdfMapAuditReport(
        output=output_path,
        output_bytes=len(encoded),
        output_digest_sha256=hashlib.sha256(encoded).hexdigest(),
        audit=audit,
    )


def load_tsdf_map_audit(path: str | Path) -> TsdfMapAudit:
    """Read an audit, refusing anything that is not exactly one."""

    audit_path = Path(path)
    if audit_path.suffix.lower() != ".sftaudit":
        raise TsdfError("audit filename must end in .sftaudit")
    try:
        with audit_path.open("rb") as audit_file:
            size = os.fstat(audit_file.fileno()).st_size
            encoded = audit_file.read(
                min(size, MAX_TSDF_MAP_AUDIT_BYTES) + 1
            )
    except OSError as error:
        raise TsdfError(f"cannot read TSDF map audit: {error}") from error
    if len(encoded) > MAX_TSDF_MAP_AUDIT_BYTES:
        raise TsdfError(
            "TSDF map audit exceeds the maximum of "
            f"{MAX_TSDF_MAP_AUDIT_BYTES} bytes"
        )
    return decode_tsdf_map_audit(encoded)


def decode_tsdf_map_audit(encoded: bytes) -> TsdfMapAudit:
    """The audit these bytes are, or a refusal.

    Read once from front to back, with every byte accounted for.
    """

    if type(encoded) is not bytes:
        raise TsdfError("a TSDF map audit is decoded from bytes")
    if len(encoded) < _PREAMBLE_BYTES or not encoded.startswith(
        TSDF_MAP_AUDIT_MAGIC
    ):
        raise TsdfError("TSDF map audit has an unrecognized signature")
    (header_length,) = struct.unpack_from(
        "<Q", encoded, len(TSDF_MAP_AUDIT_MAGIC)
    )
    if not 2 <= header_length <= MAX_TSDF_MAP_AUDIT_HEADER_BYTES:
        raise TsdfError("TSDF map audit header length is out of range")
    reader = _Reader(encoded, _PREAMBLE_BYTES, "TSDF map audit")
    header = _parse_header(reader.take(header_length, "its header"))
    image = header["image"]
    held = _decoded_held_blocks(
        reader,
        frames=header["frames"],
        of=header["blocks"]["of"],
        held=header["blocks"]["drawn"],
        tile_frames=header["tile_frames"],
        width=image["width"],
        height=image["height"],
        tile_pixels=image["tile_pixels"],
        most_blocks=MAX_TSDF_MAP_AUDIT_BLOCKS,
    )
    reader.finish()
    audit = TsdfMapAudit(
        statement_sha256=header["statement_sha256"],
        width=image["width"],
        height=image["height"],
        tile_pixels=image["tile_pixels"],
        frames=held.frames,
        block_count=header["blocks"]["of"],
        seed=header["seed"],
        block_rows=held.block_rows,
        block_indices=held.block_indices,
        tsdf_sums=held.tsdf_sums,
        weights=held.weights,
        block_paths=held.block_paths,
        tile_frames=held.tile_frames,
    )
    if audit.encoded() != encoded:
        raise TsdfError("TSDF map audit is not in the form it is written")
    return audit


def _validate_seed(seed: object) -> None:
    if not isinstance(seed, str) or not _SEED.fullmatch(seed):
        raise TsdfError(
            "an audit's seed is 1 to 128 letters, digits, '.', '_', ':' "
            "or '-'"
        )


def _parse_header(header_bytes: bytes) -> dict[str, Any]:
    try:
        text = header_bytes.decode("ascii")
    except UnicodeError as error:
        raise TsdfError("TSDF map audit header must be ASCII JSON") from error
    if text.count("{") > 8 or "[" in text:
        raise TsdfError(
            "TSDF map audit header holds an array or is nested too deeply"
        )
    try:
        document = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except _DuplicateJsonKey as error:
        raise TsdfError(
            f"TSDF map audit header contains duplicate JSON key {error.key!r}"
        ) from error
    except (json.JSONDecodeError, ValueError, RecursionError) as error:
        raise TsdfError(
            f"TSDF map audit header is invalid JSON: {error}"
        ) from error

    root = _object(
        document,
        "audit",
        (
            "schema",
            "schema_version",
            "statement_sha256",
            "frames",
            "image",
            "seed",
            "blocks",
            "tile_frames",
        ),
    )
    _exact(root["schema"], TSDF_MAP_AUDIT_SCHEMA, "audit.schema")
    _exact(
        root["schema_version"],
        TSDF_MAP_AUDIT_SCHEMA_VERSION,
        "audit.schema_version",
    )
    _sha256(root["statement_sha256"], "audit.statement_sha256")
    _integer(root["frames"], "audit.frames", minimum=1)
    _integer(root["tile_frames"], "audit.tile_frames", minimum=0)
    _validate_seed(root["seed"])
    image = _object(
        root["image"], "audit.image", ("width", "height", "tile_pixels")
    )
    for key in ("width", "height", "tile_pixels"):
        _integer(image[key], f"audit.image.{key}", minimum=1)
    blocks = _object(root["blocks"], "audit.blocks", ("of", "drawn"))
    _integer(blocks["of"], "audit.blocks.of", minimum=1)
    _integer(blocks["drawn"], "audit.blocks.drawn", minimum=1)
    if _encode_header(root) != header_bytes:
        raise TsdfError("TSDF map audit header is not in canonical form")
    return root


def _validate_audit_output(output: str | Path) -> Path:
    output_path = Path(output).resolve()
    if output_path.suffix.lower() != ".sftaudit":
        raise TsdfError("output filename must end in .sftaudit")
    if output_path.exists():
        raise TsdfError(f"output already exists: {output_path}")
    return output_path
