"""A fusion stopped part way, on disk, in a form that can be continued.

Staged fusion carries a small record between stages and the accumulators
that record describes. Both lived only in the process doing the fusing, so
an interrupted fusion lost everything it had done. A ``.sftckpt`` file is
that state saved::

    8 bytes   magic  b"SFTCKP01"
    8 bytes   header length, unsigned little-endian
    header    canonical ASCII JSON, newline terminated
    payload   TSDF sums   float64 little-endian  [blocks, 8, 8, 8]
              weights     uint32  little-endian  [blocks, 8, 8, 8]

The header holds the progress record: which plan and scan, how many of the
selected observations are behind it, the counts they produced, and a digest
of each payload array. Block indices are the plan's and are not repeated,
only their digest is, which is enough to refuse storage laid out otherwise.

A checkpoint is not a volume. It has no mesh and no place in the artifact
chain; it exists to be continued from, and continuing from it ends on the
bytes an uninterrupted fusion writes. The loader holds it to the same
standard as a volume all the same: every count the payload can confirm is
re-derived from the payload, and a file whose header and payload disagree
is refused.

Unlike every other artifact here, a checkpoint is meant to be replaced, by
the next one of the same fusion. Replacement is atomic: the new file is
written beside the old and moved over it, so an interruption leaves one or
the other, never part of each.
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
from typing import Any

import numpy as np

from .errors import TsdfError
from .tsdf_block_contributions import TSDF_CONTRIBUTION_STATUS_ORDER
from .tsdf_block_plan import TSDF_BLOCK_RESOLUTION
from .tsdf_block_plan_loader import (
    _DuplicateJsonKey,
    _reject_json_constant,
    _unique_object,
)
from .tsdf_block_storage import (
    MAX_TSDF_BLOCK_STORAGE_BLOCKS,
    TSDF_BLOCK_VOXELS,
    TsdfBlockStorage,
)
from .tsdf_stream_fusion import (
    TsdfStreamFusionProgress,
    _require_progress_describes,
)
from .tsdf_voxel_update import _validate_update_storage

TSDF_FUSION_CHECKPOINT_SCHEMA = "spatialforge.tsdf-fusion-checkpoint"
TSDF_FUSION_CHECKPOINT_SCHEMA_VERSION = "0.1.0"
TSDF_FUSION_CHECKPOINT_MAGIC = b"SFTCKP01"
MAX_TSDF_FUSION_CHECKPOINT_HEADER_BYTES = 64 * 1024

_PREAMBLE_BYTES = len(TSDF_FUSION_CHECKPOINT_MAGIC) + 8
_INDEX_DTYPE = np.dtype("<i4")
_SUM_DTYPE = np.dtype("<f8")
_WEIGHT_DTYPE = np.dtype("<u4")
_BYTES_PER_BLOCK = TSDF_BLOCK_VOXELS * (
    _SUM_DTYPE.itemsize + _WEIGHT_DTYPE.itemsize
)
MAX_TSDF_FUSION_CHECKPOINT_BYTES = (
    _PREAMBLE_BYTES
    + MAX_TSDF_FUSION_CHECKPOINT_HEADER_BYTES
    + MAX_TSDF_BLOCK_STORAGE_BLOCKS * _BYTES_PER_BLOCK
)

_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_HEADER_NESTING = 8
_VALIDATION_CHUNK_BLOCKS = 4096
_PAYLOAD_LAYOUT = (
    "tsdf_sums:float64:[blocks,8,8,8]",
    "weights:uint32:[blocks,8,8,8]",
)
_STATUS_BY_VALUE = {
    status.value: status for status in TSDF_CONTRIBUTION_STATUS_ORDER
}
_PROGRESS_COUNTS = (
    "frame_stride",
    "total_observations",
    "selected_observations",
    "processed_observations",
    "fused_observations",
    "skipped_missing_depth",
    "skipped_missing_pose",
    "valid_depth_samples",
    "invalid_depth_samples",
)


@dataclass(frozen=True, slots=True)
class TsdfFusionCheckpointReport:
    """What one write produced."""

    output: Path
    output_digest_sha256: str
    output_bytes: int
    processed_observations: int
    selected_observations: int


@dataclass(frozen=True, slots=True, eq=False)
class TsdfFusionCheckpoint:
    """An immutable fusion checkpoint loaded from a ``.sftckpt`` file."""

    path: Path
    artifact_digest_sha256: str
    progress: TsdfStreamFusionProgress
    block_indices_sha256: str
    tsdf_sums: np.ndarray
    weights: np.ndarray

    def __post_init__(self) -> None:
        shape = (
            self.progress.block_count,
            TSDF_BLOCK_RESOLUTION,
            TSDF_BLOCK_RESOLUTION,
            TSDF_BLOCK_RESOLUTION,
        )
        for array, dtype, label in (
            (self.tsdf_sums, _SUM_DTYPE, "sums"),
            (self.weights, _WEIGHT_DTYPE, "weights"),
        ):
            if (
                type(array) is not np.ndarray
                or array.shape != shape
                or array.dtype != dtype
                or array.flags.writeable
            ):
                raise TsdfError(
                    f"TSDF fusion checkpoint {label} must be an immutable "
                    f"{dtype.name} array of shape {shape}"
                )


def write_tsdf_fusion_checkpoint(
    storage: TsdfBlockStorage,
    progress: TsdfStreamFusionProgress,
    output: str | Path,
    *,
    replace_existing: bool = False,
) -> TsdfFusionCheckpointReport:
    """Save a staged fusion's storage and the progress that describes it.

    An existing file is refused unless ``replace_existing`` is set, which a
    caller should do only for a path it has already loaded as a checkpoint
    of this same fusion, or written itself.
    """

    if not isinstance(storage, TsdfBlockStorage):
        raise TsdfError(
            "TSDF fusion checkpoint requires allocated TsdfBlockStorage"
        )
    if not isinstance(progress, TsdfStreamFusionProgress):
        raise TsdfError(
            "TSDF fusion checkpoint requires a TsdfStreamFusionProgress"
        )
    _validate_update_storage(storage)
    output_path = _checkpoint_path(output)
    if output_path.exists() and not (
        replace_existing and output_path.is_file()
    ):
        raise TsdfError(f"output already exists: {output_path}")
    # The digests in the header are the progress's, so the storage has to
    # be shown to hold exactly those bytes before any of them are written.
    _require_progress_describes(progress, storage)

    header = _encode_header(
        _build_header(progress, _block_indices_sha256(storage))
    )
    if len(header) > MAX_TSDF_FUSION_CHECKPOINT_HEADER_BYTES:
        raise TsdfError("TSDF fusion checkpoint header exceeds its size limit")
    sums = np.ascontiguousarray(storage.tsdf_sums, dtype=_SUM_DTYPE)
    weights = np.ascontiguousarray(storage.weights, dtype=_WEIGHT_DTYPE)

    digest = hashlib.sha256()
    written = 0
    temporary_path: Path | None = None
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output_path.stem}-",
            suffix=".sftckpt",
            dir=output_path.parent,
        )
        temporary_path = Path(temporary_name)
        with open(descriptor, "wb", closefd=True) as output_file:
            for piece in (
                TSDF_FUSION_CHECKPOINT_MAGIC,
                struct.pack("<Q", len(header)),
                header,
                memoryview(sums).cast("B"),
                memoryview(weights).cast("B"),
            ):
                output_file.write(piece)
                digest.update(piece)
                written += len(piece)
            output_file.flush()
            os.fsync(output_file.fileno())
        if replace_existing:
            os.replace(temporary_path, output_path)
            temporary_path = None
        else:
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
            f"cannot write TSDF fusion checkpoint: {error}"
        ) from error
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass

    return TsdfFusionCheckpointReport(
        output=output_path,
        output_digest_sha256=digest.hexdigest(),
        output_bytes=written,
        processed_observations=progress.processed_observations,
        selected_observations=progress.selected_observations,
    )


def load_tsdf_fusion_checkpoint(path: str | Path) -> TsdfFusionCheckpoint:
    """Load and fully verify an immutable ``.sftckpt`` checkpoint."""

    input_path = Path(path).resolve()
    if input_path.suffix.lower() != ".sftckpt":
        raise TsdfError("TSDF fusion checkpoint filename must end in .sftckpt")
    if not input_path.is_file():
        raise TsdfError(f"TSDF fusion checkpoint does not exist: {input_path}")
    try:
        with input_path.open("rb") as input_file:
            # By its own size: a read of "up to the maximum" reserves the
            # maximum, whatever the file turns out to hold.
            size = os.fstat(input_file.fileno()).st_size
            encoded = input_file.read(
                min(size, MAX_TSDF_FUSION_CHECKPOINT_BYTES) + 1
            )
    except OSError as error:
        raise TsdfError(
            f"cannot read TSDF fusion checkpoint: {error}"
        ) from error
    if len(encoded) > MAX_TSDF_FUSION_CHECKPOINT_BYTES:
        raise TsdfError(
            "TSDF fusion checkpoint exceeds the maximum of "
            f"{MAX_TSDF_FUSION_CHECKPOINT_BYTES} bytes"
        )
    digest = hashlib.sha256(encoded).hexdigest()
    # Read through views: a slice of the bytes would be a copy.
    contents = memoryview(encoded)

    if len(encoded) < _PREAMBLE_BYTES or not encoded.startswith(
        TSDF_FUSION_CHECKPOINT_MAGIC
    ):
        raise TsdfError("TSDF fusion checkpoint has an unrecognized signature")
    (header_length,) = struct.unpack_from(
        "<Q",
        encoded,
        len(TSDF_FUSION_CHECKPOINT_MAGIC),
    )
    if not 2 <= header_length <= MAX_TSDF_FUSION_CHECKPOINT_HEADER_BYTES:
        raise TsdfError(
            "TSDF fusion checkpoint header length is out of range"
        )
    payload_offset = _PREAMBLE_BYTES + header_length
    if len(encoded) < payload_offset:
        raise TsdfError(
            "TSDF fusion checkpoint is truncated inside its header"
        )
    progress, payload = _parse_header(encoded[_PREAMBLE_BYTES:payload_offset])

    block_count = progress.block_count
    expected_payload = block_count * _BYTES_PER_BLOCK
    if len(encoded) - payload_offset != expected_payload:
        raise TsdfError(
            "TSDF fusion checkpoint payload is "
            f"{len(encoded) - payload_offset} bytes; its header describes "
            f"{expected_payload}"
        )
    sum_bytes = TSDF_BLOCK_VOXELS * _SUM_DTYPE.itemsize * block_count
    sum_payload = contents[payload_offset:payload_offset + sum_bytes]
    weight_payload = contents[payload_offset + sum_bytes:]
    for content, recorded, name in (
        (sum_payload, progress.tsdf_sums_sha256, "tsdf_sums"),
        (weight_payload, progress.weights_sha256, "weights"),
    ):
        if hashlib.sha256(content).hexdigest() != recorded:
            raise TsdfError(
                f"TSDF fusion checkpoint {name} do not match their "
                "recorded digest"
            )

    shape = (
        block_count,
        TSDF_BLOCK_RESOLUTION,
        TSDF_BLOCK_RESOLUTION,
        TSDF_BLOCK_RESOLUTION,
    )
    tsdf_sums = np.frombuffer(sum_payload, dtype=_SUM_DTYPE).reshape(shape)
    weights = np.frombuffer(weight_payload, dtype=_WEIGHT_DTYPE).reshape(
        shape
    )
    _validate_accumulators(tsdf_sums, weights, progress)

    return TsdfFusionCheckpoint(
        path=input_path,
        artifact_digest_sha256=digest,
        progress=progress,
        block_indices_sha256=payload["block_indices_sha256"],
        tsdf_sums=tsdf_sums,
        weights=weights,
    )


def restore_tsdf_fusion_checkpoint(
    checkpoint: TsdfFusionCheckpoint,
    storage: TsdfBlockStorage,
) -> TsdfStreamFusionProgress:
    """Put a checkpoint's accumulators into empty storage of its own plan.

    Returns the progress the storage then answers to, ready to be handed to
    the next stage.
    """

    if not isinstance(checkpoint, TsdfFusionCheckpoint):
        raise TsdfError(
            "TSDF fusion checkpoint restore requires a loaded "
            "TsdfFusionCheckpoint"
        )
    if not isinstance(storage, TsdfBlockStorage):
        raise TsdfError(
            "TSDF fusion checkpoint restore requires allocated "
            "TsdfBlockStorage"
        )
    _validate_update_storage(storage)
    progress = checkpoint.progress
    plan = storage.source_plan
    if (
        progress.session_id != plan.session_id
        or progress.source_plan_digest_sha256 != plan.artifact_digest_sha256
        or progress.replay_digest_sha256 != plan.replay_digest_sha256
        or progress.frame_stride != plan.frame_stride
        or progress.total_observations != plan.total_observations
        or progress.selected_observations != plan.selected_observations
        or progress.block_count != storage.block_count
        or progress.voxel_slots != storage.voxel_slots
        or checkpoint.block_indices_sha256 != _block_indices_sha256(storage)
    ):
        raise TsdfError(
            "TSDF fusion checkpoint does not belong to this plan and scan"
        )
    if (
        np.count_nonzero(storage.weights)
        or np.count_nonzero(storage.tsdf_sums)
        or bool(np.signbit(storage.tsdf_sums).any())
    ):
        raise TsdfError(
            "TSDF fusion checkpoint restore requires canonical empty "
            "planned storage"
        )
    storage.tsdf_sums[...] = checkpoint.tsdf_sums
    storage.weights[...] = checkpoint.weights
    return progress


def _checkpoint_path(output: str | Path) -> Path:
    output_path = Path(output).resolve()
    if output_path.suffix.lower() != ".sftckpt":
        raise TsdfError("checkpoint filename must end in .sftckpt")
    return output_path


def _block_indices_sha256(storage: TsdfBlockStorage) -> str:
    indices = np.asarray(storage.block_indices, dtype=np.int64)
    return hashlib.sha256(
        indices.astype(_INDEX_DTYPE).tobytes(order="C")
    ).hexdigest()


def _build_header(
    progress: TsdfStreamFusionProgress,
    index_digest: str,
) -> dict[str, Any]:
    return {
        "schema": TSDF_FUSION_CHECKPOINT_SCHEMA,
        "schema_version": TSDF_FUSION_CHECKPOINT_SCHEMA_VERSION,
        "session_id": progress.session_id,
        "replay_digest_sha256": progress.replay_digest_sha256,
        "source_plan_digest_sha256": progress.source_plan_digest_sha256,
        "progress": {
            **{name: getattr(progress, name) for name in _PROGRESS_COUNTS},
            "status_counts": {
                status.value: count
                for status, count in progress.status_counts
            },
        },
        "payload": {
            "byte_order": "little",
            "block_count": progress.block_count,
            "voxels_per_block": TSDF_BLOCK_VOXELS,
            "layout": list(_PAYLOAD_LAYOUT),
            "block_indices_sha256": index_digest,
            "tsdf_sums_sha256": progress.tsdf_sums_sha256,
            "weights_sha256": progress.weights_sha256,
        },
    }


def _encode_header(header: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            header,
            ensure_ascii=True,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _parse_header(
    header_bytes: bytes,
) -> tuple[TsdfStreamFusionProgress, dict[str, Any]]:
    try:
        text = header_bytes.decode("ascii")
    except UnicodeError as error:
        raise TsdfError(
            "TSDF fusion checkpoint header must be ASCII JSON"
        ) from error
    depth = 0
    for character in text:
        if character in "[{":
            depth += 1
            if depth > _MAX_HEADER_NESTING:
                raise TsdfError(
                    "TSDF fusion checkpoint header is nested too deeply"
                )
        elif character in "]}":
            depth -= 1
    try:
        document = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except _DuplicateJsonKey as error:
        raise TsdfError(
            "TSDF fusion checkpoint header contains duplicate JSON key "
            f"{error.key!r}"
        ) from error
    except (json.JSONDecodeError, ValueError, RecursionError) as error:
        raise TsdfError(
            f"TSDF fusion checkpoint header is invalid JSON: {error}"
        ) from error

    root = _object(
        document,
        "checkpoint",
        (
            "schema",
            "schema_version",
            "session_id",
            "replay_digest_sha256",
            "source_plan_digest_sha256",
            "progress",
            "payload",
        ),
    )
    _exact(root["schema"], TSDF_FUSION_CHECKPOINT_SCHEMA, "checkpoint.schema")
    _exact(
        root["schema_version"],
        TSDF_FUSION_CHECKPOINT_SCHEMA_VERSION,
        "checkpoint.schema_version",
    )
    if not isinstance(root["session_id"], str) or not _IDENTIFIER.fullmatch(
        root["session_id"]
    ):
        raise TsdfError("checkpoint.session_id: invalid identifier")
    _sha256(root["replay_digest_sha256"], "checkpoint.replay_digest_sha256")
    _sha256(
        root["source_plan_digest_sha256"],
        "checkpoint.source_plan_digest_sha256",
    )

    recorded = _object(
        root["progress"],
        "checkpoint.progress",
        _PROGRESS_COUNTS + ("status_counts",),
    )
    for name in _PROGRESS_COUNTS:
        _integer(recorded[name], f"checkpoint.progress.{name}", minimum=0)
    counts = recorded["status_counts"]
    if not isinstance(counts, dict):
        raise TsdfError(
            "checkpoint.progress.status_counts: expected an object"
        )
    unknown = sorted(set(counts) - set(_STATUS_BY_VALUE))
    if unknown:
        raise TsdfError(
            "checkpoint.progress.status_counts: unknown status "
            f"{unknown[0]!r}"
        )
    for name, count in counts.items():
        _integer(
            count,
            f"checkpoint.progress.status_counts.{name}",
            minimum=1,
        )

    payload = _object(
        root["payload"],
        "checkpoint.payload",
        (
            "byte_order",
            "block_count",
            "voxels_per_block",
            "layout",
            "block_indices_sha256",
            "tsdf_sums_sha256",
            "weights_sha256",
        ),
    )
    _exact(payload["byte_order"], "little", "checkpoint.payload.byte_order")
    _integer(
        payload["block_count"],
        "checkpoint.payload.block_count",
        minimum=1,
    )
    if payload["block_count"] > MAX_TSDF_BLOCK_STORAGE_BLOCKS:
        raise TsdfError(
            "checkpoint.payload.block_count: maximum is "
            f"{MAX_TSDF_BLOCK_STORAGE_BLOCKS}"
        )
    _exact(
        payload["voxels_per_block"],
        TSDF_BLOCK_VOXELS,
        "checkpoint.payload.voxels_per_block",
    )
    _exact(
        payload["layout"],
        list(_PAYLOAD_LAYOUT),
        "checkpoint.payload.layout",
    )
    for name in ("block_indices_sha256", "tsdf_sums_sha256", "weights_sha256"):
        _sha256(payload[name], f"checkpoint.payload.{name}")

    try:
        progress = TsdfStreamFusionProgress(
            session_id=root["session_id"],
            source_plan_digest_sha256=root["source_plan_digest_sha256"],
            replay_digest_sha256=root["replay_digest_sha256"],
            block_count=payload["block_count"],
            voxel_slots=payload["block_count"] * TSDF_BLOCK_VOXELS,
            # Canonical order is the enumeration's, whatever order the
            # object was written in.
            status_counts=tuple(
                (status, counts[status.value])
                for status in TSDF_CONTRIBUTION_STATUS_ORDER
                if status.value in counts
            ),
            tsdf_sums_sha256=payload["tsdf_sums_sha256"],
            weights_sha256=payload["weights_sha256"],
            **{name: recorded[name] for name in _PROGRESS_COUNTS},
        )
    except TsdfError as error:
        raise TsdfError(f"checkpoint.progress: {error}") from error

    # A header that parses but is not the canonical encoding of its own
    # contents would let two different files describe one state.
    if _encode_header(root) != header_bytes:
        raise TsdfError(
            "TSDF fusion checkpoint header is not in canonical form"
        )
    return progress, payload


def _validate_accumulators(
    tsdf_sums: np.ndarray,
    weights: np.ndarray,
    progress: TsdfStreamFusionProgress,
) -> None:
    """Re-derive what the header claims from the payload it describes."""

    # A run of blocks at a time, as the volume loader does it.
    applied = 0
    maximum = 0
    for first in range(0, len(weights), _VALIDATION_CHUNK_BLOCKS):
        sums = tsdf_sums[first:first + _VALIDATION_CHUNK_BLOCKS]
        counts = weights[first:first + _VALIDATION_CHUNK_BLOCKS]
        if not bool(np.all(np.isfinite(sums))):
            raise TsdfError(
                "TSDF fusion checkpoint contains a non-finite sum"
            )
        if bool(np.any(np.abs(sums) > counts)):
            raise TsdfError(
                "TSDF fusion checkpoint sum exceeds its weight envelope"
            )
        unseen = sums[counts == 0]
        if bool(np.any(unseen != 0.0)) or bool(
            np.any(np.signbit(unseen))
        ):
            raise TsdfError(
                "TSDF fusion checkpoint unobserved voxel is not "
                "canonical zero"
            )
        applied += int(counts.sum(dtype=np.uint64))
        maximum = max(maximum, int(counts.max()))
    if applied != progress.applied_count:
        raise TsdfError(
            "TSDF fusion checkpoint progress does not match its payload"
        )
    if maximum > progress.fused_observations:
        raise TsdfError(
            "TSDF fusion checkpoint weight exceeds its fused observations"
        )


def _object(
    value: Any,
    label: str,
    keys: tuple[str, ...],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TsdfError(f"{label}: expected an object")
    unknown = sorted(set(value) - set(keys))
    if unknown:
        raise TsdfError(f"{label}: unknown field {unknown[0]!r}")
    for key in keys:
        if key not in value:
            raise TsdfError(f"{label}.{key}: missing")
    return value


def _exact(value: Any, expected: Any, label: str) -> None:
    if type(value) is not type(expected) or value != expected:
        raise TsdfError(
            f"{label}: expected {expected!r}, received {value!r}"
        )


def _sha256(value: Any, label: str) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise TsdfError(f"{label}: expected a SHA-256 digest")


def _integer(value: Any, label: str, *, minimum: int) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
    ):
        raise TsdfError(f"{label}: expected an integer of at least {minimum}")
