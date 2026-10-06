"""A persisted, verifiable sparse TSDF block volume.

Until now a fused block volume lived only in the process that produced it.
It could be measured but not saved, and therefore not meshed, rendered or
handed to anything else; every picture of a reconstruction had to come from
a separate, coarser dense volume.

A ``.sftvol`` file is one fused plan on disk::

    8 bytes   magic  b"SFTVOL01"
    8 bytes   header length, unsigned little-endian
    header    canonical ASCII JSON, newline terminated
    payload   block indices   int32   little-endian  [blocks, 3]
              TSDF sums       float64 little-endian  [blocks, 8, 8, 8]
              weights         uint32  little-endian  [blocks, 8, 8, 8]

The accumulators are stored as fused -- sums and weights, not the normalised
quotient -- so loading a volume gives back the exact bytes fusion produced
and nothing is lost to a division. The header carries the digests that tie
the file to its scan and its plan, a digest of each payload array, and the
fusion counts; the loader re-derives every count it can from the payload and
refuses a file whose header and payload disagree.
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
from pathlib import Path
from typing import Any

import numpy as np

from .errors import TsdfError
from .tsdf_block_plan import (
    MAX_BLOCK_INDEX,
    MIN_BLOCK_INDEX,
    TSDF_BLOCK_RESOLUTION,
)
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
    TsdfStreamFusionReceipt,
    storage_payload_sha256,
)
from .tsdf_voxel_update import _validate_update_storage

TSDF_BLOCK_VOLUME_SCHEMA = "spatialforge.tsdf-block-volume"
TSDF_BLOCK_VOLUME_SCHEMA_VERSION = "0.1.0"
TSDF_BLOCK_VOLUME_MAGIC = b"SFTVOL01"
MAX_TSDF_BLOCK_VOLUME_HEADER_BYTES = 64 * 1024

_PREAMBLE_BYTES = len(TSDF_BLOCK_VOLUME_MAGIC) + 8
_INDEX_DTYPE = np.dtype("<i4")
_SUM_DTYPE = np.dtype("<f8")
_WEIGHT_DTYPE = np.dtype("<u4")
_BYTES_PER_BLOCK = (
    3 * _INDEX_DTYPE.itemsize
    + TSDF_BLOCK_VOXELS * (_SUM_DTYPE.itemsize + _WEIGHT_DTYPE.itemsize)
)
MAX_TSDF_BLOCK_VOLUME_BYTES = (
    _PREAMBLE_BYTES
    + MAX_TSDF_BLOCK_VOLUME_HEADER_BYTES
    + MAX_TSDF_BLOCK_STORAGE_BLOCKS * _BYTES_PER_BLOCK
)

_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_INDEX_ORDER = "x-fastest-then-y-then-z"
_BLOCK_ORDER = "strictly-increasing-z-then-y-then-x"
_TSDF_SIGN = "positive-free-space-negative-behind-surface"
_NORMALIZATION = "tsdf-sum-divided-by-weight"
_UNKNOWN_RULE = "weight-zero"
_MAX_HEADER_NESTING = 8
_PAYLOAD_LAYOUT = (
    "block_indices:int32:[blocks,3]",
    "tsdf_sums:float64:[blocks,8,8,8]",
    "weights:uint32:[blocks,8,8,8]",
)

_BlockIndex = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class TsdfBlockVolumeReport:
    """What one write produced."""

    output: Path
    output_digest_sha256: str
    output_bytes: int
    session_id: str
    block_count: int
    voxel_slots: int
    observed_voxel_count: int
    contributions_applied: int
    maximum_weight: int
    source_plan_digest_sha256: str
    replay_digest_sha256: str


@dataclass(frozen=True, slots=True, eq=False)
class TsdfBlockVolume:
    """An immutable fused block volume loaded from a ``.sftvol`` file."""

    path: Path
    artifact_digest_sha256: str
    session_id: str
    replay_digest_sha256: str
    source_plan_digest_sha256: str
    voxel_size_m: float
    truncation_m: float
    block_resolution: int
    frame_stride: int
    total_observations: int
    selected_observations: int
    fused_observations: int
    skipped_missing_depth: int
    skipped_missing_pose: int
    contributions_evaluated: int
    contributions_applied: int
    observed_voxel_count: int
    maximum_weight: int
    block_indices: tuple[_BlockIndex, ...]
    tsdf_sums: np.ndarray
    weights: np.ndarray

    def __post_init__(self) -> None:
        shape = (
            len(self.block_indices),
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
                    f"TSDF block volume {label} must be an immutable "
                    f"{dtype.name} array of shape {shape}"
                )

    @property
    def block_count(self) -> int:
        return len(self.block_indices)

    @property
    def voxel_slots(self) -> int:
        return self.block_count * TSDF_BLOCK_VOXELS

    @property
    def unknown_voxel_count(self) -> int:
        return self.voxel_slots - self.observed_voxel_count

    def normalized_tsdf(self) -> np.ndarray:
        """Return sum/weight per voxel, NaN where nothing was observed."""

        result = np.full(self.tsdf_sums.shape, np.nan, dtype=np.float64)
        observed = self.weights > 0
        result[observed] = self.tsdf_sums[observed] / self.weights[observed]
        return result


def write_tsdf_block_volume(
    storage: TsdfBlockStorage,
    receipt: TsdfStreamFusionReceipt,
    output: str | Path,
) -> TsdfBlockVolumeReport:
    """Persist fused storage, refusing anything the receipt does not cover."""

    if not isinstance(storage, TsdfBlockStorage):
        raise TsdfError(
            "TSDF block volume requires allocated TsdfBlockStorage"
        )
    if not isinstance(receipt, TsdfStreamFusionReceipt):
        raise TsdfError(
            "TSDF block volume requires a TsdfStreamFusionReceipt"
        )
    _validate_update_storage(storage)
    plan = storage.source_plan
    output_path = _validate_output(output)

    if (
        receipt.source_plan_digest_sha256 != plan.artifact_digest_sha256
        or receipt.replay_digest_sha256 != plan.replay_digest_sha256
        or receipt.session_id != plan.session_id
        or receipt.block_count != storage.block_count
        or receipt.voxel_slots != storage.voxel_slots
    ):
        raise TsdfError(
            "TSDF stream fusion receipt does not describe this storage"
        )
    sums_digest = storage_payload_sha256(storage.tsdf_sums)
    weights_digest = storage_payload_sha256(storage.weights)
    if (
        receipt.tsdf_sums_sha256 != sums_digest
        or receipt.weights_sha256 != weights_digest
    ):
        raise TsdfError(
            "TSDF block storage changed after it was fused; refusing to "
            "persist bytes the fusion receipt does not cover"
        )

    block_indices = np.asarray(storage.block_indices, dtype=np.int64)
    index_payload = block_indices.astype(_INDEX_DTYPE).tobytes(order="C")
    sum_payload = storage.tsdf_sums.astype(_SUM_DTYPE, copy=False).tobytes(
        order="C"
    )
    weight_payload = storage.weights.astype(
        _WEIGHT_DTYPE,
        copy=False,
    ).tobytes(order="C")

    header = _build_header(
        session_id=plan.session_id,
        replay_digest=plan.replay_digest_sha256,
        plan_digest=plan.artifact_digest_sha256,
        voxel_size_m=float(plan.voxel_size_m),
        truncation_m=float(plan.truncation_m),
        frame_stride=receipt.frame_stride,
        total_observations=receipt.total_observations,
        selected_observations=receipt.selected_observations,
        fused_observations=receipt.fused_observations,
        skipped_missing_depth=receipt.skipped_missing_depth,
        skipped_missing_pose=receipt.skipped_missing_pose,
        contributions_evaluated=receipt.evaluated_count,
        contributions_applied=receipt.applied_count,
        observed_voxels=receipt.observed_voxel_count,
        maximum_weight=receipt.maximum_weight,
        block_count=storage.block_count,
        index_digest=hashlib.sha256(index_payload).hexdigest(),
        sums_digest=sums_digest,
        weights_digest=weights_digest,
    )
    encoded_header = _encode_header(header)
    if len(encoded_header) > MAX_TSDF_BLOCK_VOLUME_HEADER_BYTES:
        raise TsdfError("TSDF block volume header exceeds its size limit")

    encoded = b"".join(
        (
            TSDF_BLOCK_VOLUME_MAGIC,
            struct.pack("<Q", len(encoded_header)),
            encoded_header,
            index_payload,
            sum_payload,
            weight_payload,
        )
    )
    output_digest = hashlib.sha256(encoded).hexdigest()
    _write_without_overwrite(output_path, encoded)
    return TsdfBlockVolumeReport(
        output=output_path,
        output_digest_sha256=output_digest,
        output_bytes=len(encoded),
        session_id=plan.session_id,
        block_count=storage.block_count,
        voxel_slots=storage.voxel_slots,
        observed_voxel_count=receipt.observed_voxel_count,
        contributions_applied=receipt.applied_count,
        maximum_weight=receipt.maximum_weight,
        source_plan_digest_sha256=plan.artifact_digest_sha256,
        replay_digest_sha256=plan.replay_digest_sha256,
    )


def load_tsdf_block_volume(path: str | Path) -> TsdfBlockVolume:
    """Load and fully verify an immutable ``.sftvol`` artifact."""

    input_path = Path(path).resolve()
    if input_path.suffix.lower() != ".sftvol":
        raise TsdfError("TSDF block volume filename must end in .sftvol")
    if not input_path.is_file():
        raise TsdfError(f"TSDF block volume does not exist: {input_path}")
    try:
        with input_path.open("rb") as input_file:
            encoded = input_file.read(MAX_TSDF_BLOCK_VOLUME_BYTES + 1)
    except OSError as error:
        raise TsdfError(f"cannot read TSDF block volume: {error}") from error
    if len(encoded) > MAX_TSDF_BLOCK_VOLUME_BYTES:
        raise TsdfError(
            "TSDF block volume exceeds the maximum of "
            f"{MAX_TSDF_BLOCK_VOLUME_BYTES} bytes"
        )
    digest = hashlib.sha256(encoded).hexdigest()

    if len(encoded) < _PREAMBLE_BYTES or not encoded.startswith(
        TSDF_BLOCK_VOLUME_MAGIC
    ):
        raise TsdfError("TSDF block volume has an unrecognized signature")
    (header_length,) = struct.unpack_from(
        "<Q",
        encoded,
        len(TSDF_BLOCK_VOLUME_MAGIC),
    )
    if not 2 <= header_length <= MAX_TSDF_BLOCK_VOLUME_HEADER_BYTES:
        raise TsdfError("TSDF block volume header length is out of range")
    payload_offset = _PREAMBLE_BYTES + header_length
    if len(encoded) < payload_offset:
        raise TsdfError("TSDF block volume is truncated inside its header")
    header_bytes = encoded[_PREAMBLE_BYTES:payload_offset]
    header = _parse_header(header_bytes)

    block_count = header["payload"]["block_count"]
    expected_payload = block_count * _BYTES_PER_BLOCK
    if len(encoded) - payload_offset != expected_payload:
        raise TsdfError(
            "TSDF block volume payload is "
            f"{len(encoded) - payload_offset} bytes; its header describes "
            f"{expected_payload}"
        )
    index_bytes = 3 * _INDEX_DTYPE.itemsize * block_count
    sum_bytes = TSDF_BLOCK_VOXELS * _SUM_DTYPE.itemsize * block_count
    index_payload = encoded[payload_offset:payload_offset + index_bytes]
    sum_payload = encoded[
        payload_offset + index_bytes:payload_offset + index_bytes + sum_bytes
    ]
    weight_payload = encoded[payload_offset + index_bytes + sum_bytes:]
    for payload, name in (
        (index_payload, "block_indices"),
        (sum_payload, "tsdf_sums"),
        (weight_payload, "weights"),
    ):
        if (
            hashlib.sha256(payload).hexdigest()
            != header["payload"][f"{name}_sha256"]
        ):
            raise TsdfError(
                f"TSDF block volume {name} do not match their recorded "
                "digest"
            )

    shape = (
        block_count,
        TSDF_BLOCK_RESOLUTION,
        TSDF_BLOCK_RESOLUTION,
        TSDF_BLOCK_RESOLUTION,
    )
    indices = np.frombuffer(index_payload, dtype=_INDEX_DTYPE).reshape(
        (block_count, 3)
    )
    tsdf_sums = np.frombuffer(sum_payload, dtype=_SUM_DTYPE).reshape(shape)
    weights = np.frombuffer(weight_payload, dtype=_WEIGHT_DTYPE).reshape(
        shape
    )
    block_indices = _validate_block_order(indices)
    fusion = header["fusion"]
    _validate_accumulators(tsdf_sums, weights, fusion)

    return TsdfBlockVolume(
        path=input_path,
        artifact_digest_sha256=digest,
        session_id=header["session_id"],
        replay_digest_sha256=header["replay_digest_sha256"],
        source_plan_digest_sha256=header["source_plan_digest_sha256"],
        voxel_size_m=header["grid"]["voxel_size_m"],
        truncation_m=header["tsdf"]["truncation_m"],
        block_resolution=TSDF_BLOCK_RESOLUTION,
        frame_stride=fusion["frame_stride"],
        total_observations=fusion["total_observations"],
        selected_observations=fusion["selected_observations"],
        fused_observations=fusion["fused_observations"],
        skipped_missing_depth=fusion["skipped_missing_depth"],
        skipped_missing_pose=fusion["skipped_missing_pose"],
        contributions_evaluated=fusion["contributions_evaluated"],
        contributions_applied=fusion["contributions_applied"],
        observed_voxel_count=fusion["observed_voxels"],
        maximum_weight=fusion["maximum_weight"],
        block_indices=block_indices,
        tsdf_sums=tsdf_sums,
        weights=weights,
    )


def _build_header(
    *,
    session_id: str,
    replay_digest: str,
    plan_digest: str,
    voxel_size_m: float,
    truncation_m: float,
    frame_stride: int,
    total_observations: int,
    selected_observations: int,
    fused_observations: int,
    skipped_missing_depth: int,
    skipped_missing_pose: int,
    contributions_evaluated: int,
    contributions_applied: int,
    observed_voxels: int,
    maximum_weight: int,
    block_count: int,
    index_digest: str,
    sums_digest: str,
    weights_digest: str,
) -> dict[str, Any]:
    return {
        "schema": TSDF_BLOCK_VOLUME_SCHEMA,
        "schema_version": TSDF_BLOCK_VOLUME_SCHEMA_VERSION,
        "session_id": session_id,
        "replay_digest_sha256": replay_digest,
        "source_plan_digest_sha256": plan_digest,
        "grid": {
            "world_anchor_m": [0.0, 0.0, 0.0],
            "voxel_size_m": voxel_size_m,
            "block_resolution": TSDF_BLOCK_RESOLUTION,
            "index_order": _INDEX_ORDER,
            "block_order": _BLOCK_ORDER,
        },
        "tsdf": {
            "truncation_m": truncation_m,
            "sign": _TSDF_SIGN,
            "normalization": _NORMALIZATION,
            "unknown": _UNKNOWN_RULE,
        },
        "fusion": {
            "frame_stride": frame_stride,
            "total_observations": total_observations,
            "selected_observations": selected_observations,
            "fused_observations": fused_observations,
            "skipped_missing_depth": skipped_missing_depth,
            "skipped_missing_pose": skipped_missing_pose,
            "contributions_evaluated": contributions_evaluated,
            "contributions_applied": contributions_applied,
            "observed_voxels": observed_voxels,
            "maximum_weight": maximum_weight,
        },
        "payload": {
            "byte_order": "little",
            "block_count": block_count,
            "voxels_per_block": TSDF_BLOCK_VOXELS,
            "layout": list(_PAYLOAD_LAYOUT),
            "block_indices_sha256": index_digest,
            "tsdf_sums_sha256": sums_digest,
            "weights_sha256": weights_digest,
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


def _parse_header(header_bytes: bytes) -> dict[str, Any]:
    try:
        text = header_bytes.decode("ascii")
    except UnicodeError as error:
        raise TsdfError(
            "TSDF block volume header must be ASCII JSON"
        ) from error
    depth = 0
    for character in text:
        if character in "[{":
            depth += 1
            if depth > _MAX_HEADER_NESTING:
                raise TsdfError(
                    "TSDF block volume header is nested too deeply"
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
            "TSDF block volume header contains duplicate JSON key "
            f"{error.key!r}"
        ) from error
    except (json.JSONDecodeError, ValueError, RecursionError) as error:
        raise TsdfError(
            f"TSDF block volume header is invalid JSON: {error}"
        ) from error

    root = _object(
        document,
        "volume",
        (
            "schema",
            "schema_version",
            "session_id",
            "replay_digest_sha256",
            "source_plan_digest_sha256",
            "grid",
            "tsdf",
            "fusion",
            "payload",
        ),
    )
    _exact(root["schema"], TSDF_BLOCK_VOLUME_SCHEMA, "volume.schema")
    _exact(
        root["schema_version"],
        TSDF_BLOCK_VOLUME_SCHEMA_VERSION,
        "volume.schema_version",
    )
    if not isinstance(root["session_id"], str) or not _IDENTIFIER.fullmatch(
        root["session_id"]
    ):
        raise TsdfError("volume.session_id: invalid identifier")
    _sha256(root["replay_digest_sha256"], "volume.replay_digest_sha256")
    _sha256(
        root["source_plan_digest_sha256"],
        "volume.source_plan_digest_sha256",
    )

    grid = _object(
        root["grid"],
        "volume.grid",
        (
            "world_anchor_m",
            "voxel_size_m",
            "block_resolution",
            "index_order",
            "block_order",
        ),
    )
    if grid["world_anchor_m"] != [0.0, 0.0, 0.0] or any(
        type(component) is not float for component in grid["world_anchor_m"]
    ):
        raise TsdfError("volume.grid.world_anchor_m: expected the origin")
    voxel_size_m = _positive_float(
        grid["voxel_size_m"],
        "volume.grid.voxel_size_m",
    )
    _exact(
        grid["block_resolution"],
        TSDF_BLOCK_RESOLUTION,
        "volume.grid.block_resolution",
    )
    _exact(grid["index_order"], _INDEX_ORDER, "volume.grid.index_order")
    _exact(grid["block_order"], _BLOCK_ORDER, "volume.grid.block_order")

    tsdf = _object(
        root["tsdf"],
        "volume.tsdf",
        ("truncation_m", "sign", "normalization", "unknown"),
    )
    truncation_m = _positive_float(
        tsdf["truncation_m"],
        "volume.tsdf.truncation_m",
    )
    if truncation_m < voxel_size_m:
        raise TsdfError(
            "volume.tsdf.truncation_m: must be at least voxel_size_m"
        )
    _exact(tsdf["sign"], _TSDF_SIGN, "volume.tsdf.sign")
    _exact(
        tsdf["normalization"],
        _NORMALIZATION,
        "volume.tsdf.normalization",
    )
    _exact(tsdf["unknown"], _UNKNOWN_RULE, "volume.tsdf.unknown")

    fusion = _object(
        root["fusion"],
        "volume.fusion",
        (
            "frame_stride",
            "total_observations",
            "selected_observations",
            "fused_observations",
            "skipped_missing_depth",
            "skipped_missing_pose",
            "contributions_evaluated",
            "contributions_applied",
            "observed_voxels",
            "maximum_weight",
        ),
    )
    for key in (
        "frame_stride",
        "total_observations",
        "selected_observations",
    ):
        _integer(fusion[key], f"volume.fusion.{key}", minimum=1)
    for key in (
        "fused_observations",
        "skipped_missing_depth",
        "skipped_missing_pose",
        "contributions_evaluated",
        "contributions_applied",
        "observed_voxels",
        "maximum_weight",
    ):
        _integer(fusion[key], f"volume.fusion.{key}", minimum=0)
    expected_selected = (
        (fusion["total_observations"] - 1) // fusion["frame_stride"]
    ) + 1
    if fusion["selected_observations"] != expected_selected:
        raise TsdfError(
            "volume.fusion.selected_observations: does not match the stride"
        )
    if fusion["fused_observations"] > fusion["selected_observations"]:
        raise TsdfError(
            "volume.fusion.fused_observations: exceeds the selection"
        )

    payload = _object(
        root["payload"],
        "volume.payload",
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
    _exact(payload["byte_order"], "little", "volume.payload.byte_order")
    _integer(payload["block_count"], "volume.payload.block_count", minimum=1)
    if payload["block_count"] > MAX_TSDF_BLOCK_STORAGE_BLOCKS:
        raise TsdfError(
            "volume.payload.block_count: exceeds the "
            f"{MAX_TSDF_BLOCK_STORAGE_BLOCKS}-block storage limit"
        )
    _exact(
        payload["voxels_per_block"],
        TSDF_BLOCK_VOXELS,
        "volume.payload.voxels_per_block",
    )
    _exact(payload["layout"], list(_PAYLOAD_LAYOUT), "volume.payload.layout")
    for key in (
        "block_indices_sha256",
        "tsdf_sums_sha256",
        "weights_sha256",
    ):
        _sha256(payload[key], f"volume.payload.{key}")
    if fusion["contributions_evaluated"] != (
        payload["block_count"]
        * TSDF_BLOCK_VOXELS
        * fusion["selected_observations"]
    ):
        raise TsdfError(
            "volume.fusion.contributions_evaluated: does not cover every "
            "voxel-observation"
        )

    # A header that parses but is not the canonical encoding of its own
    # contents would let two different files describe one volume.
    if _encode_header(root) != header_bytes:
        raise TsdfError(
            "TSDF block volume header is not in canonical form"
        )
    return root


def _validate_block_order(indices: np.ndarray) -> tuple[_BlockIndex, ...]:
    values = indices.astype(np.int64)
    if np.any(values < MIN_BLOCK_INDEX) or np.any(values > MAX_BLOCK_INDEX):
        raise TsdfError(
            "TSDF block volume block index is outside the planning range"
        )
    if len(values) > 1:
        later = values[1:]
        earlier = values[:-1]
        increasing = (later[:, 2] > earlier[:, 2]) | (
            (later[:, 2] == earlier[:, 2])
            & (
                (later[:, 1] > earlier[:, 1])
                | (
                    (later[:, 1] == earlier[:, 1])
                    & (later[:, 0] > earlier[:, 0])
                )
            )
        )
        if not bool(np.all(increasing)):
            raise TsdfError(
                "TSDF block volume blocks are not unique and in canonical "
                "order"
            )
    return tuple((x, y, z) for x, y, z in values.tolist())


def _validate_accumulators(
    tsdf_sums: np.ndarray,
    weights: np.ndarray,
    fusion: dict[str, Any],
) -> None:
    """Re-derive what the header claims from the payload it describes."""

    if not bool(np.all(np.isfinite(tsdf_sums))):
        raise TsdfError("TSDF block volume contains a non-finite sum")
    if bool(np.any(np.abs(tsdf_sums) > weights)):
        raise TsdfError(
            "TSDF block volume sum exceeds its weight envelope"
        )
    unobserved = weights == 0
    if bool(np.any(np.signbit(tsdf_sums[unobserved]))):
        raise TsdfError(
            "TSDF block volume unobserved voxel is not canonical zero"
        )
    observed = int(np.count_nonzero(weights))
    maximum = int(weights.max())
    applied = int(weights.sum(dtype=np.uint64))
    if (
        observed != fusion["observed_voxels"]
        or maximum != fusion["maximum_weight"]
        or applied != fusion["contributions_applied"]
    ):
        raise TsdfError(
            "TSDF block volume fusion summary does not match its payload"
        )
    if maximum > fusion["fused_observations"]:
        raise TsdfError(
            "TSDF block volume weight exceeds its fused observations"
        )


def _validate_output(output: str | Path) -> Path:
    output_path = Path(output).resolve()
    if output_path.suffix.lower() != ".sftvol":
        raise TsdfError("output filename must end in .sftvol")
    if output_path.exists():
        raise TsdfError(f"output already exists: {output_path}")
    return output_path


def _write_without_overwrite(output_path: Path, encoded: bytes) -> None:
    temporary_path: Path | None = None
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output_path.stem}-",
            suffix=".sftvol",
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
        except OSError as error:
            raise TsdfError(
                "cannot publish TSDF block volume without overwriting: "
                f"{error}"
            ) from error
    except TsdfError:
        raise
    except OSError as error:
        raise TsdfError(
            f"cannot write TSDF block volume: {error}"
        ) from error
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


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


def _positive_float(value: Any, label: str) -> float:
    if type(value) is not float or not math.isfinite(value) or value <= 0.0:
        raise TsdfError(f"{label}: expected a finite positive number")
    return value
