"""Deterministic candidate TSDF block planning from known-pose depth."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .errors import PointCloudError, TsdfError
from .model import CameraCalibration, Observation, ScanSession
from .point_cloud import (
    _read_depth,
    _read_depth_array,
    _sample_path,
    _transform_point,
    _validate_reconstruction_contract,
)
from .replay import replay_session
from .tsdf import (
    _validate_positive_integer,
    _validate_positive_number,
)

TSDF_BLOCK_PLAN_SCHEMA = "spatialforge.tsdf-block-plan"
TSDF_BLOCK_PLAN_SCHEMA_VERSION = "0.1.0"
# An expanded plan is a different claim from the plan it grew from, so it
# is a different version rather than the same version with extra fields.
TSDF_EXPANDED_BLOCK_PLAN_SCHEMA_VERSION = "0.2.0"
TSDF_BLOCK_PLAN_SCHEMA_VERSIONS = (
    TSDF_BLOCK_PLAN_SCHEMA_VERSION,
    TSDF_EXPANDED_BLOCK_PLAN_SCHEMA_VERSION,
)

# A surface/truncation plan makes no free-space claim. An expanded plan adds
# blocks approved from conservative nearest-pixel footprint coverage, so it
# must say so rather than inheriting the original's silence.
TSDF_FREE_SPACE_RULE_NOT_PLANNED = "not-planned"
TSDF_FREE_SPACE_RULE_FOOTPRINT = "conservative-nearest-pixel-footprint"
TSDF_FREE_SPACE_RULES = (
    TSDF_FREE_SPACE_RULE_NOT_PLANNED,
    TSDF_FREE_SPACE_RULE_FOOTPRINT,
)
TSDF_BLOCK_RESOLUTION = 8
MAX_PLANNED_BLOCKS = 100_000
# Widest per-sample block span the vectorised planner expands itself. A
# truncation band many blocks wide is legal but pathological, and is left to
# the per-pixel reference path rather than given a fast path of its own.
MAX_VECTOR_SPAN_OFFSETS = 4_096
MIN_BLOCK_INDEX = -(2**31)
MAX_BLOCK_INDEX = 2**31 - 1

_BlockIndex = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class TsdfBlockPlanReport:
    session_id: str
    output: Path
    total_observations: int
    selected_observations: int
    paired_observations: int
    skipped_missing_depth: int
    skipped_missing_pose: int
    valid_depth_points: int
    invalid_depth_samples: int
    voxel_size_m: float
    truncation_m: float
    block_resolution: int
    block_extent_m: float
    surface_blocks: tuple[_BlockIndex, ...]
    active_blocks: tuple[_BlockIndex, ...]
    planned_voxel_slots: int
    min_block_index: _BlockIndex
    max_block_index: _BlockIndex
    replay_digest_sha256: str
    output_digest_sha256: str

    @property
    def surface_block_count(self) -> int:
        return len(self.surface_blocks)

    @property
    def active_block_count(self) -> int:
        return len(self.active_blocks)

    @property
    def halo_block_count(self) -> int:
        return self.active_block_count - self.surface_block_count


def plan_tsdf_blocks(
    session: ScanSession,
    output: str | Path,
    *,
    voxel_size_m: float,
    truncation_m: float,
    frame_stride: int = 1,
) -> TsdfBlockPlanReport:
    """Plan candidate surface-neighborhood blocks without fusing a TSDF."""

    voxel_size = _validate_positive_number(voxel_size_m, "voxel_size_m")
    truncation = _validate_positive_number(truncation_m, "truncation_m")
    if truncation < voxel_size:
        raise TsdfError("truncation_m must be greater than or equal to voxel_size_m")
    _validate_positive_integer(frame_stride, "frame_stride")
    block_extent = voxel_size * TSDF_BLOCK_RESOLUTION
    if not math.isfinite(block_extent) or block_extent <= 0.0:
        raise TsdfError("TSDF block extent must remain finite and positive")
    output_path = _validate_tsdf_block_plan_output(output)

    try:
        camera, depth_scale_m = _validate_reconstruction_contract(session)
    except PointCloudError as error:
        raise TsdfError(str(error)) from error

    replay = replay_session(session)
    selected = tuple(
        observation
        for observation in replay.observations
        if observation.sequence % frame_stride == 0
    )
    surface_blocks: set[_BlockIndex] = set()
    active_blocks: set[_BlockIndex] = set()
    paired_observations = 0
    skipped_missing_depth = 0
    skipped_missing_pose = 0
    valid_depth_points = 0
    invalid_depth_samples = 0

    for observation in selected:
        missing_depth = observation.depth is None
        missing_pose = observation.pose is None
        if missing_depth:
            skipped_missing_depth += 1
        if missing_pose:
            skipped_missing_pose += 1
        if missing_depth or missing_pose:
            continue

        paired_observations += 1
        frame_valid, frame_invalid = _plan_observation_blocks(
            session,
            observation,
            camera,
            depth_scale_m,
            block_extent,
            truncation,
            surface_blocks,
            active_blocks,
        )
        valid_depth_points += frame_valid
        invalid_depth_samples += frame_invalid

    if paired_observations == 0:
        raise TsdfError(
            "no selected RGB observation has both exact depth and pose "
            "for TSDF block planning"
        )
    if valid_depth_points == 0:
        raise TsdfError(
            "selected frames contain no positive finite depth samples "
            "for TSDF block planning"
        )
    if not surface_blocks.issubset(active_blocks):
        raise TsdfError("TSDF block planning lost a surface block")

    ending_replay = replay_session(session)
    if ending_replay.digest_sha256 != replay.digest_sha256:
        raise TsdfError(
            "session inputs changed while planning TSDF blocks; rerun the command"
        )

    ordered_surface = _ordered_blocks(surface_blocks)
    ordered_active = _ordered_blocks(active_blocks)
    minimum, maximum = _block_bounds(ordered_active)
    planned_voxel_slots = (
        len(ordered_active) * TSDF_BLOCK_RESOLUTION**3
    )
    document = _build_plan_document(
        session,
        replay.digest_sha256,
        voxel_size,
        truncation,
        block_extent,
        frame_stride,
        len(replay.observations),
        len(selected),
        paired_observations,
        skipped_missing_depth,
        skipped_missing_pose,
        valid_depth_points,
        invalid_depth_samples,
        ordered_surface,
        ordered_active,
        planned_voxel_slots,
        minimum,
        maximum,
    )
    encoded = (
        json.dumps(
            document,
            ensure_ascii=True,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")
    output_digest = hashlib.sha256(encoded).hexdigest()
    _write_plan_without_overwrite(output_path, encoded)

    return TsdfBlockPlanReport(
        session_id=session.session_id,
        output=output_path,
        total_observations=len(replay.observations),
        selected_observations=len(selected),
        paired_observations=paired_observations,
        skipped_missing_depth=skipped_missing_depth,
        skipped_missing_pose=skipped_missing_pose,
        valid_depth_points=valid_depth_points,
        invalid_depth_samples=invalid_depth_samples,
        voxel_size_m=voxel_size,
        truncation_m=truncation,
        block_resolution=TSDF_BLOCK_RESOLUTION,
        block_extent_m=block_extent,
        surface_blocks=ordered_surface,
        active_blocks=ordered_active,
        planned_voxel_slots=planned_voxel_slots,
        min_block_index=minimum,
        max_block_index=maximum,
        replay_digest_sha256=replay.digest_sha256,
        output_digest_sha256=output_digest,
    )


def _plan_observation_blocks(
    session: ScanSession,
    observation: Observation,
    camera: CameraCalibration,
    depth_scale_m: float,
    block_extent_m: float,
    truncation_m: float,
    surface_blocks: set[_BlockIndex],
    active_blocks: set[_BlockIndex],
) -> tuple[int, int]:
    """Plan one observation's blocks, vectorised over its depth image.

    The per-pixel loop in ``_plan_observation_blocks_scalar`` is the
    reference definition. This evaluates the same arithmetic in the same
    order for every valid pixel at once, so the selected blocks and the
    sample counts are identical and the written plan is byte-identical.

    Anything irregular -- a non-finite coordinate, an index outside the
    planning range, an empty or oversized span -- is not handled here at
    all. The frame is handed to the reference path untouched, which either
    raises its precise per-pixel error or completes normally. That keeps
    every failure message exactly what the reference produces without
    restating its error logic a second time.
    """

    planned = _plan_observation_blocks_vector(
        session,
        observation,
        camera,
        depth_scale_m,
        block_extent_m,
        truncation_m,
    )
    if planned is None:
        return _plan_observation_blocks_scalar(
            session,
            observation,
            camera,
            depth_scale_m,
            block_extent_m,
            truncation_m,
            surface_blocks,
            active_blocks,
        )

    frame_surface, frame_active, valid_points, invalid_samples = planned
    surface_blocks.update(frame_surface)
    active_blocks.update(frame_active)
    if len(active_blocks) > MAX_PLANNED_BLOCKS:
        raise TsdfError(
            "TSDF block plan exceeds "
            f"{MAX_PLANNED_BLOCKS} candidate blocks. "
            "Increase --voxel-size-m or clean the "
            "depth/pose input."
        )
    return valid_points, invalid_samples


def _plan_observation_blocks_vector(
    session: ScanSession,
    observation: Observation,
    camera: CameraCalibration,
    depth_scale_m: float,
    block_extent_m: float,
    truncation_m: float,
) -> tuple[set[_BlockIndex], set[_BlockIndex], int, int] | None:
    """Return one frame's blocks and counts, or None to defer to the scalar.

    Nothing outside this function is mutated, so a deferred frame reaches
    the reference path with the plan exactly as the previous frame left it.
    """

    if observation.depth is None or observation.pose is None:
        raise AssertionError("caller must filter incomplete observations")

    try:
        depth_path = _sample_path(session, observation.depth.data, "depth")
        raw_depth = _read_depth_array(
            depth_path,
            camera.width,
            camera.height,
        )
    except PointCloudError as error:
        raise TsdfError(str(error)) from error

    transform = tuple(observation.pose.data["T_world_camera"])
    try:
        matrix = [float(component) for component in transform[:12]]
    except (TypeError, ValueError, OverflowError):
        return None
    if len(matrix) != 12 or raw_depth.shape != (
        camera.width * camera.height,
    ):
        return None

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        z_all = raw_depth * depth_scale_m
        valid = (raw_depth > 0) & np.isfinite(z_all) & (z_all > 0.0)
        flat = np.flatnonzero(valid)
        valid_points = int(flat.size)
        invalid_samples = int(raw_depth.size) - valid_points
        if valid_points == 0:
            return set(), set(), 0, invalid_samples

        z_camera = z_all[flat]
        u = flat % camera.width
        v = flat // camera.width
        x_camera = (u - camera.cx) * z_camera / camera.fx
        y_camera = (v - camera.cy) * z_camera / camera.fy
        world = (
            matrix[0] * x_camera
            + matrix[1] * y_camera
            + matrix[2] * z_camera
            + matrix[3],
            matrix[4] * x_camera
            + matrix[5] * y_camera
            + matrix[6] * z_camera
            + matrix[7],
            matrix[8] * x_camera
            + matrix[9] * y_camera
            + matrix[10] * z_camera
            + matrix[11],
        )
        if not all(bool(np.all(np.isfinite(axis))) for axis in world):
            return None

        surface_axes = []
        lower_axes = []
        upper_axes = []
        for coordinate in world:
            surface = _containing_block_index_vector(
                coordinate,
                block_extent_m,
            )
            span = _candidate_block_span_vector(
                coordinate,
                truncation_m,
                block_extent_m,
            )
            if surface is None or span is None:
                return None
            surface_axes.append(surface)
            lower_axes.append(span[0])
            upper_axes.append(span[1])

    widths = [
        upper - lower + 1
        for lower, upper in zip(lower_axes, upper_axes, strict=True)
    ]
    # Exact Python integers: three in-range widths can overflow int64 when
    # multiplied, and an oversized span must reach the reference path, which
    # reports it per sample.
    widest = [int(width.max()) for width in widths]
    if widest[0] * widest[1] * widest[2] > min(
        MAX_PLANNED_BLOCKS,
        MAX_VECTOR_SPAN_OFFSETS,
    ):
        return None

    frame_surface = _unique_rows(np.stack(surface_axes, axis=1))

    # Many pixels share one span box, so the boxes are deduplicated before
    # being expanded into the blocks they cover.
    boxes = _unique_rows(
        np.stack(
            [
                lower_axes[0],
                lower_axes[1],
                lower_axes[2],
                widths[0],
                widths[1],
                widths[2],
            ],
            axis=1,
        )
    )
    covered = []
    for offset_z in range(widest[2]):
        within_z = boxes[:, 5] > offset_z
        for offset_y in range(widest[1]):
            within_zy = within_z & (boxes[:, 4] > offset_y)
            for offset_x in range(widest[0]):
                selected = boxes[within_zy & (boxes[:, 3] > offset_x)]
                if selected.size:
                    covered.append(
                        selected[:, :3]
                        + np.array(
                            [offset_x, offset_y, offset_z],
                            dtype=np.int64,
                        )
                    )
    frame_active = _unique_rows(np.concatenate(covered, axis=0))
    return (
        _block_set(frame_surface),
        _block_set(frame_active),
        valid_points,
        invalid_samples,
    )


def _containing_block_index_vector(
    coordinate: np.ndarray,
    block_extent_m: float,
) -> np.ndarray | None:
    """Vector ``_containing_block_index``; None if any sample is irregular."""

    ratio = coordinate / block_extent_m
    if not np.all(np.isfinite(ratio)):
        return None
    index = np.floor(ratio)
    lower = index * block_extent_m
    upper = (index + 1.0) * block_extent_m
    if not (np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))):
        return None
    below = lower > coordinate
    above = ~below & (upper <= coordinate)
    index = index - below + above
    if np.any(index < MIN_BLOCK_INDEX) or np.any(index > MAX_BLOCK_INDEX):
        return None
    return index.astype(np.int64)


def _candidate_block_span_vector(
    coordinate: np.ndarray,
    truncation_m: float,
    block_extent_m: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Vector ``_candidate_block_span``; None if any sample is irregular."""

    lower_bound = coordinate - truncation_m
    upper_bound = coordinate + truncation_m
    lower_ratio = lower_bound / block_extent_m
    upper_ratio = upper_bound / block_extent_m
    if not all(
        bool(np.all(np.isfinite(values)))
        for values in (lower_bound, upper_bound, lower_ratio, upper_ratio)
    ):
        return None

    lower_index = np.floor(lower_ratio)
    upper_exclusive = np.ceil(upper_ratio)
    snapped_lower = lower_index * block_extent_m
    snapped_upper = upper_exclusive * block_extent_m
    if not (
        np.all(np.isfinite(snapped_lower))
        and np.all(np.isfinite(snapped_upper))
    ):
        return None
    lower_index = lower_index - (snapped_lower > lower_bound)
    upper_exclusive = upper_exclusive + (snapped_upper < upper_bound)
    upper_index = upper_exclusive - 1.0
    for index in (lower_index, upper_index):
        if np.any(index < MIN_BLOCK_INDEX) or np.any(
            index > MAX_BLOCK_INDEX
        ):
            return None
    if np.any(upper_index < lower_index):
        return None
    return lower_index.astype(np.int64), upper_index.astype(np.int64)


def _unique_rows(rows: np.ndarray) -> np.ndarray:
    """Distinct integer rows, via one collision-free scalar key per row.

    Sorting rows directly costs a structured comparison per element and
    dominated the planner. Each column is instead shifted to start at zero
    and the columns are packed mixed-radix into a single int64, which sorts
    an order of magnitude faster and decodes exactly. Inputs whose packed
    range would not fit fall back to the row-wise sort.
    """

    minimum = rows.min(axis=0)
    radices = [int(value) for value in rows.max(axis=0) - minimum + 1]
    capacity = 1
    for radix in radices:
        capacity *= radix
    if capacity >= 2**62:
        return np.unique(rows, axis=0)

    shifted = rows - minimum
    keys = np.zeros(len(rows), dtype=np.int64)
    for column in range(rows.shape[1] - 1, -1, -1):
        keys = keys * radices[column] + shifted[:, column]
    keys = np.unique(keys)

    decoded = np.empty((len(keys), rows.shape[1]), dtype=np.int64)
    for column in range(rows.shape[1]):
        decoded[:, column] = keys % radices[column]
        keys = keys // radices[column]
    return decoded + minimum


def _block_set(rows: np.ndarray) -> set[_BlockIndex]:
    return {(x, y, z) for x, y, z in rows.tolist()}


def _plan_observation_blocks_scalar(
    session: ScanSession,
    observation: Observation,
    camera: CameraCalibration,
    depth_scale_m: float,
    block_extent_m: float,
    truncation_m: float,
    surface_blocks: set[_BlockIndex],
    active_blocks: set[_BlockIndex],
) -> tuple[int, int]:
    """The per-pixel reference definition of block planning."""

    if observation.depth is None or observation.pose is None:
        raise AssertionError("caller must filter incomplete observations")

    try:
        depth_path = _sample_path(session, observation.depth.data, "depth")
        depth_pixels = _read_depth(
            depth_path,
            camera.width,
            camera.height,
        )
    except PointCloudError as error:
        raise TsdfError(str(error)) from error

    transform = tuple(observation.pose.data["T_world_camera"])
    valid_points = 0
    invalid_samples = 0
    for v in range(camera.height):
        row_offset = v * camera.width
        for u in range(camera.width):
            raw_depth = depth_pixels[row_offset + u]
            if raw_depth <= 0:
                invalid_samples += 1
                continue
            z_camera = raw_depth * depth_scale_m
            if not math.isfinite(z_camera) or z_camera <= 0.0:
                invalid_samples += 1
                continue

            x_camera = (u - camera.cx) * z_camera / camera.fx
            y_camera = (v - camera.cy) * z_camera / camera.fy
            world = _transform_point(
                transform,
                x_camera,
                y_camera,
                z_camera,
            )
            if not all(math.isfinite(value) for value in world):
                raise TsdfError(
                    "TSDF block planning produced a non-finite coordinate "
                    f"for RGB sample {observation.rgb.id!r} at pixel ({u}, {v})"
                )

            surface_block = tuple(
                _containing_block_index(value, block_extent_m)
                for value in world
            )
            spans = tuple(
                _candidate_block_span(
                    value,
                    truncation_m,
                    block_extent_m,
                )
                for value in world
            )
            span_count = math.prod(
                upper - lower + 1 for lower, upper in spans
            )
            if span_count > MAX_PLANNED_BLOCKS:
                raise TsdfError(
                    "one depth sample requires "
                    f"{span_count} candidate TSDF blocks; maximum is "
                    f"{MAX_PLANNED_BLOCKS}. Increase --voxel-size-m or "
                    "clean the depth/pose input."
                )

            surface_blocks.add(surface_block)  # type: ignore[arg-type]
            for block_z in range(spans[2][0], spans[2][1] + 1):
                for block_y in range(spans[1][0], spans[1][1] + 1):
                    for block_x in range(spans[0][0], spans[0][1] + 1):
                        active_blocks.add((block_x, block_y, block_z))
                        if len(active_blocks) > MAX_PLANNED_BLOCKS:
                            raise TsdfError(
                                "TSDF block plan exceeds "
                                f"{MAX_PLANNED_BLOCKS} candidate blocks. "
                                "Increase --voxel-size-m or clean the "
                                "depth/pose input."
                            )
            valid_points += 1
    return valid_points, invalid_samples


def _containing_block_index(coordinate: float, block_extent_m: float) -> int:
    """Return floor ownership with multiply-back boundary correction."""

    ratio = coordinate / block_extent_m
    if not math.isfinite(ratio):
        raise TsdfError("TSDF block coordinate must remain finite")
    index = math.floor(ratio)
    lower = index * block_extent_m
    upper = (index + 1) * block_extent_m
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise TsdfError("TSDF block coordinate must remain finite")
    if lower > coordinate:
        index -= 1
    elif upper <= coordinate:
        index += 1
    _validate_block_index(index)
    return index


def _candidate_block_span(
    coordinate: float,
    truncation_m: float,
    block_extent_m: float,
) -> tuple[int, int]:
    """Return an outward-conservative cover of the half-open surface band."""

    lower_bound = coordinate - truncation_m
    upper_bound = coordinate + truncation_m
    lower_ratio = lower_bound / block_extent_m
    upper_ratio = upper_bound / block_extent_m
    if not all(
        math.isfinite(value)
        for value in (
            lower_bound,
            upper_bound,
            lower_ratio,
            upper_ratio,
        )
    ):
        raise TsdfError("TSDF block activation bounds must remain finite")

    lower_index = math.floor(lower_ratio)
    upper_exclusive = math.ceil(upper_ratio)
    snapped_lower = lower_index * block_extent_m
    snapped_upper = upper_exclusive * block_extent_m
    if not math.isfinite(snapped_lower) or not math.isfinite(snapped_upper):
        raise TsdfError("TSDF block activation bounds must remain finite")
    if snapped_lower > lower_bound:
        lower_index -= 1
    if snapped_upper < upper_bound:
        upper_exclusive += 1

    upper_index = upper_exclusive - 1
    _validate_block_index(lower_index)
    _validate_block_index(upper_index)
    if upper_index < lower_index:
        raise TsdfError("TSDF block activation span is empty")
    return lower_index, upper_index


def _validate_block_index(index: int) -> None:
    if index < MIN_BLOCK_INDEX or index > MAX_BLOCK_INDEX:
        raise TsdfError(
            "TSDF block index exceeds the signed 32-bit planning range"
        )


def _ordered_blocks(
    blocks: set[_BlockIndex],
) -> tuple[_BlockIndex, ...]:
    return tuple(sorted(blocks, key=lambda index: (index[2], index[1], index[0])))


def _block_bounds(
    blocks: tuple[_BlockIndex, ...],
) -> tuple[_BlockIndex, _BlockIndex]:
    if not blocks:
        raise TsdfError("TSDF block plan contains no candidate block")
    minimum = tuple(min(block[axis] for block in blocks) for axis in range(3))
    maximum = tuple(max(block[axis] for block in blocks) for axis in range(3))
    return minimum, maximum  # type: ignore[return-value]


def _build_plan_document(
    session: ScanSession,
    replay_digest: str,
    voxel_size_m: float,
    truncation_m: float,
    block_extent_m: float,
    frame_stride: int,
    total_observations: int,
    selected_observations: int,
    paired_observations: int,
    skipped_missing_depth: int,
    skipped_missing_pose: int,
    valid_depth_points: int,
    invalid_depth_samples: int,
    surface_blocks: tuple[_BlockIndex, ...],
    active_blocks: tuple[_BlockIndex, ...],
    planned_voxel_slots: int,
    minimum: _BlockIndex,
    maximum: _BlockIndex,
) -> dict[str, object]:
    return {
        "schema": TSDF_BLOCK_PLAN_SCHEMA,
        "schema_version": TSDF_BLOCK_PLAN_SCHEMA_VERSION,
        "session_id": session.session_id,
        "replay_digest_sha256": replay_digest,
        "grid": {
            "world_anchor_m": [0.0, 0.0, 0.0],
            "voxel_size_m": _clean_float(voxel_size_m),
            "block_resolution": TSDF_BLOCK_RESOLUTION,
            "block_extent_m": _clean_float(block_extent_m),
            "block_bounds": "lower-inclusive-upper-exclusive",
            "index_order": "x-fastest-then-y-then-z",
            "coordinate_rounding": (
                "floor-with-multiply-back-boundary-correction"
            ),
        },
        "activation": {
            "truncation_m": _clean_float(truncation_m),
            "rule": "outward-conservative-half-open-l-infinity-cover",
            "endpoint_rounding": (
                "floor-ceil-with-multiply-back-outward-correction"
            ),
            "free_space_rule": TSDF_FREE_SPACE_RULE_NOT_PLANNED,
        },
        "planning": {
            "frame_stride": frame_stride,
            "total_observations": total_observations,
            "selected_observations": selected_observations,
            "paired_observations": paired_observations,
            "skipped_missing_depth": skipped_missing_depth,
            "skipped_missing_pose": skipped_missing_pose,
            "valid_depth_points": valid_depth_points,
            "invalid_depth_samples": invalid_depth_samples,
            "surface_blocks": len(surface_blocks),
            "active_blocks": len(active_blocks),
            "halo_blocks": len(active_blocks) - len(surface_blocks),
            "planned_voxel_slots": planned_voxel_slots,
            "min_block_index": list(minimum),
            "max_block_index": list(maximum),
        },
        "surface_blocks": [list(index) for index in surface_blocks],
        "active_blocks": [list(index) for index in active_blocks],
    }


def _clean_float(value: float) -> float:
    return 0.0 if value == 0.0 else value


def _validate_tsdf_block_plan_output(output: str | Path) -> Path:
    output_path = Path(output).resolve()
    if output_path.suffix.lower() != ".sftplan":
        raise TsdfError("output filename must end in .sftplan")
    if output_path.exists():
        raise TsdfError(f"output already exists: {output_path}")
    return output_path


def _write_plan_without_overwrite(output_path: Path, encoded: bytes) -> None:
    temporary_path: Path | None = None
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output_path.stem}-",
            suffix=".sftplan",
            dir=output_path.parent,
        )
        temporary_path = Path(temporary_name)
        with open(descriptor, "wb", closefd=True) as output_file:
            output_file.write(encoded)
        try:
            os.link(temporary_path, output_path)
        except FileExistsError as error:
            raise TsdfError(
                "output appeared while planning; refusing to overwrite: "
                f"{output_path}"
            ) from error
        except OSError as error:
            raise TsdfError(
                "cannot publish TSDF block plan without overwriting: "
                f"{error}"
            ) from error
    except TsdfError:
        raise
    except OSError as error:
        raise TsdfError(f"cannot write TSDF block plan: {error}") from error
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
