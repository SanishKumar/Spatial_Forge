"""Free-space plan expansion at the scale of a real scan.

A surface plan holds the blocks within a truncation of some measured depth.
Everything between the camera and that band is space the scan looked
through and found empty, and none of it is in the plan. Expansion adds the
blocks of it: any block holding a voxel some frame observed.

The reference path gets there in stages that are each easy to believe.
Cover every pixel's wedge with blocks, resolve every voxel of that cover
against every frame, approve the blocks with an observed voxel. Each stage
keeps an outcome per pixel or per voxel, and stops at 262,144 of them. One
640x480 frame is more than that.

This path asks the question directly. "Observed" has one meaning here: the
voxel would receive a contribution from that frame if its block were fused.
So take a box of candidate blocks that is known to contain every such
voxel, run the fusion evaluator over it one frame at a time, and note which
blocks received anything. No outcome is kept, only a bit per block, and the
verdict is fusion's own because it is fusion's own code that gives it.

The box has to be shown to contain every observed voxel, or a block could
be missed without a sign. A voxel observed through pixel ``p`` at depth
``z`` satisfies ``z <= m + truncation``, where ``m`` is the depth measured
at ``p``. In the camera's frame its position is ``(z / m) s + z e``, with
``s`` the surface sample the planner back-projected for ``p`` and ``e`` a
sideways offset of at most half a pixel. So it lies within

    truncation * |r| + (m + truncation) * |half a pixel|

of the segment from the camera centre to ``s``, where ``r`` is the longest
ray of the image scaled to unit depth. ``s`` is inside a planned block, and
``m`` is no more than the distance from the camera to it. The box spanning
the planned blocks and the camera centres, grown by that bound with ``m``
replaced by the box's own diagonal, therefore contains every observed
voxel. The test that a larger box finds nothing more is a check of this
argument, not a substitute for it.

Most of such a box is behind something. A block hidden by a wall is in
the image and in front of the camera, so the two whole-block verdicts
fusion uses do not settle it, and it would be evaluated voxel by voxel on
every frame only to find, every time, that nothing lands. A third verdict
settles it from the depth image. If the nearest corner of a block is
farther than a truncation behind the largest depth measured anywhere in
the rectangle of pixels the block projects into, no voxel of it can be
within a truncation of what its own pixel measured. The largest depth in
a rectangle is read from a small table built once per frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .errors import PointCloudError, SessionReplayError, TsdfError
from .model import ScanSession
from .point_cloud import _validate_reconstruction_contract
from .replay import replay_session
from .tsdf_block_contributions import _evaluate_ready_voxels
from .tsdf_block_plan import (
    MAX_BLOCK_INDEX,
    MAX_PLANNED_BLOCKS,
    MIN_BLOCK_INDEX,
    TSDF_BLOCK_RESOLUTION,
    TSDF_FREE_SPACE_RULE_OBSERVED,
    _ordered_blocks,
)
from .tsdf_block_plan_loader import TsdfBlockPlan
from .tsdf_block_storage import TSDF_BLOCK_VOXELS
from .tsdf_plan_expansion import TsdfPlanExpansionProposal
from .tsdf_replay_depth_context import (
    TsdfReplayDepthStatus,
    _classify_observation,
    _is_rigid_transform,
    _prepared_transform,
    _validate_context_camera,
    _validate_context_plan,
    _validate_plan_associations,
)
from .tsdf_stream_fusion import (
    BLOCK_EVALUATE,
    STREAM_FUSION_CHUNK_BLOCKS,
    _chunk_voxel_centres_world_m,
    _classify_blocks,
    _decode_metric_depth,
)
from .tsdf_voxel_contribution import _validate_contribution_plan

# Whether blocks hidden behind a frame's surfaces are settled without
# evaluating their voxels. The proposal does not depend on it.
STREAM_EXPANSION_SKIPS_HIDDEN_BLOCKS = True
# Pixels along the side of one cell of the largest-depth table.
_DEPTH_TILE_PIXELS = 8
# Slack on a block's nearest depth, relative to how far away it is, and
# on the pixel rectangle it projects into. Both make the verdict rarer.
_HIDDEN_RELATIVE_DEPTH_SLACK = 1e-9
_HIDDEN_PIXEL_SLACK = 1

# A proposal lists every candidate block as approved or rejected, which
# is what limits the box it can be made from.
MAX_TSDF_STREAM_EXPANSION_CANDIDATE_BLOCKS = 500_000
# A survey lists only the blocks it approved. The box itself is one bit
# and one index triple per block, and a real scan's can be large: a few
# far depth readings stretch it well past the room.
MAX_TSDF_STREAM_EXPANSION_BOX_BLOCKS = 8_000_000
# Blocks given a whole-block verdict per pass over the box.
_BOX_CLASSIFICATION_CHUNK_BLOCKS = 65_536


def candidate_box_margin_m(
    camera,
    *,
    truncation_m: float,
    diagonal_m: float,
) -> float:
    """How far an observed voxel can lie outside the cameras-and-plan box.

    See the module docstring for the argument. ``diagonal_m`` stands in for
    the largest depth any frame can have measured.
    """

    across = max(abs(camera.cx), abs(camera.width - 1 - camera.cx)) / abs(
        camera.fx
    )
    down = max(abs(camera.cy), abs(camera.height - 1 - camera.cy)) / abs(
        camera.fy
    )
    longest_ray = math.sqrt(1.0 + across * across + down * down)
    half_pixel = 0.5 * math.hypot(1.0 / camera.fx, 1.0 / camera.fy)
    return (
        truncation_m * longest_ray
        + (diagonal_m + truncation_m) * half_pixel
    )


@dataclass(frozen=True, slots=True)
class TsdfStreamExpansionSurvey:
    """What an expansion looked at, and the proposal it came to.

    The proposal's domain is the blocks that were approved and no more.
    The blocks that were looked at and rejected are counted here and not
    listed, so the box may be far larger than a list of it could be.
    """

    box_low_block: tuple[int, int, int]
    box_shape_blocks: tuple[int, int, int]
    candidate_block_count: int
    observed_block_count: int
    frames_read: int
    proposal: TsdfPlanExpansionProposal

    @property
    def rejected_block_count(self) -> int:
        return self.candidate_block_count - self.observed_block_count


def propose_tsdf_plan_expansion_streaming(
    plan: TsdfBlockPlan,
    session: ScanSession,
    *,
    extra_margin_blocks: int = 0,
) -> TsdfPlanExpansionProposal:
    """Approve every block of the candidate box that some frame observed.

    Returns the same proposal the reference path does, over a different
    domain: the candidate box in place of the surveyed pixel cover. The
    blocks it adds to the plan are the same blocks.

    Every block of the box is listed, approved or rejected, so the box
    may hold no more than a list of it reasonably can. For a larger one
    use ``survey_tsdf_plan_expansion_streaming``.

    ``extra_margin_blocks`` grows the box beyond what the bound requires.
    It can only cost time; it exists so the bound can be tested.
    """

    candidates, observed, _, _, _ = _observe_box(
        plan,
        session,
        extra_margin_blocks,
        MAX_TSDF_STREAM_EXPANSION_CANDIDATE_BLOCKS,
    )
    approved = {(int(x), int(y), int(z)) for x, y, z in candidates[observed]}
    rejected = {
        (int(x), int(y), int(z)) for x, y, z in candidates[~observed]
    }
    return TsdfPlanExpansionProposal(
        source_plan_digest_sha256=plan.artifact_digest_sha256,
        replay_digest_sha256=plan.replay_digest_sha256,
        frame_stride=plan.frame_stride,
        total_observations=plan.total_observations,
        block_resolution=plan.block_resolution,
        source_plan_block_indices=plan.active_blocks,
        domain_block_indices=_ordered_blocks(approved | rejected),
        approved_block_indices=_ordered_blocks(approved),
        rejected_block_indices=_ordered_blocks(rejected),
        expanded_block_indices=_ordered_blocks(
            set(plan.active_blocks) | approved
        ),
        free_space_rule=TSDF_FREE_SPACE_RULE_OBSERVED,
    )


def survey_tsdf_plan_expansion_streaming(
    plan: TsdfBlockPlan,
    session: ScanSession,
    *,
    extra_margin_blocks: int = 0,
) -> TsdfStreamExpansionSurvey:
    """The same expansion, without listing the blocks it rejected.

    The blocks approved, and so the expanded plan, are those
    ``propose_tsdf_plan_expansion_streaming`` arrives at. The box can be
    sixteen times larger.
    """

    candidates, observed, low, shape, frames_read = _observe_box(
        plan,
        session,
        extra_margin_blocks,
        MAX_TSDF_STREAM_EXPANSION_BOX_BLOCKS,
    )
    approved = {(int(x), int(y), int(z)) for x, y, z in candidates[observed]}
    ordered = _ordered_blocks(approved)
    return TsdfStreamExpansionSurvey(
        box_low_block=(int(low[0]), int(low[1]), int(low[2])),
        box_shape_blocks=(shape[0], shape[1], shape[2]),
        candidate_block_count=len(candidates),
        observed_block_count=len(ordered),
        frames_read=frames_read,
        proposal=TsdfPlanExpansionProposal(
            source_plan_digest_sha256=plan.artifact_digest_sha256,
            replay_digest_sha256=plan.replay_digest_sha256,
            frame_stride=plan.frame_stride,
            total_observations=plan.total_observations,
            block_resolution=plan.block_resolution,
            source_plan_block_indices=plan.active_blocks,
            domain_block_indices=ordered,
            approved_block_indices=ordered,
            rejected_block_indices=(),
            expanded_block_indices=_ordered_blocks(
                set(plan.active_blocks) | approved
            ),
            free_space_rule=TSDF_FREE_SPACE_RULE_OBSERVED,
        ),
    )


def _observe_box(
    plan: TsdfBlockPlan,
    session: ScanSession,
    extra_margin_blocks: int,
    candidate_limit: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], int]:
    """Which blocks of the candidate box hold an observed voxel.

    Returns the box's blocks in canonical order, one flag for each, the
    box's lowest block and extent, and how many frames were read.
    """

    if not isinstance(plan, TsdfBlockPlan):
        raise TsdfError(
            "TSDF stream expansion requires a loaded TsdfBlockPlan"
        )
    if not isinstance(session, ScanSession):
        raise TsdfError("TSDF stream expansion requires a loaded ScanSession")
    if (
        isinstance(extra_margin_blocks, bool)
        or not isinstance(extra_margin_blocks, int)
        or extra_margin_blocks < 0
    ):
        raise TsdfError(
            "TSDF stream expansion extra margin must be a nonnegative "
            "integer"
        )
    _validate_contribution_plan(plan)
    _validate_context_plan(plan)
    if plan.session_id != session.session_id:
        raise TsdfError(
            "TSDF block plan session_id does not match the loaded session"
        )
    try:
        camera, depth_scale_m = _validate_reconstruction_contract(session)
    except PointCloudError as error:
        raise TsdfError(str(error)) from error
    _validate_context_camera(camera)

    starting_replay = replay_session(session)
    if starting_replay.digest_sha256 != plan.replay_digest_sha256:
        raise TsdfError(
            "TSDF block plan replay digest does not match current session "
            "inputs; regenerate the plan"
        )
    if len(starting_replay.observations) != plan.total_observations:
        raise TsdfError(
            "TSDF block plan total observations do not match current replay"
        )
    selected = tuple(
        starting_replay.observations[sequence]
        for sequence in range(0, plan.total_observations, plan.frame_stride)
    )
    if len(selected) != plan.selected_observations:
        raise TsdfError(
            "TSDF block plan selected observations do not match its stride"
        )
    _validate_plan_associations(plan, selected, camera)

    try:
        ready = []
        for observation in selected:
            status = _classify_observation(observation)
            if status is not TsdfReplayDepthStatus.READY:
                continue
            transform = _prepared_transform(observation, status)
            if transform is None or not _is_rigid_transform(transform):
                raise TsdfError(
                    "TSDF stream expansion requires a rigid T_world_camera "
                    f"for observation {observation.sequence}"
                )
            ready.append((observation, transform))

        candidates, low, shape = _candidate_blocks(
            plan,
            camera,
            [
                (transform[3], transform[7], transform[11])
                for _, transform in ready
            ],
            extra_margin_blocks,
            candidate_limit,
        )
        observed = np.zeros(len(candidates), dtype=bool)
        frames_read = 0
        # Where each planned block sits in the box, so that blocks the
        # plan already has are not counted as additions.
        planned = np.asarray(plan.active_blocks, dtype=np.int64) - low
        in_plan = np.zeros(len(candidates), dtype=bool)
        in_plan[
            (planned[:, 2] * shape[1] + planned[:, 1]) * shape[0]
            + planned[:, 0]
        ] = True
        for position, (observation, transform) in enumerate(ready):
            depth_m = _decode_metric_depth(
                session,
                observation,
                camera.width,
                camera.height,
                depth_scale_m,
            )
            # A block that already holds an observed voxel has its answer.
            pending = np.flatnonzero(~observed)
            if not len(pending):
                break
            frames_read += 1
            table = (
                _largest_depth_table(depth_m)
                if STREAM_EXPANSION_SKIPS_HIDDEN_BLOCKS
                else None
            )
            # The whole-block verdicts are given a run of the box at a
            # time: each needs a dozen arrays of eight corners a block.
            kept = []
            for start in range(
                0, len(pending), _BOX_CLASSIFICATION_CHUNK_BLOCKS
            ):
                part = pending[start:start + _BOX_CLASSIFICATION_CHUNK_BLOCKS]
                part = part[
                    _classify_blocks(
                        camera, transform, candidates[part], plan.voxel_size_m
                    )
                    == BLOCK_EVALUATE
                ]
                if table is not None and len(part):
                    part = part[
                        ~_hidden_blocks(
                            camera,
                            transform,
                            candidates[part],
                            plan.voxel_size_m,
                            plan.truncation_m,
                            table,
                        )
                    ]
                kept.append(part)
            visible = np.concatenate(kept)
            for first in range(0, len(visible), STREAM_FUSION_CHUNK_BLOCKS):
                rows = visible[first:first + STREAM_FUSION_CHUNK_BLOCKS]
                _, _, weight_deltas = _evaluate_ready_voxels(
                    camera,
                    transform,
                    depth_m,
                    plan.truncation_m,
                    _chunk_voxel_centres_world_m(
                        candidates[rows], plan.voxel_size_m
                    ),
                )
                observed[rows] |= weight_deltas.reshape(
                    (len(rows), TSDF_BLOCK_VOXELS)
                ).any(axis=1)
            # The count only grows, so once it is over there is no point
            # in reading the rest of the scan to find out by how much.
            if (
                len(plan.active_blocks)
                + int(np.count_nonzero(observed & ~in_plan))
                > MAX_PLANNED_BLOCKS
            ):
                raise TsdfError(
                    "TSDF plan expansion holds more than "
                    f"{MAX_PLANNED_BLOCKS} blocks after {position + 1} of "
                    f"{len(ready)} frames; that is the maximum. Use a "
                    "coarser voxel size"
                )

        if replay_session(session).digest_sha256 != (
            starting_replay.digest_sha256
        ):
            raise TsdfError(
                "session inputs changed during TSDF stream expansion; "
                "rerun the command"
            )

        return candidates, observed, low, shape, frames_read
    except (TsdfError, SessionReplayError):
        raise
    except PointCloudError as error:
        raise TsdfError(str(error)) from error
    except Exception as error:
        raise TsdfError(
            f"cannot complete TSDF stream expansion: {error}"
        ) from error


def _largest_depth_table(depth_m: np.ndarray) -> list[list[np.ndarray]]:
    """Tables from which the largest valid depth in any rectangle is read.

    The image is cut into tiles and each tile keeps its largest valid
    depth, or zero if it has none. ``table[i][j]`` then holds, for every
    position, the largest value in a window of ``2**i`` by ``2**j`` tiles.
    Any rectangle of tiles is covered by four such windows, so its largest
    depth is the largest of four entries.
    """

    tile = _DEPTH_TILE_PIXELS
    height, width = depth_m.shape
    rows = -(-height // tile)
    columns = -(-width // tile)
    padded = np.zeros((rows * tile, columns * tile))
    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    padded[:height, :width] = np.where(valid, depth_m, 0.0)
    base = padded.reshape((rows, tile, columns, tile)).max(axis=(1, 3))

    table: list[list[np.ndarray]] = [[base]]
    span = 1
    while 2 * span <= columns:
        previous = table[0][-1]
        table[0].append(
            np.maximum(previous[:, :-span], previous[:, span:])
        )
        span *= 2
    span = 1
    while 2 * span <= rows:
        table.append(
            [
                np.maximum(previous[:-span, :], previous[span:, :])
                for previous in table[-1]
            ]
        )
        span *= 2
    return table


def _largest_depth_in(
    table: list[list[np.ndarray]],
    first_row: np.ndarray,
    last_row: np.ndarray,
    first_column: np.ndarray,
    last_column: np.ndarray,
) -> np.ndarray:
    """The largest tile value in each inclusive rectangle of tiles."""

    result = np.zeros(len(first_row))

    def level(extent: np.ndarray, levels: int) -> np.ndarray:
        # The largest power of two not above the extent, counted in whole
        # numbers. A logarithm would have to be trusted to be exact at
        # exactly the extents where it matters.
        found = np.zeros(len(extent), dtype=np.int64)
        for power in range(1, levels):
            found[extent >= (1 << power)] = power
        return found

    row_level = level(last_row - first_row + 1, len(table))
    column_level = level(last_column - first_column + 1, len(table[0]))
    for i in range(len(table)):
        for j in range(len(table[0])):
            chosen = np.flatnonzero((row_level == i) & (column_level == j))
            if not len(chosen):
                continue
            window = table[i][j]
            top = first_row[chosen]
            bottom = last_row[chosen] - (1 << i) + 1
            left = first_column[chosen]
            right = last_column[chosen] - (1 << j) + 1
            result[chosen] = np.maximum(
                np.maximum(window[top, left], window[top, right]),
                np.maximum(window[bottom, left], window[bottom, right]),
            )
    return result


def _hidden_blocks(
    camera,
    transform: tuple[float, ...],
    blocks: np.ndarray,
    voxel_size_m: float,
    truncation_m: float,
    table: list[list[np.ndarray]],
) -> np.ndarray:
    """Blocks no voxel of which one frame can observe, by its depth image.

    ``True`` promises that the evaluator would give no voxel of the block
    a contribution from this frame. ``False`` promises nothing.

    A block's voxel centres lie in the box its eight extreme centres span.
    With all eight in front of the camera, every centre is at least as
    deep as the nearest corner, and projects inside the rectangle the
    corners' projections span. So if the nearest corner is more than a
    truncation behind the largest depth measured in that rectangle, every
    voxel is more than a truncation behind what its own pixel measured.
    """

    matrix = [float(component) for component in transform]
    base = blocks * TSDF_BLOCK_RESOLUTION
    corners = np.empty((len(blocks), 8, 3))
    low = (base.astype(np.float64) + 0.5) * voxel_size_m
    high = (
        (base + (TSDF_BLOCK_RESOLUTION - 1)).astype(np.float64) + 0.5
    ) * voxel_size_m
    for corner in range(8):
        for axis in range(3):
            corners[:, corner, axis] = (
                high if (corner >> axis) & 1 else low
            )[:, axis]
    delta_x = corners[:, :, 0] - matrix[3]
    delta_y = corners[:, :, 1] - matrix[7]
    delta_z = corners[:, :, 2] - matrix[11]
    camera_x = matrix[0] * delta_x + matrix[4] * delta_y + matrix[8] * delta_z
    camera_y = matrix[1] * delta_x + matrix[5] * delta_y + matrix[9] * delta_z
    camera_z = matrix[2] * delta_x + matrix[6] * delta_y + matrix[10] * delta_z
    reach = np.maximum(
        np.maximum(np.abs(delta_x), np.abs(delta_y)), np.abs(delta_z)
    ).max(axis=1)
    slack = _HIDDEN_RELATIVE_DEPTH_SLACK * (1.0 + reach)
    nearest = camera_z.min(axis=1)
    # Only a block wholly in front of the camera has a rectangle to speak
    # of; anything else is left to the evaluator.
    settled = np.isfinite(reach) & (nearest > slack)
    safe_z = np.where(settled[:, None], camera_z, 1.0)
    with np.errstate(over="ignore", invalid="ignore"):
        u = camera.fx * camera_x / safe_z + camera.cx
        v = camera.fy * camera_y / safe_z + camera.cy
        first_column = np.floor(u.min(axis=1) + 0.5) - _HIDDEN_PIXEL_SLACK
        last_column = np.floor(u.max(axis=1) + 0.5) + _HIDDEN_PIXEL_SLACK
        first_row = np.floor(v.min(axis=1) + 0.5) - _HIDDEN_PIXEL_SLACK
        last_row = np.floor(v.max(axis=1) + 0.5) + _HIDDEN_PIXEL_SLACK
    settled &= (
        np.isfinite(first_column)
        & np.isfinite(last_column)
        & np.isfinite(first_row)
        & np.isfinite(last_row)
    )
    # Pixels outside the image measure nothing, so the rectangle can be
    # cut to the image; a block that misses it entirely is not this
    # verdict's to give.
    tile = _DEPTH_TILE_PIXELS
    tile_rows, tile_columns = table[0][0].shape
    settled &= (
        (last_column >= 0)
        & (first_column <= camera.width - 1)
        & (last_row >= 0)
        & (first_row <= camera.height - 1)
    )
    index = np.flatnonzero(settled)
    hidden = np.zeros(len(blocks), dtype=bool)
    if not len(index):
        return hidden

    def tiles(values: np.ndarray, limit: int, count: int) -> np.ndarray:
        clipped = np.clip(values[index], 0, limit - 1).astype(np.int64)
        return np.minimum(clipped // tile, count - 1)

    largest = _largest_depth_in(
        table,
        tiles(first_row, camera.height, tile_rows),
        tiles(last_row, camera.height, tile_rows),
        tiles(first_column, camera.width, tile_columns),
        tiles(last_column, camera.width, tile_columns),
    )
    hidden[index] = (nearest[index] - slack[index]) > (largest + truncation_m)
    return hidden


def _candidate_blocks(
    plan: TsdfBlockPlan,
    camera,
    camera_centres_m: list[tuple[float, float, float]],
    extra_margin_blocks: int,
    candidate_limit: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Every block of the box that must contain all observed voxels.

    Returned in canonical order, with the box's lowest block and its
    extent in blocks on each axis.
    """

    block_extent_m = plan.voxel_size_m * TSDF_BLOCK_RESOLUTION
    planned = np.asarray(plan.active_blocks, dtype=np.int64)
    low = planned.min(axis=0)
    high = planned.max(axis=0)
    if camera_centres_m:
        centres = np.asarray(camera_centres_m, dtype=np.float64)
        if not np.all(np.isfinite(centres)):
            raise TsdfError(
                "TSDF stream expansion requires finite camera positions"
            )
        camera_blocks = np.floor(centres / block_extent_m)
        if np.any(np.abs(camera_blocks) > MAX_BLOCK_INDEX):
            raise TsdfError(
                "TSDF stream expansion camera position is outside the "
                "planning range"
            )
        camera_blocks = camera_blocks.astype(np.int64)
        low = np.minimum(low, camera_blocks.min(axis=0))
        high = np.maximum(high, camera_blocks.max(axis=0))

    diagonal_m = float(np.linalg.norm((high - low + 1) * block_extent_m))
    margin_m = candidate_box_margin_m(
        camera,
        truncation_m=plan.truncation_m,
        diagonal_m=diagonal_m,
    )
    margin = int(math.ceil(margin_m / block_extent_m)) + extra_margin_blocks
    low = low - margin
    high = high + margin
    if np.any(low < MIN_BLOCK_INDEX) or np.any(high > MAX_BLOCK_INDEX):
        raise TsdfError(
            "TSDF stream expansion candidate box is outside the planning "
            "range"
        )
    shape = (high - low + 1).tolist()
    count = shape[0] * shape[1] * shape[2]
    if count > candidate_limit:
        raise TsdfError(
            "TSDF stream expansion would consider "
            f"{count} candidate blocks ({shape[0]} x {shape[1]} x "
            f"{shape[2]}); the maximum is {candidate_limit}. Use a "
            "coarser voxel size"
        )
    # Canonical order: x fastest, then y, then z.
    z, y, x = np.meshgrid(
        np.arange(low[2], high[2] + 1),
        np.arange(low[1], high[1] + 1),
        np.arange(low[0], high[0] + 1),
        indexing="ij",
    )
    return np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1), low, shape
