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
"""

from __future__ import annotations

import math

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

# The candidate box is held as one bit and one index triple per block, and
# the proposal lists every one of them as approved or rejected.
MAX_TSDF_STREAM_EXPANSION_CANDIDATE_BLOCKS = 500_000


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

    ``extra_margin_blocks`` grows the box beyond what the bound requires.
    It can only cost time; it exists so the bound can be tested.
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
        )
        observed = np.zeros(len(candidates), dtype=bool)
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
            verdicts = _classify_blocks(
                camera, transform, candidates[pending], plan.voxel_size_m
            )
            visible = pending[verdicts == BLOCK_EVALUATE]
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

        approved = {
            (int(x), int(y), int(z)) for x, y, z in candidates[observed]
        }
        rejected = {
            (int(x), int(y), int(z)) for x, y, z in candidates[~observed]
        }
        expanded = _ordered_blocks(set(plan.active_blocks) | approved)
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
            expanded_block_indices=expanded,
            free_space_rule=TSDF_FREE_SPACE_RULE_OBSERVED,
        )
    except (TsdfError, SessionReplayError):
        raise
    except PointCloudError as error:
        raise TsdfError(str(error)) from error
    except Exception as error:
        raise TsdfError(
            f"cannot complete TSDF stream expansion: {error}"
        ) from error


def _candidate_blocks(
    plan: TsdfBlockPlan,
    camera,
    camera_centres_m: list[tuple[float, float, float]],
    extra_margin_blocks: int,
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
    if count > MAX_TSDF_STREAM_EXPANSION_CANDIDATE_BLOCKS:
        raise TsdfError(
            "TSDF stream expansion would consider "
            f"{count} candidate blocks ({shape[0]} x {shape[1]} x "
            f"{shape[2]}); the maximum is "
            f"{MAX_TSDF_STREAM_EXPANSION_CANDIDATE_BLOCKS}. Use a coarser "
            "voxel size"
        )
    # Canonical order: x fastest, then y, then z.
    z, y, x = np.meshgrid(
        np.arange(low[2], high[2] + 1),
        np.arange(low[1], high[1] + 1),
        np.arange(low[0], high[0] + 1),
        indexing="ij",
    )
    return np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1), low, shape
