"""Score a fused TUM RGB-D reconstruction against held-out depth frames.

The TUM RGB-D benchmark ships a ground-truth *trajectory*, not a ground-truth
mesh, so there is no surface to compare a reconstruction against directly.
What is available instead is cross-validation.

Fuse from one set of frames, then take depth measured by frames the fusion
never saw, back-project it with its own ground-truth pose, and ask what the
reconstruction says at those points. A correct TSDF reads zero on a real
surface, so the interpolated value scaled by the truncation is a signed
surface error in metres.

This measures geometric self-consistency across viewpoints. It does not
measure pose accuracy: held-out frames use the same motion-capture ground
truth, which is taken as exact.

Usage:

    python tools/tum_reconstruction_report.py SESSION PLAN
        [--held-out-offset N] [--pixel-step N]

`SESSION` is an imported `.vgsession`; `PLAN` is a `.sftplan` built from it.
The held-out offset must not be a multiple of the plan's frame stride, or the
evaluation frames would be the ones that were fused.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from spatialforge import (
    allocate_empty_tsdf_blocks,
    build_tsdf_replay_depth_context,
    load_scan_session,
    load_tsdf_block_plan,
    replay_session,
)
from spatialforge.errors import TsdfError
from spatialforge.point_cloud import (
    _read_depth,
    _sample_path,
    _validate_reconstruction_contract,
)
from spatialforge.tsdf_block_storage import TSDF_BLOCK_RESOLUTION
from spatialforge.tsdf_block_vector_fusion import (
    fuse_tsdf_block_from_vector_fields,
)


class BlockVolumeSampler:
    """Read a fused sparse block volume at arbitrary world coordinates."""

    def __init__(self, storage, active_blocks, voxel_size_m: float) -> None:
        blocks = np.array(active_blocks, dtype=np.int64)
        self._minimum = blocks.min(axis=0)
        self._shape = tuple(blocks.max(axis=0) - self._minimum + 1)
        self._rows = np.full(self._shape, -1, dtype=np.int64)
        self._rows[
            blocks[:, 0] - self._minimum[0],
            blocks[:, 1] - self._minimum[1],
            blocks[:, 2] - self._minimum[2],
        ] = np.arange(len(blocks))
        self._sums = storage.tsdf_sums
        self._weights = storage.weights
        self._voxel_size_m = voxel_size_m

    def at_voxels(self, global_index: np.ndarray) -> np.ndarray:
        """Normalized TSDF at integer global voxel indices; NaN if unknown."""

        block_index = np.floor_divide(global_index, TSDF_BLOCK_RESOLUTION)
        local = global_index - block_index * TSDF_BLOCK_RESOLUTION
        offset = block_index - self._minimum
        inside = np.all(
            (offset >= 0) & (offset < np.array(self._shape)),
            axis=1,
        )
        row = np.full(len(global_index), -1, dtype=np.int64)
        row[inside] = self._rows[
            offset[inside, 0],
            offset[inside, 1],
            offset[inside, 2],
        ]
        found = row >= 0
        value = np.full(len(global_index), np.nan)
        if not np.any(found):
            return value
        selected = row[found]
        local_x = local[found, 0]
        local_y = local[found, 1]
        local_z = local[found, 2]
        weight = self._weights[
            selected, local_z, local_y, local_x
        ].astype(np.float64)
        total = self._sums[selected, local_z, local_y, local_x]
        seen = weight > 0
        value[np.flatnonzero(found)[seen]] = total[seen] / weight[seen]
        return value

    def nearest(self, points: np.ndarray) -> np.ndarray:
        return self.at_voxels(
            np.floor(points / self._voxel_size_m).astype(np.int64)
        )

    def trilinear(self, points: np.ndarray) -> np.ndarray:
        """Interpolate; NaN unless all eight neighbours were observed."""

        continuous = points / self._voxel_size_m - 0.5
        base = np.floor(continuous).astype(np.int64)
        fraction = continuous - base
        result = np.zeros(len(points))
        usable = np.ones(len(points), dtype=bool)
        for corner in range(8):
            step = np.array(
                [corner & 1, (corner >> 1) & 1, (corner >> 2) & 1]
            )
            value = self.at_voxels(base + step)
            weight = np.prod(
                np.where(step == 1, fraction, 1.0 - fraction),
                axis=1,
            )
            usable &= np.isfinite(value)
            result += np.where(np.isfinite(value), value * weight, 0.0)
        return np.where(usable, result, np.nan)


def back_project(observation, camera, depth_scale_m, columns, rows):
    """World points for one observation's sampled valid depth pixels."""

    depth_path = _sample_path(observation.session, observation.depth, "depth")
    raw = np.asarray(
        _read_depth(depth_path, camera.width, camera.height),
        dtype=np.float64,
    ).reshape((camera.height, camera.width))
    metres = raw * depth_scale_m
    z = metres[rows, columns]
    usable = np.isfinite(z) & (z > 0.0)
    if not np.any(usable):
        return None, 0
    z = z[usable]
    u = columns[usable].astype(np.float64)
    v = rows[usable].astype(np.float64)
    x = (u - camera.cx) * z / camera.fx
    y = (v - camera.cy) * z / camera.fy

    matrix = [float(value) for value in observation.transform]
    world = np.empty((len(z), 3))
    world[:, 0] = (
        matrix[0] * x + matrix[1] * y + matrix[2] * z + matrix[3]
    )
    world[:, 1] = (
        matrix[4] * x + matrix[5] * y + matrix[6] * z + matrix[7]
    )
    world[:, 2] = (
        matrix[8] * x + matrix[9] * y + matrix[10] * z + matrix[11]
    )
    return world, int(usable.sum())


class HeldOutFrame:
    """One evaluation observation, decoupled from replay's record types."""

    __slots__ = ("session", "depth", "transform", "sequence")

    def __init__(self, session, depth, transform, sequence) -> None:
        self.session = session
        self.depth = depth
        self.transform = transform
        self.sequence = sequence


def describe(label: str, errors: np.ndarray, voxel_size_m: float) -> None:
    absolute = np.abs(errors)
    print(f"\n{label}: n={len(errors)}")
    print(f"  mean signed    {1000 * errors.mean():+7.1f} mm")
    print(f"  median |error| {1000 * np.median(absolute):7.1f} mm")
    print(f"  rms            {1000 * np.sqrt((errors ** 2).mean()):7.1f} mm")
    print(f"  p95 |error|    {1000 * np.percentile(absolute, 95):7.1f} mm")
    inside = np.count_nonzero(absolute <= voxel_size_m)
    print(
        f"  within one voxel ({1000 * voxel_size_m:.0f} mm): "
        f"{100 * inside / len(errors):.1f}%"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--held-out-offset", type=int, default=4)
    parser.add_argument("--pixel-step", type=int, default=4)
    arguments = parser.parse_args(argv)

    session = load_scan_session(arguments.session)
    plan = load_tsdf_block_plan(arguments.plan)
    if arguments.held_out_offset % plan.frame_stride == 0:
        raise SystemExit(
            f"--held-out-offset {arguments.held_out_offset} is a multiple of "
            f"the plan's frame_stride={plan.frame_stride}; the evaluation "
            "frames would be the fused ones"
        )
    camera, depth_scale_m = _validate_reconstruction_contract(session)
    print(
        f"plan: blocks={len(plan.active_blocks)} "
        f"surface={len(plan.surface_blocks)} "
        f"voxel={plan.voxel_size_m} truncation={plan.truncation_m} "
        f"stride={plan.frame_stride} fused_frames={plan.selected_observations}"
    )

    storage = allocate_empty_tsdf_blocks(plan, session)
    started = time.perf_counter()
    context = build_tsdf_replay_depth_context(plan, session)
    print(
        f"context: {time.perf_counter() - started:.1f}s "
        f"retained_depth={context.depth_payload_bytes / 1e6:.0f} MB"
    )

    started = time.perf_counter()
    applied = 0
    for block_index_xyz in plan.active_blocks:
        applied += fuse_tsdf_block_from_vector_fields(
            storage,
            block_index_xyz,
            context,
        ).applied_count
    seconds = time.perf_counter() - started
    evaluated = (
        len(plan.active_blocks)
        * TSDF_BLOCK_RESOLUTION**3
        * plan.selected_observations
    )
    print(
        f"fuse: {seconds:.1f}s applied={applied} evaluated={evaluated} "
        f"rate={evaluated / seconds:,.0f} voxel-observations/s"
    )
    observed = storage.nonzero_weight_count
    print(
        f"observed voxels: {observed} of {storage.voxel_slots} "
        f"({100 * observed / storage.voxel_slots:.1f}% of planned slots)"
    )

    sampler = BlockVolumeSampler(
        storage,
        plan.active_blocks,
        plan.voxel_size_m,
    )
    replay = replay_session(session)
    held_out = [
        HeldOutFrame(
            session,
            observation.depth.data,
            tuple(observation.pose.data["T_world_camera"]),
            observation.sequence,
        )
        for observation in replay.observations
        if observation.sequence % plan.frame_stride
        == arguments.held_out_offset
        and observation.depth is not None
        and observation.pose is not None
    ]
    if not held_out:
        raise SystemExit("no held-out frame has both depth and pose")
    print(f"held-out frames: {len(held_out)} (never fused)")

    columns, rows = np.meshgrid(
        np.arange(0, camera.width, arguments.pixel_step),
        np.arange(0, camera.height, arguments.pixel_step),
        indexing="xy",
    )
    columns = columns.ravel()
    rows = rows.ravel()

    nearest_chunks: list[np.ndarray] = []
    linear_chunks: list[np.ndarray] = []
    sampled = 0
    covered = 0
    for frame in held_out:
        world, count = back_project(
            frame,
            camera,
            depth_scale_m,
            columns,
            rows,
        )
        if world is None:
            continue
        sampled += count
        nearest = sampler.nearest(world)
        linear = sampler.trilinear(world)
        covered += int(np.count_nonzero(np.isfinite(nearest)))
        nearest_chunks.append(
            nearest[np.isfinite(nearest)] * plan.truncation_m
        )
        linear_chunks.append(
            linear[np.isfinite(linear)] * plan.truncation_m
        )

    print(
        f"held-out depth samples: {sampled} "
        f"(every {arguments.pixel_step} pixels); "
        f"inside observed voxels: {covered} "
        f"({100 * covered / max(sampled, 1):.1f}%)"
    )
    describe(
        "nearest voxel",
        np.concatenate(nearest_chunks),
        plan.voxel_size_m,
    )
    describe(
        "trilinear",
        np.concatenate(linear_chunks),
        plan.voxel_size_m,
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except TsdfError as error:
        print(f"TUM RECONSTRUCTION REPORT FAILED: {error}", file=sys.stderr)
        sys.exit(2)
