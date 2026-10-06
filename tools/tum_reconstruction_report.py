"""Score a persisted reconstruction against held-out depth frames.

The TUM RGB-D benchmark ships a ground-truth *trajectory*, not a ground-truth
mesh, so there is no surface to compare a reconstruction against directly.
What is available instead is cross-validation.

A volume is fused from every ``frame_stride``-th frame. This takes depth
measured by frames the fusion never saw, back-projects it with its own
ground-truth pose, and asks what the volume says at those points. A correct
TSDF reads zero on a real surface, so the interpolated value scaled by the
truncation is a signed residual in metres.

This measures geometric self-consistency across viewpoints. It is not
distance to a surveyed surface, and it does not measure pose accuracy:
held-out frames use the same motion-capture ground truth, taken as exact.

The volume is read from its ``.sftvol`` file rather than fused here, so the
thing being scored is the same artifact that gets meshed and rendered, and
the manifest can tie them together by digest: scan, plan, volume, mesh.

Usage:

    python tools/tum_reconstruction_report.py SESSION VOLUME.sftvol
        [--mesh MESH.ply] [--held-out-offset N] [--pixel-step N]
        [--manifest-out RESULT.json] [--allow-dirty]

The held-out offset must not be a multiple of the volume's frame stride, or
the evaluation frames would be the ones that were fused. It defaults to half
the stride.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):  # run as a script rather than imported
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools._output import (  # noqa: E402
    non_negative_int,
    positive_int,
    publishing,
    reserve_output,
    source_state,
)

from spatialforge import (  # noqa: E402
    load_scan_session,
    load_tsdf_block_volume,
    replay_session,
)
from spatialforge.errors import PointCloudError, TsdfError  # noqa: E402
from spatialforge.point_cloud import (  # noqa: E402
    _read_depth_array,
    _sample_path,
    _validate_reconstruction_contract,
)
from spatialforge.tsdf_block_storage import (  # noqa: E402
    TSDF_BLOCK_RESOLUTION,
)

RESULT_MANIFEST_SCHEMA = "spatialforge.result-manifest"
RESULT_MANIFEST_SCHEMA_VERSION = "0.2.0"
_VOLUME_COMMENT = "comment spatialforge_source_volume_sha256 "


class BlockVolumeSampler:
    """Read a fused sparse block volume at arbitrary world coordinates."""

    def __init__(self, volume) -> None:
        blocks = np.array(volume.block_indices, dtype=np.int64)
        self._minimum = blocks.min(axis=0)
        self._shape = tuple(blocks.max(axis=0) - self._minimum + 1)
        self._rows = np.full(self._shape, -1, dtype=np.int64)
        self._rows[
            blocks[:, 0] - self._minimum[0],
            blocks[:, 1] - self._minimum[1],
            blocks[:, 2] - self._minimum[2],
        ] = np.arange(len(blocks))
        self._sums = volume.tsdf_sums
        self._weights = volume.weights
        self._voxel_size_m = volume.voxel_size_m

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


def back_project(session, observation, camera, depth_scale_m, columns, rows):
    """World points for one observation's sampled valid depth pixels."""

    metres = (
        _read_depth_array(
            _sample_path(session, observation.depth.data, "depth"),
            camera.width,
            camera.height,
        )
        .astype(np.float64)
        .reshape((camera.height, camera.width))
        * depth_scale_m
    )
    z = metres[rows, columns]
    usable = np.isfinite(z) & (z > 0.0)
    if not np.any(usable):
        return None, 0
    z = z[usable]
    u = columns[usable].astype(np.float64)
    v = rows[usable].astype(np.float64)
    x = (u - camera.cx) * z / camera.fx
    y = (v - camera.cy) * z / camera.fy

    matrix = [
        float(value) for value in observation.pose.data["T_world_camera"]
    ]
    world = np.empty((len(z), 3))
    world[:, 0] = matrix[0] * x + matrix[1] * y + matrix[2] * z + matrix[3]
    world[:, 1] = matrix[4] * x + matrix[5] * y + matrix[6] * z + matrix[7]
    world[:, 2] = matrix[8] * x + matrix[9] * y + matrix[10] * z + matrix[11]
    return world, int(usable.sum())


def describe_mesh(path: Path, volume_digest: str) -> dict[str, object]:
    """Check that a mesh was extracted from this volume, and summarise it.

    The mesher records the digest of its source volume in the PLY header.
    A mesh that names a different volume is not a picture of the thing being
    measured, and listing it beside these numbers would say that it is.
    """

    encoded = path.read_bytes()
    end = encoded.find(b"end_header\n")
    if not encoded.startswith(b"ply\n") or end < 0:
        raise SystemExit(f"{path}: not a PLY file")
    header = encoded[:end].decode("ascii", errors="replace").splitlines()
    recorded = [
        line[len(_VOLUME_COMMENT):]
        for line in header
        if line.startswith(_VOLUME_COMMENT)
    ]
    if recorded != [volume_digest]:
        raise SystemExit(
            f"{path} was not extracted from this volume: it records source "
            f"volume {recorded[0] if recorded else 'nothing'}, and the "
            f"volume being scored is {volume_digest}"
        )

    def count(element: str) -> int:
        return next(
            int(line.split()[2])
            for line in header
            if line.startswith(f"element {element} ")
        )

    def recorded_integer(name: str) -> int:
        prefix = f"comment spatialforge_{name} "
        return next(
            int(line[len(prefix):])
            for line in header
            if line.startswith(prefix)
        )

    return {
        "path": path.as_posix(),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "source_volume_sha256": volume_digest,
        "vertices": count("vertex"),
        "triangles": count("face"),
        "minimum_weight": recorded_integer("minimum_weight"),
        "minimum_component_triangles": recorded_integer(
            "minimum_component_triangles"
        ),
    }


def summarise(errors: np.ndarray, voxel_size_m: float) -> dict[str, float]:
    absolute = np.abs(errors)
    return {
        "samples": int(len(errors)),
        "mean_signed_mm": float(1000 * errors.mean()),
        "median_absolute_mm": float(1000 * np.median(absolute)),
        "rms_mm": float(1000 * np.sqrt((errors**2).mean())),
        "p95_absolute_mm": float(1000 * np.percentile(absolute, 95)),
        "within_one_voxel_fraction": float(
            np.count_nonzero(absolute <= voxel_size_m) / len(errors)
        ),
    }


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
    parser.add_argument("volume", type=Path)
    parser.add_argument("--mesh", type=Path, default=None)
    parser.add_argument(
        "--held-out-offset",
        type=non_negative_int,
        default=None,
    )
    parser.add_argument("--pixel-step", type=positive_int, default=4)
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=None,
        help="Write a machine-readable result manifest to this JSON path.",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help=(
            "Write a manifest even though the working tree has uncommitted "
            "changes. The manifest records that it was not clean."
        ),
    )
    arguments = parser.parse_args(argv)

    manifest_path: Path | None = None
    source_commit: str | None = None
    worktree_clean: bool | None = None
    if arguments.manifest_out is not None:
        protected = [arguments.session, arguments.volume]
        if arguments.mesh is not None:
            protected.append(arguments.mesh)
        manifest_path = reserve_output(
            arguments.manifest_out,
            ".json",
            protected=protected,
        )
        source_commit, worktree_clean = source_state()
        if worktree_clean is False and not arguments.allow_dirty:
            raise SystemExit(
                "refusing to write a result manifest from a dirty working "
                "tree: the recorded commit would not describe the code that "
                "produced these numbers. Commit first, or pass --allow-dirty "
                "to record the run as unclean."
            )

    session = load_scan_session(arguments.session)
    volume = load_tsdf_block_volume(arguments.volume)
    stride = volume.frame_stride
    if stride < 2:
        raise SystemExit(
            "this volume was fused from every frame (frame_stride=1), so "
            "there are no held-out frames to score it against"
        )
    offset = (
        stride // 2
        if arguments.held_out_offset is None
        else arguments.held_out_offset
    )
    if offset % stride == 0:
        raise SystemExit(
            f"--held-out-offset {offset} is a multiple of "
            f"the volume's frame_stride={stride}; the evaluation "
            "frames would be the fused ones"
        )
    if offset >= stride:
        raise SystemExit(
            f"--held-out-offset {offset} is outside the "
            f"volume's frame_stride={stride}; use an offset in "
            f"[1, {stride - 1}]"
        )
    if session.session_id != volume.session_id:
        raise SystemExit(
            f"volume was fused from session {volume.session_id!r}, not "
            f"{session.session_id!r}"
        )
    replay = replay_session(session)
    if replay.digest_sha256 != volume.replay_digest_sha256:
        raise SystemExit(
            "the session's replay digest does not match the one recorded in "
            "the volume; this is not the scan the volume was fused from"
        )
    try:
        camera, depth_scale_m = _validate_reconstruction_contract(session)
    except PointCloudError as error:
        raise SystemExit(str(error)) from error
    mesh = (
        None
        if arguments.mesh is None
        else describe_mesh(arguments.mesh, volume.artifact_digest_sha256)
    )

    print(
        f"volume: blocks={volume.block_count} "
        f"voxel={volume.voxel_size_m} truncation={volume.truncation_m} "
        f"stride={stride} fused_frames={volume.fused_observations}"
    )
    print(
        f"observed voxels: {volume.observed_voxel_count} of "
        f"{volume.voxel_slots} "
        f"({100 * volume.observed_voxel_count / volume.voxel_slots:.1f}% "
        "of planned slots)"
    )
    print(f"volume sha256: {volume.artifact_digest_sha256}")
    if mesh is not None:
        print(
            f"mesh: vertices={mesh['vertices']} "
            f"triangles={mesh['triangles']} (extracted from this volume)"
        )

    sampler = BlockVolumeSampler(volume)
    held_out = [
        observation
        for observation in replay.observations
        if observation.sequence % stride == offset
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

    started = time.perf_counter()
    nearest_chunks: list[np.ndarray] = []
    linear_chunks: list[np.ndarray] = []
    sampled = 0
    covered = 0
    for observation in held_out:
        world, count = back_project(
            session,
            observation,
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
            nearest[np.isfinite(nearest)] * volume.truncation_m
        )
        linear_chunks.append(
            linear[np.isfinite(linear)] * volume.truncation_m
        )
    evaluation_seconds = time.perf_counter() - started
    if not nearest_chunks or not linear_chunks:
        raise SystemExit("no held-out depth sample fell inside the volume")

    print(
        f"held-out depth samples: {sampled} "
        f"(every {arguments.pixel_step} pixels); "
        f"inside observed voxels: {covered} "
        f"({100 * covered / max(sampled, 1):.1f}%)"
    )
    nearest_errors = np.concatenate(nearest_chunks)
    linear_errors = np.concatenate(linear_chunks)
    describe("nearest voxel", nearest_errors, volume.voxel_size_m)
    describe("trilinear", linear_errors, volume.voxel_size_m)

    if manifest_path is not None:
        manifest = {
            "schema": RESULT_MANIFEST_SCHEMA,
            "schema_version": RESULT_MANIFEST_SCHEMA_VERSION,
            "measurement": "held-out-tsdf-residual",
            "measures": (
                "agreement between the fused volume and depth measured by "
                "frames it never saw, using ground-truth camera poses. This "
                "is a cross-view consistency residual, not distance to a "
                "surveyed surface."
            ),
            "source_commit": source_commit,
            "source_worktree_clean": worktree_clean,
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "platform": platform.platform(),
                "processor": platform.processor(),
            },
            "inputs": {
                "session": arguments.session.as_posix(),
                "session_id": session.session_id,
                "replay_digest_sha256": volume.replay_digest_sha256,
                "plan_sha256": volume.source_plan_digest_sha256,
                "volume": arguments.volume.as_posix(),
                "volume_sha256": volume.artifact_digest_sha256,
            },
            "reconstruction": {
                "path": "sparse-block-streaming-fusion",
                "persisted": True,
                "voxel_size_m": volume.voxel_size_m,
                "truncation_m": volume.truncation_m,
                "frame_stride": stride,
                "fused_frames": volume.fused_observations,
                "active_blocks": volume.block_count,
                "planned_voxel_slots": volume.voxel_slots,
                "observed_voxel_slots": volume.observed_voxel_count,
                "contributions_applied": volume.contributions_applied,
                "contributions_evaluated": volume.contributions_evaluated,
                "maximum_weight": volume.maximum_weight,
            },
            "mesh": mesh,
            "evaluation": {
                "held_out_offset": offset,
                "held_out_frames": len(held_out),
                "pixel_step": arguments.pixel_step,
                "depth_samples": sampled,
                "samples_in_observed_voxels": covered,
                "coverage_fraction": covered / max(sampled, 1),
                "nearest_voxel": summarise(
                    nearest_errors,
                    volume.voxel_size_m,
                ),
                "trilinear": summarise(linear_errors, volume.voxel_size_m),
            },
            "timings_seconds": {
                "evaluation": round(evaluation_seconds, 3),
            },
        }
        payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        with publishing(manifest_path, ".json") as temporary:
            temporary.write_text(payload, encoding="utf-8")
        print(f"\nwrote manifest {manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except TsdfError as error:
        print(f"TUM RECONSTRUCTION REPORT FAILED: {error}", file=sys.stderr)
        sys.exit(2)
