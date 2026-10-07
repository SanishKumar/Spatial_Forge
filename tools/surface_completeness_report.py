"""Measure how much of the true surface a reconstruction recovered.

``surface_accuracy_report.py`` asks whether the surface that was built is
in the right place. A mesh of one perfect square metre of wall passes that
with full marks. This asks the other question: of the true surface the
cameras saw, how much is in the mesh.

That needs two definitions, and both are stated rather than left to taste.

What the cameras saw. A point of the ground-truth model counts as seen by a
frame if it projects inside the image, in front of the camera, on the side
of the surface that faces it, and the depth the frame measured at that
pixel is the point's own depth to within a tolerance. The last condition is
what makes it sight rather than line of sight: a point behind a sofa
projects into the image too, and the depth there is the sofa's. A point is
*observable* if enough fused frames saw it. Enough is, by default, the
number of observations the mesh itself demanded of a voxel, because a
surface seen less often than that was never going to be in the mesh.

What it means to be in the mesh. The distance from the point to the mesh,
measured exactly to its triangles and not to their vertices, is no more
than a threshold. Several thresholds are reported.

The frame the model is placed in is not fitted here. It is taken from an
accuracy report of the same model and trajectory, checked by digest, so
that accuracy and completeness are two readings in one frame.

Visibility can be decided on a different session of the same trajectory
with ``--visibility-session``. A noisy sequence's own depth disagrees with
the truth by its noise, so judging what it "saw" by that depth would shrink
the observable set for a reason that has nothing to do with the scene. The
exact sequence decides instead, and the noisy mesh is held to the same
surface the exact one is.

Usage:

    python tools/surface_completeness_report.py SESSION VOLUME.sftvol
        MESH.ply MODEL.ply --alignment ACCURACY.json
        [--alignment-session SESSION] [--visibility-session SESSION]
        [--model-stride N] [--min-views N] [--depth-tolerance-m T]
        [--reach-m R] [--manifest-out RESULT.json] [--allow-dirty]
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):  # run as a script rather than imported
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools._output import (  # noqa: E402
    positive_int,
    publishing,
    reserve_output,
    source_state,
)
from tools._triangles import TriangleIndex  # noqa: E402
from tools.render_mesh import read_mesh_ply  # noqa: E402
from tools.surface_accuracy_report import (  # noqa: E402
    ORIENTED_FRACTION,
    TRAJECTORY_TOLERANCE_M,
    TRAJECTORY_TOLERANCE_RAD,
    _apply,
    _invert_rigid,
    _sha256,
    load_alignment,
    read_surface_model,
    session_from_source,
    source_trajectory,
    trajectory_difference,
)
from tools.tum_reconstruction_report import (  # noqa: E402
    RESULT_MANIFEST_SCHEMA,
    RESULT_MANIFEST_SCHEMA_VERSION,
    describe_mesh,
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

THRESHOLDS_MM = (5, 10, 20)
# Nearer than this a point is not in front of the camera in any useful
# sense, and its projection is unstable.
_NEAR_PLANE_M = 0.05


def count_views(
    points: np.ndarray,
    normals: np.ndarray | None,
    session,
    observations,
    camera,
    depth_scale_m: float,
    *,
    depth_tolerance_m: float,
) -> np.ndarray:
    """How many of the given frames saw each point, in the session frame.

    ``normals`` restricts sight to the side of the surface that faces the
    camera; pass ``None`` when the model's normals are not known to be
    oriented.
    """

    views = np.zeros(len(points), dtype=np.int32)
    for observation in observations:
        pose = np.array(
            [float(value) for value in observation.pose.data["T_world_camera"]]
        ).reshape((4, 4))
        centre = pose[:3, 3]
        offset = points - centre
        # Row vectors times R is R transposed times each offset.
        in_camera = offset @ pose[:3, :3]
        depth = in_camera[:, 2]
        seen = depth > _NEAR_PLANE_M
        if normals is not None:
            seen &= np.einsum("ij,ij->i", normals, offset) < 0.0
        safe = np.where(seen, depth, 1.0)
        column = np.floor(camera.fx * in_camera[:, 0] / safe + camera.cx + 0.5)
        row = np.floor(camera.fy * in_camera[:, 1] / safe + camera.cy + 0.5)
        seen &= (
            (column >= 0)
            & (column < camera.width)
            & (row >= 0)
            & (row < camera.height)
        )
        measured = (
            _read_depth_array(
                _sample_path(session, observation.depth.data, "depth"),
                camera.width,
                camera.height,
            )
            .astype(np.float64)
            .reshape((camera.height, camera.width))
            * depth_scale_m
        )
        chosen = np.flatnonzero(seen)
        at_pixel = measured[
            row[chosen].astype(np.int64), column[chosen].astype(np.int64)
        ]
        agrees = (at_pixel > 0.0) & (
            np.abs(at_pixel - depth[chosen]) <= depth_tolerance_m
        )
        views[chosen[agrees]] += 1
    return views


def summarise(
    distances_m: np.ndarray,
    *,
    reach_m: float,
    voxel_size_m: float,
) -> dict[str, object]:
    """Completeness of the observable points, given their mesh distances.

    A point farther than the reach has no distance. It ranks above every
    point that has one, so percentiles stay exact for as long as they fall
    among the measured, and are not stated otherwise.
    """

    total = len(distances_m)
    if total == 0:
        raise SystemExit(
            "no point of the model was seen by enough frames; check the "
            "alignment, or lower --min-views"
        )
    found = np.sort(distances_m[np.isfinite(distances_m)])

    def ranked(percent: float) -> float | None:
        position = max(0, math.ceil(percent / 100.0 * total) - 1)
        if position >= len(found):
            return None
        return float(1000.0 * found[position])

    within = {
        f"{threshold}mm": float(
            np.count_nonzero(found <= threshold / 1000.0) / total
        )
        for threshold in THRESHOLDS_MM
        if threshold / 1000.0 <= reach_m
    }
    if voxel_size_m <= reach_m:
        within["one_voxel"] = float(
            np.count_nonzero(found <= voxel_size_m) / total
        )
    return {
        "observable_points": total,
        "reach_m": reach_m,
        "beyond_reach": total - len(found),
        "beyond_reach_fraction": (total - len(found)) / total,
        "within_fraction": within,
        "distance_to_mesh": {
            "median_mm": ranked(50),
            "p90_mm": ranked(90),
            "p95_mm": ranked(95),
            "p99_mm": ranked(99),
        },
    }


def _same_camera(first, second) -> bool:
    return all(
        getattr(first, name) == getattr(second, name)
        for name in ("width", "height", "fx", "fy", "cx", "cy")
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("session", type=Path)
    parser.add_argument("volume", type=Path)
    parser.add_argument("mesh", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--alignment-session", type=Path, default=None)
    parser.add_argument("--visibility-session", type=Path, default=None)
    parser.add_argument("--model-stride", type=positive_int, default=4)
    parser.add_argument("--min-views", type=positive_int, default=None)
    parser.add_argument("--depth-tolerance-m", type=float, default=0.02)
    parser.add_argument("--reach-m", type=float, default=0.03)
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
    for name in ("depth_tolerance_m", "reach_m"):
        value = getattr(arguments, name)
        if not (math.isfinite(value) and value > 0.0):
            parser.error(
                f"--{name.replace('_', '-')} must be finite and positive"
            )

    inputs = [
        arguments.session,
        arguments.volume,
        arguments.mesh,
        arguments.model,
        arguments.alignment,
    ]
    for optional in (arguments.alignment_session, arguments.visibility_session):
        if optional is not None:
            inputs.append(optional)
    manifest_path: Path | None = None
    source_commit: str | None = None
    worktree_clean: bool | None = None
    if arguments.manifest_out is not None:
        manifest_path = reserve_output(
            arguments.manifest_out, ".json", protected=inputs
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
    mesh = describe_mesh(arguments.mesh, volume.artifact_digest_sha256)
    minimum_views = (
        max(1, int(mesh["minimum_weight"]))
        if arguments.min_views is None
        else arguments.min_views
    )

    trajectory = source_trajectory(replay.observations)
    reference_trajectory = None
    if arguments.alignment_session is not None:
        reference_trajectory = source_trajectory(
            replay_session(
                load_scan_session(arguments.alignment_session)
            ).observations
        )
    to_session, _, _ = session_from_source(replay.observations)

    started = time.perf_counter()
    model_points, model_normals = read_surface_model(arguments.model)
    model_digest = _sha256(arguments.model)
    model_from_source, reused = load_alignment(
        arguments.alignment,
        model_sha256=model_digest,
        trajectory=trajectory,
        reference_trajectory=reference_trajectory,
    )
    recorded_facing = reused["model_normals_facing_camera_fraction"]
    oriented = recorded_facing is not None and recorded_facing >= ORIENTED_FRACTION

    # What was seen is decided on this session, or on a cleaner one of the
    # same trajectory.
    sight_session, sight_replay, sight_camera, sight_scale = (
        session,
        replay,
        camera,
        depth_scale_m,
    )
    sight: dict[str, object] = {"session": "the evaluated session"}
    if arguments.visibility_session is not None:
        sight_session = load_scan_session(arguments.visibility_session)
        sight_replay = replay_session(sight_session)
        try:
            sight_camera, sight_scale = _validate_reconstruction_contract(
                sight_session
            )
        except PointCloudError as error:
            raise SystemExit(str(error)) from error
        if not _same_camera(camera, sight_camera):
            raise SystemExit(
                "the visibility session has a different camera; what it "
                "saw says nothing about what this session saw"
            )
        difference = trajectory_difference(
            source_trajectory(sight_replay.observations), trajectory
        )
        if difference is None or (
            difference[0] > TRAJECTORY_TOLERANCE_M
            or difference[1] > TRAJECTORY_TOLERANCE_RAD
        ):
            raise SystemExit(
                "the visibility session follows a different trajectory; "
                "what it saw says nothing about what this session saw"
            )
        sight = {
            "session": arguments.visibility_session.as_posix(),
            "session_id": sight_session.session_id,
            "replay_digest_sha256": sight_replay.digest_sha256,
            "largest_translation_difference_m": difference[0],
            "largest_rotation_difference_rad": difference[1],
        }
    frames = [
        observation
        for observation in sight_replay.observations
        if observation.sequence % volume.frame_stride == 0
        and observation.depth is not None
        and observation.pose is not None
    ]
    if not frames:
        raise SystemExit("no fused frame has both depth and pose")

    session_from_model = to_session @ _invert_rigid(model_from_source)
    points = _apply(session_from_model, model_points[::arguments.model_stride])
    normals = (
        model_normals[::arguments.model_stride].astype(np.float64)
        @ session_from_model[:3, :3].T
        if oriented
        else None
    )
    print(
        f"model: {len(points)} of {len(model_points)} points "
        f"(every {arguments.model_stride})"
    )
    print(
        f"mesh: vertices={mesh['vertices']} triangles={mesh['triangles']} "
        f"(extracted from this volume, minimum weight "
        f"{mesh['minimum_weight']})"
    )
    views = count_views(
        points,
        normals,
        sight_session,
        frames,
        sight_camera,
        sight_scale,
        depth_tolerance_m=arguments.depth_tolerance_m,
    )
    observable = views >= minimum_views
    visibility_seconds = time.perf_counter() - started
    print(
        f"seen by at least {minimum_views} of {len(frames)} fused frames: "
        f"{int(np.count_nonzero(observable))} points "
        f"({100 * observable.mean():.1f}% of those considered); seen by "
        f"none: {int(np.count_nonzero(views == 0))}"
    )

    started = time.perf_counter()
    vertices, faces = read_mesh_ply(arguments.mesh)
    distances = TriangleIndex(
        vertices, faces, reach_m=arguments.reach_m
    ).distance(points[observable])
    summary = summarise(
        distances,
        reach_m=arguments.reach_m,
        voxel_size_m=volume.voxel_size_m,
    )
    distance_seconds = time.perf_counter() - started

    within = summary["within_fraction"]
    print(
        "of the surface that was seen, within "
        + " / ".join(key.replace("_", " ") for key in within)
        + " of the mesh: "
        + " / ".join(f"{100 * value:.1f}%" for value in within.values())
    )
    print(
        f"farther than {1000 * arguments.reach_m:.0f} mm from any of it: "
        f"{100 * summary['beyond_reach_fraction']:.1f}%"
    )

    if manifest_path is not None:
        manifest = {
            "schema": RESULT_MANIFEST_SCHEMA,
            "schema_version": RESULT_MANIFEST_SCHEMA_VERSION,
            "measurement": "surface-completeness-against-ground-truth-model",
            "measures": (
                "of the ground-truth model's points that enough fused "
                "frames saw, the share lying within a threshold of the "
                "reconstructed mesh, measured exactly to its triangles. "
                "Seen means inside the image, in front of the camera, "
                "facing it, and at the depth the frame measured. The "
                "model is placed by an alignment taken from an accuracy "
                "report, not fitted here."
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
                "model": arguments.model.as_posix(),
                "model_sha256": model_digest,
                "model_points": len(model_points),
            },
            "reconstruction": {
                "voxel_size_m": volume.voxel_size_m,
                "truncation_m": volume.truncation_m,
                "frame_stride": volume.frame_stride,
                "fused_frames": volume.fused_observations,
            },
            "mesh": mesh,
            "alignment": {
                "reused_from": reused,
                "model_from_source": [
                    float(value) for value in model_from_source.ravel()
                ],
            },
            "visibility": {
                "decided_on": sight,
                "frames": len(frames),
                "minimum_views": minimum_views,
                "depth_tolerance_m": arguments.depth_tolerance_m,
                "facing_test": oriented,
                "model_stride": arguments.model_stride,
                "points_considered": len(points),
                "seen_by_no_frame": int(np.count_nonzero(views == 0)),
                "observable_fraction": float(observable.mean()),
            },
            "completeness": summary,
            "timings_seconds": {
                "visibility": round(visibility_seconds, 3),
                "distance": round(distance_seconds, 3),
            },
        }
        payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        with publishing(manifest_path, ".json") as temporary:
            # Bytes, not text: text mode would translate line endings and
            # make the same manifest differ between platforms.
            temporary.write_bytes(payload.encode("utf-8"))
        print(f"\nwrote manifest {manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except TsdfError as error:
        print(f"SURFACE COMPLETENESS REPORT FAILED: {error}", file=sys.stderr)
        sys.exit(2)
