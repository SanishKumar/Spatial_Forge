"""Measure a reconstructed mesh against a ground-truth surface model.

The held-out residual in ``tum_reconstruction_report.py`` is agreement
between views. A reconstruction that is wrong the same way from every
viewpoint scores well on it. This is the other measurement: how far the
reconstructed surface is from a surface that is known.

It needs a dataset that publishes one. ICL-NUIM does: a dense, oriented
point model of the room its sequences were rendered from. For every vertex
of the mesh this finds the nearest model point, exactly, and reports

  - the distance to that point, which is the statistic ICL-NUIM's own
    evaluation tool reports, and an upper bound on the distance to the true
    surface: the model's points lie on it, but only every few millimetres;
  - the distance to the plane through that point, along the model's normal,
    which removes the model's sampling from the figure.

Two things have to be right before either number means anything, and both
are measured here rather than assumed.

The frames have to be aligned. The dataset publishes its trajectory and its
model in different frames and does not publish the transform between them.
It is fitted here, as one rigid motion, by point-to-plane registration of
the *raw depth images* to the model. Never of the reconstruction: an
alignment fitted to the thing being measured would absorb part of its error
and report the rest. The mesh is then judged in a frame it had no part in
choosing.

The measurement has a floor. Depth that the dataset itself rendered, taken
from frames the fit did not use, is scored against the model the same way.
That is the error of a perfect reconstruction under this alignment and this
model sampling, and the mesh's figures are to be read against it.

The mesh must have been extracted from the volume it is named with, and the
volume fused from the session it is named with; both are checked by digest.

Usage:

    python tools/surface_accuracy_report.py SESSION VOLUME.sftvol MESH.ply
        MODEL.ply [--initial-translation X Y Z] [--fit-frame-stride N]
        [--alignment EARLIER.json] [--pixel-step N] [--limit-m L]
        [--errors-out ERRORS.npy] [--manifest-out RESULT.json]
        [--allow-dirty]

``--initial-translation`` is a rough position of the trajectory's origin in
the model's frame, in metres. The fit only needs it to within a few
centimetres. It is expressed in the frame of the trajectory the session was
imported from or, for a session that was not imported, the session's own.

``--alignment`` takes the manifest of an earlier report and judges this
mesh in the frame that report fitted, without fitting again. It is for
comparing scans of one trajectory, a clean one and a degraded one, in a
single frame chosen from the better data. It is refused unless the model
and the source trajectory are, by digest, the ones the earlier report
used: an alignment says where one trajectory's frame sits in one model,
and nothing about any other.

``--errors-out`` writes each vertex's distance to the model's tangent
plane, in metres and in the mesh's vertex order, for
``render_mesh.py --vertex-errors`` to draw: where the error is, not
only how much of it there is.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

if __package__ in (None, ""):  # run as a script rather than imported
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools._nearest import NearestPointIndex  # noqa: E402
from tools._output import (  # noqa: E402
    positive_int,
    publishing,
    reserve_output,
    source_state,
)
from tools.render_mesh import read_mesh_ply  # noqa: E402
from tools.tum_reconstruction_report import (  # noqa: E402
    RESULT_MANIFEST_SCHEMA,
    RESULT_MANIFEST_SCHEMA_VERSION,
    back_project,
    describe_mesh,
)

from spatialforge import (  # noqa: E402
    load_scan_session,
    load_tsdf_block_volume,
    replay_session,
)
from spatialforge.errors import PointCloudError, TsdfError  # noqa: E402
from spatialforge.point_cloud import (  # noqa: E402
    _validate_reconstruction_contract,
)

# Registration runs coarse to fine. The coarse stage sees a thinned model
# through a wide gate, so a rough starting guess is enough; the fine stage
# sees every model point through a narrow one.
COARSE_LIMIT_M = 0.08
COARSE_ITERATIONS = 12
COARSE_MODEL_POINTS = 600_000
FINE_LIMIT_M = 0.02
FINE_ITERATIONS = 6
# The last fine update must be smaller than this, in metres and in radians,
# or the fit is reported as not having settled.
CONVERGED_STEP = 1e-5
# Source and session poses must describe one rigid re-anchoring.
POSE_CONSISTENCY_TOLERANCE = 1e-9
# Model normals are only trusted for a signed figure if the cameras that
# rendered the depth are in front of nearly all of them.
ORIENTED_FRACTION = 0.99

_PLY_TYPES = {
    "char": "i1", "int8": "i1",
    "uchar": "u1", "uint8": "u1",
    "short": "<i2", "int16": "<i2",
    "ushort": "<u2", "uint16": "<u2",
    "int": "<i4", "int32": "<i4",
    "uint": "<u4", "uint32": "<u4",
    "float": "<f4", "float32": "<f4",
    "double": "<f8", "float64": "<f8",
}


def read_surface_model(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Points and unit normals of a binary little-endian point model."""

    with path.open("rb") as handle:
        if handle.readline() != b"ply\n":
            raise SystemExit(f"{path}: not a PLY file")
        header: list[str] = []
        while True:
            line = handle.readline()
            if not line or len(header) > 1_000:
                raise SystemExit(f"{path}: PLY header does not end")
            if line == b"end_header\n":
                break
            header.append(line.decode("ascii", errors="replace").strip())
        offset = handle.tell()
    if "format binary_little_endian 1.0" not in header:
        raise SystemExit(f"{path}: expected a binary little-endian PLY")

    count: int | None = None
    fields: list[tuple[str, str]] = []
    element: str | None = None
    for line in header:
        parts = line.split()
        if parts[:1] == ["element"]:
            if len(parts) != 3 or not parts[2].isdigit():
                raise SystemExit(f"{path}: malformed element: {line}")
            # Vertex data is read from the end of the header, which is
            # only where it is if no other element precedes it.
            if parts[1] == "vertex" and element is not None:
                raise SystemExit(
                    f"{path}: the vertex element must come first, once"
                )
            element = parts[1]
            if element == "vertex":
                count = int(parts[2])
        elif parts[:1] == ["property"] and element == "vertex":
            if len(parts) != 3 or parts[1] not in _PLY_TYPES:
                raise SystemExit(
                    f"{path}: unsupported vertex property: {line}"
                )
            fields.append((parts[2], _PLY_TYPES[parts[1]]))
    names = [name for name, _ in fields]
    if count is None or count < 1:
        raise SystemExit(f"{path}: model has no vertices")
    if len(set(names)) != len(names):
        raise SystemExit(f"{path}: repeated vertex property")
    missing = [
        name for name in ("x", "y", "z", "nx", "ny", "nz") if name not in names
    ]
    if missing:
        raise SystemExit(
            f"{path}: model needs positions and normals; missing "
            f"{', '.join(missing)}"
        )
    record = np.dtype(fields)
    if path.stat().st_size < offset + count * record.itemsize:
        raise SystemExit(f"{path}: payload is shorter than its header says")

    data = np.fromfile(path, dtype=record, count=count, offset=offset)
    points = np.empty((count, 3))
    normals = np.empty((count, 3), dtype=np.float32)
    for column, name in enumerate(("x", "y", "z")):
        points[:, column] = data[name]
    for column, name in enumerate(("nx", "ny", "nz")):
        normals[:, column] = data[name]
    if not (np.all(np.isfinite(points)) and np.all(np.isfinite(normals))):
        raise SystemExit(f"{path}: model contains non-finite values")
    lengths = np.sqrt(
        np.einsum("ij,ij->i", normals, normals, dtype=np.float64)
    )
    if np.any(np.abs(lengths - 1.0) > 1e-3):
        raise SystemExit(f"{path}: model normals are not unit length")
    return points, normals


def quaternion_matrix(quaternion_xyzw) -> np.ndarray:
    x, y, z, w = (float(value) for value in quaternion_xyzw)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not (math.isfinite(norm) and norm > 0.0):
        raise SystemExit("source pose has an unusable quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def session_from_source(observations) -> tuple[np.ndarray, float, bool]:
    """The rigid motion from the imported trajectory's frame to the session.

    An importer re-anchors the session at its first pose and keeps each
    source pose beside the session pose it became. Every such pair implies
    the same motion; the largest disagreement between them is returned so
    that it can be checked and recorded. A session that was not imported
    has no source frame, and its own is used.
    """

    implied = []
    for observation in observations:
        if observation.pose is None:
            continue
        data = observation.pose.data
        if (
            "source_translation_m" not in data
            or "source_quaternion_xyzw" not in data
        ):
            continue
        source = np.eye(4)
        source[:3, :3] = quaternion_matrix(data["source_quaternion_xyzw"])
        source[:3, 3] = [float(v) for v in data["source_translation_m"]]
        session = np.array(
            [float(v) for v in data["T_world_camera"]]
        ).reshape((4, 4))
        implied.append(session @ _invert_rigid(source))
    if not implied:
        return np.eye(4), 0.0, False
    deviation = max(
        float(np.abs(matrix - implied[0]).max()) for matrix in implied
    )
    return implied[0], deviation, True


def source_trajectory_digest(observations) -> str:
    """One digest for the poses that define the source frame.

    The source pose where the importer kept one, the session pose where
    it did not, for every posed observation in order. Two sessions with
    the same digest were placed by the same trajectory, whatever their
    images are.
    """

    rows = []
    for observation in observations:
        if observation.pose is None:
            continue
        data = observation.pose.data
        if (
            "source_translation_m" in data
            and "source_quaternion_xyzw" in data
        ):
            values = [
                *data["source_translation_m"],
                *data["source_quaternion_xyzw"],
            ]
        else:
            values = list(data["T_world_camera"])
        rows.append([float(value) for value in values])
    encoded = json.dumps(rows, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def load_alignment(
    path: Path,
    *,
    model_sha256: str,
    trajectory_sha256: str,
) -> tuple[np.ndarray, dict[str, object]]:
    """The transform an earlier report fitted, if it applies here."""

    encoded = path.read_bytes()
    try:
        manifest = json.loads(encoded)
        if manifest["measurement"] != (
            "surface-distance-to-ground-truth-model"
        ):
            raise KeyError("measurement")
        recorded_model = manifest["inputs"]["model_sha256"]
        recorded_trajectory = manifest["inputs"][
            "source_trajectory_sha256"
        ]
        values = manifest["registration"]["model_from_source"]
        transform = np.array(values, dtype=np.float64).reshape((4, 4))
    except (ValueError, KeyError, TypeError):
        raise SystemExit(
            f"{path}: not a surface-accuracy manifest that records an "
            "alignment, its model and its source trajectory"
        ) from None
    rotation = transform[:3, :3]
    if not (
        np.all(np.isfinite(transform))
        and np.array_equal(transform[3], (0.0, 0.0, 0.0, 1.0))
        and np.abs(rotation @ rotation.T - np.eye(3)).max() < 1e-9
        and abs(float(np.linalg.det(rotation)) - 1.0) < 1e-9
    ):
        raise SystemExit(f"{path}: recorded alignment is not rigid")
    if recorded_model != model_sha256:
        raise SystemExit(
            f"{path}: that alignment was fitted to a different "
            "ground-truth model"
        )
    if recorded_trajectory != trajectory_sha256:
        raise SystemExit(
            f"{path}: that alignment belongs to a different source "
            "trajectory, so it does not say where this session is"
        )
    return transform, {
        "path": path.as_posix(),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "session_id": manifest["inputs"].get("session_id"),
        "replay_digest_sha256": manifest["inputs"].get(
            "replay_digest_sha256"
        ),
    }


def _invert_rigid(matrix: np.ndarray) -> np.ndarray:
    inverse = np.eye(4)
    inverse[:3, :3] = matrix[:3, :3].T
    inverse[:3, 3] = -matrix[:3, :3].T @ matrix[:3, 3]
    return inverse


def _apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def _rotation_from_vector(vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(vector))
    if angle < 1e-300:
        return np.eye(3)
    k = vector / angle
    cross = np.array(
        [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]]
    )
    return (
        np.eye(3)
        + math.sin(angle) * cross
        + (1.0 - math.cos(angle)) * (cross @ cross)
    )


@dataclass(frozen=True, slots=True)
class RegistrationStage:
    limit_m: float
    iterations: int
    model_points: int
    matched_fraction: float
    last_step_m: float
    last_step_rad: float


def register_point_to_plane(
    points: np.ndarray,
    model_points: np.ndarray,
    model_normals: np.ndarray,
    index: NearestPointIndex,
    transform: np.ndarray,
    *,
    limit_m: float,
    iterations: int,
) -> tuple[np.ndarray, RegistrationStage]:
    """Refine ``transform`` so ``points`` lie on the model's tangent planes.

    Each iteration pairs every point with its nearest model point within
    ``limit_m`` and solves, in closed form, for the small rigid motion that
    minimises the squared distances to the planes through those points.
    """

    transform = transform.copy()
    matched_fraction = 0.0
    step_m = step_rad = math.inf
    for _ in range(iterations):
        moved = _apply(transform, points)
        found, _ = index.nearest(moved, limit_m=limit_m)
        keep = found >= 0
        matched_fraction = float(np.count_nonzero(keep)) / len(points)
        if np.count_nonzero(keep) < 6:
            raise SystemExit(
                "registration found no model surface near the depth; the "
                "initial translation is too far off"
            )
        q = moved[keep]
        normal = model_normals[found[keep]].astype(np.float64)
        residual = np.einsum("ij,ij->i", q - model_points[found[keep]], normal)
        jacobian = np.concatenate([np.cross(q, normal), normal], axis=1)
        normal_matrix = jacobian.T @ jacobian
        spectrum = np.linalg.eigvalsh(normal_matrix)
        if spectrum[0] <= 1e-10 * spectrum[-1]:
            raise SystemExit(
                "registration is under-constrained: the depth does not see "
                "surfaces in enough directions to fix a rigid motion"
            )
        solution = np.linalg.solve(normal_matrix, -(jacobian.T @ residual))
        update = np.eye(4)
        update[:3, :3] = _rotation_from_vector(solution[:3])
        update[:3, 3] = solution[3:]
        transform = update @ transform
        step_rad = float(np.linalg.norm(solution[:3]))
        step_m = float(np.linalg.norm(solution[3:]))
    # Remove the rounding a product of rotations accumulates.
    left, _, right = np.linalg.svd(transform[:3, :3])
    transform[:3, :3] = left @ right
    return transform, RegistrationStage(
        limit_m=limit_m,
        iterations=iterations,
        model_points=len(model_points),
        matched_fraction=matched_fraction,
        last_step_m=step_m,
        last_step_rad=step_rad,
    )


@dataclass(frozen=True, slots=True)
class SurfaceDistances:
    """Per-point distances to the model; unmatched points are excluded."""

    total: int
    matched: np.ndarray
    nearest_m: np.ndarray
    signed_plane_m: np.ndarray
    model_index: np.ndarray


def measure(
    points: np.ndarray,
    model_points: np.ndarray,
    model_normals: np.ndarray,
    index: NearestPointIndex,
    *,
    limit_m: float,
) -> SurfaceDistances:
    found, distance = index.nearest(points, limit_m=limit_m)
    keep = found >= 0
    matched = found[keep]
    signed = np.einsum(
        "ij,ij->i",
        points[keep] - model_points[matched],
        model_normals[matched].astype(np.float64),
    )
    return SurfaceDistances(
        total=len(points),
        matched=keep,
        nearest_m=distance[keep],
        signed_plane_m=signed,
        model_index=matched,
    )


def _millimetres(value: float) -> float | None:
    return float(1000.0 * value) if math.isfinite(value) else None


def summarise(
    distances: SurfaceDistances,
    *,
    limit_m: float,
    signed: bool,
) -> dict[str, object]:
    """Statistics over every point; ones beyond the limit count as worse.

    Percentiles rank an unmatched point above every matched one, so they
    stay exact for as long as they fall among the matched. Means cannot
    include a distance that was never found, so they are taken over the
    matched points and the number left out is reported beside them.
    """

    matched = len(distances.nearest_m)
    if matched == 0:
        raise SystemExit(
            f"no point lies within {limit_m} m of the model; the frames "
            "are not aligned"
        )

    def block(values: np.ndarray) -> dict[str, float | None]:
        ordered = np.sort(values)

        def ranked(percent: float) -> float | None:
            # Nearest rank over every point: always an observed value,
            # never an interpolation towards one that was not found.
            position = max(
                0, math.ceil(percent / 100.0 * distances.total) - 1
            )
            if position >= matched:
                return None
            return _millimetres(float(ordered[position]))

        return {
            "mean_mm": _millimetres(float(values.mean())),
            "std_mm": _millimetres(float(values.std())),
            "rms_mm": _millimetres(float(np.sqrt((values**2).mean()))),
            "median_mm": ranked(50),
            "p90_mm": ranked(90),
            "p95_mm": ranked(95),
            "p99_mm": ranked(99),
            "max_matched_mm": _millimetres(float(ordered[-1])),
        }

    absolute_plane = np.abs(distances.signed_plane_m)
    plane = block(absolute_plane)
    plane["mean_signed_mm"] = (
        _millimetres(float(distances.signed_plane_m.mean()))
        if signed
        else None
    )
    return {
        "points": distances.total,
        "matched": matched,
        "beyond_limit": distances.total - matched,
        "limit_m": limit_m,
        "nearest_point": block(distances.nearest_m),
        "point_to_plane": plane,
        "within_fraction": {
            f"{threshold_mm}mm": float(
                np.count_nonzero(
                    distances.nearest_m <= threshold_mm / 1000.0
                )
                / distances.total
            )
            for threshold_mm in (5, 10, 20)
        },
    }


def describe(label: str, summary: dict[str, object]) -> None:
    def text(value: object) -> str:
        return "    n/a" if value is None else f"{value:7.2f}"

    print(
        f"\n{label}: n={summary['points']} "
        f"(beyond {1000 * summary['limit_m']:.0f} mm: "
        f"{summary['beyond_limit']})"
    )
    for name, title in (
        ("nearest_point", "to nearest model point"),
        ("point_to_plane", "to model tangent plane"),
    ):
        block = summary[name]
        print(
            f"  {title}: mean {text(block['mean_mm'])}  "
            f"median {text(block['median_mm'])}  "
            f"rms {text(block['rms_mm'])}  "
            f"p95 {text(block['p95_mm'])} mm"
        )
    signed = summary["point_to_plane"]["mean_signed_mm"]
    if signed is not None:
        print(f"  mean signed, towards free space: {signed:+.2f} mm")
    within = summary["within_fraction"]
    print(
        "  within 5 / 10 / 20 mm of a model point: "
        f"{100 * within['5mm']:.1f}% / {100 * within['10mm']:.1f}% / "
        f"{100 * within['20mm']:.1f}%"
    )


def sample_depth(
    session,
    observations,
    camera,
    depth_scale_m,
    pixel_step: int,
    source_from_session: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Depth points and their camera centres, in the source frame."""

    columns, rows = np.meshgrid(
        np.arange(0, camera.width, pixel_step),
        np.arange(0, camera.height, pixel_step),
        indexing="xy",
    )
    columns = columns.ravel()
    rows = rows.ravel()
    points = []
    centres = []
    for observation in observations:
        world, count = back_project(
            session, observation, camera, depth_scale_m, columns, rows
        )
        if world is None:
            continue
        pose = [
            float(value)
            for value in observation.pose.data["T_world_camera"]
        ]
        centre = np.array([pose[3], pose[7], pose[11]])
        points.append(_apply(source_from_session, world))
        centres.append(
            np.broadcast_to(
                _apply(source_from_session, centre[None, :]), (count, 3)
            )
        )
    if not points:
        raise SystemExit("no sampled frame has usable depth")
    return np.concatenate(points), np.concatenate(centres)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("session", type=Path)
    parser.add_argument("volume", type=Path)
    parser.add_argument("mesh", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument(
        "--initial-translation",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument(
        "--alignment",
        type=Path,
        default=None,
        help=(
            "Manifest of an earlier report whose fitted alignment is "
            "reused instead of fitting one. Same model and same source "
            "trajectory only."
        ),
    )
    parser.add_argument("--fit-frame-stride", type=positive_int, default=20)
    parser.add_argument("--pixel-step", type=positive_int, default=12)
    parser.add_argument("--limit-m", type=float, default=0.16)
    parser.add_argument(
        "--errors-out",
        type=Path,
        default=None,
        help=(
            "Write per-vertex distance to the model's tangent plane, in "
            "metres, as a float32 .npy; NaN where no model point was "
            "within the limit."
        ),
    )
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
    if not (math.isfinite(arguments.limit_m) and arguments.limit_m > 0.0):
        parser.error("--limit-m must be finite and positive")
    if arguments.alignment is not None:
        if arguments.initial_translation is not None:
            parser.error(
                "--alignment reuses a fitted alignment; an "
                "--initial-translation would have nothing to start"
            )
    elif arguments.initial_translation is None:
        arguments.initial_translation = (0.0, 0.0, 0.0)
    if arguments.initial_translation is not None and not all(
        math.isfinite(v) for v in arguments.initial_translation
    ):
        parser.error("--initial-translation must be finite")
    if arguments.fit_frame_stride < 2:
        parser.error(
            "--fit-frame-stride must be at least 2, so that some frames "
            "are left to check the fit against"
        )

    inputs = [
        arguments.session,
        arguments.volume,
        arguments.mesh,
        arguments.model,
    ]
    if arguments.alignment is not None:
        inputs.append(arguments.alignment)
    errors_path = (
        None
        if arguments.errors_out is None
        else reserve_output(arguments.errors_out, ".npy", protected=inputs)
    )
    manifest_path: Path | None = None
    source_commit: str | None = None
    worktree_clean: bool | None = None
    if arguments.manifest_out is not None:
        manifest_path = reserve_output(
            arguments.manifest_out,
            ".json",
            protected=inputs,
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

    to_session, pose_deviation, imported = session_from_source(
        replay.observations
    )
    if pose_deviation > POSE_CONSISTENCY_TOLERANCE:
        raise SystemExit(
            "the session's poses are not one rigid re-anchoring of its "
            f"source trajectory (they disagree by {pose_deviation:.3g}); "
            "the model cannot be placed in the session's frame"
        )
    source_from_session = _invert_rigid(to_session)
    trajectory_digest = source_trajectory_digest(replay.observations)

    posed = [
        observation
        for observation in replay.observations
        if observation.depth is not None and observation.pose is not None
    ]
    stride = arguments.fit_frame_stride
    fit_frames = [o for o in posed if o.sequence % stride == 0]
    check_frames = [o for o in posed if o.sequence % stride == stride // 2]
    if not fit_frames or not check_frames:
        raise SystemExit(
            "too few frames with depth and pose to fit an alignment and "
            "check it on others"
        )

    started = time.perf_counter()
    model_points, model_normals = read_surface_model(arguments.model)
    model_digest = _sha256(arguments.model)
    load_seconds = time.perf_counter() - started
    print(
        f"model: {len(model_points)} oriented points "
        f"({arguments.model.name})"
    )
    print(
        f"volume: blocks={volume.block_count} voxel={volume.voxel_size_m} "
        f"truncation={volume.truncation_m} "
        f"fused_frames={volume.fused_observations}"
    )
    print(
        f"mesh: vertices={mesh['vertices']} triangles={mesh['triangles']} "
        "(extracted from this volume)"
    )

    started = time.perf_counter()
    stages: list[RegistrationStage] = []
    fit_samples = 0
    reused: dict[str, object] | None = None
    if arguments.alignment is not None:
        transform, reused = load_alignment(
            arguments.alignment,
            model_sha256=model_digest,
            trajectory_sha256=trajectory_digest,
        )
        index = NearestPointIndex(model_points, cell_m=FINE_LIMIT_M)
    else:
        fit_points, _ = sample_depth(
            session,
            fit_frames,
            camera,
            depth_scale_m,
            arguments.pixel_step,
            source_from_session,
        )
        fit_samples = len(fit_points)
        thinning = max(
            1, math.ceil(len(model_points) / COARSE_MODEL_POINTS)
        )
        coarse_points = model_points[::thinning]
        coarse_normals = model_normals[::thinning]
        transform = np.eye(4)
        transform[:3, 3] = arguments.initial_translation
        transform, coarse = register_point_to_plane(
            fit_points,
            coarse_points,
            coarse_normals,
            NearestPointIndex(coarse_points, cell_m=COARSE_LIMIT_M),
            transform,
            limit_m=COARSE_LIMIT_M,
            iterations=COARSE_ITERATIONS,
        )
        index = NearestPointIndex(model_points, cell_m=FINE_LIMIT_M)
        transform, fine = register_point_to_plane(
            fit_points,
            model_points,
            model_normals,
            index,
            transform,
            limit_m=FINE_LIMIT_M,
            iterations=FINE_ITERATIONS,
        )
        if max(fine.last_step_m, fine.last_step_rad) > CONVERGED_STEP:
            raise SystemExit(
                "registration did not settle: its last update was "
                f"{fine.last_step_m:.3g} m and {fine.last_step_rad:.3g} "
                "rad. Give a closer --initial-translation."
            )
        stages = [coarse, fine]
    registration_seconds = time.perf_counter() - started
    angle = math.degrees(
        math.acos(
            max(-1.0, min(1.0, (float(np.trace(transform[:3, :3])) - 1) / 2))
        )
    )
    placed = (
        f"rotation {angle:.4f} deg, translation "
        f"({transform[0, 3]:+.5f}, {transform[1, 3]:+.5f}, "
        f"{transform[2, 3]:+.5f}) m"
    )
    if reused is not None:
        print(
            f"alignment: reused from {arguments.alignment.name}, not "
            f"fitted to this session; {placed}"
        )
    else:
        print(
            f"registration: {len(fit_frames)} frames, {fit_samples} "
            f"depth samples; {placed}; last step "
            f"{stages[-1].last_step_m:.1e} m"
        )

    started = time.perf_counter()
    check_points, check_centres = sample_depth(
        session,
        check_frames,
        camera,
        depth_scale_m,
        arguments.pixel_step,
        source_from_session,
    )
    depth_distances = measure(
        _apply(transform, check_points),
        model_points,
        model_normals,
        index,
        limit_m=arguments.limit_m,
    )
    to_camera = (
        _apply(transform, check_centres[depth_distances.matched])
        - model_points[depth_distances.model_index]
    )
    facing = float(
        np.count_nonzero(
            np.einsum(
                "ij,ij->i",
                to_camera,
                model_normals[depth_distances.model_index].astype(np.float64),
            )
            > 0.0
        )
        / max(len(to_camera), 1)
    )
    oriented = facing >= ORIENTED_FRACTION
    depth_summary = summarise(
        depth_distances, limit_m=arguments.limit_m, signed=oriented
    )

    vertices, _ = read_mesh_ply(arguments.mesh)
    mesh_distances = measure(
        _apply(transform @ source_from_session, vertices),
        model_points,
        model_normals,
        index,
        limit_m=arguments.limit_m,
    )
    mesh_summary = summarise(
        mesh_distances, limit_m=arguments.limit_m, signed=oriented
    )
    evaluation_seconds = time.perf_counter() - started

    describe(
        f"raw depth, {len(check_frames)} frames"
        + ("" if reused is not None else " the fit did not use"),
        depth_summary,
    )
    describe("mesh vertices", mesh_summary)
    if not oriented:
        print(
            "\nmodel normals face the camera for only "
            f"{100 * facing:.1f}% of depth samples; signed figures omitted"
        )

    vertex_errors: dict[str, object] | None = None
    if errors_path is not None:
        errors = np.full(mesh_distances.total, np.nan, dtype=np.float32)
        errors[mesh_distances.matched] = np.abs(
            mesh_distances.signed_plane_m
        )
        with publishing(errors_path, ".npy") as temporary:
            with temporary.open("wb") as handle:
                np.save(handle, errors)
        vertex_errors = {
            "path": arguments.errors_out.as_posix(),
            "sha256": _sha256(errors_path),
            "quantity": "absolute distance to model tangent plane, m",
        }
        print(f"\nwrote vertex errors {errors_path}")

    if manifest_path is not None:
        manifest = {
            "schema": RESULT_MANIFEST_SCHEMA,
            "schema_version": RESULT_MANIFEST_SCHEMA_VERSION,
            "measurement": "surface-distance-to-ground-truth-model",
            "measures": (
                "distance from each vertex of the reconstructed mesh to a "
                "ground-truth surface model: to the nearest model point, "
                "and to the tangent plane through it. The frames are "
                "aligned by one rigid motion fitted to raw depth, not to "
                "the reconstruction, and the same distances for raw depth "
                "are given as the floor of the measurement. Camera poses "
                "are the dataset's ground truth."
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
                "source_trajectory_sha256": trajectory_digest,
                "model_points": len(model_points),
            },
            "reconstruction": {
                "path": "sparse-block-streaming-fusion",
                "persisted": True,
                "voxel_size_m": volume.voxel_size_m,
                "truncation_m": volume.truncation_m,
                "frame_stride": volume.frame_stride,
                "fused_frames": volume.fused_observations,
                "active_blocks": volume.block_count,
                "planned_voxel_slots": volume.voxel_slots,
                "observed_voxel_slots": volume.observed_voxel_count,
                "contributions_applied": volume.contributions_applied,
                "contributions_evaluated": volume.contributions_evaluated,
                "maximum_weight": volume.maximum_weight,
            },
            "mesh": mesh,
            "registration": {
                "method": (
                    "reused from an earlier report"
                    if reused is not None
                    else "point-to-plane, raw depth to model, coarse "
                    "then fine"
                ),
                "fitted_to": (
                    "raw depth of the session named in reused_from"
                    if reused is not None
                    else "raw depth"
                ),
                "reused_from": reused,
                "session_was_imported": imported,
                "source_pose_consistency": pose_deviation,
                "initial_translation_m": (
                    None
                    if reused is not None
                    else list(arguments.initial_translation)
                ),
                "fit_frame_stride": stride,
                "fit_frames": 0 if reused is not None else len(fit_frames),
                "pixel_step": arguments.pixel_step,
                "fit_samples": fit_samples,
                "stages": [
                    {
                        "limit_m": stage.limit_m,
                        "iterations": stage.iterations,
                        "model_points": stage.model_points,
                        "matched_fraction": stage.matched_fraction,
                        "last_step_m": stage.last_step_m,
                        "last_step_rad": stage.last_step_rad,
                    }
                    for stage in stages
                ],
                "model_from_source": [
                    float(value) for value in transform.ravel()
                ],
                "rotation_degrees": angle,
            },
            "depth_reference": {
                "frames": len(check_frames),
                "model_normals_facing_camera_fraction": facing,
                **depth_summary,
            },
            "evaluation": mesh_summary,
            "vertex_errors": vertex_errors,
            "timings_seconds": {
                "model_load": round(load_seconds, 3),
                "registration": round(registration_seconds, 3),
                "evaluation": round(evaluation_seconds, 3),
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
        print(f"SURFACE ACCURACY REPORT FAILED: {error}", file=sys.stderr)
        sys.exit(2)
