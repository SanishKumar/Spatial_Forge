"""The surface-accuracy report, on scenes whose answer is known.

The report turns a mesh and a model into a handful of millimetre figures,
and nothing about a millimetre figure says whether it is right. Each piece
is therefore driven with a planted answer: distances from points placed a
known amount off a known plane, a registration that has to recover a rigid
motion it was not told, and a whole run on the synthetic room, whose walls
are where the generator put them.

The property the tool exists for gets its own test: a mesh that is rigidly
displaced must score worse. An evaluation that aligned the reconstruction
itself to the model would hide exactly that.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import shutil
import tempfile
import unittest
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from spatialforge import (
    allocate_empty_tsdf_blocks,
    fuse_tsdf_plan_streaming,
    load_tsdf_block_plan,
    write_tsdf_block_volume,
)
from spatialforge.replay import replay_session
from spatialforge.session_loader import load_scan_session
from spatialforge.tsdf_block_mesh import extract_tsdf_block_mesh
from spatialforge.tsdf_block_plan import plan_tsdf_blocks
from spatialforge.tum_importer import T_RIG_CAMERA, import_tum_dataset

from tests import room_fixture
from tests.heavy_fixtures import ROOM_PLAN_ARGUMENTS, shared_room_case
from tests.room_fixture import (
    BOX,
    CEILING_Z,
    FAR_WALL_X,
    FLOOR_Z,
    LEFT_WALL_Y,
    NOISE_SIGMA_M,
    RIGHT_WALL_Y,
)
from tools._frame_search import AMBIGUITY_RATIO
from tools._nearest import NearestPointIndex
from tools.surface_accuracy_report import (
    SurfaceDistances,
    main,
    measure,
    read_surface_model,
    register_point_to_plane,
    load_alignment,
    session_from_source,
    source_trajectory,
    source_trajectory_digest,
    summarise,
    trajectory_difference,
)

TEST_ROOT = Path(__file__).resolve().parent
FIXTURE = TEST_ROOT / "fixtures" / "minimal.vgsession"
TUM_FIXTURE = TEST_ROOT / "fixtures" / "tum" / "rgbd_dataset_freiburg1_tiny"
SPACING_M = 0.01


def rigid(axis, degrees: float, translation) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    angle = math.radians(degrees)
    cross = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    matrix = np.eye(4)
    matrix[:3, :3] = (
        np.eye(3)
        + math.sin(angle) * cross
        + (1.0 - math.cos(angle)) * (cross @ cross)
    )
    matrix[:3, 3] = translation
    return matrix


def apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def lattice(axis: int, value: float, first, second, normal):
    """Points every ``SPACING_M`` on an axis-aligned rectangle."""

    a = np.arange(first[0], first[1] + SPACING_M / 2, SPACING_M)
    b = np.arange(second[0], second[1] + SPACING_M / 2, SPACING_M)
    grid_a, grid_b = np.meshgrid(a, b, indexing="ij")
    points = np.empty((grid_a.size, 3))
    others = [index for index in range(3) if index != axis]
    points[:, axis] = value
    points[:, others[0]] = grid_a.ravel()
    points[:, others[1]] = grid_b.ravel()
    normals = np.tile(np.asarray(normal, dtype=np.float64), (len(points), 1))
    return points, normals


def write_model(
    path: Path,
    points: np.ndarray,
    normals: np.ndarray,
    *,
    with_normals: bool = True,
    form: str = "binary_little_endian",
) -> None:
    """A point model in the layout ICL-NUIM publishes: colour in between."""

    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    if with_normals:
        fields += [("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4")]
    record = np.zeros(len(points), dtype=fields)
    for column, name in enumerate(("x", "y", "z")):
        record[name] = points[:, column]
    if with_normals:
        for column, name in enumerate(("nx", "ny", "nz")):
            record[name] = normals[:, column]
    kinds = {"<f4": "float", "u1": "uchar"}
    header = "ply\nformat " + form + " 1.0\n"
    header += f"element vertex {len(points)}\n"
    header += "".join(
        f"property {kinds[kind]} {name}\n" for name, kind in fields
    )
    header += "end_header\n"
    path.write_bytes(header.encode("ascii") + record.tobytes())


class ModelReaderTests(unittest.TestCase):
    def test_reads_positions_and_normals_past_other_properties(self) -> None:
        points, normals = lattice(2, 0.25, (0.0, 0.05), (0.0, 0.03), (0, 0, 1))
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            path = Path(temporary) / "model.ply"
            write_model(path, points, normals)
            read_points, read_normals = read_surface_model(path)

        self.assertEqual(read_points.dtype, np.float64)
        np.testing.assert_array_equal(
            read_points, points.astype(np.float32).astype(np.float64)
        )
        np.testing.assert_array_equal(read_normals, normals)

    def test_unusable_models_are_refused(self) -> None:
        points, normals = lattice(2, 0.0, (0.0, 0.02), (0.0, 0.02), (0, 0, 1))
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            root = Path(temporary)

            def refused(name: str, message: str, **arguments) -> None:
                path = root / name
                write_model(path, points, normals, **arguments)
                with self.assertRaisesRegex(SystemExit, message):
                    read_surface_model(path)

            refused("flat.ply", "missing nx, ny, nz", with_normals=False)
            refused("ascii.ply", "binary little-endian", form="ascii")
            path = root / "long.ply"
            write_model(path, points, normals * 1.5)
            with self.assertRaisesRegex(SystemExit, "not unit length"):
                read_surface_model(path)
            bad = points.copy()
            bad[0, 0] = np.nan
            write_model(root / "nan.ply", bad, normals)
            with self.assertRaisesRegex(SystemExit, "non-finite"):
                read_surface_model(root / "nan.ply")
            good = root / "good.ply"
            write_model(good, points, normals)
            encoded = good.read_bytes()
            (root / "short.ply").write_bytes(encoded[:-1])
            with self.assertRaisesRegex(SystemExit, "shorter than"):
                read_surface_model(root / "short.ply")
            (root / "text.ply").write_bytes(b"not a ply\n")
            with self.assertRaisesRegex(SystemExit, "not a PLY"):
                read_surface_model(root / "text.ply")
            (root / "endless.ply").write_bytes(b"ply\nformat ascii 1.0\n")
            with self.assertRaisesRegex(SystemExit, "does not end"):
                read_surface_model(root / "endless.ply")
            # Vertex data is read from the end of the header, so an element
            # ahead of it would be misread as vertices.
            (root / "faces-first.ply").write_bytes(
                encoded.replace(
                    b"element vertex", b"element face 0\nelement vertex"
                )
            )
            with self.assertRaisesRegex(SystemExit, "must come first"):
                read_surface_model(root / "faces-first.ply")
            (root / "list.ply").write_bytes(
                encoded.replace(
                    b"property float x", b"property list uchar int x"
                )
            )
            with self.assertRaisesRegex(SystemExit, "unsupported vertex"):
                read_surface_model(root / "list.ply")


class PlantedDistanceTests(unittest.TestCase):
    def test_distances_from_a_known_plane_are_exact(self) -> None:
        model, normals = lattice(2, 0.0, (0.0, 1.0), (0.0, 1.0), (0, 0, 1))
        index = NearestPointIndex(model, cell_m=0.02)
        # Above a lattice point, the nearest model point is straight down.
        # Above the middle of a lattice square it is a corner, half a
        # diagonal away sideways; the plane distance does not care.
        above_point = np.array([[0.5, 0.5, 0.0078125]])
        above_centre = np.array([[0.505, 0.505, -0.015625]])
        result = measure(
            np.concatenate([above_point, above_centre]),
            model,
            normals.astype(np.float32),
            index,
            limit_m=0.1,
        )

        self.assertEqual(result.total, 2)
        self.assertTrue(np.all(result.matched))
        self.assertAlmostEqual(result.nearest_m[0], 0.0078125, places=12)
        self.assertAlmostEqual(
            result.nearest_m[1],
            math.sqrt(0.015625**2 + 2 * 0.005**2),
            places=12,
        )
        np.testing.assert_allclose(
            result.signed_plane_m, [0.0078125, -0.015625], atol=1e-15
        )

    def test_a_point_with_no_model_nearby_is_left_out_not_guessed(
        self,
    ) -> None:
        model, normals = lattice(2, 0.0, (0.0, 0.2), (0.0, 0.2), (0, 0, 1))
        result = measure(
            np.array([[0.1, 0.1, 0.004], [0.1, 0.1, 3.0]]),
            model,
            normals.astype(np.float32),
            NearestPointIndex(model, cell_m=0.02),
            limit_m=0.05,
        )
        self.assertEqual(result.matched.tolist(), [True, False])
        self.assertEqual(len(result.nearest_m), 1)

    def test_statistics_are_the_ones_computed_by_hand(self) -> None:
        nearest = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float64) / 1000
        signed = np.array([1, -2, 3, -4, 5, -6, 7, -8], dtype=np.float64) / 1000
        # Ten points, two of them farther than the limit.
        distances = SurfaceDistances(
            total=10,
            matched=np.array([True] * 8 + [False] * 2),
            nearest_m=nearest,
            signed_plane_m=signed,
            model_index=np.zeros(8, dtype=np.int64),
        )
        summary = summarise(distances, limit_m=0.05, signed=True)

        self.assertEqual(summary["points"], 10)
        self.assertEqual(summary["matched"], 8)
        self.assertEqual(summary["beyond_limit"], 2)
        block = summary["nearest_point"]
        self.assertAlmostEqual(block["mean_mm"], 4.5)
        self.assertAlmostEqual(block["rms_mm"], math.sqrt(204 / 8))
        self.assertAlmostEqual(block["std_mm"], math.sqrt(204 / 8 - 4.5**2))
        # Ranked over all ten: the fifth smallest is the median, and the
        # ninth and tenth were never found, so nothing above the 80th
        # percentile can be stated.
        self.assertAlmostEqual(block["median_mm"], 5.0)
        self.assertIsNone(block["p90_mm"])
        self.assertIsNone(block["p95_mm"])
        self.assertIsNone(block["p99_mm"])
        self.assertAlmostEqual(block["max_matched_mm"], 8.0)
        plane = summary["point_to_plane"]
        self.assertAlmostEqual(plane["mean_mm"], 4.5)
        self.assertAlmostEqual(plane["mean_signed_mm"], -0.5)
        self.assertEqual(
            summary["within_fraction"],
            {"5mm": 0.5, "10mm": 0.8, "20mm": 0.8},
        )
        self.assertIsNone(
            summarise(distances, limit_m=0.05, signed=False)[
                "point_to_plane"
            ]["mean_signed_mm"]
        )

    def test_nothing_matched_is_an_error_not_a_set_of_zeros(self) -> None:
        empty = SurfaceDistances(
            total=3,
            matched=np.zeros(3, dtype=bool),
            nearest_m=np.empty(0),
            signed_plane_m=np.empty(0),
            model_index=np.empty(0, dtype=np.int64),
        )
        with self.assertRaisesRegex(SystemExit, "not aligned"):
            summarise(empty, limit_m=0.05, signed=True)


def corner_model():
    """Three mutually perpendicular unit squares meeting at the origin."""

    parts = [
        lattice(0, 0.0, (0.0, 1.0), (0.0, 1.0), (1, 0, 0)),
        lattice(1, 0.0, (0.0, 1.0), (0.0, 1.0), (0, 1, 0)),
        lattice(2, 0.0, (0.0, 1.0), (0.0, 1.0), (0, 0, 1)),
    ]
    return (
        np.concatenate([points for points, _ in parts]),
        np.concatenate([normals for _, normals in parts]).astype(np.float32),
    )


def corner_samples(seed: int, count: int) -> np.ndarray:
    """Points exactly on the three squares, clear of their shared edges."""

    rng = np.random.default_rng(seed)
    samples = rng.uniform(0.1, 0.9, size=(count, 3))
    samples[np.arange(count), rng.integers(0, 3, size=count)] = 0.0
    return samples


class RegistrationTests(unittest.TestCase):
    def test_recovers_a_rigid_motion_it_was_not_told(self) -> None:
        model, normals = corner_model()
        planted = rigid((0.3, 1.0, -0.2), 1.3, (0.031, -0.024, 0.017))
        # What a sensor would report: the surface, in a frame the planted
        # motion carries onto the model.
        observed = apply(np.linalg.inv(planted), corner_samples(21, 4_000))

        start = np.eye(4)
        start[:3, 3] = (0.02, -0.01, 0.03)
        coarse, _ = register_point_to_plane(
            observed,
            model[::4],
            normals[::4],
            NearestPointIndex(model[::4], cell_m=0.08),
            start,
            limit_m=0.08,
            iterations=12,
        )
        fitted, stage = register_point_to_plane(
            observed,
            model,
            normals,
            NearestPointIndex(model, cell_m=0.02),
            coarse,
            limit_m=0.02,
            iterations=6,
        )

        np.testing.assert_allclose(fitted, planted, atol=1e-10)
        self.assertEqual(stage.matched_fraction, 1.0)
        self.assertLess(stage.last_step_m, 1e-10)
        self.assertLess(stage.last_step_rad, 1e-10)
        self.assertAlmostEqual(float(np.linalg.det(fitted[:3, :3])), 1.0)

    def test_a_single_plane_cannot_fix_a_rigid_motion(self) -> None:
        model, normals = lattice(2, 0.0, (0.0, 1.0), (0.0, 1.0), (0, 0, 1))
        observed = np.random.default_rng(22).uniform(0.1, 0.9, size=(500, 3))
        observed[:, 2] = 0.001
        with self.assertRaisesRegex(SystemExit, "under-constrained"):
            register_point_to_plane(
                observed,
                model,
                normals.astype(np.float32),
                NearestPointIndex(model, cell_m=0.02),
                np.eye(4),
                limit_m=0.02,
                iterations=3,
            )

    def test_depth_nowhere_near_the_model_is_an_error(self) -> None:
        model, normals = corner_model()
        far = np.eye(4)
        far[:3, 3] = (5.0, 5.0, 5.0)
        with self.assertRaisesRegex(SystemExit, "too far off"):
            register_point_to_plane(
                corner_samples(23, 200),
                model,
                normals,
                NearestPointIndex(model, cell_m=0.08),
                far,
                limit_m=0.08,
                iterations=3,
            )


class SourceFrameTests(unittest.TestCase):
    def test_an_imported_session_remembers_its_source_frame(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            output = Path(temporary) / "tiny.vgsession"
            import_tum_dataset(TUM_FIXTURE, output)
            replay = replay_session(load_scan_session(output))
            to_session, deviation, imported = session_from_source(
                replay.observations
            )

        self.assertTrue(imported)
        self.assertLess(deviation, 1e-12)
        # The fixture's first pose is a pure translation to (1, 2, 3), and
        # the importer anchors the session there in rig axes.
        first = np.eye(4)
        first[:3, 3] = (1.0, 2.0, 3.0)
        expected = np.array(T_RIG_CAMERA).reshape((4, 4)) @ np.linalg.inv(
            first
        )
        np.testing.assert_allclose(to_session, expected, atol=1e-12)

    def test_a_session_that_was_not_imported_is_its_own_frame(self) -> None:
        replay = replay_session(load_scan_session(FIXTURE))
        to_session, deviation, imported = session_from_source(
            replay.observations
        )
        self.assertFalse(imported)
        self.assertEqual(deviation, 0.0)
        np.testing.assert_array_equal(to_session, np.eye(4))

    def test_poses_that_disagree_about_the_frame_are_measured(self) -> None:
        def observation(translation, session_x):
            session = np.eye(4)
            session[0, 3] = session_x
            return SimpleNamespace(
                pose=SimpleNamespace(
                    data={
                        "source_translation_m": translation,
                        "source_quaternion_xyzw": (0.0, 0.0, 0.0, 1.0),
                        "T_world_camera": tuple(session.ravel()),
                    }
                )
            )

        consistent = [
            observation((1.0, 0.0, 0.0), 0.0),
            observation((1.5, 0.0, 0.0), 0.5),
            SimpleNamespace(pose=None),
        ]
        self.assertLess(session_from_source(consistent)[1], 1e-15)
        broken = [*consistent, observation((2.0, 0.0, 0.0), 1.25)]
        self.assertAlmostEqual(session_from_source(broken)[1], 0.25)


def room_model() -> tuple[np.ndarray, np.ndarray]:
    """The planes the room generator drew, sampled every centimetre."""

    x_range = (-0.5, FAR_WALL_X)
    y_range = (RIGHT_WALL_Y, LEFT_WALL_Y)
    z_range = (FLOOR_Z, CEILING_Z)
    x0, x1, y0, y1, z0, z1 = BOX
    parts = [
        lattice(0, FAR_WALL_X, y_range, z_range, (-1, 0, 0)),
        lattice(1, LEFT_WALL_Y, x_range, z_range, (0, -1, 0)),
        lattice(1, RIGHT_WALL_Y, x_range, z_range, (0, 1, 0)),
        lattice(2, FLOOR_Z, x_range, y_range, (0, 0, 1)),
        lattice(2, CEILING_Z, x_range, y_range, (0, 0, -1)),
        lattice(0, x0, (y0, y1), (z0, z1), (-1, 0, 0)),
        lattice(0, x1, (y0, y1), (z0, z1), (1, 0, 0)),
        lattice(1, y0, (x0, x1), (z0, z1), (0, -1, 0)),
        lattice(1, y1, (x0, x1), (z0, z1), (0, 1, 0)),
        lattice(2, z1, (x0, x1), (y0, y1), (0, 0, 1)),
    ]
    return (
        np.concatenate([points for points, _ in parts]),
        np.concatenate([normals for _, normals in parts]),
    )


# Where the model sits relative to the scan. The report is told only a
# rough translation and has to find the rest.
MODEL_FROM_SESSION = rigid((0.1, -0.2, 1.0), 1.1, (0.31, -0.22, 0.13))
ROUGH_TRANSLATION = ("0.33", "-0.20", "0.11")


@contextmanager
def source_state(commit: str | None, clean: bool | None):
    with patch(
        "tools.surface_accuracy_report.source_state",
        return_value=(commit, clean),
    ):
        yield


ROOM: SimpleNamespace | None = None


def setUpModule() -> None:
    """One room for the module: scan, volume, mesh and displaced model."""

    global ROOM
    root = Path(tempfile.mkdtemp(dir=TEST_ROOT))
    unittest.addModuleCleanup(shutil.rmtree, root, True)
    case = shared_room_case()
    storage = allocate_empty_tsdf_blocks(case.plan, case.session)
    receipt = fuse_tsdf_plan_streaming(storage, case.session)
    volume = root / "room.sftvol"
    write_tsdf_block_volume(storage, receipt, volume)
    mesh = root / "room.ply"
    extract_tsdf_block_mesh(volume, mesh)
    points, normals = room_model()
    model = root / "model.ply"
    write_model(
        model,
        apply(MODEL_FROM_SESSION, points),
        normals @ MODEL_FROM_SESSION[:3, :3].T,
    )
    ROOM = SimpleNamespace(
        root=root,
        session=case.session_path,
        volume=volume,
        mesh=mesh,
        model=model,
    )


def report_arguments(
    *,
    translation=ROUGH_TRANSLATION,
    **replaced: Path,
) -> list[str]:
    values = {
        "session": ROOM.session,
        "volume": ROOM.volume,
        "mesh": ROOM.mesh,
        "model": ROOM.model,
    }
    values.update(replaced)
    return [
        str(values["session"]),
        str(values["volume"]),
        str(values["mesh"]),
        str(values["model"]),
        *(
            []
            if translation is None
            else ["--initial-translation", *translation]
        ),
        "--fit-frame-stride",
        "2",
        "--pixel-step",
        "2",
    ]


def run_report(name: str, *extra: str, **arguments) -> dict:
    manifest = ROOM.root / f"{name}.json"
    with source_state("a" * 40, True), redirect_stdout(io.StringIO()):
        exit_code = main(
            [
                *report_arguments(**arguments),
                "--manifest-out",
                str(manifest),
                *extra,
            ]
        )
    assert exit_code == 0
    encoded = manifest.read_bytes()
    # Written as bytes: the same manifest on every platform.
    assert b"\r" not in encoded
    return json.loads(encoded)


_BASELINE: dict | None = None


def baseline() -> dict:
    """The report on the untouched room, run once for the module."""

    global _BASELINE
    if _BASELINE is None:
        _BASELINE = run_report(
            "baseline",
            "--errors-out",
            str(ROOM.root / "baseline-errors.npy"),
        )
    return _BASELINE


class RoomReportTests(unittest.TestCase):

    def test_finds_the_alignment_and_measures_against_it(self) -> None:
        manifest = baseline()

        self.assertEqual(
            manifest["measurement"], "surface-distance-to-ground-truth-model"
        )
        self.assertEqual(manifest["source_commit"], "a" * 40)
        self.assertTrue(manifest["source_worktree_clean"])
        registration = manifest["registration"]
        self.assertEqual(registration["fitted_to"], "raw depth")
        self.assertFalse(registration["session_was_imported"])
        self.assertEqual(registration["fit_frames"], 10)
        fitted = np.array(registration["model_from_source"]).reshape((4, 4))
        # Not told the rotation, and told the translation only to within
        # about three centimetres.
        self.assertLess(
            float(np.abs(fitted[:3, 3] - MODEL_FROM_SESSION[:3, 3]).max()),
            0.001,
        )
        self.assertLess(
            float(np.abs(fitted[:3, :3] - MODEL_FROM_SESSION[:3, :3]).max()),
            0.001,
        )
        self.assertAlmostEqual(registration["rotation_degrees"], 1.1, places=1)

        # Raw depth carries the generator's 4 mm noise along the ray and
        # nothing else, so that is what its plane distance has to show.
        depth = manifest["depth_reference"]
        self.assertEqual(depth["frames"], 10)
        self.assertEqual(depth["beyond_limit"], 0)
        self.assertGreater(depth["model_normals_facing_camera_fraction"], 0.99)
        noise_mm = 1000 * NOISE_SIGMA_M
        self.assertLess(depth["point_to_plane"]["rms_mm"], 1.2 * noise_mm)
        self.assertGreater(depth["point_to_plane"]["rms_mm"], 0.3 * noise_mm)
        self.assertLess(abs(depth["point_to_plane"]["mean_signed_mm"]), 1.0)

        # Twenty noisy frames fused at 40 mm voxels land within a tenth
        # of a voxel of the true walls, at the median.
        evaluation = manifest["evaluation"]
        self.assertEqual(evaluation["points"], manifest["mesh"]["vertices"])
        self.assertLess(evaluation["point_to_plane"]["median_mm"], 4.0)
        self.assertGreater(evaluation["within_fraction"]["20mm"], 0.95)
        # The model is sampled every centimetre, so the distance to its
        # nearest point cannot be smaller than the distance to its plane.
        self.assertGreaterEqual(
            evaluation["nearest_point"]["median_mm"],
            evaluation["point_to_plane"]["median_mm"],
        )

        inputs = manifest["inputs"]
        self.assertEqual(
            manifest["mesh"]["source_volume_sha256"], inputs["volume_sha256"]
        )
        self.assertEqual(len(inputs["model_sha256"]), 64)
        self.assertEqual(
            inputs["model_points"], len(room_model()[0])
        )

    def test_per_vertex_errors_are_the_ones_that_were_summarised(
        self,
    ) -> None:
        manifest = baseline()
        path = ROOM.root / "baseline-errors.npy"
        errors = np.load(path, allow_pickle=False)
        evaluation = manifest["evaluation"]

        self.assertEqual(errors.dtype, np.float32)
        self.assertEqual(errors.shape, (manifest["mesh"]["vertices"],))
        self.assertEqual(
            int(np.count_nonzero(np.isnan(errors))),
            evaluation["beyond_limit"],
        )
        measured = np.sort(errors[np.isfinite(errors)].astype(np.float64))
        self.assertTrue(bool(np.all(measured >= 0.0)))
        # The same nearest-rank median the summary states, to the
        # precision float32 keeps.
        median = measured[math.ceil(0.5 * len(errors)) - 1]
        self.assertAlmostEqual(
            1000 * median,
            evaluation["point_to_plane"]["median_mm"],
            places=3,
        )
        self.assertAlmostEqual(
            1000 * float(measured.mean()),
            evaluation["point_to_plane"]["mean_mm"],
            places=3,
        )
        recorded = manifest["vertex_errors"]
        self.assertEqual(
            recorded["sha256"], hashlib.sha256(path.read_bytes()).hexdigest()
        )
        self.assertIsNone(
            run_report("no_errors_requested")["vertex_errors"]
        )

    def test_a_rigidly_displaced_mesh_scores_worse(self) -> None:
        # The check an alignment fitted to the reconstruction would fail:
        # move every vertex 25 mm and the surface is wrong by up to 25 mm,
        # yet a best-fit alignment of that mesh would move it straight back.
        encoded = ROOM.mesh.read_bytes()
        end = encoded.index(b"end_header\n") + len(b"end_header\n")
        count = int(
            next(
                line.split()[2]
                for line in encoded[:end].split(b"\n")
                if line.startswith(b"element vertex")
            )
        )
        vertices = np.frombuffer(
            encoded, dtype="<f8", count=3 * count, offset=end
        ).reshape((count, 3)) + np.array([0.025, 0.0, 0.0])
        displaced = ROOM.root / "displaced.ply"
        displaced.write_bytes(
            encoded[:end] + vertices.tobytes() + encoded[end + 24 * count:]
        )

        honest = baseline()["evaluation"]["point_to_plane"]
        moved = run_report("moved", mesh=displaced)["evaluation"][
            "point_to_plane"
        ]
        # Walls facing along x take the whole displacement; walls it
        # slides along take none. The far tail is the first kind.
        self.assertGreater(moved["p95_mm"], honest["p95_mm"] + 15.0)
        self.assertGreater(moved["rms_mm"], honest["rms_mm"] + 5.0)

    def test_the_answer_does_not_depend_on_the_rough_guess(self) -> None:
        first = baseline()
        second = run_report(
            "second_guess", translation=("0.29", "-0.25", "0.16")
        )
        np.testing.assert_allclose(
            second["registration"]["model_from_source"],
            first["registration"]["model_from_source"],
            atol=1e-6,
        )
        self.assertAlmostEqual(
            second["evaluation"]["point_to_plane"]["median_mm"],
            first["evaluation"]["point_to_plane"]["median_mm"],
            places=3,
        )


LEVEL_POSE = room_fixture._pose_for


def tipped_pose(frame: int) -> tuple[float, ...]:
    """The room fixture's arc, with the camera tipped 25 degrees down."""

    level = LEVEL_POSE(frame)
    tip = math.radians(25.0)
    pose = list(level)
    for row in range(3):
        down, forward = level[4 * row + 1], level[4 * row + 2]
        pose[4 * row + 1] = math.cos(tip) * down - math.sin(tip) * forward
        pose[4 * row + 2] = math.cos(tip) * forward + math.sin(tip) * down
    return tuple(pose)


_TIPPED: SimpleNamespace | None = None


def tipped_room() -> SimpleNamespace:
    """The same room scanned looking down: scan, volume and mesh.

    The fixture's camera is level, and the floor and ceiling stay out of
    all but the last rows of its image. A scan that is to be placed by
    its planes has to see planes facing three ways, so this one looks at
    the floor as well as the walls.
    """

    global _TIPPED
    if _TIPPED is None:
        session_path = ROOM.root / "tipped.vgsession"
        with patch("tests.room_fixture._pose_for", tipped_pose):
            room_fixture._write_session(session_path)
        session = load_scan_session(session_path)
        plan_path = ROOM.root / "tipped.sftplan"
        plan_tsdf_blocks(
            session, plan_path, frame_stride=1, **ROOM_PLAN_ARGUMENTS
        )
        storage = allocate_empty_tsdf_blocks(
            load_tsdf_block_plan(plan_path), session
        )
        receipt = fuse_tsdf_plan_streaming(storage, session)
        volume = ROOM.root / "tipped.sftvol"
        write_tsdf_block_volume(storage, receipt, volume)
        mesh = ROOM.root / "tipped.ply"
        extract_tsdf_block_mesh(volume, mesh)
        _TIPPED = SimpleNamespace(
            session=session_path, volume=volume, mesh=mesh
        )
    return _TIPPED


class FoundFrameTests(unittest.TestCase):
    """The report with no starting guess, placed by the room's planes."""

    def tipped(self, name: str, *extra: str, **arguments) -> dict:
        room = tipped_room()
        return run_report(
            name,
            *extra,
            session=room.session,
            volume=room.volume,
            mesh=room.mesh,
            **arguments,
        )

    def test_the_frame_it_finds_is_the_frame_a_guess_leads_to(self) -> None:
        guessed = self.tipped("tipped_guessed")
        searched = self.tipped(
            "tipped_searched", "--find-frame", translation=None
        )

        registration = searched["registration"]
        self.assertIsNone(registration["initial_translation_m"])
        self.assertEqual(registration["fitted_to"], "raw depth")
        found = registration["frame_search"]
        # One placement, and the scan can tell. Nearly all of the depth
        # lands on the model. The next best still puts five sixths of it
        # on some wall facing the right way, in a room with one box in
        # it, and that is inside what the search will call a difference.
        self.assertGreater(found["agreement"], 0.95)
        self.assertLess(
            found["runner_up_agreement"],
            AMBIGUITY_RATIO * found["agreement"],
        )
        self.assertGreater(found["runner_up_agreement"], 0.7)
        start = np.array(found["start_model_from_source"]).reshape((4, 4))
        # The search is a guess, good to centimetres and to a degree.
        self.assertLess(
            float(np.abs(start[:3, 3] - MODEL_FROM_SESSION[:3, 3]).max()),
            0.05,
        )
        self.assertLess(
            float(np.abs(start[:3, :3] - MODEL_FROM_SESSION[:3, :3]).max()),
            0.03,
        )
        # The fit from it settles where the fit from a given guess does.
        np.testing.assert_allclose(
            registration["model_from_source"],
            guessed["registration"]["model_from_source"],
            atol=1e-6,
        )
        for figure in ("median_mm", "rms_mm", "p95_mm"):
            self.assertAlmostEqual(
                searched["evaluation"]["point_to_plane"][figure],
                guessed["evaluation"]["point_to_plane"][figure],
                places=3,
            )
        # A run that was given its guess says so, and searched nothing.
        self.assertIsNone(guessed["registration"]["frame_search"])
        self.assertEqual(
            guessed["registration"]["initial_translation_m"],
            [float(value) for value in ROUGH_TRANSLATION],
        )

    def test_a_model_turned_and_moved_any_way_is_found(self) -> None:
        # The same room, published in a frame that has nothing to do with
        # the scan's: 131 degrees round a tilted axis and five metres off.
        turn = rigid((0.4, -1.0, 0.7), 131.0, (4.0, -3.0, 2.5))
        placed = turn @ MODEL_FROM_SESSION
        points, normals = room_model()
        turned = ROOM.root / "turned.ply"
        write_model(turned, apply(placed, points), normals @ placed[:3, :3].T)

        searched = self.tipped(
            "tipped_turned", "--find-frame", translation=None, model=turned
        )

        guessed = self.tipped("tipped_guessed_again")
        fitted = np.array(
            searched["registration"]["model_from_source"]
        ).reshape((4, 4))
        expected = turn @ np.array(
            guessed["registration"]["model_from_source"]
        ).reshape((4, 4))
        # The model is stored in single precision, and the turned one
        # rounds differently: agreement to a hundredth of a millimetre.
        np.testing.assert_allclose(fitted, expected, atol=1e-5)
        self.assertAlmostEqual(
            searched["evaluation"]["point_to_plane"]["median_mm"],
            guessed["evaluation"]["point_to_plane"]["median_mm"],
            places=2,
        )
        self.assertEqual(
            searched["evaluation"]["beyond_limit"],
            guessed["evaluation"]["beyond_limit"],
        )
        # No translation alone gets there. The true one is the best a
        # caller could give, and from it the fit is refused.
        room = tipped_room()
        with (
            self.assertRaises(SystemExit) as raised,
            redirect_stdout(io.StringIO()),
        ):
            main(
                report_arguments(
                    translation=[f"{value:.3f}" for value in placed[:3, 3]],
                    session=room.session,
                    volume=room.volume,
                    mesh=room.mesh,
                    model=turned,
                )
            )
        self.assertRegex(
            str(raised.exception), "did not settle|too far off|constrained"
        )

    def test_a_scan_that_sees_walls_facing_two_ways_is_refused(self) -> None:
        # The level camera: one wall ahead, one to each side, and no
        # floor or ceiling a normal can be taken on.
        with (
            self.assertRaises(SystemExit) as raised,
            redirect_stdout(io.StringIO()),
        ):
            main([*report_arguments(translation=None), "--find-frame"])
        self.assertIn("three perpendicular", str(raised.exception))

    def test_a_guess_and_a_search_are_exclusive(self) -> None:
        baseline()
        for arguments in (
            [*report_arguments(), "--find-frame"],
            [
                *report_arguments(translation=None),
                "--find-frame",
                "--alignment",
                str(ROOM.root / "baseline.json"),
            ],
        ):
            with self.subTest(arguments=arguments[4:]):
                with (
                    self.assertRaises(SystemExit) as raised,
                    redirect_stderr(io.StringIO()),
                ):
                    main(arguments)
                self.assertEqual(raised.exception.code, 2)


class ReusedAlignmentTests(unittest.TestCase):
    """Judging a scan in the frame an earlier report fitted."""

    def reuse(self, name: str, alignment: Path, **replaced) -> dict:
        return run_report(
            name,
            "--alignment",
            str(alignment),
            translation=None,
            **replaced,
        )

    def edited_baseline(self, name: str, edit) -> Path:
        baseline()
        manifest = json.loads((ROOM.root / "baseline.json").read_bytes())
        edit(manifest)
        path = ROOM.root / f"{name}.json"
        path.write_bytes(json.dumps(manifest).encode("utf-8"))
        return path

    def assert_refused(self, message: str, alignment: Path, **replaced) -> None:
        with (
            self.assertRaises(SystemExit) as raised,
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
        ):
            main(
                [
                    *report_arguments(translation=None, **replaced),
                    "--alignment",
                    str(alignment),
                ]
            )
        self.assertIn(message, str(raised.exception))

    def test_the_trajectory_is_recorded_by_digest(self) -> None:
        digest = baseline()["inputs"]["source_trajectory_sha256"]
        self.assertEqual(len(digest), 64)
        replay = replay_session(load_scan_session(ROOM.session))
        self.assertEqual(
            digest, source_trajectory_digest(replay.observations)
        )
        # A different trajectory, a different digest; a frame with no pose
        # is not part of either.
        self.assertNotEqual(
            digest, source_trajectory_digest(replay.observations[1:])
        )
        self.assertEqual(
            source_trajectory_digest(
                [*replay.observations, SimpleNamespace(pose=None)]
            ),
            digest,
        )

    def test_a_reused_frame_gives_the_numbers_of_the_run_that_fitted_it(
        self,
    ) -> None:
        fitted = baseline()
        # Reuse must not fit anything: registration is made to fail if it
        # is so much as called.
        with patch(
            "tools.surface_accuracy_report.register_point_to_plane",
            side_effect=AssertionError("fitted again"),
        ):
            reused = self.reuse("reused", ROOM.root / "baseline.json")

        registration = reused["registration"]
        self.assertEqual(registration["method"], "reused from an earlier report")
        self.assertEqual(
            registration["model_from_source"],
            fitted["registration"]["model_from_source"],
        )
        self.assertEqual(registration["stages"], [])
        self.assertEqual(registration["fit_frames"], 0)
        self.assertIsNone(registration["initial_translation_m"])
        self.assertEqual(
            registration["reused_from"]["sha256"],
            hashlib.sha256(
                (ROOM.root / "baseline.json").read_bytes()
            ).hexdigest(),
        )
        self.assertEqual(
            registration["reused_from"]["replay_digest_sha256"],
            fitted["inputs"]["replay_digest_sha256"],
        )
        self.assertEqual(reused["evaluation"], fitted["evaluation"])

        def measured(reference: dict) -> dict:
            return {
                key: value
                for key, value in reference.items()
                if not key.startswith("normals_trusted")
            }

        self.assertEqual(
            measured(reused["depth_reference"]),
            measured(fitted["depth_reference"]),
        )
        self.assertIsNone(fitted["registration"]["reused_from"])

    def test_which_way_normals_face_comes_with_the_reused_alignment(
        self,
    ) -> None:
        fitted = baseline()
        own = fitted["depth_reference"]["model_normals_facing_camera_fraction"]
        self.assertEqual(
            fitted["depth_reference"]["normals_trusted_from"],
            "this session's depth",
        )
        self.assertEqual(
            fitted["depth_reference"]["normals_trusted_at_fraction"], own
        )

        def facing(value):
            def edit(manifest: dict) -> None:
                manifest["depth_reference"][
                    "model_normals_facing_camera_fraction"
                ] = value

            return edit

        # A bar this session's own depth would not clear, and an earlier
        # report that does: the earlier report decides, so signed figures
        # are stated.
        self.assertLess(own, 0.9995)
        with patch("tools.surface_accuracy_report.ORIENTED_FRACTION", 0.9995):
            trusted = self.reuse(
                "trusted", self.edited_baseline("faces", facing(1.0))
            )
            alone = run_report("alone")
        self.assertIsNone(
            alone["evaluation"]["point_to_plane"]["mean_signed_mm"]
        )
        self.assertEqual(
            trusted["evaluation"]["point_to_plane"]["mean_signed_mm"],
            fitted["evaluation"]["point_to_plane"]["mean_signed_mm"],
        )
        reference = trusted["depth_reference"]
        self.assertEqual(
            reference["normals_trusted_from"],
            "the report the alignment came from",
        )
        self.assertEqual(reference["normals_trusted_at_fraction"], 1.0)
        # This session's own figure is still measured and still recorded.
        self.assertEqual(reference["model_normals_facing_camera_fraction"], own)

        # An earlier report that did not trust them is not overruled.
        doubted = self.reuse(
            "doubted", self.edited_baseline("faces-away", facing(0.5))
        )
        self.assertIsNone(
            doubted["evaluation"]["point_to_plane"]["mean_signed_mm"]
        )
        self.assertIsNone(
            doubted["depth_reference"]["point_to_plane"]["mean_signed_mm"]
        )

        # One that never recorded the figure leaves it to this session.
        silent = self.reuse(
            "silent", self.edited_baseline("no-facing", facing(None))
        )
        self.assertEqual(
            silent["depth_reference"]["normals_trusted_from"],
            "this session's depth",
        )
        self.assertIsNotNone(
            silent["evaluation"]["point_to_plane"]["mean_signed_mm"]
        )

    def test_an_alignment_is_only_reused_where_it_applies(self) -> None:
        other_trajectory = self.edited_baseline(
            "other-trajectory",
            lambda m: m["inputs"].__setitem__(
                "source_trajectory_sha256", "0" * 64
            ),
        )
        self.assert_refused("different source trajectory", other_trajectory)

        points, normals = room_model()
        moved = ROOM.root / "other-model.ply"
        write_model(
            moved,
            apply(MODEL_FROM_SESSION, points) + 0.5,
            normals @ MODEL_FROM_SESSION[:3, :3].T,
        )
        self.assert_refused(
            "different ground-truth model",
            ROOM.root / "baseline.json",
            model=moved,
        )

        def skew(manifest: dict) -> None:
            manifest["registration"]["model_from_source"][0] = 1.5

        self.assert_refused(
            "is not rigid", self.edited_baseline("skewed", skew)
        )

        def mirror(manifest: dict) -> None:
            values = manifest["registration"]["model_from_source"]
            for index in (0, 4, 8):
                values[index] = -values[index]

        self.assert_refused(
            "is not rigid", self.edited_baseline("mirrored", mirror)
        )
        for name, edit in (
            ("no-digest", lambda m: m["inputs"].pop("source_trajectory_sha256")),
            ("no-transform", lambda m: m["registration"].pop("model_from_source")),
            ("short", lambda m: m["registration"]["model_from_source"].pop()),
            ("held-out", lambda m: m.__setitem__("measurement", "held-out-tsdf-residual")),
        ):
            with self.subTest(name=name):
                self.assert_refused(
                    "not a surface-accuracy manifest",
                    self.edited_baseline(name, edit),
                )
        garbage = ROOM.root / "garbage.json"
        garbage.write_bytes(b"not json")
        self.assert_refused("not a surface-accuracy manifest", garbage)

    def test_the_same_motion_rounded_differently_is_matched_pose_by_pose(
        self,
    ) -> None:
        fitted = baseline()
        manifest = ROOM.root / "baseline.json"
        model = fitted["inputs"]["model_sha256"]
        reference = source_trajectory(
            replay_session(load_scan_session(ROOM.session)).observations
        )
        # The room's poses are 4x4 matrices, row-major: index 3 is x.
        self.assertEqual(len(reference[0]), 16)

        def moved(by_m: float) -> list[list[float]]:
            rows = [list(row) for row in reference]
            for row in rows:
                row[3] += by_m
            return rows

        # Five micrometres is another digest and the same trajectory.
        rounded = moved(5e-6)
        with self.assertRaisesRegex(SystemExit, "--alignment-session"):
            load_alignment(manifest, model_sha256=model, trajectory=rounded)
        transform, used = load_alignment(
            manifest,
            model_sha256=model,
            trajectory=rounded,
            reference_trajectory=reference,
        )
        np.testing.assert_array_equal(
            transform.ravel(), fitted["registration"]["model_from_source"]
        )
        match = used["trajectory_match"]
        self.assertEqual(match["by"], "poses")
        self.assertEqual(match["poses"], len(reference))
        self.assertAlmostEqual(
            match["largest_translation_difference_m"], 5e-6, places=12
        )
        self.assertLess(match["largest_rotation_difference_rad"], 1e-7)

        # The same trajectory exactly needs no reference at all.
        _, exact = load_alignment(
            manifest, model_sha256=model, trajectory=reference
        )
        self.assertEqual(exact["trajectory_match"], {"by": "digest"})

        # A millimetre is a different trajectory, with or without one.
        with self.assertRaisesRegex(SystemExit, "poses differ by up to"):
            load_alignment(
                manifest,
                model_sha256=model,
                trajectory=moved(1e-3),
                reference_trajectory=reference,
            )
        turned = [list(row) for row in reference]
        cosine, sine = math.cos(1e-3), math.sin(1e-3)
        first = np.array(turned[4]).reshape((4, 4))
        first[:3, :3] = first[:3, :3] @ np.array(
            [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
        )
        turned[4] = [float(value) for value in first.ravel()]
        with self.assertRaisesRegex(SystemExit, "poses differ by up to"):
            load_alignment(
                manifest,
                model_sha256=model,
                trajectory=turned,
                reference_trajectory=reference,
            )
        # A frame fewer cannot be paired.
        with self.assertRaisesRegex(SystemExit, "same poses to compare"):
            load_alignment(
                manifest,
                model_sha256=model,
                trajectory=reference[:-1],
                reference_trajectory=reference,
            )
        # And the reference has to be the session the report was made on.
        with self.assertRaisesRegex(SystemExit, "not the one that report"):
            load_alignment(
                manifest,
                model_sha256=model,
                trajectory=rounded,
                reference_trajectory=moved(2e-6),
            )

    def test_pose_differences_are_measured_for_both_kinds_of_pose(
        self,
    ) -> None:
        half = math.sin(1e-5)
        quaternion = [
            [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0],
            [1.5, 2.0, 3.0, 0.0, 0.0, 1.0, 0.0],
        ]
        nudged = [
            [1.0, 2.0, 3.00003, 0.0, 0.0, half, math.cos(1e-5)],
            # The same rotation with every sign flipped is the same pose.
            [1.5, 2.0, 3.0, 0.0, 0.0, -1.0, 0.0],
        ]
        worst_m, worst_rad = trajectory_difference(quaternion, nudged)
        self.assertAlmostEqual(worst_m, 3e-5, places=12)
        self.assertAlmostEqual(worst_rad, 2e-5, places=9)
        self.assertEqual(trajectory_difference(quaternion, quaternion), (0.0, 0.0))
        matrix = [[float(v) for v in np.eye(4).ravel()]] * 2
        self.assertIsNone(trajectory_difference(quaternion, matrix))
        self.assertIsNone(trajectory_difference(quaternion, quaternion[:1]))
        self.assertIsNone(trajectory_difference([], []))

    def test_a_starting_guess_and_a_reused_alignment_are_exclusive(
        self,
    ) -> None:
        baseline()
        with (
            self.assertRaises(SystemExit) as raised,
            redirect_stderr(io.StringIO()),
        ):
            main(
                [
                    *report_arguments(),
                    "--alignment",
                    str(ROOM.root / "baseline.json"),
                ]
            )
        self.assertEqual(raised.exception.code, 2)

    def test_the_reused_manifest_is_an_input_and_is_not_written_over(
        self,
    ) -> None:
        baseline()
        source = ROOM.root / "baseline.json"
        before = source.read_bytes()
        with (
            source_state("a" * 40, True),
            self.assertRaises(SystemExit) as raised,
            redirect_stdout(io.StringIO()),
        ):
            main(
                [
                    *report_arguments(translation=None),
                    "--alignment",
                    str(source),
                    "--manifest-out",
                    str(source),
                ]
            )
        self.assertIn("write over an input", str(raised.exception))
        self.assertEqual(source.read_bytes(), before)


class RoomRefusalTests(unittest.TestCase):
    def arguments(self, **replaced) -> list[str]:
        return report_arguments(**replaced)

    def assert_refused(self, message: str, arguments: list[str]) -> None:
        with (
            self.assertRaises(SystemExit) as raised,
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
        ):
            main(arguments)
        self.assertIn(message, str(raised.exception))

    def test_a_mesh_of_some_other_volume_is_refused(self) -> None:
        encoded = ROOM.mesh.read_bytes()
        marker = b"comment spatialforge_source_volume_sha256 "
        at = encoded.index(marker) + len(marker)
        foreign = ROOM.root / "foreign.ply"
        foreign.write_bytes(encoded[:at] + b"0" * 64 + encoded[at + 64:])
        self.assert_refused(
            "was not extracted from this volume",
            self.arguments(mesh=foreign),
        )

    def test_a_volume_is_only_measured_with_its_own_scan(self) -> None:
        self.assert_refused(
            "volume was fused from session",
            self.arguments(session=FIXTURE),
        )

    def test_a_model_without_normals_is_refused(self) -> None:
        points, normals = room_model()
        flat = ROOM.root / "flat.ply"
        write_model(flat, points, normals, with_normals=False)
        self.assert_refused(
            "missing nx, ny, nz", self.arguments(model=flat)
        )

    def test_a_guess_far_from_the_model_is_an_error_not_a_number(
        self,
    ) -> None:
        arguments = self.arguments()
        at = arguments.index("--initial-translation")
        arguments[at + 1:at + 4] = ["7.0", "7.0", "7.0"]
        self.assert_refused("too far off", arguments)

    def test_manifest_provenance_and_output_safety(self) -> None:
        target = ROOM.root / "result.json"
        with source_state("b" * 40, False):
            self.assert_refused(
                "dirty working tree",
                [*self.arguments(), "--manifest-out", str(target)],
            )
        self.assertFalse(target.exists())
        existing = ROOM.root / "existing.json"
        existing.write_bytes(b"keep")
        with source_state("b" * 40, True):
            self.assert_refused(
                "already exists",
                [*self.arguments(), "--manifest-out", str(existing)],
            )
            taken = ROOM.root / "taken.npy"
            taken.write_bytes(b"keep")
            self.assert_refused(
                "already exists",
                [*self.arguments(), "--errors-out", str(taken)],
            )
            self.assertEqual(taken.read_bytes(), b"keep")
            self.assert_refused(
                "must end in .npy",
                [*self.arguments(), "--errors-out", str(target)],
            )
            self.assert_refused(
                "must end in .json",
                [*self.arguments(), "--manifest-out", str(ROOM.model)],
            )
            self.assert_refused(
                "write over an input",
                [
                    *self.arguments(),
                    "--manifest-out",
                    str(ROOM.session / "result.json"),
                ],
            )
        self.assertEqual(existing.read_bytes(), b"keep")

    def test_a_dirty_tree_is_recorded_when_explicitly_allowed(self) -> None:
        target = ROOM.root / "dirty.json"
        with source_state("c" * 40, False), redirect_stdout(io.StringIO()):
            exit_code = main(
                [
                    *self.arguments(),
                    "--manifest-out",
                    str(target),
                    "--allow-dirty",
                ]
            )
        self.assertEqual(exit_code, 0)
        self.assertFalse(json.loads(target.read_bytes())["source_worktree_clean"])

    def test_unusable_arguments_are_refused(self) -> None:
        for extra in (
            ["--fit-frame-stride", "1"],
            ["--fit-frame-stride", "0"],
            ["--pixel-step", "0"],
            ["--limit-m", "0"],
            ["--limit-m", "nan"],
            ["--initial-translation", "nan", "0", "0"],
            ["--alignment-session", str(FIXTURE)],
        ):
            with self.subTest(extra=extra):
                with (
                    self.assertRaises(SystemExit) as raised,
                    redirect_stderr(io.StringIO()),
                ):
                    main([*self.arguments(), *extra])
                self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
