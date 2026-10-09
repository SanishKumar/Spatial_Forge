from __future__ import annotations

import io
import json
import math
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from spatialforge.cli import main
from spatialforge.errors import TumImportError
from spatialforge.replay import replay_session
from spatialforge.session_loader import load_scan_session
from spatialforge.tum_importer import (
    T_RIG_CAMERA,
    TUM_DEFAULT_INTRINSICS,
    TUM_SOURCE_UP_AXES,
    TumCameraIntrinsics,
    _associate_timestamps,
    import_tum_dataset,
)


TEST_ROOT = Path(__file__).resolve().parent
TUM_FIXTURE = (
    TEST_ROOT
    / "fixtures"
    / "tum"
    / "rgbd_dataset_freiburg1_tiny"
)


def tree_snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


class TumImporterTests(unittest.TestCase):
    def test_import_produces_valid_replayable_known_pose_session(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            output = Path(temporary_directory) / "tiny.vgsession"
            report = import_tum_dataset(TUM_FIXTURE, output)
            session = load_scan_session(output)
            replay = replay_session(session)

        self.assertEqual(report.source_rgb_count, 3)
        self.assertEqual(report.source_depth_count, 3)
        self.assertEqual(report.source_pose_count, 2)
        self.assertEqual(report.matched_rgbd_count, 2)
        self.assertEqual(report.matched_pose_count, 2)
        self.assertEqual(report.unmatched_rgb_count, 1)
        self.assertEqual(report.unmatched_depth_count, 1)
        self.assertEqual(len(session.streams["rgb"]), 2)
        self.assertEqual(len(session.streams["depth"]), 2)
        self.assertEqual(len(session.streams["pose"]), 2)
        self.assertEqual(session.streams["rgb"][1].timestamp_ns, 33_333_000)
        self.assertEqual(
            session.streams["depth"][0].data["association_delta_ns"],
            1_000_000,
        )
        self.assertEqual(
            session.streams["depth"][1].data["association_delta_ns"],
            667_000,
        )
        self.assertAlmostEqual(
            session.stream_definitions["depth"].depth_scale_m,
            0.0002,
        )
        self.assertEqual(len(replay.observations), 2)
        self.assertTrue(
            all(observation.depth is not None for observation in replay.observations)
        )
        self.assertTrue(
            all(observation.pose is not None for observation in replay.observations)
        )
        self.assertEqual(
            replay.digest_sha256,
            "2545dbf336019d34890cf05b069c2ad89664e45d45de06ebb4040e642f105027",
        )

    def test_pose_is_rebased_into_initial_forward_left_up_rig_frame(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            output = Path(temporary_directory) / "pose.vgsession"
            import_tum_dataset(TUM_FIXTURE, output)
            session = load_scan_session(output)
            first_pose = session.streams["pose"][0].data["T_world_camera"]
            second_pose = session.streams["pose"][1].data["T_world_camera"]

        self.assertEqual(tuple(first_pose), T_RIG_CAMERA)
        self.assertEqual(
            tuple(second_pose),
            (
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                1.0,
                0.0,
                -1.0,
                -1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ),
        )

    def test_import_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            first_output = temporary_root / "first.vgsession"
            second_output = temporary_root / "second.vgsession"

            import_tum_dataset(TUM_FIXTURE, first_output)
            import_tum_dataset(TUM_FIXTURE, second_output)
            first_replay = replay_session(load_scan_session(first_output))
            second_replay = replay_session(load_scan_session(second_output))

            self.assertEqual(
                tree_snapshot(first_output),
                tree_snapshot(second_output),
            )

        self.assertEqual(
            first_replay.digest_sha256,
            second_replay.digest_sha256,
        )

    def test_groundtruth_is_optional_and_no_pose_is_fabricated(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            source = temporary_root / "tum-without-groundtruth"
            output = temporary_root / "without-pose.vgsession"
            shutil.copytree(TUM_FIXTURE, source)
            (source / "groundtruth.txt").unlink()

            report = import_tum_dataset(source, output)
            session = load_scan_session(output)
            replay = replay_session(session)

        self.assertEqual(report.source_pose_count, 0)
        self.assertEqual(report.matched_pose_count, 0)
        self.assertNotIn("pose", session.streams)
        self.assertTrue(
            all(observation.pose is None for observation in replay.observations)
        )

    def test_groundtruth_gaps_remain_missing(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            source = temporary_root / "tum-with-pose-gap"
            output = temporary_root / "pose-gap.vgsession"
            shutil.copytree(TUM_FIXTURE, source)
            (source / "groundtruth.txt").write_text(
                "1305031102.000500 1 2 3 0 0 0 1\n",
                encoding="utf-8",
            )

            report = import_tum_dataset(source, output)
            replay = replay_session(load_scan_session(output))

        self.assertEqual(report.matched_pose_count, 1)
        self.assertIsNotNone(replay.observations[0].pose)
        self.assertIsNone(replay.observations[1].pose)

    def test_association_is_strict_and_one_to_one(self) -> None:
        self.assertEqual(
            _associate_timestamps([0], [20_000_000]),
            {},
        )
        self.assertEqual(
            _associate_timestamps(
                [0, 40_000_000],
                [20_000_000],
                max_difference_ns=20_000_001,
            ),
            {0: 0},
        )

    def test_source_path_traversal_is_rejected_without_output(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            source = temporary_root / "unsafe-tum"
            output = temporary_root / "unsafe.vgsession"
            shutil.copytree(TUM_FIXTURE, source)
            (source / "rgb.txt").write_text(
                "1305031102.000000 ../outside.ppm\n",
                encoding="utf-8",
            )

            with self.assertRaises(TumImportError) as raised:
                import_tum_dataset(source, output)

            self.assertFalse(output.exists())

        self.assertIn(
            "path must be relative and remain inside the dataset",
            str(raised.exception),
        )

    def test_existing_output_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            output = Path(temporary_directory) / "existing.vgsession"
            output.mkdir()
            marker = output / "keep.txt"
            marker.write_text("keep", encoding="utf-8")

            with self.assertRaises(TumImportError):
                import_tum_dataset(TUM_FIXTURE, output)

            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")


class TumImporterIntrinsicsTests(unittest.TestCase):
    def test_default_intrinsics_are_the_benchmark_projection(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            implicit = temporary_root / "implicit.vgsession"
            explicit = temporary_root / "explicit.vgsession"
            import_tum_dataset(TUM_FIXTURE, implicit)
            import_tum_dataset(
                TUM_FIXTURE,
                explicit,
                intrinsics=TUM_DEFAULT_INTRINSICS,
            )
            implicit_files = tree_snapshot(implicit)
            explicit_files = tree_snapshot(explicit)

        self.assertEqual(implicit_files, explicit_files)
        self.assertEqual(
            json.loads(implicit_files["calibration/cameras.json"])["cameras"][
                0
            ]["intrinsics"],
            {"fx": 525.0, "fy": 525.0, "cx": 319.5, "cy": 239.5},
        )

    def test_override_changes_the_calibration_and_nothing_else(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            default = temporary_root / "default.vgsession"
            overridden = temporary_root / "overridden.vgsession"
            import_tum_dataset(TUM_FIXTURE, default)
            import_tum_dataset(
                TUM_FIXTURE,
                overridden,
                intrinsics=TumCameraIntrinsics(
                    fx=481.2,
                    fy=480.0,
                    cx=319.5,
                    cy=239.5,
                ),
            )
            load_scan_session(overridden)
            default_files = tree_snapshot(default)
            overridden_files = tree_snapshot(overridden)

        self.assertEqual(set(default_files), set(overridden_files))
        self.assertEqual(
            [
                name
                for name in sorted(default_files)
                if default_files[name] != overridden_files[name]
            ],
            ["calibration/cameras.json"],
        )
        self.assertEqual(
            json.loads(overridden_files["calibration/cameras.json"])[
                "cameras"
            ][0]["intrinsics"],
            {"fx": 481.2, "fy": 480.0, "cx": 319.5, "cy": 239.5},
        )

    def test_a_mirrored_or_malformed_camera_is_refused(self) -> None:
        cases = (
            ({"fy": -480.0}, "fy must be positive"),
            ({"fx": 0.0}, "fx must be positive"),
            ({"cx": float("nan")}, "cx must be a finite number"),
            ({"cy": float("inf")}, "cy must be a finite number"),
            ({"fx": True}, "fx must be a finite number"),
            ({"fy": "480"}, "fy must be a finite number"),
        )
        for override, message in cases:
            values = {"fx": 481.2, "fy": 480.0, "cx": 319.5, "cy": 239.5}
            values.update(override)
            with self.subTest(override=override):
                with tempfile.TemporaryDirectory(
                    dir=TEST_ROOT
                ) as temporary_directory:
                    output = Path(temporary_directory) / "refused.vgsession"
                    with self.assertRaises(TumImportError) as raised:
                        import_tum_dataset(
                            TUM_FIXTURE,
                            output,
                            intrinsics=TumCameraIntrinsics(**values),
                        )
                    self.assertFalse(output.exists())
                    self.assertEqual(
                        list(Path(temporary_directory).iterdir()),
                        [],
                    )
                self.assertIn(message, str(raised.exception))

    def test_a_negative_focal_length_is_named_as_a_mirrored_frame(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            output = Path(temporary_directory) / "mirrored.vgsession"
            with self.assertRaises(TumImportError) as raised:
                import_tum_dataset(
                    TUM_FIXTURE,
                    output,
                    intrinsics=TumCameraIntrinsics(
                        fx=481.2,
                        fy=-480.0,
                        cx=319.5,
                        cy=239.5,
                    ),
                )

        self.assertIn("mirrored camera frame", str(raised.exception))


def quaternion_of(rotation) -> tuple[float, float, float, float]:
    """The unit quaternion (x, y, z, w) of a rotation with w well off zero."""

    w = math.sqrt(1.0 + rotation[0][0] + rotation[1][1] + rotation[2][2]) / 2.0
    return (
        (rotation[2][1] - rotation[1][2]) / (4.0 * w),
        (rotation[0][2] - rotation[2][0]) / (4.0 * w),
        (rotation[1][0] - rotation[0][1]) / (4.0 * w),
        w,
    )


def camera_rotation(forward, up) -> list[list[float]]:
    """Columns right, down, forward for a camera facing ``forward``.

    ``up`` is the way the top of its image points, made perpendicular.
    """

    length = math.sqrt(sum(value * value for value in forward))
    forward = [value / length for value in forward]
    along = sum(a * b for a, b in zip(up, forward))
    down = [-(a - along * b) for a, b in zip(up, forward)]
    length = math.sqrt(sum(value * value for value in down))
    down = [value / length for value in down]
    right = [
        down[1] * forward[2] - down[2] * forward[1],
        down[2] * forward[0] - down[0] * forward[2],
        down[0] * forward[1] - down[1] * forward[0],
    ]
    return [[right[row], down[row], forward[row]] for row in range(3)]


def matrix(values) -> list[list[float]]:
    return [list(values[4 * row:4 * row + 4]) for row in range(4)]


def product(left, right) -> list[list[float]]:
    return [
        [sum(left[i][k] * right[k][j] for k in range(4)) for j in range(4)]
        for i in range(4)
    ]


def inverse(pose) -> list[list[float]]:
    rotation = [[pose[j][i] for j in range(3)] for i in range(3)]
    moved = [-sum(rotation[i][k] * pose[k][3] for k in range(3)) for i in range(3)]
    return [[*rotation[i], moved[i]] for i in range(3)] + [[0.0, 0.0, 0.0, 1.0]]


class TumImporterLevelTests(unittest.TestCase):
    """A session whose z is the dataset's up, not the first camera's."""

    def dataset(self, root: Path, poses) -> Path:
        """The tiny fixture with its two poses replaced."""

        source = root / "rgbd_dataset_freiburg1_tiny"
        shutil.copytree(TUM_FIXTURE, source)
        lines = ["# timestamp tx ty tz qx qy qz qw"]
        for stamp, (position, rotation) in zip(
            ("1305031102.000500", "1305031102.033000"), poses
        ):
            lines.append(
                " ".join(
                    [stamp]
                    + [repr(float(value)) for value in position]
                    + [repr(value) for value in quaternion_of(rotation)]
                )
            )
        (source / "groundtruth.txt").write_text(
            "\n".join(lines) + "\n", encoding="ascii"
        )
        return source

    def poses_of(self, session_path: Path):
        session = load_scan_session(session_path)
        return [
            matrix(sample.data["T_world_camera"])
            for sample in session.streams["pose"]
        ]

    def test_a_level_first_camera_gives_the_session_it_always_gave(
        self,
    ) -> None:
        # Facing along the dataset's x with the top of its image along z:
        # this camera's own forward, left and up are already the level
        # ones, so asking for a level session changes nothing at all.
        level = camera_rotation((1, 0, 0), (0, 0, 1))
        turned = camera_rotation((0.6, 0.8, 0), (0, 0, 1))
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            root = Path(temporary)
            source = self.dataset(
                root, [((1.0, 2.0, 3.0), level), ((1.5, 2.0, 3.25), turned)]
            )
            import_tum_dataset(source, root / "as-held.vgsession")
            import_tum_dataset(source, root / "level.vgsession", source_up="z")
            self.assertEqual(
                tree_snapshot(root / "level.vgsession"),
                tree_snapshot(root / "as-held.vgsession"),
            )
            first, second = self.poses_of(root / "level.vgsession")
        self.assertEqual(
            [value for row in first for value in row], list(T_RIG_CAMERA)
        )
        # Half a metre along x and a quarter of a metre up.
        self.assertEqual([row[3] for row in second[:3]], [0.5, 0.0, 0.25])

    def test_a_tilted_first_camera_keeps_its_tilt(self) -> None:
        # Looking along x and 30 degrees down. As held, the session
        # takes this camera for level and the whole room tips up to
        # meet it. Level, the camera is the thing that is tipped.
        dive = math.radians(30.0)
        tilted = camera_rotation(
            (math.cos(dive), 0.0, -math.sin(dive)), (0, 0, 1)
        )
        other = camera_rotation((0.2, 1.0, -0.4), (0.1, 0, 1))
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            root = Path(temporary)
            source = self.dataset(
                root, [((1.0, 2.0, 3.0), tilted), ((1.5, 2.5, 3.25), other)]
            )
            import_tum_dataset(source, root / "as-held.vgsession")
            import_tum_dataset(source, root / "level.vgsession", source_up="z")
            held = self.poses_of(root / "as-held.vgsession")
            level = self.poses_of(root / "level.vgsession")

        self.assertEqual(
            [value for row in held[0] for value in row], list(T_RIG_CAMERA)
        )
        # The first camera is at the origin and faces along x and down.
        self.assertEqual([row[3] for row in level[0][:3]], [0.0, 0.0, 0.0])
        facing = [row[2] for row in level[0][:3]]
        self.assertAlmostEqual(facing[0], math.cos(dive), places=12)
        self.assertEqual(facing[1], 0.0)
        self.assertAlmostEqual(facing[2], -math.sin(dive), places=12)
        # The second is where the dataset put it relative to the first:
        # the dataset's x, y and z here are the session's.
        for axis, expected in enumerate((0.5, 0.5, 0.25)):
            self.assertAlmostEqual(level[1][axis][3], expected, places=12)
        # As held, the same quarter of a metre of height is spread over
        # two axes.
        self.assertNotAlmostEqual(held[1][2][3], 0.25, places=2)
        # And nothing between the frames has changed: the motion from
        # one camera to the next is the same motion.
        relative_held = product(inverse(held[0]), held[1])
        relative_level = product(inverse(level[0]), level[1])
        for row in range(4):
            for column in range(4):
                self.assertAlmostEqual(
                    relative_level[row][column],
                    relative_held[row][column],
                    places=12,
                )

    def test_any_axis_can_be_the_one_that_points_up(self) -> None:
        position_first, position_second = (0.3, -1.1, 0.7), (0.9, -0.2, 1.6)
        first = camera_rotation((1.0, 0.7, -0.4), (0.2, -0.1, 1.0))
        second = camera_rotation((-0.3, 1.0, 0.5), (0.0, 0.3, 1.0))
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            root = Path(temporary)
            source = self.dataset(
                root, [(position_first, first), (position_second, second)]
            )
            for name, up in TUM_SOURCE_UP_AXES.items():
                with self.subTest(up=name):
                    output = root / f"level{name}.vgsession"
                    report = import_tum_dataset(source, output, source_up=name)
                    level = self.poses_of(output)
                    self.assertEqual(report.source_up, name)
                    # Height in the session is distance along the named
                    # axis in the dataset.
                    climbed = sum(
                        (position_second[axis] - position_first[axis])
                        * up[axis]
                        for axis in range(3)
                    )
                    self.assertAlmostEqual(level[1][2][3], climbed, places=12)
                    self.assertEqual(
                        [row[3] for row in level[0][:3]], [0.0, 0.0, 0.0]
                    )
                    # The first camera faces along x and to neither side.
                    self.assertAlmostEqual(level[0][1][2], 0.0, places=12)
                    self.assertGreater(level[0][0][2], 0.0)
                    # Still a rotation, and a right-handed one.
                    rotation = [row[:3] for row in level[1][:3]]
                    for i in range(3):
                        for j in range(3):
                            dot = sum(
                                rotation[k][i] * rotation[k][j]
                                for k in range(3)
                            )
                            self.assertAlmostEqual(
                                dot, 1.0 if i == j else 0.0, places=12
                            )
                    self.assertGreater(
                        rotation[0][0]
                        * (
                            rotation[1][1] * rotation[2][2]
                            - rotation[1][2] * rotation[2][1]
                        )
                        - rotation[0][1]
                        * (
                            rotation[1][0] * rotation[2][2]
                            - rotation[1][2] * rotation[2][0]
                        )
                        + rotation[0][2]
                        * (
                            rotation[1][0] * rotation[2][1]
                            - rotation[1][1] * rotation[2][0]
                        ),
                        0.999,
                    )

    def test_what_cannot_be_levelled_is_refused_and_nothing_written(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            root = Path(temporary)
            # The fixture's own first camera looks along the dataset's z.
            for name, source_up, message in (
                ("along", "z", "straight along the up axis"),
                ("against", "-z", "straight along the up axis"),
                ("unknown", "w", "source_up must be one of"),
                ("upper", "Z", "source_up must be one of"),
            ):
                with self.subTest(case=name):
                    with self.assertRaises(TumImportError) as raised:
                        import_tum_dataset(
                            TUM_FIXTURE,
                            root / f"{name}.vgsession",
                            source_up=source_up,
                        )
                    self.assertIn(message, str(raised.exception))
            # A scan with no poses has nothing to be levelled by.
            unposed = root / "rgbd_dataset_freiburg1_tiny"
            shutil.copytree(TUM_FIXTURE, unposed)
            (unposed / "groundtruth.txt").unlink()
            with self.assertRaises(TumImportError) as raised:
                import_tum_dataset(
                    unposed, root / "unposed.vgsession", source_up="y"
                )
            self.assertIn("no frame has a pose", str(raised.exception))
            # The same fixture is level about its x and y.
            import_tum_dataset(TUM_FIXTURE, root / "x.vgsession", source_up="x")
            self.assertEqual(
                sorted(path.name for path in root.iterdir()),
                ["rgbd_dataset_freiburg1_tiny", "x.vgsession"],
            )

    def test_cli_names_the_axis_and_says_which_frame_it_wrote(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            root = Path(temporary)
            outputs = {}
            for name, extra in (
                ("held", []),
                ("up", ["--up", "x"]),
                ("down", ["--down", "y"]),
            ):
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    exit_code = main(
                        [
                            "scan",
                            "import-tum",
                            str(TUM_FIXTURE),
                            str(root / f"{name}.vgsession"),
                            *extra,
                        ]
                    )
                self.assertEqual(exit_code, 0)
                outputs[name] = stdout.getvalue()
            down = self.poses_of(root / "down.vgsession")
            by_name = root / "by-name.vgsession"
            import_tum_dataset(TUM_FIXTURE, by_name, source_up="-y")
            self.assertEqual(
                tree_snapshot(root / "down.vgsession"), tree_snapshot(by_name)
            )
            # Both at once, or an axis the dataset does not have.
            for extra in (["--up", "x", "--down", "y"], ["--up", "w"]):
                with (
                    self.assertRaises(SystemExit) as raised,
                    redirect_stderr(io.StringIO()),
                ):
                    main(
                        [
                            "scan",
                            "import-tum",
                            str(TUM_FIXTURE),
                            str(root / "refused.vgsession"),
                            *extra,
                        ]
                    )
                self.assertEqual(raised.exception.code, 2)
            # The first camera looks along z: refused, and it says why.
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                exit_code = main(
                    [
                        "scan",
                        "import-tum",
                        str(TUM_FIXTURE),
                        str(root / "refused.vgsession"),
                        "--up",
                        "z",
                    ]
                )
            self.assertEqual(exit_code, 2)
            self.assertIn("straight along the up axis", stderr.getvalue())
            self.assertFalse((root / "refused.vgsession").exists())

        self.assertIn("frame: first camera\n", outputs["held"])
        self.assertIn("frame: level, up is source x\n", outputs["up"])
        self.assertIn("frame: level, up is source -y\n", outputs["down"])
        # The second camera is a metre along the dataset's x from the
        # first, which with y down is no height at all.
        self.assertEqual(down[1][2][3], 0.0)


class TumImporterCliTests(unittest.TestCase):
    def test_cli_imports_and_reports_digest(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            output = Path(temporary_directory) / "cli.vgsession"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "scan",
                        "import-tum",
                        str(TUM_FIXTURE),
                        str(output),
                    ]
                )

            self.assertTrue(output.is_dir())

        self.assertEqual(exit_code, 0)
        self.assertIn("IMPORTED tum-rgbd_dataset_freiburg1_tiny", stdout.getvalue())
        self.assertIn("matched: rgb_depth=2 poses=2", stdout.getvalue())
        self.assertIn("digest_sha256: ", stdout.getvalue())

    def test_cli_failure_is_nonzero_and_actionable(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            output = Path(temporary_directory) / "bad.vgsession"
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                exit_code = main(
                    [
                        "scan",
                        "import-tum",
                        str(TEST_ROOT / "does-not-exist"),
                        str(output),
                    ]
                )

        self.assertEqual(exit_code, 2)
        self.assertIn("IMPORT FAILED", stderr.getvalue())
        self.assertIn("source directory does not exist", stderr.getvalue())


class TumImporterCliIntrinsicsTests(unittest.TestCase):
    def test_cli_intrinsics_reach_the_calibration(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            output = Path(temporary_directory) / "cli.vgsession"
            with redirect_stdout(io.StringIO()):
                exit_code = main(
                    [
                        "scan",
                        "import-tum",
                        str(TUM_FIXTURE),
                        str(output),
                        "--fx",
                        "481.2",
                        "--fy",
                        "480",
                    ]
                )
            calibration = json.loads(
                (output / "calibration" / "cameras.json").read_bytes()
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            calibration["cameras"][0]["intrinsics"],
            {"fx": 481.2, "fy": 480.0, "cx": 319.5, "cy": 239.5},
        )

    def test_cli_refuses_a_negative_focal_length(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            output = Path(temporary_directory) / "mirrored.vgsession"
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                exit_code = main(
                    [
                        "scan",
                        "import-tum",
                        str(TUM_FIXTURE),
                        str(output),
                        "--fy=-480",
                    ]
                )
            self.assertFalse(output.exists())

        self.assertEqual(exit_code, 2)
        self.assertIn("IMPORT FAILED", stderr.getvalue())
        self.assertIn("fy must be positive", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
