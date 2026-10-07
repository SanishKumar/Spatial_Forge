"""The ICL-NUIM conversion, checked against the geometry it claims.

The tool rewrites a pose by permuting four numbers and negating one. That
is either exactly the change of basis its documentation describes or a
mirrored room, and the two cannot be told apart by looking at the output.
So the permutation is checked against the matrix form it stands for, and
the matrix form is checked against the only thing that defines it: a point
seen by the published left-handed camera must land on the same pixel, at
the same depth, when seen by the converted right-handed one.
"""

from __future__ import annotations

import io
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from spatialforge.session_loader import load_scan_session
from spatialforge.tum_importer import TumCameraIntrinsics, import_tum_dataset

from tools.prepare_icl_nuim import (
    ICL_NUIM_INTRINSICS,
    convert_pose,
    main,
    prepare_icl_nuim,
)

TEST_ROOT = Path(__file__).resolve().parent
TUM_FIXTURE = TEST_ROOT / "fixtures" / "tum" / "rgbd_dataset_freiburg1_tiny"

WORLD_FLIP = np.diag([1.0, 1.0, -1.0])
CAMERA_FLIP = np.diag([1.0, -1.0, 1.0])


def rotation_matrix(quaternion_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = quaternion_xyzw / np.linalg.norm(quaternion_xyzw)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def random_pose(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    quaternion = rng.normal(size=4)
    return quaternion / np.linalg.norm(quaternion), rng.uniform(-3, 3, size=3)


def converted(
    quaternion: np.ndarray,
    translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    fields = [repr(float(value)) for value in (*translation, *quaternion)]
    result = [float(token) for token in convert_pose(fields)]
    return np.array(result[3:]), np.array(result[:3])


class ConversionGeometryTests(unittest.TestCase):
    def test_the_permutation_is_the_documented_change_of_basis(self) -> None:
        rng = np.random.default_rng(11)
        for _ in range(200):
            quaternion, translation = random_pose(rng)
            new_quaternion, new_translation = converted(
                quaternion, translation
            )
            np.testing.assert_allclose(
                rotation_matrix(new_quaternion),
                WORLD_FLIP @ rotation_matrix(quaternion) @ CAMERA_FLIP,
                atol=1e-14,
            )
            np.testing.assert_array_equal(
                new_translation, WORLD_FLIP @ translation
            )

    def test_the_result_is_a_proper_rotation(self) -> None:
        rng = np.random.default_rng(12)
        for _ in range(50):
            quaternion, translation = random_pose(rng)
            rotation = rotation_matrix(converted(quaternion, translation)[0])
            self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0)
            np.testing.assert_allclose(
                rotation @ rotation.T, np.eye(3), atol=1e-14
            )

    def test_both_descriptions_put_a_point_on_the_same_pixel(self) -> None:
        fx, cx, cy = 481.2, 319.5, 239.5
        published_fy = -480.0
        rng = np.random.default_rng(13)
        for _ in range(200):
            quaternion, translation = random_pose(rng)
            # A point in front of the published, left-handed camera.
            camera_point = np.array(
                [rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(0.5, 4)]
            )
            world_point = rotation_matrix(quaternion) @ camera_point + (
                translation
            )
            published_pixel = (
                fx * camera_point[0] / camera_point[2] + cx,
                published_fy * camera_point[1] / camera_point[2] + cy,
            )

            new_quaternion, new_translation = converted(
                quaternion, translation
            )
            seen = rotation_matrix(new_quaternion).T @ (
                WORLD_FLIP @ world_point - new_translation
            )
            pixel = (
                fx * seen[0] / seen[2] + cx,
                ICL_NUIM_INTRINSICS["fy"] * seen[1] / seen[2] + cy,
            )
            np.testing.assert_allclose(pixel, published_pixel, atol=1e-9)
            self.assertAlmostEqual(seen[2], camera_point[2], places=12)

    def test_keeping_the_poses_and_dropping_the_sign_is_a_different_room(
        self,
    ) -> None:
        # The shortcut the conversion exists to prevent: positive fy with
        # the published pose. It disagrees with the dataset everywhere off
        # the image's horizontal centre line.
        quaternion, translation = random_pose(np.random.default_rng(14))
        camera_point = np.array([0.3, 0.4, 2.0])
        world_point = rotation_matrix(quaternion) @ camera_point + translation
        seen = rotation_matrix(quaternion).T @ (world_point - translation)
        shortcut_row = 480.0 * seen[1] / seen[2] + 239.5
        published_row = -480.0 * camera_point[1] / camera_point[2] + 239.5
        self.assertGreater(abs(shortcut_row - published_row), 100.0)

    def test_digits_are_carried_not_recomputed(self) -> None:
        self.assertEqual(
            convert_pose(
                ["-0.00473952", "1e-08", "-2.25009", "0.1", "-4.2262e-05",
                 "0.000114325", "0.999998"]
            ),
            ["-0.00473952", "1e-08", "2.25009", "0.999998", "0.000114325",
             "-4.2262e-05", "0.1"],
        )
        self.assertEqual(convert_pose(["0", "0", "2.5", *"0001"])[2], "-2.5")
        # Negative zero would be a different token for the same pose.
        for zero in ("0", "0.0", "-0", "0e0"):
            self.assertEqual(convert_pose(["0", "0", zero, *"0001"])[2], "0")
        self.assertEqual(
            float(convert_pose(["0", "0", "3.5e-07", *"0001"])[2]), -3.5e-07
        )


def write_sequence(root: Path, *, poses: str, associations: str) -> Path:
    source = root / "living_room_traj9_frei_png"
    (source / "rgb").mkdir(parents=True)
    (source / "depth").mkdir()
    for index in range(3):
        shutil.copyfile(
            TUM_FIXTURE / "rgb" / f"00000{index}.ppm",
            source / "rgb" / f"{index}.ppm",
        )
        shutil.copyfile(
            TUM_FIXTURE / "depth" / f"00000{index}.pgm",
            source / "depth" / f"{index}.pgm",
        )
    (source / "associations.txt").write_bytes(associations.encode("ascii"))
    (source / "livingRoom9.gt.freiburg").write_bytes(poses.encode("ascii"))
    return source


ASSOCIATIONS = (
    "0 depth/0.pgm 0 rgb/0.ppm\n"
    "1 depth/1.pgm 1 rgb/1.ppm\n"
    "2 depth/2.pgm 2 rgb/2.ppm\n"
)
# Frame 0 has no pose, as in the published sequences.
POSES = (
    "1 0 0 -2.25 0 0 0 1\n"
    "2 -0.25 0.5 -2.5 0.5 -0.5 0.5 0.5\n"
)


class PrepareSequenceTests(unittest.TestCase):
    def test_writes_exactly_the_tum_layout(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            root = Path(temporary)
            source = write_sequence(
                root, poses=POSES, associations=ASSOCIATIONS
            )
            output = root / "prepared"
            counts = prepare_icl_nuim(source, output)
            written = {
                path.relative_to(output).as_posix(): path.read_bytes()
                for path in sorted(output.rglob("*"))
                if path.is_file()
            }
            originals = {
                f"{kind}/{index}.{suffix}": (
                    source / kind / f"{index}.{suffix}"
                ).read_bytes()
                for kind, suffix in (("rgb", "ppm"), ("depth", "pgm"))
                for index in (1, 2)
            }
            leftovers = sorted(path.name for path in root.iterdir())

        self.assertEqual(
            counts,
            {"frames": 2, "frames_without_pose": 1, "poses_without_frame": 0},
        )
        self.assertEqual(
            written["rgb.txt"], b"1 rgb/1.ppm\n2 rgb/2.ppm\n"
        )
        self.assertEqual(
            written["depth.txt"], b"1 depth/1.pgm\n2 depth/2.pgm\n"
        )
        self.assertEqual(
            written["groundtruth.txt"],
            b"1 0 0 2.25 1 0 0 0\n2 -0.25 0.5 2.5 0.5 0.5 -0.5 0.5\n",
        )
        self.assertEqual(
            sorted(written),
            sorted([*originals, "rgb.txt", "depth.txt", "groundtruth.txt"]),
        )
        for name, content in originals.items():
            self.assertEqual(written[name], content)
        self.assertEqual(
            leftovers, ["living_room_traj9_frei_png", "prepared"]
        )

    def test_the_result_imports_with_the_stated_camera(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            root = Path(temporary)
            source = write_sequence(
                root, poses=POSES, associations=ASSOCIATIONS
            )
            prepare_icl_nuim(source, root / "prepared")
            report = import_tum_dataset(
                root / "prepared",
                root / "prepared.vgsession",
                intrinsics=TumCameraIntrinsics(**ICL_NUIM_INTRINSICS),
            )
            session = load_scan_session(root / "prepared.vgsession")
            poses = [
                list(record.data["source_quaternion_xyzw"])
                for record in session.streams["pose"]
            ]

        self.assertEqual(report.matched_rgbd_count, 2)
        self.assertEqual(report.matched_pose_count, 2)
        self.assertEqual(report.unmatched_rgb_count, 0)
        self.assertEqual(poses, [[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, -0.5, 0.5]])

    def test_timestamps_match_by_value(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            root = Path(temporary)
            source = write_sequence(
                root,
                poses="1.0 0 0 0 0 0 0 1\n2.00 0 0 0 0 0 0 1\n",
                associations=ASSOCIATIONS,
            )
            counts = prepare_icl_nuim(source, root / "prepared")
            trajectory = (root / "prepared" / "groundtruth.txt").read_bytes()

        self.assertEqual(counts["frames"], 2)
        self.assertEqual(
            trajectory, b"1 0 0 0 1 0 0 0\n2 0 0 0 1 0 0 0\n"
        )

    def test_command_line_reports_what_it_left_out(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            root = Path(temporary)
            source = write_sequence(
                root, poses=POSES, associations=ASSOCIATIONS
            )
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main([str(source), str(root / "prepared")])
            self.assertTrue((root / "prepared" / "groundtruth.txt").is_file())

        self.assertEqual(exit_code, 0)
        self.assertIn("frames: 2 (left out, no pose: 1;", stdout.getvalue())
        self.assertIn(
            "--fx 481.2 --fy 480.0 --cx 319.5 --cy 239.5", stdout.getvalue()
        )


class PrepareRefusalTests(unittest.TestCase):
    def assert_refused(
        self,
        message: str,
        *,
        poses: str = POSES,
        associations: str = ASSOCIATIONS,
        extra_trajectory: bool = False,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            root = Path(temporary)
            source = write_sequence(
                root, poses=poses, associations=associations
            )
            if extra_trajectory:
                (source / "second.gt.freiburg").write_bytes(b"")
            with self.assertRaises(SystemExit) as raised:
                prepare_icl_nuim(source, root / "prepared")
            # Nothing published and no staging directory left behind.
            self.assertEqual(
                sorted(path.name for path in root.iterdir()),
                ["living_room_traj9_frei_png"],
            )
        self.assertIn(message, str(raised.exception))

    def test_malformed_sequences_are_refused_and_leave_nothing(self) -> None:
        self.assert_refused(
            "expected 'timestamp depth timestamp rgb'",
            associations="0 depth/0.pgm 0\n",
        )
        self.assert_refused(
            "depth and rgb timestamps differ",
            associations="0 depth/0.pgm 1 rgb/0.ppm\n",
        )
        self.assert_refused(
            "repeated timestamp",
            associations="1 depth/1.pgm 1 rgb/1.ppm\n"
            "1.0 depth/2.pgm 1.0 rgb/2.ppm\n",
        )
        self.assert_refused(
            "repeated file name",
            associations="1 depth/1.pgm 1 rgb/1.ppm\n"
            "2 depth/1.pgm 2 rgb/2.ppm\n",
        )
        self.assert_refused(
            "must be relative and stay inside",
            associations="1 ../depth/1.pgm 1 rgb/1.ppm\n",
        )
        self.assert_refused(
            "no such file",
            associations="1 depth/9.pgm 1 rgb/1.ppm\n",
        )
        self.assert_refused(
            "expected 'timestamp tx ty tz qx qy qz qw'",
            poses="1 0 0 0 0 0 1\n",
        )
        self.assert_refused("is not a number", poses="1 0 0 x 0 0 0 1\n")
        self.assert_refused("is not finite", poses="1 0 0 nan 0 0 0 1\n")
        self.assert_refused(
            "repeated timestamp", poses="1 0 0 0 0 0 0 1\n1 0 0 0 0 0 0 1\n"
        )
        self.assert_refused(
            "no frame in associations.txt has a pose",
            poses="7 0 0 0 0 0 0 1\n",
        )
        self.assert_refused(
            "expected exactly one *.gt.freiburg", extra_trajectory=True
        )

    def test_an_output_that_exists_or_sits_in_the_source_is_refused(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            root = Path(temporary)
            source = write_sequence(
                root, poses=POSES, associations=ASSOCIATIONS
            )
            existing = root / "existing"
            existing.mkdir()
            (existing / "keep.txt").write_bytes(b"keep")
            with self.assertRaisesRegex(SystemExit, "already exists"):
                prepare_icl_nuim(source, existing)
            self.assertEqual((existing / "keep.txt").read_bytes(), b"keep")
            with self.assertRaisesRegex(SystemExit, "into the source"):
                prepare_icl_nuim(source, source / "prepared")
            with self.assertRaisesRegex(SystemExit, "into the source"):
                prepare_icl_nuim(source, source)
            with self.assertRaisesRegex(SystemExit, "must be a directory"):
                prepare_icl_nuim(root / "absent", root / "prepared")
            self.assertEqual(
                sorted(path.name for path in source.iterdir()),
                ["associations.txt", "depth", "livingRoom9.gt.freiburg", "rgb"],
            )

    def test_a_sequence_without_associations_is_refused(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            root = Path(temporary)
            source = write_sequence(
                root, poses=POSES, associations=ASSOCIATIONS
            )
            (source / "associations.txt").unlink()
            with self.assertRaisesRegex(SystemExit, "missing"):
                prepare_icl_nuim(source, root / "prepared")


if __name__ == "__main__":
    unittest.main()
