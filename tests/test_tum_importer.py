from __future__ import annotations

import io
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


if __name__ == "__main__":
    unittest.main()
