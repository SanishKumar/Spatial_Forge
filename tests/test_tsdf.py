from __future__ import annotations

import io
import json
import math
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from spatialforge.cli import main
from spatialforge.errors import TsdfError
from spatialforge.session_loader import load_scan_session
from spatialforge.tsdf import MAX_REFERENCE_VOXELS, integrate_tsdf


TEST_ROOT = Path(__file__).resolve().parent
FIXTURE = TEST_ROOT / "fixtures" / "minimal.vgsession"
REFERENCE_ARGUMENTS = {
    "origin_world_m": (0.0, -0.25, -0.25),
    "dimensions": (4, 1, 1),
    "voxel_size_m": 0.5,
    "truncation_m": 0.5,
}


def copy_fixture(parent: Path, name: str = "case.vgsession") -> Path:
    target = parent / name
    shutil.copytree(FIXTURE, target)
    return target


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2) + "\n",
        encoding="utf-8",
    )


class TsdfTests(unittest.TestCase):
    def test_two_frames_fuse_into_exact_plane_tsdf(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            output = Path(temporary_directory) / "plane.sftsdf"
            report = integrate_tsdf(
                load_scan_session(FIXTURE),
                output,
                **REFERENCE_ARGUMENTS,
            )
            document = read_json(output)

        self.assertEqual(report.integrated_frames, 2)
        self.assertEqual(report.total_voxels, 4)
        self.assertEqual(report.observed_voxels, 3)
        self.assertEqual(report.fused_voxels, 3)
        self.assertEqual(report.voxel_updates, 6)
        self.assertEqual(report.max_weight, 2)
        self.assertEqual(
            document["voxels"],
            [
                {"index": [0, 0, 0], "tsdf": 1.0, "weight": 2},
                {"index": [1, 0, 0], "tsdf": 0.5, "weight": 2},
                {"index": [2, 0, 0], "tsdf": -0.5, "weight": 2},
            ],
        )
        self.assertEqual(document["integration"]["unknown_voxels"], 1)
        self.assertEqual(
            report.output_digest_sha256,
            "7d3f30121fa5537a4f77d881b12087d04e5805a84066598c6a62f988e3247495",
        )

    def test_frame_stride_integrates_only_first_frame(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            output = Path(temporary_directory) / "one-frame.sftsdf"
            report = integrate_tsdf(
                load_scan_session(FIXTURE),
                output,
                frame_stride=2,
                **REFERENCE_ARGUMENTS,
            )
            document = read_json(output)

        self.assertEqual(report.selected_observations, 1)
        self.assertEqual(report.integrated_frames, 1)
        self.assertEqual(report.fused_voxels, 0)
        self.assertEqual(report.voxel_updates, 3)
        self.assertEqual(report.max_weight, 1)
        self.assertEqual(
            [voxel["weight"] for voxel in document["voxels"]],
            [1, 1, 1],
        )

    def test_missing_pose_is_skipped_without_identity_fallback(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            fixture = copy_fixture(temporary_root)
            pose_path = fixture / "streams" / "poses.jsonl"
            first_pose = pose_path.read_text(encoding="utf-8").splitlines()[0]
            pose_path.write_text(first_pose + "\n", encoding="utf-8")
            output = temporary_root / "pose-gap.sftsdf"

            report = integrate_tsdf(
                load_scan_session(fixture),
                output,
                **REFERENCE_ARGUMENTS,
            )

        self.assertEqual(report.integrated_frames, 1)
        self.assertEqual(report.skipped_missing_pose, 1)
        self.assertEqual(report.max_weight, 1)

    def test_no_known_pose_fails_without_partial_output(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            fixture = copy_fixture(temporary_root)
            manifest_path = fixture / "manifest.json"
            manifest = read_json(manifest_path)
            del manifest["streams"]["pose"]
            write_json(manifest_path, manifest)
            output = temporary_root / "no-pose.sftsdf"

            with self.assertRaises(TsdfError) as raised:
                integrate_tsdf(
                    load_scan_session(fixture),
                    output,
                    **REFERENCE_ARGUMENTS,
                )

            self.assertFalse(output.exists())

        self.assertIn("no selected RGB observation", str(raised.exception))

    def test_zero_depth_is_invalid_and_not_fused(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            fixture = copy_fixture(temporary_root)
            depth_path = fixture / "data" / "depth" / "000000.pgm"
            depth_path.write_text(
                "P2\n2 2\n65535\n0 0\n0 0\n",
                encoding="ascii",
            )
            output = temporary_root / "zero-depth.sftsdf"

            report = integrate_tsdf(
                load_scan_session(fixture),
                output,
                **REFERENCE_ARGUMENTS,
            )

        self.assertEqual(report.invalid_depth_pixels, 4)
        self.assertEqual(report.fused_voxels, 0)
        self.assertEqual(report.max_weight, 1)

    def test_non_finite_scaled_depth_fails_cleanly(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            fixture = copy_fixture(temporary_root)
            manifest_path = fixture / "manifest.json"
            manifest = read_json(manifest_path)
            manifest["streams"]["depth"]["depth_scale_m"] = 1e308
            write_json(manifest_path, manifest)
            output = temporary_root / "overflow-depth.sftsdf"

            with self.assertRaises(TsdfError) as raised:
                integrate_tsdf(
                    load_scan_session(fixture),
                    output,
                    **REFERENCE_ARGUMENTS,
                )

            self.assertFalse(output.exists())

        self.assertIn("do not observe any voxel", str(raised.exception))

    def test_volume_outside_view_fails_without_output(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            output = Path(temporary_directory) / "outside.sftsdf"

            with self.assertRaises(TsdfError) as raised:
                integrate_tsdf(
                    load_scan_session(FIXTURE),
                    output,
                    origin_world_m=(-4.0, -0.25, -0.25),
                    dimensions=(2, 1, 1),
                    voxel_size_m=0.5,
                    truncation_m=0.5,
                )

            self.assertFalse(output.exists())

        self.assertIn("do not observe any voxel", str(raised.exception))

    def test_parameter_validation_is_bounded_and_actionable(self) -> None:
        session = load_scan_session(FIXTURE)
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            invalid_cases = (
                (
                    {"origin_world_m": None},
                    "origin_world_m",
                ),
                (
                    {"origin_world_m": (math.inf, 0.0, 0.0)},
                    "origin_world_m",
                ),
                (
                    {"origin_world_m": (10**400, 0.0, 0.0)},
                    "origin_world_m",
                ),
                (
                    {"dimensions": (0, 1, 1)},
                    "dimensions",
                ),
                (
                    {
                        "dimensions": (MAX_REFERENCE_VOXELS + 1, 1, 1),
                    },
                    "maximum",
                ),
                (
                    {"truncation_m": 0.25},
                    "greater than or equal",
                ),
                (
                    {"voxel_size_m": 10**400},
                    "voxel_size_m",
                ),
            )
            for index, (overrides, expected_message) in enumerate(invalid_cases):
                arguments = dict(REFERENCE_ARGUMENTS)
                arguments.update(overrides)
                with self.subTest(overrides=overrides):
                    with self.assertRaises(TsdfError) as raised:
                        integrate_tsdf(
                            session,
                            temporary_root / f"invalid-{index}.sftsdf",
                            **arguments,
                        )
                    self.assertIn(expected_message, str(raised.exception))

    def test_output_is_byte_deterministic_and_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            first = temporary_root / "first.sftsdf"
            second = temporary_root / "second.sftsdf"
            integrate_tsdf(
                load_scan_session(FIXTURE),
                first,
                **REFERENCE_ARGUMENTS,
            )
            integrate_tsdf(
                load_scan_session(FIXTURE),
                second,
                **REFERENCE_ARGUMENTS,
            )
            self.assertEqual(first.read_bytes(), second.read_bytes())

            existing = temporary_root / "existing.sftsdf"
            existing.write_text("keep", encoding="ascii")
            with self.assertRaises(TsdfError):
                integrate_tsdf(
                    load_scan_session(FIXTURE),
                    existing,
                    **REFERENCE_ARGUMENTS,
                )
            self.assertEqual(existing.read_text(encoding="ascii"), "keep")

    def test_racing_output_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            output = Path(temporary_directory) / "raced.sftsdf"
            actual_link = os.link

            def create_competing_output(
                staging_path: str | Path,
                output_path: str | Path,
            ) -> None:
                Path(output_path).write_text("competitor", encoding="ascii")
                actual_link(staging_path, output_path)

            with patch(
                "spatialforge.tsdf.os.link",
                side_effect=create_competing_output,
            ):
                with self.assertRaises(TsdfError) as raised:
                    integrate_tsdf(
                        load_scan_session(FIXTURE),
                        output,
                        **REFERENCE_ARGUMENTS,
                    )

            self.assertEqual(output.read_text(encoding="ascii"), "competitor")

        self.assertIn("refusing to overwrite", str(raised.exception))


class TsdfCliTests(unittest.TestCase):
    def test_cli_reports_numerical_fusion_evidence(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            output = Path(temporary_directory) / "cli.sftsdf"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "reconstruct",
                        "tsdf",
                        str(FIXTURE),
                        str(output),
                        "--origin",
                        "0",
                        "-0.25",
                        "-0.25",
                        "--dimensions",
                        "4",
                        "1",
                        "1",
                        "--voxel-size-m",
                        "0.5",
                        "--truncation-m",
                        "0.5",
                    ]
                )
            output_exists = output.is_file()

        self.assertEqual(exit_code, 0)
        self.assertTrue(output_exists)
        self.assertIn("TSDF scan-synthetic-0001", stdout.getvalue())
        self.assertIn("observed=3 fused=3", stdout.getvalue())
        self.assertIn("voxel_updates: 6 max_weight=2", stdout.getvalue())

    def test_cli_failure_is_actionable(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            output = Path(temporary_directory) / "missing.sftsdf"
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                exit_code = main(
                    [
                        "reconstruct",
                        "tsdf",
                        str(TEST_ROOT / "missing.vgsession"),
                        str(output),
                        "--origin",
                        "0",
                        "0",
                        "0",
                        "--dimensions",
                        "1",
                        "1",
                        "1",
                    ]
                )

        self.assertEqual(exit_code, 2)
        self.assertIn("TSDF FAILED", stderr.getvalue())
        self.assertIn("directory does not exist", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
