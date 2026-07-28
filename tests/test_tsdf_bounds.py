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
from spatialforge.tsdf_bounds import infer_tsdf_bounds


TEST_ROOT = Path(__file__).resolve().parent
FIXTURE = TEST_ROOT / "fixtures" / "minimal.vgsession"


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


def set_pose_translation(
    fixture: Path,
    line_index: int,
    translation: tuple[float, float, float],
) -> None:
    pose_path = fixture / "streams" / "poses.jsonl"
    poses = [
        json.loads(line)
        for line in pose_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    transform = poses[line_index]["T_world_camera"]
    transform[3], transform[7], transform[11] = translation
    pose_path.write_text(
        "".join(json.dumps(pose, separators=(",", ":")) + "\n" for pose in poses),
        encoding="utf-8",
    )


def infer_fixture_bounds(
    fixture: Path = FIXTURE,
    *,
    frame_stride: int = 1,
):
    return infer_tsdf_bounds(
        load_scan_session(fixture),
        voxel_size_m=0.5,
        truncation_m=0.5,
        frame_stride=frame_stride,
    )


class TsdfBoundsTests(unittest.TestCase):
    def test_fixture_bounds_match_equivalent_manual_tsdf(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            session = load_scan_session(FIXTURE)
            bounds = infer_tsdf_bounds(
                session,
                voxel_size_m=0.5,
                truncation_m=0.5,
            )
            automatic = temporary_root / "automatic.sftsdf"
            manual = temporary_root / "manual.sftsdf"

            automatic_report = integrate_tsdf(
                session,
                automatic,
                origin_world_m=bounds.origin_world_m,
                dimensions=bounds.dimensions_xyz,
                voxel_size_m=0.5,
                truncation_m=0.5,
            )
            manual_report = integrate_tsdf(
                session,
                manual,
                origin_world_m=(0.5, -1.0, -1.0),
                dimensions=(2, 4, 4),
                voxel_size_m=0.5,
                truncation_m=0.5,
            )
            document = read_json(automatic)
            automatic_bytes = automatic.read_bytes()
            manual_bytes = manual.read_bytes()

        self.assertEqual(bounds.surface_min_world_m, (1.0, -0.25, -0.25))
        self.assertEqual(bounds.surface_max_world_m, (1.0, 0.25, 0.25))
        self.assertEqual(bounds.origin_world_m, (0.5, -1.0, -1.0))
        self.assertEqual(bounds.upper_world_m, (1.5, 1.0, 1.0))
        self.assertEqual(bounds.dimensions_xyz, (2, 4, 4))
        self.assertEqual(bounds.total_voxels, 32)
        self.assertEqual(bounds.total_observations, 2)
        self.assertEqual(bounds.selected_observations, 2)
        self.assertEqual(bounds.paired_observations, 2)
        self.assertEqual(bounds.valid_depth_points, 8)
        self.assertEqual(bounds.invalid_depth_samples, 0)
        self.assertEqual(automatic_bytes, manual_bytes)
        self.assertEqual(
            automatic_report.output_digest_sha256,
            manual_report.output_digest_sha256,
        )
        self.assertEqual(
            automatic_report.output_digest_sha256,
            "e61803737cdd68b209459fb644cc2f67f18e0420d306273316808e27d2e89994",
        )
        self.assertEqual(document["volume"]["origin_world_m"], [0.5, -1.0, -1.0])
        self.assertEqual(document["volume"]["dimensions_xyz"], [2, 4, 4])
        self.assertEqual(automatic_report.observed_voxels, 8)
        self.assertEqual(automatic_report.fused_voxels, 8)
        self.assertEqual(automatic_report.voxel_updates, 16)
        self.assertEqual(automatic_report.max_weight, 2)

    def test_exact_negative_grid_boundaries_do_not_add_voxels(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            fixture = copy_fixture(Path(temporary_directory))
            set_pose_translation(fixture, 0, (-2.0, -0.25, 0.25))

            bounds = infer_fixture_bounds(fixture, frame_stride=2)

        self.assertEqual(bounds.origin_world_m, (-1.5, -1.0, -0.5))
        self.assertEqual(bounds.upper_world_m, (-0.5, 0.5, 1.0))
        self.assertEqual(bounds.dimensions_xyz, (2, 3, 3))

    def test_off_grid_negative_bounds_snap_outward(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            fixture = copy_fixture(Path(temporary_directory))
            set_pose_translation(fixture, 0, (-1.76, -0.13, 0.12))

            bounds = infer_fixture_bounds(fixture, frame_stride=2)

        self.assertEqual(bounds.origin_world_m, (-1.5, -1.0, -1.0))
        self.assertEqual(bounds.upper_world_m, (0.0, 1.0, 1.0))
        self.assertEqual(bounds.dimensions_xyz, (3, 4, 4))
        self.assertEqual(math.copysign(1.0, bounds.upper_world_m[0]), 1.0)

    def test_padding_uses_truncation_not_one_voxel(self) -> None:
        bounds = infer_tsdf_bounds(
            load_scan_session(FIXTURE),
            voxel_size_m=0.25,
            truncation_m=0.5,
        )

        self.assertEqual(bounds.origin_world_m, (0.5, -0.75, -0.75))
        self.assertEqual(bounds.upper_world_m, (1.5, 0.75, 0.75))
        self.assertEqual(bounds.dimensions_xyz, (4, 6, 6))
        self.assertEqual(bounds.total_voxels, 144)
        self.assertEqual(bounds.padding_m, 0.5)

    def test_frame_stride_controls_bound_discovery(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            fixture = copy_fixture(Path(temporary_directory))
            set_pose_translation(fixture, 1, (2.05, 0.0, 0.0))

            all_frames = infer_fixture_bounds(fixture)
            first_frame = infer_fixture_bounds(fixture, frame_stride=2)

        self.assertEqual(all_frames.origin_world_m, (0.5, -1.0, -1.0))
        self.assertEqual(all_frames.dimensions_xyz, (6, 4, 4))
        self.assertEqual(all_frames.selected_observations, 2)
        self.assertEqual(all_frames.paired_observations, 2)
        self.assertEqual(first_frame.dimensions_xyz, (2, 4, 4))
        self.assertEqual(first_frame.selected_observations, 1)
        self.assertEqual(first_frame.paired_observations, 1)

    def test_invalid_depth_samples_are_excluded_from_bounds(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            fixture = copy_fixture(Path(temporary_directory))
            (fixture / "data" / "depth" / "000000.pgm").write_text(
                "P2\n2 2\n65535\n0 500\n1000 2000\n",
                encoding="ascii",
            )

            bounds = infer_fixture_bounds(fixture, frame_stride=2)

        self.assertEqual(bounds.origin_world_m, (0.0, -1.0, -1.0))
        self.assertEqual(bounds.dimensions_xyz, (5, 4, 4))
        self.assertEqual(bounds.valid_depth_points, 3)
        self.assertEqual(bounds.invalid_depth_samples, 1)

    def test_missing_depth_and_pose_are_counted_and_skipped(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            missing_depth = copy_fixture(temporary_root, "depth-gap.vgsession")
            depth_path = missing_depth / "streams" / "depth.jsonl"
            depth_path.write_text(
                depth_path.read_text(encoding="utf-8").splitlines()[0] + "\n",
                encoding="utf-8",
            )
            depth_bounds = infer_fixture_bounds(missing_depth)

            missing_pose = copy_fixture(temporary_root, "pose-gap.vgsession")
            pose_path = missing_pose / "streams" / "poses.jsonl"
            pose_path.write_text(
                pose_path.read_text(encoding="utf-8").splitlines()[0] + "\n",
                encoding="utf-8",
            )
            pose_bounds = infer_fixture_bounds(missing_pose)

        self.assertEqual(depth_bounds.skipped_missing_depth, 1)
        self.assertEqual(depth_bounds.skipped_missing_pose, 0)
        self.assertEqual(depth_bounds.paired_observations, 1)
        self.assertEqual(depth_bounds.dimensions_xyz, (2, 4, 4))
        self.assertEqual(pose_bounds.skipped_missing_depth, 0)
        self.assertEqual(pose_bounds.skipped_missing_pose, 1)
        self.assertEqual(pose_bounds.paired_observations, 1)
        self.assertEqual(pose_bounds.dimensions_xyz, (2, 4, 4))

    def test_no_usable_depth_fails_actionably(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)

            no_pose = copy_fixture(temporary_root, "no-pose.vgsession")
            manifest_path = no_pose / "manifest.json"
            manifest = read_json(manifest_path)
            del manifest["streams"]["pose"]
            write_json(manifest_path, manifest)

            zero_depth = copy_fixture(temporary_root, "zero.vgsession")
            for filename in ("000000.pgm", "000001.pgm"):
                (zero_depth / "data" / "depth" / filename).write_text(
                    "P2\n2 2\n65535\n0 0\n0 0\n",
                    encoding="ascii",
                )

            overflow = copy_fixture(temporary_root, "overflow.vgsession")
            overflow_manifest_path = overflow / "manifest.json"
            overflow_manifest = read_json(overflow_manifest_path)
            overflow_manifest["streams"]["depth"]["depth_scale_m"] = 1e308
            write_json(overflow_manifest_path, overflow_manifest)

            cases = (
                (no_pose, "both exact depth and pose"),
                (zero_depth, "no positive finite depth samples"),
                (overflow, "no positive finite depth samples"),
            )
            for fixture, expected in cases:
                with self.subTest(fixture=fixture.name):
                    with self.assertRaises(TsdfError) as raised:
                        infer_fixture_bounds(fixture)
                    self.assertIn(expected, str(raised.exception))

    def test_derived_volume_respects_dense_voxel_limit(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            fixture = copy_fixture(Path(temporary_directory))
            set_pose_translation(fixture, 1, (500000.0, 0.0, 0.0))

            with self.assertRaises(TsdfError) as raised:
                infer_fixture_bounds(fixture)

        self.assertIn("16000032 voxels", str(raised.exception))
        self.assertIn(f"maximum is {MAX_REFERENCE_VOXELS}", str(raised.exception))
        self.assertIn("Increase --voxel-size-m", str(raised.exception))

    def test_exact_dense_voxel_cap_is_allowed(self) -> None:
        with patch("spatialforge.tsdf_bounds.MAX_REFERENCE_VOXELS", 32):
            bounds = infer_fixture_bounds()
        self.assertEqual(bounds.total_voxels, 32)

        with patch("spatialforge.tsdf_bounds.MAX_REFERENCE_VOXELS", 31):
            with self.assertRaises(TsdfError) as raised:
                infer_fixture_bounds()
        self.assertIn("32 voxels", str(raised.exception))
        self.assertIn("maximum is 31", str(raised.exception))

    def test_parameter_validation_is_actionable(self) -> None:
        session = load_scan_session(FIXTURE)
        cases = (
            (
                {"voxel_size_m": 0.0, "truncation_m": 0.5},
                "voxel_size_m",
            ),
            (
                {"voxel_size_m": math.nan, "truncation_m": 0.5},
                "voxel_size_m",
            ),
            (
                {"voxel_size_m": 0.5, "truncation_m": math.inf},
                "truncation_m",
            ),
            (
                {"voxel_size_m": 0.5, "truncation_m": 0.25},
                "greater than or equal",
            ),
            (
                {
                    "voxel_size_m": 0.5,
                    "truncation_m": 0.5,
                    "frame_stride": 0,
                },
                "frame_stride",
            ),
            (
                {
                    "voxel_size_m": 0.5,
                    "truncation_m": 0.5,
                    "frame_stride": True,
                },
                "frame_stride",
            ),
        )
        for arguments, expected in cases:
            with self.subTest(arguments=arguments):
                with self.assertRaises(TsdfError) as raised:
                    infer_tsdf_bounds(session, **arguments)
                self.assertIn(expected, str(raised.exception))

    def test_auto_cli_is_deterministic_and_preserves_outputs(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            first = temporary_root / "first.sftsdf"
            second = temporary_root / "second.sftsdf"
            def arguments(output: Path) -> list[str]:
                return [
                    "reconstruct",
                    "tsdf-auto",
                    str(FIXTURE),
                    str(output),
                    "--voxel-size-m",
                    "0.5",
                    "--truncation-m",
                    "0.5",
                ]

            with redirect_stdout(io.StringIO()):
                first_exit = main(arguments(first))
                second_exit = main(arguments(second))
            self.assertEqual(first_exit, 0)
            self.assertEqual(second_exit, 0)
            self.assertEqual(first.read_bytes(), second.read_bytes())

            existing = temporary_root / "existing.sftsdf"
            existing.write_text("keep", encoding="ascii")
            existing_stdout = io.StringIO()
            existing_stderr = io.StringIO()
            with patch(
                "spatialforge.cli.infer_tsdf_bounds"
            ) as infer_mock:
                with (
                    redirect_stdout(existing_stdout),
                    redirect_stderr(existing_stderr),
                ):
                    existing_exit = main(arguments(existing))
                infer_mock.assert_not_called()
            self.assertEqual(existing_exit, 2)
            self.assertEqual(existing_stdout.getvalue(), "")
            self.assertIn("output already exists", existing_stderr.getvalue())
            self.assertEqual(existing.read_text(encoding="ascii"), "keep")

            raced = temporary_root / "raced.sftsdf"
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
                raced_stdout = io.StringIO()
                raced_stderr = io.StringIO()
                with (
                    redirect_stdout(raced_stdout),
                    redirect_stderr(raced_stderr),
                ):
                    raced_exit = main(arguments(raced))

            self.assertEqual(raced_exit, 2)
            self.assertEqual(raced_stdout.getvalue(), "")
            self.assertIn("refusing to overwrite", raced_stderr.getvalue())
            self.assertEqual(raced.read_text(encoding="ascii"), "competitor")

            invalid = temporary_root / "invalid.json"
            invalid_stderr = io.StringIO()
            with patch(
                "spatialforge.cli.infer_tsdf_bounds"
            ) as infer_mock:
                with redirect_stderr(invalid_stderr):
                    invalid_exit = main(arguments(invalid))
                infer_mock.assert_not_called()
            self.assertEqual(invalid_exit, 2)
            self.assertIn("must end in .sftsdf", invalid_stderr.getvalue())
            self.assertFalse(invalid.exists())

    def test_changed_input_between_bounds_and_fusion_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            fixture = copy_fixture(temporary_root)
            output = temporary_root / "changed.sftsdf"
            actual_infer = infer_tsdf_bounds

            def infer_then_change(session, **arguments):
                bounds = actual_infer(session, **arguments)
                (fixture / "data" / "depth" / "000001.pgm").write_text(
                    "P2\n2 2\n65535\n2000 2000\n2000 2000\n",
                    encoding="ascii",
                )
                return bounds

            stderr = io.StringIO()
            with (
                patch(
                    "spatialforge.cli.infer_tsdf_bounds",
                    side_effect=infer_then_change,
                ),
                redirect_stderr(stderr),
            ):
                exit_code = main(
                    [
                        "reconstruct",
                        "tsdf-auto",
                        str(fixture),
                        str(output),
                        "--voxel-size-m",
                        "0.5",
                        "--truncation-m",
                        "0.5",
                    ]
                )

            self.assertEqual(exit_code, 2)
            self.assertIn(
                "inputs changed after automatic bounds",
                stderr.getvalue(),
            )
            self.assertFalse(output.exists())


class TsdfBoundsCliTests(unittest.TestCase):
    def test_cli_infers_bounds_and_integrates_tsdf(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            output = Path(temporary_directory) / "automatic.sftsdf"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "reconstruct",
                        "tsdf-auto",
                        str(FIXTURE),
                        str(output),
                        "--voxel-size-m",
                        "0.5",
                        "--truncation-m",
                        "0.5",
                    ]
                )
            output_exists = output.is_file()

        self.assertEqual(exit_code, 0)
        self.assertTrue(output_exists)
        self.assertIn("AUTO TSDF scan-synthetic-0001", stdout.getvalue())
        self.assertIn("bounds_depth: valid=8 invalid=0", stdout.getvalue())
        self.assertIn(
            "origin=(0.500000000, -1.000000000, -1.000000000) "
            "dimensions=(2, 4, 4) voxels=32",
            stdout.getvalue(),
        )
        self.assertIn(
            "integration: observed=8 fused=8 updates=16 max_weight=2",
            stdout.getvalue(),
        )
        self.assertIn(
            "output_sha256: "
            "e61803737cdd68b209459fb644cc2f67f18e0420d306273316808e27d2e89994",
            stdout.getvalue(),
        )

    def test_cli_failure_is_actionable(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            output = Path(temporary_directory) / "missing.sftsdf"
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                exit_code = main(
                    [
                        "reconstruct",
                        "tsdf-auto",
                        str(TEST_ROOT / "missing.vgsession"),
                        str(output),
                    ]
                )

        self.assertEqual(exit_code, 2)
        self.assertIn("AUTO TSDF FAILED", stderr.getvalue())
        self.assertIn("directory does not exist", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_cli_frame_stride_controls_bounds_and_integration(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            fixture = copy_fixture(temporary_root)
            set_pose_translation(fixture, 1, (2.05, 0.0, 0.0))
            output = temporary_root / "stride.sftsdf"
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "reconstruct",
                        "tsdf-auto",
                        str(fixture),
                        str(output),
                        "--voxel-size-m",
                        "0.5",
                        "--truncation-m",
                        "0.5",
                        "--frame-stride",
                        "2",
                    ]
                )
            document = read_json(output)

        self.assertEqual(exit_code, 0)
        self.assertIn(
            "bounds_frames: total=2 selected=1 paired=1",
            stdout.getvalue(),
        )
        self.assertIn("dimensions=(2, 4, 4)", stdout.getvalue())
        self.assertIn(
            "integration: observed=8 fused=0 updates=8 max_weight=1",
            stdout.getvalue(),
        )
        self.assertEqual(document["integration"]["frame_stride"], 2)
        self.assertEqual(document["integration"]["selected_observations"], 1)
        self.assertEqual(document["integration"]["integrated_frames"], 1)


if __name__ == "__main__":
    unittest.main()
