from __future__ import annotations

import io
import json
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from spatialforge.cli import main
from spatialforge.errors import PointCloudError
from spatialforge.point_cloud import reconstruct_point_cloud
from spatialforge.session_loader import load_scan_session


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


def parse_ply(path: Path) -> tuple[list[str], list[tuple[float, ...]]]:
    lines = path.read_text(encoding="ascii").splitlines()
    end_header = lines.index("end_header")
    vertices = [
        tuple(float(value) for value in line.split())
        for line in lines[end_header + 1 :]
        if line
    ]
    return lines[: end_header + 1], vertices


class PointCloudTests(unittest.TestCase):
    def test_first_frame_has_exact_world_vertices_colors_and_header(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            output = Path(temporary_directory) / "first-frame.ply"
            report = reconstruct_point_cloud(
                load_scan_session(FIXTURE),
                output,
                frame_stride=2,
            )
            header, vertices = parse_ply(output)

        self.assertEqual(report.selected_observations, 1)
        self.assertEqual(report.integrated_frames, 1)
        self.assertEqual(report.points_written, 4)
        self.assertIn("element vertex 4", header)
        property_start = header.index("property double x")
        self.assertEqual(
            header[property_start:],
            [
                "property double x",
                "property double y",
                "property double z",
                "property uchar red",
                "property uchar green",
                "property uchar blue",
                "end_header",
            ],
        )
        self.assertEqual(
            vertices,
            [
                (1.0, 0.25, 0.25, 255.0, 0.0, 0.0),
                (1.0, -0.25, 0.25, 0.0, 255.0, 0.0),
                (1.0, 0.25, -0.25, 0.0, 0.0, 255.0),
                (1.0, -0.25, -0.25, 255.0, 255.0, 255.0),
            ],
        )

    def test_depth_scale_is_applied_and_zero_depth_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            fixture = copy_fixture(temporary_root)
            depth_path = fixture / "data" / "depth" / "000000.pgm"
            depth_path.write_text(
                "P2\n2 2\n65535\n0 500\n1000 2000\n",
                encoding="ascii",
            )
            output = temporary_root / "zero-depth.ply"

            report = reconstruct_point_cloud(
                load_scan_session(fixture),
                output,
                frame_stride=2,
            )
            _, vertices = parse_ply(output)

        self.assertEqual(report.invalid_depth_samples, 1)
        self.assertEqual(report.points_written, 3)
        self.assertEqual(
            vertices,
            [
                (0.5, -0.125, 0.125, 0.0, 255.0, 0.0),
                (1.0, 0.25, -0.25, 0.0, 0.0, 255.0),
                (2.0, -0.5, -0.5, 255.0, 255.0, 255.0),
            ],
        )

    def test_frame_and_pixel_stride_are_zero_based(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            output = Path(temporary_directory) / "strided.ply"
            report = reconstruct_point_cloud(
                load_scan_session(FIXTURE),
                output,
                frame_stride=2,
                pixel_stride=2,
            )
            _, vertices = parse_ply(output)

        self.assertEqual(report.total_observations, 2)
        self.assertEqual(report.selected_observations, 1)
        self.assertEqual(report.integrated_frames, 1)
        self.assertEqual(report.points_written, 1)
        self.assertEqual(vertices[0], (1.0, 0.25, 0.25, 255.0, 0.0, 0.0))

    def test_missing_pose_is_skipped_and_never_replaced_with_identity(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            fixture = copy_fixture(temporary_root)
            pose_path = fixture / "streams" / "poses.jsonl"
            first_pose = pose_path.read_text(encoding="utf-8").splitlines()[0]
            pose_path.write_text(first_pose + "\n", encoding="utf-8")
            output = temporary_root / "pose-gap.ply"

            report = reconstruct_point_cloud(
                load_scan_session(fixture),
                output,
            )

        self.assertEqual(report.integrated_frames, 1)
        self.assertEqual(report.skipped_missing_pose, 1)
        self.assertEqual(report.points_written, 4)

    def test_no_known_pose_fails_without_partial_output(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            fixture = copy_fixture(temporary_root)
            manifest_path = fixture / "manifest.json"
            manifest = read_json(manifest_path)
            del manifest["streams"]["pose"]
            write_json(manifest_path, manifest)
            output = temporary_root / "no-pose.ply"

            with self.assertRaises(PointCloudError) as raised:
                reconstruct_point_cloud(load_scan_session(fixture), output)

            self.assertFalse(output.exists())

        self.assertIn("no selected RGB observation", str(raised.exception))

    def test_output_is_byte_deterministic(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            first_output = temporary_root / "first.ply"
            second_output = temporary_root / "second.ply"

            first_report = reconstruct_point_cloud(
                load_scan_session(FIXTURE), first_output
            )
            second_report = reconstruct_point_cloud(
                load_scan_session(FIXTURE), second_output
            )

            self.assertEqual(first_output.read_bytes(), second_output.read_bytes())

        self.assertEqual(
            first_report.output_digest_sha256,
            "097a3bb73c9a28dce65ae926c981d19ee303538e6a4d539b6ab472bee08813b3",
        )
        self.assertEqual(
            first_report.output_digest_sha256,
            second_report.output_digest_sha256,
        )

    def test_real_16_bit_png_depth_and_rgb_are_decoded(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            fixture = copy_fixture(temporary_root)
            for frame_index in range(2):
                rgb_source = fixture / "data" / "rgb" / f"{frame_index:06d}.ppm"
                depth_source = (
                    fixture / "data" / "depth" / f"{frame_index:06d}.pgm"
                )
                rgb_target = rgb_source.with_suffix(".png")
                depth_target = depth_source.with_suffix(".png")
                with Image.open(rgb_source) as rgb_image:
                    rgb_image.save(rgb_target, format="PNG")
                with Image.open(depth_source) as depth_image:
                    depth_image.convert("I;16").save(
                        depth_target, format="PNG"
                    )

            for stream_name, old_suffix in (("rgb", ".ppm"), ("depth", ".pgm")):
                index_path = fixture / "streams" / f"{stream_name}.jsonl"
                index_path.write_text(
                    index_path.read_text(encoding="utf-8").replace(
                        old_suffix, ".png"
                    ),
                    encoding="utf-8",
                )

            output = temporary_root / "png.ply"
            report = reconstruct_point_cloud(load_scan_session(fixture), output)

        self.assertEqual(report.integrated_frames, 2)
        self.assertEqual(report.points_written, 8)

    def test_dimension_mismatch_is_actionable_and_atomic(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            fixture = copy_fixture(temporary_root)
            calibration_path = fixture / "calibration" / "cameras.json"
            calibration = read_json(calibration_path)
            calibration["cameras"][0]["width"] = 3
            write_json(calibration_path, calibration)
            output = temporary_root / "wrong-size.ply"

            with self.assertRaises(PointCloudError) as raised:
                reconstruct_point_cloud(load_scan_session(fixture), output)

            self.assertFalse(output.exists())

        self.assertIn("expected 3x2", str(raised.exception))

    def test_non_finite_projection_is_rejected_without_output(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            fixture = copy_fixture(temporary_root)
            calibration_path = fixture / "calibration" / "cameras.json"
            calibration = read_json(calibration_path)
            calibration["cameras"][0]["intrinsics"]["fx"] = 5e-324
            write_json(calibration_path, calibration)
            output = temporary_root / "overflow.ply"

            with self.assertRaises(PointCloudError) as raised:
                reconstruct_point_cloud(
                    load_scan_session(fixture),
                    output,
                    frame_stride=2,
                )

            self.assertFalse(output.exists())

        self.assertIn("non-finite coordinate", str(raised.exception))

    def test_unaligned_depth_and_distortion_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)

            unaligned_fixture = copy_fixture(
                temporary_root, "unaligned.vgsession"
            )
            manifest_path = unaligned_fixture / "manifest.json"
            manifest = read_json(manifest_path)
            del manifest["streams"]["depth"]["aligned_to"]
            write_json(manifest_path, manifest)
            with self.assertRaises(PointCloudError) as unaligned:
                reconstruct_point_cloud(
                    load_scan_session(unaligned_fixture),
                    temporary_root / "unaligned.ply",
                )

            distorted_fixture = copy_fixture(
                temporary_root, "distorted.vgsession"
            )
            calibration_path = (
                distorted_fixture / "calibration" / "cameras.json"
            )
            calibration = read_json(calibration_path)
            calibration["cameras"][0]["distortion"] = {
                "model": "opencv-radtan",
                "coefficients": [0, 0, 0, 0, 0],
            }
            write_json(calibration_path, calibration)
            with self.assertRaises(PointCloudError) as distorted:
                reconstruct_point_cloud(
                    load_scan_session(distorted_fixture),
                    temporary_root / "distorted.ply",
                )

        self.assertIn("aligned_to='rgb'", str(unaligned.exception))
        self.assertIn("unsupported until undistortion", str(distorted.exception))

    def test_existing_output_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            output = Path(temporary_directory) / "existing.ply"
            output.write_text("keep", encoding="ascii")

            with self.assertRaises(PointCloudError):
                reconstruct_point_cloud(load_scan_session(FIXTURE), output)

            self.assertEqual(output.read_text(encoding="ascii"), "keep")

    def test_racing_output_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            output = Path(temporary_directory) / "raced.ply"
            actual_link = os.link

            def create_competing_output(
                staging_path: str | Path,
                output_path: str | Path,
            ) -> None:
                Path(output_path).write_text("competitor", encoding="ascii")
                actual_link(staging_path, output_path)

            with patch(
                "spatialforge.point_cloud.os.link",
                side_effect=create_competing_output,
            ):
                with self.assertRaises(PointCloudError) as raised:
                    reconstruct_point_cloud(load_scan_session(FIXTURE), output)

            self.assertEqual(output.read_text(encoding="ascii"), "competitor")

        self.assertIn("refusing to overwrite", str(raised.exception))

    def test_pillow_safety_limit_is_reported_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            output = Path(temporary_directory) / "limited.ply"
            previous_limit = Image.MAX_IMAGE_PIXELS
            try:
                Image.MAX_IMAGE_PIXELS = 1
                with self.assertRaises(PointCloudError) as raised:
                    reconstruct_point_cloud(load_scan_session(FIXTURE), output)
            finally:
                Image.MAX_IMAGE_PIXELS = previous_limit

            self.assertFalse(output.exists())

        self.assertIn("cannot decode RGB image", str(raised.exception))


class PointCloudCliTests(unittest.TestCase):
    def test_cli_reconstructs_and_reports_counts(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            output = Path(temporary_directory) / "cli.ply"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "reconstruct",
                        "point-cloud",
                        str(FIXTURE),
                        str(output),
                        "--frame-stride",
                        "2",
                    ]
                )
            output_exists = output.is_file()

        self.assertEqual(exit_code, 0)
        self.assertTrue(output_exists)
        self.assertIn("POINT CLOUD scan-synthetic-0001", stdout.getvalue())
        self.assertIn("integrated=1", stdout.getvalue())
        self.assertIn("points: 4", stdout.getvalue())

    def test_cli_reconstruction_failure_is_actionable(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            output = Path(temporary_directory) / "missing.ply"
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                exit_code = main(
                    [
                        "reconstruct",
                        "point-cloud",
                        str(TEST_ROOT / "missing.vgsession"),
                        str(output),
                    ]
                )

        self.assertEqual(exit_code, 2)
        self.assertIn("RECONSTRUCT FAILED", stderr.getvalue())
        self.assertIn("directory does not exist", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
