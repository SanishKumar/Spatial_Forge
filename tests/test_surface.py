from __future__ import annotations

import copy
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from spatialforge.cli import main
from spatialforge.errors import SurfaceExtractionError
from spatialforge.session_loader import load_scan_session
from spatialforge.surface import extract_surface_points
from spatialforge.tsdf import integrate_tsdf


TEST_ROOT = Path(__file__).resolve().parent
FIXTURE = TEST_ROOT / "fixtures" / "minimal.vgsession"
REFERENCE_ARGUMENTS = {
    "origin_world_m": (0.0, -0.25, -0.25),
    "dimensions": (4, 1, 1),
    "voxel_size_m": 0.5,
    "truncation_m": 0.5,
}


def parse_ply(path: Path) -> tuple[list[str], list[tuple[float, ...]]]:
    lines = path.read_text(encoding="ascii").splitlines()
    end_header = lines.index("end_header")
    vertices = [
        tuple(float(value) for value in line.split())
        for line in lines[end_header + 1 :]
        if line
    ]
    return lines[: end_header + 1], vertices


def build_artifact(
    path: Path,
    *,
    origin: tuple[float, float, float],
    dimensions: tuple[int, int, int],
    voxel_size: float,
    voxels: list[dict],
) -> dict:
    total_voxels = dimensions[0] * dimensions[1] * dimensions[2]
    integrated_frames = max(voxel["weight"] for voxel in voxels)
    document = {
        "schema": "spatialforge.reference-tsdf",
        "schema_version": "0.1.0",
        "session_id": "surface-synthetic",
        "replay_digest_sha256": "a" * 64,
        "volume": {
            "origin_world_m": list(origin),
            "dimensions_xyz": list(dimensions),
            "voxel_size_m": voxel_size,
            "truncation_m": voxel_size,
            "index_order": "x-fastest-then-y-then-z",
            "tsdf_sign": "positive-free-space-negative-behind-surface",
            "unknown_rule": "weight-zero",
        },
        "integration": {
            "frame_stride": 1,
            "total_observations": integrated_frames,
            "selected_observations": integrated_frames,
            "integrated_frames": integrated_frames,
            "skipped_missing_depth": 0,
            "skipped_missing_pose": 0,
            "invalid_depth_pixels": 0,
            "total_voxels": total_voxels,
            "observed_voxels": len(voxels),
            "unknown_voxels": total_voxels - len(voxels),
            "fused_voxels": sum(voxel["weight"] > 1 for voxel in voxels),
            "voxel_updates": sum(voxel["weight"] for voxel in voxels),
            "max_weight": max(voxel["weight"] for voxel in voxels),
        },
        "voxels": voxels,
    }
    write_artifact(path, document)
    return document


def write_artifact(path: Path, document: dict) -> None:
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


class SurfacePointTests(unittest.TestCase):
    def test_reference_plane_extracts_exact_zero_crossing(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            tsdf_path = temporary_root / "plane.sftsdf"
            output = temporary_root / "surface.ply"
            integrate_tsdf(
                load_scan_session(FIXTURE),
                tsdf_path,
                **REFERENCE_ARGUMENTS,
            )

            report = extract_surface_points(tsdf_path, output)
            header, vertices = parse_ply(output)

        self.assertEqual(report.total_voxels, 4)
        self.assertEqual(report.observed_voxels, 3)
        self.assertEqual(report.observed_edges, 2)
        self.assertEqual(report.exact_zero_points, 0)
        self.assertEqual(report.crossing_x_points, 1)
        self.assertEqual(report.crossing_y_points, 0)
        self.assertEqual(report.crossing_z_points, 0)
        self.assertEqual(report.points_written, 1)
        self.assertEqual(vertices, [(1.0, 0.0, 0.0)])
        self.assertIn("element vertex 1", header)
        self.assertNotIn("property list uchar int vertex_indices", header)
        self.assertNotIn("property uchar red", header)
        self.assertEqual(
            report.output_digest_sha256,
            "bdb05ea65e4f5fcfa59213958a0f4ab33581e2f4d8d6df0f2f410c2c65fd9154",
        )

    def test_crossings_use_tsdf_interpolation_and_axis_order(self) -> None:
        voxels = [
            {"index": [0, 0, 0], "tsdf": 1.0, "weight": 2},
            {"index": [2, 0, 0], "tsdf": 0.75, "weight": 3},
            {"index": [4, 0, 0], "tsdf": 0.25, "weight": 4},
            {"index": [5, 0, 0], "tsdf": -0.75, "weight": 7},
            {"index": [2, 1, 0], "tsdf": -0.25, "weight": 5},
            {"index": [0, 2, 0], "tsdf": 1.0, "weight": 6},
            {"index": [1, 2, 0], "tsdf": -0.5, "weight": 9},
            {"index": [0, 0, 1], "tsdf": -1.0, "weight": 8},
        ]
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            tsdf_path = temporary_root / "axes.sftsdf"
            output = temporary_root / "axes.ply"
            build_artifact(
                tsdf_path,
                origin=(10.0, -2.0, 3.0),
                dimensions=(8, 3, 2),
                voxel_size=2.0,
                voxels=voxels,
            )

            report = extract_surface_points(tsdf_path, output)
            _, vertices = parse_ply(output)

        self.assertEqual(report.crossing_x_points, 2)
        self.assertEqual(report.crossing_y_points, 1)
        self.assertEqual(report.crossing_z_points, 1)
        self.assertEqual(
            vertices,
            [
                (19.5, -1.0, 4.0),
                (12.333333333, 3.0, 4.0),
                (15.0, 0.5, 4.0),
                (11.0, -1.0, 5.0),
            ],
        )

    def test_exact_zero_centers_are_emitted_once_before_edges(self) -> None:
        voxels = [
            {"index": [0, 0, 0], "tsdf": 1.0, "weight": 1},
            {"index": [1, 0, 0], "tsdf": 0.0, "weight": 1},
            {"index": [2, 0, 0], "tsdf": 0.0, "weight": 1},
            {"index": [3, 0, 0], "tsdf": -1.0, "weight": 1},
            {"index": [1, 1, 0], "tsdf": 1.0, "weight": 1},
        ]
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            tsdf_path = temporary_root / "zeros.sftsdf"
            output = temporary_root / "zeros.ply"
            build_artifact(
                tsdf_path,
                origin=(0.0, 0.0, 0.0),
                dimensions=(4, 2, 1),
                voxel_size=1.0,
                voxels=voxels,
            )

            report = extract_surface_points(tsdf_path, output)
            _, vertices = parse_ply(output)

        self.assertEqual(report.exact_zero_points, 2)
        self.assertEqual(
            report.crossing_x_points
            + report.crossing_y_points
            + report.crossing_z_points,
            0,
        )
        self.assertEqual(
            vertices,
            [(1.5, 0.5, 0.5), (2.5, 0.5, 0.5)],
        )

    def test_unknown_gap_never_creates_a_surface(self) -> None:
        voxels = [
            {"index": [0, 0, 0], "tsdf": 1.0, "weight": 1},
            {"index": [2, 0, 0], "tsdf": -1.0, "weight": 1},
            {"index": [3, 0, 0], "tsdf": -1.0, "weight": 1},
        ]
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            tsdf_path = temporary_root / "gap.sftsdf"
            output = temporary_root / "gap.ply"
            build_artifact(
                tsdf_path,
                origin=(0.0, 0.0, 0.0),
                dimensions=(4, 1, 1),
                voxel_size=1.0,
                voxels=voxels,
            )

            with self.assertRaises(SurfaceExtractionError) as raised:
                extract_surface_points(tsdf_path, output)

            self.assertFalse(output.exists())

        self.assertIn("no exact-zero voxel", str(raised.exception))

    def test_malformed_artifacts_fail_actionably(self) -> None:
        base_voxels = [
            {"index": [0, 0, 0], "tsdf": 0.5, "weight": 1},
            {"index": [1, 0, 0], "tsdf": -0.5, "weight": 1},
        ]
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            base_path = temporary_root / "base.sftsdf"
            base = build_artifact(
                base_path,
                origin=(0.0, 0.0, 0.0),
                dimensions=(2, 1, 1),
                voxel_size=1.0,
                voxels=base_voxels,
            )
            cases: list[tuple[str, dict, str]] = []

            bad_schema = copy.deepcopy(base)
            bad_schema["schema"] = "wrong"
            cases.append(("schema", bad_schema, "TSDF.schema"))

            reversed_voxels = copy.deepcopy(base)
            reversed_voxels["voxels"].reverse()
            cases.append(("order", reversed_voxels, "strictly x-fastest ordered"))

            out_of_bounds = copy.deepcopy(base)
            out_of_bounds["voxels"][1]["index"] = [2, 0, 0]
            cases.append(("bounds", out_of_bounds, "outside volume dimensions"))

            invalid_tsdf = copy.deepcopy(base)
            invalid_tsdf["voxels"][0]["tsdf"] = 2.0
            cases.append(("tsdf", invalid_tsdf, "expected value in [-1, 1]"))

            invalid_weight = copy.deepcopy(base)
            invalid_weight["voxels"][0]["weight"] = 0
            cases.append(("weight", invalid_weight, "positive integer"))

            bad_count = copy.deepcopy(base)
            bad_count["integration"]["observed_voxels"] = 1
            cases.append(("count", bad_count, "observed_voxels"))

            unknown_field = copy.deepcopy(base)
            unknown_field["volume"]["future_option"] = True
            cases.append(("unknown", unknown_field, "unexpected field"))

            impossible_frames = copy.deepcopy(base)
            impossible_frames["integration"]["integrated_frames"] = 0
            cases.append(
                ("frames", impossible_frames, "integrated_frames"),
            )

            impossible_weight = copy.deepcopy(base)
            impossible_weight["voxels"][0]["weight"] = 2
            impossible_weight["integration"]["fused_voxels"] = 1
            impossible_weight["integration"]["voxel_updates"] = 3
            impossible_weight["integration"]["max_weight"] = 2
            cases.append(
                ("max-weight", impossible_weight, "max_weight"),
            )

            for name, document, expected_message in cases:
                with self.subTest(name=name):
                    input_path = temporary_root / f"{name}.sftsdf"
                    output = temporary_root / f"{name}.ply"
                    write_artifact(input_path, document)
                    with self.assertRaises(SurfaceExtractionError) as raised:
                        extract_surface_points(input_path, output)
                    self.assertIn(expected_message, str(raised.exception))
                    self.assertFalse(output.exists())

            duplicate_path = temporary_root / "duplicate.sftsdf"
            duplicate_path.write_text(
                '{"schema":"first","schema":"second"}\n',
                encoding="utf-8",
            )
            with self.assertRaises(SurfaceExtractionError) as duplicate:
                extract_surface_points(
                    duplicate_path,
                    temporary_root / "duplicate.ply",
                )
            self.assertIn("duplicate JSON key", str(duplicate.exception))

            nested_path = temporary_root / "nested.sftsdf"
            nested_path.write_text(
                "[" * 100_000 + "0" + "]" * 100_000,
                encoding="utf-8",
            )
            with self.assertRaises(SurfaceExtractionError) as nested:
                extract_surface_points(
                    nested_path,
                    temporary_root / "nested.ply",
                )
            self.assertIn("nested too deeply", str(nested.exception))

    def test_output_is_deterministic_and_existing_file_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            tsdf_path = temporary_root / "plane.sftsdf"
            first = temporary_root / "first.ply"
            second = temporary_root / "second.ply"
            integrate_tsdf(
                load_scan_session(FIXTURE),
                tsdf_path,
                **REFERENCE_ARGUMENTS,
            )
            extract_surface_points(tsdf_path, first)
            extract_surface_points(tsdf_path, second)
            self.assertEqual(first.read_bytes(), second.read_bytes())

            existing = temporary_root / "existing.ply"
            existing.write_text("keep", encoding="ascii")
            with self.assertRaises(SurfaceExtractionError):
                extract_surface_points(tsdf_path, existing)
            self.assertEqual(existing.read_text(encoding="ascii"), "keep")

    def test_racing_output_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            tsdf_path = temporary_root / "plane.sftsdf"
            output = temporary_root / "raced.ply"
            integrate_tsdf(
                load_scan_session(FIXTURE),
                tsdf_path,
                **REFERENCE_ARGUMENTS,
            )
            actual_link = os.link

            def create_competing_output(
                staging_path: str | Path,
                output_path: str | Path,
            ) -> None:
                Path(output_path).write_text("competitor", encoding="ascii")
                actual_link(staging_path, output_path)

            with patch(
                "spatialforge.surface.os.link",
                side_effect=create_competing_output,
            ):
                with self.assertRaises(SurfaceExtractionError) as raised:
                    extract_surface_points(tsdf_path, output)

            self.assertEqual(output.read_text(encoding="ascii"), "competitor")

        self.assertIn("refusing to overwrite", str(raised.exception))


class SurfacePointCliTests(unittest.TestCase):
    def test_cli_extracts_and_reports_the_surface_point(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            tsdf_path = temporary_root / "plane.sftsdf"
            output = temporary_root / "surface.ply"
            integrate_tsdf(
                load_scan_session(FIXTURE),
                tsdf_path,
                **REFERENCE_ARGUMENTS,
            )
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "reconstruct",
                        "surface-points",
                        str(tsdf_path),
                        str(output),
                    ]
                )
            output_exists = output.is_file()

        self.assertEqual(exit_code, 0)
        self.assertTrue(output_exists)
        self.assertIn("SURFACE POINTS scan-synthetic-0001", stdout.getvalue())
        self.assertIn("crossings: x=1 y=0 z=0", stdout.getvalue())
        self.assertIn("exact_zero=0 crossing=1 total=1", stdout.getvalue())

    def test_cli_failure_is_actionable(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            output = Path(temporary_directory) / "missing.ply"
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                exit_code = main(
                    [
                        "reconstruct",
                        "surface-points",
                        str(TEST_ROOT / "missing.sftsdf"),
                        str(output),
                    ]
                )

        self.assertEqual(exit_code, 2)
        self.assertIn("SURFACE EXTRACTION FAILED", stderr.getvalue())
        self.assertIn("does not exist", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
