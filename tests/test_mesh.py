from __future__ import annotations

import io
import json
import math
import os
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from spatialforge.cli import main
from spatialforge.errors import MeshExtractionError
from spatialforge.mesh import extract_triangle_mesh
from spatialforge.session_loader import load_scan_session
from spatialforge.tsdf import integrate_tsdf


TEST_ROOT = Path(__file__).resolve().parent
FIXTURE = TEST_ROOT / "fixtures" / "minimal.vgsession"
MESH_REFERENCE_ARGUMENTS = {
    "origin_world_m": (0.5, -0.5, -0.5),
    "dimensions": (2, 2, 2),
    "voxel_size_m": 0.5,
    "truncation_m": 0.5,
}


def parse_mesh_ply(
    path: Path,
) -> tuple[list[str], list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    lines = path.read_text(encoding="ascii").splitlines()
    end_header = lines.index("end_header")
    header = lines[: end_header + 1]
    vertex_count = int(
        next(line for line in header if line.startswith("element vertex ")).split()[2]
    )
    face_count = int(
        next(line for line in header if line.startswith("element face ")).split()[2]
    )
    body = lines[end_header + 1 :]
    vertices = [
        tuple(float(value) for value in line.split())
        for line in body[:vertex_count]
    ]
    faces = [
        tuple(int(value) for value in line.split()[1:])
        for line in body[vertex_count : vertex_count + face_count]
    ]
    return header, vertices, faces  # type: ignore[return-value]


def grid_voxels(
    dimensions: tuple[int, int, int],
    scalar,
    *,
    missing: set[tuple[int, int, int]] | None = None,
) -> list[dict]:
    omitted = missing or set()
    nx, ny, nz = dimensions
    return [
        {
            "index": [x, y, z],
            "tsdf": scalar(x, y, z),
            "weight": 1,
        }
        for z in range(nz)
        for y in range(ny)
        for x in range(nx)
        if (x, y, z) not in omitted
    ]


def build_artifact(
    path: Path,
    *,
    dimensions: tuple[int, int, int],
    voxels: list[dict],
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    voxel_size: float = 1.0,
) -> None:
    total_voxels = math.prod(dimensions)
    integrated_frames = max(voxel["weight"] for voxel in voxels)
    document = {
        "schema": "spatialforge.reference-tsdf",
        "schema_version": "0.1.0",
        "session_id": "mesh-synthetic",
        "replay_digest_sha256": "b" * 64,
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
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def face_normal(
    vertices: list[tuple[float, float, float]],
    face: tuple[int, int, int],
) -> tuple[float, float, float]:
    first, second, third = (vertices[index] for index in face)
    first_edge = tuple(b - a for a, b in zip(first, second, strict=True))
    second_edge = tuple(b - a for a, b in zip(first, third, strict=True))
    return (
        first_edge[1] * second_edge[2] - first_edge[2] * second_edge[1],
        first_edge[2] * second_edge[0] - first_edge[0] * second_edge[2],
        first_edge[0] * second_edge[1] - first_edge[1] * second_edge[0],
    )


def edge_incidence_counts(
    faces: list[tuple[int, int, int]],
) -> Counter[tuple[int, int]]:
    return Counter(
        tuple(sorted(edge))
        for first, second, third in faces
        for edge in ((first, second), (second, third), (third, first))
    )


class TriangleMeshTests(unittest.TestCase):
    def test_reference_plane_has_exact_topology_winding_and_area(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            tsdf_path = temporary_root / "plane.sftsdf"
            output = temporary_root / "plane.ply"
            integrate_tsdf(
                load_scan_session(FIXTURE),
                tsdf_path,
                **MESH_REFERENCE_ARGUMENTS,
            )

            report = extract_triangle_mesh(tsdf_path, output)
            header, vertices, faces = parse_mesh_ply(output)
            encoded = output.read_bytes()

        self.assertEqual(report.total_voxels, 8)
        self.assertEqual(report.observed_voxels, 8)
        self.assertEqual(report.total_cells, 1)
        self.assertEqual(report.skipped_unknown_cells, 0)
        self.assertEqual(report.skipped_exact_zero_cells, 0)
        self.assertEqual(report.eligible_cells, 1)
        self.assertEqual(report.active_cells, 1)
        self.assertEqual(report.vertices_written, 9)
        self.assertEqual(report.triangles_written, 8)
        self.assertEqual(report.boundary_edges, 8)
        self.assertEqual(
            vertices,
            [
                (1.0, -0.25, -0.25),
                (1.0, 0.0, -0.25),
                (1.0, 0.0, 0.0),
                (1.0, 0.25, -0.25),
                (1.0, 0.25, 0.0),
                (1.0, 0.25, 0.25),
                (1.0, 0.0, 0.25),
                (1.0, -0.25, 0.0),
                (1.0, -0.25, 0.25),
            ],
        )
        self.assertEqual(
            faces,
            [
                (0, 2, 1),
                (1, 4, 3),
                (1, 2, 4),
                (2, 5, 4),
                (2, 6, 5),
                (7, 8, 6),
                (7, 6, 2),
                (0, 7, 2),
            ],
        )
        normals = [face_normal(vertices, face) for face in faces]
        self.assertTrue(all(normal[0] < 0.0 for normal in normals))
        self.assertTrue(
            all(normal[1] == 0.0 and normal[2] == 0.0 for normal in normals)
        )
        area = sum(
            math.sqrt(sum(component * component for component in normal)) / 2.0
            for normal in normals
        )
        self.assertAlmostEqual(area, 0.25)
        self.assertIn("element vertex 9", header)
        self.assertIn("element face 8", header)
        self.assertIn("property list uchar int vertex_indices", header)
        self.assertNotIn("property float nx", header)
        self.assertNotIn("property uchar red", header)
        self.assertTrue(encoded.endswith(b"\n"))
        self.assertEqual(
            report.output_digest_sha256,
            "b240a3a8286eb1025dd5e64047a63d8aa56519a21b201a050eca7c951d4b1a35",
        )

    def test_adjacent_cells_share_vertices_and_output_is_deterministic(self) -> None:
        dimensions = (2, 3, 2)
        voxels = grid_voxels(
            dimensions,
            lambda x, _y, _z: 0.5 if x == 0 else -0.5,
        )
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            tsdf_path = temporary_root / "two-cells.sftsdf"
            first = temporary_root / "first.ply"
            second = temporary_root / "second.ply"
            build_artifact(
                tsdf_path,
                dimensions=dimensions,
                voxels=voxels,
            )

            first_report = extract_triangle_mesh(tsdf_path, first)
            second_report = extract_triangle_mesh(tsdf_path, second)
            _, vertices, faces = parse_mesh_ply(first)
            first_bytes = first.read_bytes()
            second_bytes = second.read_bytes()

        self.assertEqual(first_bytes, second_bytes)
        self.assertEqual(first_report.output_digest_sha256, second_report.output_digest_sha256)
        self.assertEqual(first_report.active_cells, 2)
        self.assertEqual(first_report.vertices_written, 15)
        self.assertEqual(first_report.triangles_written, 16)
        self.assertEqual(first_report.boundary_edges, 12)
        self.assertEqual(len(vertices), len(set(vertices)))
        self.assertTrue(
            all(0 <= index < len(vertices) for face in faces for index in face)
        )
        self.assertEqual(
            {index for face in faces for index in face},
            set(range(len(vertices))),
        )
        self.assertTrue(
            all(face_normal(vertices, face)[0] < 0.0 for face in faces)
        )
        vertex_ids = {point: index for index, point in enumerate(vertices)}
        seam = [
            vertex_ids[(1.0, 1.5, 0.5)],
            vertex_ids[(1.0, 1.5, 1.0)],
            vertex_ids[(1.0, 1.5, 1.5)],
        ]
        incidences = edge_incidence_counts(faces)
        self.assertEqual(incidences[tuple(sorted(seam[:2]))], 2)
        self.assertEqual(incidences[tuple(sorted(seam[1:]))], 2)

    def test_unequal_tsdf_values_interpolate_a_non_x_plane(self) -> None:
        dimensions = (2, 2, 2)
        voxels = grid_voxels(
            dimensions,
            lambda _x, y, _z: 0.25 if y == 0 else -0.75,
        )
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            tsdf_path = temporary_root / "unequal.sftsdf"
            output = temporary_root / "unequal.ply"
            build_artifact(
                tsdf_path,
                dimensions=dimensions,
                voxels=voxels,
            )

            report = extract_triangle_mesh(tsdf_path, output)
            _, vertices, faces = parse_mesh_ply(output)

        self.assertEqual(report.vertices_written, 9)
        self.assertEqual(report.triangles_written, 8)
        self.assertTrue(all(vertex[1] == 0.75 for vertex in vertices))
        normals = [face_normal(vertices, face) for face in faces]
        self.assertTrue(all(normal[1] < 0.0 for normal in normals))
        self.assertTrue(
            all(normal[0] == 0.0 and normal[2] == 0.0 for normal in normals)
        )

    def test_extreme_coordinates_preserve_reference_winding(self) -> None:
        dimensions = (2, 2, 2)
        voxels = grid_voxels(
            dimensions,
            lambda x, y, z: -0.5 if (x, y, z) == (0, 0, 0) else 0.5,
        )
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            small_tsdf = temporary_root / "small.sftsdf"
            large_tsdf = temporary_root / "large.sftsdf"
            small_output = temporary_root / "small.ply"
            large_output = temporary_root / "large.ply"
            build_artifact(
                small_tsdf,
                dimensions=dimensions,
                voxels=voxels,
                voxel_size=1.0,
            )
            build_artifact(
                large_tsdf,
                dimensions=dimensions,
                voxels=voxels,
                voxel_size=1e200,
            )

            extract_triangle_mesh(small_tsdf, small_output)
            extract_triangle_mesh(large_tsdf, large_output)
            _, _, small_faces = parse_mesh_ply(small_output)
            _, large_vertices, large_faces = parse_mesh_ply(large_output)

        self.assertEqual(large_faces, small_faces)
        self.assertTrue(
            all(math.isfinite(component) for point in large_vertices for component in point)
        )

    def test_serialization_refuses_collapsed_tiny_triangles(self) -> None:
        dimensions = (2, 2, 2)
        voxels = grid_voxels(
            dimensions,
            lambda x, y, z: -1e-9 if (x, y, z) == (0, 0, 0) else 1.0,
        )
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            tsdf_path = temporary_root / "tiny.sftsdf"
            output = temporary_root / "tiny.ply"
            build_artifact(
                tsdf_path,
                dimensions=dimensions,
                voxels=voxels,
                voxel_size=0.05,
            )

            with self.assertRaises(MeshExtractionError) as raised:
                extract_triangle_mesh(tsdf_path, output)

            self.assertIn("precision would merge", str(raised.exception))
            self.assertFalse(output.exists())

    def test_unknown_gaps_cannot_create_a_bow_tie_vertex(self) -> None:
        dimensions = (3, 3, 2)
        voxels = grid_voxels(
            dimensions,
            lambda _x, _y, z: 0.5 if z == 0 else -0.5,
            missing={
                (x, y, z)
                for z in range(2)
                for y in range(3)
                for x in range(3)
                if not (
                    (x <= 1 and y <= 1)
                    or (x >= 1 and y >= 1)
                )
            },
        )
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            tsdf_path = temporary_root / "bow-tie.sftsdf"
            output = temporary_root / "bow-tie.ply"
            build_artifact(
                tsdf_path,
                dimensions=dimensions,
                voxels=voxels,
            )

            with self.assertRaises(MeshExtractionError) as raised:
                extract_triangle_mesh(tsdf_path, output)

            self.assertIn("non-manifold triangle vertices", str(raised.exception))
            self.assertFalse(output.exists())

    def test_checkerboard_case_has_fixed_tetrahedral_resolution(self) -> None:
        dimensions = (2, 2, 2)
        voxels = grid_voxels(
            dimensions,
            lambda x, y, z: 0.5 if (x + y + z) % 2 == 0 else -0.5,
        )
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            tsdf_path = temporary_root / "checkerboard.sftsdf"
            output = temporary_root / "checkerboard.ply"
            build_artifact(
                tsdf_path,
                dimensions=dimensions,
                voxels=voxels,
            )

            report = extract_triangle_mesh(tsdf_path, output)
            _, vertices, faces = parse_mesh_ply(output)

        self.assertEqual(report.active_cells, 1)
        self.assertEqual(report.vertices_written, 13)
        self.assertEqual(report.triangles_written, 12)
        self.assertEqual(report.boundary_edges, 12)
        self.assertEqual(
            faces,
            [
                (0, 3, 1),
                (0, 2, 3),
                (4, 5, 3),
                (4, 3, 2),
                (4, 7, 6),
                (4, 2, 7),
                (8, 9, 7),
                (8, 7, 2),
                (8, 11, 10),
                (8, 2, 11),
                (0, 12, 11),
                (0, 11, 2),
            ],
        )
        self.assertEqual(len(faces), len({tuple(sorted(face)) for face in faces}))
        self.assertTrue(all(len(set(face)) == 3 for face in faces))
        self.assertEqual(len(vertices), 13)

    def test_successful_mesh_reports_skipped_cell_precedence(self) -> None:
        dimensions = (3, 2, 2)
        base_voxels = grid_voxels(
            dimensions,
            lambda x, _y, _z: 0.5 if x == 0 else -0.5,
        )
        zero_index = (2, 0, 0)
        missing_index = (2, 1, 1)
        cases = (
            ("unknown", {missing_index}, set(), (1, 0)),
            ("zero", set(), {zero_index}, (0, 1)),
            (
                "unknown-before-zero",
                {missing_index},
                {zero_index},
                (1, 0),
            ),
        )
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            for name, missing, zeros, expected_skips in cases:
                with self.subTest(name=name):
                    voxels = [
                        {
                            **voxel,
                            "tsdf": (
                                0.0
                                if tuple(voxel["index"]) in zeros
                                else voxel["tsdf"]
                            ),
                        }
                        for voxel in base_voxels
                        if tuple(voxel["index"]) not in missing
                    ]
                    tsdf_path = temporary_root / f"{name}.sftsdf"
                    output = temporary_root / f"{name}.ply"
                    build_artifact(
                        tsdf_path,
                        dimensions=dimensions,
                        voxels=voxels,
                    )

                    report = extract_triangle_mesh(tsdf_path, output)
                    _, vertices, faces = parse_mesh_ply(output)

                    self.assertEqual(report.total_cells, 2)
                    self.assertEqual(
                        (
                            report.skipped_unknown_cells,
                            report.skipped_exact_zero_cells,
                        ),
                        expected_skips,
                    )
                    self.assertEqual(report.eligible_cells, 1)
                    self.assertEqual(report.active_cells, 1)
                    self.assertEqual(
                        report.total_cells,
                        report.skipped_unknown_cells
                        + report.skipped_exact_zero_cells
                        + report.eligible_cells,
                    )
                    self.assertEqual(
                        {index for face in faces for index in face},
                        set(range(len(vertices))),
                    )

    def test_unknown_zero_uniform_and_flat_volumes_fail_without_output(self) -> None:
        dimensions = (2, 2, 2)
        cases = (
            (
                "unknown",
                grid_voxels(
                    dimensions,
                    lambda x, _y, _z: 0.5 if x == 0 else -0.5,
                    missing={(1, 1, 1)},
                ),
                "unknown=1",
            ),
            (
                "exact-zero",
                grid_voxels(
                    dimensions,
                    lambda x, y, z: (
                        0.0
                        if (x, y, z) == (1, 1, 1)
                        else (0.5 if x == 0 else -0.5)
                    ),
                ),
                "exact_zero=1",
            ),
            (
                "uniform",
                grid_voxels(dimensions, lambda _x, _y, _z: 0.5),
                "eligible=1",
            ),
        )
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            for name, voxels, expected in cases:
                with self.subTest(name=name):
                    tsdf_path = temporary_root / f"{name}.sftsdf"
                    output = temporary_root / f"{name}.ply"
                    build_artifact(
                        tsdf_path,
                        dimensions=dimensions,
                        voxels=voxels,
                    )
                    with self.assertRaises(MeshExtractionError) as raised:
                        extract_triangle_mesh(tsdf_path, output)
                    self.assertIn(expected, str(raised.exception))
                    self.assertFalse(output.exists())

            flat_path = temporary_root / "flat.sftsdf"
            flat_output = temporary_root / "flat.ply"
            build_artifact(
                flat_path,
                dimensions=(2, 2, 1),
                voxels=grid_voxels(
                    (2, 2, 1),
                    lambda x, _y, _z: 0.5 if x == 0 else -0.5,
                ),
            )
            with self.assertRaises(MeshExtractionError) as flat:
                extract_triangle_mesh(flat_path, flat_output)
            self.assertIn("dimensions must each be at least 2", str(flat.exception))
            self.assertFalse(flat_output.exists())

    def test_strict_tsdf_validation_is_reused(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            malformed = temporary_root / "malformed.sftsdf"
            malformed.write_text(
                '{"schema":"spatialforge.reference-tsdf","future":true}\n',
                encoding="utf-8",
            )
            output = temporary_root / "malformed.ply"

            with self.assertRaises(MeshExtractionError) as raised:
                extract_triangle_mesh(malformed, output)
            self.assertFalse(output.exists())

        self.assertIn("unexpected field", str(raised.exception))

    def test_triangle_limit_boundary_and_failure_are_exact(self) -> None:
        dimensions = (2, 2, 2)
        voxels = grid_voxels(
            dimensions,
            lambda x, _y, _z: 0.5 if x == 0 else -0.5,
        )
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            tsdf_path = temporary_root / "plane.sftsdf"
            allowed = temporary_root / "allowed.ply"
            limited = temporary_root / "limited.ply"
            build_artifact(
                tsdf_path,
                dimensions=dimensions,
                voxels=voxels,
            )

            with patch("spatialforge.mesh.MAX_REFERENCE_TRIANGLES", 8):
                report = extract_triangle_mesh(tsdf_path, allowed)
            self.assertEqual(report.triangles_written, 8)

            with patch("spatialforge.mesh.MAX_REFERENCE_TRIANGLES", 7):
                with self.assertRaises(MeshExtractionError) as raised:
                    extract_triangle_mesh(tsdf_path, limited)

            self.assertIn("triangle limit exceeded", str(raised.exception))
            self.assertFalse(limited.exists())

    def test_existing_and_racing_outputs_are_preserved(self) -> None:
        dimensions = (2, 2, 2)
        voxels = grid_voxels(
            dimensions,
            lambda x, _y, _z: 0.5 if x == 0 else -0.5,
        )
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            tsdf_path = temporary_root / "plane.sftsdf"
            build_artifact(
                tsdf_path,
                dimensions=dimensions,
                voxels=voxels,
            )

            existing = temporary_root / "existing.ply"
            existing.write_text("keep", encoding="ascii")
            with self.assertRaises(MeshExtractionError):
                extract_triangle_mesh(tsdf_path, existing)
            self.assertEqual(existing.read_text(encoding="ascii"), "keep")

            raced = temporary_root / "raced.ply"
            actual_link = os.link

            def create_competing_output(
                staging_path: str | Path,
                output_path: str | Path,
            ) -> None:
                Path(output_path).write_text("competitor", encoding="ascii")
                actual_link(staging_path, output_path)

            with patch(
                "spatialforge.mesh.os.link",
                side_effect=create_competing_output,
            ):
                with self.assertRaises(MeshExtractionError) as raised:
                    extract_triangle_mesh(tsdf_path, raced)

            self.assertIn("refusing to overwrite", str(raised.exception))
            self.assertEqual(raced.read_text(encoding="ascii"), "competitor")
            self.assertEqual(list(temporary_root.glob(".raced-*.ply")), [])


class TriangleMeshCliTests(unittest.TestCase):
    def test_cli_extracts_and_reports_reference_mesh(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            tsdf_path = temporary_root / "plane.sftsdf"
            output = temporary_root / "plane.ply"
            integrate_tsdf(
                load_scan_session(FIXTURE),
                tsdf_path,
                **MESH_REFERENCE_ARGUMENTS,
            )
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "reconstruct",
                        "triangle-mesh",
                        str(tsdf_path),
                        str(output),
                    ]
                )
            output_exists = output.is_file()

        self.assertEqual(exit_code, 0)
        self.assertTrue(output_exists)
        self.assertIn("TRIANGLE MESH scan-synthetic-0001", stdout.getvalue())
        self.assertIn("cells: total=1 eligible=1 active=1", stdout.getvalue())
        self.assertIn(
            "mesh: vertices=9 triangles=8 boundary_edges=8",
            stdout.getvalue(),
        )

    def test_cli_failure_is_actionable(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            output = Path(temporary_directory) / "missing.ply"
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                exit_code = main(
                    [
                        "reconstruct",
                        "triangle-mesh",
                        str(TEST_ROOT / "missing.sftsdf"),
                        str(output),
                    ]
                )

        self.assertEqual(exit_code, 2)
        self.assertIn("TRIANGLE MESH FAILED", stderr.getvalue())
        self.assertIn("does not exist", stderr.getvalue())

    def test_cli_reports_valid_volume_with_no_surface(self) -> None:
        dimensions = (2, 2, 2)
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            tsdf_path = temporary_root / "uniform.sftsdf"
            output = temporary_root / "uniform.ply"
            build_artifact(
                tsdf_path,
                dimensions=dimensions,
                voxels=grid_voxels(
                    dimensions,
                    lambda _x, _y, _z: 0.5,
                ),
            )
            stderr = io.StringIO()

            with redirect_stderr(stderr):
                exit_code = main(
                    [
                        "reconstruct",
                        "triangle-mesh",
                        str(tsdf_path),
                        str(output),
                    ]
                )
            output_exists = output.exists()

        self.assertEqual(exit_code, 2)
        self.assertFalse(output_exists)
        self.assertIn("TRIANGLE MESH FAILED", stderr.getvalue())
        self.assertIn("produced no triangles", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
