from __future__ import annotations

import io
import json
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np

from spatialforge import integrate_sparse_tsdf
from spatialforge.cli import main
from spatialforge.errors import TsdfError
from spatialforge.mesh import extract_triangle_mesh
from spatialforge.session_loader import load_scan_session
from spatialforge.sparse_tsdf import _SparseTsdfAccumulator
from spatialforge.surface import extract_surface_points
from spatialforge.tsdf import MAX_REFERENCE_VOXELS, integrate_tsdf


TEST_ROOT = Path(__file__).resolve().parent
FIXTURE = TEST_ROOT / "fixtures" / "minimal.vgsession"
REFERENCE_ARGUMENTS = {
    "origin_world_m": (0.0, -0.25, -0.25),
    "dimensions": (4, 1, 1),
    "voxel_size_m": 0.5,
    "truncation_m": 0.5,
}
MESH_ARGUMENTS = {
    "origin_world_m": (0.5, -0.5, -0.5),
    "dimensions": (2, 2, 2),
    "voxel_size_m": 0.5,
    "truncation_m": 0.5,
}


def copy_fixture(parent: Path, name: str = "case.vgsession") -> Path:
    target = parent / name
    shutil.copytree(FIXTURE, target)
    return target


def reports_without_output(report) -> dict:
    values = asdict(report)
    del values["output"]
    return values


class SparseAccumulatorTests(unittest.TestCase):
    def test_accumulator_is_lazy_accumulates_and_emits_sorted_indices(self) -> None:
        accumulator = _SparseTsdfAccumulator(1_000_000)
        self.assertEqual(accumulator.stored_voxels, 0)

        accumulator.add(
            np.asarray([700_000, 9, 2], dtype=np.int64),
            np.asarray([1.0, 0.5, -0.5], dtype=np.float64),
        )
        accumulator.add(
            np.asarray([9], dtype=np.int64),
            np.asarray([0.25], dtype=np.float64),
        )
        indices, sums, weights = accumulator.observed_arrays()

        self.assertEqual(accumulator.stored_voxels, 3)
        self.assertEqual(indices.tolist(), [2, 9, 700_000])
        self.assertEqual(sums.tolist(), [-0.5, 0.75, 1.0])
        self.assertEqual(weights.tolist(), [1, 2, 1])

    def test_empty_updates_allocate_no_voxel_entries(self) -> None:
        accumulator = _SparseTsdfAccumulator(1_000_000)
        accumulator.add(
            np.asarray([], dtype=np.int64),
            np.asarray([], dtype=np.float64),
        )
        indices, sums, weights = accumulator.observed_arrays()

        self.assertEqual(accumulator.stored_voxels, 0)
        self.assertEqual(indices.size, 0)
        self.assertEqual(sums.size, 0)
        self.assertEqual(weights.size, 0)


class SparseTsdfTests(unittest.TestCase):
    def assert_dense_sparse_parity(
        self,
        session_path: Path,
        temporary_root: Path,
        *,
        name: str,
        arguments: dict | None = None,
    ):
        dense_output = temporary_root / f"{name}-dense.sftsdf"
        sparse_output = temporary_root / f"{name}-sparse.sftsdf"
        integration_arguments = arguments or REFERENCE_ARGUMENTS
        dense_report = integrate_tsdf(
            load_scan_session(session_path),
            dense_output,
            **integration_arguments,
        )
        sparse_report = integrate_sparse_tsdf(
            load_scan_session(session_path),
            sparse_output,
            **integration_arguments,
        )
        self.assertEqual(dense_output.read_bytes(), sparse_output.read_bytes())
        self.assertEqual(
            reports_without_output(dense_report),
            reports_without_output(sparse_report),
        )
        return dense_report, sparse_report

    def test_fixture_is_byte_identical_to_dense_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            dense_report, sparse_report = self.assert_dense_sparse_parity(
                FIXTURE,
                temporary_root,
                name="reference",
            )
            repeated = temporary_root / "repeated.sftsdf"
            repeated_report = integrate_sparse_tsdf(
                load_scan_session(FIXTURE),
                repeated,
                **REFERENCE_ARGUMENTS,
            )
            sparse_bytes = sparse_report.output.read_bytes()
            repeated_bytes = repeated.read_bytes()

        self.assertEqual(sparse_bytes, repeated_bytes)
        self.assertEqual(
            sparse_report.output_digest_sha256,
            "7d3f30121fa5537a4f77d881b12087d04e5805a84066598c6a62f988e3247495",
        )
        self.assertEqual(
            repeated_report.output_digest_sha256,
            dense_report.output_digest_sha256,
        )

    def test_replay_conditions_remain_byte_identical_to_dense(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            stride_arguments = dict(REFERENCE_ARGUMENTS)
            stride_arguments["frame_stride"] = 2
            self.assert_dense_sparse_parity(
                FIXTURE,
                temporary_root,
                name="stride",
                arguments=stride_arguments,
            )

            missing_pose = copy_fixture(temporary_root, "missing-pose.vgsession")
            pose_path = missing_pose / "streams" / "poses.jsonl"
            pose_path.write_text(
                pose_path.read_text(encoding="utf-8").splitlines()[0] + "\n",
                encoding="utf-8",
            )
            self.assert_dense_sparse_parity(
                missing_pose,
                temporary_root,
                name="missing-pose",
            )

            missing_depth = copy_fixture(
                temporary_root,
                "missing-depth.vgsession",
            )
            depth_index = missing_depth / "streams" / "depth.jsonl"
            depth_index.write_text(
                depth_index.read_text(encoding="utf-8").splitlines()[0] + "\n",
                encoding="utf-8",
            )
            self.assert_dense_sparse_parity(
                missing_depth,
                temporary_root,
                name="missing-depth",
            )

            zero_depth = copy_fixture(temporary_root, "zero-depth.vgsession")
            (zero_depth / "data" / "depth" / "000000.pgm").write_text(
                "P2\n2 2\n65535\n0 0\n0 0\n",
                encoding="ascii",
            )
            self.assert_dense_sparse_parity(
                zero_depth,
                temporary_root,
                name="zero-depth",
            )

    def test_chunk_boundary_remains_byte_identical_to_dense(self) -> None:
        arguments = {
            "origin_world_m": (-65535.5, -0.25, -0.25),
            "dimensions": (131_073, 1, 1),
            "voxel_size_m": 0.5,
            "truncation_m": 0.5,
        }
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            dense_report, sparse_report = self.assert_dense_sparse_parity(
                FIXTURE,
                Path(temporary_directory),
                name="chunk-boundary",
                arguments=arguments,
            )
            document = json.loads(
                sparse_report.output.read_text(encoding="utf-8")
            )

        self.assertGreater(sparse_report.observed_voxels, 0)
        self.assertEqual(sparse_report.voxel_updates, dense_report.voxel_updates)
        self.assertEqual(
            [voxel["index"][0] for voxel in document["voxels"]],
            [131_071, 131_072],
        )

    def test_report_observed_count_equals_sparse_accumulator_entries(self) -> None:
        created_accumulators: list[_SparseTsdfAccumulator] = []

        def create_accumulator(total_voxels: int) -> _SparseTsdfAccumulator:
            accumulator = _SparseTsdfAccumulator(total_voxels)
            created_accumulators.append(accumulator)
            return accumulator

        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            output = Path(temporary_directory) / "instrumented.sftsdf"
            with patch(
                "spatialforge.sparse_tsdf._SparseTsdfAccumulator",
                side_effect=create_accumulator,
            ):
                report = integrate_sparse_tsdf(
                    load_scan_session(FIXTURE),
                    output,
                    **REFERENCE_ARGUMENTS,
                )

        self.assertEqual(len(created_accumulators), 1)
        self.assertEqual(
            created_accumulators[0].stored_voxels,
            report.observed_voxels,
        )

    def test_validation_and_digest_fail_before_sparse_allocation(self) -> None:
        session = load_scan_session(FIXTURE)
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            with patch(
                "spatialforge.sparse_tsdf._SparseTsdfAccumulator"
            ) as accumulator_factory:
                with self.assertRaises(TsdfError) as oversized:
                    integrate_sparse_tsdf(
                        session,
                        temporary_root / "oversized.sftsdf",
                        origin_world_m=(0.0, 0.0, 0.0),
                        dimensions=(MAX_REFERENCE_VOXELS + 1, 1, 1),
                        voxel_size_m=0.5,
                        truncation_m=0.5,
                    )
                accumulator_factory.assert_not_called()

            digest_output = temporary_root / "digest.sftsdf"
            with patch(
                "spatialforge.sparse_tsdf._SparseTsdfAccumulator"
            ) as accumulator_factory:
                with self.assertRaises(TsdfError) as digest_error:
                    integrate_sparse_tsdf(
                        session,
                        digest_output,
                        expected_replay_digest_sha256="0" * 64,
                        **REFERENCE_ARGUMENTS,
                    )
                accumulator_factory.assert_not_called()

        self.assertIn("maximum", str(oversized.exception))
        self.assertIn("inputs changed", str(digest_error.exception))
        self.assertFalse(digest_output.exists())

    def test_unobserved_volume_fails_without_output(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            output = Path(temporary_directory) / "outside.sftsdf"
            with self.assertRaises(TsdfError) as raised:
                integrate_sparse_tsdf(
                    load_scan_session(FIXTURE),
                    output,
                    origin_world_m=(-4.0, -0.25, -0.25),
                    dimensions=(2, 1, 1),
                    voxel_size_m=0.5,
                    truncation_m=0.5,
                )
            output_exists = output.exists()

        self.assertIn("do not observe any voxel", str(raised.exception))
        self.assertFalse(output_exists)

    def test_output_is_never_overwritten_even_during_publish_race(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            invalid = temporary_root / "invalid.json"
            with self.assertRaises(TsdfError) as invalid_error:
                integrate_sparse_tsdf(
                    load_scan_session(FIXTURE),
                    invalid,
                    **REFERENCE_ARGUMENTS,
                )
            self.assertFalse(invalid.exists())

            existing = temporary_root / "existing.sftsdf"
            existing.write_text("keep", encoding="ascii")
            with self.assertRaises(TsdfError):
                integrate_sparse_tsdf(
                    load_scan_session(FIXTURE),
                    existing,
                    **REFERENCE_ARGUMENTS,
                )
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
                with self.assertRaises(TsdfError) as raced_error:
                    integrate_sparse_tsdf(
                        load_scan_session(FIXTURE),
                        raced,
                        **REFERENCE_ARGUMENTS,
                    )
            raced_contents = raced.read_text(encoding="ascii")

        self.assertEqual(raced_contents, "competitor")
        self.assertIn("must end in .sftsdf", str(invalid_error.exception))
        self.assertIn("refusing to overwrite", str(raced_error.exception))

    def test_existing_surface_and_mesh_consumers_accept_sparse_artifacts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            surface_tsdf = temporary_root / "surface.sftsdf"
            surface_output = temporary_root / "surface.ply"
            integrate_sparse_tsdf(
                load_scan_session(FIXTURE),
                surface_tsdf,
                **REFERENCE_ARGUMENTS,
            )
            surface_report = extract_surface_points(
                surface_tsdf,
                surface_output,
            )

            mesh_tsdf = temporary_root / "mesh.sftsdf"
            mesh_output = temporary_root / "mesh.ply"
            integrate_sparse_tsdf(
                load_scan_session(FIXTURE),
                mesh_tsdf,
                **MESH_ARGUMENTS,
            )
            mesh_report = extract_triangle_mesh(mesh_tsdf, mesh_output)

        self.assertEqual(
            surface_report.output_digest_sha256,
            "bdb05ea65e4f5fcfa59213958a0f4ab33581e2f4d8d6df0f2f410c2c65fd9154",
        )
        self.assertEqual(surface_report.points_written, 1)
        self.assertEqual(
            mesh_report.output_digest_sha256,
            "b240a3a8286eb1025dd5e64047a63d8aa56519a21b201a050eca7c951d4b1a35",
        )
        self.assertEqual(mesh_report.triangles_written, 8)


class SparseTsdfCliTests(unittest.TestCase):
    def test_cli_reports_sparse_storage_and_exact_fixture_hash(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            output = Path(temporary_directory) / "sparse.sftsdf"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "reconstruct",
                        "tsdf-sparse",
                        str(FIXTURE),
                        str(output),
                        "--origin",
                        "0.5",
                        "-1",
                        "-1",
                        "--dimensions",
                        "2",
                        "4",
                        "4",
                        "--voxel-size-m",
                        "0.5",
                        "--truncation-m",
                        "0.5",
                    ]
                )
            output_exists = output.is_file()
            output_text = stdout.getvalue()

        self.assertEqual(exit_code, 0)
        self.assertTrue(output_exists)
        self.assertIn("SPARSE TSDF scan-synthetic-0001", output_text)
        self.assertIn("observed=8 fused=8", output_text)
        self.assertIn("storage: sparse accumulator_entries=8", output_text)
        self.assertIn(
            "e61803737cdd68b209459fb644cc2f67f18e0420d306273316808e27d2e89994",
            output_text,
        )

    def test_auto_cli_remains_on_dense_backend(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            output = Path(temporary_directory) / "automatic.sftsdf"
            with patch(
                "spatialforge.cli.integrate_sparse_tsdf"
            ) as sparse_integrator:
                with redirect_stdout(io.StringIO()):
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
        sparse_integrator.assert_not_called()

    def test_cli_failure_is_actionable(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            output = Path(temporary_directory) / "missing.sftsdf"
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                exit_code = main(
                    [
                        "reconstruct",
                        "tsdf-sparse",
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
        self.assertIn("SPARSE TSDF FAILED", stderr.getvalue())
        self.assertIn("directory does not exist", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
