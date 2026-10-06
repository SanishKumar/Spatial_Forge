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

import numpy as np

from spatialforge import plan_tsdf_blocks
from spatialforge.cli import main
from spatialforge.errors import TsdfError
from spatialforge.session_loader import load_scan_session
from spatialforge.tsdf_block_plan import (
    MAX_BLOCK_INDEX,
    MIN_BLOCK_INDEX,
    _candidate_block_span,
    _containing_block_index,
    _plan_observation_blocks,
)


TEST_ROOT = Path(__file__).resolve().parent
FIXTURE = TEST_ROOT / "fixtures" / "minimal.vgsession"
REFERENCE_ARGUMENTS = {
    "voxel_size_m": 0.125,
    "truncation_m": 0.5,
}
EXPECTED_SURFACE_BLOCKS = (
    (1, -1, -1),
    (1, 0, -1),
    (1, -1, 0),
    (1, 0, 0),
)
EXPECTED_ACTIVE_BLOCKS = (
    (0, -1, -1),
    (1, -1, -1),
    (0, 0, -1),
    (1, 0, -1),
    (0, -1, 0),
    (1, -1, 0),
    (0, 0, 0),
    (1, 0, 0),
)
EXPECTED_SECOND_SURFACE_BLOCKS = tuple(
    (x + 2, y, z) for x, y, z in EXPECTED_SURFACE_BLOCKS
)
EXPECTED_SECOND_ACTIVE_BLOCKS = tuple(
    (x + 2, y, z) for x, y, z in EXPECTED_ACTIVE_BLOCKS
)
EXPECTED_COMBINED_SURFACE_BLOCKS = tuple(
    sorted(
        EXPECTED_SURFACE_BLOCKS + EXPECTED_SECOND_SURFACE_BLOCKS,
        key=lambda index: (index[2], index[1], index[0]),
    )
)
EXPECTED_COMBINED_ACTIVE_BLOCKS = tuple(
    sorted(
        EXPECTED_ACTIVE_BLOCKS + EXPECTED_SECOND_ACTIVE_BLOCKS,
        key=lambda index: (index[2], index[1], index[0]),
    )
)


def copy_fixture(parent: Path, name: str = "case.vgsession") -> Path:
    target = parent / name
    shutil.copytree(FIXTURE, target)
    return target


def move_second_pose_to_x(session_path: Path, translation_x: float) -> None:
    pose_path = session_path / "streams" / "poses.jsonl"
    records = [
        json.loads(line)
        for line in pose_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records[1]["T_world_camera"][3] = translation_x
    pose_path.write_text(
        "".join(
            json.dumps(record, separators=(",", ":")) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


class TsdfBlockMathTests(unittest.TestCase):
    def test_half_open_activation_pins_exact_and_adjacent_boundaries(self) -> None:
        point = -0.5
        self.assertEqual(_candidate_block_span(point, 0.5, 1.0), (-1, -1))
        self.assertEqual(
            _candidate_block_span(
                math.nextafter(point, -math.inf),
                0.5,
                1.0,
            ),
            (-1, -1),
        )
        twice_below = math.nextafter(
            math.nextafter(point, -math.inf),
            -math.inf,
        )
        self.assertEqual(
            _candidate_block_span(
                twice_below,
                0.5,
                1.0,
            ),
            (-2, -1),
        )
        self.assertEqual(
            _candidate_block_span(
                math.nextafter(point, math.inf),
                0.5,
                1.0,
            ),
            (-1, 0),
        )

    def test_surface_ownership_uses_signed_half_open_blocks(self) -> None:
        boundary = -1.0
        self.assertEqual(_containing_block_index(boundary, 1.0), -1)
        self.assertEqual(
            _containing_block_index(
                math.nextafter(boundary, -math.inf),
                1.0,
            ),
            -2,
        )
        self.assertEqual(
            _containing_block_index(
                math.nextafter(boundary, math.inf),
                1.0,
            ),
            -1,
        )
        self.assertEqual(_containing_block_index(-0.0, 1.0), 0)

    def test_signed_block_index_limits_are_exact(self) -> None:
        self.assertEqual(
            _containing_block_index(float(MIN_BLOCK_INDEX), 1.0),
            MIN_BLOCK_INDEX,
        )
        self.assertEqual(
            _containing_block_index(float(MAX_BLOCK_INDEX), 1.0),
            MAX_BLOCK_INDEX,
        )
        with self.assertRaises(TsdfError):
            _containing_block_index(float(MIN_BLOCK_INDEX - 1), 1.0)
        with self.assertRaises(TsdfError):
            _containing_block_index(float(MAX_BLOCK_INDEX + 1), 1.0)

    def test_truncation_expands_a_cartesian_surface_neighborhood(self) -> None:
        tight = _candidate_block_span(0.5, 0.25, 1.0)
        expanded = _candidate_block_span(0.5, 0.75, 1.0)

        self.assertEqual(tight, (0, 0))
        self.assertEqual(expanded, (-1, 1))
        self.assertEqual((tight[1] - tight[0] + 1) ** 3, 1)
        self.assertEqual((expanded[1] - expanded[0] + 1) ** 3, 27)

    def test_rounded_boundaries_are_conservatively_overcovered(self) -> None:
        lower_extent = 0.01697795914275709
        lower_endpoint = 8262.64639804645
        lower_truncation = lower_extent
        lower_coordinate = lower_endpoint + lower_truncation
        lower_span = _candidate_block_span(
            lower_coordinate,
            lower_truncation,
            lower_extent,
        )

        upper_extent = 1.0297901307057234e-05
        upper_endpoint = -1.158225555807341
        upper_truncation = upper_extent
        upper_coordinate = upper_endpoint - upper_truncation
        upper_span = _candidate_block_span(
            upper_coordinate,
            upper_truncation,
            upper_extent,
        )

        self.assertEqual(lower_endpoint, 486_669 * lower_extent)
        self.assertEqual(upper_endpoint, -112_472 * upper_extent)
        self.assertEqual(lower_span[0], 486_668)
        self.assertEqual(upper_span[1], -112_472)


class TsdfBlockPlanTests(unittest.TestCase):
    def test_fixture_plan_has_exact_geometry_schema_and_hash(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            output = Path(temporary_directory) / "fixture.sftplan"
            report = plan_tsdf_blocks(
                load_scan_session(FIXTURE),
                output,
                **REFERENCE_ARGUMENTS,
            )
            document = json.loads(output.read_text(encoding="utf-8"))
            encoded = output.read_bytes()

        self.assertEqual(report.total_observations, 2)
        self.assertEqual(report.selected_observations, 2)
        self.assertEqual(report.paired_observations, 2)
        self.assertEqual(report.valid_depth_points, 8)
        self.assertEqual(report.invalid_depth_samples, 0)
        self.assertEqual(report.block_resolution, 8)
        self.assertEqual(report.block_extent_m, 1.0)
        self.assertEqual(report.surface_blocks, EXPECTED_SURFACE_BLOCKS)
        self.assertEqual(report.active_blocks, EXPECTED_ACTIVE_BLOCKS)
        self.assertEqual(report.surface_block_count, 4)
        self.assertEqual(report.active_block_count, 8)
        self.assertEqual(report.halo_block_count, 4)
        self.assertEqual(report.planned_voxel_slots, 4096)
        self.assertEqual(report.min_block_index, (0, -1, -1))
        self.assertEqual(report.max_block_index, (1, 0, 0))
        self.assertEqual(
            report.replay_digest_sha256,
            "dc001ae0ca01004a21ad227b12d57a7350bbc23904040453ca73fd955ef050b8",
        )
        self.assertEqual(
            report.output_digest_sha256,
            "372c7c5d49eff1a30317ceb8b67cb3c40683f1d9d2a6179a049772f763d6f79d",
        )
        self.assertEqual(
            document["schema"],
            "spatialforge.tsdf-block-plan",
        )
        self.assertEqual(document["schema_version"], "0.1.0")
        self.assertEqual(
            tuple(tuple(index) for index in document["surface_blocks"]),
            EXPECTED_SURFACE_BLOCKS,
        )
        self.assertEqual(
            tuple(tuple(index) for index in document["active_blocks"]),
            EXPECTED_ACTIVE_BLOCKS,
        )
        self.assertIn(
            b'"rule": "outward-conservative-half-open-l-infinity-cover"',
            encoded,
        )
        self.assertNotIn(str(output.parent).encode("ascii"), encoded)
        self.assertNotIn(b"created_at", encoded)

    def test_output_is_byte_deterministic(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            first = temporary_root / "first.sftplan"
            second = temporary_root / "second.sftplan"
            first_report = plan_tsdf_blocks(
                load_scan_session(FIXTURE),
                first,
                **REFERENCE_ARGUMENTS,
            )
            second_report = plan_tsdf_blocks(
                load_scan_session(FIXTURE),
                second,
                **REFERENCE_ARGUMENTS,
            )
            first_bytes = first.read_bytes()
            second_bytes = second.read_bytes()

        self.assertEqual(first_bytes, second_bytes)
        self.assertEqual(
            first_report.output_digest_sha256,
            second_report.output_digest_sha256,
        )

    def test_stride_and_missing_streams_are_explicit(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            separated = copy_fixture(temporary_root, "separated.vgsession")
            move_second_pose_to_x(separated, 2.05)
            combined_report = plan_tsdf_blocks(
                load_scan_session(separated),
                temporary_root / "combined.sftplan",
                **REFERENCE_ARGUMENTS,
            )
            self.assertEqual(
                combined_report.surface_blocks,
                EXPECTED_COMBINED_SURFACE_BLOCKS,
            )
            self.assertEqual(
                combined_report.active_blocks,
                EXPECTED_COMBINED_ACTIVE_BLOCKS,
            )

            stride_report = plan_tsdf_blocks(
                load_scan_session(separated),
                temporary_root / "stride.sftplan",
                frame_stride=2,
                **REFERENCE_ARGUMENTS,
            )
            self.assertEqual(stride_report.selected_observations, 1)
            self.assertEqual(stride_report.paired_observations, 1)
            self.assertEqual(stride_report.valid_depth_points, 4)
            self.assertEqual(
                stride_report.active_blocks,
                EXPECTED_ACTIVE_BLOCKS,
            )

            missing_pose = copy_fixture(temporary_root, "missing-pose.vgsession")
            move_second_pose_to_x(missing_pose, 2.05)
            pose_path = missing_pose / "streams" / "poses.jsonl"
            pose_path.write_text(
                pose_path.read_text(encoding="utf-8").splitlines()[0] + "\n",
                encoding="utf-8",
            )
            pose_report = plan_tsdf_blocks(
                load_scan_session(missing_pose),
                temporary_root / "missing-pose.sftplan",
                **REFERENCE_ARGUMENTS,
            )
            self.assertEqual(pose_report.paired_observations, 1)
            self.assertEqual(pose_report.skipped_missing_pose, 1)
            self.assertEqual(pose_report.active_blocks, EXPECTED_ACTIVE_BLOCKS)

            missing_depth = copy_fixture(
                temporary_root,
                "missing-depth.vgsession",
            )
            move_second_pose_to_x(missing_depth, 2.05)
            depth_path = missing_depth / "streams" / "depth.jsonl"
            depth_path.write_text(
                depth_path.read_text(encoding="utf-8").splitlines()[0] + "\n",
                encoding="utf-8",
            )
            depth_report = plan_tsdf_blocks(
                load_scan_session(missing_depth),
                temporary_root / "missing-depth.sftplan",
                **REFERENCE_ARGUMENTS,
            )
            self.assertEqual(depth_report.paired_observations, 1)
            self.assertEqual(depth_report.skipped_missing_depth, 1)
            self.assertEqual(depth_report.active_blocks, EXPECTED_ACTIVE_BLOCKS)

    def test_invalid_depth_is_counted_and_all_invalid_fails_atomically(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            partly_invalid = copy_fixture(
                temporary_root,
                "partly-invalid.vgsession",
            )
            move_second_pose_to_x(partly_invalid, 2.05)
            first_depth = partly_invalid / "data" / "depth" / "000000.pgm"
            first_depth.write_text(
                "P2\n2 2\n65535\n0 0\n0 0\n",
                encoding="ascii",
            )
            partial_report = plan_tsdf_blocks(
                load_scan_session(partly_invalid),
                temporary_root / "partly-invalid.sftplan",
                **REFERENCE_ARGUMENTS,
            )
            self.assertEqual(partial_report.valid_depth_points, 4)
            self.assertEqual(partial_report.invalid_depth_samples, 4)
            self.assertEqual(
                partial_report.surface_blocks,
                EXPECTED_SECOND_SURFACE_BLOCKS,
            )
            self.assertEqual(
                partial_report.active_blocks,
                EXPECTED_SECOND_ACTIVE_BLOCKS,
            )

            all_invalid = copy_fixture(
                temporary_root,
                "all-invalid.vgsession",
            )
            for filename in ("000000.pgm", "000001.pgm"):
                (all_invalid / "data" / "depth" / filename).write_text(
                    "P2\n2 2\n65535\n0 0\n0 0\n",
                    encoding="ascii",
                )
            failed_output = temporary_root / "all-invalid.sftplan"
            with self.assertRaises(TsdfError) as raised:
                plan_tsdf_blocks(
                    load_scan_session(all_invalid),
                    failed_output,
                    **REFERENCE_ARGUMENTS,
                )
            failed_output_exists = failed_output.exists()

        self.assertIn("no positive finite depth", str(raised.exception))
        self.assertFalse(failed_output_exists)

    def test_parameters_and_derived_extent_are_bounded(self) -> None:
        session = load_scan_session(FIXTURE)
        invalid_cases = (
            ({"voxel_size_m": 0.0, "truncation_m": 0.5}, "voxel_size_m"),
            ({"voxel_size_m": math.nan, "truncation_m": 0.5}, "voxel_size_m"),
            ({"voxel_size_m": 0.5, "truncation_m": math.inf}, "truncation_m"),
            (
                {"voxel_size_m": 0.5, "truncation_m": 0.25},
                "greater than or equal",
            ),
            (
                {"voxel_size_m": 1e308, "truncation_m": 1e308},
                "block extent",
            ),
            (
                {
                    "voxel_size_m": 0.5,
                    "truncation_m": 0.5,
                    "frame_stride": 0,
                },
                "frame_stride",
            ),
        )
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            for index, (arguments, message) in enumerate(invalid_cases):
                output = temporary_root / f"invalid-{index}.sftplan"
                with self.subTest(arguments=arguments):
                    with self.assertRaises(TsdfError) as raised:
                        plan_tsdf_blocks(session, output, **arguments)
                    self.assertIn(message, str(raised.exception))
                    self.assertFalse(output.exists())

    def test_active_block_cap_has_an_exact_boundary(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            with patch(
                "spatialforge.tsdf_block_plan.MAX_PLANNED_BLOCKS",
                8,
            ):
                allowed = plan_tsdf_blocks(
                    load_scan_session(FIXTURE),
                    temporary_root / "allowed.sftplan",
                    **REFERENCE_ARGUMENTS,
                )
            self.assertEqual(allowed.active_block_count, 8)

            rejected_output = temporary_root / "rejected.sftplan"
            with patch(
                "spatialforge.tsdf_block_plan.MAX_PLANNED_BLOCKS",
                7,
            ):
                with self.assertRaises(TsdfError) as rejected:
                    plan_tsdf_blocks(
                        load_scan_session(FIXTURE),
                        rejected_output,
                        **REFERENCE_ARGUMENTS,
                    )
            self.assertFalse(rejected_output.exists())

            single_output = temporary_root / "single-span.sftplan"

            def wide_spans(coordinate, truncation_m, block_extent_m):
                return (
                    np.full(len(coordinate), -1, dtype=np.int64),
                    np.full(len(coordinate), 1, dtype=np.int64),
                )

            # Both forms of the span rule are widened, so the vectorised
            # planner sees a 27-block sample, defers, and the per-sample
            # limit is reported by the reference path that owns it.
            with (
                patch(
                    "spatialforge.tsdf_block_plan.MAX_PLANNED_BLOCKS",
                    26,
                ),
                patch(
                    "spatialforge.tsdf_block_plan._candidate_block_span",
                    return_value=(-1, 1),
                ),
                patch(
                    "spatialforge.tsdf_block_plan."
                    "_candidate_block_span_vector",
                    side_effect=wide_spans,
                ),
            ):
                with self.assertRaises(TsdfError) as single:
                    plan_tsdf_blocks(
                        load_scan_session(FIXTURE),
                        single_output,
                        **REFERENCE_ARGUMENTS,
                    )
            self.assertFalse(single_output.exists())

        self.assertIn("maximum", str(rejected.exception))
        self.assertIn("one depth sample", str(single.exception))

    def test_invalid_existing_and_racing_outputs_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            with patch(
                "spatialforge.tsdf_block_plan._plan_observation_blocks"
            ) as planner:
                invalid = temporary_root / "invalid.json"
                with self.assertRaises(TsdfError) as invalid_error:
                    plan_tsdf_blocks(
                        load_scan_session(FIXTURE),
                        invalid,
                        **REFERENCE_ARGUMENTS,
                    )
                planner.assert_not_called()

                existing = temporary_root / "existing.sftplan"
                existing.write_text("keep", encoding="ascii")
                with self.assertRaises(TsdfError):
                    plan_tsdf_blocks(
                        load_scan_session(FIXTURE),
                        existing,
                        **REFERENCE_ARGUMENTS,
                    )
                planner.assert_not_called()
            self.assertEqual(existing.read_text(encoding="ascii"), "keep")

            raced = temporary_root / "raced.sftplan"
            actual_link = os.link

            def create_competing_output(
                staging_path: str | Path,
                output_path: str | Path,
            ) -> None:
                Path(output_path).write_text("competitor", encoding="ascii")
                actual_link(staging_path, output_path)

            with patch(
                "spatialforge.tsdf_block_plan.os.link",
                side_effect=create_competing_output,
            ):
                with self.assertRaises(TsdfError) as raced_error:
                    plan_tsdf_blocks(
                        load_scan_session(FIXTURE),
                        raced,
                        **REFERENCE_ARGUMENTS,
                    )
            raced_contents = raced.read_text(encoding="ascii")

        self.assertIn("must end in .sftplan", str(invalid_error.exception))
        self.assertEqual(raced_contents, "competitor")
        self.assertIn("refusing to overwrite", str(raced_error.exception))

    def test_changed_input_digest_fails_before_publication(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            fixture = copy_fixture(temporary_root, "changed.vgsession")
            session = load_scan_session(fixture)
            output = temporary_root / "changed.sftplan"
            mutation_done = False

            def plan_then_change_input(*arguments, **keyword_arguments):
                nonlocal mutation_done
                result = _plan_observation_blocks(
                    *arguments,
                    **keyword_arguments,
                )
                if not mutation_done:
                    depth_path = fixture / "data" / "depth" / "000000.pgm"
                    depth_path.write_text(
                        depth_path.read_text(encoding="ascii").replace(
                            "1000",
                            "999",
                            1,
                        ),
                        encoding="ascii",
                    )
                    mutation_done = True
                return result

            with patch(
                "spatialforge.tsdf_block_plan._plan_observation_blocks",
                side_effect=plan_then_change_input,
            ):
                with self.assertRaises(TsdfError) as raised:
                    plan_tsdf_blocks(
                        session,
                        output,
                        **REFERENCE_ARGUMENTS,
                    )
            output_exists = output.exists()

        self.assertIn("inputs changed", str(raised.exception))
        self.assertFalse(output_exists)


class TsdfBlockPlanCliTests(unittest.TestCase):
    def test_cli_writes_and_reports_the_fixture_plan(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            output = Path(temporary_directory) / "fixture.sftplan"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "reconstruct",
                        "tsdf-block-plan",
                        str(FIXTURE),
                        str(output),
                        "--voxel-size-m",
                        "0.125",
                        "--truncation-m",
                        "0.5",
                    ]
                )
            output_exists = output.is_file()
            output_text = stdout.getvalue()

        self.assertEqual(exit_code, 0)
        self.assertTrue(output_exists)
        self.assertIn("TSDF BLOCK PLAN scan-synthetic-0001", output_text)
        self.assertIn("valid=8 invalid=0", output_text)
        self.assertIn(
            "surface=4 active=8 halo=4 voxel_slots=4096",
            output_text,
        )
        self.assertIn(
            "372c7c5d49eff1a30317ceb8b67cb3c40683f1d9d2a6179a049772f763d6f79d",
            output_text,
        )

    def test_cli_failure_is_actionable(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            output = Path(temporary_directory) / "missing.sftplan"
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                exit_code = main(
                    [
                        "reconstruct",
                        "tsdf-block-plan",
                        str(TEST_ROOT / "missing.vgsession"),
                        str(output),
                    ]
                )
            output_exists = output.exists()

        self.assertEqual(exit_code, 2)
        self.assertFalse(output_exists)
        self.assertIn("TSDF BLOCK PLAN FAILED", stderr.getvalue())
        self.assertIn("directory does not exist", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
