from __future__ import annotations

import io
import shutil
import tempfile
import unittest
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path
from unittest.mock import MagicMock, patch

from spatialforge import build_tsdf_replay_depth_context, load_tsdf_block_plan
from spatialforge.cli import main
from spatialforge.errors import TsdfError
from spatialforge.session_loader import load_scan_session
from spatialforge.tsdf_block_plan import plan_tsdf_blocks
from spatialforge.tsdf_block_plan_loader import TsdfBlockPlan
from spatialforge.tsdf_observation_block_rays import (
    trace_tsdf_observation_block_rays_from_context,
)
from spatialforge.tsdf_observation_footprint import (
    MAX_TSDF_OBSERVATION_FOOTPRINT_CANDIDATE_BLOCKS,
    TsdfObservationFootprintReceipt,
    survey_tsdf_observation_pixel_footprints_from_context,
)
from spatialforge.tsdf_pixel_footprint_coverage import (
    TsdfPixelFootprintStatus,
    evaluate_tsdf_pixel_footprint_coverage_from_context,
)
from spatialforge.tsdf_replay_depth_context import (
    TsdfReplayDepthContext,
    TsdfReplayDepthStatus,
)


TEST_ROOT = Path(__file__).resolve().parent
FIXTURE = TEST_ROOT / "fixtures" / "minimal.vgsession"
PLAN_ARGUMENTS = {
    "voxel_size_m": 0.125,
    "truncation_m": 0.5,
}
PLAN_SHA256 = (
    "372c7c5d49eff1a30317ceb8b67cb3c40683f1d9d2a6179a049772f763d6f79d"
)
REPLAY_SHA256 = (
    "dc001ae0ca01004a21ad227b12d57a7350bbc23904040453ca73fd955ef050b8"
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

_FORBIDDEN_FOOTPRINT_TARGETS = (
    "spatialforge.tsdf_observation_footprint.replay_session",
    "spatialforge.tsdf_observation_footprint."
    "build_tsdf_replay_depth_context",
    "spatialforge.tsdf_observation_footprint.allocate_empty_tsdf_blocks",
    "spatialforge.tsdf_observation_footprint.locate_tsdf_voxel",
    "spatialforge.tsdf_observation_footprint."
    "evaluate_tsdf_voxel_contribution_from_context",
    "spatialforge.tsdf_observation_footprint."
    "apply_tsdf_voxel_contribution_from_context",
    "spatialforge.tsdf_observation_footprint."
    "traverse_tsdf_plan_blocks_from_context",
    "spatialforge.tsdf_observation_footprint."
    "trace_tsdf_observation_block_rays_from_context",
    "spatialforge.replay._file_digest",
    "pathlib.Path.open",
    "PIL.Image.open",
)

_FORBIDDEN_CLI_TARGETS = (
    "spatialforge.cli.allocate_empty_tsdf_blocks",
    "spatialforge.cli.locate_tsdf_voxel",
    "spatialforge.cli.classify_tsdf_voxel_sampling_from_context",
    "spatialforge.cli.classify_tsdf_voxel_across_observations_from_context",
    "spatialforge.cli.trace_tsdf_observation_block_rays_from_context",
    "spatialforge.cli.survey_tsdf_plan_block_rays_from_context",
    "spatialforge.cli.evaluate_tsdf_voxel_contribution",
    "spatialforge.cli.evaluate_tsdf_voxel_contribution_from_context",
    "spatialforge.cli.apply_tsdf_voxel_contribution",
    "spatialforge.cli.apply_tsdf_voxel_contribution_from_context",
    "spatialforge.cli.traverse_tsdf_voxel_observations",
    "spatialforge.cli.traverse_tsdf_voxel_observations_from_context",
    "spatialforge.cli.traverse_tsdf_block_voxels_from_context",
    "spatialforge.cli.traverse_tsdf_plan_blocks_from_context",
    "spatialforge.cli.reconstruct_point_cloud",
    "spatialforge.cli.integrate_tsdf",
    "spatialforge.cli.integrate_sparse_tsdf",
    "spatialforge.cli.infer_tsdf_bounds",
    "spatialforge.cli.plan_tsdf_blocks",
    "spatialforge.cli.extract_surface_points",
    "spatialforge.cli.extract_triangle_mesh",
    "spatialforge.tsdf._write_output_without_overwrite",
    "spatialforge.tsdf_block_plan._write_plan_without_overwrite",
    "spatialforge.surface._write_surface_ply",
    "spatialforge.mesh._write_mesh_ply",
)


def copy_fixture(parent: Path, name: str = "case.vgsession") -> Path:
    target = parent / name
    shutil.copytree(FIXTURE, target)
    return target


def create_plan(
    parent: Path,
    *,
    session_path: Path = FIXTURE,
    name: str = "fixture.sftplan",
    frame_stride: int = 1,
) -> Path:
    output = parent / name
    plan_tsdf_blocks(
        load_scan_session(session_path),
        output,
        frame_stride=frame_stride,
        **PLAN_ARGUMENTS,
    )
    return output


def load_case(
    parent: Path,
    *,
    session_path: Path = FIXTURE,
    frame_stride: int = 1,
    name: str = "fixture.sftplan",
) -> tuple[TsdfBlockPlan, TsdfReplayDepthContext]:
    plan = load_tsdf_block_plan(
        create_plan(
            parent,
            session_path=session_path,
            frame_stride=frame_stride,
            name=name,
        )
    )
    session = load_scan_session(session_path)
    context = build_tsdf_replay_depth_context(plan, session)
    return plan, context


def tree_snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def plan_snapshot(plan: TsdfBlockPlan) -> tuple[tuple[str, object], ...]:
    return tuple((field.name, getattr(plan, field.name)) for field in fields(plan))


def set_depth_sample(session_path: Path, filename: str, raw_depth: int) -> None:
    (session_path / "data" / "depth" / filename).write_text(
        "P2\n2 2\n65535\n"
        f"{raw_depth} {raw_depth}\n{raw_depth} {raw_depth}\n",
        encoding="ascii",
    )


def set_first_depth_sample_invalid(session_path: Path) -> None:
    (session_path / "data" / "depth" / "000000.pgm").write_text(
        "P2\n2 2\n65535\n0 1000\n1000 1000\n",
        encoding="ascii",
    )


def remove_first_record(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text(lines[1] + "\n", encoding="utf-8")


@contextmanager
def forbidden_footprint_calls():
    with ExitStack() as stack:
        mocks: list[MagicMock] = []
        for target in _FORBIDDEN_FOOTPRINT_TARGETS:
            mocks.append(stack.enter_context(patch(target, create=True)))
        yield mocks


@contextmanager
def forbidden_cli_calls():
    with ExitStack() as stack:
        mocks: list[MagicMock] = []
        for target in _FORBIDDEN_CLI_TARGETS:
            mocks.append(stack.enter_context(patch(target, create=True)))
        yield mocks


class TsdfObservationFootprintTests(unittest.TestCase):
    def test_fixture_observation_covers_every_pixel_wedge_exactly(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            plan, context = load_case(temporary_root)
            before_tree = tree_snapshot(temporary_root)
            before_plan = plan_snapshot(plan)

            with forbidden_footprint_calls() as forbidden:
                receipt = (
                    survey_tsdf_observation_pixel_footprints_from_context(
                        plan,
                        context,
                        0,
                    )
                )
            repeated = survey_tsdf_observation_pixel_footprints_from_context(
                plan,
                context,
                0,
            )
            after_tree = tree_snapshot(temporary_root)

        self.assertIsInstance(receipt, TsdfObservationFootprintReceipt)
        self.assertEqual(receipt, repeated)
        self.assertEqual(receipt.source_plan_digest_sha256, PLAN_SHA256)
        self.assertEqual(receipt.replay_digest_sha256, REPLAY_SHA256)
        self.assertIs(
            receipt.observation_status,
            TsdfReplayDepthStatus.READY,
        )
        self.assertEqual(receipt.observation_sequence, 0)
        self.assertEqual(receipt.image_size, (2, 2))
        self.assertEqual(receipt.camera_origin_world_m, (0.0, 0.0, 0.0))
        self.assertEqual(
            tuple(child.pixel_uv for child in receipt.pixel_receipts),
            ((0, 0), (1, 0), (0, 1), (1, 1)),
        )
        self.assertEqual(receipt.pixel_count, 4)
        self.assertEqual(receipt.covered_pixel_count, 4)
        self.assertEqual(receipt.depth_invalid_count, 0)
        self.assertEqual(receipt.candidate_block_count, 18)
        self.assertEqual(receipt.rejected_candidate_count, 0)
        self.assertEqual(receipt.block_visit_count, 18)
        self.assertEqual(receipt.duplicate_block_visit_count, 10)
        self.assertEqual(receipt.maximum_blocks_per_pixel, 8)
        self.assertEqual(receipt.covered_block_indices, EXPECTED_ACTIVE_BLOCKS)
        self.assertEqual(
            receipt.existing_plan_block_indices,
            EXPECTED_ACTIVE_BLOCKS,
        )
        self.assertEqual(receipt.unplanned_block_indices, ())
        self.assertEqual(receipt.maximum_block_pixel_count, 4)
        # Per pixel the wedge is strictly wider than its centreline, but this
        # 2x2 image's four centrelines already reach all eight blocks, so the
        # observation-level union does not widen. The far-depth test covers
        # the case where it does.
        self.assertEqual(
            receipt.centerline_block_indices,
            EXPECTED_ACTIVE_BLOCKS,
        )
        self.assertEqual(receipt.footprint_only_block_indices, ())
        self.assertFalse(receipt.widens_centerline_coverage)
        self.assertTrue(receipt.prepared_depth_accessed)
        self.assertEqual(plan_snapshot(plan), before_plan)
        self.assertEqual(after_tree, before_tree)
        for forbidden_call in forbidden:
            forbidden_call.assert_not_called()

    def test_coverage_contains_the_centerline_ray_union(self) -> None:
        cases = ((1000, "near", 8, 8), (3000, "far", 52, 16))
        for raw_depth, name, expected_covered, expected_rays in cases:
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory(
                    dir=TEST_ROOT
                ) as temporary_directory:
                    temporary_root = Path(temporary_directory)
                    session_path = copy_fixture(temporary_root)
                    for filename in ("000000.pgm", "000001.pgm"):
                        set_depth_sample(session_path, filename, raw_depth)
                    plan, context = load_case(
                        temporary_root,
                        session_path=session_path,
                    )

                    footprint = (
                        survey_tsdf_observation_pixel_footprints_from_context(
                            plan,
                            context,
                            0,
                        )
                    )
                    rays = trace_tsdf_observation_block_rays_from_context(
                        plan,
                        context,
                        0,
                    )

                    self.assertEqual(
                        len(footprint.covered_block_indices),
                        expected_covered,
                    )
                    self.assertEqual(
                        len(rays.covered_block_indices),
                        expected_rays,
                    )
                    self.assertEqual(
                        set(rays.covered_block_indices),
                        set(footprint.centerline_block_indices),
                    )
                    self.assertTrue(
                        set(rays.covered_block_indices)
                        <= set(footprint.covered_block_indices),
                        "centreline union escaped the footprint coverage",
                    )

    def test_far_observation_widens_coverage_and_reports_unplanned(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            session_path = copy_fixture(temporary_root)
            for filename in ("000000.pgm", "000001.pgm"):
                set_depth_sample(session_path, filename, 3000)
            plan, context = load_case(
                temporary_root,
                session_path=session_path,
            )
            before_plan = plan_snapshot(plan)
            before_tree = tree_snapshot(temporary_root)

            receipt = survey_tsdf_observation_pixel_footprints_from_context(
                plan,
                context,
                0,
            )
            after_tree = tree_snapshot(temporary_root)

        self.assertEqual(len(plan.active_blocks), 32)
        self.assertEqual(receipt.candidate_block_count, 100)
        self.assertEqual(receipt.rejected_candidate_count, 16)
        self.assertEqual(receipt.block_visit_count, 84)
        self.assertEqual(len(receipt.covered_block_indices), 52)
        self.assertEqual(receipt.duplicate_block_visit_count, 32)
        self.assertEqual(receipt.maximum_blocks_per_pixel, 31)
        self.assertEqual(len(receipt.centerline_block_indices), 16)
        self.assertEqual(len(receipt.footprint_only_block_indices), 36)
        self.assertTrue(receipt.widens_centerline_coverage)
        self.assertEqual(len(receipt.existing_plan_block_indices), 32)
        self.assertEqual(len(receipt.unplanned_block_indices), 20)
        self.assertEqual(
            set(receipt.unplanned_block_indices) & set(plan.active_blocks),
            set(),
        )
        self.assertEqual(plan_snapshot(plan), before_plan)
        self.assertEqual(after_tree, before_tree)

    def test_children_equal_independent_per_pixel_evaluations(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            plan, context = load_case(temporary_root)
            receipt = survey_tsdf_observation_pixel_footprints_from_context(
                plan,
                context,
                0,
            )
            expected = tuple(
                evaluate_tsdf_pixel_footprint_coverage_from_context(
                    plan,
                    context,
                    0,
                    pixel_uv,
                )
                for pixel_uv in ((0, 0), (1, 0), (0, 1), (1, 1))
            )

        self.assertEqual(receipt.pixel_receipts, expected)

    def test_invalid_depth_pixels_contribute_no_coverage(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            session_path = copy_fixture(temporary_root)
            set_first_depth_sample_invalid(session_path)
            plan, context = load_case(
                temporary_root,
                session_path=session_path,
            )
            before_tree = tree_snapshot(temporary_root)

            receipt = survey_tsdf_observation_pixel_footprints_from_context(
                plan,
                context,
                0,
            )
            after_tree = tree_snapshot(temporary_root)

        invalid = receipt.pixel_receipts[0]
        self.assertIs(invalid.status, TsdfPixelFootprintStatus.DEPTH_INVALID)
        self.assertEqual(invalid.covered_block_indices, ())
        self.assertEqual(invalid.candidate_block_count, 0)
        self.assertEqual(receipt.pixel_count, 4)
        self.assertEqual(receipt.covered_pixel_count, 3)
        self.assertEqual(receipt.depth_invalid_count, 1)
        self.assertEqual(receipt.candidate_block_count, 16)
        self.assertEqual(receipt.block_visit_count, 16)
        self.assertEqual(receipt.duplicate_block_visit_count, 8)
        self.assertEqual(receipt.covered_block_indices, EXPECTED_ACTIVE_BLOCKS)
        self.assertEqual(receipt.maximum_block_pixel_count, 3)
        self.assertEqual(after_tree, before_tree)

    def test_missing_inputs_return_stable_zero_coverage_receipts(
        self,
    ) -> None:
        cases = (
            (
                "missing-depth",
                True,
                False,
                TsdfReplayDepthStatus.MISSING_DEPTH,
                (0.0, 0.0, 0.0),
            ),
            (
                "missing-pose",
                False,
                True,
                TsdfReplayDepthStatus.MISSING_POSE,
                None,
            ),
            (
                "missing-both",
                True,
                True,
                TsdfReplayDepthStatus.MISSING_DEPTH_AND_POSE,
                None,
            ),
        )
        for name, remove_depth, remove_pose, status, origin in cases:
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory(
                    dir=TEST_ROOT
                ) as temporary_directory:
                    temporary_root = Path(temporary_directory)
                    session_path = copy_fixture(temporary_root)
                    if remove_depth:
                        remove_first_record(
                            session_path / "streams" / "depth.jsonl"
                        )
                    if remove_pose:
                        remove_first_record(
                            session_path / "streams" / "poses.jsonl"
                        )
                    plan, context = load_case(
                        temporary_root,
                        session_path=session_path,
                    )
                    before_tree = tree_snapshot(temporary_root)

                    receipt = (
                        survey_tsdf_observation_pixel_footprints_from_context(
                            plan,
                            context,
                            0,
                        )
                    )

                    self.assertIs(receipt.observation_status, status)
                    self.assertEqual(receipt.camera_origin_world_m, origin)
                    self.assertEqual(receipt.pixel_receipts, ())
                    self.assertEqual(receipt.covered_block_indices, ())
                    self.assertEqual(receipt.centerline_block_indices, ())
                    self.assertEqual(receipt.pixel_count, 0)
                    self.assertEqual(receipt.covered_pixel_count, 0)
                    self.assertEqual(receipt.candidate_block_count, 0)
                    self.assertEqual(receipt.block_visit_count, 0)
                    self.assertEqual(receipt.maximum_blocks_per_pixel, 0)
                    self.assertFalse(receipt.widens_centerline_coverage)
                    self.assertFalse(receipt.prepared_depth_accessed)
                    self.assertEqual(tree_snapshot(temporary_root), before_tree)

    def test_selection_and_provenance_are_preflighted(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            plan, context = load_case(temporary_root)
            stride_plan, stride_context = load_case(
                temporary_root,
                frame_stride=2,
                name="stride.sftplan",
            )
            before_tree = tree_snapshot(temporary_root)
            cases = (
                ("plan-type", object(), context, 0, "TsdfBlockPlan"),
                (
                    "context-type",
                    plan,
                    object(),
                    0,
                    "TsdfReplayDepthContext",
                ),
                ("bool", plan, context, True, "sequence"),
                ("negative", plan, context, -1, "sequence"),
                ("out-of-range", plan, context, 2, "sequence"),
                (
                    "unselected",
                    stride_plan,
                    stride_context,
                    1,
                    "selected",
                ),
                (
                    "digest",
                    plan,
                    replace(context, source_plan_digest_sha256="0" * 64),
                    0,
                    "source plan digest",
                ),
                (
                    "extent",
                    replace(plan, block_extent_m=2.0),
                    context,
                    0,
                    "block_extent_m",
                ),
            )
            for name, case_plan, case_context, sequence, message in cases:
                with self.subTest(name=name):
                    with self.assertRaises(TsdfError) as raised:
                        survey_tsdf_observation_pixel_footprints_from_context(
                            case_plan,  # type: ignore[arg-type]
                            case_context,  # type: ignore[arg-type]
                            sequence,  # type: ignore[arg-type]
                        )
                    self.assertIn(message, str(raised.exception))
            after_tree = tree_snapshot(temporary_root)

        self.assertEqual(after_tree, before_tree)

    def test_candidate_cap_is_accumulated_and_atomic(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            plan, context = load_case(temporary_root)
            before_tree = tree_snapshot(temporary_root)
            before_plan = plan_snapshot(plan)

            self.assertEqual(
                MAX_TSDF_OBSERVATION_FOOTPRINT_CANDIDATE_BLOCKS,
                262_144,
            )
            with patch(
                "spatialforge.tsdf_observation_footprint."
                "MAX_TSDF_OBSERVATION_FOOTPRINT_CANDIDATE_BLOCKS",
                10,
            ):
                with self.assertRaises(TsdfError) as raised:
                    survey_tsdf_observation_pixel_footprints_from_context(
                        plan,
                        context,
                        0,
                    )
            with patch(
                "spatialforge.tsdf_observation_footprint."
                "MAX_TSDF_OBSERVATION_FOOTPRINT_CANDIDATE_BLOCKS",
                18,
            ):
                receipt = (
                    survey_tsdf_observation_pixel_footprints_from_context(
                        plan,
                        context,
                        0,
                    )
                )
            after_tree = tree_snapshot(temporary_root)

        self.assertIn("candidate blocks reach", str(raised.exception))
        self.assertIn("maximum is 10", str(raised.exception))
        self.assertEqual(receipt.candidate_block_count, 18)
        self.assertEqual(plan_snapshot(plan), before_plan)
        self.assertEqual(after_tree, before_tree)

    def test_receipts_are_frozen_slotted_and_strict(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            plan, context = load_case(Path(temporary_directory))
            receipt = survey_tsdf_observation_pixel_footprints_from_context(
                plan,
                context,
                0,
            )

        self.assertFalse(hasattr(receipt, "__dict__"))
        with self.assertRaises(FrozenInstanceError):
            receipt.observation_sequence = 1  # type: ignore[misc]

        for arguments in (
            {"pixel_receipts": tuple(reversed(receipt.pixel_receipts))},
            {"pixel_receipts": receipt.pixel_receipts[:2]},
            {"covered_block_indices": receipt.covered_block_indices[:-1]},
            {
                "covered_block_indices": tuple(
                    reversed(receipt.covered_block_indices)
                )
            },
            {
                "existing_plan_block_indices": (),
                "unplanned_block_indices": (),
            },
            {
                "existing_plan_block_indices": (),
                "unplanned_block_indices": receipt.covered_block_indices,
            },
            {"source_plan_block_indices": ()},
            {"source_plan_digest_sha256": "0"},
            {"observation_sequence": 999},
            {"block_resolution": 7},
            {"block_extent_m": 0.0},
            {"image_size": (2, 3)},
            {"camera_origin_world_m": (0.0, 0.0, 5.0)},
            {"observation_status": TsdfReplayDepthStatus.MISSING_POSE},
        ):
            with self.subTest(arguments=arguments):
                with self.assertRaises(TsdfError):
                    replace(receipt, **arguments)

    def test_late_receipt_failure_leaves_inputs_and_files_unchanged(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            plan, context = load_case(temporary_root)
            before_tree = tree_snapshot(temporary_root)
            before_plan = plan_snapshot(plan)
            injected = RuntimeError("injected final footprint survey failure")

            with patch(
                "spatialforge.tsdf_observation_footprint."
                "TsdfObservationFootprintReceipt",
                side_effect=injected,
            ):
                with self.assertRaises(TsdfError) as raised:
                    survey_tsdf_observation_pixel_footprints_from_context(
                        plan,
                        context,
                        0,
                    )
            after_tree = tree_snapshot(temporary_root)

        self.assertIs(raised.exception.__cause__, injected)
        self.assertIn(
            "injected final footprint survey failure",
            str(raised.exception),
        )
        self.assertEqual(plan_snapshot(plan), before_plan)
        self.assertEqual(after_tree, before_tree)


class TsdfObservationFootprintCliTests(unittest.TestCase):
    def test_cli_reports_exact_read_only_footprint_survey(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            session_path = copy_fixture(temporary_root)
            plan_path = create_plan(
                temporary_root,
                session_path=session_path,
            )
            before_tree = tree_snapshot(temporary_root)
            stdout = io.StringIO()
            stderr = io.StringIO()
            actual = survey_tsdf_observation_pixel_footprints_from_context

            with (
                forbidden_cli_calls() as forbidden,
                patch(
                    "spatialforge.cli."
                    "survey_tsdf_observation_pixel_footprints_from_context",
                    wraps=actual,
                ) as surveyor,
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                exit_code = main(
                    [
                        "reconstruct",
                        "tsdf-block-context-observation-footprint",
                        str(plan_path),
                        str(session_path),
                        "--observation-sequence",
                        "0",
                    ]
                )
            after_tree = tree_snapshot(temporary_root)

        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(after_tree, before_tree)
        surveyor.assert_called_once()
        for forbidden_call in forbidden:
            forbidden_call.assert_not_called()

        output = stdout.getvalue()
        for expected in (
            "TSDF BLOCK CONTEXT OBSERVATION FOOTPRINT CHECK "
            "scan-synthetic-0001\n",
            "observation: sequence=0 status=ready\n",
            "image: width=2 height=2\n",
            "pixel_outcomes: total=4 covered=4 depth_invalid=0\n",
            "candidate_blocks: total=18 rejected=0\n",
            "pixel_block_visits: total=18 unique=8 duplicate=10 "
            "maximum_per_pixel=8\n",
            "coverage_blocks: total=8 centerline=8 footprint_only=0\n",
            "centerline_contained_in_coverage: yes\n",
            "widens_centerline_coverage: no\n",
            "coverage_partition: existing_plan=8 unplanned=0\n",
            "coverage_support: maximum_pixels_per_block=4\n",
            "footprint_workload: candidate_blocks=18 maximum=262144\n",
            "coverage_scope: one-prepared-observation-all-pixels\n",
            "sampling_rule: nearest-pixel-half-open-unit-square\n",
            "coverage_rule: conservative-plane-superset-of-half-open-cells\n",
            "multi_observation_coverage_computed: no\n",
            "per_voxel_verdict_applied: no\n",
            "coverage_approved_for_expansion: no\n",
            "free_space_carving_applied: no\n",
            "plan_expanded: no\n",
            "storage_allocated: no\n",
            "storage_mutated: no\n",
            "full_fusion_performed: no\n",
            "artifact_written: no\n",
            f"plan_sha256: {PLAN_SHA256}\n",
            f"replay_digest_sha256: {REPLAY_SHA256}\n",
        ):
            self.assertIn(expected, output)

    def test_cli_reports_failure_without_writing_or_traceback(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            session_path = copy_fixture(temporary_root)
            plan_path = create_plan(
                temporary_root,
                session_path=session_path,
            )
            before_tree = tree_snapshot(temporary_root)
            stdout = io.StringIO()
            stderr = io.StringIO()

            with (
                patch(
                    "spatialforge.cli."
                    "survey_tsdf_observation_pixel_footprints_from_context",
                    side_effect=TsdfError("injected footprint survey failure"),
                ) as surveyor,
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                exit_code = main(
                    [
                        "reconstruct",
                        "tsdf-block-context-observation-footprint",
                        str(plan_path),
                        str(session_path),
                        "--observation-sequence",
                        "0",
                    ]
                )
            after_tree = tree_snapshot(temporary_root)

        self.assertEqual(exit_code, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(after_tree, before_tree)
        surveyor.assert_called_once()
        error = stderr.getvalue()
        self.assertIn("TSDF BLOCK CONTEXT OBSERVATION FOOTPRINT FAILED", error)
        self.assertIn("injected footprint survey failure", error)
        self.assertNotIn("Traceback", error)


if __name__ == "__main__":
    unittest.main()
