from __future__ import annotations

import io
import shutil
import tempfile
import unittest
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path
from unittest.mock import MagicMock, patch

from spatialforge import (
    allocate_empty_tsdf_blocks,
    build_tsdf_replay_depth_context,
    load_tsdf_block_plan,
)
from spatialforge.cli import main
from spatialforge.errors import TsdfError
from spatialforge.session_loader import load_scan_session
from spatialforge.tsdf_block_cross_view import (
    classify_tsdf_block_voxels_across_observations_from_context,
)
from spatialforge.tsdf_block_plan import plan_tsdf_blocks
from spatialforge.tsdf_block_plan_loader import TsdfBlockPlan
from spatialforge.tsdf_domain_cross_view import (
    MAX_TSDF_COVERAGE_DOMAIN_CROSS_VIEW_OUTCOMES,
    TsdfCoverageDomainCrossViewReceipt,
    sweep_tsdf_coverage_domain_cross_view_from_context,
)
from spatialforge.tsdf_plan_footprint_survey import (
    survey_tsdf_plan_pixel_footprints_from_context,
)
from spatialforge.tsdf_plan_traversal import (
    traverse_tsdf_plan_blocks_from_context,
)
from spatialforge.tsdf_replay_depth_context import TsdfReplayDepthContext
from spatialforge.tsdf_voxel_cross_view import TsdfVoxelCrossViewVerdict


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

_FORBIDDEN_SWEEP_TARGETS = (
    "spatialforge.tsdf_domain_cross_view.replay_session",
    "spatialforge.tsdf_domain_cross_view.build_tsdf_replay_depth_context",
    "spatialforge.tsdf_domain_cross_view.allocate_empty_tsdf_blocks",
    "spatialforge.tsdf_domain_cross_view.locate_tsdf_voxel",
    "spatialforge.tsdf_domain_cross_view."
    "survey_tsdf_plan_pixel_footprints_from_context",
    "spatialforge.tsdf_domain_cross_view."
    "traverse_tsdf_plan_blocks_from_context",
    "spatialforge.replay._file_digest",
    "pathlib.Path.open",
    "PIL.Image.open",
)

_FORBIDDEN_CLI_TARGETS = (
    "spatialforge.cli.allocate_empty_tsdf_blocks",
    "spatialforge.cli.locate_tsdf_voxel",
    "spatialforge.cli.classify_tsdf_voxel_across_observations_from_context",
    "spatialforge.cli.classify_tsdf_voxel_sampling_from_context",
    "spatialforge.cli.survey_tsdf_plan_block_rays_from_context",
    "spatialforge.cli.trace_tsdf_observation_block_rays_from_context",
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


def sweep_case(
    plan: TsdfBlockPlan,
    context: TsdfReplayDepthContext,
) -> TsdfCoverageDomainCrossViewReceipt:
    coverage = survey_tsdf_plan_pixel_footprints_from_context(plan, context)
    return sweep_tsdf_coverage_domain_cross_view_from_context(
        plan,
        context,
        coverage,
    )


@contextmanager
def forbidden_sweep_calls():
    with ExitStack() as stack:
        mocks: list[MagicMock] = []
        for target in _FORBIDDEN_SWEEP_TARGETS:
            mocks.append(stack.enter_context(patch(target, create=True)))
        yield mocks


@contextmanager
def forbidden_cli_calls():
    with ExitStack() as stack:
        mocks: list[MagicMock] = []
        for target in _FORBIDDEN_CLI_TARGETS:
            mocks.append(stack.enter_context(patch(target, create=True)))
        yield mocks


class TsdfDomainCrossViewTests(unittest.TestCase):
    def test_fixture_domain_resolves_every_covered_voxel(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            plan, context = load_case(temporary_root)
            before_tree = tree_snapshot(temporary_root)
            before_plan = plan_snapshot(plan)
            coverage = survey_tsdf_plan_pixel_footprints_from_context(
                plan,
                context,
            )

            with forbidden_sweep_calls() as forbidden:
                receipt = sweep_tsdf_coverage_domain_cross_view_from_context(
                    plan,
                    context,
                    coverage,
                )
            after_tree = tree_snapshot(temporary_root)

        self.assertIsInstance(receipt, TsdfCoverageDomainCrossViewReceipt)
        self.assertEqual(receipt.source_plan_digest_sha256, PLAN_SHA256)
        self.assertEqual(receipt.replay_digest_sha256, REPLAY_SHA256)
        self.assertEqual(
            receipt.domain_block_indices,
            coverage.covered_block_indices,
        )
        self.assertEqual(receipt.block_count, 8)
        self.assertEqual(len(receipt.existing_plan_block_indices), 8)
        self.assertEqual(receipt.unplanned_block_indices, ())
        self.assertEqual(receipt.voxel_count, 4096)
        self.assertEqual(receipt.observation_count, 2)
        self.assertEqual(receipt.retained_outcome_count, 8192)
        self.assertEqual(
            (
                receipt.surface_voxel_count,
                receipt.free_space_voxel_count,
                receipt.occluded_voxel_count,
                receipt.unseen_voxel_count,
            ),
            (560, 24, 792, 2720),
        )
        self.assertEqual(receipt.observed_voxel_count, 584)
        self.assertEqual(receipt.carvable_free_space_voxel_count, 24)
        self.assertEqual(receipt.planned_carvable_voxel_count, 24)
        self.assertEqual(receipt.unplanned_carvable_voxel_count, 0)
        self.assertEqual(len(receipt.carvable_blocks), 4)
        self.assertEqual(receipt.unplanned_carvable_blocks, ())
        self.assertEqual(receipt.reference_weight_total, 1168)
        self.assertAlmostEqual(
            receipt.reference_tsdf_sum_total,
            -304.0,
            places=9,
        )
        self.assertEqual(receipt.maximum_voxel_weight, 2)
        self.assertEqual(
            len(receipt.carvable_free_space_voxel_indices),
            24,
        )
        self.assertEqual(plan_snapshot(plan), before_plan)
        self.assertEqual(after_tree, before_tree)
        for forbidden_call in forbidden:
            forbidden_call.assert_not_called()

    def test_totals_match_the_fusing_plan_traversal(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            plan, context = load_case(temporary_root)
            session = load_scan_session(FIXTURE)
            storage = allocate_empty_tsdf_blocks(plan, session)
            traversal = traverse_tsdf_plan_blocks_from_context(
                storage,
                context,
            )
            receipt = sweep_case(plan, context)

        self.assertEqual(receipt.domain_block_indices, plan.active_blocks)
        self.assertEqual(
            receipt.reference_weight_total,
            traversal.weight_delta,
        )
        self.assertEqual(
            receipt.observed_voxel_count,
            traversal.observed_voxel_count,
        )
        self.assertEqual(
            receipt.planned_reference_weight_total,
            traversal.weight_delta,
        )
        self.assertEqual(
            receipt.maximum_voxel_weight,
            traversal.maximum_weight_after,
        )

    def test_children_equal_independent_per_block_resolutions(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            plan, context = load_case(temporary_root)
            receipt = sweep_case(plan, context)
            expected_first = (
                classify_tsdf_block_voxels_across_observations_from_context(
                    plan,
                    context,
                    receipt.domain_block_indices[0],
                )
            )
            expected_last = (
                classify_tsdf_block_voxels_across_observations_from_context(
                    plan,
                    context,
                    receipt.domain_block_indices[-1],
                )
            )

        self.assertEqual(receipt.block_receipts[0], expected_first)
        self.assertEqual(receipt.block_receipts[-1], expected_last)
        self.assertEqual(
            tuple(
                child.block_index_xyz for child in receipt.block_receipts
            ),
            receipt.domain_block_indices,
        )
        self.assertEqual(
            receipt.carvable_free_space_voxel_indices,
            tuple(
                voxel_index
                for child in receipt.block_receipts
                for voxel_index in child.carvable_free_space_voxel_indices
            ),
        )

    def test_far_domain_finds_carvable_space_outside_the_plan(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            session_path = copy_fixture(temporary_root)
            for filename in ("000000.pgm", "000001.pgm"):
                set_depth_sample(session_path, filename, 3000)
            plan, context = load_case(
                temporary_root,
                session_path=session_path,
                frame_stride=2,
            )
            before_plan = plan_snapshot(plan)
            before_tree = tree_snapshot(temporary_root)

            receipt = sweep_case(plan, context)
            after_tree = tree_snapshot(temporary_root)

        self.assertEqual(len(plan.active_blocks), 32)
        self.assertEqual(receipt.block_count, 52)
        self.assertEqual(len(receipt.existing_plan_block_indices), 32)
        self.assertEqual(len(receipt.unplanned_block_indices), 20)
        self.assertEqual(receipt.voxel_count, 26624)
        self.assertEqual(
            (
                receipt.surface_voxel_count,
                receipt.free_space_voxel_count,
                receipt.occluded_voxel_count,
                receipt.unseen_voxel_count,
            ),
            (4656, 2680, 3608, 15680),
        )
        self.assertEqual(receipt.carvable_free_space_voxel_count, 2680)
        self.assertEqual(receipt.planned_carvable_voxel_count, 1304)
        self.assertEqual(receipt.unplanned_carvable_voxel_count, 1376)
        self.assertGreater(
            receipt.unplanned_carvable_voxel_count,
            receipt.planned_carvable_voxel_count,
        )
        self.assertEqual(len(receipt.carvable_blocks), 24)
        self.assertEqual(len(receipt.unplanned_carvable_blocks), 8)
        self.assertEqual(
            set(receipt.unplanned_carvable_blocks) & set(plan.active_blocks),
            set(),
        )
        self.assertEqual(plan_snapshot(plan), before_plan)
        self.assertEqual(after_tree, before_tree)

    def test_domain_is_bound_to_its_own_coverage_survey(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            plan, context = load_case(temporary_root)
            stride_plan, stride_context = load_case(
                temporary_root,
                frame_stride=2,
                name="stride.sftplan",
            )
            coverage = survey_tsdf_plan_pixel_footprints_from_context(
                plan,
                context,
            )
            foreign_coverage = (
                survey_tsdf_plan_pixel_footprints_from_context(
                    stride_plan,
                    stride_context,
                )
            )
            before_tree = tree_snapshot(temporary_root)

            with self.assertRaises(TsdfError) as foreign:
                sweep_tsdf_coverage_domain_cross_view_from_context(
                    plan,
                    context,
                    foreign_coverage,
                )
            with self.assertRaises(TsdfError) as coverage_type:
                sweep_tsdf_coverage_domain_cross_view_from_context(
                    plan,
                    context,
                    object(),  # type: ignore[arg-type]
                )
            receipt = sweep_tsdf_coverage_domain_cross_view_from_context(
                plan,
                context,
                coverage,
            )
            after_tree = tree_snapshot(temporary_root)

        self.assertIn("provenance does not match", str(foreign.exception))
        self.assertIn(
            "TsdfPlanFootprintSurveyReceipt",
            str(coverage_type.exception),
        )
        self.assertEqual(
            receipt.domain_block_indices,
            coverage.covered_block_indices,
        )
        self.assertEqual(after_tree, before_tree)

    def test_types_and_provenance_are_preflighted(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            plan, context = load_case(temporary_root)
            coverage = survey_tsdf_plan_pixel_footprints_from_context(
                plan,
                context,
            )
            before_tree = tree_snapshot(temporary_root)
            cases = (
                ("plan-type", object(), context, "TsdfBlockPlan"),
                ("context-type", plan, object(), "TsdfReplayDepthContext"),
                (
                    "digest",
                    plan,
                    replace(context, source_plan_digest_sha256="0" * 64),
                    "source plan digest",
                ),
                (
                    "extent",
                    replace(plan, block_extent_m=2.0),
                    context,
                    "block_extent_m",
                ),
            )
            for name, case_plan, case_context, message in cases:
                with self.subTest(name=name):
                    with self.assertRaises(TsdfError) as raised:
                        sweep_tsdf_coverage_domain_cross_view_from_context(
                            case_plan,  # type: ignore[arg-type]
                            case_context,  # type: ignore[arg-type]
                            coverage,
                        )
                    self.assertIn(message, str(raised.exception))
            after_tree = tree_snapshot(temporary_root)

        self.assertEqual(after_tree, before_tree)

    def test_outcome_cap_is_preflighted_and_atomic(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            plan, context = load_case(temporary_root)
            coverage = survey_tsdf_plan_pixel_footprints_from_context(
                plan,
                context,
            )
            before_tree = tree_snapshot(temporary_root)

            self.assertEqual(
                MAX_TSDF_COVERAGE_DOMAIN_CROSS_VIEW_OUTCOMES,
                262_144,
            )
            with patch(
                "spatialforge.tsdf_domain_cross_view."
                "MAX_TSDF_COVERAGE_DOMAIN_CROSS_VIEW_OUTCOMES",
                8191,
            ):
                with self.assertRaises(TsdfError) as raised:
                    sweep_tsdf_coverage_domain_cross_view_from_context(
                        plan,
                        context,
                        coverage,
                    )
            after_tree = tree_snapshot(temporary_root)

        self.assertIn("requires 8192 retained outcomes", str(raised.exception))
        self.assertIn("maximum is 8191", str(raised.exception))
        self.assertEqual(after_tree, before_tree)

    def test_receipts_are_frozen_slotted_and_strict(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            plan, context = load_case(Path(temporary_directory))
            receipt = sweep_case(plan, context)

        self.assertFalse(hasattr(receipt, "__dict__"))
        with self.assertRaises(FrozenInstanceError):
            receipt.frame_stride = 2  # type: ignore[misc]

        for arguments in (
            {"block_receipts": tuple(reversed(receipt.block_receipts))},
            {"block_receipts": receipt.block_receipts[:4]},
            {
                "domain_block_indices": receipt.domain_block_indices[:-1],
            },
            {
                "domain_block_indices": tuple(
                    reversed(receipt.domain_block_indices)
                )
            },
            {
                "existing_plan_block_indices": (),
                "unplanned_block_indices": (),
            },
            {
                "existing_plan_block_indices": (),
                "unplanned_block_indices": receipt.domain_block_indices,
            },
            {"selected_observation_sequences": (0,)},
            {"frame_stride": 2},
            {"total_observations": 3},
            {"block_resolution": 7},
            {"source_plan_digest_sha256": "0"},
            {"replay_digest_sha256": "0"},
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
            coverage = survey_tsdf_plan_pixel_footprints_from_context(
                plan,
                context,
            )
            before_tree = tree_snapshot(temporary_root)
            before_plan = plan_snapshot(plan)
            injected = RuntimeError("injected final domain sweep failure")

            with patch(
                "spatialforge.tsdf_domain_cross_view."
                "TsdfCoverageDomainCrossViewReceipt",
                side_effect=injected,
            ):
                with self.assertRaises(TsdfError) as raised:
                    sweep_tsdf_coverage_domain_cross_view_from_context(
                        plan,
                        context,
                        coverage,
                    )
            after_tree = tree_snapshot(temporary_root)

        self.assertIs(raised.exception.__cause__, injected)
        self.assertIn(
            "injected final domain sweep failure",
            str(raised.exception),
        )
        self.assertEqual(plan_snapshot(plan), before_plan)
        self.assertEqual(after_tree, before_tree)


class TsdfDomainCrossViewCliTests(unittest.TestCase):
    def test_cli_reports_exact_read_only_domain_sweep(self) -> None:
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
            actual = sweep_tsdf_coverage_domain_cross_view_from_context

            with (
                forbidden_cli_calls() as forbidden,
                patch(
                    "spatialforge.cli."
                    "sweep_tsdf_coverage_domain_cross_view_from_context",
                    wraps=actual,
                ) as sweeper,
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                exit_code = main(
                    [
                        "reconstruct",
                        "tsdf-block-context-domain-cross-view",
                        str(plan_path),
                        str(session_path),
                    ]
                )
            after_tree = tree_snapshot(temporary_root)

        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(after_tree, before_tree)
        sweeper.assert_called_once()
        for forbidden_call in forbidden:
            forbidden_call.assert_not_called()

        output = stdout.getvalue()
        for expected in (
            "TSDF BLOCK CONTEXT DOMAIN CROSS-VIEW CHECK scan-synthetic-0001\n",
            "coverage_source: conservative-pixel-footprint-survey\n",
            "coverage_domain: blocks=8 existing_plan=8 unplanned=0\n",
            "domain_voxels: total=4096 observations=2\n",
            "voxel_verdicts: surface=560 free_space=24 occluded=792 "
            "unseen=2720\n",
            "observed_voxels: 584\n",
            "carvable_free_space_voxels: total=24 in_plan=24 unplanned=0\n",
            "carvable_blocks: total=4 unplanned=0\n",
            "reference_weight_total: 1168\n",
            "planned_observed_voxels: 584\n",
            "maximum_voxel_weight: 2\n",
            "cross_view_workload: retained_outcomes=8192 maximum=262144\n",
            "cross_view_scope: surveyed-coverage-domain-all-voxels\n",
            "whole_scan_carvable_set_computed: yes\n",
            "coverage_approved_for_expansion: no\n",
            "free_space_carving_applied: no\n",
            "plan_expanded: no\n",
            "missing_blocks_created: no\n",
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
                    "sweep_tsdf_coverage_domain_cross_view_from_context",
                    side_effect=TsdfError("injected domain sweep failure"),
                ) as sweeper,
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                exit_code = main(
                    [
                        "reconstruct",
                        "tsdf-block-context-domain-cross-view",
                        str(plan_path),
                        str(session_path),
                    ]
                )
            after_tree = tree_snapshot(temporary_root)

        self.assertEqual(exit_code, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(after_tree, before_tree)
        sweeper.assert_called_once()
        error = stderr.getvalue()
        self.assertIn("TSDF BLOCK CONTEXT DOMAIN CROSS-VIEW FAILED", error)
        self.assertIn("injected domain sweep failure", error)
        self.assertNotIn("Traceback", error)


if __name__ == "__main__":
    unittest.main()
