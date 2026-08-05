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
    MAX_TSDF_BLOCK_CROSS_VIEW_OUTCOMES,
    TsdfBlockCrossViewReceipt,
    _local_index_for_flat,
    classify_tsdf_block_voxels_across_observations_from_context,
)
from spatialforge.tsdf_block_plan import plan_tsdf_blocks
from spatialforge.tsdf_block_plan_loader import TsdfBlockPlan
from spatialforge.tsdf_block_traversal import (
    traverse_tsdf_block_voxels_from_context,
)
from spatialforge.tsdf_replay_depth_context import TsdfReplayDepthContext
from spatialforge.tsdf_voxel_address import compose_tsdf_global_voxel_index
from spatialforge.tsdf_voxel_cross_view import (
    TsdfVoxelCrossViewVerdict,
    classify_tsdf_voxel_across_observations_from_context,
)
from spatialforge.tsdf_voxel_sampling import TsdfVoxelSamplingStatus


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

_FORBIDDEN_CROSS_VIEW_TARGETS = (
    "spatialforge.tsdf_block_cross_view.replay_session",
    "spatialforge.tsdf_block_cross_view.build_tsdf_replay_depth_context",
    "spatialforge.tsdf_block_cross_view.allocate_empty_tsdf_blocks",
    "spatialforge.tsdf_block_cross_view.locate_tsdf_voxel",
    "spatialforge.tsdf_block_cross_view."
    "evaluate_tsdf_voxel_contribution_from_context",
    "spatialforge.tsdf_block_cross_view."
    "apply_tsdf_voxel_contribution_from_context",
    "spatialforge.tsdf_block_cross_view."
    "traverse_tsdf_block_voxels_from_context",
    "spatialforge.replay._file_digest",
    "pathlib.Path.open",
    "PIL.Image.open",
)

_FORBIDDEN_CLI_TARGETS = (
    "spatialforge.cli.allocate_empty_tsdf_blocks",
    "spatialforge.cli.locate_tsdf_voxel",
    "spatialforge.cli.classify_tsdf_voxel_across_observations_from_context",
    "spatialforge.cli.classify_tsdf_voxel_sampling_from_context",
    "spatialforge.cli.survey_tsdf_plan_pixel_footprints_from_context",
    "spatialforge.cli.survey_tsdf_observation_pixel_footprints_from_context",
    "spatialforge.cli.survey_tsdf_plan_block_rays_from_context",
    "spatialforge.cli.trace_tsdf_observation_block_rays_from_context",
    "spatialforge.cli.evaluate_tsdf_pixel_footprint_coverage_from_context",
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


def remove_first_record(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text(lines[1] + "\n", encoding="utf-8")


@contextmanager
def forbidden_cross_view_calls():
    with ExitStack() as stack:
        mocks: list[MagicMock] = []
        for target in _FORBIDDEN_CROSS_VIEW_TARGETS:
            mocks.append(stack.enter_context(patch(target, create=True)))
        yield mocks


@contextmanager
def forbidden_cli_calls():
    with ExitStack() as stack:
        mocks: list[MagicMock] = []
        for target in _FORBIDDEN_CLI_TARGETS:
            mocks.append(stack.enter_context(patch(target, create=True)))
        yield mocks


class TsdfBlockCrossViewTests(unittest.TestCase):
    def test_fixture_blocks_separate_surface_free_space_and_occlusion(
        self,
    ) -> None:
        cases = (
            ((1, -1, -1), True, (102, 0, 198, 212), 0, 204, -117.0),
            ((0, 0, 0), True, (38, 6, 0, 468), 6, 88, 41.0),
            ((7, 0, 0), False, (0, 0, 512, 0), 0, 0, 0.0),
        )
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            plan, context = load_case(temporary_root)
            before_tree = tree_snapshot(temporary_root)
            before_plan = plan_snapshot(plan)

            for (
                block_index,
                planned,
                counts,
                carvable,
                weight_total,
                sum_total,
            ) in cases:
                with self.subTest(block=block_index):
                    with forbidden_cross_view_calls() as forbidden:
                        receipt = (
                            classify_tsdf_block_voxels_across_observations_from_context(
                                plan,
                                context,
                                block_index,
                            )
                        )
                    repeated = (
                        classify_tsdf_block_voxels_across_observations_from_context(
                            plan,
                            context,
                            block_index,
                        )
                    )

                    self.assertIsInstance(receipt, TsdfBlockCrossViewReceipt)
                    self.assertEqual(receipt, repeated)
                    self.assertEqual(receipt.block_index_xyz, block_index)
                    self.assertEqual(receipt.planned_block, planned)
                    self.assertEqual(receipt.voxel_count, 512)
                    self.assertEqual(receipt.observation_count, 2)
                    self.assertEqual(receipt.retained_outcome_count, 1024)
                    self.assertEqual(
                        (
                            receipt.surface_voxel_count,
                            receipt.free_space_voxel_count,
                            receipt.occluded_voxel_count,
                            receipt.unseen_voxel_count,
                        ),
                        counts,
                    )
                    self.assertEqual(
                        receipt.observed_voxel_count,
                        counts[0] + counts[1],
                    )
                    self.assertEqual(
                        receipt.carvable_free_space_voxel_count,
                        carvable,
                    )
                    self.assertEqual(receipt.carves_free_space, carvable > 0)
                    self.assertEqual(
                        receipt.reference_weight_total,
                        weight_total,
                    )
                    self.assertAlmostEqual(
                        receipt.reference_tsdf_sum_total,
                        sum_total,
                        places=9,
                    )
                    self.assertEqual(
                        receipt.source_plan_digest_sha256,
                        PLAN_SHA256,
                    )
                    self.assertEqual(
                        receipt.replay_digest_sha256,
                        REPLAY_SHA256,
                    )
                    for forbidden_call in forbidden:
                        forbidden_call.assert_not_called()

            after_tree = tree_snapshot(temporary_root)

        self.assertEqual(plan_snapshot(plan), before_plan)
        self.assertEqual(after_tree, before_tree)

    def test_totals_match_the_fusing_block_traversal(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            plan, context = load_case(temporary_root)
            session = load_scan_session(FIXTURE)
            verdicts: set[TsdfVoxelCrossViewVerdict] = set()
            compared_blocks = 0

            for block_index in plan.active_blocks:
                storage = allocate_empty_tsdf_blocks(plan, session)
                traversal = traverse_tsdf_block_voxels_from_context(
                    storage,
                    block_index,
                    context,
                )
                receipt = (
                    classify_tsdf_block_voxels_across_observations_from_context(
                        plan,
                        context,
                        block_index,
                    )
                )
                compared_blocks += 1
                self.assertEqual(
                    len(traversal.voxel_receipts),
                    len(receipt.voxel_receipts),
                )
                for position, voxel in enumerate(traversal.voxel_receipts):
                    mine = receipt.voxel_receipts[position]
                    verdicts.add(mine.verdict)
                    self.assertEqual(
                        mine.local_index_xyz,
                        voxel.address.local_index_xyz,
                        f"{block_index} position {position} order",
                    )
                    self.assertEqual(
                        mine.reference_weight,
                        voxel.weight_after,
                        f"{block_index} position {position} weight",
                    )
                    self.assertEqual(
                        mine.reference_tsdf_sum,
                        voxel.tsdf_sum_after,
                        f"{block_index} position {position} sum",
                    )
                self.assertEqual(
                    receipt.reference_weight_total,
                    sum(
                        voxel.weight_after
                        for voxel in traversal.voxel_receipts
                    ),
                )

        self.assertEqual(compared_blocks, 8)
        self.assertIn(TsdfVoxelCrossViewVerdict.SURFACE, verdicts)
        self.assertIn(TsdfVoxelCrossViewVerdict.FREE_SPACE, verdicts)
        self.assertIn(TsdfVoxelCrossViewVerdict.OCCLUDED, verdicts)
        self.assertIn(TsdfVoxelCrossViewVerdict.UNSEEN, verdicts)

    def test_carvable_voxels_are_free_space_and_never_banded(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            plan, context = load_case(temporary_root)
            receipt = (
                classify_tsdf_block_voxels_across_observations_from_context(
                    plan,
                    context,
                    (0, 0, 0),
                )
            )

        carvable = set(receipt.carvable_free_space_voxel_indices)
        self.assertEqual(len(carvable), 6)
        for voxel in receipt.voxel_receipts:
            if voxel.global_index_xyz in carvable:
                self.assertIs(
                    voxel.verdict,
                    TsdfVoxelCrossViewVerdict.FREE_SPACE,
                )
                self.assertEqual(voxel.surface_band_count, 0)
                self.assertGreater(voxel.free_space_count, 0)
                self.assertTrue(voxel.carvable_free_space)
            else:
                self.assertFalse(voxel.carvable_free_space)
        self.assertEqual(
            receipt.carvable_free_space_voxel_indices,
            tuple(
                voxel.global_index_xyz
                for voxel in receipt.voxel_receipts
                if voxel.carvable_free_space
            ),
        )

    def test_children_equal_independent_per_voxel_cross_views(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            plan, context = load_case(temporary_root)
            receipt = (
                classify_tsdf_block_voxels_across_observations_from_context(
                    plan,
                    context,
                    (0, 0, 0),
                )
            )
            for local_flat_index in (0, 1, 8, 64, 511):
                expected = classify_tsdf_voxel_across_observations_from_context(
                    plan,
                    context,
                    compose_tsdf_global_voxel_index(
                        (0, 0, 0),
                        _local_index_for_flat(local_flat_index),
                    ),
                )
                self.assertEqual(
                    receipt.voxel_receipts[local_flat_index],
                    expected,
                )

    def test_voxel_order_is_canonical_x_fastest_local_flat(self) -> None:
        self.assertEqual(_local_index_for_flat(0), (0, 0, 0))
        self.assertEqual(_local_index_for_flat(1), (1, 0, 0))
        self.assertEqual(_local_index_for_flat(8), (0, 1, 0))
        self.assertEqual(_local_index_for_flat(64), (0, 0, 1))
        self.assertEqual(_local_index_for_flat(511), (7, 7, 7))

        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            plan, context = load_case(Path(temporary_directory))
            receipt = (
                classify_tsdf_block_voxels_across_observations_from_context(
                    plan,
                    context,
                    (0, -1, 0),
                )
            )

        self.assertEqual(
            tuple(
                voxel.local_index_xyz for voxel in receipt.voxel_receipts
            ),
            tuple(_local_index_for_flat(index) for index in range(512)),
        )
        self.assertEqual(
            receipt.voxel_receipts[0].global_index_xyz,
            (0, -8, 0),
        )
        self.assertEqual(
            receipt.voxel_receipts[511].global_index_xyz,
            (7, -1, 7),
        )

    def test_unplanned_blocks_are_resolved_without_touching_the_plan(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            plan, context = load_case(temporary_root)
            before_plan = plan_snapshot(plan)
            before_tree = tree_snapshot(temporary_root)

            receipt = (
                classify_tsdf_block_voxels_across_observations_from_context(
                    plan,
                    context,
                    (-5, -5, -5),
                )
            )
            after_tree = tree_snapshot(temporary_root)

        self.assertFalse(receipt.planned_block)
        self.assertNotIn((-5, -5, -5), plan.active_blocks)
        self.assertEqual(receipt.voxel_count, 512)
        self.assertEqual(receipt.unseen_voxel_count, 512)
        self.assertEqual(receipt.carvable_free_space_voxel_count, 0)
        self.assertEqual(receipt.reference_weight_total, 0)
        self.assertEqual(plan_snapshot(plan), before_plan)
        self.assertEqual(after_tree, before_tree)

    def test_missing_pose_reduces_evidence_without_inverting_verdicts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            session_path = copy_fixture(temporary_root)
            remove_first_record(session_path / "streams" / "poses.jsonl")
            plan, context = load_case(
                temporary_root,
                session_path=session_path,
            )
            before_tree = tree_snapshot(temporary_root)

            receipt = (
                classify_tsdf_block_voxels_across_observations_from_context(
                    plan,
                    context,
                    (1, -1, -1),
                )
            )
            after_tree = tree_snapshot(temporary_root)

        for voxel in receipt.voxel_receipts:
            self.assertIs(
                voxel.observation_receipts[0].status,
                TsdfVoxelSamplingStatus.MISSING_POSE,
            )
        self.assertEqual(receipt.maximum_voxel_weight, 1)
        self.assertGreater(receipt.surface_voxel_count, 0)
        self.assertEqual(
            receipt.reference_weight_total,
            receipt.observed_voxel_count,
        )
        self.assertEqual(after_tree, before_tree)

    def test_block_provenance_and_selection_are_preflighted(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            plan, context = load_case(temporary_root)
            stride_plan, _ = load_case(
                temporary_root,
                frame_stride=2,
                name="stride.sftplan",
            )
            before_tree = tree_snapshot(temporary_root)
            cases = (
                ("plan-type", object(), context, (0, 0, 0), "TsdfBlockPlan"),
                (
                    "context-type",
                    plan,
                    object(),
                    (0, 0, 0),
                    "TsdfReplayDepthContext",
                ),
                (
                    "foreign-context",
                    stride_plan,
                    context,
                    (0, 0, 0),
                    "does not match",
                ),
                ("block-length", plan, context, (0, 0), "block index"),
                ("block-bool", plan, context, (True, 0, 0), "block index"),
                (
                    "digest",
                    plan,
                    replace(context, source_plan_digest_sha256="0" * 64),
                    (0, 0, 0),
                    "source plan digest",
                ),
                (
                    "extent",
                    replace(plan, block_extent_m=2.0),
                    context,
                    (0, 0, 0),
                    "block_extent_m",
                ),
            )
            for name, case_plan, case_context, block, message in cases:
                with self.subTest(name=name):
                    with self.assertRaises(TsdfError) as raised:
                        classify_tsdf_block_voxels_across_observations_from_context(
                            case_plan,  # type: ignore[arg-type]
                            case_context,  # type: ignore[arg-type]
                            block,  # type: ignore[arg-type]
                        )
                    self.assertIn(message, str(raised.exception))
            after_tree = tree_snapshot(temporary_root)

        self.assertEqual(after_tree, before_tree)

    def test_outcome_cap_is_preflighted_and_atomic(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            plan, context = load_case(temporary_root)
            before_tree = tree_snapshot(temporary_root)

            self.assertEqual(MAX_TSDF_BLOCK_CROSS_VIEW_OUTCOMES, 262_144)
            with patch(
                "spatialforge.tsdf_block_cross_view."
                "MAX_TSDF_BLOCK_CROSS_VIEW_OUTCOMES",
                1023,
            ):
                with self.assertRaises(TsdfError) as raised:
                    classify_tsdf_block_voxels_across_observations_from_context(
                        plan,
                        context,
                        (0, 0, 0),
                    )
            with patch(
                "spatialforge.tsdf_block_cross_view."
                "MAX_TSDF_BLOCK_CROSS_VIEW_OUTCOMES",
                1024,
            ):
                receipt = (
                    classify_tsdf_block_voxels_across_observations_from_context(
                        plan,
                        context,
                        (0, 0, 0),
                    )
                )
            after_tree = tree_snapshot(temporary_root)

        self.assertIn("requires 1024 retained outcomes", str(raised.exception))
        self.assertIn("maximum is 1023", str(raised.exception))
        self.assertEqual(receipt.retained_outcome_count, 1024)
        self.assertEqual(after_tree, before_tree)

    def test_receipts_are_frozen_slotted_and_strict(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            plan, context = load_case(Path(temporary_directory))
            receipt = (
                classify_tsdf_block_voxels_across_observations_from_context(
                    plan,
                    context,
                    (0, 0, 0),
                )
            )
            other = (
                classify_tsdf_block_voxels_across_observations_from_context(
                    plan,
                    context,
                    (1, 0, 0),
                )
            )

        self.assertFalse(hasattr(receipt, "__dict__"))
        with self.assertRaises(FrozenInstanceError):
            receipt.planned_block = False  # type: ignore[misc]

        for arguments in (
            {"voxel_receipts": tuple(reversed(receipt.voxel_receipts))},
            {"voxel_receipts": receipt.voxel_receipts[:511]},
            {"voxel_receipts": other.voxel_receipts},
            {"block_index_xyz": (1, 0, 0)},
            {"selected_observation_sequences": (0,)},
            {"frame_stride": 2},
            {"total_observations": 3},
            {"planned_block": False},
            {"block_resolution": 7},
            {"voxel_size_m": 0.25},
            {"truncation_m": 0.0},
            {"image_size": (2, 3)},
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
            before_tree = tree_snapshot(temporary_root)
            before_plan = plan_snapshot(plan)
            injected = RuntimeError("injected final block cross-view failure")

            with patch(
                "spatialforge.tsdf_block_cross_view."
                "TsdfBlockCrossViewReceipt",
                side_effect=injected,
            ):
                with self.assertRaises(TsdfError) as raised:
                    classify_tsdf_block_voxels_across_observations_from_context(
                        plan,
                        context,
                        (0, 0, 0),
                    )
            after_tree = tree_snapshot(temporary_root)

        self.assertIs(raised.exception.__cause__, injected)
        self.assertIn(
            "injected final block cross-view failure",
            str(raised.exception),
        )
        self.assertEqual(plan_snapshot(plan), before_plan)
        self.assertEqual(after_tree, before_tree)


class TsdfBlockCrossViewCliTests(unittest.TestCase):
    def test_cli_reports_exact_read_only_block_cross_view(self) -> None:
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
            actual = (
                classify_tsdf_block_voxels_across_observations_from_context
            )

            with (
                forbidden_cli_calls() as forbidden,
                patch(
                    "spatialforge.cli."
                    "classify_tsdf_block_voxels_across_observations_from_context",
                    wraps=actual,
                ) as classifier,
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                exit_code = main(
                    [
                        "reconstruct",
                        "tsdf-block-context-block-cross-view",
                        str(plan_path),
                        str(session_path),
                        "--block",
                        "0",
                        "0",
                        "0",
                    ]
                )
            after_tree = tree_snapshot(temporary_root)

        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(after_tree, before_tree)
        classifier.assert_called_once()
        for forbidden_call in forbidden:
            forbidden_call.assert_not_called()

        output = stdout.getvalue()
        for expected in (
            "TSDF BLOCK CONTEXT BLOCK CROSS-VIEW CHECK scan-synthetic-0001\n",
            "block: index=(0, 0, 0) resolution=8 voxels=512\n",
            "block_planned: yes\n",
            "voxel_order: canonical-local-flat-x-fastest local_flat=0..511\n",
            "voxel_verdicts: surface=38 free_space=6 occluded=0 unseen=468\n",
            "observed_voxels: 44\n",
            "carvable_free_space_voxels: 6\n",
            "reference_weight_total: 88\n",
            "maximum_voxel_weight: 2\n",
            "verdict_precedence: surface-then-free-space-then-occluded-then-"
            "unseen\n",
            "occlusion_rule: carries-no-evidence-never-becomes-free-space\n",
            "cross_view_workload: retained_outcomes=1024 maximum=262144\n",
            "cross_view_scope: one-block-all-voxels-all-observations\n",
            "unplanned_blocks_accepted: yes\n",
            "carvable_free_space_set_computed: yes\n",
            "multi_block_cross_view_computed: no\n",
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
                    "classify_tsdf_block_voxels_across_observations_from_context",
                    side_effect=TsdfError("injected block cross-view failure"),
                ) as classifier,
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                exit_code = main(
                    [
                        "reconstruct",
                        "tsdf-block-context-block-cross-view",
                        str(plan_path),
                        str(session_path),
                        "--block",
                        "0",
                        "0",
                        "0",
                    ]
                )
            after_tree = tree_snapshot(temporary_root)

        self.assertEqual(exit_code, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(after_tree, before_tree)
        classifier.assert_called_once()
        error = stderr.getvalue()
        self.assertIn("TSDF BLOCK CONTEXT BLOCK CROSS-VIEW FAILED", error)
        self.assertIn("injected block cross-view failure", error)
        self.assertNotIn("Traceback", error)


if __name__ == "__main__":
    unittest.main()
