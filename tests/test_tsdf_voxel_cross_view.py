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
from spatialforge.tsdf_block_plan import plan_tsdf_blocks
from spatialforge.tsdf_block_plan_loader import TsdfBlockPlan
from spatialforge.tsdf_replay_depth_context import TsdfReplayDepthContext
from spatialforge.tsdf_voxel_address import (
    compose_tsdf_global_voxel_index,
    locate_tsdf_voxel,
)
from spatialforge.tsdf_voxel_cross_view import (
    MAX_TSDF_VOXEL_CROSS_VIEW_OBSERVATIONS,
    TsdfVoxelCrossViewReceipt,
    TsdfVoxelCrossViewVerdict,
    _combine_sampling_statuses,
    classify_tsdf_voxel_across_observations_from_context,
)
from spatialforge.tsdf_voxel_sampling import (
    TsdfVoxelSamplingStatus,
    classify_tsdf_voxel_sampling_from_context,
)
from spatialforge.tsdf_voxel_traversal import (
    traverse_tsdf_voxel_observations_from_context,
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

_FORBIDDEN_CROSS_VIEW_TARGETS = (
    "spatialforge.tsdf_voxel_cross_view.replay_session",
    "spatialforge.tsdf_voxel_cross_view.build_tsdf_replay_depth_context",
    "spatialforge.tsdf_voxel_cross_view.allocate_empty_tsdf_blocks",
    "spatialforge.tsdf_voxel_cross_view.locate_tsdf_voxel",
    "spatialforge.tsdf_voxel_cross_view.evaluate_tsdf_voxel_contribution",
    "spatialforge.tsdf_voxel_cross_view."
    "evaluate_tsdf_voxel_contribution_from_context",
    "spatialforge.tsdf_voxel_cross_view."
    "apply_tsdf_voxel_contribution_from_context",
    "spatialforge.tsdf_voxel_cross_view."
    "traverse_tsdf_voxel_observations_from_context",
    "spatialforge.replay._file_digest",
    "pathlib.Path.open",
    "PIL.Image.open",
)

_FORBIDDEN_CLI_TARGETS = (
    "spatialforge.cli.allocate_empty_tsdf_blocks",
    "spatialforge.cli.locate_tsdf_voxel",
    "spatialforge.cli.classify_tsdf_voxel_sampling_from_context",
    "spatialforge.cli.evaluate_tsdf_voxel_contribution",
    "spatialforge.cli.evaluate_tsdf_voxel_contribution_from_context",
    "spatialforge.cli.evaluate_tsdf_pixel_footprint_coverage_from_context",
    "spatialforge.cli.trace_tsdf_observation_block_rays_from_context",
    "spatialforge.cli.survey_tsdf_plan_block_rays_from_context",
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


def planned_global_indices(plan: TsdfBlockPlan, stride: int = 1):
    for block_index in plan.active_blocks:
        for local_z in range(0, 8, stride):
            for local_y in range(0, 8, stride):
                for local_x in range(0, 8, stride):
                    yield compose_tsdf_global_voxel_index(
                        block_index,
                        (local_x, local_y, local_z),
                    )


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


class TsdfVoxelCrossViewTests(unittest.TestCase):
    def test_fixture_voxels_resolve_surface_free_space_and_occluded(
        self,
    ) -> None:
        cases = (
            (
                (8, -1, -1),
                TsdfVoxelCrossViewVerdict.SURFACE,
                (2, 0, 0, 0),
                2,
                -0.25,
                -0.125,
                (0, 1),
                (),
                False,
            ),
            (
                (2, 0, 0),
                TsdfVoxelCrossViewVerdict.FREE_SPACE,
                (0, 2, 0, 0),
                2,
                2.0,
                1.0,
                (0, 1),
                (0, 1),
                True,
            ),
            (
                (60, 0, 0),
                TsdfVoxelCrossViewVerdict.OCCLUDED,
                (0, 0, 2, 0),
                0,
                0.0,
                None,
                (),
                (),
                False,
            ),
        )
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            plan, context = load_case(temporary_root)
            before_tree = tree_snapshot(temporary_root)
            before_plan = plan_snapshot(plan)

            for (
                global_index,
                verdict,
                counts,
                weight,
                total,
                value,
                contributing,
                wedge,
                carvable,
            ) in cases:
                with self.subTest(voxel=global_index):
                    with forbidden_cross_view_calls() as forbidden:
                        receipt = (
                            classify_tsdf_voxel_across_observations_from_context(
                                plan,
                                context,
                                global_index,
                            )
                        )
                    repeated = (
                        classify_tsdf_voxel_across_observations_from_context(
                            plan,
                            context,
                            global_index,
                        )
                    )

                    self.assertIsInstance(receipt, TsdfVoxelCrossViewReceipt)
                    self.assertEqual(receipt, repeated)
                    self.assertIs(receipt.verdict, verdict)
                    self.assertEqual(
                        (
                            receipt.surface_band_count,
                            receipt.free_space_count,
                            receipt.occluded_count,
                            receipt.unseen_count,
                        ),
                        counts,
                    )
                    self.assertEqual(receipt.observation_count, 2)
                    self.assertEqual(
                        receipt.selected_observation_sequences,
                        (0, 1),
                    )
                    self.assertEqual(receipt.reference_weight, weight)
                    # The fixture's two poses differ by 0.05 m, so the two
                    # per-observation values are not identical and their
                    # float64 sum is not exactly the rounded figure the CLI
                    # prints. Bit-exact agreement is proved separately
                    # against the fusing traversal.
                    self.assertAlmostEqual(
                        receipt.reference_tsdf_sum,
                        total,
                        places=12,
                    )
                    if value is None:
                        self.assertIsNone(receipt.reference_tsdf_value)
                    else:
                        self.assertAlmostEqual(
                            receipt.reference_tsdf_value,
                            value,
                            places=12,
                        )
                    self.assertEqual(
                        receipt.contributing_observation_sequences,
                        contributing,
                    )
                    self.assertEqual(
                        receipt.wedge_observation_sequences,
                        wedge,
                    )
                    self.assertEqual(receipt.carvable_free_space, carvable)
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

    def test_reference_totals_match_the_fusing_voxel_traversal(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            plan, context = load_case(temporary_root)
            session = load_scan_session(FIXTURE)
            compared = 0
            fused = 0
            verdicts: set[TsdfVoxelCrossViewVerdict] = set()

            for global_index in planned_global_indices(plan, stride=2):
                storage = allocate_empty_tsdf_blocks(plan, session)
                address = locate_tsdf_voxel(storage, global_index)
                self.assertIsNotNone(address)
                traversal = traverse_tsdf_voxel_observations_from_context(
                    storage,
                    address,
                    context,
                )
                receipt = (
                    classify_tsdf_voxel_across_observations_from_context(
                        plan,
                        context,
                        global_index,
                    )
                )
                compared += 1
                verdicts.add(receipt.verdict)

                self.assertEqual(
                    receipt.reference_weight,
                    traversal.weight_after,
                    f"{global_index} weight",
                )
                self.assertEqual(
                    receipt.reference_tsdf_sum,
                    traversal.tsdf_sum_after,
                    f"{global_index} sum",
                )
                self.assertEqual(
                    receipt.contributing_observation_sequences,
                    tuple(
                        update.contribution.observation_sequence
                        for update in traversal.update_receipts
                    ),
                )
                self.assertEqual(
                    receipt.observed,
                    traversal.weight_after > 0,
                )
                if traversal.weight_after:
                    fused += 1

        self.assertGreater(compared, 300)
        self.assertGreater(fused, 0)
        self.assertIn(TsdfVoxelCrossViewVerdict.SURFACE, verdicts)
        self.assertIn(TsdfVoxelCrossViewVerdict.FREE_SPACE, verdicts)
        self.assertIn(TsdfVoxelCrossViewVerdict.OCCLUDED, verdicts)

    def test_children_equal_independent_per_observation_classifications(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            plan, context = load_case(temporary_root)
            receipt = classify_tsdf_voxel_across_observations_from_context(
                plan,
                context,
                (8, -1, -1),
            )
            expected = tuple(
                classify_tsdf_voxel_sampling_from_context(
                    plan,
                    context,
                    observation_sequence,
                    (8, -1, -1),
                )
                for observation_sequence in (0, 1)
            )

        self.assertEqual(receipt.observation_receipts, expected)
        self.assertEqual(
            tuple(
                child.observation_sequence
                for child in receipt.observation_receipts
            ),
            (0, 1),
        )

    def test_verdict_precedence_never_turns_absence_into_free_space(
        self,
    ) -> None:
        band = TsdfVoxelSamplingStatus.OBSERVED_SURFACE_BAND
        free = TsdfVoxelSamplingStatus.OBSERVED_FREE_SPACE
        occluded = TsdfVoxelSamplingStatus.UNOBSERVED_OCCLUDED
        behind = TsdfVoxelSamplingStatus.UNOBSERVED_BEHIND_CAMERA
        missing = TsdfVoxelSamplingStatus.MISSING_POSE
        invalid = TsdfVoxelSamplingStatus.UNOBSERVED_DEPTH_INVALID
        cases = (
            ((band, free), TsdfVoxelCrossViewVerdict.SURFACE),
            ((free, band), TsdfVoxelCrossViewVerdict.SURFACE),
            ((band, occluded), TsdfVoxelCrossViewVerdict.SURFACE),
            ((occluded, band), TsdfVoxelCrossViewVerdict.SURFACE),
            ((free, occluded), TsdfVoxelCrossViewVerdict.FREE_SPACE),
            ((occluded, free), TsdfVoxelCrossViewVerdict.FREE_SPACE),
            ((free, missing), TsdfVoxelCrossViewVerdict.FREE_SPACE),
            ((occluded, missing), TsdfVoxelCrossViewVerdict.OCCLUDED),
            ((occluded, behind), TsdfVoxelCrossViewVerdict.OCCLUDED),
            ((behind, missing), TsdfVoxelCrossViewVerdict.UNSEEN),
            ((invalid, missing), TsdfVoxelCrossViewVerdict.UNSEEN),
            ((), TsdfVoxelCrossViewVerdict.UNSEEN),
        )
        for statuses, expected in cases:
            with self.subTest(statuses=statuses):
                self.assertIs(
                    _combine_sampling_statuses(statuses),
                    expected,
                )

    def test_missing_and_unplanned_inputs_never_become_free_space(
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
            before_plan = plan_snapshot(plan)
            before_tree = tree_snapshot(temporary_root)

            surface = classify_tsdf_voxel_across_observations_from_context(
                plan,
                context,
                (8, -1, -1),
            )
            occluded = classify_tsdf_voxel_across_observations_from_context(
                plan,
                context,
                (60, 0, 0),
            )
            behind = classify_tsdf_voxel_across_observations_from_context(
                plan,
                context,
                (-40, -40, -40),
            )
            after_tree = tree_snapshot(temporary_root)

        self.assertIs(
            surface.observation_receipts[0].status,
            TsdfVoxelSamplingStatus.MISSING_POSE,
        )
        self.assertIs(surface.verdict, TsdfVoxelCrossViewVerdict.SURFACE)
        self.assertEqual(surface.reference_weight, 1)
        self.assertEqual(surface.contributing_observation_sequences, (1,))

        self.assertIs(occluded.verdict, TsdfVoxelCrossViewVerdict.OCCLUDED)
        self.assertFalse(occluded.carvable_free_space)
        self.assertFalse(occluded.observed)
        self.assertEqual(occluded.reference_weight, 0)
        self.assertIsNone(occluded.reference_tsdf_value)

        self.assertIs(behind.verdict, TsdfVoxelCrossViewVerdict.UNSEEN)
        self.assertFalse(behind.planned_block)
        self.assertFalse(behind.carvable_free_space)
        self.assertEqual(behind.unseen_count, 2)
        self.assertEqual(plan_snapshot(plan), before_plan)
        self.assertEqual(after_tree, before_tree)

    def test_selection_voxel_and_provenance_are_preflighted(self) -> None:
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
                (
                    "voxel-length",
                    plan,
                    context,
                    (0, 0),
                    "global_index_xyz",
                ),
                (
                    "voxel-bool",
                    plan,
                    context,
                    (True, 0, 0),
                    "global_index_xyz",
                ),
                (
                    "voxel-range",
                    plan,
                    context,
                    (10**12, 0, 0),
                    "global_index_xyz",
                ),
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
            for name, case_plan, case_context, voxel, message in cases:
                with self.subTest(name=name):
                    with self.assertRaises(TsdfError) as raised:
                        classify_tsdf_voxel_across_observations_from_context(
                            case_plan,  # type: ignore[arg-type]
                            case_context,  # type: ignore[arg-type]
                            voxel,  # type: ignore[arg-type]
                        )
                    self.assertIn(message, str(raised.exception))
            after_tree = tree_snapshot(temporary_root)

        self.assertEqual(after_tree, before_tree)

    def test_stride_selection_is_complete_and_canonical(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            plan, context = load_case(
                temporary_root,
                frame_stride=2,
                name="stride.sftplan",
            )
            receipt = classify_tsdf_voxel_across_observations_from_context(
                plan,
                context,
                (8, -1, -1),
            )

        self.assertEqual(receipt.frame_stride, 2)
        self.assertEqual(receipt.selected_observation_sequences, (0,))
        self.assertEqual(receipt.observation_count, 1)
        self.assertEqual(receipt.reference_weight, 1)
        self.assertIs(receipt.verdict, TsdfVoxelCrossViewVerdict.SURFACE)

    def test_receipts_are_frozen_slotted_and_strict(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            plan, context = load_case(Path(temporary_directory))
            receipt = classify_tsdf_voxel_across_observations_from_context(
                plan,
                context,
                (8, -1, -1),
            )
            other = classify_tsdf_voxel_across_observations_from_context(
                plan,
                context,
                (2, 0, 0),
            )

        self.assertFalse(hasattr(receipt, "__dict__"))
        self.assertEqual(
            MAX_TSDF_VOXEL_CROSS_VIEW_OBSERVATIONS,
            262_144,
        )
        with self.assertRaises(FrozenInstanceError):
            receipt.verdict = TsdfVoxelCrossViewVerdict.UNSEEN  # type: ignore[misc]

        for arguments in (
            {"verdict": TsdfVoxelCrossViewVerdict.FREE_SPACE},
            {"verdict": TsdfVoxelCrossViewVerdict.UNSEEN},
            {
                "observation_receipts": tuple(
                    reversed(receipt.observation_receipts)
                )
            },
            {"observation_receipts": receipt.observation_receipts[:1]},
            {"observation_receipts": other.observation_receipts},
            {"selected_observation_sequences": (0,)},
            {"frame_stride": 2},
            {"total_observations": 3},
            {"global_index_xyz": (2, 0, 0)},
            {"block_index_xyz": (0, 0, 0)},
            {"local_index_xyz": (0, 0, 0)},
            {"world_xyz_m": (0.0, 0.0, 0.0)},
            {"voxel_size_m": 0.25},
            {"truncation_m": 0.0},
            {"planned_block": 1},
            {"block_resolution": 7},
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
            injected = RuntimeError("injected final cross-view failure")

            with patch(
                "spatialforge.tsdf_voxel_cross_view."
                "TsdfVoxelCrossViewReceipt",
                side_effect=injected,
            ):
                with self.assertRaises(TsdfError) as raised:
                    classify_tsdf_voxel_across_observations_from_context(
                        plan,
                        context,
                        (8, -1, -1),
                    )
            after_tree = tree_snapshot(temporary_root)

        self.assertIs(raised.exception.__cause__, injected)
        self.assertIn(
            "injected final cross-view failure",
            str(raised.exception),
        )
        self.assertEqual(plan_snapshot(plan), before_plan)
        self.assertEqual(after_tree, before_tree)


class TsdfVoxelCrossViewCliTests(unittest.TestCase):
    def test_cli_reports_exact_read_only_cross_view_checkpoint(self) -> None:
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
            actual = classify_tsdf_voxel_across_observations_from_context

            with (
                forbidden_cli_calls() as forbidden,
                patch(
                    "spatialforge.cli."
                    "classify_tsdf_voxel_across_observations_from_context",
                    wraps=actual,
                ) as classifier,
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                exit_code = main(
                    [
                        "reconstruct",
                        "tsdf-block-context-voxel-cross-view",
                        str(plan_path),
                        str(session_path),
                        "--voxel",
                        "8",
                        "-1",
                        "-1",
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
            "TSDF BLOCK CONTEXT VOXEL CROSS-VIEW CHECK scan-synthetic-0001\n",
            "voxel: global=(8, -1, -1) block=(1, -1, -1) local=(0, 7, 7)\n",
            "voxel_block_planned: yes\n",
            "observations: selected=2 surface_band=2 free_space=0 "
            "occluded=0 unseen=0\n",
            "status_counts: observed-surface-band=2\n",
            "cross_view_verdict: surface\n",
            "verdict_precedence: surface-then-free-space-then-occluded-then-"
            "unseen\n",
            "occlusion_rule: carries-no-evidence-never-becomes-free-space\n",
            "contributing_observations: (0, 1)\n",
            "reference_weight: 2\n",
            "reference_tsdf_sum: -0.250000000\n",
            "reference_tsdf_value: -0.125000000\n",
            "voxel_observed: yes\n",
            "carvable_free_space: no\n",
            "cross_view_scope: one-voxel-all-observations\n",
            "unplanned_voxels_accepted: yes\n",
            "multi_voxel_cross_view_computed: no\n",
            "visibility_culling_rule_defined: no\n",
            "confidence_or_sensor_weighting_applied: no\n",
            "free_space_carving_applied: no\n",
            "plan_expanded: no\n",
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
                    "classify_tsdf_voxel_across_observations_from_context",
                    side_effect=TsdfError("injected cross-view failure"),
                ) as classifier,
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                exit_code = main(
                    [
                        "reconstruct",
                        "tsdf-block-context-voxel-cross-view",
                        str(plan_path),
                        str(session_path),
                        "--voxel",
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
        self.assertIn("TSDF BLOCK CONTEXT VOXEL CROSS-VIEW FAILED", error)
        self.assertIn("injected cross-view failure", error)
        self.assertNotIn("Traceback", error)


if __name__ == "__main__":
    unittest.main()
