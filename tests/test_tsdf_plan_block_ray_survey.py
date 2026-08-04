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
from spatialforge.tsdf_plan_block_ray_survey import (
    MAX_TSDF_PLAN_BLOCK_RAY_SURVEY_OUTCOMES,
    TsdfPlanBlockRaySurveyReceipt,
    survey_tsdf_plan_block_rays_from_context,
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
EXPECTED_SURFACE_BLOCKS = (
    (1, -1, -1),
    (1, 0, -1),
    (1, -1, 0),
    (1, 0, 0),
)
EXPECTED_NONTERMINAL_BLOCKS = (
    (0, -1, -1),
    (0, 0, -1),
    (0, -1, 0),
    (0, 0, 0),
)

_FORBIDDEN_SURVEY_TARGETS = (
    "spatialforge.tsdf_plan_block_ray_survey.replay_session",
    "spatialforge.tsdf_plan_block_ray_survey."
    "build_tsdf_replay_depth_context",
    "spatialforge.tsdf_plan_block_ray_survey.allocate_empty_tsdf_blocks",
    "spatialforge.tsdf_plan_block_ray_survey.locate_tsdf_voxel",
    "spatialforge.tsdf_plan_block_ray_survey."
    "evaluate_tsdf_voxel_contribution",
    "spatialforge.tsdf_plan_block_ray_survey."
    "evaluate_tsdf_voxel_contribution_from_context",
    "spatialforge.tsdf_plan_block_ray_survey."
    "apply_tsdf_voxel_contribution",
    "spatialforge.tsdf_plan_block_ray_survey."
    "apply_tsdf_voxel_contribution_from_context",
    "spatialforge.tsdf_plan_block_ray_survey."
    "traverse_tsdf_voxel_observations_from_context",
    "spatialforge.tsdf_plan_block_ray_survey."
    "traverse_tsdf_block_voxels_from_context",
    "spatialforge.tsdf_plan_block_ray_survey."
    "traverse_tsdf_plan_blocks_from_context",
    "spatialforge.replay._file_digest",
    "pathlib.Path.open",
    "PIL.Image.open",
)

_FORBIDDEN_CLI_TARGETS = (
    "spatialforge.cli.allocate_empty_tsdf_blocks",
    "spatialforge.cli.locate_tsdf_voxel",
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


def context_snapshot(context: TsdfReplayDepthContext) -> tuple[object, ...]:
    observations: list[object] = []
    for observation in context.observations:
        depth = observation.depth_m
        observations.append(
            (
                observation.observation_sequence,
                observation.status,
                observation.t_world_camera,
                None
                if depth is None
                else (
                    depth.shape,
                    depth.dtype.str,
                    depth.strides,
                    bool(depth.flags.c_contiguous),
                    bool(depth.flags.owndata),
                    bool(depth.flags.writeable),
                    depth.tobytes(),
                ),
            )
        )
    return (
        context.session_id,
        context.source_plan_digest_sha256,
        context.replay_digest_sha256,
        context.frame_stride,
        context.total_observations,
        context.selected_observation_sequences,
        context.camera,
        context.depth_scale_m,
        tuple(observations),
        context.valid_depth_samples,
        context.invalid_depth_samples,
    )


def canonical_quadrant_blocks(x_values: range | tuple[int, ...]) -> tuple[
    tuple[int, int, int], ...
]:
    return tuple(
        (x, y, z)
        for z in (-1, 0)
        for y in (-1, 0)
        for x in x_values
    )


def set_depth_sample(session_path: Path, filename: str, raw_depth: int) -> None:
    (session_path / "data" / "depth" / filename).write_text(
        "P2\n2 2\n65535\n"
        f"{raw_depth} {raw_depth}\n{raw_depth} {raw_depth}\n",
        encoding="ascii",
    )


def remove_first_record(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text(lines[1] + "\n", encoding="utf-8")


@contextmanager
def forbidden_survey_calls():
    with ExitStack() as stack:
        mocks: list[MagicMock] = []
        for target in _FORBIDDEN_SURVEY_TARGETS:
            mocks.append(stack.enter_context(patch(target, create=True)))
        yield mocks


@contextmanager
def forbidden_cli_calls():
    with ExitStack() as stack:
        mocks: list[MagicMock] = []
        for target in _FORBIDDEN_CLI_TARGETS:
            mocks.append(stack.enter_context(patch(target, create=True)))
        yield mocks


class TsdfPlanBlockRaySurveyTests(unittest.TestCase):
    def test_fixture_survey_matches_independent_per_observation_traces(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            plan, context = load_case(temporary_root)
            before_tree = tree_snapshot(temporary_root)
            before_plan = plan_snapshot(plan)
            before_context = context_snapshot(context)

            with (
                forbidden_survey_calls() as forbidden,
                patch(
                    "spatialforge.tsdf_plan_block_ray_survey."
                    "trace_tsdf_observation_block_rays_from_context",
                    wraps=trace_tsdf_observation_block_rays_from_context,
                ) as tracer,
            ):
                receipt = survey_tsdf_plan_block_rays_from_context(
                    plan,
                    context,
                )

            repeated = survey_tsdf_plan_block_rays_from_context(plan, context)
            expected_children = tuple(
                trace_tsdf_observation_block_rays_from_context(
                    plan,
                    context,
                    observation_sequence,
                )
                for observation_sequence in (0, 1)
            )
            after_tree = tree_snapshot(temporary_root)

        self.assertIsInstance(receipt, TsdfPlanBlockRaySurveyReceipt)
        self.assertEqual(receipt, repeated)
        self.assertEqual(
            [call.args[2] for call in tracer.call_args_list],
            [0, 1],
        )
        self.assertEqual(receipt.observation_receipts, expected_children)
        self.assertEqual(receipt.source_plan_digest_sha256, PLAN_SHA256)
        self.assertEqual(receipt.replay_digest_sha256, REPLAY_SHA256)
        self.assertEqual(receipt.frame_stride, 1)
        self.assertEqual(receipt.total_observations, 2)
        self.assertEqual(receipt.selected_observation_sequences, (0, 1))
        self.assertEqual(
            receipt.source_plan_block_indices,
            EXPECTED_ACTIVE_BLOCKS,
        )
        self.assertEqual(receipt.block_resolution, 8)
        self.assertEqual(receipt.block_extent_m, 1.0)
        self.assertEqual(receipt.image_size, (2, 2))
        self.assertEqual(receipt.observation_count, 2)
        self.assertEqual(receipt.traced_observation_count, 2)
        self.assertEqual(
            receipt.observation_status_counts,
            ((TsdfReplayDepthStatus.READY, 2),),
        )
        self.assertEqual(receipt.covered_block_indices, EXPECTED_ACTIVE_BLOCKS)
        self.assertEqual(
            receipt.existing_plan_block_indices,
            EXPECTED_ACTIVE_BLOCKS,
        )
        self.assertEqual(receipt.unplanned_block_indices, ())
        self.assertEqual(receipt.surface_block_indices, EXPECTED_SURFACE_BLOCKS)
        self.assertEqual(
            receipt.nonterminal_block_indices,
            EXPECTED_NONTERMINAL_BLOCKS,
        )
        self.assertEqual(
            receipt.covered_block_observation_counts,
            tuple((block, 2) for block in EXPECTED_ACTIVE_BLOCKS),
        )
        self.assertEqual(
            receipt.multi_observation_block_indices,
            EXPECTED_ACTIVE_BLOCKS,
        )
        self.assertEqual(receipt.maximum_block_observation_count, 2)
        self.assertEqual(receipt.pixel_count, 8)
        self.assertEqual(receipt.traversed_ray_count, 8)
        self.assertEqual(receipt.invalid_depth_count, 0)
        self.assertEqual(receipt.block_visit_count, 22)
        self.assertEqual(receipt.duplicate_block_visit_count, 14)
        self.assertEqual(receipt.maximum_blocks_per_ray, 3)
        self.assertEqual(receipt.retained_outcome_count, 30)
        self.assertTrue(receipt.prepared_depth_accessed)
        self.assertEqual(plan_snapshot(plan), before_plan)
        self.assertEqual(context_snapshot(context), before_context)
        self.assertEqual(after_tree, before_tree)
        for forbidden_call in forbidden:
            forbidden_call.assert_not_called()

    def test_union_grows_with_observations_and_records_support_counts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            session_path = copy_fixture(temporary_root)
            set_depth_sample(session_path, "000000.pgm", 1000)
            set_depth_sample(session_path, "000001.pgm", 3000)
            plan, context = load_case(
                temporary_root,
                session_path=session_path,
            )
            before_tree = tree_snapshot(temporary_root)
            before_plan = plan_snapshot(plan)
            before_context = context_snapshot(context)

            receipt = survey_tsdf_plan_block_rays_from_context(plan, context)
            after_tree = tree_snapshot(temporary_root)

        near = receipt.observation_receipts[0]
        far = receipt.observation_receipts[1]
        expected_covered = canonical_quadrant_blocks(range(0, 4))
        expected_near = canonical_quadrant_blocks((0, 1))
        expected_far_only = canonical_quadrant_blocks((2, 3))

        self.assertEqual(near.covered_block_indices, expected_near)
        self.assertEqual(far.covered_block_indices, expected_covered)
        self.assertEqual(receipt.covered_block_indices, expected_covered)
        self.assertEqual(len(receipt.covered_block_indices), 16)
        self.assertGreater(
            len(receipt.covered_block_indices),
            len(near.covered_block_indices),
        )
        self.assertEqual(
            receipt.covered_block_observation_counts,
            tuple(
                (block, 2 if block in set(expected_near) else 1)
                for block in expected_covered
            ),
        )
        self.assertEqual(
            receipt.multi_observation_block_indices,
            expected_near,
        )
        self.assertEqual(receipt.maximum_block_observation_count, 2)
        self.assertEqual(
            receipt.surface_block_indices,
            canonical_quadrant_blocks((1, 3)),
        )
        self.assertEqual(
            receipt.nonterminal_block_indices,
            canonical_quadrant_blocks(range(0, 3)),
        )
        self.assertEqual(
            set(receipt.covered_block_indices)
            - set(near.covered_block_indices),
            set(expected_far_only),
        )
        self.assertEqual(receipt.pixel_count, 8)
        self.assertEqual(receipt.traversed_ray_count, 8)
        self.assertEqual(receipt.block_visit_count, 30)
        self.assertEqual(receipt.duplicate_block_visit_count, 14)
        self.assertEqual(receipt.maximum_blocks_per_ray, 5)
        self.assertEqual(near.maximum_blocks_per_ray, 3)
        self.assertEqual(receipt.retained_outcome_count, 38)
        self.assertEqual(plan_snapshot(plan), before_plan)
        self.assertEqual(context_snapshot(context), before_context)
        self.assertEqual(after_tree, before_tree)

    def test_unplanned_coverage_is_reported_without_expanding_the_plan(
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
            before_tree = tree_snapshot(temporary_root)
            before_plan = plan_snapshot(plan)

            receipt = survey_tsdf_plan_block_rays_from_context(plan, context)
            after_tree = tree_snapshot(temporary_root)

        self.assertEqual(len(plan.active_blocks), 32)
        self.assertEqual(
            receipt.covered_block_indices,
            canonical_quadrant_blocks(range(0, 4)),
        )
        self.assertEqual(
            receipt.existing_plan_block_indices,
            canonical_quadrant_blocks((2, 3)),
        )
        self.assertEqual(
            receipt.unplanned_block_indices,
            canonical_quadrant_blocks((0, 1)),
        )
        self.assertEqual(receipt.block_visit_count, 38)
        self.assertEqual(receipt.duplicate_block_visit_count, 22)
        self.assertEqual(receipt.retained_outcome_count, 46)
        self.assertEqual(receipt.maximum_block_observation_count, 2)
        self.assertEqual(
            set(receipt.unplanned_block_indices) & set(plan.active_blocks),
            set(),
        )
        self.assertEqual(plan_snapshot(plan), before_plan)
        self.assertEqual(
            plan.active_blocks,
            dict(before_plan)["active_blocks"],
        )
        self.assertEqual(after_tree, before_tree)

    def test_missing_inputs_are_surveyed_without_inventing_rays(self) -> None:
        cases = (
            ("missing-depth", True, False, TsdfReplayDepthStatus.MISSING_DEPTH),
            ("missing-pose", False, True, TsdfReplayDepthStatus.MISSING_POSE),
            (
                "missing-both",
                True,
                True,
                TsdfReplayDepthStatus.MISSING_DEPTH_AND_POSE,
            ),
        )
        for name, remove_depth, remove_pose, status in cases:
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
                    before_context = context_snapshot(context)

                    receipt = survey_tsdf_plan_block_rays_from_context(
                        plan,
                        context,
                    )

                    self.assertEqual(receipt.observation_count, 2)
                    self.assertEqual(
                        tuple(
                            child.observation_status
                            for child in receipt.observation_receipts
                        ),
                        (status, TsdfReplayDepthStatus.READY),
                    )
                    self.assertEqual(
                        receipt.observation_receipts[0].ray_receipts,
                        (),
                    )
                    self.assertEqual(
                        receipt.observation_receipts[0].covered_block_indices,
                        (),
                    )
                    self.assertEqual(
                        receipt.observation_status_counts,
                        (
                            (TsdfReplayDepthStatus.READY, 1),
                            (status, 1),
                        ),
                    )
                    self.assertEqual(receipt.traced_observation_count, 1)
                    self.assertEqual(
                        receipt.covered_block_indices,
                        EXPECTED_ACTIVE_BLOCKS,
                    )
                    self.assertEqual(
                        receipt.multi_observation_block_indices,
                        (),
                    )
                    self.assertEqual(
                        receipt.maximum_block_observation_count,
                        1,
                    )
                    self.assertEqual(receipt.pixel_count, 4)
                    self.assertEqual(receipt.traversed_ray_count, 4)
                    self.assertEqual(receipt.invalid_depth_count, 0)
                    self.assertEqual(receipt.block_visit_count, 11)
                    self.assertEqual(receipt.retained_outcome_count, 15)
                    self.assertTrue(receipt.prepared_depth_accessed)
                    self.assertEqual(
                        context_snapshot(context),
                        before_context,
                    )
                    self.assertEqual(tree_snapshot(temporary_root), before_tree)

    def test_survey_follows_the_complete_canonical_stride_selection(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            plan, context = load_case(temporary_root)
            stride_plan, stride_context = load_case(
                temporary_root,
                frame_stride=2,
                name="stride.sftplan",
            )
            before_tree = tree_snapshot(temporary_root)

            receipt = survey_tsdf_plan_block_rays_from_context(
                stride_plan,
                stride_context,
            )
            with self.assertRaises(TsdfError) as foreign:
                survey_tsdf_plan_block_rays_from_context(
                    stride_plan,
                    context,
                )
            with self.assertRaises(TsdfError) as unselected:
                trace_tsdf_observation_block_rays_from_context(
                    stride_plan,
                    stride_context,
                    1,
                )
            after_tree = tree_snapshot(temporary_root)

        self.assertEqual(receipt.frame_stride, 2)
        self.assertEqual(receipt.total_observations, 2)
        self.assertEqual(receipt.selected_observation_sequences, (0,))
        self.assertEqual(receipt.observation_count, 1)
        self.assertEqual(
            receipt.observation_receipts[0].observation_sequence,
            0,
        )
        self.assertEqual(receipt.traced_observation_count, 1)
        self.assertEqual(receipt.covered_block_indices, EXPECTED_ACTIVE_BLOCKS)
        self.assertEqual(receipt.multi_observation_block_indices, ())
        self.assertEqual(receipt.retained_outcome_count, 15)
        self.assertIn("does not match", str(foreign.exception))
        self.assertIn("frame_stride=2", str(unselected.exception))
        self.assertEqual(after_tree, before_tree)

    def test_types_provenance_and_geometry_are_preflighted(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            plan, context = load_case(temporary_root)
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
                    "replay-digest",
                    plan,
                    replace(context, replay_digest_sha256="0" * 64),
                    "replay digest",
                ),
                (
                    "extent",
                    replace(plan, block_extent_m=2.0),
                    context,
                    "block_extent_m",
                ),
            )
            for name, candidate_plan, candidate_context, message in cases:
                with self.subTest(name=name):
                    with self.assertRaises(TsdfError) as raised:
                        survey_tsdf_plan_block_rays_from_context(
                            candidate_plan,  # type: ignore[arg-type]
                            candidate_context,  # type: ignore[arg-type]
                        )
                    self.assertIn(message, str(raised.exception))
            after_tree = tree_snapshot(temporary_root)

        self.assertEqual(after_tree, before_tree)

    def test_outcome_cap_is_preflighted_accumulated_and_atomic(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            plan, context = load_case(temporary_root)
            before_tree = tree_snapshot(temporary_root)
            before_plan = plan_snapshot(plan)
            before_context = context_snapshot(context)

            self.assertEqual(
                MAX_TSDF_PLAN_BLOCK_RAY_SURVEY_OUTCOMES,
                262_144,
            )
            with patch(
                "spatialforge.tsdf_plan_block_ray_survey."
                "MAX_TSDF_PLAN_BLOCK_RAY_SURVEY_OUTCOMES",
                7,
            ):
                with self.assertRaises(TsdfError) as preflight:
                    survey_tsdf_plan_block_rays_from_context(plan, context)
            with patch(
                "spatialforge.tsdf_plan_block_ray_survey."
                "MAX_TSDF_PLAN_BLOCK_RAY_SURVEY_OUTCOMES",
                15,
            ):
                with self.assertRaises(TsdfError) as accumulated:
                    survey_tsdf_plan_block_rays_from_context(plan, context)
            with patch(
                "spatialforge.tsdf_plan_block_ray_survey."
                "MAX_TSDF_PLAN_BLOCK_RAY_SURVEY_OUTCOMES",
                30,
            ):
                receipt = survey_tsdf_plan_block_rays_from_context(
                    plan,
                    context,
                )
            after_tree = tree_snapshot(temporary_root)

        self.assertIn(
            "pixel receipts alone reach 8",
            str(preflight.exception),
        )
        self.assertIn("maximum is 7", str(preflight.exception))
        self.assertIn(
            "retained outcomes reach 30",
            str(accumulated.exception),
        )
        self.assertIn("maximum is 15", str(accumulated.exception))
        self.assertEqual(receipt.retained_outcome_count, 30)
        self.assertEqual(plan_snapshot(plan), before_plan)
        self.assertEqual(context_snapshot(context), before_context)
        self.assertEqual(after_tree, before_tree)

    def test_receipts_are_frozen_slotted_and_strict(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            plan, context = load_case(Path(temporary_directory))
            receipt = survey_tsdf_plan_block_rays_from_context(plan, context)

        self.assertFalse(hasattr(receipt, "__dict__"))
        with self.assertRaises(FrozenInstanceError):
            receipt.frame_stride = 2  # type: ignore[misc]

        for arguments in (
            {
                "observation_receipts": tuple(
                    reversed(receipt.observation_receipts)
                )
            },
            {"observation_receipts": receipt.observation_receipts[:1]},
            {"selected_observation_sequences": (0,)},
            {"frame_stride": 2},
            {"total_observations": 3},
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
            {"replay_digest_sha256": "0"},
            {"block_resolution": 7},
            {"block_extent_m": 0.0},
            {"image_size": (2, 3)},
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
            before_context = context_snapshot(context)
            injected = RuntimeError("injected final survey receipt failure")

            with patch(
                "spatialforge.tsdf_plan_block_ray_survey."
                "TsdfPlanBlockRaySurveyReceipt",
                side_effect=injected,
            ):
                with self.assertRaises(TsdfError) as raised:
                    survey_tsdf_plan_block_rays_from_context(plan, context)
            after_tree = tree_snapshot(temporary_root)

        self.assertIs(raised.exception.__cause__, injected)
        self.assertIn(
            "injected final survey receipt failure",
            str(raised.exception),
        )
        self.assertEqual(plan_snapshot(plan), before_plan)
        self.assertEqual(context_snapshot(context), before_context)
        self.assertEqual(after_tree, before_tree)


class TsdfPlanBlockRaySurveyCliTests(unittest.TestCase):
    def test_cli_reports_exact_read_only_survey_checkpoint(self) -> None:
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
            actual_survey = survey_tsdf_plan_block_rays_from_context

            with (
                forbidden_cli_calls() as forbidden,
                patch(
                    "spatialforge.cli."
                    "survey_tsdf_plan_block_rays_from_context",
                    wraps=actual_survey,
                ) as surveyor,
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                exit_code = main(
                    [
                        "reconstruct",
                        "tsdf-block-context-plan-rays",
                        str(plan_path),
                        str(session_path),
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
            "TSDF BLOCK CONTEXT PLAN RAY SURVEY CHECK scan-synthetic-0001\n",
            "context_selection: frame_stride=1 total=2 selected=2\n",
            "observations: selected=2 traced=2 ready=2 missing_depth=0 "
            "missing_pose=0 missing_depth_and_pose=0\n",
            "observation_order: canonical-frame-stride sequences=0..1\n",
            "first_observation: sequence=0 status=ready covered_blocks=8\n",
            "last_observation: sequence=1 status=ready covered_blocks=8\n",
            "image: width=2 height=2\n",
            "pixel_outcomes: total=8 traversed=8 depth_invalid=0\n",
            "ray_block_visits: total=22 unique=8 duplicate=14 "
            "maximum_per_ray=3\n",
            "coverage_blocks: total=8 nonterminal=4 surface_endpoint=4\n",
            "coverage_partition: existing_plan=8 unplanned=0\n",
            "coverage_support: multi_observation=8 maximum_observations=2\n",
            "first_covered_block: (0, -1, -1)\n",
            "last_covered_block: (1, 0, 0)\n",
            "survey_session_replay: no\n",
            "survey_depth_decoding: no\n",
            "survey_prepared_depth_access: yes\n",
            "survey_workload: retained_outcomes=30 maximum=262144\n",
            "coverage_scope: all-plan-selected-observations\n",
            "block_traversal_rule: closed-half-open-grid-thin-dda-"
            "simultaneous-exact-ties\n",
            "multiple_observation_coverage_computed: yes\n",
            "all_selected_observations_surveyed: yes\n",
            "conservative_nearest_pixel_free_space_coverage_proven: no\n",
            "coverage_approved_for_expansion: no\n",
            "plan_expanded: no\n",
            "missing_blocks_created: no\n",
            "storage_allocated: no\n",
            "storage_mutated: no\n",
            "full_fusion_performed: no\n",
            "artifact_written: no\n",
            f"plan_sha256: {PLAN_SHA256}\n",
            f"replay_digest_sha256: {REPLAY_SHA256}\n",
        ):
            self.assertIn(expected, output)

    def test_cli_reports_survey_failure_without_writing_or_traceback(
        self,
    ) -> None:
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
                    "survey_tsdf_plan_block_rays_from_context",
                    side_effect=TsdfError("injected plan ray survey failure"),
                ) as surveyor,
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                exit_code = main(
                    [
                        "reconstruct",
                        "tsdf-block-context-plan-rays",
                        str(plan_path),
                        str(session_path),
                    ]
                )
            after_tree = tree_snapshot(temporary_root)

        self.assertEqual(exit_code, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(after_tree, before_tree)
        surveyor.assert_called_once()
        error = stderr.getvalue()
        self.assertIn("TSDF BLOCK CONTEXT PLAN RAY SURVEY FAILED", error)
        self.assertIn("injected plan ray survey failure", error)
        self.assertNotIn("Traceback", error)


if __name__ == "__main__":
    unittest.main()
