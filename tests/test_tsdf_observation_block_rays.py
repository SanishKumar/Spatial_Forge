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
    MAX_TSDF_OBSERVATION_BLOCK_RAY_OUTCOMES,
    TsdfObservationBlockRayReceipt,
    TsdfObservationBlockRayStatus,
    TsdfObservationBlockRayTraceReceipt,
    _trace_closed_block_segment,
    trace_tsdf_observation_block_rays_from_context,
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
EXPECTED_FIXTURE_RAYS = (
    (
        (0, 0),
        1.0,
        (1.0, 0.25, 0.25),
        (1, 0, 0),
        ((0, 0, 0), (1, 0, 0)),
    ),
    (
        (1, 0),
        1.0,
        (1.0, -0.25, 0.25),
        (1, -1, 0),
        ((0, 0, 0), (0, -1, 0), (1, -1, 0)),
    ),
    (
        (0, 1),
        1.0,
        (1.0, 0.25, -0.25),
        (1, 0, -1),
        ((0, 0, 0), (0, 0, -1), (1, 0, -1)),
    ),
    (
        (1, 1),
        1.0,
        (1.0, -0.25, -0.25),
        (1, -1, -1),
        ((0, 0, 0), (0, -1, -1), (1, -1, -1)),
    ),
)

_FORBIDDEN_TRACE_TARGETS = (
    "spatialforge.tsdf_observation_block_rays.replay_session",
    "spatialforge.tsdf_observation_block_rays.build_tsdf_replay_depth_context",
    "spatialforge.tsdf_observation_block_rays.allocate_empty_tsdf_blocks",
    "spatialforge.tsdf_observation_block_rays.locate_tsdf_voxel",
    "spatialforge.tsdf_observation_block_rays."
    "evaluate_tsdf_voxel_contribution",
    "spatialforge.tsdf_observation_block_rays."
    "evaluate_tsdf_voxel_contribution_from_context",
    "spatialforge.tsdf_observation_block_rays."
    "apply_tsdf_voxel_contribution",
    "spatialforge.tsdf_observation_block_rays."
    "apply_tsdf_voxel_contribution_from_context",
    "spatialforge.tsdf_observation_block_rays."
    "traverse_tsdf_voxel_observations_from_context",
    "spatialforge.tsdf_observation_block_rays."
    "traverse_tsdf_block_voxels_from_context",
    "spatialforge.tsdf_observation_block_rays."
    "traverse_tsdf_plan_blocks_from_context",
    "spatialforge.tsdf_observation_block_rays._read_depth",
    "spatialforge.tsdf_observation_block_rays._sample_path",
    "spatialforge.replay._file_digest",
    "pathlib.Path.open",
    "PIL.Image.open",
)

_FORBIDDEN_CLI_TARGETS = (
    "spatialforge.cli.allocate_empty_tsdf_blocks",
    "spatialforge.cli.locate_tsdf_voxel",
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


def ray_tuple(
    receipt: TsdfObservationBlockRayReceipt,
) -> tuple[object, ...]:
    return (
        receipt.pixel_uv,
        receipt.measured_depth_m,
        receipt.surface_world_m,
        receipt.surface_block_index,
        receipt.block_indices,
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


def set_all_depth_samples(session_path: Path, raw_depth: int) -> None:
    payload = (
        "P2\n2 2\n65535\n"
        f"{raw_depth} {raw_depth}\n{raw_depth} {raw_depth}\n"
    )
    for filename in ("000000.pgm", "000001.pgm"):
        (session_path / "data" / "depth" / filename).write_text(
            payload,
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
def forbidden_trace_calls():
    with ExitStack() as stack:
        mocks: list[MagicMock] = []
        for target in _FORBIDDEN_TRACE_TARGETS:
            mocks.append(stack.enter_context(patch(target, create=True)))
        yield mocks


@contextmanager
def forbidden_cli_calls():
    with ExitStack() as stack:
        mocks: list[MagicMock] = []
        for target in _FORBIDDEN_CLI_TARGETS:
            mocks.append(stack.enter_context(patch(target, create=True)))
        yield mocks


class TsdfObservationBlockRayTests(unittest.TestCase):
    def test_closed_half_open_thin_dda_pins_boundary_and_tie_rules(self) -> None:
        cases = (
            (
                (0.0, 0.0, 0.0),
                (1.0, 1.0, 1.0),
                ((0, 0, 0), (1, 1, 1)),
            ),
            (
                (0.0, 0.0, 0.0),
                (-1.0, -1.0, -1.0),
                ((0, 0, 0), (-1, -1, -1)),
            ),
            (
                (0.5, 0.5, 0.5),
                (1.0, 0.5, 0.5),
                ((0, 0, 0), (1, 0, 0)),
            ),
            (
                (1.0, 0.5, 0.5),
                (0.5, 0.5, 0.5),
                ((1, 0, 0), (0, 0, 0)),
            ),
            (
                (0.5, 0.5, 0.5),
                (2.5, 0.5, 0.5),
                ((0, 0, 0), (1, 0, 0), (2, 0, 0)),
            ),
        )
        for start, end, expected in cases:
            with self.subTest(start=start, end=end):
                self.assertEqual(
                    _trace_closed_block_segment(start, end, 1.0),
                    expected,
                )

    def test_fixture_has_exact_thin_ray_geometry_order_and_partition(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            plan, context = load_case(temporary_root)
            before_tree = tree_snapshot(temporary_root)
            before_plan = plan_snapshot(plan)
            before_context = context_snapshot(context)

            with forbidden_trace_calls() as forbidden:
                receipt = trace_tsdf_observation_block_rays_from_context(
                    plan,
                    context,
                    0,
                )

            repeated = trace_tsdf_observation_block_rays_from_context(
                plan,
                context,
                0,
            )
            after_tree = tree_snapshot(temporary_root)

        self.assertIsInstance(receipt, TsdfObservationBlockRayTraceReceipt)
        self.assertEqual(receipt, repeated)
        self.assertEqual(receipt.source_plan_digest_sha256, PLAN_SHA256)
        self.assertEqual(receipt.replay_digest_sha256, REPLAY_SHA256)
        self.assertEqual(receipt.frame_stride, 1)
        self.assertEqual(receipt.total_observations, 2)
        self.assertEqual(receipt.selected_observation_sequences, (0, 1))
        self.assertEqual(
            receipt.source_plan_block_indices,
            EXPECTED_ACTIVE_BLOCKS,
        )
        self.assertEqual(receipt.observation_sequence, 0)
        self.assertIs(receipt.observation_status, TsdfReplayDepthStatus.READY)
        self.assertEqual(receipt.block_resolution, 8)
        self.assertEqual(receipt.block_extent_m, 1.0)
        self.assertEqual(receipt.camera_origin_world_m, (0.0, 0.0, 0.0))
        self.assertEqual(receipt.image_size, (2, 2))
        self.assertEqual(
            tuple(ray.status for ray in receipt.ray_receipts),
            (TsdfObservationBlockRayStatus.TRAVERSED,) * 4,
        )
        self.assertEqual(
            tuple(
                ray.camera_origin_world_m for ray in receipt.ray_receipts
            ),
            ((0.0, 0.0, 0.0),) * 4,
        )
        self.assertEqual(
            tuple(ray.block_extent_m for ray in receipt.ray_receipts),
            (1.0,) * 4,
        )
        self.assertEqual(
            tuple(ray_tuple(ray) for ray in receipt.ray_receipts),
            EXPECTED_FIXTURE_RAYS,
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
        self.assertEqual(receipt.pixel_count, 4)
        self.assertEqual(receipt.traversed_ray_count, 4)
        self.assertEqual(receipt.invalid_depth_count, 0)
        self.assertEqual(receipt.block_visit_count, 11)
        self.assertEqual(receipt.duplicate_block_visit_count, 3)
        self.assertEqual(receipt.retained_outcome_count, 15)
        self.assertEqual(receipt.maximum_blocks_per_ray, 3)
        self.assertTrue(receipt.prepared_depth_accessed)
        self.assertEqual(plan_snapshot(plan), before_plan)
        self.assertEqual(context_snapshot(context), before_context)
        self.assertEqual(after_tree, before_tree)
        for forbidden_call in forbidden:
            forbidden_call.assert_not_called()

    def test_far_depth_reports_unplanned_nonterminal_blocks_without_mutation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            session_path = copy_fixture(temporary_root)
            set_all_depth_samples(session_path, 3000)
            plan, context = load_case(
                temporary_root,
                session_path=session_path,
            )
            before_tree = tree_snapshot(temporary_root)
            before_plan = plan_snapshot(plan)
            before_context = context_snapshot(context)

            receipt = trace_tsdf_observation_block_rays_from_context(
                plan,
                context,
                0,
            )
            after_tree = tree_snapshot(temporary_root)

        expected_covered = canonical_quadrant_blocks(range(0, 4))
        expected_existing = canonical_quadrant_blocks((2, 3))
        expected_unplanned = canonical_quadrant_blocks((0, 1))
        expected_nonterminal = canonical_quadrant_blocks(range(0, 3))
        expected_surface = canonical_quadrant_blocks((3,))
        expected_paths = (
            ((0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)),
            (
                (0, 0, 0),
                (0, -1, 0),
                (1, -1, 0),
                (2, -1, 0),
                (3, -1, 0),
            ),
            (
                (0, 0, 0),
                (0, 0, -1),
                (1, 0, -1),
                (2, 0, -1),
                (3, 0, -1),
            ),
            (
                (0, 0, 0),
                (0, -1, -1),
                (1, -1, -1),
                (2, -1, -1),
                (3, -1, -1),
            ),
        )

        self.assertEqual(len(plan.active_blocks), 32)
        self.assertEqual(
            tuple(ray.block_indices for ray in receipt.ray_receipts),
            expected_paths,
        )
        self.assertEqual(receipt.covered_block_indices, expected_covered)
        self.assertEqual(receipt.existing_plan_block_indices, expected_existing)
        self.assertEqual(receipt.unplanned_block_indices, expected_unplanned)
        self.assertEqual(receipt.surface_block_indices, expected_surface)
        self.assertEqual(
            receipt.nonterminal_block_indices,
            expected_nonterminal,
        )
        self.assertEqual(receipt.pixel_count, 4)
        self.assertEqual(receipt.traversed_ray_count, 4)
        self.assertEqual(receipt.block_visit_count, 19)
        self.assertEqual(receipt.duplicate_block_visit_count, 3)
        self.assertEqual(receipt.retained_outcome_count, 23)
        self.assertEqual(receipt.maximum_blocks_per_ray, 5)
        self.assertEqual(max(block[0] for block in receipt.covered_block_indices), 3)
        self.assertNotIn((4, 0, 0), receipt.covered_block_indices)
        self.assertEqual(plan_snapshot(plan), before_plan)
        self.assertEqual(context_snapshot(context), before_context)
        self.assertEqual(after_tree, before_tree)

    def test_invalid_depth_is_unknown_and_does_not_create_a_ray(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            session_path = copy_fixture(temporary_root)
            set_first_depth_sample_invalid(session_path)
            plan, context = load_case(
                temporary_root,
                session_path=session_path,
            )
            before_tree = tree_snapshot(temporary_root)
            before_context = context_snapshot(context)

            receipt = trace_tsdf_observation_block_rays_from_context(
                plan,
                context,
                0,
            )
            after_tree = tree_snapshot(temporary_root)

        invalid = receipt.ray_receipts[0]
        self.assertEqual(invalid.pixel_uv, (0, 0))
        self.assertIs(invalid.status, TsdfObservationBlockRayStatus.DEPTH_INVALID)
        self.assertIsNone(invalid.measured_depth_m)
        self.assertIsNone(invalid.surface_world_m)
        self.assertIsNone(invalid.surface_block_index)
        self.assertEqual(invalid.block_indices, ())
        self.assertEqual(
            tuple(ray.status for ray in receipt.ray_receipts[1:]),
            (TsdfObservationBlockRayStatus.TRAVERSED,) * 3,
        )
        self.assertEqual(receipt.pixel_count, 4)
        self.assertEqual(receipt.traversed_ray_count, 3)
        self.assertEqual(receipt.invalid_depth_count, 1)
        self.assertEqual(receipt.block_visit_count, 9)
        self.assertEqual(receipt.duplicate_block_visit_count, 2)
        self.assertEqual(receipt.retained_outcome_count, 13)
        self.assertEqual(receipt.maximum_blocks_per_ray, 3)
        self.assertEqual(
            receipt.covered_block_indices,
            (
                (0, -1, -1),
                (1, -1, -1),
                (0, 0, -1),
                (1, 0, -1),
                (0, -1, 0),
                (1, -1, 0),
                (0, 0, 0),
            ),
        )
        self.assertEqual(
            receipt.existing_plan_block_indices,
            receipt.covered_block_indices,
        )
        self.assertEqual(receipt.unplanned_block_indices, ())
        self.assertTrue(receipt.prepared_depth_accessed)
        self.assertEqual(context_snapshot(context), before_context)
        self.assertEqual(after_tree, before_tree)

    def test_missing_prepared_inputs_return_stable_zero_ray_receipts(self) -> None:
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
                    before_context = context_snapshot(context)

                    receipt = trace_tsdf_observation_block_rays_from_context(
                        plan,
                        context,
                        0,
                    )

                    self.assertIs(receipt.observation_status, status)
                    self.assertEqual(receipt.camera_origin_world_m, origin)
                    self.assertEqual(receipt.image_size, (2, 2))
                    self.assertEqual(receipt.ray_receipts, ())
                    self.assertEqual(receipt.covered_block_indices, ())
                    self.assertEqual(receipt.existing_plan_block_indices, ())
                    self.assertEqual(receipt.unplanned_block_indices, ())
                    self.assertEqual(receipt.pixel_count, 0)
                    self.assertEqual(receipt.traversed_ray_count, 0)
                    self.assertEqual(receipt.invalid_depth_count, 0)
                    self.assertEqual(receipt.block_visit_count, 0)
                    self.assertEqual(receipt.duplicate_block_visit_count, 0)
                    self.assertEqual(receipt.retained_outcome_count, 0)
                    self.assertEqual(receipt.maximum_blocks_per_ray, 0)
                    self.assertEqual(receipt.surface_block_indices, ())
                    self.assertEqual(receipt.nonterminal_block_indices, ())
                    self.assertFalse(receipt.prepared_depth_accessed)
                    self.assertEqual(context_snapshot(context), before_context)
                    self.assertEqual(tree_snapshot(temporary_root), before_tree)

    def test_selection_types_provenance_and_geometry_are_preflighted(self) -> None:
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
            for name, candidate_plan, candidate_context, sequence, message in cases:
                with self.subTest(name=name):
                    with self.assertRaises(TsdfError) as raised:
                        trace_tsdf_observation_block_rays_from_context(
                            candidate_plan,  # type: ignore[arg-type]
                            candidate_context,  # type: ignore[arg-type]
                            sequence,  # type: ignore[arg-type]
                        )
                    self.assertIn(message, str(raised.exception))
            after_tree = tree_snapshot(temporary_root)

        self.assertEqual(after_tree, before_tree)

    def test_outcome_cap_uses_a_conservative_bound_and_is_atomic(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            plan, context = load_case(temporary_root)
            before_tree = tree_snapshot(temporary_root)
            before_plan = plan_snapshot(plan)
            before_context = context_snapshot(context)

            self.assertEqual(MAX_TSDF_OBSERVATION_BLOCK_RAY_OUTCOMES, 262_144)
            with patch(
                "spatialforge.tsdf_observation_block_rays."
                "MAX_TSDF_OBSERVATION_BLOCK_RAY_OUTCOMES",
                14,
            ):
                with self.assertRaises(TsdfError) as raised:
                    trace_tsdf_observation_block_rays_from_context(
                        plan,
                        context,
                        0,
                    )
            with patch(
                "spatialforge.tsdf_observation_block_rays."
                "MAX_TSDF_OBSERVATION_BLOCK_RAY_OUTCOMES",
                16,
            ):
                receipt = trace_tsdf_observation_block_rays_from_context(
                    plan,
                    context,
                    0,
                )
            after_tree = tree_snapshot(temporary_root)

        self.assertIn(
            "conservative retained-outcome bound reaches 16",
            str(raised.exception),
        )
        self.assertIn("maximum is 14", str(raised.exception))
        self.assertEqual(receipt.retained_outcome_count, 15)
        self.assertEqual(plan_snapshot(plan), before_plan)
        self.assertEqual(context_snapshot(context), before_context)
        self.assertEqual(after_tree, before_tree)

    def test_receipts_are_frozen_slotted_and_strict(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            plan, context = load_case(Path(temporary_directory))
            receipt = trace_tsdf_observation_block_rays_from_context(
                plan,
                context,
                0,
            )

        self.assertFalse(hasattr(receipt, "__dict__"))
        self.assertFalse(hasattr(receipt.ray_receipts[0], "__dict__"))
        with self.assertRaises(FrozenInstanceError):
            receipt.observation_sequence = 1  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            receipt.ray_receipts[0].pixel_uv = (1, 1)  # type: ignore[misc]

        with self.assertRaises(TsdfError):
            replace(
                receipt.ray_receipts[0],
                status=TsdfObservationBlockRayStatus.DEPTH_INVALID,
            )
        with self.assertRaises(TsdfError):
            replace(
                receipt.ray_receipts[0],
                block_indices=((0, 0, 0), (99, 0, 0), (1, 0, 0)),
            )
        with self.assertRaises(TsdfError):
            replace(
                receipt.ray_receipts[0],
                block_extent_m=10**10_000,
            )
        for arguments in (
            {"ray_receipts": tuple(reversed(receipt.ray_receipts))},
            {"observation_sequence": 999},
            {"covered_block_indices": receipt.covered_block_indices[:-1]},
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
            {"block_resolution": 7},
            {"block_extent_m": 0.0},
            {"image_size": (2, 3)},
        ):
            with self.subTest(arguments=arguments):
                with self.assertRaises(TsdfError):
                    replace(receipt, **arguments)

    def test_late_receipt_failure_leaves_inputs_and_files_unchanged(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            plan, context = load_case(temporary_root)
            before_tree = tree_snapshot(temporary_root)
            before_plan = plan_snapshot(plan)
            before_context = context_snapshot(context)
            injected = RuntimeError("injected final ray receipt failure")

            with patch(
                "spatialforge.tsdf_observation_block_rays."
                "TsdfObservationBlockRayTraceReceipt",
                side_effect=injected,
            ):
                with self.assertRaises(TsdfError) as raised:
                    trace_tsdf_observation_block_rays_from_context(
                        plan,
                        context,
                        0,
                    )
            after_tree = tree_snapshot(temporary_root)

        self.assertIs(raised.exception.__cause__, injected)
        self.assertIn("injected final ray receipt failure", str(raised.exception))
        self.assertEqual(plan_snapshot(plan), before_plan)
        self.assertEqual(context_snapshot(context), before_context)
        self.assertEqual(after_tree, before_tree)


class TsdfObservationBlockRayCliTests(unittest.TestCase):
    def test_cli_reports_exact_read_only_thin_ray_checkpoint(self) -> None:
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
            actual_trace = trace_tsdf_observation_block_rays_from_context

            with (
                forbidden_cli_calls() as forbidden,
                patch(
                    "spatialforge.cli."
                    "trace_tsdf_observation_block_rays_from_context",
                    wraps=actual_trace,
                ) as tracer,
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                exit_code = main(
                    [
                        "reconstruct",
                        "tsdf-block-context-observation-rays",
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
        tracer.assert_called_once()
        for forbidden_call in forbidden:
            forbidden_call.assert_not_called()

        output = stdout.getvalue()
        for expected in (
            "TSDF BLOCK CONTEXT OBSERVATION RAY TRACE CHECK "
            "scan-synthetic-0001\n",
            "observation: sequence=0 status=ready\n",
            "camera_origin_world_m: (0.000000000, 0.000000000, "
            "0.000000000)\n",
            "image: width=2 height=2\n",
            "pixel_outcomes: total=4 traversed=4 depth_invalid=0\n",
            "ray_block_visits: total=11 unique=8 duplicate=3 "
            "maximum_per_ray=3\n",
            "coverage_blocks: total=8 nonterminal=4 surface_endpoint=4\n",
            "coverage_partition: existing_plan=8 unplanned=0\n",
            "trace_workload: retained_outcomes=15 maximum=262144\n",
            "visibility_rule: positive-finite-depth-stops-at-measured-"
            "surface\n",
            "block_traversal_rule: closed-half-open-grid-thin-dda-"
            "simultaneous-exact-ties\n",
            "trace_prepared_depth_access: yes\n",
            "storage_allocated: no\n",
            "storage_mutated: no\n",
            "full_fusion_performed: no\n",
            "conservative_nearest_pixel_free_space_coverage_proven: no\n",
            "artifact_written: no\n",
            f"plan_sha256: {PLAN_SHA256}\n",
            f"replay_digest_sha256: {REPLAY_SHA256}\n",
        ):
            self.assertIn(expected, output)

    def test_cli_reports_trace_failure_without_writing_or_traceback(self) -> None:
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
                    "trace_tsdf_observation_block_rays_from_context",
                    side_effect=TsdfError("injected observation ray failure"),
                ) as tracer,
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                exit_code = main(
                    [
                        "reconstruct",
                        "tsdf-block-context-observation-rays",
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
        tracer.assert_called_once()
        error = stderr.getvalue()
        self.assertIn("TSDF BLOCK CONTEXT OBSERVATION RAY TRACE FAILED", error)
        self.assertIn("injected observation ray failure", error)
        self.assertNotIn("Traceback", error)


if __name__ == "__main__":
    unittest.main()
