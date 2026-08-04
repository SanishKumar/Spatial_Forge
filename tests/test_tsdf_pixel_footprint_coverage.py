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
from spatialforge.point_cloud import _transform_point
from spatialforge.session_loader import load_scan_session
from spatialforge.tsdf_block_plan import (
    _containing_block_index,
    plan_tsdf_blocks,
)
from spatialforge.tsdf_block_plan_loader import TsdfBlockPlan
from spatialforge.tsdf_observation_block_rays import (
    trace_tsdf_observation_block_rays_from_context,
)
from spatialforge.tsdf_pixel_footprint_coverage import (
    MAX_TSDF_PIXEL_FOOTPRINT_CANDIDATE_BLOCKS,
    TsdfPixelFootprintCoverageReceipt,
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

_FORBIDDEN_COVERAGE_TARGETS = (
    "spatialforge.tsdf_pixel_footprint_coverage.replay_session",
    "spatialforge.tsdf_pixel_footprint_coverage."
    "build_tsdf_replay_depth_context",
    "spatialforge.tsdf_pixel_footprint_coverage.allocate_empty_tsdf_blocks",
    "spatialforge.tsdf_pixel_footprint_coverage.locate_tsdf_voxel",
    "spatialforge.tsdf_pixel_footprint_coverage."
    "evaluate_tsdf_voxel_contribution",
    "spatialforge.tsdf_pixel_footprint_coverage."
    "evaluate_tsdf_voxel_contribution_from_context",
    "spatialforge.tsdf_pixel_footprint_coverage."
    "apply_tsdf_voxel_contribution_from_context",
    "spatialforge.tsdf_pixel_footprint_coverage."
    "traverse_tsdf_plan_blocks_from_context",
    "spatialforge.tsdf_pixel_footprint_coverage."
    "survey_tsdf_plan_block_rays_from_context",
    "spatialforge.replay._file_digest",
    "pathlib.Path.open",
    "PIL.Image.open",
)

_FORBIDDEN_CLI_TARGETS = (
    "spatialforge.cli.allocate_empty_tsdf_blocks",
    "spatialforge.cli.locate_tsdf_voxel",
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


def sampled_wedge_blocks(
    context: TsdfReplayDepthContext,
    observation_sequence: int,
    pixel_uv: tuple[int, int],
    measured_depth_m: float,
    block_extent_m: float,
    *,
    steps: int = 9,
) -> set[tuple[int, int, int]]:
    """Own the blocks of a deterministic lattice inside the sampling wedge.

    Every sampled point projects into ``pixel_uv`` under the evaluator's
    ``floor(projected + 0.5)`` rule and lies at a positive camera depth no
    greater than the measurement, so a conservative coverage rule must
    contain every block that owns one.
    """

    camera = context.camera
    observation = context.observations[observation_sequence]
    transform = observation.t_world_camera
    assert transform is not None
    blocks: set[tuple[int, int, int]] = set()
    for depth_step in range(1, steps + 1):
        z_camera = measured_depth_m * depth_step / steps
        for u_step in range(steps):
            for v_step in range(steps):
                offset_u = -0.5 + u_step / steps
                offset_v = -0.5 + v_step / steps
                x_camera = (
                    (pixel_uv[0] + offset_u - camera.cx) * z_camera / camera.fx
                )
                y_camera = (
                    (pixel_uv[1] + offset_v - camera.cy) * z_camera / camera.fy
                )
                world = _transform_point(
                    transform,
                    x_camera,
                    y_camera,
                    z_camera,
                )
                blocks.add(
                    tuple(
                        _containing_block_index(component, block_extent_m)
                        for component in world
                    )
                )
    return blocks


@contextmanager
def forbidden_coverage_calls():
    with ExitStack() as stack:
        mocks: list[MagicMock] = []
        for target in _FORBIDDEN_COVERAGE_TARGETS:
            mocks.append(stack.enter_context(patch(target, create=True)))
        yield mocks


@contextmanager
def forbidden_cli_calls():
    with ExitStack() as stack:
        mocks: list[MagicMock] = []
        for target in _FORBIDDEN_CLI_TARGETS:
            mocks.append(stack.enter_context(patch(target, create=True)))
        yield mocks


class TsdfPixelFootprintCoverageTests(unittest.TestCase):
    def test_fixture_pixel_has_exact_wedge_geometry_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            plan, context = load_case(temporary_root)
            before_tree = tree_snapshot(temporary_root)
            before_plan = plan_snapshot(plan)
            before_context = context_snapshot(context)

            with forbidden_coverage_calls() as forbidden:
                receipt = evaluate_tsdf_pixel_footprint_coverage_from_context(
                    plan,
                    context,
                    0,
                    (1, 1),
                )

            repeated = evaluate_tsdf_pixel_footprint_coverage_from_context(
                plan,
                context,
                0,
                (1, 1),
            )
            after_tree = tree_snapshot(temporary_root)

        self.assertIsInstance(receipt, TsdfPixelFootprintCoverageReceipt)
        self.assertEqual(receipt, repeated)
        self.assertEqual(receipt.source_plan_digest_sha256, PLAN_SHA256)
        self.assertEqual(receipt.replay_digest_sha256, REPLAY_SHA256)
        self.assertIs(receipt.status, TsdfPixelFootprintStatus.COVERED)
        self.assertIs(
            receipt.observation_status,
            TsdfReplayDepthStatus.READY,
        )
        self.assertEqual(receipt.pixel_uv, (1, 1))
        self.assertEqual(receipt.image_size, (2, 2))
        self.assertEqual(receipt.measured_depth_m, 1.0)
        self.assertEqual(receipt.camera_origin_world_m, (0.0, 0.0, 0.0))
        self.assertEqual(receipt.block_extent_m, 1.0)
        self.assertEqual(
            receipt.footprint_corners_world_m,
            (
                (1.0, 0.0, 0.0),
                (1.0, -0.5, 0.0),
                (1.0, -0.5, -0.5),
                (1.0, 0.0, -0.5),
            ),
        )
        self.assertEqual(receipt.candidate_min_block_index, (0, -1, -1))
        self.assertEqual(receipt.candidate_max_block_index, (1, 0, 0))
        self.assertEqual(receipt.candidate_block_count, 8)
        self.assertEqual(
            receipt.centerline_block_indices,
            ((0, 0, 0), (0, -1, -1), (1, -1, -1)),
        )
        self.assertEqual(receipt.covered_block_indices, EXPECTED_ACTIVE_BLOCKS)
        self.assertEqual(receipt.covered_block_count, 8)
        self.assertEqual(receipt.rejected_candidate_count, 0)
        self.assertEqual(
            receipt.existing_plan_block_indices,
            EXPECTED_ACTIVE_BLOCKS,
        )
        self.assertEqual(receipt.unplanned_block_indices, ())
        self.assertTrue(receipt.prepared_depth_accessed)
        self.assertEqual(plan_snapshot(plan), before_plan)
        self.assertEqual(context_snapshot(context), before_context)
        self.assertEqual(after_tree, before_tree)
        for forbidden_call in forbidden:
            forbidden_call.assert_not_called()

    def test_coverage_contains_every_sampled_wedge_point_block(self) -> None:
        cases = ((1000, "near"), (3000, "far"))
        for raw_depth, name in cases:
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

                    for pixel_uv in ((0, 0), (1, 0), (0, 1), (1, 1)):
                        receipt = (
                            evaluate_tsdf_pixel_footprint_coverage_from_context(
                                plan,
                                context,
                                0,
                                pixel_uv,
                            )
                        )
                        self.assertIsNotNone(receipt.measured_depth_m)
                        assert receipt.measured_depth_m is not None
                        sampled = sampled_wedge_blocks(
                            context,
                            0,
                            pixel_uv,
                            receipt.measured_depth_m,
                            receipt.block_extent_m,
                        )
                        covered = set(receipt.covered_block_indices)
                        self.assertTrue(
                            sampled <= covered,
                            f"{pixel_uv} missed {sorted(sampled - covered)}",
                        )
                        self.assertTrue(
                            set(receipt.centerline_block_indices) <= covered
                        )

    def test_footprint_is_a_strict_superset_of_the_centerline(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            plan, context = load_case(temporary_root)
            trace = trace_tsdf_observation_block_rays_from_context(
                plan,
                context,
                0,
            )
            widened = 0
            for ray in trace.ray_receipts:
                receipt = evaluate_tsdf_pixel_footprint_coverage_from_context(
                    plan,
                    context,
                    0,
                    ray.pixel_uv,
                )
                self.assertEqual(
                    receipt.centerline_block_indices,
                    ray.block_indices,
                )
                self.assertTrue(
                    set(ray.block_indices)
                    <= set(receipt.covered_block_indices)
                )
                if receipt.widens_centerline_coverage:
                    widened += 1
                    self.assertTrue(receipt.footprint_only_block_indices)

        self.assertEqual(widened, 3)

    def test_far_measurement_rejects_candidates_and_reports_unplanned(
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

            receipt = evaluate_tsdf_pixel_footprint_coverage_from_context(
                plan,
                context,
                0,
                (1, 1),
            )
            after_tree = tree_snapshot(temporary_root)

        self.assertEqual(receipt.measured_depth_m, 3.0)
        self.assertEqual(receipt.candidate_min_block_index, (0, -2, -2))
        self.assertEqual(receipt.candidate_max_block_index, (3, 0, 0))
        self.assertEqual(receipt.candidate_block_count, 36)
        self.assertEqual(receipt.covered_block_count, 31)
        self.assertEqual(receipt.rejected_candidate_count, 5)
        for rejected in (
            (0, -2, -2),
            (0, -1, -2),
            (0, 0, -2),
            (0, -2, -1),
            (0, -2, 0),
        ):
            self.assertNotIn(rejected, receipt.covered_block_indices)
        self.assertEqual(len(receipt.existing_plan_block_indices), 18)
        self.assertEqual(len(receipt.unplanned_block_indices), 13)
        self.assertEqual(
            set(receipt.unplanned_block_indices) & set(plan.active_blocks),
            set(),
        )
        self.assertEqual(receipt.centerline_block_count, 5)
        self.assertEqual(len(receipt.footprint_only_block_indices), 26)
        self.assertEqual(plan_snapshot(plan), before_plan)
        self.assertEqual(after_tree, before_tree)

    def test_invalid_depth_and_missing_inputs_retain_no_geometry(self) -> None:
        cases = (
            (
                "depth-invalid",
                lambda path: set_first_depth_sample_invalid(path),
                TsdfReplayDepthStatus.READY,
                TsdfPixelFootprintStatus.DEPTH_INVALID,
                (0.0, 0.0, 0.0),
            ),
            (
                "missing-depth",
                lambda path: remove_first_record(
                    path / "streams" / "depth.jsonl"
                ),
                TsdfReplayDepthStatus.MISSING_DEPTH,
                TsdfPixelFootprintStatus.MISSING_DEPTH,
                (0.0, 0.0, 0.0),
            ),
            (
                "missing-pose",
                lambda path: remove_first_record(
                    path / "streams" / "poses.jsonl"
                ),
                TsdfReplayDepthStatus.MISSING_POSE,
                TsdfPixelFootprintStatus.MISSING_POSE,
                None,
            ),
        )
        for name, mutate, observation_status, status, origin in cases:
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory(
                    dir=TEST_ROOT
                ) as temporary_directory:
                    temporary_root = Path(temporary_directory)
                    session_path = copy_fixture(temporary_root)
                    mutate(session_path)
                    plan, context = load_case(
                        temporary_root,
                        session_path=session_path,
                    )
                    before_tree = tree_snapshot(temporary_root)
                    before_context = context_snapshot(context)

                    receipt = (
                        evaluate_tsdf_pixel_footprint_coverage_from_context(
                            plan,
                            context,
                            0,
                            (0, 0),
                        )
                    )

                    self.assertIs(receipt.status, status)
                    self.assertIs(
                        receipt.observation_status,
                        observation_status,
                    )
                    self.assertEqual(receipt.camera_origin_world_m, origin)
                    self.assertFalse(receipt.covered)
                    self.assertIsNone(receipt.measured_depth_m)
                    self.assertEqual(receipt.footprint_corners_world_m, ())
                    self.assertEqual(receipt.centerline_block_indices, ())
                    self.assertIsNone(receipt.candidate_min_block_index)
                    self.assertIsNone(receipt.candidate_max_block_index)
                    self.assertEqual(receipt.candidate_block_count, 0)
                    self.assertEqual(receipt.covered_block_indices, ())
                    self.assertEqual(receipt.existing_plan_block_indices, ())
                    self.assertEqual(receipt.unplanned_block_indices, ())
                    self.assertFalse(receipt.widens_centerline_coverage)
                    self.assertEqual(
                        context_snapshot(context),
                        before_context,
                    )
                    self.assertEqual(tree_snapshot(temporary_root), before_tree)

    def test_selection_pixel_and_provenance_are_preflighted(self) -> None:
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
                ("plan-type", object(), context, 0, (0, 0), "TsdfBlockPlan"),
                (
                    "context-type",
                    plan,
                    object(),
                    0,
                    (0, 0),
                    "TsdfReplayDepthContext",
                ),
                ("bool", plan, context, True, (0, 0), "sequence"),
                ("negative", plan, context, -1, (0, 0), "sequence"),
                ("out-of-range", plan, context, 2, (0, 0), "sequence"),
                (
                    "unselected",
                    stride_plan,
                    stride_context,
                    1,
                    (0, 0),
                    "selected",
                ),
                ("pixel-type", plan, context, 0, (0,), "pixel"),
                ("pixel-bool", plan, context, 0, (True, 0), "pixel"),
                ("pixel-negative", plan, context, 0, (-1, 0), "pixel"),
                ("pixel-outside", plan, context, 0, (2, 0), "outside"),
                (
                    "digest",
                    plan,
                    replace(context, source_plan_digest_sha256="0" * 64),
                    0,
                    (0, 0),
                    "source plan digest",
                ),
                (
                    "extent",
                    replace(plan, block_extent_m=2.0),
                    context,
                    0,
                    (0, 0),
                    "block_extent_m",
                ),
            )
            for name, candidate_plan, candidate_context, sequence, pixel, message in (
                cases
            ):
                with self.subTest(name=name):
                    with self.assertRaises(TsdfError) as raised:
                        evaluate_tsdf_pixel_footprint_coverage_from_context(
                            candidate_plan,  # type: ignore[arg-type]
                            candidate_context,  # type: ignore[arg-type]
                            sequence,  # type: ignore[arg-type]
                            pixel,  # type: ignore[arg-type]
                        )
                    self.assertIn(message, str(raised.exception))
            after_tree = tree_snapshot(temporary_root)

        self.assertEqual(after_tree, before_tree)

    def test_candidate_cap_is_preflighted_and_atomic(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            plan, context = load_case(temporary_root)
            before_tree = tree_snapshot(temporary_root)
            before_plan = plan_snapshot(plan)

            self.assertEqual(
                MAX_TSDF_PIXEL_FOOTPRINT_CANDIDATE_BLOCKS,
                262_144,
            )
            with patch(
                "spatialforge.tsdf_pixel_footprint_coverage."
                "MAX_TSDF_PIXEL_FOOTPRINT_CANDIDATE_BLOCKS",
                7,
            ):
                with self.assertRaises(TsdfError) as raised:
                    evaluate_tsdf_pixel_footprint_coverage_from_context(
                        plan,
                        context,
                        0,
                        (1, 1),
                    )
            with patch(
                "spatialforge.tsdf_pixel_footprint_coverage."
                "MAX_TSDF_PIXEL_FOOTPRINT_CANDIDATE_BLOCKS",
                8,
            ):
                receipt = evaluate_tsdf_pixel_footprint_coverage_from_context(
                    plan,
                    context,
                    0,
                    (1, 1),
                )
            after_tree = tree_snapshot(temporary_root)

        self.assertIn("requires 8 candidate blocks", str(raised.exception))
        self.assertIn("maximum is 7", str(raised.exception))
        self.assertEqual(receipt.candidate_block_count, 8)
        self.assertEqual(plan_snapshot(plan), before_plan)
        self.assertEqual(after_tree, before_tree)

    def test_receipts_are_frozen_slotted_and_strict(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            plan, context = load_case(Path(temporary_directory))
            receipt = evaluate_tsdf_pixel_footprint_coverage_from_context(
                plan,
                context,
                0,
                (1, 1),
            )

        self.assertFalse(hasattr(receipt, "__dict__"))
        with self.assertRaises(FrozenInstanceError):
            receipt.pixel_uv = (0, 0)  # type: ignore[misc]

        # measured_depth_m is a retained metre measurement, not a re-derivable
        # quantity: the receipt keeps no intrinsics or pose, so only its sign
        # and finiteness are checkable. Reversing the corner tuple likewise
        # describes the identical wedge. Both match the one-observation ray
        # receipt's existing contract.
        self.assertEqual(
            replace(
                receipt,
                footprint_corners_world_m=tuple(
                    reversed(receipt.footprint_corners_world_m)
                ),
            ).covered_block_indices,
            receipt.covered_block_indices,
        )

        for arguments in (
            {"covered_block_indices": receipt.covered_block_indices[:-1]},
            {
                "covered_block_indices": tuple(
                    reversed(receipt.covered_block_indices)
                )
            },
            {"centerline_block_indices": ((9, 9, 9),)},
            {"candidate_min_block_index": (0, 0, 0)},
            {"candidate_max_block_index": (2, 0, 0)},
            {
                "footprint_corners_world_m": receipt.footprint_corners_world_m[
                    :3
                ]
            },
            {
                "footprint_corners_world_m": tuple(
                    (corner[0] + 5.0, corner[1], corner[2])
                    for corner in receipt.footprint_corners_world_m
                )
            },
            {"camera_origin_world_m": (0.0, 0.0, 5.0)},
            {"measured_depth_m": None},
            {"measured_depth_m": 0.0},
            {
                "existing_plan_block_indices": (),
                "unplanned_block_indices": (),
            },
            {"status": TsdfPixelFootprintStatus.DEPTH_INVALID},
            {"observation_status": TsdfReplayDepthStatus.MISSING_POSE},
            {"source_plan_block_indices": ()},
            {"source_plan_digest_sha256": "0"},
            {"block_resolution": 7},
            {"block_extent_m": 0.0},
            {"pixel_uv": (5, 5)},
            {"observation_sequence": 999},
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
            injected = RuntimeError("injected final footprint receipt failure")

            with patch(
                "spatialforge.tsdf_pixel_footprint_coverage."
                "TsdfPixelFootprintCoverageReceipt",
                side_effect=injected,
            ):
                with self.assertRaises(TsdfError) as raised:
                    evaluate_tsdf_pixel_footprint_coverage_from_context(
                        plan,
                        context,
                        0,
                        (1, 1),
                    )
            after_tree = tree_snapshot(temporary_root)

        self.assertIs(raised.exception.__cause__, injected)
        self.assertIn(
            "injected final footprint receipt failure",
            str(raised.exception),
        )
        self.assertEqual(plan_snapshot(plan), before_plan)
        self.assertEqual(context_snapshot(context), before_context)
        self.assertEqual(after_tree, before_tree)


class TsdfPixelFootprintCoverageCliTests(unittest.TestCase):
    def test_cli_reports_exact_read_only_footprint_checkpoint(self) -> None:
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
            actual = evaluate_tsdf_pixel_footprint_coverage_from_context

            with (
                forbidden_cli_calls() as forbidden,
                patch(
                    "spatialforge.cli."
                    "evaluate_tsdf_pixel_footprint_coverage_from_context",
                    wraps=actual,
                ) as coverage,
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                exit_code = main(
                    [
                        "reconstruct",
                        "tsdf-block-context-pixel-footprint",
                        str(plan_path),
                        str(session_path),
                        "--observation-sequence",
                        "0",
                        "--pixel",
                        "1",
                        "1",
                    ]
                )
            after_tree = tree_snapshot(temporary_root)

        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(after_tree, before_tree)
        coverage.assert_called_once()
        for forbidden_call in forbidden:
            forbidden_call.assert_not_called()

        output = stdout.getvalue()
        for expected in (
            "TSDF BLOCK CONTEXT PIXEL FOOTPRINT CHECK scan-synthetic-0001\n",
            "observation: sequence=0 status=ready\n",
            "pixel: uv=(1, 1) image=2x2\n",
            "footprint_status: covered\n",
            "measured_depth_m: 1.000000000\n",
            "sampling_rule: nearest-pixel-half-open-unit-square\n",
            "wedge_rule: apex-to-measured-depth-convex-pyramid\n",
            "coverage_rule: conservative-plane-superset-of-half-open-cells\n",
            "first_footprint_corner_world_m: (1.000000000, 0.000000000, "
            "0.000000000)\n",
            "candidate_blocks: total=8 min=(0, -1, -1) max=(1, 0, 0)\n",
            "coverage_blocks: covered=8 rejected=0\n",
            "centerline_blocks: total=3 footprint_only=5\n",
            "centerline_contained_in_coverage: yes\n",
            "widens_centerline_coverage: yes\n",
            "coverage_partition: existing_plan=8 unplanned=0\n",
            "coverage_workload: candidate_blocks=8 maximum=262144\n",
            "coverage_scope: one-prepared-pixel-only\n",
            "conservative_nearest_pixel_footprint_coverage_computed: yes\n",
            "per_voxel_sampling_proof_computed: no\n",
            "occlusion_rule_defined: no\n",
            "visibility_culling_rule_defined: no\n",
            "multi_pixel_coverage_computed: no\n",
            "multi_observation_coverage_computed: no\n",
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
                    "evaluate_tsdf_pixel_footprint_coverage_from_context",
                    side_effect=TsdfError("injected pixel footprint failure"),
                ) as coverage,
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                exit_code = main(
                    [
                        "reconstruct",
                        "tsdf-block-context-pixel-footprint",
                        str(plan_path),
                        str(session_path),
                        "--observation-sequence",
                        "0",
                        "--pixel",
                        "0",
                        "0",
                    ]
                )
            after_tree = tree_snapshot(temporary_root)

        self.assertEqual(exit_code, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(after_tree, before_tree)
        coverage.assert_called_once()
        error = stderr.getvalue()
        self.assertIn("TSDF BLOCK CONTEXT PIXEL FOOTPRINT FAILED", error)
        self.assertIn("injected pixel footprint failure", error)
        self.assertNotIn("Traceback", error)


if __name__ == "__main__":
    unittest.main()
