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
from spatialforge.tsdf_pixel_footprint_coverage import (
    evaluate_tsdf_pixel_footprint_coverage_from_context,
)
from spatialforge.tsdf_replay_depth_context import (
    TsdfReplayDepthContext,
    TsdfReplayDepthStatus,
)
from spatialforge.tsdf_voxel_address import (
    compose_tsdf_global_voxel_index,
    locate_tsdf_voxel,
)
from spatialforge.tsdf_voxel_contribution import (
    TsdfContributionStatus,
    evaluate_tsdf_voxel_contribution_from_context,
)
from spatialforge.tsdf_voxel_sampling import (
    TsdfVoxelSamplingReceipt,
    TsdfVoxelSamplingStatus,
    classify_tsdf_voxel_sampling_from_context,
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

_EQUIVALENT_STATUSES = {
    TsdfContributionStatus.CONTRIBUTES: (
        TsdfVoxelSamplingStatus.OBSERVED_FREE_SPACE,
        TsdfVoxelSamplingStatus.OBSERVED_SURFACE_BAND,
    ),
    TsdfContributionStatus.BEHIND_TRUNCATION: (
        TsdfVoxelSamplingStatus.UNOBSERVED_OCCLUDED,
    ),
    TsdfContributionStatus.CAMERA_Z_NONPOSITIVE: (
        TsdfVoxelSamplingStatus.UNOBSERVED_BEHIND_CAMERA,
    ),
    TsdfContributionStatus.PROJECTION_OUTSIDE_IMAGE: (
        TsdfVoxelSamplingStatus.UNOBSERVED_OUTSIDE_IMAGE,
    ),
    TsdfContributionStatus.DEPTH_INVALID: (
        TsdfVoxelSamplingStatus.UNOBSERVED_DEPTH_INVALID,
    ),
    TsdfContributionStatus.CAMERA_POINT_NONFINITE: (
        TsdfVoxelSamplingStatus.UNOBSERVED_NONFINITE,
    ),
    TsdfContributionStatus.PROJECTION_NONFINITE: (
        TsdfVoxelSamplingStatus.UNOBSERVED_NONFINITE,
    ),
    TsdfContributionStatus.SIGNED_DISTANCE_NONFINITE: (
        TsdfVoxelSamplingStatus.UNOBSERVED_NONFINITE,
    ),
    TsdfContributionStatus.MISSING_DEPTH: (
        TsdfVoxelSamplingStatus.MISSING_DEPTH,
    ),
    TsdfContributionStatus.MISSING_POSE: (
        TsdfVoxelSamplingStatus.MISSING_POSE,
    ),
    TsdfContributionStatus.MISSING_DEPTH_AND_POSE: (
        TsdfVoxelSamplingStatus.MISSING_DEPTH_AND_POSE,
    ),
}

_FORBIDDEN_SAMPLING_TARGETS = (
    "spatialforge.tsdf_voxel_sampling.replay_session",
    "spatialforge.tsdf_voxel_sampling.build_tsdf_replay_depth_context",
    "spatialforge.tsdf_voxel_sampling.allocate_empty_tsdf_blocks",
    "spatialforge.tsdf_voxel_sampling.locate_tsdf_voxel",
    "spatialforge.tsdf_voxel_sampling.evaluate_tsdf_voxel_contribution",
    "spatialforge.tsdf_voxel_sampling."
    "evaluate_tsdf_voxel_contribution_from_context",
    "spatialforge.tsdf_voxel_sampling."
    "apply_tsdf_voxel_contribution_from_context",
    "spatialforge.tsdf_voxel_sampling."
    "traverse_tsdf_plan_blocks_from_context",
    "spatialforge.replay._file_digest",
    "pathlib.Path.open",
    "PIL.Image.open",
)

_FORBIDDEN_CLI_TARGETS = (
    "spatialforge.cli.allocate_empty_tsdf_blocks",
    "spatialforge.cli.locate_tsdf_voxel",
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
                else (depth.shape, depth.dtype.str, depth.tobytes()),
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
    )


def remove_first_record(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text(lines[1] + "\n", encoding="utf-8")


def set_first_depth_sample_invalid(session_path: Path) -> None:
    (session_path / "data" / "depth" / "000000.pgm").write_text(
        "P2\n2 2\n65535\n0 0\n0 0\n",
        encoding="ascii",
    )


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
def forbidden_sampling_calls():
    with ExitStack() as stack:
        mocks: list[MagicMock] = []
        for target in _FORBIDDEN_SAMPLING_TARGETS:
            mocks.append(stack.enter_context(patch(target, create=True)))
        yield mocks


@contextmanager
def forbidden_cli_calls():
    with ExitStack() as stack:
        mocks: list[MagicMock] = []
        for target in _FORBIDDEN_CLI_TARGETS:
            mocks.append(stack.enter_context(patch(target, create=True)))
        yield mocks


class TsdfVoxelSamplingTests(unittest.TestCase):
    def test_fixture_voxels_split_free_space_band_and_occluded(self) -> None:
        cases = (
            (
                (8, -1, -1),
                TsdfVoxelSamplingStatus.OBSERVED_SURFACE_BAND,
                (0.0625, 0.0625, 1.0625),
                (1, 1),
                -0.0625,
                -0.125,
                True,
                False,
            ),
            (
                (2, 0, 0),
                TsdfVoxelSamplingStatus.OBSERVED_FREE_SPACE,
                (-0.0625, -0.0625, 0.3125),
                (0, 0),
                0.6875,
                1.0,
                True,
                True,
            ),
            (
                (60, 0, 0),
                TsdfVoxelSamplingStatus.UNOBSERVED_OCCLUDED,
                (-0.0625, -0.0625, 7.5625),
                (0, 0),
                -6.5625,
                -1.0,
                False,
                False,
            ),
        )
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            plan, context = load_case(temporary_root)
            before_tree = tree_snapshot(temporary_root)
            before_plan = plan_snapshot(plan)
            before_context = context_snapshot(context)

            for (
                global_index,
                status,
                camera_xyz,
                pixel_uv,
                signed,
                value,
                planned,
                wedge,
            ) in cases:
                with self.subTest(voxel=global_index):
                    with forbidden_sampling_calls() as forbidden:
                        receipt = classify_tsdf_voxel_sampling_from_context(
                            plan,
                            context,
                            0,
                            global_index,
                        )
                    repeated = classify_tsdf_voxel_sampling_from_context(
                        plan,
                        context,
                        0,
                        global_index,
                    )

                    self.assertIsInstance(receipt, TsdfVoxelSamplingReceipt)
                    self.assertEqual(receipt, repeated)
                    self.assertIs(receipt.status, status)
                    self.assertEqual(receipt.global_index_xyz, global_index)
                    self.assertEqual(receipt.camera_xyz_m, camera_xyz)
                    self.assertEqual(receipt.signed_distance_m, signed)
                    self.assertEqual(receipt.truncated_tsdf_value, value)
                    self.assertEqual(receipt.planned_block, planned)
                    self.assertEqual(receipt.inside_sampling_wedge, wedge)
                    self.assertEqual(receipt.pixel_uv, pixel_uv)
                    self.assertEqual(receipt.measured_depth_m, 1.0)
                    self.assertEqual(receipt.voxel_size_m, 0.125)
                    self.assertEqual(receipt.truncation_m, 0.5)
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
        self.assertEqual(context_snapshot(context), before_context)
        self.assertEqual(after_tree, before_tree)

    def test_classification_agrees_with_the_reference_evaluator(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            plan, context = load_case(temporary_root)
            session = load_scan_session(FIXTURE)
            storage = allocate_empty_tsdf_blocks(plan, session)
            seen: set[TsdfVoxelSamplingStatus] = set()
            compared = 0

            for observation_sequence in (0, 1):
                for global_index in planned_global_indices(plan, stride=2):
                    address = locate_tsdf_voxel(storage, global_index)
                    self.assertIsNotNone(address)
                    contribution = (
                        evaluate_tsdf_voxel_contribution_from_context(
                            storage,
                            address,
                            context,
                            observation_sequence,
                        )
                    )
                    receipt = classify_tsdf_voxel_sampling_from_context(
                        plan,
                        context,
                        observation_sequence,
                        global_index,
                    )
                    compared += 1
                    seen.add(receipt.status)

                    self.assertIn(
                        receipt.status,
                        _EQUIVALENT_STATUSES[contribution.status],
                        f"{global_index} at {observation_sequence}",
                    )
                    self.assertEqual(
                        receipt.world_xyz_m,
                        contribution.world_xyz_m,
                    )
                    self.assertEqual(
                        receipt.camera_xyz_m,
                        contribution.camera_xyz_m,
                    )
                    self.assertEqual(
                        receipt.projected_uv,
                        contribution.projected_uv,
                    )
                    self.assertEqual(receipt.pixel_uv, contribution.pixel_uv)
                    self.assertEqual(
                        receipt.measured_depth_m,
                        contribution.measured_depth_m,
                    )
                    self.assertEqual(
                        receipt.signed_distance_m,
                        contribution.signed_distance_m,
                    )
                    self.assertEqual(
                        receipt.contributes_to_reference_tsdf,
                        contribution.status
                        is TsdfContributionStatus.CONTRIBUTES,
                    )
                    if (
                        contribution.status
                        is TsdfContributionStatus.CONTRIBUTES
                    ):
                        self.assertEqual(
                            receipt.truncated_tsdf_value,
                            contribution.tsdf_sum_delta,
                        )

        self.assertGreater(compared, 500)
        self.assertIn(TsdfVoxelSamplingStatus.OBSERVED_FREE_SPACE, seen)
        self.assertIn(TsdfVoxelSamplingStatus.OBSERVED_SURFACE_BAND, seen)
        self.assertIn(TsdfVoxelSamplingStatus.UNOBSERVED_OCCLUDED, seen)

    def test_wedge_voxels_lie_in_their_pixel_footprint_coverage(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            plan, context = load_case(temporary_root)
            coverage_cache: dict[tuple[int, int], frozenset] = {}
            checked = 0

            for global_index in planned_global_indices(plan, stride=1):
                receipt = classify_tsdf_voxel_sampling_from_context(
                    plan,
                    context,
                    0,
                    global_index,
                )
                if not receipt.inside_sampling_wedge:
                    continue
                self.assertIsNotNone(receipt.pixel_uv)
                assert receipt.pixel_uv is not None
                if receipt.pixel_uv not in coverage_cache:
                    footprint = (
                        evaluate_tsdf_pixel_footprint_coverage_from_context(
                            plan,
                            context,
                            0,
                            receipt.pixel_uv,
                        )
                    )
                    coverage_cache[receipt.pixel_uv] = frozenset(
                        footprint.covered_block_indices
                    )
                checked += 1
                self.assertIn(
                    receipt.block_index_xyz,
                    coverage_cache[receipt.pixel_uv],
                    f"{global_index} escaped its pixel footprint",
                )

        self.assertGreater(checked, 100)
        self.assertEqual(len(coverage_cache), 4)

    def test_truncation_boundaries_belong_to_the_surface_band(self) -> None:
        from spatialforge.tsdf_voxel_sampling import (
            _classify_signed_distance,
        )

        truncation_m = 0.5
        cases = (
            (0.5, TsdfVoxelSamplingStatus.OBSERVED_SURFACE_BAND),
            (-0.5, TsdfVoxelSamplingStatus.OBSERVED_SURFACE_BAND),
            (0.0, TsdfVoxelSamplingStatus.OBSERVED_SURFACE_BAND),
            (0.5000000000000001, TsdfVoxelSamplingStatus.OBSERVED_FREE_SPACE),
            (
                -0.5000000000000001,
                TsdfVoxelSamplingStatus.UNOBSERVED_OCCLUDED,
            ),
        )
        for signed_distance_m, expected in cases:
            with self.subTest(signed_distance_m=signed_distance_m):
                self.assertIs(
                    _classify_signed_distance(signed_distance_m, truncation_m),
                    expected,
                )

    def test_unplanned_voxels_are_classified_without_touching_the_plan(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            plan, context = load_case(temporary_root)
            before_plan = plan_snapshot(plan)
            before_tree = tree_snapshot(temporary_root)

            unplanned = classify_tsdf_voxel_sampling_from_context(
                plan,
                context,
                0,
                (60, 0, 0),
            )
            negative = classify_tsdf_voxel_sampling_from_context(
                plan,
                context,
                0,
                (-40, -40, -40),
            )
            after_tree = tree_snapshot(temporary_root)

        self.assertFalse(unplanned.planned_block)
        self.assertEqual(unplanned.block_index_xyz, (7, 0, 0))
        self.assertNotIn((7, 0, 0), plan.active_blocks)
        self.assertFalse(negative.planned_block)
        self.assertEqual(negative.block_index_xyz, (-5, -5, -5))
        self.assertIs(
            negative.status,
            TsdfVoxelSamplingStatus.UNOBSERVED_BEHIND_CAMERA,
        )
        self.assertIsNone(negative.pixel_uv)
        self.assertIsNone(negative.measured_depth_m)
        self.assertEqual(plan_snapshot(plan), before_plan)
        self.assertEqual(after_tree, before_tree)

    def test_invalid_depth_and_missing_inputs_retain_no_measurement(
        self,
    ) -> None:
        cases = (
            (
                "depth-invalid",
                lambda path: set_first_depth_sample_invalid(path),
                TsdfReplayDepthStatus.READY,
                TsdfVoxelSamplingStatus.UNOBSERVED_DEPTH_INVALID,
                True,
            ),
            (
                "missing-depth",
                lambda path: remove_first_record(
                    path / "streams" / "depth.jsonl"
                ),
                TsdfReplayDepthStatus.MISSING_DEPTH,
                TsdfVoxelSamplingStatus.MISSING_DEPTH,
                False,
            ),
            (
                "missing-pose",
                lambda path: remove_first_record(
                    path / "streams" / "poses.jsonl"
                ),
                TsdfReplayDepthStatus.MISSING_POSE,
                TsdfVoxelSamplingStatus.MISSING_POSE,
                False,
            ),
        )
        for name, mutate, observation_status, status, keeps_camera in cases:
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

                    receipt = classify_tsdf_voxel_sampling_from_context(
                        plan,
                        context,
                        0,
                        (2, 0, 0),
                    )

                    self.assertIs(receipt.status, status)
                    self.assertIs(
                        receipt.observation_status,
                        observation_status,
                    )
                    self.assertFalse(receipt.observed)
                    self.assertFalse(receipt.inside_sampling_wedge)
                    self.assertIsNone(receipt.measured_depth_m)
                    self.assertIsNone(receipt.signed_distance_m)
                    self.assertIsNone(receipt.truncated_tsdf_value)
                    self.assertEqual(
                        receipt.camera_xyz_m is not None,
                        keeps_camera,
                    )
                    self.assertEqual(tree_snapshot(temporary_root), before_tree)

    def test_selection_voxel_and_provenance_are_preflighted(self) -> None:
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
                ("plan-type", object(), context, 0, (0, 0, 0), "TsdfBlockPlan"),
                (
                    "context-type",
                    plan,
                    object(),
                    0,
                    (0, 0, 0),
                    "TsdfReplayDepthContext",
                ),
                ("bool", plan, context, True, (0, 0, 0), "sequence"),
                ("negative", plan, context, -1, (0, 0, 0), "sequence"),
                ("out-of-range", plan, context, 2, (0, 0, 0), "sequence"),
                (
                    "unselected",
                    stride_plan,
                    stride_context,
                    1,
                    (0, 0, 0),
                    "selected",
                ),
                ("voxel-length", plan, context, 0, (0, 0), "global_index_xyz"),
                (
                    "voxel-bool",
                    plan,
                    context,
                    0,
                    (True, 0, 0),
                    "global_index_xyz",
                ),
                (
                    "voxel-range",
                    plan,
                    context,
                    0,
                    (10**12, 0, 0),
                    "global_index_xyz",
                ),
                (
                    "digest",
                    plan,
                    replace(context, source_plan_digest_sha256="0" * 64),
                    0,
                    (0, 0, 0),
                    "source plan digest",
                ),
                (
                    "extent",
                    replace(plan, block_extent_m=2.0),
                    context,
                    0,
                    (0, 0, 0),
                    "block_extent_m",
                ),
            )
            for name, case_plan, case_context, sequence, voxel, message in cases:
                with self.subTest(name=name):
                    with self.assertRaises(TsdfError) as raised:
                        classify_tsdf_voxel_sampling_from_context(
                            case_plan,  # type: ignore[arg-type]
                            case_context,  # type: ignore[arg-type]
                            sequence,  # type: ignore[arg-type]
                            voxel,  # type: ignore[arg-type]
                        )
                    self.assertIn(message, str(raised.exception))
            after_tree = tree_snapshot(temporary_root)

        self.assertEqual(after_tree, before_tree)

    def test_receipts_are_frozen_slotted_and_strict(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            plan, context = load_case(Path(temporary_directory))
            receipt = classify_tsdf_voxel_sampling_from_context(
                plan,
                context,
                0,
                (8, -1, -1),
            )

        self.assertFalse(hasattr(receipt, "__dict__"))
        with self.assertRaises(FrozenInstanceError):
            receipt.status = TsdfVoxelSamplingStatus.OBSERVED_FREE_SPACE  # type: ignore[misc]

        for arguments in (
            {"status": TsdfVoxelSamplingStatus.OBSERVED_FREE_SPACE},
            {"status": TsdfVoxelSamplingStatus.UNOBSERVED_OCCLUDED},
            {"status": TsdfVoxelSamplingStatus.MISSING_POSE},
            {"signed_distance_m": 0.25},
            {"signed_distance_m": None},
            {"truncated_tsdf_value": 0.5},
            {"truncated_tsdf_value": None},
            {"measured_depth_m": 2.0},
            {"measured_depth_m": None},
            {"pixel_uv": (0, 0)},
            {"pixel_uv": None},
            {"block_index_xyz": (0, 0, 0)},
            {"local_index_xyz": (0, 0, 0)},
            {"world_xyz_m": (0.0, 0.0, 0.0)},
            {"voxel_size_m": 0.25},
            {"truncation_m": 0.0},
            {"planned_block": 1},
            {"block_resolution": 7},
            {"observation_sequence": 999},
            {"source_plan_digest_sha256": "0"},
            {"observation_status": TsdfReplayDepthStatus.MISSING_DEPTH},
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
            injected = RuntimeError("injected final sampling receipt failure")

            with patch(
                "spatialforge.tsdf_voxel_sampling.TsdfVoxelSamplingReceipt",
                side_effect=injected,
            ):
                with self.assertRaises(TsdfError) as raised:
                    classify_tsdf_voxel_sampling_from_context(
                        plan,
                        context,
                        0,
                        (8, -1, -1),
                    )
            after_tree = tree_snapshot(temporary_root)

        self.assertIs(raised.exception.__cause__, injected)
        self.assertIn(
            "injected final sampling receipt failure",
            str(raised.exception),
        )
        self.assertEqual(plan_snapshot(plan), before_plan)
        self.assertEqual(context_snapshot(context), before_context)
        self.assertEqual(after_tree, before_tree)


class TsdfVoxelSamplingCliTests(unittest.TestCase):
    def test_cli_reports_exact_read_only_sampling_checkpoint(self) -> None:
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
            actual = classify_tsdf_voxel_sampling_from_context

            with (
                forbidden_cli_calls() as forbidden,
                patch(
                    "spatialforge.cli."
                    "classify_tsdf_voxel_sampling_from_context",
                    wraps=actual,
                ) as classifier,
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                exit_code = main(
                    [
                        "reconstruct",
                        "tsdf-block-context-voxel-sampling",
                        str(plan_path),
                        str(session_path),
                        "--observation-sequence",
                        "0",
                        "--voxel",
                        "2",
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
            "TSDF BLOCK CONTEXT VOXEL SAMPLING CHECK scan-synthetic-0001\n",
            "observation: sequence=0 status=ready\n",
            "voxel: global=(2, 0, 0) block=(0, 0, 0) local=(2, 0, 0)\n",
            "voxel_block_planned: yes\n",
            "grid: voxel_size_m=0.125000000 truncation_m=0.500000000 "
            "block_resolution=8\n",
            "camera_xyz_m: (-0.062500000, -0.062500000, 0.312500000)\n",
            "sampled_pixel: (0, 0)\n",
            "measured_depth_m: 1.000000000\n",
            "signed_distance_m: 0.687500000\n",
            "truncated_tsdf_value: 1.000000000\n",
            "sampling_status: observed-free-space\n",
            "sampling_rule: nearest-pixel-floor-projected-plus-half\n",
            "free_space_rule: signed-distance-above-positive-truncation\n",
            "occluded_rule: signed-distance-below-negative-truncation\n",
            "voxel_observed: yes\n",
            "inside_sampling_wedge: yes\n",
            "reference_evaluator_accepts: yes\n",
            "sampling_scope: one-voxel-one-observation-only\n",
            "unplanned_voxels_accepted: yes\n",
            "multi_observation_sampling_computed: no\n",
            "cross_view_occlusion_rule_defined: no\n",
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
                    "classify_tsdf_voxel_sampling_from_context",
                    side_effect=TsdfError("injected voxel sampling failure"),
                ) as classifier,
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                exit_code = main(
                    [
                        "reconstruct",
                        "tsdf-block-context-voxel-sampling",
                        str(plan_path),
                        str(session_path),
                        "--observation-sequence",
                        "0",
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
        self.assertIn("TSDF BLOCK CONTEXT VOXEL SAMPLING FAILED", error)
        self.assertIn("injected voxel sampling failure", error)
        self.assertNotIn("Traceback", error)


if __name__ == "__main__":
    unittest.main()
