from __future__ import annotations

import io
import json
import shutil
import tempfile
import unittest
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch


from spatialforge import (
    allocate_empty_tsdf_blocks,
    build_tsdf_replay_depth_context,
    evaluate_tsdf_voxel_contribution,
    evaluate_tsdf_voxel_contribution_from_context,
    locate_tsdf_voxel,
)
from spatialforge.cli import main
from spatialforge.errors import TsdfError
from spatialforge.model import ScanSession
from spatialforge.session_loader import load_scan_session
from spatialforge.tsdf_block_plan import plan_tsdf_blocks
from spatialforge.tsdf_block_plan_loader import (
    TsdfBlockPlan,
    load_tsdf_block_plan,
)
from spatialforge.tsdf_block_storage import (
    TSDF_BLOCK_VOXELS,
    TsdfBlockStorage,
)
from spatialforge.tsdf_replay_depth_context import (
    TsdfReplayDepthContext,
    TsdfReplayDepthObservation,
)
from spatialforge.tsdf_voxel_address import TsdfVoxelAddress
from spatialforge.tsdf_voxel_contribution import (
    TsdfContributionStatus,
    TsdfVoxelContribution,
)


TEST_ROOT = Path(__file__).resolve().parent
FIXTURE = TEST_ROOT / "fixtures" / "minimal.vgsession"
PLAN_ARGUMENTS = {
    "voxel_size_m": 0.125,
    "truncation_m": 0.5,
}
PLAN_SHA256 = "372c7c5d49eff1a30317ceb8b67cb3c40683f1d9d2a6179a049772f763d6f79d"
REPLAY_SHA256 = "dc001ae0ca01004a21ad227b12d57a7350bbc23904040453ca73fd955ef050b8"

_FORBIDDEN_EVALUATION_TARGETS = (
    "spatialforge.tsdf_voxel_contribution.replay_session",
    "spatialforge.tsdf_voxel_contribution._read_depth",
    "spatialforge.tsdf_voxel_contribution._sample_path",
    "spatialforge.tsdf_voxel_contribution._validate_reconstruction_contract",
    "spatialforge.tsdf_voxel_contribution.evaluate_tsdf_voxel_contribution",
    "spatialforge.replay._file_digest",
    "hashlib.sha256",
    "pathlib.Path.open",
    "PIL.Image.open",
    "spatialforge.tsdf_voxel_update.apply_tsdf_voxel_contribution",
    "spatialforge.tsdf_voxel_traversal.traverse_tsdf_voxel_observations",
)

_FORBIDDEN_CONTEXT_CLI_TARGETS = (
    "spatialforge.cli.evaluate_tsdf_voxel_contribution",
    "spatialforge.cli.apply_tsdf_voxel_contribution",
    "spatialforge.cli.traverse_tsdf_voxel_observations",
    "spatialforge.cli.reconstruct_point_cloud",
    "spatialforge.cli.integrate_tsdf",
    "spatialforge.cli.integrate_sparse_tsdf",
    "spatialforge.cli.infer_tsdf_bounds",
    "spatialforge.cli.plan_tsdf_blocks",
    "spatialforge.cli.extract_surface_points",
    "spatialforge.cli.extract_triangle_mesh",
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
    plan_name: str = "fixture.sftplan",
    frame_stride: int = 1,
) -> tuple[
    TsdfBlockPlan,
    ScanSession,
    TsdfBlockStorage,
    TsdfReplayDepthContext,
]:
    plan = load_tsdf_block_plan(
        create_plan(
            parent,
            session_path=session_path,
            name=plan_name,
            frame_stride=frame_stride,
        )
    )
    session = load_scan_session(session_path)
    storage = allocate_empty_tsdf_blocks(plan, session)
    context = build_tsdf_replay_depth_context(plan, session)
    return plan, session, storage, context


def require_address(
    storage: TsdfBlockStorage,
    global_index: tuple[int, int, int],
) -> TsdfVoxelAddress:
    address = locate_tsdf_voxel(storage, global_index)
    if address is None:
        raise AssertionError(f"expected planned address for {global_index}")
    return address


def storage_for_blocks(
    plan: TsdfBlockPlan,
    session: ScanSession,
    block_indices: tuple[tuple[int, int, int], ...],
) -> TsdfBlockStorage:
    minimum = tuple(
        min(block[axis] for block in block_indices)
        for axis in range(3)
    )
    maximum = tuple(
        max(block[axis] for block in block_indices)
        for axis in range(3)
    )
    changed_plan = replace(
        plan,
        surface_blocks=(block_indices[0],),
        active_blocks=block_indices,
        planned_voxel_slots=len(block_indices) * TSDF_BLOCK_VOXELS,
        min_block_index=minimum,
        max_block_index=maximum,
    )
    return allocate_empty_tsdf_blocks(changed_plan, session)


def remove_first_record(path: Path) -> None:
    remaining = path.read_text(encoding="utf-8").splitlines()[1:]
    path.write_text(
        "".join(line + "\n" for line in remaining),
        encoding="utf-8",
    )


def set_first_pose_translation_x(session_path: Path, value: float) -> None:
    pose_path = session_path / "streams" / "poses.jsonl"
    records = [
        json.loads(line)
        for line in pose_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    records[0]["T_world_camera"][3] = value
    pose_path.write_text(
        "".join(
            json.dumps(record, separators=(",", ":")) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def tree_snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def storage_snapshot(storage: TsdfBlockStorage) -> tuple[object, ...]:
    return (
        storage.block_indices,
        id(storage.tsdf_sums),
        id(storage.weights),
        storage.tsdf_sums.shape,
        storage.weights.shape,
        storage.tsdf_sums.strides,
        storage.weights.strides,
        storage.tsdf_sums.dtype.str,
        storage.weights.dtype.str,
        bool(storage.tsdf_sums.flags.writeable),
        bool(storage.weights.flags.writeable),
        storage.tsdf_sums.tobytes(),
        storage.weights.tobytes(),
    )


def context_snapshot(context: TsdfReplayDepthContext) -> tuple[object, ...]:
    observations: list[object] = []
    for record in context.observations:
        depth = record.depth_m
        observations.append(
            (
                record.observation_sequence,
                record.status,
                record.t_world_camera,
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


@contextmanager
def forbidden_evaluation_calls():
    with ExitStack() as stack:
        mocks: list[MagicMock] = []
        for target in _FORBIDDEN_EVALUATION_TARGETS:
            mocks.append(stack.enter_context(patch(target)))
        yield mocks


@contextmanager
def forbidden_context_cli_calls():
    with ExitStack() as stack:
        mocks: list[MagicMock] = []
        for target in _FORBIDDEN_CONTEXT_CLI_TARGETS:
            mocks.append(stack.enter_context(patch(target)))
        yield mocks


def evaluate_isolated(
    storage: TsdfBlockStorage,
    address: TsdfVoxelAddress,
    context: TsdfReplayDepthContext,
    observation_sequence: int,
) -> TsdfVoxelContribution:
    with forbidden_evaluation_calls() as forbidden:
        try:
            return evaluate_tsdf_voxel_contribution_from_context(
                storage,
                address,
                context,
                observation_sequence,
            )
        finally:
            for forbidden_call in forbidden:
                forbidden_call.assert_not_called()


class _CountingTuple(tuple):
    def __new__(cls, values):
        instance = super().__new__(cls, values)
        instance.iterations = 0
        instance.lookups = 0
        return instance

    def __iter__(self):
        self.iterations += 1
        return super().__iter__()

    def __getitem__(self, index):
        self.lookups += 1
        return super().__getitem__(index)


class TsdfContextContributionTests(unittest.TestCase):
    def test_fixture_matches_legacy_exactly_without_evaluation_io_or_mutation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            fixture = copy_fixture(temporary_root)
            _, session, storage, context = load_case(
                temporary_root,
                session_path=fixture,
            )
            address = require_address(storage, (8, -1, -1))
            expected = tuple(
                evaluate_tsdf_voxel_contribution(
                    storage,
                    address,
                    session,
                    sequence,
                )
                for sequence in (0, 1)
            )
            before_storage = storage_snapshot(storage)
            before_context = context_snapshot(context)
            before_tree = tree_snapshot(temporary_root)

            actual = tuple(
                evaluate_isolated(storage, address, context, sequence)
                for sequence in (0, 1)
            )
            repeated = evaluate_isolated(storage, address, context, 0)

            after_storage = storage_snapshot(storage)
            after_context = context_snapshot(context)
            after_tree = tree_snapshot(temporary_root)

        self.assertEqual(actual, expected)
        self.assertEqual(repeated, expected[0])
        self.assertEqual(before_storage, after_storage)
        self.assertEqual(before_context, after_context)
        self.assertEqual(before_tree, after_tree)
        self.assertEqual(actual[0].status, TsdfContributionStatus.CONTRIBUTES)
        self.assertEqual(actual[0].tsdf_sum_delta, -0.125)
        self.assertEqual(
            actual[1].tsdf_sum_delta,
            -0.12499999999999978,
        )
        self.assertEqual(address.storage_flat_index, 1016)

    def test_surface_band_math_and_skip_statuses_match_legacy(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            fixture = copy_fixture(temporary_root, "shifted.vgsession")
            set_first_pose_translation_x(fixture, -0.0625)
            plan, session, storage, context = load_case(
                temporary_root,
                session_path=fixture,
                plan_name="shifted.sftplan",
            )
            cases = (
                (1, 0.75, 1.0, TsdfContributionStatus.CONTRIBUTES),
                (3, 0.50, 1.0, TsdfContributionStatus.CONTRIBUTES),
                (5, 0.25, 0.5, TsdfContributionStatus.CONTRIBUTES),
                (7, 0.00, 0.0, TsdfContributionStatus.CONTRIBUTES),
                (9, -0.25, -0.5, TsdfContributionStatus.CONTRIBUTES),
                (11, -0.50, -1.0, TsdfContributionStatus.CONTRIBUTES),
                (12, -0.625, None, TsdfContributionStatus.BEHIND_TRUNCATION),
            )
            for global_x, signed_distance, delta, status in cases:
                with self.subTest(global_x=global_x):
                    address = require_address(storage, (global_x, -1, -1))
                    expected = evaluate_tsdf_voxel_contribution(
                        storage,
                        address,
                        session,
                        0,
                    )
                    actual = evaluate_isolated(storage, address, context, 0)
                    self.assertEqual(actual, expected)
                    self.assertEqual(actual.status, status)
                    self.assertEqual(actual.signed_distance_m, signed_distance)
                    self.assertEqual(actual.tsdf_sum_delta, delta)

            out_of_view_address = require_address(storage, (0, 7, -1))
            out_of_view_expected = evaluate_tsdf_voxel_contribution(
                storage,
                out_of_view_address,
                session,
                0,
            )
            out_of_view = evaluate_isolated(
                storage,
                out_of_view_address,
                context,
                0,
            )

            behind_storage = storage_for_blocks(
                plan,
                session,
                ((-1, -1, -1),),
            )
            behind_address = require_address(behind_storage, (-1, -1, -1))
            behind_expected = evaluate_tsdf_voxel_contribution(
                behind_storage,
                behind_address,
                session,
                0,
            )
            behind = evaluate_isolated(
                behind_storage,
                behind_address,
                context,
                0,
            )

        self.assertEqual(out_of_view, out_of_view_expected)
        self.assertEqual(
            out_of_view.status,
            TsdfContributionStatus.PROJECTION_OUTSIDE_IMAGE,
        )
        self.assertEqual(behind, behind_expected)
        self.assertEqual(
            behind.status,
            TsdfContributionStatus.CAMERA_Z_NONPOSITIVE,
        )

    def test_invalid_depth_matches_legacy_without_source_access(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            fixture = copy_fixture(temporary_root, "invalid.vgsession")
            (fixture / "data" / "depth" / "000000.pgm").write_text(
                "P2\n2 2\n65535\n1000 1000\n1000 0\n",
                encoding="ascii",
            )
            _, session, storage, context = load_case(
                temporary_root,
                session_path=fixture,
                plan_name="invalid.sftplan",
            )
            address = require_address(storage, (8, -1, -1))
            expected = evaluate_tsdf_voxel_contribution(
                storage,
                address,
                session,
                0,
            )
            actual = evaluate_isolated(storage, address, context, 0)

        self.assertEqual(actual, expected)
        self.assertEqual(actual.status, TsdfContributionStatus.DEPTH_INVALID)
        self.assertTrue(actual.depth_decoded)
        self.assertEqual(actual.pixel_uv, (1, 1))
        self.assertIsNone(actual.measured_depth_m)

    def test_missing_inputs_map_exactly_without_accessing_depth(self) -> None:
        cases = (
            (
                "depth",
                ("streams/depth.jsonl",),
                TsdfContributionStatus.MISSING_DEPTH,
            ),
            (
                "pose",
                ("streams/poses.jsonl",),
                TsdfContributionStatus.MISSING_POSE,
            ),
            (
                "both",
                ("streams/depth.jsonl", "streams/poses.jsonl"),
                TsdfContributionStatus.MISSING_DEPTH_AND_POSE,
            ),
        )
        for name, references, expected_status in cases:
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory(
                    dir=TEST_ROOT
                ) as temporary_directory:
                    temporary_root = Path(temporary_directory)
                    fixture = copy_fixture(
                        temporary_root,
                        f"{name}.vgsession",
                    )
                    for reference in references:
                        remove_first_record(fixture / reference)
                    _, session, storage, context = load_case(
                        temporary_root,
                        session_path=fixture,
                        plan_name=f"{name}.sftplan",
                    )
                    address = require_address(storage, (8, -1, -1))
                    expected = evaluate_tsdf_voxel_contribution(
                        storage,
                        address,
                        session,
                        0,
                    )
                    with (
                        patch.object(
                            TsdfReplayDepthObservation,
                            "depth_m",
                            new_callable=PropertyMock,
                            side_effect=AssertionError(
                                "missing observation depth was accessed"
                            ),
                        ),
                        forbidden_evaluation_calls() as forbidden,
                    ):
                        actual = (
                            evaluate_tsdf_voxel_contribution_from_context(
                                storage,
                                address,
                                context,
                                0,
                            )
                        )
                    for forbidden_call in forbidden:
                        forbidden_call.assert_not_called()

                self.assertEqual(actual, expected)
                self.assertEqual(actual.status, expected_status)
                self.assertFalse(actual.depth_decoded)

    def test_validation_rejects_bad_inputs_without_evaluation_io(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            _, _, storage, context = load_case(temporary_root)
            address = require_address(storage, (8, -1, -1))
            cases = (
                (object(), address, context, 0, "TsdfBlockStorage"),
                (storage, object(), context, 0, "TsdfVoxelAddress"),
                (storage, address, object(), 0, "TsdfReplayDepthContext"),
                (storage, address, context, True, "expected an integer"),
                (storage, address, context, 0.0, "expected an integer"),
                (storage, address, context, -1, "non-negative"),
                (storage, address, context, 2, "outside context range"),
                (
                    storage,
                    replace(
                        address,
                        storage_flat_index=address.storage_flat_index + 1,
                    ),
                    context,
                    0,
                    "does not match",
                ),
            )
            for (
                changed_storage,
                changed_address,
                changed_context,
                sequence,
                message,
            ) in cases:
                with self.subTest(message=message):
                    with forbidden_evaluation_calls() as forbidden:
                        with self.assertRaises(TsdfError) as raised:
                            evaluate_tsdf_voxel_contribution_from_context(
                                changed_storage,
                                changed_address,
                                changed_context,
                                sequence,
                            )
                    self.assertIn(message, str(raised.exception))
                    for forbidden_call in forbidden:
                        forbidden_call.assert_not_called()

            _, _, stride_storage, stride_context = load_case(
                temporary_root,
                plan_name="stride.sftplan",
                frame_stride=2,
            )
            stride_address = require_address(stride_storage, (8, -1, -1))
            with self.assertRaises(TsdfError) as raised:
                evaluate_isolated(
                    stride_storage,
                    stride_address,
                    stride_context,
                    1,
                )
        self.assertIn("not selected", str(raised.exception))

    def test_context_plan_provenance_selection_and_sample_mismatches_reject(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            plan, _, storage, context = load_case(temporary_root)
            address = require_address(storage, (8, -1, -1))
            stride_context = replace(
                context,
                frame_stride=2,
                selected_observation_sequences=(0,),
                observations=(context.observations[0],),
                valid_depth_samples=4,
            )
            context_cases = (
                (
                    replace(context, source_plan_digest_sha256="0" * 64),
                    "source plan digest",
                ),
                (
                    replace(context, replay_digest_sha256="0" * 64),
                    "replay digest",
                ),
                (
                    replace(context, session_id="different-session"),
                    "session_id",
                ),
                (stride_context, "selection"),
            )
            for changed_context, message in context_cases:
                with self.subTest(message=message):
                    with self.assertRaises(TsdfError) as raised:
                        evaluate_isolated(
                            storage,
                            address,
                            changed_context,
                            0,
                        )
                    self.assertIn(message, str(raised.exception))

            sample_plan = replace(
                plan,
                valid_depth_points=7,
                invalid_depth_samples=1,
            )
            sample_storage = TsdfBlockStorage(
                source_plan=sample_plan,
                block_indices=storage.block_indices,
                tsdf_sums=storage.tsdf_sums,
                weights=storage.weights,
            )
            with self.assertRaises(TsdfError) as raised:
                evaluate_isolated(
                    sample_storage,
                    address,
                    context,
                    0,
                )
            self.assertIn("sample counts", str(raised.exception))

            malformed_plan_cases = (
                ("stride-bool", {"frame_stride": True}, "frame_stride"),
                (
                    "total-bool",
                    {"total_observations": True},
                    "total_observations",
                ),
                (
                    "selected-negative",
                    {"selected_observations": -1},
                    "selected_observations",
                ),
                (
                    "paired-bool",
                    {"paired_observations": True},
                    "paired_observations",
                ),
                (
                    "valid-negative",
                    {"valid_depth_points": -1},
                    "valid_depth_points",
                ),
                (
                    "missing-depth-bool",
                    {"skipped_missing_depth": True},
                    "skipped_missing_depth",
                ),
                (
                    "missing-pose-negative",
                    {"skipped_missing_pose": -1},
                    "skipped_missing_pose",
                ),
                (
                    "invalid-depth-bool",
                    {"invalid_depth_samples": True},
                    "invalid_depth_samples",
                ),
                (
                    "paired-exceeds-selection",
                    {"paired_observations": 3},
                    "exceed selection",
                ),
                (
                    "missing-structure",
                    {
                        "paired_observations": 1,
                        "skipped_missing_depth": 0,
                        "skipped_missing_pose": 0,
                        "valid_depth_points": 4,
                    },
                    "missing-input counts",
                ),
            )
            for name, changes, message in malformed_plan_cases:
                with self.subTest(plan=name):
                    changed_plan = replace(plan, **changes)
                    changed_storage = TsdfBlockStorage(
                        source_plan=changed_plan,
                        block_indices=storage.block_indices,
                        tsdf_sums=storage.tsdf_sums,
                        weights=storage.weights,
                    )
                    with self.assertRaises(TsdfError) as raised:
                        evaluate_isolated(
                            changed_storage,
                            address,
                            context,
                            0,
                        )
                    self.assertIn(message, str(raised.exception))

    def test_evaluation_indexes_one_record_without_scanning_context(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            _, _, storage, context = load_case(Path(temporary_directory))
            address = require_address(storage, (8, -1, -1))
            observations = _CountingTuple(context.observations)
            counted_context = replace(context, observations=observations)
            observations.iterations = 0
            observations.lookups = 0

            contribution = evaluate_isolated(
                storage,
                address,
                counted_context,
                1,
            )

        self.assertTrue(contribution.contributes)
        self.assertEqual(contribution.observation_sequence, 1)
        self.assertEqual(observations.iterations, 0)
        self.assertEqual(observations.lookups, 1)

    def test_snapshot_remains_evaluable_after_source_depth_changes(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            fixture = copy_fixture(temporary_root)
            _, session, storage, context = load_case(
                temporary_root,
                session_path=fixture,
            )
            address = require_address(storage, (8, -1, -1))
            expected = evaluate_tsdf_voxel_contribution(
                storage,
                address,
                session,
                0,
            )
            depth_path = fixture / "data" / "depth" / "000000.pgm"
            depth_path.write_bytes(b"source changed after context build\n")
            changed_tree = tree_snapshot(temporary_root)

            actual = evaluate_isolated(storage, address, context, 0)
            after_tree = tree_snapshot(temporary_root)

        self.assertEqual(actual, expected)
        self.assertEqual(after_tree, changed_tree)


class TsdfContextContributionCliTests(unittest.TestCase):
    def test_cli_reports_exact_context_evaluation_and_isolates_inner_call(
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
            actual_evaluate = evaluate_tsdf_voxel_contribution_from_context
            isolated_forbidden: list[list[MagicMock]] = []

            def evaluate_with_isolation(storage, address, context, sequence):
                with forbidden_evaluation_calls() as forbidden:
                    result = actual_evaluate(
                        storage,
                        address,
                        context,
                        sequence,
                    )
                isolated_forbidden.append(forbidden)
                return result

            with (
                forbidden_context_cli_calls() as forbidden_cli,
                patch(
                    "spatialforge.cli."
                    "evaluate_tsdf_voxel_contribution_from_context",
                    side_effect=evaluate_with_isolation,
                ) as evaluator,
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                exit_code = main(
                    [
                        "reconstruct",
                        "tsdf-block-context-contribution",
                        str(plan_path),
                        str(session_path),
                        "--observation-sequence",
                        "0",
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
        evaluator.assert_called_once()
        self.assertEqual(len(isolated_forbidden), 1)
        for forbidden_call in isolated_forbidden[0]:
            forbidden_call.assert_not_called()
        for forbidden_call in forbidden_cli:
            forbidden_call.assert_not_called()

        expected_output = (
            "TSDF BLOCK CONTEXT CONTRIBUTION CHECK scan-synthetic-0001\n"
            "artifact: valid\n"
            "session_replay: matched\n"
            "context_selection: frame_stride=1 total=2 selected=2\n"
            "context_immutable: yes\n"
            "depth_source: replay-depth-context\n"
            "observation_sequence: 0\n"
            "voxel: global=(8, -1, -1) block=(1, -1, -1) "
            "local=(0, 7, 7) row=1 array=(1, 7, 7, 0) storage_flat=1016\n"
            "world_xyz_m: (1.062500000, -0.062500000, -0.062500000)\n"
            "camera_xyz_m: (0.062500000, 0.062500000, 1.062500000)\n"
            "projected_uv: (0.617647059, 0.617647059)\n"
            "pixel_uv: (1, 1)\n"
            "depth_decoded: yes\n"
            "measured_depth_m: 1.000000000\n"
            "signed_distance_m: -0.062500000\n"
            "evaluation: contributes\n"
            "proposed_delta: tsdf_sum=-0.125000000 weight=1\n"
            "evaluation_replay_hashing: no\n"
            "evaluation_depth_decoding: no\n"
            "contributions_applied: 0\n"
            "storage_mutated: no\n"
            "voxel_observation_traversal_performed: no\n"
            "voxel_address_traversal_performed: no\n"
            "fusion_block_traversal_performed: no\n"
            "full_fusion_performed: no\n"
            "missing_blocks_created: no\n"
            "artifact_written: no\n"
            "context_persisted: no\n"
            f"plan_sha256: {PLAN_SHA256}\n"
            f"replay_digest_sha256: {REPLAY_SHA256}\n"
        )
        self.assertEqual(stdout.getvalue(), expected_output)


if __name__ == "__main__":
    unittest.main()
