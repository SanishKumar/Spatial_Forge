from __future__ import annotations

import io
import json
import shutil
import tempfile
import unittest
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from spatialforge import (
    allocate_empty_tsdf_blocks,
    apply_tsdf_voxel_contribution_from_context,
    build_tsdf_replay_depth_context,
    evaluate_tsdf_voxel_contribution_from_context,
    locate_tsdf_voxel,
    traverse_tsdf_voxel_observations,
    traverse_tsdf_voxel_observations_from_context,
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
from spatialforge.tsdf_block_storage import TsdfBlockStorage
from spatialforge.tsdf_replay_depth_context import TsdfReplayDepthContext
from spatialforge.tsdf_voxel_address import TsdfVoxelAddress
from spatialforge.tsdf_voxel_contribution import TsdfContributionStatus
from spatialforge.tsdf_voxel_traversal import TsdfVoxelTraversalReceipt


TEST_ROOT = Path(__file__).resolve().parent
FIXTURE = TEST_ROOT / "fixtures" / "minimal.vgsession"
PLAN_ARGUMENTS = {
    "voxel_size_m": 0.125,
    "truncation_m": 0.5,
}
PLAN_SHA256 = "372c7c5d49eff1a30317ceb8b67cb3c40683f1d9d2a6179a049772f763d6f79d"
REPLAY_SHA256 = "dc001ae0ca01004a21ad227b12d57a7350bbc23904040453ca73fd955ef050b8"

_FORBIDDEN_TRAVERSAL_TARGETS = (
    "spatialforge.tsdf_voxel_traversal.replay_session",
    "spatialforge.tsdf_voxel_contribution.replay_session",
    "spatialforge.tsdf_voxel_update.replay_session",
    "spatialforge.tsdf_replay_depth_context.replay_session",
    "spatialforge.tsdf_voxel_contribution._read_depth",
    "spatialforge.tsdf_replay_depth_context._read_depth",
    "spatialforge.tsdf_voxel_contribution._sample_path",
    "spatialforge.tsdf_replay_depth_context._sample_path",
    "spatialforge.tsdf_voxel_contribution."
    "_validate_reconstruction_contract",
    "spatialforge.tsdf_replay_depth_context."
    "_validate_reconstruction_contract",
    "spatialforge.replay._file_digest",
    "hashlib.sha256",
    "pathlib.Path.open",
    "PIL.Image.open",
    "spatialforge.tsdf_voxel_traversal."
    "evaluate_tsdf_voxel_contribution",
    "spatialforge.tsdf_voxel_traversal."
    "apply_tsdf_voxel_contribution",
    "spatialforge.tsdf_replay_depth_context."
    "build_tsdf_replay_depth_context",
    "spatialforge.tsdf_block_storage.allocate_empty_tsdf_blocks",
)

_FORBIDDEN_CONTEXT_CLI_TARGETS = (
    "spatialforge.cli.evaluate_tsdf_voxel_contribution",
    "spatialforge.cli.apply_tsdf_voxel_contribution",
    "spatialforge.cli.traverse_tsdf_voxel_observations",
    "spatialforge.cli.evaluate_tsdf_voxel_contribution_from_context",
    "spatialforge.cli.apply_tsdf_voxel_contribution_from_context",
    "spatialforge.cli.reconstruct_point_cloud",
    "spatialforge.cli.integrate_tsdf",
    "spatialforge.cli.integrate_sparse_tsdf",
    "spatialforge.cli.infer_tsdf_bounds",
    "spatialforge.cli.plan_tsdf_blocks",
    "spatialforge.cli.extract_surface_points",
    "spatialforge.cli.extract_triangle_mesh",
    "spatialforge.tsdf._integrate_tsdf",
    "spatialforge.tsdf_block_plan._plan_observation_blocks",
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
    global_index_xyz: tuple[int, int, int],
) -> TsdfVoxelAddress:
    address = locate_tsdf_voxel(storage, global_index_xyz)
    if address is None:
        raise AssertionError(
            f"expected planned address for {global_index_xyz}"
        )
    return address


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


def storage_layout_snapshot(storage: TsdfBlockStorage) -> tuple[object, ...]:
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
        bool(storage.tsdf_sums.flags.c_contiguous),
        bool(storage.weights.flags.c_contiguous),
        bool(storage.tsdf_sums.flags.writeable),
        bool(storage.weights.flags.writeable),
        id(storage.tsdf_sums.base),
        id(storage.weights.base),
        int(storage.tsdf_sums.ctypes.data),
        int(storage.weights.ctypes.data),
    )


def storage_bytes(storage: TsdfBlockStorage) -> tuple[bytes, bytes]:
    return storage.tsdf_sums.tobytes(), storage.weights.tobytes()


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


@contextmanager
def forbidden_traversal_calls():
    with ExitStack() as stack:
        mocks: list[MagicMock] = []
        for target in _FORBIDDEN_TRAVERSAL_TARGETS:
            mocks.append(stack.enter_context(patch(target)))
        yield mocks


@contextmanager
def forbidden_context_cli_calls():
    with ExitStack() as stack:
        mocks: list[MagicMock] = []
        for target in _FORBIDDEN_CONTEXT_CLI_TARGETS:
            mocks.append(stack.enter_context(patch(target)))
        yield mocks


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


class TsdfContextTraversalTests(unittest.TestCase):
    def assert_only_flat_value_changed(
        self,
        before: bytes,
        after: bytes,
        flat_index: int,
        dtype: np.dtype,
        expected_value: float | int,
    ) -> None:
        start = flat_index * dtype.itemsize
        end = start + dtype.itemsize
        self.assertEqual(after[:start], before[:start])
        self.assertEqual(after[end:], before[end:])
        self.assertEqual(
            after[start:end],
            np.asarray(expected_value, dtype=dtype).tobytes(),
        )

    def test_exact_legacy_parity_is_linear_and_has_no_traversal_io(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            plan, session, legacy_storage, context = load_case(temporary_root)
            context_storage = allocate_empty_tsdf_blocks(plan, session)
            legacy_address = require_address(legacy_storage, (8, -1, -1))
            context_address = require_address(context_storage, (8, -1, -1))
            legacy_storage.tsdf_sums[0, 0, 0, 0] = 0.375
            legacy_storage.weights[0, 0, 0, 0] = 7
            context_storage.tsdf_sums[0, 0, 0, 0] = 0.375
            context_storage.weights[0, 0, 0, 0] = 7
            expected = traverse_tsdf_voxel_observations(
                legacy_storage,
                legacy_address,
                session,
            )

            counted_observations = _CountingTuple(context.observations)
            counted_context = replace(
                context,
                observations=counted_observations,
            )
            context_before = context_snapshot(counted_context)
            counted_observations.iterations = 0
            counted_observations.lookups = 0
            before_layout = storage_layout_snapshot(context_storage)
            before_sums, before_weights = storage_bytes(context_storage)
            before_tree = tree_snapshot(temporary_root)
            actual_evaluate = evaluate_tsdf_voxel_contribution_from_context
            actual_apply = apply_tsdf_voxel_contribution_from_context

            with (
                forbidden_traversal_calls() as forbidden,
                patch(
                    "spatialforge.tsdf_voxel_traversal."
                    "evaluate_tsdf_voxel_contribution_from_context",
                    wraps=actual_evaluate,
                ) as evaluator,
                patch(
                    "spatialforge.tsdf_voxel_traversal."
                    "apply_tsdf_voxel_contribution_from_context",
                    wraps=actual_apply,
                ) as updater,
            ):
                actual = traverse_tsdf_voxel_observations_from_context(
                    context_storage,
                    context_address,
                    counted_context,
                )
            traversal_iterations = counted_observations.iterations
            traversal_lookups = counted_observations.lookups
            after_sums, after_weights = storage_bytes(context_storage)
            after_tree = tree_snapshot(temporary_root)
            context_after = context_snapshot(counted_context)

        self.assertIsInstance(actual, TsdfVoxelTraversalReceipt)
        self.assertEqual(actual, expected)
        self.assertEqual(storage_bytes(context_storage), storage_bytes(legacy_storage))
        self.assertEqual(storage_layout_snapshot(context_storage), before_layout)
        self.assert_only_flat_value_changed(
            before_sums,
            after_sums,
            1016,
            np.dtype(np.float64),
            -0.24999999999999978,
        )
        self.assert_only_flat_value_changed(
            before_weights,
            after_weights,
            1016,
            np.dtype(np.uint32),
            2,
        )
        self.assertEqual(after_tree, before_tree)
        self.assertEqual(context_after, context_before)
        self.assertEqual(traversal_iterations, 0)
        self.assertEqual(traversal_lookups, 4)
        self.assertEqual(
            [call.args[3] for call in evaluator.call_args_list],
            [0, 1],
        )
        self.assertEqual(
            [
                call.args[1].observation_sequence
                for call in updater.call_args_list
            ],
            [0, 1],
        )
        for forbidden_call in forbidden:
            forbidden_call.assert_not_called()

    def test_stride_and_skip_transcripts_preserve_canonical_order(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            _, _, stride_storage, stride_context = load_case(
                temporary_root,
                plan_name="stride.sftplan",
                frame_stride=2,
            )
            stride_address = require_address(stride_storage, (8, -1, -1))
            stride_receipt = traverse_tsdf_voxel_observations_from_context(
                stride_storage,
                stride_address,
                stride_context,
            )

            fixture = copy_fixture(temporary_root, "missing-pose.vgsession")
            remove_first_record(fixture / "streams" / "poses.jsonl")
            plan, session, legacy_storage, context = load_case(
                temporary_root,
                session_path=fixture,
                plan_name="missing-pose.sftplan",
            )
            context_storage = allocate_empty_tsdf_blocks(plan, session)
            expected = traverse_tsdf_voxel_observations(
                legacy_storage,
                require_address(legacy_storage, (8, -1, -1)),
                session,
            )
            actual_evaluate = evaluate_tsdf_voxel_contribution_from_context
            actual_apply = apply_tsdf_voxel_contribution_from_context
            with (
                patch(
                    "spatialforge.tsdf_voxel_traversal."
                    "evaluate_tsdf_voxel_contribution_from_context",
                    wraps=actual_evaluate,
                ) as evaluator,
                patch(
                    "spatialforge.tsdf_voxel_traversal."
                    "apply_tsdf_voxel_contribution_from_context",
                    wraps=actual_apply,
                ) as updater,
            ):
                actual = traverse_tsdf_voxel_observations_from_context(
                    context_storage,
                    require_address(context_storage, (8, -1, -1)),
                    context,
                )

        self.assertEqual(stride_receipt.selected_observation_sequences, (0,))
        self.assertEqual(stride_receipt.evaluated_count, 1)
        self.assertEqual(stride_receipt.applied_count, 1)
        self.assertEqual(stride_receipt.tsdf_sum_after, -0.125)
        self.assertEqual(stride_receipt.weight_after, 1)
        self.assertEqual(actual, expected)
        self.assertEqual(
            tuple(item.status for item in actual.contributions),
            (
                TsdfContributionStatus.MISSING_POSE,
                TsdfContributionStatus.CONTRIBUTES,
            ),
        )
        self.assertEqual(
            [call.args[3] for call in evaluator.call_args_list],
            [0, 1],
        )
        self.assertEqual(updater.call_count, 1)
        self.assertEqual(
            updater.call_args.args[1].observation_sequence,
            1,
        )

    def test_skipped_only_target_is_repeatable_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            _, _, storage, context = load_case(Path(temporary_directory))
            address = require_address(storage, (12, -1, -1))
            before_layout = storage_layout_snapshot(storage)
            before_bytes = storage_bytes(storage)
            actual_evaluate = evaluate_tsdf_voxel_contribution_from_context

            with (
                patch(
                    "spatialforge.tsdf_voxel_traversal."
                    "evaluate_tsdf_voxel_contribution_from_context",
                    wraps=actual_evaluate,
                ) as evaluator,
                patch(
                    "spatialforge.tsdf_voxel_traversal."
                    "apply_tsdf_voxel_contribution_from_context",
                ) as updater,
            ):
                first = traverse_tsdf_voxel_observations_from_context(
                    storage,
                    address,
                    context,
                )
                second = traverse_tsdf_voxel_observations_from_context(
                    storage,
                    address,
                    context,
                )

        self.assertEqual(first, second)
        for receipt in (first, second):
            self.assertEqual(receipt.evaluated_count, 2)
            self.assertEqual(receipt.applied_count, 0)
            self.assertEqual(receipt.skipped_count, 2)
            self.assertEqual(receipt.storage_slots_updated, 0)
            self.assertEqual(receipt.update_receipts, ())
            self.assertEqual(receipt.tsdf_sum_after, 0.0)
            self.assertFalse(np.signbit(receipt.tsdf_sum_after))
            self.assertEqual(receipt.weight_after, 0)
            self.assertEqual(
                tuple(item.status for item in receipt.contributions),
                (
                    TsdfContributionStatus.BEHIND_TRUNCATION,
                    TsdfContributionStatus.BEHIND_TRUNCATION,
                ),
            )
        self.assertEqual(
            [call.args[3] for call in evaluator.call_args_list],
            [0, 1, 0, 1],
        )
        updater.assert_not_called()
        self.assertEqual(storage_layout_snapshot(storage), before_layout)
        self.assertEqual(storage_bytes(storage), before_bytes)

    def test_validation_and_nonempty_guard_reject_before_evaluation(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            _, _, storage, context = load_case(Path(temporary_directory))
            address = require_address(storage, (8, -1, -1))
            changed_context = replace(
                context,
                source_plan_digest_sha256="0" * 64,
            )

            for invalid_context, expected_message in (
                (object(), "TsdfReplayDepthContext"),
                (changed_context, "source plan digest"),
            ):
                with self.subTest(expected_message=expected_message):
                    before_layout = storage_layout_snapshot(storage)
                    before_bytes = storage_bytes(storage)
                    with (
                        patch(
                            "spatialforge.tsdf_voxel_traversal."
                            "evaluate_tsdf_voxel_contribution_from_context",
                        ) as evaluator,
                        patch(
                            "spatialforge.tsdf_voxel_traversal."
                            "apply_tsdf_voxel_contribution_from_context",
                        ) as updater,
                    ):
                        with self.assertRaises(TsdfError) as raised:
                            traverse_tsdf_voxel_observations_from_context(
                                storage,
                                address,
                                invalid_context,  # type: ignore[arg-type]
                            )
                    self.assertIn(expected_message, str(raised.exception))
                    evaluator.assert_not_called()
                    updater.assert_not_called()
                    self.assertEqual(
                        storage_layout_snapshot(storage),
                        before_layout,
                    )
                    self.assertEqual(storage_bytes(storage), before_bytes)

            storage.tsdf_sums[address.array_index_bzyx] = 0.25
            storage.weights[address.array_index_bzyx] = 1
            before_layout = storage_layout_snapshot(storage)
            before_bytes = storage_bytes(storage)
            with (
                patch(
                    "spatialforge.tsdf_voxel_traversal."
                    "evaluate_tsdf_voxel_contribution_from_context",
                ) as evaluator,
                patch(
                    "spatialforge.tsdf_voxel_traversal."
                    "apply_tsdf_voxel_contribution_from_context",
                ) as updater,
            ):
                with self.assertRaises(TsdfError) as raised:
                    traverse_tsdf_voxel_observations_from_context(
                        storage,
                        address,
                        context,
                    )

        self.assertIn("empty", str(raised.exception))
        evaluator.assert_not_called()
        updater.assert_not_called()
        self.assertEqual(storage_layout_snapshot(storage), before_layout)
        self.assertEqual(storage_bytes(storage), before_bytes)

    def test_all_evaluations_finish_and_transcript_validates_before_update(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            _, _, storage, context = load_case(temporary_root)
            address = require_address(storage, (8, -1, -1))
            before_layout = storage_layout_snapshot(storage)
            before_bytes = storage_bytes(storage)
            before_tree = tree_snapshot(temporary_root)
            actual_evaluate = evaluate_tsdf_voxel_contribution_from_context
            sequence_zero = actual_evaluate(storage, address, context, 0)

            with (
                patch(
                    "spatialforge.tsdf_voxel_traversal."
                    "evaluate_tsdf_voxel_contribution_from_context",
                    return_value=sequence_zero,
                ) as evaluator,
                patch(
                    "spatialforge.tsdf_voxel_traversal."
                    "apply_tsdf_voxel_contribution_from_context",
                ) as updater,
            ):
                with self.assertRaises(TsdfError) as duplicate_error:
                    traverse_tsdf_voxel_observations_from_context(
                        storage,
                        address,
                        context,
                    )
            self.assertEqual(evaluator.call_count, 2)
            updater.assert_not_called()
            self.assertEqual(storage_bytes(storage), before_bytes)

            def fail_second_evaluation(
                changed_storage: TsdfBlockStorage,
                changed_address: TsdfVoxelAddress,
                changed_context: TsdfReplayDepthContext,
                observation_sequence: int,
            ) -> object:
                if observation_sequence == 1:
                    raise TsdfError("injected later context evaluation failure")
                return actual_evaluate(
                    changed_storage,
                    changed_address,
                    changed_context,
                    observation_sequence,
                )

            with (
                patch(
                    "spatialforge.tsdf_voxel_traversal."
                    "evaluate_tsdf_voxel_contribution_from_context",
                    side_effect=fail_second_evaluation,
                ) as evaluator,
                patch(
                    "spatialforge.tsdf_voxel_traversal."
                    "apply_tsdf_voxel_contribution_from_context",
                ) as updater,
            ):
                with self.assertRaises(TsdfError) as evaluation_error:
                    traverse_tsdf_voxel_observations_from_context(
                        storage,
                        address,
                        context,
                    )
            after_tree = tree_snapshot(temporary_root)

        self.assertIn("transcript is inconsistent", str(duplicate_error.exception))
        self.assertIn(
            "injected later context evaluation failure",
            str(evaluation_error.exception),
        )
        self.assertEqual(evaluator.call_count, 2)
        self.assertEqual(
            [call.args[3] for call in evaluator.call_args_list],
            [0, 1],
        )
        updater.assert_not_called()
        self.assertEqual(storage_layout_snapshot(storage), before_layout)
        self.assertEqual(storage_bytes(storage), before_bytes)
        self.assertEqual(after_tree, before_tree)

    def test_later_update_failure_rolls_back_the_entire_traversal(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            _, _, storage, context = load_case(temporary_root)
            address = require_address(storage, (8, -1, -1))
            before_layout = storage_layout_snapshot(storage)
            before_bytes = storage_bytes(storage)
            before_tree = tree_snapshot(temporary_root)
            actual_apply = apply_tsdf_voxel_contribution_from_context

            def apply_then_fail(
                changed_storage: TsdfBlockStorage,
                contribution: object,
                changed_context: TsdfReplayDepthContext,
            ) -> object:
                update = actual_apply(
                    changed_storage,
                    contribution,  # type: ignore[arg-type]
                    changed_context,
                )
                if update.contribution.observation_sequence == 1:
                    raise TsdfError("injected second context update failure")
                return update

            with (
                forbidden_traversal_calls() as forbidden,
                patch(
                    "spatialforge.tsdf_voxel_traversal."
                    "apply_tsdf_voxel_contribution_from_context",
                    side_effect=apply_then_fail,
                ) as updater,
            ):
                with self.assertRaises(TsdfError) as raised:
                    traverse_tsdf_voxel_observations_from_context(
                        storage,
                        address,
                        context,
                    )
            after_tree = tree_snapshot(temporary_root)

        self.assertIn("injected second context update failure", str(raised.exception))
        self.assertEqual(updater.call_count, 2)
        self.assertEqual(storage_layout_snapshot(storage), before_layout)
        self.assertEqual(storage_bytes(storage), before_bytes)
        self.assertEqual(after_tree, before_tree)
        self.assertEqual(float(storage.tsdf_sums[address.array_index_bzyx]), 0.0)
        self.assertFalse(np.signbit(storage.tsdf_sums[address.array_index_bzyx]))
        self.assertEqual(int(storage.weights[address.array_index_bzyx]), 0)
        for forbidden_call in forbidden:
            forbidden_call.assert_not_called()

    def test_snapshot_traverses_after_depth_and_pose_sources_change(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            fixture = copy_fixture(temporary_root)
            plan, session, baseline_storage, context = load_case(
                temporary_root,
                session_path=fixture,
            )
            snapshot_storage = allocate_empty_tsdf_blocks(plan, session)
            baseline = traverse_tsdf_voxel_observations_from_context(
                baseline_storage,
                require_address(baseline_storage, (8, -1, -1)),
                context,
            )
            context_before = context_snapshot(context)
            depth_path = fixture / "data" / "depth" / "000000.pgm"
            depth_path.write_bytes(b"source changed after context build\n")
            set_first_pose_translation_x(fixture, 10.0)
            changed_tree = tree_snapshot(temporary_root)

            with forbidden_traversal_calls() as forbidden:
                actual = traverse_tsdf_voxel_observations_from_context(
                    snapshot_storage,
                    require_address(snapshot_storage, (8, -1, -1)),
                    context,
                )
            after_tree = tree_snapshot(temporary_root)
            context_after = context_snapshot(context)

        self.assertEqual(actual, baseline)
        self.assertEqual(actual.weight_after, 2)
        self.assertEqual(actual.tsdf_sum_after, -0.24999999999999978)
        self.assertEqual(context_after, context_before)
        self.assertEqual(after_tree, changed_tree)
        for forbidden_call in forbidden:
            forbidden_call.assert_not_called()


class TsdfContextTraversalCliTests(unittest.TestCase):
    def test_cli_reports_exact_isolated_context_traversal(self) -> None:
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
            captured_storage: list[TsdfBlockStorage] = []
            actual_allocate = allocate_empty_tsdf_blocks
            actual_build = build_tsdf_replay_depth_context
            actual_traverse = traverse_tsdf_voxel_observations_from_context
            actual_evaluate = evaluate_tsdf_voxel_contribution_from_context
            actual_apply = apply_tsdf_voxel_contribution_from_context
            actual_zeros = np.zeros

            def capture_allocation(
                plan: TsdfBlockPlan,
                session: ScanSession,
            ) -> TsdfBlockStorage:
                storage = actual_allocate(plan, session)
                captured_storage.append(storage)
                return storage

            def traverse_isolated(
                storage: TsdfBlockStorage,
                address: TsdfVoxelAddress,
                context: TsdfReplayDepthContext,
            ) -> TsdfVoxelTraversalReceipt:
                with forbidden_traversal_calls() as inner_forbidden:
                    receipt = actual_traverse(storage, address, context)
                for forbidden_call in inner_forbidden:
                    forbidden_call.assert_not_called()
                return receipt

            with (
                forbidden_context_cli_calls() as forbidden_cli,
                patch(
                    "spatialforge.cli.allocate_empty_tsdf_blocks",
                    side_effect=capture_allocation,
                ) as allocator,
                patch(
                    "spatialforge.cli.build_tsdf_replay_depth_context",
                    wraps=actual_build,
                ) as builder,
                patch(
                    "spatialforge.cli."
                    "traverse_tsdf_voxel_observations_from_context",
                    side_effect=traverse_isolated,
                ) as traverser,
                patch(
                    "spatialforge.tsdf_voxel_traversal."
                    "evaluate_tsdf_voxel_contribution_from_context",
                    wraps=actual_evaluate,
                ) as evaluator,
                patch(
                    "spatialforge.tsdf_voxel_traversal."
                    "apply_tsdf_voxel_contribution_from_context",
                    wraps=actual_apply,
                ) as updater,
                patch(
                    "spatialforge.tsdf_block_storage.np.zeros",
                    wraps=actual_zeros,
                ) as zeros,
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                exit_code = main(
                    [
                        "reconstruct",
                        "tsdf-block-context-voxel-traverse",
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
        self.assertEqual(before_tree, after_tree)
        allocator.assert_called_once()
        builder.assert_called_once()
        traverser.assert_called_once()
        self.assertEqual(evaluator.call_count, 2)
        self.assertEqual(updater.call_count, 2)
        self.assertEqual(zeros.call_count, 2)
        for forbidden_call in forbidden_cli:
            forbidden_call.assert_not_called()
        self.assertEqual(len(captured_storage), 1)
        storage = captured_storage[0]
        address = require_address(storage, (8, -1, -1))
        self.assertEqual(address.storage_flat_index, 1016)
        self.assertEqual(
            float(storage.tsdf_sums[address.array_index_bzyx]),
            -0.24999999999999978,
        )
        self.assertEqual(int(storage.weights[address.array_index_bzyx]), 2)
        self.assertEqual(
            np.flatnonzero(storage.tsdf_sums.reshape(-1)).tolist(),
            [1016],
        )
        self.assertEqual(
            np.flatnonzero(storage.weights.reshape(-1)).tolist(),
            [1016],
        )
        expected_output = (
            "TSDF BLOCK CONTEXT VOXEL TRAVERSAL CHECK "
            "scan-synthetic-0001\n"
            "artifact: valid\n"
            "session_replay: matched\n"
            "context_selection: frame_stride=1 total=2 selected=2\n"
            "context_immutable: yes\n"
            "depth_source: replay-depth-context\n"
            "voxel: global=(8, -1, -1) block=(1, -1, -1) "
            "local=(0, 7, 7) row=1 array=(1, 7, 7, 0) "
            "storage_flat=1016\n"
            "selection: frame_stride=1 total=2 selected=2\n"
            "slot_before: tsdf_sum=0.000000000 weight=0\n"
            "observation[0]: sequence=0 status=contributes "
            "delta_sum=-0.125000000 delta_weight=1\n"
            "observation[1]: sequence=1 status=contributes "
            "delta_sum=-0.125000000 delta_weight=1\n"
            "status_counts: contributes=2\n"
            "accumulated_delta: tsdf_sum=-0.250000000 weight=2\n"
            "slot_after: tsdf_sum=-0.250000000 weight=2\n"
            "storage_before: nonzero_sums=0 nonzero_weights=0 "
            "unknown_voxels=4096\n"
            "storage_after: nonzero_sums=1 nonzero_weights=1 "
            "unknown_voxels=4095\n"
            "context_provenance: matched\n"
            "traversal_source_freshness: construction-time-context\n"
            "traversal_session_replay: no\n"
            "traversal_replay_hashing: no\n"
            "traversal_source_io: no\n"
            "traversal_depth_decoding: no\n"
            "traversal_prepared_depth_access: yes\n"
            "contributions_evaluated: 2\n"
            "contributions_applied: 2\n"
            "contributions_skipped: 0\n"
            "duplicate_observation_applications: 0\n"
            "storage_slots_updated: 1\n"
            "voxel_observation_traversal_performed: yes\n"
            "voxel_address_traversal_performed: no\n"
            "fusion_block_traversal_performed: no\n"
            "ray_traversal_performed: no\n"
            "full_fusion_performed: no\n"
            "missing_blocks_created: no\n"
            "artifact_written: no\n"
            "storage_persisted: no\n"
            "context_persisted: no\n"
            f"plan_sha256: {PLAN_SHA256}\n"
            f"replay_digest_sha256: {REPLAY_SHA256}\n"
        )
        self.assertEqual(stdout.getvalue(), expected_output)


if __name__ == "__main__":
    unittest.main()
