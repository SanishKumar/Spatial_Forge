from __future__ import annotations

import io
import shutil
import tempfile
import unittest
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from spatialforge import (
    allocate_empty_tsdf_blocks,
    apply_tsdf_voxel_contribution,
    evaluate_tsdf_voxel_contribution,
    locate_tsdf_voxel,
    traverse_tsdf_voxel_observations,
)
from spatialforge.cli import main
from spatialforge.errors import TsdfError
from spatialforge.model import ScanSession
from spatialforge.replay import replay_session
from spatialforge.session_loader import load_scan_session
from spatialforge.tsdf_block_plan import plan_tsdf_blocks
from spatialforge.tsdf_block_plan_loader import (
    TsdfBlockPlan,
    load_tsdf_block_plan,
)
from spatialforge.tsdf_block_storage import TsdfBlockStorage
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

_FORBIDDEN_FULL_PIPELINE_TARGETS = (
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
) -> tuple[TsdfBlockPlan, ScanSession, TsdfBlockStorage]:
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
    return plan, session, storage


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


def tree_snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def storage_layout_snapshot(
    storage: TsdfBlockStorage,
) -> tuple[object, ...]:
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
    return (
        storage.tsdf_sums.tobytes(),
        storage.weights.tobytes(),
    )


@contextmanager
def forbidden_full_pipeline_calls():
    with ExitStack() as stack:
        mocks: list[MagicMock] = []
        for target in _FORBIDDEN_FULL_PIPELINE_TARGETS:
            mocks.append(stack.enter_context(patch(target)))
        yield mocks


class TsdfVoxelTraversalTests(unittest.TestCase):
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

    def test_exact_fixture_traverses_selected_observations_in_order(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            plan, session, storage = load_case(temporary_root)
            address = require_address(storage, (8, -1, -1))

            # A valid non-target sentinel catches array replacement or broad
            # writes while leaving the requested target canonically empty.
            storage.tsdf_sums[0, 0, 0, 0] = 0.375
            storage.weights[0, 0, 0, 0] = 7
            before_layout = storage_layout_snapshot(storage)
            before_sums, before_weights = storage_bytes(storage)
            before_tree = tree_snapshot(temporary_root)

            actual_evaluate = evaluate_tsdf_voxel_contribution
            actual_apply = apply_tsdf_voxel_contribution
            import spatialforge.tsdf_voxel_contribution as contribution_module

            actual_read_depth = contribution_module._read_depth
            with (
                forbidden_full_pipeline_calls() as forbidden,
                patch(
                    "spatialforge.tsdf_voxel_traversal."
                    "evaluate_tsdf_voxel_contribution",
                    wraps=actual_evaluate,
                ) as evaluator,
                patch(
                    "spatialforge.tsdf_voxel_traversal."
                    "apply_tsdf_voxel_contribution",
                    wraps=actual_apply,
                ) as updater,
                patch(
                    "spatialforge.tsdf_voxel_contribution._read_depth",
                    wraps=actual_read_depth,
                ) as depth_reader,
            ):
                receipt = traverse_tsdf_voxel_observations(
                    storage,
                    address,
                    session,
                )
            after_sums, after_weights = storage_bytes(storage)
            after_tree = tree_snapshot(temporary_root)

        self.assertIsInstance(receipt, TsdfVoxelTraversalReceipt)
        self.assertIs(receipt.address, address)
        self.assertEqual(
            receipt.source_plan_digest_sha256,
            plan.artifact_digest_sha256,
        )
        self.assertEqual(
            receipt.replay_digest_sha256,
            plan.replay_digest_sha256,
        )
        self.assertEqual(receipt.frame_stride, 1)
        self.assertEqual(receipt.total_observations, 2)
        self.assertEqual(receipt.selected_observation_sequences, (0, 1))
        self.assertEqual(receipt.evaluated_count, 2)
        self.assertEqual(receipt.applied_count, 2)
        self.assertEqual(receipt.skipped_count, 0)
        self.assertEqual(receipt.storage_slots_updated, 1)
        self.assertEqual(receipt.tsdf_sum_before, 0.0)
        self.assertFalse(np.signbit(receipt.tsdf_sum_before))
        self.assertEqual(receipt.weight_before, 0)
        self.assertEqual(receipt.tsdf_sum_after, -0.24999999999999978)
        self.assertEqual(receipt.weight_after, 2)
        self.assertEqual(
            tuple(
                contribution.observation_sequence
                for contribution in receipt.contributions
            ),
            (0, 1),
        )
        self.assertEqual(
            tuple(
                contribution.status
                for contribution in receipt.contributions
            ),
            (
                TsdfContributionStatus.CONTRIBUTES,
                TsdfContributionStatus.CONTRIBUTES,
            ),
        )
        self.assertEqual(
            tuple(
                contribution.tsdf_sum_delta
                for contribution in receipt.contributions
            ),
            (-0.125, -0.12499999999999978),
        )
        self.assertEqual(len(receipt.update_receipts), 2)
        self.assertEqual(
            tuple(
                update.contribution
                for update in receipt.update_receipts
            ),
            receipt.contributions,
        )
        self.assertEqual(
            tuple(
                (
                    update.tsdf_sum_before,
                    update.weight_before,
                    update.tsdf_sum_after,
                    update.weight_after,
                )
                for update in receipt.update_receipts
            ),
            (
                (0.0, 0, -0.125, 1),
                (
                    -0.125,
                    1,
                    -0.24999999999999978,
                    2,
                ),
            ),
        )
        self.assertEqual(address.storage_flat_index, 1016)
        self.assertEqual(address.array_index_bzyx, (1, 7, 7, 0))
        self.assertEqual(storage_layout_snapshot(storage), before_layout)
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
        self.assertEqual(
            [call.args[3] for call in evaluator.call_args_list],
            [0, 1],
        )
        self.assertEqual(
            [call.args[1] for call in evaluator.call_args_list],
            [address, address],
        )
        self.assertEqual(
            [
                call.args[1].observation_sequence
                for call in updater.call_args_list
            ],
            [0, 1],
        )
        self.assertEqual(depth_reader.call_count, 2)
        for forbidden_call in forbidden:
            forbidden_call.assert_not_called()
        with self.assertRaises(FrozenInstanceError):
            receipt.weight_after = 9  # type: ignore[misc]
        self.assertFalse(hasattr(receipt, "__dict__"))

    def test_receipt_rejects_a_truncated_canonical_sequence_prefix(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            _, session, storage = load_case(Path(temporary_directory))
            receipt = traverse_tsdf_voxel_observations(
                storage,
                require_address(storage, (8, -1, -1)),
                session,
            )

        self.assertEqual(receipt.total_observations, 2)
        self.assertEqual(receipt.selected_observation_sequences, (0, 1))
        with self.assertRaises(TsdfError) as raised:
            replace(
                receipt,
                selected_observation_sequences=(0,),
                contributions=receipt.contributions[:1],
                update_receipts=receipt.update_receipts[:1],
                tsdf_sum_after=-0.125,
                weight_after=1,
            )
        self.assertIn(
            "complete canonical stride order",
            str(raised.exception),
        )

    def test_frame_stride_two_selects_only_sequence_zero(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            _, session, storage = load_case(
                Path(temporary_directory),
                plan_name="stride.sftplan",
                frame_stride=2,
            )
            address = require_address(storage, (8, -1, -1))
            actual_evaluate = evaluate_tsdf_voxel_contribution
            actual_apply = apply_tsdf_voxel_contribution

            with (
                patch(
                    "spatialforge.tsdf_voxel_traversal."
                    "evaluate_tsdf_voxel_contribution",
                    wraps=actual_evaluate,
                ) as evaluator,
                patch(
                    "spatialforge.tsdf_voxel_traversal."
                    "apply_tsdf_voxel_contribution",
                    wraps=actual_apply,
                ) as updater,
            ):
                receipt = traverse_tsdf_voxel_observations(
                    storage,
                    address,
                    session,
                )

        self.assertEqual(receipt.frame_stride, 2)
        self.assertEqual(receipt.selected_observation_sequences, (0,))
        self.assertEqual(receipt.evaluated_count, 1)
        self.assertEqual(receipt.applied_count, 1)
        self.assertEqual(receipt.skipped_count, 0)
        self.assertEqual(receipt.tsdf_sum_after, -0.125)
        self.assertEqual(receipt.weight_after, 1)
        self.assertEqual(
            [call.args[3] for call in evaluator.call_args_list],
            [0],
        )
        self.assertEqual(
            [
                call.args[1].observation_sequence
                for call in updater.call_args_list
            ],
            [0],
        )

    def test_skipped_only_voxel_is_repeatable_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            _, session, storage = load_case(Path(temporary_directory))
            address = require_address(storage, (12, -1, -1))
            before_layout = storage_layout_snapshot(storage)
            before_bytes = storage_bytes(storage)
            actual_evaluate = evaluate_tsdf_voxel_contribution

            with (
                patch(
                    "spatialforge.tsdf_voxel_traversal."
                    "evaluate_tsdf_voxel_contribution",
                    wraps=actual_evaluate,
                ) as evaluator,
                patch(
                    "spatialforge.tsdf_voxel_traversal."
                    "apply_tsdf_voxel_contribution",
                ) as updater,
            ):
                first = traverse_tsdf_voxel_observations(
                    storage,
                    address,
                    session,
                )
                second = traverse_tsdf_voxel_observations(
                    storage,
                    address,
                    session,
                )

        self.assertEqual(address.block_index_xyz, (1, -1, -1))
        self.assertEqual(address.local_index_xyz, (4, 7, 7))
        self.assertEqual(address.block_row, 1)
        self.assertEqual(address.storage_flat_index, 1020)
        self.assertEqual(first, second)
        for receipt in (first, second):
            self.assertEqual(receipt.selected_observation_sequences, (0, 1))
            self.assertEqual(receipt.evaluated_count, 2)
            self.assertEqual(receipt.applied_count, 0)
            self.assertEqual(receipt.skipped_count, 2)
            self.assertEqual(receipt.storage_slots_updated, 0)
            self.assertEqual(receipt.tsdf_sum_before, 0.0)
            self.assertEqual(receipt.tsdf_sum_after, 0.0)
            self.assertFalse(np.signbit(receipt.tsdf_sum_after))
            self.assertEqual(receipt.weight_before, 0)
            self.assertEqual(receipt.weight_after, 0)
            self.assertEqual(receipt.update_receipts, ())
            self.assertEqual(
                tuple(
                    contribution.status
                    for contribution in receipt.contributions
                ),
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

    def test_missing_pose_skip_precedes_later_accepted_observation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            fixture = copy_fixture(temporary_root, "missing-pose.vgsession")
            remove_first_record(fixture / "streams" / "poses.jsonl")
            plan, session, storage = load_case(
                temporary_root,
                session_path=fixture,
                plan_name="missing-pose.sftplan",
            )
            address = require_address(storage, (8, -1, -1))
            actual_evaluate = evaluate_tsdf_voxel_contribution
            actual_apply = apply_tsdf_voxel_contribution

            with (
                patch(
                    "spatialforge.tsdf_voxel_traversal."
                    "evaluate_tsdf_voxel_contribution",
                    wraps=actual_evaluate,
                ) as evaluator,
                patch(
                    "spatialforge.tsdf_voxel_traversal."
                    "apply_tsdf_voxel_contribution",
                    wraps=actual_apply,
                ) as updater,
            ):
                receipt = traverse_tsdf_voxel_observations(
                    storage,
                    address,
                    session,
                )

        self.assertEqual(plan.total_observations, 2)
        self.assertEqual(plan.selected_observations, 2)
        self.assertEqual(plan.paired_observations, 1)
        self.assertEqual(receipt.selected_observation_sequences, (0, 1))
        self.assertEqual(receipt.evaluated_count, 2)
        self.assertEqual(receipt.applied_count, 1)
        self.assertEqual(receipt.skipped_count, 1)
        self.assertEqual(receipt.storage_slots_updated, 1)
        self.assertEqual(
            tuple(
                contribution.status
                for contribution in receipt.contributions
            ),
            (
                TsdfContributionStatus.MISSING_POSE,
                TsdfContributionStatus.CONTRIBUTES,
            ),
        )
        self.assertEqual(receipt.contributions[0].tsdf_sum_delta, None)
        self.assertEqual(
            receipt.contributions[1].tsdf_sum_delta,
            -0.12499999999999978,
        )
        self.assertEqual(len(receipt.update_receipts), 1)
        self.assertEqual(
            receipt.update_receipts[0].contribution.observation_sequence,
            1,
        )
        self.assertEqual(receipt.tsdf_sum_after, -0.12499999999999978)
        self.assertEqual(receipt.weight_after, 1)
        self.assertEqual(
            [call.args[3] for call in evaluator.call_args_list],
            [0, 1],
        )
        self.assertEqual(updater.call_count, 1)

    def test_nonempty_repeat_rejects_before_evaluation(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            _, session, storage = load_case(Path(temporary_directory))
            address = require_address(storage, (8, -1, -1))
            first = traverse_tsdf_voxel_observations(
                storage,
                address,
                session,
            )
            before_layout = storage_layout_snapshot(storage)
            before_bytes = storage_bytes(storage)

            with (
                patch(
                    "spatialforge.tsdf_voxel_traversal."
                    "evaluate_tsdf_voxel_contribution",
                ) as evaluator,
                patch(
                    "spatialforge.tsdf_voxel_traversal."
                    "apply_tsdf_voxel_contribution",
                ) as updater,
            ):
                with self.assertRaises(TsdfError) as raised:
                    traverse_tsdf_voxel_observations(
                        storage,
                        address,
                        session,
                    )

        self.assertEqual(first.weight_after, 2)
        self.assertIn("empty", str(raised.exception))
        evaluator.assert_not_called()
        updater.assert_not_called()
        self.assertEqual(storage_layout_snapshot(storage), before_layout)
        self.assertEqual(storage_bytes(storage), before_bytes)

    def test_duplicate_sequence_transcript_rejects_before_update(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            _, session, storage = load_case(Path(temporary_directory))
            address = require_address(storage, (8, -1, -1))
            sequence_zero = evaluate_tsdf_voxel_contribution(
                storage,
                address,
                session,
                0,
            )
            before_layout = storage_layout_snapshot(storage)
            before_bytes = storage_bytes(storage)

            with (
                patch(
                    "spatialforge.tsdf_voxel_traversal."
                    "evaluate_tsdf_voxel_contribution",
                    return_value=sequence_zero,
                ) as evaluator,
                patch(
                    "spatialforge.tsdf_voxel_traversal."
                    "apply_tsdf_voxel_contribution",
                ) as updater,
            ):
                with self.assertRaises(TsdfError) as raised:
                    traverse_tsdf_voxel_observations(
                        storage,
                        address,
                        session,
                    )

        self.assertIn(
            "contribution transcript is inconsistent",
            str(raised.exception),
        )
        self.assertEqual(evaluator.call_count, 2)
        self.assertEqual(
            [call.args[3] for call in evaluator.call_args_list],
            [0, 1],
        )
        updater.assert_not_called()
        self.assertEqual(storage_layout_snapshot(storage), before_layout)
        self.assertEqual(storage_bytes(storage), before_bytes)

    def test_later_evaluation_failure_occurs_before_any_update(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            _, session, storage = load_case(temporary_root)
            address = require_address(storage, (8, -1, -1))
            before_layout = storage_layout_snapshot(storage)
            before_bytes = storage_bytes(storage)
            before_tree = tree_snapshot(temporary_root)
            actual_evaluate = evaluate_tsdf_voxel_contribution

            def fail_second_evaluation(
                changed_storage: TsdfBlockStorage,
                changed_address: TsdfVoxelAddress,
                changed_session: ScanSession,
                observation_sequence: int,
            ) -> object:
                if observation_sequence == 1:
                    raise TsdfError("injected later evaluation failure")
                return actual_evaluate(
                    changed_storage,
                    changed_address,
                    changed_session,
                    observation_sequence,
                )

            with (
                patch(
                    "spatialforge.tsdf_voxel_traversal."
                    "evaluate_tsdf_voxel_contribution",
                    side_effect=fail_second_evaluation,
                ) as evaluator,
                patch(
                    "spatialforge.tsdf_voxel_traversal."
                    "apply_tsdf_voxel_contribution",
                ) as updater,
            ):
                with self.assertRaises(TsdfError) as raised:
                    traverse_tsdf_voxel_observations(
                        storage,
                        address,
                        session,
                    )

            after_tree = tree_snapshot(temporary_root)

        self.assertIn("injected later evaluation failure", str(raised.exception))
        self.assertEqual(evaluator.call_count, 2)
        self.assertEqual(
            [call.args[3] for call in evaluator.call_args_list],
            [0, 1],
        )
        updater.assert_not_called()
        self.assertEqual(storage_layout_snapshot(storage), before_layout)
        self.assertEqual(storage_bytes(storage), before_bytes)
        self.assertEqual(after_tree, before_tree)

    def test_failure_after_second_write_rolls_back_entire_traversal(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            _, session, storage = load_case(temporary_root)
            address = require_address(storage, (8, -1, -1))
            before_layout = storage_layout_snapshot(storage)
            before_bytes = storage_bytes(storage)
            before_tree = tree_snapshot(temporary_root)
            actual_apply = apply_tsdf_voxel_contribution

            def apply_then_fail(
                changed_storage: TsdfBlockStorage,
                contribution: object,
                changed_session: ScanSession,
            ) -> object:
                update = actual_apply(
                    changed_storage,
                    contribution,  # type: ignore[arg-type]
                    changed_session,
                )
                if update.contribution.observation_sequence == 1:
                    raise TsdfError("injected second update failure")
                return update

            with patch(
                "spatialforge.tsdf_voxel_traversal."
                "apply_tsdf_voxel_contribution",
                side_effect=apply_then_fail,
            ) as updater:
                with self.assertRaises(TsdfError) as raised:
                    traverse_tsdf_voxel_observations(
                        storage,
                        address,
                        session,
                    )

            after_tree = tree_snapshot(temporary_root)

        self.assertIn("injected second update failure", str(raised.exception))
        self.assertEqual(updater.call_count, 2)
        self.assertEqual(storage_layout_snapshot(storage), before_layout)
        self.assertEqual(storage_bytes(storage), before_bytes)
        self.assertEqual(after_tree, before_tree)
        self.assertEqual(float(storage.tsdf_sums[address.array_index_bzyx]), 0.0)
        self.assertFalse(np.signbit(storage.tsdf_sums[address.array_index_bzyx]))
        self.assertEqual(int(storage.weights[address.array_index_bzyx]), 0)

    def test_final_replay_mismatch_rolls_back_all_updates(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            _, session, storage = load_case(Path(temporary_directory))
            address = require_address(storage, (8, -1, -1))
            matched = replay_session(session)
            changed = replace(matched, digest_sha256="0" * 64)
            before_layout = storage_layout_snapshot(storage)
            before_bytes = storage_bytes(storage)

            with patch(
                "spatialforge.tsdf_voxel_traversal.replay_session",
                side_effect=(matched, matched, changed),
            ) as replay:
                with self.assertRaises(TsdfError) as raised:
                    traverse_tsdf_voxel_observations(
                        storage,
                        address,
                        session,
                    )

        self.assertIn("changed", str(raised.exception))
        self.assertEqual(replay.call_count, 3)
        self.assertEqual(storage_layout_snapshot(storage), before_layout)
        self.assertEqual(storage_bytes(storage), before_bytes)
        self.assertFalse(np.signbit(storage.tsdf_sums[address.array_index_bzyx]))

    def test_final_replay_target_mutation_is_detected_and_rolled_back(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            _, session, storage = load_case(Path(temporary_directory))
            address = require_address(storage, (8, -1, -1))
            matched = replay_session(session)
            before_layout = storage_layout_snapshot(storage)
            before_bytes = storage_bytes(storage)
            replay_calls = 0

            def mutate_during_final_replay(
                changed_session: ScanSession,
            ) -> object:
                nonlocal replay_calls
                self.assertIs(changed_session, session)
                replay_calls += 1
                if replay_calls == 3:
                    storage.tsdf_sums[address.array_index_bzyx] = 0.25
                return matched

            with patch(
                "spatialforge.tsdf_voxel_traversal.replay_session",
                side_effect=mutate_during_final_replay,
            ) as replay:
                with self.assertRaises(TsdfError):
                    traverse_tsdf_voxel_observations(
                        storage,
                        address,
                        session,
                    )

        self.assertEqual(replay.call_count, 3)
        self.assertEqual(storage_layout_snapshot(storage), before_layout)
        self.assertEqual(storage_bytes(storage), before_bytes)
        self.assertEqual(float(storage.tsdf_sums[address.array_index_bzyx]), 0.0)
        self.assertFalse(np.signbit(storage.tsdf_sums[address.array_index_bzyx]))
        self.assertEqual(int(storage.weights[address.array_index_bzyx]), 0)

    def test_final_replay_readonly_flip_reports_failed_rollback(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            _, session, storage = load_case(Path(temporary_directory))
            address = require_address(storage, (8, -1, -1))
            matched = replay_session(session)
            replay_calls = 0

            def make_weights_readonly_during_final_replay(
                changed_session: ScanSession,
            ) -> object:
                nonlocal replay_calls
                self.assertIs(changed_session, session)
                replay_calls += 1
                if replay_calls == 3:
                    storage.weights.setflags(write=False)
                return matched

            try:
                with patch(
                    "spatialforge.tsdf_voxel_traversal.replay_session",
                    side_effect=make_weights_readonly_during_final_replay,
                ) as replay:
                    with self.assertRaises(TsdfError) as raised:
                        traverse_tsdf_voxel_observations(
                            storage,
                            address,
                            session,
                        )
            finally:
                storage.weights.setflags(write=True)

        self.assertEqual(replay.call_count, 3)
        self.assertIn(
            "rollback failed; storage may be inconsistent",
            str(raised.exception),
        )
        self.assertTrue(storage.weights.flags.writeable)


class TsdfVoxelTraversalCliTests(unittest.TestCase):
    def test_cli_reports_exact_temporary_one_voxel_traversal(self) -> None:
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
            actual_traverse = traverse_tsdf_voxel_observations
            actual_evaluate = evaluate_tsdf_voxel_contribution
            actual_apply = apply_tsdf_voxel_contribution
            actual_zeros = np.zeros
            import spatialforge.tsdf_voxel_contribution as contribution_module

            actual_read_depth = contribution_module._read_depth

            def capture_allocation(
                plan: TsdfBlockPlan,
                session: ScanSession,
            ) -> TsdfBlockStorage:
                storage = actual_allocate(plan, session)
                captured_storage.append(storage)
                return storage

            with (
                forbidden_full_pipeline_calls() as forbidden,
                patch(
                    "spatialforge.cli.allocate_empty_tsdf_blocks",
                    side_effect=capture_allocation,
                ) as allocator,
                patch(
                    "spatialforge.cli.traverse_tsdf_voxel_observations",
                    wraps=actual_traverse,
                ) as traverser,
                patch(
                    "spatialforge.tsdf_voxel_traversal."
                    "evaluate_tsdf_voxel_contribution",
                    wraps=actual_evaluate,
                ) as evaluator,
                patch(
                    "spatialforge.tsdf_voxel_traversal."
                    "apply_tsdf_voxel_contribution",
                    wraps=actual_apply,
                ) as updater,
                patch(
                    "spatialforge.tsdf_block_storage.np.zeros",
                    wraps=actual_zeros,
                ) as zeros,
                patch(
                    "spatialforge.tsdf_voxel_contribution._read_depth",
                    wraps=actual_read_depth,
                ) as depth_reader,
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                exit_code = main(
                    [
                        "reconstruct",
                        "tsdf-block-voxel-traverse",
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
        traverser.assert_called_once()
        self.assertEqual(evaluator.call_count, 2)
        self.assertEqual(updater.call_count, 2)
        self.assertEqual(depth_reader.call_count, 2)
        self.assertEqual(zeros.call_count, 2)
        for forbidden_call in forbidden:
            forbidden_call.assert_not_called()
        self.assertEqual(len(captured_storage), 1)
        storage = captured_storage[0]
        address = require_address(storage, (8, -1, -1))
        self.assertEqual(address.storage_flat_index, 1016)
        self.assertEqual(
            float(storage.tsdf_sums[address.array_index_bzyx]),
            -0.24999999999999978,
        )
        self.assertEqual(
            int(storage.weights[address.array_index_bzyx]),
            2,
        )
        self.assertEqual(
            np.flatnonzero(storage.tsdf_sums.reshape(-1)).tolist(),
            [1016],
        )
        self.assertEqual(
            np.flatnonzero(storage.weights.reshape(-1)).tolist(),
            [1016],
        )
        expected_output = (
            "TSDF BLOCK VOXEL TRAVERSAL CHECK scan-synthetic-0001\n"
            "artifact: valid\n"
            "session_replay: matched\n"
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
            f"plan_sha256: {PLAN_SHA256}\n"
            f"replay_digest_sha256: {REPLAY_SHA256}\n"
        )
        self.assertEqual(stdout.getvalue(), expected_output)


if __name__ == "__main__":
    unittest.main()
