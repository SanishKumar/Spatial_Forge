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
    MAX_TSDF_PLAN_TRAVERSAL_OUTCOMES,
    TsdfPlanTraversalReceipt,
    allocate_empty_tsdf_blocks,
    build_tsdf_replay_depth_context,
    load_tsdf_block_plan,
    traverse_tsdf_block_voxels_from_context,
    traverse_tsdf_plan_blocks_from_context,
)
from spatialforge.cli import main
from spatialforge.errors import TsdfError
from spatialforge.model import ScanSession
from spatialforge.session_loader import load_scan_session
from spatialforge.tsdf_block_plan import plan_tsdf_blocks
from spatialforge.tsdf_block_plan_loader import TsdfBlockPlan
from spatialforge.tsdf_block_storage import (
    TSDF_BLOCK_VOXELS,
    TsdfBlockStorage,
)
from spatialforge.tsdf_block_traversal import TsdfBlockTraversalReceipt
from spatialforge.tsdf_replay_depth_context import TsdfReplayDepthContext
from spatialforge.tsdf_voxel_contribution import TsdfContributionStatus
from spatialforge.tsdf_voxel_traversal import TsdfVoxelTraversalReceipt


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
EXPECTED_BLOCKS = (
    (0, -1, -1),
    (1, -1, -1),
    (0, 0, -1),
    (1, 0, -1),
    (0, -1, 0),
    (1, -1, 0),
    (0, 0, 0),
    (1, 0, 0),
)

_FORBIDDEN_PLAN_TARGETS = (
    "spatialforge.tsdf_plan_traversal.replay_session",
    "spatialforge.tsdf_plan_traversal.build_tsdf_replay_depth_context",
    "spatialforge.tsdf_plan_traversal.allocate_empty_tsdf_blocks",
    "spatialforge.tsdf_block_traversal.replay_session",
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
    "spatialforge.tsdf_replay_depth_context."
    "build_tsdf_replay_depth_context",
    "spatialforge.tsdf_block_storage.allocate_empty_tsdf_blocks",
    "spatialforge.tsdf_block_storage.np.zeros",
)

_FORBIDDEN_PLAN_CLI_TARGETS = (
    "spatialforge.cli.traverse_tsdf_block_voxels_from_context",
    "spatialforge.cli.traverse_tsdf_voxel_observations",
    "spatialforge.cli.traverse_tsdf_voxel_observations_from_context",
    "spatialforge.cli.evaluate_tsdf_voxel_contribution",
    "spatialforge.cli.apply_tsdf_voxel_contribution",
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
) -> Path:
    output = parent / name
    plan_tsdf_blocks(
        load_scan_session(session_path),
        output,
        **PLAN_ARGUMENTS,
    )
    return output


def load_case(
    parent: Path,
    *,
    session_path: Path = FIXTURE,
) -> tuple[
    TsdfBlockPlan,
    ScanSession,
    TsdfBlockStorage,
    TsdfReplayDepthContext,
]:
    plan = load_tsdf_block_plan(
        create_plan(parent, session_path=session_path)
    )
    session = load_scan_session(session_path)
    storage = allocate_empty_tsdf_blocks(plan, session)
    context = build_tsdf_replay_depth_context(plan, session)
    return plan, session, storage, context


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
def forbidden_plan_calls():
    with ExitStack() as stack:
        mocks: list[MagicMock] = []
        for target in _FORBIDDEN_PLAN_TARGETS:
            mocks.append(stack.enter_context(patch(target, create=True)))
        yield mocks


@contextmanager
def forbidden_plan_cli_calls():
    with ExitStack() as stack:
        mocks: list[MagicMock] = []
        for target in _FORBIDDEN_PLAN_CLI_TARGETS:
            mocks.append(stack.enter_context(patch(target, create=True)))
        yield mocks


class TsdfContextPlanTraversalTests(unittest.TestCase):
    def test_exact_reference_parity_order_context_reuse_and_no_source_work(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            plan, session, expected_storage, context = load_case(
                temporary_root
            )
            actual_storage = allocate_empty_tsdf_blocks(plan, session)
            expected_receipts = tuple(
                traverse_tsdf_block_voxels_from_context(
                    expected_storage,
                    block_index_xyz,
                    context,
                )
                for block_index_xyz in plan.active_blocks
            )
            before_layout = storage_layout_snapshot(actual_storage)
            before_tree = tree_snapshot(temporary_root)
            context_before = context_snapshot(context)
            actual_child = traverse_tsdf_block_voxels_from_context

            with (
                forbidden_plan_calls() as forbidden,
                patch(
                    "spatialforge.tsdf_plan_traversal."
                    "traverse_tsdf_block_voxels_from_context",
                    wraps=actual_child,
                ) as child,
            ):
                actual = traverse_tsdf_plan_blocks_from_context(
                    actual_storage,
                    context,
                )
            after_tree = tree_snapshot(temporary_root)
            context_after = context_snapshot(context)

        self.assertIsInstance(actual, TsdfPlanTraversalReceipt)
        self.assertEqual(actual.block_indices, EXPECTED_BLOCKS)
        self.assertEqual(actual.block_receipts, expected_receipts)
        self.assertEqual(actual.source_plan_digest_sha256, PLAN_SHA256)
        self.assertEqual(actual.replay_digest_sha256, REPLAY_SHA256)
        self.assertEqual(actual.block_resolution, 8)
        self.assertEqual(actual.planned_voxel_slots, 4096)
        self.assertEqual(actual.frame_stride, 1)
        self.assertEqual(actual.total_observations, 2)
        self.assertEqual(actual.selected_observation_sequences, (0, 1))
        self.assertEqual(actual.block_count, 8)
        self.assertEqual(actual.voxel_count, 4096)
        self.assertEqual(actual.evaluated_count, 8192)
        self.assertEqual(actual.applied_count, 1168)
        self.assertEqual(actual.skipped_count, 7024)
        self.assertEqual(actual.storage_slots_updated, 584)
        self.assertEqual(actual.weight_delta, 1168)
        self.assertEqual(actual.maximum_weight_after, 2)
        self.assertEqual(actual.nonzero_sum_count, 584)
        self.assertEqual(actual.observed_voxel_count, 584)
        self.assertEqual(actual.unknown_voxel_count, 3512)
        self.assertTrue(actual.prepared_depth_accessed)
        self.assertEqual(
            actual.status_counts,
            (
                (TsdfContributionStatus.CONTRIBUTES, 1168),
                (TsdfContributionStatus.PROJECTION_OUTSIDE_IMAGE, 5440),
                (TsdfContributionStatus.BEHIND_TRUNCATION, 1584),
            ),
        )
        self.assertEqual(
            tuple(
                (receipt.applied_count, receipt.storage_slots_updated)
                for receipt in actual.block_receipts
            ),
            (
                (88, 44),
                (204, 102),
                (88, 44),
                (204, 102),
                (88, 44),
                (204, 102),
                (88, 44),
                (204, 102),
            ),
        )
        self.assertEqual(storage_bytes(actual_storage), storage_bytes(expected_storage))
        self.assertEqual(storage_layout_snapshot(actual_storage), before_layout)
        self.assertEqual(after_tree, before_tree)
        self.assertEqual(context_after, context_before)
        self.assertEqual(child.call_count, 8)
        self.assertEqual(
            tuple(call.args[1] for call in child.call_args_list),
            EXPECTED_BLOCKS,
        )
        self.assertTrue(
            all(call.args[2] is context for call in child.call_args_list)
        )
        self.assertEqual(
            tuple(receipt.block_row for receipt in actual.block_receipts),
            tuple(range(8)),
        )
        first = actual.block_receipts[0].voxel_receipts[0].address
        last = actual.block_receipts[-1].voxel_receipts[-1].address
        self.assertEqual(first.global_index_xyz, (0, -8, -8))
        self.assertEqual(first.storage_flat_index, 0)
        self.assertEqual(last.global_index_xyz, (15, 7, 7))
        self.assertEqual(last.storage_flat_index, 4095)
        self.assertEqual(actual_storage.nonzero_sum_count, 584)
        self.assertEqual(actual_storage.nonzero_weight_count, 584)
        self.assertEqual(actual_storage.unknown_voxel_count, 3512)
        self.assertEqual(
            int(np.sum(actual_storage.weights, dtype=np.uint64)),
            1168,
        )
        for forbidden_call in forbidden:
            forbidden_call.assert_not_called()

        after_success = storage_bytes(actual_storage)
        with patch(
            "spatialforge.tsdf_plan_traversal."
            "traverse_tsdf_block_voxels_from_context"
        ) as repeated_child:
            with self.assertRaises(TsdfError) as repeated_error:
                traverse_tsdf_plan_blocks_from_context(
                    actual_storage,
                    context,
                )
        self.assertIn("empty", str(repeated_error.exception))
        repeated_child.assert_not_called()
        self.assertEqual(storage_bytes(actual_storage), after_success)

    def test_whole_plan_empty_state_is_preflighted_before_first_child(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            plan, session, _, context = load_case(Path(temporary_directory))

            def nonzero_sum(storage: TsdfBlockStorage) -> None:
                storage.tsdf_sums[7, 7, 7, 7] = 0.25

            def negative_zero(storage: TsdfBlockStorage) -> None:
                storage.tsdf_sums[7, 7, 7, 7] = -0.0

            def nonzero_weight(storage: TsdfBlockStorage) -> None:
                storage.weights[7, 7, 7, 7] = 1

            for name, mutate, expected_message in (
                ("nonzero-sum", nonzero_sum, "canonical empty storage"),
                ("negative-zero", negative_zero, "canonical empty storage"),
                ("nonzero-weight", nonzero_weight, "canonical empty storage"),
            ):
                with self.subTest(name=name):
                    storage = allocate_empty_tsdf_blocks(plan, session)
                    mutate(storage)
                    before_layout = storage_layout_snapshot(storage)
                    before_bytes = storage_bytes(storage)
                    with patch(
                        "spatialforge.tsdf_plan_traversal."
                        "traverse_tsdf_block_voxels_from_context"
                    ) as child:
                        with self.assertRaises(TsdfError) as raised:
                            traverse_tsdf_plan_blocks_from_context(
                                storage,
                                context,
                            )
                    self.assertIn(expected_message, str(raised.exception))
                    child.assert_not_called()
                    self.assertEqual(
                        storage_layout_snapshot(storage),
                        before_layout,
                    )
                    self.assertEqual(storage_bytes(storage), before_bytes)

    def test_retained_outcome_limit_rejects_before_first_child(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            _, _, storage, context = load_case(Path(temporary_directory))
            before_layout = storage_layout_snapshot(storage)
            before_bytes = storage_bytes(storage)
            self.assertEqual(MAX_TSDF_PLAN_TRAVERSAL_OUTCOMES, 262_144)

            with (
                patch(
                    "spatialforge.tsdf_plan_traversal."
                    "MAX_TSDF_PLAN_TRAVERSAL_OUTCOMES",
                    8191,
                ),
                patch(
                    "spatialforge.tsdf_plan_traversal."
                    "traverse_tsdf_block_voxels_from_context"
                ) as child,
            ):
                with self.assertRaises(TsdfError) as raised:
                    traverse_tsdf_plan_blocks_from_context(storage, context)

        self.assertIn("8192 retained contribution outcomes", str(raised.exception))
        self.assertIn("maximum is 8191", str(raised.exception))
        child.assert_not_called()
        self.assertEqual(storage_layout_snapshot(storage), before_layout)
        self.assertEqual(storage_bytes(storage), before_bytes)

    def test_types_provenance_and_layout_reject_before_first_child(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            plan, session, source, context = load_case(
                Path(temporary_directory)
            )
            changed_context = replace(
                context,
                source_plan_digest_sha256="0" * 64,
            )
            for name, candidate_storage, candidate_context, message in (
                (
                    "storage-type",
                    object(),
                    context,
                    "TsdfBlockStorage",
                ),
                (
                    "context-type",
                    source,
                    object(),
                    "TsdfReplayDepthContext",
                ),
                (
                    "context-plan",
                    source,
                    changed_context,
                    "source plan digest",
                ),
            ):
                with self.subTest(name=name):
                    before = storage_bytes(source)
                    with patch(
                        "spatialforge.tsdf_plan_traversal."
                        "traverse_tsdf_block_voxels_from_context"
                    ) as child:
                        with self.assertRaises(TsdfError) as raised:
                            traverse_tsdf_plan_blocks_from_context(
                                candidate_storage,  # type: ignore[arg-type]
                                candidate_context,  # type: ignore[arg-type]
                            )
                    self.assertIn(message, str(raised.exception))
                    child.assert_not_called()
                    self.assertEqual(storage_bytes(source), before)

            invalid: list[tuple[str, TsdfBlockStorage, str]] = []
            readonly = allocate_empty_tsdf_blocks(plan, session)
            readonly.weights.setflags(write=False)
            invalid.append(
                (
                    "readonly",
                    readonly,
                    "writable sum and weight arrays",
                )
            )

            backing = np.zeros(source.tsdf_sums.nbytes, dtype=np.uint8)
            aliased = TsdfBlockStorage(
                source_plan=plan,
                block_indices=source.block_indices,
                tsdf_sums=np.ndarray(
                    shape=source.tsdf_sums.shape,
                    dtype=np.float64,
                    buffer=backing,
                ),
                weights=np.ndarray(
                    shape=source.weights.shape,
                    dtype=np.uint32,
                    buffer=backing,
                ),
            )
            invalid.append(
                (
                    "aliased",
                    aliased,
                    "non-overlapping storage arrays",
                )
            )

            malformed = allocate_empty_tsdf_blocks(plan, session)
            malformed.tsdf_sums.shape = (malformed.tsdf_sums.size,)
            invalid.append(("malformed", malformed, "canonical"))

            for name, storage, message in invalid:
                with self.subTest(name=name):
                    before = storage_bytes(storage)
                    with patch(
                        "spatialforge.tsdf_plan_traversal."
                        "traverse_tsdf_block_voxels_from_context"
                    ) as child:
                        with self.assertRaises(TsdfError) as raised:
                            traverse_tsdf_plan_blocks_from_context(
                                storage,
                                context,
                            )
                    self.assertIn(message, str(raised.exception))
                    child.assert_not_called()
                    self.assertEqual(storage_bytes(storage), before)

    def test_late_failures_restore_every_planned_row(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            plan, session, _, context = load_case(temporary_root)
            actual_child = traverse_tsdf_block_voxels_from_context

            for name, raised_error, expected_message in (
                (
                    "domain",
                    TsdfError("injected final planned block failure"),
                    "injected final planned block failure",
                ),
                (
                    "unexpected",
                    RuntimeError(
                        "injected unexpected final planned block failure"
                    ),
                    "cannot apply context-bound TSDF plan traversal",
                ),
                (
                    "memory",
                    MemoryError(
                        "injected final planned block memory failure"
                    ),
                    "cannot apply context-bound TSDF plan traversal",
                ),
            ):
                with self.subTest(name=name):
                    storage = allocate_empty_tsdf_blocks(plan, session)
                    before_layout = storage_layout_snapshot(storage)
                    before_bytes = storage_bytes(storage)
                    before_tree = tree_snapshot(temporary_root)
                    context_before = context_snapshot(context)

                    def traverse_then_fail(
                        changed_storage: TsdfBlockStorage,
                        block_index_xyz: tuple[int, int, int],
                        changed_context: TsdfReplayDepthContext,
                    ) -> TsdfBlockTraversalReceipt:
                        receipt = actual_child(
                            changed_storage,
                            block_index_xyz,
                            changed_context,
                        )
                        if receipt.block_row == 7:
                            raise raised_error
                        return receipt

                    with (
                        forbidden_plan_calls() as forbidden,
                        patch(
                            "spatialforge.tsdf_plan_traversal."
                            "traverse_tsdf_block_voxels_from_context",
                            side_effect=traverse_then_fail,
                        ) as child,
                    ):
                        with self.assertRaises(TsdfError) as raised:
                            traverse_tsdf_plan_blocks_from_context(
                                storage,
                                context,
                            )
                    after_tree = tree_snapshot(temporary_root)
                    context_after = context_snapshot(context)

                    self.assertIn(expected_message, str(raised.exception))
                    self.assertEqual(child.call_count, 8)
                    self.assertEqual(
                        storage_layout_snapshot(storage),
                        before_layout,
                    )
                    self.assertEqual(storage_bytes(storage), before_bytes)
                    self.assertEqual(storage.tsdf_sums.tobytes(), bytes(32768))
                    self.assertEqual(storage.weights.tobytes(), bytes(16384))
                    self.assertEqual(after_tree, before_tree)
                    self.assertEqual(context_after, context_before)
                    if name in {"unexpected", "memory"}:
                        self.assertIs(raised.exception.__cause__, raised_error)
                    for forbidden_call in forbidden:
                        forbidden_call.assert_not_called()

    def test_receipt_is_strict_immutable_and_cannot_be_empty(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            _, _, storage, context = load_case(Path(temporary_directory))
            receipt = traverse_tsdf_plan_blocks_from_context(storage, context)

        self.assertFalse(hasattr(receipt, "__dict__"))
        with self.assertRaises(FrozenInstanceError):
            receipt.block_indices = ()  # type: ignore[misc]

        invalid_rows = (
            (receipt.block_indices[:-1], receipt.block_receipts[:-1]),
            (
                (),
                (),
            ),
            (
                (
                    receipt.block_indices[1],
                    receipt.block_indices[0],
                    *receipt.block_indices[2:],
                ),
                (
                    receipt.block_receipts[1],
                    receipt.block_receipts[0],
                    *receipt.block_receipts[2:],
                ),
            ),
            (
                receipt.block_indices,
                (
                    receipt.block_receipts[0],
                    receipt.block_receipts[0],
                    *receipt.block_receipts[2:],
                ),
            ),
        )
        for block_indices, block_receipts in invalid_rows:
            with self.subTest(length=len(block_receipts)):
                with self.assertRaises(TsdfError):
                    replace(
                        receipt,
                        block_indices=block_indices,
                        block_receipts=block_receipts,
                    )

        for field, value in (
            ("source_plan_digest_sha256", "0" * 64),
            ("replay_digest_sha256", "0" * 64),
            ("block_resolution", 7),
            ("planned_voxel_slots", 3584),
            ("frame_stride", 2),
            ("total_observations", 3),
            ("selected_observation_sequences", (1,)),
        ):
            with self.subTest(field=field):
                with self.assertRaises(TsdfError):
                    replace(receipt, **{field: value})

    def test_final_cross_row_corruption_is_detected_and_fully_restored(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            _, _, storage, context = load_case(Path(temporary_directory))
            before_layout = storage_layout_snapshot(storage)
            before_bytes = storage_bytes(storage)
            actual_child = traverse_tsdf_block_voxels_from_context

            def corrupt_first_row_after_last_child(
                changed_storage: TsdfBlockStorage,
                block_index_xyz: tuple[int, int, int],
                changed_context: TsdfReplayDepthContext,
            ) -> TsdfBlockTraversalReceipt:
                receipt = actual_child(
                    changed_storage,
                    block_index_xyz,
                    changed_context,
                )
                if receipt.block_row == 7:
                    changed_storage.tsdf_sums[0, 0, 0, 0] = 1.0
                return receipt

            with patch(
                "spatialforge.tsdf_plan_traversal."
                "traverse_tsdf_block_voxels_from_context",
                side_effect=corrupt_first_row_after_last_child,
            ) as child:
                with self.assertRaises(TsdfError) as raised:
                    traverse_tsdf_plan_blocks_from_context(storage, context)

        self.assertIn(
            "storage does not match its block receipts",
            str(raised.exception),
        )
        self.assertEqual(child.call_count, 8)
        self.assertEqual(storage_layout_snapshot(storage), before_layout)
        self.assertEqual(storage_bytes(storage), before_bytes)
        self.assertEqual(storage.tsdf_sums.tobytes(), bytes(32768))
        self.assertEqual(storage.weights.tobytes(), bytes(16384))

    def test_one_row_plan_is_supported_without_multirow_assumptions(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            plan, _, _, context = load_case(Path(temporary_directory))
            selected_block = (1, -1, -1)
            one_row_plan = replace(
                plan,
                surface_blocks=(selected_block,),
                active_blocks=(selected_block,),
                planned_voxel_slots=TSDF_BLOCK_VOXELS,
                min_block_index=selected_block,
                max_block_index=selected_block,
            )
            storage = TsdfBlockStorage(
                source_plan=one_row_plan,
                block_indices=(selected_block,),
                tsdf_sums=np.zeros((1, 8, 8, 8), dtype=np.float64),
                weights=np.zeros((1, 8, 8, 8), dtype=np.uint32),
            )
            actual_child = traverse_tsdf_block_voxels_from_context
            with patch(
                "spatialforge.tsdf_plan_traversal."
                "traverse_tsdf_block_voxels_from_context",
                wraps=actual_child,
            ) as child:
                receipt = traverse_tsdf_plan_blocks_from_context(
                    storage,
                    context,
                )

        self.assertEqual(child.call_count, 1)
        self.assertEqual(receipt.block_indices, (selected_block,))
        self.assertEqual(receipt.block_count, 1)
        self.assertEqual(receipt.planned_voxel_slots, 512)
        self.assertEqual(receipt.voxel_count, 512)
        self.assertEqual(receipt.evaluated_count, 1024)
        self.assertEqual(receipt.applied_count, 204)
        self.assertEqual(receipt.skipped_count, 820)
        self.assertEqual(receipt.storage_slots_updated, 102)
        self.assertEqual(receipt.weight_delta, 204)
        self.assertEqual(receipt.block_receipts[0].block_row, 0)
        self.assertEqual(
            receipt.block_receipts[0]
            .voxel_receipts[0]
            .address
            .storage_flat_index,
            0,
        )
        self.assertEqual(
            receipt.block_receipts[0]
            .voxel_receipts[-1]
            .address
            .storage_flat_index,
            511,
        )


class TsdfContextPlanTraversalCliTests(unittest.TestCase):
    def test_cli_reports_exact_isolated_plan_traversal(self) -> None:
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
            actual_plan_traverse = traverse_tsdf_plan_blocks_from_context
            actual_block_traverse = traverse_tsdf_block_voxels_from_context
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
                context: TsdfReplayDepthContext,
            ) -> TsdfPlanTraversalReceipt:
                with forbidden_plan_calls() as inner_forbidden:
                    receipt = actual_plan_traverse(storage, context)
                for forbidden_call in inner_forbidden:
                    forbidden_call.assert_not_called()
                return receipt

            with (
                forbidden_plan_cli_calls() as forbidden_cli,
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
                    "traverse_tsdf_plan_blocks_from_context",
                    side_effect=traverse_isolated,
                ) as traverser,
                patch(
                    "spatialforge.tsdf_plan_traversal."
                    "traverse_tsdf_block_voxels_from_context",
                    wraps=actual_block_traverse,
                ) as block_child,
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
                        "tsdf-block-context-plan-traverse",
                        str(plan_path),
                        str(session_path),
                    ]
                )
            after_tree = tree_snapshot(temporary_root)

        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(before_tree, after_tree)
        allocator.assert_called_once()
        builder.assert_called_once()
        traverser.assert_called_once()
        self.assertEqual(block_child.call_count, 8)
        self.assertEqual(zeros.call_count, 2)
        for forbidden_call in forbidden_cli:
            forbidden_call.assert_not_called()

        self.assertEqual(len(captured_storage), 1)
        storage = captured_storage[0]
        self.assertEqual(storage.block_indices, EXPECTED_BLOCKS)
        self.assertEqual(storage.nonzero_sum_count, 584)
        self.assertEqual(storage.nonzero_weight_count, 584)
        self.assertEqual(storage.unknown_voxel_count, 3512)

        expected_output = (
            "TSDF BLOCK CONTEXT PLAN TRAVERSAL CHECK "
            "scan-synthetic-0001\n"
            "artifact: valid\n"
            "session_replay: matched\n"
            "context_selection: frame_stride=1 total=2 selected=2\n"
            "context_immutable: yes\n"
            "depth_source: replay-depth-context\n"
            "plan_blocks: active=8 surface=4 halo=4 resolution=8 "
            "voxel_slots=4096\n"
            "block_order: plan-canonical-x-fastest rows=0..7\n"
            "voxel_order: block-row-then-local-flat-x-fastest "
            "local_flat=0..511\n"
            "first_block: index=(0, -1, -1) row=0 "
            "storage_flat_range=0..511\n"
            "last_block: index=(1, 0, 0) row=7 "
            "storage_flat_range=3584..4095\n"
            "first_voxel: global=(0, -8, -8) local=(0, 0, 0) "
            "array=(0, 0, 0, 0) storage_flat=0\n"
            "last_voxel: global=(15, 7, 7) local=(7, 7, 7) "
            "array=(7, 7, 7, 7) storage_flat=4095\n"
            "selection: frame_stride=1 total=2 selected=2\n"
            "storage_before: nonzero_sums=0 nonzero_weights=0 "
            "unknown_voxels=4096\n"
            "storage_after: nonzero_sums=584 nonzero_weights=584 "
            "unknown_voxels=3512\n"
            "status_counts: contributes=1168 "
            "projection-outside-image=5440 behind-truncation=1584\n"
            "plan_weight_sum_after: 1168\n"
            "plan_max_weight_after: 2\n"
            "context_provenance: matched\n"
            "traversal_source_freshness: construction-time-context\n"
            "traversal_session_replay: no\n"
            "traversal_replay_hashing: no\n"
            "traversal_source_io: no\n"
            "traversal_depth_decoding: no\n"
            "traversal_prepared_depth_access: yes\n"
            "traversal_workload: retained_outcomes=8192 maximum=262144\n"
            "blocks_traversed: 8\n"
            "block_transcripts_retained: 8\n"
            "voxel_addresses_traversed: 4096\n"
            "voxel_transcripts_retained: 4096\n"
            "voxel_observation_traversals: 4096\n"
            "contributions_evaluated: 8192\n"
            "contributions_applied: 1168\n"
            "contributions_skipped: 7024\n"
            "storage_slots_updated: 584\n"
            "voxel_observation_traversal_performed: yes\n"
            "voxel_address_traversal_performed: yes\n"
            "fusion_block_traversal_performed: yes\n"
            "fusion_block_traversal_scope: existing-plan-block-set-only\n"
            "multiple_block_traversal_performed: yes\n"
            "planned_block_set_traversal_performed: yes\n"
            "all_existing_plan_blocks_traversed: yes\n"
            "unplanned_blocks_visited: 0\n"
            "free_space_coverage_planned: no\n"
            "ray_traversal_performed: no\n"
            "full_fusion_performed: no\n"
            "missing_blocks_created: no\n"
            "caught_failure_rollback_scope: complete-planned-storage\n"
            "artifact_written: no\n"
            "storage_persisted: no\n"
            "context_persisted: no\n"
            f"plan_sha256: {PLAN_SHA256}\n"
            f"replay_digest_sha256: {REPLAY_SHA256}\n"
        )
        self.assertEqual(stdout.getvalue(), expected_output)

    def test_cli_one_row_plan_reports_no_multiple_block_traversal(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            session_path = copy_fixture(temporary_root)
            plan_path = create_plan(
                temporary_root,
                session_path=session_path,
            )
            plan = load_tsdf_block_plan(plan_path)
            session = load_scan_session(session_path)
            context = build_tsdf_replay_depth_context(plan, session)
            selected_block = (1, -1, -1)
            one_row_plan = replace(
                plan,
                surface_blocks=(selected_block,),
                active_blocks=(selected_block,),
                planned_voxel_slots=TSDF_BLOCK_VOXELS,
                min_block_index=selected_block,
                max_block_index=selected_block,
            )
            storage = TsdfBlockStorage(
                source_plan=one_row_plan,
                block_indices=(selected_block,),
                tsdf_sums=np.zeros((1, 8, 8, 8), dtype=np.float64),
                weights=np.zeros((1, 8, 8, 8), dtype=np.uint32),
            )
            before_tree = tree_snapshot(temporary_root)
            stdout = io.StringIO()
            stderr = io.StringIO()

            with (
                patch(
                    "spatialforge.cli.load_tsdf_block_plan",
                    return_value=one_row_plan,
                ),
                patch(
                    "spatialforge.cli.load_scan_session",
                    return_value=session,
                ),
                patch(
                    "spatialforge.cli.allocate_empty_tsdf_blocks",
                    return_value=storage,
                ),
                patch(
                    "spatialforge.cli.build_tsdf_replay_depth_context",
                    return_value=context,
                ),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                exit_code = main(
                    [
                        "reconstruct",
                        "tsdf-block-context-plan-traverse",
                        str(plan_path),
                        str(session_path),
                    ]
                )
            after_tree = tree_snapshot(temporary_root)

        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(after_tree, before_tree)
        output = stdout.getvalue()
        self.assertIn("blocks_traversed: 1\n", output)
        self.assertIn("block_transcripts_retained: 1\n", output)
        self.assertIn("multiple_block_traversal_performed: no\n", output)
        self.assertNotIn("multiple_block_traversal_performed: yes\n", output)
        self.assertIn("planned_block_set_traversal_performed: yes\n", output)

    def test_cli_reports_plan_traversal_failure_without_traceback(self) -> None:
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
                    "traverse_tsdf_plan_blocks_from_context",
                    side_effect=TsdfError("injected plan traversal failure"),
                ) as traverser,
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                exit_code = main(
                    [
                        "reconstruct",
                        "tsdf-block-context-plan-traverse",
                        str(plan_path),
                        str(session_path),
                    ]
                )
            after_tree = tree_snapshot(temporary_root)

        self.assertEqual(exit_code, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(before_tree, after_tree)
        traverser.assert_called_once()
        error = stderr.getvalue()
        self.assertIn("TSDF BLOCK CONTEXT PLAN TRAVERSAL FAILED", error)
        self.assertIn("injected plan traversal failure", error)
        self.assertNotIn("Traceback", error)


if __name__ == "__main__":
    unittest.main()
