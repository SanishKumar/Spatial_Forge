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
    TsdfBlockTraversalReceipt,
    allocate_empty_tsdf_blocks,
    build_tsdf_replay_depth_context,
    compose_tsdf_global_voxel_index,
    load_tsdf_block_plan,
    locate_tsdf_voxel,
    traverse_tsdf_block_voxels_from_context,
    traverse_tsdf_voxel_observations_from_context,
)
from spatialforge.cli import main
from spatialforge.errors import TsdfError
from spatialforge.model import ScanSession
from spatialforge.session_loader import load_scan_session
from spatialforge.tsdf_block_plan import (
    MAX_BLOCK_INDEX,
    MIN_BLOCK_INDEX,
    TSDF_BLOCK_RESOLUTION,
    plan_tsdf_blocks,
)
from spatialforge.tsdf_block_plan_loader import TsdfBlockPlan
from spatialforge.tsdf_block_storage import (
    TSDF_BLOCK_VOXELS,
    TsdfBlockStorage,
)
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
SELECTED_BLOCK = (1, -1, -1)
PLAN_SHA256 = (
    "372c7c5d49eff1a30317ceb8b67cb3c40683f1d9d2a6179a049772f763d6f79d"
)
REPLAY_SHA256 = (
    "dc001ae0ca01004a21ad227b12d57a7350bbc23904040453ca73fd955ef050b8"
)

_FORBIDDEN_BLOCK_TARGETS = (
    "spatialforge.tsdf_block_traversal.replay_session",
    "spatialforge.tsdf_block_traversal.build_tsdf_replay_depth_context",
    "spatialforge.tsdf_block_traversal.allocate_empty_tsdf_blocks",
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

_FORBIDDEN_CONTEXT_CLI_TARGETS = (
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


def canonical_block_addresses(
    storage: TsdfBlockStorage,
    block_index_xyz: tuple[int, int, int] = SELECTED_BLOCK,
) -> tuple[TsdfVoxelAddress, ...]:
    addresses: list[TsdfVoxelAddress] = []
    for local_flat_index in range(TSDF_BLOCK_VOXELS):
        local_index_xyz = (
            local_flat_index % TSDF_BLOCK_RESOLUTION,
            (local_flat_index // TSDF_BLOCK_RESOLUTION)
            % TSDF_BLOCK_RESOLUTION,
            local_flat_index // (
                TSDF_BLOCK_RESOLUTION * TSDF_BLOCK_RESOLUTION
            ),
        )
        addresses.append(
            require_address(
                storage,
                compose_tsdf_global_voxel_index(
                    block_index_xyz,
                    local_index_xyz,
                ),
            )
        )
    return tuple(addresses)


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


def seed_unselected_blocks(
    storage: TsdfBlockStorage,
    selected_row: int,
) -> None:
    for block_row in range(storage.block_count):
        if block_row == selected_row:
            continue
        storage.tsdf_sums[block_row].fill((block_row + 1) / 16.0)
        storage.weights[block_row].fill(block_row + 1)


@contextmanager
def forbidden_block_calls():
    with ExitStack() as stack:
        mocks: list[MagicMock] = []
        for target in _FORBIDDEN_BLOCK_TARGETS:
            mocks.append(stack.enter_context(patch(target, create=True)))
        yield mocks


@contextmanager
def forbidden_context_cli_calls():
    with ExitStack() as stack:
        mocks: list[MagicMock] = []
        for target in _FORBIDDEN_CONTEXT_CLI_TARGETS:
            mocks.append(stack.enter_context(patch(target)))
        yield mocks


class TsdfContextBlockTraversalTests(unittest.TestCase):
    def test_exact_reference_parity_canonical_order_and_isolation(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            plan, session, expected_storage, context = load_case(
                temporary_root
            )
            actual_storage = allocate_empty_tsdf_blocks(plan, session)
            selected_row = plan.active_blocks.index(SELECTED_BLOCK)
            seed_unselected_blocks(expected_storage, selected_row)
            seed_unselected_blocks(actual_storage, selected_row)
            unselected_before = tuple(
                (
                    actual_storage.tsdf_sums[block_row].tobytes(),
                    actual_storage.weights[block_row].tobytes(),
                )
                for block_row in range(actual_storage.block_count)
                if block_row != selected_row
            )

            expected_receipts = tuple(
                traverse_tsdf_voxel_observations_from_context(
                    expected_storage,
                    address,
                    context,
                )
                for address in canonical_block_addresses(expected_storage)
            )
            before_layout = storage_layout_snapshot(actual_storage)
            before_tree = tree_snapshot(temporary_root)
            context_before = context_snapshot(context)
            actual_child = traverse_tsdf_voxel_observations_from_context

            with (
                forbidden_block_calls() as forbidden,
                patch(
                    "spatialforge.tsdf_block_traversal."
                    "traverse_tsdf_voxel_observations_from_context",
                    wraps=actual_child,
                ) as child,
            ):
                actual = traverse_tsdf_block_voxels_from_context(
                    actual_storage,
                    SELECTED_BLOCK,
                    context,
                )
            after_tree = tree_snapshot(temporary_root)
            context_after = context_snapshot(context)

        self.assertIsInstance(actual, TsdfBlockTraversalReceipt)
        self.assertEqual(actual.block_index_xyz, SELECTED_BLOCK)
        self.assertEqual(actual.block_row, 1)
        self.assertEqual(actual.source_plan_digest_sha256, PLAN_SHA256)
        self.assertEqual(actual.replay_digest_sha256, REPLAY_SHA256)
        self.assertEqual(actual.frame_stride, 1)
        self.assertEqual(actual.total_observations, 2)
        self.assertEqual(actual.selected_observation_sequences, (0, 1))
        self.assertEqual(actual.voxel_receipts, expected_receipts)
        self.assertEqual(actual.voxel_count, TSDF_BLOCK_VOXELS)
        self.assertEqual(actual.evaluated_count, 1024)
        self.assertEqual(actual.applied_count, 204)
        self.assertEqual(actual.skipped_count, 820)
        self.assertEqual(actual.storage_slots_updated, 102)
        self.assertEqual(actual.weight_delta, 204)
        self.assertEqual(
            actual.status_counts,
            (
                (TsdfContributionStatus.CONTRIBUTES, 204),
                (TsdfContributionStatus.PROJECTION_OUTSIDE_IMAGE, 424),
                (TsdfContributionStatus.BEHIND_TRUNCATION, 396),
            ),
        )

        self.assertEqual(storage_bytes(actual_storage), storage_bytes(expected_storage))
        self.assertEqual(storage_layout_snapshot(actual_storage), before_layout)
        self.assertEqual(after_tree, before_tree)
        self.assertEqual(context_after, context_before)
        self.assertEqual(child.call_count, TSDF_BLOCK_VOXELS)
        self.assertEqual(
            tuple(
                (
                    actual_storage.tsdf_sums[block_row].tobytes(),
                    actual_storage.weights[block_row].tobytes(),
                )
                for block_row in range(actual_storage.block_count)
                if block_row != selected_row
            ),
            unselected_before,
        )

        called_addresses = tuple(call.args[1] for call in child.call_args_list)
        self.assertEqual(
            tuple(address.local_flat_index for address in called_addresses),
            tuple(range(TSDF_BLOCK_VOXELS)),
        )
        self.assertEqual(
            tuple(address.storage_flat_index for address in called_addresses),
            tuple(range(512, 1024)),
        )
        self.assertTrue(
            all(
                address.block_index_xyz == SELECTED_BLOCK
                and address.block_row == selected_row
                for address in called_addresses
            )
        )
        self.assertTrue(
            all(call.args[2] is context for call in child.call_args_list)
        )
        first = actual.voxel_receipts[0].address
        last = actual.voxel_receipts[-1].address
        self.assertEqual(first.global_index_xyz, (8, -8, -8))
        self.assertEqual(first.local_index_xyz, (0, 0, 0))
        self.assertEqual(first.array_index_bzyx, (1, 0, 0, 0))
        self.assertEqual(first.storage_flat_index, 512)
        self.assertEqual(last.global_index_xyz, (15, -1, -1))
        self.assertEqual(last.local_index_xyz, (7, 7, 7))
        self.assertEqual(last.array_index_bzyx, (1, 7, 7, 7))
        self.assertEqual(last.storage_flat_index, 1023)

        self.assertEqual(
            np.count_nonzero(actual_storage.tsdf_sums[selected_row]),
            102,
        )
        self.assertEqual(
            np.count_nonzero(actual_storage.weights[selected_row]),
            102,
        )
        self.assertEqual(
            int(
                np.sum(
                    actual_storage.weights[selected_row],
                    dtype=np.uint64,
                )
            ),
            204,
        )
        self.assertEqual(
            float(
                np.sum(
                    actual_storage.tsdf_sums[selected_row],
                    dtype=np.float64,
                )
            ),
            -116.99999999999997,
        )
        for forbidden_call in forbidden:
            forbidden_call.assert_not_called()

    def test_entire_block_empty_state_is_preflighted_before_first_child(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            plan, session, _, context = load_case(temporary_root)
            selected_row = plan.active_blocks.index(SELECTED_BLOCK)

            def nonzero_sum(storage: TsdfBlockStorage) -> None:
                storage.tsdf_sums[selected_row, 7, 7, 7] = 0.25

            def negative_zero(storage: TsdfBlockStorage) -> None:
                storage.tsdf_sums[selected_row, 7, 7, 7] = -0.0

            def nonzero_weight(storage: TsdfBlockStorage) -> None:
                storage.weights[selected_row, 7, 7, 7] = 1

            for name, mutate, expected_message in (
                ("nonzero-sum", nonzero_sum, "canonical positive zero"),
                ("negative-zero", negative_zero, "canonical positive zero"),
                ("nonzero-weight", nonzero_weight, "empty"),
            ):
                with self.subTest(name=name):
                    storage = allocate_empty_tsdf_blocks(plan, session)
                    seed_unselected_blocks(storage, selected_row)
                    mutate(storage)
                    before_layout = storage_layout_snapshot(storage)
                    before_bytes = storage_bytes(storage)
                    with patch(
                        "spatialforge.tsdf_block_traversal."
                        "traverse_tsdf_voxel_observations_from_context"
                    ) as child:
                        with self.assertRaises(TsdfError) as raised:
                            traverse_tsdf_block_voxels_from_context(
                                storage,
                                SELECTED_BLOCK,
                                context,
                            )
                    self.assertIn(expected_message, str(raised.exception))
                    child.assert_not_called()
                    self.assertEqual(
                        storage_layout_snapshot(storage),
                        before_layout,
                    )
                    self.assertEqual(storage_bytes(storage), before_bytes)

    def test_successful_contributing_block_cannot_be_applied_twice(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            _, _, storage, context = load_case(Path(temporary_directory))
            first = traverse_tsdf_block_voxels_from_context(
                storage,
                SELECTED_BLOCK,
                context,
            )
            before_layout = storage_layout_snapshot(storage)
            before_bytes = storage_bytes(storage)

            with patch(
                "spatialforge.tsdf_block_traversal."
                "traverse_tsdf_voxel_observations_from_context"
            ) as child:
                with self.assertRaises(TsdfError) as raised:
                    traverse_tsdf_block_voxels_from_context(
                        storage,
                        SELECTED_BLOCK,
                        context,
                    )

        self.assertEqual(first.storage_slots_updated, 102)
        self.assertIn("empty", str(raised.exception))
        child.assert_not_called()
        self.assertEqual(storage_layout_snapshot(storage), before_layout)
        self.assertEqual(storage_bytes(storage), before_bytes)

    def test_types_provenance_and_unplanned_block_reject_before_child(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            _, _, storage, context = load_case(Path(temporary_directory))
            changed_context = replace(
                context,
                source_plan_digest_sha256="0" * 64,
            )
            cases = (
                (
                    "storage",
                    object(),
                    SELECTED_BLOCK,
                    context,
                    "TsdfBlockStorage",
                ),
                (
                    "block-list",
                    storage,
                    [1, -1, -1],
                    context,
                    "tuple of 3 integers",
                ),
                (
                    "block-arity",
                    storage,
                    (1, -1),
                    context,
                    "tuple of 3 integers",
                ),
                (
                    "block-bool",
                    storage,
                    (True, -1, -1),
                    context,
                    "expected an integer",
                ),
                (
                    "block-below",
                    storage,
                    (MIN_BLOCK_INDEX - 1, 0, 0),
                    context,
                    "expected an integer in",
                ),
                (
                    "block-above",
                    storage,
                    (MAX_BLOCK_INDEX + 1, 0, 0),
                    context,
                    "expected an integer in",
                ),
                (
                    "unplanned",
                    storage,
                    (-1, 0, 0),
                    context,
                    "not planned",
                ),
                (
                    "context-type",
                    storage,
                    SELECTED_BLOCK,
                    object(),
                    "TsdfReplayDepthContext",
                ),
                (
                    "context-plan",
                    storage,
                    SELECTED_BLOCK,
                    changed_context,
                    "source plan digest",
                ),
            )

            for name, candidate_storage, block, candidate_context, message in cases:
                with self.subTest(name=name):
                    before_layout = storage_layout_snapshot(storage)
                    before_bytes = storage_bytes(storage)
                    with patch(
                        "spatialforge.tsdf_block_traversal."
                        "traverse_tsdf_voxel_observations_from_context"
                    ) as child:
                        with self.assertRaises(TsdfError) as raised:
                            traverse_tsdf_block_voxels_from_context(
                                candidate_storage,  # type: ignore[arg-type]
                                block,  # type: ignore[arg-type]
                                candidate_context,  # type: ignore[arg-type]
                            )
                    self.assertIn(message, str(raised.exception))
                    child.assert_not_called()
                    self.assertEqual(
                        storage_layout_snapshot(storage),
                        before_layout,
                    )
                    self.assertEqual(storage_bytes(storage), before_bytes)

    def test_nonwritable_aliased_and_malformed_storage_rejects_preflight(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            plan, session, source, context = load_case(
                Path(temporary_directory)
            )

            invalid: list[tuple[str, TsdfBlockStorage, str]] = []
            readonly = allocate_empty_tsdf_blocks(plan, session)
            readonly.tsdf_sums.setflags(write=False)
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
                    before_bytes = storage_bytes(storage)
                    with patch(
                        "spatialforge.tsdf_block_traversal."
                        "traverse_tsdf_voxel_observations_from_context"
                    ) as child:
                        with self.assertRaises(TsdfError) as raised:
                            traverse_tsdf_block_voxels_from_context(
                                storage,
                                SELECTED_BLOCK,
                                context,
                            )
                    self.assertIn(message, str(raised.exception))
                    child.assert_not_called()
                    self.assertEqual(storage_bytes(storage), before_bytes)

    def test_late_child_failures_restore_the_whole_selected_block(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            plan, session, _, context = load_case(temporary_root)
            selected_row = plan.active_blocks.index(SELECTED_BLOCK)
            actual_child = traverse_tsdf_voxel_observations_from_context

            for name, raised_error, expected_message in (
                (
                    "domain",
                    TsdfError("injected final voxel failure"),
                    "injected final voxel failure",
                ),
                (
                    "unexpected",
                    RuntimeError("injected unexpected final voxel failure"),
                    "cannot apply context-bound TSDF block traversal",
                ),
            ):
                with self.subTest(name=name):
                    storage = allocate_empty_tsdf_blocks(plan, session)
                    seed_unselected_blocks(storage, selected_row)
                    before_layout = storage_layout_snapshot(storage)
                    before_bytes = storage_bytes(storage)
                    before_tree = tree_snapshot(temporary_root)
                    context_before = context_snapshot(context)

                    def traverse_then_fail(
                        changed_storage: TsdfBlockStorage,
                        address: TsdfVoxelAddress,
                        changed_context: TsdfReplayDepthContext,
                    ) -> TsdfVoxelTraversalReceipt:
                        receipt = actual_child(
                            changed_storage,
                            address,
                            changed_context,
                        )
                        if address.local_flat_index == 511:
                            raise raised_error
                        return receipt

                    with (
                        forbidden_block_calls() as forbidden,
                        patch(
                            "spatialforge.tsdf_block_traversal."
                            "traverse_tsdf_voxel_observations_from_context",
                            side_effect=traverse_then_fail,
                        ) as child,
                    ):
                        with self.assertRaises(TsdfError) as raised:
                            traverse_tsdf_block_voxels_from_context(
                                storage,
                                SELECTED_BLOCK,
                                context,
                            )
                    after_tree = tree_snapshot(temporary_root)
                    context_after = context_snapshot(context)

                    self.assertIn(expected_message, str(raised.exception))
                    self.assertEqual(child.call_count, TSDF_BLOCK_VOXELS)
                    self.assertEqual(
                        storage_layout_snapshot(storage),
                        before_layout,
                    )
                    self.assertEqual(storage_bytes(storage), before_bytes)
                    self.assertEqual(after_tree, before_tree)
                    self.assertEqual(context_after, context_before)
                    selected_sum = storage.tsdf_sums[selected_row]
                    selected_weight = storage.weights[selected_row]
                    self.assertEqual(selected_sum.tobytes(), bytes(4096))
                    self.assertEqual(selected_weight.tobytes(), bytes(2048))
                    if name == "unexpected":
                        self.assertIs(raised.exception.__cause__, raised_error)
                    for forbidden_call in forbidden:
                        forbidden_call.assert_not_called()

    def test_receipt_is_immutable_and_rejects_incomplete_or_reordered_rows(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            _, _, storage, context = load_case(Path(temporary_directory))
            receipt = traverse_tsdf_block_voxels_from_context(
                storage,
                SELECTED_BLOCK,
                context,
            )

        self.assertFalse(hasattr(receipt, "__dict__"))
        with self.assertRaises(FrozenInstanceError):
            receipt.block_row = 9  # type: ignore[misc]

        invalid_rows = (
            receipt.voxel_receipts[:-1],
            (
                receipt.voxel_receipts[1],
                receipt.voxel_receipts[0],
                *receipt.voxel_receipts[2:],
            ),
            (
                receipt.voxel_receipts[0],
                receipt.voxel_receipts[0],
                *receipt.voxel_receipts[2:],
            ),
        )
        for voxel_receipts in invalid_rows:
            with self.subTest(length=len(voxel_receipts)):
                with self.assertRaises(TsdfError):
                    replace(receipt, voxel_receipts=voxel_receipts)

        for field, value in (
            ("block_row", 0),
            ("source_plan_digest_sha256", "0" * 64),
            ("replay_digest_sha256", "0" * 64),
            ("block_resolution", 8.0),
            ("frame_stride", 2),
            ("total_observations", 3),
            ("selected_observation_sequences", (1,)),
            (
                "selected_observation_sequences",
                np.asarray([0, 1], dtype=np.int64),
            ),
        ):
            with self.subTest(field=field):
                with self.assertRaises(TsdfError):
                    replace(receipt, **{field: value})


class TsdfContextBlockTraversalCliTests(unittest.TestCase):
    def test_cli_reports_exact_isolated_one_block_traversal(self) -> None:
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
            actual_traverse = traverse_tsdf_block_voxels_from_context
            actual_child = traverse_tsdf_voxel_observations_from_context
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
                block_index_xyz: tuple[int, int, int],
                context: TsdfReplayDepthContext,
            ) -> TsdfBlockTraversalReceipt:
                with forbidden_block_calls() as inner_forbidden:
                    receipt = actual_traverse(
                        storage,
                        block_index_xyz,
                        context,
                    )
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
                    "traverse_tsdf_block_voxels_from_context",
                    side_effect=traverse_isolated,
                ) as traverser,
                patch(
                    "spatialforge.tsdf_block_traversal."
                    "traverse_tsdf_voxel_observations_from_context",
                    wraps=actual_child,
                ) as child,
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
                        "tsdf-block-context-block-traverse",
                        str(plan_path),
                        str(session_path),
                        "--block",
                        "1",
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
        self.assertEqual(child.call_count, TSDF_BLOCK_VOXELS)
        self.assertEqual(zeros.call_count, 2)
        for forbidden_call in forbidden_cli:
            forbidden_call.assert_not_called()

        self.assertEqual(len(captured_storage), 1)
        storage = captured_storage[0]
        self.assertEqual(storage.nonzero_sum_count, 102)
        self.assertEqual(storage.nonzero_weight_count, 102)
        self.assertEqual(storage.unknown_voxel_count, 3994)
        self.assertEqual(
            np.flatnonzero(storage.tsdf_sums.reshape(-1)).tolist(),
            np.flatnonzero(storage.weights.reshape(-1)).tolist(),
        )
        self.assertTrue(
            all(
                512 <= flat_index < 1024
                for flat_index in np.flatnonzero(
                    storage.weights.reshape(-1)
                ).tolist()
            )
        )

        expected_output = (
            "TSDF BLOCK CONTEXT BLOCK TRAVERSAL CHECK "
            "scan-synthetic-0001\n"
            "artifact: valid\n"
            "session_replay: matched\n"
            "context_selection: frame_stride=1 total=2 selected=2\n"
            "context_immutable: yes\n"
            "depth_source: replay-depth-context\n"
            "block: index=(1, -1, -1) row=1 resolution=8 "
            "voxel_slots=512\n"
            "storage_flat_range: 512..1023\n"
            "address_order: local-flat-x-fastest local_flat=0..511\n"
            "first_voxel: global=(8, -8, -8) local=(0, 0, 0) "
            "array=(1, 0, 0, 0) storage_flat=512\n"
            "last_voxel: global=(15, -1, -1) local=(7, 7, 7) "
            "array=(1, 7, 7, 7) storage_flat=1023\n"
            "selection: frame_stride=1 total=2 selected=2\n"
            "block_before: nonzero_sums=0 nonzero_weights=0 "
            "unknown_voxels=512\n"
            "block_after: nonzero_sums=102 nonzero_weights=102 "
            "unknown_voxels=410\n"
            "storage_before: nonzero_sums=0 nonzero_weights=0 "
            "unknown_voxels=4096\n"
            "storage_after: nonzero_sums=102 nonzero_weights=102 "
            "unknown_voxels=3994\n"
            "status_counts: contributes=204 "
            "projection-outside-image=424 behind-truncation=396\n"
            "block_weight_sum_after: 204\n"
            "block_max_weight_after: 2\n"
            "context_provenance: matched\n"
            "traversal_source_freshness: construction-time-context\n"
            "traversal_session_replay: no\n"
            "traversal_replay_hashing: no\n"
            "traversal_source_io: no\n"
            "traversal_depth_decoding: no\n"
            "traversal_prepared_depth_access: yes\n"
            "voxel_addresses_traversed: 512\n"
            "voxel_transcripts_retained: 512\n"
            "voxel_observation_traversals: 512\n"
            "contributions_evaluated: 1024\n"
            "contributions_applied: 204\n"
            "contributions_skipped: 820\n"
            "storage_slots_updated: 102\n"
            "blocks_traversed: 1\n"
            "additional_blocks_visited: 0\n"
            "voxel_observation_traversal_performed: yes\n"
            "voxel_address_traversal_performed: yes\n"
            "selected_block_traversal_performed: yes\n"
            "fusion_block_traversal_performed: yes\n"
            "fusion_block_traversal_scope: selected-planned-block-only\n"
            "multiple_block_traversal_performed: no\n"
            "planned_block_set_traversal_performed: no\n"
            "free_space_coverage_planned: no\n"
            "ray_traversal_performed: no\n"
            "full_fusion_performed: no\n"
            "missing_blocks_created: no\n"
            "caught_failure_rollback_scope: selected-block\n"
            "artifact_written: no\n"
            "storage_persisted: no\n"
            "context_persisted: no\n"
            f"plan_sha256: {PLAN_SHA256}\n"
            f"replay_digest_sha256: {REPLAY_SHA256}\n"
        )
        self.assertEqual(stdout.getvalue(), expected_output)

    def test_cli_unplanned_block_fails_actionably_without_traversal(self) -> None:
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
                    "traverse_tsdf_block_voxels_from_context"
                ) as traverser,
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                exit_code = main(
                    [
                        "reconstruct",
                        "tsdf-block-context-block-traverse",
                        str(plan_path),
                        str(session_path),
                        "--block",
                        "-1",
                        "0",
                        "0",
                    ]
                )
            after_tree = tree_snapshot(temporary_root)

        self.assertEqual(exit_code, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(before_tree, after_tree)
        traverser.assert_not_called()
        error = stderr.getvalue()
        self.assertIn("TSDF BLOCK CONTEXT BLOCK TRAVERSAL FAILED", error)
        self.assertIn("block (-1, 0, 0) is not planned", error)
        self.assertNotIn("Traceback", error)


if __name__ == "__main__":
    unittest.main()
