from __future__ import annotations

import io
import json
import shutil
import tempfile
import unittest
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from spatialforge import (
    MAX_TSDF_VOXEL_WEIGHT,
    allocate_empty_tsdf_blocks,
    apply_tsdf_voxel_contribution,
    evaluate_tsdf_voxel_contribution,
    locate_tsdf_voxel,
)
from spatialforge.cli import main
from spatialforge.errors import SessionReplayError, TsdfError
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
from spatialforge.tsdf_voxel_contribution import TsdfVoxelContribution
from spatialforge.tsdf_voxel_update import TsdfVoxelUpdateReceipt


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
    plan_name: str = "fixture.sftplan",
) -> tuple[TsdfBlockPlan, ScanSession, TsdfBlockStorage]:
    plan = load_tsdf_block_plan(
        create_plan(
            parent,
            session_path=session_path,
            name=plan_name,
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


def evaluate_fixture_contribution(
    storage: TsdfBlockStorage,
    session: ScanSession,
) -> TsdfVoxelContribution:
    return evaluate_tsdf_voxel_contribution(
        storage,
        require_address(storage, (8, -1, -1)),
        session,
        0,
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


def storage_bytes(
    storage: TsdfBlockStorage,
) -> tuple[bytes, bytes]:
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


class TsdfVoxelUpdateTests(unittest.TestCase):
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

    def assert_apply_fails_unchanged(
        self,
        storage: TsdfBlockStorage,
        contribution: object,
        session: object,
        expected_message: str,
    ) -> None:
        before_layout = storage_layout_snapshot(storage)
        before_bytes = storage_bytes(storage)
        with self.assertRaises(TsdfError) as raised:
            apply_tsdf_voxel_contribution(  # type: ignore[arg-type]
                storage,
                contribution,
                session,
            )
        self.assertIn(expected_message, str(raised.exception))
        self.assertEqual(storage_layout_snapshot(storage), before_layout)
        self.assertEqual(storage_bytes(storage), before_bytes)

    def test_exact_fixture_update_changes_only_target_byte_slices(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            _, session, storage = load_case(temporary_root)
            contribution = evaluate_fixture_contribution(storage, session)
            address = contribution.address

            # A valid non-target sentinel catches replacement or broad writes.
            storage.tsdf_sums[0, 0, 0, 0] = 0.375
            storage.weights[0, 0, 0, 0] = 7
            before_layout = storage_layout_snapshot(storage)
            before_sums, before_weights = storage_bytes(storage)
            before_tree = tree_snapshot(temporary_root)

            import spatialforge.tsdf_voxel_update as update_module

            actual_locate = update_module.locate_tsdf_voxel
            actual_replay = update_module.replay_session
            with (
                forbidden_full_pipeline_calls() as forbidden,
                patch(
                    "spatialforge.cli.allocate_empty_tsdf_blocks"
                ) as allocator,
                patch(
                    "spatialforge.tsdf_block_storage.np.zeros"
                ) as zeros,
                patch(
                    "spatialforge.tsdf_voxel_contribution."
                    "evaluate_tsdf_voxel_contribution"
                ) as evaluator,
                patch(
                    "spatialforge.tsdf_voxel_contribution._read_depth"
                ) as depth_reader,
                patch(
                    "spatialforge.tsdf_voxel_update.locate_tsdf_voxel",
                    wraps=actual_locate,
                ) as locator,
                patch(
                    "spatialforge.tsdf_voxel_update.replay_session",
                    wraps=actual_replay,
                ) as replay,
            ):
                receipt = apply_tsdf_voxel_contribution(
                    storage,
                    contribution,
                    session,
                )
            after_sums, after_weights = storage_bytes(storage)
            after_tree = tree_snapshot(temporary_root)

        self.assertIsInstance(receipt, TsdfVoxelUpdateReceipt)
        self.assertIs(receipt.contribution, contribution)
        self.assertEqual(receipt.tsdf_sum_before, 0.0)
        self.assertEqual(receipt.weight_before, 0)
        self.assertEqual(receipt.tsdf_sum_after, -0.125)
        self.assertEqual(receipt.weight_after, 1)
        self.assertEqual(contribution.tsdf_sum_delta, -0.125)
        self.assertEqual(contribution.weight_delta, 1)
        self.assertEqual(address.storage_flat_index, 1016)
        self.assertEqual(address.array_index_bzyx, (1, 7, 7, 0))
        self.assertEqual(storage_layout_snapshot(storage), before_layout)
        self.assert_only_flat_value_changed(
            before_sums,
            after_sums,
            1016,
            np.dtype(np.float64),
            -0.125,
        )
        self.assert_only_flat_value_changed(
            before_weights,
            after_weights,
            1016,
            np.dtype(np.uint32),
            1,
        )
        self.assertEqual(after_tree, before_tree)
        locator.assert_called_once_with(
            storage,
            address.global_index_xyz,
        )
        self.assertEqual(replay.call_count, 2)
        allocator.assert_not_called()
        zeros.assert_not_called()
        evaluator.assert_not_called()
        depth_reader.assert_not_called()
        for forbidden_call in forbidden:
            forbidden_call.assert_not_called()
        with self.assertRaises(FrozenInstanceError):
            receipt.weight_after = 2  # type: ignore[misc]
        self.assertFalse(hasattr(receipt, "__dict__"))

    def test_repeated_updates_accumulate_the_same_one_slot(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            _, session, storage = load_case(Path(temporary_directory))
            contribution = evaluate_fixture_contribution(storage, session)
            before_sums, before_weights = storage_bytes(storage)

            first = apply_tsdf_voxel_contribution(
                storage,
                contribution,
                session,
            )
            second = apply_tsdf_voxel_contribution(
                storage,
                contribution,
                session,
            )
            after_sums, after_weights = storage_bytes(storage)

        self.assertEqual(
            (
                first.tsdf_sum_before,
                first.weight_before,
                first.tsdf_sum_after,
                first.weight_after,
            ),
            (0.0, 0, -0.125, 1),
        )
        self.assertEqual(
            (
                second.tsdf_sum_before,
                second.weight_before,
                second.tsdf_sum_after,
                second.weight_after,
            ),
            (-0.125, 1, -0.25, 2),
        )
        self.assert_only_flat_value_changed(
            before_sums,
            after_sums,
            contribution.address.storage_flat_index,
            np.dtype(np.float64),
            -0.25,
        )
        self.assert_only_flat_value_changed(
            before_weights,
            after_weights,
            contribution.address.storage_flat_index,
            np.dtype(np.uint32),
            2,
        )

    def test_same_plan_fresh_storage_accepts_bound_contribution(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            plan, session, source = load_case(Path(temporary_directory))
            destination = allocate_empty_tsdf_blocks(plan, session)
            contribution = evaluate_fixture_contribution(source, session)
            source_before = storage_bytes(source)

            receipt = apply_tsdf_voxel_contribution(
                destination,
                contribution,
                session,
            )

        self.assertEqual(storage_bytes(source), source_before)
        self.assertEqual(receipt.tsdf_sum_after, -0.125)
        self.assertEqual(receipt.weight_after, 1)
        self.assertEqual(
            float(destination.tsdf_sums[contribution.address.array_index_bzyx]),
            -0.125,
        )
        self.assertEqual(
            int(destination.weights[contribution.address.array_index_bzyx]),
            1,
        )

    def test_surface_zero_delta_updates_weight_but_not_sum_bytes(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            fixture = copy_fixture(temporary_root)
            set_first_pose_translation_x(fixture, -0.0625)
            _, session, storage = load_case(
                temporary_root,
                session_path=fixture,
                plan_name="surface.sftplan",
            )
            address = require_address(storage, (7, -1, -1))
            contribution = evaluate_tsdf_voxel_contribution(
                storage,
                address,
                session,
                0,
            )
            before_sums, before_weights = storage_bytes(storage)

            receipt = apply_tsdf_voxel_contribution(
                storage,
                contribution,
                session,
            )
            after_sums, after_weights = storage_bytes(storage)

        self.assertEqual(contribution.signed_distance_m, 0.0)
        self.assertEqual(contribution.tsdf_sum_delta, 0.0)
        self.assertEqual(receipt.tsdf_sum_before, 0.0)
        self.assertEqual(receipt.tsdf_sum_after, 0.0)
        self.assertEqual(receipt.weight_before, 0)
        self.assertEqual(receipt.weight_after, 1)
        self.assertEqual(after_sums, before_sums)
        self.assert_only_flat_value_changed(
            before_weights,
            after_weights,
            address.storage_flat_index,
            np.dtype(np.uint32),
            1,
        )

    def test_skipped_type_address_provenance_and_session_mismatches_fail(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            _, session, storage = load_case(Path(temporary_directory))
            contribution = evaluate_fixture_contribution(storage, session)
            skipped = evaluate_tsdf_voxel_contribution(
                storage,
                require_address(storage, (0, 7, -1)),
                session,
                0,
            )
            forged_address = replace(
                contribution.address,
                storage_flat_index=(
                    contribution.address.storage_flat_index + 1
                ),
            )
            cases = (
                (
                    "storage-type",
                    object(),
                    contribution,
                    session,
                    "allocated TsdfBlockStorage",
                ),
                (
                    "contribution-type",
                    storage,
                    object(),
                    session,
                    "TsdfVoxelContribution",
                ),
                (
                    "session-type",
                    storage,
                    contribution,
                    object(),
                    "loaded ScanSession",
                ),
                (
                    "skipped",
                    storage,
                    skipped,
                    session,
                    "accepted finite contribution",
                ),
                (
                    "address",
                    storage,
                    replace(contribution, address=forged_address),
                    session,
                    "address does not match",
                ),
                (
                    "source-plan",
                    storage,
                    replace(
                        contribution,
                        source_plan_digest_sha256="0" * 64,
                    ),
                    session,
                    "source plan does not match",
                ),
                (
                    "replay-provenance",
                    storage,
                    replace(
                        contribution,
                        replay_digest_sha256="0" * 64,
                    ),
                    session,
                    "replay digest does not match destination",
                ),
                (
                    "session-id",
                    storage,
                    contribution,
                    replace(session, session_id="different-session"),
                    "session_id does not match",
                ),
            )
            original_layout = storage_layout_snapshot(storage)
            original_bytes = storage_bytes(storage)
            for (
                name,
                changed_storage,
                changed_contribution,
                changed_session,
                expected_message,
            ) in cases:
                with self.subTest(name=name):
                    with self.assertRaises(TsdfError) as raised:
                        apply_tsdf_voxel_contribution(  # type: ignore[arg-type]
                            changed_storage,
                            changed_contribution,
                            changed_session,
                        )
                    self.assertIn(
                        expected_message,
                        str(raised.exception),
                    )
                    self.assertEqual(
                        storage_layout_snapshot(storage),
                        original_layout,
                    )
                    self.assertEqual(storage_bytes(storage), original_bytes)

    def test_read_only_and_aliased_arrays_fail_before_any_write(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            plan, session, source = load_case(Path(temporary_directory))
            contribution = evaluate_fixture_contribution(source, session)

            for name, read_only_buffer in (
                ("sums", "tsdf_sums"),
                ("weights", "weights"),
            ):
                with self.subTest(name=name):
                    storage = allocate_empty_tsdf_blocks(plan, session)
                    getattr(storage, read_only_buffer).setflags(write=False)
                    self.assert_apply_fails_unchanged(
                        storage,
                        contribution,
                        session,
                        "writable sum and weight arrays",
                    )

            shape = source.tsdf_sums.shape
            backing = np.zeros(source.tsdf_sums.nbytes, dtype=np.uint8)
            aliased_sums = np.ndarray(
                shape=shape,
                dtype=np.float64,
                buffer=backing,
            )
            aliased_weights = np.ndarray(
                shape=shape,
                dtype=np.uint32,
                buffer=backing,
            )
            aliased = TsdfBlockStorage(
                source_plan=plan,
                block_indices=source.block_indices,
                tsdf_sums=aliased_sums,
                weights=aliased_weights,
            )
            self.assertTrue(
                np.shares_memory(aliased.tsdf_sums, aliased.weights)
            )
            self.assert_apply_fails_unchanged(
                aliased,
                contribution,
                session,
                "non-overlapping storage arrays",
            )

    def test_corrupt_target_prestates_and_uint32_max_fail_atomically(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            plan, session, source = load_case(Path(temporary_directory))
            contribution = evaluate_fixture_contribution(source, session)
            index = contribution.address.array_index_bzyx

            def nonfinite_nan(storage: TsdfBlockStorage) -> None:
                storage.tsdf_sums[index] = np.nan

            def nonfinite_infinity(storage: TsdfBlockStorage) -> None:
                storage.tsdf_sums[index] = np.inf

            def negative_zero_unknown(storage: TsdfBlockStorage) -> None:
                storage.tsdf_sums[index] = -0.0

            def nonzero_unknown(storage: TsdfBlockStorage) -> None:
                storage.tsdf_sums[index] = 0.25

            def outside_weight_envelope(storage: TsdfBlockStorage) -> None:
                storage.tsdf_sums[index] = 1.25
                storage.weights[index] = 1

            def maximum_weight(storage: TsdfBlockStorage) -> None:
                storage.weights[index] = np.uint32(MAX_TSDF_VOXEL_WEIGHT)

            cases = (
                ("nan", nonfinite_nan, "sum must be finite"),
                ("infinity", nonfinite_infinity, "sum must be finite"),
                (
                    "negative-zero",
                    negative_zero_unknown,
                    "canonical positive zero",
                ),
                (
                    "nonzero-unknown",
                    nonzero_unknown,
                    "canonical positive zero",
                ),
                (
                    "weight-envelope",
                    outside_weight_envelope,
                    "exceeds its weight envelope",
                ),
                (
                    "uint32-maximum",
                    maximum_weight,
                    "cannot exceed uint32 maximum",
                ),
            )
            for name, corrupt, expected_message in cases:
                with self.subTest(name=name):
                    storage = allocate_empty_tsdf_blocks(plan, session)
                    corrupt(storage)
                    self.assert_apply_fails_unchanged(
                        storage,
                        contribution,
                        session,
                        expected_message,
                    )

    def test_starting_replay_mismatch_rejects_before_any_write(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            _, session, storage = load_case(Path(temporary_directory))
            contribution = evaluate_fixture_contribution(storage, session)
            changed = replace(
                replay_session(session),
                digest_sha256="0" * 64,
            )
            before_layout = storage_layout_snapshot(storage)
            before_bytes = storage_bytes(storage)

            with patch(
                "spatialforge.tsdf_voxel_update.replay_session",
                return_value=changed,
            ) as replay:
                with self.assertRaises(TsdfError) as raised:
                    apply_tsdf_voxel_contribution(
                        storage,
                        contribution,
                        session,
                    )

        self.assertIn(
            "does not match current session inputs",
            str(raised.exception),
        )
        replay.assert_called_once_with(session)
        self.assertEqual(storage_layout_snapshot(storage), before_layout)
        self.assertEqual(storage_bytes(storage), before_bytes)

    def test_ending_replay_mismatch_rolls_back_both_target_scalars(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            _, session, storage = load_case(Path(temporary_directory))
            contribution = evaluate_fixture_contribution(storage, session)
            matched = replay_session(session)
            changed = replace(matched, digest_sha256="0" * 64)
            before_layout = storage_layout_snapshot(storage)
            before_bytes = storage_bytes(storage)

            with patch(
                "spatialforge.tsdf_voxel_update.replay_session",
                side_effect=(matched, changed),
            ) as replay:
                with self.assertRaises(TsdfError) as raised:
                    apply_tsdf_voxel_contribution(
                        storage,
                        contribution,
                        session,
                    )

        self.assertIn(
            "session inputs changed while applying",
            str(raised.exception),
        )
        self.assertEqual(replay.call_count, 2)
        self.assertEqual(storage_layout_snapshot(storage), before_layout)
        self.assertEqual(storage_bytes(storage), before_bytes)

    def test_ending_replay_exception_rolls_back_both_target_scalars(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            _, session, storage = load_case(Path(temporary_directory))
            contribution = evaluate_fixture_contribution(storage, session)
            matched = replay_session(session)
            ending_error = SessionReplayError(
                "injected ending replay failure"
            )
            before_layout = storage_layout_snapshot(storage)
            before_bytes = storage_bytes(storage)

            with patch(
                "spatialforge.tsdf_voxel_update.replay_session",
                side_effect=(matched, ending_error),
            ) as replay:
                with self.assertRaises(SessionReplayError) as raised:
                    apply_tsdf_voxel_contribution(
                        storage,
                        contribution,
                        session,
                    )

        self.assertEqual(
            str(raised.exception),
            "injected ending replay failure",
        )
        self.assertEqual(replay.call_count, 2)
        self.assertEqual(storage_layout_snapshot(storage), before_layout)
        self.assertEqual(storage_bytes(storage), before_bytes)

    def test_direct_receipt_requires_canonical_enveloped_transition(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            _, session, storage = load_case(Path(temporary_directory))
            contribution = evaluate_fixture_contribution(storage, session)

        valid = TsdfVoxelUpdateReceipt(
            contribution=contribution,
            tsdf_sum_before=0.0,
            weight_before=0,
            tsdf_sum_after=-0.125,
            weight_after=1,
        )
        self.assertEqual(valid.tsdf_sum_before, 0.0)
        self.assertEqual(valid.weight_before, 0)
        self.assertEqual(valid.tsdf_sum_after, -0.125)
        self.assertEqual(valid.weight_after, 1)
        with self.assertRaises(FrozenInstanceError):
            valid.weight_after = 2  # type: ignore[misc]
        self.assertFalse(hasattr(valid, "__dict__"))

        cases = (
            (
                "negative-zero-unknown",
                -0.0,
                0,
                -0.125,
                1,
                "canonical positive zero",
            ),
            (
                "nonzero-unknown",
                0.25,
                0,
                0.125,
                1,
                "canonical positive zero",
            ),
            (
                "prior-envelope",
                1.25,
                1,
                1.125,
                2,
                "prior sum exceeds its weight envelope",
            ),
            (
                "result-envelope",
                1.0,
                1,
                2.125,
                2,
                "resulting sum exceeds its weight envelope",
            ),
        )
        for (
            name,
            sum_before,
            weight_before,
            sum_after,
            weight_after,
            expected_message,
        ) in cases:
            with self.subTest(name=name):
                with self.assertRaises(TsdfError) as raised:
                    TsdfVoxelUpdateReceipt(
                        contribution=contribution,
                        tsdf_sum_before=sum_before,
                        weight_before=weight_before,
                        tsdf_sum_after=sum_after,
                        weight_after=weight_after,
                    )
                self.assertIn(
                    expected_message,
                    str(raised.exception),
                )


class TsdfVoxelUpdateCliTests(unittest.TestCase):
    def test_cli_exactly_reports_one_temporary_update_without_artifact(
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
            captured_storage: list[TsdfBlockStorage] = []

            actual_allocate = allocate_empty_tsdf_blocks
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
                    "spatialforge.cli.evaluate_tsdf_voxel_contribution",
                    wraps=actual_evaluate,
                ) as evaluator,
                patch(
                    "spatialforge.cli.apply_tsdf_voxel_contribution",
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
                        "tsdf-block-contribution-apply",
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
        self.assertEqual(before_tree, after_tree)
        allocator.assert_called_once()
        evaluator.assert_called_once()
        updater.assert_called_once()
        depth_reader.assert_called_once()
        self.assertEqual(zeros.call_count, 2)
        for forbidden_call in forbidden:
            forbidden_call.assert_not_called()
        self.assertEqual(len(captured_storage), 1)
        storage = captured_storage[0]
        address = require_address(storage, (8, -1, -1))
        self.assertEqual(address.storage_flat_index, 1016)
        self.assertEqual(
            float(storage.tsdf_sums[address.array_index_bzyx]),
            -0.125,
        )
        self.assertEqual(
            int(storage.weights[address.array_index_bzyx]),
            1,
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
            "TSDF BLOCK CONTRIBUTION APPLY CHECK scan-synthetic-0001\n"
            "artifact: valid\n"
            "session_replay: matched\n"
            "observation_sequence: 0\n"
            "voxel: global=(8, -1, -1) block=(1, -1, -1) "
            "local=(0, 7, 7) row=1 array=(1, 7, 7, 0) "
            "storage_flat=1016\n"
            "evaluation: contributes\n"
            "slot_before: tsdf_sum=0.000000000 weight=0\n"
            "applied_delta: tsdf_sum=-0.125000000 weight=1\n"
            "slot_after: tsdf_sum=-0.125000000 weight=1\n"
            "storage_before: nonzero_sums=0 nonzero_weights=0 "
            "unknown_voxels=4096\n"
            "storage_after: nonzero_sums=1 nonzero_weights=1 "
            "unknown_voxels=4095\n"
            "contributions_evaluated: 1\n"
            "contributions_applied: 1\n"
            "storage_slots_updated: 1\n"
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
