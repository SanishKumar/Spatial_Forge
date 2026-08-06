"""Field-driven block fusion, pinned byte-for-byte to the scalar traversal.

`traverse_tsdf_block_voxels_from_context` decides what a fused block contains.
These tests require the vector path to produce the identical bytes, because
nothing weaker would justify substituting it: a reconstruction that is nearly
the same is a different reconstruction.
"""

from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from spatialforge import (
    TsdfBlockVectorFusionReceipt,
    allocate_empty_tsdf_blocks,
    build_tsdf_replay_depth_context,
    fuse_tsdf_block_from_vector_fields,
    load_tsdf_block_plan,
    traverse_tsdf_block_voxels_from_context,
)
from spatialforge.cli import main
from spatialforge.errors import TsdfError
from spatialforge.model import ScanSession
from spatialforge.session_loader import load_scan_session
from spatialforge.tsdf_block_contributions import _freeze
from spatialforge.tsdf_block_plan import plan_tsdf_blocks
from spatialforge.tsdf_block_plan_loader import TsdfBlockPlan
from spatialforge.tsdf_block_storage import (
    TSDF_BLOCK_VOXELS,
    TsdfBlockStorage,
)
from spatialforge.tsdf_plan_traversal import (
    traverse_tsdf_plan_blocks_from_context,
)
from spatialforge.tsdf_replay_depth_context import TsdfReplayDepthContext

from tests.heavy_fixtures import shared_room_case

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

ROOM_ACCEPTED_CONTRIBUTIONS = 1_127_112
ROOM_OBSERVED_VOXELS = 81_292

_FORBIDDEN_FUSION_TARGETS = (
    "spatialforge.tsdf_voxel_contribution.replay_session",
    "spatialforge.tsdf_replay_depth_context.replay_session",
    "spatialforge.tsdf_voxel_contribution._read_depth",
    "spatialforge.tsdf_replay_depth_context._read_depth",
    "spatialforge.tsdf_voxel_contribution._sample_path",
    "spatialforge.tsdf_replay_depth_context._sample_path",
    "spatialforge.replay._file_digest",
    "hashlib.sha256",
    "pathlib.Path.open",
    "PIL.Image.open",
)


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
        frame_stride=1,
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


def storage_bytes(storage: TsdfBlockStorage) -> tuple[bytes, bytes]:
    return storage.tsdf_sums.tobytes(), storage.weights.tobytes()


@contextmanager
def forbidden_calls(targets: tuple[str, ...]):
    with ExitStack() as stack:
        patches = {
            target: stack.enter_context(patch(target))
            for target in targets
        }
        yield
        for target, mock in patches.items():
            if mock.called:
                raise AssertionError(f"{target} was called")


class VectorFusionParityTests(unittest.TestCase):
    def test_every_block_is_byte_identical_to_the_scalar_traversal(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            plan, session, storage, context = load_case(Path(temporary_dir))
            reference_storage = allocate_empty_tsdf_blocks(plan, session)
            for block_index_xyz in plan.active_blocks:
                receipt = fuse_tsdf_block_from_vector_fields(
                    storage,
                    block_index_xyz,
                    context,
                )
                reference = traverse_tsdf_block_voxels_from_context(
                    reference_storage,
                    block_index_xyz,
                    context,
                )
                with self.subTest(block=block_index_xyz):
                    self.assertEqual(
                        receipt.evaluated_count,
                        reference.evaluated_count,
                    )
                    self.assertEqual(
                        receipt.applied_count,
                        reference.applied_count,
                    )
                    self.assertEqual(
                        receipt.skipped_count,
                        reference.skipped_count,
                    )
                    self.assertEqual(
                        receipt.storage_slots_updated,
                        reference.storage_slots_updated,
                    )
                    self.assertEqual(
                        receipt.weight_delta,
                        reference.weight_delta,
                    )
                    self.assertEqual(
                        receipt.maximum_weight_after,
                        reference.maximum_weight_after,
                    )
                    self.assertEqual(
                        receipt.observed_voxel_count,
                        reference.observed_voxel_count,
                    )
                    self.assertEqual(
                        receipt.nonzero_sum_count,
                        reference.nonzero_sum_count,
                    )
                    self.assertEqual(
                        receipt.status_counts,
                        reference.status_counts,
                    )
            fused = storage_bytes(storage)
            expected = storage_bytes(reference_storage)

        self.assertEqual(fused, expected)

    def test_whole_plan_is_byte_identical_to_the_one_shot_traversal(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            plan, session, storage, context = load_case(Path(temporary_dir))
            applied = 0
            for block_index_xyz in plan.active_blocks:
                applied += fuse_tsdf_block_from_vector_fields(
                    storage,
                    block_index_xyz,
                    context,
                ).applied_count
            reference_storage = allocate_empty_tsdf_blocks(plan, session)
            reference = traverse_tsdf_plan_blocks_from_context(
                reference_storage,
                context,
            )
            fused = storage_bytes(storage)
            expected = storage_bytes(reference_storage)

        self.assertEqual(fused, expected)
        self.assertEqual(applied, reference.applied_count)
        self.assertEqual(applied, 1168)
        self.assertEqual(storage.nonzero_weight_count, 584)

    def test_room_blocks_are_byte_identical_to_the_scalar_traversal(
        self,
    ) -> None:
        case = shared_room_case()
        plan, session, context = case.plan, case.session, case.context
        storage = allocate_empty_tsdf_blocks(plan, session)
        reference_storage = allocate_empty_tsdf_blocks(plan, session)
        blocks = plan.active_blocks
        for block_index_xyz in (
            blocks[0],
            blocks[len(blocks) // 3],
            blocks[len(blocks) // 2],
            blocks[-1],
        ):
            receipt = fuse_tsdf_block_from_vector_fields(
                storage,
                block_index_xyz,
                context,
            )
            reference = traverse_tsdf_block_voxels_from_context(
                reference_storage,
                block_index_xyz,
                context,
            )
            with self.subTest(block=block_index_xyz):
                self.assertEqual(
                    receipt.applied_count,
                    reference.applied_count,
                )
                self.assertEqual(
                    receipt.status_counts,
                    reference.status_counts,
                )

        self.assertEqual(
            storage_bytes(storage),
            storage_bytes(reference_storage),
        )

    def test_whole_room_plan_fuses_to_the_recorded_totals(self) -> None:
        """The scan the one-shot traversal refuses, fused block by block."""

        case = shared_room_case()
        plan, context = case.plan, case.context
        storage = allocate_empty_tsdf_blocks(plan, case.session)
        applied = 0
        for block_index_xyz in plan.active_blocks:
            applied += fuse_tsdf_block_from_vector_fields(
                storage,
                block_index_xyz,
                context,
            ).applied_count

        self.assertEqual(applied, ROOM_ACCEPTED_CONTRIBUTIONS)
        self.assertEqual(
            storage.nonzero_weight_count,
            ROOM_OBSERVED_VOXELS,
        )


class VectorFusionReceiptTests(unittest.TestCase):
    def fuse(self) -> TsdfBlockVectorFusionReceipt:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            _, _, storage, context = load_case(Path(temporary_dir))
            return fuse_tsdf_block_from_vector_fields(
                storage,
                SELECTED_BLOCK,
                context,
            )

    def test_receipt_is_frozen_with_immutable_arrays(self) -> None:
        receipt = self.fuse()
        with self.assertRaises(FrozenInstanceError):
            receipt.block_row = 4  # type: ignore[misc]
        for array in (
            receipt.tsdf_sums_before,
            receipt.weights_before,
            receipt.tsdf_sums_after,
            receipt.weights_after,
        ):
            self.assertFalse(array.flags.writeable)
            self.assertEqual(array.shape, (TSDF_BLOCK_VOXELS,))

    def test_receipt_replays_its_own_fields(self) -> None:
        receipt = self.fuse()
        self.assertEqual(len(receipt.fields), 2)
        self.assertEqual(receipt.applied_count, 204)
        self.assertEqual(receipt.storage_slots_updated, 102)
        self.assertEqual(receipt.maximum_weight_after, 2)
        self.assertEqual(receipt.source_plan_digest_sha256, PLAN_SHA256)
        self.assertEqual(receipt.replay_digest_sha256, REPLAY_SHA256)

        derived = receipt.tsdf_sums_before.astype(np.float64)
        for field in receipt.fields:
            derived = derived + field.tsdf_sum_deltas
        self.assertEqual(
            derived.tobytes(),
            receipt.tsdf_sums_after.tobytes(),
        )

    def test_tampered_result_is_rejected(self) -> None:
        receipt = self.fuse()
        cases = {
            "tsdf_sums_after": _freeze(
                np.zeros(TSDF_BLOCK_VOXELS, dtype=np.float64)
            ),
            "weights_after": _freeze(
                np.zeros(TSDF_BLOCK_VOXELS, dtype=np.uint32)
            ),
            "fields": receipt.fields[:1],
        }
        for name, value in cases.items():
            with self.subTest(field=name):
                with self.assertRaises(TsdfError):
                    replace(receipt, **{name: value})

    def test_reordered_fields_are_rejected(self) -> None:
        receipt = self.fuse()
        with self.assertRaises(TsdfError) as caught:
            replace(receipt, fields=tuple(reversed(receipt.fields)))
        self.assertIn("inconsistent", str(caught.exception))


class VectorFusionGuardTests(unittest.TestCase):
    def test_second_fusion_of_the_same_block_is_refused(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            _, _, storage, context = load_case(Path(temporary_dir))
            fuse_tsdf_block_from_vector_fields(
                storage,
                SELECTED_BLOCK,
                context,
            )
            after_first = storage_bytes(storage)
            with self.assertRaises(TsdfError) as caught:
                fuse_tsdf_block_from_vector_fields(
                    storage,
                    SELECTED_BLOCK,
                    context,
                )
            unchanged = storage_bytes(storage)

        self.assertIn("canonical empty", str(caught.exception))
        self.assertEqual(after_first, unchanged)

    def test_caught_failure_restores_the_whole_block_row(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            plan, _, storage, context = load_case(Path(temporary_dir))
            fuse_tsdf_block_from_vector_fields(
                storage,
                plan.active_blocks[0],
                context,
            )
            before_failure = storage_bytes(storage)
            with patch(
                "spatialforge.tsdf_block_vector_fusion."
                "TsdfBlockVectorFusionReceipt",
                side_effect=TsdfError("receipt refused"),
            ):
                with self.assertRaises(TsdfError):
                    fuse_tsdf_block_from_vector_fields(
                        storage,
                        SELECTED_BLOCK,
                        context,
                    )
            after_failure = storage_bytes(storage)

        self.assertEqual(before_failure, after_failure)

    def test_unplanned_block_is_refused(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            _, _, storage, context = load_case(Path(temporary_dir))
            before = storage_bytes(storage)
            with self.assertRaises(TsdfError) as caught:
                fuse_tsdf_block_from_vector_fields(
                    storage,
                    (-1, 0, 0),
                    context,
                )
            after = storage_bytes(storage)

        self.assertIn("is not planned", str(caught.exception))
        self.assertEqual(before, after)

    def test_invalid_arguments_are_refused(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            _, _, storage, context = load_case(Path(temporary_dir))
            cases = (
                (
                    "storage",
                    (None, SELECTED_BLOCK, context),
                    "TsdfBlockStorage",
                ),
                (
                    "block",
                    (storage, (1, -1), context),
                    "expected a tuple of 3 integers",
                ),
                (
                    "context",
                    (storage, SELECTED_BLOCK, None),
                    "TsdfReplayDepthContext",
                ),
            )
            for name, arguments, expected in cases:
                with self.subTest(case=name):
                    with self.assertRaises(TsdfError) as caught:
                        fuse_tsdf_block_from_vector_fields(*arguments)
                    self.assertIn(expected, str(caught.exception))

    def test_fusion_performs_no_replay_hashing_or_depth_io(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            temporary_root = Path(temporary_dir)
            _, _, storage, context = load_case(temporary_root)
            before_tree = tree_snapshot(temporary_root)
            with forbidden_calls(_FORBIDDEN_FUSION_TARGETS):
                receipt = fuse_tsdf_block_from_vector_fields(
                    storage,
                    SELECTED_BLOCK,
                    context,
                )
            after_tree = tree_snapshot(temporary_root)

        self.assertEqual(receipt.applied_count, 204)
        self.assertEqual(before_tree, after_tree)

    def test_fusion_touches_no_other_block_row(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            plan, _, storage, context = load_case(Path(temporary_dir))
            receipt = fuse_tsdf_block_from_vector_fields(
                storage,
                SELECTED_BLOCK,
                context,
            )
            other_rows = [
                (
                    storage.tsdf_sums[row].tobytes(),
                    storage.weights[row].tobytes(),
                )
                for row in range(len(plan.active_blocks))
                if row != receipt.block_row
            ]

        empty_sums = np.zeros((8, 8, 8), dtype=np.float64).tobytes()
        empty_weights = np.zeros((8, 8, 8), dtype=np.uint32).tobytes()
        for sums, weights in other_rows:
            self.assertEqual(sums, empty_sums)
            self.assertEqual(weights, empty_weights)


class VectorFusionCliTests(unittest.TestCase):
    def test_cli_reports_the_fused_block_and_its_byte_parity(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            temporary_root = Path(temporary_dir)
            plan_path = create_plan(temporary_root)
            before_tree = tree_snapshot(temporary_root)
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                exit_code = main(
                    [
                        "reconstruct",
                        "tsdf-block-context-block-fuse",
                        str(plan_path),
                        str(FIXTURE),
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
        self.assertEqual(
            stdout.getvalue(),
            "TSDF BLOCK CONTEXT BLOCK FUSION CHECK scan-synthetic-0001\n"
            "artifact: valid\n"
            "session_replay: matched\n"
            "context_selection: frame_stride=1 total=2 selected=2\n"
            "context_immutable: yes\n"
            "depth_source: replay-depth-context\n"
            "block: index=(1, -1, -1) row=1 resolution=8 voxel_slots=512\n"
            "fusion_path: vectorised-field-per-observation\n"
            "field_application_order: "
            "canonical-observation-then-elementwise\n"
            "fields_applied: 2\n"
            "array_writes_per_field: 2\n"
            "contributions_evaluated: 1024\n"
            "contributions_applied: 204\n"
            "contributions_skipped: 820\n"
            "status_counts: contributes=204 "
            "projection-outside-image=424 behind-truncation=396\n"
            "storage_slots_updated: 102\n"
            "block_weight_sum_after: 204\n"
            "block_max_weight_after: 2\n"
            "block_observed_voxels: 102\n"
            "storage_after: nonzero_sums=102 nonzero_weights=102 "
            "unknown_voxels=3994\n"
            "scalar_reference_path: voxel-by-voxel-block-traversal\n"
            "scalar_reference_evaluations: 1024\n"
            "scalar_reference_applied: 204\n"
            "scalar_reference_sum_bytes: identical\n"
            "scalar_reference_weight_bytes: identical\n"
            "scalar_reference_parity: byte-identical\n"
            "receipt_rederived_from_fields: yes\n"
            "context_provenance: matched\n"
            "fusion_session_replay: no\n"
            "fusion_replay_hashing: no\n"
            "fusion_source_io: no\n"
            "fusion_depth_decoding: no\n"
            "blocks_fused: 1\n"
            "additional_blocks_visited: 0\n"
            "empty_block_precondition: required\n"
            "caught_failure_rollback_scope: selected-block\n"
            "ledger_used: no\n"
            "resumable: no\n"
            "planned_block_set_fusion_performed: no\n"
            "plan_expanded: no\n"
            "artifact_written: no\n"
            "storage_persisted: no\n"
            "context_persisted: no\n"
            f"plan_sha256: {PLAN_SHA256}\n"
            f"replay_digest_sha256: {REPLAY_SHA256}\n",
        )

    def test_cli_unplanned_block_fails_actionably(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            temporary_root = Path(temporary_dir)
            plan_path = create_plan(temporary_root)
            before_tree = tree_snapshot(temporary_root)
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                exit_code = main(
                    [
                        "reconstruct",
                        "tsdf-block-context-block-fuse",
                        str(plan_path),
                        str(FIXTURE),
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
        error = stderr.getvalue()
        self.assertIn("TSDF BLOCK CONTEXT BLOCK FUSION FAILED", error)
        self.assertIn("block (-1, 0, 0) is not planned", error)
        self.assertNotIn("Traceback", error)


if __name__ == "__main__":
    unittest.main()
