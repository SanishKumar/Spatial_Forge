"""Frame-at-a-time fusion, pinned byte-for-byte to the block-by-block paths.

Streaming changes which voxel is visited when. It must not change the order
in which any one voxel receives its contributions, because that order is
what fixes the last bits of a float64 sum. The tests therefore compare whole
accumulator buffers, not totals.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from contextlib import ExitStack, contextmanager
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from spatialforge import (
    allocate_empty_tsdf_blocks,
    build_tsdf_replay_depth_context,
    fuse_tsdf_block_from_vector_fields,
    load_tsdf_block_plan,
)
from spatialforge.errors import TsdfError
from spatialforge.session_loader import load_scan_session
from spatialforge.tsdf_block_plan import plan_tsdf_blocks
from spatialforge.tsdf_block_storage import TsdfBlockStorage
from spatialforge.tsdf_plan_traversal import (
    traverse_tsdf_plan_blocks_from_context,
)
from spatialforge.tsdf_stream_fusion import (
    TsdfStreamFusionReceipt,
    fuse_tsdf_plan_streaming,
    storage_payload_sha256,
)
from spatialforge.tsdf_voxel_contribution import TsdfContributionStatus

from tests.heavy_fixtures import shared_room_case

TEST_ROOT = Path(__file__).resolve().parent
FIXTURE = TEST_ROOT / "fixtures" / "minimal.vgsession"
PLAN_ARGUMENTS = {"voxel_size_m": 0.125, "truncation_m": 0.5}
PLAN_SHA256 = (
    "372c7c5d49eff1a30317ceb8b67cb3c40683f1d9d2a6179a049772f763d6f79d"
)
REPLAY_SHA256 = (
    "dc001ae0ca01004a21ad227b12d57a7350bbc23904040453ca73fd955ef050b8"
)
ROOM_ACCEPTED_CONTRIBUTIONS = 1_127_112
ROOM_OBSERVED_VOXELS = 81_292

# A streaming pass must not build or consult the all-frames context, replay
# mid-pass, or decode through the per-pixel reference decoder.
_FORBIDDEN_STREAM_TARGETS = (
    "spatialforge.tsdf_replay_depth_context."
    "build_tsdf_replay_depth_context",
    "spatialforge.tsdf_replay_depth_context._read_depth",
    "spatialforge.tsdf_voxel_contribution._read_depth",
    "spatialforge.tsdf_voxel_contribution."
    "evaluate_tsdf_voxel_contribution_from_context",
)


def storage_bytes(storage: TsdfBlockStorage) -> tuple[bytes, bytes]:
    return storage.tsdf_sums.tobytes(), storage.weights.tobytes()


def load_case(parent: Path, *, session_path: Path = FIXTURE, stride: int = 1):
    plan_path = parent / f"stride{stride}.sftplan"
    plan_tsdf_blocks(
        load_scan_session(session_path),
        plan_path,
        frame_stride=stride,
        **PLAN_ARGUMENTS,
    )
    plan = load_tsdf_block_plan(plan_path)
    session = load_scan_session(session_path)
    return plan, session


def scalar_reference(plan, session) -> TsdfBlockStorage:
    """The one-shot voxel-by-voxel traversal that defines a fused plan."""

    storage = allocate_empty_tsdf_blocks(plan, session)
    traverse_tsdf_plan_blocks_from_context(
        storage,
        build_tsdf_replay_depth_context(plan, session),
    )
    return storage


def remove_first_record(path: Path) -> None:
    remaining = path.read_text(encoding="utf-8").splitlines()[1:]
    path.write_text(
        "".join(line + "\n" for line in remaining),
        encoding="utf-8",
    )


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


class StreamFusionParityTests(unittest.TestCase):
    def test_fixture_is_byte_identical_to_the_scalar_traversal(self) -> None:
        for stride in (1, 2):
            with self.subTest(stride=stride):
                with tempfile.TemporaryDirectory(
                    dir=TEST_ROOT
                ) as temporary_dir:
                    plan, session = load_case(
                        Path(temporary_dir),
                        stride=stride,
                    )
                    storage = allocate_empty_tsdf_blocks(plan, session)
                    receipt = fuse_tsdf_plan_streaming(storage, session)
                    reference = scalar_reference(plan, session)

                self.assertEqual(
                    storage_bytes(storage),
                    storage_bytes(reference),
                )
                self.assertEqual(
                    receipt.observed_voxel_count,
                    reference.nonzero_weight_count,
                )

    def test_fixture_receipt_reports_the_agreed_invariants(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            plan, session = load_case(Path(temporary_dir))
            storage = allocate_empty_tsdf_blocks(plan, session)
            receipt = fuse_tsdf_plan_streaming(storage, session)

        self.assertEqual(receipt.source_plan_digest_sha256, PLAN_SHA256)
        self.assertEqual(receipt.replay_digest_sha256, REPLAY_SHA256)
        self.assertEqual(receipt.block_count, 8)
        self.assertEqual(receipt.voxel_slots, 4096)
        self.assertEqual(receipt.selected_observations, 2)
        self.assertEqual(receipt.fused_observations, 2)
        self.assertEqual(receipt.evaluated_count, 8192)
        self.assertEqual(receipt.applied_count, 1168)
        self.assertEqual(receipt.observed_voxel_count, 584)
        self.assertEqual(receipt.maximum_weight, 2)
        self.assertEqual(
            receipt.status_counts,
            (
                (TsdfContributionStatus.CONTRIBUTES, 1168),
                (TsdfContributionStatus.PROJECTION_OUTSIDE_IMAGE, 5440),
                (TsdfContributionStatus.BEHIND_TRUNCATION, 1584),
            ),
        )
        self.assertEqual(
            receipt.tsdf_sums_sha256,
            storage_payload_sha256(storage.tsdf_sums),
        )
        self.assertEqual(
            receipt.weights_sha256,
            storage_payload_sha256(storage.weights),
        )
        # One 2x2 float64 frame, however many frames the scan holds.
        self.assertEqual(receipt.peak_retained_depth_bytes, 32)

    def test_room_scan_is_byte_identical_to_block_major_fusion(self) -> None:
        case = shared_room_case()
        storage = allocate_empty_tsdf_blocks(case.plan, case.session)
        receipt = fuse_tsdf_plan_streaming(storage, case.session)
        reference = allocate_empty_tsdf_blocks(case.plan, case.session)
        for block_index_xyz in case.plan.active_blocks:
            fuse_tsdf_block_from_vector_fields(
                reference,
                block_index_xyz,
                case.context,
            )

        self.assertEqual(storage_bytes(storage), storage_bytes(reference))
        self.assertEqual(receipt.applied_count, ROOM_ACCEPTED_CONTRIBUTIONS)
        self.assertEqual(receipt.observed_voxel_count, ROOM_OBSERVED_VOXELS)
        self.assertEqual(receipt.fused_observations, 20)

    def test_chunk_size_never_changes_the_fused_bytes(self) -> None:
        case = shared_room_case()
        expected = allocate_empty_tsdf_blocks(case.plan, case.session)
        fuse_tsdf_plan_streaming(expected, case.session)
        self.assertGreater(case.plan.active_block_count, 7)
        for chunk in (1, 7, 1_000_003):
            with self.subTest(chunk=chunk):
                storage = allocate_empty_tsdf_blocks(case.plan, case.session)
                with patch(
                    "spatialforge.tsdf_stream_fusion."
                    "STREAM_FUSION_CHUNK_BLOCKS",
                    chunk,
                ):
                    fuse_tsdf_plan_streaming(storage, case.session)
                self.assertEqual(
                    storage_bytes(storage),
                    storage_bytes(expected),
                )

    def test_missing_inputs_match_the_scalar_traversal(self) -> None:
        cases = (
            ("depth", ("streams/depth.jsonl",), 1, 0),
            ("pose", ("streams/poses.jsonl",), 0, 1),
            ("both", ("streams/depth.jsonl", "streams/poses.jsonl"), 1, 1),
        )
        for name, references, missing_depth, missing_pose in cases:
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory(
                    dir=TEST_ROOT
                ) as temporary_dir:
                    temporary_root = Path(temporary_dir)
                    session_path = temporary_root / "case.vgsession"
                    shutil.copytree(FIXTURE, session_path)
                    for reference in references:
                        remove_first_record(session_path / reference)
                    plan, session = load_case(
                        temporary_root,
                        session_path=session_path,
                    )
                    storage = allocate_empty_tsdf_blocks(plan, session)
                    receipt = fuse_tsdf_plan_streaming(storage, session)
                    reference_storage = scalar_reference(plan, session)

                self.assertEqual(
                    storage_bytes(storage),
                    storage_bytes(reference_storage),
                )
                self.assertEqual(receipt.fused_observations, 1)
                self.assertEqual(
                    receipt.skipped_missing_depth,
                    missing_depth,
                )
                self.assertEqual(receipt.skipped_missing_pose, missing_pose)
                self.assertEqual(
                    receipt.evaluated_count,
                    receipt.voxel_slots * 2,
                )


class StreamFusionMemoryTests(unittest.TestCase):
    def test_it_fuses_a_scan_the_all_frames_context_refuses(self) -> None:
        """The point of streaming: sequence length stops being a limit."""

        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            plan, session = load_case(Path(temporary_dir))
            reference = scalar_reference(plan, session)
            storage = allocate_empty_tsdf_blocks(plan, session)
            # Room for one 2x2 float64 frame, not the two this scan selects.
            with patch(
                "spatialforge.tsdf_replay_depth_context."
                "MAX_TSDF_REPLAY_DEPTH_CONTEXT_BYTES",
                32,
            ):
                with self.assertRaises(TsdfError) as refused:
                    build_tsdf_replay_depth_context(plan, session)
                fuse_tsdf_plan_streaming(storage, session)

        self.assertIn("retained depth bytes", str(refused.exception))
        self.assertEqual(storage_bytes(storage), storage_bytes(reference))

    def test_each_ready_frame_is_decoded_exactly_once(self) -> None:
        from spatialforge import tsdf_stream_fusion

        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            plan, session = load_case(Path(temporary_dir))
            storage = allocate_empty_tsdf_blocks(plan, session)
            with (
                forbidden_calls(_FORBIDDEN_STREAM_TARGETS),
                patch(
                    "spatialforge.tsdf_stream_fusion._read_depth_array",
                    wraps=tsdf_stream_fusion._read_depth_array,
                ) as decoder,
            ):
                receipt = fuse_tsdf_plan_streaming(storage, session)

        self.assertEqual(decoder.call_count, receipt.fused_observations)
        self.assertEqual(decoder.call_count, 2)


class StreamFusionGuardTests(unittest.TestCase):
    def test_storage_that_is_not_empty_is_refused_untouched(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            plan, session = load_case(Path(temporary_dir))
            for name, mutate in (
                ("weight", lambda s: s.weights.__setitem__((0, 0, 0, 0), 1)),
                ("sum", lambda s: s.tsdf_sums.__setitem__((1, 2, 3, 4), 0.5)),
                (
                    "negative zero",
                    lambda s: s.tsdf_sums.__setitem__((7, 7, 7, 7), -0.0),
                ),
            ):
                with self.subTest(case=name):
                    storage = allocate_empty_tsdf_blocks(plan, session)
                    mutate(storage)
                    before = storage_bytes(storage)
                    with self.assertRaises(TsdfError) as caught:
                        fuse_tsdf_plan_streaming(storage, session)
                    self.assertIn("canonical empty", str(caught.exception))
                    self.assertEqual(storage_bytes(storage), before)

    def test_a_failure_part_way_through_restores_empty_storage(self) -> None:
        from spatialforge import tsdf_stream_fusion

        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            plan, session = load_case(Path(temporary_dir))
            storage = allocate_empty_tsdf_blocks(plan, session)
            empty = storage_bytes(storage)
            real = tsdf_stream_fusion._evaluate_ready_voxels
            calls = []

            def fail_on_second_frame(*arguments):
                calls.append(1)
                if len(calls) == 2:
                    raise RuntimeError("simulated evaluation failure")
                return real(*arguments)

            with patch(
                "spatialforge.tsdf_stream_fusion._evaluate_ready_voxels",
                side_effect=fail_on_second_frame,
            ):
                with self.assertRaises(TsdfError) as caught:
                    fuse_tsdf_plan_streaming(storage, session)

        self.assertIn("simulated evaluation failure", str(caught.exception))
        self.assertEqual(len(calls), 2)
        self.assertEqual(storage_bytes(storage), empty)

    def test_a_session_that_changed_since_planning_is_refused(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            temporary_root = Path(temporary_dir)
            session_path = temporary_root / "case.vgsession"
            shutil.copytree(FIXTURE, session_path)
            plan, session = load_case(
                temporary_root,
                session_path=session_path,
            )
            storage = allocate_empty_tsdf_blocks(plan, session)
            empty = storage_bytes(storage)
            (session_path / "data" / "depth" / "000000.pgm").write_text(
                "P2\n2 2\n65535\n900 900\n900 900\n",
                encoding="ascii",
            )
            with self.assertRaises(TsdfError) as caught:
                fuse_tsdf_plan_streaming(
                    storage,
                    load_scan_session(session_path),
                )

        self.assertIn("replay digest", str(caught.exception))
        self.assertEqual(storage_bytes(storage), empty)

    def test_invalid_arguments_are_refused(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            plan, session = load_case(Path(temporary_dir))
            storage = allocate_empty_tsdf_blocks(plan, session)
            with self.assertRaises(TsdfError) as no_storage:
                fuse_tsdf_plan_streaming(None, session)  # type: ignore
            with self.assertRaises(TsdfError) as no_session:
                fuse_tsdf_plan_streaming(storage, None)  # type: ignore

        self.assertIn("TsdfBlockStorage", str(no_storage.exception))
        self.assertIn("ScanSession", str(no_session.exception))


class StreamFusionReceiptTests(unittest.TestCase):
    def receipt(self) -> TsdfStreamFusionReceipt:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            plan, session = load_case(Path(temporary_dir))
            return fuse_tsdf_plan_streaming(
                allocate_empty_tsdf_blocks(plan, session),
                session,
            )

    def test_receipt_is_frozen(self) -> None:
        receipt = self.receipt()
        with self.assertRaises(FrozenInstanceError):
            receipt.block_count = 9  # type: ignore[misc]

    def test_inconsistent_receipts_are_rejected(self) -> None:
        receipt = self.receipt()
        cases = {
            "voxel slots": {"voxel_slots": 4095},
            "selection": {"selected_observations": 3},
            "status coverage": {
                "status_counts": (
                    (TsdfContributionStatus.CONTRIBUTES, 1168),
                ),
            },
            "status order": {
                "status_counts": tuple(reversed(receipt.status_counts)),
            },
            "weight above frames": {"maximum_weight": 3},
            "observed above slots": {"observed_voxel_count": 4097},
            "applied below observed": {"observed_voxel_count": 1169},
            "digest": {"weights_sha256": "not-a-digest"},
            "skipped frames": {"fused_observations": 1},
        }
        for name, changes in cases.items():
            with self.subTest(case=name):
                with self.assertRaises(TsdfError):
                    replace(receipt, **changes)

    def test_payload_digest_is_byte_order_independent(self) -> None:
        values = np.arange(24, dtype=np.float64).reshape((1, 2, 3, 4))
        swapped = values.astype(values.dtype.newbyteorder(">"))
        self.assertNotEqual(values.tobytes(), swapped.tobytes())
        self.assertEqual(
            storage_payload_sha256(values),
            storage_payload_sha256(swapped),
        )


if __name__ == "__main__":
    unittest.main()
