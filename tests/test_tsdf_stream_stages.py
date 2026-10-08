"""Streaming fusion taken in stages, held to the single pass byte for byte.

A stage boundary is a promise: stop after any frame, continue later, and the
accumulators end up exactly where one uninterrupted pass would have left
them. The tests cut the same scan in many places and compare whole buffers
and whole receipts. The state between stages is also checked against the
observation-ledger path, which reaches "the first k frames" by a different
route: frames held in a context, rows fused one block at a time.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from spatialforge import (
    allocate_empty_tsdf_blocks,
    load_tsdf_block_plan,
)
from spatialforge.errors import TsdfError
from spatialforge.session_loader import load_scan_session
from spatialforge.tsdf_block_plan import plan_tsdf_blocks
from spatialforge.tsdf_block_storage import TSDF_BLOCK_VOXELS
from spatialforge.tsdf_observation_fusion import (
    begin_tsdf_observation_ledger,
    fuse_tsdf_plan_observations_from_context,
)
from spatialforge.tsdf_stream_fusion import (
    TsdfStreamFusionProgress,
    advance_tsdf_plan_streaming,
    finish_tsdf_plan_streaming,
    fuse_tsdf_plan_streaming,
    storage_payload_sha256,
)
from spatialforge.tsdf_voxel_contribution import TsdfContributionStatus

from tests.heavy_fixtures import shared_room_case
from tests.test_tsdf_stream_fusion import (
    FIXTURE,
    load_case,
    remove_first_record,
    storage_bytes,
)

TEST_ROOT = Path(__file__).resolve().parent

_SINGLE_PASS = None


def single_pass():
    """The room scan fused in one pass: the bytes and receipt to match."""

    global _SINGLE_PASS
    if _SINGLE_PASS is None:
        room = shared_room_case()
        storage = allocate_empty_tsdf_blocks(room.plan, room.session)
        receipt = fuse_tsdf_plan_streaming(storage, room.session)
        _SINGLE_PASS = (storage_bytes(storage), receipt)
    return _SINGLE_PASS


def fuse_in_stages(plan, session, stages):
    """Fuse with the given stage lengths, then run whatever is left."""

    storage = allocate_empty_tsdf_blocks(plan, session)
    progress = None
    trail = []
    for length in stages:
        progress = advance_tsdf_plan_streaming(
            storage, session, progress, observations=length
        )
        trail.append(progress)
    if progress is None or not progress.is_complete:
        progress = advance_tsdf_plan_streaming(storage, session, progress)
        trail.append(progress)
    return storage, progress, trail


class StagesAddUpTests(unittest.TestCase):
    def test_any_cut_of_the_room_scan_fuses_to_the_same_bytes(self) -> None:
        room = shared_room_case()
        expected_bytes, expected_receipt = single_pass()
        selected = room.plan.selected_observations
        self.assertEqual(selected, 20)
        for stages in (
            (),
            (1,),
            (7,),
            (19,),
            (20,),
            (25,),
            (5, 5, 5),
            (1, 2, 3, 4),
            (1,) * 20,
        ):
            with self.subTest(stages=stages):
                storage, progress, trail = fuse_in_stages(
                    room.plan, room.session, stages
                )
                self.assertEqual(storage_bytes(storage), expected_bytes)
                self.assertTrue(progress.is_complete)
                self.assertEqual(
                    finish_tsdf_plan_streaming(
                        storage, room.session, progress
                    ),
                    expected_receipt,
                )
                # Each stage ends where its length says, short of the end.
                position = 0
                for length, reached in zip(stages, trail):
                    position = min(selected, position + length)
                    self.assertEqual(
                        reached.processed_observations, position
                    )

    def test_the_state_between_stages_is_the_first_frames_and_no_more(
        self,
    ) -> None:
        room = shared_room_case()
        rows = len(room.plan.active_blocks)
        storage = allocate_empty_tsdf_blocks(room.plan, room.session)
        reference = allocate_empty_tsdf_blocks(room.plan, room.session)
        ledger = begin_tsdf_observation_ledger(room.plan, room.context)
        progress = None
        # The ledger path is slow, so three frames: enough for the second
        # and third to add to sums the first left behind.
        for frames in (1, 2, 3):
            with self.subTest(frames=frames):
                progress = advance_tsdf_plan_streaming(
                    storage, room.session, progress, observations=1
                )
                # The ledger path walks pairs frame by frame, every row of
                # one frame before the next, so a pass limited to as many
                # pairs as there are rows absorbs exactly one frame.
                ledger = fuse_tsdf_plan_observations_from_context(
                    reference,
                    room.context,
                    ledger,
                    pair_limit=rows,
                ).ledger_after
                self.assertEqual(ledger.absorbed_pair_count, frames * rows)
                self.assertEqual(
                    storage_bytes(storage), storage_bytes(reference)
                )
                self.assertEqual(progress.processed_observations, frames)
                self.assertEqual(progress.fused_observations, frames)
                self.assertEqual(
                    progress.remaining_observations,
                    room.plan.selected_observations - frames,
                )
                self.assertFalse(progress.is_complete)
                self.assertEqual(
                    progress.evaluated_count,
                    rows * TSDF_BLOCK_VOXELS * frames,
                )
                self.assertEqual(
                    progress.applied_count, int(storage.weights.sum())
                )
                self.assertEqual(
                    progress.tsdf_sums_sha256,
                    storage_payload_sha256(storage.tsdf_sums),
                )
                self.assertEqual(
                    progress.weights_sha256,
                    storage_payload_sha256(storage.weights),
                )
        # Not a trivial agreement: the frames overlap, so later ones added
        # to voxels earlier ones had already written.
        self.assertGreater(int(storage.weights.max()), 1)

    def test_frames_skipped_for_a_missing_input_count_as_processed(
        self,
    ) -> None:
        for stream in ("depth", "poses"):
            with self.subTest(missing=stream):
                with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
                    root = Path(temporary)
                    session_path = root / "case.vgsession"
                    shutil.copytree(FIXTURE, session_path)
                    remove_first_record(
                        session_path / "streams" / f"{stream}.jsonl"
                    )
                    plan, session = load_case(root, session_path=session_path)
                    whole = allocate_empty_tsdf_blocks(plan, session)
                    receipt = fuse_tsdf_plan_streaming(whole, session)
                    storage = allocate_empty_tsdf_blocks(plan, session)
                    first = advance_tsdf_plan_streaming(
                        storage, session, observations=1
                    )
                    second = advance_tsdf_plan_streaming(
                        storage, session, first, observations=1
                    )
                    finished = finish_tsdf_plan_streaming(
                        storage, session, second
                    )

                self.assertEqual(receipt.selected_observations, 2)
                self.assertEqual(receipt.fused_observations, 1)
                # The frame with the missing record is the first one. Its
                # stage fuses nothing and is still a stage that happened.
                self.assertEqual(first.processed_observations, 1)
                self.assertEqual(first.fused_observations, 0)
                self.assertEqual(first.applied_count, 0)
                self.assertEqual(first.evaluated_count, first.voxel_slots)
                self.assertEqual(
                    first.skipped_missing_depth + first.skipped_missing_pose,
                    1,
                )
                self.assertEqual(second.processed_observations, 2)
                self.assertEqual(second.fused_observations, 1)
                self.assertEqual(storage_bytes(storage), storage_bytes(whole))
                self.assertEqual(finished, receipt)


class StageGuardTests(unittest.TestCase):
    def advanced(self, frames: int = 4):
        room = shared_room_case()
        storage = allocate_empty_tsdf_blocks(room.plan, room.session)
        progress = advance_tsdf_plan_streaming(
            storage, room.session, observations=frames
        )
        return room, storage, progress

    def test_storage_that_moved_on_is_refused_untouched(self) -> None:
        room, storage, progress = self.advanced()
        for name, mutate in (
            ("a weight", lambda s: s.weights.__setitem__((0, 0, 0, 0), 9)),
            ("a sum", lambda s: s.tsdf_sums.__setitem__((1, 2, 3, 4), 0.25)),
        ):
            with self.subTest(changed=name):
                changed = allocate_empty_tsdf_blocks(room.plan, room.session)
                changed.tsdf_sums[...] = storage.tsdf_sums
                changed.weights[...] = storage.weights
                mutate(changed)
                before = storage_bytes(changed)
                with self.assertRaises(TsdfError) as caught:
                    advance_tsdf_plan_streaming(
                        changed, room.session, progress
                    )
                self.assertIn(
                    "does not hold the bytes", str(caught.exception)
                )
                self.assertEqual(storage_bytes(changed), before)
                with self.assertRaises(TsdfError):
                    finish_tsdf_plan_streaming(
                        changed, room.session, progress
                    )

    def test_progress_from_another_plan_is_refused_untouched(self) -> None:
        room, storage, progress = self.advanced()
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            other_path = Path(temporary) / "other.sftplan"
            plan_tsdf_blocks(
                load_scan_session(room.session_path),
                other_path,
                frame_stride=2,
                voxel_size_m=room.plan.voxel_size_m,
                truncation_m=room.plan.truncation_m,
            )
            other_plan = load_tsdf_block_plan(other_path)
            other = allocate_empty_tsdf_blocks(other_plan, room.session)
            before = storage_bytes(other)
            with self.assertRaises(TsdfError) as caught:
                advance_tsdf_plan_streaming(other, room.session, progress)
        self.assertIn("does not describe this plan", str(caught.exception))
        self.assertEqual(storage_bytes(other), before)

    def test_nothing_is_fused_past_the_end(self) -> None:
        room = shared_room_case()
        storage, progress, _ = fuse_in_stages(room.plan, room.session, (20,))
        before = storage_bytes(storage)
        with self.assertRaises(TsdfError) as caught:
            advance_tsdf_plan_streaming(storage, room.session, progress)
        self.assertIn("no observation left", str(caught.exception))
        self.assertEqual(storage_bytes(storage), before)

    def test_unfinished_stages_have_no_receipt(self) -> None:
        room, storage, progress = self.advanced()
        with self.assertRaises(TsdfError) as caught:
            finish_tsdf_plan_streaming(storage, room.session, progress)
        self.assertIn("4 of 20", str(caught.exception))

    def test_a_stage_that_fails_leaves_nothing_to_continue_from(self) -> None:
        from spatialforge import tsdf_stream_fusion

        room, storage, progress = self.advanced()
        real = tsdf_stream_fusion._decode_metric_depth
        calls = []

        def fail_on_second_frame(*arguments):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("simulated decoding failure")
            return real(*arguments)

        with patch(
            "spatialforge.tsdf_stream_fusion._decode_metric_depth",
            side_effect=fail_on_second_frame,
        ):
            with self.assertRaises(TsdfError) as caught:
                advance_tsdf_plan_streaming(storage, room.session, progress)
        self.assertIn("simulated decoding failure", str(caught.exception))
        # One frame of the stage went in before the failure. The storage is
        # cleared rather than left holding it, and the old progress, which
        # described four frames, no longer matches.
        self.assertEqual(int(np.count_nonzero(storage.weights)), 0)
        self.assertEqual(int(np.count_nonzero(storage.tsdf_sums)), 0)
        with self.assertRaises(TsdfError) as stale:
            advance_tsdf_plan_streaming(storage, room.session, progress)
        self.assertIn("does not hold the bytes", str(stale.exception))

    def test_invalid_arguments_are_refused(self) -> None:
        room, storage, progress = self.advanced()
        before = storage_bytes(storage)
        for name, call, message in (
            (
                "a length of zero",
                lambda: advance_tsdf_plan_streaming(
                    storage, room.session, progress, observations=0
                ),
                "positive integer",
            ),
            (
                "a length that is a bool",
                lambda: advance_tsdf_plan_streaming(
                    storage, room.session, progress, observations=True
                ),
                "positive integer",
            ),
            (
                "progress that is not progress",
                lambda: advance_tsdf_plan_streaming(
                    storage, room.session, object()  # type: ignore
                ),
                "TsdfStreamFusionProgress",
            ),
            (
                "no progress to finish",
                lambda: finish_tsdf_plan_streaming(
                    storage, room.session, None  # type: ignore
                ),
                "TsdfStreamFusionProgress",
            ),
            (
                "no storage",
                lambda: advance_tsdf_plan_streaming(
                    None, room.session  # type: ignore
                ),
                "TsdfBlockStorage",
            ),
            (
                "no session",
                lambda: finish_tsdf_plan_streaming(
                    storage, None, progress  # type: ignore
                ),
                "ScanSession",
            ),
        ):
            with self.subTest(case=name):
                with self.assertRaises(TsdfError) as caught:
                    call()
                self.assertIn(message, str(caught.exception))
        self.assertEqual(storage_bytes(storage), before)


class ProgressRecordTests(unittest.TestCase):
    def genuine(self) -> TsdfStreamFusionProgress:
        room = shared_room_case()
        storage = allocate_empty_tsdf_blocks(room.plan, room.session)
        return advance_tsdf_plan_streaming(
            storage, room.session, observations=4
        )

    def test_progress_is_frozen(self) -> None:
        progress = self.genuine()
        with self.assertRaises(FrozenInstanceError):
            progress.processed_observations = 5  # type: ignore[misc]

    def test_a_genuine_record_rebuilt_field_for_field_is_accepted(
        self,
    ) -> None:
        progress = self.genuine()
        self.assertEqual(replace(progress), progress)

    def test_a_record_that_could_not_be_true_is_refused(self) -> None:
        progress = self.genuine()
        contributes = dict(progress.status_counts)[
            TsdfContributionStatus.CONTRIBUTES
        ]
        cases = (
            ("more processed than selected", {"processed_observations": 21}),
            ("more fused than processed", {"fused_observations": 5}),
            (
                "a frame processed that the counts do not cover",
                {"processed_observations": 5, "fused_observations": 5},
            ),
            ("a skip nothing accounts for", {"fused_observations": 3}),
            ("a skip that did not happen", {"skipped_missing_depth": 1}),
            ("slots that are not its blocks", {"voxel_slots": 7}),
            ("a selection its stride denies", {"selected_observations": 19}),
            ("a negative count", {"valid_depth_samples": -1}),
            ("a count that is a bool", {"invalid_depth_samples": True}),
            ("a digest that is not one", {"tsdf_sums_sha256": "0" * 63}),
            ("a plan digest that is not one", {
                "source_plan_digest_sha256": "g" * 64,
            }),
            ("no session", {"session_id": ""}),
            (
                "status counts out of order",
                {"status_counts": tuple(reversed(progress.status_counts))},
            ),
            (
                "a status counted twice",
                {
                    "status_counts": progress.status_counts[:1]
                    + progress.status_counts,
                },
            ),
            (
                "status counts as a list",
                {"status_counts": list(progress.status_counts)},
            ),
            (
                "a status with nothing in it",
                {
                    "status_counts": tuple(
                        (status, 0 if count == contributes else count)
                        for status, count in progress.status_counts
                    ),
                },
            ),
        )
        for name, changes in cases:
            with self.subTest(case=name):
                with self.assertRaises(TsdfError):
                    replace(progress, **changes)

    def test_frames_that_were_not_fused_report_no_depth(self) -> None:
        progress = self.genuine()
        skipped = TsdfContributionStatus.MISSING_DEPTH
        only_skips = {
            "fused_observations": 0,
            "skipped_missing_depth": 4,
            "status_counts": ((skipped, progress.voxel_slots * 4),),
        }
        with self.assertRaises(TsdfError) as caught:
            replace(progress, **only_skips)
        self.assertIn("did not fuse", str(caught.exception))
        accepted = replace(
            progress,
            valid_depth_samples=0,
            invalid_depth_samples=0,
            **only_skips,
        )
        self.assertEqual(accepted.applied_count, 0)


if __name__ == "__main__":
    unittest.main()
