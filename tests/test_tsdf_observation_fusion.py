from __future__ import annotations

import io
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from unittest.mock import patch

from spatialforge import allocate_empty_tsdf_blocks
from spatialforge.cli import main
from spatialforge.errors import TsdfError
from spatialforge.session_loader import load_scan_session
from spatialforge.tsdf_block_plan import plan_tsdf_blocks
from spatialforge.tsdf_block_plan_loader import TsdfBlockPlan
from spatialforge.tsdf_observation_fusion import (
    begin_tsdf_observation_ledger,
    fuse_tsdf_plan_observations_from_context,
)
from spatialforge.tsdf_plan_traversal import (
    traverse_tsdf_plan_blocks_from_context,
)
from spatialforge.tsdf_replay_depth_context import TsdfReplayDepthContext

from tests.heavy_fixtures import shared_case

TEST_ROOT = Path(__file__).resolve().parent
FIXTURE = TEST_ROOT / "fixtures" / "minimal.vgsession"
PLAN_ARGUMENTS = {"voxel_size_m": 0.125, "truncation_m": 0.5}


def storage_bytes(storage) -> tuple[bytes, bytes]:
    return (storage.tsdf_sums.tobytes(), storage.weights.tobytes())


def fresh_storage(plan: TsdfBlockPlan):
    return allocate_empty_tsdf_blocks(plan, load_scan_session(FIXTURE))


def shared_plan_and_context() -> tuple[TsdfBlockPlan, TsdfReplayDepthContext]:
    case = shared_case()
    return case.plan, case.context


class TsdfObservationFusionTests(unittest.TestCase):
    def test_frame_major_chunks_equal_the_one_shot_traversal(self) -> None:
        # Chunking must not perturb the float64 accumulation, so every pair
        # limit has to reproduce the one-shot bytes exactly.
        plan, context = shared_plan_and_context()
        reference = fresh_storage(plan)
        one_shot = traverse_tsdf_plan_blocks_from_context(reference, context)
        expected = storage_bytes(reference)

        for pair_limit, expected_passes in (
            (1, 16),
            (5, 4),
            (8, 2),
            (None, 1),
        ):
            with self.subTest(pair_limit=pair_limit):
                storage = fresh_storage(plan)
                ledger = begin_tsdf_observation_ledger(plan, context)
                passes = 0
                applied = 0
                while not ledger.is_complete:
                    receipt = fuse_tsdf_plan_observations_from_context(
                        storage,
                        context,
                        ledger,
                        pair_limit=pair_limit,
                    )
                    ledger = receipt.ledger_after
                    applied += receipt.applied_count
                    passes += 1
                    self.assertLessEqual(passes, 16)

                self.assertEqual(storage_bytes(storage), expected)
                self.assertEqual(applied, one_shot.weight_delta)
                self.assertEqual(applied, 1168)
                self.assertEqual(passes, expected_passes)
                self.assertTrue(ledger.is_complete)
                self.assertEqual(ledger.absorbed_pair_count, 16)

    def test_a_row_can_hold_a_partial_observation_prefix(self) -> None:
        plan, context = shared_plan_and_context()
        storage = fresh_storage(plan)
        ledger = begin_tsdf_observation_ledger(plan, context)

        # Eight pairs is exactly observation 0 across all eight rows, so
        # every row ends up holding a one-observation prefix.
        receipt = fuse_tsdf_plan_observations_from_context(
            storage,
            context,
            ledger,
            pair_limit=8,
        )

        self.assertEqual(receipt.pairs_fused_now, 8)
        self.assertEqual(receipt.pairs_pending, 8)
        self.assertFalse(receipt.is_complete)
        self.assertEqual(
            receipt.ledger_after.absorbed_counts,
            (1,) * 8,
        )
        self.assertEqual(
            tuple(sequence for _, sequence in receipt.fused_pairs),
            (0,) * 8,
        )
        self.assertEqual(
            tuple(block for block, _ in receipt.fused_pairs),
            plan.active_blocks,
        )
        for block_index in plan.active_blocks:
            self.assertEqual(receipt.ledger_after.absorbed_for(block_index), 1)
        self.assertEqual(receipt.ledger_after.untouched_block_indices, ())
        # A one-observation prefix can only produce weight one anywhere.
        self.assertLessEqual(receipt.applied_count, 512 * 8)

    def test_repeating_a_completed_pass_is_a_no_op(self) -> None:
        plan, context = shared_plan_and_context()
        storage = fresh_storage(plan)
        ledger = begin_tsdf_observation_ledger(plan, context)

        first = fuse_tsdf_plan_observations_from_context(
            storage,
            context,
            ledger,
        )
        after = storage_bytes(storage)
        repeated = fuse_tsdf_plan_observations_from_context(
            storage,
            context,
            first.ledger_after,
        )

        self.assertTrue(first.is_complete)
        self.assertEqual(repeated.pairs_fused_now, 0)
        self.assertEqual(repeated.applied_count, 0)
        self.assertEqual(repeated.evaluated_count, 0)
        self.assertEqual(repeated.fused_pairs, ())
        self.assertEqual(storage_bytes(storage), after)
        self.assertEqual(
            repeated.ledger_after.absorbed_counts,
            first.ledger_after.absorbed_counts,
        )

    def test_untouched_row_must_still_be_empty(self) -> None:
        plan, context = shared_plan_and_context()
        storage = fresh_storage(plan)
        ledger = begin_tsdf_observation_ledger(plan, context)
        storage.weights[5][0][0][0] = 3

        with self.assertRaises(TsdfError) as raised:
            fuse_tsdf_plan_observations_from_context(storage, context, ledger)

        self.assertIn("records as untouched", str(raised.exception))

    def test_ledger_provenance_and_arguments_are_preflighted(self) -> None:
        plan, context = shared_plan_and_context()
        storage = fresh_storage(plan)
        ledger = begin_tsdf_observation_ledger(plan, context)

        for name, candidate in (
            ("digest", replace(ledger, source_plan_digest_sha256="0" * 64)),
            ("replay", replace(ledger, replay_digest_sha256="0" * 64)),
            (
                "rows",
                replace(
                    ledger,
                    plan_block_indices=plan.active_blocks[:4],
                    absorbed_counts=(0, 0, 0, 0),
                ),
            ),
            (
                "selection",
                replace(ledger, selected_observation_sequences=(0,)),
            ),
        ):
            with self.subTest(name=name):
                with self.assertRaises(TsdfError) as raised:
                    fuse_tsdf_plan_observations_from_context(
                        storage,
                        context,
                        candidate,
                    )
                self.assertIn(
                    "does not match",
                    str(raised.exception),
                )

        for name, bad in (
            ("storage", (object(), context, ledger)),
            ("context", (storage, object(), ledger)),
            ("ledger", (storage, context, object())),
        ):
            with self.subTest(name=name):
                with self.assertRaises(TsdfError):
                    fuse_tsdf_plan_observations_from_context(*bad)

        with self.assertRaises(TsdfError):
            fuse_tsdf_plan_observations_from_context(
                storage,
                context,
                ledger,
                pair_limit=0,
            )
        self.assertEqual(
            storage_bytes(storage),
            storage_bytes(fresh_storage(plan)),
        )

    def test_failed_pass_restores_the_rows_it_wrote(self) -> None:
        plan, context = shared_plan_and_context()
        storage = fresh_storage(plan)
        ledger = begin_tsdf_observation_ledger(plan, context)

        done = fuse_tsdf_plan_observations_from_context(
            storage,
            context,
            ledger,
            pair_limit=8,
        )
        after_first = storage_bytes(storage)
        injected = RuntimeError("injected observation pass failure")

        with patch(
            "spatialforge.tsdf_observation_fusion."
            "TsdfObservationFusionReceipt",
            side_effect=injected,
        ):
            with self.assertRaises(TsdfError) as raised:
                fuse_tsdf_plan_observations_from_context(
                    storage,
                    context,
                    done.ledger_after,
                    pair_limit=8,
                )

        # Unlike the block ledger, a failing pass here may have written rows
        # that already held earlier observations, so rollback restores the
        # exact pre-pass bytes rather than zeroing.
        self.assertIs(raised.exception.__cause__, injected)
        self.assertEqual(storage_bytes(storage), after_first)

        resumed = fuse_tsdf_plan_observations_from_context(
            storage,
            context,
            done.ledger_after,
        )
        self.assertEqual(resumed.pairs_fused_now, 8)
        self.assertTrue(resumed.is_complete)

    def test_ledger_and_receipt_are_frozen_and_strict(self) -> None:
        plan, context = shared_plan_and_context()
        storage = fresh_storage(plan)
        ledger = begin_tsdf_observation_ledger(plan, context)
        receipt = fuse_tsdf_plan_observations_from_context(
            storage,
            context,
            ledger,
            pair_limit=4,
        )

        self.assertFalse(hasattr(ledger, "__dict__"))
        self.assertFalse(hasattr(receipt, "__dict__"))
        with self.assertRaises(FrozenInstanceError):
            ledger.absorbed_counts = ()  # type: ignore[misc]

        for arguments in (
            {"plan_block_indices": ()},
            {"selected_observation_sequences": ()},
            {"absorbed_counts": (0,)},
            {"absorbed_counts": (0,) * 7 + (3,)},
            {"absorbed_counts": (0,) * 7 + (-1,)},
            {"source_plan_digest_sha256": "0"},
            {"replay_digest_sha256": "0"},
        ):
            with self.subTest(arguments=arguments):
                with self.assertRaises(TsdfError):
                    replace(ledger, **arguments)

        for arguments in (
            {"fused_pairs": ()},
            {"applied_count": receipt.applied_count + 1},
            {"evaluated_count": 0},
            {"ledger_after": receipt.ledger_before},
        ):
            with self.subTest(arguments=arguments):
                with self.assertRaises(TsdfError):
                    replace(receipt, **arguments)

    def test_a_receipt_cannot_describe_a_pass_that_did_not_happen(
        self,
    ) -> None:
        plan, context = shared_plan_and_context()
        storage = fresh_storage(plan)
        receipt = fuse_tsdf_plan_observations_from_context(
            storage,
            context,
            begin_tsdf_observation_ledger(plan, context),
            pair_limit=4,
        )
        # A genuine receipt rebuilt field for field is accepted, so each
        # refusal below is about the one field that was changed.
        self.assertEqual(replace(receipt), receipt)
        self.assertEqual(len(receipt.fused_pairs), 4)
        self.assertGreater(len(receipt.status_counts), 1)
        self.assertGreater(receipt.applied_count, 0)
        statuses = dict(receipt.status_counts)
        other = "f" * 64
        self.assertNotEqual(receipt.replay_digest_sha256, other)
        first, second = receipt.status_counts[:2]

        cases = {
            # Provenance that does not match the ledgers it carries.
            "another replay": (
                {"replay_digest_sha256": other},
                "provenance is inconsistent",
            ),
            "another plan": (
                {"source_plan_digest_sha256": other},
                "provenance is inconsistent",
            ),
            "not a digest": (
                {"replay_digest_sha256": "replay"},
                "replay digest is invalid",
            ),
            # The right number of pairs, but not the ones the ledgers say.
            "pairs reversed": (
                {"fused_pairs": receipt.fused_pairs[::-1]},
                "must equal the ledger delta",
            ),
            "a pair repeated": (
                {"fused_pairs": (receipt.fused_pairs[0],) * 4},
                "must equal the ledger delta",
            ),
            "a pair from nowhere": (
                {
                    "fused_pairs": receipt.fused_pairs[:3]
                    + (((99, 99, 99), 0),)
                },
                "must equal the ledger delta",
            ),
            # Counts that cannot all be true at once.
            "more applied than evaluated": (
                {
                    "applied_count": receipt.evaluated_count + 1,
                    "weight_delta": receipt.evaluated_count + 1,
                },
                "cannot apply more contributions than it evaluated",
            ),
            "applied disagrees with the statuses": (
                {
                    "applied_count": receipt.applied_count - 1,
                    "weight_delta": receipt.applied_count - 1,
                },
                "must equal the number of contributing voxels",
            ),
            "statuses missing a voxel": (
                {
                    "status_counts": ((first[0], first[1] - 1),)
                    + receipt.status_counts[1:]
                },
                "must",
            ),
            "statuses out of order": (
                {
                    "status_counts": (second, first)
                    + receipt.status_counts[2:]
                },
                "once, in canonical order",
            ),
            "a status listed twice": (
                {
                    "status_counts": (
                        (first[0], 1),
                        (first[0], first[1] - 1),
                    )
                    + receipt.status_counts[1:]
                },
                "once, in canonical order",
            ),
            "a status with no voxels": (
                {"status_counts": receipt.status_counts + ((first[0], 0),)},
                "with a positive count",
            ),
            "statuses by name": (
                {
                    "status_counts": tuple(
                        (status.value, count)
                        for status, count in receipt.status_counts
                    )
                },
                "with a positive count",
            ),
            "statuses as a list": (
                {"status_counts": list(receipt.status_counts)},
                "must be a tuple",
            ),
        }
        for name, (arguments, message) in cases.items():
            with self.subTest(name=name):
                with self.assertRaises(TsdfError) as raised:
                    replace(receipt, **arguments)
                self.assertIn(message, str(raised.exception))
        self.assertEqual(sum(statuses.values()), receipt.evaluated_count)

    def test_ledger_never_un_absorbs_an_observation(self) -> None:
        plan, context = shared_plan_and_context()
        storage = fresh_storage(plan)
        receipt = fuse_tsdf_plan_observations_from_context(
            storage,
            context,
            begin_tsdf_observation_ledger(plan, context),
            pair_limit=6,
        )

        for before, after in zip(
            receipt.ledger_before.absorbed_counts,
            receipt.ledger_after.absorbed_counts,
        ):
            self.assertLessEqual(before, after)
        with self.assertRaises(TsdfError):
            replace(
                receipt,
                ledger_before=receipt.ledger_after,
                ledger_after=receipt.ledger_before,
            )


class TsdfObservationFusionCliTests(unittest.TestCase):
    def test_cli_reports_frame_major_passes(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            root = Path(temporary)
            session_path = root / "case.vgsession"
            shutil.copytree(FIXTURE, session_path)
            plan_path = root / "fixture.sftplan"
            plan_tsdf_blocks(
                load_scan_session(session_path),
                plan_path,
                frame_stride=1,
                **PLAN_ARGUMENTS,
            )
            stdout = io.StringIO()
            stderr = io.StringIO()
            before = sorted(p.name for p in root.rglob("*"))

            with redirect_stdout(stdout), redirect_stderr(stderr):
                exit_code = main(
                    [
                        "reconstruct",
                        "tsdf-block-context-observation-fuse",
                        str(plan_path),
                        str(session_path),
                        "--pair-limit",
                        "5",
                    ]
                )
            after = sorted(p.name for p in root.rglob("*"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(after, before)

        output = stdout.getvalue()
        for expected in (
            "TSDF BLOCK CONTEXT OBSERVATION FUSION CHECK "
            "scan-synthetic-0001\n",
            "ledger_scope: rows=8 observations=2 pairs=16\n",
            "absorption_order: frame-major-canonical-per-row\n",
            "fusion_passes: count=4 pair_limit=5\n",
            "pass_0: pairs=5 pending_after=11 applied=336\n",
            "pass_3: pairs=1 pending_after=0 applied=102\n",
            "ledger: absorbed=16 pending=0 complete=yes\n",
            "repeat_pass: pairs=0 applied=0\n",
            "idempotent_repeat: yes\n",
            "storage_after: nonzero_sums=584 nonzero_weights=584 "
            "unknown_voxels=3512\n",
            "contributions_applied: 1168\n",
            "plan_weight_sum_after: 1168\n",
            "empty_storage_required: no\n",
            "observation_level_ledger: yes\n",
            "partial_row_absorption_supported: yes\n",
            "ledger_persisted: no\n",
            "artifact_written: no\n",
        ):
            self.assertIn(expected, output)

    def test_cli_reports_failure_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            plan_path = Path(temporary) / "fixture.sftplan"
            plan_tsdf_blocks(
                load_scan_session(FIXTURE),
                plan_path,
                frame_stride=1,
                **PLAN_ARGUMENTS,
            )
            stdout = io.StringIO()
            stderr = io.StringIO()

            with (
                patch(
                    "spatialforge.cli."
                    "fuse_tsdf_plan_observations_from_context",
                    side_effect=TsdfError("injected observation failure"),
                ),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                exit_code = main(
                    [
                        "reconstruct",
                        "tsdf-block-context-observation-fuse",
                        str(plan_path),
                        str(FIXTURE),
                    ]
                )

        self.assertEqual(exit_code, 2)
        self.assertEqual(stdout.getvalue(), "")
        error = stderr.getvalue()
        self.assertIn("TSDF BLOCK CONTEXT OBSERVATION FUSION FAILED", error)
        self.assertIn("injected observation failure", error)
        self.assertNotIn("Traceback", error)


if __name__ == "__main__":
    unittest.main()
