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
from spatialforge.tsdf_plan_fusion import (
    begin_tsdf_fusion_ledger,
    fuse_tsdf_plan_blocks_from_context,
)
from spatialforge.tsdf_plan_traversal import (
    traverse_tsdf_plan_blocks_from_context,
)
from spatialforge.tsdf_replay_depth_context import TsdfReplayDepthContext

from tests.heavy_fixtures import shared_case

TEST_ROOT = Path(__file__).resolve().parent
FIXTURE = TEST_ROOT / "fixtures" / "minimal.vgsession"
PLAN_ARGUMENTS = {"voxel_size_m": 0.125, "truncation_m": 0.5}


def create_plan(parent: Path, *, name: str = "fixture.sftplan") -> Path:
    output = parent / name
    plan_tsdf_blocks(
        load_scan_session(FIXTURE),
        output,
        frame_stride=1,
        **PLAN_ARGUMENTS,
    )
    return output


def storage_bytes(storage) -> tuple[bytes, bytes]:
    return (storage.tsdf_sums.tobytes(), storage.weights.tobytes())


def fresh_storage(plan: TsdfBlockPlan):
    """Storage is mutable, so every test gets its own zeroed allocation."""

    return allocate_empty_tsdf_blocks(plan, load_scan_session(FIXTURE))


def shared_plan_and_context() -> tuple[TsdfBlockPlan, TsdfReplayDepthContext]:
    case = shared_case()
    return case.plan, case.context


class TsdfPlanFusionTests(unittest.TestCase):
    def test_chunked_fusion_equals_one_shot_traversal(self) -> None:
        # The plan, context and one-shot reference are identical for every
        # chunk size, so build them once and vary only the resumption.
        plan, context = shared_plan_and_context()
        reference = fresh_storage(plan)
        one_shot = traverse_tsdf_plan_blocks_from_context(reference, context)
        expected = storage_bytes(reference)

        for block_limit, expected_passes in ((1, 8), (3, 3), (None, 1)):
            with self.subTest(block_limit=block_limit):
                storage = fresh_storage(plan)
                ledger = begin_tsdf_fusion_ledger(plan)
                passes = 0
                weight_total = 0
                while not ledger.is_complete:
                    receipt = fuse_tsdf_plan_blocks_from_context(
                        storage,
                        context,
                        ledger,
                        block_limit=block_limit,
                    )
                    ledger = receipt.ledger_after
                    weight_total += receipt.weight_delta
                    passes += 1
                    self.assertLessEqual(passes, 8)

                self.assertEqual(storage_bytes(storage), expected)
                self.assertEqual(weight_total, one_shot.weight_delta)
                self.assertEqual(weight_total, 1168)
                self.assertTrue(ledger.is_complete)
                self.assertEqual(ledger.fused_block_count, 8)
                self.assertEqual(passes, expected_passes)

    def test_repeating_a_completed_pass_is_a_no_op(self) -> None:
        plan, context = shared_plan_and_context()
        storage = fresh_storage(plan)
        ledger = begin_tsdf_fusion_ledger(plan)

        first = fuse_tsdf_plan_blocks_from_context(storage, context, ledger)
        after_first = storage_bytes(storage)
        repeated = fuse_tsdf_plan_blocks_from_context(
            storage,
            context,
            first.ledger_after,
        )
        again = fuse_tsdf_plan_blocks_from_context(
            storage,
            context,
            repeated.ledger_after,
        )

        self.assertEqual(first.blocks_fused_now, 8)
        self.assertEqual(repeated.blocks_fused_now, 0)
        self.assertEqual(again.blocks_fused_now, 0)
        self.assertEqual(repeated.weight_delta, 0)
        self.assertEqual(repeated.block_receipts, ())
        self.assertEqual(storage_bytes(storage), after_first)
        self.assertEqual(
            repeated.ledger_after.fused_block_indices,
            first.ledger_after.fused_block_indices,
        )

    def test_partial_ledger_resumes_only_pending_rows(self) -> None:
        plan, context = shared_plan_and_context()
        storage = fresh_storage(plan)
        ledger = begin_tsdf_fusion_ledger(plan)

        first = fuse_tsdf_plan_blocks_from_context(
            storage,
            context,
            ledger,
            block_limit=2,
        )
        second = fuse_tsdf_plan_blocks_from_context(
            storage,
            context,
            first.ledger_after,
            block_limit=2,
        )

        self.assertEqual(first.blocks_fused_now, 2)
        self.assertEqual(first.blocks_already_fused, 0)
        self.assertEqual(first.blocks_pending, 6)
        self.assertEqual(
            first.ledger_after.fused_block_indices,
            plan.active_blocks[:2],
        )
        self.assertEqual(second.blocks_fused_now, 2)
        self.assertEqual(second.blocks_already_fused, 2)
        self.assertEqual(second.blocks_pending, 4)
        self.assertEqual(
            second.ledger_after.fused_block_indices,
            plan.active_blocks[:4],
        )
        self.assertEqual(
            tuple(r.block_index_xyz for r in second.block_receipts),
            plan.active_blocks[2:4],
        )
        self.assertFalse(second.is_complete)

    def test_unclaimed_nonempty_row_is_rejected(self) -> None:
        plan, context = shared_plan_and_context()
        storage = fresh_storage(plan)
        ledger = begin_tsdf_fusion_ledger(plan)
        storage.weights[3][0][0][0] = 7

        with self.assertRaises(TsdfError) as raised:
            fuse_tsdf_plan_blocks_from_context(storage, context, ledger)

        self.assertIn("does not claim", str(raised.exception))

    def test_ledger_provenance_is_preflighted(self) -> None:
        plan, context = shared_plan_and_context()
        storage = fresh_storage(plan)
        ledger = begin_tsdf_fusion_ledger(plan)
        cases = (
            ("digest", replace(ledger, source_plan_digest_sha256="0" * 64)),
            ("replay", replace(ledger, replay_digest_sha256="0" * 64)),
            (
                "rows",
                replace(
                    ledger,
                    plan_block_indices=plan.active_blocks[:4],
                    fused_block_indices=(),
                ),
            ),
        )
        for name, candidate in cases:
            with self.subTest(name=name):
                with self.assertRaises(TsdfError) as raised:
                    fuse_tsdf_plan_blocks_from_context(
                        storage,
                        context,
                        candidate,
                    )
                self.assertIn(
                    "provenance does not match",
                    str(raised.exception),
                )

        for name, bad in (
            ("storage", (object(), context, ledger)),
            ("context", (storage, object(), ledger)),
            ("ledger", (storage, context, object())),
        ):
            with self.subTest(name=name):
                with self.assertRaises(TsdfError):
                    fuse_tsdf_plan_blocks_from_context(*bad)

        with self.assertRaises(TsdfError):
            fuse_tsdf_plan_blocks_from_context(
                storage,
                context,
                ledger,
                block_limit=0,
            )
        self.assertEqual(
            storage_bytes(storage),
            storage_bytes(fresh_storage(plan)),
        )

    def test_failed_pass_restores_only_its_own_rows(self) -> None:
        plan, context = shared_plan_and_context()
        storage = fresh_storage(plan)
        ledger = begin_tsdf_fusion_ledger(plan)

        done = fuse_tsdf_plan_blocks_from_context(
            storage,
            context,
            ledger,
            block_limit=3,
        )
        after_first = storage_bytes(storage)
        injected = RuntimeError("injected fusion pass failure")

        with patch(
            "spatialforge.tsdf_plan_fusion.TsdfPlanFusionReceipt",
            side_effect=injected,
        ):
            with self.assertRaises(TsdfError) as raised:
                fuse_tsdf_plan_blocks_from_context(
                    storage,
                    context,
                    done.ledger_after,
                    block_limit=3,
                )

        # The three rows fused before the failing pass survive; the rows that
        # pass touched are zeroed, so storage still matches the ledger the
        # caller is still holding and the work can simply be retried.
        self.assertIs(raised.exception.__cause__, injected)
        self.assertEqual(storage_bytes(storage), after_first)

        resumed = fuse_tsdf_plan_blocks_from_context(
            storage,
            context,
            done.ledger_after,
        )
        self.assertEqual(resumed.blocks_fused_now, 5)
        self.assertTrue(resumed.is_complete)

    def test_ledger_and_receipt_are_frozen_and_strict(self) -> None:
        plan, context = shared_plan_and_context()
        storage = fresh_storage(plan)
        ledger = begin_tsdf_fusion_ledger(plan)
        receipt = fuse_tsdf_plan_blocks_from_context(
            storage,
            context,
            ledger,
            block_limit=2,
        )

        self.assertFalse(hasattr(ledger, "__dict__"))
        self.assertFalse(hasattr(receipt, "__dict__"))
        with self.assertRaises(FrozenInstanceError):
            ledger.fused_block_indices = ()  # type: ignore[misc]

        for arguments in (
            {"plan_block_indices": ()},
            {"fused_block_indices": ((99, 99, 99),)},
            {"fused_block_indices": tuple(reversed(plan.active_blocks))},
            {"source_plan_digest_sha256": "0"},
            {"replay_digest_sha256": "0"},
        ):
            with self.subTest(arguments=arguments):
                with self.assertRaises(TsdfError):
                    replace(ledger, **arguments)

        for arguments in (
            {"block_receipts": ()},
            {"ledger_after": receipt.ledger_before},
            {"selected_observation_sequences": (0,)},
            {"source_plan_digest_sha256": "0"},
        ):
            with self.subTest(arguments=arguments):
                with self.assertRaises(TsdfError):
                    replace(receipt, **arguments)

    def test_fusion_never_unfuses_a_row(self) -> None:
        plan, context = shared_plan_and_context()
        storage = fresh_storage(plan)
        receipt = fuse_tsdf_plan_blocks_from_context(
            storage,
            context,
            begin_tsdf_fusion_ledger(plan),
            block_limit=4,
        )

        self.assertTrue(
            set(receipt.ledger_before.fused_block_indices)
            <= set(receipt.ledger_after.fused_block_indices)
        )
        with self.assertRaises(TsdfError):
            replace(
                receipt,
                ledger_before=receipt.ledger_after,
                ledger_after=receipt.ledger_before,
                block_receipts=(),
            )


class TsdfPlanFusionCliTests(unittest.TestCase):
    def test_cli_reports_resumable_passes(self) -> None:
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
                        "tsdf-block-context-plan-fuse",
                        str(plan_path),
                        str(session_path),
                        "--block-limit",
                        "3",
                    ]
                )
            after = sorted(p.name for p in root.rglob("*"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(after, before)

        output = stdout.getvalue()
        for expected in (
            "TSDF BLOCK CONTEXT PLAN FUSION CHECK scan-synthetic-0001\n",
            "fusion_passes: count=3 block_limit=3\n",
            "pass_0: fused=3 already_fused=0 pending_after=5 "
            "weight_delta=380\n",
            "pass_1: fused=3 already_fused=3 pending_after=2 "
            "weight_delta=496\n",
            "pass_2: fused=2 already_fused=6 pending_after=0 "
            "weight_delta=292\n",
            "ledger: fused=8 pending=0 complete=yes\n",
            "repeat_pass: fused=0 pending=0\n",
            "idempotent_repeat: yes\n",
            "storage_after: nonzero_sums=584 nonzero_weights=584 "
            "unknown_voxels=3512\n",
            "contributions_applied: 1168\n",
            "plan_weight_sum_after: 1168\n",
            "empty_storage_required: no\n",
            "resumable_fusion_performed: yes\n",
            "ledger_guard: nonempty-unclaimed-row-rejected\n",
            "caught_failure_rollback_scope: rows-fused-by-this-pass\n",
            "observation_level_ledger: no\n",
            "ledger_persisted: no\n",
            "artifact_written: no\n",
        ):
            self.assertIn(expected, output)

    def test_cli_reports_failure_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            plan_path = create_plan(Path(temporary))
            stdout = io.StringIO()
            stderr = io.StringIO()

            with (
                patch(
                    "spatialforge.cli.fuse_tsdf_plan_blocks_from_context",
                    side_effect=TsdfError("injected fusion failure"),
                ),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                exit_code = main(
                    [
                        "reconstruct",
                        "tsdf-block-context-plan-fuse",
                        str(plan_path),
                        str(FIXTURE),
                    ]
                )

        self.assertEqual(exit_code, 2)
        self.assertEqual(stdout.getvalue(), "")
        error = stderr.getvalue()
        self.assertIn("TSDF BLOCK CONTEXT PLAN FUSION FAILED", error)
        self.assertIn("injected fusion failure", error)
        self.assertNotIn("Traceback", error)


if __name__ == "__main__":
    unittest.main()
