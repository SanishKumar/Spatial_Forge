"""The volume command, stopped and run again.

What ``--checkpoint`` promises is narrow and checkable: however a fusion is
interrupted, running the same command again finishes it, and the ``.sftvol``
that comes out is the one an uninterrupted run writes. The tests stop a run
on purpose, kill one in the middle, fail one at the very last step, and
compare bytes each time.
"""

from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from spatialforge import (
    allocate_empty_tsdf_blocks,
    load_tsdf_block_plan,
)
from spatialforge import cli
from spatialforge.cli import main
from spatialforge.errors import TsdfError
from spatialforge.session_loader import load_scan_session
from spatialforge.tsdf_block_plan import plan_tsdf_blocks
from spatialforge.tsdf_fusion_checkpoint import (
    load_tsdf_fusion_checkpoint,
    write_tsdf_fusion_checkpoint,
)
from spatialforge.tsdf_stream_fusion import advance_tsdf_plan_streaming

from tests.heavy_fixtures import shared_room_case

TEST_ROOT = Path(__file__).resolve().parent
STAGED_ONLY = (
    "resumed_from_selected_observation:",
    "checkpoints_written:",
    "checkpoint_removed:",
    "output:",
)


def run_cli(arguments: list[str]) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        exit_code = main(arguments)
    return exit_code, stdout.getvalue(), stderr.getvalue()


def volume_command(output: Path, *extra: str) -> list[str]:
    room = shared_room_case()
    return [
        "reconstruct",
        "tsdf-block-volume",
        str(room.plan_path),
        str(room.session_path),
        str(output),
        *extra,
    ]


def shared_lines(stdout: str) -> list[str]:
    """A report without the lines only a checkpointed run prints."""

    return [
        line
        for line in stdout.splitlines()
        if not line.startswith(STAGED_ONLY)
    ]


class ResumeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory(dir=TEST_ROOT)
        self.addCleanup(self._directory.cleanup)
        self.root = Path(self._directory.name)
        self.checkpoint = self.root / "progress.sftckpt"
        self.output = self.root / "resumed.sftvol"
        exit_code, self.plain_stdout, stderr = run_cli(
            volume_command(self.root / "plain.sftvol")
        )
        self.assertEqual((exit_code, stderr), (0, ""))
        self.plain = (self.root / "plain.sftvol").read_bytes()

    def files(self) -> list[str]:
        return sorted(path.name for path in self.root.iterdir())

    def test_a_run_stopped_on_purpose_continues_to_the_same_volume(
        self,
    ) -> None:
        exit_code, stdout, stderr = run_cli(
            volume_command(
                self.output,
                "--checkpoint", str(self.checkpoint),
                "--checkpoint-every", "3",
                "--stop-after", "7",
            )
        )
        self.assertEqual((exit_code, stderr), (0, ""))
        self.assertTrue(stdout.startswith("TSDF BLOCK VOLUME PAUSED "))
        self.assertIn(
            "frames: total=20 selected=20 processed=7 fused=7 remaining=13\n",
            stdout,
        )
        self.assertIn("resumed_from_selected_observation: none\n", stdout)
        # Three, three and one.
        self.assertIn("checkpoints_written: 3\n", stdout)
        self.assertIn("volume_written: no\n", stdout)
        self.assertEqual(self.files(), ["plain.sftvol", "progress.sftckpt"])
        saved = load_tsdf_fusion_checkpoint(self.checkpoint)
        self.assertEqual(saved.progress.processed_observations, 7)
        self.assertIn(
            f"checkpoint_sha256: {saved.artifact_digest_sha256}\n", stdout
        )

        exit_code, stdout, stderr = run_cli(
            volume_command(
                self.output,
                "--checkpoint", str(self.checkpoint),
                "--checkpoint-every", "6",
            )
        )
        self.assertEqual((exit_code, stderr), (0, ""))
        self.assertEqual(self.output.read_bytes(), self.plain)
        self.assertIn("resumed_from_selected_observation: 7\n", stdout)
        # Six, six and one.
        self.assertIn("checkpoints_written: 3\n", stdout)
        self.assertIn("checkpoint_removed: yes\n", stdout)
        self.assertEqual(self.files(), ["plain.sftvol", "resumed.sftvol"])
        # Every count and digest the report prints is the plain run's.
        self.assertEqual(shared_lines(stdout), shared_lines(self.plain_stdout))
        self.assertGreater(
            len(stdout.splitlines()), len(shared_lines(stdout)) + 3
        )

    def test_a_run_that_dies_part_way_can_simply_be_run_again(self) -> None:
        real = cli.advance_tsdf_plan_streaming
        calls = []

        def die_on_the_third_stage(*arguments, **keywords):
            calls.append(1)
            if len(calls) == 3:
                raise TsdfError("simulated power cut")
            return real(*arguments, **keywords)

        command = volume_command(
            self.output,
            "--checkpoint", str(self.checkpoint),
            "--checkpoint-every", "4",
        )
        with patch(
            "spatialforge.cli.advance_tsdf_plan_streaming",
            side_effect=die_on_the_third_stage,
        ):
            exit_code, stdout, stderr = run_cli(command)
        self.assertEqual((exit_code, stdout), (2, ""))
        self.assertIn("TSDF BLOCK VOLUME FAILED", stderr)
        self.assertIn("simulated power cut", stderr)
        self.assertIn("the first 8 selected observations are saved", stderr)
        self.assertIn("run the same command to continue", stderr)
        self.assertNotIn("Traceback", stderr)
        self.assertEqual(self.files(), ["plain.sftvol", "progress.sftckpt"])

        exit_code, stdout, stderr = run_cli(command)
        self.assertEqual((exit_code, stderr), (0, ""))
        self.assertIn("resumed_from_selected_observation: 8\n", stdout)
        self.assertEqual(self.output.read_bytes(), self.plain)
        self.assertEqual(self.files(), ["plain.sftvol", "resumed.sftvol"])

    def test_a_run_that_dies_before_its_first_save_leaves_nothing(
        self,
    ) -> None:
        with patch(
            "spatialforge.cli.advance_tsdf_plan_streaming",
            side_effect=TsdfError("simulated early failure"),
        ):
            exit_code, stdout, stderr = run_cli(
                volume_command(
                    self.output, "--checkpoint", str(self.checkpoint)
                )
            )
        self.assertEqual((exit_code, stdout), (2, ""))
        self.assertIn("simulated early failure", stderr)
        self.assertNotIn("are saved", stderr)
        self.assertEqual(self.files(), ["plain.sftvol"])

    def test_a_run_that_fails_writing_the_volume_only_needs_finishing(
        self,
    ) -> None:
        command = volume_command(
            self.output,
            "--checkpoint", str(self.checkpoint),
            "--checkpoint-every", "8",
        )
        with patch(
            "spatialforge.cli.write_tsdf_block_volume",
            side_effect=TsdfError("simulated full disk"),
        ):
            exit_code, _, stderr = run_cli(command)
        self.assertEqual(exit_code, 2)
        self.assertIn("simulated full disk", stderr)
        self.assertIn("the first 20 selected observations are saved", stderr)
        self.assertEqual(self.files(), ["plain.sftvol", "progress.sftckpt"])

        # Everything was fused. The second run must not fuse a frame.
        with patch(
            "spatialforge.cli.advance_tsdf_plan_streaming",
            side_effect=AssertionError("fused again"),
        ):
            exit_code, stdout, stderr = run_cli(command)
        self.assertEqual((exit_code, stderr), (0, ""))
        self.assertIn("resumed_from_selected_observation: 20\n", stdout)
        self.assertIn("checkpoints_written: 0\n", stdout)
        self.assertEqual(self.output.read_bytes(), self.plain)
        self.assertEqual(self.files(), ["plain.sftvol", "resumed.sftvol"])

    def test_stopping_after_more_than_is_left_just_finishes(self) -> None:
        exit_code, stdout, stderr = run_cli(
            volume_command(
                self.output,
                "--checkpoint", str(self.checkpoint),
                "--stop-after", "500",
            )
        )
        self.assertEqual((exit_code, stderr), (0, ""))
        # A finished volume's report, not a pause.
        self.assertTrue(
            stdout.startswith("TSDF BLOCK VOLUME scan-room-0001\n"), stdout
        )
        self.assertIn("checkpoints_written: 1\n", stdout)
        self.assertEqual(self.output.read_bytes(), self.plain)
        self.assertEqual(self.files(), ["plain.sftvol", "resumed.sftvol"])


class RefusalTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory(dir=TEST_ROOT)
        self.addCleanup(self._directory.cleanup)
        self.root = Path(self._directory.name)
        self.checkpoint = self.root / "progress.sftckpt"
        self.output = self.root / "out.sftvol"

    def refused(self, *extra: str) -> str:
        exit_code, stdout, stderr = run_cli(
            volume_command(self.output, *extra)
        )
        self.assertEqual((exit_code, stdout), (2, ""))
        self.assertIn("TSDF BLOCK VOLUME FAILED", stderr)
        self.assertNotIn("Traceback", stderr)
        self.assertFalse(self.output.exists())
        return stderr

    def test_a_file_that_is_not_a_checkpoint_is_left_alone(self) -> None:
        self.checkpoint.write_bytes(b"somebody's notes")
        stderr = self.refused("--checkpoint", str(self.checkpoint))
        self.assertIn("unrecognized signature", stderr)
        self.assertNotIn("are saved", stderr)
        self.assertEqual(self.checkpoint.read_bytes(), b"somebody's notes")

    def test_a_checkpoint_of_another_fusion_is_left_alone(self) -> None:
        room = shared_room_case()
        other_path = self.root / "other.sftplan"
        plan_tsdf_blocks(
            load_scan_session(room.session_path),
            other_path,
            frame_stride=2,
            voxel_size_m=room.plan.voxel_size_m,
            truncation_m=room.plan.truncation_m,
        )
        session = load_scan_session(room.session_path)
        storage = allocate_empty_tsdf_blocks(
            load_tsdf_block_plan(other_path), session
        )
        write_tsdf_fusion_checkpoint(
            storage,
            advance_tsdf_plan_streaming(storage, session, observations=2),
            self.checkpoint,
        )
        before = self.checkpoint.read_bytes()
        stderr = self.refused("--checkpoint", str(self.checkpoint))
        self.assertIn("does not belong to this plan and scan", stderr)
        self.assertEqual(self.checkpoint.read_bytes(), before)

    def test_options_that_make_no_sense_are_refused(self) -> None:
        self.assertIn(
            "requires --checkpoint", self.refused("--stop-after", "3")
        )
        self.assertIn(
            "must end in .sftckpt",
            self.refused("--checkpoint", str(self.root / "progress.sftvol")),
        )
        self.assertEqual(
            sorted(path.name for path in self.root.iterdir()), []
        )


if __name__ == "__main__":
    unittest.main()
