"""Fusion checkpoints: saved part way, continued to the same volume.

The claim worth testing is the last one. A fusion that is stopped, written
to disk, read back in what might as well be another process and carried on
must produce the ``.sftvol`` an uninterrupted fusion produces, byte for
byte. Everything else here is about refusing a file that could not have
come from a real fusion, changed one field or one byte at a time.
"""

from __future__ import annotations

import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from spatialforge import (
    allocate_empty_tsdf_blocks,
    load_tsdf_block_plan,
    write_tsdf_block_volume,
)
from spatialforge.errors import TsdfError
from spatialforge.session_loader import load_scan_session
from spatialforge.tsdf_block_plan import plan_tsdf_blocks
from spatialforge.tsdf_fusion_checkpoint import (
    TSDF_FUSION_CHECKPOINT_MAGIC,
    TsdfFusionCheckpoint,
    load_tsdf_fusion_checkpoint,
    restore_tsdf_fusion_checkpoint,
    write_tsdf_fusion_checkpoint,
)
from spatialforge.tsdf_stream_fusion import (
    advance_tsdf_plan_streaming,
    finish_tsdf_plan_streaming,
    fuse_tsdf_plan_streaming,
)

from tests.heavy_fixtures import shared_room_case
from tests.test_tsdf_stream_fusion import load_case, storage_bytes

TEST_ROOT = Path(__file__).resolve().parent
PREAMBLE = len(TSDF_FUSION_CHECKPOINT_MAGIC) + 8


def split(encoded: bytes) -> tuple[dict, bytes, bytes]:
    """A checkpoint's header, sums and weights, without the loader."""

    (length,) = struct.unpack_from("<Q", encoded, 8)
    header = json.loads(encoded[PREAMBLE:PREAMBLE + length])
    payload = encoded[PREAMBLE + length:]
    sum_bytes = header["payload"]["block_count"] * 512 * 8
    return header, payload[:sum_bytes], payload[sum_bytes:]


def assemble(
    header: dict,
    sums: bytes,
    weights: bytes,
    *,
    redigest: bool = True,
) -> bytes:
    """Put a checkpoint back together, by default with honest digests."""

    if redigest:
        header["payload"]["tsdf_sums_sha256"] = hashlib.sha256(
            sums
        ).hexdigest()
        header["payload"]["weights_sha256"] = hashlib.sha256(
            weights
        ).hexdigest()
    encoded = (
        json.dumps(header, indent=2, sort_keys=True) + "\n"
    ).encode("ascii")
    return b"".join(
        (
            TSDF_FUSION_CHECKPOINT_MAGIC,
            struct.pack("<Q", len(encoded)),
            encoded,
            sums,
            weights,
        )
    )


class ContinuationTests(unittest.TestCase):
    def test_a_fusion_continued_from_disk_writes_the_same_volume(
        self,
    ) -> None:
        room = shared_room_case()
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            root = Path(temporary)
            whole = allocate_empty_tsdf_blocks(room.plan, room.session)
            receipt = fuse_tsdf_plan_streaming(whole, room.session)
            write_tsdf_block_volume(whole, receipt, root / "whole.sftvol")

            first = allocate_empty_tsdf_blocks(room.plan, room.session)
            progress = advance_tsdf_plan_streaming(
                first, room.session, observations=7
            )
            report = write_tsdf_fusion_checkpoint(
                first, progress, root / "seven.sftckpt"
            )
            saved = storage_bytes(first)
            del first, progress

            # Nothing below uses anything from above but the file.
            checkpoint = load_tsdf_fusion_checkpoint(root / "seven.sftckpt")
            second = allocate_empty_tsdf_blocks(room.plan, room.session)
            restored = restore_tsdf_fusion_checkpoint(checkpoint, second)
            self.assertEqual(storage_bytes(second), saved)
            advanced = advance_tsdf_plan_streaming(
                second, room.session, restored, observations=6
            )
            # And once more through a file, replacing the first.
            write_tsdf_fusion_checkpoint(
                second,
                advanced,
                root / "seven.sftckpt",
                replace_existing=True,
            )
            third = allocate_empty_tsdf_blocks(room.plan, room.session)
            again = load_tsdf_fusion_checkpoint(root / "seven.sftckpt")
            finished = advance_tsdf_plan_streaming(
                third,
                room.session,
                restore_tsdf_fusion_checkpoint(again, third),
            )
            resumed_receipt = finish_tsdf_plan_streaming(
                third, room.session, finished
            )
            write_tsdf_block_volume(
                third, resumed_receipt, root / "resumed.sftvol"
            )
            whole_volume = (root / "whole.sftvol").read_bytes()
            resumed_volume = (root / "resumed.sftvol").read_bytes()
            leftovers = sorted(path.name for path in root.iterdir())

        self.assertEqual(checkpoint.progress.processed_observations, 7)
        self.assertEqual(again.progress.processed_observations, 13)
        self.assertEqual(report.processed_observations, 7)
        self.assertEqual(report.selected_observations, 20)
        self.assertEqual(
            report.output_digest_sha256, checkpoint.artifact_digest_sha256
        )
        self.assertEqual(resumed_receipt, receipt)
        self.assertEqual(resumed_volume, whole_volume)
        # One checkpoint, replaced in place, and no temporary file left.
        self.assertEqual(
            leftovers, ["resumed.sftvol", "seven.sftckpt", "whole.sftvol"]
        )

    def test_a_checkpoint_at_the_very_end_only_needs_finishing(self) -> None:
        room = shared_room_case()
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            root = Path(temporary)
            storage = allocate_empty_tsdf_blocks(room.plan, room.session)
            progress = advance_tsdf_plan_streaming(storage, room.session)
            expected = finish_tsdf_plan_streaming(
                storage, room.session, progress
            )
            write_tsdf_fusion_checkpoint(
                storage, progress, root / "end.sftckpt"
            )
            fresh = allocate_empty_tsdf_blocks(room.plan, room.session)
            restored = restore_tsdf_fusion_checkpoint(
                load_tsdf_fusion_checkpoint(root / "end.sftckpt"), fresh
            )
            receipt = finish_tsdf_plan_streaming(
                fresh, room.session, restored
            )
        self.assertTrue(restored.is_complete)
        self.assertEqual(receipt, expected)

    def test_the_same_state_is_the_same_file(self) -> None:
        room = shared_room_case()
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            root = Path(temporary)
            digests = []
            for name in ("a", "b"):
                storage = allocate_empty_tsdf_blocks(room.plan, room.session)
                progress = advance_tsdf_plan_streaming(
                    storage, room.session, observations=3
                )
                digests.append(
                    write_tsdf_fusion_checkpoint(
                        storage, progress, root / f"{name}.sftckpt"
                    ).output_digest_sha256
                )
            first = (root / "a.sftckpt").read_bytes()
            second = (root / "b.sftckpt").read_bytes()
        self.assertEqual(first, second)
        self.assertEqual(digests[0], hashlib.sha256(first).hexdigest())
        header, sums, weights = split(first)
        # No time, no path, no host: nothing that is not the fusion.
        self.assertEqual(
            sorted(header),
            [
                "payload",
                "progress",
                "replay_digest_sha256",
                "schema",
                "schema_version",
                "session_id",
                "source_plan_digest_sha256",
            ],
        )
        self.assertEqual(header["progress"]["processed_observations"], 3)
        self.assertEqual(len(sums), 351 * 512 * 8)
        self.assertEqual(len(weights), 351 * 512 * 4)


class SmallCase(unittest.TestCase):
    """The two-frame fixture, stopped after its first frame."""

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory(dir=TEST_ROOT)
        self.addCleanup(self._directory.cleanup)
        self.root = Path(self._directory.name)
        self.plan, self.session = load_case(self.root)
        self.storage = allocate_empty_tsdf_blocks(self.plan, self.session)
        self.progress = advance_tsdf_plan_streaming(
            self.storage, self.session, observations=1
        )
        self.path = self.root / "one.sftckpt"
        write_tsdf_fusion_checkpoint(self.storage, self.progress, self.path)
        self.encoded = self.path.read_bytes()

    def refused(self, encoded: bytes, name: str = "tampered") -> str:
        path = self.root / f"{name}.sftckpt"
        path.write_bytes(encoded)
        with self.assertRaises(TsdfError) as caught:
            load_tsdf_fusion_checkpoint(path)
        return str(caught.exception)


class LoaderTests(SmallCase):
    def test_a_genuine_checkpoint_loads_as_what_was_saved(self) -> None:
        checkpoint = load_tsdf_fusion_checkpoint(self.path)
        self.assertIsInstance(checkpoint, TsdfFusionCheckpoint)
        self.assertEqual(checkpoint.progress, self.progress)
        self.assertEqual(
            (checkpoint.tsdf_sums.tobytes(), checkpoint.weights.tobytes()),
            storage_bytes(self.storage),
        )
        self.assertEqual(
            checkpoint.artifact_digest_sha256,
            hashlib.sha256(self.encoded).hexdigest(),
        )
        self.assertFalse(checkpoint.tsdf_sums.flags.writeable)
        self.assertFalse(checkpoint.weights.flags.writeable)
        # Taking the file apart and putting it back changes nothing, so a
        # refusal below is about the one thing that test changed.
        self.assertEqual(assemble(*split(self.encoded)), self.encoded)

    def test_a_damaged_file_is_refused(self) -> None:
        header, sums, weights = split(self.encoded)
        flipped_sums = bytearray(sums)
        flipped_sums[100] ^= 1
        flipped_weights = bytearray(weights)
        flipped_weights[40] ^= 1
        cases = (
            ("empty", b"", "signature"),
            ("another-magic", b"SFTVOL01" + self.encoded[8:], "signature"),
            ("cut-in-the-header", self.encoded[:40], "truncated"),
            ("cut-in-the-payload", self.encoded[:-8], "payload is"),
            ("with-a-tail", self.encoded + b"\0" * 8, "payload is"),
            (
                "no-header",
                self.encoded[:8] + struct.pack("<Q", 1) + self.encoded[16:],
                "header length",
            ),
            (
                "a-sum-bit",
                assemble(header, bytes(flipped_sums), weights, redigest=False),
                "tsdf_sums do not match",
            ),
            (
                "a-weight-bit",
                assemble(
                    header, sums, bytes(flipped_weights), redigest=False
                ),
                "weights do not match",
            ),
        )
        for name, encoded, message in cases:
            with self.subTest(case=name):
                self.assertIn(message, self.refused(encoded, name))

    def test_a_header_that_is_not_the_format_is_refused(self) -> None:
        _, sums, weights = split(self.encoded)

        def changed(edit) -> bytes:
            header, _, _ = split(self.encoded)
            edit(header)
            return assemble(header, sums, weights)

        cases = (
            (
                "an unknown field",
                lambda h: h.__setitem__("written_at", "noon"),
                "unknown field 'written_at'",
            ),
            (
                "a missing field",
                lambda h: h.pop("session_id"),
                "session_id: missing",
            ),
            (
                "another schema",
                lambda h: h.__setitem__(
                    "schema", "spatialforge.tsdf-block-volume"
                ),
                "checkpoint.schema",
            ),
            (
                "a version nobody wrote",
                lambda h: h.__setitem__("schema_version", "0.2.0"),
                "schema_version",
            ),
            (
                "a plan digest that is not one",
                lambda h: h.__setitem__("source_plan_digest_sha256", "abc"),
                "source_plan_digest_sha256",
            ),
            (
                "a count that is a float",
                lambda h: h["progress"].__setitem__(
                    "processed_observations", 1.0
                ),
                "processed_observations",
            ),
            (
                "a status nobody defined",
                lambda h: h["progress"]["status_counts"].__setitem__(
                    "vanished", 5
                ),
                "unknown status 'vanished'",
            ),
            (
                "a status with nothing in it",
                lambda h: h["progress"]["status_counts"].__setitem__(
                    "missing-pose", 0
                ),
                "status_counts.missing-pose",
            ),
            (
                "another layout",
                lambda h: h["payload"]["layout"].reverse(),
                "payload.layout",
            ),
            (
                "big-endian",
                lambda h: h["payload"].__setitem__("byte_order", "big"),
                "byte_order",
            ),
        )
        for name, edit, message in cases:
            with self.subTest(case=name):
                self.assertIn(message, self.refused(changed(edit)))

        # The same contents written another way are a different file, and
        # one state must not have two.
        compact = (
            json.dumps(split(self.encoded)[0], sort_keys=True) + "\n"
        ).encode("ascii")
        self.assertIn(
            "not in canonical form",
            self.refused(
                b"".join(
                    (
                        TSDF_FUSION_CHECKPOINT_MAGIC,
                        struct.pack("<Q", len(compact)),
                        compact,
                        sums,
                        weights,
                    )
                )
            ),
        )

        header_text = json.dumps(split(self.encoded)[0], indent=2)
        doubled = header_text.replace(
            '"schema":', '"schema": "x",\n  "schema":', 1
        ).encode("ascii")
        self.assertIn(
            "duplicate JSON key 'schema'",
            self.refused(
                b"".join(
                    (
                        TSDF_FUSION_CHECKPOINT_MAGIC,
                        struct.pack("<Q", len(doubled)),
                        doubled,
                        sums,
                        weights,
                    )
                )
            ),
        )

    def test_progress_the_payload_denies_is_refused(self) -> None:
        header, sums, weights = split(self.encoded)
        weight_values = np.frombuffer(weights, dtype="<u4").copy()
        sum_values = np.frombuffer(sums, dtype="<f8").copy()
        seen = int(np.flatnonzero(weight_values)[0])
        unseen = int(np.flatnonzero(weight_values == 0)[0])

        def with_weights(edit) -> bytes:
            values = weight_values.copy()
            edit(values)
            return assemble(split(self.encoded)[0], sums, values.tobytes())

        def with_sums(edit) -> bytes:
            values = sum_values.copy()
            edit(values)
            return assemble(split(self.encoded)[0], values.tobytes(), weights)

        def with_progress(edit) -> bytes:
            fresh, _, _ = split(self.encoded)
            edit(fresh["progress"])
            return assemble(fresh, sums, weights)

        # One frame was fused, so no voxel can have been seen twice. Give
        # one a second view and take another's only view away entirely, so
        # that the total still matches and only the maximum is wrong.
        moved_weights = weight_values.copy()
        moved_sums = sum_values.copy()
        emptied = int(np.flatnonzero(weight_values)[-1])
        self.assertNotEqual(emptied, seen)
        moved_weights[seen] += 1
        moved_weights[emptied] = 0
        moved_sums[emptied] = 0.0
        twice_seen = assemble(
            split(self.encoded)[0],
            moved_sums.tobytes(),
            moved_weights.tobytes(),
        )

        cases = (
            (
                "a contribution the counts do not have",
                with_weights(lambda w: w.__setitem__(unseen, 1)),
                "does not match its payload",
            ),
            (
                "a voxel seen more often than frames were fused",
                twice_seen,
                "weight exceeds its fused observations",
            ),
            (
                "a sum no weight allows",
                with_sums(lambda s: s.__setitem__(seen, 5.0)),
                "weight envelope",
            ),
            (
                "a sum that is not a number",
                with_sums(lambda s: s.__setitem__(seen, np.nan)),
                "non-finite",
            ),
            (
                "an unseen voxel holding negative zero",
                with_sums(lambda s: s.__setitem__(unseen, -0.0)),
                "not canonical zero",
            ),
            (
                "a second frame the counts do not cover",
                with_progress(
                    lambda p: p.update(
                        processed_observations=2, fused_observations=2
                    )
                ),
                "do not cover every voxel-observation",
            ),
            (
                "more processed than selected",
                with_progress(
                    lambda p: p.update(processed_observations=3)
                ),
                "checkpoint.progress",
            ),
            (
                "a selection its stride denies",
                with_progress(lambda p: p.update(selected_observations=5)),
                "does not match its stride",
            ),
        )
        for name, encoded, message in cases:
            with self.subTest(case=name):
                self.assertIn(message, self.refused(encoded))

    def test_only_a_checkpoint_file_is_opened(self) -> None:
        (self.root / "volume.sftvol").write_bytes(self.encoded)
        for path, message in (
            (self.root / "volume.sftvol", "must end in .sftckpt"),
            (self.root / "absent.sftckpt", "does not exist"),
            (self.root, "must end in .sftckpt"),
        ):
            with self.subTest(path=path.name):
                with self.assertRaises(TsdfError) as caught:
                    load_tsdf_fusion_checkpoint(path)
                self.assertIn(message, str(caught.exception))


class WriterTests(SmallCase):
    def test_an_existing_file_is_kept_unless_replacement_is_asked_for(
        self,
    ) -> None:
        occupied = self.root / "occupied.sftckpt"
        occupied.write_bytes(b"something else")
        with self.assertRaises(TsdfError) as caught:
            write_tsdf_fusion_checkpoint(self.storage, self.progress, occupied)
        self.assertIn("already exists", str(caught.exception))
        self.assertEqual(occupied.read_bytes(), b"something else")

        write_tsdf_fusion_checkpoint(
            self.storage, self.progress, occupied, replace_existing=True
        )
        self.assertEqual(occupied.read_bytes(), self.encoded)
        # A directory is never something to replace.
        folder = self.root / "folder.sftckpt"
        folder.mkdir()
        with self.assertRaises(TsdfError):
            write_tsdf_fusion_checkpoint(
                self.storage, self.progress, folder, replace_existing=True
            )
        self.assertTrue(folder.is_dir())

    def test_a_replacement_that_fails_leaves_the_old_checkpoint(self) -> None:
        advanced = advance_tsdf_plan_streaming(
            self.storage, self.session, self.progress
        )
        with patch(
            "spatialforge.tsdf_fusion_checkpoint.os.replace",
            side_effect=OSError("simulated full disk"),
        ):
            with self.assertRaises(TsdfError) as caught:
                write_tsdf_fusion_checkpoint(
                    self.storage, advanced, self.path, replace_existing=True
                )
        self.assertIn("simulated full disk", str(caught.exception))
        self.assertEqual(self.path.read_bytes(), self.encoded)
        self.assertEqual(
            sorted(path.name for path in self.root.iterdir()),
            ["one.sftckpt", "stride1.sftplan"],
        )

    def test_storage_the_progress_does_not_describe_is_not_written(
        self,
    ) -> None:
        self.storage.weights[0, 0, 0, 0] += 1
        target = self.root / "wrong.sftckpt"
        with self.assertRaises(TsdfError) as caught:
            write_tsdf_fusion_checkpoint(self.storage, self.progress, target)
        self.assertIn("does not hold the bytes", str(caught.exception))
        self.assertFalse(target.exists())

    def test_invalid_arguments_are_refused(self) -> None:
        for name, call, message in (
            (
                "another suffix",
                lambda: write_tsdf_fusion_checkpoint(
                    self.storage, self.progress, self.root / "x.sftvol"
                ),
                "must end in .sftckpt",
            ),
            (
                "no storage",
                lambda: write_tsdf_fusion_checkpoint(
                    None, self.progress, self.root / "x.sftckpt"  # type: ignore
                ),
                "TsdfBlockStorage",
            ),
            (
                "no progress",
                lambda: write_tsdf_fusion_checkpoint(
                    self.storage, None, self.root / "x.sftckpt"  # type: ignore
                ),
                "TsdfStreamFusionProgress",
            ),
        ):
            with self.subTest(case=name):
                with self.assertRaises(TsdfError) as caught:
                    call()
                self.assertIn(message, str(caught.exception))
        self.assertFalse((self.root / "x.sftckpt").exists())


class RestoreTests(SmallCase):
    def test_restore_needs_empty_storage_of_the_same_plan(self) -> None:
        checkpoint = load_tsdf_fusion_checkpoint(self.path)

        used = allocate_empty_tsdf_blocks(self.plan, self.session)
        used.weights[0, 0, 0, 0] = 1
        before = storage_bytes(used)
        with self.assertRaises(TsdfError) as caught:
            restore_tsdf_fusion_checkpoint(checkpoint, used)
        self.assertIn("canonical empty", str(caught.exception))
        self.assertEqual(storage_bytes(used), before)

        other_path = self.root / "other.sftplan"
        plan_tsdf_blocks(
            load_scan_session(self.session.root),
            other_path,
            frame_stride=2,
            voxel_size_m=self.plan.voxel_size_m,
            truncation_m=self.plan.truncation_m,
        )
        other = allocate_empty_tsdf_blocks(
            load_tsdf_block_plan(other_path), self.session
        )
        before = storage_bytes(other)
        with self.assertRaises(TsdfError) as caught:
            restore_tsdf_fusion_checkpoint(checkpoint, other)
        self.assertIn("does not belong to this plan", str(caught.exception))
        self.assertEqual(storage_bytes(other), before)

        for name, call in (
            (
                "no checkpoint",
                lambda: restore_tsdf_fusion_checkpoint(
                    None, used  # type: ignore
                ),
            ),
            (
                "no storage",
                lambda: restore_tsdf_fusion_checkpoint(
                    checkpoint, None  # type: ignore
                ),
            ),
        ):
            with self.subTest(case=name):
                with self.assertRaises(TsdfError):
                    call()


if __name__ == "__main__":
    unittest.main()
