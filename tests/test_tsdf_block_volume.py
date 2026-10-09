"""The persisted sparse volume: exact round trips and refused forgeries.

A ``.sftvol`` exists so that a fused volume can leave the process that made
it. Two things have to hold for that to be worth anything. Loading must give
back the fused accumulators bit for bit, and a file whose header and payload
disagree -- in any way the loader can check -- must be refused rather than
believed.
"""

from __future__ import annotations

import hashlib
import io
import json
import struct
import tempfile
import tracemalloc
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

from spatialforge import (
    TsdfBlockVolume,
    allocate_empty_tsdf_blocks,
    fuse_tsdf_plan_streaming,
    load_tsdf_block_plan,
    load_tsdf_block_volume,
    write_tsdf_block_volume,
)
from spatialforge.cli import main
from spatialforge.errors import TsdfError
from spatialforge.session_loader import load_scan_session
from spatialforge.tsdf_block_plan import plan_tsdf_blocks
from spatialforge.tsdf_block_volume import (
    MAX_TSDF_BLOCK_VOLUME_HEADER_BYTES,
    TSDF_BLOCK_VOLUME_MAGIC,
    _encode_header,
)

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
# Pinned literally on purpose. The suite runs on Linux, macOS and Windows and
# on more than one interpreter; these only hold if the same scan fuses to the
# same bytes on all of them.
FIXTURE_SUMS_SHA256 = (
    "b70a4f65097cf2f0ed786051080272bbe4dabf4a9332f7ee3a20df56bb623908"
)
FIXTURE_WEIGHTS_SHA256 = (
    "cefdd8c161b8f528192cae8f6a1c031fba04d79830fc32be23bfa6e3948553b3"
)
FIXTURE_VOLUME_SHA256 = (
    "29f9f427b43c2a9e8ec416dad9a3ba1572044c716407ef2b30989daed9c2adc3"
)
FIXTURE_VOLUME_BYTES = 50_830


def fuse_fixture(parent: Path):
    plan_path = parent / "fixture.sftplan"
    if not plan_path.exists():
        plan_tsdf_blocks(
            load_scan_session(FIXTURE),
            plan_path,
            **PLAN_ARGUMENTS,
        )
    plan = load_tsdf_block_plan(plan_path)
    session = load_scan_session(FIXTURE)
    storage = allocate_empty_tsdf_blocks(plan, session)
    receipt = fuse_tsdf_plan_streaming(storage, session)
    return plan_path, storage, receipt


def write_fixture_volume(parent: Path, name: str = "fixture.sftvol"):
    _, storage, receipt = fuse_fixture(parent)
    output = parent / name
    report = write_tsdf_block_volume(storage, receipt, output)
    return output, storage, receipt, report


def split(encoded: bytes):
    """Split a volume file into header document and payload arrays."""

    magic_length = len(TSDF_BLOCK_VOLUME_MAGIC)
    (header_length,) = struct.unpack_from("<Q", encoded, magic_length)
    start = magic_length + 8
    header = json.loads(encoded[start:start + header_length])
    payload = encoded[start + header_length:]
    blocks = header["payload"]["block_count"]
    index_bytes = blocks * 12
    sum_bytes = blocks * 512 * 8
    indices = np.frombuffer(payload[:index_bytes], dtype="<i4").copy()
    sums = np.frombuffer(
        payload[index_bytes:index_bytes + sum_bytes],
        dtype="<f8",
    ).copy()
    weights = np.frombuffer(
        payload[index_bytes + sum_bytes:],
        dtype="<u4",
    ).copy()
    return header, indices, sums, weights


def assemble(header, indices, sums, weights, *, redigest: bool = True):
    """Re-encode a volume, by default with digests that match the payload.

    Recomputing the digests is what lets a test reach the semantic checks:
    a forgery with stale digests is caught earlier, by the digest check.
    """

    index_payload = indices.astype("<i4").tobytes()
    sum_payload = sums.astype("<f8").tobytes()
    weight_payload = weights.astype("<u4").tobytes()
    if redigest:
        header["payload"]["block_indices_sha256"] = hashlib.sha256(
            index_payload
        ).hexdigest()
        header["payload"]["tsdf_sums_sha256"] = hashlib.sha256(
            sum_payload
        ).hexdigest()
        header["payload"]["weights_sha256"] = hashlib.sha256(
            weight_payload
        ).hexdigest()
    encoded_header = _encode_header(header)
    return b"".join(
        (
            TSDF_BLOCK_VOLUME_MAGIC,
            struct.pack("<Q", len(encoded_header)),
            encoded_header,
            index_payload,
            sum_payload,
            weight_payload,
        )
    )


class BlockVolumeRoundTripTests(unittest.TestCase):
    def test_loading_returns_the_fused_accumulators_bit_for_bit(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            output, storage, receipt, report = write_fixture_volume(
                Path(temporary_dir)
            )
            volume = load_tsdf_block_volume(output)
            file_digest = hashlib.sha256(output.read_bytes()).hexdigest()

        self.assertIsInstance(volume, TsdfBlockVolume)
        self.assertEqual(
            volume.tsdf_sums.tobytes(),
            storage.tsdf_sums.tobytes(),
        )
        self.assertEqual(volume.weights.tobytes(), storage.weights.tobytes())
        self.assertEqual(volume.block_indices, storage.block_indices)
        self.assertEqual(volume.artifact_digest_sha256, file_digest)
        self.assertEqual(report.output_digest_sha256, file_digest)
        self.assertEqual(volume.source_plan_digest_sha256, PLAN_SHA256)
        self.assertEqual(volume.replay_digest_sha256, REPLAY_SHA256)
        self.assertEqual(volume.voxel_size_m, 0.125)
        self.assertEqual(volume.truncation_m, 0.5)
        self.assertEqual(volume.block_count, 8)
        self.assertEqual(volume.voxel_slots, 4096)
        self.assertEqual(volume.fused_observations, 2)
        self.assertEqual(volume.contributions_evaluated, 8192)
        self.assertEqual(
            volume.contributions_applied,
            receipt.applied_count,
        )
        self.assertEqual(volume.contributions_applied, 1168)
        self.assertEqual(volume.observed_voxel_count, 584)
        self.assertEqual(volume.unknown_voxel_count, 3512)
        self.assertEqual(volume.maximum_weight, 2)

    def test_loaded_arrays_cannot_be_modified(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            output, _, _, _ = write_fixture_volume(Path(temporary_dir))
            volume = load_tsdf_block_volume(output)

        for array in (volume.tsdf_sums, volume.weights):
            self.assertFalse(array.flags.writeable)
            with self.assertRaises(ValueError):
                array[0, 0, 0, 0] = 1

    def test_the_same_scan_always_writes_the_same_bytes(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            temporary_root = Path(temporary_dir)
            first, _, receipt, report = write_fixture_volume(
                temporary_root,
                "first.sftvol",
            )
            second, _, _, _ = write_fixture_volume(
                temporary_root,
                "second.sftvol",
            )
            first_bytes = first.read_bytes()
            second_bytes = second.read_bytes()

        self.assertEqual(first_bytes, second_bytes)
        self.assertEqual(receipt.tsdf_sums_sha256, FIXTURE_SUMS_SHA256)
        self.assertEqual(receipt.weights_sha256, FIXTURE_WEIGHTS_SHA256)
        self.assertEqual(len(first_bytes), FIXTURE_VOLUME_BYTES)
        self.assertEqual(report.output_bytes, FIXTURE_VOLUME_BYTES)
        self.assertEqual(report.output_digest_sha256, FIXTURE_VOLUME_SHA256)

    def test_normalized_tsdf_divides_only_where_observed(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            output, storage, _, _ = write_fixture_volume(Path(temporary_dir))
            volume = load_tsdf_block_volume(output)
            normalized = volume.normalized_tsdf()

        observed = storage.weights > 0
        self.assertEqual(int(observed.sum()), 584)
        self.assertTrue(bool(np.all(np.isnan(normalized[~observed]))))
        self.assertEqual(
            normalized[observed].tobytes(),
            (storage.tsdf_sums[observed] / storage.weights[observed])
            .tobytes(),
        )
        self.assertTrue(bool(np.all(np.abs(normalized[observed]) <= 1.0)))

    def test_room_scan_round_trips(self) -> None:
        case = shared_room_case()
        storage = allocate_empty_tsdf_blocks(case.plan, case.session)
        receipt = fuse_tsdf_plan_streaming(storage, case.session)
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            output = Path(temporary_dir) / "room.sftvol"
            write_tsdf_block_volume(storage, receipt, output)
            volume = load_tsdf_block_volume(output)

        self.assertEqual(
            volume.tsdf_sums.tobytes(),
            storage.tsdf_sums.tobytes(),
        )
        self.assertEqual(volume.weights.tobytes(), storage.weights.tobytes())
        self.assertEqual(volume.block_count, 351)
        self.assertEqual(volume.contributions_applied, 1_127_112)
        self.assertEqual(volume.observed_voxel_count, 81_292)

    def test_a_volume_is_written_and_loaded_without_a_copy_of_it(
        self,
    ) -> None:
        # A volume is as large as anything this project holds, and the
        # ceiling on its size is set by how much room handling one takes.
        # Writing must take none beyond the storage already there, and
        # loading the file's own bytes and nothing like twice that.
        case = shared_room_case()
        storage = allocate_empty_tsdf_blocks(case.plan, case.session)
        receipt = fuse_tsdf_plan_streaming(storage, case.session)
        payload = storage.tsdf_sums.nbytes + storage.weights.nbytes
        self.assertEqual(payload, 351 * 6144)
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            output = Path(temporary_dir) / "room.sftvol"
            tracemalloc.start()
            try:
                tracemalloc.reset_peak()
                before, _ = tracemalloc.get_traced_memory()
                write_tsdf_block_volume(storage, receipt, output)
                _, peak = tracemalloc.get_traced_memory()
                written = peak - before

                tracemalloc.reset_peak()
                before, _ = tracemalloc.get_traced_memory()
                # The payload is verified a run of blocks at a time. This
                # room is smaller than one run, so the run is made short
                # here to check what a large volume would be given.
                with patch(
                    "spatialforge.tsdf_block_volume."
                    "_VALIDATION_CHUNK_BLOCKS",
                    16,
                ):
                    volume = load_tsdf_block_volume(output)
                _, peak = tracemalloc.get_traced_memory()
                loaded = peak - before
            finally:
                tracemalloc.stop()
            size = output.stat().st_size

        self.assertGreater(size, payload)
        # Joining the pieces into one bytes object took three times the
        # payload. Written from where they are, the accumulators cost
        # nothing; what is left is the block indices and the header.
        self.assertLess(written, payload // 10)
        # The file is read once. Slicing it into its arrays, as bytes,
        # used to be a second copy.
        self.assertGreater(loaded, size)
        self.assertLess(loaded, size + payload // 4)
        self.assertEqual(
            volume.tsdf_sums.tobytes(), storage.tsdf_sums.tobytes()
        )


class BlockVolumeWriterGuardTests(unittest.TestCase):
    def test_storage_changed_after_fusion_is_not_persisted(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            temporary_root = Path(temporary_dir)
            _, storage, receipt = fuse_fixture(temporary_root)
            storage.weights[0, 0, 0, 0] += 1
            output = temporary_root / "changed.sftvol"
            with self.assertRaises(TsdfError) as caught:
                write_tsdf_block_volume(storage, receipt, output)
            self.assertFalse(output.exists())

        self.assertIn("changed after it was fused", str(caught.exception))

    def test_a_receipt_for_another_plan_is_refused(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            temporary_root = Path(temporary_dir)
            _, storage, _ = fuse_fixture(temporary_root)
            other_path = temporary_root / "other.sftplan"
            plan_tsdf_blocks(
                load_scan_session(FIXTURE),
                other_path,
                voxel_size_m=0.25,
                truncation_m=0.5,
            )
            other_plan = load_tsdf_block_plan(other_path)
            session = load_scan_session(FIXTURE)
            other_receipt = fuse_tsdf_plan_streaming(
                allocate_empty_tsdf_blocks(other_plan, session),
                session,
            )
            output = temporary_root / "mismatch.sftvol"
            with self.assertRaises(TsdfError) as caught:
                write_tsdf_block_volume(storage, other_receipt, output)
            self.assertFalse(output.exists())

        self.assertIn("does not describe this storage", str(caught.exception))

    def test_existing_output_and_wrong_suffix_are_refused(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            temporary_root = Path(temporary_dir)
            _, storage, receipt = fuse_fixture(temporary_root)
            existing = temporary_root / "existing.sftvol"
            existing.write_bytes(b"keep me")
            with self.assertRaises(TsdfError) as exists:
                write_tsdf_block_volume(storage, receipt, existing)
            with self.assertRaises(TsdfError) as suffix:
                write_tsdf_block_volume(
                    storage,
                    receipt,
                    temporary_root / "volume.bin",
                )
            leftovers = sorted(
                path.name
                for path in temporary_root.iterdir()
                if path.suffix != ".sftplan"
            )
            kept = existing.read_bytes()

        self.assertIn("already exists", str(exists.exception))
        self.assertIn(".sftvol", str(suffix.exception))
        self.assertEqual(kept, b"keep me")
        self.assertEqual(leftovers, ["existing.sftvol"])

    def test_invalid_arguments_are_refused(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            temporary_root = Path(temporary_dir)
            _, storage, receipt = fuse_fixture(temporary_root)
            output = temporary_root / "volume.sftvol"
            with self.assertRaises(TsdfError):
                write_tsdf_block_volume(None, receipt, output)  # type: ignore
            with self.assertRaises(TsdfError):
                write_tsdf_block_volume(storage, None, output)  # type: ignore
            self.assertFalse(output.exists())


class BlockVolumeForgeryTests(unittest.TestCase):
    """Every way the header and payload can disagree must be refused."""

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory(dir=TEST_ROOT)
        self.root = Path(self._directory.name)
        self.addCleanup(self._directory.cleanup)
        output, _, _, _ = write_fixture_volume(self.root)
        self.encoded = output.read_bytes()
        self.counter = 0

    def refuse(self, encoded: bytes) -> str:
        self.counter += 1
        path = self.root / f"forged-{self.counter}.sftvol"
        path.write_bytes(encoded)
        with self.assertRaises(TsdfError) as caught:
            load_tsdf_block_volume(path)
        return str(caught.exception)

    def test_the_untouched_file_loads(self) -> None:
        path = self.root / "intact.sftvol"
        path.write_bytes(assemble(*split(self.encoded)))
        self.assertEqual(path.read_bytes(), self.encoded)
        self.assertEqual(load_tsdf_block_volume(path).block_count, 8)

    def test_any_changed_payload_byte_is_detected(self) -> None:
        header, indices, sums, weights = split(self.encoded)
        cases = {}
        changed = sums.copy()
        changed[5] = 0.25
        cases["tsdf_sums"] = (indices, changed, weights)
        changed = weights.copy()
        changed[7] += 1
        cases["weights"] = (indices, sums, changed)
        changed = indices.copy()
        changed[0] += 1
        cases["block_indices"] = (changed, sums, weights)
        for name, arrays in cases.items():
            with self.subTest(array=name):
                message = self.refuse(
                    assemble(
                        json.loads(json.dumps(header)),
                        *arrays,
                        redigest=False,
                    )
                )
                self.assertIn(name, message)
                self.assertIn("recorded digest", message)

    def test_container_damage_is_refused(self) -> None:
        cases = {
            "signature": b"SFTVOL99" + self.encoded[8:],
            "empty": b"",
            "truncated inside its header": self.encoded[:40],
            "payload is": self.encoded[:-1],
            "payload is ": self.encoded + b"\x00",
            "header length": (
                self.encoded[:8]
                + struct.pack("<Q", MAX_TSDF_BLOCK_VOLUME_HEADER_BYTES + 1)
                + self.encoded[16:]
            ),
        }
        for expected, encoded in cases.items():
            with self.subTest(case=expected.strip()):
                message = self.refuse(encoded)
                self.assertIn(
                    "signature" if expected == "empty" else expected.strip(),
                    message,
                )

    def test_header_must_be_strict_and_canonical(self) -> None:
        header, indices, sums, weights = split(self.encoded)
        magic_length = len(TSDF_BLOCK_VOLUME_MAGIC)
        (header_length,) = struct.unpack_from(
            "<Q",
            self.encoded,
            magic_length,
        )
        payload = self.encoded[magic_length + 8 + header_length:]

        def with_header_bytes(header_bytes: bytes) -> bytes:
            return (
                TSDF_BLOCK_VOLUME_MAGIC
                + struct.pack("<Q", len(header_bytes))
                + header_bytes
                + payload
            )

        compact = json.dumps(header, sort_keys=True).encode("ascii")
        self.assertIn(
            "canonical form",
            self.refuse(with_header_bytes(compact)),
        )
        self.assertIn(
            "duplicate JSON key",
            self.refuse(
                with_header_bytes(b'{"schema": "a", "schema": "b"}\n')
            ),
        )
        self.assertIn(
            "invalid JSON",
            self.refuse(with_header_bytes(b'{"schema": NaN}\n')),
        )
        self.assertIn(
            "nested too deeply",
            self.refuse(with_header_bytes(b"[" * 64 + b"]" * 64)),
        )
        self.assertIn(
            "ASCII",
            self.refuse(with_header_bytes("{é}".encode("utf-8"))),
        )

        def mutated(change) -> bytes:
            document = json.loads(json.dumps(header))
            change(document)
            return assemble(document, indices, sums, weights)

        cases = {
            "volume.schema": lambda d: d.__setitem__("schema", "other"),
            "unknown field": lambda d: d.__setitem__("extra", 1),
            "volume.fusion.frame_stride: missing": (
                lambda d: d["fusion"].pop("frame_stride")
            ),
            "volume.session_id": lambda d: d.__setitem__("session_id", "A"),
            "volume.grid.voxel_size_m": (
                lambda d: d["grid"].__setitem__("voxel_size_m", 1)
            ),
            "volume.tsdf.truncation_m": (
                lambda d: d["tsdf"].__setitem__("truncation_m", 0.01)
            ),
            "volume.payload.byte_order": (
                lambda d: d["payload"].__setitem__("byte_order", "big")
            ),
            "volume.payload.layout": (
                lambda d: d["payload"].__setitem__("layout", [])
            ),
            "volume.fusion.selected_observations": (
                lambda d: d["fusion"].__setitem__("selected_observations", 5)
            ),
            "volume.fusion.contributions_evaluated": (
                lambda d: d["fusion"].__setitem__(
                    "contributions_evaluated",
                    1,
                )
            ),
            "volume.replay_digest_sha256": (
                lambda d: d.__setitem__("replay_digest_sha256", "abc")
            ),
        }
        for expected, change in cases.items():
            with self.subTest(case=expected):
                self.assertIn(expected, self.refuse(mutated(change)))

    def test_a_consistently_redigested_forgery_is_still_refused(self) -> None:
        """Matching digests are not enough; the payload is re-derived."""

        header, indices, sums, weights = split(self.encoded)

        def forge(change) -> str:
            document = json.loads(json.dumps(header))
            new_indices = indices.copy()
            new_sums = sums.copy()
            new_weights = weights.copy()
            change(document, new_indices, new_sums, new_weights)
            return self.refuse(
                assemble(document, new_indices, new_sums, new_weights)
            )

        observed = int(np.flatnonzero(weights)[0])
        unobserved = int(np.flatnonzero(weights == 0)[0])

        def extra_weight(document, i, s, w):
            w[unobserved] = 1

        def envelope(document, i, s, w):
            s[observed] = 5.0

        def non_finite(document, i, s, w):
            s[observed] = np.inf

        def negative_zero(document, i, s, w):
            s[unobserved] = -0.0

        def heavy(document, i, s, w):
            w[observed] = 9
            document["fusion"]["maximum_weight"] = 9
            document["fusion"]["contributions_applied"] = int(w.sum())

        def swapped_blocks(document, i, s, w):
            i[0:3], i[3:6] = i[3:6].copy(), i[0:3].copy()

        def duplicate_block(document, i, s, w):
            i[3:6] = i[0:3]

        cases = (
            ("fusion summary does not match", extra_weight),
            ("weight envelope", envelope),
            ("non-finite sum", non_finite),
            ("not canonical zero", negative_zero),
            ("exceeds its fused observations", heavy),
            ("canonical order", swapped_blocks),
            ("canonical order", duplicate_block),
        )
        for expected, change in cases:
            with self.subTest(case=change.__name__):
                self.assertIn(expected, forge(change))

    def test_wrong_suffix_and_missing_file_are_refused(self) -> None:
        with self.assertRaises(TsdfError) as suffix:
            load_tsdf_block_volume(self.root / "volume.bin")
        with self.assertRaises(TsdfError) as missing:
            load_tsdf_block_volume(self.root / "absent.sftvol")
        self.assertIn(".sftvol", str(suffix.exception))
        self.assertIn("does not exist", str(missing.exception))


class BlockVolumeCliTests(unittest.TestCase):
    def run_cli(self, arguments: list[str]) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main(arguments)
        return exit_code, stdout.getvalue(), stderr.getvalue()

    def test_cli_fuses_and_writes_the_fixture_volume(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            temporary_root = Path(temporary_dir)
            plan_path, _, _ = fuse_fixture(temporary_root)
            output = temporary_root / "fixture.sftvol"
            exit_code, stdout, stderr = self.run_cli(
                [
                    "reconstruct",
                    "tsdf-block-volume",
                    str(plan_path),
                    str(FIXTURE),
                    str(output),
                ]
            )
            written = sorted(path.name for path in temporary_root.iterdir())
            volume = load_tsdf_block_volume(output)

        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr, "")
        self.assertEqual(written, ["fixture.sftplan", "fixture.sftvol"])
        self.assertEqual(volume.artifact_digest_sha256, FIXTURE_VOLUME_SHA256)
        self.assertEqual(
            stdout,
            "TSDF BLOCK VOLUME scan-synthetic-0001\n"
            "artifact: valid\n"
            "session_replay: matched\n"
            "frames: total=2 selected=2 fused=2\n"
            "skipped: missing_depth=0 missing_pose=0\n"
            "grid: voxel_size_m=0.125000000 truncation_m=0.500000000 "
            "block_resolution=8\n"
            "blocks: active=8 voxel_slots=4096\n"
            "fusion_path: streaming-frame-major\n"
            "frames_retained_at_once: 1\n"
            "peak_retained_depth_bytes: 32\n"
            "contributions_evaluated: 8192\n"
            "contributions_applied: 1168\n"
            "contributions_skipped: 7024\n"
            "status_counts: contributes=1168 "
            "projection-outside-image=5440 behind-truncation=1584\n"
            "observed_voxels: 584\n"
            "unknown_voxels: 3512\n"
            "max_weight: 2\n"
            f"tsdf_sums_sha256: {FIXTURE_SUMS_SHA256}\n"
            f"weights_sha256: {FIXTURE_WEIGHTS_SHA256}\n"
            "storage_persisted: yes\n"
            "ledger_persisted: no\n"
            "free_space_coverage_planned: no\n"
            "plan_expanded: no\n"
            f"volume_bytes: {FIXTURE_VOLUME_BYTES}\n"
            f"output: {output.resolve()}\n"
            f"output_sha256: {FIXTURE_VOLUME_SHA256}\n"
            f"plan_sha256: {PLAN_SHA256}\n"
            f"replay_digest_sha256: {REPLAY_SHA256}\n",
        )

    def test_cli_refuses_an_existing_output_before_fusing(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            temporary_root = Path(temporary_dir)
            plan_path, _, _ = fuse_fixture(temporary_root)
            output = temporary_root / "taken.sftvol"
            output.write_bytes(b"keep me")
            exit_code, stdout, stderr = self.run_cli(
                [
                    "reconstruct",
                    "tsdf-block-volume",
                    str(plan_path),
                    str(FIXTURE),
                    str(output),
                ]
            )
            kept = output.read_bytes()

        self.assertEqual(exit_code, 2)
        self.assertEqual(stdout, "")
        self.assertEqual(kept, b"keep me")
        self.assertIn("TSDF BLOCK VOLUME FAILED", stderr)
        self.assertIn("output already exists", stderr)
        self.assertNotIn("Traceback", stderr)


if __name__ == "__main__":
    unittest.main()
