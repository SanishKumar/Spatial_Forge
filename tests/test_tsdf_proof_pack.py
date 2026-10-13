"""A packed proof, held to streams written here by hand.

Three things are claimed of packing.

What is unpacked is what was packed, whoever packed it. The same bytes are
put here into streams no compressor made (stored blocks set down from RFC
1950 and RFC 1951, their checksum added up byte by byte) and into streams
zlib makes at every level, in a small window, flushed part way and with
other strategies. Every one of them is read back as the same proof.

Unpacking is bounded before it starts. A length over what a kind may be
is refused with the inflater taken away, and a stream of sixteen
mebibytes that declares a thousand bytes is refused with the bytes
inflated counted.

And packing changes nothing a proof says. A packed file is read as the
proof it carries, with the digest the proof's own tests pin; it checks the
same; the commands print the same lines of it. No bit of a packed proof
can be changed into another proof. Some bits can be changed and leave the
same proof, which is why no proof is named by its packed bytes.

Nothing here is a literal of packed bytes or of their digest. Two zlib
libraries need not pack the same bytes the same way.
"""

from __future__ import annotations

import hashlib
import io
import shutil
import struct
import tempfile
import unittest
import zlib
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from spatialforge import (
    MAX_TSDF_BLOCK_PROOF_BYTES,
    MAX_TSDF_ROUTE_CERTIFICATE_BYTES,
    TSDF_PACKED_PROOF_MAGIC,
    TSDF_PROOF_PACK_LEVEL,
    TsdfBlockProofVerdict,
    TsdfMapAuditVerdict,
    TsdfRouteVerdict,
    audit_tsdf_map,
    certify_tsdf_route,
    check_tsdf_block_proof,
    check_tsdf_map_audit,
    check_tsdf_route_certificate,
    decode_tsdf_block_proof,
    decode_tsdf_map_audit,
    decode_tsdf_route_certificate,
    load_tsdf_block_proof,
    load_tsdf_block_volume,
    load_tsdf_map_audit,
    load_tsdf_route_certificate,
    pack_tsdf_proof_bytes,
    prove_tsdf_block,
    state_tsdf_map,
    tsdf_proof_bytes_are_packed,
    unpack_tsdf_proof_bytes,
    write_tsdf_block_proof,
    write_tsdf_block_volume,
    write_tsdf_map_audit,
    write_tsdf_map_statement,
    write_tsdf_route_certificate,
)
from spatialforge.cli import main
from spatialforge.errors import TsdfError
from spatialforge.tsdf_map_audit import MAX_TSDF_MAP_AUDIT_BYTES

from tests.heavy_fixtures import shared_room_case
from tests.test_tsdf_block_evidence import forbidden_calls, fused
from tests.test_tsdf_block_evidence import tree_snapshot
from tests.test_tsdf_block_proof import (
    FIXTURE_PROOF_BYTES,
    FIXTURE_PROOFS,
    fixture_volume,
)
from tests.test_tsdf_map_audit import (
    FIXTURE_AUDIT_BYTES,
    FIXTURE_AUDIT_SHA256,
)
from tests.test_tsdf_route_certificate import (
    FIXTURE_CLEAR,
    FIXTURE_CLEAR_BYTES,
    FIXTURE_CLEAR_SHA256,
)

TEST_ROOT = Path(__file__).resolve().parent
FIXTURE = TEST_ROOT / "fixtures" / "minimal.vgsession"
FIXTURE_STATEMENT_SHA256 = (
    "7b771f1ef5ff5249f48cf29affdbce16df2349f02b67cf7d9accf4884414e780"
)
FIXTURE_BLOCK, FIXTURE_PROOF_SHA256 = FIXTURE_PROOFS[3]
_INFLATER = "spatialforge.tsdf_proof_pack.zlib.decompressobj"


def adler32_by_hand(data: bytes) -> int:
    """RFC 1950's checksum, one byte at a time."""

    low, high = 1, 0
    for byte in data:
        low = (low + byte) % 65521
        high = (high + low) % 65521
    return (high << 16) | low


def stored_by_hand(data: bytes, *, spare_bits: int = 0) -> bytes:
    """A zlib stream of these bytes that no compressor made.

    Two header bytes, the bytes themselves in stored blocks of at most
    65,535, and their checksum. A stored block begins with three bits
    that say what it is and five that say nothing; ``spare_bits`` is what
    is put in those five.
    """

    parts = [b"\x78\x01"]
    for start in range(0, len(data), 65535):
        chunk = data[start:start + 65535]
        last = start + 65535 >= len(data)
        parts.append(bytes([int(last) | (spare_bits << 3)]))
        parts.append(struct.pack("<HH", len(chunk), len(chunk) ^ 0xFFFF))
        parts.append(chunk)
    parts.append(struct.pack(">I", adler32_by_hand(data)))
    return b"".join(parts)


def packed_by_hand(stream: bytes, declared: int) -> bytes:
    return b"SFTPACK1" + struct.pack("<Q", declared) + stream


def squeezed(data: bytes, *arguments, flush_at=None, flush=None) -> bytes:
    packer = zlib.compressobj(*arguments)
    if flush_at is None:
        return packer.compress(data) + packer.flush()
    return (
        packer.compress(data[:flush_at])
        + packer.flush(flush)
        + packer.compress(data[flush_at:])
        + packer.flush()
    )


def streams_of(data: bytes):
    """The same bytes as many different zlib streams."""

    yield "stored by hand", stored_by_hand(data)
    yield "stored by hand, spare bits set", stored_by_hand(
        data, spare_bits=31
    )
    for level in range(10):
        yield f"level {level}", zlib.compress(data, level)
    yield "a window of 512 bytes", squeezed(data, 6, zlib.DEFLATED, 9)
    yield "flushed part way", squeezed(
        data, 6, flush_at=len(data) // 3, flush=zlib.Z_FULL_FLUSH
    )
    yield "synchronised part way", squeezed(
        data, 6, flush_at=len(data) // 2, flush=zlib.Z_SYNC_FLUSH
    )
    yield "codes alone", squeezed(
        data, 6, zlib.DEFLATED, 15, 8, zlib.Z_HUFFMAN_ONLY
    )
    yield "filtered", squeezed(data, 9, zlib.DEFLATED, 15, 9, zlib.Z_FILTERED)


def setUpModule() -> None:
    global WORK, KINDS
    root = Path(tempfile.mkdtemp(dir=TEST_ROOT))
    unittest.addModuleCleanup(shutil.rmtree, root, True)

    volume_path, volume, session = fixture_volume(root, "fixture", FIXTURE)
    statement = state_tsdf_map(volume, session)
    statement_path = root / "fixture.sftcommit"
    write_tsdf_map_statement(statement, statement_path)

    room = shared_room_case()
    room_storage, room_receipt = fused(room.plan, room.session)
    room_volume_path = root / "room.sftvol"
    write_tsdf_block_volume(room_storage, room_receipt, room_volume_path)
    room_volume = load_tsdf_block_volume(room_volume_path)
    room_statement = state_tsdf_map(room_volume, room.session)

    WORK = SimpleNamespace(
        root=root,
        volume_path=volume_path,
        statement=statement,
        statement_path=statement_path,
        room_statement=room_statement,
        room_audit=audit_tsdf_map(
            room_statement,
            room_volume,
            room.session,
            seed="auditor-1",
            count=20,
        ),
    )
    KINDS = (
        SimpleNamespace(
            name="proof",
            noun="TSDF block proof",
            suffix=".sftproof",
            held=prove_tsdf_block(statement, volume, session, FIXTURE_BLOCK),
            write=write_tsdf_block_proof,
            load=load_tsdf_block_proof,
            decode=decode_tsdf_block_proof,
            check=lambda proof: check_tsdf_block_proof(statement, proof),
            good=TsdfBlockProofVerdict.VERIFIED,
            invalid=TsdfBlockProofVerdict.INVALID,
            most=MAX_TSDF_BLOCK_PROOF_BYTES,
            bytes=FIXTURE_PROOF_BYTES,
            sha256=FIXTURE_PROOF_SHA256,
        ),
        SimpleNamespace(
            name="certificate",
            noun="TSDF route certificate",
            suffix=".sftroute",
            held=certify_tsdf_route(
                statement, volume, session, FIXTURE_CLEAR
            ),
            write=write_tsdf_route_certificate,
            load=load_tsdf_route_certificate,
            decode=decode_tsdf_route_certificate,
            check=lambda certificate: check_tsdf_route_certificate(
                statement, certificate
            ),
            good=TsdfRouteVerdict.CLEAR,
            invalid=TsdfRouteVerdict.INVALID,
            most=MAX_TSDF_ROUTE_CERTIFICATE_BYTES,
            bytes=FIXTURE_CLEAR_BYTES,
            sha256=FIXTURE_CLEAR_SHA256,
        ),
        SimpleNamespace(
            name="audit",
            noun="TSDF map audit",
            suffix=".sftaudit",
            held=audit_tsdf_map(
                statement, volume, session, seed="auditor-1", count=3
            ),
            write=write_tsdf_map_audit,
            load=load_tsdf_map_audit,
            decode=decode_tsdf_map_audit,
            check=lambda audit: check_tsdf_map_audit(
                statement, audit, seed="auditor-1", count=3
            ),
            good=TsdfMapAuditVerdict.PASSED,
            invalid=TsdfMapAuditVerdict.INVALID,
            most=MAX_TSDF_MAP_AUDIT_BYTES,
            bytes=FIXTURE_AUDIT_BYTES,
            sha256=FIXTURE_AUDIT_SHA256,
        ),
    )


def payloads():
    for kind in KINDS:
        yield f"the {kind.name}", kind.held.encoded()
    # More than three stored blocks, and every value a byte can take.
    yield "every byte, many times", bytes(range(256)) * 800
    yield "one byte", b"\x00"
    yield "nothing but zeros", bytes(100_000)


class PackTests(unittest.TestCase):
    def test_what_is_packed_is_what_is_unpacked(self) -> None:
        self.assertEqual(TSDF_PACKED_PROOF_MAGIC, b"SFTPACK1")
        self.assertEqual(TSDF_PROOF_PACK_LEVEL, 6)
        for name, data in payloads():
            with self.subTest(name=name):
                packed = pack_tsdf_proof_bytes(data)
                self.assertIs(type(packed), bytes)
                self.assertEqual(packed[:8], b"SFTPACK1")
                self.assertEqual(
                    struct.unpack("<Q", packed[8:16]), (len(data),)
                )
                self.assertEqual(zlib.decompress(packed[16:]), data)
                self.assertEqual(
                    packed, packed_by_hand(zlib.compress(data, 6), len(data))
                )
                self.assertTrue(tsdf_proof_bytes_are_packed(packed))
                self.assertFalse(tsdf_proof_bytes_are_packed(data))
                for most in (len(data), len(data) + 1, 1 << 40):
                    self.assertEqual(
                        unpack_tsdf_proof_bytes(packed, most_bytes=most),
                        data,
                    )

    def test_any_stream_of_the_bytes_carries_them(self) -> None:
        for name, data in payloads():
            streams = dict(streams_of(data))
            for made, stream in streams.items():
                with self.subTest(name=name, made=made):
                    self.assertEqual(
                        unpack_tsdf_proof_bytes(
                            packed_by_hand(stream, len(data)),
                            most_bytes=len(data),
                        ),
                        data,
                    )
            # They are not one stream said many ways.
            self.assertGreater(len(set(streams.values())), 4)
        # The checksum added up here is the one zlib adds up.
        for _, data in payloads():
            self.assertEqual(adler32_by_hand(data), zlib.adler32(data))

    def test_a_length_over_the_most_is_refused_with_nothing_inflated(
        self,
    ) -> None:
        data = KINDS[0].held.encoded()
        stream = zlib.compress(data, 6)
        cases = (
            (len(data), len(data) - 1),
            (len(data), 1),
            (0, len(data)),
            (1 << 63, 1 << 40),
            ((1 << 64) - 1, 1 << 40),
        )
        for declared, most in cases:
            with self.subTest(declared=declared, most=most):
                with forbidden_calls((_INFLATER,)):
                    with self.assertRaisesRegex(
                        TsdfError,
                        f"^packed TSDF proof declares {declared} bytes; "
                        f"one of this kind is from 1 to {most}$",
                    ):
                        unpack_tsdf_proof_bytes(
                            packed_by_hand(stream, declared), most_bytes=most
                        )
        for most in (0, -1, True, 7009.0, "7009", None):
            with self.subTest(most=most):
                with self.assertRaisesRegex(
                    TsdfError, "the most a TSDF proof may be is a positive"
                ):
                    unpack_tsdf_proof_bytes(
                        pack_tsdf_proof_bytes(data), most_bytes=most
                    )

    def test_no_more_is_inflated_than_is_declared(self) -> None:
        inflater = zlib.decompressobj
        made = []

        class Counted:
            def __init__(self) -> None:
                self.inner = inflater()
                self.inflated = 0
                made.append(self)

            def decompress(self, data, most=0):
                given = self.inner.decompress(data, most)
                self.inflated += len(given)
                return given

            def __getattr__(self, name):
                return getattr(self.inner, name)

        huge = bytes(16 * 1024 * 1024)
        stream = zlib.compress(huge, 9)
        self.assertLess(len(stream), 64 * 1024)
        for declared in (1, 1000, len(huge) - 1):
            with self.subTest(declared=declared):
                made.clear()
                with patch(_INFLATER, Counted):
                    with self.assertRaisesRegex(
                        TsdfError,
                        "^packed TSDF proof inflates to more than the "
                        f"{declared} bytes it declares$",
                    ):
                        unpack_tsdf_proof_bytes(
                            packed_by_hand(stream, declared),
                            most_bytes=1 << 40,
                        )
                self.assertEqual(
                    [counted.inflated for counted in made], [declared + 1]
                )
        # Counted, the whole of it is what was packed.
        made.clear()
        with patch(_INFLATER, Counted):
            unpacked = unpack_tsdf_proof_bytes(
                packed_by_hand(stream, len(huge)), most_bytes=len(huge)
            )
        self.assertEqual(unpacked, huge)
        self.assertEqual([counted.inflated for counted in made], [len(huge)])

    def test_a_stream_has_to_end_where_its_length_says(self) -> None:
        data = KINDS[0].held.encoded()
        size = len(data)
        stream = zlib.compress(data, 6)
        cases = (
            (
                "declares a byte less",
                packed_by_hand(stream, size - 1),
                f"inflates to more than the {size - 1} bytes it declares",
            ),
            (
                "declares a byte more",
                packed_by_hand(stream, size + 1),
                f"inflates to {size} bytes and declares {size + 1}",
            ),
            (
                "a byte after its end",
                packed_by_hand(stream + b"\x00", size),
                "holds 1 bytes after its end",
            ),
            (
                "twice over",
                packed_by_hand(stream + stream, size),
                f"holds {len(stream)} bytes after its end",
            ),
            (
                "another packed proof after it",
                packed_by_hand(stream, size) + packed_by_hand(stream, size),
                f"holds {len(stream) + 16} bytes after its end",
            ),
            (
                "a wrong checksum",
                packed_by_hand(
                    stream[:-1] + bytes([stream[-1] ^ 1]), size
                ),
                "cannot be inflated: .*incorrect data check",
            ),
            (
                "no stream",
                packed_by_hand(b"", size),
                "is cut short",
            ),
            (
                "not a stream",
                packed_by_hand(bytes(40), size),
                "cannot be inflated",
            ),
            (
                "the bytes themselves",
                packed_by_hand(data, size),
                "cannot be inflated",
            ),
            (
                "deflate with no wrapper",
                packed_by_hand(squeezed(data, 6, zlib.DEFLATED, -15), size),
                "cannot be inflated",
            ),
            (
                "gzip",
                packed_by_hand(squeezed(data, 6, zlib.DEFLATED, 31), size),
                "cannot be inflated",
            ),
            (
                "a stream that wants a dictionary",
                packed_by_hand(
                    squeezed(
                        data, 6, zlib.DEFLATED, 15, 8,
                        zlib.Z_DEFAULT_STRATEGY, data[:512],
                    ),
                    size,
                ),
                "cannot be inflated",
            ),
            (
                "not packed",
                data,
                "has an unrecognized signature",
            ),
            (
                "another signature",
                b"SFTPACK2" + packed_by_hand(stream, size)[8:],
                "has an unrecognized signature",
            ),
        )
        for name, packed, message in cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(
                    TsdfError, f"^packed TSDF proof {message}"
                ):
                    unpack_tsdf_proof_bytes(packed, most_bytes=1 << 40)

        packed = pack_tsdf_proof_bytes(data)
        for length in range(len(packed)):
            if length < 8:
                message = "has an unrecognized signature"
            elif length < 16:
                message = "is cut short before its length"
            else:
                message = "is cut short"
            with self.subTest(length=length):
                with self.assertRaisesRegex(
                    TsdfError, f"^packed TSDF proof {message}$"
                ):
                    unpack_tsdf_proof_bytes(
                        packed[:length], most_bytes=1 << 40
                    )

    def test_only_a_proofs_bytes_are_packed_and_only_once(self) -> None:
        data = KINDS[0].held.encoded()
        for wrong in (b"", bytearray(data), memoryview(data), "text", None):
            with self.subTest(wrong=type(wrong).__name__):
                with self.assertRaisesRegex(
                    TsdfError, "a TSDF proof is packed from its bytes"
                ):
                    pack_tsdf_proof_bytes(wrong)
        with self.assertRaisesRegex(
            TsdfError, "a packed TSDF proof is not packed again"
        ):
            pack_tsdf_proof_bytes(pack_tsdf_proof_bytes(data))
        packed = pack_tsdf_proof_bytes(data)
        for wrong in (bytearray(packed), memoryview(packed), "SFTPACK1", 7):
            with self.subTest(wrong=type(wrong).__name__):
                self.assertFalse(tsdf_proof_bytes_are_packed(wrong))
                with self.assertRaisesRegex(
                    TsdfError, "has an unrecognized signature"
                ):
                    unpack_tsdf_proof_bytes(wrong, most_bytes=1 << 40)

    def test_every_bit_of_a_stream_written_by_hand(self) -> None:
        # In a stream of stored blocks every bit has a meaning that
        # RFC 1950 and RFC 1951 give it, so what each bit does when it
        # is changed is known before it is tried: the five spare bits
        # of the one block leave the same proof, and every other bit
        # leaves nothing that can be read.
        kind = KINDS[0]
        data = kind.held.encoded()
        packed = packed_by_hand(stored_by_hand(data), len(data))
        self.assertEqual(len(packed), 16 + 2 + 5 + len(data) + 4)
        same = []
        refused = 0
        for position in range(len(packed)):
            for bit in range(8):
                changed = bytearray(packed)
                changed[position] ^= 1 << bit
                changed = bytes(changed)
                try:
                    if tsdf_proof_bytes_are_packed(changed):
                        unpacked = unpack_tsdf_proof_bytes(
                            changed, most_bytes=kind.most
                        )
                    else:
                        # Not packed, so read as a proof's own bytes.
                        self.assertLess(position, 8)
                        kind.decode(changed)
                        self.fail("a packed proof was read as a proof")
                except TsdfError:
                    refused += 1
                    continue
                self.assertEqual(unpacked, data)
                same.append((position, bit))
        self.assertEqual(same, [(18, bit) for bit in range(3, 8)])
        self.assertEqual(refused, 8 * len(packed) - 5)


class PackedFileTests(unittest.TestCase):
    def test_a_proof_written_packed_is_the_proof_read(self) -> None:
        for kind in KINDS:
            with (
                self.subTest(kind=kind.name),
                tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary,
            ):
                folder = Path(temporary)
                encoded = kind.held.encoded()
                plain = kind.write(kind.held, folder / f"plain{kind.suffix}")
                self.assertIs(plain.packed, False)
                self.assertEqual(
                    (
                        plain.output_bytes,
                        plain.output_digest_sha256,
                        plain.unpacked_bytes,
                        plain.unpacked_digest_sha256,
                    ),
                    (kind.bytes, kind.sha256, kind.bytes, kind.sha256),
                )
                self.assertEqual(plain.output.read_bytes(), encoded)

                report = kind.write(
                    kind.held, folder / f"packed{kind.suffix}", packed=True
                )
                written = report.output.read_bytes()
                self.assertIs(report.packed, True)
                self.assertEqual(
                    (report.unpacked_bytes, report.unpacked_digest_sha256),
                    (kind.bytes, kind.sha256),
                )
                self.assertEqual(
                    (report.output_bytes, report.output_digest_sha256),
                    (len(written), hashlib.sha256(written).hexdigest()),
                )
                self.assertEqual(written, pack_tsdf_proof_bytes(encoded))
                self.assertEqual(written[:8], b"SFTPACK1")
                self.assertEqual(
                    struct.unpack("<Q", written[8:16]), (kind.bytes,)
                )
                self.assertEqual(zlib.decompress(written[16:]), encoded)
                self.assertLess(len(written), len(encoded) // 5)
                self.assertEqual(
                    sorted(path.name for path in folder.iterdir()),
                    [f"packed{kind.suffix}", f"plain{kind.suffix}"],
                )

                read = kind.load(report.output)
                self.assertEqual(read.encoded(), encoded)
                self.assertEqual(read.digest_sha256, kind.sha256)
                self.assertEqual(
                    kind.load(plain.output).encoded(), encoded
                )
                self.assertIs(kind.check(read).verdict, kind.good)

                # Decoding is of a proof's own bytes. A packed proof
                # is unpacked first, by whoever reads the file.
                with self.assertRaisesRegex(
                    TsdfError, f"^{kind.noun} has an unrecognized signature$"
                ):
                    kind.decode(written)
                with self.assertRaisesRegex(
                    TsdfError, "output already exists"
                ):
                    kind.write(kind.held, report.output, packed=True)
                self.assertEqual(report.output.read_bytes(), written)
                for wrong in (1, 0, "yes", None):
                    with self.assertRaisesRegex(
                        TsdfError, "^packed must be true or false$"
                    ):
                        kind.write(
                            kind.held,
                            folder / f"other{kind.suffix}",
                            packed=wrong,
                        )
                self.assertFalse((folder / f"other{kind.suffix}").exists())

    def test_depth_that_was_measured_packs_smaller(self) -> None:
        audit = WORK.room_audit
        encoded = audit.encoded()
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            report = write_tsdf_map_audit(
                audit, Path(temporary) / "room.sftaudit", packed=True
            )
            self.assertEqual(report.unpacked_bytes, len(encoded))
            self.assertLess(3 * report.output_bytes, 2 * len(encoded))
            read = load_tsdf_map_audit(report.output)
        self.assertEqual(read.encoded(), encoded)
        self.assertIs(
            check_tsdf_map_audit(
                WORK.room_statement, read, seed="auditor-1", count=20
            ).verdict,
            TsdfMapAuditVerdict.PASSED,
        )

    def test_a_file_packed_by_anyone_is_the_same_proof(self) -> None:
        for kind in KINDS:
            encoded = kind.held.encoded()
            files = set()
            with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
                path = Path(temporary) / f"made{kind.suffix}"
                for made, stream in streams_of(encoded):
                    with self.subTest(kind=kind.name, made=made):
                        packed = packed_by_hand(stream, len(encoded))
                        path.write_bytes(packed)
                        read = kind.load(path)
                        self.assertEqual(read.encoded(), encoded)
                        self.assertEqual(read.digest_sha256, kind.sha256)
                        self.assertIs(kind.check(read).verdict, kind.good)
                        files.add(hashlib.sha256(packed).hexdigest())
            # Many files, and one proof with one name.
            self.assertGreater(len(files), 4)
            self.assertNotIn(kind.sha256, files)

        # Two files a single byte apart that are the same proof: the
        # spare bits of a stored block, which no reader looks at.
        kind = KINDS[0]
        encoded = kind.held.encoded()
        one = packed_by_hand(stored_by_hand(encoded), len(encoded))
        other = packed_by_hand(
            stored_by_hand(encoded, spare_bits=31), len(encoded)
        )
        self.assertEqual(
            [at for at in range(len(one)) if one[at] != other[at]], [18]
        )
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            digests = []
            for name, packed in (("one", one), ("other", other)):
                path = Path(temporary) / f"{name}.sftproof"
                path.write_bytes(packed)
                digests.append(kind.load(path).digest_sha256)
        self.assertEqual(digests, [kind.sha256, kind.sha256])

    def test_what_is_unpacked_is_read_as_strictly_as_what_never_was_packed(
        self,
    ) -> None:
        for position, kind in enumerate(KINDS):
            encoded = kind.held.encoded()
            another = KINDS[(position + 1) % len(KINDS)].held.encoded()
            written = pack_tsdf_proof_bytes(encoded)

            def carrying(data: bytes) -> bytes:
                return packed_by_hand(zlib.compress(data, 6), len(data))

            cases = (
                (
                    "a byte after the proof",
                    carrying(encoded + b"\x00"),
                    f"^{kind.noun} holds 1 bytes after its end$",
                ),
                (
                    "a byte short of the proof",
                    carrying(encoded[:-1]),
                    f"^{kind.noun} ",
                ),
                (
                    "a proof of another kind",
                    carrying(another),
                    f"^{kind.noun} has an unrecognized signature$",
                ),
                (
                    "a proof packed twice",
                    carrying(written),
                    f"^{kind.noun} has an unrecognized signature$",
                ),
                (
                    "a byte after the stream",
                    written + b"\x00",
                    "^packed TSDF proof holds 1 bytes after its end$",
                ),
                (
                    "a byte short of the stream",
                    written[:-1],
                    "^packed TSDF proof is cut short$",
                ),
                (
                    "a signature and no more",
                    b"SFTPACK1",
                    "^packed TSDF proof is cut short before its length$",
                ),
                (
                    "a wrong length",
                    packed_by_hand(written[16:], len(encoded) + 1),
                    f"^packed TSDF proof inflates to {len(encoded)} bytes "
                    f"and declares {len(encoded) + 1}$",
                ),
            )
            with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
                path = Path(temporary) / f"held{kind.suffix}"
                for name, data, message in cases:
                    with self.subTest(kind=kind.name, name=name):
                        path.write_bytes(data)
                        with self.assertRaisesRegex(TsdfError, message):
                            kind.load(path)
                # More than a file of this kind may be, and so not
                # inflated to find out.
                path.write_bytes(packed_by_hand(written[16:], kind.most + 1))
                with self.subTest(kind=kind.name, name="too much"):
                    with forbidden_calls((_INFLATER,)):
                        with self.assertRaisesRegex(
                            TsdfError,
                            f"^packed TSDF proof declares {kind.most + 1} "
                            "bytes; one of this kind is from 1 to "
                            f"{kind.most}$",
                        ):
                            kind.load(path)

    def test_no_byte_of_a_packed_file_can_be_changed_into_another_proof(
        self,
    ) -> None:
        for kind in KINDS:
            encoded = kind.held.encoded()
            packed = pack_tsdf_proof_bytes(encoded)
            outcomes: Counter = Counter()
            with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
                path = Path(temporary) / f"changed{kind.suffix}"
                for position in range(len(packed)):
                    changed = bytearray(packed)
                    changed[position] ^= 1 << (position % 8)
                    path.write_bytes(changed)
                    try:
                        read = kind.load(path)
                    except TsdfError:
                        outcomes["refused"] += 1
                        continue
                    if read.encoded() == encoded:
                        outcomes["the same proof"] += 1
                        continue
                    # Other bytes that the stream's own checksum let
                    # through. They are still only what the statement
                    # says they are.
                    verdict = kind.check(read).verdict
                    if verdict is not kind.invalid:
                        self.fail(
                            f"byte {position} of a packed {kind.name} "
                            f"changed and it is {verdict.value}"
                        )
                    outcomes["another, and invalid"] += 1
            with self.subTest(kind=kind.name):
                self.assertEqual(sum(outcomes.values()), len(packed))
                self.assertGreater(
                    outcomes["refused"], 9 * len(packed) // 10
                )
                # The signature and the length are not the stream's
                # to forgive.
                self.assertGreaterEqual(outcomes["refused"], 16)


class PackCliTests(unittest.TestCase):
    def run_cli(self, *arguments: str) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                code = main(["reconstruct", *arguments])
            except SystemExit as stop:
                code = stop.code
        return code, stdout.getvalue(), stderr.getvalue()

    def cases(self):
        return (
            SimpleNamespace(
                kind=KINDS[0],
                write=("tsdf-block-prove",),
                asked=("--block", "1", "0", "-1"),
                title="TSDF BLOCK PROVE",
                described=(
                    "block: 1,0,-1 row=3 of=8\n"
                    "proof_holds: frames=2 frames_with_tiles=2 tiles=2 "
                    "tile_pixels=16 depth_bytes=16 tile_proof_hashes=0 "
                    "block_path_hashes=3\n"
                ),
                check=("tsdf-block-proof-check",),
                check_asked=(),
                failed="TSDF BLOCK PROOF CHECK FAILED",
                verdict="verdict: verified",
            ),
            SimpleNamespace(
                kind=KINDS[1],
                write=("tsdf-route-certify",),
                asked=(
                    "--band", "0", "0", "--min-weight", "2",
                    "--through", "2", "0", "--through", "5", "0",
                ),
                title="TSDF ROUTE CERTIFY",
                described=(
                    "route: up=z band=0..0 radius=0 min_weight=2 "
                    "waypoints=2\n"
                    "route_metres: length=0.375 width=0.125 from=0.000 "
                    "to=0.125\n"
                    "certificate_holds: blocks=1 frames=2 "
                    "frames_with_tiles=2 tiles=2 tile_pixels=16 "
                    "depth_bytes=16 tile_proof_hashes=0 "
                    "block_path_hashes=3\n"
                    "map_says: clear unobserved_voxels=0 occupied_voxels=0\n"
                ),
                check=("tsdf-route-check",),
                check_asked=(),
                failed="TSDF ROUTE CHECK FAILED",
                verdict="verdict: clear",
            ),
            SimpleNamespace(
                kind=KINDS[2],
                write=("tsdf-map-audit",),
                asked=("--seed", "auditor-1", "--blocks", "3"),
                title="TSDF MAP AUDIT",
                described=(
                    "draw: seed=auditor-1 blocks=3 of=8\n"
                    "audit_holds: frames=2 frames_with_tiles=2 tiles=2 "
                    "tile_pixels=16 depth_bytes=16 tile_proof_hashes=0 "
                    "block_path_hashes=9\n"
                ),
                check=("tsdf-map-audit-check",),
                check_asked=("--seed", "auditor-1", "--blocks", "3"),
                failed="TSDF MAP AUDIT CHECK FAILED",
                verdict="verdict: passed",
            ),
        )

    def test_cli_writes_a_packed_file_and_says_what_it_carries(self) -> None:
        before = tree_snapshot(WORK.root)
        for case in self.cases():
            kind = case.kind
            with (
                self.subTest(kind=kind.name),
                tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary,
            ):
                folder = Path(temporary)
                plain = folder / f"plain{kind.suffix}"
                packed = folder / f"packed{kind.suffix}"

                def written(output: Path, *more: str):
                    return self.run_cli(
                        *case.write,
                        str(WORK.statement_path), str(WORK.volume_path),
                        str(FIXTURE), str(output), *case.asked, *more,
                    )

                def expected(output: Path, packed_line: str) -> str:
                    return (
                        f"{case.title} scan-synthetic-0001\n"
                        "statement: valid\n"
                        f"statement_sha256: {FIXTURE_STATEMENT_SHA256}\n"
                        "volume: verified\n"
                        "statement_recomputed: matched\n"
                        + case.described
                        + f"{kind.name}_bytes: {kind.bytes}\n"
                        f"{kind.name}_sha256: {kind.sha256}\n"
                        + packed_line
                        + f"output: {output.resolve()}\n"
                        "statement_modified: no\n"
                        "volume_modified: no\n"
                        "session_modified: no\n"
                    )

                code, printed, errors = written(plain)
                self.assertEqual((code, errors), (0, ""))
                self.assertEqual(printed, expected(plain, ""))
                code, printed, errors = written(packed, "--packed")
                self.assertEqual((code, errors), (0, ""))
                size = packed.stat().st_size
                self.assertEqual(
                    printed, expected(packed, f"packed_bytes: {size}\n")
                )
                self.assertLess(size, kind.bytes // 5)
                self.assertEqual(plain.read_bytes(), kind.held.encoded())
                self.assertEqual(
                    unpack_tsdf_proof_bytes(
                        packed.read_bytes(), most_bytes=kind.bytes
                    ),
                    kind.held.encoded(),
                )

                # The check says of the packed file, line for line,
                # what it says of the other.
                said = []
                for held in (plain, packed):
                    with forbidden_calls(
                        (
                            "spatialforge.cli.load_scan_session",
                            "spatialforge.cli.load_tsdf_block_volume",
                        )
                    ):
                        code, printed, errors = self.run_cli(
                            *case.check, str(WORK.statement_path),
                            str(held), *case.check_asked,
                        )
                    self.assertEqual((code, errors), (0, ""))
                    said.append(printed)
                self.assertEqual(said[0], said[1])
                self.assertIn(
                    f"{kind.name}_sha256: {kind.sha256}\n", said[1]
                )
                self.assertIn(case.verdict + "\n", said[1])

                # A packed file cut short is not checked at all.
                cut = folder / f"cut{kind.suffix}"
                cut.write_bytes(packed.read_bytes()[:-1])
                code, printed, errors = self.run_cli(
                    *case.check, str(WORK.statement_path), str(cut),
                    *case.check_asked,
                )
                self.assertEqual((code, printed), (2, ""))
                self.assertEqual(
                    errors,
                    f"{case.failed} {cut}\n"
                    "- packed TSDF proof is cut short\n",
                )
        self.assertEqual(tree_snapshot(WORK.root), before)


if __name__ == "__main__":
    unittest.main()
