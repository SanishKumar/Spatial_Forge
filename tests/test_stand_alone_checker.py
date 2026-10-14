"""The stand-alone checker, held to the package's check of the same proofs.

``checker/check_block_proof.py`` checks a block's proof with nothing but
Python's standard library, from the documents. Three things are claimed
of it.

It stands alone. It imports nothing but the standard library, and it is
run here by an interpreter told to load nothing else: no site packages,
no environment, not the directory it is in.

It says what the package says. On every block of the committed fixture
and of the generated room, on every block after each of seven
alterations of the map or the scan, and on every proof made by changing
one bit of a proof, the two give the same verdict, fail the same check,
and count the same frames and the same differing voxels. Where proofs
are made to fail the two checks one-bit changes never reach, a pixel
left out, a tile to spare and a pose that is not rigid, they fail them
alike.

And it fuses another way. The package rules frames out by the corners of
the block before it looks at a voxel. The checker looks at every voxel of
every frame. That the two agree bit for bit is the claim.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import io
import shutil
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from spatialforge import (
    check_tsdf_block_proof,
    commit_tsdf_scan,
    commit_tsdf_volume,
    decode_tsdf_block_proof,
    load_tsdf_block_volume,
    pack_tsdf_proof_bytes,
    prove_tsdf_blocks,
    state_tsdf_map,
    write_tsdf_block_proof,
    write_tsdf_block_volume,
    write_tsdf_map_statement,
)
from spatialforge.errors import TsdfError
from spatialforge.merkle import merkle_root
from spatialforge.session_loader import load_scan_session
from spatialforge.tsdf_commitment import _statement

from tests.heavy_fixtures import shared_room_case
from tests.test_tsdf_block_evidence import fused
from tests.test_tsdf_block_proof import (
    FIXTURE_PROOFS,
    change_one_sample,
    drop_a_frame,
    exchange_two_frames,
    fixture_volume,
    flip_a_bit,
    forge,
    move_the_far_wall,
    nudge_a_pose,
    remove_the_box,
    tiles_of_the_scan,
    with_frame,
)
from tests.test_tsdf_block_volume import assemble, split
from tests.test_tsdf_commitment import copied

TEST_ROOT = Path(__file__).resolve().parent
CHECKER = TEST_ROOT.parent / "checker" / "check_block_proof.py"
FIXTURE = TEST_ROOT / "fixtures" / "minimal.vgsession"
FIXTURE_STATEMENT_SHA256 = (
    "7b771f1ef5ff5249f48cf29affdbce16df2349f02b67cf7d9accf4884414e780"
)
FORGED_STATEMENT_SHA256 = (
    "ffc316e93e4602857282cde9225207e7027a53578b4af5de83cdec42d1f87883"
)
CHECKS = (
    "statement",
    "shape",
    "scan root",
    "map root",
    "depth roots",
    "poses",
    "pixels",
    "fusion",
)


def load_checker():
    spec = importlib.util.spec_from_file_location("stand_alone", CHECKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ALONE = load_checker()


def package_says(statement, encoded: bytes) -> tuple:
    try:
        proof = decode_tsdf_block_proof(encoded)
    except TsdfError:
        return ("not a proof",)
    check = check_tsdf_block_proof(statement, proof)
    return (
        check.verdict.value,
        check.failed_check,
        check.differing_voxels,
        check.ready_observations,
        check.contributing_observations,
    )


def checker_says(statement_bytes: bytes, encoded: bytes) -> tuple:
    try:
        proof = ALONE.Proof(encoded)
    except ALONE.Unreadable:
        return ("not a proof",)
    outcome = ALONE.check(ALONE.Statement(statement_bytes), proof)
    return (
        outcome.verdict,
        outcome.failed_check,
        outcome.differing_voxels,
        outcome.ready,
        outcome.contributing,
    )


def setUpModule() -> None:
    global WORK
    root = Path(tempfile.mkdtemp(dir=TEST_ROOT))
    unittest.addModuleCleanup(shutil.rmtree, root, True)

    volume_path, volume, session = fixture_volume(root, "fixture", FIXTURE)
    statement = state_tsdf_map(volume, session)
    statement_path = root / "fixture.sftcommit"
    write_tsdf_map_statement(statement, statement_path)
    header, indices, sums, weights = split(volume_path.read_bytes())
    target = int((weights != 0).nonzero()[0][0])
    sums[target] = -sums[target]
    forged_path = root / "forged.sftvol"
    forged_path.write_bytes(assemble(header, indices, sums, weights))
    forged = load_tsdf_block_volume(forged_path)
    forged_statement = state_tsdf_map(forged, session)
    forged_statement_path = root / "forged.sftcommit"
    write_tsdf_map_statement(forged_statement, forged_statement_path)

    room = shared_room_case()
    room_storage, room_receipt = fused(room.plan, room.session)
    room_volume_path = root / "room.sftvol"
    write_tsdf_block_volume(room_storage, room_receipt, room_volume_path)
    room_volume = load_tsdf_block_volume(room_volume_path)
    room_statement = state_tsdf_map(room_volume, room.session)
    room_proofs = prove_tsdf_blocks(
        room_statement, room_volume, room.session, room_volume.block_indices
    )
    # The smallest block that some frames reach and some do not, with a
    # frame it reads two tiles of.
    subject = min(
        (
            proof
            for proof in room_proofs
            if 8 <= len(proof.tile_frames) <= 14
            and any(tiles.tile_count >= 2 for tiles in proof.tile_frames)
        ),
        key=lambda proof: len(proof.encoded()),
    )

    WORK = SimpleNamespace(
        root=root,
        session=session,
        volume=volume,
        statement=statement,
        statement_path=statement_path,
        proofs=prove_tsdf_blocks(
            statement, volume, session, volume.block_indices
        ),
        forged=forged,
        forged_statement=forged_statement,
        forged_statement_path=forged_statement_path,
        room=room,
        room_volume_path=room_volume_path,
        room_volume=room_volume,
        room_statement=room_statement,
        room_proofs=room_proofs,
        subject=subject,
    )


class StandsAloneTests(unittest.TestCase):
    def test_it_imports_only_the_standard_library(self) -> None:
        tree = ast.parse(CHECKER.read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(
                    alias.name.split(".")[0] for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom):
                self.assertEqual(node.level, 0)
                imported.add(node.module.split(".")[0])
        self.assertEqual(
            imported,
            {
                "__future__",
                "hashlib",
                "json",
                "math",
                "re",
                "struct",
                "sys",
                "zlib",
            },
        )
        self.assertLessEqual(
            imported - {"__future__"}, set(sys.stdlib_module_names)
        )

    def test_it_runs_where_nothing_else_can_be_loaded(self) -> None:
        bare = [sys.executable, "-I", "-S"]
        for module in ("numpy", "spatialforge"):
            with self.subTest(module=module):
                ran = subprocess.run(
                    [*bare, "-c", f"import {module}"],
                    cwd=TEST_ROOT.parent,
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(ran.returncode, 0)
                self.assertIn("ModuleNotFoundError", ran.stderr)
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            proof = Path(temporary) / "block.sftproof"
            write_tsdf_block_proof(WORK.proofs[3], proof)
            ran = subprocess.run(
                [*bare, str(CHECKER), str(WORK.statement_path), str(proof)],
                cwd=TEST_ROOT.parent,
                capture_output=True,
                text=True,
            )
        self.assertEqual((ran.returncode, ran.stderr), (0, ""))
        self.assertEqual(
            ran.stdout.replace("\r\n", "\n"),
            verified_output(),
        )


def verified_output() -> str:
    return (
        "BLOCK PROOF CHECK, STANDARD LIBRARY ONLY\n"
        f"statement_sha256: {FIXTURE_STATEMENT_SHA256}\n"
        f"proof_sha256: {FIXTURE_PROOFS[3][1]}\n"
        "packed: no\n"
        "block: 1,0,-1 row=3 of=8\n"
        + "".join(f"check {name}: passed\n" for name in CHECKS)
        + "fusion: ready=2 read=2 contributing=2 differing_voxels=0\n"
        "verdict: verified\n"
    )


class AgreementTests(unittest.TestCase):
    def agree(self, statement, proofs) -> Counter:
        statement_bytes = statement.encoded()
        verdicts: Counter = Counter()
        for proof in proofs:
            encoded = proof if isinstance(proof, bytes) else proof.encoded()
            said = package_says(statement, encoded)
            self.assertEqual(checker_says(statement_bytes, encoded), said)
            verdicts[said[:2]] += 1
        return verdicts

    def test_every_honest_block_is_verified_by_both(self) -> None:
        self.assertEqual(
            self.agree(WORK.statement, WORK.proofs),
            {("verified", None): 8},
        )
        self.assertEqual(
            self.agree(WORK.room_statement, WORK.room_proofs),
            {("verified", None): 351},
        )

    def test_every_one_bit_change_is_judged_alike(self) -> None:
        for name, statement, proof, outcomes in (
            (
                "the fixture's",
                WORK.statement,
                WORK.proofs[3],
                {
                    ("not a proof",): 403,
                    ("invalid", "statement"): 17,
                    ("invalid", "shape"): 1,
                    ("invalid", "scan root"): 320,
                    ("invalid", "map root"): 6252,
                    ("invalid", "depth roots"): 16,
                },
            ),
            ("the room's", WORK.room_statement, WORK.subject, None),
        ):
            with self.subTest(proof=name):
                encoded = proof.encoded()
                changed = []
                for position in range(len(encoded)):
                    one = bytearray(encoded)
                    one[position] ^= 1 << (position % 8)
                    changed.append(bytes(one))
                verdicts = self.agree(statement, changed)
                self.assertEqual(sum(verdicts.values()), len(encoded))
                self.assertNotIn(("verified", None), verdicts)
                self.assertNotIn(("refuted", "fusion"), verdicts)
                if outcomes is not None:
                    self.assertEqual(dict(verdicts), outcomes)

    def test_altered_maps_are_refuted_alike(self) -> None:
        self.assertEqual(
            self.agree(
                WORK.forged_statement,
                prove_tsdf_blocks(
                    WORK.forged_statement,
                    WORK.forged,
                    WORK.session,
                    WORK.forged.block_indices,
                ),
            ),
            {("refuted", "fusion"): 1, ("verified", None): 7},
        )
        true_bytes = WORK.room_volume_path.read_bytes()
        for name, edit, refuted in (
            ("one bit flipped", flip_a_bit, 1),
            ("a wall moved one voxel", move_the_far_wall, 176),
            ("an obstacle removed", remove_the_box, 20),
        ):
            with self.subTest(alteration=name):
                with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
                    path = Path(temporary) / "altered.sftvol"
                    path.write_bytes(forge(true_bytes, edit))
                    altered = load_tsdf_block_volume(path)
                statement = state_tsdf_map(altered, WORK.room.session)
                self.assertEqual(
                    self.agree(
                        statement,
                        prove_tsdf_blocks(
                            statement,
                            altered,
                            WORK.room.session,
                            altered.block_indices,
                        ),
                    ),
                    {
                        ("refuted", "fusion"): refuted,
                        ("verified", None): 351 - refuted,
                    },
                )

    def test_altered_scans_are_refuted_alike(self) -> None:
        committed = commit_tsdf_volume(WORK.room_volume)
        for name, alter, refuted in (
            ("one depth sample changed", change_one_sample, 3),
            ("a frame dropped", drop_a_frame, 253),
            ("two frames exchanged", exchange_two_frames, 234),
            ("a pose nudged", nudge_a_pose, 229),
        ):
            with self.subTest(alteration=name):
                with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
                    session_path = copied(WORK.room.session_path, temporary)
                    alter(session_path)
                    session = load_scan_session(session_path)
                    statement = _statement(
                        commit_tsdf_scan(session), committed
                    )
                    proofs = prove_tsdf_blocks(
                        statement,
                        WORK.room_volume,
                        session,
                        WORK.room_volume.block_indices,
                    )
                self.assertEqual(
                    self.agree(statement, proofs),
                    {
                        ("refuted", "fusion"): refuted,
                        ("verified", None): 351 - refuted,
                    },
                )

    def test_the_checks_one_bit_never_reaches_fail_alike(self) -> None:
        proof = WORK.subject
        session_path = WORK.room.session_path
        position, tiles = next(
            (position, tiles)
            for position, tiles in enumerate(proof.tile_frames)
            if tiles.tile_count >= 2
        )

        def with_tiles(places, at=position):
            return replace(
                proof,
                tile_frames=proof.tile_frames[:at]
                + (tiles_of_the_scan(session_path, tiles.sequence, places),)
                + proof.tile_frames[at + 1:],
            )

        spare = next(
            place for place in range(12) if place not in tiles.places
        )
        reading = {held.sequence for held in proof.tile_frames}
        unread = next(
            sequence for sequence in range(20) if sequence not in reading
        )
        made = [
            with_tiles(tiles.places[:left_out] + tiles.places[left_out + 1:])
            for left_out in range(tiles.tile_count)
        ]
        made += [
            replace(
                proof,
                tile_frames=proof.tile_frames[:left_out]
                + proof.tile_frames[left_out + 1:],
            )
            for left_out in range(len(proof.tile_frames))
        ]
        made.append(with_tiles(tiles.places + (spare,)))
        made.append(
            replace(
                proof,
                tile_frames=tuple(
                    sorted(
                        proof.tile_frames
                        + (tiles_of_the_scan(session_path, unread, [5]),),
                        key=lambda held: held.sequence,
                    )
                ),
            )
        )
        self.assertEqual(
            self.agree(WORK.room_statement, made),
            {("invalid", "pixels"): tiles.tile_count + len(proof.tile_frames)
             + 2},
        )

        # A scan committed with a pose fusion would refuse.
        read = proof.tile_frames[0].sequence
        pose = proof.frames[read].transform
        for name, transform in (
            (
                "stretched",
                tuple(
                    value * 1.5 if place in (0, 1, 2, 4, 5, 6, 8, 9, 10)
                    else value
                    for place, value in enumerate(pose)
                ),
            ),
            ("endless", pose[:3] + (float("inf"),) + pose[4:]),
        ):
            with self.subTest(pose=name):
                frames = with_frame(proof, read, transform=transform).frames
                statement = replace(
                    WORK.room_statement,
                    scan_root_sha256=merkle_root(
                        [frame.leaf_hash() for frame in frames]
                    ).hex(),
                )
                self.assertEqual(
                    self.agree(
                        statement,
                        [
                            replace(
                                proof,
                                frames=frames,
                                statement_sha256=statement.digest_sha256,
                            )
                        ],
                    ),
                    {("invalid", "poses"): 1},
                )

    def test_a_packed_proof_is_read_as_the_proof_it_carries(self) -> None:
        encoded = WORK.proofs[3].encoded()
        packed = pack_tsdf_proof_bytes(encoded)
        self.assertEqual(ALONE.unpacked(packed), (encoded, True))
        self.assertEqual(ALONE.unpacked(encoded), (encoded, False))
        for name, wrong in (
            ("cut short", packed[:-1]),
            ("a byte after its end", packed + b"\x00"),
            ("a wrong length", packed[:8] + (1).to_bytes(8, "little")
             + packed[16:]),
            ("its length alone", packed[:12]),
        ):
            with self.subTest(packed=name):
                with self.assertRaises(ALONE.Unreadable):
                    ALONE.unpacked(wrong)


class StatementTests(unittest.TestCase):
    def test_a_statement_is_read_as_strictly_as_the_package_reads_it(
        self,
    ) -> None:
        data = WORK.statement_path.read_bytes()
        statement = ALONE.Statement(data)
        self.assertEqual(statement.digest, FIXTURE_STATEMENT_SHA256)
        self.assertEqual(
            (
                statement.frames,
                statement.width,
                statement.height,
                statement.tile_pixels,
                statement.frame_stride,
                statement.blocks,
                statement.voxel_size_m,
                statement.truncation_m,
            ),
            (2, 2, 2, 16, 1, 8, 0.125, 0.5),
        )
        text = data.decode("ascii")
        for name, wrong in (
            ("indented otherwise", text.replace("\n  ", "\n   ")),
            ("without its last newline", text[:-1]),
            ("a field taken away", text.replace('  "hash": "sha-256",\n', "")),
            ("a count as text", text.replace('"frames": 2', '"frames": "2"')),
            ("a focal length as a whole number",
             text.replace('"fx": 2.0', '"fx": 2')),
            ("a field named twice",
             text.replace('{\n  "hash"', '{\n  "hash": "sha-256",\n  "hash"')),
            ("not JSON", text[:-3]),
        ):
            with self.subTest(statement=name):
                with self.assertRaises(ALONE.Unreadable):
                    ALONE.Statement(wrong.encode("ascii"))

    def test_it_checks_only_the_rule_it_knows(self) -> None:
        other = replace(
            WORK.statement, rule="projective-tsdf/unit-weight/nearest-pixel/2"
        )
        with self.assertRaisesRegex(
            ALONE.Unreadable,
            "made under the rule 'projective-tsdf/unit-weight/nearest-"
            "pixel/2', and this checker knows only 'projective-tsdf/"
            "unit-weight/nearest-pixel/1'",
        ):
            ALONE.Statement(other.encoded())


class CommandTests(unittest.TestCase):
    def run_checker(self, *arguments: str) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = ALONE.main(list(arguments))
        return code, stdout.getvalue(), stderr.getvalue()

    def test_what_the_command_says(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            folder = Path(temporary)
            proof = folder / "block.sftproof"
            write_tsdf_block_proof(WORK.proofs[3], proof)
            code, printed, errors = self.run_checker(
                str(WORK.statement_path), str(proof)
            )
            self.assertEqual((code, errors), (0, ""))
            self.assertEqual(printed, verified_output())

            packed = folder / "packed.sftproof"
            write_tsdf_block_proof(WORK.proofs[3], packed, packed=True)
            code, printed, errors = self.run_checker(
                str(WORK.statement_path), str(packed)
            )
            self.assertEqual((code, errors), (0, ""))
            self.assertEqual(
                printed,
                verified_output().replace("packed: no", "packed: yes"),
            )

            # The forged map's one altered block, under its own statement.
            refuted = folder / "forged.sftproof"
            write_tsdf_block_proof(
                prove_tsdf_blocks(
                    WORK.forged_statement,
                    WORK.forged,
                    WORK.session,
                    WORK.forged.block_indices[:1],
                )[0],
                refuted,
            )
            code, printed, errors = self.run_checker(
                str(WORK.forged_statement_path), str(refuted)
            )
            self.assertEqual((code, errors), (1, ""))
            self.assertEqual(
                printed,
                "BLOCK PROOF CHECK, STANDARD LIBRARY ONLY\n"
                f"statement_sha256: {FORGED_STATEMENT_SHA256}\n"
                "proof_sha256: "
                f"{hashlib.sha256(refuted.read_bytes()).hexdigest()}\n"
                "packed: no\n"
                "block: 0,-1,-1 row=0 of=8\n"
                + "".join(f"check {name}: passed\n" for name in CHECKS[:-1])
                + "check fusion: failed\n"
                "detail: 1 of 512 voxels are not what the scan gives\n"
                "fusion: ready=2 read=2 contributing=2 differing_voxels=1\n"
                "verdict: refuted\n",
            )

            # The honest block, under the forged map's statement.
            code, printed, errors = self.run_checker(
                str(WORK.forged_statement_path), str(proof)
            )
            self.assertEqual((code, errors), (1, ""))
            self.assertEqual(
                printed,
                "BLOCK PROOF CHECK, STANDARD LIBRARY ONLY\n"
                f"statement_sha256: {FORGED_STATEMENT_SHA256}\n"
                f"proof_sha256: {FIXTURE_PROOFS[3][1]}\n"
                "packed: no\n"
                "block: 1,0,-1 row=3 of=8\n"
                "check statement: failed\n"
                + "".join(
                    f"check {name}: not reached\n" for name in CHECKS[1:]
                )
                + "detail: the proof names statement "
                f"{FIXTURE_STATEMENT_SHA256}\n"
                "verdict: invalid\n",
            )

            cut = folder / "cut.sftproof"
            cut.write_bytes(proof.read_bytes()[:100])
            for arguments, path, message in (
                (
                    (str(WORK.statement_path), str(cut)),
                    cut,
                    "the proof ends inside its header",
                ),
                (
                    (str(proof), str(proof)),
                    proof,
                    "the statement is not ASCII",
                ),
                (
                    (str(WORK.statement_path), str(folder / "gone")),
                    folder / "gone",
                    None,
                ),
            ):
                with self.subTest(path=path.name):
                    code, printed, errors = self.run_checker(*arguments)
                    self.assertEqual((code, printed), (2, ""))
                    lines = errors.splitlines()
                    self.assertEqual(lines[0], f"CANNOT CHECK {path}")
                    if message is not None:
                        self.assertEqual(lines[1:], [f"- {message}"])
            code, printed, errors = self.run_checker(str(proof))
            self.assertEqual((code, printed), (2, ""))
            self.assertIn("usage:", errors)


if __name__ == "__main__":
    unittest.main()
