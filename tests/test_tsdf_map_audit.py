"""An audit by lot, held to the draw worked out by hand and to the proofs.

Three things are claimed of an audit.

Which rows a seed draws is a rule anyone can run. It is run here a second
time from the hash alone, and held to the module's draw for many seeds,
counts and sizes of map.

What a passed audit is worth is a count: how few wrong blocks it would
almost surely have found. That is worked out a second way, as a product
of fractions, for every draw from every map up to thirty blocks.

And an audit says of each block what that block's own proof says. On maps
altered in one block, in twenty and in half of them, an audit is refuted
exactly when its seed draws an altered row, and names exactly those. The
map with one block altered passes most audits, which is the limit of
auditing by lot and why a claim should carry its own blocks.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import shutil
import struct
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from spatialforge import (
    MAX_TSDF_MAP_AUDIT_BLOCKS,
    TSDF_MAP_AUDIT_CHECKS,
    TsdfMapAudit,
    TsdfMapAuditVerdict,
    audit_tsdf_map,
    check_tsdf_block_proof,
    check_tsdf_map_audit,
    decode_tsdf_map_audit,
    draw_tsdf_audit_rows,
    load_tsdf_block_volume,
    load_tsdf_map_audit,
    prove_tsdf_blocks,
    state_tsdf_map,
    tsdf_audit_wrong_blocks_caught,
    write_tsdf_block_volume,
    write_tsdf_map_audit,
    write_tsdf_map_statement,
)
from spatialforge.cli import main
from spatialforge.errors import TsdfError

from tests.heavy_fixtures import shared_room_case
from tests.test_tsdf_block_evidence import forbidden_calls, fused
from tests.test_tsdf_block_evidence import tree_snapshot
from tests.test_tsdf_block_proof import (
    ANOTHER_RULE,
    RULE_REFUSED,
    fixture_volume,
    flip_a_bit,
    forge,
    frozen,
    move_the_far_wall,
    remove_the_box,
    rows_that_differ,
    with_frame,
)
from tests.test_tsdf_block_volume import assemble, split

TEST_ROOT = Path(__file__).resolve().parent
FIXTURE = TEST_ROOT / "fixtures" / "minimal.vgsession"
FIXTURE_STATEMENT_SHA256 = (
    "7b771f1ef5ff5249f48cf29affdbce16df2349f02b67cf7d9accf4884414e780"
)
FORGED_STATEMENT_SHA256 = (
    "ffc316e93e4602857282cde9225207e7027a53578b4af5de83cdec42d1f87883"
)
# The fixture's audit of three blocks under the seed "auditor-1".
FIXTURE_AUDIT_SHA256 = (
    "44e1f3f4267c8c5ef14ab88d738016c99bea1024c7a36b241e37bf61b1bf230d"
)
FIXTURE_AUDIT_BYTES = 19_573
FIXTURE_ONE_SHA256 = (
    "6892984d3eae44122c972e316b7232acce39f051398dd873295790008ccfeab2"
)
FIXTURE_AUDIT_HEADER = "\n".join(
    (
        "{",
        '  "blocks": {',
        '    "drawn": 3,',
        '    "of": 8',
        "  },",
        '  "frames": 2,',
        '  "image": {',
        '    "height": 2,',
        '    "tile_pixels": 16,',
        '    "width": 2',
        "  },",
        '  "schema": "spatialforge.tsdf-map-audit",',
        '  "schema_version": "0.1.0",',
        '  "seed": "auditor-1",',
        f'  "statement_sha256": "{FIXTURE_STATEMENT_SHA256}",',
        '  "tile_frames": 2',
        "}",
        "",
    )
).encode("ascii")
CHECKS = (
    "statement",
    "shape",
    "draw",
    "scan root",
    "map root",
    "depth roots",
    "poses",
    "pixels",
    "fusion",
)

_FORBIDDEN_TO_A_CHECK = (
    "spatialforge.tsdf_held_blocks._gather_pooled_frames",
    "spatialforge.tsdf_held_blocks._commit_tsdf_scan",
    "spatialforge.tsdf_held_blocks.tsdf_volume_merkle_tree",
    "spatialforge.tsdf_block_evidence.replay_session",
    "spatialforge.tsdf_commitment.replay_session",
    "spatialforge.tsdf_stream_fusion._read_depth_array",
    "pathlib.Path.open",
    "PIL.Image.open",
)

WORK: SimpleNamespace | None = None
_GONE = object()


def rows_by_hand(digest: str, seed: str, count: int, blocks: int):
    """The rows a seed draws, from the hash and the rule and nothing else."""

    stem = (
        b"SFAUDIT1"
        + bytes.fromhex(digest)
        + len(seed).to_bytes(2, "little")
        + seed.encode("ascii")
    )
    even = 2**64 - 2**64 % blocks
    drawn: list[int] = []
    draw = 0
    while len(drawn) < count:
        number = int.from_bytes(
            hashlib.sha256(stem + draw.to_bytes(8, "little")).digest()[:8],
            "little",
        )
        draw += 1
        if number < even and number % blocks not in drawn:
            drawn.append(number % blocks)
    return tuple(sorted(drawn))


def chance_of_missing(blocks: int, wrong: int, drawn: int) -> Fraction:
    """One draw after another, none of them a wrong block."""

    chance = Fraction(1)
    for taken in range(drawn):
        chance *= Fraction(max(blocks - wrong - taken, 0), blocks - taken)
    return chance


def setUpModule() -> None:
    global WORK
    root = Path(tempfile.mkdtemp(dir=TEST_ROOT))
    unittest.addModuleCleanup(shutil.rmtree, root, True)

    volume_path, volume, session = fixture_volume(root, "fixture", FIXTURE)
    statement = state_tsdf_map(volume, session)
    statement_path = root / "fixture.sftcommit"
    write_tsdf_map_statement(statement, statement_path)
    header, indices, sums, weights = split(volume_path.read_bytes())
    target = int(np.flatnonzero(weights)[0])
    sums[target] = -sums[target]
    forged_path = root / "forged.sftvol"
    forged_path.write_bytes(assemble(header, indices, sums, weights))

    room = shared_room_case()
    room_storage, room_receipt = fused(room.plan, room.session)
    room_volume_path = root / "room.sftvol"
    write_tsdf_block_volume(room_storage, room_receipt, room_volume_path)
    room_volume = load_tsdf_block_volume(room_volume_path)
    room_statement = state_tsdf_map(room_volume, room.session)

    WORK = SimpleNamespace(
        root=root,
        session=session,
        volume_path=volume_path,
        volume=volume,
        statement=statement,
        statement_path=statement_path,
        forged_path=forged_path,
        fixture_audit=audit_tsdf_map(
            statement, volume, session, seed="auditor-1", count=3
        ),
        room=room,
        room_volume_path=room_volume_path,
        room_volume=room_volume,
        room_statement=room_statement,
        room_audit=audit_tsdf_map(
            room_statement,
            room_volume,
            room.session,
            seed="auditor-1",
            count=60,
        ),
    )


class DrawTests(unittest.TestCase):
    def test_the_fixtures_draws_written_out(self) -> None:
        digest = FIXTURE_STATEMENT_SHA256
        self.assertEqual(
            [
                draw_tsdf_audit_rows(digest, "auditor-1", count, 8)
                for count in (1, 2, 3)
            ],
            [(2,), (2, 5), (2, 3, 5)],
        )
        self.assertEqual(
            draw_tsdf_audit_rows(digest, "auditor-2", 3, 8), (1, 3, 5)
        )
        self.assertEqual(
            draw_tsdf_audit_rows(digest, "auditor-3", 3, 8), (0, 4, 7)
        )
        self.assertEqual(
            draw_tsdf_audit_rows(digest, "auditor-1", 8, 8),
            tuple(range(8)),
        )
        # Another statement, another draw.
        self.assertEqual(
            draw_tsdf_audit_rows(FORGED_STATEMENT_SHA256, "auditor-3", 3, 8),
            rows_by_hand(FORGED_STATEMENT_SHA256, "auditor-3", 3, 8),
        )

    def test_a_draw_is_the_rule_run_by_hand(self) -> None:
        digests = (
            FIXTURE_STATEMENT_SHA256,
            FORGED_STATEMENT_SHA256,
            "0" * 64,
        )
        for blocks in (1, 2, 3, 7, 8, 351, 1000, 69_415):
            for number, count in enumerate((1, 2, 5, 60, 459)):
                if count > blocks:
                    continue
                digest = digests[number % 3]
                for seed in ("a", "auditor-1", "2026-10-12T09:00:00.x_Y"):
                    with self.subTest(blocks=blocks, count=count, seed=seed):
                        rows = draw_tsdf_audit_rows(
                            digest, seed, count, blocks
                        )
                        self.assertEqual(
                            rows, rows_by_hand(digest, seed, count, blocks)
                        )
                        self.assertEqual(len(set(rows)), count)
                        self.assertEqual(list(rows), sorted(rows))
                        self.assertTrue(
                            all(0 <= row < blocks for row in rows)
                        )
                        # Asking for one more block keeps the ones
                        # already drawn.
                        if count < blocks:
                            self.assertLess(
                                set(rows),
                                set(
                                    draw_tsdf_audit_rows(
                                        digest, seed, count + 1, blocks
                                    )
                                ),
                            )

    def test_every_row_is_as_likely_as_another(self) -> None:
        # Two thousand seeds each draw one row of eight.
        drawn = Counter(
            draw_tsdf_audit_rows(
                FIXTURE_STATEMENT_SHA256, f"seed-{number}", 1, 8
            )[0]
            for number in range(2000)
        )
        self.assertEqual(sorted(drawn), list(range(8)))
        self.assertGreater(min(drawn.values()), 200)
        self.assertLess(max(drawn.values()), 300)

    def test_what_is_not_a_draw_is_refused(self) -> None:
        digest = FIXTURE_STATEMENT_SHA256
        seeds = ("", "two words", "a" * 129, "caf" + chr(233), 7, None, b"a")
        for seed in seeds:
            with self.subTest(seed=seed):
                with self.assertRaisesRegex(
                    TsdfError, "seed is 1 to 128 letters"
                ):
                    draw_tsdf_audit_rows(digest, seed, 1, 8)
        self.assertEqual(
            len(draw_tsdf_audit_rows(digest, "a" * 128, 1, 8)), 1
        )
        for count in (0, -1, True, 2.0, "2"):
            with self.subTest(count=count):
                with self.assertRaisesRegex(TsdfError, "positive integer"):
                    draw_tsdf_audit_rows(digest, "s", count, 8)
        with self.assertRaisesRegex(TsdfError, "block count is a positive"):
            draw_tsdf_audit_rows(digest, "s", 1, 0)
        with self.assertRaisesRegex(
            TsdfError, "cannot draw 9 different blocks from a map of 8"
        ):
            draw_tsdf_audit_rows(digest, "s", 9, 8)
        with self.assertRaisesRegex(TsdfError, "draws at most 65536"):
            draw_tsdf_audit_rows(
                digest, "s", MAX_TSDF_MAP_AUDIT_BLOCKS + 1, 10**6
            )
        for bad in (digest.upper(), digest[:-1], bytes.fromhex(digest)):
            with self.subTest(digest=bad):
                with self.assertRaisesRegex(TsdfError, "statement's SHA-256"):
                    draw_tsdf_audit_rows(bad, "s", 1, 8)


class BoundTests(unittest.TestCase):
    def test_the_bound_is_the_chance_worked_out_draw_by_draw(self) -> None:
        for blocks in range(1, 31):
            for drawn in range(1, blocks + 1):
                for one_in in (2, 100):
                    with self.subTest(
                        blocks=blocks, drawn=drawn, one_in=one_in
                    ):
                        caught = tsdf_audit_wrong_blocks_caught(
                            blocks, drawn, one_in=one_in
                        )
                        self.assertGreaterEqual(caught, 1)
                        self.assertLessEqual(
                            chance_of_missing(blocks, caught, drawn),
                            Fraction(1, one_in),
                        )
                        if caught > 1:
                            self.assertGreater(
                                chance_of_missing(blocks, caught - 1, drawn),
                                Fraction(1, one_in),
                            )

    def test_the_bounds_the_pages_quote(self) -> None:
        # The office, the living room and the desk, 459 blocks each.
        self.assertEqual(tsdf_audit_wrong_blocks_caught(69_415, 459), 691)
        self.assertEqual(tsdf_audit_wrong_blocks_caught(27_965, 459), 277)
        self.assertEqual(tsdf_audit_wrong_blocks_caught(5_401, 459), 52)
        self.assertEqual(tsdf_audit_wrong_blocks_caught(69_415, 200), 1578)
        # The design page's rule of thumb: 459 draws find a map that is
        # one part in a hundred wrong 99 times in 100.
        self.assertLessEqual(
            tsdf_audit_wrong_blocks_caught(1_000_000, 459), 10_000
        )
        self.assertLess(0.99**459, 0.01)
        self.assertGreater(0.99**458, 0.01)
        # Every block drawn: one wrong block cannot be missed. One
        # block drawn of eight: all eight must be wrong.
        self.assertEqual(tsdf_audit_wrong_blocks_caught(8, 8), 1)
        self.assertEqual(tsdf_audit_wrong_blocks_caught(8, 1), 8)
        self.assertEqual(tsdf_audit_wrong_blocks_caught(8, 3), 6)
        self.assertEqual(tsdf_audit_wrong_blocks_caught(351, 60), 24)
        self.assertEqual(
            tsdf_audit_wrong_blocks_caught(351, 60, one_in=2), 4
        )
        self.assertEqual(math.comb(8, 3), 56)
        for arguments in ((0, 1), (8, 0), (8, 9), (8.0, 3), (8, True)):
            with self.subTest(arguments=arguments):
                with self.assertRaisesRegex(TsdfError, "audit"):
                    tsdf_audit_wrong_blocks_caught(*arguments)
        with self.assertRaisesRegex(TsdfError, "odds is a positive"):
            tsdf_audit_wrong_blocks_caught(8, 3, one_in=0)


class AuditTests(unittest.TestCase):
    def test_an_honest_map_passes_and_its_blocks_are_their_proofs(
        self,
    ) -> None:
        audit = WORK.room_audit
        volume = WORK.room_volume
        self.assertEqual(TSDF_MAP_AUDIT_CHECKS, CHECKS)
        rows = rows_by_hand(
            WORK.room_statement.digest_sha256, "auditor-1", 60, 351
        )
        self.assertEqual(audit.block_rows, rows)
        self.assertEqual(
            audit.block_indices, tuple(volume.block_indices[r] for r in rows)
        )
        self.assertEqual(
            (audit.drawn_block_count, audit.block_count, audit.seed),
            (60, 351, "auditor-1"),
        )
        encoded = audit.encoded()
        decoded = decode_tsdf_map_audit(encoded)
        self.assertIsInstance(decoded, TsdfMapAudit)
        self.assertEqual(decoded.encoded(), encoded)
        with forbidden_calls(_FORBIDDEN_TO_A_CHECK):
            check = check_tsdf_map_audit(
                WORK.room_statement, decoded, seed="auditor-1", count=60
            )
        self.assertIs(check.verdict, TsdfMapAuditVerdict.PASSED)
        self.assertTrue(check.holds)
        self.assertIsNone(check.failed_check)
        self.assertEqual(check.passed_checks, CHECKS)
        self.assertEqual(check.refuted_blocks, ())
        self.assertEqual(
            (
                check.block_count,
                check.drawn_block_count,
                check.selected_observations,
                check.ready_observations,
            ),
            (351, 60, 20, 20),
        )
        # Each block held as its own proof holds it, and each proof
        # verified.
        proofs = prove_tsdf_blocks(
            WORK.room_statement, volume, WORK.room.session, audit.block_indices
        )
        pooled: dict[int, dict] = {}
        for place, proof in enumerate(proofs):
            self.assertTrue(
                check_tsdf_block_proof(WORK.room_statement, proof).holds
            )
            self.assertEqual(proof.block_row, audit.block_rows[place])
            self.assertEqual(proof.block_path, audit.block_paths[place])
            self.assertEqual(
                proof.tsdf_sums.tobytes(), audit.tsdf_sums[place].tobytes()
            )
            self.assertEqual(
                proof.weights.tobytes(), audit.weights[place].tobytes()
            )
            for tiles in proof.tile_frames:
                of_frame = pooled.setdefault(tiles.sequence, {})
                for tile in tiles.tiles.tiles:
                    of_frame[tile[:2]] = tile[2].tobytes()
        self.assertEqual(
            {
                tiles.sequence: {
                    tile[:2]: tile[2].tobytes()
                    for tile in tiles.tiles.tiles
                }
                for tiles in audit.tile_frames
            },
            pooled,
        )
        self.assertLess(
            len(encoded), sum(len(proof.encoded()) for proof in proofs) // 2
        )

    def test_an_altered_map_is_refuted_exactly_when_its_rows_are_drawn(
        self,
    ) -> None:
        true_bytes = WORK.room_volume_path.read_bytes()
        cases = (
            ("one block of 351", flip_a_bit, 1, 60, 20),
            ("twenty blocks", remove_the_box, 20, 30, 20),
            ("half the map", move_the_far_wall, 176, 10, 12),
        )
        refuted_counts = {}
        for name, edit, altered_blocks, count, seeds in cases:
            with self.subTest(alteration=name):
                with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
                    path = Path(temporary) / "altered.sftvol"
                    path.write_bytes(forge(true_bytes, edit))
                    altered = load_tsdf_block_volume(path)
                    statement = state_tsdf_map(altered, WORK.room.session)
                    changed = rows_that_differ(
                        WORK.room_volume, altered.tsdf_sums, altered.weights
                    )
                    self.assertEqual(len(changed), altered_blocks)
                    refuted = 0
                    for number in range(seeds):
                        seed = f"s{number}"
                        audit = audit_tsdf_map(
                            statement,
                            altered,
                            WORK.room.session,
                            seed=seed,
                            count=count,
                        )
                        check = check_tsdf_map_audit(
                            statement, audit, seed=seed, count=count
                        )
                        expected = {
                            altered.block_indices[row]: changed[row]
                            for row in audit.block_rows
                            if row in changed
                        }
                        self.assertEqual(
                            dict(check.refuted_blocks), expected
                        )
                        if expected:
                            self.assertIs(
                                check.verdict, TsdfMapAuditVerdict.REFUTED
                            )
                            self.assertEqual(check.failed_check, "fusion")
                            self.assertEqual(
                                check.passed_checks, CHECKS[:-1]
                            )
                            first = check.refuted_blocks[0]
                            self.assertEqual(
                                check.detail,
                                f"{len(expected)} of {count} blocks drawn "
                                "are not what the scan gives, the first of "
                                f"them {first[0]} in {first[1]} of 512 "
                                "voxels",
                            )
                            refuted += 1
                        else:
                            self.assertIs(
                                check.verdict, TsdfMapAuditVerdict.PASSED
                            )
                    refuted_counts[name] = (refuted, seeds)
        # Which rows a seed draws depends on the statement's digest, and
        # the room is generated, with poses from this machine's sines
        # and cosines. So these are held loosely where the fixture's
        # are held to the digit. A map half wrong is caught by nearly
        # every audit of ten, and one with twenty blocks wrong by most
        # audits of thirty. A map with one block wrong passes most
        # audits of sixty.
        self.assertGreaterEqual(refuted_counts["half the map"][0], 11)
        self.assertGreaterEqual(refuted_counts["twenty blocks"][0], 10)
        self.assertLessEqual(refuted_counts["one block of 351"][0], 10)

    def test_the_seed_and_the_count_are_the_auditors(self) -> None:
        audit = WORK.room_audit
        statement = WORK.room_statement

        def stopped(check, detail):
            self.assertIs(check.verdict, TsdfMapAuditVerdict.INVALID)
            self.assertEqual(check.failed_check, "draw")
            self.assertEqual(check.passed_checks, CHECKS[:2])
            self.assertRegex(check.detail, detail)

        stopped(
            check_tsdf_map_audit(
                statement, audit, seed="auditor-2", count=60
            ),
            "the audit was drawn with the seed 'auditor-1', which is not "
            "the seed it is checked with",
        )
        # Asked for more than it holds, and for fewer.
        stopped(
            check_tsdf_map_audit(
                statement, audit, seed="auditor-1", count=61
            ),
            r"the seed draws row \d+ of the map, and the audit does not "
            "hold it",
        )
        stopped(
            check_tsdf_map_audit(
                statement, audit, seed="auditor-1", count=59
            ),
            r"the audit holds row \d+ of the map, and the seed does not "
            "draw it",
        )
        stopped(
            check_tsdf_map_audit(
                statement, audit, seed="auditor-1", count=352
            ),
            "352 different blocks cannot be drawn from a map of 351",
        )
        # An answer made for another seed and relabelled with the
        # auditor's: true blocks of the map, and not the ones drawn.
        other = audit_tsdf_map(
            statement,
            WORK.room_volume,
            WORK.room.session,
            seed="auditor-2",
            count=60,
        )
        self.assertNotEqual(other.block_rows, audit.block_rows)
        self.assertTrue(
            check_tsdf_map_audit(
                statement, other, seed="auditor-2", count=60
            ).holds
        )
        stopped(
            check_tsdf_map_audit(
                statement,
                replace(other, seed="auditor-1"),
                seed="auditor-1",
                count=60,
            ),
            r"the seed draws row \d+ of the map, and the audit does not "
            "hold it",
        )
        for seed in ("", "two words", None):
            with self.subTest(seed=seed):
                with self.assertRaisesRegex(TsdfError, "seed is 1 to 128"):
                    check_tsdf_map_audit(statement, audit, seed=seed, count=60)
        with self.assertRaisesRegex(TsdfError, "count is a positive"):
            check_tsdf_map_audit(statement, audit, seed="auditor-1", count=0)
        with self.assertRaisesRegex(TsdfError, "TsdfMapStatement"):
            check_tsdf_map_audit(
                WORK.statement_path, audit, seed="auditor-1", count=60
            )
        with self.assertRaisesRegex(TsdfError, "TsdfMapAudit"):
            check_tsdf_map_audit(
                statement, audit.encoded(), seed="auditor-1", count=60
            )

    def test_an_altered_audit_is_invalid_at_a_named_check(self) -> None:
        audit = WORK.room_audit
        statement = WORK.room_statement

        def stopped(altered, by, **given):
            check = check_tsdf_map_audit(
                given.get("statement", statement),
                altered,
                seed="auditor-1",
                count=given.get("count", 60),
            )
            self.assertIs(check.verdict, TsdfMapAuditVerdict.INVALID)
            self.assertEqual(check.failed_check, by)
            self.assertEqual(check.passed_checks, CHECKS[:CHECKS.index(by)])
            self.assertEqual(check.refuted_blocks, ())
            return check

        stopped(replace(audit, statement_sha256="0" * 64), "statement")
        stopped(WORK.fixture_audit, "statement", count=3)
        stopped(
            replace(
                audit,
                frames=audit.frames[:-1],
                tile_frames=tuple(
                    tiles
                    for tiles in audit.tile_frames
                    if tiles.sequence < 19
                ),
            ),
            "shape",
        )
        nudged = list(audit.frames[4].transform)
        nudged[3] += 0.001
        stopped(with_frame(audit, 4, transform=tuple(nudged)), "scan root")
        flipped = audit.tsdf_sums.copy()
        observed = np.flatnonzero(audit.weights.reshape(-1))
        flipped.reshape(-1).view(np.uint64)[int(observed[0])] ^= 1
        check = stopped(
            replace(audit, tsdf_sums=frozen(flipped)), "map root"
        )
        self.assertRegex(check.detail, r"^block \(.*\) and its path lead to ")
        tiles = audit.tile_frames[0]
        first = tiles.tiles.tiles[0]
        changed = first[2].copy()
        changed[0, 0] += 1
        altered_tiles = replace(
            tiles,
            tiles=replace(
                tiles.tiles,
                tiles=((*first[:2], frozen(changed)),) + tiles.tiles.tiles[1:],
            ),
        )
        check = stopped(
            replace(
                audit, tile_frames=(altered_tiles,) + audit.tile_frames[1:]
            ),
            "depth roots",
        )
        self.assertEqual(
            check.detail,
            f"observation {tiles.sequence}: its {tiles.tile_count} tiles "
            "and the hashes with them do not lead to the frame's depth "
            "root",
        )
        check = stopped(
            replace(audit, tile_frames=audit.tile_frames[1:]), "pixels"
        )
        self.assertRegex(
            check.detail,
            f"observation {tiles.sequence}: the block reads the depth pixel",
        )

    def test_no_audit_is_checked_under_another_rule(self) -> None:
        other = replace(WORK.room_statement, rule=ANOTHER_RULE)
        audit = WORK.room_audit
        named = replace(audit, statement_sha256=other.digest_sha256)
        for held in (named, audit):
            check = check_tsdf_map_audit(
                other, held, seed="auditor-1", count=60
            )
            self.assertIs(check.verdict, TsdfMapAuditVerdict.INVALID)
            self.assertEqual(
                (check.failed_check, check.passed_checks, check.detail),
                ("statement", (), RULE_REFUSED),
            )
            self.assertEqual(check.refuted_blocks, ())
        with self.assertRaisesRegex(
            TsdfError,
            "not a statement of this volume and this scan: fusion rule$",
        ):
            audit_tsdf_map(
                other,
                WORK.room_volume,
                WORK.room.session,
                seed="auditor-1",
                count=60,
            )

    def test_an_answer_is_made_only_under_a_statement_of_these_two(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            TsdfError, "not a statement of this volume: block count"
        ):
            audit_tsdf_map(
                WORK.statement,
                WORK.room_volume,
                WORK.room.session,
                seed="s",
                count=3,
            )
        forged = load_tsdf_block_volume(WORK.forged_path)
        with self.assertRaisesRegex(
            TsdfError, "not a statement of this volume: map root"
        ):
            audit_tsdf_map(
                WORK.statement, forged, WORK.session, seed="s", count=3
            )
        with self.assertRaisesRegex(TsdfError, "cannot draw 9 different"):
            audit_tsdf_map(
                WORK.statement, WORK.volume, WORK.session, seed="s", count=9
            )
        with self.assertRaisesRegex(TsdfError, "seed is 1 to 128"):
            audit_tsdf_map(
                WORK.statement, WORK.volume, WORK.session, seed="", count=1
            )
        for arguments, message in (
            ((WORK.statement_path, WORK.volume, WORK.session),
             "TsdfMapStatement"),
            ((WORK.statement, WORK.volume_path, WORK.session),
             "TsdfBlockVolume"),
            ((WORK.statement, WORK.volume, FIXTURE), "loaded ScanSession"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(TsdfError, message):
                    audit_tsdf_map(*arguments, seed="s", count=1)
        before = tree_snapshot(WORK.root)
        audit_tsdf_map(
            WORK.statement, WORK.volume, WORK.session, seed="s", count=8
        )
        self.assertEqual(tree_snapshot(WORK.root), before)


class AuditFileTests(unittest.TestCase):
    def test_the_fixtures_audit_written_out(self) -> None:
        audit = WORK.fixture_audit
        encoded = audit.encoded()
        self.assertEqual(len(encoded), FIXTURE_AUDIT_BYTES)
        self.assertEqual(audit.digest_sha256, FIXTURE_AUDIT_SHA256)
        self.assertEqual(encoded[:8], b"SFTAUD01")
        self.assertEqual(struct.unpack_from("<Q", encoded, 8), (343,))
        self.assertEqual(encoded[16:359], FIXTURE_AUDIT_HEADER)
        self.assertEqual(audit.block_rows, (2, 3, 5))
        self.assertEqual(
            audit.block_indices, ((0, 0, -1), (1, 0, -1), (1, -1, 0))
        )
        # The two frames' leaves, then each block: its row, the block
        # as the map's tree holds it, and its three steps to the root.
        at = 359 + 2 * 177
        for place, row in enumerate(audit.block_rows):
            self.assertEqual(encoded[at:at + 8], struct.pack("<Q", row))
            self.assertEqual(
                encoded[at + 8:at + 28],
                b"SFBLOCK1" + struct.pack("<3i", *audit.block_indices[place]),
            )
            self.assertEqual(
                encoded[at + 28:at + 28 + 6144],
                WORK.volume.tsdf_sums[row].tobytes()
                + WORK.volume.weights[row].tobytes(),
            )
            at += 8 + 6164 + 96
        self.assertEqual(len(encoded) - at, 56)
        check = check_tsdf_map_audit(
            WORK.statement, audit, seed="auditor-1", count=3
        )
        self.assertIs(check.verdict, TsdfMapAuditVerdict.PASSED)
        one = audit_tsdf_map(
            WORK.statement, WORK.volume, WORK.session, seed="auditor-1",
            count=1,
        )
        self.assertEqual(one.digest_sha256, FIXTURE_ONE_SHA256)

    def test_no_byte_of_an_audit_can_be_changed(self) -> None:
        one = audit_tsdf_map(
            WORK.statement, WORK.volume, WORK.session, seed="auditor-1",
            count=1,
        )
        encoded = one.encoded()
        self.assertEqual(len(encoded), 7037)
        outcomes: Counter = Counter()
        for position in range(len(encoded)):
            changed = bytearray(encoded)
            changed[position] ^= 1 << (position % 8)
            try:
                check = check_tsdf_map_audit(
                    WORK.statement,
                    decode_tsdf_map_audit(bytes(changed)),
                    seed="auditor-1",
                    count=1,
                )
            except TsdfError:
                outcomes["not an audit"] += 1
                continue
            if check.verdict is not TsdfMapAuditVerdict.INVALID:
                self.fail(
                    f"byte {position} changed and the audit "
                    f"{check.verdict.value}"
                )
            outcomes[check.failed_check] += 1
        self.assertEqual(
            dict(outcomes),
            {
                "not an audit": 422,
                "statement": 19,
                "shape": 2,
                "draw": 6,
                "scan root": 320,
                "map root": 6252,
                "depth roots": 16,
            },
        )
        for length in range(len(encoded)):
            with self.assertRaises(TsdfError):
                decode_tsdf_map_audit(encoded[:length])
        with self.assertRaisesRegex(
            TsdfError, "TSDF map audit holds 1 bytes after its end"
        ):
            decode_tsdf_map_audit(encoded + b"\x00")

    def test_an_audit_written_is_the_audit_read(self) -> None:
        audit = WORK.fixture_audit
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            output = Path(temporary) / "made" / "answer.sftaudit"
            report = write_tsdf_map_audit(audit, output)
            self.assertEqual(report.output, output.resolve())
            self.assertEqual(report.output_bytes, FIXTURE_AUDIT_BYTES)
            self.assertEqual(
                report.output_digest_sha256, FIXTURE_AUDIT_SHA256
            )
            self.assertIs(report.audit, audit)
            self.assertEqual(output.read_bytes(), audit.encoded())
            self.assertEqual(
                [path.name for path in output.parent.iterdir()],
                ["answer.sftaudit"],
            )
            self.assertEqual(
                load_tsdf_map_audit(output).encoded(), audit.encoded()
            )
            with self.assertRaisesRegex(TsdfError, "output already exists"):
                write_tsdf_map_audit(audit, output)
            output.unlink()

            def link(source, target):
                Path(target).write_bytes(b"someone else's")
                raise FileExistsError(target)

            with patch("spatialforge.tsdf_map_audit.os.link", link):
                with self.assertRaisesRegex(
                    TsdfError, "appeared while writing"
                ):
                    write_tsdf_map_audit(audit, output)
            self.assertEqual(output.read_bytes(), b"someone else's")
            for name in ("answer.sftroute", "answer"):
                with self.subTest(name=name):
                    with self.assertRaisesRegex(TsdfError, ".sftaudit"):
                        write_tsdf_map_audit(audit, Path(temporary) / name)
            with self.assertRaisesRegex(TsdfError, "TsdfMapAudit"):
                write_tsdf_map_audit(
                    audit.encoded(), Path(temporary) / "other.sftaudit"
                )
            with self.assertRaisesRegex(TsdfError, "must end in .sftaudit"):
                load_tsdf_map_audit(WORK.statement_path)
            with self.assertRaisesRegex(
                TsdfError, "cannot read TSDF map audit"
            ):
                load_tsdf_map_audit(Path(temporary) / "gone.sftaudit")

    def test_bytes_that_are_not_exactly_an_audit_are_refused(self) -> None:
        encoded = WORK.fixture_audit.encoded()
        payload = 16 + 343

        def put(path, value):
            (length,) = struct.unpack_from("<Q", encoded, 8)
            document = json.loads(encoded[16:16 + length])
            target = document
            for key in path[:-1]:
                target = target[key]
            if value is _GONE:
                del target[path[-1]]
            else:
                target[path[-1]] = value
            header = (
                json.dumps(document, indent=2, sort_keys=True) + "\n"
            ).encode("ascii")
            return (
                encoded[:8] + struct.pack("<Q", len(header)) + header
                + encoded[payload:]
            )

        text = FIXTURE_AUDIT_HEADER.decode("ascii")

        def raw_header(changed: str) -> bytes:
            data = changed.encode("utf-8")
            return (
                encoded[:8] + struct.pack("<Q", len(data)) + data
                + encoded[payload:]
            )

        for held in (bytearray(encoded), memoryview(encoded), "SFTAUD01"):
            with self.subTest(held=type(held).__name__):
                with self.assertRaisesRegex(TsdfError, "decoded from bytes"):
                    decode_tsdf_map_audit(held)
        cases = {
            "nothing": (b"", "unrecognized signature"),
            "a certificate's signature": (
                b"SFTRTE01" + encoded[8:],
                "unrecognized signature",
            ),
            "a header of no length": (
                encoded[:8] + struct.pack("<Q", 1) + encoded[16:],
                "header length is out of range",
            ),
            "a header packed onto one line": (
                raw_header(json.dumps(json.loads(text), sort_keys=True)),
                "not in canonical form",
            ),
            "another schema": (
                put(("schema",), "spatialforge.tsdf-route-certificate"),
                "audit.schema: expected",
            ),
            "a field left out": (
                put(("seed",), _GONE),
                "audit.seed: missing",
            ),
            "a field nobody asked for": (
                put(("verdict",), "passed"),
                "audit: unknown field 'verdict'",
            ),
            "a seed of two words": (
                put(("seed",), "two words"),
                "seed is 1 to 128 letters",
            ),
            "a seed that is a number": (
                put(("seed",), 7),
                "seed is 1 to 128 letters",
            ),
            "no blocks drawn": (
                put(("blocks", "drawn"), 0),
                "audit.blocks.drawn",
            ),
            "one block more than it holds": (
                put(("blocks", "drawn"), 4),
                "counts more than its bytes can hold",
            ),
            "one block fewer than it holds": (
                put(("blocks", "drawn"), 2),
                "counts more than its bytes can hold",
            ),
            "a row past the map's last": (
                put(("blocks", "of"), 5),
                "TSDF map audit row lies outside the blocks it counts",
            ),
            "an array": (
                raw_header(text.replace('"frames": 2', '"frames": [2]')),
                "holds an array",
            ),
            "a key twice": (
                raw_header(
                    text.replace(
                        '  "tile_frames": 2',
                        '  "tile_frames": 2,\n  "tile_frames": 2',
                    )
                ),
                "duplicate JSON key 'tile_frames'",
            ),
            "not ascii": (
                raw_header(text.replace("0.1.0", "0.1." + chr(233))),
                "must be ASCII JSON",
            ),
            "rows out of order": (
                encoded[:payload + 354]
                + struct.pack("<Q", 3)
                + encoded[payload + 362:payload + 354 + 6268]
                + struct.pack("<Q", 2)
                + encoded[payload + 354 + 6276:],
                "distinct and in order of their rows",
            ),
            "the same row twice": (
                encoded[:payload + 354 + 6268]
                + struct.pack("<Q", 2)
                + encoded[payload + 354 + 6276:],
                "distinct and in order of their rows",
            ),
        }
        for name, (changed, message) in cases.items():
            with self.subTest(case=name):
                self.assertNotEqual(changed, encoded)
                with self.assertRaisesRegex(TsdfError, message):
                    decode_tsdf_map_audit(changed)

    def test_audits_validate_themselves(self) -> None:
        audit = WORK.room_audit
        cases = {
            "a seed of two words": ({"seed": "two words"}, "seed is 1 to"),
            "no seed": ({"seed": ""}, "seed is 1 to"),
            "a statement digest that is not one": (
                {"statement_sha256": "abc"},
                "statement digest",
            ),
            "no frames": ({"frames": ()}, "at least one"),
            "no blocks": ({"block_indices": ()}, "from 1 to 65536 blocks"),
            "rows out of order": (
                {"block_rows": audit.block_rows[::-1]},
                "distinct and in order of their rows",
            ),
            "a row twice": (
                {
                    "block_rows": (audit.block_rows[1],)
                    + audit.block_rows[1:]
                },
                "distinct and in order of their rows",
            ),
            "a row past the map's last": (
                {"block_rows": audit.block_rows[:-1] + (351,)},
                "a row of the map for each block",
            ),
            "sums that can be written": (
                {"tsdf_sums": audit.tsdf_sums.copy()},
                "immutable float64",
            ),
            "a path too few": (
                {"block_paths": audit.block_paths[:-1]},
                "a path for each block",
            ),
            "tile frames out of order": (
                {"tile_frames": audit.tile_frames[::-1]},
                "increasing frame order",
            ),
        }
        for name, (changes, message) in cases.items():
            with self.subTest(case=name):
                with self.assertRaisesRegex(TsdfError, message):
                    replace(audit, **changes)


class AuditCliTests(unittest.TestCase):
    def run_cli(self, *arguments: str) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                code = main(["reconstruct", *arguments])
            except SystemExit as stop:
                code = stop.code
        return code, stdout.getvalue(), stderr.getvalue()

    def test_cli_answers_an_audit_and_checks_it_alone(self) -> None:
        before = tree_snapshot(WORK.root)
        described = (
            "draw: seed=auditor-1 blocks=3 of=8\n"
            "audit_holds: frames=2 frames_with_tiles=2 tiles=2 "
            "tile_pixels=16 depth_bytes=16 tile_proof_hashes=0 "
            "block_path_hashes=9\n"
        )
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            output = Path(temporary) / "answer.sftaudit"
            arguments = (
                "tsdf-map-audit",
                str(WORK.statement_path), str(WORK.volume_path),
                str(FIXTURE), str(output),
                "--seed", "auditor-1", "--blocks", "3",
            )
            code, printed, errors = self.run_cli(*arguments)
            self.assertEqual(errors, "")
            self.assertEqual(code, 0)
            self.assertEqual(
                printed,
                "TSDF MAP AUDIT scan-synthetic-0001\n"
                "statement: valid\n"
                f"statement_sha256: {FIXTURE_STATEMENT_SHA256}\n"
                "volume: verified\n"
                "statement_recomputed: matched\n"
                + described
                + "audit_bytes: 19573\n"
                f"audit_sha256: {FIXTURE_AUDIT_SHA256}\n"
                f"output: {output.resolve()}\n"
                "statement_modified: no\n"
                "volume_modified: no\n"
                "session_modified: no\n",
            )
            code, printed, errors = self.run_cli(*arguments)
            self.assertEqual((code, printed), (2, ""))
            self.assertEqual(
                errors,
                f"TSDF MAP AUDIT FAILED {WORK.statement_path}\n"
                f"- output already exists: {output.resolve()}\n",
            )

            alone = Path(temporary) / "alone"
            alone.mkdir()
            statement = alone / "map.sftcommit"
            answer = alone / "answer.sftaudit"
            shutil.copyfile(WORK.statement_path, statement)
            shutil.copyfile(output, answer)
            with forbidden_calls(
                (
                    "spatialforge.cli.load_scan_session",
                    "spatialforge.cli.load_tsdf_block_volume",
                )
            ):
                code, printed, errors = self.run_cli(
                    "tsdf-map-audit-check", str(statement), str(answer),
                    "--seed", "auditor-1", "--blocks", "3",
                )
            self.assertEqual(errors, "")
            self.assertEqual(code, 0)
            self.assertEqual(
                printed,
                "TSDF MAP AUDIT CHECK scan-synthetic-0001\n"
                "statement: valid\n"
                f"statement_sha256: {FIXTURE_STATEMENT_SHA256}\n"
                "audit: well formed\n"
                f"audit_sha256: {FIXTURE_AUDIT_SHA256}\n"
                "asked: seed=auditor-1 blocks=3\n"
                + described
                + "".join(f"check {name}: passed\n" for name in CHECKS)
                + "fusion: blocks=3 refuted=0 selected=2 ready=2\n"
                "session_read: no\n"
                "volume_read: no\n"
                "statement_modified: no\n"
                "audit_modified: no\n"
                "verdict: passed\n"
                "bound: a map with 6 or more of its 8 blocks wrong passes "
                "such an audit less than 1 time in 100\n"
                "result: every block drawn is what the scan gives\n",
            )

            # Checked with a seed that is not the one it was drawn with.
            code, printed, errors = self.run_cli(
                "tsdf-map-audit-check", str(statement), str(answer),
                "--seed", "auditor-2", "--blocks", "3",
            )
            self.assertEqual((code, errors), (1, ""))
            self.assertEqual(
                printed.splitlines()[8:],
                [
                    "check statement: passed",
                    "check shape: passed",
                    "check draw: failed",
                ]
                + [f"check {name}: not reached" for name in CHECKS[3:]]
                + [
                    "detail: the audit was drawn with the seed "
                    "'auditor-1', which is not the seed it is checked "
                    "with",
                    "session_read: no",
                    "volume_read: no",
                    "statement_modified: no",
                    "audit_modified: no",
                    "verdict: invalid",
                    "result: the audit shows nothing about the map; it "
                    "failed the check of its draw",
                ],
            )
        self.assertEqual(tree_snapshot(WORK.root), before)

    def test_cli_refutes_an_altered_map_when_its_row_is_drawn(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            statement = Path(temporary) / "forged.sftcommit"
            code, _, errors = self.run_cli(
                "tsdf-map-commit",
                str(WORK.forged_path), str(FIXTURE), str(statement),
            )
            self.assertEqual((code, errors), (0, ""))
            self.assertEqual(
                hashlib.sha256(statement.read_bytes()).hexdigest(),
                FORGED_STATEMENT_SHA256,
            )
            verdicts = {}
            for seed in ("auditor-1", "auditor-3"):
                answer = Path(temporary) / f"{seed}.sftaudit"
                code, _, errors = self.run_cli(
                    "tsdf-map-audit",
                    str(statement), str(WORK.forged_path), str(FIXTURE),
                    str(answer), "--seed", seed, "--blocks", "3",
                )
                self.assertEqual((code, errors), (0, ""))
                code, printed, errors = self.run_cli(
                    "tsdf-map-audit-check", str(statement), str(answer),
                    "--seed", seed, "--blocks", "3",
                )
                self.assertEqual(errors, "")
                verdicts[seed] = (code, printed.splitlines())
        # The first seed does not draw the altered row, and the altered
        # map passes. The third does.
        code, lines = verdicts["auditor-1"]
        self.assertEqual(code, 0)
        self.assertEqual(lines[-3], "verdict: passed")
        code, lines = verdicts["auditor-3"]
        self.assertEqual(code, 1)
        self.assertEqual(
            lines[8:19],
            [f"check {name}: passed" for name in CHECKS[:-1]]
            + [
                "check fusion: failed",
                "detail: 1 of 3 blocks drawn are not what the scan gives, "
                "the first of them (0, -1, -1) in 1 of 512 voxels",
                "fusion: blocks=3 refuted=1 selected=2 ready=2",
            ],
        )
        self.assertEqual(
            lines[-2:],
            [
                "verdict: refuted",
                "result: a block drawn is not what the scan gives; the map "
                "does not follow from the scan",
            ],
        )

    def test_cli_refuses_what_it_cannot_answer_or_read(self) -> None:
        before = tree_snapshot(WORK.root)
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            folder = Path(temporary)
            output = folder / "answer.sftaudit"
            cases = (
                (
                    "more blocks than the map has",
                    (str(output), "--seed", "s", "--blocks", "9"),
                    "- an audit cannot draw 9 different blocks from a map "
                    "of 8\n",
                ),
                (
                    "a seed of two words",
                    (str(output), "--seed", "two words", "--blocks", "1"),
                    "- an audit's seed is 1 to 128 letters, digits, '.', "
                    "'_', ':' or '-'\n",
                ),
                (
                    "an output of another kind",
                    (str(folder / "answer.bin"), "--seed", "s",
                     "--blocks", "1"),
                    "- output filename must end in .sftaudit\n",
                ),
            )
            for name, arguments, message in cases:
                with self.subTest(answer=name):
                    code, printed, errors = self.run_cli(
                        "tsdf-map-audit",
                        str(WORK.statement_path), str(WORK.volume_path),
                        str(FIXTURE), *arguments,
                    )
                    self.assertEqual((code, printed), (2, ""))
                    self.assertEqual(
                        errors,
                        f"TSDF MAP AUDIT FAILED {WORK.statement_path}\n"
                        + message,
                    )
            for missing in (("--seed", "s"), ("--blocks", "1")):
                with self.subTest(given=missing):
                    code, printed, errors = self.run_cli(
                        "tsdf-map-audit",
                        str(WORK.statement_path), str(WORK.volume_path),
                        str(FIXTURE), str(output), *missing,
                    )
                    self.assertEqual((code, printed), (2, ""))
                    self.assertIn("required", errors)
            self.assertEqual(list(folder.iterdir()), [])

            encoded = WORK.fixture_audit.encoded()
            (folder / "cut.sftaudit").write_bytes(encoded[:-1])
            code, printed, errors = self.run_cli(
                "tsdf-map-audit-check",
                str(WORK.statement_path), str(folder / "cut.sftaudit"),
                "--seed", "auditor-1", "--blocks", "3",
            )
            self.assertEqual((code, printed), (2, ""))
            self.assertEqual(
                errors,
                f"TSDF MAP AUDIT CHECK FAILED {folder / 'cut.sftaudit'}\n"
                "- TSDF map audit is truncated inside a tile's samples\n",
            )
            (folder / "whole.sftaudit").write_bytes(encoded)
            code, printed, errors = self.run_cli(
                "tsdf-map-audit-check",
                str(WORK.statement_path), str(folder / "whole.sftaudit"),
                "--seed", "two words", "--blocks", "3",
            )
            self.assertEqual((code, printed), (2, ""))
            self.assertEqual(
                errors,
                f"TSDF MAP AUDIT CHECK FAILED {folder / 'whole.sftaudit'}\n"
                "- an audit's seed is 1 to 128 letters, digits, '.', '_', "
                "':' or '-'\n",
            )
        self.assertEqual(tree_snapshot(WORK.root), before)


if __name__ == "__main__":
    unittest.main()
