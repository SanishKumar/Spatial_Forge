"""A block's proof, held to the block fused from whole frames.

The check reads a statement and a proof. What it says of a block has to be
what the first step says of it, which fuses the block from the session's
whole frames and compares: verified where that is the map's block, refuted
where it is not, for every block of a volume. That is required here of an
honest map, of three maps altered after fusion, and of four scans altered
before they were committed to.

Then the proof itself is altered, one thing at a time, and each alteration
has to be stopped by the check the page says stops it. An altered proof is
never a verified block, and it is never an accusation either: a proof
that does not tie to the statement shows nothing.

The committed fixture's proofs are written out as digests, which holds
every platform to the same bytes, and every byte of one of them is changed
in turn.
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import struct
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from spatialforge import (
    MAX_TSDF_BLOCK_PROOF_BYTES,
    TSDF_BLOCK_PROOF_CHECKS,
    TsdfBlockProof,
    TsdfBlockProofVerdict,
    TsdfDepthTiles,
    TsdfFrameCommitment,
    TsdfProofFrameTiles,
    check_tsdf_block_proof,
    commit_tsdf_scan,
    commit_tsdf_volume,
    decode_tsdf_block_proof,
    fuse_tsdf_blocks_alone,
    load_tsdf_block_plan,
    load_tsdf_block_proof,
    load_tsdf_block_volume,
    prove_tsdf_block,
    prove_tsdf_blocks,
    state_tsdf_map,
    tsdf_depth_tile_tree,
    write_tsdf_block_proof,
    write_tsdf_block_volume,
    write_tsdf_map_statement,
)
from spatialforge import tsdf_block_proof
from spatialforge.cli import main
from spatialforge.errors import TsdfError
from spatialforge.merkle import merkle_path_length, merkle_root
from spatialforge.session_loader import load_scan_session
from spatialforge.tsdf_block_plan import plan_tsdf_blocks
from spatialforge.tsdf_commitment import _statement

from tests.heavy_fixtures import shared_room_case
from tests.room_fixture import BOX
from tests.test_tsdf_block_evidence import forbidden_calls, fused
from tests.test_tsdf_block_evidence import tree_snapshot
from tests.test_tsdf_block_volume import assemble, split
from tests.test_tsdf_commitment import (
    copied,
    pgm,
    raw_depth,
    without_first_line,
    write_pgm,
)

TEST_ROOT = Path(__file__).resolve().parent
FIXTURE = TEST_ROOT / "fixtures" / "minimal.vgsession"
PLAN_ARGUMENTS = {"voxel_size_m": 0.125, "truncation_m": 0.5}
FIXTURE_STATEMENT_SHA256 = (
    "7b771f1ef5ff5249f48cf29affdbce16df2349f02b67cf7d9accf4884414e780"
)
# Each block of the committed fixture and the SHA-256 of its proof file.
FIXTURE_PROOFS = (
    ((0, -1, -1),
     "3fffaefa70689be445fa0bfdf63f00697e61c959d36ead952de6849049aa0dd7"),
    ((1, -1, -1),
     "5dd7c5ea510f7e93fd564655bdd332b58c17cde02fa8e349a7850f99fe4ce6fd"),
    ((0, 0, -1),
     "70258d138b314cc00c49855a5e8e21966340f160f5836e585dd29408d592b8d1"),
    ((1, 0, -1),
     "50929ac2c1cabc3ef1e78759087718966036cd1147213c278f1113786612f626"),
    ((0, -1, 0),
     "4f5ba2c86c824a1568657d44d6ffc56dd6cf74276de58666d1f360f17b885f38"),
    ((1, -1, 0),
     "f6eae50308bf563340ed795ca18a3f2fa3c2afc02b0c66e3993d6aef2217a1e4"),
    ((0, 0, 0),
     "45e392a835e56a207a562cbf6df4e45573a0b3118d92ee5b577b7429ec2a8d2f"),
    ((1, 0, 0),
     "e6c26b8f57760c073631f4c4dd10bc55bb244e29b4d76e63df1537cf072f9d4c"),
)
FIXTURE_PROOF_BYTES = 7009
FIXTURE_PROOF_HEADER = "\n".join(
    (
        "{",
        '  "block": {',
        '    "blocks": 8,',
        '    "row": 0',
        "  },",
        '  "frames": 2,',
        '  "image": {',
        '    "height": 2,',
        '    "tile_pixels": 16,',
        '    "width": 2',
        "  },",
        '  "schema": "spatialforge.tsdf-block-proof",',
        '  "schema_version": "0.1.0",',
        f'  "statement_sha256": "{FIXTURE_STATEMENT_SHA256}",',
        '  "tile_frames": 2',
        "}",
        "",
    )
).encode("ascii")
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
ROOM_VOXEL_M = 0.04

# A check is given a statement and a proof. It may not go looking for
# anything else.
_FORBIDDEN_TO_A_CHECK = (
    "spatialforge.tsdf_block_proof.gather_tsdf_blocks_evidence",
    "spatialforge.tsdf_block_proof._commit_tsdf_scan",
    "spatialforge.tsdf_block_proof.tsdf_volume_merkle_tree",
    "spatialforge.tsdf_block_evidence.replay_session",
    "spatialforge.tsdf_commitment.replay_session",
    "spatialforge.tsdf_lone_block.replay_session",
    "spatialforge.tsdf_stream_fusion._read_depth_array",
    "spatialforge.tsdf_block_evidence._decode_raw_depth",
    "spatialforge.tsdf_commitment._decode_raw_depth",
    "pathlib.Path.open",
    "PIL.Image.open",
)

WORK: SimpleNamespace | None = None


def forge(volume_bytes: bytes, edit) -> bytes:
    """A volume altered after fusion that still loads.

    The digests are recomputed, and so is everything the header says of
    its payload, as someone altering a map would.
    """

    header, indices, sums, weights = split(volume_bytes)
    edit(
        indices.reshape(-1, 3),
        sums.reshape(-1, 8, 8, 8),
        weights.reshape(-1, 8, 8, 8),
    )
    fusion = header["fusion"]
    fusion["observed_voxels"] = int(np.count_nonzero(weights))
    fusion["maximum_weight"] = int(weights.max())
    fusion["contributions_applied"] = int(weights.sum(dtype=np.uint64))
    return assemble(header, indices, sums, weights)


def flip_a_bit(blocks, sums, weights) -> None:
    """The last bit of one observed voxel's sum."""

    flat = sums.reshape(-1)
    counts = weights.reshape(-1)
    target = int(np.flatnonzero((counts > 0) & (np.abs(flat) < counts))[1000])
    flat.view(np.uint64)[target] ^= 1


def move_the_far_wall(blocks, sums, weights) -> None:
    """Everything in the far wall's two layers of blocks, one voxel on.

    The wall stands at x = 2.537 m, in the blocks whose x index is 7,
    and its band reaches the blocks at 8. Every voxel of those blocks
    takes the value of its neighbour one voxel nearer the camera.
    """

    rows = {tuple(block): row for row, block in enumerate(blocks.tolist())}
    old_sums, old_weights = sums.copy(), weights.copy()
    for (bx, by, bz), row in rows.items():
        if bx not in (7, 8):
            continue
        sums[row, :, :, 1:] = old_sums[row, :, :, :-1]
        weights[row, :, :, 1:] = old_weights[row, :, :, :-1]
        nearer = rows.get((bx - 1, by, bz))
        if nearer is None:
            sums[row, :, :, 0] = 0.0
            weights[row, :, :, 0] = 0
        else:
            sums[row, :, :, 0] = old_sums[nearer, :, :, 7]
            weights[row, :, :, 0] = old_weights[nearer, :, :, 7]


def remove_the_box(blocks, sums, weights) -> None:
    """Every observed voxel around the box, said to be empty space.

    A sum equal to its weight is a voxel a full truncation in front of
    any surface. The floor under the box is left alone.
    """

    centre = (np.arange(8) + 0.5) * ROOM_VOXEL_M
    band = 0.12
    for row, (bx, by, bz) in enumerate(blocks.tolist()):
        x = bx * 8 * ROOM_VOXEL_M + centre[None, None, :]
        y = by * 8 * ROOM_VOXEL_M + centre[None, :, None]
        z = bz * 8 * ROOM_VOXEL_M + centre[:, None, None]
        around = (
            (x > BOX[0] - band) & (x < BOX[1] + band)
            & (y > BOX[2] - band) & (y < BOX[3] + band)
            & (z > BOX[4] + 2 * band) & (z < BOX[5] + band)
            & (weights[row] > 0)
        )
        sums[row][around] = weights[row][around]


def rows_that_differ(volume, sums, weights) -> dict[int, int]:
    """Rows of a volume whose block is not these accumulators, by voxels."""

    differ = (
        volume.tsdf_sums.view(np.uint64).reshape(-1, 512)
        != np.ascontiguousarray(sums).view(np.uint64).reshape(-1, 512)
    ) | (volume.weights.reshape(-1, 512) != weights.reshape(-1, 512))
    return {
        row: int(count)
        for row, count in enumerate(differ.sum(axis=1).tolist())
        if count
    }


def what_whole_frames_say(volume, session) -> dict[int, int]:
    """The first step's account: rows the scan does not give, by voxels.

    Every block of the volume is fused alone from the session's whole
    frames and compared. No statement, no proof, no tile and no tree.
    """

    lone = fuse_tsdf_blocks_alone(
        session,
        volume.block_indices,
        voxel_size_m=volume.voxel_size_m,
        truncation_m=volume.truncation_m,
        frame_stride=volume.frame_stride,
    )
    return rows_that_differ(
        volume,
        np.stack([block.tsdf_sums for block in lone]),
        np.stack([block.weights for block in lone]),
    )


def refuted(checks) -> dict[int, int]:
    return {
        row: check.differing_voxels
        for row, check in enumerate(checks)
        if check.verdict is TsdfBlockProofVerdict.REFUTED
    }


def lines_of(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def put_lines(path: Path, rows: list[str]) -> None:
    path.write_text("".join(row + "\n" for row in rows), encoding="utf-8")


def change_one_sample(session_path: Path) -> None:
    depth = session_path / "data" / "depth" / "000007.pgm"
    width, height, samples = pgm(depth)
    samples[20 * width + 33] += 1
    write_pgm(depth, width, height, samples)


def drop_a_frame(session_path: Path) -> None:
    for name in ("rgb.jsonl", "depth.jsonl", "poses.jsonl"):
        path = session_path / "streams" / name
        rows = lines_of(path)
        put_lines(path, rows[:9] + rows[10:])


def exchange_two_frames(session_path: Path) -> None:
    depth = session_path / "data" / "depth"
    third = (depth / "000003.pgm").read_bytes()
    (depth / "000003.pgm").write_bytes((depth / "000004.pgm").read_bytes())
    (depth / "000004.pgm").write_bytes(third)


def nudge_a_pose(session_path: Path) -> None:
    path = session_path / "streams" / "poses.jsonl"
    rows = lines_of(path)
    entry = json.loads(rows[12])
    entry["T_world_camera"][3] += 0.001
    rows[12] = json.dumps(entry)
    put_lines(path, rows)


def with_tile_frame(proof, position: int, **changes):
    """A proof with one frame's tiles replaced."""

    return replace(
        proof,
        tile_frames=(
            proof.tile_frames[:position]
            + (replace(proof.tile_frames[position], **changes),)
            + proof.tile_frames[position + 1:]
        ),
    )


def with_frame(proof, sequence: int, **changes):
    """A proof with one frame's leaf replaced."""

    return replace(
        proof,
        frames=(
            proof.frames[:sequence]
            + (replace(proof.frames[sequence], **changes),)
            + proof.frames[sequence + 1:]
        ),
    )


def frozen(array: np.ndarray) -> np.ndarray:
    copy = array.copy()
    copy.setflags(write=False)
    return copy


def tile_of_the_scan(session_path: Path, sequence: int, place: int):
    """One tile of a frame and its path, cut from the scan itself."""

    raw = raw_depth(session_path / "data" / "depth" / f"{sequence:06d}.pgm")
    tree = tsdf_depth_tile_tree(raw, 16)
    columns = -(-raw.shape[1] // 16)
    tile_row, tile_column = divmod(place, columns)
    samples = frozen(
        raw[
            tile_row * 16:(tile_row + 1) * 16,
            tile_column * 16:(tile_column + 1) * 16,
        ]
    )
    return (tile_row, tile_column, samples), tree.path(place)


def rebuilt(encoded: bytes, change) -> bytes:
    """A proof file with its header document changed and re-encoded."""

    (length,) = struct.unpack_from("<Q", encoded, 8)
    document = json.loads(encoded[16:16 + length])
    change(document)
    header = (
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=True)
        + "\n"
    ).encode("ascii")
    return (
        encoded[:8] + struct.pack("<Q", len(header)) + header
        + encoded[16 + length:]
    )


def fixture_volume(root: Path, name: str, session_path: Path, **plan):
    plan_path = root / f"{name}.sftplan"
    session = load_scan_session(session_path)
    plan_tsdf_blocks(session, plan_path, **PLAN_ARGUMENTS, **plan)
    storage, receipt = fused(load_tsdf_block_plan(plan_path), session)
    volume_path = root / f"{name}.sftvol"
    write_tsdf_block_volume(storage, receipt, volume_path)
    return volume_path, load_tsdf_block_volume(volume_path), session


def setUpModule() -> None:
    global WORK
    root = Path(tempfile.mkdtemp(dir=TEST_ROOT))
    unittest.addModuleCleanup(shutil.rmtree, root, True)

    volume_path, volume, session = fixture_volume(root, "fixture", FIXTURE)
    statement = state_tsdf_map(volume, session)
    statement_path = root / "fixture.sftcommit"
    write_tsdf_map_statement(statement, statement_path)
    proofs = prove_tsdf_blocks(
        statement, volume, session, volume.block_indices
    )
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
    room_proofs = prove_tsdf_blocks(
        room_statement, room_volume, room.session, room_volume.block_indices
    )
    # A block that some frames reach and some do not, with a frame it
    # reads two tiles of, and with every one of its tiles deep enough in
    # a frame's tree to have a path.
    subject = next(
        proof
        for proof in room_proofs
        if 8 <= len(proof.tile_frames) <= 14
        and any(tiles.tile_count >= 2 for tiles in proof.tile_frames)
    )

    WORK = SimpleNamespace(
        root=root,
        session=session,
        volume_path=volume_path,
        volume=volume,
        statement=statement,
        statement_path=statement_path,
        proofs=proofs,
        forged_path=forged_path,
        forged=load_tsdf_block_volume(forged_path),
        room=room,
        room_volume_path=room_volume_path,
        room_volume=room_volume,
        room_statement=room_statement,
        room_proofs=room_proofs,
        room_checks=[
            check_tsdf_block_proof(room_statement, proof)
            for proof in room_proofs
        ],
        subject=subject,
    )


class ProofVerifiesTests(unittest.TestCase):
    def test_every_block_of_the_fixture_is_verified_from_its_proof(
        self,
    ) -> None:
        self.assertEqual(TSDF_BLOCK_PROOF_CHECKS, CHECKS)
        self.assertEqual(
            WORK.statement.digest_sha256, FIXTURE_STATEMENT_SHA256
        )
        for row, (proof, (block, digest)) in enumerate(
            zip(WORK.proofs, FIXTURE_PROOFS, strict=True)
        ):
            with self.subTest(block=block):
                # What the proof is, written out. The image is two
                # pixels square, so a frame is one tile and the tile is
                # its own depth root: no path.
                self.assertEqual(proof.block_index, block)
                self.assertEqual(proof.block_row, row)
                self.assertEqual(proof.block_count, 8)
                self.assertEqual(
                    proof.statement_sha256, FIXTURE_STATEMENT_SHA256
                )
                self.assertEqual(
                    (proof.width, proof.height, proof.tile_pixels),
                    (2, 2, 16),
                )
                self.assertEqual(proof.frame_count, 2)
                self.assertEqual(len(proof.block_path), 3)
                self.assertEqual(
                    [tiles.sequence for tiles in proof.tile_frames], [0, 1]
                )
                self.assertEqual(proof.tile_count, 2)
                self.assertEqual(proof.depth_byte_count, 16)
                self.assertEqual(proof.tile_path_hash_count, 0)
                self.assertEqual(
                    proof.tsdf_sums.tobytes(),
                    WORK.volume.tsdf_sums[row].tobytes(),
                )
                self.assertEqual(
                    proof.weights.tobytes(),
                    WORK.volume.weights[row].tobytes(),
                )
                encoded = proof.encoded()
                self.assertEqual(len(encoded), FIXTURE_PROOF_BYTES)
                self.assertEqual(hashlib.sha256(encoded).hexdigest(), digest)
                self.assertEqual(proof.digest_sha256, digest)

                check = check_tsdf_block_proof(WORK.statement, proof)
                self.assertIs(check.verdict, TsdfBlockProofVerdict.VERIFIED)
                self.assertTrue(check.holds)
                self.assertIsNone(check.failed_check)
                self.assertIsNone(check.detail)
                self.assertEqual(check.passed_checks, CHECKS)
                self.assertEqual(check.block_index, block)
                self.assertEqual(
                    check.statement_digest_sha256, FIXTURE_STATEMENT_SHA256
                )
                self.assertEqual(
                    (
                        check.differing_voxels,
                        check.selected_observations,
                        check.ready_observations,
                        check.ruled_out_observations,
                        check.evaluated_observations,
                        check.contributing_observations,
                    ),
                    (0, 2, 2, 0, 2, 2),
                )

    def test_every_block_of_the_room_is_what_whole_frames_say_it_is(
        self,
    ) -> None:
        room = WORK.room
        volume = WORK.room_volume
        self.assertEqual(len(WORK.room_proofs), 351)
        self.assertEqual(what_whole_frames_say(volume, room.session), {})
        self.assertEqual(
            Counter(check.verdict for check in WORK.room_checks),
            {TsdfBlockProofVerdict.VERIFIED: 351},
        )
        # And it got there the way the first step did: the same frames
        # ruled out by pose, the same evaluated, the same contributing.
        lone = fuse_tsdf_blocks_alone(
            room.session,
            volume.block_indices,
            voxel_size_m=volume.voxel_size_m,
            truncation_m=volume.truncation_m,
            frame_stride=volume.frame_stride,
        )
        for check, block, proof in zip(
            WORK.room_checks, lone, WORK.room_proofs, strict=True
        ):
            self.assertEqual(
                (
                    check.block_index,
                    check.selected_observations,
                    check.ready_observations,
                    check.ruled_out_observations,
                    check.evaluated_observations,
                    check.contributing_observations,
                ),
                (
                    block.block_index,
                    block.selected_observations,
                    block.ready_observations,
                    block.ruled_out_observations,
                    block.evaluated_observations,
                    block.contributing_observations,
                ),
            )
            # A frame is in the proof if the block reads a pixel of it,
            # which is no more frames than were evaluated.
            self.assertLessEqual(
                len(proof.tile_frames), check.evaluated_observations
            )
        # Blocks no frame reaches are proved empty with no depth at all.
        empty = [
            proof for proof in WORK.room_proofs if not proof.tile_frames
        ]
        self.assertEqual(len(empty), 34)
        self.assertTrue(all(not proof.weights.any() for proof in empty))

    def test_a_proof_holds_the_scans_leaves_and_the_maps_block(self) -> None:
        room = WORK.room
        scan = commit_tsdf_scan(room.session)
        for proof in WORK.room_proofs[::25]:
            with self.subTest(block=proof.block_index):
                self.assertEqual(proof.frames, scan.frames)
                self.assertEqual(
                    proof.block_index,
                    WORK.room_volume.block_indices[proof.block_row],
                )
                self.assertEqual(
                    len(proof.block_path),
                    merkle_path_length(proof.block_row, 351),
                )
                self.assertEqual(
                    proof.statement_sha256,
                    WORK.room_statement.digest_sha256,
                )

    def test_a_block_has_one_proof_however_it_is_asked_for(self) -> None:
        blocks = WORK.room_volume.block_indices[40:340:60]
        together = prove_tsdf_blocks(
            WORK.room_statement, WORK.room_volume, WORK.room.session, blocks
        )
        backwards = prove_tsdf_blocks(
            WORK.room_statement,
            WORK.room_volume,
            WORK.room.session,
            blocks[::-1],
        )
        self.assertEqual(
            [proof.block_index for proof in backwards], list(blocks[::-1])
        )
        for block, proof, other in zip(
            blocks, together, backwards[::-1], strict=True
        ):
            with self.subTest(block=block):
                alone = prove_tsdf_block(
                    WORK.room_statement,
                    WORK.room_volume,
                    WORK.room.session,
                    block,
                )
                row = WORK.room_volume.block_indices.index(block)
                expected = WORK.room_proofs[row].encoded()
                self.assertEqual(proof.encoded(), expected)
                self.assertEqual(other.encoded(), expected)
                self.assertEqual(alone.encoded(), expected)

    def test_a_proof_is_made_at_the_statements_tile_size(self) -> None:
        blocks = WORK.room_volume.block_indices[40:340:60]
        sizes = {}
        # Five pixels to a tile divides neither side of the image, so
        # the tiles along two edges are short; sixty-four is one tile
        # that is shorter than it is wide.
        for tile in (1, 5, 64):
            statement = state_tsdf_map(
                WORK.room_volume, WORK.room.session, tile_pixels=tile
            )
            proofs = prove_tsdf_blocks(
                statement, WORK.room_volume, WORK.room.session, blocks
            )
            for proof in proofs:
                with self.subTest(tile=tile, block=proof.block_index):
                    self.assertEqual(proof.tile_pixels, tile)
                    encoded = proof.encoded()
                    read = decode_tsdf_block_proof(encoded)
                    self.assertEqual(read.encoded(), encoded)
                    self.assertIs(
                        check_tsdf_block_proof(statement, read).verdict,
                        TsdfBlockProofVerdict.VERIFIED,
                    )
                    # Under the statement at sixteen it is a proof of
                    # nothing.
                    other = check_tsdf_block_proof(WORK.room_statement, proof)
                    self.assertIs(
                        other.verdict, TsdfBlockProofVerdict.INVALID
                    )
                    self.assertEqual(other.failed_check, "statement")
            sizes[tile] = (
                sum(proof.depth_byte_count for proof in proofs),
                sum(proof.tile_path_hash_count for proof in proofs),
            )
        # Smaller tiles: less depth, more path. At the whole image there
        # is no path at all.
        self.assertLess(sizes[1][0], sizes[5][0])
        self.assertLess(sizes[5][0], sizes[64][0])
        self.assertGreater(sizes[1][1], sizes[5][1])
        self.assertEqual(sizes[64][1], 0)

    def test_frames_fusion_does_not_select_bring_no_tiles(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            _, volume, session = fixture_volume(
                Path(temporary), "strided", FIXTURE, frame_stride=2
            )
        statement = state_tsdf_map(volume, session)
        self.assertEqual(statement.frame_stride, 2)
        proofs = prove_tsdf_blocks(
            statement, volume, session, volume.block_indices
        )
        tile = (0, 0, frozen(np.full((2, 2), 950, dtype=np.uint16)))
        for proof in proofs:
            with self.subTest(block=proof.block_index):
                # Both frames' leaves, and tiles of the first alone.
                self.assertEqual(proof.frame_count, 2)
                self.assertEqual(
                    [tiles.sequence for tiles in proof.tile_frames], [0]
                )
                check = check_tsdf_block_proof(statement, proof)
                self.assertIs(check.verdict, TsdfBlockProofVerdict.VERIFIED)
                self.assertEqual(check.selected_observations, 1)
                # The second frame's tile, true as it is, has no place.
                padded = replace(
                    proof,
                    tile_frames=proof.tile_frames
                    + (
                        TsdfProofFrameTiles(
                            sequence=1,
                            tiles=TsdfDepthTiles(2, 2, 16, (tile,)),
                            paths=((),),
                        ),
                    ),
                )
                check = check_tsdf_block_proof(statement, padded)
                self.assertIs(check.verdict, TsdfBlockProofVerdict.INVALID)
                self.assertEqual(check.failed_check, "pixels")
                self.assertEqual(
                    check.detail,
                    "observation 1 is not fused, and the proof holds "
                    "tiles of it",
                )

    def test_frames_without_depth_or_pose_bring_no_tiles(self) -> None:
        tile = (0, 0, frozen(np.full((2, 2), 1000, dtype=np.uint16)))
        cases = (
            (
                "depth",
                ("streams/depth.jsonl",),
                "depth roots",
                "observation 0 has no depth, and the proof holds tiles "
                "of it",
            ),
            (
                "pose",
                ("streams/poses.jsonl",),
                "pixels",
                "observation 0 is not fused, and the proof holds tiles "
                "of it",
            ),
        )
        for name, references, stopped_by, detail in cases:
            with self.subTest(missing=name):
                with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
                    session_path = copied(FIXTURE, temporary)
                    without_first_line(session_path, *references)
                    _, volume, session = fixture_volume(
                        Path(temporary), "lacking", session_path
                    )
                    statement = state_tsdf_map(volume, session)
                    proofs = prove_tsdf_blocks(
                        statement, volume, session, volume.block_indices
                    )
                for proof in proofs:
                    self.assertEqual(
                        [tiles.sequence for tiles in proof.tile_frames], [1]
                    )
                    check = check_tsdf_block_proof(statement, proof)
                    self.assertIs(
                        check.verdict, TsdfBlockProofVerdict.VERIFIED
                    )
                    self.assertEqual(check.ready_observations, 1)
                    padded = replace(
                        proof,
                        tile_frames=(
                            TsdfProofFrameTiles(
                                sequence=0,
                                tiles=TsdfDepthTiles(2, 2, 16, (tile,)),
                                paths=((),),
                            ),
                        )
                        + proof.tile_frames,
                    )
                    check = check_tsdf_block_proof(statement, padded)
                    self.assertIs(
                        check.verdict, TsdfBlockProofVerdict.INVALID
                    )
                    self.assertEqual(check.failed_check, stopped_by)
                    self.assertEqual(check.detail, detail)

    def test_a_check_reads_the_statement_and_the_proof_and_no_more(
        self,
    ) -> None:
        encoded = [proof.encoded() for proof in WORK.room_proofs[::40]]
        before = tree_snapshot(WORK.room.session_path)
        with forbidden_calls(_FORBIDDEN_TO_A_CHECK):
            for proof_bytes in encoded:
                check = check_tsdf_block_proof(
                    WORK.room_statement, decode_tsdf_block_proof(proof_bytes)
                )
                self.assertIs(check.verdict, TsdfBlockProofVerdict.VERIFIED)
        self.assertEqual(tree_snapshot(WORK.room.session_path), before)
        with self.assertRaisesRegex(TsdfError, "TsdfMapStatement"):
            check_tsdf_block_proof(WORK.statement_path, WORK.proofs[0])
        with self.assertRaisesRegex(TsdfError, "TsdfBlockProof"):
            check_tsdf_block_proof(WORK.statement, WORK.proofs[0].encoded())


class AlteredMapTests(unittest.TestCase):
    """A map changed after it was fused, and then committed to.

    The statement is a true statement of the altered map, so every proof
    ties to it. What is wrong is the map, and the fusion check says so:
    for exactly the blocks that were changed, and by exactly the voxels.
    """

    def test_each_alteration_is_refuted_in_the_blocks_it_touched(
        self,
    ) -> None:
        true_bytes = WORK.room_volume_path.read_bytes()
        cases = (
            ("one bit flipped", flip_a_bit, 1, 1),
            ("a wall moved one voxel", move_the_far_wall, 176, 40_309),
            ("an obstacle removed", remove_the_box, 20, 1_433),
        )
        for name, edit, blocks, voxels in cases:
            with self.subTest(alteration=name):
                with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
                    path = Path(temporary) / "altered.sftvol"
                    path.write_bytes(forge(true_bytes, edit))
                    altered = load_tsdf_block_volume(path)
                    statement = state_tsdf_map(altered, WORK.room.session)
                    proofs = prove_tsdf_blocks(
                        statement,
                        altered,
                        WORK.room.session,
                        altered.block_indices,
                    )
                changed = rows_that_differ(
                    WORK.room_volume, altered.tsdf_sums, altered.weights
                )
                self.assertEqual(
                    (len(changed), sum(changed.values())), (blocks, voxels)
                )
                checks = [
                    check_tsdf_block_proof(statement, proof)
                    for proof in proofs
                ]
                self.assertEqual(refuted(checks), changed)
                for row, check in enumerate(checks):
                    if row in changed:
                        self.assertEqual(check.failed_check, "fusion")
                        self.assertEqual(check.passed_checks, CHECKS[:-1])
                        self.assertEqual(
                            check.detail,
                            f"{changed[row]} of 512 voxels are not what "
                            "the scan gives",
                        )
                        self.assertFalse(check.holds)
                    else:
                        self.assertIs(
                            check.verdict, TsdfBlockProofVerdict.VERIFIED
                        )
                # The altered block was not smuggled into the true
                # map's proof either: there it is under no root.
                row = next(iter(changed))
                smuggled = replace(
                    WORK.room_proofs[row],
                    tsdf_sums=proofs[row].tsdf_sums,
                    weights=proofs[row].weights,
                )
                check = check_tsdf_block_proof(WORK.room_statement, smuggled)
                self.assertIs(check.verdict, TsdfBlockProofVerdict.INVALID)
                self.assertEqual(check.failed_check, "map root")
                # And no proof of it can be made under the true
                # statement.
                with self.assertRaisesRegex(
                    TsdfError,
                    "not a statement of this volume: map root",
                ):
                    prove_tsdf_block(
                        WORK.room_statement,
                        altered,
                        WORK.room.session,
                        altered.block_indices[row],
                    )

    def test_the_fixtures_forgery_is_refuted_in_its_one_block(self) -> None:
        statement = state_tsdf_map(WORK.forged, WORK.session)
        proofs = prove_tsdf_blocks(
            statement, WORK.forged, WORK.session, WORK.forged.block_indices
        )
        checks = [check_tsdf_block_proof(statement, proof) for proof in proofs]
        self.assertEqual(refuted(checks), {0: 1})
        self.assertEqual(checks[0].block_index, (0, -1, -1))
        self.assertEqual(
            (
                checks[0].selected_observations,
                checks[0].evaluated_observations,
                checks[0].contributing_observations,
            ),
            (2, 2, 2),
        )


class AlteredScanTests(unittest.TestCase):
    """A scan changed before it was committed to, beside an honest map.

    Again every proof ties to the statement. The blocks the change
    reaches no longer follow from the scan that was committed, and they
    are the blocks whole-frame fusion of the altered scan disagrees on.
    """

    def test_each_alteration_is_refuted_where_whole_frames_say_so(
        self,
    ) -> None:
        volume = WORK.room_volume
        committed = commit_tsdf_volume(volume)
        cases = (
            ("one depth sample changed", change_one_sample, 20, 3, 10),
            ("a frame dropped", drop_a_frame, 19, 253, 59_181),
            ("two frames exchanged", exchange_two_frames, 20, 234, 35_707),
            ("a pose nudged", nudge_a_pose, 20, 229, 28_552),
        )
        for name, alter, frames, blocks, voxels in cases:
            with self.subTest(alteration=name):
                with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
                    session_path = copied(WORK.room.session_path, temporary)
                    alter(session_path)
                    session = load_scan_session(session_path)
                    statement = _statement(
                        commit_tsdf_scan(session), committed
                    )
                    proofs = prove_tsdf_blocks(
                        statement, volume, session, volume.block_indices
                    )
                    expected = what_whole_frames_say(volume, session)
                self.assertEqual(statement.frame_count, frames)
                self.assertEqual(
                    (len(expected), sum(expected.values())), (blocks, voxels)
                )
                checks = [
                    check_tsdf_block_proof(statement, proof)
                    for proof in proofs
                ]
                self.assertEqual(refuted(checks), expected)
                self.assertEqual(
                    Counter(check.verdict for check in checks),
                    {
                        TsdfBlockProofVerdict.REFUTED: blocks,
                        TsdfBlockProofVerdict.VERIFIED: 351 - blocks,
                    },
                )
                for row in expected:
                    self.assertEqual(checks[row].failed_check, "fusion")
                # The honest statement takes none of these proofs.
                check = check_tsdf_block_proof(
                    WORK.room_statement, proofs[next(iter(expected))]
                )
                self.assertIs(check.verdict, TsdfBlockProofVerdict.INVALID)
                self.assertEqual(check.failed_check, "statement")

    def test_tiles_of_an_altered_scan_fail_under_the_honest_statement(
        self,
    ) -> None:
        # The proof names the honest statement and carries the honest
        # frames' leaves, with one tile from the altered scan.
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            session_path = copied(WORK.room.session_path, temporary)
            change_one_sample(session_path)
            (tile_row, tile_column, samples), path = tile_of_the_scan(
                session_path, 7, 6
            )
        self.assertEqual((tile_row, tile_column), (1, 2))
        carried = 0
        for proof in WORK.room_proofs:
            for position, tiles in enumerate(proof.tile_frames):
                if tiles.sequence != 7:
                    continue
                held = [tile[:2] for tile in tiles.tiles.tiles]
                if (1, 2) not in held:
                    continue
                place = held.index((1, 2))
                self.assertEqual(tiles.paths[place], path)
                altered = with_tile_frame(
                    proof,
                    position,
                    tiles=replace(
                        tiles.tiles,
                        tiles=tiles.tiles.tiles[:place]
                        + ((1, 2, samples),)
                        + tiles.tiles.tiles[place + 1:],
                    ),
                )
                check = check_tsdf_block_proof(WORK.room_statement, altered)
                self.assertIs(check.verdict, TsdfBlockProofVerdict.INVALID)
                self.assertEqual(check.failed_check, "depth roots")
                self.assertEqual(
                    check.detail,
                    "observation 7: the tile at row 1, column 2 does not "
                    "lead to the frame's depth root",
                )
                carried += 1
        self.assertGreater(carried, 10)


class AlteredProofTests(unittest.TestCase):
    """An honest statement and a proof that has been tampered with.

    Each alteration is stopped by one check, named here, and by none
    before it. None is taken for a verified block, and none for a
    refuted one.
    """

    def stopped(self, proof, by: str, detail: str | None = None, **given):
        statement = given.get("statement", WORK.room_statement)
        check = check_tsdf_block_proof(statement, proof)
        self.assertIs(check.verdict, TsdfBlockProofVerdict.INVALID)
        self.assertEqual(check.failed_check, by)
        self.assertEqual(check.passed_checks, CHECKS[:CHECKS.index(by)])
        self.assertEqual(check.differing_voxels, 0)
        self.assertFalse(check.holds)
        if detail is not None:
            self.assertRegex(check.detail, detail)
        return check

    def test_the_subject_is_a_block_worth_tampering_with(self) -> None:
        proof = WORK.subject
        self.assertIs(
            check_tsdf_block_proof(WORK.room_statement, proof).verdict,
            TsdfBlockProofVerdict.VERIFIED,
        )
        self.assertLess(len(proof.tile_frames), 20)
        self.assertTrue(
            all(len(path) in (3, 4) for tiles in proof.tile_frames
                for path in tiles.paths)
        )
        self.assertEqual(len(proof.block_path), 9)

    def test_a_proof_under_another_statement(self) -> None:
        proof = WORK.subject
        self.stopped(
            replace(proof, statement_sha256="0" * 64),
            "statement",
            "the proof was made under statement 0{64}",
        )
        other = state_tsdf_map(
            WORK.room_volume, WORK.room.session, tile_pixels=8
        )
        self.stopped(proof, "statement", statement=other)
        # A proof of the fixture's block against the room's statement.
        self.stopped(WORK.proofs[0], "statement")

    def test_a_proof_that_is_not_shaped_as_the_statement_says(self) -> None:
        # Each of these names the right statement and lies about what
        # the statement says.
        fixture = WORK.proofs[0]
        self.stopped(
            replace(fixture, block_count=7),
            "shape",
            "the proof has blocks 7 and the statement 8",
            statement=WORK.statement,
        )
        # The same tiles said to be cut two pixels to a tile. They are
        # the same samples, since the image is two pixels square.
        self.stopped(
            replace(
                fixture,
                tile_pixels=2,
                tile_frames=tuple(
                    replace(tiles, tiles=replace(tiles.tiles, tile_pixels=2))
                    for tiles in fixture.tile_frames
                ),
            ),
            "shape",
            "the proof has tile size 2 and the statement 16",
            statement=WORK.statement,
        )
        proof = WORK.subject
        kept = tuple(
            tiles for tiles in proof.tile_frames if tiles.sequence < 19
        )
        self.stopped(
            replace(proof, frames=proof.frames[:-1], tile_frames=kept),
            "shape",
            "the proof has frames 19 and the statement 20",
        )
        # A frame left out of the middle, the rest moved up to hide it.
        moved_up = proof.frames[:9] + tuple(
            replace(frame, sequence=frame.sequence - 1)
            for frame in proof.frames[10:]
        )
        self.stopped(
            replace(proof, frames=moved_up, tile_frames=kept),
            "shape",
            "the proof has frames 19 and the statement 20",
        )
        # Left out and not moved up, it is not a proof at all.
        with self.assertRaisesRegex(TsdfError, "every frame, in replay"):
            replace(proof, frames=proof.frames[:9] + proof.frames[10:])

    def test_frames_that_are_not_the_scans(self) -> None:
        proof = WORK.subject
        read = proof.tile_frames[0].sequence
        frames = proof.frames
        nudged = list(frames[read].transform)
        nudged[3] += 0.001
        cases = {
            "a pose nudged": with_frame(
                proof, read, transform=tuple(nudged)
            ),
            "a pose nudged by the least a float can move": with_frame(
                proof,
                read,
                transform=frames[read].transform[:3]
                + (float(np.nextafter(frames[read].transform[3], 9.0)),)
                + frames[read].transform[4:],
            ),
            "a pose taken away": with_frame(proof, 19, transform=None),
            "another depth root": with_frame(
                proof, read, depth_root=frames[read + 1].depth_root
            ),
            "a frame said to have no depth": with_frame(
                proof, 19, depth_root=None
            ),
            "two frames exchanged": replace(
                proof,
                frames=frames[:3]
                + (
                    replace(frames[4], sequence=3),
                    replace(frames[3], sequence=4),
                )
                + frames[5:],
            ),
            "a frame put in place of its neighbour": replace(
                proof,
                frames=frames[:9]
                + (replace(frames[10], sequence=9),)
                + frames[10:],
            ),
        }
        for name, altered in cases.items():
            with self.subTest(alteration=name):
                self.stopped(
                    altered, "scan root", "the frames' leaves lead to "
                )

    def test_a_block_that_is_not_the_maps(self) -> None:
        proof = WORK.subject
        row = proof.block_row
        self.assertEqual(
            merkle_path_length(row ^ 1, 351), len(proof.block_path)
        )
        flipped = proof.tsdf_sums.copy()
        flipped.reshape(-1).view(np.uint64)[
            int(np.flatnonzero(proof.weights.reshape(-1))[0])
        ] ^= 1
        heavier = proof.weights.copy()
        heavier.reshape(-1)[0] += 1
        x, y, z = proof.block_index
        other = WORK.room_proofs[row ^ 1]
        cases = {
            "one bit of one sum flipped": replace(
                proof, tsdf_sums=frozen(flipped)
            ),
            "one weight raised by one": replace(
                proof, weights=frozen(heavier)
            ),
            "said to be the block beside it": replace(
                proof, block_index=(x + 1, y, z)
            ),
            "said to sit at the row beside it": replace(
                proof, block_row=row ^ 1
            ),
            "the first step of its path replaced": replace(
                proof,
                block_path=(bytes(32),) + proof.block_path[1:],
            ),
            "the last step of its path replaced": replace(
                proof,
                block_path=proof.block_path[:-1] + (bytes(32),),
            ),
            "its path taken from the row beside it": replace(
                proof, block_path=other.block_path
            ),
            "another block's accumulators on its path": replace(
                proof, tsdf_sums=other.tsdf_sums, weights=other.weights
            ),
        }
        for name, altered in cases.items():
            with self.subTest(alteration=name):
                self.stopped(
                    altered, "map root", "the block and its path lead to "
                )

    def test_tiles_that_are_not_the_frames(self) -> None:
        proof = WORK.subject
        position, tiles = next(
            (position, tiles)
            for position, tiles in enumerate(proof.tile_frames)
            if tiles.tile_count >= 2
        )
        first = tiles.tiles.tiles[0]
        changed = first[2].copy()
        changed[5, 9] += 1
        held = {tile[:2] for tile in tiles.tiles.tiles}
        columns = tiles.tiles.tile_columns
        length = len(tiles.paths[0])
        # Somewhere else in the same frame that a path of this length
        # could lead from, and that the proof does not hold.
        elsewhere = next(
            divmod(place, columns)
            for place in range(12)
            if divmod(place, columns) not in held
            and merkle_path_length(place, 12) == length
        )
        moved = sorted(
            [(*elsewhere, first[2])] + list(tiles.tiles.tiles[1:]),
            key=lambda tile: tile[:2],
        )
        moved_paths = tuple(
            tiles.paths[0] if tile[:2] == elsewhere
            else tiles.paths[tiles.tiles.tiles.index(tile)]
            for tile in moved
        )
        later = proof.tile_frames[position + 1]
        cases = {
            "one depth sample changed": with_tile_frame(
                proof,
                position,
                tiles=replace(
                    tiles.tiles,
                    tiles=((*first[:2], frozen(changed)),)
                    + tiles.tiles.tiles[1:],
                ),
            ),
            "a step of a tile's path replaced": with_tile_frame(
                proof,
                position,
                paths=((bytes(32),) + tiles.paths[0][1:],) + tiles.paths[1:],
            ),
            "a tile said to sit elsewhere in its frame": with_tile_frame(
                proof,
                position,
                tiles=replace(tiles.tiles, tiles=tuple(moved)),
                paths=moved_paths,
            ),
            "one frame's tiles given as another's": replace(
                proof,
                tile_frames=proof.tile_frames[:position]
                + (
                    replace(later, sequence=tiles.sequence),
                    replace(tiles, sequence=later.sequence),
                )
                + proof.tile_frames[position + 2:],
            ),
        }
        for name, altered in cases.items():
            with self.subTest(alteration=name):
                self.stopped(
                    altered,
                    "depth roots",
                    r"observation \d+: the tile at row \d, column \d does "
                    "not lead to the frame's depth root",
                )

    def test_a_proof_that_holds_less_than_the_block_reads(self) -> None:
        proof = WORK.subject
        position, tiles = next(
            (position, tiles)
            for position, tiles in enumerate(proof.tile_frames)
            if tiles.tile_count >= 2
        )
        for left_out in range(tiles.tile_count):
            with self.subTest(tile=left_out):
                self.stopped(
                    with_tile_frame(
                        proof,
                        position,
                        tiles=replace(
                            tiles.tiles,
                            tiles=tiles.tiles.tiles[:left_out]
                            + tiles.tiles.tiles[left_out + 1:],
                        ),
                        paths=tiles.paths[:left_out]
                        + tiles.paths[left_out + 1:],
                    ),
                    "pixels",
                    f"observation {tiles.sequence}: the block reads the "
                    r"depth pixel at row \d+, column \d+, and the evidence "
                    "holds no tile for it",
                )
        # Every frame's tiles, left out whole in turn: a frame cannot
        # be dropped by not mentioning it.
        for left_out, tiles in enumerate(proof.tile_frames):
            with self.subTest(frame=tiles.sequence):
                self.stopped(
                    replace(
                        proof,
                        tile_frames=proof.tile_frames[:left_out]
                        + proof.tile_frames[left_out + 1:],
                    ),
                    "pixels",
                    f"observation {tiles.sequence}: the block reads the "
                    "depth pixel",
                )

    def test_a_proof_that_holds_more_than_the_block_reads(self) -> None:
        proof = WORK.subject
        session_path = WORK.room.session_path
        position, tiles = next(
            (position, tiles)
            for position, tiles in enumerate(proof.tile_frames)
            if tiles.tile_count < 12
        )
        columns = tiles.tiles.tile_columns
        held = {
            tile[0] * columns + tile[1] for tile in tiles.tiles.tiles
        }
        spare = next(place for place in range(12) if place not in held)
        tile, path = tile_of_the_scan(session_path, tiles.sequence, spare)
        entries = sorted(
            list(zip(tiles.tiles.tiles, tiles.paths)) + [(tile, path)],
            key=lambda entry: entry[0][:2],
        )
        # A true tile of the right frame with its true path, which the
        # block reads no pixel of.
        self.stopped(
            with_tile_frame(
                proof,
                position,
                tiles=replace(
                    tiles.tiles, tiles=tuple(entry[0] for entry in entries)
                ),
                paths=tuple(entry[1] for entry in entries),
            ),
            "pixels",
            f"observation {tiles.sequence}: the proof holds the tile at "
            f"row {tile[0]}, column {tile[1]}, and the block reads no "
            "pixel of it",
        )
        # A true tile of a frame the block reads nothing of.
        reading = {tiles.sequence for tiles in proof.tile_frames}
        unread = next(
            sequence for sequence in range(20) if sequence not in reading
        )
        tile, path = tile_of_the_scan(session_path, unread, 5)
        extra = TsdfProofFrameTiles(
            sequence=unread,
            tiles=TsdfDepthTiles(64, 48, 16, (tile,)),
            paths=(path,),
        )
        self.stopped(
            replace(
                proof,
                tile_frames=tuple(
                    sorted(
                        proof.tile_frames + (extra,),
                        key=lambda tiles: tiles.sequence,
                    )
                ),
            ),
            "pixels",
            f"observation {unread}: the proof holds the tile at row 1, "
            "column 1, and the block reads no pixel of it",
        )

    def test_a_scan_committed_with_a_pose_fusion_would_refuse(self) -> None:
        # Such a scan has no honest map, so nothing of it is verified.
        # The statement here is made by hand over the altered leaves, as
        # whoever committed to such a scan would have to make it.
        proof = WORK.subject
        read = proof.tile_frames[0].sequence
        pose = proof.frames[read].transform
        stretched = tuple(
            value * 1.5 if place in (0, 1, 2, 4, 5, 6, 8, 9, 10) else value
            for place, value in enumerate(pose)
        )
        endless = pose[:3] + (float("inf"),) + pose[4:]
        for name, transform in (
            ("stretched", stretched),
            ("endless", endless),
        ):
            with self.subTest(pose=name):
                frames = with_frame(proof, read, transform=transform).frames
                statement = replace(
                    WORK.room_statement,
                    scan_root_sha256=merkle_root(
                        [frame.leaf_hash() for frame in frames]
                    ).hex(),
                )
                self.stopped(
                    replace(
                        proof,
                        frames=frames,
                        statement_sha256=statement.digest_sha256,
                    ),
                    "poses",
                    f"observation {read} is committed with a pose that is "
                    "not a rigid transform; this scan cannot be fused",
                    statement=statement,
                )


class EveryByteTests(unittest.TestCase):
    def test_no_byte_of_a_proof_can_be_changed(self) -> None:
        proof = WORK.proofs[3]
        encoded = proof.encoded()
        self.assertEqual(len(encoded), FIXTURE_PROOF_BYTES)
        outcomes: Counter = Counter()
        for position in range(len(encoded)):
            changed = bytearray(encoded)
            changed[position] ^= 1 << (position % 8)
            try:
                check = check_tsdf_block_proof(
                    WORK.statement, decode_tsdf_block_proof(bytes(changed))
                )
            except TsdfError:
                outcomes["not a proof"] += 1
                continue
            if check.verdict is not TsdfBlockProofVerdict.INVALID:
                self.fail(
                    f"byte {position} changed and the block was "
                    f"{check.verdict.value}"
                )
            outcomes[check.failed_check] += 1
        # One bit in each of 7,009 bytes. Most of a proof this small is
        # the block; the frames' leaves and the tiles are the rest.
        self.assertEqual(
            dict(outcomes),
            {
                "not a proof": 403,
                "statement": 17,
                "shape": 1,
                "scan root": 320,
                "map root": 6252,
                "depth roots": 16,
            },
        )

    def test_no_proof_is_a_beginning_of_another(self) -> None:
        encoded = WORK.proofs[3].encoded()
        for length in range(len(encoded)):
            with self.assertRaises(TsdfError):
                decode_tsdf_block_proof(encoded[:length])
        with self.assertRaisesRegex(
            TsdfError, "holds 1 bytes after its end"
        ):
            decode_tsdf_block_proof(encoded + b"\x00")
        room = WORK.subject.encoded()
        for length in range(0, len(room), 97):
            with self.assertRaises(TsdfError):
                decode_tsdf_block_proof(room[:length])


class ProofFileTests(unittest.TestCase):
    def test_the_fixtures_proof_file_written_out(self) -> None:
        proof = WORK.proofs[0]
        encoded = proof.encoded()
        self.assertEqual(encoded[:8], b"SFTPRF01")
        self.assertEqual(struct.unpack_from("<Q", encoded, 8), (323,))
        self.assertEqual(encoded[16:339], FIXTURE_PROOF_HEADER)
        # Then the two frames' leaves, the block as the map's tree
        # holds it, and its three steps to the map's root.
        frames = b"".join(frame.leaf_data() for frame in proof.frames)
        self.assertEqual(len(frames), 2 * 177)
        self.assertEqual(encoded[339:693], frames)
        self.assertEqual(
            encoded[693:693 + 20], b"SFBLOCK1" + struct.pack("<3i", 0, -1, -1)
        )
        self.assertEqual(
            encoded[713:713 + 6144],
            WORK.volume.tsdf_sums[0].tobytes()
            + WORK.volume.weights[0].tobytes(),
        )
        self.assertEqual(encoded[6857:6953], b"".join(proof.block_path))
        # Then each frame's tiles: its place, how many, and for each
        # tile where it sits and its samples. One tile is a whole
        # frame here, so no path follows it.
        self.assertEqual(
            encoded[6953:],
            struct.pack("<QI", 0, 1)
            + struct.pack("<II", 0, 0)
            + struct.pack("<4H", *[1000] * 4)
            + struct.pack("<QI", 1, 1)
            + struct.pack("<II", 0, 0)
            + struct.pack("<4H", *[950] * 4),
        )
        decoded = decode_tsdf_block_proof(encoded)
        self.assertIsInstance(decoded, TsdfBlockProof)
        self.assertEqual(decoded.encoded(), encoded)
        self.assertFalse(decoded.tsdf_sums.flags.writeable)
        self.assertFalse(decoded.weights.flags.writeable)

    def test_a_proof_written_is_the_proof_read(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            for name, proof, statement in (
                ("fixture", WORK.proofs[5], WORK.statement),
                ("room", WORK.subject, WORK.room_statement),
            ):
                with self.subTest(proof=name):
                    output = Path(temporary) / name / "block.sftproof"
                    report = write_tsdf_block_proof(proof, output)
                    self.assertEqual(report.output, output.resolve())
                    self.assertEqual(
                        report.output_bytes, len(proof.encoded())
                    )
                    self.assertEqual(
                        report.output_digest_sha256, proof.digest_sha256
                    )
                    self.assertIs(report.proof, proof)
                    self.assertEqual(output.read_bytes(), proof.encoded())
                    self.assertEqual(
                        [path.name for path in output.parent.iterdir()],
                        ["block.sftproof"],
                    )
                    loaded = load_tsdf_block_proof(output)
                    self.assertEqual(loaded.encoded(), proof.encoded())
                    self.assertIs(
                        check_tsdf_block_proof(statement, loaded).verdict,
                        TsdfBlockProofVerdict.VERIFIED,
                    )
                    self.assertEqual(
                        load_tsdf_block_proof(str(output)).digest_sha256,
                        proof.digest_sha256,
                    )

    def test_writing_never_replaces_a_file(self) -> None:
        proof = WORK.proofs[0]
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            output = Path(temporary) / "block.sftproof"
            output.write_bytes(b"something else")
            with self.assertRaisesRegex(TsdfError, "output already exists"):
                write_tsdf_block_proof(proof, output)
            self.assertEqual(output.read_bytes(), b"something else")
            output.unlink()

            def link(source, target):
                Path(target).write_bytes(b"someone else's")
                raise FileExistsError(target)

            with patch("spatialforge.tsdf_block_proof.os.link", link):
                with self.assertRaisesRegex(
                    TsdfError, "appeared while writing"
                ):
                    write_tsdf_block_proof(proof, output)
            self.assertEqual(output.read_bytes(), b"someone else's")
            self.assertEqual(
                [path.name for path in Path(temporary).iterdir()],
                ["block.sftproof"],
            )
            output.unlink()

            def refuse(source, target):
                raise PermissionError("no links here")

            with patch("spatialforge.tsdf_block_proof.os.link", refuse):
                with self.assertRaisesRegex(
                    TsdfError, "cannot write TSDF block proof"
                ):
                    write_tsdf_block_proof(proof, output)
            for name in ("block.sftcommit", "block.sftproof.bin", "block"):
                with self.subTest(name=name):
                    with self.assertRaisesRegex(TsdfError, ".sftproof"):
                        write_tsdf_block_proof(proof, Path(temporary) / name)
            with self.assertRaisesRegex(TsdfError, "TsdfBlockProof"):
                write_tsdf_block_proof(proof.encoded(), output)
            self.assertEqual(list(Path(temporary).iterdir()), [])

            (Path(temporary) / "block.sftcommit").write_bytes(
                proof.encoded()
            )
            with self.assertRaisesRegex(TsdfError, "must end in .sftproof"):
                load_tsdf_block_proof(Path(temporary) / "block.sftcommit")
            with self.assertRaisesRegex(
                TsdfError, "cannot read TSDF block proof"
            ):
                load_tsdf_block_proof(Path(temporary) / "gone.sftproof")
            self.assertGreater(MAX_TSDF_BLOCK_PROOF_BYTES, 2**29)

    def test_bytes_that_are_not_exactly_a_proof_are_refused(self) -> None:
        encoded = WORK.proofs[0].encoded()
        payload = 16 + 323

        def header(change):
            return rebuilt(encoded, change)

        def put(path, value):
            def change(document):
                target = document
                for key in path[:-1]:
                    target = target[key]
                if value is _GONE:
                    del target[path[-1]]
                else:
                    target[path[-1]] = value

            return header(change)

        def raw_header(text: str) -> bytes:
            data = text.encode("utf-8")
            return (
                encoded[:8] + struct.pack("<Q", len(data)) + data
                + encoded[payload:]
            )

        text = FIXTURE_PROOF_HEADER.decode("ascii")
        last_two = encoded[-56:]
        for held in (bytearray(encoded), memoryview(encoded), "SFTPRF01"):
            with self.subTest(held=type(held).__name__):
                with self.assertRaisesRegex(TsdfError, "decoded from bytes"):
                    decode_tsdf_block_proof(held)
        cases = {
            "nothing": (b"", "unrecognized signature"),
            "half a preamble": (encoded[:12], "unrecognized signature"),
            "a volume's signature": (
                b"SFTVOL01" + encoded[8:],
                "unrecognized signature",
            ),
            "a header of no length": (
                encoded[:8] + struct.pack("<Q", 0) + encoded[16:],
                "header length is out of range",
            ),
            "a header of impossible length": (
                encoded[:8] + struct.pack("<Q", 2**40) + encoded[16:],
                "header length is out of range",
            ),
            "a header longer than the file": (
                encoded[:8] + struct.pack("<Q", 16_000) + encoded[16:],
                "truncated inside its header",
            ),
            "a header packed onto one line": (
                raw_header(json.dumps(json.loads(text), sort_keys=True)),
                "not in canonical form",
            ),
            "a header with its keys in another order": (
                raw_header(
                    json.dumps(
                        dict(reversed(json.loads(text).items())), indent=2
                    )
                    + "\n"
                ),
                "not in canonical form",
            ),
            "another schema": (
                put(("schema",), "spatialforge.tsdf-map-statement"),
                "proof.schema: expected",
            ),
            "another version": (
                put(("schema_version",), "0.2.0"),
                "proof.schema_version: expected",
            ),
            "a field left out": (
                put(("tile_frames",), _GONE),
                "proof.tile_frames: missing",
            ),
            "a field nobody asked for": (
                put(("verdict",), "verified"),
                "proof: unknown field 'verdict'",
            ),
            "a block field nobody asked for": (
                put(("block", "x"), 0),
                "proof.block: unknown field 'x'",
            ),
            "a statement digest in capitals": (
                put(("statement_sha256",), FIXTURE_STATEMENT_SHA256.upper()),
                "proof.statement_sha256: expected a SHA-256",
            ),
            "no frames": (put(("frames",), 0), "proof.frames"),
            "frames as a flag": (put(("frames",), True), "proof.frames"),
            "a row below the first": (
                put(("block", "row"), -1),
                "proof.block.row",
            ),
            "a row past the last": (
                put(("block", "row"), 8),
                "row lies outside the blocks it counts",
            ),
            "no blocks": (put(("block", "blocks"), 0), "proof.block.blocks"),
            "an image of no width": (
                put(("image", "width"), 0),
                "proof.image.width",
            ),
            "tiles too large": (
                put(("image", "tile_pixels"), 1025),
                "tile size",
            ),
            "more frames than there are bytes for": (
                put(("frames",), 10**9),
                "counts more than its bytes can hold",
            ),
            "more tile frames than there are bytes for": (
                put(("tile_frames",), 10**9),
                "counts more than its bytes can hold",
            ),
            "one frame more than it holds": (
                put(("frames",), 3),
                "a TSDF frame leaf starts with its tag",
            ),
            "one frame fewer than it holds": (
                put(("frames",), 1),
                "block does not start with its tag",
            ),
            "one tile frame more than it holds": (
                put(("tile_frames",), 3),
                "truncated inside a frame's tiles",
            ),
            "one tile frame fewer than it holds": (
                put(("tile_frames",), 1),
                "holds 28 bytes after its end",
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
            "not a number": (
                raw_header(text.replace('"frames": 2', '"frames": NaN')),
                "invalid JSON",
            ),
            "not ascii": (
                raw_header(text.replace("0.1.0", "0.1." + chr(233))),
                "must be ASCII JSON",
            ),
            "a frame's tag damaged": (
                encoded[:payload] + b"X" + encoded[payload + 1:],
                "a TSDF frame leaf starts with its tag",
            ),
            "a frame out of its place": (
                encoded[:payload + 8]
                + struct.pack("<Q", 1)
                + encoded[payload + 16:],
                "every frame, in replay order",
            ),
            "a flag no frame has": (
                encoded[:payload + 16] + b"\x07" + encoded[payload + 17:],
                "unknown flag",
            ),
            "a pose in a frame said to have none": (
                encoded[:payload + 16] + b"\x01" + encoded[payload + 17:],
                "with no pose holds one",
            ),
            "a depth root in a frame said to have none": (
                encoded[:payload + 16] + b"\x02" + encoded[payload + 17:],
                "with no depth holds a depth root",
            ),
            "the block's tag damaged": (
                encoded[:693] + b"X" + encoded[694:],
                "block does not start with its tag",
            ),
            "a tile outside the image": (
                encoded[:-16] + struct.pack("<II", 5, 0) + encoded[-8:],
                "lies outside the image",
            ),
            "a frame named and no tile of it": (
                encoded[:-28] + struct.pack("<QI", 1, 0) + encoded[-16:],
                "holds no tile of it",
            ),
            "tile frames out of order": (
                encoded[:-56] + last_two[28:] + last_two[:28],
                "increasing frame order",
            ),
            "tiles of a frame it does not hold": (
                encoded[:-28] + struct.pack("<QI", 2, 1) + encoded[-16:],
                "tiles of a frame it does not hold",
            ),
            "the same frame's tiles twice": (
                encoded[:-28] + struct.pack("<QI", 0, 1) + encoded[-16:],
                "increasing frame order",
            ),
        }
        for name, (changed, message) in cases.items():
            with self.subTest(case=name):
                self.assertNotEqual(changed, encoded)
                with self.assertRaisesRegex(TsdfError, message):
                    decode_tsdf_block_proof(changed)
        self.assertEqual(
            decode_tsdf_block_proof(encoded).encoded(), encoded
        )

    def test_proofs_validate_themselves(self) -> None:
        proof = WORK.subject
        tiles = proof.tile_frames[0]
        writable = proof.tsdf_sums.copy()
        cases = {
            "a statement digest that is not one": (
                {"statement_sha256": "abc"},
                "statement digest",
            ),
            "tiles of no pixels": ({"tile_pixels": 0}, "tile size"),
            "an image of no height": ({"height": 0}, "image dimensions"),
            "no frames": ({"frames": ()}, "at least one"),
            "frames as a list": (
                {"frames": list(proof.frames)},
                "at least one",
            ),
            "something that is not a frame": (
                {"frames": proof.frames[:-1] + (None,)},
                "at least one",
            ),
            "frames out of order": (
                {"frames": proof.frames[::-1]},
                "every frame, in replay order",
            ),
            "no blocks": ({"block_count": 0}, "outside the blocks"),
            "a row past the last": ({"block_row": 351}, "outside the blocks"),
            "a row that is a flag": ({"block_row": True}, "an integer"),
            "an index of two": (
                {"block_index": proof.block_index[:2]},
                "three integers",
            ),
            "sums that can be written": (
                {"tsdf_sums": writable},
                "immutable float64",
            ),
            "sums of another shape": (
                {"tsdf_sums": frozen(proof.tsdf_sums.reshape(-1))},
                "immutable float64",
            ),
            "weights of another kind": (
                {"weights": frozen(proof.weights.astype(np.int64))},
                "immutable uint32",
            ),
            "a path one step short": (
                {"block_path": proof.block_path[:-1]},
                "block path here is 9 hashes",
            ),
            "a path of something else": (
                {"block_path": proof.block_path[:-1] + ("ab" * 32,)},
                "block path here is 9 hashes",
            ),
            "a path as a list": (
                {"block_path": list(proof.block_path)},
                "block path here is 9 hashes",
            ),
            "tiles as a list": (
                {"tile_frames": list(proof.tile_frames)},
                "must be TsdfProofFrameTiles",
            ),
            "tiles that are not tiles": (
                {"tile_frames": (tiles.tiles,)},
                "must be TsdfProofFrameTiles",
            ),
            "tile frames out of order": (
                {"tile_frames": proof.tile_frames[::-1]},
                "increasing frame order",
            ),
            "tiles of a frame it does not hold": (
                {"tile_frames": (replace(tiles, sequence=20),)},
                "a frame it does not hold",
            ),
            "tiles cut on another grid": (
                {"tile_pixels": 8},
                "not cut on its tile grid",
            ),
        }
        for name, (changes, message) in cases.items():
            with self.subTest(case=name):
                with self.assertRaisesRegex(TsdfError, message):
                    replace(proof, **changes)
        frame_cases = {
            "a place before the first": ({"sequence": -1}, "invalid frame"),
            "a place that is a flag": ({"sequence": True}, "invalid frame"),
            "samples and no grid": (
                {"tiles": tiles.tiles.tiles},
                "must be TsdfDepthTiles",
            ),
            "a path too few": (
                {"paths": tiles.paths[:-1]},
                "one path a tile",
            ),
            "paths as a list": ({"paths": list(tiles.paths)}, "one path a"),
            "a path one step long": (
                {"paths": (tiles.paths[0] + (bytes(32),),) + tiles.paths[1:]},
                r"tile path here is \d hashes",
            ),
            "a path of short hashes": (
                {
                    "paths": (
                        tuple(step[:31] for step in tiles.paths[0]),
                    )
                    + tiles.paths[1:]
                },
                r"tile path here is \d hashes",
            ),
        }
        for name, (changes, message) in frame_cases.items():
            with self.subTest(case=name):
                with self.assertRaisesRegex(TsdfError, message):
                    replace(tiles, **changes)
        self.assertEqual(tiles.tile_count, len(tiles.paths))
        self.assertEqual(
            tiles.path_hash_count, sum(len(path) for path in tiles.paths)
        )
        self.assertIsInstance(proof.frames[0], TsdfFrameCommitment)


class ProverTests(unittest.TestCase):
    def test_a_proof_is_made_only_under_a_statement_of_these_two(
        self,
    ) -> None:
        block = WORK.volume.block_indices[0]
        with self.assertRaisesRegex(
            TsdfError, "not a statement of this volume: map root"
        ):
            prove_tsdf_block(WORK.statement, WORK.forged, WORK.session, block)
        with self.assertRaisesRegex(
            TsdfError,
            "not a statement of this volume: session, voxel size, "
            "truncation, block count, map root",
        ):
            prove_tsdf_block(
                WORK.room_statement, WORK.volume, WORK.session, block
            )
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            session_path = copied(FIXTURE, temporary)
            write_pgm(
                session_path / "data" / "depth" / "000001.pgm",
                2, 2, [950, 950, 950, 951],
            )
            with self.assertRaisesRegex(
                TsdfError,
                "not a statement of this volume and this scan: scan root",
            ):
                prove_tsdf_block(
                    WORK.statement,
                    WORK.volume,
                    load_scan_session(session_path),
                    block,
                )
        with self.assertRaisesRegex(TsdfError, "its own scan"):
            prove_tsdf_block(
                WORK.statement, WORK.volume, WORK.room.session, block
            )

    def test_only_a_block_the_map_holds_can_be_proved(self) -> None:
        with self.assertRaisesRegex(
            TsdfError,
            r"the volume holds no block \(9, 9, 9\); only a block the map "
            "holds can be proved",
        ):
            prove_tsdf_block(
                WORK.statement, WORK.volume, WORK.session, (9, 9, 9)
            )
        with self.assertRaisesRegex(TsdfError, "at least one block"):
            prove_tsdf_blocks(WORK.statement, WORK.volume, WORK.session, ())
        for blocks in ("0,0,0", b"000", None, 7):
            with self.subTest(blocks=blocks):
                with self.assertRaisesRegex(TsdfError, "by block index"):
                    prove_tsdf_blocks(
                        WORK.statement, WORK.volume, WORK.session, blocks
                    )
        with self.assertRaisesRegex(TsdfError, "three integers"):
            prove_tsdf_blocks(
                WORK.statement, WORK.volume, WORK.session, [(0, 0)]
            )
        with self.assertRaisesRegex(TsdfError, "TsdfMapStatement"):
            prove_tsdf_block(
                WORK.statement_path, WORK.volume, WORK.session, (0, 0, 0)
            )
        with self.assertRaisesRegex(TsdfError, "TsdfBlockVolume"):
            prove_tsdf_block(
                WORK.statement, WORK.volume_path, WORK.session, (0, 0, 0)
            )
        with self.assertRaisesRegex(TsdfError, "loaded ScanSession"):
            prove_tsdf_block(
                WORK.statement, WORK.volume, FIXTURE, (0, 0, 0)
            )

    def test_a_scan_that_changes_between_its_two_readings_is_refused(
        self,
    ) -> None:
        # Proving reads the scan once for the tiles and once for the
        # trees their paths come from. Here the first reading is of a
        # scan with one depth sample different, as if the session had
        # been rewritten in between.
        real_gather = tsdf_block_proof.gather_tsdf_blocks_evidence
        reading = [
            proof.block_index
            for proof in WORK.room_proofs
            if any(
                tiles.sequence == 7
                and (1, 2) in [tile[:2] for tile in tiles.tiles.tiles]
                for tiles in proof.tile_frames
            )
        ]
        elsewhere = next(
            proof
            for proof in WORK.room_proofs
            if proof.tile_frames
            and all(tiles.sequence != 7 for tiles in proof.tile_frames)
        )
        self.assertGreater(len(reading), 10)
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            session_path = copied(WORK.room.session_path, temporary)
            change_one_sample(session_path)
            earlier = load_scan_session(session_path)

            def gather(session, *arguments, **options):
                return real_gather(earlier, *arguments, **options)

            with patch(
                "spatialforge.tsdf_block_proof.gather_tsdf_blocks_evidence",
                gather,
            ):
                for block in reading[::5]:
                    with self.subTest(block=block):
                        with self.assertRaisesRegex(
                            TsdfError,
                            "session inputs changed while the block was "
                            "proved; rerun the command",
                        ):
                            prove_tsdf_block(
                                WORK.room_statement,
                                WORK.room_volume,
                                WORK.room.session,
                                block,
                            )
                # A block that reads nothing of the changed tile is
                # proved as it always is.
                self.assertEqual(
                    prove_tsdf_block(
                        WORK.room_statement,
                        WORK.room_volume,
                        WORK.room.session,
                        elsewhere.block_index,
                    ).encoded(),
                    elsewhere.encoded(),
                )

        # Or the second reading finds a frame's depth gone.
        real_commit = tsdf_block_proof._commit_tsdf_scan
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            session_path = copied(FIXTURE, temporary)
            without_first_line(session_path, "streams/depth.jsonl")
            later = load_scan_session(session_path)

            def commit(session, *arguments):
                return real_commit(later, *arguments)

            with patch(
                "spatialforge.tsdf_block_proof._commit_tsdf_scan", commit
            ):
                with self.assertRaisesRegex(
                    TsdfError, "changed while the block was proved"
                ):
                    prove_tsdf_block(
                        WORK.statement, WORK.volume, WORK.session, (0, 0, 0)
                    )

    def test_proving_writes_nothing(self) -> None:
        before = tree_snapshot(WORK.root)
        session_before = tree_snapshot(WORK.room.session_path)
        prove_tsdf_blocks(
            WORK.room_statement,
            WORK.room_volume,
            WORK.room.session,
            WORK.room_volume.block_indices[::70],
        )
        self.assertEqual(tree_snapshot(WORK.root), before)
        self.assertEqual(
            tree_snapshot(WORK.room.session_path), session_before
        )


_GONE = object()


class BlockProofCliTests(unittest.TestCase):
    def run_cli(self, *arguments: str) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                code = main(["reconstruct", *arguments])
            except SystemExit as stop:
                code = stop.code
        return code, stdout.getvalue(), stderr.getvalue()

    def shape(self, row: int, block: str) -> str:
        return (
            f"block: {block} row={row} of=8\n"
            "proof_holds: frames=2 frames_with_tiles=2 tiles=2 "
            "tile_pixels=16 depth_bytes=16 tile_path_hashes=0 "
            "block_path_hashes=3\n"
        )

    def test_cli_proves_a_block_and_checks_it_with_nothing_else(
        self,
    ) -> None:
        before = tree_snapshot(WORK.root)
        digest = FIXTURE_PROOFS[3][1]
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            output = Path(temporary) / "block.sftproof"
            arguments = (
                "tsdf-block-prove",
                str(WORK.statement_path), str(WORK.volume_path),
                str(FIXTURE), str(output), "--block", "1", "0", "-1",
            )
            code, printed, errors = self.run_cli(*arguments)
            self.assertEqual(errors, "")
            self.assertEqual(code, 0)
            self.assertEqual(
                printed,
                "TSDF BLOCK PROVE scan-synthetic-0001\n"
                "statement: valid\n"
                f"statement_sha256: {FIXTURE_STATEMENT_SHA256}\n"
                "volume: verified\n"
                "statement_recomputed: matched\n"
                + self.shape(3, "1,0,-1")
                + "proof_bytes: 7009\n"
                f"proof_sha256: {digest}\n"
                f"output: {output.resolve()}\n"
                "statement_modified: no\n"
                "volume_modified: no\n"
                "session_modified: no\n",
            )
            self.assertEqual(
                hashlib.sha256(output.read_bytes()).hexdigest(), digest
            )

            # A second time, the output is in the way and stays.
            code, printed, errors = self.run_cli(*arguments)
            self.assertEqual((code, printed), (2, ""))
            self.assertEqual(
                errors,
                f"TSDF BLOCK PROVE FAILED {WORK.statement_path}\n"
                f"- output already exists: {output.resolve()}\n",
            )

            # The check is run where there is a statement and a proof
            # and nothing else, and with both loaders it has no use for
            # taken away.
            alone = Path(temporary) / "alone"
            alone.mkdir()
            statement = alone / "map.sftcommit"
            proof = alone / "block.sftproof"
            shutil.copyfile(WORK.statement_path, statement)
            shutil.copyfile(output, proof)
            with forbidden_calls(
                (
                    "spatialforge.cli.load_scan_session",
                    "spatialforge.cli.load_tsdf_block_volume",
                )
            ):
                code, printed, errors = self.run_cli(
                    "tsdf-block-proof-check", str(statement), str(proof)
                )
            self.assertEqual(errors, "")
            self.assertEqual(code, 0)
            self.assertEqual(
                printed,
                "TSDF BLOCK PROOF CHECK scan-synthetic-0001\n"
                "statement: valid\n"
                f"statement_sha256: {FIXTURE_STATEMENT_SHA256}\n"
                "proof: well formed\n"
                f"proof_sha256: {digest}\n"
                + self.shape(3, "1,0,-1")
                + "check statement: passed\n"
                "check shape: passed\n"
                "check scan root: passed\n"
                "check map root: passed\n"
                "check depth roots: passed\n"
                "check poses: passed\n"
                "check pixels: passed\n"
                "check fusion: passed\n"
                "fusion: selected=2 ready=2 ruled_out_by_pose=0 "
                "evaluated=2 contributing=2 differing_voxels=0\n"
                "session_read: no\n"
                "volume_read: no\n"
                "statement_modified: no\n"
                "proof_modified: no\n"
                "verdict: verified\n"
                "result: the block is the map's, and it is the block the "
                "scan gives\n",
            )
            self.assertEqual(
                sorted(path.name for path in alone.iterdir()),
                ["block.sftproof", "map.sftcommit"],
            )
        self.assertEqual(tree_snapshot(WORK.root), before)

    def test_cli_refutes_a_block_of_an_altered_map(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            statement = Path(temporary) / "forged.sftcommit"
            proof = Path(temporary) / "block.sftproof"
            code, _, errors = self.run_cli(
                "tsdf-map-commit",
                str(WORK.forged_path), str(FIXTURE), str(statement),
            )
            self.assertEqual((code, errors), (0, ""))
            code, _, errors = self.run_cli(
                "tsdf-block-prove",
                str(statement), str(WORK.forged_path), str(FIXTURE),
                str(proof), "--block", "0", "-1", "-1",
            )
            self.assertEqual((code, errors), (0, ""))
            digest = hashlib.sha256(statement.read_bytes()).hexdigest()
            proof_digest = hashlib.sha256(proof.read_bytes()).hexdigest()
            code, printed, errors = self.run_cli(
                "tsdf-block-proof-check", str(statement), str(proof)
            )
            self.assertEqual(errors, "")
            self.assertEqual(code, 1)
            self.assertEqual(
                printed,
                "TSDF BLOCK PROOF CHECK scan-synthetic-0001\n"
                "statement: valid\n"
                f"statement_sha256: {digest}\n"
                "proof: well formed\n"
                f"proof_sha256: {proof_digest}\n"
                + self.shape(0, "0,-1,-1")
                + "check statement: passed\n"
                "check shape: passed\n"
                "check scan root: passed\n"
                "check map root: passed\n"
                "check depth roots: passed\n"
                "check poses: passed\n"
                "check pixels: passed\n"
                "check fusion: failed\n"
                "detail: 1 of 512 voxels are not what the scan gives\n"
                "fusion: selected=2 ready=2 ruled_out_by_pose=0 "
                "evaluated=2 contributing=2 differing_voxels=1\n"
                "session_read: no\n"
                "volume_read: no\n"
                "statement_modified: no\n"
                "proof_modified: no\n"
                "verdict: refuted\n"
                "result: the block is the map's, and it is not the block "
                "the scan gives\n",
            )
            # The same proof against the true map's statement shows
            # nothing.
            code, printed, errors = self.run_cli(
                "tsdf-block-proof-check",
                str(WORK.statement_path), str(proof),
            )
            self.assertEqual((code, errors), (1, ""))
            self.assertEqual(
                printed.splitlines()[7:],
                [
                    "check statement: failed",
                    "check shape: not reached",
                    "check scan root: not reached",
                    "check map root: not reached",
                    "check depth roots: not reached",
                    "check poses: not reached",
                    "check pixels: not reached",
                    "check fusion: not reached",
                    f"detail: the proof was made under statement {digest}",
                    "session_read: no",
                    "volume_read: no",
                    "statement_modified: no",
                    "proof_modified: no",
                    "verdict: invalid",
                    "result: the proof shows nothing about the map; it "
                    "failed the check of its statement",
                ],
            )

    def test_cli_names_the_check_an_altered_proof_fails(self) -> None:
        proof = WORK.proofs[3]
        flipped = proof.tsdf_sums.copy()
        flipped.reshape(-1).view(np.uint64)[
            int(np.flatnonzero(proof.weights.reshape(-1))[0])
        ] ^= 1
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            path = Path(temporary) / "altered.sftproof"
            write_tsdf_block_proof(
                replace(proof, tsdf_sums=frozen(flipped)), path
            )
            code, printed, errors = self.run_cli(
                "tsdf-block-proof-check", str(WORK.statement_path), str(path)
            )
        self.assertEqual((code, errors), (1, ""))
        lines = printed.splitlines()
        self.assertEqual(
            lines[7:15],
            [
                "check statement: passed",
                "check shape: passed",
                "check scan root: passed",
                "check map root: failed",
                "check depth roots: not reached",
                "check poses: not reached",
                "check pixels: not reached",
                "check fusion: not reached",
            ],
        )
        self.assertRegex(
            lines[15], "^detail: the block and its path lead to [0-9a-f]{64}$"
        )
        self.assertEqual(
            lines[-2:],
            [
                "verdict: invalid",
                "result: the proof shows nothing about the map; it failed "
                "the check of its map root",
            ],
        )

    def test_cli_refuses_what_it_cannot_prove_or_read(self) -> None:
        before = tree_snapshot(WORK.root)
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            folder = Path(temporary)
            output = folder / "block.sftproof"
            proves = (
                (
                    "a block the map does not hold",
                    (str(WORK.statement_path), str(WORK.volume_path),
                     str(FIXTURE), str(output), "--block", "9", "9", "9"),
                    "- the volume holds no block (9, 9, 9); only a block "
                    "the map holds can be proved\n",
                ),
                (
                    "another map",
                    (str(WORK.statement_path), str(WORK.forged_path),
                     str(FIXTURE), str(output), "--block", "0", "0", "0"),
                    "- the statement is not a statement of this volume: "
                    "map root\n",
                ),
                (
                    "an output of another kind",
                    (str(WORK.statement_path), str(WORK.volume_path),
                     str(FIXTURE), str(folder / "block.bin"),
                     "--block", "0", "0", "0"),
                    "- output filename must end in .sftproof\n",
                ),
            )
            for name, arguments, message in proves:
                with self.subTest(prove=name):
                    code, printed, errors = self.run_cli(
                        "tsdf-block-prove", *arguments
                    )
                    self.assertEqual((code, printed), (2, ""))
                    self.assertEqual(
                        errors,
                        f"TSDF BLOCK PROVE FAILED {WORK.statement_path}\n"
                        + message,
                    )
            code, printed, errors = self.run_cli(
                "tsdf-block-prove",
                str(WORK.statement_path), str(WORK.volume_path),
                str(FIXTURE), str(output),
            )
            self.assertEqual((code, printed), (2, ""))
            self.assertIn("--block", errors)
            self.assertEqual(list(folder.iterdir()), [])

            encoded = WORK.proofs[0].encoded()
            (folder / "cut.sftproof").write_bytes(encoded[:-1])
            (folder / "long.sftproof").write_bytes(encoded + b"\x00")
            checks = (
                (
                    "a proof cut short",
                    WORK.statement_path,
                    folder / "cut.sftproof",
                    "- TSDF block proof is truncated inside a tile's "
                    "samples\n",
                ),
                (
                    "a proof with a byte to spare",
                    WORK.statement_path,
                    folder / "long.sftproof",
                    "- TSDF block proof holds 1 bytes after its end\n",
                ),
                (
                    "a statement given as the proof",
                    WORK.statement_path,
                    WORK.statement_path,
                    "- proof filename must end in .sftproof\n",
                ),
                (
                    "a proof given as the statement",
                    folder / "cut.sftproof",
                    folder / "cut.sftproof",
                    "- statement filename must end in .sftcommit\n",
                ),
            )
            for name, statement, proof, message in checks:
                with self.subTest(check=name):
                    code, printed, errors = self.run_cli(
                        "tsdf-block-proof-check", str(statement), str(proof)
                    )
                    self.assertEqual((code, printed), (2, ""))
                    self.assertEqual(
                        errors,
                        f"TSDF BLOCK PROOF CHECK FAILED {proof}\n" + message,
                    )
            code, printed, errors = self.run_cli(
                "tsdf-block-proof-check",
                str(WORK.statement_path), str(folder / "gone.sftproof"),
            )
            self.assertEqual((code, printed), (2, ""))
            self.assertTrue(
                errors.startswith(
                    "TSDF BLOCK PROOF CHECK FAILED "
                    f"{folder / 'gone.sftproof'}\n"
                    "- cannot read TSDF block proof: "
                )
            )
        self.assertEqual(tree_snapshot(WORK.root), before)


if __name__ == "__main__":
    unittest.main()
