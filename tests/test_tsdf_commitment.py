"""Two roots and a statement, held to a recomputation by hand.

A commitment is only worth something if a stranger can recompute it. So
both roots are worked out here a second time from the files themselves:
the scan's from the session's text indexes and its ASCII depth images,
with no session loader, no array and no tree but the RFC's, written out
recursively in ``test_merkle``; the map's from the bytes of the volume
file, taken apart by hand.

The committed fixture's roots and its whole statement are then written
out as literals, which is what holds every platform to the same bytes.
The generated room is held to the recomputation only.

What a root must and must not depend on is tested by changing one thing
at a time: one depth sample, one pose, the encoding of an image, which
frames have depth, one voxel of the map, which block holds which
accumulators.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import struct
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from spatialforge import (
    MAX_TSDF_DEPTH_TILE_PIXELS,
    MAX_TSDF_MAP_STATEMENT_BYTES,
    TSDF_DEPTH_TILE_PIXELS,
    TSDF_FUSION_RULE,
    TsdfFrameCommitment,
    TsdfMapStatement,
    TsdfPinholeCamera,
    TsdfScanCommitment,
    TsdfVolumeCommitment,
    check_tsdf_map_statement,
    commit_tsdf_scan,
    commit_tsdf_volume,
    gather_tsdf_blocks_evidence,
    load_tsdf_block_plan,
    load_tsdf_block_volume,
    load_tsdf_map_statement,
    state_tsdf_map,
    tsdf_depth_tile_leaf_hash,
    tsdf_depth_tile_tree,
    tsdf_volume_block_leaf_hash,
    tsdf_volume_merkle_tree,
    write_tsdf_block_volume,
    write_tsdf_map_statement,
)
from spatialforge import tsdf_commitment
from spatialforge.cli import main
from spatialforge.errors import TsdfError
from spatialforge.merkle import merkle_root_from_path
from spatialforge.replay import replay_session
from spatialforge.session_loader import load_scan_session
from spatialforge.tsdf_block_plan import plan_tsdf_blocks

from tests.heavy_fixtures import shared_room_case
from tests.test_merkle import rfc_path, rfc_root
from tests.test_tsdf_block_evidence import (
    forbidden_calls,
    fused,
    options,
    tree_snapshot,
)
from tests.test_tsdf_block_volume import assemble, split

TEST_ROOT = Path(__file__).resolve().parent
FIXTURE = TEST_ROOT / "fixtures" / "minimal.vgsession"
PLAN_ARGUMENTS = {"voxel_size_m": 0.125, "truncation_m": 0.5}
REPLAY_SHA256 = (
    "dc001ae0ca01004a21ad227b12d57a7350bbc23904040453ca73fd955ef050b8"
)
RULE = "projective-tsdf/unit-weight/nearest-pixel/1"

# The committed fixture, committed to. Its image is two pixels square, so
# at sixteen pixels to a tile each frame is one tile of four samples.
FIXTURE_POSES = (
    (0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0,
     0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    (0.0, 0.0, 1.0, 0.05, -1.0, 0.0, 0.0, 0.0,
     0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
)
FIXTURE_DEPTH_ROOTS = (
    "a9b155df38281863776164cec59c3724553097a2033f9cf9ddbe5f80cba4869d",
    "36507737a81427c9778dab57c16f1aa47f3e832172a165c99908484ffda9917d",
)
FIXTURE_FRAME_LEAVES = (
    "07c65dd976bade19bb93d36ae5e832b5614e17f2906e88627e11cec285c4434a",
    "f7d7ef28cf4b2f84cab8b7ace2698edd606a38e2f04142db597045a5b4213d88",
)
FIXTURE_SCAN_ROOT = (
    "c5740130e75365c7dd3489543e3a8e70a93c798cf835e44c84b7774ce9906248"
)
# The same scan at one pixel to a tile: four leaves a frame.
FIXTURE_DEPTH_ROOTS_BY_PIXEL = (
    "c71a354572d0e59fe9612054cf3cdbbb5db9242bd39a1cf8b55d12562b1e7f59",
    "50bcf7d87945e119daa65cd64b005526aeb13e36ae95a1d415ed187339da3468",
)
FIXTURE_SCAN_ROOT_BY_PIXEL = (
    "82f8e3552f510a68fbba18f1c2fa1442596ac2aa395a7b2d6ee87861a113831e"
)
FIXTURE_MAP_ROOT = (
    "2132eeb769f580eb6a273fce5e32fdabfee8a62a2d70b156678530c5422852f0"
)
FORGED_MAP_ROOT = (
    "ba86185fcbabf926b2c51e5ec38f850a8e49d530451fdd88bae7b4e2779069e4"
)
FIXTURE_STATEMENT_SHA256 = (
    "7b771f1ef5ff5249f48cf29affdbce16df2349f02b67cf7d9accf4884414e780"
)
FIXTURE_STATEMENT = "\n".join(
    (
        "{",
        '  "hash": "sha-256",',
        '  "map": {',
        '    "block_resolution": 8,',
        '    "blocks": 8,',
        '    "frame_stride": 1,',
        f'    "root_sha256": "{FIXTURE_MAP_ROOT}",',
        '    "truncation_m": 0.5,',
        '    "voxel_size_m": 0.125',
        "  },",
        '  "rule": "projective-tsdf/unit-weight/nearest-pixel/1",',
        '  "scan": {',
        '    "camera": {',
        '      "cx": 0.5,',
        '      "cy": 0.5,',
        '      "fx": 2.0,',
        '      "fy": 2.0,',
        '      "height": 2,',
        '      "width": 2',
        "    },",
        '    "depth_sample": "uint16-le",',
        '    "depth_scale_m": 0.001,',
        '    "frames": 2,',
        f'    "root_sha256": "{FIXTURE_SCAN_ROOT}",',
        '    "tile_pixels": 16',
        "  },",
        '  "schema": "spatialforge.tsdf-map-statement",',
        '  "schema_version": "0.1.0",',
        '  "session_id": "scan-synthetic-0001",',
        '  "tree": "rfc6962"',
        "}",
        "",
    )
).encode("ascii")

# Committing hashes what is there. It fuses nothing.
_FUSION = (
    "spatialforge.tsdf_stream_fusion._classify_blocks",
    "spatialforge.tsdf_block_contributions._evaluate_ready_voxels_reading",
    "spatialforge.tsdf_lone_block._prepare_lone_fusion",
)

WORK: SimpleNamespace | None = None


def pgm(path: Path) -> tuple[int, int, list[int]]:
    """An ASCII depth image's width, height and samples, row by row."""

    tokens = path.read_text(encoding="ascii").split()
    if tokens[0] != "P2" or tokens[3] != "65535":
        raise AssertionError(f"{path.name} is not a 16-bit ASCII PGM")
    width, height = int(tokens[1]), int(tokens[2])
    samples = [int(token) for token in tokens[4:]]
    if len(samples) != width * height:
        raise AssertionError(f"{path.name} is not {width} by {height}")
    return width, height, samples


def write_pgm(path: Path, width: int, height: int, samples: list[int]):
    rows = (
        " ".join(str(sample) for sample in samples[top:top + width])
        for top in range(0, width * height, width)
    )
    path.write_text(
        f"P2\n{width} {height}\n65535\n" + "\n".join(rows) + "\n",
        encoding="ascii",
    )


def raw_depth(path: Path) -> np.ndarray:
    width, height, samples = pgm(path)
    return np.array(samples, dtype=np.uint16).reshape(height, width)


def tile_items_by_hand(
    width: int, height: int, samples: list[int], tile: int
) -> list[bytes]:
    """What each tile's leaf is the hash of, tiles row by row."""

    items = []
    for top in range(0, height, tile):
        for left in range(0, width, tile):
            cut = [
                samples[row * width + column]
                for row in range(top, min(top + tile, height))
                for column in range(left, min(left + tile, width))
            ]
            items.append(b"SFTILE01" + struct.pack(f"<{len(cut)}H", *cut))
    return items


def frame_items_by_hand(session_path: Path, tile: int) -> list[bytes]:
    """What each frame's leaf is the hash of, from the session's files.

    An RGB frame is a frame; its depth and its pose are the ones at its
    timestamp, if there are any. Nothing here is shared with the code
    under test but the layout it is meant to follow.
    """

    def index(name: str) -> list[dict]:
        text = (session_path / "streams" / name).read_text(encoding="utf-8")
        return [json.loads(line) for line in text.splitlines() if line]

    depth_at = {
        entry["timestamp_ns"]: entry["path"] for entry in index("depth.jsonl")
    }
    pose_at = {
        entry["timestamp_ns"]: entry["T_world_camera"]
        for entry in index("poses.jsonl")
    }
    items = []
    for sequence, rgb in enumerate(index("rgb.jsonl")):
        stamp = rgb["timestamp_ns"]
        flags = 0
        pose = bytes(16 * 8)
        depth_root = bytes(32)
        if stamp in depth_at:
            flags |= 1
            depth_root = rfc_root(
                tile_items_by_hand(*pgm(session_path / depth_at[stamp]), tile)
            )
        if stamp in pose_at:
            flags |= 2
            pose = struct.pack(
                "<16d", *(float(value) for value in pose_at[stamp])
            )
        items.append(
            b"SFFRAME1"
            + struct.pack("<Q", sequence)
            + bytes([flags])
            + pose
            + depth_root
        )
    return items


def block_items_by_hand(volume_bytes: bytes) -> list[bytes]:
    """What each block's leaf is the hash of, from a volume file's bytes."""

    header, indices, sums, weights = split(volume_bytes)
    return [
        b"SFBLOCK1"
        + indices[3 * row:3 * row + 3].astype("<i4").tobytes()
        + sums[512 * row:512 * (row + 1)].astype("<f8").tobytes()
        + weights[512 * row:512 * (row + 1)].astype("<u4").tobytes()
        for row in range(header["payload"]["block_count"])
    ]


def canonical(document: dict) -> bytes:
    return (
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def copied(session_path: Path, temporary: str) -> Path:
    target = Path(temporary) / "case.vgsession"
    shutil.copytree(session_path, target)
    return target


def without_first_line(session_path: Path, *references: str) -> None:
    for reference in references:
        path = session_path / reference
        lines = path.read_text(encoding="utf-8").splitlines()
        path.write_text(
            "".join(line + "\n" for line in lines[1:]), encoding="utf-8"
        )


def setUpModule() -> None:
    global WORK
    root = Path(tempfile.mkdtemp(dir=TEST_ROOT))
    unittest.addModuleCleanup(shutil.rmtree, root, True)

    plan_path = root / "fixture.sftplan"
    plan_tsdf_blocks(load_scan_session(FIXTURE), plan_path, **PLAN_ARGUMENTS)
    session = load_scan_session(FIXTURE)
    storage, receipt = fused(load_tsdf_block_plan(plan_path), session)
    volume_path = root / "fixture.sftvol"
    write_tsdf_block_volume(storage, receipt, volume_path)
    header, indices, sums, weights = split(volume_path.read_bytes())
    target = int(np.flatnonzero(weights)[0])
    sums[target] = -sums[target]
    forged_path = root / "forged.sftvol"
    forged_path.write_bytes(assemble(header, indices, sums, weights))

    room = shared_room_case()
    room_storage, room_receipt = fused(room.plan, room.session)
    room_volume_path = root / "room.sftvol"
    write_tsdf_block_volume(room_storage, room_receipt, room_volume_path)

    volume = load_tsdf_block_volume(volume_path)
    statement = state_tsdf_map(volume, session)
    statement_path = root / "fixture.sftcommit"
    write_tsdf_map_statement(statement, statement_path)

    WORK = SimpleNamespace(
        root=root,
        session=session,
        volume_path=volume_path,
        volume=volume,
        forged_path=forged_path,
        forged=load_tsdf_block_volume(forged_path),
        room=room,
        room_volume_path=room_volume_path,
        room_volume=load_tsdf_block_volume(room_volume_path),
        statement=statement,
        statement_path=statement_path,
    )


class ScanCommitmentTests(unittest.TestCase):
    def test_the_fixture_scan_written_out(self) -> None:
        scan = commit_tsdf_scan(WORK.session)
        self.assertEqual(scan.session_id, "scan-synthetic-0001")
        self.assertEqual(
            scan.camera, TsdfPinholeCamera(2, 2, 2.0, 2.0, 0.5, 0.5)
        )
        self.assertEqual(scan.depth_scale_m, 0.001)
        self.assertEqual(scan.tile_pixels, TSDF_DEPTH_TILE_PIXELS)
        self.assertEqual(scan.tile_pixels, 16)
        self.assertEqual(scan.frame_count, 2)
        for frame, pose, raw, depth_root, leaf in zip(
            scan.frames,
            FIXTURE_POSES,
            (1000, 950),
            FIXTURE_DEPTH_ROOTS,
            FIXTURE_FRAME_LEAVES,
            strict=True,
        ):
            with self.subTest(frame=frame.sequence):
                self.assertEqual(frame.transform, pose)
                self.assertTrue(frame.has_depth)
                # One tile: a tag and four samples, hashed as a leaf.
                # A tree of one leaf is that leaf.
                self.assertEqual(
                    frame.depth_root,
                    hashlib.sha256(
                        b"\x00SFTILE01" + struct.pack("<4H", *[raw] * 4)
                    ).digest(),
                )
                self.assertEqual(frame.depth_root.hex(), depth_root)
                # A frame's leaf: tag, place, flags, pose, depth root.
                data = (
                    b"SFFRAME1"
                    + frame.sequence.to_bytes(8, "little")
                    + b"\x03"
                    + struct.pack("<16d", *pose)
                    + bytes.fromhex(depth_root)
                )
                self.assertEqual(len(data), 177)
                self.assertEqual(frame.leaf_data(), data)
                self.assertEqual(
                    frame.leaf_hash(),
                    hashlib.sha256(b"\x00" + data).digest(),
                )
                self.assertEqual(frame.leaf_hash().hex(), leaf)
        self.assertEqual(
            scan.root,
            hashlib.sha256(
                b"\x01"
                + bytes.fromhex(FIXTURE_FRAME_LEAVES[0])
                + bytes.fromhex(FIXTURE_FRAME_LEAVES[1])
            ).digest(),
        )
        self.assertEqual(scan.root_sha256, FIXTURE_SCAN_ROOT)
        self.assertEqual(scan.frame_table_bytes, 354)

    def test_the_fixture_scan_at_one_pixel_to_a_tile(self) -> None:
        scan = commit_tsdf_scan(WORK.session, tile_pixels=1)
        self.assertEqual(scan.tile_pixels, 1)
        self.assertEqual(
            tuple(frame.depth_root.hex() for frame in scan.frames),
            FIXTURE_DEPTH_ROOTS_BY_PIXEL,
        )
        self.assertEqual(scan.root_sha256, FIXTURE_SCAN_ROOT_BY_PIXEL)
        self.assertEqual(
            tuple(frame.transform for frame in scan.frames), FIXTURE_POSES
        )

    def test_a_tile_larger_than_the_image_is_the_whole_image(self) -> None:
        # Two pixels to a tile already takes the whole fixture image, so
        # the scan's root is the same from there up. What tells such
        # statements apart is the tile size the statement names.
        for tile in (2, 3, 16, MAX_TSDF_DEPTH_TILE_PIXELS):
            with self.subTest(tile=tile):
                self.assertEqual(
                    commit_tsdf_scan(WORK.session, tile_pixels=tile)
                    .root_sha256,
                    FIXTURE_SCAN_ROOT,
                )
        self.assertNotEqual(
            state_tsdf_map(WORK.volume, WORK.session, tile_pixels=2)
            .digest_sha256,
            FIXTURE_STATEMENT_SHA256,
        )

    def test_the_scan_root_is_the_root_by_hand_at_every_tile_size(
        self,
    ) -> None:
        cases = (
            ("fixture", FIXTURE, WORK.session, (1, 2, 16)),
            (
                "room",
                WORK.room.session_path,
                WORK.room.session,
                (1, 5, 8, 16, 48, 64, 1024),
            ),
        )
        roots = set()
        for name, session_path, session, tiles in cases:
            for tile in tiles:
                with self.subTest(scan=name, tile=tile):
                    scan = commit_tsdf_scan(session, tile_pixels=tile)
                    items = frame_items_by_hand(session_path, tile)
                    self.assertEqual(
                        [frame.leaf_data() for frame in scan.frames], items
                    )
                    self.assertEqual(scan.root, rfc_root(items))
                    self.assertEqual(
                        scan.frame_table_bytes, 177 * len(items)
                    )
                    roots.add((name, scan.root))
        # The fixture is one tile from two pixels up; the room is one
        # tile only from sixty-four.
        self.assertEqual(len(roots), 2 + 6)

    def test_a_frames_tiles_are_leaves_row_by_row_with_ragged_edges(
        self,
    ) -> None:
        raw = (np.arange(35, dtype=np.int64).reshape(5, 7) * 1873) % 65536
        tree = tsdf_depth_tile_tree(raw, 3)
        self.assertEqual(tree.size, 2 * 3)
        items = tile_items_by_hand(7, 5, [int(v) for v in raw.ravel()], 3)
        self.assertEqual(
            [len(item) for item in items],
            [8 + 2 * count for count in (9, 9, 3, 6, 6, 2)],
        )
        self.assertEqual(
            [tree.leaf(index) for index in range(6)],
            [hashlib.sha256(b"\x00" + item).digest() for item in items],
        )
        self.assertEqual(tree.root, rfc_root(items))
        for index in range(6):
            self.assertEqual(list(tree.path(index)), rfc_path(index, items))
        # The samples are what is hashed, not how an array holds them.
        for held in (np.uint16, ">u2", np.int32, np.uint64):
            with self.subTest(dtype=held):
                self.assertEqual(
                    tsdf_depth_tile_tree(raw.astype(held), 3).root, tree.root
                )
        self.assertEqual(
            tsdf_depth_tile_tree(
                np.asfortranarray(raw.astype(np.uint16)), 3
            ).root,
            tree.root,
        )
        samples = raw.astype(np.uint16)
        self.assertEqual(
            tsdf_depth_tile_leaf_hash(samples[0:3, 3:6]), tree.leaf(1)
        )
        self.assertEqual(
            tsdf_depth_tile_leaf_hash(samples[3:5, 6:7]), tree.leaf(5)
        )
        # Transposed, a tile holds the same samples in another order.
        self.assertNotEqual(
            tsdf_depth_tile_leaf_hash(samples[0:3, 3:6].T.copy()),
            tree.leaf(1),
        )

    def test_a_tile_gathered_as_evidence_is_a_leaf_under_its_frames_root(
        self,
    ) -> None:
        room = WORK.room
        for tile in (8, 16):
            scan = commit_tsdf_scan(room.session, tile_pixels=tile)
            trees = {
                sequence: tsdf_depth_tile_tree(
                    raw_depth(
                        room.session_path
                        / "data" / "depth" / f"{sequence:06d}.pgm"
                    ),
                    tile,
                )
                for sequence in range(scan.frame_count)
            }
            gathered = gather_tsdf_blocks_evidence(
                room.session,
                room.plan.active_blocks[::50],
                tile_pixels=tile,
                **options(room.plan),
            )
            tied = 0
            for evidence in gathered:
                self.assertEqual(evidence.camera, scan.camera)
                self.assertEqual(evidence.depth_scale_m, scan.depth_scale_m)
                for frame in evidence.frames:
                    committed = scan.frames[frame.sequence]
                    self.assertEqual(frame.transform, committed.transform)
                    if frame.tiles is None:
                        continue
                    tree = trees[frame.sequence]
                    columns = frame.tiles.tile_columns
                    size = frame.tiles.tile_rows * columns
                    self.assertEqual(tree.size, size)
                    for tile_row, tile_column, samples in frame.tiles.tiles:
                        place = tile_row * columns + tile_column
                        self.assertEqual(
                            merkle_root_from_path(
                                tsdf_depth_tile_leaf_hash(samples),
                                place,
                                size,
                                tree.path(place),
                            ),
                            committed.depth_root,
                        )
                        tied += 1
            with self.subTest(tile=tile):
                self.assertGreater(tied, 100)

    def test_one_depth_sample_changes_one_tile_one_frame_and_the_root(
        self,
    ) -> None:
        room = WORK.room
        before = commit_tsdf_scan(room.session)
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            session_path = copied(room.session_path, temporary)
            depth_path = session_path / "data" / "depth" / "000007.pgm"
            old_tree = tsdf_depth_tile_tree(raw_depth(depth_path), 16)
            width, height, samples = pgm(depth_path)
            samples[20 * width + 33] += 1
            write_pgm(depth_path, width, height, samples)
            new_tree = tsdf_depth_tile_tree(raw_depth(depth_path), 16)
            after = commit_tsdf_scan(load_scan_session(session_path))
            items = frame_items_by_hand(session_path, 16)
        self.assertEqual(
            [
                frame.sequence
                for frame, other in zip(
                    before.frames, after.frames, strict=True
                )
                if frame != other
            ],
            [7],
        )
        self.assertEqual(after.frames[7].transform, before.frames[7].transform)
        self.assertEqual(before.frames[7].depth_root, old_tree.root)
        self.assertEqual(after.frames[7].depth_root, new_tree.root)
        self.assertNotEqual(old_tree.root, new_tree.root)
        # Row 20, column 33 is in the second row of tiles and the third
        # column; four tiles to a row.
        self.assertEqual(
            [
                place
                for place in range(old_tree.size)
                if old_tree.leaf(place) != new_tree.leaf(place)
            ],
            [1 * 4 + 2],
        )
        self.assertNotEqual(after.root, before.root)
        self.assertEqual(after.root, rfc_root(items))

    def test_one_pose_changes_one_frame_and_the_root_and_no_depth_root(
        self,
    ) -> None:
        before = commit_tsdf_scan(WORK.session)
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            session_path = copied(FIXTURE, temporary)
            poses = session_path / "streams" / "poses.jsonl"
            text = poses.read_text(encoding="utf-8")
            self.assertEqual(text.count("0.05"), 1)
            poses.write_text(
                text.replace("0.05", "0.05000000000000001"),
                encoding="utf-8",
            )
            after = commit_tsdf_scan(load_scan_session(session_path))
            items = frame_items_by_hand(session_path, 16)
        self.assertEqual(after.frames[0], before.frames[0])
        self.assertEqual(
            after.frames[1].depth_root, before.frames[1].depth_root
        )
        # The next float up from 0.05: the smallest change a pose can
        # have.
        self.assertEqual(
            after.frames[1].transform[3], np.nextafter(0.05, 1.0)
        )
        self.assertNotEqual(after.frames[1].leaf_hash().hex(),
                            FIXTURE_FRAME_LEAVES[1])
        self.assertNotEqual(after.root_sha256, FIXTURE_SCAN_ROOT)
        self.assertEqual(after.root, rfc_root(items))

    def test_two_encodings_of_the_same_depth_are_the_same_scan(self) -> None:
        encodings = {
            "binary": b"P5\n2 2\n65535\n" + struct.pack(">4H", *[1000] * 4),
            "spaced": (
                b"P2\n# the same four samples\n2 2\n65535\n"
                b"1000\n1000\n1000\n1000\n"
            ),
        }
        digests = {REPLAY_SHA256}
        for name, payload in encodings.items():
            with self.subTest(encoding=name):
                with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
                    session_path = copied(FIXTURE, temporary)
                    (
                        session_path / "data" / "depth" / "000000.pgm"
                    ).write_bytes(payload)
                    session = load_scan_session(session_path)
                    scan = commit_tsdf_scan(session)
                    digests.add(replay_session(session).digest_sha256)
                    # The volume was fused from the same samples, and
                    # the statement is as true of this scan as of the
                    # one it was made from.
                    check = check_tsdf_map_statement(
                        WORK.statement, WORK.volume, session
                    )
                    with self.assertRaisesRegex(TsdfError, "replay digest"):
                        state_tsdf_map(WORK.volume, session)
                self.assertEqual(scan.root_sha256, FIXTURE_SCAN_ROOT)
                self.assertEqual(
                    tuple(frame.leaf_hash().hex() for frame in scan.frames),
                    FIXTURE_FRAME_LEAVES,
                )
                self.assertTrue(check.holds)
        # Three sessions the replay digest tells apart, and one scan.
        self.assertEqual(len(digests), 3)

    def test_a_frame_without_depth_or_pose_says_so_in_its_leaf(self) -> None:
        cases = (
            ("depth", ("streams/depth.jsonl",), 2),
            ("pose", ("streams/poses.jsonl",), 1),
            ("both", ("streams/depth.jsonl", "streams/poses.jsonl"), 0),
        )
        roots = {FIXTURE_SCAN_ROOT}
        for name, references, flags in cases:
            with self.subTest(missing=name):
                with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
                    session_path = copied(FIXTURE, temporary)
                    without_first_line(session_path, *references)
                    scan = commit_tsdf_scan(load_scan_session(session_path))
                    items = frame_items_by_hand(session_path, 16)
                first, second = scan.frames
                self.assertEqual(first.has_depth, bool(flags & 1))
                self.assertEqual(first.transform is not None, bool(flags & 2))
                data = first.leaf_data()
                self.assertEqual(len(data), 177)
                self.assertEqual(data[:8], b"SFFRAME1")
                self.assertEqual(data[8:16], bytes(8))
                self.assertEqual(data[16], flags)
                if not flags & 2:
                    self.assertEqual(data[17:145], bytes(128))
                if not flags & 1:
                    self.assertEqual(data[145:], bytes(32))
                self.assertEqual(
                    second.leaf_hash().hex(), FIXTURE_FRAME_LEAVES[1]
                )
                self.assertEqual(
                    [frame.leaf_data() for frame in scan.frames], items
                )
                self.assertEqual(scan.root, rfc_root(items))
                roots.add(scan.root_sha256)
        self.assertEqual(len(roots), 4)

    def test_an_absent_pose_is_not_a_pose_of_zeros(self) -> None:
        depth_root = bytes.fromhex(FIXTURE_DEPTH_ROOTS[0])
        absent = TsdfFrameCommitment(0, None, depth_root)
        zeros = TsdfFrameCommitment(0, (0.0,) * 16, depth_root)
        self.assertEqual(absent.leaf_data()[17:], zeros.leaf_data()[17:])
        self.assertEqual(
            (absent.leaf_data()[16], zeros.leaf_data()[16]), (1, 3)
        )
        self.assertNotEqual(absent.leaf_hash(), zeros.leaf_hash())
        # And an absent image is not an image whose root is zeros.
        self.assertNotEqual(
            TsdfFrameCommitment(0, FIXTURE_POSES[0], None).leaf_hash(),
            TsdfFrameCommitment(0, FIXTURE_POSES[0], bytes(32)).leaf_hash(),
        )

    def test_exchanging_two_frames_depth_changes_both_and_the_root(
        self,
    ) -> None:
        room = WORK.room
        before = commit_tsdf_scan(room.session)
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            session_path = copied(room.session_path, temporary)
            depth = session_path / "data" / "depth"
            third = (depth / "000003.pgm").read_bytes()
            (depth / "000003.pgm").write_bytes(
                (depth / "000004.pgm").read_bytes()
            )
            (depth / "000004.pgm").write_bytes(third)
            after = commit_tsdf_scan(load_scan_session(session_path))
        self.assertEqual(
            [
                frame.sequence
                for frame, other in zip(
                    before.frames, after.frames, strict=True
                )
                if frame != other
            ],
            [3, 4],
        )
        self.assertEqual(after.frames[3].depth_root,
                         before.frames[4].depth_root)
        self.assertEqual(after.frames[4].depth_root,
                         before.frames[3].depth_root)
        self.assertNotEqual(after.root, before.root)

    def test_committing_writes_nothing_and_fuses_nothing(self) -> None:
        room = WORK.room
        before = tree_snapshot(room.session_path)
        with forbidden_calls(_FUSION):
            first = commit_tsdf_scan(room.session)
            second = commit_tsdf_scan(room.session)
            commit_tsdf_volume(WORK.room_volume)
        self.assertEqual(tree_snapshot(room.session_path), before)
        self.assertEqual(first.root, second.root)
        self.assertEqual(first.frames, second.frames)

    def test_a_scan_that_changes_while_it_is_committed_is_refused(
        self,
    ) -> None:
        real = tsdf_commitment.replay_session
        replays = []

        def replay(session):
            result = real(session)
            replays.append(result)
            if len(replays) == 2:
                return SimpleNamespace(
                    observations=result.observations,
                    digest_sha256="0" * 64,
                )
            return result

        with patch("spatialforge.tsdf_commitment.replay_session", replay):
            with self.assertRaisesRegex(
                TsdfError, "changed while the scan was committed"
            ):
                commit_tsdf_scan(WORK.session)
        self.assertEqual(len(replays), 2)

    def test_invalid_scan_inputs_are_refused(self) -> None:
        with self.assertRaisesRegex(TsdfError, "loaded ScanSession"):
            commit_tsdf_scan(FIXTURE)
        for tile in (0, -16, MAX_TSDF_DEPTH_TILE_PIXELS + 1, True, 16.0, "16"):
            with self.subTest(tile=tile):
                with self.assertRaisesRegex(TsdfError, "tile size"):
                    commit_tsdf_scan(WORK.session, tile_pixels=tile)
        good = np.zeros((4, 4), dtype=np.uint16)
        cases = {
            "a list": [[0, 0], [0, 0]],
            "one row": np.zeros(4, dtype=np.uint16),
            "a stack": np.zeros((2, 2, 2), dtype=np.uint16),
            "metres": np.zeros((4, 4), dtype=np.float64),
            "flags": np.zeros((4, 4), dtype=np.bool_),
        }
        for name, depth in cases.items():
            with self.subTest(depth=name):
                with self.assertRaisesRegex(TsdfError, "2-D integer array"):
                    tsdf_depth_tile_tree(depth, 2)
        for sample in (-1, 65536):
            with self.subTest(sample=sample):
                wide = np.zeros((4, 4), dtype=np.int64)
                wide[3, 1] = sample
                with self.assertRaisesRegex(TsdfError, "does not fit"):
                    tsdf_depth_tile_tree(wide, 2)
        with self.assertRaisesRegex(TsdfError, "tile size"):
            tsdf_depth_tile_tree(good, 0)
        for name, tile in (
            ("wide samples", np.zeros((2, 2), dtype=np.int64)),
            ("a row", np.zeros(4, dtype=np.uint16)),
            ("nothing", np.zeros((0, 2), dtype=np.uint16)),
            ("a list", [[0, 0]]),
        ):
            with self.subTest(tile=name):
                with self.assertRaisesRegex(TsdfError, "2-D uint16"):
                    tsdf_depth_tile_leaf_hash(tile)


class VolumeCommitmentTests(unittest.TestCase):
    def test_the_fixture_map_root_and_what_it_is_a_root_of(self) -> None:
        commitment = commit_tsdf_volume(WORK.volume)
        self.assertEqual(
            commitment,
            TsdfVolumeCommitment(
                session_id="scan-synthetic-0001",
                voxel_size_m=0.125,
                truncation_m=0.5,
                frame_stride=1,
                block_count=8,
                root_sha256=FIXTURE_MAP_ROOT,
            ),
        )

    def test_the_map_root_is_the_root_by_hand_from_the_files_bytes(
        self,
    ) -> None:
        cases = (
            ("fixture", WORK.volume_path, WORK.volume, 8),
            ("forged", WORK.forged_path, WORK.forged, 8),
            ("room", WORK.room_volume_path, WORK.room_volume, 351),
        )
        for name, path, volume, blocks in cases:
            with self.subTest(volume=name):
                items = block_items_by_hand(path.read_bytes())
                self.assertEqual(len(items), blocks)
                self.assertEqual(
                    {len(item) for item in items}, {8 + 12 + 6144}
                )
                self.assertEqual(
                    commit_tsdf_volume(volume).root_sha256,
                    rfc_root(items).hex(),
                )
                tree = tsdf_volume_merkle_tree(volume)
                self.assertEqual(tree.size, blocks)
                self.assertEqual(tree.root, rfc_root(items))
                for row in range(0, blocks, 7):
                    leaf = hashlib.sha256(b"\x00" + items[row]).digest()
                    self.assertEqual(tree.leaf(row), leaf)
                    self.assertEqual(
                        tsdf_volume_block_leaf_hash(volume, row), leaf
                    )
                    self.assertEqual(
                        list(tree.path(row)), rfc_path(row, items)
                    )
                    self.assertEqual(
                        merkle_root_from_path(
                            leaf, row, blocks, tree.path(row)
                        ),
                        tree.root,
                    )

    def test_one_voxel_changed_is_one_leaf_changed_and_another_root(
        self,
    ) -> None:
        self.assertEqual(
            commit_tsdf_volume(WORK.forged).root_sha256, FORGED_MAP_ROOT
        )
        true_tree = tsdf_volume_merkle_tree(WORK.volume)
        forged_tree = tsdf_volume_merkle_tree(WORK.forged)
        self.assertEqual(
            [
                WORK.volume.block_indices[row]
                for row in range(8)
                if true_tree.leaf(row) != forged_tree.leaf(row)
            ],
            [(0, -1, -1)],
        )
        # The forged block's own path still leads to the forged root,
        # and not to the true one.
        self.assertEqual(
            merkle_root_from_path(
                forged_tree.leaf(0), 0, 8, forged_tree.path(0)
            ).hex(),
            FORGED_MAP_ROOT,
        )
        self.assertNotEqual(
            merkle_root_from_path(
                forged_tree.leaf(0), 0, 8, true_tree.path(0)
            ).hex(),
            FIXTURE_MAP_ROOT,
        )

    def test_a_block_is_committed_at_its_own_index(self) -> None:
        # The same accumulators under two blocks exchanged: every digest
        # of the payload as a multiset of blocks is unchanged, and the
        # root is not.
        items = block_items_by_hand(WORK.volume_path.read_bytes())
        exchanged = list(items)
        exchanged[2] = items[2][:20] + items[5][20:]
        exchanged[5] = items[5][:20] + items[2][20:]
        self.assertNotEqual(items[2][20:], items[5][20:])
        self.assertNotEqual(rfc_root(exchanged).hex(), FIXTURE_MAP_ROOT)
        # And the index is in the leaf, not only in the leaf's place.
        moved = list(items)
        moved[2] = items[2][:8] + struct.pack("<3i", 9, 9, 9) + items[2][20:]
        self.assertNotEqual(rfc_root(moved).hex(), FIXTURE_MAP_ROOT)
        self.assertEqual(
            [struct.unpack("<3i", item[8:20]) for item in items],
            list(WORK.volume.block_indices),
        )

    def test_invalid_volume_inputs_are_refused(self) -> None:
        for function in (commit_tsdf_volume, tsdf_volume_merkle_tree):
            with self.subTest(function=function.__name__):
                with self.assertRaisesRegex(TsdfError, "TsdfBlockVolume"):
                    function(WORK.volume_path)
        with self.assertRaisesRegex(TsdfError, "TsdfBlockVolume"):
            tsdf_volume_block_leaf_hash(WORK.volume_path, 0)
        for row in (-1, 8, True, 0.0, "0", None):
            with self.subTest(row=row):
                with self.assertRaisesRegex(
                    TsdfError, "a volume of 8 blocks has no block at row"
                ):
                    tsdf_volume_block_leaf_hash(WORK.volume, row)
        good = commit_tsdf_volume(WORK.volume)
        cases = {
            "no session": {"session_id": ""},
            "voxel size of zero": {"voxel_size_m": 0.0},
            "voxel size as an integer": {"voxel_size_m": 1},
            "truncation not finite": {"truncation_m": float("inf")},
            "stride of zero": {"frame_stride": 0},
            "stride as a flag": {"frame_stride": True},
            "no blocks": {"block_count": 0},
            "a root in capitals": {"root_sha256": FIXTURE_MAP_ROOT.upper()},
            "a short root": {"root_sha256": FIXTURE_MAP_ROOT[:-1]},
            "a root as bytes": {
                "root_sha256": bytes.fromhex(FIXTURE_MAP_ROOT)
            },
        }
        for name, changes in cases.items():
            with self.subTest(case=name):
                with self.assertRaisesRegex(TsdfError, "volume commitment"):
                    replace(good, **changes)


class MapStatementTests(unittest.TestCase):
    def test_the_fixture_statement_written_out(self) -> None:
        statement = WORK.statement
        self.assertEqual(
            statement,
            TsdfMapStatement(
                session_id="scan-synthetic-0001",
                rule=RULE,
                frame_count=2,
                camera=TsdfPinholeCamera(2, 2, 2.0, 2.0, 0.5, 0.5),
                depth_scale_m=0.001,
                tile_pixels=16,
                scan_root_sha256=FIXTURE_SCAN_ROOT,
                voxel_size_m=0.125,
                truncation_m=0.5,
                frame_stride=1,
                block_count=8,
                map_root_sha256=FIXTURE_MAP_ROOT,
            ),
        )
        self.assertEqual(TSDF_FUSION_RULE, RULE)
        self.assertEqual(statement.encoded(), FIXTURE_STATEMENT)
        self.assertEqual(len(statement.encoded()), 769)
        self.assertEqual(statement.digest_sha256, FIXTURE_STATEMENT_SHA256)
        self.assertEqual(
            hashlib.sha256(FIXTURE_STATEMENT).hexdigest(),
            FIXTURE_STATEMENT_SHA256,
        )
        self.assertEqual(canonical(statement.document()), FIXTURE_STATEMENT)

    def test_the_statement_names_the_tile_size_it_was_made_at(self) -> None:
        statement = state_tsdf_map(WORK.volume, WORK.session, tile_pixels=1)
        self.assertEqual(
            statement,
            replace(
                WORK.statement,
                tile_pixels=1,
                scan_root_sha256=FIXTURE_SCAN_ROOT_BY_PIXEL,
            ),
        )
        self.assertNotEqual(statement.digest_sha256, FIXTURE_STATEMENT_SHA256)

    def test_the_room_statement_holds_the_roots_by_hand(self) -> None:
        room = WORK.room
        statement = state_tsdf_map(WORK.room_volume, room.session)
        self.assertEqual(statement.session_id, "scan-room-0001")
        self.assertEqual(statement.frame_count, 20)
        self.assertEqual(statement.block_count, 351)
        self.assertEqual(
            statement.camera, TsdfPinholeCamera(64, 48, 48.0, 48.0, 31.5, 23.5)
        )
        self.assertEqual(
            (statement.voxel_size_m, statement.truncation_m), (0.04, 0.12)
        )
        self.assertEqual(
            statement.scan_root_sha256,
            rfc_root(frame_items_by_hand(room.session_path, 16)).hex(),
        )
        self.assertEqual(
            statement.map_root_sha256,
            rfc_root(
                block_items_by_hand(WORK.room_volume_path.read_bytes())
            ).hex(),
        )
        check = check_tsdf_map_statement(
            statement, WORK.room_volume, room.session
        )
        self.assertTrue(check.holds)
        self.assertEqual(check.statement_digest_sha256,
                         statement.digest_sha256)

    def test_a_statement_is_made_only_of_a_volume_and_its_own_scan(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            TsdfError,
            "volume was fused from session 'scan-synthetic-0001', not "
            "'scan-room-0001'",
        ):
            state_tsdf_map(WORK.volume, WORK.room.session)
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            session_path = copied(FIXTURE, temporary)
            write_pgm(
                session_path / "data" / "depth" / "000001.pgm",
                2, 2, [950, 950, 950, 951],
            )
            with self.assertRaisesRegex(
                TsdfError, "not the scan the volume was fused from"
            ):
                state_tsdf_map(WORK.volume, load_scan_session(session_path))
        with self.assertRaisesRegex(TsdfError, "TsdfBlockVolume"):
            state_tsdf_map(WORK.volume_path, WORK.session)
        with self.assertRaisesRegex(TsdfError, "loaded ScanSession"):
            state_tsdf_map(WORK.volume, FIXTURE)
        with self.assertRaisesRegex(TsdfError, "tile size"):
            state_tsdf_map(WORK.volume, WORK.session, tile_pixels=0)

    def test_the_check_recomputes_both_roots_and_names_what_differs(
        self,
    ) -> None:
        check = check_tsdf_map_statement(
            WORK.statement, WORK.volume, WORK.session
        )
        self.assertTrue(check.holds)
        self.assertEqual(check.differences, ())
        self.assertEqual(
            check.statement_digest_sha256, FIXTURE_STATEMENT_SHA256
        )
        self.assertEqual(check.scan_root_sha256, FIXTURE_SCAN_ROOT)
        self.assertEqual(check.map_root_sha256, FIXTURE_MAP_ROOT)

        forged = check_tsdf_map_statement(
            WORK.statement, WORK.forged, WORK.session
        )
        self.assertFalse(forged.holds)
        self.assertEqual(forged.differences, ("map root",))
        self.assertEqual(forged.map_root_sha256, FORGED_MAP_ROOT)
        self.assertEqual(forged.scan_root_sha256, FIXTURE_SCAN_ROOT)

        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            session_path = copied(FIXTURE, temporary)
            write_pgm(
                session_path / "data" / "depth" / "000001.pgm",
                2, 2, [950, 950, 950, 951],
            )
            changed = check_tsdf_map_statement(
                WORK.statement, WORK.volume, load_scan_session(session_path)
            )
            both = check_tsdf_map_statement(
                WORK.statement, WORK.forged, load_scan_session(session_path)
            )
        self.assertEqual(changed.differences, ("scan root",))
        self.assertEqual(changed.map_root_sha256, FIXTURE_MAP_ROOT)
        self.assertNotEqual(changed.scan_root_sha256, FIXTURE_SCAN_ROOT)
        self.assertEqual(both.differences, ("scan root", "map root"))

    def test_every_line_of_a_statement_is_held_to_the_recomputation(
        self,
    ) -> None:
        other = "0" * 64
        cases = (
            ("session", {"session_id": "scan-synthetic-0002"}),
            ("fusion rule", {"rule": "projective-tsdf/another/2"}),
            ("frame count", {"frame_count": 3}),
            (
                "camera",
                {"camera": TsdfPinholeCamera(2, 2, 2.5, 2.0, 0.5, 0.5)},
            ),
            ("depth scale", {"depth_scale_m": 0.0002}),
            ("scan root", {"scan_root_sha256": other}),
            ("voxel size", {"voxel_size_m": 0.25}),
            ("truncation", {"truncation_m": 0.75}),
            ("frame stride", {"frame_stride": 2}),
            ("block count", {"block_count": 9}),
            ("map root", {"map_root_sha256": other}),
        )
        everything = {}
        for label, changes in cases:
            with self.subTest(line=label):
                claimed = replace(WORK.statement, **changes)
                check = check_tsdf_map_statement(
                    claimed, WORK.volume, WORK.session
                )
                self.assertEqual(check.differences, (label,))
                self.assertFalse(check.holds)
                self.assertEqual(
                    check.statement_digest_sha256, claimed.digest_sha256
                )
                self.assertNotEqual(
                    claimed.digest_sha256, FIXTURE_STATEMENT_SHA256
                )
            everything.update(changes)
        self.assertEqual(
            check_tsdf_map_statement(
                replace(WORK.statement, **everything),
                WORK.volume,
                WORK.session,
            ).differences,
            tuple(label for label, _ in cases),
        )
        # The tile size is held by way of the scan root it leads to.
        self.assertEqual(
            check_tsdf_map_statement(
                replace(WORK.statement, tile_pixels=1),
                WORK.volume,
                WORK.session,
            ).differences,
            ("scan root",),
        )

    def test_the_check_reads_and_does_not_write_or_fuse(self) -> None:
        before = tree_snapshot(WORK.root)
        session_before = tree_snapshot(WORK.room.session_path)
        with forbidden_calls(_FUSION):
            check_tsdf_map_statement(
                WORK.statement, WORK.volume, WORK.session
            )
            state_tsdf_map(WORK.room_volume, WORK.room.session)
        self.assertEqual(tree_snapshot(WORK.root), before)
        self.assertEqual(
            tree_snapshot(WORK.room.session_path), session_before
        )

    def test_the_check_refuses_what_is_not_a_statement_volume_or_scan(
        self,
    ) -> None:
        with self.assertRaisesRegex(TsdfError, "TsdfMapStatement"):
            check_tsdf_map_statement(
                WORK.statement.document(), WORK.volume, WORK.session
            )
        with self.assertRaisesRegex(TsdfError, "TsdfBlockVolume"):
            check_tsdf_map_statement(
                WORK.statement, WORK.volume_path, WORK.session
            )
        with self.assertRaisesRegex(TsdfError, "loaded ScanSession"):
            check_tsdf_map_statement(WORK.statement, WORK.volume, FIXTURE)
        # A volume and a scan of two different sessions are not two
        # things a statement could be of.
        with self.assertRaisesRegex(TsdfError, "its own scan"):
            check_tsdf_map_statement(
                WORK.statement, WORK.volume, WORK.room.session
            )

    def test_statements_validate_themselves(self) -> None:
        cases = {
            "no session": ({"session_id": ""}, "session_id"),
            "a session in capitals": (
                {"session_id": "Scan-0001"},
                "session identifier",
            ),
            "a session with a bracket in it": (
                {"session_id": "scan[1]"},
                "session identifier",
            ),
            "a session with a quote in it": (
                {"session_id": 'scan"1'},
                "session identifier",
            ),
            "no rule": ({"rule": ""}, "rule must be named"),
            "a rule that is not text": ({"rule": 1}, "rule must be named"),
            "a rule with a space in it": (
                {"rule": "projective tsdf/1"},
                "rule must be named",
            ),
            "a rule with a brace in it": (
                {"rule": "projective-tsdf/{1}"},
                "rule must be named",
            ),
            "a rule of 129 characters": (
                {"rule": "r" * 129},
                "rule must be named",
            ),
            "a camera that is six numbers": (
                {"camera": (2, 2, 2.0, 2.0, 0.5, 0.5)},
                "TsdfPinholeCamera",
            ),
            "depth scale of zero": ({"depth_scale_m": 0.0}, "depth_scale_m"),
            "depth scale not a number": (
                {"depth_scale_m": float("nan")},
                "depth_scale_m",
            ),
            "voxel size as an integer": ({"voxel_size_m": 1}, "voxel_size_m"),
            "truncation below zero": ({"truncation_m": -0.5}, "truncation_m"),
            "truncation under a voxel": (
                {"truncation_m": 0.0625},
                "greater than or equal to voxel_size_m",
            ),
            "no frames": ({"frame_count": 0}, "frame_count"),
            "frames as a flag": ({"frame_count": True}, "frame_count"),
            "stride of zero": ({"frame_stride": 0}, "frame_stride"),
            "blocks as a fraction": ({"block_count": 8.0}, "block_count"),
            "tiles of no pixels": ({"tile_pixels": 0}, "tile size"),
            "tiles too large": (
                {"tile_pixels": MAX_TSDF_DEPTH_TILE_PIXELS + 1},
                "tile size",
            ),
            "a scan root in capitals": (
                {"scan_root_sha256": FIXTURE_SCAN_ROOT.upper()},
                "scan root is invalid",
            ),
            "a short map root": (
                {"map_root_sha256": FIXTURE_MAP_ROOT[:63]},
                "map root is invalid",
            ),
        }
        for name, (changes, message) in cases.items():
            with self.subTest(case=name):
                with self.assertRaisesRegex(TsdfError, message):
                    replace(WORK.statement, **changes)

    def test_frame_and_scan_commitments_validate_themselves(self) -> None:
        depth_root = bytes.fromhex(FIXTURE_DEPTH_ROOTS[0])
        pose = FIXTURE_POSES[0]
        frames = {
            "a place before the first": (-1, pose, depth_root),
            "a place that is a flag": (True, pose, depth_root),
            "a place past 63 bits": (2**63, pose, depth_root),
            "a pose of fifteen numbers": (0, pose[:15], depth_root),
            "a pose of integers": (0, (1,) * 16, depth_root),
            "a pose as a list": (0, list(pose), depth_root),
            "a short depth root": (0, pose, depth_root[:31]),
            "a depth root as text": (0, pose, depth_root.hex()),
            "a depth root as a bytearray": (0, pose, bytearray(depth_root)),
        }
        for name, arguments in frames.items():
            with self.subTest(frame=name):
                with self.assertRaisesRegex(TsdfError, "frame commitment"):
                    TsdfFrameCommitment(*arguments)
        self.assertEqual(
            TsdfFrameCommitment(2**63 - 1, None, None).leaf_data(),
            b"SFFRAME1" + b"\xff" * 7 + b"\x7f" + bytes(1 + 128 + 32),
        )

        scan = commit_tsdf_scan(WORK.session)
        first, second = scan.frames
        scans = {
            "no session": ({"session_id": ""}, "session_id"),
            "a camera that is not one": ({"camera": None}, "camera"),
            "depth scale of zero": ({"depth_scale_m": 0.0}, "depth scale"),
            "depth scale as an integer": ({"depth_scale_m": 1}, "depth scale"),
            "tiles of no pixels": ({"tile_pixels": 0}, "tile size"),
            "no frames": ({"frames": ()}, "at least one"),
            "frames as a list": ({"frames": [first, second]}, "at least one"),
            "something that is not a frame": (
                {"frames": (first, second.leaf_data())},
                "at least one",
            ),
            "a frame left out": ({"frames": (second,)}, "every frame"),
            "frames out of order": (
                {"frames": (second, first)},
                "every frame",
            ),
            "a frame twice": ({"frames": (first, first)}, "every frame"),
        }
        for name, (changes, message) in scans.items():
            with self.subTest(scan=name):
                with self.assertRaisesRegex(TsdfError, message):
                    replace(scan, **changes)
        self.assertIsInstance(scan, TsdfScanCommitment)
        # A scan of one frame is that frame's leaf.
        self.assertEqual(
            replace(scan, frames=(first,)).root, first.leaf_hash()
        )


class MapStatementFileTests(unittest.TestCase):
    def test_a_statement_written_is_the_statement_read(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            output = Path(temporary) / "made" / "here" / "map.sftcommit"
            report = write_tsdf_map_statement(WORK.statement, output)
            self.assertEqual(report.output, output.resolve())
            self.assertEqual(report.output_bytes, 769)
            self.assertEqual(
                report.output_digest_sha256, FIXTURE_STATEMENT_SHA256
            )
            self.assertIs(report.statement, WORK.statement)
            self.assertEqual(output.read_bytes(), FIXTURE_STATEMENT)
            # Nothing is left beside it.
            self.assertEqual(
                [path.name for path in output.parent.iterdir()],
                ["map.sftcommit"],
            )
            self.assertEqual(load_tsdf_map_statement(output), WORK.statement)
            self.assertEqual(
                load_tsdf_map_statement(str(output)).digest_sha256,
                FIXTURE_STATEMENT_SHA256,
            )
            capitals = Path(temporary) / "MAP.SFTCOMMIT"
            write_tsdf_map_statement(WORK.statement, capitals)
            self.assertEqual(load_tsdf_map_statement(capitals), WORK.statement)

            # Every character a session's name or a rule's may hold, at
            # the longest either may be: what can be made can be read.
            widest = replace(
                WORK.statement,
                session_id=("a.b_c-d0" * 16),
                rule=("x/y.z_w-" * 16),
                frame_count=2**40,
                block_count=2**40,
                frame_stride=2**20,
                depth_scale_m=5e-324,
                truncation_m=1.7976931348623157e308,
            )
            self.assertEqual(
                (len(widest.session_id), len(widest.rule)), (128, 128)
            )
            report = write_tsdf_map_statement(
                widest, Path(temporary) / "widest.sftcommit"
            )
            self.assertEqual(load_tsdf_map_statement(report.output), widest)
            self.assertLess(
                report.output_bytes, MAX_TSDF_MAP_STATEMENT_BYTES // 8
            )

    def test_writing_never_replaces_a_file(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            output = Path(temporary) / "map.sftcommit"
            output.write_bytes(b"something else")
            with self.assertRaisesRegex(TsdfError, "output already exists"):
                write_tsdf_map_statement(WORK.statement, output)
            self.assertEqual(output.read_bytes(), b"something else")
            output.unlink()

            # The file appears between the check and the link.
            def link(source, target):
                Path(target).write_bytes(b"someone else's")
                raise FileExistsError(target)

            with patch("spatialforge.tsdf_commitment.os.link", link):
                with self.assertRaisesRegex(
                    TsdfError, "appeared while writing"
                ):
                    write_tsdf_map_statement(WORK.statement, output)
            self.assertEqual(output.read_bytes(), b"someone else's")
            self.assertEqual(
                [path.name for path in Path(temporary).iterdir()],
                ["map.sftcommit"],
            )
            output.unlink()

            def refuse(source, target):
                raise PermissionError("no links here")

            with patch("spatialforge.tsdf_commitment.os.link", refuse):
                with self.assertRaisesRegex(
                    TsdfError, "cannot write TSDF map statement"
                ):
                    write_tsdf_map_statement(WORK.statement, output)
            self.assertEqual(list(Path(temporary).iterdir()), [])

            for name in ("map.json", "map.sftcommit.txt", "map"):
                with self.subTest(name=name):
                    with self.assertRaisesRegex(TsdfError, ".sftcommit"):
                        write_tsdf_map_statement(
                            WORK.statement, Path(temporary) / name
                        )
            with self.assertRaisesRegex(TsdfError, "TsdfMapStatement"):
                write_tsdf_map_statement(FIXTURE_STATEMENT, output)
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_only_the_canonical_bytes_are_a_statement(self) -> None:
        text = FIXTURE_STATEMENT.decode("ascii")
        document = json.loads(text)
        cases = {
            "packed onto one line": json.dumps(document, sort_keys=True),
            "keys in another order": (
                json.dumps(dict(reversed(document.items())), indent=2) + "\n"
            ),
            "indented by four": (
                json.dumps(document, indent=4, sort_keys=True) + "\n"
            ),
            "no final newline": text[:-1],
            "two final newlines": text + "\n",
            "a space before the end": text[:-1] + " \n",
            "carriage returns": text.replace("\n", "\r\n"),
            "a number written longer": text.replace(
                '"truncation_m": 0.5', '"truncation_m": 0.50'
            ),
            "a number written as a power": text.replace(
                '"depth_scale_m": 0.001', '"depth_scale_m": 1e-3'
            ),
        }
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            for number, (name, changed) in enumerate(cases.items()):
                with self.subTest(case=name):
                    self.assertNotEqual(changed, text)
                    self.assertEqual(json.loads(changed), document)
                    path = Path(temporary) / f"{number}.sftcommit"
                    path.write_bytes(changed.encode("ascii"))
                    with self.assertRaisesRegex(
                        TsdfError, "not in canonical form"
                    ):
                        load_tsdf_map_statement(path)

    def test_a_statement_that_is_not_exactly_one_is_refused(self) -> None:
        def changed(*path_and_value):
            *path, value = path_and_value
            document = json.loads(FIXTURE_STATEMENT)
            target = document
            for key in path[:-1]:
                target = target[key]
            if value is _GONE:
                del target[path[-1]]
            else:
                target[path[-1]] = value
            return canonical(document)

        text = FIXTURE_STATEMENT.decode("ascii")
        cases = {
            "another schema": (
                changed("schema", "spatialforge.tsdf-block-volume"),
                "statement.schema: expected",
            ),
            "another version": (
                changed("schema_version", "0.2.0"),
                "statement.schema_version: expected",
            ),
            "another hash": (
                changed("hash", "sha-512"),
                "statement.hash: expected",
            ),
            "another tree": (
                changed("tree", "binary"),
                "statement.tree: expected",
            ),
            "another sample": (
                changed("scan", "depth_sample", "uint16-be"),
                "statement.scan.depth_sample: expected",
            ),
            "another block size": (
                changed("map", "block_resolution", 16),
                "statement.map.block_resolution: expected",
            ),
            "no rule": (changed("rule", ""), "statement.rule"),
            "a rule with a space in it": (
                changed("rule", "projective tsdf/1"),
                "rule must be named",
            ),
            "a session with a bracket in it": (
                changed("session_id", "scan[1]"),
                "holds an array",
            ),
            "a session in capitals": (
                changed("session_id", "Scan-0001"),
                "session identifier",
            ),
            "a rule that is a number": (
                changed("rule", 1),
                "statement.rule",
            ),
            "no session": (changed("session_id", ""), "statement.session_id"),
            "a field left out": (
                changed("tree", _GONE),
                "statement.tree: missing",
            ),
            "a scan field left out": (
                changed("scan", "tile_pixels", _GONE),
                "statement.scan.tile_pixels: missing",
            ),
            "a field nobody asked for": (
                changed("replay_digest_sha256", REPLAY_SHA256),
                "statement: unknown field 'replay_digest_sha256'",
            ),
            "a map field nobody asked for": (
                changed("map", "note", "trust me"),
                "statement.map: unknown field 'note'",
            ),
            "a camera field nobody asked for": (
                changed("scan", "camera", "k1", 0.5),
                "statement.scan.camera: unknown field 'k1'",
            ),
            "no frames": (
                changed("scan", "frames", 0),
                "statement.scan.frames",
            ),
            "frames as a flag": (
                changed("scan", "frames", True),
                "statement.scan.frames",
            ),
            "frames as a fraction": (
                changed("scan", "frames", 2.0),
                "statement.scan.frames",
            ),
            "tiles of no pixels": (
                changed("scan", "tile_pixels", 0),
                "statement.scan.tile_pixels",
            ),
            "tiles too large": (
                changed("scan", "tile_pixels", 1025),
                "tile size",
            ),
            "a focal length as an integer": (
                changed("scan", "camera", "fx", 2),
                "statement.scan.camera.fx: expected a number",
            ),
            "a focal length as text": (
                changed("scan", "camera", "fy", "2.0"),
                "statement.scan.camera.fy: expected a number",
            ),
            "a centre off the image": (
                changed("scan", "camera", "cx", 2.0),
                "camera intrinsics are invalid",
            ),
            "an image of no width": (
                changed("scan", "camera", "width", 0),
                "statement.scan.camera.width",
            ),
            "depth scale of zero": (
                changed("scan", "depth_scale_m", 0.0),
                "statement.scan.depth_scale_m",
            ),
            "depth scale as an integer": (
                changed("scan", "depth_scale_m", 1),
                "statement.scan.depth_scale_m",
            ),
            "voxel size below zero": (
                changed("map", "voxel_size_m", -0.125),
                "statement.map.voxel_size_m",
            ),
            "truncation under a voxel": (
                changed("map", "truncation_m", 0.0625),
                "greater than or equal to voxel_size_m",
            ),
            "stride of zero": (
                changed("map", "frame_stride", 0),
                "statement.map.frame_stride",
            ),
            "no blocks": (changed("map", "blocks", 0), "statement.map.blocks"),
            "a scan root in capitals": (
                changed("scan", "root_sha256", FIXTURE_SCAN_ROOT.upper()),
                "statement.scan.root_sha256: expected a SHA-256",
            ),
            "a short map root": (
                changed("map", "root_sha256", FIXTURE_MAP_ROOT[:63]),
                "statement.map.root_sha256: expected a SHA-256",
            ),
            "a map root that is nothing": (
                changed("map", "root_sha256", None),
                "statement.map.root_sha256: expected a SHA-256",
            ),
            "a scan that is not an object": (
                changed("scan", "frames=2"),
                "statement.scan: expected an object",
            ),
            "a number where the statement goes": (
                b"3\n",
                "statement: expected an object",
            ),
            "a list": (b'["statement"]\n', "holds an array"),
            "a list inside": (
                text.replace('"blocks": 8', '"blocks": [8]').encode(),
                "holds an array",
            ),
            "nested too deeply": (
                text.replace('"blocks": 8', '"blocks": ' + '{"a":' * 6 + "8"
                             + "}" * 6).encode(),
                "nested too deeply",
            ),
            "a key twice": (
                text.replace(
                    '  "tree": "rfc6962"',
                    '  "tree": "rfc6962",\n  "tree": "rfc6962"',
                ).encode(),
                "duplicate JSON key 'tree'",
            ),
            "not a number": (
                text.replace('"depth_scale_m": 0.001',
                             '"depth_scale_m": NaN').encode(),
                "invalid JSON",
            ),
            "more than a number can be": (
                text.replace('"fx": 2.0', '"fx": 1e999').encode(),
                "camera intrinsics are invalid",
            ),
            "cut short": (FIXTURE_STATEMENT[:400], "invalid JSON"),
            "empty": (b"", "invalid JSON"),
            "not ascii": (
                text.replace("scan-synthetic-0001", "scan-synth" + chr(233))
                .encode("utf-8"),
                "must be ASCII JSON",
            ),
            "too large": (
                FIXTURE_STATEMENT
                + b" " * (MAX_TSDF_MAP_STATEMENT_BYTES - 768),
                "larger than 16384 bytes",
            ),
        }
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            for number, (name, (encoded, message)) in enumerate(
                cases.items()
            ):
                with self.subTest(case=name):
                    self.assertNotEqual(encoded, FIXTURE_STATEMENT)
                    path = Path(temporary) / f"{number}.sftcommit"
                    path.write_bytes(encoded)
                    with self.assertRaisesRegex(TsdfError, message):
                        load_tsdf_map_statement(path)
            # At the limit exactly, a statement is read and then found
            # not to be canonical: the limit is on reading, not a form.
            path = Path(temporary) / "limit.sftcommit"
            path.write_bytes(
                FIXTURE_STATEMENT
                + b" " * (MAX_TSDF_MAP_STATEMENT_BYTES - 769)
            )
            with self.assertRaisesRegex(TsdfError, "not in canonical form"):
                load_tsdf_map_statement(path)

            good = Path(temporary) / "good.sftcommit"
            good.write_bytes(FIXTURE_STATEMENT)
            renamed = Path(temporary) / "good.json"
            renamed.write_bytes(FIXTURE_STATEMENT)
            with self.assertRaisesRegex(TsdfError, "must end in .sftcommit"):
                load_tsdf_map_statement(renamed)
            with self.assertRaisesRegex(
                TsdfError, "cannot read TSDF map statement"
            ):
                load_tsdf_map_statement(Path(temporary) / "gone.sftcommit")
            folder = Path(temporary) / "folder.sftcommit"
            folder.mkdir()
            with self.assertRaisesRegex(
                TsdfError, "cannot read TSDF map statement"
            ):
                load_tsdf_map_statement(folder)
            self.assertEqual(load_tsdf_map_statement(good), WORK.statement)

    def test_a_statement_under_another_rule_is_read_and_then_differs(
        self,
    ) -> None:
        # The loader does not know every rule there will be. It reads
        # the name; the check is what holds it against this build's.
        claimed = replace(WORK.statement, rule="projective-tsdf/another/2")
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            path = Path(temporary) / "other.sftcommit"
            write_tsdf_map_statement(claimed, path)
            loaded = load_tsdf_map_statement(path)
        self.assertEqual(loaded, claimed)
        self.assertEqual(
            check_tsdf_map_statement(
                loaded, WORK.volume, WORK.session
            ).differences,
            ("fusion rule",),
        )


_GONE = object()


class MapCommitCliTests(unittest.TestCase):
    def run_cli(self, *arguments: str) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                code = main(["reconstruct", *arguments])
            except SystemExit as stop:
                code = stop.code
        return code, stdout.getvalue(), stderr.getvalue()

    def test_cli_commits_to_a_volume_and_its_scan(self) -> None:
        before = tree_snapshot(WORK.root)
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            output = Path(temporary) / "fixture.sftcommit"
            code, printed, errors = self.run_cli(
                "tsdf-map-commit",
                str(WORK.volume_path), str(FIXTURE), str(output),
            )
            self.assertEqual(errors, "")
            self.assertEqual(code, 0)
            self.assertEqual(
                printed,
                "TSDF MAP COMMIT scan-synthetic-0001\n"
                "volume: verified\n"
                "session_replay: matched\n"
                "rule: projective-tsdf/unit-weight/nearest-pixel/1\n"
                "hash: sha-256\n"
                "tree: rfc6962\n"
                "scan: frames=2 with_depth=2 with_pose=2 tile_pixels=16 "
                "frame_table_bytes=354\n"
                f"scan_root_sha256: {FIXTURE_SCAN_ROOT}\n"
                "map: blocks=8 voxel_size_m=0.125000000 "
                "truncation_m=0.500000000 frame_stride=1\n"
                f"map_root_sha256: {FIXTURE_MAP_ROOT}\n"
                "statement_bytes: 769\n"
                f"statement_sha256: {FIXTURE_STATEMENT_SHA256}\n"
                f"output: {output.resolve()}\n"
                "volume_modified: no\n"
                "session_modified: no\n"
                "blocks_fused: no\n",
            )
            self.assertEqual(output.read_bytes(), FIXTURE_STATEMENT)
            self.assertEqual(
                [path.name for path in Path(temporary).iterdir()],
                ["fixture.sftcommit"],
            )

            # A second time, the output is in the way and stays.
            code, printed, errors = self.run_cli(
                "tsdf-map-commit",
                str(WORK.volume_path), str(FIXTURE), str(output),
            )
            self.assertEqual(code, 2)
            self.assertEqual(printed, "")
            self.assertEqual(
                errors,
                f"TSDF MAP COMMIT FAILED {WORK.volume_path}\n"
                f"- output already exists: {output.resolve()}\n",
            )
            self.assertEqual(output.read_bytes(), FIXTURE_STATEMENT)
        self.assertEqual(tree_snapshot(WORK.root), before)

    def test_cli_commits_at_the_tile_size_asked_for(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            output = Path(temporary) / "by-pixel.sftcommit"
            code, printed, errors = self.run_cli(
                "tsdf-map-commit",
                str(WORK.volume_path), str(FIXTURE), str(output),
                "--tiles", "1",
            )
            self.assertEqual((code, errors), (0, ""))
            lines = printed.splitlines()
            self.assertEqual(
                lines[6],
                "scan: frames=2 with_depth=2 with_pose=2 tile_pixels=1 "
                "frame_table_bytes=354",
            )
            self.assertEqual(
                lines[7], f"scan_root_sha256: {FIXTURE_SCAN_ROOT_BY_PIXEL}"
            )
            self.assertEqual(
                lines[9], f"map_root_sha256: {FIXTURE_MAP_ROOT}"
            )
            self.assertEqual(lines[10], "statement_bytes: 768")
            statement = load_tsdf_map_statement(output)
            self.assertEqual(statement.tile_pixels, 1)
            self.assertEqual(
                lines[11], f"statement_sha256: {statement.digest_sha256}"
            )

            code, printed, errors = self.run_cli(
                "tsdf-map-commit-check",
                str(output), str(WORK.volume_path), str(FIXTURE),
            )
            self.assertEqual((code, errors), (0, ""))
            self.assertIn("differences: none\n", printed)

    def test_cli_says_what_a_scan_lacks(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            session_path = copied(FIXTURE, temporary)
            without_first_line(session_path, "streams/depth.jsonl")
            session = load_scan_session(session_path)
            plan_path = Path(temporary) / "one.sftplan"
            plan_tsdf_blocks(session, plan_path, **PLAN_ARGUMENTS)
            storage, receipt = fused(load_tsdf_block_plan(plan_path), session)
            volume_path = Path(temporary) / "one.sftvol"
            write_tsdf_block_volume(storage, receipt, volume_path)
            output = Path(temporary) / "one.sftcommit"
            code, printed, errors = self.run_cli(
                "tsdf-map-commit",
                str(volume_path), str(session_path), str(output),
            )
            self.assertEqual((code, errors), (0, ""))
            self.assertEqual(
                printed.splitlines()[6],
                "scan: frames=2 with_depth=1 with_pose=2 tile_pixels=16 "
                "frame_table_bytes=354",
            )
            self.assertEqual(
                printed.splitlines()[7],
                "scan_root_sha256: "
                + rfc_root(frame_items_by_hand(session_path, 16)).hex(),
            )

    def test_cli_checks_a_statement_and_exits_by_what_it_finds(self) -> None:
        before = tree_snapshot(WORK.root)
        fixture_before = tree_snapshot(FIXTURE)
        code, printed, errors = self.run_cli(
            "tsdf-map-commit-check",
            str(WORK.statement_path), str(WORK.volume_path), str(FIXTURE),
        )
        self.assertEqual(errors, "")
        self.assertEqual(code, 0)
        self.assertEqual(
            printed,
            "TSDF MAP COMMIT CHECK scan-synthetic-0001\n"
            "statement: valid\n"
            f"statement_sha256: {FIXTURE_STATEMENT_SHA256}\n"
            "rule: projective-tsdf/unit-weight/nearest-pixel/1\n"
            f"scan_root_stated: {FIXTURE_SCAN_ROOT}\n"
            f"scan_root_recomputed: {FIXTURE_SCAN_ROOT}\n"
            f"map_root_stated: {FIXTURE_MAP_ROOT}\n"
            f"map_root_recomputed: {FIXTURE_MAP_ROOT}\n"
            "differences: none\n"
            "statement_modified: no\n"
            "volume_modified: no\n"
            "session_modified: no\n"
            "blocks_fused: no\n"
            "result: the statement is a statement of this volume and this "
            "scan\n",
        )

        code, printed, errors = self.run_cli(
            "tsdf-map-commit-check",
            str(WORK.statement_path), str(WORK.forged_path), str(FIXTURE),
        )
        self.assertEqual(errors, "")
        self.assertEqual(code, 1)
        self.assertEqual(
            printed,
            "TSDF MAP COMMIT CHECK scan-synthetic-0001\n"
            "statement: valid\n"
            f"statement_sha256: {FIXTURE_STATEMENT_SHA256}\n"
            "rule: projective-tsdf/unit-weight/nearest-pixel/1\n"
            f"scan_root_stated: {FIXTURE_SCAN_ROOT}\n"
            f"scan_root_recomputed: {FIXTURE_SCAN_ROOT}\n"
            f"map_root_stated: {FIXTURE_MAP_ROOT}\n"
            f"map_root_recomputed: {FORGED_MAP_ROOT}\n"
            "differences: map root\n"
            "statement_modified: no\n"
            "volume_modified: no\n"
            "session_modified: no\n"
            "blocks_fused: no\n"
            "result: the statement is not a statement of this volume and "
            "this scan\n",
        )
        self.assertEqual(tree_snapshot(WORK.root), before)
        self.assertEqual(tree_snapshot(FIXTURE), fixture_before)

    def test_cli_check_names_a_changed_scan(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            session_path = copied(FIXTURE, temporary)
            write_pgm(
                session_path / "data" / "depth" / "000001.pgm",
                2, 2, [950, 950, 950, 951],
            )
            code, printed, errors = self.run_cli(
                "tsdf-map-commit-check",
                str(WORK.statement_path),
                str(WORK.forged_path),
                str(session_path),
            )
            # Making a statement of the two is refused outright.
            made = self.run_cli(
                "tsdf-map-commit",
                str(WORK.volume_path),
                str(session_path),
                str(Path(temporary) / "changed.sftcommit"),
            )
            self.assertFalse(
                (Path(temporary) / "changed.sftcommit").exists()
            )
        self.assertEqual((code, errors), (1, ""))
        self.assertIn("differences: scan root, map root\n", printed)
        self.assertEqual(made[0], 2)
        self.assertEqual(made[1], "")
        self.assertEqual(
            made[2],
            f"TSDF MAP COMMIT FAILED {WORK.volume_path}\n"
            "- the session's replay digest does not match the one recorded "
            "in the volume; this is not the scan the volume was fused "
            "from\n",
        )

    def test_cli_refuses_what_it_cannot_commit_or_check(self) -> None:
        before = tree_snapshot(WORK.root)
        room_path = WORK.room.session_path
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            folder = Path(temporary)
            (folder / "bent.sftcommit").write_bytes(FIXTURE_STATEMENT[:-1])
            commits = (
                (
                    "another scan",
                    (str(WORK.volume_path), str(room_path),
                     str(folder / "a.sftcommit")),
                    "- volume was fused from session 'scan-synthetic-0001', "
                    "not 'scan-room-0001'\n",
                ),
                (
                    "an output of another kind",
                    (str(WORK.volume_path), str(FIXTURE),
                     str(folder / "a.json")),
                    "- output filename must end in .sftcommit\n",
                ),
            )
            for name, arguments, message in commits:
                with self.subTest(commit=name):
                    code, printed, errors = self.run_cli(
                        "tsdf-map-commit", *arguments
                    )
                    self.assertEqual((code, printed), (2, ""))
                    self.assertEqual(
                        errors,
                        f"TSDF MAP COMMIT FAILED {WORK.volume_path}\n"
                        + message,
                    )
            code, printed, errors = self.run_cli(
                "tsdf-map-commit",
                str(folder / "none.sftvol"), str(FIXTURE),
                str(folder / "a.sftcommit"),
            )
            self.assertEqual((code, printed), (2, ""))
            self.assertTrue(
                errors.startswith(
                    f"TSDF MAP COMMIT FAILED {folder / 'none.sftvol'}\n- "
                )
            )
            code, printed, errors = self.run_cli(
                "tsdf-map-commit",
                str(WORK.volume_path), str(folder / "none.vgsession"),
                str(folder / "a.sftcommit"),
            )
            self.assertEqual((code, printed), (2, ""))
            self.assertTrue(
                errors.startswith(
                    f"TSDF MAP COMMIT FAILED {WORK.volume_path}\n- "
                )
            )

            checks = (
                (
                    "a statement not in canonical form",
                    folder / "bent.sftcommit",
                    (str(WORK.volume_path), str(FIXTURE)),
                    "- TSDF map statement is not in canonical form\n",
                ),
                (
                    "a volume and a scan of two sessions",
                    WORK.statement_path,
                    (str(WORK.volume_path), str(room_path)),
                    "- a TSDF map statement binds a volume to its own "
                    "scan\n",
                ),
                (
                    "a statement of another kind",
                    WORK.volume_path,
                    (str(WORK.volume_path), str(FIXTURE)),
                    "- statement filename must end in .sftcommit\n",
                ),
            )
            for name, statement, arguments, message in checks:
                with self.subTest(check=name):
                    code, printed, errors = self.run_cli(
                        "tsdf-map-commit-check", str(statement), *arguments
                    )
                    self.assertEqual((code, printed), (2, ""))
                    self.assertEqual(
                        errors,
                        f"TSDF MAP COMMIT CHECK FAILED {statement}\n"
                        + message,
                    )
            code, printed, errors = self.run_cli(
                "tsdf-map-commit-check",
                str(WORK.statement_path), str(WORK.volume_path),
                str(folder / "none.vgsession"),
            )
            self.assertEqual((code, printed), (2, ""))
            self.assertTrue(
                errors.startswith(
                    f"TSDF MAP COMMIT CHECK FAILED {WORK.statement_path}\n- "
                )
            )
            self.assertEqual(
                sorted(path.name for path in folder.iterdir()),
                ["bent.sftcommit"],
            )

            for tiles in ("0", "-4", "sixteen"):
                with self.subTest(tiles=tiles):
                    code, printed, errors = self.run_cli(
                        "tsdf-map-commit",
                        str(WORK.volume_path), str(FIXTURE),
                        str(folder / "a.sftcommit"), "--tiles", tiles,
                    )
                    self.assertEqual((code, printed), (2, ""))
                    self.assertIn("--tiles", errors)
            code, printed, errors = self.run_cli(
                "tsdf-map-commit",
                str(WORK.volume_path), str(FIXTURE),
                str(folder / "a.sftcommit"), "--tiles", "1025",
            )
            self.assertEqual((code, printed), (2, ""))
            self.assertEqual(
                errors,
                f"TSDF MAP COMMIT FAILED {WORK.volume_path}\n"
                "- TSDF depth tile size must be an integer from 1 to 1024 "
                "pixels\n",
            )
            self.assertFalse((folder / "a.sftcommit").exists())
        self.assertEqual(tree_snapshot(WORK.root), before)
        self.assertTrue(os.path.isdir(room_path))


if __name__ == "__main__":
    unittest.main()
