"""A block fused from the pixels it reads, held to the block fused alone.

Two halves are under test. Gathering takes a session and returns, for a
block, the poses of every selected frame and the tiles of depth the block
reads. Fusing takes that and nothing else. If the second is to stand in
for the first step's whole-frame fusion it has to give the same bytes, and
where it cannot it has to refuse: a pixel that was not given is an error,
never a depth of zero.

So every block of every fixture is fused both ways and compared, at tile
sizes from one pixel to the whole image. Every tile gathered is then taken
away in turn, and each time the fusion must refuse and name a pixel. And
which pixels a block reads is worked out a second time here, by plain
arithmetic that shares nothing with the evaluator.
"""

from __future__ import annotations

import io
import shutil
import tempfile
import unittest
from contextlib import ExitStack, contextmanager, redirect_stderr
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from spatialforge import (
    MAX_TSDF_DEPTH_TILE_PIXELS,
    TSDF_DEPTH_TILE_PIXELS,
    TsdfBlockCheckStatus,
    TsdfBlockEvidence,
    TsdfBlockFrameEvidence,
    TsdfDepthTiles,
    TsdfPinholeCamera,
    allocate_empty_tsdf_blocks,
    check_tsdf_volume_blocks,
    check_tsdf_volume_blocks_from_tiles,
    fuse_tsdf_block_alone,
    fuse_tsdf_block_from_evidence,
    fuse_tsdf_blocks_alone,
    fuse_tsdf_plan_streaming,
    gather_tsdf_block_evidence,
    gather_tsdf_blocks_evidence,
    load_tsdf_block_plan,
    load_tsdf_block_volume,
    write_tsdf_block_volume,
)
from spatialforge import tsdf_stream_fusion
from spatialforge.cli import main
from spatialforge.errors import TsdfError
from spatialforge.point_cloud import _validate_reconstruction_contract
from spatialforge.replay import replay_session
from spatialforge.session_loader import load_scan_session
from spatialforge.tsdf_block_plan import plan_tsdf_blocks
from spatialforge.tsdf_expanded_plan import write_tsdf_expanded_block_plan
from spatialforge.tsdf_stream_expansion import (
    propose_tsdf_plan_expansion_streaming,
)
from spatialforge.tsdf_stream_fusion import (
    _decode_metric_depth,
    _decode_raw_depth,
    _metric_depth,
)

from tests.heavy_fixtures import shared_room_case
from tests.test_tsdf_block_volume import assemble, split

TEST_ROOT = Path(__file__).resolve().parent
FIXTURE = TEST_ROOT / "fixtures" / "minimal.vgsession"
PLAN_ARGUMENTS = {"voxel_size_m": 0.125, "truncation_m": 0.5}
REPLAY_SHA256 = (
    "dc001ae0ca01004a21ad227b12d57a7350bbc23904040453ca73fd955ef050b8"
)
FIXTURE_VOLUME_SHA256 = (
    "29f9f427b43c2a9e8ec416dad9a3ba1572044c716407ef2b30989daed9c2adc3"
)
FIXTURE_BLOCKS = (
    (0, -1, -1),
    (1, -1, -1),
    (0, 0, -1),
    (1, 0, -1),
    (0, -1, 0),
    (1, -1, 0),
    (0, 0, 0),
    (1, 0, 0),
)
# Over the room's 351 blocks and 20 frames, by tile size: tiles kept and
# depth samples in them. One pixel to a tile counts the pixels read.
ROOM_TILES = {1: 294_483, 4: 33_745, 16: 8_614, 64: 5_136}
ROOM_SAMPLES = {1: 294_483, 4: 539_920, 16: 2_205_184, 64: 15_777_792}
ROOM_FRAMES_READ = 5_136
ROOM_SCAN_DEPTH_BYTES = 20 * 64 * 48 * 2

# Fusing from evidence reads no session and opens no file.
_FORBIDDEN_WITHOUT_A_SESSION = (
    "spatialforge.tsdf_block_evidence.replay_session",
    "spatialforge.tsdf_lone_block.replay_session",
    "spatialforge.tsdf_stream_fusion._read_depth_array",
    "spatialforge.tsdf_block_evidence._decode_raw_depth",
    "spatialforge.tsdf_block_evidence._prepare_lone_fusion",
    "pathlib.Path.open",
    "PIL.Image.open",
)

WORK: SimpleNamespace | None = None


@contextmanager
def forbidden_calls(targets: tuple[str, ...]):
    with ExitStack() as stack:
        patches = {
            target: stack.enter_context(patch(target)) for target in targets
        }
        yield
        for target, mock in patches.items():
            if mock.called:
                raise AssertionError(f"{target} was called")


def tree_snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def fused(plan, session):
    storage = allocate_empty_tsdf_blocks(plan, session)
    receipt = fuse_tsdf_plan_streaming(storage, session)
    return storage, receipt


def options(plan) -> dict:
    return {
        "voxel_size_m": plan.voxel_size_m,
        "truncation_m": plan.truncation_m,
        "frame_stride": plan.frame_stride,
    }


def described(lone) -> tuple:
    """A lone block's bytes and everything it says it took."""

    return (
        lone.block_index,
        lone.tsdf_sums.tobytes(),
        lone.weights.tobytes(),
        lone.session_id,
        lone.replay_digest_sha256,
        lone.voxel_size_m,
        lone.truncation_m,
        lone.frame_stride,
        lone.selected_observations,
        lone.ready_observations,
        lone.ruled_out_behind_camera,
        lone.ruled_out_outside_image,
        lone.evaluated_observations,
        lone.contributing_observations,
    )


def without_tile(evidence: TsdfBlockEvidence, frame_at: int, tile_at: int):
    """The same evidence with one tile of one frame taken away."""

    frame = evidence.frames[frame_at]
    kept = frame.tiles.tiles[:tile_at] + frame.tiles.tiles[tile_at + 1:]
    tiles = None if not kept else replace(frame.tiles, tiles=kept)
    frames = (
        evidence.frames[:frame_at]
        + (replace(frame, tiles=tiles),)
        + evidence.frames[frame_at + 1:]
    )
    return replace(evidence, frames=frames)


def pixels_by_a_second_account(camera, transform, block, voxel_size_m):
    """The pixels a block's voxels land on, by plain arithmetic.

    Written out from the rule, one coordinate at a time, sharing no code
    with the evaluator: a voxel reads the pixel nearest its projection if
    it is in front of the camera and that pixel is on the image.
    """

    local = np.arange(8)
    z, y, x = np.meshgrid(local, local, local, indexing="ij")
    centres = (
        np.asarray(block) * 8
        + np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
        + 0.5
    ) * voxel_size_m
    offset_x = centres[:, 0] - transform[3]
    offset_y = centres[:, 1] - transform[7]
    offset_z = centres[:, 2] - transform[11]
    across, down, depth = (
        transform[axis] * offset_x
        + transform[4 + axis] * offset_y
        + transform[8 + axis] * offset_z
        for axis in range(3)
    )
    pixels = set()
    for voxel in range(512):
        if not depth[voxel] > 0.0:
            continue
        u = camera.fx * across[voxel] / depth[voxel] + camera.cx
        v = camera.fy * down[voxel] / depth[voxel] + camera.cy
        column = int(np.floor(u + 0.5))
        row = int(np.floor(v + 0.5))
        if 0 <= column < camera.width and 0 <= row < camera.height:
            pixels.add((row, column))
    return pixels


def setUpModule() -> None:
    global WORK
    root = Path(tempfile.mkdtemp(dir=TEST_ROOT))
    unittest.addModuleCleanup(shutil.rmtree, root, True)

    plan_path = root / "fixture.sftplan"
    plan_tsdf_blocks(load_scan_session(FIXTURE), plan_path, **PLAN_ARGUMENTS)
    plan = load_tsdf_block_plan(plan_path)
    session = load_scan_session(FIXTURE)
    storage, receipt = fused(plan, session)
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
    expanded_path = root / "room-expanded.sftplan"
    write_tsdf_expanded_block_plan(
        room.plan,
        propose_tsdf_plan_expansion_streaming(room.plan, room.session),
        expanded_path,
    )
    expanded_plan = load_tsdf_block_plan(expanded_path)
    expanded_storage, _ = fused(expanded_plan, room.session)

    WORK = SimpleNamespace(
        root=root,
        plan=plan,
        session=session,
        volume_path=volume_path,
        forged_path=forged_path,
        room=room,
        room_alone=fuse_tsdf_blocks_alone(
            room.session, room.plan.active_blocks, **options(room.plan)
        ),
        room_volume_path=room_volume_path,
        expanded_plan=expanded_plan,
        expanded_storage=expanded_storage,
    )


class EvidenceEqualsLoneBlockTests(unittest.TestCase):
    def test_every_fixture_block_from_its_evidence_is_the_block_alone(
        self,
    ) -> None:
        for block in FIXTURE_BLOCKS:
            with self.subTest(block=block):
                evidence = gather_tsdf_block_evidence(
                    WORK.session, block, frame_stride=1, **PLAN_ARGUMENTS
                )
                alone = fuse_tsdf_block_alone(
                    WORK.session, block, frame_stride=1, **PLAN_ARGUMENTS
                )
                self.assertEqual(
                    described(fuse_tsdf_block_from_evidence(evidence)),
                    described(alone),
                )
                # What the evidence is, written out. The image is two
                # pixels square, so its one tile is the whole of it.
                self.assertEqual(
                    evidence.camera,
                    TsdfPinholeCamera(2, 2, 2.0, 2.0, 0.5, 0.5),
                )
                self.assertEqual(evidence.depth_scale_m, 0.001)
                self.assertEqual(evidence.replay_digest_sha256, REPLAY_SHA256)
                self.assertEqual(evidence.tile_pixels, TSDF_DEPTH_TILE_PIXELS)
                self.assertEqual(evidence.tile_pixels, 16)
                self.assertEqual(
                    [frame.sequence for frame in evidence.frames], [0, 1]
                )
                for frame, raw in zip(
                    evidence.frames, (1000, 950), strict=True
                ):
                    self.assertTrue(frame.is_ready)
                    ((tile_row, tile_column, samples),) = frame.tiles.tiles
                    self.assertEqual((tile_row, tile_column), (0, 0))
                    self.assertEqual(samples.dtype, np.uint16)
                    self.assertEqual(samples.tolist(), [[raw, raw]] * 2)
                self.assertEqual(evidence.frames_with_tiles, 2)
                self.assertEqual(evidence.depth_tile_count, 2)
                self.assertEqual(evidence.depth_sample_count, 8)
                self.assertEqual(evidence.depth_byte_count, 16)
                self.assertEqual(evidence.scan_depth_byte_count, 16)

    def test_every_room_block_at_every_tile_size(self) -> None:
        room = WORK.room
        for tile_pixels in (1, 4, 16, 64):
            with self.subTest(tile_pixels=tile_pixels):
                gathered = gather_tsdf_blocks_evidence(
                    room.session,
                    room.plan.active_blocks,
                    tile_pixels=tile_pixels,
                    **options(room.plan),
                )
                self.assertEqual(len(gathered), 351)
                for evidence, alone in zip(
                    gathered, WORK.room_alone, strict=True
                ):
                    self.assertEqual(
                        described(fuse_tsdf_block_from_evidence(evidence)),
                        described(alone),
                    )
                # The tile size changes how much depth is kept and
                # nothing about which frames are read.
                self.assertEqual(
                    sum(e.depth_tile_count for e in gathered),
                    ROOM_TILES[tile_pixels],
                )
                self.assertEqual(
                    sum(e.depth_sample_count for e in gathered),
                    ROOM_SAMPLES[tile_pixels],
                )
                self.assertEqual(
                    sum(e.frames_with_tiles for e in gathered),
                    ROOM_FRAMES_READ,
                )
                self.assertEqual(
                    {e.scan_depth_byte_count for e in gathered},
                    {ROOM_SCAN_DEPTH_BYTES},
                )
        # At one pixel to a tile, a tile is a pixel read; at the width of
        # the image, a tile is a whole frame.
        self.assertEqual(ROOM_TILES[1], ROOM_SAMPLES[1])
        self.assertEqual(ROOM_TILES[64], ROOM_FRAMES_READ)
        self.assertEqual(ROOM_SAMPLES[64], ROOM_FRAMES_READ * 64 * 48)

    def test_a_block_no_plan_held_from_its_evidence(self) -> None:
        room = WORK.room
        added = tuple(
            block
            for block in WORK.expanded_plan.active_blocks
            if block not in set(room.plan.active_blocks)
        )
        self.assertEqual(len(added), 173)
        for evidence in gather_tsdf_blocks_evidence(
            room.session, added, **options(room.plan)
        ):
            lone = fuse_tsdf_block_from_evidence(evidence)
            row = WORK.expanded_storage.block_indices.index(lone.block_index)
            self.assertEqual(
                lone.tsdf_sums.tobytes(),
                WORK.expanded_storage.tsdf_sums[row].tobytes(),
            )
            self.assertEqual(
                lone.weights.tobytes(),
                WORK.expanded_storage.weights[row].tobytes(),
            )

    def test_the_pixels_kept_are_the_ones_a_second_account_finds(
        self,
    ) -> None:
        room = WORK.room
        camera, _ = _validate_reconstruction_contract(room.session)
        blocks = room.plan.active_blocks[::13]
        gathered = gather_tsdf_blocks_evidence(
            room.session, blocks, tile_pixels=1, **options(room.plan)
        )
        compared = 0
        for evidence in gathered:
            for frame in evidence.frames:
                expected = pixels_by_a_second_account(
                    camera,
                    frame.transform,
                    evidence.block_index,
                    room.plan.voxel_size_m,
                )
                kept = (
                    set()
                    if frame.tiles is None
                    else {
                        (tile_row, tile_column)
                        for tile_row, tile_column, _ in frame.tiles.tiles
                    }
                )
                self.assertEqual(kept, expected)
                compared += len(expected)
        # Not a comparison of nothing with nothing.
        self.assertGreater(compared, 20_000)
        self.assertEqual(len(blocks), 27)

    def test_blocks_gathered_together_each_get_their_own(self) -> None:
        room = WORK.room
        some = room.plan.active_blocks[::40]
        together = gather_tsdf_blocks_evidence(
            room.session, some, **options(room.plan)
        )
        self.assertEqual(
            tuple(evidence.block_index for evidence in together), some
        )
        for evidence in together:
            alone = gather_tsdf_block_evidence(
                room.session, evidence.block_index, **options(room.plan)
            )
            self.assertEqual(alone.depth_tile_count, evidence.depth_tile_count)
            for mine, theirs in zip(
                alone.frames, evidence.frames, strict=True
            ):
                self.assertEqual(mine.sequence, theirs.sequence)
                self.assertEqual(mine.transform, theirs.transform)
                self.assertEqual(mine.tiles is None, theirs.tiles is None)
                if mine.tiles is None:
                    continue
                self.assertEqual(
                    [
                        (row, column, samples.tobytes())
                        for row, column, samples in mine.tiles.tiles
                    ],
                    [
                        (row, column, samples.tobytes())
                        for row, column, samples in theirs.tiles.tiles
                    ],
                )

    def test_frames_without_depth_or_pose_carry_no_tiles_and_are_skipped(
        self,
    ) -> None:
        cases = (
            ("depth", ("streams/depth.jsonl",), False, True),
            ("pose", ("streams/poses.jsonl",), True, False),
            (
                "both",
                ("streams/depth.jsonl", "streams/poses.jsonl"),
                False,
                False,
            ),
        )
        for name, references, has_depth, has_pose in cases:
            with self.subTest(missing=name):
                with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
                    session_path = Path(temporary) / "case.vgsession"
                    shutil.copytree(FIXTURE, session_path)
                    for reference in references:
                        path = session_path / reference
                        lines = path.read_text(encoding="utf-8").splitlines()
                        path.write_text(
                            "".join(line + "\n" for line in lines[1:]),
                            encoding="utf-8",
                        )
                    session = load_scan_session(session_path)
                    gathered = gather_tsdf_blocks_evidence(
                        session, FIXTURE_BLOCKS, frame_stride=1,
                        **PLAN_ARGUMENTS,
                    )
                    alone = fuse_tsdf_blocks_alone(
                        session, FIXTURE_BLOCKS, frame_stride=1,
                        **PLAN_ARGUMENTS,
                    )
                for evidence, lone in zip(gathered, alone, strict=True):
                    first, second = evidence.frames
                    self.assertEqual(first.has_depth, has_depth)
                    self.assertEqual(first.transform is not None, has_pose)
                    self.assertFalse(first.is_ready)
                    self.assertIsNone(first.tiles)
                    self.assertTrue(second.is_ready)
                    self.assertEqual(
                        described(fuse_tsdf_block_from_evidence(evidence)),
                        described(lone),
                    )
                    self.assertEqual(lone.ready_observations, 1)
                    # A frame with no depth image adds nothing to what
                    # the scan holds: eight bytes a frame that has one.
                    self.assertEqual(
                        evidence.scan_depth_byte_count,
                        16 if has_depth else 8,
                    )

    def test_fusing_from_evidence_reads_no_session_and_opens_no_file(
        self,
    ) -> None:
        room = WORK.room
        gathered = gather_tsdf_blocks_evidence(
            room.session, room.plan.active_blocks[::50], **options(room.plan)
        )
        before = tree_snapshot(room.session_path)
        with forbidden_calls(_FORBIDDEN_WITHOUT_A_SESSION):
            blocks = [
                fuse_tsdf_block_from_evidence(evidence)
                for evidence in gathered
            ]
        self.assertEqual(tree_snapshot(room.session_path), before)
        self.assertEqual(len(blocks), 8)
        self.assertTrue(any(block.contributions_applied for block in blocks))

    def test_a_tile_and_a_frame_scale_to_the_same_metres(self) -> None:
        room = WORK.room
        camera, depth_scale_m = _validate_reconstruction_contract(room.session)
        observation = replay_session(room.session).observations[7]
        raw = _decode_raw_depth(
            room.session, observation, camera.width, camera.height
        )
        whole = _decode_metric_depth(
            room.session,
            observation,
            camera.width,
            camera.height,
            depth_scale_m,
        )
        self.assertEqual(raw.shape, (48, 64))
        self.assertEqual(
            _metric_depth(raw, depth_scale_m).tobytes(), whole.tobytes()
        )
        tile = raw[16:32, 48:64].astype(np.uint16)
        self.assertEqual(
            _metric_depth(tile, depth_scale_m).tobytes(),
            np.ascontiguousarray(whole[16:32, 48:64]).tobytes(),
        )


class EvidenceRefusalTests(unittest.TestCase):
    def gathered(self, step: int = 25, tile_pixels: int = 16):
        room = WORK.room
        return gather_tsdf_blocks_evidence(
            room.session,
            room.plan.active_blocks[::step],
            tile_pixels=tile_pixels,
            **options(room.plan),
        )

    def test_every_tile_gathered_is_one_the_block_cannot_do_without(
        self,
    ) -> None:
        removed = 0
        for evidence in self.gathered(tile_pixels=8):
            for frame_at, frame in enumerate(evidence.frames):
                if frame.tiles is None:
                    continue
                for tile_at in range(frame.tiles.tile_count):
                    with self.assertRaises(TsdfError) as raised:
                        fuse_tsdf_block_from_evidence(
                            without_tile(evidence, frame_at, tile_at)
                        )
                    message = str(raised.exception)
                    self.assertIn(
                        f"observation {frame.sequence}: the block reads "
                        "the depth pixel at row ",
                        message,
                    )
                    self.assertIn("holds no tile for it", message)
                    removed += 1
        # Fifteen blocks: each tile of each frame of each, 687 in all,
        # taken away one at a time.
        self.assertGreater(removed, 600)

    def test_depth_that_was_not_given_is_not_taken_for_empty(self) -> None:
        evidence = max(self.gathered(), key=lambda e: e.depth_tile_count)
        stripped = replace(
            evidence,
            frames=tuple(
                replace(frame, tiles=None) for frame in evidence.frames
            ),
        )
        self.assertEqual(stripped.depth_tile_count, 0)
        with self.assertRaises(TsdfError) as raised:
            fuse_tsdf_block_from_evidence(stripped)
        self.assertIn("holds no tile for it", str(raised.exception))
        # A block that reads nothing needs nothing, and is empty.
        nothing_read = min(self.gathered(), key=lambda e: e.depth_tile_count)
        if nothing_read.depth_tile_count == 0:
            self.assertEqual(
                fuse_tsdf_block_from_evidence(
                    nothing_read
                ).contributions_applied,
                0,
            )

    def test_one_blocks_evidence_does_not_serve_another(self) -> None:
        gathered = self.gathered(step=5)
        mismatched = 0
        for first, second in zip(gathered, gathered[::-1], strict=True):
            if first.block_index == second.block_index:
                continue
            borrowed = replace(first, block_index=second.block_index)
            try:
                lone = fuse_tsdf_block_from_evidence(borrowed)
            except TsdfError as error:
                self.assertIn("holds no tile for it", str(error))
                mismatched += 1
            else:
                # It may happen to read only pixels the other also
                # read. Then it is that block, correctly.
                genuine = fuse_tsdf_block_from_evidence(second)
                self.assertEqual(described(lone), described(genuine))
        self.assertGreater(mismatched, 30)

    def test_evidence_is_not_proof_that_it_is_the_scans(self) -> None:
        # Nothing in evidence says its depth is the scan's. Change the
        # depth and a different block comes out, with no complaint. That
        # is what a commitment is for, and this step has none.
        evidence = max(self.gathered(), key=lambda e: e.depth_tile_count)
        genuine = fuse_tsdf_block_from_evidence(evidence)
        self.assertGreater(genuine.contributions_applied, 0)

        def blanked(frame: TsdfBlockFrameEvidence):
            if frame.tiles is None:
                return frame
            tiles = []
            for tile_row, tile_column, samples in frame.tiles.tiles:
                empty = np.zeros_like(samples)
                empty.setflags(write=False)
                tiles.append((tile_row, tile_column, empty))
            return replace(frame, tiles=replace(frame.tiles, tiles=tuple(tiles)))

        blank = fuse_tsdf_block_from_evidence(
            replace(
                evidence,
                frames=tuple(blanked(frame) for frame in evidence.frames),
            )
        )
        self.assertEqual(blank.contributions_applied, 0)
        self.assertEqual(
            blank.evaluated_observations, genuine.evaluated_observations
        )

    def test_tiles_that_are_not_tiles_are_refused(self) -> None:
        def samples(shape, dtype=np.uint16, writable=False):
            array = np.zeros(shape, dtype=dtype)
            array.setflags(write=writable)
            return array

        good = ((0, 0, samples((16, 16))),)
        self.assertEqual(TsdfDepthTiles(64, 48, 16, good).byte_count, 512)
        # The image is 40 wide: its third column of tiles is 8 pixels.
        edge = TsdfDepthTiles(40, 48, 16, ((2, 2, samples((16, 8))),))
        self.assertEqual((edge.tile_rows, edge.tile_columns), (3, 3))
        cases = (
            ("no tile", (64, 48, 16, ()), "at least one tile"),
            ("a list", (64, 48, 16, [good[0]]), "at least one tile"),
            ("a pair", (64, 48, 16, ((0, 0),)), "a row, a column"),
            (
                "outside the image",
                (64, 48, 16, ((3, 0, samples((16, 16))),)),
                "outside the image",
            ),
            (
                "a negative tile",
                (64, 48, 16, ((0, -1, samples((16, 16))),)),
                "outside the image",
            ),
            (
                "out of order",
                (
                    64, 48, 16,
                    ((0, 1, samples((16, 16))), (0, 0, samples((16, 16)))),
                ),
                "increasing order",
            ),
            (
                "the same tile twice",
                (64, 48, 16, (good[0], good[0])),
                "increasing order",
            ),
            (
                "the wrong shape",
                (64, 48, 16, ((0, 0, samples((16, 8))),)),
                "shape (16, 16)",
            ),
            (
                "an edge tile at full size",
                (40, 48, 16, ((0, 2, samples((16, 16))),)),
                "shape (16, 8)",
            ),
            (
                "signed samples",
                (64, 48, 16, ((0, 0, samples((16, 16), np.int16)),)),
                "immutable uint16",
            ),
            (
                "writable samples",
                (64, 48, 16, ((0, 0, samples((16, 16), writable=True)),)),
                "immutable uint16",
            ),
            ("no tile size", (64, 48, 0, good), "from 1 to 1024"),
            ("a huge tile", (64, 48, 1025, good), "from 1 to 1024"),
            ("a true tile size", (64, 48, True, good), "from 1 to 1024"),
            ("no width", (0, 48, 16, good), "dimensions"),
        )
        for name, arguments, message in cases:
            with self.subTest(case=name):
                with self.assertRaises(TsdfError) as raised:
                    TsdfDepthTiles(*arguments)
                self.assertIn(message, str(raised.exception))
        self.assertEqual(MAX_TSDF_DEPTH_TILE_PIXELS, 1024)

    def test_frames_and_evidence_that_contradict_themselves_are_refused(
        self,
    ) -> None:
        evidence = max(self.gathered(), key=lambda e: e.depth_tile_count)
        read = next(frame for frame in evidence.frames if frame.tiles)
        frame_cases = {
            "tiles without a pose": {"transform": None},
            "tiles without depth": {"has_depth": False},
            "a short pose": {"transform": read.transform[:15]},
            "a pose that is not finite": {
                "transform": (float("nan"),) + read.transform[1:]
            },
            "a pose that is a list": {"transform": list(read.transform)},
            "a negative place": {"sequence": -1},
            "a flag that is a number": {"has_depth": 1},
            "tiles that are a tuple": {"tiles": read.tiles.tiles},
        }
        for name, changes in frame_cases.items():
            with self.subTest(frame=name):
                with self.assertRaises(TsdfError):
                    replace(read, **changes)

        smaller = np.zeros((8, 8), dtype=np.uint16)
        smaller.setflags(write=False)
        on_another_grid = replace(
            read, tiles=TsdfDepthTiles(64, 48, 8, ((0, 0, smaller),))
        )
        position = evidence.frames.index(read)
        evidence_cases = {
            "a frame missing": {"frames": evidence.frames[:-1]},
            "frames out of order": {"frames": evidence.frames[::-1]},
            "another stride": {"frame_stride": 2},
            "more frames than it says": {"total_observations": 19},
            "tiles from another grid": {
                "frames": evidence.frames[:position]
                + (on_another_grid,)
                + evidence.frames[position + 1:]
            },
            "another tile size": {"tile_pixels": 8},
            "truncation under a voxel": {"truncation_m": 0.01},
            "no depth scale": {"depth_scale_m": 0.0},
            "a camera that is a tuple": {"camera": (64, 48, 48.0, 48.0)},
            "a digest that is not one": {"replay_digest_sha256": "abc"},
            "no session": {"session_id": ""},
            "an index out of range": {"block_index": (0, 0, 2**31)},
            "frames that are a list": {"frames": list(evidence.frames)},
        }
        for name, changes in evidence_cases.items():
            with self.subTest(evidence=name):
                with self.assertRaises(TsdfError):
                    replace(evidence, **changes)
        with self.assertRaises(TsdfError):
            fuse_tsdf_block_from_evidence(None)  # type: ignore[arg-type]
        for name, arguments in (
            ("no width", (0, 48, 48.0, 48.0, 31.5, 23.5)),
            ("a focal length of zero", (64, 48, 0.0, 48.0, 31.5, 23.5)),
            ("a centre off the image", (64, 48, 48.0, 48.0, 64.0, 23.5)),
            ("a focal length of nan", (64, 48, 48.0, float("nan"), 1.0, 1.0)),
        ):
            with self.subTest(camera=name):
                with self.assertRaises(TsdfError):
                    TsdfPinholeCamera(*arguments)

    def test_gathering_refuses_what_it_cannot_gather(self) -> None:
        room = WORK.room
        block = room.plan.active_blocks[0]
        for name, changes, message in (
            ("no tile size", {"tile_pixels": 0}, "from 1 to 1024"),
            ("a tile too large", {"tile_pixels": 1025}, "from 1 to 1024"),
            ("a fraction of a pixel", {"tile_pixels": 1.5}, "from 1 to 1024"),
            ("stride zero", {"frame_stride": 0}, "frame_stride"),
        ):
            with self.subTest(case=name):
                with self.assertRaises(TsdfError) as raised:
                    gather_tsdf_block_evidence(
                        room.session,
                        block,
                        **{**options(room.plan), **changes},
                    )
                self.assertIn(message, str(raised.exception))
        with self.assertRaises(TsdfError) as raised:
            gather_tsdf_blocks_evidence(
                room.session, (block, block), **options(room.plan)
            )
        self.assertIn("a block twice", str(raised.exception))
        with self.assertRaises(TsdfError) as raised:
            gather_tsdf_block_evidence(None, block, **options(room.plan))
        self.assertIn("ScanSession", str(raised.exception))

        # Depth that does not fit the sixteen bits a tile holds.
        seen = max(WORK.room_alone, key=lambda lone: lone.contributions_applied)
        wide = np.full((48, 64), 70_000, dtype=np.int64)
        with patch(
            "spatialforge.tsdf_block_evidence._decode_raw_depth",
            return_value=wide,
        ):
            with self.assertRaises(TsdfError) as raised:
                gather_tsdf_block_evidence(
                    room.session, seen.block_index, **options(room.plan)
                )
        self.assertIn("does not fit 16 bits", str(raised.exception))


class CheckFromTilesTests(unittest.TestCase):
    def test_a_genuine_volume_and_a_forged_one(self) -> None:
        genuine = load_tsdf_block_volume(WORK.volume_path)
        pairs = check_tsdf_volume_blocks_from_tiles(
            genuine, WORK.session, genuine.block_indices
        )
        self.assertEqual(
            [check.status for check, _ in pairs],
            [TsdfBlockCheckStatus.IDENTICAL] * 8,
        )
        self.assertEqual(
            [evidence.block_index for _, evidence in pairs],
            list(FIXTURE_BLOCKS),
        )
        forged = load_tsdf_block_volume(WORK.forged_path)
        by_tiles = check_tsdf_volume_blocks_from_tiles(
            forged, WORK.session, forged.block_indices
        )
        by_frames = check_tsdf_volume_blocks(
            forged, WORK.session, forged.block_indices
        )
        # The two ways of fusing a block alone say the same of every
        # block, down to the voxel count.
        self.assertEqual(
            [(check.status, check.differing_voxels) for check, _ in by_tiles],
            [(check.status, check.differing_voxels) for check in by_frames],
        )
        self.assertEqual(
            [
                check.block_index
                for check, _ in by_tiles
                if check.status is TsdfBlockCheckStatus.DIFFERENT
            ],
            [(0, -1, -1)],
        )

    def test_a_sample_of_the_room_and_a_block_it_does_not_hold(self) -> None:
        volume = load_tsdf_block_volume(WORK.room_volume_path)
        missing = next(
            block
            for block in WORK.expanded_plan.active_blocks
            if block not in set(volume.block_indices)
        )
        blocks = volume.block_indices[::30] + (missing,)
        pairs = check_tsdf_volume_blocks_from_tiles(
            volume, WORK.room.session, blocks, tile_pixels=4
        )
        self.assertEqual(
            [check.status for check, _ in pairs],
            [TsdfBlockCheckStatus.IDENTICAL] * 12
            + [TsdfBlockCheckStatus.ABSENT],
        )
        absent_check, absent_evidence = pairs[-1]
        self.assertGreater(absent_check.lone.observed_voxel_count, 0)
        self.assertGreater(absent_evidence.depth_tile_count, 0)
        self.assertEqual(absent_evidence.tile_pixels, 4)

    def test_another_scan_is_refused_before_a_frame_is_opened(self) -> None:
        volume = load_tsdf_block_volume(WORK.volume_path)
        with self.assertRaises(TsdfError) as raised:
            check_tsdf_volume_blocks_from_tiles(
                volume, WORK.room.session, volume.block_indices
            )
        self.assertIn("was fused from session", str(raised.exception))
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            session_path = Path(temporary) / "changed.vgsession"
            shutil.copytree(FIXTURE, session_path)
            depth = sorted((session_path / "data" / "depth").iterdir())[0]
            text = depth.read_text(encoding="ascii")
            last = text.rstrip().split()[-1]
            depth.write_text(
                text.rstrip()[: -len(last)] + str(int(last) + 1) + "\n",
                encoding="ascii",
            )
            with patch(
                "spatialforge.tsdf_stream_fusion._read_depth_array",
                wraps=tsdf_stream_fusion._read_depth_array,
            ) as decoder:
                with self.assertRaises(TsdfError) as raised:
                    check_tsdf_volume_blocks_from_tiles(
                        volume,
                        load_scan_session(session_path),
                        volume.block_indices,
                    )
        self.assertIn(
            "not the scan the volume was fused from", str(raised.exception)
        )
        decoder.assert_not_called()


class BlockCheckTilesCliTests(unittest.TestCase):
    def run_cli(self, *arguments: str) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                code = main(["reconstruct", "tsdf-block-check", *arguments])
            except SystemExit as stop:
                code = stop.code
        return code, stdout.getvalue(), stderr.getvalue()

    def test_cli_fuses_from_tiles_and_says_how_much_depth_that_was(
        self,
    ) -> None:
        before = tree_snapshot(WORK.root)
        code, output, errors = self.run_cli(
            str(WORK.volume_path), str(FIXTURE),
            "--sample", "3", "--tiles", "16",
        )
        self.assertEqual(errors, "")
        self.assertEqual(code, 0)
        self.assertEqual(
            output,
            "TSDF BLOCK CHECK scan-synthetic-0001\n"
            "volume: verified\n"
            "session_replay: matched\n"
            "grid: voxel_size_m=0.125000000 truncation_m=0.500000000 "
            "frame_stride=1\n"
            "blocks: in_volume=8 named=0 sampled=3 sample_seed=0 "
            "checked=3\n"
            "frames: selected=2 ready=2\n"
            "frames_ruled_out_by_pose: yes\n"
            "depth_read_from: tiles of 16 x 16 pixels\n"
            "block[0]: index=(0, -1, -1) status=identical frames_seeing=2 "
            "frames_contributing=2 observed_voxels=44 contributions=88 "
            "frames_read=2 depth_tiles=2 depth_bytes=16\n"
            "block[1]: index=(1, -1, -1) status=identical frames_seeing=2 "
            "frames_contributing=2 observed_voxels=102 contributions=204 "
            "frames_read=2 depth_tiles=2 depth_bytes=16\n"
            "block[2]: index=(1, 0, -1) status=identical frames_seeing=2 "
            "frames_contributing=2 observed_voxels=102 contributions=204 "
            "frames_read=2 depth_tiles=2 depth_bytes=16\n"
            "identical: 3\n"
            "different: 0\n"
            "absent: 0\n"
            "frames_seeing_a_block: min=2 max=2\n"
            "observed_voxels_recomputed: 248\n"
            "contributions_recomputed: 496\n"
            "depth_tiles_read: 6\n"
            "depth_bytes_read: total=48 least=16 most=16\n"
            "scan_depth_bytes: 16\n"
            f"volume_sha256: {FIXTURE_VOLUME_SHA256}\n"
            f"replay_digest_sha256: {REPLAY_SHA256}\n"
            "volume_modified: no\n"
            "session_modified: no\n"
            "whole_volume_fused: no\n"
            "result: no checked block differs from what the scan gives\n",
        )
        self.assertEqual(tree_snapshot(WORK.root), before)

        # Without --tiles the report has no line about tiles: it is the
        # first step's report, unchanged.
        code, plain, _ = self.run_cli(
            str(WORK.volume_path), str(FIXTURE), "--sample", "3"
        )
        self.assertEqual(code, 0)
        self.assertNotIn("depth_", plain)
        self.assertNotIn("frames_read", plain)

    def test_cli_names_a_forged_block_from_tiles_too(self) -> None:
        code, output, errors = self.run_cli(
            str(WORK.forged_path), str(FIXTURE),
            "--block", "0", "-1", "-1", "--tiles", "1",
        )
        self.assertEqual(errors, "")
        self.assertEqual(code, 1)
        self.assertIn("depth_read_from: tiles of 1 x 1 pixels\n", output)
        self.assertRegex(
            output,
            r"block\[0\]: index=\(0, -1, -1\) status=different .* "
            r"frames_read=2 depth_tiles=\d+ depth_bytes=\d+ "
            r"differing_voxels=1\n",
        )
        self.assertIn("result: 1 of 1 checked blocks differ", output)

    def test_cli_refuses_tiles_with_every_frame_and_sizes_that_are_not(
        self,
    ) -> None:
        volume = str(WORK.volume_path)
        code, output, errors = self.run_cli(
            volume, str(FIXTURE), "--sample", "1",
            "--tiles", "16", "--evaluate-every-frame",
        )
        self.assertEqual(code, 2)
        self.assertEqual(output, "")
        self.assertIn("TSDF BLOCK CHECK FAILED", errors)
        self.assertIn(
            "cannot be combined with --evaluate-every-frame", errors
        )
        code, output, errors = self.run_cli(
            volume, str(FIXTURE), "--sample", "1", "--tiles", "2000"
        )
        self.assertEqual(code, 2)
        self.assertEqual(output, "")
        self.assertIn("from 1 to 1024", errors)
        code, _, _ = self.run_cli(
            volume, str(FIXTURE), "--sample", "1", "--tiles", "0"
        )
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
