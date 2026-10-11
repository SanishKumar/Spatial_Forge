"""One block fused by itself, held to the fusion of the whole plan.

The claim under test is that a block of a volume is a function of the scan
and three numbers, and of nothing else: not of the plan, not of the other
blocks, not of whether frames that cannot reach it were opened. If that is
so, a block can be checked without fusing the volume it came from.

Every block of every fixture volume is therefore fused alone and required
to be, byte for byte, what streaming fusion put there, and on the committed
fixture what the voxel-by-voxel traversal put there, which shares no
vectorised code with either. Then the uses: a volume that has been altered
and re-digested so that it loads without complaint is caught, and the
block that was altered is named.
"""

from __future__ import annotations

import io
import shutil
import tempfile
import unittest
from contextlib import ExitStack, contextmanager, redirect_stderr
from contextlib import redirect_stdout
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from spatialforge import (
    MAX_TSDF_LONE_BLOCKS,
    TsdfBlockCheckStatus,
    TsdfLoneBlock,
    TsdfVolumeBlockCheck,
    allocate_empty_tsdf_blocks,
    build_tsdf_replay_depth_context,
    check_tsdf_volume_blocks,
    fuse_tsdf_block_alone,
    fuse_tsdf_blocks_alone,
    fuse_tsdf_plan_streaming,
    load_tsdf_block_plan,
    load_tsdf_block_volume,
    sample_tsdf_volume_blocks,
    write_tsdf_block_volume,
)
from spatialforge import tsdf_stream_fusion
from spatialforge.cli import main
from spatialforge.errors import TsdfError
from spatialforge.replay import replay_session
from spatialforge.session_loader import load_scan_session
from spatialforge.tsdf_block_plan import plan_tsdf_blocks
from spatialforge.tsdf_expanded_plan import write_tsdf_expanded_block_plan
from spatialforge.tsdf_plan_traversal import (
    traverse_tsdf_plan_blocks_from_context,
)
from spatialforge.tsdf_stream_expansion import (
    propose_tsdf_plan_expansion_streaming,
)

from tests.heavy_fixtures import shared_case, shared_room_case
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
# The numbers everything in this repository agrees on for the fixture.
FIXTURE_CONTRIBUTIONS = 1168
FIXTURE_OBSERVED_VOXELS = 584
ROOM_ACCEPTED_CONTRIBUTIONS = 1_127_112
ROOM_OBSERVED_VOXELS = 81_292
# Raw depth and stride of the rewritten two-pixel fixtures.
REWRITTEN_CASES = ((900, 2), (1500, 1), (3000, 2), (4500, 1), (6000, 1))

# Fusing a block alone plans nothing, allocates nothing, fuses no plan and
# builds no all-frames context.
_FORBIDDEN = (
    "spatialforge.tsdf_stream_fusion._fuse_stage",
    "spatialforge.tsdf_block_storage.allocate_empty_tsdf_blocks",
    "spatialforge.tsdf_block_plan.plan_tsdf_blocks",
    "spatialforge.tsdf_block_plan_loader.load_tsdf_block_plan",
    "spatialforge.tsdf_replay_depth_context.build_tsdf_replay_depth_context",
)

WORK: SimpleNamespace | None = None


def tree_snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


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


def fused(plan, session):
    storage = allocate_empty_tsdf_blocks(plan, session)
    receipt = fuse_tsdf_plan_streaming(storage, session)
    return storage, receipt


def alone(session, plan, blocks, **options):
    return fuse_tsdf_blocks_alone(
        session,
        blocks,
        voxel_size_m=plan.voxel_size_m,
        truncation_m=plan.truncation_m,
        frame_stride=plan.frame_stride,
        **options,
    )


def lone_bytes(lone: TsdfLoneBlock) -> tuple[bytes, bytes]:
    return lone.tsdf_sums.tobytes(), lone.weights.tobytes()


def stored_bytes(storage, block) -> tuple[bytes, bytes]:
    row = storage.block_indices.index(block)
    return storage.tsdf_sums[row].tobytes(), storage.weights[row].tobytes()


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

    # A volume that is wrong and says nothing about it: one observed
    # voxel's sum turned from in front of the surface to behind it, and
    # the file's own digests recomputed to match.
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
        storage=storage,
        volume_path=volume_path,
        forged_path=forged_path,
        forged_block=FIXTURE_BLOCKS[target // 512],
        room=room,
        room_storage=room_storage,
        room_volume_path=room_volume_path,
        expanded_plan=expanded_plan,
        expanded_storage=expanded_storage,
    )


class LoneBlockEqualsFusionTests(unittest.TestCase):
    def test_every_block_of_the_fixture_alone_is_the_fused_block(self) -> None:
        self.assertEqual(WORK.plan.active_blocks, FIXTURE_BLOCKS)
        for rule_out in (True, False):
            with self.subTest(rule_out_by_pose=rule_out):
                blocks = alone(
                    WORK.session,
                    WORK.plan,
                    FIXTURE_BLOCKS,
                    rule_out_by_pose=rule_out,
                )
                for lone in blocks:
                    self.assertEqual(
                        lone_bytes(lone),
                        stored_bytes(WORK.storage, lone.block_index),
                    )
                # The fixture's two numbers, arrived at one block at a
                # time by a path that never saw the plan.
                self.assertEqual(
                    sum(lone.contributions_applied for lone in blocks),
                    FIXTURE_CONTRIBUTIONS,
                )
                self.assertEqual(
                    sum(lone.observed_voxel_count for lone in blocks),
                    FIXTURE_OBSERVED_VOXELS,
                )
                self.assertEqual(
                    {lone.maximum_weight for lone in blocks}, {2}
                )
                self.assertEqual(
                    {lone.replay_digest_sha256 for lone in blocks},
                    {REPLAY_SHA256},
                )

    def test_it_is_the_voxel_by_voxel_traversal_too(self) -> None:
        # The traversal that defines a fused plan visits one voxel and one
        # frame at a time through the scalar evaluator.
        reference = allocate_empty_tsdf_blocks(WORK.plan, WORK.session)
        traverse_tsdf_plan_blocks_from_context(
            reference,
            build_tsdf_replay_depth_context(WORK.plan, WORK.session),
        )
        for block in FIXTURE_BLOCKS:
            lone = fuse_tsdf_block_alone(
                WORK.session, block, frame_stride=1, **PLAN_ARGUMENTS
            )
            self.assertEqual(lone_bytes(lone), stored_bytes(reference, block))

    def test_rewritten_fixtures_and_strides_agree_as_well(self) -> None:
        for raw_depth, stride in REWRITTEN_CASES:
            with self.subTest(raw_depth=raw_depth, stride=stride):
                case = shared_case(raw_depth, stride)
                session = load_scan_session(case.session_path)
                storage, receipt = fused(case.plan, session)
                blocks = alone(session, case.plan, case.plan.active_blocks)
                for lone in blocks:
                    self.assertEqual(
                        lone_bytes(lone),
                        stored_bytes(storage, lone.block_index),
                    )
                self.assertEqual(
                    sum(lone.contributions_applied for lone in blocks),
                    receipt.applied_count,
                )
                self.assertEqual(
                    blocks[0].selected_observations,
                    receipt.selected_observations,
                )

    def test_every_block_of_the_room_alone_is_the_fused_block(self) -> None:
        room = WORK.room
        for rule_out in (True, False):
            with self.subTest(rule_out_by_pose=rule_out):
                blocks = alone(
                    room.session,
                    room.plan,
                    room.plan.active_blocks,
                    rule_out_by_pose=rule_out,
                )
                self.assertEqual(len(blocks), 351)
                for lone in blocks:
                    self.assertEqual(
                        lone_bytes(lone),
                        stored_bytes(WORK.room_storage, lone.block_index),
                    )
                self.assertEqual(
                    sum(lone.contributions_applied for lone in blocks),
                    ROOM_ACCEPTED_CONTRIBUTIONS,
                )
                self.assertEqual(
                    sum(lone.observed_voxel_count for lone in blocks),
                    ROOM_OBSERVED_VOXELS,
                )

    def test_a_block_no_plan_held_is_what_a_larger_plan_fuses(self) -> None:
        # Blocks the surface plan never had, fused with no reference to
        # the plan that does have them.
        room = WORK.room
        added = tuple(
            block
            for block in WORK.expanded_plan.active_blocks
            if block not in set(room.plan.active_blocks)
        )
        self.assertEqual(len(added), 173)
        blocks = alone(room.session, room.plan, added)
        for lone in blocks:
            self.assertEqual(
                lone_bytes(lone),
                stored_bytes(WORK.expanded_storage, lone.block_index),
            )
        # Each was added for holding an observed voxel, and does.
        self.assertTrue(all(lone.observed_voxel_count for lone in blocks))

    def test_asking_for_other_blocks_changes_none_of_them(self) -> None:
        room = WORK.room
        some = room.plan.active_blocks[::29]
        together = alone(room.session, room.plan, some)
        self.assertEqual(
            tuple(lone.block_index for lone in together), some
        )
        backwards = alone(room.session, room.plan, some[::-1])
        for lone, mirrored in zip(together, backwards[::-1], strict=True):
            self.assertEqual(lone.block_index, mirrored.block_index)
            self.assertEqual(lone_bytes(lone), lone_bytes(mirrored))
        for lone in together[:4]:
            single = fuse_tsdf_block_alone(
                room.session,
                lone.block_index,
                voxel_size_m=room.plan.voxel_size_m,
                truncation_m=room.plan.truncation_m,
                frame_stride=room.plan.frame_stride,
            )
            self.assertEqual(lone_bytes(single), lone_bytes(lone))
            # Not only the accumulators: what it took is its own too.
            for field in (
                "ready_observations",
                "ruled_out_behind_camera",
                "ruled_out_outside_image",
                "evaluated_observations",
                "contributing_observations",
            ):
                self.assertEqual(
                    getattr(single, field), getattr(lone, field), field
                )

    def test_frames_that_cannot_reach_a_block_are_not_opened(self) -> None:
        room = WORK.room
        everything = alone(room.session, room.plan, room.plan.active_blocks)
        # The block fewest frames can reach.
        target = min(everything, key=lambda lone: lone.evaluated_observations)
        self.assertLess(
            target.evaluated_observations, target.ready_observations
        )
        self.assertEqual(target.ready_observations, 20)
        calls = {}
        results = {}
        for rule_out in (True, False):
            with patch(
                "spatialforge.tsdf_stream_fusion._read_depth_array",
                wraps=tsdf_stream_fusion._read_depth_array,
            ) as decoder:
                results[rule_out] = alone(
                    room.session,
                    room.plan,
                    (target.block_index,),
                    rule_out_by_pose=rule_out,
                )[0]
            calls[rule_out] = decoder.call_count
        self.assertEqual(calls[True], target.evaluated_observations)
        self.assertEqual(calls[False], 20)
        self.assertEqual(lone_bytes(results[True]), lone_bytes(results[False]))
        self.assertEqual(results[False].evaluated_observations, 20)
        self.assertEqual(results[False].ruled_out_observations, 0)
        self.assertEqual(
            results[True].ruled_out_observations,
            20 - target.evaluated_observations,
        )
        # Ruling out is by pose; it is not a guess at who contributes.
        # Every frame that gave the block anything was among those kept.
        self.assertEqual(
            results[True].contributing_observations,
            results[False].contributing_observations,
        )

    def test_a_block_nothing_reaches_is_empty_and_costs_no_frame(self) -> None:
        room = WORK.room
        with patch(
            "spatialforge.tsdf_stream_fusion._read_depth_array",
            wraps=tsdf_stream_fusion._read_depth_array,
        ) as decoder:
            far = alone(room.session, room.plan, ((1000, 1000, 1000),))[0]
        self.assertEqual(decoder.call_count, 0)
        self.assertEqual(far.evaluated_observations, 0)
        self.assertEqual(far.ruled_out_observations, 20)
        self.assertEqual(far.contributions_applied, 0)
        self.assertEqual(far.tsdf_sums.tobytes(), bytes(512 * 8))
        self.assertEqual(far.weights.tobytes(), bytes(512 * 4))
        # Opening every frame finds the same nothing.
        opened = alone(
            room.session,
            room.plan,
            ((1000, 1000, 1000),),
            rule_out_by_pose=False,
        )[0]
        self.assertEqual(opened.evaluated_observations, 20)
        self.assertEqual(lone_bytes(opened), lone_bytes(far))

    def test_frames_without_depth_or_pose_are_skipped_as_fusion_skips(
        self,
    ) -> None:
        cases = (
            ("depth", ("streams/depth.jsonl",)),
            ("pose", ("streams/poses.jsonl",)),
            ("both", ("streams/depth.jsonl", "streams/poses.jsonl")),
        )
        for name, references in cases:
            with self.subTest(missing=name):
                with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
                    root = Path(temporary)
                    session_path = root / "case.vgsession"
                    shutil.copytree(FIXTURE, session_path)
                    for reference in references:
                        path = session_path / reference
                        lines = path.read_text(encoding="utf-8").splitlines()
                        path.write_text(
                            "".join(line + "\n" for line in lines[1:]),
                            encoding="utf-8",
                        )
                    plan_path = root / "case.sftplan"
                    plan_tsdf_blocks(
                        load_scan_session(session_path),
                        plan_path,
                        **PLAN_ARGUMENTS,
                    )
                    plan = load_tsdf_block_plan(plan_path)
                    session = load_scan_session(session_path)
                    storage, receipt = fused(plan, session)
                    blocks = alone(session, plan, plan.active_blocks)
                self.assertEqual(receipt.fused_observations, 1)
                for lone in blocks:
                    self.assertEqual(lone.selected_observations, 2)
                    self.assertEqual(lone.ready_observations, 1)
                    self.assertEqual(
                        lone_bytes(lone),
                        stored_bytes(storage, lone.block_index),
                    )


class LoneBlockGuardTests(unittest.TestCase):
    def test_it_plans_nothing_fuses_no_plan_and_writes_nothing(self) -> None:
        before = tree_snapshot(FIXTURE)
        work_before = sorted(path.name for path in WORK.root.iterdir())
        with forbidden_calls(_FORBIDDEN):
            blocks = fuse_tsdf_blocks_alone(
                load_scan_session(FIXTURE),
                FIXTURE_BLOCKS,
                frame_stride=1,
                **PLAN_ARGUMENTS,
            )
        self.assertEqual(len(blocks), 8)
        self.assertEqual(tree_snapshot(FIXTURE), before)
        self.assertEqual(
            sorted(path.name for path in WORK.root.iterdir()), work_before
        )

    def test_invalid_arguments_are_refused(self) -> None:
        session = WORK.session
        good = {"frame_stride": 1, **PLAN_ARGUMENTS}
        cases = (
            ("not a session", (None, FIXTURE_BLOCKS), good, "ScanSession"),
            ("no block", (session, ()), good, "at least one block"),
            (
                "a block twice",
                (session, ((0, 0, 0), (0, 0, 0))),
                good,
                "a block twice",
            ),
            ("a string", (session, "000"), good, "sequence of block"),
            ("a list index", (session, ([0, 0, 0],)), good, "three integers"),
            (
                "a float index",
                (session, ((0.0, 0, 0),)),
                good,
                "three integers",
            ),
            (
                "a boolean index",
                (session, ((True, 0, 0),)),
                good,
                "three integers",
            ),
            (
                "an index outside the range",
                (session, ((2**31, 0, 0),)),
                good,
                "planning range",
            ),
            (
                "truncation under a voxel",
                (session, FIXTURE_BLOCKS),
                {"voxel_size_m": 0.5, "truncation_m": 0.125},
                "greater than or equal to voxel_size_m",
            ),
            (
                "no voxel size",
                (session, FIXTURE_BLOCKS),
                {"voxel_size_m": 0.0, "truncation_m": 0.5},
                "voxel_size_m",
            ),
            (
                "stride zero",
                (session, FIXTURE_BLOCKS),
                {**PLAN_ARGUMENTS, "frame_stride": 0},
                "frame_stride",
            ),
            (
                "a rule that is not a rule",
                (session, FIXTURE_BLOCKS),
                {**good, "rule_out_by_pose": 1},
                "true or false",
            ),
        )
        for name, arguments, options, message in cases:
            with self.subTest(case=name):
                with self.assertRaises(TsdfError) as raised:
                    fuse_tsdf_blocks_alone(*arguments, **options)
                self.assertIn(message, str(raised.exception))

    def test_more_blocks_than_the_limit_is_a_job_for_fusion(self) -> None:
        self.assertEqual(MAX_TSDF_LONE_BLOCKS, 4096)
        with patch("spatialforge.tsdf_lone_block.MAX_TSDF_LONE_BLOCKS", 3):
            with self.assertRaises(TsdfError) as raised:
                fuse_tsdf_blocks_alone(
                    WORK.session,
                    FIXTURE_BLOCKS[:4],
                    frame_stride=1,
                    **PLAN_ARGUMENTS,
                )
            three = fuse_tsdf_blocks_alone(
                WORK.session,
                FIXTURE_BLOCKS[:3],
                frame_stride=1,
                **PLAN_ARGUMENTS,
            )
        self.assertIn("the maximum is 3", str(raised.exception))
        self.assertEqual(len(three), 3)

    def test_a_scan_that_changes_while_it_is_read_is_refused(self) -> None:
        genuine = replay_session(WORK.session)
        changed = SimpleNamespace(
            digest_sha256="0" * 64, observations=genuine.observations
        )
        with patch(
            "spatialforge.tsdf_lone_block.replay_session",
            side_effect=(genuine, changed),
        ):
            with self.assertRaises(TsdfError) as raised:
                fuse_tsdf_block_alone(
                    WORK.session, (0, 0, 0), frame_stride=1, **PLAN_ARGUMENTS
                )
        self.assertIn("changed during", str(raised.exception))


class LoneBlockReceiptTests(unittest.TestCase):
    def lone(self) -> TsdfLoneBlock:
        return fuse_tsdf_block_alone(
            WORK.session, (1, 0, 0), frame_stride=1, **PLAN_ARGUMENTS
        )

    def test_it_is_frozen_and_its_arrays_cannot_be_written(self) -> None:
        lone = self.lone()
        with self.assertRaises(FrozenInstanceError):
            lone.frame_stride = 2  # type: ignore[misc]
        with self.assertRaises(ValueError):
            lone.tsdf_sums[0, 0, 0] = 1.0
        with self.assertRaises(ValueError):
            lone.weights[0, 0, 0] = 1
        self.assertEqual(lone.tsdf_sums.shape, (8, 8, 8))
        self.assertEqual(lone.contributions_applied, 204)
        self.assertEqual(lone.observed_voxel_count, 102)
        self.assertEqual(len(lone.tsdf_sums_sha256), 64)
        self.assertNotEqual(lone.tsdf_sums_sha256, lone.weights_sha256)

    def test_it_refuses_to_describe_what_could_not_have_happened(
        self,
    ) -> None:
        lone = self.lone()

        def frozen(array: np.ndarray) -> np.ndarray:
            array.setflags(write=False)
            return array

        observed = np.argwhere(lone.weights > 0)[0]
        unobserved = np.argwhere(lone.weights == 0)[0]
        heavy = lone.weights.copy()
        heavy[tuple(observed)] = 3
        outside = lone.tsdf_sums.copy()
        outside[tuple(observed)] = 5.0
        infinite = lone.tsdf_sums.copy()
        infinite[tuple(observed)] = np.inf
        negative_zero = lone.tsdf_sums.copy()
        negative_zero[tuple(unobserved)] = -0.0
        cases = {
            "frames unaccounted for": {"ready_observations": 1},
            "more ruled out than ready": {"ruled_out_behind_camera": 1},
            "contributing without evaluating": {
                "contributing_observations": 3
            },
            "no frame selected": {
                "selected_observations": 0,
                "ready_observations": 0,
                "evaluated_observations": 0,
                "contributing_observations": 0,
            },
            "a negative count": {"ruled_out_outside_image": -1},
            "a boolean count": {"selected_observations": True},
            "weight above its frames": {"weights": frozen(heavy)},
            "weight with no frame": {
                "contributing_observations": 0,
            },
            "sum outside its envelope": {"tsdf_sums": frozen(outside)},
            "a sum that is not a number": {"tsdf_sums": frozen(infinite)},
            "an unseen voxel at minus zero": {
                "tsdf_sums": frozen(negative_zero)
            },
            "writable sums": {"tsdf_sums": lone.tsdf_sums.copy()},
            "writable weights": {"weights": lone.weights.copy()},
            "flat sums": {"tsdf_sums": frozen(lone.tsdf_sums.reshape(512))},
            "weights of another type": {
                "weights": frozen(lone.weights.astype(np.int64))
            },
            "a digest that is not one": {"replay_digest_sha256": "abc"},
            "no session": {"session_id": ""},
            "an index out of range": {"block_index": (0, 0, 2**31)},
            "an index of two": {"block_index": (0, 0)},
        }
        for name, changes in cases.items():
            with self.subTest(case=name):
                with self.assertRaises(TsdfError):
                    replace(lone, **changes)
        # The same receipt, rebuilt unchanged, is accepted.
        self.assertEqual(
            lone_bytes(replace(lone, frame_stride=1)), lone_bytes(lone)
        )


class VolumeBlockCheckTests(unittest.TestCase):
    def test_every_block_of_a_genuine_volume_is_identical(self) -> None:
        volume = load_tsdf_block_volume(WORK.volume_path)
        self.assertEqual(volume.artifact_digest_sha256, FIXTURE_VOLUME_SHA256)
        checks = check_tsdf_volume_blocks(
            volume, WORK.session, volume.block_indices
        )
        self.assertEqual(
            [check.status for check in checks],
            [TsdfBlockCheckStatus.IDENTICAL] * 8,
        )
        self.assertEqual({check.differing_voxels for check in checks}, {0})
        self.assertEqual(
            {check.volume_digest_sha256 for check in checks},
            {FIXTURE_VOLUME_SHA256},
        )
        self.assertEqual(
            tuple(check.block_index for check in checks), FIXTURE_BLOCKS
        )

    def test_a_sample_of_the_room_volume_is_identical(self) -> None:
        volume = load_tsdf_block_volume(WORK.room_volume_path)
        blocks = sample_tsdf_volume_blocks(volume, 60, seed=11)
        checks = check_tsdf_volume_blocks(volume, WORK.room.session, blocks)
        self.assertEqual(len(checks), 60)
        self.assertTrue(
            all(
                check.status is TsdfBlockCheckStatus.IDENTICAL
                for check in checks
            )
        )
        # Blocks of a real-shaped scan are not all seen by every frame.
        seeing = {check.lone.evaluated_observations for check in checks}
        self.assertGreater(len(seeing), 3)

    def test_a_forgery_that_loads_is_caught_and_its_block_named(self) -> None:
        # The file's own digests were recomputed by whoever changed it,
        # and every re-derived check a loader makes still holds: total
        # weight, heaviest voxel, sums inside their envelope.
        forged = load_tsdf_block_volume(WORK.forged_path)
        genuine = load_tsdf_block_volume(WORK.volume_path)
        self.assertNotEqual(
            forged.artifact_digest_sha256, genuine.artifact_digest_sha256
        )
        self.assertEqual(
            forged.contributions_applied, genuine.contributions_applied
        )
        self.assertEqual(WORK.forged_block, (0, -1, -1))
        checks = check_tsdf_volume_blocks(
            forged, WORK.session, forged.block_indices
        )
        different = [
            check
            for check in checks
            if check.status is TsdfBlockCheckStatus.DIFFERENT
        ]
        self.assertEqual(len(different), 1)
        self.assertEqual(different[0].block_index, WORK.forged_block)
        self.assertEqual(different[0].differing_voxels, 1)
        self.assertEqual(
            sum(
                check.status is TsdfBlockCheckStatus.IDENTICAL
                for check in checks
            ),
            7,
        )
        # Asked about that one block and no other, it says the same.
        alone_check = check_tsdf_volume_blocks(
            forged, WORK.session, (WORK.forged_block,)
        )[0]
        self.assertIs(alone_check.status, TsdfBlockCheckStatus.DIFFERENT)
        # And the block it was fused into is what the scan gives.
        self.assertEqual(
            lone_bytes(alone_check.lone),
            stored_bytes(WORK.storage, WORK.forged_block),
        )

    def test_a_block_left_out_is_absent_and_the_scan_says_what_was_there(
        self,
    ) -> None:
        # The surface plan's volume does not hold the blocks expansion
        # adds. Leaving a block out is not a difference, and it cannot
        # hide what the scan would have put there from this check.
        volume = load_tsdf_block_volume(WORK.room_volume_path)
        missing = tuple(
            block
            for block in WORK.expanded_plan.active_blocks
            if block not in set(volume.block_indices)
        )[:5]
        held = volume.block_indices[:2]
        checks = check_tsdf_volume_blocks(
            volume, WORK.room.session, missing + held
        )
        self.assertEqual(
            [check.status for check in checks],
            [TsdfBlockCheckStatus.ABSENT] * 5
            + [TsdfBlockCheckStatus.IDENTICAL] * 2,
        )
        for check in checks[:5]:
            self.assertEqual(check.differing_voxels, 0)
            self.assertGreater(check.lone.observed_voxel_count, 0)

    def test_a_volume_is_only_checked_against_its_own_scan(self) -> None:
        volume = load_tsdf_block_volume(WORK.volume_path)
        room_volume = load_tsdf_block_volume(WORK.room_volume_path)
        with self.assertRaises(TsdfError) as raised:
            check_tsdf_volume_blocks(
                volume, WORK.room.session, volume.block_indices
            )
        self.assertIn("was fused from session", str(raised.exception))
        with self.assertRaises(TsdfError) as raised:
            check_tsdf_volume_blocks(
                room_volume, WORK.session, room_volume.block_indices[:1]
            )
        self.assertIn("was fused from session", str(raised.exception))

        # The same session, with one depth sample different since.
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
                "spatialforge.tsdf_stream_fusion._read_depth_array"
            ) as decoder:
                with self.assertRaises(TsdfError) as raised:
                    check_tsdf_volume_blocks(
                        volume,
                        load_scan_session(session_path),
                        volume.block_indices,
                    )
        self.assertIn(
            "not the scan the volume was fused from", str(raised.exception)
        )
        # Refused before a frame was opened.
        decoder.assert_not_called()

        for name, arguments, message in (
            ("no volume", (None, WORK.session), "TsdfBlockVolume"),
            ("no session", (volume, None), "ScanSession"),
        ):
            with self.subTest(case=name):
                with self.assertRaises(TsdfError) as raised:
                    check_tsdf_volume_blocks(*arguments, ((0, 0, 0),))
                self.assertIn(message, str(raised.exception))

    def test_a_check_cannot_say_two_things_at_once(self) -> None:
        volume = load_tsdf_block_volume(WORK.volume_path)
        check = check_tsdf_volume_blocks(volume, WORK.session, ((0, 0, 0),))[0]
        cases = {
            "identical with differences": {"differing_voxels": 3},
            "different with none": {
                "status": TsdfBlockCheckStatus.DIFFERENT
            },
            "absent with differences": {
                "status": TsdfBlockCheckStatus.ABSENT,
                "differing_voxels": 1,
            },
            "more than a block holds": {
                "status": TsdfBlockCheckStatus.DIFFERENT,
                "differing_voxels": 513,
            },
            "a status that is a word": {"status": "identical"},
            "a digest that is not one": {"volume_digest_sha256": "x"},
            "no lone block": {"lone": None},
        }
        for name, changes in cases.items():
            with self.subTest(case=name):
                with self.assertRaises(TsdfError):
                    replace(check, **changes)
        self.assertIsInstance(
            replace(check, differing_voxels=0), TsdfVolumeBlockCheck
        )


class BlockSampleTests(unittest.TestCase):
    def test_the_draw_is_the_same_on_every_machine(self) -> None:
        # Pinned literally: the draw is a SHA-256 of the seed and the
        # block's index, so this holds wherever the suite runs.
        volume = load_tsdf_block_volume(WORK.volume_path)
        self.assertEqual(
            sample_tsdf_volume_blocks(volume, 1), ((1, 0, -1),)
        )
        self.assertEqual(
            sample_tsdf_volume_blocks(volume, 3, seed=0),
            ((0, -1, -1), (1, -1, -1), (1, 0, -1)),
        )
        self.assertEqual(
            sample_tsdf_volume_blocks(volume, 3, seed=1),
            ((0, -1, -1), (0, 0, -1), (1, 0, -1)),
        )
        self.assertEqual(
            sample_tsdf_volume_blocks(volume, 8, seed=5), FIXTURE_BLOCKS
        )

    def test_it_is_distinct_canonical_and_grows_with_its_count(self) -> None:
        volume = load_tsdf_block_volume(WORK.room_volume_path)
        forty = sample_tsdf_volume_blocks(volume, 40, seed=3)
        self.assertEqual(len(set(forty)), 40)
        self.assertTrue(set(forty) <= set(volume.block_indices))
        # Canonical block order: z, then y, then x.
        self.assertEqual(
            list(forty),
            sorted(forty, key=lambda block: (block[2], block[1], block[0])),
        )
        # A larger draw of the same seed contains the smaller one.
        self.assertTrue(
            set(sample_tsdf_volume_blocks(volume, 10, seed=3)) <= set(forty)
        )
        self.assertNotEqual(
            sample_tsdf_volume_blocks(volume, 40, seed=4), forty
        )

    def test_impossible_draws_are_refused(self) -> None:
        volume = load_tsdf_block_volume(WORK.volume_path)
        for name, arguments, options, message in (
            ("no volume", (None, 1), {}, "TsdfBlockVolume"),
            ("none", (volume, 0), {}, "positive integer"),
            ("a fraction", (volume, 1.5), {}, "positive integer"),
            ("more than there are", (volume, 9), {}, "that holds 8"),
            ("a negative seed", (volume, 1), {"seed": -1}, "nonnegative"),
            ("a seed that is true", (volume, 1), {"seed": True}, "nonneg"),
        ):
            with self.subTest(case=name):
                with self.assertRaises(TsdfError) as raised:
                    sample_tsdf_volume_blocks(*arguments, **options)
                self.assertIn(message, str(raised.exception))
        with patch("spatialforge.tsdf_lone_block.MAX_TSDF_LONE_BLOCKS", 4):
            with self.assertRaises(TsdfError) as raised:
                sample_tsdf_volume_blocks(volume, 5)
        self.assertIn("at most 4", str(raised.exception))


class BlockCheckCliTests(unittest.TestCase):
    def run_cli(self, *arguments: str) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                code = main(["reconstruct", "tsdf-block-check", *arguments])
            except SystemExit as stop:
                code = stop.code
        return code, stdout.getvalue(), stderr.getvalue()

    def block_lines(self, differing: bool) -> str:
        lines = []
        for position, block in enumerate(FIXTURE_BLOCKS):
            observed, applied = (102, 204) if block[0] == 1 else (44, 88)
            wrong = differing and block == (0, -1, -1)
            lines.append(
                f"block[{position}]: index={block} "
                f"status={'different' if wrong else 'identical'} "
                "frames_seeing=2 frames_contributing=2 "
                f"observed_voxels={observed} contributions={applied}"
                + (" differing_voxels=1" if wrong else "")
                + "\n"
            )
        return "".join(lines)

    def test_cli_checks_the_fixture_volume_and_says_what_it_did(self) -> None:
        before = tree_snapshot(WORK.root)
        fixture_before = tree_snapshot(FIXTURE)
        code, output, errors = self.run_cli(
            str(WORK.volume_path), str(FIXTURE), "--sample", "8"
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
            "blocks: in_volume=8 named=0 sampled=8 sample_seed=0 "
            "checked=8\n"
            "frames: selected=2 ready=2\n"
            "frames_ruled_out_by_pose: yes\n"
            + self.block_lines(differing=False)
            + "identical: 8\n"
            "different: 0\n"
            "absent: 0\n"
            "frames_seeing_a_block: min=2 max=2\n"
            f"observed_voxels_recomputed: {FIXTURE_OBSERVED_VOXELS}\n"
            f"contributions_recomputed: {FIXTURE_CONTRIBUTIONS}\n"
            f"volume_sha256: {FIXTURE_VOLUME_SHA256}\n"
            f"replay_digest_sha256: {REPLAY_SHA256}\n"
            "volume_modified: no\n"
            "session_modified: no\n"
            "whole_volume_fused: no\n"
            "result: no checked block differs from what the scan gives\n",
        )
        self.assertEqual(tree_snapshot(WORK.root), before)
        self.assertEqual(tree_snapshot(FIXTURE), fixture_before)

        # Opening every frame is the same report but for the line that
        # says so.
        code, every_frame, _ = self.run_cli(
            str(WORK.volume_path),
            str(FIXTURE),
            "--sample",
            "8",
            "--evaluate-every-frame",
        )
        self.assertEqual(code, 0)
        self.assertEqual(
            every_frame,
            output.replace(
                "frames_ruled_out_by_pose: yes", "frames_ruled_out_by_pose: no"
            ),
        )

    def test_cli_exits_1_on_a_forged_volume_and_names_the_block(self) -> None:
        code, output, errors = self.run_cli(
            str(WORK.forged_path), str(FIXTURE), "--sample", "8"
        )
        self.assertEqual(errors, "")
        self.assertEqual(code, 1)
        self.assertIn(self.block_lines(differing=True), output)
        self.assertIn("identical: 7\ndifferent: 1\nabsent: 0\n", output)
        self.assertTrue(
            output.endswith(
                "whole_volume_fused: no\n"
                "result: 1 of 8 checked blocks differ from what the scan "
                "gives\n"
            )
        )
        # Named, the one block is enough.
        code, output, _ = self.run_cli(
            str(WORK.forged_path), str(FIXTURE), "--block", "0", "-1", "-1"
        )
        self.assertEqual(code, 1)
        self.assertIn(
            "blocks: in_volume=8 named=1 sampled=0 sample_seed=none "
            "checked=1\n",
            output,
        )
        self.assertIn("result: 1 of 1 checked blocks differ", output)

    def test_cli_takes_named_and_drawn_blocks_together(self) -> None:
        code, output, errors = self.run_cli(
            str(WORK.volume_path),
            str(FIXTURE),
            "--block", "1", "0", "-1",
            "--block", "7", "7", "7",
            "--sample", "3",
        )
        self.assertEqual(errors, "")
        self.assertEqual(code, 0)
        # The draw of three includes the first named block, which is
        # checked once.
        self.assertIn(
            "blocks: in_volume=8 named=2 sampled=3 sample_seed=0 "
            "checked=4\n",
            output,
        )
        self.assertIn(
            "block[0]: index=(1, 0, -1) status=identical", output
        )
        self.assertIn(
            "block[1]: index=(7, 7, 7) status=absent frames_seeing=0 "
            "frames_contributing=0 observed_voxels=0 contributions=0\n",
            output,
        )
        self.assertIn("identical: 3\ndifferent: 0\nabsent: 1\n", output)
        self.assertIn("frames_seeing_a_block: min=0 max=2\n", output)

    def test_cli_failures_are_exit_2_and_say_what_to_do(self) -> None:
        volume = str(WORK.volume_path)
        fixture = str(FIXTURE)
        cases = (
            (
                "nothing to check",
                (volume, fixture),
                "name a block with --block or draw some with --sample",
            ),
            (
                "the same block twice",
                (volume, fixture, "--block", "0", "0", "0",
                 "--block", "0", "0", "0"),
                "--block was given the same block twice",
            ),
            (
                "a draw larger than the volume",
                (volume, fixture, "--sample", "9"),
                "cannot sample 9 blocks",
            ),
            (
                "a negative seed",
                (volume, fixture, "--sample", "1", "--seed", "-3"),
                "nonnegative",
            ),
            (
                "another scan",
                (volume, str(WORK.room.session_path), "--sample", "1"),
                "was fused from session",
            ),
            (
                "a plan in place of a volume",
                (str(WORK.root / "fixture.sftplan"), fixture,
                 "--sample", "1"),
                ".sftvol",
            ),
            (
                "no such volume",
                (str(WORK.root / "missing.sftvol"), fixture,
                 "--sample", "1"),
                "TSDF BLOCK CHECK FAILED",
            ),
            (
                "no such session",
                (volume, str(WORK.root / "missing.vgsession"),
                 "--sample", "1"),
                "TSDF BLOCK CHECK FAILED",
            ),
        )
        for name, arguments, message in cases:
            with self.subTest(case=name):
                code, output, errors = self.run_cli(*arguments)
                self.assertEqual(code, 2)
                self.assertEqual(output, "")
                self.assertIn("TSDF BLOCK CHECK FAILED", errors)
                self.assertIn(message, errors)
                self.assertNotIn("Traceback", errors)
        # What argparse itself refuses.
        for arguments in (
            (volume, fixture, "--sample", "0"),
            (volume, fixture, "--block", "0", "0"),
            (volume,),
        ):
            with self.subTest(arguments=arguments[1:]):
                code, _, _ = self.run_cli(*arguments)
                self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
