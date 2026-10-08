"""Free-space expansion at real scale, held to three other accounts of it.

The streaming path approves every block of a candidate box that holds a
voxel some frame observed. Three things could be wrong with that: the
verdict on a block, the box, and the relation to the reference path it
replaces for real data. Each has its own check.

The verdict is compared with a second implementation written here, with a
matrix product and no shared code. The box is checked by growing it and
requiring nothing new, and its margin by building the worst case the
argument allows and measuring it. The reference is a different rule, so it
is not expected to agree everywhere: it must never approve a block this
path does not, and where this path approves more, the extra blocks must be
exactly the kind the reference's wedge cannot reach.
"""

from __future__ import annotations

import math
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from spatialforge import (
    allocate_empty_tsdf_blocks,
    fuse_tsdf_plan_streaming,
    load_tsdf_block_plan,
    verify_tsdf_block_plan_replay,
)
from spatialforge.errors import TsdfError
from spatialforge.point_cloud import (
    _sample_path,
    _validate_reconstruction_contract,
)
from spatialforge.replay import replay_session
from spatialforge.session_loader import load_scan_session
from spatialforge.tsdf_block_plan import (
    TSDF_EXPANSION_APPROVAL_RULES,
    TSDF_FREE_SPACE_RULE_FOOTPRINT,
    TSDF_FREE_SPACE_RULE_OBSERVED,
    plan_tsdf_blocks,
)
from spatialforge.tsdf_expanded_plan import write_tsdf_expanded_block_plan
from spatialforge.tsdf_plan_expansion import (
    propose_tsdf_plan_expansion_from_domain,
)
from spatialforge.tsdf_stream_expansion import (
    candidate_box_margin_m,
    propose_tsdf_plan_expansion_streaming,
)

from tests.heavy_fixtures import shared_case, shared_room_case

TEST_ROOT = Path(__file__).resolve().parent

# Raw depth and frame stride of the two-pixel-wide fixture. At 6000 one
# pixel is three metres across where it meets the surface.
REFERENCE_CASES = (
    (0, 1),
    (900, 2),
    (1500, 1),
    (3000, 2),
    (4500, 1),
    (5200, 2),
    (6000, 1),
)


def observed_blocks(plan, session, blocks):
    """Which blocks hold an observed voxel, and which hold one in free space.

    Written out again from the rule, sharing no code with the fusion
    evaluator, one block at a time. The rotation is spelled out term by
    term and not left to a matrix product: a BLAS is free to round that
    differently from one platform to the next, and a voxel that sits
    exactly on a threshold would then be judged differently too.
    """

    camera, depth_scale_m = _validate_reconstruction_contract(session)
    local = np.arange(8)
    z, y, x = np.meshgrid(local, local, local, indexing="ij")
    offsets = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    seen = np.zeros(len(blocks), dtype=bool)
    in_free_space = np.zeros(len(blocks), dtype=bool)
    for observation in replay_session(session).observations:
        if observation.sequence % plan.frame_stride:
            continue
        if observation.depth is None or observation.pose is None:
            continue
        pose = np.array(observation.pose.data["T_world_camera"]).reshape(4, 4)
        with Image.open(
            _sample_path(session, observation.depth.data, "depth")
        ) as image:
            depth = np.asarray(image, dtype=np.float64) * depth_scale_m
        for index, block in enumerate(blocks):
            centres = (
                np.asarray(block) * 8 + offsets + 0.5
            ) * plan.voxel_size_m
            offset_x = centres[:, 0] - pose[0, 3]
            offset_y = centres[:, 1] - pose[1, 3]
            offset_z = centres[:, 2] - pose[2, 3]
            across, down, depth_of_voxel = (
                pose[0, axis] * offset_x
                + pose[1, axis] * offset_y
                + pose[2, axis] * offset_z
                for axis in range(3)
            )
            ahead = depth_of_voxel > 0.0
            safe = np.where(ahead, depth_of_voxel, 1.0)
            column = np.floor(
                camera.fx * across / safe + camera.cx + 0.5
            ).astype(np.int64)
            row = np.floor(
                camera.fy * down / safe + camera.cy + 0.5
            ).astype(np.int64)
            inside = (
                ahead
                & (column >= 0)
                & (column < camera.width)
                & (row >= 0)
                & (row < camera.height)
            )
            measured = np.zeros(len(centres))
            measured[inside] = depth[row[inside], column[inside]]
            signed = measured - depth_of_voxel
            observed = inside & (measured > 0.0) & (
                signed >= -plan.truncation_m
            )
            seen[index] |= bool(observed.any())
            in_free_space[index] |= bool((observed & (signed >= 0.0)).any())
    return seen, in_free_space


def as_set(blocks, chosen) -> set:
    return {tuple(block) for block, keep in zip(blocks, chosen) if keep}


class AgainstTheReferenceTests(unittest.TestCase):
    def test_the_reference_lies_between_two_rules_and_this_is_the_wider(
        self,
    ) -> None:
        identical = 0
        for raw_depth, stride in REFERENCE_CASES:
            with self.subTest(raw_depth=raw_depth, stride=stride):
                case = shared_case(raw_depth, stride)
                session = load_scan_session(case.session_path)
                reference = propose_tsdf_plan_expansion_from_domain(
                    case.plan, case.domain
                )
                streaming = propose_tsdf_plan_expansion_streaming(
                    case.plan, session
                )
                self.assertEqual(
                    reference.free_space_rule, TSDF_FREE_SPACE_RULE_FOOTPRINT
                )
                self.assertEqual(
                    streaming.free_space_rule, TSDF_FREE_SPACE_RULE_OBSERVED
                )
                plan_blocks = set(case.plan.active_blocks)
                from_reference = set(reference.expanded_block_indices)
                from_streaming = set(streaming.expanded_block_indices)
                # Never fewer than the reference.
                self.assertLessEqual(from_reference, from_streaming)

                box = streaming.domain_block_indices
                seen, in_free_space = observed_blocks(case.plan, session, box)
                # The verdicts are the second implementation's verdicts.
                self.assertEqual(
                    set(streaming.approved_block_indices), as_set(box, seen)
                )
                # A block with an observed voxel in front of the surface is
                # inside some pixel's wedge, so the reference has it too.
                self.assertLessEqual(
                    as_set(box, in_free_space) | plan_blocks, from_reference
                )
                # Whatever this path adds beyond the reference holds only
                # voxels behind the surface, which the wedge stops short of.
                extra = from_streaming - from_reference
                self.assertFalse(extra & as_set(box, in_free_space))
                identical += from_reference == from_streaming
                if raw_depth == 6000:
                    self.assertEqual(
                        (len(from_reference), len(from_streaming)),
                        (161, 176),
                    )
        # Only the three-metre-wide pixel tells the two rules apart.
        self.assertEqual(identical, len(REFERENCE_CASES) - 1)

    def test_where_they_agree_only_the_rule_they_record_differs(self) -> None:
        case = shared_case(3000, 2)
        session = load_scan_session(case.session_path)
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            root = Path(temporary)
            write_tsdf_expanded_block_plan(
                case.plan,
                propose_tsdf_plan_expansion_from_domain(
                    case.plan, case.domain
                ),
                root / "reference.sftplan",
            )
            report = write_tsdf_expanded_block_plan(
                case.plan,
                propose_tsdf_plan_expansion_streaming(case.plan, session),
                root / "streaming.sftplan",
            )
            reference = load_tsdf_block_plan(root / "reference.sftplan")
            streaming = load_tsdf_block_plan(root / "streaming.sftplan")
            verify_tsdf_block_plan_replay(streaming, session)
            reference_text = (root / "reference.sftplan").read_text("ascii")
            streaming_text = (root / "streaming.sftplan").read_text("ascii")

        self.assertEqual(streaming.active_blocks, reference.active_blocks)
        self.assertEqual(len(streaming.active_blocks), 40)
        self.assertEqual(report.added_block_count, 8)
        self.assertEqual(report.free_space_rule, TSDF_FREE_SPACE_RULE_OBSERVED)
        self.assertEqual(
            streaming.free_space_rule, TSDF_FREE_SPACE_RULE_OBSERVED
        )
        self.assertEqual(
            reference.free_space_rule, TSDF_FREE_SPACE_RULE_FOOTPRINT
        )
        changed = [
            (old.strip(), new.strip())
            for old, new in zip(
                reference_text.splitlines(), streaming_text.splitlines()
            )
            if old != new
        ]
        self.assertEqual(
            changed,
            [
                (
                    '"free_space_rule": '
                    f'"{TSDF_FREE_SPACE_RULE_FOOTPRINT}",',
                    '"free_space_rule": '
                    f'"{TSDF_FREE_SPACE_RULE_OBSERVED}",',
                ),
                (
                    '"approval_rule": "'
                    + TSDF_EXPANSION_APPROVAL_RULES[
                        TSDF_FREE_SPACE_RULE_FOOTPRINT
                    ]
                    + '",',
                    '"approval_rule": "'
                    + TSDF_EXPANSION_APPROVAL_RULES[
                        TSDF_FREE_SPACE_RULE_OBSERVED
                    ]
                    + '",',
                ),
            ],
        )


class RoomScanTests(unittest.TestCase):
    """A scan the reference path cannot take: twenty 64x48 frames."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.room = shared_room_case()
        cls.proposal = propose_tsdf_plan_expansion_streaming(
            cls.room.plan, cls.room.session
        )

    def test_the_verdict_on_every_block_is_the_second_implementations(
        self,
    ) -> None:
        box = self.proposal.domain_block_indices
        seen, _ = observed_blocks(self.room.plan, self.room.session, box)
        self.assertEqual(
            set(self.proposal.approved_block_indices), as_set(box, seen)
        )
        # Not a trivial agreement: most of the box is empty, and most of
        # what is approved was not in the plan's halo for nothing.
        self.assertEqual(self.proposal.source_block_count, 351)
        self.assertEqual(self.proposal.added_block_count, 173)
        self.assertEqual(self.proposal.domain_block_count, 1584)
        self.assertGreater(self.proposal.rejected_block_count, 1000)
        self.assertEqual(
            self.proposal.expanded_block_count,
            351 + self.proposal.added_block_count,
        )

    def test_a_larger_box_finds_nothing_more(self) -> None:
        larger = propose_tsdf_plan_expansion_streaming(
            self.room.plan, self.room.session, extra_margin_blocks=2
        )
        self.assertGreater(
            larger.domain_block_count, self.proposal.domain_block_count
        )
        self.assertEqual(
            larger.approved_block_indices,
            self.proposal.approved_block_indices,
        )
        self.assertEqual(
            larger.expanded_block_indices,
            self.proposal.expanded_block_indices,
        )

    def test_the_expanded_plan_is_closed_under_fusion(self) -> None:
        room = self.room
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            path = Path(temporary) / "expanded.sftplan"
            write_tsdf_expanded_block_plan(room.plan, self.proposal, path)
            expanded_plan = load_tsdf_block_plan(path)
            verify_tsdf_block_plan_replay(expanded_plan, room.session)
            expanded = allocate_empty_tsdf_blocks(expanded_plan, room.session)
            expanded_receipt = fuse_tsdf_plan_streaming(
                expanded, room.session
            )
        base = allocate_empty_tsdf_blocks(room.plan, room.session)
        base_receipt = fuse_tsdf_plan_streaming(base, room.session)

        rows = {
            block: row for row, block in enumerate(expanded.block_indices)
        }
        # Blocks the plan already had are fused to the same bytes: adding
        # blocks changes nothing about the ones that were there.
        shared = [rows[block] for block in base.block_indices]
        self.assertEqual(
            expanded.tsdf_sums[shared].tobytes(), base.tsdf_sums.tobytes()
        )
        self.assertEqual(
            expanded.weights[shared].tobytes(), base.weights.tobytes()
        )
        # Every added block was added for a reason.
        added = [rows[block] for block in self.proposal.added_block_indices]
        self.assertEqual(len(added), 173)
        self.assertTrue(
            np.all(expanded.weights[added].reshape(len(added), -1).any(axis=1))
        )
        self.assertGreater(
            expanded_receipt.applied_count, base_receipt.applied_count
        )
        # And nothing was left outside: a block of the larger box that is
        # not in the expanded plan would receive no contribution at all.
        larger = propose_tsdf_plan_expansion_streaming(
            room.plan, room.session, extra_margin_blocks=1
        )
        outside = [
            block
            for block in larger.domain_block_indices
            if block not in rows
        ]
        self.assertGreater(len(outside), 1000)
        seen, _ = observed_blocks(room.plan, room.session, outside)
        self.assertFalse(seen.any())


class CandidateBoxTests(unittest.TestCase):
    CAMERA = SimpleNamespace(
        width=101, height=81, fx=100.0, fy=80.0, cx=50.0, cy=40.0
    )

    def test_the_margin_is_the_one_computed_by_hand(self) -> None:
        # The farthest corner is half the focal length away on each axis,
        # so the longest ray is sqrt(1.5). Half a pixel is half of
        # hypot(1/100, 1/80). With 0.06 m of truncation and a 5 m box:
        #   0.06 * 1.2247449 + 5.06 * 0.5 * 0.0160078 = 0.1139844
        self.assertAlmostEqual(
            candidate_box_margin_m(
                self.CAMERA, truncation_m=0.06, diagonal_m=5.0
            ),
            0.1139844,
            places=6,
        )

    def test_no_observed_voxel_is_farther_from_its_ray_than_the_margin(
        self,
    ) -> None:
        # The argument, built instead of trusted. Take a pixel, a measured
        # depth, and a voxel that pixel could observe: anywhere inside the
        # pixel's square, at any depth up to a truncation behind the
        # surface. Its distance from the segment between the camera and
        # the pixel's surface sample must not exceed the margin.
        camera = self.CAMERA
        truncation = 0.09
        diagonal = 6.0
        margin = candidate_box_margin_m(
            camera, truncation_m=truncation, diagonal_m=diagonal
        )
        generator = np.random.default_rng(83)
        worst = 0.0
        for _ in range(4000):
            column = int(generator.integers(0, camera.width))
            row = int(generator.integers(0, camera.height))
            ray = np.array(
                [
                    (column - camera.cx) / camera.fx,
                    (row - camera.cy) / camera.fy,
                    1.0,
                ]
            )
            # A depth whose sample is still inside a box of that diagonal.
            measured = generator.uniform(0.2, diagonal / np.linalg.norm(ray))
            sample = measured * ray
            # Extremes are where the bound is tight: the corners of the
            # pixel's square, at the far end of the band.
            depth = (
                measured + truncation
                if generator.random() < 0.5
                else generator.uniform(0.01, measured + truncation)
            )
            offset = generator.choice([-0.5, 0.5], size=2)
            voxel = depth * np.array(
                [
                    (column + offset[0] - camera.cx) / camera.fx,
                    (row + offset[1] - camera.cy) / camera.fy,
                    1.0,
                ]
            )
            along = np.clip(
                (voxel @ sample) / (sample @ sample), 0.0, 1.0
            )
            distance = float(np.linalg.norm(voxel - along * sample))
            worst = max(worst, distance)
            self.assertLessEqual(distance, margin)
        # The bound is not slack by an order of magnitude: the worst case
        # found uses a good part of it.
        self.assertGreater(worst, 0.5 * margin)


class GuardTests(unittest.TestCase):
    def test_a_plan_too_large_to_hold_is_refused(self) -> None:
        room = shared_room_case()
        with patch(
            "spatialforge.tsdf_stream_expansion.MAX_PLANNED_BLOCKS", 400
        ):
            with self.assertRaises(TsdfError) as caught:
                propose_tsdf_plan_expansion_streaming(room.plan, room.session)
        # Refused as soon as the count passes the ceiling, not after the
        # whole scan has been read: the room has twenty frames.
        message = str(caught.exception)
        self.assertIn("holds more than 400 blocks after ", message)
        self.assertIn(" of 20 frames", message)
        self.assertNotIn("after 20 of 20", message)
        self.assertIn("coarser voxel size", message)

    def test_a_box_too_large_to_consider_is_refused(self) -> None:
        room = shared_room_case()
        with patch(
            "spatialforge.tsdf_stream_expansion."
            "MAX_TSDF_STREAM_EXPANSION_CANDIDATE_BLOCKS",
            1000,
        ):
            with self.assertRaises(TsdfError) as caught:
                propose_tsdf_plan_expansion_streaming(room.plan, room.session)
        self.assertIn("1584 candidate blocks", str(caught.exception))

    def test_a_scan_that_changed_since_planning_is_refused(self) -> None:
        room = shared_room_case()
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            root = Path(temporary)
            session_path = root / "room.vgsession"
            shutil.copytree(room.session_path, session_path)
            plan_path = root / "room.sftplan"
            plan_tsdf_blocks(
                load_scan_session(session_path),
                plan_path,
                voxel_size_m=room.plan.voxel_size_m,
                truncation_m=room.plan.truncation_m,
            )
            plan = load_tsdf_block_plan(plan_path)
            depth = sorted((session_path / "data" / "depth").iterdir())[3]
            with Image.open(depth) as image:
                changed = np.asarray(image).copy()
            changed[0, 0] += 1
            Image.fromarray(changed).save(depth)
            with self.assertRaises(TsdfError) as caught:
                propose_tsdf_plan_expansion_streaming(
                    plan, load_scan_session(session_path)
                )
        self.assertIn("replay digest", str(caught.exception))

    def test_invalid_arguments_are_refused(self) -> None:
        room = shared_room_case()
        other = shared_case(3000, 2)
        for name, call, message in (
            (
                "no plan",
                lambda: propose_tsdf_plan_expansion_streaming(
                    None, room.session  # type: ignore
                ),
                "TsdfBlockPlan",
            ),
            (
                "no session",
                lambda: propose_tsdf_plan_expansion_streaming(
                    room.plan, None  # type: ignore
                ),
                "ScanSession",
            ),
            (
                "another scan's plan",
                lambda: propose_tsdf_plan_expansion_streaming(
                    other.plan, room.session
                ),
                "session_id",
            ),
            (
                "a negative margin",
                lambda: propose_tsdf_plan_expansion_streaming(
                    room.plan, room.session, extra_margin_blocks=-1
                ),
                "nonnegative",
            ),
        ):
            with self.subTest(case=name):
                with self.assertRaises(TsdfError) as caught:
                    call()
                self.assertIn(message, str(caught.exception))


if __name__ == "__main__":
    unittest.main()
