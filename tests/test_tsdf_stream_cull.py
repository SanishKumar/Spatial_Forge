"""Skipping blocks a frame cannot see, held to the per-voxel evaluator.

Streaming fusion settles some blocks without evaluating them: all 512
voxels behind the camera, or all 512 projecting outside the image. That is
only an optimisation if it is never wrong, and "never" has to include a
block that straddles the camera plane by a hair or sits one pixel off the
edge of the image.

So the block verdict is compared with what the evaluator says about every
voxel of the same block, over thousands of poses chosen to put blocks on
those boundaries, and whole fusions are run both ways and compared byte for
byte, receipt included.
"""

from __future__ import annotations

import json
import math
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from spatialforge import allocate_empty_tsdf_blocks, load_tsdf_block_plan
from spatialforge import tsdf_stream_fusion
from spatialforge.point_cloud import _validate_reconstruction_contract
from spatialforge.session_loader import load_scan_session
from spatialforge.tsdf_block_contributions import (
    _STATUS_CODE,
    _evaluate_ready_voxels,
)
from spatialforge.tsdf_block_plan import plan_tsdf_blocks
from spatialforge.tsdf_stream_fusion import (
    BLOCK_BEHIND_CAMERA,
    BLOCK_EVALUATE,
    BLOCK_OUTSIDE_IMAGE,
    _chunk_voxel_centres_world_m,
    _classify_blocks,
    fuse_tsdf_plan_streaming,
)
from spatialforge.tsdf_voxel_contribution import TsdfContributionStatus

from tests.heavy_fixtures import ROOM_PLAN_ARGUMENTS, shared_room_case
from tests.room_fixture import room_session

TEST_ROOT = Path(__file__).resolve().parent
BEHIND = _STATUS_CODE[TsdfContributionStatus.CAMERA_Z_NONPOSITIVE]
OUTSIDE = _STATUS_CODE[TsdfContributionStatus.PROJECTION_OUTSIDE_IMAGE]


def room_camera():
    camera, _ = _validate_reconstruction_contract(
        load_scan_session(room_session())
    )
    return camera


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    quaternion = rng.normal(size=4)
    x, y, z, w = quaternion / np.linalg.norm(quaternion)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def transform_from(rotation: np.ndarray, position: np.ndarray) -> tuple:
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = position
    return tuple(float(value) for value in matrix.ravel())


def voxel_statuses(camera, transform, block, voxel_size_m) -> np.ndarray:
    depth_m = np.full((camera.height, camera.width), 1.5)
    codes, _, _ = _evaluate_ready_voxels(
        camera,
        transform,
        depth_m,
        3.0 * voxel_size_m,
        _chunk_voxel_centres_world_m(block[None, :], voxel_size_m),
    )
    return codes


class WholeBlockVerdictTests(unittest.TestCase):
    def check(self, camera, transform, block, voxel_size_m) -> int:
        """The verdict, after holding it to all 512 voxels of the block."""

        verdict = int(
            _classify_blocks(camera, transform, block[None, :], voxel_size_m)[
                0
            ]
        )
        if verdict != BLOCK_EVALUATE:
            expected = BEHIND if verdict == BLOCK_BEHIND_CAMERA else OUTSIDE
            statuses = voxel_statuses(camera, transform, block, voxel_size_m)
            self.assertTrue(
                bool(np.all(statuses == expected)),
                (verdict, np.unique(statuses), transform, block),
            )
        return verdict

    def test_a_verdict_is_what_every_voxel_would_have_got(self) -> None:
        rng = np.random.default_rng(31)
        small = room_camera()
        cameras = (
            small,
            replace(
                small, width=640, height=480, fx=525.0, fy=525.0,
                cx=319.5, cy=239.5,
            ),
            replace(
                small, width=640, height=480, fx=481.2, fy=480.0,
                cx=300.0, cy=260.0,
            ),
        )
        counts = {BLOCK_EVALUATE: 0, BLOCK_BEHIND_CAMERA: 0, BLOCK_OUTSIDE_IMAGE: 0}
        for _ in range(3_000):
            camera = cameras[int(rng.integers(len(cameras)))]
            voxel_size_m = float(rng.choice([0.01, 0.04, 0.125]))
            block = rng.integers(-30, 31, size=3)
            position = rng.uniform(-4.0, 4.0, size=3)
            verdict = self.check(
                camera,
                transform_from(random_rotation(rng), position),
                block,
                voxel_size_m,
            )
            counts[verdict] += 1
        # The test settles a real share of blocks either way, and leaves a
        # real share to be evaluated.
        for verdict, count in counts.items():
            self.assertGreater(count, 150, (verdict, counts))

    def test_blocks_on_the_camera_plane_and_the_image_edge(self) -> None:
        # Poses built around the block itself: the camera a few block
        # widths from it, looking so that the block lands near the plane
        # z = 0 or near the border of the image, where a verdict given
        # without slack would sometimes be wrong.
        rng = np.random.default_rng(32)
        camera = replace(
            room_camera(), width=640, height=480, fx=525.0, fy=525.0,
            cx=319.5, cy=239.5,
        )
        half_fov = math.atan(320.0 / 525.0)
        counts = {BLOCK_EVALUATE: 0, BLOCK_BEHIND_CAMERA: 0, BLOCK_OUTSIDE_IMAGE: 0}
        for _ in range(3_000):
            voxel_size_m = float(rng.choice([0.01, 0.04]))
            block = rng.integers(-5, 6, size=3)
            extent = 8 * voxel_size_m
            centre = (block + 0.5) * extent
            offset = rng.normal(size=3)
            offset *= rng.uniform(0.2, 12.0) * extent / np.linalg.norm(offset)
            position = centre + offset
            # Look at an angle from the block that is close to either the
            # edge of the field of view or ninety degrees.
            to_block = (centre - position) / np.linalg.norm(centre - position)
            angle = float(
                rng.choice([half_fov, math.pi / 2.0])
                + rng.normal(scale=0.2)
            )
            side = np.cross(to_block, rng.normal(size=3))
            side /= np.linalg.norm(side)
            forward = math.cos(angle) * to_block + math.sin(angle) * side
            right = np.cross(forward, rng.normal(size=3))
            right /= np.linalg.norm(right)
            down = np.cross(forward, right)
            rotation = np.stack([right, down, forward], axis=1)
            counts[
                self.check(
                    camera,
                    transform_from(rotation, position),
                    block,
                    voxel_size_m,
                )
            ] += 1
        for verdict, count in counts.items():
            self.assertGreater(count, 100, (verdict, counts))

    def test_a_block_a_voxel_can_see_is_never_settled(self) -> None:
        # The other direction, stated on its own: whenever any voxel of a
        # block gets past the two visibility gates, the verdict is to
        # evaluate.
        rng = np.random.default_rng(33)
        camera = room_camera()
        seen = 0
        for _ in range(2_000):
            block = rng.integers(-6, 7, size=3)
            transform = transform_from(
                random_rotation(rng), rng.uniform(-1.5, 1.5, size=3)
            )
            statuses = voxel_statuses(camera, transform, block, 0.04)
            if np.any((statuses != BEHIND) & (statuses != OUTSIDE)):
                seen += 1
                self.assertEqual(
                    int(
                        _classify_blocks(
                            camera, transform, block[None, :], 0.04
                        )[0]
                    ),
                    BLOCK_EVALUATE,
                )
        self.assertGreater(seen, 100)

    def test_clear_cases_are_settled(self) -> None:
        camera = room_camera()
        looking_along_x = transform_from(
            np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]),
            np.zeros(3),
        )
        blocks = np.array(
            [
                [10, 0, 0],    # straight ahead
                [-10, 0, 0],   # straight behind
                [3, 40, 0],    # in front, far off to the side
                [3, 0, 40],    # in front, far above
                [0, 0, 0],     # the camera is inside it
            ]
        )
        self.assertEqual(
            _classify_blocks(camera, looking_along_x, blocks, 0.04).tolist(),
            [
                BLOCK_EVALUATE,
                BLOCK_BEHIND_CAMERA,
                BLOCK_OUTSIDE_IMAGE,
                BLOCK_OUTSIDE_IMAGE,
                BLOCK_EVALUATE,
            ],
        )

    def test_a_camera_outside_the_analysed_bounds_settles_nothing(
        self,
    ) -> None:
        camera = room_camera()
        transform = transform_from(np.eye(3), np.zeros(3))
        blocks = np.array([[0, 0, -50], [0, 0, 50], [60, 0, 5]])
        self.assertIn(
            BLOCK_BEHIND_CAMERA,
            _classify_blocks(camera, transform, blocks, 0.04).tolist(),
        )
        for odd in (
            replace(camera, fx=5.0e4),
            replace(camera, fy=5.0e4),
            replace(camera, width=200_000),
            replace(camera, cx=-3.0e5),
        ):
            self.assertEqual(
                _classify_blocks(odd, transform, blocks, 0.04).tolist(),
                [BLOCK_EVALUATE] * 3,
            )

    def test_a_grid_too_large_to_reason_about_settles_nothing(self) -> None:
        camera = room_camera()
        transform = transform_from(np.eye(3), np.zeros(3))
        # The first block's coordinates overflow; the second's are merely
        # enormous, and it really is behind the camera.
        blocks = np.array([[0, 0, -(2**40)], [0, 0, -50]])
        verdicts = _classify_blocks(camera, transform, blocks, 1e300)
        self.assertEqual(
            verdicts.tolist(), [BLOCK_EVALUATE, BLOCK_BEHIND_CAMERA]
        )
        self.assertEqual(
            self.check(camera, transform, blocks[1], 1e300),
            BLOCK_BEHIND_CAMERA,
        )


def storage_bytes(storage) -> tuple[bytes, bytes]:
    return storage.tsdf_sums.tobytes(), storage.weights.tobytes()


def fuse_both_ways(plan, session):
    """Fuse with and without block verdicts; return both and the workload."""

    results = {}
    for culling in (True, False):
        storage = allocate_empty_tsdf_blocks(plan, session)
        evaluated = 0
        verdict_counts = np.zeros(3, dtype=np.int64)
        real_evaluate = tsdf_stream_fusion._evaluate_ready_voxels
        real_classify = tsdf_stream_fusion._classify_blocks

        def counting_evaluate(*arguments):
            nonlocal evaluated
            evaluated += len(arguments[4][0])
            return real_evaluate(*arguments)

        def counting_classify(*arguments):
            verdicts = real_classify(*arguments)
            verdict_counts[:] += np.bincount(verdicts, minlength=3)
            return verdicts

        with (
            patch.object(
                tsdf_stream_fusion, "STREAM_FUSION_CULLS_BLOCKS", culling
            ),
            patch.object(
                tsdf_stream_fusion, "_evaluate_ready_voxels", counting_evaluate
            ),
            patch.object(
                tsdf_stream_fusion, "_classify_blocks", counting_classify
            ),
        ):
            receipt = fuse_tsdf_plan_streaming(storage, session)
        results[culling] = (
            storage_bytes(storage),
            receipt,
            evaluated,
            verdict_counts,
        )
    return results


class WholeFusionTests(unittest.TestCase):
    def test_the_room_fuses_to_the_same_bytes_with_less_work(self) -> None:
        case = shared_room_case()
        results = fuse_both_ways(case.plan, case.session)
        culled_bytes, culled_receipt, culled_work, verdicts = results[True]
        plain_bytes, plain_receipt, plain_work, unused = results[False]

        self.assertEqual(culled_bytes, plain_bytes)
        self.assertEqual(culled_receipt, plain_receipt)
        self.assertEqual(int(unused.sum()), 0)
        total = plain_receipt.voxel_slots * plain_receipt.fused_observations
        self.assertEqual(plain_work, total)
        self.assertEqual(
            sum(count for _, count in plain_receipt.status_counts), total
        )
        self.assertGreater(int(verdicts[BLOCK_OUTSIDE_IMAGE]), 0)
        self.assertLess(culled_work, plain_work)
        # Every voxel-observation is still accounted for in the receipt.
        self.assertEqual(
            culled_work
            + 512 * int(verdicts[BLOCK_BEHIND_CAMERA])
            + 512 * int(verdicts[BLOCK_OUTSIDE_IMAGE]),
            total,
        )

    def test_a_scan_that_turns_around_fuses_to_the_same_bytes(self) -> None:
        # The room scan only ever looks forwards, so nothing in its plan is
        # behind a camera. Turning every other pose half a turn about the
        # vertical puts half the plan behind each frame.
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            root = Path(temporary)
            session_path = root / "turning.vgsession"
            shutil.copytree(room_session(), session_path)
            poses = session_path / "streams" / "poses.jsonl"
            half_turn = np.diag([-1.0, -1.0, 1.0, 1.0])
            lines = []
            for index, line in enumerate(
                poses.read_text(encoding="utf-8").splitlines()
            ):
                record = json.loads(line)
                if index % 2 == 1:
                    matrix = np.array(record["T_world_camera"]).reshape((4, 4))
                    record["T_world_camera"] = [
                        float(value) for value in (half_turn @ matrix).ravel()
                    ]
                lines.append(json.dumps(record))
            poses.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
            plan_path = root / "turning.sftplan"
            plan_tsdf_blocks(
                load_scan_session(session_path),
                plan_path,
                frame_stride=1,
                **ROOM_PLAN_ARGUMENTS,
            )
            results = fuse_both_ways(
                load_tsdf_block_plan(plan_path),
                load_scan_session(session_path),
            )
        culled_bytes, culled_receipt, culled_work, verdicts = results[True]
        plain_bytes, plain_receipt, plain_work, _ = results[False]

        self.assertEqual(culled_bytes, plain_bytes)
        self.assertEqual(culled_receipt, plain_receipt)
        self.assertGreater(int(verdicts[BLOCK_BEHIND_CAMERA]), 0)
        self.assertGreater(int(verdicts[BLOCK_OUTSIDE_IMAGE]), 0)
        self.assertLess(culled_work, plain_work / 2)
        status_counts = dict(culled_receipt.status_counts)
        self.assertGreaterEqual(
            status_counts[TsdfContributionStatus.CAMERA_Z_NONPOSITIVE],
            512 * int(verdicts[BLOCK_BEHIND_CAMERA]),
        )


if __name__ == "__main__":
    unittest.main()
