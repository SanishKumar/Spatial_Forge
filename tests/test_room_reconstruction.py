"""End-to-end reconstruction accuracy on a non-degenerate room scan.

Every other test in this suite pins exact values against a 2x2 fixture whose
geometry lands on voxel boundaries. These assert that the pipeline recovers
known planes from noisy, off-grid, multi-frame data to a stated tolerance,
which is the property that actually matters once real sensors are involved.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from spatialforge import (
    build_tsdf_replay_depth_context,
    extract_surface_points,
    extract_triangle_mesh,
    infer_tsdf_bounds,
    integrate_tsdf,
    load_scan_session,
    load_tsdf_block_plan,
    reconstruct_point_cloud,
)
from spatialforge.errors import TsdfError
from spatialforge.tsdf_block_plan import plan_tsdf_blocks
from spatialforge.tsdf_block_storage import allocate_empty_tsdf_blocks
from spatialforge.tsdf_plan_fusion import (
    begin_tsdf_fusion_ledger,
    fuse_tsdf_plan_blocks_from_context,
)
from spatialforge.tsdf_plan_traversal import (
    traverse_tsdf_plan_blocks_from_context,
)

from tests.room_fixture import room_session, room_truth

TEST_ROOT = Path(__file__).resolve().parent
VOXEL_M = 0.04
TRUNCATION_M = 0.12


def read_ply_points(path: Path) -> list[tuple[float, float, float]]:
    lines = path.read_text(encoding="ascii").splitlines()
    start = next(
        index for index, line in enumerate(lines)
        if line.strip() == "end_header"
    ) + 1
    points = []
    for line in lines[start:]:
        parts = line.split()
        if len(parts) >= 3:
            points.append(
                (float(parts[0]), float(parts[1]), float(parts[2]))
            )
    return points


def plane_error(points, axis, truth, keep):
    selected = [p[axis] - truth for p in points if keep(p)]
    if not selected:
        return None
    mean = sum(selected) / len(selected)
    rms = (sum(value * value for value in selected) / len(selected)) ** 0.5
    return len(selected), mean, rms


class RoomReconstructionTests(unittest.TestCase):
    def test_session_validates_and_back_projects(self) -> None:
        session_path = room_session()
        truth = room_truth()
        session = load_scan_session(session_path)

        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            report = reconstruct_point_cloud(
                session,
                Path(temporary) / "cloud.ply",
            )

        self.assertEqual(session.session_id, "scan-room-0001")
        self.assertEqual(report.integrated_frames, truth.frames)
        self.assertEqual(report.invalid_depth_samples, 0)
        self.assertEqual(
            report.points_written,
            truth.frames * truth.width * truth.height,
        )

    def test_inferred_bounds_bracket_the_known_room(self) -> None:
        session = load_scan_session(room_session())
        truth = room_truth()

        bounds = infer_tsdf_bounds(
            session,
            voxel_size_m=VOXEL_M,
            truncation_m=TRUNCATION_M,
        )

        # Inference pads by the truncation distance and snaps outward, so the
        # volume must contain every true plane with room to spare.
        self.assertLess(bounds.origin_world_m[0], truth.far_wall_x)
        self.assertGreater(
            bounds.origin_world_m[0] + bounds.dimensions_xyz[0] * VOXEL_M,
            truth.far_wall_x,
        )
        self.assertLess(bounds.origin_world_m[1], truth.right_wall_y)
        self.assertGreater(
            bounds.origin_world_m[1] + bounds.dimensions_xyz[1] * VOXEL_M,
            truth.left_wall_y,
        )
        self.assertLess(bounds.origin_world_m[2], truth.floor_z)
        self.assertGreater(
            bounds.origin_world_m[2] + bounds.dimensions_xyz[2] * VOXEL_M,
            truth.ceiling_z,
        )

    def test_surface_recovers_known_planes_within_tolerance(self) -> None:
        session = load_scan_session(room_session())
        truth = room_truth()

        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            root = Path(temporary)
            bounds = infer_tsdf_bounds(
                session,
                voxel_size_m=VOXEL_M,
                truncation_m=TRUNCATION_M,
            )
            integrate_tsdf(
                session,
                root / "room.sftsdf",
                origin_world_m=bounds.origin_world_m,
                dimensions=bounds.dimensions_xyz,
                voxel_size_m=VOXEL_M,
                truncation_m=TRUNCATION_M,
            )
            surface = extract_surface_points(
                root / "room.sftsdf",
                root / "surface.ply",
            )
            points = read_ply_points(root / "surface.ply")

        self.assertGreater(surface.points_written, 2000)
        self.assertEqual(len(points), surface.points_written)

        band = 0.09
        checks = (
            (
                "far wall",
                0,
                truth.far_wall_x,
                lambda p: p[0] > truth.far_wall_x - band,
                0.030,
            ),
            (
                "left wall",
                1,
                truth.left_wall_y,
                lambda p: p[1] > truth.left_wall_y - band,
                0.030,
            ),
            (
                "right wall",
                1,
                truth.right_wall_y,
                lambda p: p[1] < truth.right_wall_y + band,
                0.030,
            ),
        )
        for label, axis, value, keep, tolerance in checks:
            with self.subTest(plane=label):
                result = plane_error(points, axis, value, keep)
                self.assertIsNotNone(result)
                count, mean, rms = result
                self.assertGreater(count, 200)
                # Mean bias well inside one voxel, and spread inside one
                # voxel, with 4 mm depth noise and 40 mm voxels.
                self.assertLess(
                    abs(mean),
                    tolerance,
                    f"{label} mean bias {mean * 1000:.1f}mm",
                )
                self.assertLess(
                    rms,
                    VOXEL_M,
                    f"{label} rms {rms * 1000:.1f}mm",
                )

    def test_mesh_is_non_degenerate(self) -> None:
        session = load_scan_session(room_session())

        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            root = Path(temporary)
            bounds = infer_tsdf_bounds(
                session,
                voxel_size_m=VOXEL_M,
                truncation_m=TRUNCATION_M,
            )
            integrate_tsdf(
                session,
                root / "room.sftsdf",
                origin_world_m=bounds.origin_world_m,
                dimensions=bounds.dimensions_xyz,
                voxel_size_m=VOXEL_M,
                truncation_m=TRUNCATION_M,
            )
            mesh = extract_triangle_mesh(
                root / "room.sftsdf",
                root / "mesh.ply",
            )

        self.assertGreater(mesh.vertices_written, 10000)
        self.assertGreater(mesh.triangles_written, 20000)
        self.assertGreater(mesh.active_cells, 1000)

    def test_one_shot_traversal_refuses_a_real_scale_plan(self) -> None:
        """The scale limit this run exposed, pinned so it cannot regress.

        A 20-frame 64x48 scan plans 351 blocks, which is 3,594,240 retained
        contribution outcomes against a 262,144 cap. The one-shot traversal
        is a reference path for the tiny fixture, not a scan-scale one.
        """

        session = load_scan_session(room_session())

        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            root = Path(temporary)
            plan_tsdf_blocks(
                session,
                root / "plan.sftplan",
                voxel_size_m=VOXEL_M,
                truncation_m=TRUNCATION_M,
            )
            plan = load_tsdf_block_plan(root / "plan.sftplan")
            context = build_tsdf_replay_depth_context(plan, session)
            storage = allocate_empty_tsdf_blocks(plan, session)

            with self.assertRaises(TsdfError) as raised:
                traverse_tsdf_plan_blocks_from_context(storage, context)

            self.assertIn(
                "retained contribution outcomes",
                str(raised.exception),
            )
            self.assertGreater(plan.active_block_count, 300)

            # The resumable path is what makes the same plan tractable: a
            # bounded pass stays under the cap and makes real progress.
            ledger = begin_tsdf_fusion_ledger(plan)
            receipt = fuse_tsdf_plan_blocks_from_context(
                storage,
                context,
                ledger,
                block_limit=2,
            )

        self.assertEqual(receipt.blocks_fused_now, 2)
        self.assertEqual(receipt.evaluated_count, 2 * 512 * 20)
        self.assertFalse(receipt.is_complete)
        self.assertEqual(
            receipt.ledger_after.pending_block_count,
            plan.active_block_count - 2,
        )


if __name__ == "__main__":
    unittest.main()
