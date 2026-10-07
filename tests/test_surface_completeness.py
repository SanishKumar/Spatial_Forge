"""The completeness report, on a room where what was seen is known.

Completeness needs a ruling on what the cameras saw, and a ruling can be
wrong in either direction: count a wall behind another wall as seen and
every reconstruction looks incomplete; miss half of what was seen and every
reconstruction looks whole. So the ruling is checked against a second
implementation written one point at a time, and against surfaces planted
where no camera could have seen them. The measure itself is checked by
cutting a wall out of a mesh and requiring the report to notice exactly
that.
"""

from __future__ import annotations

import io
import json
import shutil
import tempfile
import unittest
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from spatialforge import (
    allocate_empty_tsdf_blocks,
    fuse_tsdf_plan_streaming,
    write_tsdf_block_volume,
)
from spatialforge.point_cloud import (
    _sample_path,
    _validate_reconstruction_contract,
)
from spatialforge.replay import replay_session
from spatialforge.session_loader import load_scan_session
from spatialforge.tsdf_block_mesh import extract_tsdf_block_mesh

from tests.heavy_fixtures import shared_room_case
from tests.room_fixture import CEILING_Z, FAR_WALL_X, FLOOR_Z, LEFT_WALL_Y, RIGHT_WALL_Y
from tests.test_surface_accuracy import (
    MODEL_FROM_SESSION,
    ROUGH_TRANSLATION,
    apply,
    lattice,
    room_model,
    write_model,
)
from tools.surface_accuracy_report import main as accuracy_main
from tools.surface_completeness_report import count_views, main, summarise

TEST_ROOT = Path(__file__).resolve().parent
FIXTURE = TEST_ROOT / "fixtures" / "minimal.vgsession"

# Surfaces no camera in the room scan could have seen: a second wall half a
# metre behind the far wall, and a wall behind where the cameras stand,
# which all look the other way.
HIDDEN_BEHIND_WALL = lattice(
    0, FAR_WALL_X + 0.5, (RIGHT_WALL_Y, LEFT_WALL_Y), (FLOOR_Z, CEILING_Z), (-1, 0, 0)
)
HIDDEN_BEHIND_CAMERAS = lattice(
    0, -1.5, (RIGHT_WALL_Y, LEFT_WALL_Y), (FLOOR_Z, CEILING_Z), (1, 0, 0)
)

ROOM: SimpleNamespace | None = None


@contextmanager
def clean_tree(module: str):
    with patch(f"{module}.source_state", return_value=("a" * 40, True)):
        yield


def model_in_session_frame() -> tuple[np.ndarray, np.ndarray, int]:
    """The room's true surfaces, then the hidden ones; and where they start."""

    points, normals = room_model()
    visible = len(points)
    return (
        np.concatenate([points, HIDDEN_BEHIND_WALL[0], HIDDEN_BEHIND_CAMERAS[0]]),
        np.concatenate([normals, HIDDEN_BEHIND_WALL[1], HIDDEN_BEHIND_CAMERAS[1]]),
        visible,
    )


def setUpModule() -> None:
    global ROOM
    root = Path(tempfile.mkdtemp(dir=TEST_ROOT))
    unittest.addModuleCleanup(shutil.rmtree, root, True)
    case = shared_room_case()
    storage = allocate_empty_tsdf_blocks(case.plan, case.session)
    receipt = fuse_tsdf_plan_streaming(storage, case.session)
    volume = root / "room.sftvol"
    write_tsdf_block_volume(storage, receipt, volume)
    mesh = root / "room.ply"
    extract_tsdf_block_mesh(volume, mesh)
    points, normals, _ = model_in_session_frame()
    model = root / "model.ply"
    write_model(
        model,
        apply(MODEL_FROM_SESSION, points),
        normals @ MODEL_FROM_SESSION[:3, :3].T,
    )
    accuracy = root / "accuracy.json"
    with clean_tree("tools.surface_accuracy_report"), redirect_stdout(io.StringIO()):
        accuracy_main(
            [
                str(case.session_path),
                str(volume),
                str(mesh),
                str(model),
                "--initial-translation",
                *ROUGH_TRANSLATION,
                "--fit-frame-stride",
                "2",
                "--pixel-step",
                "2",
                "--manifest-out",
                str(accuracy),
            ]
        )
    ROOM = SimpleNamespace(
        root=root,
        session=case.session_path,
        volume=volume,
        mesh=mesh,
        model=model,
        accuracy=accuracy,
    )


def report(name: str, *extra: str, **replaced: Path) -> dict:
    values = {
        "session": ROOM.session,
        "volume": ROOM.volume,
        "mesh": ROOM.mesh,
        "model": ROOM.model,
    }
    values.update(replaced)
    manifest = ROOM.root / f"{name}.json"
    with (
        clean_tree("tools.surface_completeness_report"),
        redirect_stdout(io.StringIO()),
    ):
        exit_code = main(
            [
                str(values["session"]),
                str(values["volume"]),
                str(values["mesh"]),
                str(values["model"]),
                "--alignment",
                str(ROOM.accuracy),
                "--model-stride",
                "1",
                "--manifest-out",
                str(manifest),
                *extra,
            ]
        )
    assert exit_code == 0
    encoded = manifest.read_bytes()
    assert b"\r" not in encoded
    return json.loads(encoded)


_BASELINE: dict | None = None


def baseline() -> dict:
    global _BASELINE
    if _BASELINE is None:
        _BASELINE = report("baseline")
    return _BASELINE


def room_frames():
    session = load_scan_session(ROOM.session)
    camera, depth_scale_m = _validate_reconstruction_contract(session)
    observations = replay_session(session).observations
    return session, camera, depth_scale_m, observations


def views_one_point_at_a_time(points, normals, tolerance_m) -> np.ndarray:
    """The visibility rule again, as a loop over points and frames."""

    session, camera, depth_scale_m, observations = room_frames()
    views = np.zeros(len(points), dtype=np.int32)
    for observation in observations:
        pose = np.array(observation.pose.data["T_world_camera"]).reshape((4, 4))
        with Image.open(
            _sample_path(session, observation.depth.data, "depth")
        ) as image:
            depth = np.asarray(image, dtype=np.float64) * depth_scale_m
        for index, (point, normal) in enumerate(zip(points, normals)):
            x, y, z = pose[:3, :3].T @ (point - pose[:3, 3])
            if z <= 0.05 or normal @ (pose[:3, 3] - point) <= 0.0:
                continue
            column = int(np.floor(camera.fx * x / z + camera.cx + 0.5))
            row = int(np.floor(camera.fy * y / z + camera.cy + 0.5))
            if not (0 <= column < camera.width and 0 <= row < camera.height):
                continue
            measured = depth[row, column]
            if measured > 0.0 and abs(measured - z) <= tolerance_m:
                views[index] += 1
    return views


class VisibilityTests(unittest.TestCase):
    def test_the_ruling_matches_one_made_a_point_at_a_time(self) -> None:
        points, normals, _ = model_in_session_frame()
        chosen = np.random.default_rng(61).choice(len(points), 400, replace=False)
        session, camera, depth_scale_m, observations = room_frames()
        views = count_views(
            points[chosen],
            normals[chosen],
            session,
            observations,
            camera,
            depth_scale_m,
            depth_tolerance_m=0.02,
        )
        expected = views_one_point_at_a_time(
            points[chosen], normals[chosen], 0.02
        )
        np.testing.assert_array_equal(views, expected)
        # Not a trivial agreement: some points are seen by every frame,
        # some by a few, some by none.
        self.assertEqual(int(views.max()), len(observations))
        self.assertTrue(np.any((views > 0) & (views < len(observations))))
        self.assertTrue(np.any(views == 0))

    def test_what_no_camera_could_see_is_not_seen(self) -> None:
        points, normals, visible = model_in_session_frame()
        session, camera, depth_scale_m, observations = room_frames()
        views = count_views(
            points,
            normals,
            session,
            observations,
            camera,
            depth_scale_m,
            depth_tolerance_m=0.02,
        )
        # Behind the far wall the image has the far wall's depth; behind
        # the cameras nothing projects at all.
        self.assertEqual(int(views[visible:].max()), 0)
        self.assertGreater(int(np.count_nonzero(views[:visible])), visible // 4)

    def test_only_the_side_facing_the_camera_is_seen(self) -> None:
        # The far wall, with its normals turned to face away. Its depth
        # still agrees, so without normals it is seen, and with them not.
        points, normals = lattice(
            0, FAR_WALL_X, (-0.3, 0.3), (-0.3, 0.3), (1, 0, 0)
        )
        session, camera, depth_scale_m, observations = room_frames()

        def seen(with_normals) -> int:
            return int(
                np.count_nonzero(
                    count_views(
                        points,
                        with_normals,
                        session,
                        observations,
                        camera,
                        depth_scale_m,
                        depth_tolerance_m=0.02,
                    )
                )
            )

        self.assertEqual(seen(normals), 0)
        self.assertEqual(seen(None), len(points))
        self.assertEqual(seen(-normals), len(points))

    def test_a_tighter_tolerance_sees_less_of_a_noisy_wall(self) -> None:
        points, normals = lattice(
            0, FAR_WALL_X, (-0.3, 0.3), (-0.3, 0.3), (-1, 0, 0)
        )
        session, camera, depth_scale_m, observations = room_frames()

        def total(tolerance_m: float) -> int:
            return int(
                count_views(
                    points,
                    normals,
                    session,
                    observations,
                    camera,
                    depth_scale_m,
                    depth_tolerance_m=tolerance_m,
                ).sum()
            )

        # The scan's depth noise is 4 mm, so 20 mm admits every frame and
        # 2 mm turns most of them away.
        self.assertEqual(total(0.02), len(points) * len(observations))
        self.assertLess(total(0.002), total(0.02) // 2)


class SummaryTests(unittest.TestCase):
    def test_shares_and_percentiles_are_the_ones_computed_by_hand(self) -> None:
        distances = np.array([0.001, 0.004, 0.006, 0.012, np.inf])
        summary = summarise(distances, reach_m=0.03, voxel_size_m=0.01)
        self.assertEqual(summary["observable_points"], 5)
        self.assertEqual(summary["beyond_reach"], 1)
        self.assertEqual(summary["beyond_reach_fraction"], 0.2)
        self.assertEqual(
            summary["within_fraction"],
            {"5mm": 0.4, "10mm": 0.6, "20mm": 0.8, "one_voxel": 0.6},
        )
        distance = summary["distance_to_mesh"]
        self.assertAlmostEqual(distance["median_mm"], 6.0)
        # The fifth point was never found, so nothing above the 80th
        # percentile can be stated.
        self.assertIsNone(distance["p90_mm"])
        self.assertIsNone(distance["p99_mm"])

    def test_a_threshold_beyond_the_reach_is_not_reported(self) -> None:
        summary = summarise(
            np.array([0.001, 0.006]), reach_m=0.008, voxel_size_m=0.01
        )
        self.assertEqual(summary["within_fraction"], {"5mm": 0.5})

    def test_nothing_observable_is_an_error(self) -> None:
        with self.assertRaisesRegex(SystemExit, "seen by enough frames"):
            summarise(np.empty(0), reach_m=0.03, voxel_size_m=0.01)


def mesh_without(name: str, drop) -> Path:
    """A copy of the room mesh with some triangles cut out."""

    encoded = ROOM.mesh.read_bytes()
    end = encoded.index(b"end_header\n") + len(b"end_header\n")
    header = encoded[:end].decode("ascii")
    vertex_count = int(
        next(line for line in header.splitlines() if line.startswith("element vertex")).split()[2]
    )
    face_count = int(
        next(line for line in header.splitlines() if line.startswith("element face")).split()[2]
    )
    vertices = np.frombuffer(
        encoded, dtype="<f8", count=3 * vertex_count, offset=end
    ).reshape((-1, 3))
    record = np.dtype([("count", "u1"), ("corners", "<i4", (3,))])
    faces = np.frombuffer(
        encoded, dtype=record, count=face_count, offset=end + 24 * vertex_count
    )
    kept = faces[~drop(vertices[faces["corners"]])]
    assert 0 < len(kept) < face_count
    path = ROOM.root / f"{name}.ply"
    path.write_bytes(
        header.replace(
            f"element face {face_count}", f"element face {len(kept)}"
        ).encode("ascii")
        + vertices.tobytes()
        + kept.tobytes()
    )
    return path


class RoomReportTests(unittest.TestCase):
    def test_most_of_what_was_seen_is_in_the_mesh(self) -> None:
        manifest = baseline()
        self.assertEqual(
            manifest["measurement"],
            "surface-completeness-against-ground-truth-model",
        )
        visibility = manifest["visibility"]
        _, _, visible = model_in_session_frame()
        hidden = len(HIDDEN_BEHIND_WALL[0]) + len(HIDDEN_BEHIND_CAMERAS[0])
        self.assertEqual(visibility["frames"], 20)
        # The fixture's mesh asks one observation of a voxel, so one view
        # makes a point observable.
        self.assertEqual(visibility["minimum_views"], 1)
        self.assertTrue(visibility["facing_test"])
        self.assertEqual(visibility["points_considered"], visible + hidden)
        self.assertGreaterEqual(visibility["seen_by_no_frame"], hidden)
        completeness = manifest["completeness"]
        self.assertLessEqual(completeness["observable_points"], visible)
        self.assertGreater(completeness["observable_points"], visible // 4)
        # Twenty frames at 40 mm voxels: nearly everything seen is within a
        # voxel of the mesh.
        self.assertGreater(completeness["within_fraction"]["20mm"], 0.9)
        self.assertGreaterEqual(
            completeness["within_fraction"]["20mm"],
            completeness["within_fraction"]["10mm"],
        )
        self.assertGreaterEqual(
            completeness["within_fraction"]["10mm"],
            completeness["within_fraction"]["5mm"],
        )
        self.assertNotIn("one_voxel", completeness["within_fraction"])
        self.assertEqual(
            manifest["alignment"]["reused_from"]["trajectory_match"],
            {"by": "digest"},
        )
        accuracy = json.loads(ROOM.accuracy.read_bytes())
        self.assertEqual(
            manifest["alignment"]["model_from_source"],
            accuracy["registration"]["model_from_source"],
        )
        self.assertEqual(manifest["source_commit"], "a" * 40)

    def test_a_wall_cut_out_of_the_mesh_is_missed(self) -> None:
        whole = baseline()["completeness"]
        # Every triangle of the far wall.
        holed = report(
            "holed",
            mesh=mesh_without(
                "holed",
                lambda corners: np.all(
                    corners[:, :, 0] > FAR_WALL_X - 0.1, axis=1
                ),
            ),
        )["completeness"]
        self.assertEqual(holed["observable_points"], whole["observable_points"])
        # What was seen of the far wall is now far from any mesh at all.
        # The cut also takes the last 10 cm of floor, ceiling and side
        # walls, so what is lost lies between the wall itself and
        # everything within the reach of the cut.
        points, normals, _ = model_in_session_frame()
        session, camera, depth_scale_m, observations = room_frames()

        def seen(chosen: np.ndarray) -> int:
            return int(
                np.count_nonzero(
                    count_views(
                        points[chosen],
                        normals[chosen],
                        session,
                        observations,
                        camera,
                        depth_scale_m,
                        depth_tolerance_m=0.02,
                    )
                )
            )

        wall_seen = seen(np.abs(points[:, 0] - FAR_WALL_X) < 1e-9)
        slab_seen = seen(
            (points[:, 0] > FAR_WALL_X - 0.1 - whole["reach_m"])
            & (points[:, 0] < FAR_WALL_X + 0.1)
        )
        self.assertGreater(wall_seen, 1_000)
        lost = holed["beyond_reach"] - whole["beyond_reach"]
        self.assertGreater(lost, 0.8 * wall_seen)
        self.assertLessEqual(lost, slab_seen)
        self.assertLess(
            holed["within_fraction"]["20mm"],
            whole["within_fraction"]["20mm"] - 0.1,
        )

    def test_more_views_demanded_means_less_counted_as_seen(self) -> None:
        strict = report("strict", "--min-views", "20")
        self.assertEqual(strict["visibility"]["minimum_views"], 20)
        self.assertLess(
            strict["completeness"]["observable_points"],
            baseline()["completeness"]["observable_points"],
        )
        self.assertGreater(strict["completeness"]["observable_points"], 0)

    def test_sight_can_be_decided_on_another_session_of_the_trajectory(
        self,
    ) -> None:
        same = report(
            "same-sight", "--visibility-session", str(ROOM.session)
        )
        self.assertEqual(same["completeness"], baseline()["completeness"])
        decided = same["visibility"]["decided_on"]
        self.assertEqual(decided["largest_translation_difference_m"], 0.0)
        self.assertEqual(
            decided["replay_digest_sha256"],
            same["inputs"]["replay_digest_sha256"],
        )
        self.assertEqual(
            baseline()["visibility"]["decided_on"],
            {"session": "the evaluated session"},
        )


class RefusalTests(unittest.TestCase):
    def arguments(self, **replaced) -> list[str]:
        values = {
            "session": ROOM.session,
            "volume": ROOM.volume,
            "mesh": ROOM.mesh,
            "model": ROOM.model,
            "alignment": ROOM.accuracy,
        }
        values.update(replaced)
        return [
            str(values["session"]),
            str(values["volume"]),
            str(values["mesh"]),
            str(values["model"]),
            "--alignment",
            str(values["alignment"]),
            "--model-stride",
            "1",
        ]

    def assert_refused(self, message: str, arguments: list[str]) -> None:
        with (
            self.assertRaises(SystemExit) as raised,
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
        ):
            main(arguments)
        self.assertIn(message, str(raised.exception))

    def test_the_frame_must_come_from_this_model_and_trajectory(self) -> None:
        points, normals = room_model()
        other = ROOM.root / "other-model.ply"
        write_model(other, apply(MODEL_FROM_SESSION, points), normals)
        self.assert_refused(
            "different ground-truth model", self.arguments(model=other)
        )
        manifest = json.loads(ROOM.accuracy.read_bytes())
        manifest["inputs"]["source_trajectory_sha256"] = "0" * 64
        elsewhere = ROOM.root / "elsewhere.json"
        elsewhere.write_bytes(json.dumps(manifest).encode("utf-8"))
        self.assert_refused(
            "different source trajectory", self.arguments(alignment=elsewhere)
        )

    def test_the_mesh_volume_and_scan_must_belong_together(self) -> None:
        encoded = ROOM.mesh.read_bytes()
        marker = b"comment spatialforge_source_volume_sha256 "
        at = encoded.index(marker) + len(marker)
        foreign = ROOM.root / "foreign.ply"
        foreign.write_bytes(encoded[:at] + b"0" * 64 + encoded[at + 64:])
        self.assert_refused(
            "was not extracted from this volume", self.arguments(mesh=foreign)
        )
        self.assert_refused(
            "volume was fused from session", self.arguments(session=FIXTURE)
        )

    def test_a_visibility_session_must_be_the_same_camera_and_path(
        self,
    ) -> None:
        self.assert_refused(
            "different camera",
            [*self.arguments(), "--visibility-session", str(FIXTURE)],
        )
        # The room again with its camera's path moved a centimetre.
        moved = ROOM.root / "moved.vgsession"
        shutil.copytree(ROOM.session, moved)
        poses = moved / "streams" / "poses.jsonl"
        lines = []
        for line in poses.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            record["T_world_camera"][3] += 0.01
            lines.append(json.dumps(record))
        poses.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
        self.assert_refused(
            "different trajectory",
            [*self.arguments(), "--visibility-session", str(moved)],
        )

    def test_manifest_provenance_and_output_safety(self) -> None:
        target = ROOM.root / "refused.json"
        with patch(
            "tools.surface_completeness_report.source_state",
            return_value=("b" * 40, False),
        ):
            self.assert_refused(
                "dirty working tree",
                [*self.arguments(), "--manifest-out", str(target)],
            )
        self.assertFalse(target.exists())
        with clean_tree("tools.surface_completeness_report"):
            self.assert_refused(
                "write over an input",
                [*self.arguments(), "--manifest-out", str(ROOM.accuracy)],
            )
            self.assert_refused(
                "must end in .json",
                [*self.arguments(), "--manifest-out", str(ROOM.model)],
            )

    def test_unusable_arguments_are_refused(self) -> None:
        for extra in (
            ["--reach-m", "0"],
            ["--depth-tolerance-m", "nan"],
            ["--model-stride", "0"],
            ["--min-views", "0"],
        ):
            with self.subTest(extra=extra):
                with (
                    self.assertRaises(SystemExit) as raised,
                    redirect_stderr(io.StringIO()),
                ):
                    main([*self.arguments(), *extra])
                self.assertEqual(raised.exception.code, 2)
        with (
            self.assertRaises(SystemExit) as raised,
            redirect_stderr(io.StringIO()),
        ):
            main(self.arguments()[:4])
        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
