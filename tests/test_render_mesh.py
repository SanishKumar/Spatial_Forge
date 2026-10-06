"""The mesh renderer: a software rasteriser has to be right pixel by pixel.

Pictures are what most people will judge a reconstruction by, so the tool
that draws them is held to things that can be checked exactly: which pixels
a triangle covers, which of two overlapping triangles wins, and that a
vertex is only ever coloured by a frame that really sees it.
"""

from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
from PIL import Image

from spatialforge import (
    allocate_empty_tsdf_blocks,
    fuse_tsdf_plan_streaming,
    load_tsdf_block_plan,
    write_tsdf_block_volume,
)
from spatialforge.session_loader import load_scan_session
from spatialforge.tsdf_block_mesh import extract_tsdf_block_mesh
from spatialforge.tsdf_block_plan import plan_tsdf_blocks

from tools.render_mesh import (
    _BACK_TINT,
    _UNSEEN,
    animation_azimuths,
    main as render_main,
    rasterise,
    read_mesh_ply,
    scan_vertex_colours,
    vertex_normals,
)
from tools.render_point_cloud import _BACKGROUND

TEST_ROOT = Path(__file__).resolve().parent
FIXTURE = TEST_ROOT / "fixtures" / "minimal.vgsession"
RED = np.array([200.0, 10.0, 10.0])
GREEN = np.array([10.0, 200.0, 10.0])


def fixture_mesh(parent: Path) -> Path:
    plan_path = parent / "fixture.sftplan"
    plan_tsdf_blocks(
        load_scan_session(FIXTURE),
        plan_path,
        voxel_size_m=0.125,
        truncation_m=0.5,
    )
    plan = load_tsdf_block_plan(plan_path)
    session = load_scan_session(FIXTURE)
    storage = allocate_empty_tsdf_blocks(plan, session)
    receipt = fuse_tsdf_plan_streaming(storage, session)
    volume_path = parent / "fixture.sftvol"
    write_tsdf_block_volume(storage, receipt, volume_path)
    mesh_path = parent / "fixture.ply"
    extract_tsdf_block_mesh(volume_path, mesh_path)
    return mesh_path


def covered(canvas: np.ndarray) -> np.ndarray:
    return np.any(canvas != _BACKGROUND, axis=2)


def draw(screen, depth, faces, colours, resolution=12) -> np.ndarray:
    return rasterise(
        np.array(screen, dtype=np.float64),
        np.array(depth, dtype=np.float64),
        np.array(faces, dtype=np.int64),
        np.array(colours, dtype=np.float64),
        resolution,
    )


class RasteriserTests(unittest.TestCase):
    # In image coordinates y runs downwards, so this winding is the one a
    # free-space-facing triangle has on screen.
    FRONT = [[1.0, 1.0], [1.0, 9.0], [9.0, 1.0]]

    def test_a_triangle_covers_exactly_its_lattice_points(self) -> None:
        canvas = draw(self.FRONT, [1.0] * 3, [[0, 1, 2]], [RED] * 3)
        mask = covered(canvas)
        expected = np.zeros((12, 12), dtype=bool)
        for row in range(12):
            for column in range(12):
                expected[row, column] = (
                    row >= 1 and column >= 1 and row + column <= 10
                )

        self.assertEqual(int(mask.sum()), 45)
        self.assertTrue(bool(np.array_equal(mask, expected)))
        self.assertTrue(bool(np.all(canvas[mask] == RED)))

    def test_the_nearer_triangle_wins_whatever_the_order(self) -> None:
        screen = self.FRONT + [[1.0, 1.0], [1.0, 9.0], [9.0, 1.0]]
        colours = [RED] * 3 + [GREEN] * 3
        depth = [2.0] * 3 + [1.0] * 3
        near_last = draw(screen, depth, [[0, 1, 2], [3, 4, 5]], colours)
        near_first = draw(screen, depth, [[3, 4, 5], [0, 1, 2]], colours)

        self.assertTrue(bool(np.array_equal(near_last, near_first)))
        self.assertTrue(bool(np.all(near_last[covered(near_last)] == GREEN)))

    def test_colour_is_interpolated_between_the_vertices(self) -> None:
        blue = np.array([10.0, 10.0, 200.0])
        canvas = draw(self.FRONT, [1.0] * 3, [[0, 1, 2]], [RED, GREEN, blue])

        np.testing.assert_allclose(canvas[1, 1], RED)
        np.testing.assert_allclose(canvas[9, 1], GREEN)
        np.testing.assert_allclose(canvas[1, 9], blue)
        np.testing.assert_allclose(canvas[5, 1], (RED + GREEN) / 2.0)

    def test_the_reverse_side_is_drawn_tinted_not_culled(self) -> None:
        back = [self.FRONT[0], self.FRONT[2], self.FRONT[1]]
        canvas = draw(back, [1.0] * 3, [[0, 1, 2]], [RED] * 3)
        mask = covered(canvas)

        self.assertEqual(int(mask.sum()), 45)
        np.testing.assert_allclose(canvas[mask][0], RED * _BACK_TINT)

    def test_geometry_that_cannot_be_drawn_is_skipped(self) -> None:
        behind = draw(self.FRONT, [1.0, -1.0, 1.0], [[0, 1, 2]], [RED] * 3)
        sliver = draw(
            [[1.0, 1.0], [5.0, 5.0], [9.0, 9.0]],
            [1.0] * 3,
            [[0, 1, 2]],
            [RED] * 3,
        )
        outside = draw(
            [[40.0, 40.0], [40.0, 50.0], [50.0, 40.0]],
            [1.0] * 3,
            [[0, 1, 2]],
            [RED] * 3,
        )
        for canvas in (behind, sliver, outside):
            self.assertFalse(bool(covered(canvas).any()))

    def test_a_large_triangle_is_filled_without_gaps(self) -> None:
        canvas = draw(
            [[0.0, 0.0], [0.0, 199.0], [199.0, 0.0]],
            [1.0] * 3,
            [[0, 1, 2]],
            [RED] * 3,
            resolution=200,
        )
        self.assertEqual(int(covered(canvas).sum()), 200 * 201 // 2)


class MeshInputTests(unittest.TestCase):
    def test_reader_returns_what_the_mesher_wrote(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            vertices, faces = read_mesh_ply(fixture_mesh(Path(temporary_dir)))

        self.assertEqual(vertices.shape, (225, 3))
        self.assertEqual(faces.shape, (392, 3))
        self.assertEqual(int(faces.max()), 224)

    def test_reader_refuses_files_it_cannot_trust(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            temporary_root = Path(temporary_dir)
            good = fixture_mesh(temporary_root).read_bytes()
            split = good.index(b"end_header\n") + len(b"end_header\n")
            first_face = split + 225 * 24
            cases = {
                "not a PLY": b"hello",
                "binary little-endian": (
                    b"ply\nformat ascii 1.0\nelement vertex 1\n"
                    b"property double x\nend_header\n0\n"
                ),
                "payload size": good[:-1],
                "must be a triangle": (
                    good[:first_face] + b"\x04" + good[first_face + 1:]
                ),
                "outside the vertex list": (
                    good[:first_face + 1]
                    + (9_999).to_bytes(4, "little")
                    + good[first_face + 5:]
                ),
            }
            for expected, encoded in cases.items():
                with self.subTest(case=expected):
                    path = temporary_root / "bad.ply"
                    path.write_bytes(encoded)
                    with self.assertRaises(SystemExit) as caught:
                        read_mesh_ply(path)
                    self.assertIn(expected, str(caught.exception))

    def test_normals_follow_the_winding(self) -> None:
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [5.0, 5.0, 5.0],
            ]
        )
        normals = vertex_normals(vertices, np.array([[0, 1, 2]]))
        flipped = vertex_normals(vertices, np.array([[0, 2, 1]]))

        np.testing.assert_allclose(normals[:3], [[0.0, 0.0, 1.0]] * 3)
        np.testing.assert_allclose(flipped[:3], [[0.0, 0.0, -1.0]] * 3)
        # A vertex no triangle uses still gets a unit normal.
        np.testing.assert_allclose(normals[3], [0.0, 0.0, 1.0])


class ScanColourTests(unittest.TestCase):
    def test_a_vertex_is_coloured_only_by_frames_that_see_it(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            vertices, faces = read_mesh_ply(fixture_mesh(Path(temporary_dir)))
        normals = vertex_normals(vertices, faces)
        colours, frames_used, seen = scan_vertex_colours(
            FIXTURE,
            vertices,
            normals,
            1,
        )

        self.assertEqual(frames_used, 2)
        self.assertGreater(seen, 0)
        self.assertTrue(bool(np.all((colours >= 0.0) & (colours <= 255.0))))
        unseen = np.all(colours == _UNSEEN, axis=1)
        self.assertEqual(int((~unseen).sum()), seen)

        # The same vertices pushed well behind the surface they lie on are
        # still inside the image, but no frame measures that depth there.
        buried, _, buried_seen = scan_vertex_colours(
            FIXTURE,
            vertices - normals * 0.4,
            normals,
            1,
        )
        self.assertEqual(buried_seen, 0)
        self.assertTrue(bool(np.all(buried == _UNSEEN)))

        # And a surface facing away from every camera is not painted with
        # what the cameras saw on its other side.
        _, _, reversed_seen = scan_vertex_colours(
            FIXTURE,
            vertices,
            -normals,
            1,
        )
        self.assertEqual(reversed_seen, 0)


class AnimationPathTests(unittest.TestCase):
    def test_orbit_and_sweep_paths(self) -> None:
        self.assertEqual(
            animation_azimuths(10.0, 4, None),
            [10.0, 100.0, 190.0, 280.0],
        )
        sweep = animation_azimuths(180.0, 24, 30.0)
        self.assertEqual(len(sweep), 24)
        self.assertEqual(sweep[0], 180.0)
        self.assertAlmostEqual(max(sweep), 210.0)
        self.assertAlmostEqual(min(sweep), 150.0)
        self.assertEqual(animation_azimuths(180.0, 1, 30.0), [180.0])


class RenderCommandTests(unittest.TestCase):
    def run_main(self, arguments: list[str]) -> str:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = render_main(arguments)
        self.assertEqual(exit_code, 0)
        return stdout.getvalue()

    def test_it_writes_a_still_and_an_animation_deterministically(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            temporary_root = Path(temporary_dir)
            mesh_path = fixture_mesh(temporary_root)
            outputs = []
            for name in ("first", "second"):
                printed = self.run_main(
                    [
                        str(mesh_path),
                        str(temporary_root / name),
                        "--session",
                        str(FIXTURE),
                        "--colour-stride",
                        "1",
                        "--size",
                        "72",
                        "--frames",
                        "3",
                        "--sweep",
                        "20",
                    ]
                )
                outputs.append(
                    (
                        (temporary_root / f"{name}.png").read_bytes(),
                        (temporary_root / f"{name}.gif").read_bytes(),
                    )
                )
            with Image.open(temporary_root / "first.png") as still:
                size = still.size
                pixels = np.asarray(still.convert("RGB"))
            with Image.open(temporary_root / "first.gif") as animation:
                frames = animation.n_frames

        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(size, (72, 72))
        self.assertEqual(frames, 3)
        self.assertGreater(len(np.unique(pixels.reshape(-1, 3), axis=0)), 8)
        self.assertIn("225 vertices, 392 triangles", printed)
        self.assertIn("colour: scan RGB from 2 frames", printed)

    def test_outputs_are_never_overwritten_and_inputs_never_touched(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            temporary_root = Path(temporary_dir)
            mesh_path = fixture_mesh(temporary_root)
            original = mesh_path.read_bytes()
            existing = temporary_root / "taken.png"
            existing.write_bytes(b"keep me")
            with self.assertRaises(SystemExit) as taken:
                render_main(
                    [str(mesh_path), str(temporary_root / "taken")]
                )
            self.assertIn("already exists", str(taken.exception))
            self.assertEqual(existing.read_bytes(), b"keep me")
            self.assertFalse((temporary_root / "taken.gif").exists())
            self.assertEqual(mesh_path.read_bytes(), original)

    def test_unusable_arguments_are_refused(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            temporary_root = Path(temporary_dir)
            mesh_path = fixture_mesh(temporary_root)
            cases = {
                "degenerate elevation": ["--elevation", "90"],
                "zero sweep": ["--sweep", "0"],
                "zero frames": ["--frames", "0"],
                "zero size": ["--size", "0"],
                "zero colour stride": ["--colour-stride", "0"],
                "fill above one": ["--fill", "2"],
            }
            for position, (name, extra) in enumerate(cases.items()):
                with self.subTest(case=name):
                    with (
                        redirect_stderr(io.StringIO()),
                        self.assertRaises(SystemExit),
                    ):
                        render_main(
                            [
                                str(mesh_path),
                                str(temporary_root / f"out{position}"),
                                *extra,
                            ]
                        )
            leftovers = sorted(
                path.name
                for path in temporary_root.iterdir()
                if path.suffix in (".png", ".gif")
            )

        self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
