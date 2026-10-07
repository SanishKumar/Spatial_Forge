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
    BACK_CULLED,
    BACK_PLAIN,
    BACK_TINTED,
    _BACK_TINT,
    _ERROR_RAMP,
    _UNSEEN,
    draw_error_legend,
    error_colours,
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


class BackFaceModeTests(unittest.TestCase):
    FRONT = RasteriserTests.FRONT
    BACK = [FRONT[0], FRONT[2], FRONT[1]]

    def draw(self, screen, depth, faces, colours, back) -> np.ndarray:
        return rasterise(
            np.array(screen, dtype=np.float64),
            np.array(depth, dtype=np.float64),
            np.array(faces, dtype=np.int64),
            np.array(colours, dtype=np.float64),
            12,
            back,
        )

    def test_a_culled_reverse_side_is_not_drawn(self) -> None:
        canvas = self.draw(
            self.BACK, [1.0] * 3, [[0, 1, 2]], [RED] * 3, BACK_CULLED
        )
        self.assertFalse(bool(covered(canvas).any()))
        front = self.draw(
            self.FRONT, [1.0] * 3, [[0, 1, 2]], [RED] * 3, BACK_CULLED
        )
        self.assertEqual(int(covered(front).sum()), 45)
        self.assertTrue(bool(np.all(front[covered(front)] == RED)))

    def test_a_culled_reverse_side_does_not_hide_what_is_behind_it(
        self,
    ) -> None:
        # Looking into a room: the near wall is seen from behind and must
        # not win the depth test against the far wall it would occlude.
        screen = self.BACK + self.FRONT
        depth = [1.0] * 3 + [2.0] * 3
        colours = [RED] * 3 + [GREEN] * 3
        faces = [[0, 1, 2], [3, 4, 5]]
        culled = self.draw(screen, depth, faces, colours, BACK_CULLED)
        tinted = self.draw(screen, depth, faces, colours, BACK_TINTED)

        self.assertTrue(bool(np.all(culled[covered(culled)] == GREEN)))
        np.testing.assert_allclose(
            tinted[covered(tinted)][0], RED * _BACK_TINT
        )

    def test_a_plain_reverse_side_ignores_the_vertex_colour(self) -> None:
        canvas = self.draw(
            self.BACK, [1.0] * 3, [[0, 1, 2]], [RED, GREEN, RED], BACK_PLAIN
        )
        mask = covered(canvas)
        self.assertEqual(int(mask.sum()), 45)
        self.assertTrue(bool(np.all(canvas[mask] == _UNSEEN * _BACK_TINT)))
        # The measured side is untouched by the mode.
        front = self.draw(
            self.FRONT, [1.0] * 3, [[0, 1, 2]], [RED] * 3, BACK_PLAIN
        )
        self.assertTrue(bool(np.all(front[covered(front)] == RED)))

    def test_an_unknown_mode_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown back-face mode"):
            self.draw(self.FRONT, [1.0] * 3, [[0, 1, 2]], [RED] * 3, "x")


class ErrorColourTests(unittest.TestCase):
    def test_the_ramp_runs_from_zero_to_the_scale_and_saturates(self) -> None:
        colours = error_colours(
            np.array([0.0, 0.0025, 0.005, 0.0075, 0.01, 0.5]), 0.01
        )
        np.testing.assert_allclose(colours[:5], _ERROR_RAMP)
        np.testing.assert_allclose(colours[5], _ERROR_RAMP[-1])
        between = error_colours(np.array([0.00125]), 0.01)[0]
        np.testing.assert_allclose(
            between, (_ERROR_RAMP[0] + _ERROR_RAMP[1]) / 2.0
        )

    def test_an_unmeasured_vertex_is_grey_not_zero_error(self) -> None:
        colours = error_colours(np.array([np.nan, 0.0]), 0.01)
        np.testing.assert_allclose(colours[0], _UNSEEN)
        self.assertFalse(bool(np.allclose(colours[0], colours[1])))

    def test_the_legend_shows_both_ends_of_the_scale(self) -> None:
        blank = Image.new("RGB", (200, 200), (14, 16, 22))
        drawn = np.asarray(draw_error_legend(blank, 5.0), dtype=np.float64)
        # margin 7, bar 68 by 4, against the bottom-left corner.
        np.testing.assert_allclose(drawn[190, 7], np.round(_ERROR_RAMP[0]))
        np.testing.assert_allclose(drawn[190, 74], np.round(_ERROR_RAMP[-1]))
        np.testing.assert_allclose(drawn[190, 75], (14, 16, 22))
        # Labels were written above it.
        self.assertTrue(bool(np.any(drawn[170:188, 7:75] != (14, 16, 22))))
        self.assertEqual(blank.getpixel((7, 190)), (14, 16, 22))


class ErrorMapCommandTests(unittest.TestCase):
    def test_it_draws_measured_error_with_its_scale(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            temporary_root = Path(temporary_dir)
            mesh_path = fixture_mesh(temporary_root)
            errors = np.linspace(0.0, 0.02, 225, dtype=np.float32)
            errors[17] = np.nan
            errors_path = temporary_root / "errors.npy"
            np.save(errors_path, errors)
            original = errors_path.read_bytes()
            outputs = []
            for name in ("first", "second"):
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    exit_code = render_main(
                        [
                            str(mesh_path),
                            str(temporary_root / name),
                            "--vertex-errors",
                            str(errors_path),
                            "--error-scale-mm",
                            "10",
                            "--size",
                            "72",
                            "--frames",
                            "2",
                            "--sweep",
                            "20",
                        ]
                    )
                self.assertEqual(exit_code, 0)
                outputs.append(
                    (
                        (temporary_root / f"{name}.png").read_bytes(),
                        (temporary_root / f"{name}.gif").read_bytes(),
                    )
                )
            with Image.open(temporary_root / "first.png") as still:
                pixels = np.asarray(still.convert("RGB"), dtype=np.float64)
            self.assertEqual(errors_path.read_bytes(), original)

        self.assertEqual(outputs[0], outputs[1])
        self.assertIn(
            "colour: distance to ground truth, 0 to 10 mm, 224 of 225 "
            "vertices measured",
            stdout.getvalue(),
        )
        # At 72 pixels the scale bar is 24 by 2, three pixels in from the
        # bottom-left corner.
        np.testing.assert_allclose(pixels[67, 3], np.round(_ERROR_RAMP[0]))
        np.testing.assert_allclose(pixels[67, 26], np.round(_ERROR_RAMP[-1]))

    def test_scan_colour_and_error_are_drawn_side_by_side(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            temporary_root = Path(temporary_dir)
            mesh_path = fixture_mesh(temporary_root)
            errors_path = temporary_root / "errors.npy"
            np.save(errors_path, np.linspace(0.0, 0.02, 225, dtype=np.float32))
            common = [
                str(mesh_path),
                "--colour-stride",
                "1",
                "--size",
                "72",
                "--frames",
                "3",
                "--sweep",
                "20",
            ]

            def run(name: str, *colour: str) -> tuple[np.ndarray, str]:
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    render_main(
                        [common[0], str(temporary_root / name), *common[1:], *colour]
                    )
                with Image.open(temporary_root / f"{name}.png") as still:
                    pixels = np.asarray(
                        still.convert("RGB"), dtype=np.float64
                    )
                return pixels, stdout.getvalue()

            scan, _ = run("scan", "--session", str(FIXTURE))
            error, _ = run("error", "--vertex-errors", str(errors_path))
            both, printed = run(
                "both",
                "--session",
                str(FIXTURE),
                "--vertex-errors",
                str(errors_path),
            )
            with Image.open(temporary_root / "both.gif") as animation:
                gif_size = animation.size
                gif_frames = animation.n_frames

        # Two 72-pixel panels and a one-pixel gap between them.
        self.assertEqual(both.shape, (72, 145, 3))
        self.assertEqual(gif_size, (145, 72))
        self.assertEqual(gif_frames, 3)
        self.assertIn("colour: scan RGB", printed)
        self.assertIn("colour: distance to ground truth", printed)
        # Each panel is exactly the picture that colouring gives alone, so
        # the two are the same camera and the same crop.
        self.assertTrue(bool(np.array_equal(both[:, :72], scan)))
        self.assertTrue(bool(np.array_equal(both[:, 73:], error)))
        self.assertTrue(bool(np.all(both[:, 72] == (14.0, 16.0, 22.0))))
        # The scale belongs to the error panel only.
        np.testing.assert_allclose(both[67, 73 + 3], np.round(_ERROR_RAMP[0]))
        self.assertFalse(
            bool(np.allclose(both[67, 3], np.round(_ERROR_RAMP[0])))
        )

    def test_errors_that_do_not_belong_to_the_mesh_are_refused(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            temporary_root = Path(temporary_dir)
            mesh_path = fixture_mesh(temporary_root)
            good = np.zeros(225, dtype=np.float32)
            negative = good.copy()
            negative[3] = -0.001
            infinite = good.copy()
            infinite[3] = np.inf
            cases = {
                "short": (np.zeros(224, dtype=np.float32), "not measured on"),
                "grid": (np.zeros((225, 1), dtype=np.float32), "one-dimens"),
                "integer": (np.zeros(225, dtype=np.int32), "floating-point"),
                "negative": (negative, "non-negative"),
                "infinite": (infinite, "non-negative"),
            }
            for name, (array, message) in cases.items():
                with self.subTest(name=name):
                    path = temporary_root / f"{name}.npy"
                    np.save(path, array)
                    with (
                        self.assertRaises(SystemExit) as raised,
                        redirect_stdout(io.StringIO()),
                    ):
                        render_main(
                            [
                                str(mesh_path),
                                str(temporary_root / f"out-{name}"),
                                "--vertex-errors",
                                str(path),
                            ]
                        )
                    self.assertIn(message, str(raised.exception))
                    self.assertFalse(
                        (temporary_root / f"out-{name}.png").exists()
                    )
            garbage = temporary_root / "garbage.npy"
            garbage.write_bytes(b"not an array")
            with (
                self.assertRaisesRegex(SystemExit, "not a readable"),
                redirect_stdout(io.StringIO()),
            ):
                render_main(
                    [
                        str(mesh_path),
                        str(temporary_root / "out-garbage"),
                        "--vertex-errors",
                        str(garbage),
                    ]
                )
            np.save(temporary_root / "good.npy", good)
            for scale in ("0", "-1", "nan", "inf"):
                with self.assertRaisesRegex(SystemExit, "error-scale-mm"):
                    render_main(
                        [
                            str(mesh_path),
                            str(temporary_root / "out-scale"),
                            "--vertex-errors",
                            str(temporary_root / "good.npy"),
                            f"--error-scale-mm={scale}",
                        ]
                    )

    def test_culling_is_available_from_the_command_line(self) -> None:
        # The fixture mesh is one flat wall. From one side culling changes
        # nothing; from the other it removes the whole picture.
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            temporary_root = Path(temporary_dir)
            mesh_path = fixture_mesh(temporary_root)
            counts = {}
            for azimuth in (35, 215):
                for mode, extra in (
                    ("tinted", []),
                    ("culled", ["--cull-back-faces"]),
                ):
                    name = f"{mode}-{azimuth}"
                    with redirect_stdout(io.StringIO()):
                        render_main(
                            [
                                str(mesh_path),
                                str(temporary_root / name),
                                "--size",
                                "72",
                                "--frames",
                                "1",
                                "--azimuth",
                                str(azimuth),
                                *extra,
                            ]
                        )
                    with Image.open(temporary_root / f"{name}.png") as still:
                        counts[(mode, azimuth)] = int(
                            covered(
                                np.asarray(
                                    still.convert("RGB"), dtype=np.float64
                                )
                            ).sum()
                        )

        self.assertGreater(counts[("tinted", 35)], 0)
        self.assertGreater(counts[("tinted", 215)], 0)
        self.assertEqual(
            sorted(
                counts[("culled", azimuth)] == counts[("tinted", azimuth)]
                for azimuth in (35, 215)
            ),
            [False, True],
        )
        self.assertEqual(
            min(counts[("culled", 35)], counts[("culled", 215)]), 0
        )


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
