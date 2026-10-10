"""The free-space map, on a room whose walls are where the fixture put them.

The map makes one claim per column, and a claim of "free" is the one that
would matter if it were wrong. So the rule is checked by hand on a slab
small enough to read, and then on the room scan from both kinds of volume:
the surface plan's, which knows only a shell around the walls, and the
expanded plan's, which knows the space in front of them.
"""

from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from spatialforge import (
    allocate_empty_tsdf_blocks,
    fuse_tsdf_plan_streaming,
    load_tsdf_block_plan,
    load_tsdf_block_volume,
    write_tsdf_block_volume,
)
from spatialforge.tsdf_expanded_plan import write_tsdf_expanded_block_plan
from spatialforge.tsdf_stream_expansion import (
    propose_tsdf_plan_expansion_streaming,
)

from tests.heavy_fixtures import shared_room_case
from tests.room_fixture import FAR_WALL_X, LEFT_WALL_Y, RIGHT_WALL_Y
from tools.free_space_map import (
    FREE,
    OCCUPIED,
    UNKNOWN,
    band_slab,
    camera_centres_m,
    camera_columns,
    camera_voxel_verdicts,
    classify_columns,
    main,
    window_columns,
)

TEST_ROOT = Path(__file__).resolve().parent
# A band at the cameras' own height, clear of the floor and the ceiling.
BAND = (-0.2, 0.2)
VOXEL = 0.04

ROOM: SimpleNamespace | None = None


def setUpModule() -> None:
    global ROOM
    room = shared_room_case()
    directory = tempfile.TemporaryDirectory(dir=TEST_ROOT)
    root = Path(directory.name)
    write_tsdf_expanded_block_plan(
        room.plan,
        propose_tsdf_plan_expansion_streaming(room.plan, room.session),
        root / "expanded.sftplan",
    )
    for name, plan in (
        ("surface", room.plan),
        ("expanded", load_tsdf_block_plan(root / "expanded.sftplan")),
    ):
        storage = allocate_empty_tsdf_blocks(plan, room.session)
        receipt = fuse_tsdf_plan_streaming(storage, room.session)
        write_tsdf_block_volume(storage, receipt, root / f"{name}.sftvol")
    ROOM = SimpleNamespace(
        directory=directory,
        root=root,
        session=room.session_path,
        surface=root / "surface.sftvol",
        expanded=root / "expanded.sftvol",
    )


def tearDownModule() -> None:
    ROOM.directory.cleanup()


def room_map(path: Path):
    """Each column's verdict, keyed by the global voxel index of its x, y."""

    volume = load_tsdf_block_volume(path)
    values, weights, origin, across = band_slab(volume, 2, *BAND)
    columns = classify_columns(values, weights, min_weight=1)
    verdicts = {
        (origin[0] + a, origin[1] + b): int(columns[a, b])
        for a in range(columns.shape[0])
        for b in range(columns.shape[1])
    }
    return verdicts, columns, origin, across


def column_at(x_m: float, y_m: float) -> tuple[int, int]:
    return (int(np.floor(x_m / VOXEL)), int(np.floor(y_m / VOXEL)))


class RuleTests(unittest.TestCase):
    def test_each_verdict_is_the_one_read_off_by_hand(self) -> None:
        # Five columns of four voxels. A weight of 0 is an unobserved voxel
        # and its value means nothing.
        values = np.array(
            [
                [
                    [1.0, 1.0, 0.4, 1.0],     # all seen, all in front: free
                    [1.0, -0.2, 1.0, 1.0],    # one behind a surface
                    [1.0, 1.0, 1.0, 1.0],     # one never seen
                    [1.0, 0.0, 1.0, 1.0],     # one exactly on a surface
                    [-9.0, -9.0, -9.0, -9.0],  # nothing seen at all
                ]
            ]
        )
        weights = np.array(
            [
                [
                    [3, 3, 3, 3],
                    [3, 3, 3, 3],
                    [3, 0, 3, 3],
                    [3, 3, 3, 3],
                    [0, 0, 0, 0],
                ]
            ],
            dtype=np.uint32,
        )
        self.assertEqual(
            classify_columns(values, weights, min_weight=1).tolist(),
            [[FREE, OCCUPIED, UNKNOWN, OCCUPIED, UNKNOWN]],
        )
        # Asking more of a voxel than it has makes it unobserved: nothing
        # is free any more, and a surface seen too rarely is not one.
        self.assertEqual(
            classify_columns(values, weights, min_weight=4).tolist(),
            [[UNKNOWN] * 5],
        )

    def test_one_unseen_voxel_keeps_a_column_from_being_free(self) -> None:
        values = np.ones((1, 1, 50))
        weights = np.full((1, 1, 50), 7, dtype=np.uint32)
        self.assertEqual(
            classify_columns(values, weights, min_weight=3)[0, 0], FREE
        )
        weights[0, 0, 31] = 2
        self.assertEqual(
            classify_columns(values, weights, min_weight=3)[0, 0], UNKNOWN
        )


class RoomTests(unittest.TestCase):
    def test_the_expanded_volume_knows_the_room_and_the_shell_does_not(
        self,
    ) -> None:
        surface, surface_columns, _, _ = room_map(ROOM.surface)
        expanded, expanded_columns, _, _ = room_map(ROOM.expanded)

        def count(columns, kind) -> int:
            return int(np.count_nonzero(columns == kind))

        self.assertEqual(count(surface_columns, FREE), 935)
        self.assertEqual(count(expanded_columns, FREE), 3016)
        # The middle of the room, between the cameras and the far wall.
        middle = column_at(1.6, 0.0)
        self.assertEqual(surface[middle], UNKNOWN)
        self.assertEqual(expanded[middle], FREE)

    def test_adding_free_space_moves_no_wall(self) -> None:
        surface, surface_columns, _, _ = room_map(ROOM.surface)
        expanded, expanded_columns, _, _ = room_map(ROOM.expanded)
        occupied = {key for key, kind in surface.items() if kind == OCCUPIED}
        self.assertEqual(
            occupied,
            {key for key, kind in expanded.items() if kind == OCCUPIED},
        )
        self.assertEqual(len(occupied), 406)
        # And nothing the shell knew to be free stops being free.
        for key, kind in surface.items():
            if kind == FREE:
                self.assertEqual(expanded[key], FREE)

    def test_walls_are_where_the_room_was_built(self) -> None:
        expanded, _, _, _ = room_map(ROOM.expanded)
        # Just in front of the far wall is free; the wall is occupied; what
        # is behind it was never seen.
        self.assertEqual(expanded[column_at(FAR_WALL_X - 0.2, 0.0)], FREE)
        self.assertEqual(expanded[column_at(FAR_WALL_X + 0.02, 0.0)], OCCUPIED)
        self.assertEqual(
            expanded.get(column_at(FAR_WALL_X + 0.3, 0.0), UNKNOWN), UNKNOWN
        )
        for wall_y, inward in ((LEFT_WALL_Y, -1), (RIGHT_WALL_Y, 1)):
            with self.subTest(wall_y=wall_y):
                self.assertEqual(
                    expanded[column_at(2.0, wall_y + inward * 0.2)], FREE
                )
                self.assertEqual(
                    expanded[column_at(2.0, wall_y - inward * 0.02)],
                    OCCUPIED,
                )
        # No occupied column lies in the open room.
        for key, kind in expanded.items():
            x_m = (key[0] + 0.5) * VOXEL
            y_m = (key[1] + 0.5) * VOXEL
            if kind == OCCUPIED:
                self.assertTrue(
                    x_m > FAR_WALL_X - 0.05
                    or y_m > LEFT_WALL_Y - 0.05
                    or y_m < RIGHT_WALL_Y + 0.05,
                    (x_m, y_m),
                )

    def test_no_camera_was_behind_a_surface(self) -> None:
        volume = load_tsdf_block_volume(ROOM.expanded)
        centres = camera_centres_m(ROOM.session)
        self.assertEqual(len(centres), 20)
        own = camera_voxel_verdicts(volume, centres, min_weight=1)
        self.assertFalse(np.any(own == OCCUPIED))
        # The lookup is not vacuous: moved into the far wall, every camera
        # is behind a surface, and moved out past it, in a voxel nobody saw.
        in_the_wall = centres.copy()
        in_the_wall[:, 0] = FAR_WALL_X + 0.06
        in_the_wall[:, 1:] = 0.0
        self.assertTrue(
            np.all(
                camera_voxel_verdicts(volume, in_the_wall, min_weight=1)
                == OCCUPIED
            )
        )
        beyond = in_the_wall + [1.0, 0.0, 0.0]
        self.assertTrue(
            np.all(
                camera_voxel_verdicts(volume, beyond, min_weight=1)
                == UNKNOWN
            )
        )
        in_the_room = in_the_wall - [0.9, 0.0, 0.0]
        self.assertTrue(
            np.all(
                camera_voxel_verdicts(volume, in_the_room, min_weight=1)
                == FREE
            )
        )
        # And the path lands on the map where the cameras were.
        _, columns, origin, across = room_map(ROOM.expanded)
        path = camera_columns(centres, VOXEL, origin, across)
        self.assertEqual(path.shape, (20, 2))
        self.assertTrue(np.all(path >= 0))
        self.assertTrue(np.all(path < columns.shape))


class CommandTests(unittest.TestCase):
    def run_tool(self, *arguments: str) -> tuple[int, str]:
        stdout = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(io.StringIO()):
            try:
                code = main(list(arguments))
            except SystemExit as stop:
                return 2, str(stop)
        return code, stdout.getvalue()

    def test_it_draws_the_map_and_says_what_is_on_it(self) -> None:
        output = ROOM.root / "map.png"
        code, text = self.run_tool(
            str(ROOM.expanded), str(output),
            "--from-m", str(BAND[0]), "--to-m", str(BAND[1]),
            "--min-weight", "1", "--session", str(ROOM.session),
            "--pixels-per-voxel", "4",
        )
        self.assertEqual(code, 0, text)
        self.assertIn("band: 10 voxels along z", text)
        self.assertIn("map: 80 x 80 columns along x and y, 10.24 m2", text)
        self.assertIn("free: 3016 columns, 4.83 m2 (47.1%)", text)
        self.assertIn("occupied: 406 columns, 0.65 m2 (6.3%)", text)
        self.assertIn("unknown: 2978 columns, 4.76 m2 (46.5%)", text)
        self.assertIn(
            "cameras: 20 posed; the voxel each was in is free for", text
        )
        self.assertTrue(
            text.splitlines()[-2].endswith("behind a surface for 0"), text
        )
        with Image.open(output) as image:
            width, height = image.size
            pixels = np.asarray(image.convert("RGB"))
        self.assertEqual(width, 320)
        self.assertGreater(height, 320)
        # The middle of the room is drawn in the colour of free space:
        # column (40, 40) of the map, four pixels to the voxel, y upward.
        self.assertEqual(tuple(pixels[320 - 162, 162]), (246, 246, 242))

    def test_a_window_is_a_cut_of_the_map_and_changes_no_verdict(
        self,
    ) -> None:
        _, columns, origin, _ = room_map(ROOM.expanded)
        # The map runs from -0.32 m on x and -1.60 m on y, 40 mm a column.
        self.assertEqual(origin[:2], [-8, -40])
        cut, moved = window_columns(
            columns, origin, VOXEL, (0.0, 1.0, -0.52, 0.28)
        )
        # Columns whose centres lie inside: 0.02 to 0.98 m, and -0.50 to
        # 0.26 m. Every edge here falls on a column's side, and keeps the
        # column inside it and not the one beyond.
        self.assertEqual(cut.shape, (25, 20))
        self.assertEqual(moved, [0, -13, origin[2]])
        np.testing.assert_array_equal(cut, columns[8:33, 27:47])
        # A window larger than the map is the map.
        whole, same = window_columns(
            columns, origin, VOXEL, (-50.0, 50.0, -50.0, 50.0)
        )
        np.testing.assert_array_equal(whole, columns)
        self.assertEqual(same, list(origin))
        for within, message in (
            ((1.0, 0.0, -0.5, 0.3), "in that order"),
            ((0.0, 1.0, 0.3, 0.3), "in that order"),
            ((0.0, float("nan"), -0.5, 0.3), "in that order"),
            ((40.0, 41.0, -0.5, 0.3), "no column of the map"),
            ((0.001, 0.002, -0.5, 0.3), "no column of the map"),
        ):
            with self.subTest(within=within):
                with self.assertRaises(SystemExit) as raised:
                    window_columns(columns, origin, VOXEL, within)
                self.assertIn(message, str(raised.exception))

    def test_it_draws_a_window_and_counts_what_is_in_it(self) -> None:
        _, columns, origin, _ = room_map(ROOM.expanded)
        cut = columns[8:33, 27:47]
        output = ROOM.root / "window.png"
        code, text = self.run_tool(
            str(ROOM.expanded), str(output),
            "--from-m", str(BAND[0]), "--to-m", str(BAND[1]),
            "--min-weight", "1", "--session", str(ROOM.session),
            "--pixels-per-voxel", "4",
            "--within", "0.0", "1.0", "-0.52", "0.28",
        )
        self.assertEqual(code, 0, text)
        self.assertIn("map: 25 x 20 columns along x and y, 0.80 m2", text)
        for label, kind in (
            ("free", FREE), ("occupied", OCCUPIED), ("unknown", UNKNOWN)
        ):
            count = int(np.count_nonzero(cut == kind))
            self.assertIn(f"{label}: {count} columns, ", text)
        # Every camera is still looked up, in the window or not.
        self.assertIn("cameras: 20 posed", text)
        with Image.open(output) as image:
            self.assertEqual(image.size[0], 100)
        code, text = self.run_tool(
            str(ROOM.expanded), str(ROOM.root / "nowhere.png"),
            "--from-m", str(BAND[0]), "--to-m", str(BAND[1]),
            "--within", "40", "41", "0", "1",
        )
        self.assertEqual(code, 2)
        self.assertIn("no column of the map", text)
        self.assertFalse((ROOM.root / "nowhere.png").exists())

    def test_it_refuses_what_it_cannot_draw(self) -> None:
        taken = ROOM.root / "taken.png"
        taken.write_bytes(b"keep me")
        for name, arguments, message in (
            (
                "an existing output",
                (str(ROOM.expanded), str(taken),
                 "--from-m", "-0.2", "--to-m", "0.2"),
                "already exists",
            ),
            (
                "a band thinner than a voxel",
                (str(ROOM.expanded), str(ROOM.root / "thin.png"),
                 "--from-m", "0.001", "--to-m", "0.002"),
                "narrower than a 40 mm voxel",
            ),
            (
                "something that is not a volume",
                (str(ROOM.root / "expanded.sftplan"),
                 str(ROOM.root / "plan.png"),
                 "--from-m", "-0.2", "--to-m", "0.2"),
                ".sftvol",
            ),
        ):
            with self.subTest(case=name):
                code, text = self.run_tool(*arguments)
                self.assertEqual(code, 2)
                self.assertIn(message, text)
        self.assertEqual(taken.read_bytes(), b"keep me")
        self.assertFalse((ROOM.root / "thin.png").exists())


if __name__ == "__main__":
    unittest.main()
