"""Placing a scan in a model with no starting guess, where the answer is planted.

The search returns a placement and a registration then refines it, so a
wrong placement would not show as a wrong number: it would show as a fit
that settles somewhere else, or not at all. Each piece is therefore given a
scene whose answer is known.

The room planted here is not a rectangle. One wall steps back half way
along, and a cupboard stands in a corner, so that turning it half way round
does not lay it on itself. It is turned 117 degrees about a tilted axis and
moved five metres, which no registration recovers from an identity start.
The bare rectangle the search must refuse is tested beside it.
"""

from __future__ import annotations

import math
import unittest
from unittest.mock import patch

import numpy as np

from spatialforge.replay import replay_session

from tests.heavy_fixtures import shared_room_case
from tests.room_fixture import (
    BOX,
    CEILING_Z,
    FAR_WALL_X,
    FLOOR_Z,
    LEFT_WALL_Y,
    RIGHT_WALL_Y,
)
from tools._frame_search import (
    AMBIGUITY_RATIO,
    MIN_AGREEMENT,
    PLANE_BIN_M,
    FrameCandidate,
    _histogram,
    choose_frame,
    dominant_axes,
    oriented_depth,
    proper_signed_permutations,
    search_frame,
)
from tools._nearest import NearestPointIndex
from tools.surface_accuracy_report import register_point_to_plane

from spatialforge.point_cloud import _validate_reconstruction_contract

SPACING_M = 0.02


def rigid(axis, degrees: float, translation) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    angle = math.radians(degrees)
    cross = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    matrix = np.eye(4)
    matrix[:3, :3] = (
        np.eye(3)
        + math.sin(angle) * cross
        + (1.0 - math.cos(angle)) * (cross @ cross)
    )
    matrix[:3, 3] = translation
    return matrix


def apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def difference(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    """Angle in degrees and distance in metres between two placements."""

    relative = first[:3, :3].T @ second[:3, :3]
    cosine = max(-1.0, min(1.0, (float(np.trace(relative)) - 1.0) / 2.0))
    return (
        math.degrees(math.acos(cosine)),
        float(np.linalg.norm(first[:3, 3] - second[:3, 3])),
    )


def rectangle(axis: int, value: float, first, second, normal):
    """Oriented points every ``SPACING_M`` on an axis-aligned rectangle."""

    a = np.arange(first[0] + SPACING_M / 2, first[1], SPACING_M)
    b = np.arange(second[0] + SPACING_M / 2, second[1], SPACING_M)
    grid_a, grid_b = np.meshgrid(a, b, indexing="ij")
    points = np.empty((grid_a.size, 3))
    others = [index for index in range(3) if index != axis]
    points[:, axis] = value
    points[:, others[0]] = grid_a.ravel()
    points[:, others[1]] = grid_b.ravel()
    return points, np.tile(np.asarray(normal, dtype=np.float64), (len(points), 1))


def gather(parts) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.concatenate([points for points, _ in parts]),
        np.concatenate([normals for _, normals in parts]),
    )


def stepped_room() -> tuple[np.ndarray, np.ndarray]:
    """A room about 4 m by 3 m whose back wall steps out, and a cupboard.

    Normals face into the room. No plane is a whole number of histogram
    bins from another.
    """

    long, short, deep, step = 4.037, 3.013, 3.629, 2.011
    height = (0.0, 2.409)
    return gather(
        [
            # Walls, round the outline.
            rectangle(1, 0.0, (0.0, long), height, (0, 1, 0)),
            rectangle(0, long, (0.0, deep), height, (-1, 0, 0)),
            rectangle(1, deep, (step, long), height, (0, -1, 0)),
            rectangle(0, step, (short, deep), height, (1, 0, 0)),
            rectangle(1, short, (0.0, step), height, (0, -1, 0)),
            rectangle(0, 0.0, (0.0, short), height, (1, 0, 0)),
            # Floor and ceiling, in the two rectangles of the outline.
            rectangle(2, 0.0, (0.0, long), (0.0, short), (0, 0, 1)),
            rectangle(2, 0.0, (step, long), (short, deep), (0, 0, 1)),
            rectangle(2, height[1], (0.0, long), (0.0, short), (0, 0, -1)),
            rectangle(2, height[1], (step, long), (short, deep), (0, 0, -1)),
            # A cupboard in the corner at the origin.
            rectangle(0, 0.613, (0.0, 1.227), (0.0, 1.811), (1, 0, 0)),
            rectangle(1, 1.227, (0.0, 0.613), (0.0, 1.811), (0, 1, 0)),
            rectangle(2, 1.811, (0.0, 0.613), (0.0, 1.227), (0, 0, 1)),
        ]
    )


def bare_room() -> tuple[np.ndarray, np.ndarray]:
    """A plain rectangular room: the same turned half way round."""

    height = (0.0, 2.4)
    return gather(
        [
            rectangle(1, 0.0, (0.0, 4.0), height, (0, 1, 0)),
            rectangle(1, 3.0, (0.0, 4.0), height, (0, -1, 0)),
            rectangle(0, 0.0, (0.0, 3.0), height, (1, 0, 0)),
            rectangle(0, 4.0, (0.0, 3.0), height, (-1, 0, 0)),
            rectangle(2, 0.0, (0.0, 4.0), (0.0, 3.0), (0, 0, 1)),
            rectangle(2, 2.4, (0.0, 4.0), (0.0, 3.0), (0, 0, -1)),
        ]
    )


# Where the model's frame sits relative to the room as it was drawn, so
# that the model's planes are not along its coordinate axes either.
MODEL_FROM_ROOM = rigid((1.0, 0.4, -0.3), 23.0, (0.7, -1.1, 0.4))
# The motion the search is not told.
MODEL_FROM_SCAN = rigid((0.3, 1.0, 0.5), 117.0, (3.1, -2.6, 4.2))


def outsides() -> tuple[np.ndarray, np.ndarray]:
    """The outer faces of the stepped room's walls, a quarter metre out.

    Facing away from the room. A model made from a drawing of a building
    has them; a scan made from inside it has none.
    """

    x = (-0.25, 4.037 + 0.25)
    y = (-0.25, 3.629 + 0.25)
    z = (-0.25, 2.409 + 0.25)
    return gather(
        [
            rectangle(0, x[0], y, z, (-1, 0, 0)),
            rectangle(0, x[1], y, z, (1, 0, 0)),
            rectangle(1, y[0], x, z, (0, -1, 0)),
            rectangle(1, y[1], x, z, (0, 1, 0)),
            rectangle(2, z[0], x, y, (0, 0, -1)),
            rectangle(2, z[1], x, y, (0, 0, 1)),
        ]
    )


def by_size(model, source) -> float:
    """The slide that matches large planes to large planes.

    What the search did before it counted how much of the scan a slide
    explains. Kept here to show a scene on which the two differ.
    """

    model_all = np.concatenate(model)
    source_all = np.concatenate(source)
    model_low = float(model_all.min())
    source_low = float(source_all.min())
    model_bins = int((float(model_all.max()) - model_low) / PLANE_BIN_M) + 1
    source_bins = int((float(source_all.max()) - source_low) / PLANE_BIN_M) + 1
    agreement = np.zeros(model_bins + source_bins - 1)
    for facing in range(2):
        agreement += np.correlate(
            np.sqrt(_histogram(model[facing], model_low, model_bins)),
            np.sqrt(_histogram(source[facing], source_low, source_bins)),
            mode="full",
        )
    lag = int(np.argmax(agreement)) - (source_bins - 1)
    return model_low - source_low + lag * PLANE_BIN_M


def scene(room, beyond=None):
    """A model, and a scan of part of it in a frame of its own.

    ``beyond`` is surface the model holds and the scan never saw.
    """

    points, normals = room
    if beyond is None:
        model_points, model_normals = points, normals
    else:
        model_points = np.concatenate([points, beyond[0]])
        model_normals = np.concatenate([normals, beyond[1]])
    model_points = apply(MODEL_FROM_ROOM, model_points)
    model_normals = model_normals @ MODEL_FROM_ROOM[:3, :3].T
    # The scan saw the surface near every third point of the model, and
    # none of the room's first half metre. Not the model's own points:
    # each is slid up to 4 mm within its plane, so it is on the surface
    # and between the model's samples.
    seen = np.zeros(len(points), dtype=bool)
    seen[1::3] = True
    seen &= (points[:, 0] > 0.5) & (points[:, 1] > 0.4)
    slide = np.random.default_rng(33).uniform(
        -0.004, 0.004, size=(int(seen.sum()), 3)
    )
    slide -= (slide * normals[seen]).sum(axis=1, keepdims=True) * normals[seen]
    scan_from_room = np.linalg.inv(MODEL_FROM_SCAN) @ MODEL_FROM_ROOM
    return (
        model_points,
        model_normals,
        apply(scan_from_room, points[seen] + slide),
        normals[seen] @ scan_from_room[:3, :3].T,
    )


class DominantAxesTests(unittest.TestCase):
    def test_finds_planted_axes_among_noise(self) -> None:
        generator = np.random.default_rng(31)
        planted = rigid((0.2, -0.7, 1.0), 41.0, (0, 0, 0))[:3, :3]
        # Unequal shares, either way round, a degree of scatter, and a
        # third of the normals facing nowhere in particular.
        counts = (9000, 5000, 2500)
        parts = []
        for k, count in enumerate(counts):
            signs = generator.choice([-1.0, 1.0], size=count)
            parts.append(signs[:, None] * planted[:, k])
        normals = np.concatenate(parts)
        normals = normals + generator.normal(0.0, 0.017, size=normals.shape)
        stray = generator.normal(size=(8000, 3))
        normals = np.concatenate([normals, stray])
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        generator.shuffle(normals)

        axes, shares = dominant_axes(normals)

        self.assertAlmostEqual(float(np.linalg.det(axes)), 1.0, places=12)
        np.testing.assert_allclose(axes.T @ axes, np.eye(3), atol=1e-12)
        # Most normals first, and each within a quarter of a degree of
        # the axis that was planted.
        for k in range(3):
            cosine = abs(float(axes[:, k] @ planted[:, k]))
            self.assertGreater(cosine, math.cos(math.radians(0.25)))
        self.assertGreater(shares[0], shares[1])
        self.assertGreater(shares[1], shares[2])
        self.assertAlmostEqual(shares[0], 9000 / 24500, delta=0.03)

    def test_surfaces_that_do_not_face_three_ways_are_refused(self) -> None:
        generator = np.random.default_rng(32)
        ball = generator.normal(size=(20000, 3))
        ball /= np.linalg.norm(ball, axis=1, keepdims=True)
        with self.assertRaisesRegex(SystemExit, "three perpendicular"):
            dominant_axes(ball)
        # One wall, however much of it, is one direction.
        wall = np.tile([0.0, 0.6, 0.8], (5000, 1))
        with self.assertRaisesRegex(SystemExit, "three perpendicular"):
            dominant_axes(wall)
        # Two walls and nothing across them.
        two = np.concatenate([wall, np.tile([1.0, 0.0, 0.0], (5000, 1))])
        with self.assertRaisesRegex(SystemExit, "three perpendicular"):
            dominant_axes(two)
        with self.assertRaises(ValueError):
            dominant_axes(np.zeros((5, 2)))

    def test_there_are_24_ways_to_lay_axes_on_axes(self) -> None:
        rotations = proper_signed_permutations()
        self.assertEqual(len(rotations), 24)
        self.assertEqual(
            len({tuple(matrix.ravel().tolist()) for matrix in rotations}), 24
        )
        for matrix in rotations:
            np.testing.assert_array_equal(matrix @ matrix.T, np.eye(3))
            self.assertEqual(float(np.linalg.det(matrix)), 1.0)
            self.assertEqual(int(np.count_nonzero(matrix)), 3)


class PlantedPlacementTests(unittest.TestCase):
    def test_finds_a_placement_it_was_not_told_and_the_fit_finishes_it(
        self,
    ) -> None:
        model_points, model_normals, points, normals = scene(stepped_room())
        coarse = NearestPointIndex(model_points[::4], cell_m=0.08)

        candidates = search_frame(
            points, normals, model_points[::4], model_normals[::4], coarse
        )
        best = choose_frame(candidates)

        self.assertEqual(len(candidates), 24)
        # A starting guess, not an answer. The scan does not hold the
        # wall one axis of the model starts at, so along it the planes
        # are laid on each other to a histogram bin: it is 7 mm out.
        angle, distance = difference(best.model_from_source, MODEL_FROM_SCAN)
        self.assertLess(angle, 0.01)
        self.assertGreater(distance, 0.001)
        self.assertLess(distance, PLANE_BIN_M)
        self.assertGreater(best.agreement, 0.99)
        # The next best is the room turned a quarter of the way round.
        # Four fifths of the scan still lands on some wall facing the
        # right way, in a room this bare, which is why a placement is
        # judged against the next and not alone.
        self.assertLess(
            candidates[1].agreement, AMBIGUITY_RATIO * best.agreement
        )
        self.assertGreater(candidates[1].agreement, 0.7)
        angle, _ = difference(
            best.model_from_source, candidates[1].model_from_source
        )
        self.assertAlmostEqual(angle, 90.0, delta=1.0)

        # From an identity start the registration goes nowhere: it pairs
        # the eighth of the scan that happens to lie near something and
        # stays 117 degrees and nearly six metres from the answer.
        lost, stage = register_point_to_plane(
            points,
            model_points[::4],
            model_normals[::4],
            coarse,
            np.eye(4),
            limit_m=0.08,
            iterations=12,
        )
        angle, distance = difference(lost, MODEL_FROM_SCAN)
        self.assertLess(stage.matched_fraction, 0.2)
        self.assertGreater(angle, 100.0)
        self.assertGreater(distance, 5.0)
        # From the placement it arrives at the motion that was planted.
        fitted, _ = register_point_to_plane(
            points,
            model_points[::4],
            model_normals[::4],
            coarse,
            best.model_from_source,
            limit_m=0.08,
            iterations=12,
        )
        fitted, stage = register_point_to_plane(
            points,
            model_points,
            model_normals,
            NearestPointIndex(model_points, cell_m=0.02),
            fitted,
            limit_m=0.02,
            iterations=6,
        )
        # To a micron, and it has stopped moving. Not to the last digit:
        # two of the 33,813 samples sit at the cupboard's top edge and pair
        # with a model point of the face next to theirs, which holds the
        # fit a third of a micron from where it was planted.
        np.testing.assert_allclose(fitted, MODEL_FROM_SCAN, atol=1e-6)
        self.assertEqual(len(points), 33813)
        self.assertEqual(stage.matched_fraction, 1.0)
        self.assertLess(stage.last_step_m, 1e-12)
        self.assertLess(stage.last_step_rad, 1e-12)

    def test_the_outsides_of_the_walls_do_not_take_the_scan(self) -> None:
        # The living room's model has them, and each is larger than the
        # wall inside it: nothing stands against the outside of a house.
        # A slide chosen by matching large planes to large planes lays
        # the scan's walls on those, and the scan ends up outside.
        model_points, model_normals, points, normals = scene(
            stepped_room(), beyond=outsides()
        )
        thinned = model_points[::4], model_normals[::4]
        index = NearestPointIndex(thinned[0], cell_m=0.08)

        best = choose_frame(search_frame(points, normals, *thinned, index))

        angle, distance = difference(best.model_from_source, MODEL_FROM_SCAN)
        self.assertLess(angle, 0.01)
        self.assertLess(distance, math.sqrt(3.0) * PLANE_BIN_M)
        self.assertGreater(best.agreement, 0.99)

        # The scene is one where it matters: the rule this replaced puts
        # a fifth of the scan on the model, a quarter turn and eight
        # metres from where it belongs.
        with patch("tools._frame_search._best_shift", by_size):
            lost = search_frame(points, normals, *thinned, index)
        angle, distance = difference(lost[0].model_from_source, MODEL_FROM_SCAN)
        self.assertLess(lost[0].agreement, 0.3)
        self.assertGreater(angle, 80.0)
        self.assertGreater(distance, 5.0)
        with self.assertRaisesRegex(SystemExit, "no placement"):
            choose_frame(lost)

    def test_a_room_that_is_the_same_turned_round_is_refused(self) -> None:
        points, normals = bare_room()
        model_points = apply(MODEL_FROM_ROOM, points)
        model_normals = normals @ MODEL_FROM_ROOM[:3, :3].T
        scan_from_model = np.linalg.inv(MODEL_FROM_SCAN)
        candidates = search_frame(
            apply(scan_from_model, model_points[1::3]),
            model_normals[1::3] @ scan_from_model[:3, :3].T,
            model_points,
            model_normals,
            NearestPointIndex(model_points, cell_m=0.08),
        )
        # Two placements fit, a half turn apart, and nothing in the scan
        # prefers one.
        self.assertGreater(candidates[0].agreement, 0.9)
        self.assertGreater(candidates[1].agreement, 0.9)
        angle, _ = difference(
            candidates[0].model_from_source, candidates[1].model_from_source
        )
        self.assertAlmostEqual(angle, 180.0, delta=1.0)
        with self.assertRaisesRegex(SystemExit, "two placements"):
            choose_frame(candidates)

    def test_what_is_trusted_is_decided_by_two_thresholds(self) -> None:
        def ranked(*agreements: float) -> list[FrameCandidate]:
            return [FrameCandidate(np.eye(4), value) for value in agreements]

        clear = ranked(0.9, 0.6, 0.1)
        self.assertIs(choose_frame(clear), clear[0])
        with self.assertRaisesRegex(SystemExit, "no placement"):
            choose_frame(ranked(MIN_AGREEMENT - 0.01, 0.1))
        with self.assertRaisesRegex(SystemExit, "two placements"):
            choose_frame(ranked(0.9, 0.9 * AMBIGUITY_RATIO + 0.01))
        just = ranked(0.9, 0.9 * AMBIGUITY_RATIO - 0.01)
        self.assertIs(choose_frame(just), just[0])


class OrientedDepthTests(unittest.TestCase):
    def test_normals_are_the_room_generators_planes(self) -> None:
        room = shared_room_case()
        camera, depth_scale_m = _validate_reconstruction_contract(room.session)
        observations = replay_session(room.session).observations
        points, normals = oriented_depth(
            room.session, observations, camera, depth_scale_m, 2, np.eye(4)
        )

        self.assertEqual(points.shape, normals.shape)
        self.assertGreater(len(points), 5000)
        np.testing.assert_allclose(
            np.linalg.norm(normals, axis=1), 1.0, atol=1e-12
        )
        # Each plane the generator drew, and the way it faces the room.
        x0, x1, y0, y1, _, z1 = BOX
        planes = [
            (0, FAR_WALL_X, (-1, 0, 0)),
            (1, LEFT_WALL_Y, (0, -1, 0)),
            (1, RIGHT_WALL_Y, (0, 1, 0)),
            (2, FLOOR_Z, (0, 0, 1)),
            (2, CEILING_Z, (0, 0, -1)),
            (0, x0, (-1, 0, 0)),
            (0, x1, (1, 0, 0)),
            (1, y0, (0, -1, 0)),
            (1, y1, (0, 1, 0)),
            (2, z1, (0, 0, 1)),
        ]
        judged = []
        agreeing = 0
        for axis, value, normal in planes:
            # Well clear of the 4 mm of noise, and of any other plane.
            on_it = np.abs(points[:, axis] - value) < 0.02
            for other_axis, other_value, _ in planes:
                if (other_axis, other_value) != (axis, value):
                    on_it &= np.abs(points[:, other_axis] - other_value) > 0.1
            cosine = normals[on_it] @ np.asarray(normal, dtype=np.float64)
            judged.append(len(cosine))
            agreeing += int(np.count_nonzero(cosine > math.cos(math.radians(8))))
        # Most samples are on one plane and clear of the rest, and nearly
        # all of those face the way that plane does.
        self.assertGreater(sum(judged), 0.7 * len(points))
        self.assertGreater(agreeing, 0.95 * sum(judged))
        # Not only the wall the camera faces. The side walls are seen
        # along their length, where depth changes by a tenth from one
        # pixel to the next but one, and a rule that took a change in
        # depth for an edge would keep none of them.
        for count in judged[:3]:
            self.assertGreater(count, 400)
        # This camera is level and the floor and ceiling only enter the
        # last rows of its image, inside the margin a normal needs.
        self.assertEqual(judged[3:5], [0, 0])

    def test_an_image_too_small_for_a_normal_is_refused(self) -> None:
        room = shared_room_case()
        camera, depth_scale_m = _validate_reconstruction_contract(room.session)
        tiny = type(
            "Camera",
            (),
            dict(width=8, height=8, fx=6.0, fy=6.0, cx=3.5, cy=3.5),
        )()
        with self.assertRaisesRegex(SystemExit, "too small"):
            oriented_depth(room.session, [], tiny, depth_scale_m, 2, np.eye(4))


if __name__ == "__main__":
    unittest.main()
