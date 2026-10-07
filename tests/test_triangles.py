"""Exact point-to-mesh distance, pinned to two independent references.

Completeness is a count of true-surface points within a threshold of the
mesh. A distance that was slightly long near triangle edges, or an index
that sometimes missed the nearest triangle, would not fail anything; it
would report a slightly less complete reconstruction. So the per-triangle
distance is checked against a different algorithm for the same quantity,
and the grid index against measuring every triangle.
"""

from __future__ import annotations

import math
import unittest
from unittest.mock import patch

import numpy as np

from tools import _triangles
from tools._triangles import (
    TriangleIndex,
    mesh_distance_brute_force,
    point_triangle_distance_squared,
)


def closest_point_by_regions(p, a, b, c) -> np.ndarray:
    """Closest point on a triangle by Voronoi regions (Ericson, 5.1.5).

    A different derivation from the one under test: it classifies the query
    into a vertex, edge or face region from six dot products and never
    measures to an edge it does not need.
    """

    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = ab @ ap, ac @ ap
    if d1 <= 0 and d2 <= 0:
        return a
    bp = p - b
    d3, d4 = ab @ bp, ac @ bp
    if d3 >= 0 and d4 <= d3:
        return b
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        return a + ab * (d1 / (d1 - d3))
    cp = p - c
    d5, d6 = ab @ cp, ac @ cp
    if d6 >= 0 and d5 <= d6:
        return c
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        return a + ac * (d2 / (d2 - d6))
    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)))
    denominator = 1.0 / (va + vb + vc)
    return a + ab * (vb * denominator) + ac * (vc * denominator)


def soup(seed: int, triangles: int, extent: float, size: float):
    """Random small triangles scattered through a cube."""

    rng = np.random.default_rng(seed)
    centres = rng.uniform(-extent, extent, size=(triangles, 1, 3))
    vertices = (centres + rng.uniform(-size, size, size=(triangles, 3, 3))).reshape(
        (-1, 3)
    )
    faces = np.arange(3 * triangles, dtype=np.int64).reshape((-1, 3))
    return vertices, faces


class PointTriangleTests(unittest.TestCase):
    def test_agrees_with_the_region_method_everywhere_around_a_triangle(
        self,
    ) -> None:
        rng = np.random.default_rng(51)
        for _ in range(300):
            a, b, c = rng.uniform(-1.0, 1.0, size=(3, 3))
            points = rng.uniform(-2.0, 2.0, size=(40, 3))
            measured = point_triangle_distance_squared(
                points,
                np.broadcast_to(a, points.shape),
                np.broadcast_to(b, points.shape),
                np.broadcast_to(c, points.shape),
            )
            for point, squared in zip(points, measured):
                closest = closest_point_by_regions(point, a, b, c)
                self.assertAlmostEqual(
                    squared, float((point - closest) @ (point - closest)), places=11
                )

    def test_known_distances(self) -> None:
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([4.0, 0.0, 0.0])
        c = np.array([0.0, 4.0, 0.0])
        cases = (
            ((1.0, 1.0, 3.0), 9.0),      # over the face: straight down
            ((1.0, 1.0, 0.0), 0.0),      # on the face
            ((2.0, -3.0, 0.0), 9.0),     # beside edge ab
            ((-3.0, -4.0, 0.0), 25.0),   # beyond vertex a
            ((5.0, 0.0, 12.0), 145.0),   # beyond vertex b, and above
            ((3.0, 3.0, 0.0), 2.0),      # beside the hypotenuse
        )
        points = np.array([point for point, _ in cases])
        measured = point_triangle_distance_squared(
            points,
            np.broadcast_to(a, points.shape),
            np.broadcast_to(b, points.shape),
            np.broadcast_to(c, points.shape),
        )
        np.testing.assert_allclose(
            measured, [expected for _, expected in cases], atol=1e-12
        )

    def test_a_triangle_with_no_area_is_the_segment_it_collapses_to(
        self,
    ) -> None:
        a = np.array([[0.0, 0.0, 0.0]])
        b = np.array([[2.0, 0.0, 0.0]])
        point = np.array([[1.0, 3.0, 0.0]])
        # Three collinear corners, two coincident corners, and a point.
        self.assertEqual(
            float(
                point_triangle_distance_squared(
                    point, a, b, np.array([[1.0, 0.0, 0.0]])
                )[0]
            ),
            9.0,
        )
        self.assertEqual(
            float(point_triangle_distance_squared(point, a, b, b)[0]), 9.0
        )
        self.assertEqual(
            float(point_triangle_distance_squared(point, a, a, a)[0]), 10.0
        )


class IndexAgainstBruteForceTests(unittest.TestCase):
    def assert_same(self, vertices, faces, queries, reach_m) -> np.ndarray:
        indexed = TriangleIndex(vertices, faces, reach_m=reach_m).distance(
            queries
        )
        expected = mesh_distance_brute_force(
            vertices, faces, queries, reach_m=reach_m
        )
        # The same number, not a close one: both take the smallest of the
        # same per-triangle distances.
        self.assertEqual(indexed.tobytes(), expected.tobytes())
        return indexed

    def test_random_triangle_soup_matches_exactly(self) -> None:
        vertices, faces = soup(52, 1_500, 1.0, 0.03)
        queries = np.random.default_rng(53).uniform(-1.2, 1.2, size=(800, 3))
        for reach_m in (0.02, 0.05, 0.2):
            with self.subTest(reach_m=reach_m):
                distances = self.assert_same(vertices, faces, queries, reach_m)
                self.assertTrue(np.any(np.isfinite(distances)))
                self.assertTrue(np.any(np.isinf(distances)))
                self.assertLessEqual(
                    float(distances[np.isfinite(distances)].max()), reach_m
                )

    def test_triangles_much_larger_than_the_reach(self) -> None:
        # Each triangle is filed in dozens of cells and a query deep inside
        # one is nowhere near any of its vertices.
        vertices, faces = soup(54, 40, 1.0, 0.6)
        queries = np.random.default_rng(55).uniform(-1.5, 1.5, size=(600, 3))
        self.assert_same(vertices, faces, queries, 0.05)

    def test_a_flat_sheet_is_at_the_height_of_the_query(self) -> None:
        steps = np.arange(0.0, 1.0001, 0.05)
        grid_x, grid_y = np.meshgrid(steps, steps, indexing="ij")
        vertices = np.stack(
            [grid_x.ravel(), grid_y.ravel(), np.zeros(grid_x.size)], axis=1
        )
        side = len(steps)
        faces = []
        for i in range(side - 1):
            for j in range(side - 1):
                corner = i * side + j
                faces.append([corner, corner + side, corner + 1])
                faces.append([corner + 1, corner + side, corner + side + 1])
        faces = np.array(faces)
        rng = np.random.default_rng(56)
        queries = np.column_stack(
            [
                rng.uniform(0.0, 1.0, 400),
                rng.uniform(0.0, 1.0, 400),
                rng.uniform(-0.03, 0.03, 400),
            ]
        )
        distances = self.assert_same(vertices, faces, queries, 0.04)
        np.testing.assert_allclose(distances, np.abs(queries[:, 2]), atol=1e-12)
        # The nearest vertex would have been up to 35 mm away.
        beside = np.array([[0.525, 0.525, 0.001]])
        self.assertAlmostEqual(
            float(TriangleIndex(vertices, faces, reach_m=0.04).distance(beside)[0]),
            0.001,
        )
        self.assertGreater(
            float(np.linalg.norm(vertices - beside, axis=1).min()), 0.035
        )

    def test_working_in_small_pieces_changes_nothing(self) -> None:
        vertices, faces = soup(57, 600, 1.0, 0.05)
        queries = np.random.default_rng(58).uniform(-1.1, 1.1, size=(300, 3))
        whole = TriangleIndex(vertices, faces, reach_m=0.1).distance(queries)
        with (
            patch.object(_triangles, "_MAX_CANDIDATES", 5),
            patch.object(_triangles, "_MAX_QUERIES", 7),
        ):
            pieces = TriangleIndex(vertices, faces, reach_m=0.1).distance(
                queries
            )
        self.assertEqual(pieces.tobytes(), whole.tobytes())

    def test_the_reach_is_inclusive_and_beyond_it_is_no_answer(self) -> None:
        vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        faces = np.array([[0, 1, 2]])
        index = TriangleIndex(vertices, faces, reach_m=0.25)
        distances = index.distance(
            np.array(
                [
                    [0.25, 0.25, 0.25],
                    [0.25, 0.25, 0.2500001],
                    [50.0, 50.0, 50.0],
                    [1e12, -1e12, 0.0],
                ]
            )
        )
        self.assertEqual(distances[0], 0.25)
        self.assertTrue(np.all(np.isinf(distances[1:])))
        self.assertEqual(index.reach_m, 0.25)
        self.assertEqual(index.distance(np.empty((0, 3))).shape, (0,))


class ValidationTests(unittest.TestCase):
    def test_unusable_meshes_and_queries_are_refused(self) -> None:
        vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        faces = np.array([[0, 1, 2]])
        for bad_vertices, message in (
            (np.zeros((3, 2)), "vertices must have shape"),
            (np.full((3, 3), np.nan), "vertices must be finite"),
        ):
            with self.assertRaisesRegex(ValueError, message):
                TriangleIndex(bad_vertices, faces, reach_m=0.1)
        for bad_faces, message in (
            (np.zeros((0, 3), dtype=np.int64), "non-empty"),
            (np.zeros((1, 4), dtype=np.int64), "non-empty"),
            (np.array([[0.0, 1.0, 2.0]]), "non-empty"),
            (np.array([[0, 1, 3]]), "do not exist"),
            (np.array([[0, 1, -1]]), "do not exist"),
        ):
            with self.assertRaisesRegex(ValueError, message):
                TriangleIndex(vertices, bad_faces, reach_m=0.1)
        for reach_m in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "reach_m"):
                TriangleIndex(vertices, faces, reach_m=reach_m)
        with self.assertRaisesRegex(ValueError, "too large for this reach"):
            TriangleIndex(vertices * 1e9, faces, reach_m=1e-12)
        index = TriangleIndex(vertices, faces, reach_m=0.1)
        with self.assertRaisesRegex(ValueError, "queries must have shape"):
            index.distance(np.zeros(3))
        with self.assertRaisesRegex(ValueError, "queries must be finite"):
            index.distance(np.full((1, 3), np.inf))

    def test_triangles_far_larger_than_the_reach_are_refused(self) -> None:
        vertices = np.array(
            [[0.0, 0.0, 0.0], [900.0, 0.0, 0.0], [0.0, 900.0, 900.0]]
        )
        with self.assertRaisesRegex(ValueError, "too large for this reach"):
            TriangleIndex(vertices, np.array([[0, 1, 2]]), reach_m=1.0)
        self.assertTrue(math.isfinite(_triangles._MAX_FILINGS))


if __name__ == "__main__":
    unittest.main()
