"""The grid nearest-neighbour search, pinned against comparing everything.

A surface-accuracy figure is a statistic of nearest-neighbour distances, so
an index that occasionally returns a near point instead of the nearest one
would not crash; it would quietly report a slightly worse reconstruction.
Every answer here is therefore checked against the quadratic search, to the
last bit of the distance and to the index.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from tools import _nearest
from tools._nearest import NearestPointIndex, nearest_points_brute_force


def cloud(seed: int, count: int, low: float, high: float) -> np.ndarray:
    return np.random.default_rng(seed).uniform(low, high, size=(count, 3))


class GridAgainstBruteForceTests(unittest.TestCase):
    def assert_same_answer(
        self,
        reference: np.ndarray,
        queries: np.ndarray,
        *,
        cell_m: float,
        limit_m: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        indices, distances = NearestPointIndex(
            reference, cell_m=cell_m
        ).nearest(queries, limit_m=limit_m)
        expected_indices, expected_distances = nearest_points_brute_force(
            reference, queries, limit_m=limit_m
        )
        np.testing.assert_array_equal(indices, expected_indices)
        # Bit-for-bit, not approximately: both paths add the same three
        # squares in the same order.
        self.assertEqual(distances.tobytes(), expected_distances.tobytes())
        return indices, distances

    def test_random_clouds_match_exactly(self) -> None:
        reference = cloud(1, 3_000, -1.0, 1.0)
        # Wider than the reference, so some queries start outside its grid.
        queries = cloud(2, 700, -1.4, 1.4)
        for cell_m, limit_m in (
            (0.05, 0.05),
            (0.05, 0.4),
            (0.2, 0.1),
            (0.01, 1.0),
            (3.0, 0.5),
        ):
            with self.subTest(cell_m=cell_m, limit_m=limit_m):
                indices, _ = self.assert_same_answer(
                    reference, queries, cell_m=cell_m, limit_m=limit_m
                )
                self.assertTrue(np.any(indices >= 0))

    def test_far_neighbours_are_found_by_coarser_grids(self) -> None:
        # Nothing lies within thirty cells of any query, so every answer
        # has to come from a fallback level rather than the base grid.
        reference = cloud(3, 400, 0.0, 1.0)
        queries = cloud(4, 60, 0.0, 1.0) + np.array([0.0, 0.0, 1.3])
        indices, distances = self.assert_same_answer(
            reference, queries, cell_m=0.01, limit_m=4.0
        )
        self.assertTrue(np.all(indices >= 0))
        self.assertGreater(float(distances.min()), 0.3)

    def test_the_answer_does_not_depend_on_the_cell_size(self) -> None:
        reference = cloud(5, 2_000, -1.0, 1.0)
        queries = cloud(6, 300, -1.0, 1.0)
        answers = [
            NearestPointIndex(reference, cell_m=cell_m).nearest(
                queries, limit_m=0.5
            )
            for cell_m in (0.02, 0.07, 0.5)
        ]
        for indices, distances in answers[1:]:
            np.testing.assert_array_equal(indices, answers[0][0])
            self.assertEqual(distances.tobytes(), answers[0][1].tobytes())

    def test_working_in_small_pieces_changes_nothing(self) -> None:
        reference = cloud(7, 1_500, -1.0, 1.0)
        queries = cloud(8, 200, -1.2, 1.2)
        whole = NearestPointIndex(reference, cell_m=0.1).nearest(
            queries, limit_m=0.6
        )
        with (
            patch.object(_nearest, "_MAX_CANDIDATES", 7),
            patch.object(_nearest, "_MAX_QUERIES", 5),
        ):
            pieces = NearestPointIndex(reference, cell_m=0.1).nearest(
                queries, limit_m=0.6
            )
        np.testing.assert_array_equal(pieces[0], whole[0])
        self.assertEqual(pieces[1].tobytes(), whole[1].tobytes())


class TieAndLimitTests(unittest.TestCase):
    def test_ties_resolve_to_the_lowest_reference_index(self) -> None:
        # Four points at the same distance from the query, listed so that
        # the lowest index is not the first one any grid ordering meets.
        reference = np.array(
            [
                [0.25, 0.0, 0.0],
                [-0.25, 0.0, 0.0],
                [0.0, 0.25, 0.0],
                [0.0, -0.25, 0.0],
                [0.25, 0.0, 0.0],
            ]
        )
        query = np.zeros((1, 3))
        for cell_m in (0.05, 0.3, 2.0):
            with self.subTest(cell_m=cell_m):
                indices, distances = NearestPointIndex(
                    reference, cell_m=cell_m
                ).nearest(query, limit_m=1.0)
                self.assertEqual(indices.tolist(), [0])
                self.assertEqual(distances.tolist(), [0.25])

    def test_a_point_exactly_at_the_limit_counts(self) -> None:
        reference = np.zeros((1, 3))
        queries = np.array([[0.5, 0.0, 0.0], [0.75, 0.0, 0.0]])
        indices, distances = NearestPointIndex(
            reference, cell_m=0.125
        ).nearest(queries, limit_m=0.5)
        self.assertEqual(indices.tolist(), [0, -1])
        self.assertEqual(distances.tolist(), [0.5, float("inf")])

    def test_beyond_the_limit_is_no_answer_rather_than_a_guess(self) -> None:
        reference = cloud(9, 500, 0.0, 1.0)
        queries = np.array([[5.0, 5.0, 5.0], [0.5, 0.5, 0.5]])
        indices, distances = NearestPointIndex(
            reference, cell_m=0.05
        ).nearest(queries, limit_m=0.3)
        self.assertEqual(int(indices[0]), -1)
        self.assertTrue(np.isinf(distances[0]))
        self.assertGreaterEqual(int(indices[1]), 0)
        self.assertLessEqual(float(distances[1]), 0.3)

    def test_a_query_absurdly_far_away_is_merely_unanswered(self) -> None:
        # Its cell index would not fit an integer; that must not surface
        # as an overflow or as a wrapped index that lands inside the grid.
        reference = cloud(10, 50, 0.0, 1e-6)
        queries = np.array([[1e9, -1e9, 1e9], [5e-7, 5e-7, 5e-7]])
        indices, distances = NearestPointIndex(
            reference, cell_m=1e-10
        ).nearest(queries, limit_m=1e-9)
        self.assertEqual(int(indices[0]), -1)
        self.assertTrue(np.isinf(distances[0]))
        expected = nearest_points_brute_force(
            reference, queries, limit_m=1e-9
        )
        np.testing.assert_array_equal(indices, expected[0])
        self.assertEqual(distances.tobytes(), expected[1].tobytes())

    def test_no_queries_is_an_empty_answer(self) -> None:
        indices, distances = NearestPointIndex(
            np.zeros((1, 3)), cell_m=1.0
        ).nearest(np.empty((0, 3)), limit_m=1.0)
        self.assertEqual(indices.shape, (0,))
        self.assertEqual(distances.shape, (0,))


class ValidationTests(unittest.TestCase):
    def test_unusable_inputs_are_refused(self) -> None:
        good = np.zeros((2, 3))
        with self.assertRaisesRegex(ValueError, "reference must have shape"):
            NearestPointIndex(np.zeros((2, 2)), cell_m=1.0)
        with self.assertRaisesRegex(ValueError, "at least one point"):
            NearestPointIndex(np.empty((0, 3)), cell_m=1.0)
        with self.assertRaisesRegex(ValueError, "reference must be finite"):
            NearestPointIndex(np.full((1, 3), np.nan), cell_m=1.0)
        for cell_m in (0.0, -1.0, float("inf"), float("nan")):
            with self.assertRaisesRegex(ValueError, "cell_m"):
                NearestPointIndex(good, cell_m=cell_m)
        index = NearestPointIndex(good, cell_m=1.0)
        self.assertEqual(index.cell_m, 1.0)
        with self.assertRaisesRegex(ValueError, "queries must have shape"):
            index.nearest(np.zeros(3), limit_m=1.0)
        with self.assertRaisesRegex(ValueError, "queries must be finite"):
            index.nearest(np.full((1, 3), np.inf), limit_m=1.0)
        for limit_m in (0.0, -1.0, float("inf"), float("nan")):
            with self.assertRaisesRegex(ValueError, "limit_m"):
                index.nearest(good, limit_m=limit_m)

    def test_an_extent_too_large_to_index_is_refused(self) -> None:
        reference = np.array([[0.0, 0.0, 0.0], [1e9, 1e9, 1e9]])
        with self.assertRaisesRegex(ValueError, "too large"):
            NearestPointIndex(reference, cell_m=1e-12)


if __name__ == "__main__":
    unittest.main()
