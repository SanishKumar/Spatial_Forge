"""Exact nearest neighbour in a fixed point set, on a uniform grid.

Measuring a reconstruction against a ground-truth model means asking, for a
few hundred thousand points, which of ten million model points is closest.
Brute force is 10^12 distances. This sorts the model into cubic cells and,
for each query, looks only at the 27 cells around it.

That shortcut is exact under one condition, and the condition is checked
rather than assumed: a neighbour found at distance ``d`` no greater than the
cell size is the true nearest, because anything nearer would also lie within
``d`` and therefore inside those 27 cells. Queries whose best candidate is
farther than a cell are not answered at that level; they are retried on a
grid of twice the size, up to a stated limit. Beyond the limit a query is
reported as having no neighbour, never as having an approximate one.

Ties resolve to the lowest index in the reference array, so the answer does
not depend on the grid.
"""

from __future__ import annotations

import numpy as np

# Distances for this many candidate pairs are held in memory at once.
_MAX_CANDIDATES = 4_000_000
_MAX_QUERIES = 262_144
_NO_INDEX = np.iinfo(np.int64).max


class NearestPointIndex:
    """A fixed reference point set, sorted once into cells of ``cell_m``.

    The base grid is kept; the coarser grids that far-away queries fall
    back to are built for the call that needs them and then dropped, since
    each is another sorted copy of the reference.
    """

    def __init__(self, reference: np.ndarray, *, cell_m: float) -> None:
        self._reference = _points(reference, "reference")
        if len(self._reference) == 0:
            raise ValueError("reference must contain at least one point")
        if not (np.isfinite(cell_m) and cell_m > 0.0):
            raise ValueError("cell_m must be finite and positive")
        self._cell_m = float(cell_m)
        self._base = _Level(self._reference, self._cell_m)

    @property
    def cell_m(self) -> float:
        return self._cell_m

    def nearest(
        self,
        queries: np.ndarray,
        *,
        limit_m: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Index of and distance to the nearest reference point per query.

        A query with no reference point within ``limit_m`` gets index ``-1``
        and an infinite distance.
        """

        queries = _points(queries, "queries")
        if not (np.isfinite(limit_m) and limit_m > 0.0):
            raise ValueError("limit_m must be finite and positive")

        indices = np.full(len(queries), -1, dtype=np.int64)
        distances = np.full(len(queries), np.inf)
        pending = np.arange(len(queries), dtype=np.int64)
        level = self._base
        while len(pending) > 0:
            # At the last level the cell is at least the limit, so
            # everything within the limit is answered exactly there.
            trusted = min(level.cell, float(limit_m))
            found_index, found_squared = level.search(queries[pending])
            answered = found_squared <= trusted * trusted
            resolved = pending[answered]
            indices[resolved] = found_index[answered]
            distances[resolved] = np.sqrt(found_squared[answered])
            if level.cell >= limit_m:
                break
            pending = pending[~answered]
            if len(pending) > 0:
                level = _Level(self._reference, level.cell * 2.0)
        return indices, distances


def nearest_points_brute_force(
    reference: np.ndarray,
    queries: np.ndarray,
    *,
    limit_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """The same answer by comparing every query with every point.

    Quadratic, and only for checking ``NearestPointIndex`` on small inputs.
    """

    reference = _points(reference, "reference")
    queries = _points(queries, "queries")
    indices = np.full(len(queries), -1, dtype=np.int64)
    distances = np.full(len(queries), np.inf)
    for row, query in enumerate(queries):
        squared = _squared_distance(reference, query[None, :])
        best = int(np.argmin(squared))
        if squared[best] <= limit_m * limit_m:
            indices[row] = best
            distances[row] = np.sqrt(squared[best])
    return indices, distances


def _points(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (count, 3)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _squared_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    # Written out so both search paths add the terms in the same order and
    # agree to the last bit.
    delta_x = left[:, 0] - right[:, 0]
    delta_y = left[:, 1] - right[:, 1]
    delta_z = left[:, 2] - right[:, 2]
    return delta_x * delta_x + delta_y * delta_y + delta_z * delta_z


class _Level:
    """The reference sorted by cell key at one cell size."""

    def __init__(self, reference: np.ndarray, cell: float) -> None:
        self.cell = cell
        # Decided in floating point, before anything is cast: a cell far
        # smaller than the extent overflows the integer cell index.
        low = np.floor(reference.min(axis=0) / cell)
        high = np.floor(reference.max(axis=0) / cell)
        span = high - low + 1.0
        if (
            max(float(np.abs(low).max()), float(np.abs(high).max()))
            >= 2.0**52
            or float(span[0]) * float(span[1]) * float(span[2]) >= 2.0**62
        ):
            raise ValueError(
                "reference extent is too large for this cell size"
            )
        self._origin = low.astype(np.int64)
        self._shape = span.astype(np.int64)
        # Keys increase fastest along x. Built an axis at a time so that
        # the transient arrays stay one column wide.
        keys = self._axis_cells(reference, 2)
        keys *= self._shape[1]
        keys += self._axis_cells(reference, 1)
        keys *= self._shape[0]
        keys += self._axis_cells(reference, 0)
        self._order = np.argsort(keys, kind="stable")
        self._keys = keys[self._order]
        self._points = reference[self._order]

    def _axis_cells(self, points: np.ndarray, axis: int) -> np.ndarray:
        return (
            np.floor(points[:, axis] / self.cell).astype(np.int64)
            - self._origin[axis]
        )

    def search(self, queries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Best candidate among the 27 cells around each query.

        Returns the reference index and squared distance; a query with no
        candidate gets index ``-1`` and an infinite squared distance.
        """

        best_index = np.full(len(queries), _NO_INDEX, dtype=np.int64)
        best_squared = np.full(len(queries), np.inf)
        for first in range(0, len(queries), _MAX_QUERIES):
            last = min(first + _MAX_QUERIES, len(queries))
            starts, counts = self._candidate_ranges(queries[first:last])
            per_query = counts.sum(axis=1)
            cumulative = np.cumsum(per_query)
            begin = 0
            while begin < len(per_query):
                consumed = cumulative[begin - 1] if begin > 0 else 0
                end = int(
                    np.searchsorted(
                        cumulative,
                        consumed + _MAX_CANDIDATES,
                        side="right",
                    )
                )
                end = max(end, begin + 1)
                self._reduce(
                    queries[first + begin:first + end],
                    starts[begin:end],
                    counts[begin:end],
                    best_index[first + begin:first + end],
                    best_squared[first + begin:first + end],
                )
                begin = end
        best_index[best_index == _NO_INDEX] = -1
        return best_index, best_squared

    def _candidate_ranges(
        self,
        queries: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Nine runs of the sorted reference per query, one per (y, z).

        The three x-neighbours of a cell are one contiguous run.
        """

        shape = self._shape
        # Clipped to just outside the grid before the cast, so a query
        # arbitrarily far away is merely unreachable, not an overflow.
        cells = np.clip(
            np.floor(queries / self.cell) - self._origin,
            -2.0,
            (shape + 1).astype(np.float64),
        ).astype(np.int64)
        low_x = np.clip(cells[:, 0] - 1, 0, shape[0] - 1)
        high_x = np.clip(cells[:, 0] + 1, 0, shape[0] - 1)
        x_reachable = (cells[:, 0] + 1 >= 0) & (cells[:, 0] - 1 < shape[0])

        starts = np.zeros((len(queries), 9), dtype=np.int64)
        counts = np.zeros((len(queries), 9), dtype=np.int64)
        column = 0
        for offset_z in (-1, 0, 1):
            for offset_y in (-1, 0, 1):
                y = cells[:, 1] + offset_y
                z = cells[:, 2] + offset_z
                reachable = (
                    x_reachable
                    & (y >= 0)
                    & (y < shape[1])
                    & (z >= 0)
                    & (z < shape[2])
                )
                row = np.where(reachable, z, 0) * shape[1] + np.where(
                    reachable, y, 0
                )
                low = np.searchsorted(
                    self._keys, row * shape[0] + low_x, side="left"
                )
                high = np.searchsorted(
                    self._keys, row * shape[0] + high_x, side="right"
                )
                starts[:, column] = low
                counts[:, column] = np.where(reachable, high - low, 0)
                column += 1
        return starts, counts

    def _reduce(
        self,
        queries: np.ndarray,
        starts: np.ndarray,
        counts: np.ndarray,
        best_index: np.ndarray,
        best_squared: np.ndarray,
    ) -> None:
        flat_counts = counts.ravel()
        total = int(flat_counts.sum())
        if total == 0:
            return
        per_query = counts.sum(axis=1)
        run_first = np.cumsum(flat_counts) - flat_counts
        within_run = np.arange(total, dtype=np.int64) - np.repeat(
            run_first, flat_counts
        )
        candidate = np.repeat(starts.ravel(), flat_counts) + within_run
        owner = np.repeat(np.arange(len(queries), dtype=np.int64), per_query)
        squared = _squared_distance(self._points[candidate], queries[owner])

        occupied = np.flatnonzero(per_query > 0)
        segment_first = (np.cumsum(per_query) - per_query)[occupied]
        minimum = np.minimum.reduceat(squared, segment_first)
        is_minimum = squared == np.repeat(minimum, per_query[occupied])
        lowest_index = np.minimum.reduceat(
            np.where(is_minimum, self._order[candidate], _NO_INDEX),
            segment_first,
        )
        best_squared[occupied] = minimum
        best_index[occupied] = lowest_index
