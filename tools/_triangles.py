"""Exact distance from points to a triangle mesh, within a stated reach.

Accuracy asks how far each vertex of a reconstruction is from the true
surface, and a vertex is a point, so a nearest-point search answers it.
Completeness asks the opposite question, how far each point of the true
surface is from the reconstruction, and the reconstruction is not points.
It is triangles. The distance to the nearest vertex overstates the distance
to the mesh by up to the size of a triangle, which at a 10 mm voxel is the
size of the threshold being tested.

So this measures to the triangles themselves. Each triangle is filed in
every cell of a uniform grid that its bounding box touches. For a query,
the 27 cells around it are searched, and with cells no smaller than the
reach that is exact: a triangle within reach of the query has a point
within reach of it, that point lies in one of those 27 cells, and the
triangle's bounding box touches the cell its own point is in.

Beyond the reach a query has no distance, rather than an approximate one.
"""

from __future__ import annotations

import numpy as np

# Point-triangle pairs evaluated at once.
_MAX_CANDIDATES = 1_000_000
_MAX_QUERIES = 262_144
# A mesh whose triangles are filed more times than this in total has
# triangles far larger than the reach; refuse rather than fill memory.
_MAX_FILINGS = 400_000_000


def _segment_distance_squared(
    point: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> np.ndarray:
    along = end - start
    offset = point - start
    length_squared = np.einsum("ij,ij->i", along, along)
    # A segment of zero length is its start point.
    fraction = np.clip(
        np.einsum("ij,ij->i", offset, along)
        / np.where(length_squared > 0.0, length_squared, 1.0),
        0.0,
        1.0,
    )
    gap = offset - fraction[:, None] * along
    return np.einsum("ij,ij->i", gap, gap)


def point_triangle_distance_squared(
    point: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
) -> np.ndarray:
    """Squared distance from each point to its own triangle ``a b c``.

    The nearest point of a triangle is either the foot of the perpendicular
    onto its plane, when that foot is inside, or a point on one of its three
    edges. The smaller of those is taken, so a degenerate triangle, which
    has no inside, is measured as the segments it collapses to.
    """

    best = np.minimum(
        np.minimum(
            _segment_distance_squared(point, a, b),
            _segment_distance_squared(point, b, c),
        ),
        _segment_distance_squared(point, c, a),
    )
    normal = np.cross(b - a, c - a)
    normal_squared = np.einsum("ij,ij->i", normal, normal)
    inside = (
        (normal_squared > 0.0)
        & (np.einsum("ij,ij->i", np.cross(b - a, point - a), normal) >= 0.0)
        & (np.einsum("ij,ij->i", np.cross(c - b, point - b), normal) >= 0.0)
        & (np.einsum("ij,ij->i", np.cross(a - c, point - c), normal) >= 0.0)
    )
    height = np.einsum("ij,ij->i", point - a, normal)
    plane = height * height / np.where(normal_squared > 0.0, normal_squared, 1.0)
    return np.where(inside, np.minimum(best, plane), best)


class TriangleIndex:
    """A triangle mesh, filed for exact distance queries within a reach."""

    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        *,
        reach_m: float,
    ) -> None:
        vertices = np.asarray(vertices, dtype=np.float64)
        faces = np.asarray(faces)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("vertices must have shape (count, 3)")
        if not np.all(np.isfinite(vertices)):
            raise ValueError("vertices must be finite")
        if (
            faces.ndim != 2
            or faces.shape[1] != 3
            or faces.dtype.kind not in "iu"
            or len(faces) == 0
        ):
            raise ValueError("faces must be a non-empty (count, 3) integer array")
        if faces.min() < 0 or faces.max() >= len(vertices):
            raise ValueError("faces refer to vertices that do not exist")
        if not (np.isfinite(reach_m) and reach_m > 0.0):
            raise ValueError("reach_m must be finite and positive")

        self._reach = float(reach_m)
        self._a = vertices[faces[:, 0]]
        self._b = vertices[faces[:, 1]]
        self._c = vertices[faces[:, 2]]
        corners = np.stack([self._a, self._b, self._c])
        low = np.floor(corners.min(axis=0) / self._reach)
        high = np.floor(corners.max(axis=0) / self._reach)
        origin = low.min(axis=0)
        shape = high.max(axis=0) - origin + 1.0
        if (
            max(float(np.abs(low).max()), float(np.abs(high).max())) >= 2.0**52
            or float(shape[0]) * float(shape[1]) * float(shape[2]) >= 2.0**62
        ):
            raise ValueError("mesh extent is too large for this reach")
        self._origin = origin.astype(np.int64)
        self._shape = shape.astype(np.int64)
        first = low.astype(np.int64) - self._origin
        last = high.astype(np.int64) - self._origin
        spans = last - first + 1
        if int(np.prod(spans, axis=1, dtype=np.float64).sum()) > _MAX_FILINGS:
            raise ValueError(
                "triangles are too large for this reach to index them"
            )

        keys = []
        owners = []
        triangle = np.arange(len(faces), dtype=np.int64)
        for step_z in range(int(spans[:, 2].max())):
            in_z = spans[:, 2] > step_z
            for step_y in range(int(spans[in_z, 1].max())):
                in_y = in_z & (spans[:, 1] > step_y)
                for step_x in range(int(spans[in_y, 0].max())):
                    chosen = in_y & (spans[:, 0] > step_x)
                    keys.append(
                        (
                            (first[chosen, 2] + step_z) * self._shape[1]
                            + first[chosen, 1]
                            + step_y
                        )
                        * self._shape[0]
                        + first[chosen, 0]
                        + step_x
                    )
                    owners.append(triangle[chosen])
        keys = np.concatenate(keys)
        owners = np.concatenate(owners)
        order = np.argsort(keys, kind="stable")
        self._keys = keys[order]
        self._owners = owners[order]

    @property
    def reach_m(self) -> float:
        return self._reach

    def distance(self, queries: np.ndarray) -> np.ndarray:
        """Distance from each query to the mesh; infinite beyond the reach."""

        queries = np.asarray(queries, dtype=np.float64)
        if queries.ndim != 2 or queries.shape[1] != 3:
            raise ValueError("queries must have shape (count, 3)")
        if not np.all(np.isfinite(queries)):
            raise ValueError("queries must be finite")
        squared = np.full(len(queries), np.inf)
        for first in range(0, len(queries), _MAX_QUERIES):
            last = min(first + _MAX_QUERIES, len(queries))
            starts, counts = self._candidate_runs(queries[first:last])
            per_query = counts.sum(axis=1)
            cumulative = np.cumsum(per_query)
            begin = 0
            while begin < len(per_query):
                consumed = cumulative[begin - 1] if begin > 0 else 0
                end = int(
                    np.searchsorted(
                        cumulative, consumed + _MAX_CANDIDATES, side="right"
                    )
                )
                end = max(end, begin + 1)
                self._reduce(
                    queries[first + begin:first + end],
                    starts[begin:end],
                    counts[begin:end],
                    squared[first + begin:first + end],
                )
                begin = end
        limit = self._reach * self._reach
        return np.where(squared <= limit, np.sqrt(squared), np.inf)

    def _candidate_runs(
        self,
        queries: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Nine runs of the filed triangles per query, one per (y, z).

        Keys increase fastest along x, so the three x-neighbours of a cell
        are one contiguous run.
        """

        shape = self._shape
        cells = np.clip(
            np.floor(queries / self._reach) - self._origin,
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
        best: np.ndarray,
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
        filed = np.repeat(starts.ravel(), flat_counts) + within_run
        owner = np.repeat(np.arange(len(queries), dtype=np.int64), per_query)
        triangle = self._owners[filed]
        squared = point_triangle_distance_squared(
            queries[owner],
            self._a[triangle],
            self._b[triangle],
            self._c[triangle],
        )
        occupied = np.flatnonzero(per_query > 0)
        segment_first = (np.cumsum(per_query) - per_query)[occupied]
        best[occupied] = np.minimum.reduceat(squared, segment_first)


def mesh_distance_brute_force(
    vertices: np.ndarray,
    faces: np.ndarray,
    queries: np.ndarray,
    *,
    reach_m: float,
) -> np.ndarray:
    """The same answer by measuring every query against every triangle.

    Quadratic, and only for checking ``TriangleIndex`` on small inputs.
    """

    vertices = np.asarray(vertices, dtype=np.float64)
    queries = np.asarray(queries, dtype=np.float64)
    a = vertices[faces[:, 0]]
    b = vertices[faces[:, 1]]
    c = vertices[faces[:, 2]]
    distances = np.full(len(queries), np.inf)
    for row, query in enumerate(queries):
        squared = float(
            point_triangle_distance_squared(
                np.broadcast_to(query, a.shape), a, b, c
            ).min()
        )
        if squared <= reach_m * reach_m:
            distances[row] = np.sqrt(squared)
    return distances
