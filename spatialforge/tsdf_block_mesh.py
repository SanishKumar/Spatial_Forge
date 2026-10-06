"""Triangle meshing directly from a persisted sparse block volume.

The reference mesher in :mod:`mesh` walks a dense grid one cell at a time in
Python and refuses any surface with a non-manifold vertex. Both are right for
a reference and wrong for real scans: a sparse volume at a useful voxel size
has no dense grid to walk, and real data always produces a few such vertices.

This mesher reads a ``.sftvol`` and triangulates every fully observed cell
with the same six-tetrahedron Freudenthal split, the same strict sign rule
and the same vertex rule as the reference, vectorised over all cells at once.
For the same volume it produces the same vertices, in the same order, and the
same triangles -- which is how it is tested.

It then does the one thing the reference declines to. Where the observed
region is ragged, the cells around a grid edge can be present on two
opposite sides and missing between them, so two separate fans of triangles
meet at a single vertex. Such a pinch is not an error in the data. It is
resolved the standard way, by giving each fan its own copy of the vertex,
after which the surface is a manifold with boundary.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .errors import MeshExtractionError, TsdfError
from .mesh import _CORNER_OFFSETS, _TETRAHEDRA
from .tsdf_block_plan import TSDF_BLOCK_RESOLUTION
from .tsdf_block_volume import TsdfBlockVolume, load_tsdf_block_volume

MAX_TSDF_BLOCK_MESH_TRIANGLES = 16_000_000

_OFFSETS = np.array(_CORNER_OFFSETS, dtype=np.int64)
_SIDE = TSDF_BLOCK_RESOLUTION
_MESH_RULE = "freudenthal_six_tetra_strict_signs"
_WINDING = "toward_positive_tsdf_free_space"


def _build_case_tables() -> tuple[
    tuple[tuple[int, ...], ...],
    dict[tuple[int, int], tuple[tuple[tuple[int, int], ...], tuple]],
]:
    """Enumerate, per tetrahedron and sign pattern, the reference's output.

    For each pattern the table holds the crossing edges in the order the
    reference creates their vertices, and the triangles over them with the
    winding the reference would choose. That winding does not depend on
    where along its edge each vertex falls: for crossings strictly inside
    their edges the sign of the orientation test is a fixed property of the
    pattern, so it is evaluated once here, at the midpoints.
    """

    sorted_tetrahedra = tuple(
        tuple(sorted(tetrahedron)) for tetrahedron in _TETRAHEDRA
    )
    cases: dict[
        tuple[int, int],
        tuple[tuple[tuple[int, int], ...], tuple],
    ] = {}
    for tetra_index, corners in enumerate(sorted_tetrahedra):
        for pattern in range(1, 15):
            negative = [
                corner
                for position, corner in enumerate(corners)
                if pattern >> position & 1
            ]
            positive = [
                corner
                for position, corner in enumerate(corners)
                if not pattern >> position & 1
            ]
            if len(negative) == 1:
                calls = tuple((negative[0], outside) for outside in positive)
                triangles = [(0, 1, 2)]
            elif len(positive) == 1:
                calls = tuple((inside, positive[0]) for inside in negative)
                triangles = [(0, 1, 2)]
            else:
                calls = (
                    (negative[0], positive[0]),
                    (negative[0], positive[1]),
                    (negative[1], positive[0]),
                    (negative[1], positive[1]),
                )
                triangles = [(0, 1, 3), (0, 3, 2)]

            midpoints = [
                (_OFFSETS[inside] + _OFFSETS[outside]) / 2.0
                for inside, outside in calls
            ]
            toward_positive = (
                _OFFSETS[positive].mean(axis=0)
                - _OFFSETS[negative].mean(axis=0)
            )
            wound = []
            for first, second, third in triangles:
                normal = np.cross(
                    midpoints[second] - midpoints[first],
                    midpoints[third] - midpoints[first],
                )
                alignment = float(normal @ toward_positive)
                if alignment == 0.0:
                    raise AssertionError("degenerate marching case")
                wound.append(
                    (first, third, second)
                    if alignment < 0.0
                    else (first, second, third)
                )
            for inside, outside in calls:
                low, high = min(inside, outside), max(inside, outside)
                if low & high != low:
                    raise AssertionError("edge is not monotone")
            cases[(tetra_index, pattern)] = (calls, tuple(wound))
    return sorted_tetrahedra, cases


_SORTED_TETRAHEDRA, _CASES = _build_case_tables()


@dataclass(frozen=True, slots=True, eq=False)
class TsdfBlockMesh:
    """An indexed triangle mesh and the account of how it was built."""

    vertices: np.ndarray
    faces: np.ndarray
    considered_cells: int
    skipped_unknown_cells: int
    skipped_exact_zero_cells: int
    eligible_cells: int
    active_cells: int
    boundary_edges: int
    pinch_vertices_split: int
    vertices_added_by_splitting: int
    components: int
    components_removed: int
    triangles_removed: int

    @property
    def vertex_count(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def triangle_count(self) -> int:
        return int(self.faces.shape[0])


@dataclass(frozen=True, slots=True)
class TsdfBlockMeshReport:
    session_id: str
    input: Path
    output: Path
    minimum_weight: int
    minimum_component_triangles: int
    considered_cells: int
    skipped_unknown_cells: int
    skipped_exact_zero_cells: int
    eligible_cells: int
    active_cells: int
    vertices_written: int
    triangles_written: int
    boundary_edges: int
    pinch_vertices_split: int
    vertices_added_by_splitting: int
    components: int
    components_removed: int
    triangles_removed: int
    source_volume_digest_sha256: str
    output_digest_sha256: str
    output_bytes: int


def extract_tsdf_block_mesh(
    input: str | Path,
    output: str | Path,
    *,
    minimum_weight: int = 1,
    minimum_component_triangles: int = 1,
) -> TsdfBlockMeshReport:
    """Mesh a ``.sftvol`` and write a binary little-endian PLY."""

    output_path = Path(output).resolve()
    if output_path.suffix.lower() != ".ply":
        raise MeshExtractionError("output filename must end in .ply")
    if output_path.exists():
        raise MeshExtractionError(f"output already exists: {output_path}")
    try:
        volume = load_tsdf_block_volume(input)
    except TsdfError as error:
        raise MeshExtractionError(str(error)) from error

    mesh = build_tsdf_block_mesh(
        volume,
        minimum_weight=minimum_weight,
        minimum_component_triangles=minimum_component_triangles,
    )
    encoded = _encode_ply(
        volume,
        mesh,
        minimum_weight,
        minimum_component_triangles,
    )
    digest = hashlib.sha256(encoded).hexdigest()
    _write_without_overwrite(output_path, encoded)
    return TsdfBlockMeshReport(
        session_id=volume.session_id,
        input=volume.path,
        output=output_path,
        minimum_weight=minimum_weight,
        minimum_component_triangles=minimum_component_triangles,
        considered_cells=mesh.considered_cells,
        skipped_unknown_cells=mesh.skipped_unknown_cells,
        skipped_exact_zero_cells=mesh.skipped_exact_zero_cells,
        eligible_cells=mesh.eligible_cells,
        active_cells=mesh.active_cells,
        vertices_written=mesh.vertex_count,
        triangles_written=mesh.triangle_count,
        boundary_edges=mesh.boundary_edges,
        pinch_vertices_split=mesh.pinch_vertices_split,
        vertices_added_by_splitting=mesh.vertices_added_by_splitting,
        components=mesh.components,
        components_removed=mesh.components_removed,
        triangles_removed=mesh.triangles_removed,
        source_volume_digest_sha256=volume.artifact_digest_sha256,
        output_digest_sha256=digest,
        output_bytes=len(encoded),
    )


def build_tsdf_block_mesh(
    volume: TsdfBlockVolume,
    *,
    minimum_weight: int = 1,
    minimum_component_triangles: int = 1,
) -> TsdfBlockMesh:
    """Triangulate a volume, split pinch vertices, and filter fragments."""

    if not isinstance(volume, TsdfBlockVolume):
        raise MeshExtractionError(
            "block meshing requires a loaded TsdfBlockVolume"
        )
    for value, label in (
        (minimum_weight, "minimum_weight"),
        (minimum_component_triangles, "minimum_component_triangles"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise MeshExtractionError(
                f"{label}: expected a positive integer"
            )

    vertices, faces, counts = triangulate_tsdf_block_volume(
        volume,
        minimum_weight=minimum_weight,
    )
    boundary_edges, pinch_corners = _edge_topology(faces, len(vertices))
    vertices, faces, pinched, added = _split_pinch_vertices(
        vertices,
        faces,
        pinch_corners,
    )
    vertices, faces, components, removed, dropped = _filter_components(
        vertices,
        faces,
        minimum_component_triangles,
    )
    if dropped:
        boundary_edges, pinch_corners = _edge_topology(faces, len(vertices))
    _require_manifold_with_boundary(faces, len(vertices))
    vertices.setflags(write=False)
    faces.setflags(write=False)
    return TsdfBlockMesh(
        vertices=vertices,
        faces=faces,
        considered_cells=counts[0],
        skipped_unknown_cells=counts[1],
        skipped_exact_zero_cells=counts[2],
        eligible_cells=counts[3],
        active_cells=counts[4],
        boundary_edges=boundary_edges,
        pinch_vertices_split=pinched,
        vertices_added_by_splitting=added,
        components=components,
        components_removed=removed,
        triangles_removed=dropped,
    )


def triangulate_tsdf_block_volume(
    volume: TsdfBlockVolume,
    *,
    minimum_weight: int = 1,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int, int]]:
    """Return the reference triangulation of every fully observed cell.

    The vertices and faces are exactly what the reference mesher builds for
    the same values, before its topology verdict: same vertex order, same
    face order, same winding.
    """

    values = volume.normalized_tsdf()
    if minimum_weight > 1:
        values[volume.weights < minimum_weight] = np.nan
    blocks = np.asarray(volume.block_indices, dtype=np.int64)

    patch = _neighbour_patches(values, volume.block_indices)
    known = np.ones(values.shape, dtype=bool)
    nonzero = np.ones(values.shape, dtype=bool)
    any_negative = np.zeros(values.shape, dtype=bool)
    any_positive = np.zeros(values.shape, dtype=bool)
    for offset_x, offset_y, offset_z in _CORNER_OFFSETS:
        corner = patch[
            :,
            offset_z:offset_z + _SIDE,
            offset_y:offset_y + _SIDE,
            offset_x:offset_x + _SIDE,
        ]
        known &= ~np.isnan(corner)
        nonzero &= corner != 0.0
        any_negative |= corner < 0.0
        any_positive |= corner > 0.0

    considered = int(values.size)
    unknown_cells = considered - int(np.count_nonzero(known))
    eligible_mask = known & nonzero
    eligible = int(np.count_nonzero(eligible_mask))
    exact_zero = int(np.count_nonzero(known)) - eligible
    active_mask = eligible_mask & any_negative & any_positive
    row, local_z, local_y, local_x = np.nonzero(active_mask)
    active = int(row.size)
    if active == 0:
        raise MeshExtractionError(
            "TSDF block volume produced no triangles "
            f"(cells={considered}, unknown={unknown_cells}, "
            f"exact_zero={exact_zero}, eligible={eligible})"
        )

    # Linearise global voxel indices over the block bounding box. This is
    # the dense flat index the reference uses, so "lower voxel of an edge"
    # and cell traversal order mean the same thing here as there.
    minimum = blocks.min(axis=0) * _SIDE
    extent = [
        int(value)
        for value in (blocks.max(axis=0) - blocks.min(axis=0) + 1) * _SIDE
    ]
    if extent[0] * extent[1] * extent[2] >= 2**59:
        raise MeshExtractionError(
            "TSDF block volume spans too large an index range to mesh"
        )
    anchor = np.stack(
        [
            blocks[row, 0] * _SIDE + local_x,
            blocks[row, 1] * _SIDE + local_y,
            blocks[row, 2] * _SIDE + local_z,
        ],
        axis=1,
    )
    relative = anchor - minimum
    flat = (
        relative[:, 2] * extent[1] + relative[:, 1]
    ) * extent[0] + relative[:, 0]
    order = np.argsort(flat, kind="stable")
    row, local_z, local_y, local_x = (
        row[order],
        local_z[order],
        local_y[order],
        local_x[order],
    )
    anchor = anchor[order]
    flat = flat[order]
    corner_values = np.stack(
        [
            patch[
                row,
                local_z + offset_z,
                local_y + offset_y,
                local_x + offset_x,
            ]
            for offset_x, offset_y, offset_z in _CORNER_OFFSETS
        ],
        axis=1,
    )
    del patch, known, nonzero, any_negative, any_positive, eligible_mask
    flat_offset = (
        _OFFSETS[:, 2] * extent[1] + _OFFSETS[:, 1]
    ) * extent[0] + _OFFSETS[:, 0]

    call_sort = []
    call_key = []
    call_cell = []
    call_inside = []
    call_outside = []
    face_sort = []
    face_calls = []
    call_total = 0
    for tetra_index, corners in enumerate(_SORTED_TETRAHEDRA):
        negative = corner_values[:, corners] < 0.0
        pattern = (
            negative[:, 0]
            + 2 * negative[:, 1]
            + 4 * negative[:, 2]
            + 8 * negative[:, 3]
        )
        for case in range(1, 15):
            cells = np.flatnonzero(pattern == case)
            if cells.size == 0:
                continue
            calls, triangles = _CASES[(tetra_index, case)]
            starts = []
            for position, (inside, outside) in enumerate(calls):
                low, high = min(inside, outside), max(inside, outside)
                starts.append(call_total)
                call_total += int(cells.size)
                call_sort.append(cells * 32 + tetra_index * 4 + position)
                call_key.append(
                    (flat[cells] + flat_offset[low]) * 8 + (high - low)
                )
                call_cell.append(cells)
                call_inside.append(
                    np.full(cells.size, inside, dtype=np.int8)
                )
                call_outside.append(
                    np.full(cells.size, outside, dtype=np.int8)
                )
            step = np.arange(cells.size, dtype=np.int64)
            for position, triangle in enumerate(triangles):
                face_sort.append(cells * 32 + tetra_index * 4 + position)
                face_calls.append(
                    np.stack(
                        [starts[call] + step for call in triangle],
                        axis=1,
                    )
                )

    call_sort_all = np.concatenate(call_sort)
    call_key_all = np.concatenate(call_key)
    call_cell_all = np.concatenate(call_cell)
    call_inside_all = np.concatenate(call_inside)
    call_outside_all = np.concatenate(call_outside)
    del call_sort, call_key, call_cell, call_inside, call_outside

    # Vertex identifiers follow first creation in the reference's traversal:
    # cells in dense order, tetrahedra in order, crossings in call order.
    traversal = np.argsort(call_sort_all, kind="stable")
    unique_keys, first_seen, inverse = np.unique(
        call_key_all[traversal],
        return_index=True,
        return_inverse=True,
    )
    creation = np.argsort(first_seen, kind="stable")
    identifier_of_key = np.empty(len(unique_keys), dtype=np.int64)
    identifier_of_key[creation] = np.arange(len(unique_keys), dtype=np.int64)
    call_vertex = np.empty(len(traversal), dtype=np.int64)
    call_vertex[traversal] = identifier_of_key[inverse]
    defining_call = traversal[first_seen[creation]]

    cell = call_cell_all[defining_call]
    inside = call_inside_all[defining_call].astype(np.int64)
    outside = call_outside_all[defining_call].astype(np.int64)
    inside_value = corner_values[cell, inside]
    outside_value = corner_values[cell, outside]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        alpha = inside_value / (inside_value - outside_value)
        inside_centre = (
            (anchor[cell] + _OFFSETS[inside]).astype(np.float64) + 0.5
        ) * volume.voxel_size_m
        outside_centre = (
            (anchor[cell] + _OFFSETS[outside]).astype(np.float64) + 0.5
        ) * volume.voxel_size_m
        vertices = inside_centre + alpha[:, None] * (
            outside_centre - inside_centre
        )
    if not bool(np.all(np.isfinite(vertices))):
        raise MeshExtractionError(
            "triangle-mesh interpolation produced a non-finite coordinate"
        )

    face_order = np.argsort(np.concatenate(face_sort), kind="stable")
    faces = call_vertex[np.concatenate(face_calls, axis=0)[face_order]]
    if len(faces) > MAX_TSDF_BLOCK_MESH_TRIANGLES:
        raise MeshExtractionError(
            "triangle limit exceeded; block meshing supports at most "
            f"{MAX_TSDF_BLOCK_MESH_TRIANGLES} triangles"
        )
    _require_sound_triangles(vertices, faces)
    return (
        vertices,
        faces,
        (considered, unknown_cells, exact_zero, eligible, active),
    )


def _neighbour_patches(
    values: np.ndarray,
    block_indices: tuple[tuple[int, int, int], ...],
) -> np.ndarray:
    """Extend each block by one voxel into its +x, +y and +z neighbours.

    A cell anchored on a block's upper faces needs corners that belong to up
    to seven neighbouring blocks. Blocks that are not planned contribute
    unknown voxels, exactly as unobserved voxels do.
    """

    block_count = len(block_indices)
    row_of = {index: row for row, index in enumerate(block_indices)}
    padded = np.concatenate(
        [values, np.full((1, _SIDE, _SIDE, _SIDE), np.nan)],
        axis=0,
    )

    def rows(offset_x: int, offset_y: int, offset_z: int) -> np.ndarray:
        return np.fromiter(
            (
                row_of.get(
                    (x + offset_x, y + offset_y, z + offset_z),
                    block_count,
                )
                for x, y, z in block_indices
            ),
            dtype=np.int64,
            count=block_count,
        )

    patch = np.full(
        (block_count, _SIDE + 1, _SIDE + 1, _SIDE + 1),
        np.nan,
    )
    patch[:, :_SIDE, :_SIDE, :_SIDE] = values
    patch[:, :_SIDE, :_SIDE, _SIDE] = padded[rows(1, 0, 0), :, :, 0]
    patch[:, :_SIDE, _SIDE, :_SIDE] = padded[rows(0, 1, 0), :, 0, :]
    patch[:, _SIDE, :_SIDE, :_SIDE] = padded[rows(0, 0, 1), 0, :, :]
    patch[:, :_SIDE, _SIDE, _SIDE] = padded[rows(1, 1, 0), :, 0, 0]
    patch[:, _SIDE, :_SIDE, _SIDE] = padded[rows(1, 0, 1), 0, :, 0]
    patch[:, _SIDE, _SIDE, :_SIDE] = padded[rows(0, 1, 1), 0, 0, :]
    patch[:, _SIDE, _SIDE, _SIDE] = padded[rows(1, 1, 1), 0, 0, 0]
    return patch


def _require_sound_triangles(vertices: np.ndarray, faces: np.ndarray) -> None:
    first = vertices[faces[:, 0]]
    normal = np.cross(
        vertices[faces[:, 1]] - first,
        vertices[faces[:, 2]] - first,
    )
    if bool(np.any(np.einsum("ij,ij->i", normal, normal) == 0.0)):
        raise MeshExtractionError(
            "triangle-mesh construction produced a degenerate face"
        )
    ordered = np.sort(faces, axis=1)
    vertex_count = len(vertices)
    packed = (
        ordered[:, 0] * vertex_count + ordered[:, 1]
    ) * vertex_count + ordered[:, 2]
    if vertex_count**3 < 2**62 and len(np.unique(packed)) != len(packed):
        raise MeshExtractionError(
            "mesh construction produced a duplicate triangle"
        )
    if vertex_count**3 >= 2**62 and len(
        np.unique(ordered, axis=0)
    ) != len(ordered):
        raise MeshExtractionError(
            "mesh construction produced a duplicate triangle"
        )


def _half_edges(
    faces: np.ndarray,
    vertex_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sorted undirected edge keys with the corners at each end."""

    start = faces.reshape(-1)
    end = np.roll(faces, -1, axis=1).reshape(-1)
    corner_at_start = np.arange(faces.size, dtype=np.int64)
    corner_at_end = (
        corner_at_start - corner_at_start % 3 + (corner_at_start % 3 + 1) % 3
    )
    forward = start < end
    key = np.where(forward, start, end) * vertex_count + np.where(
        forward,
        end,
        start,
    )
    corner_low = np.where(forward, corner_at_start, corner_at_end)
    corner_high = np.where(forward, corner_at_end, corner_at_start)
    order = np.argsort(key, kind="stable")
    return key[order], corner_low[order], corner_high[order], forward[order]


def _edge_topology(
    faces: np.ndarray,
    vertex_count: int,
) -> tuple[int, tuple[np.ndarray, np.ndarray]]:
    """Count boundary edges and return corner pairs joined across edges.

    Raises if any edge carries more than two triangles or two triangles
    cross an edge in the same direction; neither can come out of a
    consistent tetrahedral split, so either means the mesher is wrong.
    """

    key, corner_low, corner_high, forward = _half_edges(faces, vertex_count)
    boundaries = np.flatnonzero(np.diff(key, prepend=-1) != 0)
    counts = np.diff(np.append(boundaries, len(key)))
    if bool(np.any(counts > 2)):
        raise MeshExtractionError(
            "mesh construction produced "
            f"{int(np.count_nonzero(counts > 2))} non-manifold triangle "
            "edges"
        )
    paired = boundaries[counts == 2]
    if bool(np.any(forward[paired] == forward[paired + 1])):
        raise MeshExtractionError(
            "mesh construction produced inconsistently wound triangles"
        )
    joined = (
        np.concatenate([corner_low[paired], corner_high[paired]]),
        np.concatenate([corner_low[paired + 1], corner_high[paired + 1]]),
    )
    return int(np.count_nonzero(counts == 1)), joined


def _connected_labels(
    count: int,
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray:
    """Label connected components with the smallest member of each.

    Hook each root to the smallest root it is joined to, then compress
    every path, and repeat. Each round removes every root that has a
    smaller neighbour, so the number of rounds grows with the logarithm of
    the component size rather than with its diameter.
    """

    parent = np.arange(count, dtype=np.int64)
    while True:
        root_first = parent[first]
        root_second = parent[second]
        differing = root_first != root_second
        if not bool(np.any(differing)):
            return parent
        low = np.minimum(root_first[differing], root_second[differing])
        high = np.maximum(root_first[differing], root_second[differing])
        np.minimum.at(parent, high, low)
        while True:
            grandparent = parent[parent]
            if np.array_equal(grandparent, parent):
                break
            parent = grandparent


def _split_pinch_vertices(
    vertices: np.ndarray,
    faces: np.ndarray,
    joined: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Give every fan of triangles around a vertex its own vertex.

    Two triangle corners at the same vertex belong to one fan when the
    triangles share an edge through that vertex. A vertex with more than
    one fan is a pinch; its first fan keeps the vertex and each further fan
    receives a copy, in a fixed order, so the result is deterministic.
    """

    corner_count = faces.size
    fan = _connected_labels(corner_count, joined[0], joined[1])
    corner_vertex = faces.reshape(-1)
    pair = corner_vertex * corner_count + fan
    unique_pairs, pair_of_corner = np.unique(pair, return_inverse=True)
    pair_vertex = unique_pairs // corner_count
    first_of_vertex = np.flatnonzero(np.diff(pair_vertex, prepend=-1) != 0)
    fans_per_vertex = np.diff(np.append(first_of_vertex, len(pair_vertex)))
    pinched = int(np.count_nonzero(fans_per_vertex > 1))
    if pinched == 0:
        return vertices, faces, 0, 0

    is_first_fan = np.zeros(len(unique_pairs), dtype=bool)
    is_first_fan[first_of_vertex] = True
    extra = np.flatnonzero(~is_first_fan)
    identifier = pair_vertex.copy()
    identifier[extra] = len(vertices) + np.arange(len(extra), dtype=np.int64)
    split_faces = identifier[pair_of_corner].reshape(faces.shape)
    split_vertices = np.concatenate(
        [vertices, vertices[pair_vertex[extra]]],
        axis=0,
    )
    return split_vertices, split_faces, pinched, int(len(extra))


def _filter_components(
    vertices: np.ndarray,
    faces: np.ndarray,
    minimum_triangles: int,
) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    """Drop connected fragments smaller than a triangle count."""

    label = _connected_labels(
        len(vertices),
        np.concatenate([faces[:, 0], faces[:, 1]]),
        np.concatenate([faces[:, 1], faces[:, 2]]),
    )
    face_label = label[faces[:, 0]]
    sizes = np.bincount(face_label, minlength=len(vertices))
    components = int(np.count_nonzero(sizes))
    if minimum_triangles <= 1:
        return vertices, faces, components, 0, 0

    keep = sizes[face_label] >= minimum_triangles
    removed = int(np.count_nonzero((sizes > 0) & (sizes < minimum_triangles)))
    dropped = int(np.count_nonzero(~keep))
    if dropped == 0:
        return vertices, faces, components, 0, 0
    if not bool(np.any(keep)):
        raise MeshExtractionError(
            "every mesh component is smaller than "
            f"minimum_component_triangles={minimum_triangles}"
        )
    kept_faces = faces[keep]
    used = np.zeros(len(vertices), dtype=bool)
    used[kept_faces.reshape(-1)] = True
    renumber = np.cumsum(used) - 1
    return (
        vertices[used],
        renumber[kept_faces],
        components,
        removed,
        dropped,
    )


def _require_manifold_with_boundary(
    faces: np.ndarray,
    vertex_count: int,
) -> None:
    """Final verdict: every vertex has one fan, open or closed."""

    _, joined = _edge_topology(faces, vertex_count)
    corner_count = faces.size
    fan = _connected_labels(corner_count, joined[0], joined[1])
    pairs = np.unique(faces.reshape(-1) * corner_count + fan)
    if len(pairs) != vertex_count:
        raise MeshExtractionError(
            "mesh construction left "
            f"{len(pairs) - vertex_count} non-manifold triangle vertices"
        )


def _encode_ply(
    volume: TsdfBlockVolume,
    mesh: TsdfBlockMesh,
    minimum_weight: int,
    minimum_component_triangles: int,
) -> bytes:
    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            f"comment spatialforge_session {volume.session_id}",
            "comment spatialforge_replay_sha256 "
            f"{volume.replay_digest_sha256}",
            "comment spatialforge_source_volume_sha256 "
            f"{volume.artifact_digest_sha256}",
            "comment coordinates metres world_x_forward world_y_left "
            "world_z_up",
            f"comment spatialforge_mesh_rule {_MESH_RULE}",
            f"comment spatialforge_mesh_winding {_WINDING}",
            "comment spatialforge_unknown_cells skipped",
            "comment spatialforge_exact_zero_cells skipped",
            "comment spatialforge_pinch_vertices split_per_triangle_fan",
            f"comment spatialforge_minimum_weight {minimum_weight}",
            "comment spatialforge_minimum_component_triangles "
            f"{minimum_component_triangles}",
            f"element vertex {mesh.vertex_count}",
            "property double x",
            "property double y",
            "property double z",
            f"element face {mesh.triangle_count}",
            "property list uchar int vertex_indices",
            "end_header",
            "",
        ]
    ).encode("ascii")
    if mesh.vertex_count > np.iinfo(np.int32).max:
        raise MeshExtractionError("mesh has too many vertices for PLY int")
    face_records = np.empty(
        mesh.triangle_count,
        dtype=[("count", "u1"), ("indices", "<i4", (3,))],
    )
    face_records["count"] = 3
    face_records["indices"] = mesh.faces
    return b"".join(
        (
            header,
            mesh.vertices.astype("<f8").tobytes(order="C"),
            face_records.tobytes(order="C"),
        )
    )


def _write_without_overwrite(output_path: Path, encoded: bytes) -> None:
    temporary_path: Path | None = None
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output_path.stem}-",
            suffix=".ply",
            dir=output_path.parent,
        )
        temporary_path = Path(temporary_name)
        with open(descriptor, "wb", closefd=True) as output_file:
            output_file.write(encoded)
        try:
            os.link(temporary_path, output_path)
        except FileExistsError as error:
            raise MeshExtractionError(
                "output appeared while meshing; refusing to overwrite: "
                f"{output_path}"
            ) from error
        except OSError as error:
            raise MeshExtractionError(
                f"cannot publish mesh output without overwriting: {error}"
            ) from error
    except MeshExtractionError:
        raise
    except OSError as error:
        raise MeshExtractionError(
            f"cannot write mesh output: {error}"
        ) from error
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
