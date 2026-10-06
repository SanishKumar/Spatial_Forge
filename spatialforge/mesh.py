"""Deterministic reference triangle meshing from a validated TSDF volume."""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .errors import MeshExtractionError, SurfaceExtractionError
from .surface import (
    _ReferenceTsdf,
    _TsdfVoxel,
    _load_reference_tsdf,
    _voxel_center,
)

MAX_REFERENCE_TRIANGLES = 1_000_000

# Corners use x as the low bit, then y, then z.
_CORNER_OFFSETS = (
    (0, 0, 0),
    (1, 0, 0),
    (0, 1, 0),
    (1, 1, 0),
    (0, 0, 1),
    (1, 0, 1),
    (0, 1, 1),
    (1, 1, 1),
)

# Translation-invariant Freudenthal split around the 000 -> 111 diagonal.
_TETRAHEDRA = (
    (0, 1, 3, 7),
    (0, 3, 2, 7),
    (0, 2, 6, 7),
    (0, 6, 4, 7),
    (0, 4, 5, 7),
    (0, 5, 1, 7),
)

_VertexKey = tuple[int, int]
_Point = tuple[float, float, float]
_Face = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class TriangleMeshReport:
    session_id: str
    input: Path
    output: Path
    total_voxels: int
    observed_voxels: int
    total_cells: int
    skipped_unknown_cells: int
    skipped_exact_zero_cells: int
    eligible_cells: int
    active_cells: int
    vertices_written: int
    triangles_written: int
    boundary_edges: int
    source_tsdf_digest_sha256: str
    output_digest_sha256: str


@dataclass(frozen=True, slots=True)
class _Mesh:
    total_cells: int
    skipped_unknown_cells: int
    skipped_exact_zero_cells: int
    eligible_cells: int
    active_cells: int
    vertices: tuple[_Point, ...]
    index_vertices: tuple[_Point, ...]
    faces: tuple[_Face, ...]
    boundary_edges: int


def extract_triangle_mesh(
    input: str | Path,
    output: str | Path,
) -> TriangleMeshReport:
    """Extract a deterministic triangle mesh using six tetrahedra per cell."""

    try:
        volume = _load_reference_tsdf(input)
    except SurfaceExtractionError as error:
        raise MeshExtractionError(str(error)) from error

    output_path = Path(output).resolve()
    if output_path.suffix.lower() != ".ply":
        raise MeshExtractionError("output filename must end in .ply")
    if output_path.exists():
        raise MeshExtractionError(f"output already exists: {output_path}")

    mesh = _build_mesh(volume)
    output_digest = _write_mesh_ply(volume, mesh, output_path)
    return TriangleMeshReport(
        session_id=volume.session_id,
        input=volume.path,
        output=output_path,
        total_voxels=volume.total_voxels,
        observed_voxels=len(volume.voxels),
        total_cells=mesh.total_cells,
        skipped_unknown_cells=mesh.skipped_unknown_cells,
        skipped_exact_zero_cells=mesh.skipped_exact_zero_cells,
        eligible_cells=mesh.eligible_cells,
        active_cells=mesh.active_cells,
        vertices_written=len(mesh.vertices),
        triangles_written=len(mesh.faces),
        boundary_edges=mesh.boundary_edges,
        source_tsdf_digest_sha256=volume.digest_sha256,
        output_digest_sha256=output_digest,
    )


def _build_mesh(volume: _ReferenceTsdf) -> _Mesh:
    mesh, non_manifold_edges, non_manifold_vertices = _build_unchecked_mesh(
        volume
    )
    if non_manifold_edges:
        raise MeshExtractionError(
            "mesh construction produced "
            f"{non_manifold_edges} non-manifold triangle edges"
        )
    if non_manifold_vertices:
        raise MeshExtractionError(
            "mesh construction produced "
            f"{non_manifold_vertices} non-manifold triangle vertices"
        )
    return mesh


def _build_unchecked_mesh(volume: _ReferenceTsdf) -> tuple[_Mesh, int, int]:
    """Triangulate every eligible cell, leaving the topology verdict out.

    Returned separately so the triangulation itself can be compared against
    another mesher on volumes whose ragged observed region produces pinch
    vertices, which ``_build_mesh`` refuses.
    """

    nx, ny, nz = volume.dimensions
    if nx < 2 or ny < 2 or nz < 2:
        raise MeshExtractionError(
            "TSDF volume dimensions must each be at least 2 to form mesh cells"
        )

    total_cells = (nx - 1) * (ny - 1) * (nz - 1)
    voxel_by_index = {voxel.index: voxel for voxel in volume.voxels}
    vertex_ids: dict[_VertexKey, int] = {}
    vertices: list[_Point] = []
    index_vertices: list[_Point] = []
    faces: list[_Face] = []
    face_keys: set[tuple[int, int, int]] = set()
    skipped_unknown_cells = 0
    skipped_exact_zero_cells = 0
    eligible_cells = 0
    active_cells = 0

    for z in range(nz - 1):
        for y in range(ny - 1):
            for x in range(nx - 1):
                corners = tuple(
                    voxel_by_index.get((x + dx, y + dy, z + dz))
                    for dx, dy, dz in _CORNER_OFFSETS
                )
                if any(corner is None for corner in corners):
                    skipped_unknown_cells += 1
                    continue

                observed_corners = tuple(
                    corner for corner in corners if corner is not None
                )
                if any(corner.tsdf == 0.0 for corner in observed_corners):
                    skipped_exact_zero_cells += 1
                    continue

                eligible_cells += 1
                faces_before_cell = len(faces)
                for tetrahedron in _TETRAHEDRA:
                    tetra_corners = tuple(
                        observed_corners[position] for position in tetrahedron
                    )
                    for candidate in _triangulate_tetrahedron(
                        volume,
                        tetra_corners,
                        vertex_ids,
                        vertices,
                        index_vertices,
                    ):
                        if len(faces) >= MAX_REFERENCE_TRIANGLES:
                            raise MeshExtractionError(
                                "triangle limit exceeded; reference meshing supports "
                                f"at most {MAX_REFERENCE_TRIANGLES} triangles"
                            )
                        duplicate_key = tuple(sorted(candidate))
                        if duplicate_key in face_keys:
                            raise MeshExtractionError(
                                "mesh construction produced a duplicate triangle"
                            )
                        face_keys.add(duplicate_key)
                        faces.append(candidate)
                if len(faces) > faces_before_cell:
                    active_cells += 1

    if not faces:
        raise MeshExtractionError(
            "TSDF volume produced no triangles "
            f"(cells={total_cells}, unknown={skipped_unknown_cells}, "
            f"exact_zero={skipped_exact_zero_cells}, eligible={eligible_cells})"
        )

    (
        boundary_edges,
        non_manifold_edges,
        non_manifold_vertices,
    ) = _topology_diagnostics(faces)
    return (
        _Mesh(
            total_cells=total_cells,
            skipped_unknown_cells=skipped_unknown_cells,
            skipped_exact_zero_cells=skipped_exact_zero_cells,
            eligible_cells=eligible_cells,
            active_cells=active_cells,
            vertices=tuple(vertices),
            index_vertices=tuple(index_vertices),
            faces=tuple(faces),
            boundary_edges=boundary_edges,
        ),
        non_manifold_edges,
        non_manifold_vertices,
    )


def _triangulate_tetrahedron(
    volume: _ReferenceTsdf,
    corners: tuple[_TsdfVoxel, _TsdfVoxel, _TsdfVoxel, _TsdfVoxel],
    vertex_ids: dict[_VertexKey, int],
    vertices: list[_Point],
    index_vertices: list[_Point],
) -> tuple[_Face, ...]:
    negative = sorted(
        (corner for corner in corners if corner.tsdf < 0.0),
        key=lambda corner: _flat_index(volume, corner.index),
    )
    positive = sorted(
        (corner for corner in corners if corner.tsdf > 0.0),
        key=lambda corner: _flat_index(volume, corner.index),
    )
    if not negative or not positive:
        return ()

    candidates: tuple[tuple[int, int, int], ...]
    if len(negative) == 1:
        crossing_ids = tuple(
            _intersection_vertex(
                volume,
                negative[0],
                outside,
                vertex_ids,
                vertices,
                index_vertices,
            )
            for outside in positive
        )
        candidates = (crossing_ids,)  # type: ignore[assignment]
    elif len(positive) == 1:
        crossing_ids = tuple(
            _intersection_vertex(
                volume,
                inside,
                positive[0],
                vertex_ids,
                vertices,
                index_vertices,
            )
            for inside in negative
        )
        candidates = (crossing_ids,)  # type: ignore[assignment]
    else:
        q00 = _intersection_vertex(
            volume,
            negative[0],
            positive[0],
            vertex_ids,
            vertices,
            index_vertices,
        )
        q01 = _intersection_vertex(
            volume,
            negative[0],
            positive[1],
            vertex_ids,
            vertices,
            index_vertices,
        )
        q10 = _intersection_vertex(
            volume,
            negative[1],
            positive[0],
            vertex_ids,
            vertices,
            index_vertices,
        )
        q11 = _intersection_vertex(
            volume,
            negative[1],
            positive[1],
            vertex_ids,
            vertices,
            index_vertices,
        )
        candidates = ((q00, q01, q11), (q00, q11, q10))

    negative_centroid = _centroid(
        tuple(_index_center(corner.index) for corner in negative)
    )
    positive_centroid = _centroid(
        tuple(_index_center(corner.index) for corner in positive)
    )
    toward_positive = _subtract(positive_centroid, negative_centroid)
    return tuple(
        _orient_toward_positive(candidate, index_vertices, toward_positive)
        for candidate in candidates
    )


def _intersection_vertex(
    volume: _ReferenceTsdf,
    negative: _TsdfVoxel,
    positive: _TsdfVoxel,
    vertex_ids: dict[_VertexKey, int],
    vertices: list[_Point],
    index_vertices: list[_Point],
) -> int:
    negative_flat = _flat_index(volume, negative.index)
    positive_flat = _flat_index(volume, positive.index)
    key = (
        min(negative_flat, positive_flat),
        max(negative_flat, positive_flat),
    )
    existing = vertex_ids.get(key)
    if existing is not None:
        return existing

    alpha = negative.tsdf / (negative.tsdf - positive.tsdf)
    negative_center = _voxel_center(volume, negative.index)
    positive_center = _voxel_center(volume, positive.index)
    point = tuple(
        first + alpha * (second - first)
        for first, second in zip(
            negative_center,
            positive_center,
            strict=True,
        )
    )
    if not all(math.isfinite(value) for value in point):
        raise MeshExtractionError(
            "triangle-mesh interpolation produced a non-finite coordinate"
        )
    negative_index_center = _index_center(negative.index)
    positive_index_center = _index_center(positive.index)
    index_point = tuple(
        first + alpha * (second - first)
        for first, second in zip(
            negative_index_center,
            positive_index_center,
            strict=True,
        )
    )

    identifier = len(vertices)
    vertex_ids[key] = identifier
    vertices.append(point)  # type: ignore[arg-type]
    index_vertices.append(index_point)  # type: ignore[arg-type]
    return identifier


def _orient_toward_positive(
    face: tuple[int, int, int],
    vertices: list[_Point],
    toward_positive: _Point,
) -> _Face:
    first, second, third = (vertices[index] for index in face)
    normal = _cross(_subtract(second, first), _subtract(third, first))
    normal_squared = _dot(normal, normal)
    alignment = _dot(normal, toward_positive)
    if (
        not math.isfinite(normal_squared)
        or not math.isfinite(alignment)
        or normal_squared == 0.0
        or alignment == 0.0
    ):
        raise MeshExtractionError(
            "triangle-mesh construction produced a degenerate face"
        )
    if alignment < 0.0:
        return (face[0], face[2], face[1])
    return face


def _flat_index(
    volume: _ReferenceTsdf,
    index: tuple[int, int, int],
) -> int:
    nx, ny, _ = volume.dimensions
    return (index[2] * ny + index[1]) * nx + index[0]


def _index_center(index: tuple[int, int, int]) -> _Point:
    return (
        index[0] + 0.5,
        index[1] + 0.5,
        index[2] + 0.5,
    )


def _centroid(points: tuple[_Point, ...]) -> _Point:
    count = len(points)
    return (
        sum(point[0] for point in points) / count,
        sum(point[1] for point in points) / count,
        sum(point[2] for point in points) / count,
    )


def _subtract(first: _Point, second: _Point) -> _Point:
    return (
        first[0] - second[0],
        first[1] - second[1],
        first[2] - second[2],
    )


def _cross(first: _Point, second: _Point) -> _Point:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _dot(first: _Point, second: _Point) -> float:
    return (
        first[0] * second[0]
        + first[1] * second[1]
        + first[2] * second[2]
    )


def _topology_diagnostics(faces: list[_Face]) -> tuple[int, int, int]:
    incidences: dict[tuple[int, int], int] = {}
    vertex_links: dict[int, list[tuple[int, int]]] = {}
    for first, second, third in faces:
        for edge in (
            (first, second),
            (second, third),
            (third, first),
        ):
            key = (min(edge), max(edge))
            incidences[key] = incidences.get(key, 0) + 1
        vertex_links.setdefault(first, []).append((second, third))
        vertex_links.setdefault(second, []).append((third, first))
        vertex_links.setdefault(third, []).append((first, second))
    boundary_edges = sum(count == 1 for count in incidences.values())
    non_manifold_edges = sum(count > 2 for count in incidences.values())
    non_manifold_vertices = sum(
        not _is_manifold_vertex_link(link_edges)
        for link_edges in vertex_links.values()
    )
    return boundary_edges, non_manifold_edges, non_manifold_vertices


def _is_manifold_vertex_link(link_edges: list[tuple[int, int]]) -> bool:
    adjacency: dict[int, set[int]] = {}
    for first, second in link_edges:
        adjacency.setdefault(first, set()).add(second)
        adjacency.setdefault(second, set()).add(first)

    degrees = tuple(len(neighbors) for neighbors in adjacency.values())
    if any(degree not in (1, 2) for degree in degrees):
        return False
    if sum(degree == 1 for degree in degrees) not in (0, 2):
        return False

    start = next(iter(adjacency))
    visited: set[int] = set()
    pending = [start]
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        pending.extend(adjacency[current] - visited)
    return len(visited) == len(adjacency)


def _write_mesh_ply(
    volume: _ReferenceTsdf,
    mesh: _Mesh,
    output_path: Path,
) -> str:
    temporary_path: Path | None = None
    try:
        formatted_vertices = _prepare_serialized_vertices(mesh)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output_path.stem}-",
            suffix=".ply",
            dir=output_path.parent,
        )
        temporary_path = Path(temporary_name)
        header = "\n".join(
            [
                "ply",
                "format ascii 1.0",
                f"comment spatialforge_session {volume.session_id}",
                "comment spatialforge_replay_sha256 "
                f"{volume.replay_digest_sha256}",
                "comment spatialforge_source_tsdf_sha256 "
                f"{volume.digest_sha256}",
                "comment coordinates metres world_x_forward world_y_left "
                "world_z_up",
                "comment spatialforge_mesh_rule "
                "freudenthal_six_tetra_strict_signs",
                "comment spatialforge_mesh_winding toward_positive_tsdf_free_space",
                "comment spatialforge_unknown_cells skipped",
                "comment spatialforge_exact_zero_cells skipped",
                f"element vertex {len(mesh.vertices)}",
                "property double x",
                "property double y",
                "property double z",
                f"element face {len(mesh.faces)}",
                "property list uchar int vertex_indices",
                "end_header",
                "",
            ]
        ).encode("ascii")
        with open(descriptor, "wb", closefd=True) as output_file:
            output_file.write(header)
            for x, y, z in formatted_vertices:
                output_file.write(
                    f"{x} {y} {z}\n".encode("ascii")
                )
            for first, second, third in mesh.faces:
                output_file.write(
                    f"3 {first} {second} {third}\n".encode("ascii")
                )

        output_digest = _sha256_file(temporary_path)
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
        return output_digest
    except MeshExtractionError:
        raise
    except OSError as error:
        raise MeshExtractionError(f"cannot write mesh output: {error}") from error
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def _prepare_serialized_vertices(
    mesh: _Mesh,
) -> tuple[tuple[str, str, str], ...]:
    formatted = tuple(
        (
            _format_coordinate(point[0]),
            _format_coordinate(point[1]),
            _format_coordinate(point[2]),
        )
        for point in mesh.vertices
    )
    quantized = tuple(
        tuple(_fixed_decimal_units(component) for component in point)
        for point in formatted
    )
    if len(set(quantized)) != len(quantized):
        raise MeshExtractionError(
            "nine-decimal PLY precision would merge distinct mesh vertices"
        )

    for face in mesh.faces:
        first, second, third = (quantized[index] for index in face)
        quantized_normal = _cross_integer(
            tuple(second[axis] - first[axis] for axis in range(3)),
            tuple(third[axis] - first[axis] for axis in range(3)),
        )
        normal_scale = max(abs(component) for component in quantized_normal)
        if normal_scale == 0:
            raise MeshExtractionError(
                "nine-decimal PLY precision would collapse a mesh triangle"
            )

        index_first, index_second, index_third = (
            mesh.index_vertices[index] for index in face
        )
        reference_normal = _cross(
            _subtract(index_second, index_first),
            _subtract(index_third, index_first),
        )
        normalized_quantized = tuple(
            component / normal_scale for component in quantized_normal
        )
        alignment = _dot(
            normalized_quantized,  # type: ignore[arg-type]
            reference_normal,
        )
        if not math.isfinite(alignment) or alignment <= 0.0:
            raise MeshExtractionError(
                "nine-decimal PLY precision would invalidate mesh winding"
            )
    return formatted


def _fixed_decimal_units(value: str) -> int:
    negative = value.startswith("-")
    unsigned = value[1:] if negative else value
    whole, fraction = unsigned.split(".", maxsplit=1)
    units = int(whole) * 1_000_000_000 + int(fraction)
    return -units if negative else units


def _cross_integer(
    first: tuple[int, int, int],
    second: tuple[int, int, int],
) -> tuple[int, int, int]:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as input_file:
            while chunk := input_file.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise MeshExtractionError(f"cannot hash mesh output: {error}") from error
    return digest.hexdigest()


def _format_coordinate(value: float) -> str:
    if abs(value) < 0.5e-9:
        value = 0.0
    return f"{value:.9f}"
