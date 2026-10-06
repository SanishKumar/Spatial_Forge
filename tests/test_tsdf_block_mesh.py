"""The sparse block mesher, pinned to the reference mesher.

``mesh._build_unchecked_mesh`` walks a dense grid one cell at a time and is
the definition of the triangulation. The block mesher must produce the same
vertices in the same order with the same bits, and the same triangles with
the same winding, from the same volume. Where the two differ on purpose --
the reference refuses a surface with pinch vertices and the block mesher
splits them -- the reference's own topology check is used to confirm that
the split surface is one it would have accepted.
"""

from __future__ import annotations

import hashlib
import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np

from spatialforge import (
    allocate_empty_tsdf_blocks,
    fuse_tsdf_plan_streaming,
    load_tsdf_block_plan,
    write_tsdf_block_volume,
)
from spatialforge.cli import main
from spatialforge.errors import MeshExtractionError
from spatialforge.mesh import (
    _CORNER_OFFSETS,
    _build_unchecked_mesh,
    _topology_diagnostics,
)
from spatialforge.session_loader import load_scan_session
from spatialforge.surface import _ReferenceTsdf, _TsdfVoxel
from spatialforge.tsdf_block_mesh import (
    _CASES,
    _SORTED_TETRAHEDRA,
    _connected_labels,
    _edge_topology,
    _filter_components,
    _split_pinch_vertices,
    build_tsdf_block_mesh,
    extract_tsdf_block_mesh,
    triangulate_tsdf_block_volume,
)
from spatialforge.tsdf_block_plan import plan_tsdf_blocks
from spatialforge.tsdf_block_volume import TsdfBlockVolume

from tests.heavy_fixtures import shared_room_case

TEST_ROOT = Path(__file__).resolve().parent
FIXTURE = TEST_ROOT / "fixtures" / "minimal.vgsession"
DIGEST = "0" * 64
FIXTURE_VOLUME_SHA256 = (
    "29f9f427b43c2a9e8ec416dad9a3ba1572044c716407ef2b30989daed9c2adc3"
)
# Pinned literally: holds only if meshing is bit-reproducible across the
# operating systems and interpreters the suite runs on.
FIXTURE_MESH_SHA256 = (
    "8c019fe39e181ef1855c8c229422528cf4218ca5b4998af533ae5e00d4e5f190"
)


def frozen(array: np.ndarray, dtype: str) -> np.ndarray:
    return np.frombuffer(
        array.astype(dtype).tobytes(),
        dtype=dtype,
    ).reshape(array.shape)


def make_volume(blocks, sums, weights, voxel_size_m) -> TsdfBlockVolume:
    order = sorted(
        range(len(blocks)),
        key=lambda row: (blocks[row][2], blocks[row][1], blocks[row][0]),
    )
    weights = weights[order]
    return TsdfBlockVolume(
        path=Path("synthetic.sftvol"),
        artifact_digest_sha256=DIGEST,
        session_id="synthetic",
        replay_digest_sha256=DIGEST,
        source_plan_digest_sha256=DIGEST,
        voxel_size_m=voxel_size_m,
        truncation_m=voxel_size_m * 3,
        block_resolution=8,
        frame_stride=1,
        total_observations=1,
        selected_observations=1,
        fused_observations=1,
        skipped_missing_depth=0,
        skipped_missing_pose=0,
        contributions_evaluated=len(blocks) * 512,
        contributions_applied=int(weights.sum()),
        observed_voxel_count=int(np.count_nonzero(weights)),
        maximum_weight=int(weights.max()),
        block_indices=tuple(blocks[row] for row in order),
        tsdf_sums=frozen(sums[order], "<f8"),
        weights=frozen(weights, "<u4"),
    )


def synthetic_volume(
    seed,
    blocks,
    voxel_size_m,
    drop,
    kind,
) -> TsdfBlockVolume:
    """A small volume with a known field and randomly unobserved voxels."""

    rng = np.random.default_rng(seed)
    sums = np.zeros((len(blocks), 8, 8, 8))
    weights = np.zeros((len(blocks), 8, 8, 8), dtype=np.uint32)
    z, y, x = np.meshgrid(
        np.arange(8),
        np.arange(8),
        np.arange(8),
        indexing="ij",
    )
    for row, (block_x, block_y, block_z) in enumerate(blocks):
        global_x = block_x * 8 + x + 0.5
        global_y = block_y * 8 + y + 0.5
        global_z = block_z * 8 + z + 0.5
        if kind == "sphere":
            value = (
                np.sqrt(
                    (global_x - 9.3) ** 2
                    + (global_y - 8.1) ** 2
                    + (global_z - 7.7) ** 2
                )
                - 5.6
            )
        elif kind == "sheet":
            value = (
                0.37 * global_x
                + 0.21 * global_y
                - 0.9 * global_z
                + 3.1
                + 0.013 * (global_x - 7.2) ** 2
                - 0.007 * global_y * global_z
            )
        else:
            value = rng.normal(size=(8, 8, 8))
        value = np.clip(value / 3.0, -1.0, 1.0)
        weight = rng.integers(1, 5, size=(8, 8, 8)).astype(np.uint32)
        weight[rng.random((8, 8, 8)) < drop] = 0
        sums[row] = np.where(weight > 0, value * weight, 0.0)
        weights[row] = weight
    return make_volume(list(blocks), sums, weights, voxel_size_m)


def reference_volume(volume: TsdfBlockVolume) -> _ReferenceTsdf:
    """The dense grid the reference mesher needs, holding the same values.

    Its origin is exactly zero and its indices are the global voxel
    indices, so the reference computes voxel centres as ``0.0 + (i + 0.5) *
    voxel`` -- bit-identical to the block mesher's ``(i + 0.5) * voxel``.
    That requires every block index to be non-negative.
    """

    blocks = np.array(volume.block_indices)
    if blocks.min() < 0:
        raise AssertionError("reference comparison needs non-negative blocks")
    dimensions = tuple(int(value) for value in (blocks.max(axis=0) + 1) * 8)
    tsdf = volume.normalized_tsdf()
    voxels = []
    for row, (block_x, block_y, block_z) in enumerate(volume.block_indices):
        zs, ys, xs = np.nonzero(volume.weights[row])
        for z, y, x in zip(zs.tolist(), ys.tolist(), xs.tolist()):
            voxels.append(
                _TsdfVoxel(
                    index=(block_x * 8 + x, block_y * 8 + y, block_z * 8 + z),
                    tsdf=float(tsdf[row, z, y, x]),
                    weight=int(volume.weights[row, z, y, x]),
                )
            )
    voxels.sort(key=lambda voxel: voxel.index[::-1])
    return _ReferenceTsdf(
        path=Path("synthetic.sftsdf"),
        digest_sha256=DIGEST,
        session_id="synthetic",
        replay_digest_sha256=DIGEST,
        origin_world_m=(0.0, 0.0, 0.0),
        dimensions=dimensions,  # type: ignore[arg-type]
        voxel_size_m=volume.voxel_size_m,
        total_voxels=dimensions[0] * dimensions[1] * dimensions[2],
        voxels=tuple(voxels),
    )


def cube(blocks_x: int, blocks_y: int, blocks_z: int):
    return [
        (x, y, z)
        for z in range(blocks_z)
        for y in range(blocks_y)
        for x in range(blocks_x)
    ]


SYNTHETIC_CASES = (
    ("sphere, fully observed", 1, cube(3, 2, 2), 0.04, 0.0, "sphere"),
    ("sphere, ragged", 2, cube(3, 2, 2), 0.04, 0.25, "sphere"),
    (
        "sheet, missing blocks",
        3,
        [(0, 0, 0), (1, 0, 0), (0, 1, 0), (2, 1, 1), (1, 1, 1), (0, 0, 1)],
        0.0173,
        0.1,
        "sheet",
    ),
    ("noise, ragged", 4, cube(2, 2, 1), 0.05, 0.3, "noise"),
    ("noise, fully observed", 5, cube(2, 1, 1), 0.05, 0.0, "noise"),
    (
        "sphere, scattered blocks",
        6,
        [(0, 0, 0), (2, 0, 0), (1, 1, 0), (0, 0, 1), (1, 0, 1), (2, 1, 1)],
        0.031,
        0.15,
        "sphere",
    ),
    ("noise, mostly unobserved", 7, cube(2, 2, 2), 0.02, 0.5, "noise"),
)


def fixture_volume_file(parent: Path) -> Path:
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
    output = parent / "fixture.sftvol"
    write_tsdf_block_volume(storage, receipt, output)
    return output


def read_binary_ply(path: Path):
    encoded = path.read_bytes()
    end = encoded.index(b"end_header\n") + len(b"end_header\n")
    header = encoded[:end].decode("ascii").splitlines()
    vertex_count = next(
        int(line.split()[2])
        for line in header
        if line.startswith("element vertex")
    )
    face_count = next(
        int(line.split()[2])
        for line in header
        if line.startswith("element face")
    )
    vertices = np.frombuffer(
        encoded,
        dtype="<f8",
        count=vertex_count * 3,
        offset=end,
    ).reshape((vertex_count, 3))
    records = np.frombuffer(
        encoded,
        dtype=[("count", "u1"), ("indices", "<i4", (3,))],
        count=face_count,
        offset=end + vertex_count * 24,
    )
    assert end + vertex_count * 24 + face_count * 13 == len(encoded)
    return header, vertices, records["count"], records["indices"]


class ReferenceParityTests(unittest.TestCase):
    def assert_matches_reference(self, volume: TsdfBlockVolume) -> int:
        vertices, faces, counts = triangulate_tsdf_block_volume(volume)
        reference, non_manifold_edges, pinches = _build_unchecked_mesh(
            reference_volume(volume)
        )
        self.assertEqual(
            vertices.tobytes(),
            np.array(reference.vertices, dtype=np.float64).tobytes(),
        )
        self.assertEqual(
            faces.tolist(),
            [list(face) for face in reference.faces],
        )
        self.assertEqual(counts[2], reference.skipped_exact_zero_cells)
        self.assertEqual(counts[3], reference.eligible_cells)
        self.assertEqual(counts[4], reference.active_cells)
        self.assertEqual(non_manifold_edges, 0)

        mesh = build_tsdf_block_mesh(volume)
        self.assertEqual(mesh.pinch_vertices_split, pinches)
        self.assertEqual(mesh.boundary_edges, reference.boundary_edges)
        self.assertEqual(mesh.triangle_count, len(reference.faces))
        self.assertEqual(
            mesh.vertex_count,
            len(reference.vertices) + mesh.vertices_added_by_splitting,
        )
        # The reference's own verdict on the surface after splitting.
        boundary, edges, vertices_left = _topology_diagnostics(
            [tuple(face) for face in mesh.faces.tolist()]
        )
        self.assertEqual((edges, vertices_left), (0, 0))
        self.assertEqual(boundary, reference.boundary_edges)
        return pinches

    def test_synthetic_volumes_match_the_reference_exactly(self) -> None:
        total_pinches = 0
        for label, seed, blocks, voxel, drop, kind in SYNTHETIC_CASES:
            with self.subTest(case=label):
                total_pinches += self.assert_matches_reference(
                    synthetic_volume(seed, blocks, voxel, drop, kind)
                )
        # The comparison is only interesting if some of these volumes are
        # ones the reference would have refused.
        self.assertGreater(total_pinches, 0)

    def test_real_fused_room_values_match_the_reference_exactly(self) -> None:
        """The same check on values produced by actual fusion."""

        case = shared_room_case()
        storage = allocate_empty_tsdf_blocks(case.plan, case.session)
        fuse_tsdf_plan_streaming(storage, case.session)
        blocks = np.array(storage.block_indices)
        # A sub-volume keeps the per-cell Python reference affordable, and
        # shifting it into the positive octant is a relabelling both
        # meshers see identically.
        low = np.percentile(blocks, 35, axis=0).astype(int)
        selected = [
            row
            for row, block in enumerate(blocks)
            if all(
                low[axis] <= block[axis] < low[axis] + 4
                for axis in range(3)
            )
        ]
        self.assertGreater(len(selected), 8)
        shifted = [
            tuple(int(value) for value in blocks[row] - low)
            for row in selected
        ]
        volume = make_volume(
            shifted,
            storage.tsdf_sums[selected],
            storage.weights[selected],
            case.plan.voxel_size_m,
        )
        self.assert_matches_reference(volume)

    def test_both_meshers_refuse_a_degenerate_face_identically(self) -> None:
        """An exactly planar field puts crossings on top of one another."""

        blocks = cube(2, 1, 1)
        sums = np.zeros((2, 8, 8, 8))
        weights = np.ones((2, 8, 8, 8), dtype=np.uint32)
        z, y, x = np.meshgrid(
            np.arange(8),
            np.arange(8),
            np.arange(8),
            indexing="ij",
        )
        for row, (block_x, _, _) in enumerate(blocks):
            sums[row] = np.where(block_x * 8 + x + y + z < 9, -0.5, 1e-300)
        volume = make_volume(blocks, sums, weights, 0.05)
        with self.assertRaises(MeshExtractionError) as sparse:
            triangulate_tsdf_block_volume(volume)
        with self.assertRaises(MeshExtractionError) as reference:
            _build_unchecked_mesh(reference_volume(volume))
        self.assertIn("degenerate face", str(sparse.exception))
        self.assertEqual(str(sparse.exception), str(reference.exception))


class MarchingCaseTableTests(unittest.TestCase):
    def test_winding_never_depends_on_where_the_crossings_fall(self) -> None:
        """The table fixes winding at edge midpoints; that must be general.

        For crossings strictly inside their edges the orientation test has
        the same sign wherever they sit, which is what lets the table stand
        in for the reference's per-triangle arithmetic.
        """

        rng = np.random.default_rng(11)
        offsets = np.array(_CORNER_OFFSETS, dtype=np.float64)
        checked = 0
        for (tetra_index, pattern), (calls, triangles) in _CASES.items():
            corners = _SORTED_TETRAHEDRA[tetra_index]
            negative = [
                corner
                for position, corner in enumerate(corners)
                if pattern >> position & 1
            ]
            positive = [c for c in corners if c not in negative]
            toward = offsets[positive].mean(axis=0) - offsets[
                negative
            ].mean(axis=0)
            for _ in range(25):
                alpha = rng.uniform(0.001, 0.999, size=len(calls))
                points = [
                    offsets[inside]
                    + alpha[index] * (offsets[outside] - offsets[inside])
                    for index, (inside, outside) in enumerate(calls)
                ]
                for first, second, third in triangles:
                    normal = np.cross(
                        points[second] - points[first],
                        points[third] - points[first],
                    )
                    self.assertGreater(float(normal @ toward), 0.0)
                    checked += 1
        self.assertEqual(len(_CASES), 6 * 14)
        self.assertGreater(checked, 2_000)


class TopologyRepairTests(unittest.TestCase):
    def test_a_bow_tie_vertex_is_split_into_one_vertex_per_fan(self) -> None:
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ]
        )
        faces = np.array([[0, 1, 2], [0, 3, 4]], dtype=np.int64)
        _, pinch_pairs = _edge_topology(faces, len(vertices))
        split_vertices, split_faces, pinched, added = _split_pinch_vertices(
            vertices,
            faces,
            pinch_pairs,
        )

        self.assertEqual((pinched, added), (1, 1))
        self.assertEqual(split_faces.tolist(), [[0, 1, 2], [5, 3, 4]])
        self.assertEqual(split_vertices[5].tolist(), [0.0, 0.0, 0.0])
        _, edges, pinches = _topology_diagnostics(
            [tuple(face) for face in split_faces.tolist()]
        )
        self.assertEqual((edges, pinches), (0, 0))

    def test_a_closed_fan_is_left_alone(self) -> None:
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ]
        )
        faces = np.array(
            [[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]],
            dtype=np.int64,
        )
        boundary, pairs = _edge_topology(faces, len(vertices))
        result = _split_pinch_vertices(vertices, faces, pairs)

        self.assertEqual(boundary, 4)
        self.assertEqual(result[2:], (0, 0))
        self.assertIs(result[1], faces)

    def test_impossible_edge_topology_is_reported_not_repaired(self) -> None:
        three_on_an_edge = np.array(
            [[0, 1, 2], [1, 0, 3], [0, 1, 4]],
            dtype=np.int64,
        )
        with self.assertRaises(MeshExtractionError) as non_manifold:
            _edge_topology(three_on_an_edge, 5)
        same_direction = np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int64)
        with self.assertRaises(MeshExtractionError) as winding:
            _edge_topology(same_direction, 4)

        self.assertIn(
            "non-manifold triangle edges",
            str(non_manifold.exception),
        )
        self.assertIn("inconsistently wound", str(winding.exception))

    def test_small_fragments_are_dropped_and_vertices_renumbered(self) -> None:
        vertices = np.arange(27, dtype=np.float64).reshape((9, 3))
        faces = np.array(
            [[0, 1, 2], [3, 4, 5], [4, 6, 5], [5, 6, 7], [6, 8, 7]],
            dtype=np.int64,
        )
        kept_vertices, kept_faces, found, removed, dropped = (
            _filter_components(vertices, faces, 2)
        )

        self.assertEqual((found, removed, dropped), (2, 1, 1))
        self.assertEqual(kept_vertices.tolist(), vertices[3:].tolist())
        self.assertEqual(
            kept_faces.tolist(),
            [[0, 1, 2], [1, 3, 2], [2, 3, 4], [3, 5, 4]],
        )
        untouched = _filter_components(vertices, faces, 1)
        self.assertIs(untouched[1], faces)
        self.assertEqual(untouched[2:], (2, 0, 0))
        with self.assertRaises(MeshExtractionError):
            _filter_components(vertices, faces, 99)

    def test_component_labels_agree_with_a_plain_union_find(self) -> None:
        rng = np.random.default_rng(5)
        for count, edges in ((1, 0), (50, 20), (400, 380), (2_000, 6_000)):
            with self.subTest(count=count, edges=edges):
                first = rng.integers(0, count, size=edges)
                second = rng.integers(0, count, size=edges)
                parent = list(range(count))

                def find(node: int) -> int:
                    while parent[node] != node:
                        parent[node] = parent[parent[node]]
                        node = parent[node]
                    return node

                for a, b in zip(first.tolist(), second.tolist()):
                    root_a, root_b = find(a), find(b)
                    if root_a != root_b:
                        parent[max(root_a, root_b)] = min(root_a, root_b)
                expected = [find(node) for node in range(count)]
                self.assertEqual(
                    _connected_labels(count, first, second).tolist(),
                    expected,
                )

    def test_a_long_chain_is_one_component(self) -> None:
        count = 20_000
        order = np.random.default_rng(3).permutation(count)
        labels = _connected_labels(count, order[:-1], order[1:])
        self.assertEqual(int(labels.max()), 0)


class BlockMeshOptionTests(unittest.TestCase):
    def test_minimum_weight_treats_weak_voxels_as_unobserved(self) -> None:
        dense = synthetic_volume(21, cube(2, 2, 1), 0.04, 0.0, "sphere")
        values = dense.normalized_tsdf()
        # A contiguous slab seen only twice, in a volume otherwise seen
        # five times, so removing it leaves a meshable surface behind.
        weights = np.full(values.shape, 5, dtype=np.uint32)
        weights[:, :, :, :3] = 2
        sums = values * weights
        volume = make_volume(
            list(dense.block_indices),
            sums,
            weights,
            dense.voxel_size_m,
        )
        weak = weights < 3
        expected_volume = make_volume(
            list(volume.block_indices),
            np.where(weak, 0.0, sums),
            np.where(weak, 0, weights).astype(np.uint32),
            volume.voxel_size_m,
        )
        filtered = triangulate_tsdf_block_volume(volume, minimum_weight=3)
        expected = triangulate_tsdf_block_volume(expected_volume)

        self.assertEqual(filtered[0].tobytes(), expected[0].tobytes())
        self.assertEqual(filtered[1].tolist(), expected[1].tolist())
        self.assertNotEqual(
            len(filtered[1]),
            len(triangulate_tsdf_block_volume(volume)[1]),
        )

    def test_a_volume_with_no_surface_is_refused(self) -> None:
        blocks = cube(1, 1, 1)
        volume = make_volume(
            blocks,
            np.full((1, 8, 8, 8), 0.5),
            np.ones((1, 8, 8, 8), dtype=np.uint32),
            0.05,
        )
        with self.assertRaises(MeshExtractionError) as caught:
            build_tsdf_block_mesh(volume)
        self.assertIn("produced no triangles", str(caught.exception))
        self.assertIn("eligible=343", str(caught.exception))

    def test_exact_zero_cells_are_skipped_and_counted(self) -> None:
        volume = synthetic_volume(1, cube(2, 2, 1), 0.04, 0.0, "sphere")
        sums = np.array(volume.tsdf_sums)
        sums[0, 3, 3, 3] = 0.0
        zeroed = make_volume(
            list(volume.block_indices),
            sums,
            np.array(volume.weights),
            volume.voxel_size_m,
        )
        counts = triangulate_tsdf_block_volume(zeroed)[2]
        reference, _, _ = _build_unchecked_mesh(reference_volume(zeroed))

        self.assertEqual(counts[2], 8)
        self.assertEqual(counts[2], reference.skipped_exact_zero_cells)

    def test_invalid_options_are_refused(self) -> None:
        volume = synthetic_volume(1, cube(1, 1, 1), 0.04, 0.0, "sphere")
        for options in (
            {"minimum_weight": 0},
            {"minimum_weight": True},
            {"minimum_component_triangles": 0},
            {"minimum_component_triangles": 1.5},
        ):
            with self.subTest(options=options):
                with self.assertRaises(MeshExtractionError):
                    build_tsdf_block_mesh(volume, **options)
        with self.assertRaises(MeshExtractionError):
            build_tsdf_block_mesh(None)  # type: ignore[arg-type]


class BlockMeshFileTests(unittest.TestCase):
    def test_fixture_mesh_is_written_as_verifiable_binary_ply(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            temporary_root = Path(temporary_dir)
            volume_path = fixture_volume_file(temporary_root)
            output = temporary_root / "fixture.ply"
            report = extract_tsdf_block_mesh(volume_path, output)
            header, vertices, counts, faces = read_binary_ply(output)
            digest = hashlib.sha256(output.read_bytes()).hexdigest()
            again = temporary_root / "again.ply"
            extract_tsdf_block_mesh(volume_path, again)
            identical = again.read_bytes() == output.read_bytes()

        self.assertTrue(identical)
        self.assertEqual(report.output_digest_sha256, digest)
        self.assertEqual(digest, FIXTURE_MESH_SHA256)
        self.assertEqual(
            report.source_volume_digest_sha256,
            FIXTURE_VOLUME_SHA256,
        )
        self.assertEqual(
            (report.vertices_written, report.triangles_written),
            (225, 392),
        )
        self.assertEqual(report.boundary_edges, 56)
        self.assertEqual(vertices.shape, (225, 3))
        self.assertEqual(faces.shape, (392, 3))
        self.assertTrue(bool(np.all(counts == 3)))
        self.assertEqual(int(faces.min()), 0)
        self.assertEqual(int(faces.max()), 224)
        self.assertIn("format binary_little_endian 1.0", header)
        self.assertIn(
            f"comment spatialforge_source_volume_sha256 "
            f"{FIXTURE_VOLUME_SHA256}",
            header,
        )
        self.assertIn(
            "comment spatialforge_pinch_vertices split_per_triangle_fan",
            header,
        )

    def test_existing_outputs_and_bad_inputs_are_refused(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            temporary_root = Path(temporary_dir)
            volume_path = fixture_volume_file(temporary_root)
            existing = temporary_root / "existing.ply"
            existing.write_bytes(b"keep me")
            with self.assertRaises(MeshExtractionError) as exists:
                extract_tsdf_block_mesh(volume_path, existing)
            with self.assertRaises(MeshExtractionError) as suffix:
                extract_tsdf_block_mesh(volume_path, temporary_root / "m.obj")
            corrupt = temporary_root / "corrupt.sftvol"
            corrupt.write_bytes(volume_path.read_bytes()[:-1])
            unwritten = temporary_root / "unwritten.ply"
            with self.assertRaises(MeshExtractionError) as damaged:
                extract_tsdf_block_mesh(corrupt, unwritten)
            kept = existing.read_bytes()
            written = unwritten.exists()

        self.assertIn("already exists", str(exists.exception))
        self.assertIn(".ply", str(suffix.exception))
        self.assertIn("payload", str(damaged.exception))
        self.assertEqual(kept, b"keep me")
        self.assertFalse(written)


class BlockMeshCliTests(unittest.TestCase):
    def run_cli(self, arguments: list[str]) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main(arguments)
        return exit_code, stdout.getvalue(), stderr.getvalue()

    def test_cli_meshes_the_fixture_volume(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            temporary_root = Path(temporary_dir)
            volume_path = fixture_volume_file(temporary_root)
            output = temporary_root / "fixture.ply"
            exit_code, stdout, stderr = self.run_cli(
                [
                    "reconstruct",
                    "tsdf-block-volume-mesh",
                    str(volume_path),
                    str(output),
                ]
            )
            size = output.stat().st_size

        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr, "")
        self.assertEqual(
            stdout,
            "TSDF BLOCK VOLUME MESH scan-synthetic-0001\n"
            "volume: verified\n"
            "filters: minimum_weight=1 minimum_component_triangles=1\n"
            "cells: considered=4096 eligible=330 active=49\n"
            "skipped_cells: unknown=3766 exact_zero=0\n"
            "mesh_rule: freudenthal-six-tetra-strict-signs\n"
            "winding: toward-positive-tsdf-free-space\n"
            "pinch_vertices: split=0 vertices_added=0\n"
            "components: found=1 removed=0 triangles_removed=0\n"
            "mesh: vertices=225 triangles=392 boundary_edges=56\n"
            "non_manifold_edges: 0\n"
            "non_manifold_vertices: 0\n"
            "triangle_winding_consistent: yes\n"
            "ply_format: binary-little-endian-float64\n"
            f"mesh_bytes: {size}\n"
            f"output: {output.resolve()}\n"
            f"output_sha256: {FIXTURE_MESH_SHA256}\n"
            f"volume_sha256: {FIXTURE_VOLUME_SHA256}\n",
        )

    def test_cli_failure_is_actionable(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            temporary_root = Path(temporary_dir)
            exit_code, stdout, stderr = self.run_cli(
                [
                    "reconstruct",
                    "tsdf-block-volume-mesh",
                    str(temporary_root / "absent.sftvol"),
                    str(temporary_root / "mesh.ply"),
                ]
            )

        self.assertEqual(exit_code, 2)
        self.assertEqual(stdout, "")
        self.assertIn("TSDF BLOCK VOLUME MESH FAILED", stderr)
        self.assertIn("does not exist", stderr)
        self.assertNotIn("Traceback", stderr)


if __name__ == "__main__":
    unittest.main()
