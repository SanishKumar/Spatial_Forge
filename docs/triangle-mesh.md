# Deterministic reference triangle mesh

This milestone converts a validated `.sftsdf` volume into an indexed ASCII PLY
triangle mesh:

```powershell
.\.venv\Scripts\python.exe -m spatialforge reconstruct triangle-mesh `
  outputs/progress-mesh.sftsdf `
  outputs/progress-mesh.ply
```

It is a bounded CPU correctness reference, not the production meshing backend.

## Cell and topology rule

TSDF samples live at voxel centers, so each mesh cell spans eight neighboring
centers. A cell participates only when all eight centers are observed and all
eight TSDF values are strictly nonzero. Unknown cells are never bridged.
Exact-zero cells are conservatively skipped and reported in this milestone.

Each eligible cell is split around its `000 -> 111` diagonal into six fixed
Freudenthal tetrahedra:

```text
000-100-110-111
000-110-010-111
000-010-011-111
000-011-001-111
000-001-101-111
000-101-100-111
```

Every tetrahedron is intersected at TSDF zero. Opposite-sign endpoints use
linear interpolation:

```text
t = tsdf_negative / (tsdf_negative - tsdf_positive)
p = center_negative + t * (center_positive - center_negative)
```

Vertices shared by cells or tetrahedra are welded by their canonical global
endpoint indices, never by rounded coordinates. Cells are traversed X-fastest,
tetrahedra use the order above, and two-versus-two cases use a fixed diagonal.
Faces point toward increasing TSDF: the positive/free-space side.

The reference implementation refuses more than 1,000,000 triangles. It also
refuses existing or race-created outputs and fails without a partial PLY when
no triangles are available. Basic edge and vertex-link checks reject
non-manifold topology. The final nine-decimal coordinates are checked again
before publication so rounding cannot merge vertices, collapse triangles, or
reverse their winding.

## Exact fixture proof

Build the fully observed `2 x 2 x 2` TSDF:

```powershell
.\.venv\Scripts\python.exe -m spatialforge reconstruct tsdf `
  tests/fixtures/minimal.vgsession `
  outputs/progress-mesh.sftsdf `
  --origin 0.5 -0.5 -0.5 `
  --dimensions 2 2 2 `
  --voxel-size-m 0.5 `
  --truncation-m 0.5
```

Four centers at `X=0.75 m` have TSDF `+0.5`; four at `X=1.25 m` have
TSDF `-0.5`. Meshing reports:

```text
cells: total=1 eligible=1 active=1
skipped_cells: unknown=0 exact_zero=0
mesh: vertices=9 triangles=8 boundary_edges=8
```

The mesh is a square on `X=1.0 m`, spans `Y,Z = [-0.25, 0.25]`, has area
`0.25 m^2`, and is wound toward `-X`, where this fixture has positive TSDF.
Its deterministic output SHA-256 is:

```text
b240a3a8286eb1025dd5e64047a63d8aa56519a21b201a050eca7c951d4b1a35
```

## PLY contract

The output contains only double-precision XYZ vertices and indexed triangle
faces:

```text
element vertex V
property double x
property double y
property double z
element face F
property list uchar int vertex_indices
```

Coordinates use metres and nine decimal places. Lines use LF endings and the
file ends with a newline. Header comments record the session, replay digest,
source TSDF digest, algorithm, winding, and skipped-cell rules.

## Explicitly deferred

- exact-zero-cell triangulation;
- marching cubes or another production extraction backend;
- vertex or face normals, colors, materials, and texture coordinates;
- smoothing, decimation, hole filling, and watertightness repair;
- connected-component and higher-level mesh quality reports;
- sparse/GPU extraction and full-sequence performance; and
- floor, wall, opening, room, semantic, and visual-inspector work.
