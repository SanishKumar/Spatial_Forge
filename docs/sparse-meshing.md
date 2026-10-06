# Meshing a sparse volume

```powershell
python -m spatialforge reconstruct tsdf-block-volume-mesh `
  scan.sftvol scan.ply --min-weight 3 --min-component-triangles 200
```

There are two meshers. [`triangle-mesh.md`](triangle-mesh.md) describes the
reference: it walks a dense grid one cell at a time in Python and refuses any
surface with a non-manifold vertex. That is the right behaviour for a
definition and it made the first real scan impossible to mesh, twice over —
a sparse volume at a useful voxel size has no dense grid to walk, and real
data always contains a few such vertices.

This page describes the mesher that reads a `.sftvol`.

## The same triangulation

A cell is the cube between eight neighbouring voxel centres. It is meshed
only if all eight are observed and none is exactly zero. Each such cell is
cut into six tetrahedra around its main diagonal (the Freudenthal split),
and each tetrahedron whose corners differ in sign contributes one triangle
or two. A vertex lies on the edge between a negative and a positive voxel,
at the linear zero crossing, and is shared by every tetrahedron around that
edge. Triangles are wound toward the positive side, which is free space.

Those are the reference's rules, unchanged. What differs is that they are
applied to every cell at once:

- Each block is padded by one voxel from its `+x`, `+y` and `+z` neighbours,
  so cells that straddle blocks see all eight corners. A neighbour that is
  not planned contributes unknown voxels, exactly as an unobserved voxel
  would.
- A table indexed by tetrahedron and sign pattern — 6 × 14 entries — gives
  the crossing edges and the triangles over them.
- Vertices are numbered in the order the reference would create them: cells
  in dense `(z, y, x)` order, tetrahedra in order, crossings in call order.

For the same volume the result is the same vertices, in the same order, with
the same float64 bits, and the same triangle list. The tests compare exactly
that, on seven synthetic volumes and on a volume fused from the room scan.

### Winding from a table

The reference decides each triangle's winding by arithmetic: it compares the
triangle's normal with the direction from the negative corners to the
positive ones. The table stores the answer instead.

That is sound because the answer does not depend on where the crossings
fall. For a tetrahedron with one negative corner `n` and positive corners
`p0, p1, p2`, with crossings a fraction `a_i` of the way along each edge, the
orientation test works out to

```text
det(p0 - n, p1 - n, p2 - n) * (a0 a1 + a1 a2 + a0 a2) / 3
```

and the second factor is positive for any crossings strictly inside their
edges. The sign belongs to the pattern. A test evaluates all 84 cases at
random crossing positions and requires the table's winding every time.

## Pinch vertices

Real scans have ragged observed regions. Take the cells around one grid
edge: seen along that edge there are four of them, and the triangles they
contribute around a crossing on the edge form a fan.

```text
   all four observed          two opposite cells unobserved

      +-----+-----+               +-----+ . . .
      |  \  |  /  |               |  \  |     .
      +-----v-----+               +-----v-----+
      |  /  |  \  |               .     |  \  |
      +-----+-----+               . . . +-----+

    one closed fan              two fans meeting at v
```

When the cells on two opposite sides are observed and the ones between them
are not, the fan breaks into two that touch only at the vertex. Every edge
of the mesh still has at most two triangles, but the surface is not a
manifold at that point: it is two sheets pinched together.

The reference stops there. This mesher gives each fan its own copy of the
vertex. Two triangle corners at a vertex belong to the same fan when their
triangles share an edge through it; a union-find over corners groups them;
the first fan keeps the vertex and each further fan gets a copy at the same
position, in a fixed order. The result is a manifold with boundary.

This is a standard repair and nothing about the surface moves. To confirm it
does what is claimed, the *reference's* topology check is run on the split
mesh in the tests and must report no non-manifold vertex — so the result is
a surface the reference would have accepted, reached from one it refused.
The number of pinches found also has to equal the number the reference
counts.

What is not repaired: an edge carrying three triangles, or two triangles
crossing an edge in the same direction. Neither can come out of a consistent
tetrahedral split, so either would mean the mesher is wrong, and both raise.

## Two optional filters

`--min-weight W` treats voxels fused from fewer than `W` observations as
unknown before meshing. Voxels at the fringe of the truncation band are seen
once or twice and are mostly noise.

`--min-component-triangles T` drops connected fragments with fewer than `T`
triangles, after splitting.

Both default to off, both are recorded in the PLY header, and neither
touches the volume. On the TUM volume, `3` and `200` remove 751 fragments
holding 12,004 of 547,490 triangles.

## Output

A binary little-endian PLY with float64 vertices and a triangle list.
Nothing is rounded to decimal text, so no surface is refused for the
precision of its file format and the mesh's digest is stable. The header
records the session, the replay digest, the digest of the source volume, the
meshing rule and both filter settings.

## Limits

The volume's values come from projective distances, so the mesh inherits
that rule's bias. There are no vertex normals in the file, no colour, no
decimation and no hole filling. A cell with an exactly zero corner is
skipped rather than triangulated, as in the reference, and a triangle of
exactly zero area raises rather than being dropped.
