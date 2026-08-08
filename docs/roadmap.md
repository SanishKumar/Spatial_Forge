# Roadmap

Where the reconstruction path stands, and what is deliberately not built yet.

## Working

**Sessions and replay.** Folder-backed `ScanSession v0.1` with calibrated RGB,
depth, IMU and pose streams. Strict validation of timestamps, paths, depth
scale, camera models and coordinate conventions. Deterministic replay with
exact timestamp association and SHA-256 input digests. TUM RGB-D import.

**Dense reconstruction.** Fixed-bounds and automatically inferred-bounds
projective TSDF integration. Known-pose RGB-D back-projection to coloured
point clouds. Zero-crossing surface points and a deterministic indexed
triangle mesh.

**Sparse block volumes.** Depth-driven planning of 8×8×8 candidate blocks with
a truncation halo, written as a verifiable `.sftplan` artifact. Strict
read-only loading, replay verification, allocation, and signed voxel
addressing. Byte-identical to the dense integrator on the same input.

**Free-space reasoning.** Conservative pixel-footprint coverage and per-voxel
cross-view resolution, which together separate observed surface from carvable
free space, occlusion and never-observed space. Feeds a plan-expansion
proposal that grows a plan without ever removing a planned block.

**Fusion.** Resumable and idempotent, tracked by an in-memory ledger, at block
granularity and at `(row, observation)` granularity so a capture can arrive
one frame at a time. Byte-identical to the one-shot traversal at any chunk
size.

**Vectorised path.** Whole-block evaluation against one observation in a
single float64 NumPy pass, bit-identical to the scalar evaluator, and
whole-block fusion from those fields, byte-identical to the scalar traversal.
About 100× faster; a room scan fuses in 2 s rather than 210 s.

**Validation.** A seeded non-degenerate synthetic room scan, and a real TUM
RGB-D sequence scored against held-out frames at 9.3 mm median error.

## Next

1. **Vectorise the block planner.** At ~2 s per 640×480 frame it is now the
   dominant cost — 201 s of a 242 s pipeline, against fusion's 33 s. The
   vectorised evaluator is the template; the existing per-pixel path is the
   reference.
2. **Stream or window the depth context.** It retains float64 depth for every
   selected frame against a 512 MB ceiling, capping 640×480 at roughly 208
   frames. Full sequences need this.
3. **Fix meshing on real data.** The fixed six-tetrahedron split produces
   non-manifold vertices on real geometry. Needs connected-component and
   quality validation, exact-zero cells, and normals.
4. **Drive the ledgers from field fusion**, so resumable and partial-frame
   fusion run at the vector path's rate, with a plan-wide entry point.
5. **Persist block volumes** together with their fusion ledger, so resumption
   survives process exit, and connect the surface and mesh consumers.
6. **Culled traversal** for full sequences, plus confidence and
   sensor-dependent weighting, depth and pose outlier rejection, and
   configurable production bounds.
7. **More real datasets** — ARKitScenes and similar — with published accuracy.
8. **Structure**: gravity and floor alignment, floor and wall candidates,
   openings.

## Not started

These are the parts that turn a reconstruction engine into a mapping system.
None of it should begin before the known-pose path is numerically and visually
trustworthy.

- A portable map package format carrying quality, uncertainty and provenance.
- Pose estimation: visual keyframes, features, matching, and trajectory
  evaluation against known poses.
- IMU preintegration, visual-inertial optimisation, factor graphs, loop
  closure and place recognition.
- Structural mapping: floors, walls, openings, rooms, corridors, walkable
  regions and coverage.
- Semantic mapping: objects, doors, stairs, lifts, signs, with confidence and
  multi-view provenance.
- Visitor localization: retrieval, local matching, 2D/3D correspondences, PnP,
  pose verification and relocalization.
- Multi-session and multi-floor alignment, and change detection.

## Verifying progress

```bash
.venv/Scripts/python.exe -m pip install -e .
.venv/Scripts/python.exe -W error -m unittest discover -s tests -p "test_*.py"
```

434 tests, about 80 seconds, must finish `OK`.

Beyond the suite, the numbers that are independent of visual appearance:
deterministic replay hashes, inferred bounds, dense/sparse byte parity,
planned block coordinates, fusion weights and counts, mesh topology, winding
and boundary counts. Every CLI command prints them.
