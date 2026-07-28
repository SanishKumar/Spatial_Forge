# Deterministic sparse TSDF accumulation

This checkpoint separates in-memory TSDF working state from logical volume
traversal.
The fixed-bounds sparse command stores sums and weights only for voxel indices
that receive an update:

```powershell
.\.venv\Scripts\python.exe -m spatialforge reconstruct tsdf-sparse `
  tests/fixtures/minimal.vgsession `
  outputs/progress-sparse.sftsdf `
  --origin 0.5 -1 -1 `
  --dimensions 2 4 4 `
  --voxel-size-m 0.5 `
  --truncation-m 0.5
```

It is intentionally separate from `tsdf`, which remains the dense numerical
oracle, and from `tsdf-auto`, which remains on the dense path for this phase.

## Exact compatibility contract

Sparse and dense integration share the same:

- deterministic replay and frame-stride selection;
- world-to-camera transform and voxel-center traversal order;
- pinhole projection, nearest-pixel depth lookup, and validity rules;
- truncation, uniform weight, and float64 accumulation order;
- counters, validation, input-digest checks, and overwrite protection; and
- canonical X-fastest `.sftsdf` serialization.

Before serialization, sparse voxel indices are sorted by their global flat
index. The artifact schema remains `spatialforge.reference-tsdf` version
`0.1.0`; it contains no backend-specific field. Given identical inputs and
arguments, dense and sparse commands must therefore produce byte-identical
files.

For the command above, the proof is:

```text
voxels: total=32 observed=8 fused=8
voxel_updates: 16 max_weight=2
storage: sparse accumulator_entries=8
output_sha256: e61803737cdd68b209459fb644cc2f67f18e0420d306273316808e27d2e89994
```

The existing surface-point and triangle-mesh commands consume this artifact
without conversion or special handling.

## What is sparse in this checkpoint

Only in-memory sums and weights are sparse. An unobserved logical voxel has no
accumulator entry. The command still visits the bounded logical volume in the
same fixed chunks as the dense reference for every integrated frame.

This is sparse storage inside dense reconstruction. It is unrelated to the
sparse visual landmarks and localization map planned later in the architecture.

This distinction is important: it proves the storage abstraction and numerical
equivalence, but it is not yet a full-sequence performance backend. A Python
map can also cost more memory than dense arrays when most voxels are observed.

The existing 1,000,000-logical-voxel cap remains because traversal is still
dense and current surface and mesh diagnostics are bounded reference
consumers.

## Safety

- Invalid parameters and replay-digest mismatches fail before allocating sparse
  accumulator state.
- No output is written when no selected frame can observe the volume.
- Existing targets are never overwritten.
- Publication uses the same race-safe staging and hard-link contract as the
  dense reference.

## Explicitly deferred

- automatic-bounds wiring for the sparse backend;
- consuming the candidate-block plan during fusion, frustum culling, or
  ray-centric traversal;
- logical volumes larger than 1,000,000 voxels;
- a block-backed fused TSDF value/weight artifact and sparse-aware surface or
  mesh traversal;
- Open3D, GPU, parallel, adaptive-resolution, or submap backends;
- performance or full-sequence scalability claims;
- robust outlier filtering, color fusion, and sensor-dependent weighting; and
- production meshing, structural extraction, pose estimation, and SLAM.
