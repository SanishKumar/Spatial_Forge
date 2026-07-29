# Empty in-memory TSDF block storage

This checkpoint turns a strict-loaded candidate plan into temporary numeric
storage without decoding depth or fusing a TSDF:

```powershell
.\.venv\Scripts\python.exe -m spatialforge reconstruct tsdf-block-allocate `
  outputs/progress-blocks.sftplan `
  tests/fixtures/minimal.vgsession
```

The command writes no output artifact. The allocated arrays are discarded when
the process exits.

## API and precondition

The public API is:

```python
allocate_empty_tsdf_blocks(plan, session)
```

`plan` must be an immutable `TsdfBlockPlan` produced by the strict loader. The
allocator preflights its canonical active-block topology and numeric payload,
then replay-verifies it against `session` before allocating arrays. Replay
verification reads and hashes referenced sensor payload bytes, but neither the
verifier nor allocator decodes depth pixels or back-projects them.

## Numeric layout

For `N` active candidate coordinates, `TsdfBlockStorage` contains two
C-contiguous arrays:

```text
tsdf_sums  shape=(N, 8, 8, 8)  dtype=float64
weights    shape=(N, 8, 8, 8)  dtype=uint32
```

The axes are `(block, z, y, x)`, so local X is fastest. Block rows follow the
plan's canonical X-fastest block-coordinate order exactly.

Every value is initialized to zero. A zero weight means the voxel is unknown;
a zero sum with zero weight is not an observed TSDF zero crossing. The storage
object keeps the immutable source plan and block-coordinate tuple, while the
numeric buffers are intentionally mutable working state for a later fusion
checkpoint.

## Payload bound

Each block has `8^3 = 512` voxel slots. Each slot reserves eight bytes for its
float64 sum and four bytes for its uint32 weight:

```text
bytes per block = 512 * (8 + 4) = 6,144
```

The reference allocator permits at most `64 MiB` (`67,108,864` bytes) of
numeric payload, or 10,922 complete blocks. It checks this bound before calling
NumPy. Reported payload bytes are the arrays' numeric `nbytes`; they do not
claim to measure Python objects, NumPy headers, allocator overhead, or process
resident memory.

## Exact fixture proof

The committed fixture plan has eight active block coordinates:

```text
allocation: blocks=8 resolution=8 voxel_slots=4096
layout: shape=(8, 8, 8, 8) axes=block-z-y-x x_fastest=yes
dtypes: tsdf_sums=float64 weights=uint32
zero_state: nonzero_sums=0 nonzero_weights=0 unknown_voxels=4096
payload_bytes: tsdf_sums=32768 weights=16384 total=49152
block_rows: first=(0, -1, -1) last=(1, 0, 0)
```

The command also reports:

```text
artifact: valid
session_replay: matched
depth_decoded: no
geometry_recomputed: no
fusion_performed: no
artifact_written: no
```

These lines demonstrate the narrow checkpoint: deterministic block order,
exact shape and dtypes, a completely unknown zero state, and bounded numeric
allocation from the replay-matched plan.

## Explicitly deferred

- decoding depth, back-projecting samples, or regenerating block coordinates;
- TSDF observations, sums, weight increments, normalization, or fusion;
- camera-to-surface free-space, frustum, ray, visibility, or occlusion rules;
- a lookup index for fusion, per-block provenance, color, confidence, or
  normals;
- persistence, checkpointing, a block-backed `.sftsdf`, or any other output
  artifact;
- eviction, streaming, submaps, adaptive resolution, Open3D, GPU, or parallel
  storage;
- sparse-aware surface extraction, meshing, and full-sequence scalability or
  performance claims;
- artifact signatures or authentication; and
- structure, pose estimation, SLAM, semantics, localization, and map-package
  export.
