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

## Read-only signed addressing

The first topology operation over this storage is:

```python
address = locate_tsdf_voxel(storage, global_index_xyz)
```

For a valid signed global voxel index in a planned block, it returns a frozen
address containing the canonical block row, local XYZ coordinate, `(block, z,
y, x)` array index, local flat index, and storage flat index. A valid index
whose block is not planned returns `None`; addressing never inserts or
allocates a missing block.

The inverse integer operation is:

```python
global_index_xyz = compose_tsdf_global_voxel_index(
    block_index_xyz,
    local_index_xyz,
)
```

Both operations leave block coordinates and numeric buffers unchanged. The
full signed floor-division and range contract is enforced by
`locate_tsdf_voxel`.

## Payload bound

Each block has `8^3 = 512` voxel slots. Each slot reserves eight bytes for its
float64 sum and four bytes for its uint32 weight:

```text
bytes per block = 512 * (8 + 4) = 6,144
```

The reference allocator permits as many blocks as the planner will plan:
250,000, which is `1,536,000,000` bytes of numeric payload. The two limits are
one number, so a plan the planner accepts is never refused here for its
size. It checks this bound before calling
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

## One-selected-block consumer

The first complete-row mutation consumer is:

```python
receipt = traverse_tsdf_block_voxels_from_context(
    storage,
    block_index_xyz,
    context,
)
```

It selects one already allocated canonical block row, requires all 512 of that
row's sum/weight slots to be canonically empty, and traverses every local-flat
address from `0` through `511` in X-fastest order. The operation composes the
existing context one-voxel traversal and retains one frozen child receipt for
every address.

The selected row is the only mutation target. Other rows may already contain
valid accumulator state and must remain unchanged. A caught failure restores
the complete selected row from its exact 6,144-byte numeric starting payload.
The temporary storage is still process-local and is discarded by the CLI; the
operation neither creates missing blocks nor persists an artifact.

## Complete existing-plan consumer

The next composition consumes every row already in the source plan:

```python
receipt = traverse_tsdf_plan_blocks_from_context(
    storage,
    context,
)
```

It requires `storage.block_indices == plan.active_blocks`, traverses rows from
zero through `block_count - 1` in canonical X-fastest plan order, and retains
one complete block receipt per row. The complete numeric storage must contain
canonical all-zero bytes before the first child. The diagnostic rejects more
than 262,144 planned voxel/selected-observation outcomes before mutation.

Because preflight proves the starting payload is all zero, caught-failure
rollback fills the existing arrays with canonical float64 and uint32 zero and
verifies their bytes and layout. It does not allocate a second whole-storage
copy. The operation still creates no missing block, changes no block tuple,
and writes no artifact.

## Explicitly deferred

- allocator-time decoding, back-projection, block-coordinate regeneration, or
  TSDF mutation; these remain separate consumer operations;
- normalized storage or complete spatial-domain fusion;
- camera-to-surface free-space, frustum, ray, visibility, or occlusion rules;
- a mutable lookup cache or fusion index, per-block provenance, color,
  confidence, or normals;
- persistence, checkpointing, a block-backed `.sftsdf`, or any other output
  artifact;
- eviction, streaming, submaps, adaptive resolution, Open3D, GPU, or parallel
  storage;
- sparse-aware surface extraction, meshing, and full-sequence scalability or
  performance claims;
- artifact signatures or authentication; and
- structure, pose estimation, SLAM, semantics, localization, and map-package
  export.
