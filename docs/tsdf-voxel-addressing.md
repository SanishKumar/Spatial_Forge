# Signed global TSDF voxel addressing

This checkpoint resolves signed integer voxel coordinates inside existing
`TsdfBlockStorage`. It does not create blocks or change TSDF state.

## Public API

```python
address = locate_tsdf_voxel(storage, global_index_xyz)

global_index_xyz = compose_tsdf_global_voxel_index(
    block_index_xyz,
    local_index_xyz,
)
```

`locate_tsdf_voxel` returns a frozen `TsdfVoxelAddress` when the corresponding
block row exists. It returns `None` for a valid global coordinate whose block
is not present in the planned storage. Invalid types or out-of-range
coordinates raise `TsdfError`; a valid sparse miss is not an error.

The address records:

- signed `global_index_xyz`;
- signed `block_index_xyz`;
- local `local_index_xyz`, with every component in `[0, 7]`;
- the canonical `block_row`;
- `array_index_bzyx` for `(block, z, y, x)` NumPy access;
- X-fastest `local_flat_index`; and
- `storage_flat_index` across every block row.

## Signed split and inverse

For each axis with block resolution `R = 8`:

```text
block = floor(global / R)
local = global - block * R

global = block * R + local
```

Python's integer `divmod` implements the required mathematical floor behavior:

| Global | Block | Local |
|---:|---:|---:|
| -9 | -2 | 7 |
| -8 | -1 | 0 |
| -1 | -1 | 7 |
| 0 | 0 | 0 |
| 7 | 0 | 7 |
| 8 | 1 | 0 |

`compose_tsdf_global_voxel_index` is the inverse for a signed 32-bit block
coordinate and a local coordinate in `[0, 7]`.

The supported inclusive global range on every axis is:

```text
minimum = -17,179,869,184
maximum =  17,179,869,183
```

These endpoints correspond to signed 32-bit block indices with eight local
voxels per axis. This checkpoint accepts integer voxel indices only; it does
not quantize world-space floating-point coordinates.

## Array and flat order

For local `(lx, ly, lz)`:

```text
local_flat = (lz * 8 + ly) * 8 + lx
storage_flat = block_row * 512 + local_flat
array_index_bzyx = (block_row, lz, ly, lx)
```

Block rows are found by binary search over the immutable canonical block tuple,
ordered X-fastest, then Y, then Z. Addressing builds no mutable lookup cache.

## Exact fixture proof

Run three queries over the committed fixture plan:

```powershell
.\.venv\Scripts\python.exe -m spatialforge reconstruct tsdf-block-address `
  outputs/progress-blocks.sftplan `
  tests/fixtures/minimal.vgsession `
  --voxel 7 -1 -1 `
  --voxel 8 0 0 `
  --voxel -1 0 0
```

The relevant output is:

```text
storage_mutated: no
addressing_created_blocks: no
artifact_written: no
allocation: blocks=8 voxel_slots=4096
queries: requested=3 resolved=2 unplanned=1
voxel[0]: status=planned global=(7, -1, -1) block=(0, -1, -1) local=(7, 7, 7) row=0 array=(0, 7, 7, 7) local_flat=511 storage_flat=511
voxel[1]: status=planned global=(8, 0, 0) block=(1, 0, 0) local=(0, 0, 0) row=7 array=(7, 0, 0, 0) local_flat=0 storage_flat=3584
voxel[2]: status=unplanned global=(-1, 0, 0)
```

The first query demonstrates negative floor behavior and the final local slot
of row zero. The second crosses a positive block boundary and reaches the first
slot of row seven. The third is within the supported integer range but belongs
to an unplanned block; `locate_tsdf_voxel` returns `None` without allocating
it.

The CLI strict-loads and replay-verifies the plan, allocates the already
planned empty storage using the previous checkpoint, resolves the queries, and
then discards the storage. The addressing operation itself creates no block and
the command writes no artifact.

## Explicitly deferred

- dynamic block creation, insertion, eviction, or a mutable lookup cache;
- using TSDF sums or weights as geometry, changing their values, depth decoding,
  back-projection, replanning, fusion, or free-space decisions;
- world-coordinate quantization, interpolation, neighbor/stencil iteration,
  frustum or ray traversal, visibility, and occlusion;
- a dense global flat index or finite global volume AABB;
- persistence, checkpointing, `.sftsdf` output, or any other artifact;
- authentication, signatures, or trusted-geometry claims;
- GPU, parallel, adaptive, submap, full-sequence, or performance claims; and
- meshing, structure, pose estimation, SLAM, semantics, localization, and
  map-package export.
