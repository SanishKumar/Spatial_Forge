# The `.sftvol` sparse volume format

A `.sftvol` file is one fused block plan on disk. It exists so that a
reconstruction can leave the process that made it: be scored, meshed and
rendered as the same artifact, and be recognised later by its digest.

```powershell
python -m spatialforge reconstruct tsdf-block-volume `
  scan.sftplan scan.vgsession scan.sftvol
```

## Layout

```text
offset  size        content
0       8           magic  "SFTVOL01"
8       8           header length H, unsigned 64-bit little-endian
16      H           header: canonical ASCII JSON, newline terminated
16+H    12 N        block indices   int32   [N, 3]         (x, y, z)
        4096 N      TSDF sums       float64 [N, 8, 8, 8]   (z, y, x)
        2048 N      weights         uint32  [N, 8, 8, 8]   (z, y, x)
```

Everything is little-endian. `N` is the number of blocks. Blocks appear in
strictly increasing `(z, y, x)` order, so a set of blocks has exactly one
encoding.

The world position of a voxel is not stored; it follows from the grid. Block
`(bx, by, bz)` holds global voxel indices `8 * b + local`, and the centre of
global voxel `g` is `(g + 0.5) * voxel_size_m` on each axis, with the grid
anchored at the world origin.

## Sums and weights, not values

The file stores the accumulators exactly as fusion left them: the float64 sum
of accepted truncated distances and the count of observations that
contributed. The TSDF value of a voxel is `sum / weight`; a voxel with weight
zero was never observed and has no value.

Storing the quotient instead would be smaller to explain and would throw away
the property the rest of the project depends on. With the accumulators on
disk, loading a volume returns the exact bytes fusion produced, and two
volumes can be compared, or a fusion re-run and checked, without a rounding
step in between.

## The header

```json
{
  "schema": "spatialforge.tsdf-block-volume",
  "schema_version": "0.1.0",
  "session_id": "…",
  "replay_digest_sha256": "…",
  "source_plan_digest_sha256": "…",
  "grid":   { "world_anchor_m": [0.0, 0.0, 0.0], "voxel_size_m": 0.015,
              "block_resolution": 8, "index_order": "…", "block_order": "…" },
  "tsdf":   { "truncation_m": 0.045, "sign": "…", "normalization": "…",
              "unknown": "weight-zero" },
  "fusion": { "frame_stride": 2, "total_observations": 792,
              "selected_observations": 396, "fused_observations": 395,
              "skipped_missing_depth": 0, "skipped_missing_pose": 1,
              "contributions_evaluated": 1095063552,
              "contributions_applied": 66200108,
              "observed_voxels": 1492894, "maximum_weight": 391 },
  "payload": { "byte_order": "little", "block_count": 5401,
               "voxels_per_block": 512, "layout": ["…"],
               "block_indices_sha256": "…", "tsdf_sums_sha256": "…",
               "weights_sha256": "…" }
}
```

It is written with sorted keys and a fixed indent. The two provenance digests
tie the volume to the scan it was fused from and the plan that chose its
blocks. The digest of the whole file is the volume's identity; a mesh records
it, and so does a result manifest.

## What the writer refuses

`write_tsdf_block_volume` takes storage and the receipt of the fusion that
filled it. It re-digests the arrays and refuses to write if they no longer
match the receipt, so storage that was modified after fusion cannot be
persisted as though it were the fusion's result. It refuses a receipt for a
different plan, an existing output, and any suffix other than `.sftvol`, and
it publishes by linking a fully written temporary file into place.

## What the loader refuses

`load_tsdf_block_volume` returns immutable arrays or raises. It does not
trust anything it can check:

- **The container.** Wrong magic, a header length out of range, a file
  truncated inside the header, or a payload whose size is not exactly what
  the header's block count implies.
- **The header.** Not ASCII, not valid JSON, a duplicate key, a non-finite
  number, an unknown or missing field, a wrong type, or nesting deeper than
  the format has. It must also be the *canonical* encoding of its own
  contents, so that two different files cannot describe one volume.
- **Each array's digest.**
- **The payload against the header.** With digests that match, the loader
  still recounts: observed voxels, maximum weight, and the sum of all weights
  against `contributions_applied`. It checks that every sum is finite and
  within its weight's envelope, that unobserved voxels are exactly positive
  zero, and that blocks are unique and in order.

The last group is what makes a forgery expensive. Editing the payload and
recomputing its digest is easy; editing it so that every count the header
claims still follows from it is the same as producing a different valid
volume. The tests build such forgeries — consistently re-digested, each
wrong in one way — and require each to be refused.

## Limits

A volume holds at most 250,000 blocks, the most the planner will plan and
the most fusion will allocate: 1.5 GB of accumulators. The whole file is
read into memory to be verified. There is no partial or memory-mapped
load, no compression, and no way to append to a volume: it is written
once, complete.

The ceiling is set by memory, and what it costs to handle a volume is
one copy of it. The writer streams the accumulators from the storage
they were fused in, hashing as it goes. The loader reads the file once,
hands out views of what it read, and verifies the payload a run of
blocks at a time. A test measures both with `tracemalloc`: writing
allocates under a tenth of the payload, loading the file and under a
quarter of the payload more. Before that was so, writing took twice the
payload on top of the storage, and the ceiling was 100,000 blocks.

Measured on the largest volume here, the living room with its free space
at 10 mm, 122,688 blocks and 0.70 GiB of accumulators:

```text
fuse, with a checkpoint every 100 frames    0.90 GiB peak
mesh, 7.2 million triangles                 4.27 GiB peak
```

Fusion peaked at 1.30 GiB before its end-of-stage check was made a run of
blocks at a time, and writes the same volume, digest for digest, either
way. Meshing is the expensive step, and not because of blocks: the same
triangles cost 3.33 GiB from the 27,965-block volume without free space.
That is about 10 kB a block, and a little over 3 GiB for the mesh itself.

A fusion that is interrupted before its volume is written can be continued,
from a separate file that is not a volume:
[`fusion-checkpoint.md`](fusion-checkpoint.md).
