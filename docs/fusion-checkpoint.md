# Stopping a fusion and continuing it

Fusion is the long step: 211 s for the living room at 10 mm, and longer for
anything larger. It used to run from the first frame to the last or not at
all, so an interrupted run started again from nothing.

```powershell
python -m spatialforge reconstruct tsdf-block-volume `
  scan.sftplan scan.vgsession scan.sftvol --checkpoint scan.sftckpt
```

With `--checkpoint`, progress is saved every 100 selected frames, or as
often as `--checkpoint-every` says. If the run is interrupted, run the same
command again. It finds the checkpoint, checks that it belongs to this plan
and this scan, and continues from it. Once the volume has been written the
checkpoint is removed.

The volume is the same either way. Not equivalent: the same bytes, under the
same digest.

## Why the bytes are the same

A voxel's sum is one addition per observation, in observation order, and
that order is what fixes the last bits of a float64. Streaming fusion walks
frames in its outer loop, so stopping between two frames interrupts no
voxel part way through its additions. Whichever run makes them, they are
the same additions in the same order.

The accumulators are stored as they are held: float64 sums and integer
weights, the same representation a [`.sftvol`](sparse-volume-format.md)
uses. Nothing is rounded on the way to disk or back.

What has to cross the gap besides the accumulators is small:

- how many of the plan's selected observations are behind it, counted
  whether a frame was fused or skipped for a missing depth image or pose;
- the counts those observations produced, one per outcome, and the valid
  and invalid depth samples read;
- the digests of the two accumulator arrays at that point.

That is the progress record. A stage of fusion takes one and returns the
next, and refuses storage whose bytes are not the ones its record
describes.

## What was checked

- **Any cut.** A 20-frame room scan fused in nine different sets of stages,
  from one frame at a time to a single stage longer than the scan. Each
  must end on the accumulators and the receipt of one uninterrupted pass.
- **The state in between.** After one, two and three frames the staged
  accumulators are compared with the observation-ledger path, which
  reaches "the first k frames" by another route: frames held in a context,
  rows fused one block at a time.
- **Through a file, twice.** Seven frames, saved, loaded into fresh
  storage, six more, saved over the first, loaded again, finished. The
  `.sftvol` is compared byte for byte with the uninterrupted one.
- **The command.** Stopped on purpose with `--stop-after`; failed in the
  middle of a stage; failed at the last step, writing the volume. Run
  again each time, and the volume compared.
- **On the real room.** ICL-NUIM `lr kt2` at 20 mm, 440 frames. The
  process was killed after its second save, 80 frames in, and the command
  run again. The volume it wrote has the digest
  `f93e362a…68018078`, which is the one the
  [published result](../results/icl-nuim-lr-kt2-20mm-surface.json) was
  measured on.

## The file

```text
offset  size        content
0       8           magic  "SFTCKP01"
8       8           header length H, unsigned 64-bit little-endian
16      H           header: canonical ASCII JSON, newline terminated
16+H    4096 N      TSDF sums   float64 [N, 8, 8, 8]
        2048 N      weights     uint32  [N, 8, 8, 8]
```

```json
{
  "schema": "spatialforge.tsdf-fusion-checkpoint",
  "schema_version": "0.1.0",
  "session_id": "…",
  "replay_digest_sha256": "…",
  "source_plan_digest_sha256": "…",
  "progress": { "frame_stride": 2, "total_observations": 880,
                "selected_observations": 440, "processed_observations": 80,
                "fused_observations": 80, "skipped_missing_depth": 0,
                "skipped_missing_pose": 0, "valid_depth_samples": 24576000,
                "invalid_depth_samples": 0,
                "status_counts": { "contributes": "…", "behind-truncation": "…" } },
  "payload": { "byte_order": "little", "block_count": 7655,
               "voxels_per_block": 512, "layout": ["…"],
               "block_indices_sha256": "…", "tsdf_sums_sha256": "…",
               "weights_sha256": "…" }
}
```

Block indices are the plan's and are not repeated. Their digest is, which
is enough to refuse storage laid out any other way. The file holds no time,
path or host name, so the same state is the same bytes.

## What is refused

A checkpoint is held to the standard of a volume, because continuing from a
wrong one would write a wrong volume under a plausible digest.

- **A file that is not one.** Wrong magic, a truncated header or payload,
  trailing bytes, a header that is not JSON of exactly the expected fields
  in its one canonical encoding, or an array whose digest does not match.
- **Progress the payload denies.** With honest digests the loader still
  recounts. Total weight must equal the contributions the record claims; no
  voxel may have been seen more often than frames were fused; sums must be
  finite and inside their weight's envelope; unseen voxels must be exactly
  positive zero.
- **A record that could not be true.** More observations processed than
  selected, outcome counts that do not cover every voxel of every frame
  processed, skipped frames nothing accounts for.
- **Somebody else's.** A checkpoint of another plan, another scan, or the
  same plan at another stride is refused before anything is fused, and left
  as it was.

A file at the checkpoint path that is not a checkpoint of this fusion is
never written over. It is the one artifact here meant to be replaced, by
the next save of the same fusion, and replacement is atomic: the new file
is written beside the old one and moved over it, so an interruption during
a save leaves the previous checkpoint intact.

## What it does not do

- **It does not add frames.** A plan is made from a fixed selection of
  frames and a fixed scan. A checkpoint continues that fusion. New frames
  are a new scan digest, a new plan and a new fusion.
- **It is not a volume.** A checkpoint cannot be meshed or scored. It has
  no place in the artifact chain and no result refers to one.
- **One run at a time.** Two runs given the same checkpoint path at once
  are not guarded against each other.
- **A stage that fails is lost.** The frames since the last save are fused
  again on the next run. Saving more often costs one hash and one write of
  the accumulators each time: 6,144 bytes per block, 47 MB for the room at
  20 mm and at most 614 MB.

`--stop-after N` stops deliberately once N more selected frames have been
processed, leaving the checkpoint and no volume. It exists for running a
long fusion in slots, and it is how the tests stop a run at a known frame.
