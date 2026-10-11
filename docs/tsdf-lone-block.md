# Fusing one block alone

A fused volume is tens of thousands of blocks. Until now the only way to
find out whether one of them is what the scan gives was to fuse all of them
again.

```powershell
python -m spatialforge reconstruct tsdf-block-check `
  scan.sftvol scan.vgsession --sample 200
```

The command draws blocks of the volume, fuses each by itself from the scan,
and requires the bytes the volume holds. It reads both files and writes
nothing. It exits 0 if no checked block differs and 1 if one does.

This is the first step of [a map that can be checked](verifiable-maps.md).

## Why a block can be fused alone

A voxel's accumulators are the sum, in frame order, of one contribution
from each frame, and a contribution reads one pixel of that frame's depth.
Nothing in it depends on another voxel or on which blocks were planned. So
a block is a function of the scan and three numbers: voxel size, truncation
and frame stride.

`fuse_tsdf_block_alone` evaluates that function. It takes a session, a
block index and the three numbers. It consults no plan and allocates no
storage, and the block need not belong to any volume.

**Frames are ruled out by pose.** Fusion already settles a whole block as
behind the camera or outside the image from its eight corners and a frame's
pose, and promises that every voxel of such a block would have been given
that verdict one at a time. The same verdicts are used here. A frame that
can reach none of the blocks asked for is never decoded.

**Several at once.** `fuse_tsdf_blocks_alone` reads each frame once for
all the blocks asked for. They share the reading and nothing else: what one
of them comes to does not depend on which others were asked for.

## Checking a volume

`check_tsdf_volume_blocks` fuses blocks alone with the volume's own three
numbers and compares bits, so a negative zero is a difference. Each block
comes back as one of:

| | |
|---|---|
| `identical` | the volume holds what the scan gives |
| `different` | it does not, in this many voxels |
| `absent` | the volume has no such block |

An absent block is unknown space, not a difference, and the check still
reports what the scan would have put there. Leaving a block out can hide
what was seen. It cannot make this check pass something false.

The session has to be the scan the volume records, by replay digest, and a
session that is not is refused before a frame is opened.

**The draw.** `--sample N` picks blocks by the SHA-256 of `--seed` and each
block's index. The same seed picks the same blocks on every machine, and a
block's chance does not depend on where the volume lists it. `--block X Y Z`
names one instead, and the two can be combined.

## What it catches that the file cannot

A `.sftvol` carries digests of its own payload, and its loader re-derives
what it can: total weight, heaviest voxel, sums inside their envelope. All
of that says the file is intact. None of it says the file is the scan's.

The test builds the difference. One observed voxel of the fixture volume
has its sum turned from in front of the surface to behind it, and the
file's digests are recomputed to match. The loader accepts it: every check
it makes still holds. The block check does not:

```text
block[0]: index=(0, -1, -1) status=different frames_seeing=2 frames_contributing=2 observed_voxels=44 contributions=88 differing_voxels=1
...
identical: 7
different: 1
result: 1 of 8 checked blocks differ from what the scan gives
```

Asked about that one block alone, it says the same.

## What it is pinned to

- **Streaming fusion.** Every block of the committed fixture, of five
  rewritten fixtures at two strides, and all 351 blocks of the room scan,
  each fused alone and compared byte for byte, with frames ruled out by
  pose and without.
- **The voxel-by-voxel traversal** on the committed fixture, which shares
  no vectorised code with either.
- **The numbers everything agrees on.** Summed over the fixture's eight
  blocks, fused one at a time by a path that never saw the plan: 1,168
  contributions over 584 voxels. Over the room's 351: 1,127,112 over
  81,292.
- **Blocks no plan held.** The 173 blocks that free-space expansion adds to
  the room, fused alone with no reference to the expanded plan, are the
  bytes the expanded plan fuses.
- **Frames with no depth or no pose** are skipped as fusion skips them.
- **Frames not opened.** For the block of the room that fewest frames can
  reach, the depth decoder is called once per frame that can reach it and
  for no other, and the accumulators are the ones every frame gives.

## On real volumes

Two hundred blocks drawn with seed 0 from each of three published volumes,
whose digests are the published ones:

| Volume | Blocks | Frames | Checked | Identical | Frames that can reach a block | Whole check | Fusing the volume |
|---|---|---|---|---|---|---|---|
| Kinect desk, 15 mm | 5,401 | 395 | 200 | 200 | 2 to 395, median 104 | 5.5 s | 79 s |
| Living room, 10 mm | 27,965 | 440 | 200 | 200 | 8 to 250, median 81 | 6.0 s | 211 s |
| Kinect office with free space, 15 mm | 69,415 | 676 | 200 | 200 | 7 to 289, median 53 | 8.9 s | 460 s |

Checking one block of the office takes 5.6 s, which is not the block. It
is loading and verifying a 427 MB volume and hashing every frame of the
scan for its replay digest. Evaluating the block is hundredths of a second.
Removing that fixed cost is what the next steps are for: a check that needs
neither the whole volume nor the whole scan.

(Timed once, on one laptop. The office's 460 s was taken with other work
running on it.)

## What it does not do

- **It does not need less of the scan yet.** A frame that can reach a block
  is decoded whole, though the block reads a few hundred of its pixels,
  and the session's replay digest still hashes every file.
- **It commits to nothing.** There is no hash a third party could hold, and
  no proof to hand over. The check is run by someone who has the volume and
  the scan.
- **A sample is a sample.** Two hundred identical blocks of 69,415 say that
  a volume is not wrong in bulk. They say little about one moved wall.
  Naming the blocks a claim depends on is how that is checked, and is a
  later step.
- **It says nothing about the scan.** A fabricated depth frame gives a
  volume that passes. So do wrong poses.
- **It says nothing about accuracy.** A block that is identical is the
  block this engine makes, error and all.
- **At most 4,096 blocks a call.** Past that, fuse the volume.
