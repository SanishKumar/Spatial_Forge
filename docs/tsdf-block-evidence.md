# Fusing a block from the pixels it reads

[Fusing a block alone](tsdf-lone-block.md) still opens every frame that can
reach it. A frame is 307,200 depth samples, and a block reads a few hundred
of them: each of its 512 voxels reads one pixel.

```powershell
python -m spatialforge reconstruct tsdf-block-check `
  scan.sftvol scan.vgsession --sample 200 --tiles 16
```

With `--tiles`, each block drawn is fused from the depth tiles it reads and
from nothing else of the scan's depth, and the report says how much depth
that was.

This is the second step of [a map that can be checked](verifiable-maps.md).

## What a block is a function of

Small things, and all of them are now gathered into one value, the block's
*evidence*:

- the camera: image size, focal lengths, principal point, and the depth
  scale;
- every selected frame's place in the scan, its pose if it has one, and
  whether it has depth at all;
- for the frames that can reach the block, the tiles of depth that hold a
  pixel some voxel of the block reads.

The poses of all the frames are there, including the many that give the
block nothing. That they give it nothing is decided from their pose, and
whoever checks the block has to be able to decide it too.

Depth is cut into tiles, squares of pixels on a fixed grid from the top
left of the image, because a tile is a thing a hash can be given to. It is
kept as the 16-bit samples the scan holds, not as metres.

## Two halves

`gather_tsdf_block_evidence` reads a session and returns the evidence for a
block. `fuse_tsdf_block_from_evidence` takes the evidence and returns the
block. The second reads no session and opens no file, and a test patches
the decoder, the replay and both ways of opening a file to prove it.

It also takes nothing on trust that it can work out. Frames are ruled out
from their poses again, not from which of them arrived with tiles.

## One account of which pixel

Which pixels a block reads could have been worked out by a second piece of
code, and then two pieces of code would have to agree for ever. It is not.

The voxel evaluator touches depth in exactly one place, a gather of one
pixel a voxel. That gather is now behind a reader. A whole frame is one
reader, and is the one fusion has always used; its bytes are unchanged, and
the digests pinned across the suite are what say so.

- To gather, the evaluator is run with a reader that notes the pixels it is
  asked for.
- To fuse from evidence, the same evaluator is run with a reader over the
  tiles, which refuses a pixel it does not hold.

So a block fused from evidence is the block fused alone, to the byte, or it
is an error that names a frame and a pixel:

```text
observation 6: the block reads the depth pixel at row 31, column 31, and the evidence holds no tile for it
```

It is never a block fused from a guess. Depth that was not given is not
taken for empty space.

## What it is pinned to

- **The block fused alone.** Every block of the committed fixture, and all
  351 blocks of the room scan at four tile sizes, fused from evidence and
  compared with the first step: the accumulators byte for byte, and every
  count of what it took.

  | Tile | Tiles kept, 351 blocks | Depth samples in them |
  |---|---|---|
  | 1 pixel | 294,483 | 294,483 |
  | 4 pixels | 33,745 | 539,920 |
  | 16 pixels | 8,614 | 2,205,184 |
  | 64 pixels, the image's width | 5,136 | 15,777,792 |

  The first row counts the pixels the room's blocks read. The last keeps
  whole frames, 5,136 of the 7,020 a block and a frame can make.
- **A second account of the pixels.** For 27 blocks and all 20 frames, the
  pixels kept at one pixel to a tile are the pixels that plain arithmetic
  in the test finds, one voxel at a time, sharing no code with the
  evaluator.
- **Every tile is needed.** For 15 blocks at 8-pixel tiles, each of their
  687 tiles is taken away in turn. Every time, the fusion refuses and names
  a pixel in the frame the tile came from.
- **Blocks no plan held.** The 173 blocks expansion adds to the room, from
  their evidence, are the expanded plan's bytes.
- **Frames with no depth or no pose** carry no tiles and are skipped as
  fusion skips them.
- **Scaling.** A tile cut from a frame scales to the same metres, bit for
  bit, as those pixels of the frame scaled whole.
- **The volume check** says the same of every block from tiles as from
  whole frames, on a genuine volume and on the forged one.

## On real volumes

Two hundred blocks drawn with seed 0 from each of three published volumes,
fused from 16-pixel tiles:

| Volume | Identical | Frames read, median | Depth a block, median | Mean | Most | The scan's depth | Scan over median |
|---|---|---|---|---|---|---|---|
| Kinect desk, 15 mm | 200 of 200 | 103 of 395 | 327 kB | 694 kB | 6.5 MB | 243 MB | 743 |
| Living room, 10 mm | 200 of 200 | 76 of 440 | 120 kB | 150 kB | 0.8 MB | 270 MB | 2,247 |
| Kinect office with free space, 15 mm | 200 of 200 | 30 of 676 | 112 kB | 274 kB | 2.1 MB | 415 MB | 3,721 |

A block of the office is fused from one part in 3,700 of the depth the
scan holds, at the median. The desk's blocks need more because the camera
never leaves the desk: most frames see most blocks.

The tile size is a trade. On the office, the same 200 blocks:

| Tile | Tiles a block, median | Depth a block, median |
|---|---|---|
| 8 pixels | 554 | 71 kB |
| 16 pixels | 218 | 112 kB |
| 32 pixels | 115 | 236 kB |

Smaller tiles carry less depth and there are more of them, and each will
need its own path to a hash. Which size is cheapest once the paths are
counted is for the next step to measure.

(Measured once, on one laptop.)

## What it does not do

- **Evidence is not proof.** Nothing in it says its depth is the scan's. A
  test blanks every sample in a block's evidence and gets a different
  block with no complaint. Binding the tiles and the poses to the scan is
  what a commitment is for, and is the next step.
- **It is slower than whole frames here.** Checking 200 blocks of the
  office takes 47 s from tiles against 9 s from frames. Gathering runs the
  evaluator once for every block and frame, and fusing from evidence rules
  out 600 frames a block one pose at a time: about 70 ms a block before
  any depth is read. Nothing in it is tuned.
- **The poses are not small beside a block.** Every selected frame's pose
  is in every block's evidence: 87 kB for the office, most of what the
  median block's depth comes to. It is the same table for every block, and
  will be committed once.
- **Evidence is a value in memory.** There is no file for it yet.
- **Depth must fit 16 bits.** A scan that holds a larger sample is refused.
