# What checking a map costs

The six steps of [a map that can be checked](verifiable-maps.md) end in
three things a checker can be handed: [a block's proof](tsdf-block-proof.md),
[a route's certificate](tsdf-route-certificate.md) and
[an audit's answer](tsdf-map-audit.md). This page is what each of them
weighs and how long it takes to check, beside doing without them: holding
the whole scan and fusing the map again.

Everything here was measured once, on one laptop, with scratch scripts
that are not part of the package. Sizes are exact and times are not.

## Beside rebuilding

| | Kinect desk, 15 mm | Living room, 10 mm | Kinect office with free space, 15 mm |
|---|---|---|---|
| frames fused | 396 | 440 | 676 |
| blocks | 5,401 | 27,965 | 69,415 |
| depth the volume was fused from | 243 MB | 270 MB | 415 MB |
| the volume | 33 MB | 172 MB | 427 MB |
| fusing it | 79 s | 211 s | 460 s |
| **the statement** | 800 bytes | 797 bytes | 803 bytes |
| **one block's proof**, median of 200 | 535 kB | 323 kB | 382 kB |
| checking it | 0.11 s | 0.08 s | 0.10 s |
| **an audit of 459 blocks** | 174 MB | 73 MB | 110 MB |
| checking it | 5.8 s | 2.8 s | 3.0 s |
| **a 3.5 m route**, 0.135 m wide | | | 84 MB |
| checking it | | | 3 to 5 s |

The fusing times are the ones each volume's own page records.

**Time is saved every time.** A block is checked in a tenth of a second.
An audit that would catch a map one part in a hundred wrong is checked in
3 to 6 s, where fusing takes 79 to 460 s and needs the planner run first.

**Bytes are saved for a block and much less for many blocks.** One block
needs a thousandth of the office's depth. But 459 blocks drawn across a
map are read, between them, by every frame: the office's audit holds
tiles of all 676 frames and comes to a quarter of the scan's depth. On
the desk, where the camera never leaves and every frame sees most of the
volume, it comes to seven tenths. A route through the open middle of a
room is the same story, told on the route's own page. Those are counts of
raw samples on both sides, and the last section of this page says what
that leaves out.

What nothing here needs is the volume. A checker holds 800 bytes and is
sent the blocks in question, 6 kB each.

## One block, by voxel size

Two hundred blocks drawn with seed 0 from volumes of the same two scans at
different voxel sizes. The desk's coarsest volume was fused from every
eighth frame and the others from every second.

| Volume | Blocks | Frames fused | Proof, median | Mean | Largest | Depth in it, median | Check, a block |
|---|---|---|---|---|---|---|---|
| desk, 30 mm | 1,241 | 99 | 444 kB | 756 kB | 6.9 MB | 260 kB | 0.04 s |
| desk, 20 mm | 3,017 | 396 | 695 kB | 1,409 kB | 10.8 MB | 467 kB | 0.12 s |
| desk, 15 mm | 5,401 | 396 | 535 kB | 944 kB | 7.1 MB | 327 kB | 0.11 s |
| desk, 10 mm | 12,503 | 396 | 491 kB | 674 kB | 3.9 MB | 271 kB | 0.09 s |
| living room, 20 mm | 7,655 | 440 | 502 kB | 567 kB | 2.0 MB | 281 kB | 0.09 s |
| living room, 15 mm | 14,059 | 440 | 426 kB | 477 kB | 1.5 MB | 216 kB | 0.09 s |
| living room, 10 mm | 27,965 | 440 | 323 kB | 355 kB | 1.1 MB | 120 kB | 0.08 s |

A finer map has smaller proofs. A block is eight voxels across whatever
the voxel, so at 10 mm it is a third the width it is at 30 mm and lands on
fewer pixels of every frame that sees it. The frames' leaves do not
change: 140 kB of every desk proof and 156 kB of every living-room proof.

The time to check hardly moves. It is mostly ruling frames out by pose,
one frame at a time, and that depends on how many frames were fused.

## A route, by length and width

The line through the office of the route's own page, in the band from
0.80 m to 1.40 m above the floor.

| Of the line, 9 voxels wide | Blocks | Frames read | Tiles | Certificate | Checked |
|---|---|---|---|---|---|
| the first 34 columns, half a metre | 72 | 228 | 21,641 | 12 MB | 1.0 s |
| the first 67, a little over 1 m | 150 | 233 | 31,274 | 18 MB | 1.6 s |
| the first 134, about 2.3 m | 282 | 247 | 54,795 | 31 MB | 2.7 s |
| all 209, 3.54 m | 420 | 330 | 154,532 | 84 MB | 4.9 s |

| The whole line, 3.54 m | Blocks | Frames read | Certificate | Checked |
|---|---|---|---|---|
| 1 voxel wide | 204 | 304 | 67 MB | 1.9 s |
| 9 voxels wide | 420 | 330 | 84 MB | 3.2 s |
| 21 voxels wide | 732 | 352 | 101 MB | 6.0 s |
| 21 wide, 2 voxels high | 122 | 302 | 57 MB | 1.3 s |

Half a metre of route already needs a third of the scan's frames, each
for under a tenth of its tiles. After that a certificate grows with the
route, though not evenly: the last third of this line is two thirds of
its certificate. Width and height add little.

The two tables were measured in separate runs. The certificate that is in
both, the whole line 9 voxels wide, was checked in 4.9 s in one and 3.2 s
in the other, which is how far to trust the times.

A line of the same width 1.45 m long, 2.4 m above the floor where 30
frames looked, is 4.0 MB.

## An audit, by how much is drawn

| Drawn from the office's 69,415 | Answer | Frames read | Answered | Checked | Catches, 99 times in 100 |
|---|---|---|---|---|---|
| 100 blocks | 26 MB | 654 of 676 | 23 s | 0.7 s | 3,122 wrong blocks, 4.5% |
| 459 blocks | 110 MB | 676 of 676 | 27 s | 3.0 s | 691 wrong blocks, 1.0% |

A hundred blocks already reach nearly every frame, because a hundred
blocks drawn by lot are everywhere. What grows after that is how many
tiles of each frame are wanted.

## What sharing proofs saved

Until this step every depth tile carried its own path to its frame's
depth root. The tiles of a frame now share one proof.

| | One path a tile | One proof a frame |
|---|---|---|
| a block of the desk, median | 706 kB | 535 kB |
| a block of the living room | 368 kB | 323 kB |
| a block of the office | 437 kB | 382 kB |
| the office route, 1 voxel wide | 109 MB | 67 MB |
| the office route, 21 voxels wide | 163 MB | 101 MB |
| hashes in that last certificate | 62.5 MB | 0.65 MB |

A certificate is now nearly all depth. Its hashes went down by a factor
of 96, because a frame gives a certificate hundreds of tiles that lie in
a few blocks of the image.

## Smaller tiles, measured again

With a path for every tile, 16 pixels was the cheapest tile by a wide
margin. With paths that cheap, 8-pixel tiles were tried again on the
office:

| | 16 pixels | 8 pixels |
|---|---|---|
| making the statement | 18 s | 32 s |
| a block's proof, median | 382 kB | 359 kB |
| checking a block | 0.11 s | 0.15 s |
| the route, 1 voxel wide | 67 MB | 60 MB |
| the route, 21 voxels wide | 101 MB | 98 MB |
| checking that route | 6.0 s | 8.5 s |

Three to ten percent smaller, for half as long again to check and nearly
twice as long to commit. Sixteen stays.

## What these numbers are not

- **Not tuned.** Checking a block spends most of its time ruling out
  hundreds of frames one pose at a time in Python.
- **Not a network measurement.** Sizes are of the files. Nothing was sent
  anywhere.
- **Not many scans.** Two real scans and one synthetic room, one route.
- **Not compressed, and this matters.** Depth tiles are raw 16-bit
  samples, and "the scan's depth" above counts samples too. The office
  scan's own depth images are PNG files: 179 MB for all 1,352 frames,
  about 90 MB for the 676 its volume was fused from, a fifth of their
  samples. Beside those files, an audit of 110 MB or a route of 84 MB as
  written saves nothing in bytes. Put through zlib at its default level
  the 84 MB certificate is 12 MB and the 4.0 MB one is 0.8 MB. Nothing
  in the package compresses a proof.
