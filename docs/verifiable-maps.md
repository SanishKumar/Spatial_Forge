# A map that can be checked

**Status: design, with the first two of six steps built.** This page says
what is to be built, why it should work, what was measured before
deciding, and what it will not do. Each step in the plan at the end
becomes real only when it has its own tests and its own page. Two have:
[fusing one block alone](tsdf-lone-block.md), and
[fusing it from the pixels it reads](tsdf-block-evidence.md).

## The idea

A reconstructed map is usually taken on trust. Whoever receives it cannot
tell whether it follows from the scan it claims to come from, or whether a
wall was moved, an obstacle removed, or a corridor marked clear that nobody
looked down. The only check available is to have the whole scan and build
the whole map again.

This engine is repeatable to the bit. The same scan gives the same volume on
Linux, macOS and Windows, under two versions of Python, and the test suite
fails if it does not. So far that has been a standard of care. It is also
what makes a cheaper check possible:

> Publish one hash for the scan and one for the map. Anyone can then take a
> single block of the map, fetch only the parts of the scan that could have
> reached it, compute that block again, and compare.

One wrong block is a proof that the map does not follow from the scan. A
claim about the map, such as "this route is clear", can carry exactly the
blocks it rests on and be checked without the rest.

## Why it should work: three facts about the engine

**A block is a function of the scan and nothing else.** A voxel's value is
the sum, in frame order, of one contribution per frame, and each
contribution reads one pixel of that frame's depth. It does not depend on
any other voxel or on which other blocks the plan holds. The test suite
already requires that a plan grown with more blocks fuses the blocks it
shared to the same bytes.

**Most frames can be ruled out from their pose alone.** Fusion already
settles a whole block as behind the camera or outside the image from eight
corners, and those verdicts are tested to change no byte. A checker can
give the same verdicts from the camera poses, which are small, and needs
depth only from the frames that are left.

**It is the same arithmetic everywhere.** The check is equality of bytes, so
it only means something if two machines agree to the last bit. That is the
property the suite pins with literal digests on six platforms.

## Measured before deciding

On the real office of [`tum-validation.md`](tum-validation.md), expanded
into free space: 69,415 blocks from 676 frames at 15 mm. Eighty blocks
chosen at random were each fused again alone, from only the frames whose
whole-block verdict allows them to see it.

```text
recomputed alone, identical to the volume's bytes    80 of 80
frames whose verdict lets a block be seen            median 57 of 676   (10 to 312)
frames that give it anything                         median 12
depth pixels read, over all those frames             median 10,209
16 x 16 pixel tiles holding them                     median 260         (133 kB of depth)
evaluation, one block                                13.5 ms
the whole scan's depth                               415 MB
fusing the whole volume                              460 s
```

A block needs about three thousand times less depth than the scan holds
and about a hundredth of a second of evaluation. (Measured once with a
scratch script, on one volume, before anything was built.)

The tool that now exists reports the same picture on 200 blocks of the
same volume drawn by hash: all 200 identical, a median of 218 tiles and
112 kB of depth a block, one part in 3,700 of the scan's. Fusing a block
from its evidence takes about 70 ms, most of it ruling out 600 frames one
pose at a time, which the scratch script did for all its blocks at once.

## What is committed

Two Merkle trees, hashed as in
[RFC 6962](https://www.rfc-editor.org/rfc/rfc6962): a leaf and an inner node
are hashed with different prefixes, so one cannot be passed off as the
other.

**The scan.** One leaf per frame, in replay order. A frame's leaf holds the
root of a second tree over its depth image, cut into tiles of 16 by 16
pixels, as the raw samples the session decodes and not as file bytes, so
the commitment does not depend on how an image was compressed. The camera
poses are hashed together as one table, because a checker always needs all
of them.

**The map.** One leaf per block, in the canonical block order, holding the
block's index and its 6,144 bytes of sums and weights.

**The statement** binds the two roots to everything else the answer depends
on: voxel size, truncation, frame stride, the camera's intrinsics, the
depth scale, the tile size, and a name for the fusion rule. A change to the
arithmetic is a new rule name, and the pinned digests are what notice one.

## What can be checked

**One block.** Given the statement, the pose table, a block with its path
to the map root, and the depth tiles that block's voxels project into with
their paths to the scan root: rule out frames by pose, evaluate the rest in
order, compare bytes. The checker computes for itself which pixels the
block reads, and refuses a proof that leaves one out.

**That a block is absent.** Blocks are in sorted order, so two neighbouring
leaves show there is nothing between them. An absent block is unknown
space. Leaving a block out can hide what the scan saw there, and can never
make anything look clear.

**A route.** A chain of grid columns, a radius and a height band, all in
whole voxels so that which voxels are meant is not a matter of rounding.
The certificate is every block the swept volume touches. The route is
clear if every voxel in it was observed at least three times and lies in
front of every surface seen, which is the rule the
[free-space map](free-space-expansion.md) already draws.

**The whole map, by sampling.** Check blocks chosen at random. If a share
`f` of the blocks is wrong, `k` checks all pass with probability
`(1 - f)^k`: 459 checks catch a map that is 1% wrong 99 times in 100. A
single moved wall is a much smaller share than that and a random audit will
usually miss it. That is why claims carry their own blocks: a route's
certificate checks the blocks that route depends on, however few they are.

## What it does not prove

- **That the scan is true.** A fabricated depth frame gives a verified map
  of something that is not there. This binds a map to a scan. Binding a
  scan to the world is a signature from the device that recorded it, and
  the scan root is where that signature would attach.
- **That the poses are right.** They are inputs, and this engine has no
  tracker.
- **That the map is accurate.** A verified map is the map this engine
  makes from that scan, error and all. How large the error is belongs to
  the validation pages.
- **Anything without the scan.** Someone has to hold the frames and serve
  the tiles. If the scan is gone, the roots can still show that a map was
  changed, and nothing can be recomputed.
- **Privacy.** A proof contains pieces of the depth images.

## What is near it

Ten searches in October 2026, not a systematic review, found these
neighbours and nothing that verifies a dense reconstruction by recomputing
part of it against a commitment to the scan. That is an absence of
results, not a proof that nobody has done it.

- **Authenticated spatial indexes.** The
  [MR-tree](https://cse.hkust.edu.hk/faculty/dimitris/PAPERS/ICDE08-MR.pdf)
  (Yang, Papadopoulos, Papadias and Kollios, 2008) lets a client check that
  an outsourced spatial query returned the owner's records, all of them and
  unaltered. It authenticates stored data. It says nothing about how the
  data was computed.
- **Checking a computation by running part of it again.** Refereed
  delegation (Canetti, Riva and Rothblum, 2011) and the fraud proofs of
  [Truebit](https://arxiv.org/pdf/1806.11476) and optimistic rollups narrow
  a dispute to one step of a deterministic machine and run that step. The
  idea here is that one, with the block as the step: fusion is local enough
  that no search for the disputed step is needed.
- **Provenance for reconstructions.** Starling Lab's
  [verifiable radiance fields](https://starling.stanford.edu/prototypes/radiance-fields/)
  tie points of a NeRF or a Gaussian splat to the photographs behind them.
  Training those is not repeatable to the bit, so nothing can be recomputed
  and compared.
- **Integrity of shared maps.** Work on crowdsourced HD maps relies on
  agreement between contributors and on ledgers of who changed what
  ([overview](https://arxiv.org/pdf/2409.09726)), not on recomputation.
- **Zero-knowledge proofs**, which have been proposed for vehicle
  perception, prove a computation without revealing its inputs, at a cost
  per operation that the office's twenty-four billion voxel evaluations
  would not bear. Re-execution is cheap and reveals the inputs it uses.

## The plan

One step at a time, each pinned to a path that already exists.

1. **Fuse one block alone.** *Built:*
   [`tsdf-lone-block.md`](tsdf-lone-block.md). Pinned against streaming
   fusion: every block of the fixture volumes, to the byte, with and
   without ruling frames out by pose. On three published volumes, 600 of
   600 blocks drawn at random are the volume's bytes.
2. **Fuse it from tiles.** *Built:*
   [`tsdf-block-evidence.md`](tsdf-block-evidence.md). The pixels a block
   reads, from the evaluator's own arithmetic; a depth source holding only
   some tiles; refusal when a needed pixel is missing. Pinned against
   step 1 on every fixture block at four tile sizes.
3. **Commit.** The two trees and the statement, as a file. Pinned against a
   plain recomputation of the roots, and by literal digests on every
   platform.
4. **A block's proof, and a checker that has nothing else.** It reads the
   statement and the proof and no session. Then the tampering suite: a
   flipped bit, a wall moved one voxel, an obstacle removed, a depth pixel
   changed, a frame dropped, two frames exchanged, a pose nudged. Each must
   be caught, and the page will say by which check.
5. **A route's certificate**, on the office.
6. **Audit by sampling, and the measurements**: proof size and time against
   block, route length and voxel size, beside rebuilding the whole map.

## Open questions

- Whether tiles of 16 pixels are the right size, or whether the paths cost
  more than the pixels they save. On the office a block reads a median
  71 kB in 554 tiles of 8 pixels, 112 kB in 218 of 16, and 236 kB in 115
  of 32; what each tile's path adds is not yet counted.
- How much of a route's certificate is shared between its blocks. A long
  route may need most of the frames that look along it.
- Whether the mesh, which is a repeatable function of the volume, should be
  committed block by block as well.
