# A map that can be checked

**Status: all six steps built.** This page says what was to be built,
why it should work, what was measured before deciding, and what it does
not do. Each step in the plan at the end has its own tests and its own
page:
[fusing one block alone](tsdf-lone-block.md),
[fusing it from the pixels it reads](tsdf-block-evidence.md),
[committing to the scan and the map](tsdf-commitment.md),
[proving one block](tsdf-block-proof.md),
[certifying a route](tsdf-route-certificate.md), and
[auditing a map by lot](tsdf-map-audit.md). What it all costs is on
[one page](verifiable-map-costs.md), and what it came to is
[below](#what-it-came-to). Since then a proof can be
[written packed](tsdf-proof-pack.md).

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
the commitment does not depend on how an image was compressed. The leaf
holds the frame's pose beside that root. A checker always needs every
pose, and a leaf is 177 bytes, so the frames' leaves are handed over
whole: whoever has them recomputes the scan root, and from then on holds
every pose and every frame's depth root.

**The map.** One leaf per block, in the canonical block order, holding the
block's index and its 6,144 bytes of sums and weights.

**The statement** binds the two roots to everything else the answer depends
on: voxel size, truncation, frame stride, the camera's intrinsics, the
depth scale, the tile size, and a name for the fusion rule. A change to the
arithmetic is a new rule name, and the pinned digests are what notice one.

## What can be checked

**One block.** Given the statement, the frames' leaves, a block with its
path to the map root, and the depth tiles that block's voxels project into,
each with its path to its frame's depth root: rule out frames by pose,
evaluate the rest in order, compare bytes. The checker computes for itself
which pixels the block reads, and refuses a proof that leaves one out.

**That a block is absent.** Blocks are in sorted order, so two neighbouring
leaves show there is nothing between them. An absent block is unknown
space. Leaving a block out can hide what the scan saw there, and can never
make anything look clear. That is as well, because this check is weaker
than the others: a root does not say its leaves are in order, and two
leaves cannot show that the rest are. Nothing a clear route rests on
needs an absence. See the open questions.

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
3. **Commit.** *Built:* [`tsdf-commitment.md`](tsdf-commitment.md). The
   two trees and the statement, as a file. The tree is pinned to the
   Certificate Transparency test vectors and to the RFC's definition
   written out a second time; the roots to a recomputation by hand from
   the session's files and the volume's bytes; the fixture's statement
   to literal digests on every platform.
4. **A block's proof, and a checker that has nothing else.** *Built:*
   [`tsdf-block-proof.md`](tsdf-block-proof.md). It reads the statement
   and the proof and no session, and says verified, refuted or invalid.
   Pinned against step 1 on every block: of an honest map, verified as
   whole-frame fusion has it; of three maps altered after fusion (a
   flipped bit, a wall moved one voxel, an obstacle removed), refuted in
   exactly the blocks that were changed; of four scans altered before
   committing (a depth pixel changed, a frame dropped, two frames
   exchanged, a pose nudged), refuted in exactly the blocks whole-frame
   fusion of the altered scan disagrees on. An altered proof is stopped
   by a check the page names for each alteration, and no byte of one can
   be changed.
5. **A route's certificate**, on the office. *Built:*
   [`tsdf-route-certificate.md`](tsdf-route-certificate.md). A route in
   whole voxels, every block it touches, the frames' leaves once and each
   depth tile once. The check works out the route's blocks itself, fuses
   them all again, and only then asks whether the route is clear: clear,
   not clear, refuted or invalid. Clearance is pinned to the free-space
   map tool on 41 routes, the blocks to their single proofs and to whole
   frames. A map with an obstacle written out of a route says the route
   is clear and its certificate says refuted. On the office a 3.5 m
   route checks in 3 to 7 s from a certificate of 109 to 163 MB, a fifth
   to two fifths of the scan's depth: much less of a saving than one
   block, because half the scan looks through the middle of a room.
6. **Audit by sampling, and the measurements.** *Built:*
   [`tsdf-map-audit.md`](tsdf-map-audit.md) and
   [`verifiable-map-costs.md`](verifiable-map-costs.md). The auditor's
   seed and the statement's digest draw rows of the map by a rule anyone
   can run; the answer must hold exactly those; a passed audit says how
   many wrong blocks it would almost surely have found. The draw and the
   bound are each worked out a second way in the tests. Before
   measuring, the tiles of a frame were given one proof shared between
   them in place of a path each, pinned to the RFC's own recursion.

## What it came to

On the real office, 69,415 blocks fused from 676 frames in 460 s:

- **One block** is checked in a tenth of a second from 382 kB, a
  thousandth of the depth its volume was fused from. All 600 blocks
  drawn from three published volumes are verified.
- **A map altered** in one voxel, a wall moved one voxel, an obstacle
  removed, or a scan altered in one depth sample, one pose or one frame:
  refuted in exactly the blocks that no longer follow, on every block of
  a 351-block room, on all six CI platforms.
- **A route** 3.5 m long through the middle of the room is certified
  clear and checked in 2 to 6 s. A map with an obstacle written out of a
  route says the route is clear, and its certificate says refuted.
- **An audit** of 459 blocks is checked in 3 s and would catch a map one
  part in a hundred wrong 99 times in 100. It does not find one altered
  block, and says so.

And what it did not come to:

- **Many blocks are not small.** A route through open space or an audit
  by lot is read by most of the scan's frames. Their certificates are 56
  to 110 MB of raw depth, a seventh to a quarter of the samples the
  volume was fused from, and about the size of those frames' own PNG
  files. As written they save time and the volume, not bytes.
  [Packed](tsdf-proof-pack.md), which was built after the six steps,
  they are 9 to 19 MB, a tenth to a fifth of those files.
- **Absence is not proved.** A block the map does not hold cannot be
  shown to be missing to someone who holds only the root.
- **The scan is taken as given.** Everything here binds a map to a scan.
  Nothing binds the scan to the world.

## Open questions

- ~~Whether tiles of 16 pixels are the right size.~~ Counted in step 3,
  with one path a tile: on the office a block's depth and its paths come
  to a median 302 kB at 8 pixels, 188 kB at 16 and 267 kB at 32. Measured
  again in step 6 with one proof shared between a frame's tiles: 8-pixel
  tiles make a block's proof 6% smaller and a route's certificate 3 to
  10% smaller, and take half as long again to check. Sixteen stays.
- How a block is shown to be absent to someone who holds only the root.
  Two neighbouring leaves are enough only if the leaves are known to be
  in order. Keeping each subtree's range of indices in its node would
  make the order checkable along a path, at the cost of a tree that is
  no longer the RFC's. Undecided. A clear route does not need it, since
  every block the route touches has to be shown; a claim that somewhere
  is unknown would.
- ~~How much of a route's certificate is shared between its blocks.~~
  Measured in step 5. The leaves are carried once and each tile once,
  and on the office the route still needs half the frames and a third of
  their tiles. Two fifths of that certificate was tile paths, one a
  tile; step 6 replaced them with one proof a frame and they are now
  under one part in a hundred of it.
- ~~Whether proofs should be compressed.~~ Built after the six steps:
  [`tsdf-proof-pack.md`](tsdf-proof-pack.md). `--packed` writes a proof,
  a certificate or an audit inside a zlib stream, and the checks read
  either. Kinect certificates and audits pack about seven times and
  rendered depth about twice. A proof is still named by the SHA-256 of
  its unpacked bytes, because the same proof need not pack to the same
  bytes on every machine.
- Whether the mesh, which is a repeatable function of the volume, should be
  committed block by block as well.
