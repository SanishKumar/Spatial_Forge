# Proving one block

A [statement](tsdf-commitment.md) names a root for a scan and a root for a
map. A proof is what lets someone who holds only the statement decide, for
one block, whether the map's block follows from the scan.

```powershell
python -m spatialforge reconstruct tsdf-block-prove `
  scan.sftcommit scan.sftvol scan.vgsession block.sftproof --block 12 -3 40
python -m spatialforge reconstruct tsdf-block-proof-check `
  scan.sftcommit block.sftproof
```

The first needs the volume and the scan, and writes a proof. The second is
given the statement and the proof and reads nothing else: no session and no
volume. It exits 0 if the block is verified.

`--packed` on the first writes the same proof inside a zlib stream, and
the second reads either: [packing a proof](tsdf-proof-pack.md).

This is the fourth step of [a map that can be checked](verifiable-maps.md).

## What a proof holds

- **Every frame's leaf**, 177 bytes each: its place, its pose and its depth
  root. The scan's root is recomputed from them.
- **The block as the map holds it**: its index and its 6,144 bytes of sums
  and weights, with its path to the map's root.
- **The depth the block reads**: for each frame it reads a pixel of, the
  tiles holding those pixels, and one proof shared between them that
  takes them to that frame's depth root.

The tiles a block reads of a frame lie together in the image, and tiles
that lie together share nearly all of their paths to the root. So a frame
carries one *shared proof*: the hashes of the largest subtrees that hold
none of its tiles, left to right. Nine tiles in a square of three, of a
frame's 1,200, need 17 hashes between them where nine paths need 99.

It is a file, `.sftproof`:

```text
SFTPRF01, then the header's length                 16 bytes
header                 canonical JSON: the statement's SHA-256, the counts,
                       the image and tile size, the block's row
every frame's leaf     177 bytes each
the block's leaf       6,164 bytes: tag, index, sums, weights
the block's path       32 bytes a step
for each frame read:   its place and how many tiles, 12 bytes
  for each tile:       its row and column, 8 bytes
                       its samples, 2 bytes each
  the tiles' shared proof, 32 bytes a hash
```

Nothing in it has a length of its own. How many samples a tile has, how
many steps a path has and how many hashes a frame's tiles share follow
from the header and from which tiles they are, so the bytes can be read
one way only, and a file with a byte missing or a byte to spare is not a
proof.

## Three verdicts

They are kept apart, because they are findings about different things.

| Verdict | What was found |
|---|---|
| `verified` | Everything in the proof is under the statement's roots, and the block fused from it is the block the map holds, bit for bit. |
| `refuted` | Everything in the proof is under the statement's roots, and the block fused from it is another block. What was committed as the map does not follow from what was committed as the scan. The proof is the evidence, and anyone can check it. |
| `invalid` | Something in the proof is not under the statement's roots, or it holds less than fusion needs, or more. This is a finding about the proof. It says nothing about the map. |

An altered proof is `invalid`. It is never a verified block, and it is
never an accusation either.

## The checks, in order

The check stops at the first that fails, so an invalid proof names one.

| Check | What it holds to what |
|---|---|
| `statement` | The statement is made under the rule this check fuses by, and the proof names its SHA-256. A statement may name any rule, and under another the check would be fusing a function the statement does not name. |
| `shape` | It has the statement's count of frames and of blocks, its image size and its tile size. A root does not say how many leaves it is over; the statement does. |
| `scan root` | The frames' leaves lead to the statement's scan root. From here on every pose and every depth root is the scan's. |
| `map root` | The block's leaf and its path, at its row, lead to the statement's map root. |
| `depth roots` | Each frame's tiles, at their places in the image, and the proof they share lead to that frame's depth root. |
| `poses` | Every pose fusion would read is one fusion would accept. |
| `pixels` | The proof holds every pixel the block reads, and no tile the block reads no pixel of. |
| `fusion` | The block fused from the tiles is the block claimed. This is the one that says `refuted`. |

## A proof cannot leave something out

Which frames reach a block, and which pixels of them it reads, is not
taken from the proof. The check rules frames out from their committed
poses, as fusion does, and runs the voxel evaluator over the rest. The
evaluator asks for the pixels it reads. A pixel that is not held is a
refusal that names it:

```text
observation 6: the block reads the depth pixel at row 31, column 31, and the evidence holds no tile for it
```

So a frame cannot be dropped by not mentioning it, and depth that was not
given is never taken for empty space.

A tile no pixel of which was asked for is a refusal too, even when it is a
true tile of the scan with a true path. A proof is therefore exactly what
its block needs, and a block has one proof: the same bytes whether it is
proved alone or with others, in whatever order.

## Tampering, and what stops it

**A map or a scan altered before it was committed to.** The statement is
then a true statement of altered things, and every proof ties to it. What
is wrong is that the map no longer follows from the scan, and `fusion`
says so.

On the generated room, 351 blocks from 20 frames, every block was proved
and checked after each of seven alterations:

| Altered | Blocks refuted | Voxels differing |
|---|---|---|
| one bit of one voxel's sum, in the map | 1 | 1 |
| the far wall moved one voxel, in the map | 176 | 40,309 |
| an obstacle removed, in the map | 20 | 1,433 |
| one depth sample raised by one, in the scan | 3 | 10 |
| a frame dropped from the scan | 253 | 59,181 |
| two frames' depth exchanged | 234 | 35,707 |
| one pose moved a millimetre | 229 | 28,552 |

Each time the refuted blocks are exactly the blocks that no longer
follow, with the same count of voxels in each, and every other block is
verified. For an altered map those are the blocks whose bytes were
changed. For an altered scan they are the blocks that
[fusing each block alone from whole frames](tsdf-lone-block.md) of that
scan disagrees with the map on. A refuted block passes every check
before `fusion`.

**A proof altered, under an honest statement.**

| Altered | Stopped by |
|---|---|
| the statement it names; the statement it is checked against; a statement made under another rule | `statement` |
| its count of blocks or frames, its tile size; a frame left out of the leaves and the rest moved up to hide it | `shape` |
| a pose moved, even by the least a float can move; a pose taken away; a depth root replaced or said to be absent; two frames exchanged; a frame put in place of its neighbour | `scan root` |
| one bit of one sum; one weight; the block's index; its row; a step of its path; another block's path; another block's accumulators | `map root` |
| one depth sample; the first or last hash of a frame's shared proof; a tile's samples given at another place in its frame; one frame's tiles given as another's; a tile cut from an altered scan; tiles of a frame that has no depth | `depth roots` |
| a pose that is not a rigid transform, or not finite, in a scan committed that way | `poses` |
| any one tile left out, the rest truly tied to their frame; any one frame's tiles left out; a true tile the block does not read; true tiles of a frame the block does not read, of a frame the stride skips, or of a frame with no pose | `pixels` |

A frame left out of the leaves without renumbering the rest is not a proof
at all and cannot be read.

**Every byte.** One bit was changed in each of the 7,009 bytes of a
fixture block's proof file, in turn. Not one change is a verified block
and not one is a refuted block:

| Outcome | Bytes |
|---|---|
| not a proof: the file cannot be read as one | 403 |
| `statement` | 17 |
| `shape` | 1 |
| `scan root` | 320 |
| `map root` | 6,252 |
| `depth roots` | 16 |

Every one of the 7,009 beginnings of the file is refused, and so is the
file with a byte added.

## What it is pinned to

- **The first step.** Every block of the room: verified, and with the same
  frames ruled out by pose, the same evaluated and the same contributing
  as the block fused alone from whole frames. After each alteration of the
  scan above, the same verdict and the same count of differing voxels as
  whole-frame fusion of the altered scan gives, for all 351 blocks.
- **Literal digests.** The proof of each of the committed fixture's eight
  blocks is 7,009 bytes, and the SHA-256 of each is written out in the
  test, with the whole layout of the first. Any platform that produces
  another byte fails.
- **Nothing else is read.** A test patches session replay, depth decoding,
  the scan commitment, the volume's tree and both ways of opening a file,
  and checks proofs from their bytes. The command is run in a directory
  holding a statement and a proof, with the session and volume loaders
  taken away.
- **The file.** 45 kinds of bytes that are not exactly a proof are
  refused, each with its reason. Writing never replaces a file.
- **The prover.** It recomputes the statement from the volume and the
  scan and refuses to prove under any other. It reads the scan twice,
  once for the tiles and once for the trees their paths come from, and
  holds each tile to the tree: a scan that changed in between is refused
  and not handed on as a proof that cannot hold.

## On real volumes

Two hundred blocks drawn with seed 0 from each of three published volumes,
proved, written to bytes, read back and checked:

| Volume | Verified | Proof, median | Mean | Largest | Of which frames' leaves | Check, a block |
|---|---|---|---|---|---|---|
| Kinect desk, 15 mm | 200 of 200 | 535 kB | 944 kB | 7.1 MB | 140 kB | 0.10 s |
| Living room, 10 mm | 200 of 200 | 323 kB | 355 kB | 1.1 MB | 156 kB | 0.08 s |
| Kinect office with free space, 15 mm | 200 of 200 | 382 kB | 558 kB | 2.6 MB | 239 kB | 0.11 s |

A block of the office is checked from 382 kB, at the median, in a ninth
of a second. The scan it was fused from holds 415 MB of depth and fusing
the volume took 460 s.

With a path for every tile the office's median was 437 kB, close to the
430 kB the step before estimated from counts alone, and the desk's 706
kB. Sharing one proof among a frame's tiles took the office's hashes
from 77 kB to 21 kB a block and the desk's from 225 kB to 57 kB.

Three fifths of the office's median proof is the frames' leaves, which
are the same in every proof of that scan. The desk's proofs are the
largest because its camera never leaves the desk: a block there is read
by a hundred frames.

Making the 200 proofs took 21 to 27 s a volume, after 10 to 18 s to make
the statement.

Then one voxel of the desk's map was negated, in a block that was among
the 200, the volume given new digests so that it loads, and a statement
made of it. Of the same 200 blocks, 199 are verified and that one is
refuted, one voxel differing.

(Measured once, on one laptop, with a scratch script.)

## What it does not do

- **It does not show a block is absent.** Only a block the map holds can
  be proved. See the design page's open questions.
- **It is one block.** A verified block says nothing of the others. A
  claim that rests on many blocks is a certificate's business, and a
  whole map's is an audit's.
- **Every proof repeats every frame's leaf.** On a long scan that is most
  of a proof. [A route's certificate](tsdf-route-certificate.md) and
  [an audit](tsdf-map-audit.md) hold many blocks and carry the leaves
  once.
- **`refuted` does not say what was altered.** A map changed after fusion
  and a scan changed before committing look the same from here: the two
  that were committed do not agree.
- **It does not say the scan is true**, or that the poses are right, or
  that the map is accurate.
- **The check fuses with the same evaluator the map was fused with.** It
  shows a block is what this engine makes of that scan. A mistake in the
  evaluator would be in both.
  [A second check](stand-alone-checker.md) fuses another way, in one file
  that needs only Python's standard library, and agrees with this one on
  every proof the tests make. It was written by the same hand.
- **The check is this package.** That it reads only a statement and a
  proof is a property of what it is given, shown by the tests above.
- **A proof shows the depth it holds** to whoever is given it.
