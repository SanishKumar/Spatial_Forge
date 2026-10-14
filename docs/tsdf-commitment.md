# Committing to a scan and a map

A block can be [fused from the few depth tiles it reads](tsdf-block-evidence.md),
but nothing in those tiles says they are the scan's, and nothing in a block
says it is the map's. A commitment is what says it: one hash for the scan
and one for the map, built so that a piece of either can later be shown to
belong with a short path of other hashes.

```powershell
python -m spatialforge reconstruct tsdf-map-commit `
  scan.sftvol scan.vgsession scan.sftcommit
python -m spatialforge reconstruct tsdf-map-commit-check `
  scan.sftcommit scan.sftvol scan.vgsession
```

The first writes a *statement*: about 800 bytes of JSON naming both hashes
and everything else a block's bytes depend on. The second recomputes it
from the volume and the scan and exits 1 if anything differs.

This is the third step of [a map that can be checked](verifiable-maps.md).
It proves nothing yet. It is what the next step's proofs are checked
against.

## The tree

Both hashes are roots of Merkle trees as
[RFC 6962](https://www.rfc-editor.org/rfc/rfc6962) defines them, the tree
Certificate Transparency uses:

```text
MTH({})      = SHA-256()
MTH({d0})    = SHA-256(0x00 || d0)
MTH(D[0:n])  = SHA-256(0x01 || MTH(D[0:k]) || MTH(D[k:n]))    k the largest power of two below n
```

An item and an inner node are hashed with different first bytes, so the
hash of two children can never be passed off as an item. It was chosen
because it is written down exactly and has published test vectors.
`spatialforge/merkle.py` is that tree and nothing about maps.

## What is hashed

Every leaf starts with a tag of its own, so a tile can never be read as a
frame or a block.

**A tile of depth.** `SFTILE01`, then the tile's samples row by row as
16-bit little-endian integers. They are the samples the session decodes,
not the file's bytes: two encodings of the same depth are the same scan.
A frame's image is cut into tiles on the grid block evidence uses, and its
*depth root* is the root over them, tiles taken row by row.

**A frame.** 177 bytes:

| Bytes | |
|---|---|
| 8 | `SFFRAME1` |
| 8 | the frame's place in replay order |
| 1 | 1 if it has depth, plus 2 if it has a pose |
| 128 | its pose, sixteen 64-bit floats, or zeros |
| 32 | its depth root, or zeros |

The scan's root is over every frame in replay order, whether fusion
selects it or not, so one commitment to a scan serves volumes fused at any
stride.

**A block.** `SFBLOCK1`, the block's index as three 32-bit integers, and
its 6,144 bytes of sums and weights as the volume holds them. The map's
root is over the blocks in the canonical block order.

**The statement.** This is the committed fixture's, whole:

```json
{
  "hash": "sha-256",
  "map": {
    "block_resolution": 8,
    "blocks": 8,
    "frame_stride": 1,
    "root_sha256": "2132eeb769f580eb6a273fce5e32fdabfee8a62a2d70b156678530c5422852f0",
    "truncation_m": 0.5,
    "voxel_size_m": 0.125
  },
  "rule": "projective-tsdf/unit-weight/nearest-pixel/1",
  "scan": {
    "camera": {
      "cx": 0.5,
      "cy": 0.5,
      "fx": 2.0,
      "fy": 2.0,
      "height": 2,
      "width": 2
    },
    "depth_sample": "uint16-le",
    "depth_scale_m": 0.001,
    "frames": 2,
    "root_sha256": "c5740130e75365c7dd3489543e3a8e70a93c798cf835e44c84b7774ce9906248",
    "tile_pixels": 16
  },
  "schema": "spatialforge.tsdf-map-statement",
  "schema_version": "0.1.0",
  "session_id": "scan-synthetic-0001",
  "tree": "rfc6962"
}
```

It is canonical JSON: keys sorted, two spaces, ASCII, one newline at the
end. The same statement is always the same bytes, and the SHA-256 of those
bytes is the one value a third party needs to hold. For the fixture it is
`7b771f1e…e780`, and a test requires that on every platform.

`rule` names the arithmetic a block's bytes come from. A change to what
fusion computes is a new name, and a statement made under the old one is
then a statement about a different function. So nothing is proved or
checked under a statement that names a rule this package does not fuse
by: a proof, a certificate or an audit held to one fails its first
check, `statement`, and none is made under one.

## Why a frame's pose is in its leaf

A check of any block needs every pose, because frames are ruled out by
pose. The frames' leaves are small, so they are meant to be handed over
whole. Whoever has them recomputes the scan root from them and compares it
with the statement's. From then on they hold every frame's pose and every
frame's depth root, and a tile needs only its path to that depth root, not
a path through the scan's tree as well.

## What a root does not say

A root does not say how many leaves it is over. The first leaf of five and
the first leaf of six climb the same steps, so one path serves either
count; under another count a leaf can even sit at another place. A test
shows both. So the statement carries the counts: `frames`, `blocks`, and
the image size and tile size that fix how many tiles a frame has. A path
is checked at the stated count, and at the true count a leaf's path leads
to the root from its own place and no other.

Nor does the scan's root say how large a tile is. The fixture's image is
two pixels square, so it is one tile at any tile size from two up and has
the same scan root at all of them. The statements differ, because each
names its tile size.

## What it is pinned to

- **The published vectors.** The eight roots of the Certificate
  Transparency reference test vectors, as literals.
- **The RFC's definition, written out again.** The module builds its tree
  bottom up. The test writes the RFC's top-down definition recursively,
  from the items, and requires the same root for every size up to 130 and
  the same path for every leaf of every size up to 66. For every tree up
  to 19 leaves, a leaf's path fails with another leaf in its place, with
  its own leaf at any other place, and with any one step changed.
- **Both roots, by hand.** The scan's root is recomputed in the test from
  the session's text indexes and its ASCII depth images, with no session
  loader, no array and no tree but the RFC's: the fixture at three tile
  sizes and the 20-frame room at seven, from one pixel to the whole image.
  The map's is recomputed from the volume file's bytes taken apart by
  hand: the fixture, a forged volume, and the room's 351 blocks.
- **Literal digests.** The fixture's two depth roots, two frame leaves,
  scan root at two tile sizes, map root, and its statement's 769 bytes and
  digest are written out in the test. Any platform that computes another
  byte fails.
- **One thing changed at a time.**

  | Changed | What follows |
  |---|---|
  | one depth sample of one frame | one tile's leaf, that frame's leaf, the scan root; no other frame |
  | one pose, by the smallest step a float can take | that frame's leaf and the scan root; no depth root |
  | an image re-encoded, as binary or with other spacing | the replay digest; not the scan root |
  | a frame's depth or pose taken away | its leaf says so in one byte; an absent pose is not a pose of zeros |
  | two frames' depth exchanged | both frames' leaves and the scan root |
  | one voxel of a volume that still loads | one block's leaf and the map root |
  | two blocks' accumulators exchanged | the map root |

- **Evidence is under the root.** Every tile gathered as evidence for a
  sample of the room's blocks, at 8 and 16 pixels, hashes to a leaf that
  its path takes to its frame's committed depth root; and the pose in the
  evidence is the pose in the frame's leaf. This is the join the next step
  rests on.
- **The file.** Only the canonical bytes load. A statement reformatted,
  reordered, with a number written another way, with a field missing,
  unknown or repeated, with an array, a NaN, or the wrong schema, hash,
  tree or sample name is refused: 56 cases. A session's name and a rule's
  hold only characters JSON gives no meaning to, so every statement that
  can be made can be written and read back, and a test does so with the
  longest names and the largest numbers allowed. Writing never replaces a
  file.
- **The commands.** Both outputs are asserted whole, with exit 0 when the
  statement holds, 1 when it does not, and 2 when it cannot be checked.
  Neither changes a file it reads, and a test patches the fusion functions
  under the calls they make to prove that nothing is fused.

## On real volumes

| Volume | Frames | Blocks | Commit | Check | Statement | Frames' leaves |
|---|---|---|---|---|---|---|
| Kinect desk, 15 mm | 792 | 5,401 | 16 s | 14 s | 800 bytes | 140 kB |
| Living room, 10 mm | 880 | 27,965 | 17 s | 12 s | 797 bytes | 156 kB |
| Kinect office with free space, 15 mm | 1,352 | 69,415 | 20 s | 17 s | 803 bytes | 239 kB |

The office's statement stands for 831 MB of depth, every frame's and not
only the 676 its volume was fused from, and a 427 MB volume.
Two of the desk's frames have no pose, and their leaves say so. All three
statements hold when checked. The desk's statement held against the same
scan's 20 mm volume is refused with four differences named: voxel size,
truncation, block count and map root.

Of the office's 20 s, 15 are committing to the scan, which decodes every
depth image and replays the session before and after to see that it did
not change underneath. The map's tree takes half a second.

(Measured once, on one laptop.)

## What a tile's path costs

The step before left a question open: smaller tiles carry less depth, but
each needs its own path. For the same 200 blocks of the office, with one
path a tile from the tile to its frame's depth root:

| Tile | Tiles a block, median | Depth, median | Paths, median | Both, median | Both, mean |
|---|---|---|---|---|---|
| 4 pixels | 1,646 | 53 kB | 785 kB | 838 kB | 2,077 kB |
| 8 pixels | 554 | 71 kB | 231 kB | 302 kB | 809 kB |
| 16 pixels | 218 | 112 kB | 77 kB | 188 kB | 459 kB |
| 32 pixels | 115 | 236 kB | 32 kB | 267 kB | 506 kB |
| 64 pixels | 70 | 567 kB | 15 kB | 583 kB | 928 kB |

Sixteen pixels is the cheapest of these and stays the default. At eight,
a tile is 128 bytes and its path about 400.

So what a checker of one office block will need, at the median, is the
statement, the 239 kB of frames' leaves, and 188 kB of tiles and paths:
about 430 kB against the 415 MB of depth the volume was fused from, one
part in 970. Depth alone was one part in 3,700; the paths and the leaves
are what a checker pays to know the depth is the scan's. The frames'
leaves are the larger half, and are the same for every block of the scan.

(Counted with a scratch script from the evidence the last step gathers,
once, on one volume. No proof was built to get these numbers.)

That count is of one path a tile. The proofs that were built later give
the tiles of a frame one proof shared between them, which takes most of
the paths column away: the office's median block came to 382 kB, not 430.
With paths that cheap, 8-pixel tiles were measured again and are 6%
smaller for a block. Sixteen stayed.

## What it does not do

- **It is not a proof.** `tsdf-map-commit-check` needs the whole volume
  and the whole scan. Checking one block against a statement, with
  nothing else, is the next step.
- **It does not say the map follows from the scan.** A statement binds
  two things. A forged volume has a root as good as any; the test's
  forgery commits without complaint, to a different root. What catches a
  forgery is fusing a block again.
- **It does not say the scan is true**, or that the poses are right.
- **It cannot show a block is absent.** Blocks are committed in order,
  and two neighbouring leaves would show nothing lies between them, but
  only to someone who knows every leaf is in order. A root does not say
  so. This is open in the design page.
- **A path belongs to one tile.** Tiles of the same frame share most of
  their paths' upper hashes. `merkle.py` has since been given a proof
  shared between several leaves, and a block's proof uses it.
- **The rule's name is only a name.** Nothing ties it to the code but the
  digests the suite pins.
- **Colour, inertial samples and timestamps are not committed.** Fusion
  reads none of them.
- **Depth must fit 16 bits**, as in the step before.
