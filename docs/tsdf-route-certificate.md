# Certifying a route

A map is asked one kind of question more than any other: can something get
from here to there. An answer read off a map is as good as the map, and a
map taken on trust can have had an obstacle taken out of it, or a corridor
marked clear that nobody looked down.

```powershell
python -m spatialforge reconstruct tsdf-route-certify `
  scan.sftcommit scan.sftvol scan.vgsession way.sftroute `
  --band -51 -12 --radius 10 --through 47 105 --through 19 104 `
  --through 13 102 --through -39 40 --through 6 -72
python -m spatialforge reconstruct tsdf-route-check scan.sftcommit way.sftroute
```

The first needs the volume and the scan, and writes a certificate. The
second is given a [statement](tsdf-commitment.md) and the certificate and
reads nothing else. It works out which blocks the route touches, fuses
every one of them again from the depth the certificate carries, and only
then asks whether the route is clear. It exits 0 if it is.

`--packed` on the first writes the same certificate inside a zlib stream,
about a seventh the size on a real route, and the second reads either:
[packing a proof](tsdf-proof-pack.md).

This is the fifth step of [a map that can be checked](verifiable-maps.md).

## What a route is

It is stated in whole voxels, so that which voxels are meant is never a
matter of rounding.

- **Which axis is up.** One of the volume's own.
- **A line of columns.** A column is a pair of voxel indices on the two
  axes that are not up, in x, y, z order. `--through` gives waypoints, and
  each is joined to the next by Bresenham's line: a run of columns, each
  beside the one before it, drawn in integers.
- **A radius**, in voxels. The route is every column within it of a column
  of the line.
- **A band of heights**: the first and the last voxel index on the up
  axis.
- **How many observations make a voxel observed.** Three unless asked,
  the number a mesh asks of a voxel.

A position in metres is in the voxel `floor(position / voxel size)`.

The route is **clear** if every voxel of every one of its columns in the
band was observed at least that many times and lies in front of every
surface seen. That is the rule
[the free-space map](free-space-expansion.md) draws a column free by.

## What a certificate holds

- the route;
- every frame's leaf, once;
- every block the route touches, as the map holds it, with its row and its
  path to the map's root;
- the depth tiles those blocks read, each once however many of the
  blocks read it, with one proof shared between the tiles of each frame
  that takes them to that frame's depth root.

It is [a block's proof](tsdf-block-proof.md) for every block of the route,
with what the proofs would repeat carried once. On the generated room, a
certificate of 26 blocks is under half the size of their 26 proofs.

```text
SFTRTE01, then the header's length                 16 bytes
header                 canonical JSON: the statement's SHA-256, the counts,
                       the image and tile size, and the route but for its
                       waypoints
the waypoints          16 bytes each
every frame's leaf     177 bytes each
for each block:        its row in the map, 8 bytes
                       its leaf, 6,164 bytes
                       its path, 32 bytes a step
for each frame read:   its tiles, as a block's proof holds them
```

## Four verdicts

| Verdict | What was found |
|---|---|
| `clear` | Every block the route touches follows from the scan, and the route is clear in them. |
| `not clear` | Every block follows from the scan, and some voxel of the route was not observed enough or is at or behind a surface. The first such voxel is named. |
| `refuted` | Everything is under the statement's roots, and some block the route touches is not what the scan gives. Whatever the map says of the route, the map does not follow from the scan there. |
| `invalid` | The certificate does not hold together. It says nothing about the map or the route. |

## The checks, in order

Those of a block's proof, made once for all the blocks, with two more.

| Check | What it holds to what |
|---|---|
| `statement`, `shape` | As for a block's proof. |
| `route` | The blocks the certificate holds are exactly the blocks the route touches. They are worked out from the route here, not taken from the certificate, so a block cannot be left out. |
| `scan root`, `map root`, `depth roots`, `poses` | As for a block's proof, for every block and every tile. |
| `pixels` | The tiles hold every pixel any of the blocks reads, and no tile none of them reads. |
| `fusion` | Every block, fused from the tiles, is the block held. This is the one that says `refuted`, and it names the blocks. |
| `clearance` | In blocks now seen to follow from the scan, every voxel of the route is observed and in front. This is the one that says `not clear`. |

Whether the route is clear is asked last, and only of blocks that passed
`fusion`. The map's own word on the route is never used.

## The route is the claim

A certificate states its route, and the check answers for the route
stated. If the band, the radius, a waypoint or the count of observations
in a certificate is changed and the blocks it holds are still that route's
blocks, it is a certificate of the other route, and what it says of that
one is true. Whoever relies on a verdict has to read which route it is a
verdict on. Both commands print it.

## What the map says, and what the certificate says

On the generated room there is a box on the floor. A route two voxels wide
that runs from open floor into its side is `not clear`: 222 of its 1,078
voxels were never observed three times and 14 are at or behind a surface.

Then the map is altered. Every voxel of that route is written as observed
and empty, the volume is given new digests so that it loads, and a
statement is made of it. The altered map says the route is clear, and the
free-space map drawn from it shows every column of the route free.

The certificate of the same route through the altered map is `refuted`: 10
of its 12 blocks are not what the scan gives, in 256 voxels, which are
exactly the blocks and the voxels that were altered. Under the true map's
statement the altered blocks fail `map root`.

## What it is pinned to

- **The free-space map.** What a volume says of a route is worked out a
  second time from the map tool's own dense slab and its own column
  verdicts: for 41 routes on the room, up each of the three axes, in and
  out of the volume. The counts of unobserved and occupied voxels agree,
  the first voxel that is not clear agrees, and the route is clear exactly
  when the tool calls every one of its columns free.
- **The route's voxels.** The columns and the blocks are counted a second
  time one voxel at a time, for 28 routes. The line is held to the
  straight line in whole numbers: never more than half a voxel off it,
  for hundreds of pairs of waypoints drawn at random.
- **Block proofs.** The blocks, rows and paths of a certificate are those
  of the single proofs of the same blocks. Its tiles are the union of
  theirs, tied to each frame by the one proof the scan itself gives for
  those tiles together.
- **Whole frames.** All 524 blocks of the room, fused at once from tiles
  gathered once for all of them, are the volume's bytes, with the same
  frames ruled out, evaluated and contributing for each block as when it
  is fused alone. A scan with a frame dropped before it was committed to
  refutes all 26 blocks of a route, by the voxels whole-frame fusion of
  that scan differs in.
- **Literal digests.** Three certificates through the committed fixture
  are written out by SHA-256, and the 7,189 bytes of the first by layout.
- **Alterations of a certificate** under an honest statement: another
  statement; a block left out; a true block of the map the route does not
  touch; a pose moved; the voxels that block the route written as empty in
  the certificate; a block given another's row and path; a depth sample;
  each tile and each frame's tiles left out in turn; a true tile no block
  reads. Each is `invalid` at the check named for it.
- **Every byte.** One bit changed in each of the 7,189 bytes of a fixture
  certificate, in turn:

  | Outcome | Bytes |
  |---|---|
  | not a certificate | 569 |
  | `statement` | 19 |
  | `shape` | 1 |
  | `route` | 22 |
  | `scan root` | 320 |
  | `map root` | 6,241 |
  | `depth roots` | 16 |
  | a certificate of another route, answered truly | 1 |

  None is `refuted`. The one that still reads as a certificate asks for
  three observations where the route asked for two, and is told, truly,
  that the route is not clear at three.
- **What it costs to refuse.** A route states at most 4,096 waypoints, a
  line of 65,536 columns, a radius of 64 voxels and 65,536 blocks, and
  those are held before anything is worked out, so a certificate cannot
  ask for more work than its own size.

## On the real office

The office of [`tum-validation.md`](tum-validation.md), fused at 15 mm
into 69,415 blocks from 676 frames. Its floor is at -1.56 m in the scan's
own frame.

A line was found through the free-space map, 3.54 m long over five
waypoints, across the open middle of the room. With a radius of 10 voxels
it is 0.315 m wide, and in the band from 0.80 m to 1.40 m above the floor
it is 207,600 voxels in 732 blocks. Its certificate says `clear`.

| The same line | Blocks | Frames read | Tiles | Certificate | Checked |
|---|---|---|---|---|---|
| 1 voxel wide | 204 | 304 | 124,577 | 67 MB | 1.9 s |
| 9 voxels wide | 420 | 330 | 154,532 | 84 MB | 3.2 s |
| 21 voxels wide | 732 | 352 | 183,065 | 101 MB | 6.0 s |
| 21 wide, 2 voxels high | 122 | 302 | 105,334 | 56 MB | 1.3 s |

Making each took 24 to 30 s.

**These are large.** The scan the volume was fused from holds 415 MB of
depth, so a certificate of this line is a seventh to a quarter of it. A
single block of the same volume is proved from a thousandth.

The reason is where the route runs. The open middle of a room is what half
the scan looked through: 304 of the 676 frames read some voxel of even the
line one voxel wide, and a third of the depth tiles of those frames. A
block is a function of every frame that observed it, and free space in
the middle of a room was observed by most of them. Making the route wider
or taller adds little, because the same frames and mostly the same tiles
are already there.

A route where few frames looked is small. A line of the same width 1.45 m
long and 2.4 m above the floor, in 330 blocks that 30 frames saw, is
certified in 4.0 MB, half of it the blocks themselves, and checked in
about a second.

What a certificate saves is time, and the volume. The check takes 1 to
6 s and needs 1 to 5 MB of blocks; fusing the volume took 460 s and the
volume is 427 MB.

A certificate is nearly all depth: of the widest one's 101 MB, 94 MB is
tiles, 4.5 MB blocks and 0.65 MB the hashes that tie the tiles to their
frames. Before the tiles of a frame shared one proof, each carried its
own path and those hashes were 62.5 MB.

The tiles are raw samples. The scan's own depth images are PNG files and
hold the same samples in about a fifth of the bytes, so beside the
*files* of the frames this volume was fused from, about 90 MB, these
certificates save little or nothing as they are written. Written with
`--packed` the four are 9 to 15 MB, which is
[its own page](tsdf-proof-pack.md).
[The costs page](verifiable-map-costs.md) has the rest.

(Measured once, on one laptop, with scratch scripts. The line was found by
a breadth-first search over the free-space map, which is not part of the
package.)

## What it does not do

- **It does not find routes.** It certifies one it is given.
- **It cannot say a route is not clear because the map is empty there.**
  A block the map does not hold is unknown space, and a route that touches
  one is not clear, but nothing here can show a block is absent. The
  command refuses such a route and says which block.
- **It does not say something fits.** The radius and the band are the
  claim. What they have to be for a body to pass is the claimant's
  business.
- **Clear is what the scan saw.** A voxel observed three times and found
  empty was empty then. It says nothing of now, and nothing of what no
  frame could see.
- **It does not say the scan is true**, or that the poses are right.
- **The check fuses with the same evaluator the map was fused with**, as
  for a block's proof.
- **One route to a certificate.**
