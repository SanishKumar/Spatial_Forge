# Auditing a map by lot

A [block's proof](tsdf-block-proof.md) says one block follows from the
scan. Nobody is going to ask for every block of a map. An audit asks for
some of them, chosen so that whoever made the map could not have known
which.

```powershell
python -m spatialforge reconstruct tsdf-map-audit `
  scan.sftcommit scan.sftvol scan.vgsession answer.sftaudit `
  --seed 7f3a-2026-10-12 --blocks 459
python -m spatialforge reconstruct tsdf-map-audit-check `
  scan.sftcommit answer.sftaudit --seed 7f3a-2026-10-12 --blocks 459
```

The first is run by whoever holds the volume and the scan. The second is
run by the auditor, with the [statement](tsdf-commitment.md), the answer,
and their own seed and count. It reads no session and no volume, and exits
0 if every block drawn is what the scan gives.

This is the sixth step of [a map that can be checked](verifiable-maps.md).

## The draw

The auditor picks a **seed**: up to 128 letters, digits, `.`, `_`, `:` or
`-`. The seed, the statement's SHA-256 and a count fix which rows of the
map are drawn, by a rule anyone can run:

```text
draw i  =  SHA-256( "SFAUDIT1" || statement digest || seed length || seed || i )
```

The first eight bytes of each draw are read as a number. It is thrown away
if it falls in the uneven remainder above the largest multiple of the
map's block count, and otherwise taken modulo the count, so every row is
as likely as every other. Draws go on until that many different rows have
come up.

Rows are drawn, not block indices: the auditor does not know which blocks
a map holds, only how many, and a block's proof shows what is at a row.

**The seed has to be the auditor's.** If the maker of a map chose it, they
would choose one that misses what they changed. If it could be worked out
from the statement alone, they could try statements until the draw fell
where they liked: a map one part in a hundred wrong passes an audit of 459
blocks one time in a hundred, and a hundred statements are cheap. So the
check takes the seed and the count from the auditor and not from the
answer, and an answer drawn with any other seed is `invalid`.

## What an answer holds

The blocks at the rows drawn, held as
[a route's certificate](tsdf-route-certificate.md) holds its blocks: every
frame's leaf once, each block with its row and its path to the map's root,
and each depth tile once, with one proof shared between the tiles of a
frame.

## Three verdicts

| Verdict | What was found |
|---|---|
| `passed` | Every block drawn follows from the scan. |
| `refuted` | Everything is under the statement's roots, and some block drawn is not what the scan gives. One is enough: the map does not follow from the scan. The blocks are named. |
| `invalid` | The answer does not hold together, or was not drawn with the auditor's seed and count. It says nothing about the map. |

The checks are those of a route's certificate with `draw` in the place of
`route`: the rows the answer holds must be exactly the rows the auditor's
seed draws, worked out by the check and not taken from the answer.

## What passing is worth

`refuted` is certain. `passed` is not: it says how much of the map could
be wrong and still have got through.

If `wrong` of a map's `total` blocks are not what the scan gives, a draw
of `drawn` different blocks misses all of them with probability

```text
C(total - wrong, drawn) / C(total, drawn)
```

The check reports the smallest `wrong` for which that is under one in a
hundred, worked out in whole numbers:

```text
bound: a map with 691 or more of its 69415 blocks wrong passes such an audit less than 1 time in 100
```

| Blocks drawn | of the office's 69,415 | caught 99 times in 100 if this many are wrong |
|---|---|---|
| 100 | 0.14% | 3,122, or 4.5% |
| 200 | 0.29% | 1,578, or 2.3% |
| 459 | 0.66% | 691, or 1.0% |
| 2,000 | 2.9% | 158, or 0.23% |

**An audit by lot does not find a single altered block.** On the generated
room, a map with one block of 351 altered passed 14 of 20 audits of 60
blocks on the machine that wrote this page. One with 20 blocks altered
was caught by most audits of 30, and one with half its blocks altered by
every audit of 10. That is the arithmetic above, and it is why a claim
that rests on particular blocks should carry them: a route's certificate
checks the blocks that route depends on, however few they are.

## What it is pinned to

- **The draw, by hand.** The rule is run a second time in the test from
  the hash alone, for maps of 1 to 69,415 blocks. Asking for one more
  block keeps the ones already drawn. Two thousand seeds each drawing one
  row of eight draw every row between 200 and 300 times.
- **The bound, draw by draw.** The chance of missing is worked out a
  second way, as a product of fractions, for every draw from every map up
  to 30 blocks: the bound is the smallest count that brings it under the
  odds asked, and one fewer does not.
- **Block proofs.** The blocks of an answer are held as their own proofs
  hold them, and each of those proofs is verified.
- **Altered maps.** On three altered maps of the room, under 52 seeds, an
  audit is `refuted` exactly when its seed draws an altered row, and names
  exactly the altered blocks it drew, by the voxels they differ in.
- **The auditor's seed and count.** An answer checked with another seed,
  with a larger count or a smaller one, or made for another seed and
  relabelled, is `invalid` at `draw`.
- **Literal digests.** The committed fixture's draws and two of its
  answers are written out.
- **Every byte.** One bit changed in each of the 7,037 bytes of a fixture
  answer is never `passed` and never `refuted`: 422 are not an answer at
  all, 6 fail `draw`, and the rest fail the check of what they changed.

## On real volumes

An audit of 459 blocks, which catches a map one part in a hundred wrong 99
times in 100:

| Volume | Blocks | Answer | Frames read | Answered | Checked | Fusing the volume |
|---|---|---|---|---|---|---|
| Kinect desk, 15 mm | 5,401 | 174 MB | 395 of 396 | 18 s | 5.8 s | 79 s |
| Living room, 10 mm | 27,965 | 73 MB | 440 of 440 | 19 s | 2.8 s | 211 s |
| Kinect office with free space, 15 mm | 69,415 | 110 MB | 676 of 676 | 27 s | 3.0 s | 460 s |

All three pass, and are checked in a few seconds.

**They are not small.** Blocks drawn by lot are everywhere in a map, so
between them they are read by every frame. The office's answer holds a
quarter of the depth its volume was fused from, and the desk's seven
tenths, because the desk's camera never leaves and nearly every frame
sees nearly every block. An audit saves the time of fusing and the need
to hold the volume. It does not save much of the scan.

A hundred blocks of the office, which catch a map one part in twenty-two
wrong, are answered in 26 MB and checked in 0.7 s.

(Measured once, on one laptop, with a scratch script.)

## What it does not do

- **It does not find one altered block**, or a few. See above.
- **It does not see what a map leaves out.** Rows are drawn from the
  blocks a map holds. A block the scan would have filled and the map does
  not hold is never drawn. Leaving a block out hides what was seen there,
  and cannot make anything look clear.
- **It does not choose the seed.** An auditor who lets the maker choose,
  or who reuses a seed the maker has seen, has audited nothing. Where a
  seed nobody controls comes from is the auditor's business.
- **One answer is one audit.** Nothing here joins the results of several.
- **It does not say the scan is true**, or that the poses are right.
- **The check fuses with the same evaluator the map was fused with**, as
  for a block's proof.
