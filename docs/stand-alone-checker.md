# A checker that stands alone

[A block's proof](tsdf-block-proof.md) is checked by the package that
made it, with the evaluator that fused the map. Its own page said what
that leaves open: the check is this package, and a mistake in the
evaluator would be in both. `checker/check_block_proof.py` is a second
check of the same proofs, in one file that imports nothing but Python's
standard library:

```powershell
python checker/check_block_proof.py scan.sftcommit block.sftproof
```

It needs no numpy and no SpatialForge, and its tests run it in an
interpreter that can load neither. It exits 0 if the block is verified,
1 if it is refuted or the proof is invalid, and 2 if a file cannot be
read as a statement or a proof. A [packed proof](tsdf-proof-pack.md) is
read as the proof it carries.

```text
BLOCK PROOF CHECK, STANDARD LIBRARY ONLY
statement_sha256: 7b771f1ef5ff5249f48cf29affdbce16df2349f02b67cf7d9accf4884414e780
proof_sha256: 50929ac2c1cabc3ef1e78759087718966036cd1147213c278f1113786612f626
packed: no
block: 1,0,-1 row=3 of=8
check statement: passed
check shape: passed
check scan root: passed
check map root: passed
check depth roots: passed
check poses: passed
check pixels: passed
check fusion: passed
fusion: ready=2 read=2 contributing=2 differing_voxels=0
verdict: verified
```

It makes the same eight checks in the same order, and gives the same
three verdicts, as the [proof's page](tsdf-block-proof.md#the-checks-in-order)
describes. It does not fuse the way the package does. The package first
rules frames out by the eight corners of the block, with a margin for
rounding, and evaluates voxels only in the frames left. The checker
evaluates every voxel of every frame that has depth and a pose. A frame
the package rules out reads no pixel either way, so the two must agree to
the bit, and the tests hold them to it.

## The arithmetic, to the bit

The [reference integrator's page](reference-tsdf.md#integration-rule)
gives the rule. A check that is to agree to the bit needs the order of
every operation as well, and that was read from the evaluator to write
the checker. Written out, in IEEE doubles and one operation at a time,
for the voxel at local place `flat` of the block at index `(bx, by, bz)`:

```text
local       (i, j, k) = (flat % 8, flat // 8 % 8, flat // 64)
centre      cx_w = (float(i + 8 bx) + 0.5) * voxel_size_m      and so for y, z
            the integer is exact, then the half is added, then the product
from pose   T, sixteen numbers row by row; t = (T[3], T[7], T[11])
            dx = cx_w - T[3],  dy = cy_w - T[7],  dz = cz_w - T[11]
camera      x = (T[0] dx + T[4] dy) + T[8] dz
            y = (T[1] dx + T[5] dy) + T[9] dz
            z = (T[2] dx + T[6] dy) + T[10] dz
            kept if x, y, z are finite and z > 0
image       u = ((fx x) / z) + cx,   v = ((fy y) / z) + cy
            kept if u, v are finite, -0.5 <= u < width - 0.5 and
            -0.5 <= v < height - 0.5
pixel       column = floor(u + 0.5),  row = floor(v + 0.5)
            kept if inside the image
depth       d = float(sample) * depth_scale_m
            kept if finite and d > 0
distance    e = d - z
            kept if finite and e >= -truncation_m
update      sum += min(max(e / truncation_m, -1.0), 1.0),  weight += 1
```

Three things in it are easy to get wrong, and each would be a different
map:

- **The inverse of a pose is its rotation transposed.** The camera point
  is `Rᵀ (p - t)`, not the matrix inverse of `T`, which the reference
  page writes as `inverse(T_world_camera)`, applied to `p`. For a pose
  read from a file the two differ in the last bits, and for one rigid
  only to one part in 100,000, which fusion accepts, by far more.
- **Frames are taken in replay order and chosen by place.** The frames a
  map is fused from are every `frame_stride`-th of all the scan's frames,
  counting those with no depth or no pose, and a chosen frame is fused
  only if it has both. Each frame's contribution is added in that order.
- **Sums start at +0.0** and are compared as their 64 bits, weights as
  integers.

A pose fusion reads has to be finite, and one of a frame with depth has
to be rigid: its last row within `1e-5` of `0 0 0 1`, each pair of rows
of its rotation orthonormal within `1e-5` by the sum of their three
products taken left to right, and its determinant, expanded along the
first row, within `1e-5` of one.

## What it holds a file to

The [proof's page](tsdf-block-proof.md#what-a-proof-holds) gives the
layout. The checker refuses, as the package does, anything that is not
exactly a proof:

- **The header** is ASCII JSON in canonical form: keys sorted, two spaces,
  one newline at the end, no arrays, no key twice. It has exactly these
  fields, here for the first block of the committed fixture:

  ```json
  {
    "block": {
      "blocks": 8,
      "row": 0
    },
    "frames": 2,
    "image": {
      "height": 2,
      "tile_pixels": 16,
      "width": 2
    },
    "schema": "spatialforge.tsdf-block-proof",
    "schema_version": "0.1.0",
    "statement_sha256": "7b771f1ef5ff5249f48cf29affdbce16df2349f02b67cf7d9accf4884414e780",
    "tile_frames": 2
  }
  ```

  `tile_frames` is how many frames the proof holds tiles of. Counts are
  whole numbers, not floats or booleans: at least one, except the row and
  `tile_frames`, which may be zero. The header is 2 to 16,384 bytes.
- **Integers** are little-endian: the header's length and a frame's place
  8 bytes, a frame's count of tiles and a tile's row and column 4 bytes
  each, the block's index three signed 4-byte integers. Samples are
  unsigned 2-byte, sums 8-byte doubles and weights unsigned 4-byte, each
  block's voxels in the order `flat = x + 8 y + 64 z`.
- **The frames' leaves** are every frame of the scan, each at its own
  place in order. A leaf's flags are 1 for depth and 2 for a pose, and an
  absent pose or depth root is all zeros.
- **The frames read** are in increasing order, each with at least one
  tile, and each frame's tiles are in increasing place, inside the grid,
  with the tiles at the right and bottom edges cut short to the image.
  A tile is at most 1,024 pixels on a side.
- **Nothing** is left after the end, and no count may claim more than
  the bytes that are left.

A statement is held to the same canonical JSON, and its fields to what
the package holds them to. The checker refuses one thing the package
does not: a statement made under any rule but
`projective-tsdf/unit-weight/nearest-pixel/1`. It knows the arithmetic
of that rule and will not check a block under another. The package's
check of a proof does not look at the rule. Given a statement that names
another, it checks the block by the one arithmetic it has, which was
found by writing this checker and is not yet changed.

## What it is pinned to

Every test runs both checks on the same bytes and requires the same
verdict, the same check failed, the same count of frames with depth and
a pose and of frames that contributed, and the same count of differing
voxels.

- **Every honest block.** The eight of the committed fixture and the 351
  of the generated room: all verified by both.
- **Every one-bit change.** One bit in each of the 7,009 bytes of a
  fixture proof, and in each byte of a proof of the room. The fixture's
  come out as the proof's own page counts them: 403 not a proof, 17
  `statement`, one `shape`, 320 `scan root`, 6,252 `map root` and 16
  `depth roots`. Not one is verified or refuted by either.
- **Seven alterations.** One bit of one voxel, the far wall moved a
  voxel, an obstacle removed, one depth sample raised, a frame dropped,
  two frames exchanged, a pose moved a millimetre. Every block of the
  room after each: refuted where the package refutes, with the same
  voxels differing, and verified everywhere else.
- **The checks one bit never reaches.** Each tile of a frame left out,
  each frame's tiles left out, a true tile the block reads no pixel of,
  tiles of a frame it reads nothing of: `pixels`, in both. A scan
  committed with a stretched pose and with an endless one: `poses`, in
  both.
- **Standing alone.** The file's imports are read from its source, and
  they are hashlib, json, math, re, struct, sys and zlib, all of the
  standard library. It is run by an interpreter started with `-I -S`,
  which loads no site packages and not the directory it is in, and in
  which `import numpy` and `import spatialforge` both fail. Its output
  is held to the lines above.

## On real volumes

Two hundred blocks drawn with seed 0 from each of the three published
volumes the proof's page measures, each proved, then checked by both:

| Volume | Frames fused | Both say | Package, median | Checker, median | Checker, slowest |
|---|---|---|---|---|---|
| Kinect desk, 15 mm | 396 | verified, 200 of 200 | 0.14 s | 0.25 s | 0.75 s |
| Living room, 10 mm | 440 | verified, 200 of 200 | 0.13 s | 0.24 s | 0.37 s |
| Kinect office with free space, 15 mm | 676 | verified, 200 of 200 | 0.15 s | 0.27 s | 0.52 s |

Every one of the 600 is the same verdict with the same frames counted.
Looking at every voxel of every frame in plain Python costs a block a
quarter of a second, about twice what the package takes.

(Measured once, on one laptop, with a scratch script.)

## What it does not do

- **It checks block proofs only.** A route's certificate and an audit
  hold blocks the same way, but the checker does not read them.
- **It is not another author or another language.** It was written by
  the package's author, in Python, with doubles that round as the
  package's do. It is independent in what it runs, not in who wrote it,
  and it would share a misreading of the rule that its author made twice.
- **It does not show the rule is right.** Two programs that compute one
  rule and agree show that the rule is what both compute. Whether a map
  fused by it is a good map is
  [another page](tum-validation.md)'s question.
- **It is slower**, about twice the package's time a block, because
  it looks at every voxel of every frame in plain Python.
