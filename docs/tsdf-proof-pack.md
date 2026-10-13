# Packing a proof

[A route's certificate](tsdf-route-certificate.md) and [an audit's
answer](tsdf-map-audit.md) are nearly all raw depth samples, and raw depth
compresses. `--packed` writes either of them, or [a block's
proof](tsdf-block-proof.md), inside a zlib stream:

```powershell
python -m spatialforge reconstruct tsdf-route-certify `
  scan.sftcommit scan.sftvol scan.vgsession way.sftroute `
  --band -51 -12 --radius 10 --through 47 105 --through -39 40 --packed
python -m spatialforge reconstruct tsdf-route-check `
  scan.sftcommit way.sftroute
```

`tsdf-block-prove` and `tsdf-map-audit` take the same flag. The three
checks take none: a file is read as what it is, packed or not, and checked
as the same proof.

This is not one of the six steps of [a map that can be
checked](verifiable-maps.md). It answers what they ended on: a
certificate of many blocks saved time and the volume, and no bytes.

## What a packed file is

```text
"SFTPACK1"                      8 bytes
length of what it carries       8 bytes, little-endian
a zlib stream (RFC 1950)        the proof's bytes, deflated
```

The file keeps its kind's name: a packed certificate is still a
`.sftroute`. What the stream carries is exactly the bytes the file would
have held unpacked, and those are read as strictly as ever. A packed file
is opened, and then it is a proof's bytes or it is refused.

## A proof is not named by its packed bytes

A proof's bytes are exact: the same proof is the same bytes on every
machine, and their SHA-256 names it. Its packed bytes are not exact, in
two ways.

- **Two libraries need not pack the same bytes the same way.** A zlib
  stream is defined by what it unpacks to. CPython is not built with
  the same library everywhere: this machine's is zlib-ng, and others
  have zlib.
- **One stream can be written many ways.** Of the 5,112 bits in the
  committed fixture's packed proof as this machine wrote it, 45 can be
  flipped and leave a file that unpacks to the identical proof. Deflate
  has bits no reader looks at and more than one way to say the same
  thing.

So nothing hashes packed bytes. The commands print a packed file's size
and nothing else about it:

```text
certificate_bytes: 7189
certificate_sha256: a755eb4b31d49fdc3a2011d288044fd4657d39be2a998a0a938074fbc59cf30f
packed_bytes: 699
```

The first two lines are of the certificate and are what they would be
without `--packed`. The third is of the file, and is 699 on this machine. A check prints the same lines of a packed file as of an
unpacked one.

It follows that **any stream that unpacks to a proof's bytes carries that
proof**. How hard it was squeezed, and by what, is not asked. A file
packed by another program is as good as one packed here.

## Bounded before it starts

A stream of a few kilobytes can inflate to gigabytes. Unpacking is held
to three things:

1. The length a packed file declares is at most what a file of its kind
   may be unpacked: 1 GiB for a proof, 2 GiB for a certificate or an
   audit. One that declares more is refused before anything is inflated.
2. No more than that many bytes and one are inflated. A stream that has
   more to give is refused there.
3. The stream has to end exactly at the length declared, with its
   checksum right and nothing after it.

That bounds memory by what an unpacked file of the same kind could
already have asked for, and no further: a packed file of 2 MB may still
make its reader hold 2 GiB before finding it is not a proof.

## What it is pinned to

- **Streams no compressor made.** The tests write a zlib stream by hand
  from RFC 1950 and RFC 1951: two header bytes, the proof in stored
  blocks, and a checksum added up a byte at a time. It is read back as
  the proof, with the SHA-256 the proof's own tests write out.
- **Every bit of that stream.** In a stream of stored blocks every bit
  has a meaning the RFCs give it. All 56,288 bits of the packed fixture
  proof are flipped in turn: the five spare bits of its one block leave
  the same proof, and every other bit leaves a file that is refused.
- **Streams of every other making.** zlib at each of its ten levels, in a
  512-byte window, flushed part way, and with two other strategies. All
  are the same proof, for all three kinds, and their files have many
  different digests.
- **The bound.** A length over the limit is refused with the inflater
  taken away. A stream of 16 MiB that declares a thousand bytes is
  refused with the bytes inflated counted: a thousand and one.
- **The refusals.** A wrong length either way, a byte after the stream,
  two streams, a wrong checksum, no stream, raw deflate, gzip, a stream
  that wants a dictionary, every cut of a packed proof short of its end,
  a proof of another kind packed, and a packed proof packed again. Each
  with its reason.
- **Whatever the stream lets through.** Every byte of each kind's packed
  fixture file is changed in turn and the file read back. It is refused,
  or it is the same proof, or it is bytes the statement calls invalid.
  It is never another proof that passes.
- **The commands.** Each writing command's output with `--packed` is its
  output without, and one line more. Each check's output for a packed
  file is its output for the unpacked one, line for line.

## On real volumes

The certificates and audits were written packed, read back from their
files and checked. The block proofs were packed in memory.

| | Unpacked | Packed | |
|---|---|---|---|
| **Kinect office with free space, 15 mm** | | | |
| the 3.5 m route, 1 voxel wide | 67 MB | 9.8 MB | 1 in 6.9 |
| the same, 9 voxels wide | 84 MB | 12.1 MB | 1 in 6.9 |
| the same, 21 voxels wide | 101 MB | 14.5 MB | 1 in 6.9 |
| 21 wide, 2 voxels high | 56 MB | 8.9 MB | 1 in 6.3 |
| the route near the ceiling | 4.0 MB | 0.8 MB | 1 in 4.8 |
| an audit of 459 blocks | 110 MB | 19.0 MB | 1 in 5.8 |
| one block's proof, median of 200 | 382 kB | 214 kB | 1 in 1.8 |
| **Kinect desk, 15 mm** | | | |
| an audit of 459 blocks | 174 MB | 25.4 MB | 1 in 6.9 |
| one block's proof, median of 200 | 535 kB | 179 kB | 1 in 3.0 |
| **Living room, 10 mm** | | | |
| an audit of 459 blocks | 73 MB | 32.3 MB | 1 in 2.3 |
| one block's proof, median of 200 | 323 kB | 198 kB | 1 in 1.6 |

**Kinect depth packs about seven times, and rendered depth about twice.**
In the two Kinect scans a sample is the same as the one to its left 77
to 79 times in 100, a quarter of all samples are zero where the sensor
read nothing, and a frame holds about 300 different values. In the
rendered living room a sample equals its neighbour 20 times in 100 and a
frame holds 2,700 values.

**One block's proof packs least.** Much of it is the frames' leaves,
which are poses and hashes. They are three fifths of the office's median
proof, and its 239 kB of them pack to 175 kB.

**Beside the frames themselves.** What a checker would otherwise be sent
is the frames a volume was fused from. Here they are as the PNG files
the datasets ship, and as the same samples put through zlib, which was
estimated from every tenth frame:

| | Depth PNGs of the frames fused | Their samples through zlib | An audit of 459, packed |
|---|---|---|---|
| Kinect desk | 48 MB | about 29 MB | 25.4 MB |
| Living room | 83 MB | about 77 MB | 32.3 MB |
| Kinect office | 89 MB | about 49 MB | 19.0 MB |

Packed, the office's audit is a fifth of its frames' files and its
routes a tenth to a sixth. The files flatter that: on the two Kinect
scans zlib by itself makes the same samples smaller than the datasets'
PNG files do, 1 in 8.4 against 1 in 4.7 and 5.1. Against frames packed
the same way, the office's audit is two fifths, its 9-voxel route a
quarter, and the living room's audit two fifths.

**The desk's audit saves little.** It is nearly nine tenths of its
frames packed the same way. The desk's camera never leaves, an audit of
it reads all but one of the frames and seven tenths of their samples,
and packed it is most of the scan packed. Packing does not change what
an audit reads. It stops raw samples being weighed against packed ones.

**Time.** Packing the office's 84 MB certificate took 1.5 s and
unpacking it 0.11 s. Reading a packed file took up to 0.2 s longer than
reading the same file unpacked, on files that take 1 to 3 s to read.
Checking is the same work on the same bytes.

(Measured once, on one laptop, with scratch scripts. Packed sizes are
this machine's, whose Python is built with zlib-ng 1.3.1.)

## What it does not do

- **It is not part of a proof.** Nothing a check decides depends on
  whether its file came packed. Packing is for carrying a proof and for
  keeping it.
- **It does not make a proof small where depth is not the weight.** One
  block's proof is mostly the frames' leaves, which are poses and hashes
  and hardly compress.
- **It is not a depth codec.** zlib knows nothing of images. Nothing
  here was compared with a codec made for depth.
- **It is zlib at its default level**, because that is in every Python.
  On the office's certificate and audit the hardest level was no smaller
  and took more than twice as long, and lzma was over a quarter smaller
  and took thirty times as long.
- **Packed files need not be the same on every machine**, and are not
  meant to be compared, hashed or signed. Compare what they carry.
- **A statement is never packed.** It is 800 bytes of text.
- **Nothing is streamed.** A packed file is read whole and unpacked
  whole, as an unpacked one is read whole.
