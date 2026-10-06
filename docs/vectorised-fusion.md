# Vectorised fusion

The reference definition of fusion is scalar. `evaluate_tsdf_voxel_contribution`
answers one voxel against one observation — transform the voxel centre into
the camera, project it, sample the nearest depth pixel, classify the signed
distance — and returns a fully validated receipt. Fusing a plan that way is
one Python call per voxel per frame, and on the first non-trivial scan it ran
about a hundred times slower than the dense integrator it was meant to
replace.

Everything on this page is the same computation done in NumPy. None of it is
allowed to change the answer.

## Bit-identical, not close

The requirement is stricter than a tolerance. For every voxel the vectorised
evaluator must return the same status, the same integer weight delta, and a
sum delta with the **same float64 bit pattern** as the scalar path — compared
as `float.hex()`, so even `-0.0` and `0.0` are not interchangeable.

That is a constraint on operation order, because floating-point addition is
commutative but not associative:

- The world-to-camera product is written out term by term. A matrix product
  is free to reassociate the three-term sum or fuse the multiply-add, and
  either moves the last bits.
- `fx * x / z + cx` stays left-associated.
- The pixel rule stays `floor(u + 0.5)` after a half-open bound, and the
  integer bound after it is kept even though it looks redundant: a `u` one
  ulp below `w - 0.5` can round up to exactly `w` when `0.5` is added.
- The delta stays `clip(signed / truncation, -1, 1)`.

Each NumPy binary operation is separately rounded, so an expression written
in the scalar path's order produces the scalar path's bits.

Voxels that fail an early gate — behind the camera, outside the image —
still take part in the later arithmetic, which is what makes it a single
pass. Their values are nonsense and never read; the resulting overflow and
invalid-operation reports are silenced, and their pixel indices are pinned
to `(0, 0)` so the depth gather stays in bounds.

Statuses are assigned in the scalar evaluator's early-return order, first
failure wins:

```text
camera-point-nonfinite -> camera-z-nonpositive -> projection-nonfinite
  -> projection-outside-image -> depth-invalid
  -> signed-distance-nonfinite -> behind-truncation -> contributes
```

## Why accumulation order survives

A voxel's stored value is a running sum over the observations that accept
it, in canonical observation order. That order is what a vectorised path
must not disturb, and two facts keep it intact:

- **One contribution per voxel per frame.** A single observation accepts a
  voxel at most once, so adding whole frames in observation order performs
  exactly the scalar path's sequence of additions for each voxel.
- **Skipped voxels add `+0.0`.** `x + 0.0` preserves every value the
  accumulator can hold. The one exception, `-0.0 + 0.0 = +0.0`, cannot
  arise: the accumulator starts at `+0.0`, and round-to-nearest produces
  `-0.0` from a sum only when both operands are `-0.0`.

So no masking is needed. The skipped voxels add zero and the arithmetic is
the same arithmetic.

## Streaming

`fuse_tsdf_plan_streaming` is the path the pipeline uses. For each selected
frame, in order: decode the depth image once, evaluate every planned voxel
against it, add the result to storage, drop the frame.

The earlier paths walked blocks in the outer loop and needed every frame
decoded and held at once, which caps a 640x480 scan at roughly 208 frames.
Swapping the loops changes which voxel is visited when. It does not change
the order in which any one voxel receives its contributions, so the fused
bytes are the same — and memory is one depth image, whatever the sequence
length.

There is one evaluation function, `_evaluate_ready_voxels`, and it does not
care whether it is handed one block's 512 centres or a whole plan's. Every
vectorised path calls it, so they agree by construction; the tests then
check that they agree in fact.

Voxels are processed in chunks of 256 blocks. That is a cache decision: a
dozen temporaries of a 131,072-element pass stay resident where a
multi-million-element pass would not. A test fuses the same scan at three
other chunk sizes and requires identical storage.

Storage must be canonically empty on entry — weights zero, sums positive
zero — and any failure restores it to exactly that, which is trivially exact
because there is no earlier state to lose.

## What it is pinned against

| Path | Pinned against | Compared |
|---|---|---|
| block evaluator | scalar evaluator, every fixture block and frame | status, weight, sum bits per voxel |
| block evaluator | scalar evaluator, sampled room-scan blocks | the same |
| rare statuses | scalar metric-frame evaluator, hostile poses | the same |
| block-major fusion | scalar block and plan traversals | whole storage buffers |
| streaming fusion | scalar one-shot plan traversal (fixture) | whole storage buffers |
| streaming fusion | block-major fusion (room scan) | whole storage buffers |

The committed fixtures only ever produce three statuses: `contributes`,
`projection-outside-image` and `behind-truncation`. A non-finite camera
point or a non-positive camera depth needs a pose no plausible scan
contains, so those are compared against the scalar evaluator directly on
constructed inputs. `signed-distance-nonfinite` is unreachable in both
paths — by the time it is tested the depth and the camera z are both finite
and positive — and is kept only because the scalar path keeps it.

## Cost

The room fixture is 351 blocks at 40 mm over 20 frames, 3.6 million
voxel-observations:

```text
scalar, ledgered fusion     210 s
block-major field fusion    1.8 s
streaming fusion            0.3 s
```

All three produce the same 1,127,112 contributions over 81,292 voxels, and
the latter two the same bytes. On real 640x480 data streaming fusion runs at
about ten million voxel-observations per second on one core.

## What remains scalar

The scalar evaluator and traversals are still there and still define the
result; nothing above replaces them as the reference. The resumable fusion
ledgers also still run voxel by voxel, in memory, so fusion can be resumed
within a process but not at vector speed and not across runs.
