# How this project is built

SpatialForge advances in small, individually provable checkpoints rather than
in features. This page records that method, the invariants everything is
pinned to, and the conventions and traps that are easy to rediscover the hard
way. Read it before adding a checkpoint.

## Why checkpoints

TSDF reconstruction combines many independent risks at once: coordinate
systems, sensor association, file freshness, numerical precision, sparse
addressing, mutable storage, overflow, rollback, memory limits and
performance. Implemented together, a wrong surface could come from any of a
dozen causes and none of them would be locally testable.

So each checkpoint has: a precise contract, explicit non-goals, one narrow
implementation, a deterministic fixture, exact numerical expectations, failure
and malformed-input tests, side-effect and mutation checks, a read-only CLI
proof, documentation, a full regression run, and a commit.

## The rule that makes it work

**Pin every new layer to something that already exists and was derived
independently.** Not to a hand-computed expectation — to another code path.

Examples currently in the suite:

| new layer | pinned against |
|---|---|
| sparse TSDF | dense TSDF, byte-identical |
| context evaluator | session-backed evaluator, exact field equality |
| per-voxel sampling | reference contribution evaluator's accept rule |
| cross-view voxel | the fusing traversal's weight and float64 sum |
| cross-view block | block traversal's per-voxel `weight_after` / `tsdf_sum_after` |
| domain sweep | plan traversal's `weight_delta` and observed-voxel count |
| conservative coverage | centreline ray coverage must be a subset |
| resumable fusion | one-shot traversal, byte-identical at any chunk size |
| frame-major fusion | same, at any `(row, observation)` chunk size |

When a checkpoint cannot be pinned to an existing path, say so in its doc
rather than substituting a weaker assertion.

## The numbers everything agrees on

On `tests/fixtures/minimal.vgsession` with `voxel_size_m=0.125`,
`truncation_m=0.5`, `frame_stride=1`:

```text
plan               8 active blocks, 4 surface, 4096 voxel slots
fusion             1168 contributions applied over 584 slots, max weight 2
                   evaluated 8192  (8 blocks x 512 voxels x 2 observations)
single voxel       (8, -1, -1) -> tsdf_sum -0.125, weight 1 per observation
plan sha256        372c7c5d49eff1a30317ceb8b67cb3c40683f1d9d2a6179a049772f763d6f79d
replay sha256      dc001ae0ca01004a21ad227b12d57a7350bbc23904040453ca73fd955ef050b8
```

If a change moves `1168`, `584` or those digests, either the change is wrong
or it is a deliberate contract change that needs saying out loud.

## Receipt discipline

Every operation returns a frozen, slotted dataclass that validates itself in
`__post_init__`, and the validation **re-derives** rather than trusts:

- recompute the result from the retained inputs and require equality
  (the ray receipt re-runs its own DDA; the footprint receipt re-covers its
  own wedge; aggregate receipts recompute their union from their children);
- require children to be present in canonical order with matching provenance;
- store no aggregate that could drift — expose counts as properties derived
  from the children on access.

Where a retained value genuinely cannot be re-derived, document it rather than
pretending. `measured_depth_m` is the standing example: the receipts keep no
intrinsics or pose, so only its sign and finiteness are checkable.

## Conventions

- **Canonical block order** is x-fastest, then y, then z. `_ordered_blocks`
  produces it; validators reject anything else.
- **Canonical voxel order** within a block is local-flat `0..511`, x-fastest.
- **Provenance** travels as `source_plan_digest_sha256` + `replay_digest_sha256`
  on every receipt, and cross-object calls verify them before doing work.
- **Caps** are named constants, currently `262_144` outcomes and
  `100_000` blocks. Preflight before doing work; a limit failure returns no
  partial receipt.
- **CLI output** is `key: value` lines, one fact per line, including explicit
  negatives (`plan_expanded: no`) so a diagnostic states what it did *not* do.
- **Errors** are `TsdfError` with a message naming the fix. The CLI prints
  `<THING> FAILED <path>` to stderr, returns 2, and never shows a traceback.
- **Line width** is 79 columns in `spatialforge/`.
- Private helpers are imported across modules deliberately (`_ordered_blocks`,
  `_validate_trace_plan`) to keep one definition of a rule.

## Test conventions

- `python -W error -m unittest discover -s tests -p "test_*.py"` must end `OK`.
  Warnings are errors, so anything noisy fails.
- Tests assert **no filesystem change** via `tree_snapshot` before/after, and
  **no forbidden calls** by patching functions the operation must not use
  (replay, `Path.open`, `PIL.Image.open`) and asserting they were never hit.
- `tests/heavy_fixtures.py` caches expensive plan/context/coverage/domain
  resolutions once per run. Everything it hands out is frozen; storage is not,
  so allocate storage per test.
- `tests/room_fixture.py` generates the non-degenerate room scan from a fixed
  seed. Use it for anything about real-world behaviour.
- `tests/` is a package: import helpers as `from tests.x import y`.

## Traps already paid for

- **`minimal.vgsession` is degenerate.** 2x2 image, camera at the exact
  origin, surfaces exactly on voxel and block boundaries, depths exact binary
  fractions. It hides boundary bugs. New geometric work should also be checked
  against `room_fixture`.
- **`TSDF_SUM_DTYPE` is a dtype instance, not a callable.** Use
  `np.float64(0.0)` / `np.uint32(0)` to fill.
- **`TemporaryDirectory` registers a shutdown finalizer** that races `atexit`
  cleanup and emits a `ResourceWarning`, which `-W error` turns into a
  failure. Use `mkdtemp` + `shutil.rmtree` for anything cleaned up at exit.
- **Rewriting a session's depth files changes its digests**, so tests that
  assert the committed plan/replay sha256 must use the pristine fixture.
- **Git Bash `/tmp` is not Python's `/tmp` on Windows.** Pass real paths to
  Python.
- **The reconstruct dispatch is exhaustive** and raises on an unrouted
  command; `tests/test_cli_routing.py` asserts every registered subcommand is
  reachable. Add both parser entry and dispatch branch together.
- Float addition is commutative but **not associative**, so anything that
  changes per-voxel accumulation order changes the last bits. The observation
  ledger stores a canonical prefix specifically to avoid this.

## Health check

Worth running before a release or after a refactor:

- unused imports, `TODO`/`FIXME`, bare `except`
- every registered subcommand routes and parses `--help`
- no unresolved or duplicate names in `spatialforge.__all__`
- every markdown cross-reference resolves
- all plan+session diagnostics run clean against the fixture and leave
  `outputs/` untouched

## Definition of done for a checkpoint

1. Full suite `OK` under `-W error`.
2. The new behaviour pinned to an independently derived path.
3. A read-only CLI proof whose output is asserted verbatim in a test.
4. A doc in `docs/` stating the contract **and its non-goals**.
5. `docs/roadmap.md` moved forward, remaining items renumbered.
6. `README.md` updated with the command and what it shows.
7. One commit whose message explains *why*, not just what.
