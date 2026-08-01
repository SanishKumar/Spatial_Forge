# SpatialForge

SpatialForge is a standalone spatial mapping and localization engine. It will
turn calibrated indoor scans into metric, semantic, localizable maps while
remaining independent of navigation products such as VoiceGIS.

This repository currently implements these narrow foundations:

- a versioned, folder-backed `ScanSession` (`.vgsession`) contract;
- validation for calibration, timestamps, file references, depth scale, IMU
  samples, and rigid camera poses;
- deterministic offline replay around RGB observations;
- an extracted TUM RGB-D folder importer with known-pose support;
- calibrated, known-pose RGB-D back-projection to a deterministic colored PLY;
- fixed-bounds projective TSDF integration as a deterministic CPU reference;
- fixed-bounds TSDF integration with sparse in-memory accumulator state, dense
  traversal, and exact dense-reference parity;
- deterministic known-pose depth planning of candidate 8 x 8 x 8 voxel blocks
  around observed surfaces;
- strict immutable loading of `.sftplan` diagnostics and read-only verification
  of their current ScanSession replay binding;
- deterministic allocation of replay-matched candidate blocks into temporary,
  zeroed float64-sum and uint32-weight buffers;
- deterministic signed global-voxel addressing into planned block rows and
  local `(z, y, x)` array positions, without allocating missing blocks;
- read-only evaluation of one replay-selected observation at one planned
  voxel, returning an immutable projective TSDF sum/weight delta when exact
  depth and pose exist, or a skip diagnostic, without applying it;
- plan- and replay-bound application of one accepted contribution to exactly
  one addressed temporary storage slot, with an immutable before/after
  receipt and rollback on a caught post-write replay failure;
- deterministic traversal of every replay observation selected by the plan
  for one addressed voxel, evaluating all results before mutation and then
  accumulating accepted contributions in canonical observation order;
- plan- and replay-bound construction of a frozen in-memory
  `TsdfReplayDepthContext` for every observation selected by the block plan,
  decoding each ready aligned-depth frame exactly once into immutable,
  C-contiguous float64 metric storage;
- read-only evaluation of exactly one prepared context observation at one
  planned voxel, reusing its copied pose and immutable metric depth without
  evaluation-time replay hashing or depth decoding and returning the existing
  frozen contribution/skip contract without mutation;
- context-bound guarded application of one accepted prepared contribution to
  exactly one addressed temporary slot, using construction-time context
  provenance without application-time replay, source I/O, or depth access;
- context-backed traversal of every prepared plan-selected observation for one
  addressed voxel, evaluating the complete ordered transcript before mutation
  and then applying accepted contributions without traversal-time replay,
  source I/O, or depth decoding;
- context-backed traversal of all 512 X-fastest voxel addresses in one
  caller-selected planned block, retaining one complete observation transcript
  per voxel and restoring the selected block after a caught failure;
- deterministic, depth-derived world-aligned TSDF volume bounds;
- deterministic zero-crossing surface-point extraction from the TSDF; and
- deterministic six-tetrahedron reference triangle meshing.

The prepared replay/depth context is consumed by separate scalar evaluation,
guarded-application, one-voxel traversal, and one-selected-block traversal
APIs. The deliberately redundant session-backed scalar and one-voxel traversal
APIs remain available as reference paths. Traversal across the plan's block
rows, camera-to-surface free-space coverage, full block fusion, scalable sparse
execution, robust outlier filtering, production meshing, normals, SLAM, map
packages, mobile capture, and the visual inspector are deliberately not
implemented yet.

## Set up

Python 3.11 or newer is required.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
```

## Validate and replay a session

```powershell
.\.venv\Scripts\python.exe -m spatialforge scan validate `
  tests/fixtures/minimal.vgsession
.\.venv\Scripts\python.exe -m spatialforge scan replay `
  tests/fixtures/minimal.vgsession
```

Import the committed tiny TUM-layout fixture:

```powershell
.\.venv\Scripts\python.exe -m spatialforge scan import-tum `
  tests/fixtures/tum/rgbd_dataset_freiburg1_tiny `
  outputs/tum-tiny.vgsession
.\.venv\Scripts\python.exe -m spatialforge scan validate `
  outputs/tum-tiny.vgsession
```

## Build a known-pose point cloud

```powershell
.\.venv\Scripts\python.exe -m spatialforge reconstruct point-cloud `
  tests/fixtures/minimal.vgsession `
  outputs/minimal.ply
```

## Check current progress yourself

First, run the complete automated test suite:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The last line should be `OK`.

Then run the small numerical TSDF proof:

```powershell
.\.venv\Scripts\python.exe -m spatialforge reconstruct tsdf `
  tests/fixtures/minimal.vgsession `
  outputs/progress.sftsdf `
  --origin 0 -0.25 -0.25 `
  --dimensions 4 1 1 `
  --voxel-size-m 0.5 `
  --truncation-m 0.5
```

The important output is:

```text
voxels: total=4 observed=3 fused=3
voxel_updates: 6 max_weight=2
```

`max_weight=2` proves that both frames contributed to the same voxels. Inspect
the exact signed distances with:

```powershell
Get-Content outputs/progress.sftsdf
```

The three observed TSDF values should be `1.0`, `0.5`, and `-0.5`, each with
weight `2`. The sign change brackets the known plane at world `X=1.0 m`.

Extract that zero crossing:

```powershell
.\.venv\Scripts\python.exe -m spatialforge reconstruct surface-points `
  outputs/progress.sftsdf `
  outputs/progress-surface.ply
```

Expected:

```text
crossings: x=1 y=0 z=0
points: exact_zero=0 crossing=1 total=1
```

The final line of `outputs/progress-surface.ply` should be:

```text
1.000000000 0.000000000 0.000000000
```

The one-dimensional TSDF above proves the zero crossing but cannot contain a
triangle. Automatically infer a padded, world-aligned 3D volume from the known
depth and poses:

```powershell
.\.venv\Scripts\python.exe -m spatialforge reconstruct tsdf-auto `
  tests/fixtures/minimal.vgsession `
  outputs/progress-auto.sftsdf `
  --voxel-size-m 0.5 `
  --truncation-m 0.5
```

Expected:

```text
bounds_depth: valid=8 invalid=0
volume: origin=(0.500000000, -1.000000000, -1.000000000) dimensions=(2, 4, 4) voxels=32
integration: observed=8 fused=8 updates=16 max_weight=2
output_sha256: e61803737cdd68b209459fb644cc2f67f18e0420d306273316808e27d2e89994
```

Use those exact bounds with the first sparse-storage checkpoint:

```powershell
.\.venv\Scripts\python.exe -m spatialforge reconstruct tsdf-sparse `
  tests/fixtures/minimal.vgsession `
  outputs/progress-sparse.sftsdf `
  --origin 0.5 -1 -1 `
  --dimensions 2 4 4 `
  --voxel-size-m 0.5 `
  --truncation-m 0.5
```

Expected:

```text
voxels: total=32 observed=8 fused=8
voxel_updates: 16 max_weight=2
storage: sparse accumulator_entries=8
output_sha256: e61803737cdd68b209459fb644cc2f67f18e0420d306273316808e27d2e89994
```

The equal digest confirms byte parity for this fixture. The sparse command
stores sums and weights only for updated voxels, but deliberately retains the
bounded dense traversal in this checkpoint.

Plan candidate blocks for the eventual block-backed reconstruction path:

```powershell
.\.venv\Scripts\python.exe -m spatialforge reconstruct tsdf-block-plan `
  tests/fixtures/minimal.vgsession `
  outputs/progress-blocks.sftplan `
  --voxel-size-m 0.125 `
  --truncation-m 0.5
```

Expected:

```text
depth_samples: valid=8 invalid=0
grid: voxel_size_m=0.125000000 block_resolution=8 block_extent_m=1.000000000
candidate_blocks: surface=4 active=8 halo=4 voxel_slots=4096
block_bounds: min=(0, -1, -1) max=(1, 0, 0)
output_sha256: 372c7c5d49eff1a30317ceb8b67cb3c40683f1d9d2a6179a049772f763d6f79d
```

This `.sftplan` is a deterministic surface-neighborhood plan only. The narrow
address, contribution, update, one-voxel traversal, and one-selected-block
traversal diagnostics below consume it, but plan-wide fusion does not. It
deliberately does not plan the dense reference backend's full
camera-to-surface free-space updates.

Strictly load that artifact and check it against the current session replay:

```powershell
.\.venv\Scripts\python.exe -m spatialforge reconstruct tsdf-block-plan-verify `
  outputs/progress-blocks.sftplan `
  tests/fixtures/minimal.vgsession
```

The important status lines are:

```text
artifact: valid
session_replay: matched
geometry_recomputed: no
```

`artifact: valid` means the strict loader accepted the schema, types, ordering,
limits, and cross-field invariants. `session_replay: matched` means the
artifact's session ID and replay digest still match the current sensor inputs
and replay-derived frame metadata. `geometry_recomputed: no` is equally
important: verification does not decode depth again, regenerate block
coordinates, allocate TSDF blocks, or fuse any values.

The plan SHA-256 identifies the exact artifact bytes. It does not authenticate
the artifact or prove that its candidate geometry came from trusted planner
code. This verification command is read-only and writes no output file.

Allocate the verified candidate coordinates as empty in-memory TSDF block
storage:

```powershell
.\.venv\Scripts\python.exe -m spatialforge reconstruct tsdf-block-allocate `
  outputs/progress-blocks.sftplan `
  tests/fixtures/minimal.vgsession
```

Expected:

```text
artifact: valid
session_replay: matched
depth_decoded: no
geometry_recomputed: no
fusion_performed: no
artifact_written: no
allocation: blocks=8 resolution=8 voxel_slots=4096
layout: shape=(8, 8, 8, 8) axes=block-z-y-x x_fastest=yes
dtypes: tsdf_sums=float64 weights=uint32
zero_state: nonzero_sums=0 nonzero_weights=0 unknown_voxels=4096
payload_bytes: tsdf_sums=32768 weights=16384 total=49152
block_rows: first=(0, -1, -1) last=(1, 0, 0)
```

This command strict-loads and replay-verifies the plan, allocates the buffers,
reports their zero state, and discards them when the process exits. Replay
verification hashes sensor payload bytes but does not decode depth pixels.
The reference allocator caps numeric array payload at `64 MiB`; this does not
include Python, NumPy-header, allocator, or process-memory overhead. There is no
TSDF update, free-space decision, fusion, authentication, or output artifact in
this checkpoint.

Resolve signed global voxel indices in that temporary storage:

```powershell
.\.venv\Scripts\python.exe -m spatialforge reconstruct tsdf-block-address `
  outputs/progress-blocks.sftplan `
  tests/fixtures/minimal.vgsession `
  --voxel 7 -1 -1 `
  --voxel 8 0 0 `
  --voxel -1 0 0
```

Expected:

```text
depth_decoded: no
geometry_recomputed: no
fusion_performed: no
storage_mutated: no
addressing_created_blocks: no
artifact_written: no
allocation: blocks=8 voxel_slots=4096
queries: requested=3 resolved=2 unplanned=1
voxel[0]: status=planned global=(7, -1, -1) block=(0, -1, -1) local=(7, 7, 7) row=0 array=(0, 7, 7, 7) local_flat=511 storage_flat=511
voxel[1]: status=planned global=(8, 0, 0) block=(1, 0, 0) local=(0, 0, 0) row=7 array=(7, 0, 0, 0) local_flat=0 storage_flat=3584
voxel[2]: status=unplanned global=(-1, 0, 0)
```

The valid third query belongs to a block that is absent from the plan, so the
API returns `None` and the CLI reports `unplanned`. It does not insert a block.
Address resolution only computes indices; the zeroed sum and weight arrays are
unchanged and no output file is written.

Evaluate one resolved voxel against one selected known-pose depth observation:

```powershell
.\.venv\Scripts\python.exe -m spatialforge reconstruct tsdf-block-contribution `
  outputs/progress-blocks.sftplan `
  tests/fixtures/minimal.vgsession `
  --observation-sequence 0 `
  --voxel 8 -1 -1
```

Expected:

```text
TSDF BLOCK CONTRIBUTION CHECK scan-synthetic-0001
artifact: valid
session_replay: matched
observation_sequence: 0
voxel: global=(8, -1, -1) block=(1, -1, -1) local=(0, 7, 7) row=1 array=(1, 7, 7, 0) storage_flat=1016
world_xyz_m: (1.062500000, -0.062500000, -0.062500000)
camera_xyz_m: (0.062500000, 0.062500000, 1.062500000)
projected_uv: (0.617647059, 0.617647059)
pixel_uv: (1, 1)
depth_decoded: yes
measured_depth_m: 1.000000000
signed_distance_m: -0.062500000
evaluation: contributes
proposed_delta: tsdf_sum=-0.125000000 weight=1
contributions_applied: 0
fusion_performed: no
storage_mutated: no
missing_blocks_created: no
artifact_written: no
```

The voxel center is transformed into camera space, projected with the aligned
RGB intrinsics, and sampled at the nearest depth pixel. The measured depth of
`1.0 m` minus camera-space depth `1.0625 m` gives `-0.0625 m`; division by the
plan's `0.5 m` truncation gives the proposed TSDF sum delta `-0.125`. A proposed
weight of one describes what a later fusion operation could apply. This
checkpoint applies zero contributions: it evaluates no other observation or
voxel, runs no frame- or block-wide contribution loop, leaves both storage
arrays zero, and writes no artifact.

This single-voxel rule retains the reference TSDF sign and truncation
conventions, but it does not decide which camera-to-surface free-space blocks
should exist. This read-only command applies no delta. The one-slot command
and traversal commands below demonstrate separate mutation consumers; this
scalar command itself still visits no other observation, address, or block.
Free-space coverage, full fusion, and persistence remain later checkpoints.
The exact result and skip contract is documented in
[`docs/tsdf-voxel-contribution.md`](docs/tsdf-voxel-contribution.md).

Apply that accepted contribution to its one temporary storage slot:

```powershell
.\.venv\Scripts\python.exe -m spatialforge reconstruct tsdf-block-contribution-apply `
  outputs/progress-blocks.sftplan `
  tests/fixtures/minimal.vgsession `
  --observation-sequence 0 `
  --voxel 8 -1 -1
```

Relevant output:

```text
TSDF BLOCK CONTRIBUTION APPLY CHECK scan-synthetic-0001
artifact: valid
session_replay: matched
observation_sequence: 0
voxel: global=(8, -1, -1) block=(1, -1, -1) local=(0, 7, 7) row=1 array=(1, 7, 7, 0) storage_flat=1016
evaluation: contributes
slot_before: tsdf_sum=0.000000000 weight=0
applied_delta: tsdf_sum=-0.125000000 weight=1
slot_after: tsdf_sum=-0.125000000 weight=1
storage_before: nonzero_sums=0 nonzero_weights=0 unknown_voxels=4096
storage_after: nonzero_sums=1 nonzero_weights=1 unknown_voxels=4095
contributions_evaluated: 1
contributions_applied: 1
storage_slots_updated: 1
fusion_block_traversal_performed: no
ray_traversal_performed: no
full_fusion_performed: no
missing_blocks_created: no
artifact_written: no
storage_persisted: no
```

The immutable receipt returned by the updater records the exact transition
from sum/weight `(0, 0)`, through delta `(-0.125, 1)`, to `(-0.125, 1)`.
Only that addressed slot changes:
one previously unknown voxel becomes observed, so the nonzero counts become
one and the unknown count falls from 4,096 to 4,095. A weight of one is one
observation; it is not evidence of multi-view fusion.

The contribution carries the source plan and replay SHA-256 digests. Before
writing, the updater checks those values against the destination storage and
current session replay. It checks replay again after the scalar writes and
attempts to restore the prior slot values if a caught post-write failure or
replay change occurs; an explicit error warns if rollback itself fails. This is
an in-process rollback under exclusive access, not a persistent, crash-atomic,
thread-safe transaction.

There is no observation ledger: calling the API again with the same accepted
contribution applies the same delta again while the replay and accumulator
preconditions remain valid. Persistent per-observation duplicate tracking,
block/ray traversal, free-space planning, full fusion, normalization, and
persistent output remain deferred. The exact mutation and failure contract is
documented in
[`docs/tsdf-voxel-update.md`](docs/tsdf-voxel-update.md).

Traverse every observation selected by the plan for that same one voxel:

```powershell
.\.venv\Scripts\python.exe -m spatialforge reconstruct tsdf-block-voxel-traverse `
  outputs/progress-blocks.sftplan `
  tests/fixtures/minimal.vgsession `
  --voxel 8 -1 -1
```

Relevant output:

```text
TSDF BLOCK VOXEL TRAVERSAL CHECK scan-synthetic-0001
artifact: valid
session_replay: matched
voxel: global=(8, -1, -1) block=(1, -1, -1) local=(0, 7, 7) row=1 array=(1, 7, 7, 0) storage_flat=1016
selection: frame_stride=1 total=2 selected=2
slot_before: tsdf_sum=0.000000000 weight=0
observation[0]: sequence=0 status=contributes delta_sum=-0.125000000 delta_weight=1
observation[1]: sequence=1 status=contributes delta_sum=-0.125000000 delta_weight=1
status_counts: contributes=2
accumulated_delta: tsdf_sum=-0.250000000 weight=2
slot_after: tsdf_sum=-0.250000000 weight=2
storage_before: nonzero_sums=0 nonzero_weights=0 unknown_voxels=4096
storage_after: nonzero_sums=1 nonzero_weights=1 unknown_voxels=4095
contributions_evaluated: 2
contributions_applied: 2
contributions_skipped: 0
duplicate_observation_applications: 0
storage_slots_updated: 1
voxel_observation_traversal_performed: yes
voxel_address_traversal_performed: no
fusion_block_traversal_performed: no
ray_traversal_performed: no
full_fusion_performed: no
missing_blocks_created: no
artifact_written: no
storage_persisted: no
```

The committed plan selects observation sequences `0` and `1`. The traversal
evaluates both before applying either result. Their sequential float64 deltas
are `-0.125` and `-0.12499999999999978`, each with weight one, so the one fresh
target slot changes from raw sum/weight `(0, 0)` to
`(-0.24999999999999978, 2)`. Nine-decimal CLI formatting displays that sum as
`-0.250000000`. The derived normalized value is approximately `-0.125`, but
this checkpoint stores only the accumulator sum and weight. Exactly one voxel
becomes observed; its weight of two proves that two distinct selected
observation sequences contributed during this traversal.

The command reports selected, evaluated, contributing, skipped, and applied
counts together with the target's before/after state.
`voxel_observation_traversal_performed: yes` means it traversed the selected
observations for this address; `voxel_address_traversal_performed: no` and
`fusion_block_traversal_performed: no` preserve the one-address boundary. No
missing block is created and no artifact is written or persisted. The CLI
allocates fresh empty storage on each invocation and discards it at exit.

The traversal accepts only a canonical empty target. This is a coarse guard
against applying a second traversal to the same live slot, not a persistent
per-observation ledger. It evaluates all selected observations before the
first write and restores the target to its initial state after a caught
application-phase failure. That is bounded in-process rollback under exclusive
access, not crash atomicity or thread safety. The exact traversal, guard, and
rollback contract is documented in
[`docs/tsdf-voxel-traversal.md`](docs/tsdf-voxel-traversal.md).

Prepare the shared selected-observation replay/depth snapshot independently:

```powershell
.\.venv\Scripts\python.exe -m spatialforge reconstruct tsdf-block-replay-context `
  outputs\progress-blocks.sftplan `
  tests\fixtures\minimal.vgsession
```

Relevant output:

```text
TSDF BLOCK REPLAY DEPTH CONTEXT CHECK scan-synthetic-0001
artifact: valid
session_replay: matched
selection: frame_stride=1 total=2 selected=2
observation[0]: sequence=0 status=ready depth_decoded=yes
observation[1]: sequence=1 status=ready depth_decoded=yes
status_counts: ready=2
ready_observations: 2
depth_frames_decoded: 2
depth_layout: shape=(2, 2) dtype=float64 samples=8 payload_bytes=64
context_immutable: yes
tsdf_storage_allocated: no
voxel_evaluation_performed: no
voxel_observation_traversal_performed: no
voxel_address_traversal_performed: no
fusion_block_traversal_performed: no
ray_traversal_performed: no
full_fusion_performed: no
artifact_written: no
context_persisted: no
plan_sha256: 372c7c5d49eff1a30317ceb8b67cb3c40683f1d9d2a6179a049772f763d6f79d
replay_digest_sha256: dc001ae0ca01004a21ad227b12d57a7350bbc23904040453ca73fd955ef050b8
```

The plan selects sequences `0` and `1`. Both are ready, so the builder decodes
their aligned `2 x 2` depth payloads once. The immutable float64 metric frames
contain `1.0` and `0.9500000000000001`, respectively: eight samples and 64
retained numeric bytes in total. The builder brackets preparation with equal
replay digests and returns a frozen, bytes-backed in-memory snapshot. It does
not allocate TSDF storage, evaluate or update a voxel, traverse observations
for fusion, or write an artifact.

The replay digest proves the inputs at construction time. It does not monitor
the session folder afterward; later file changes cannot mutate an already
built context. Rebuild the context when current folder contents are required.
The retained numeric depth payload is capped at 512 MiB, although peak build
memory is higher because decoding temporarily holds the decoder result, a
mutable float64 frame, and its immutable byte copy. The context supports
read-only scalar evaluation, guarded one-slot application, and a context-backed
all-selected-observation traversal for one voxel and one selected block. The
exact snapshot, memory, and failure contract is documented in
[`docs/tsdf-replay-depth-context.md`](docs/tsdf-replay-depth-context.md).

Evaluate exactly one prepared observation at one planned voxel:

```powershell
.\.venv\Scripts\python.exe -m spatialforge reconstruct tsdf-block-context-contribution `
  outputs\progress-blocks.sftplan `
  tests\fixtures\minimal.vgsession `
  --observation-sequence 0 `
  --voxel 8 -1 -1
```

Relevant output:

```text
TSDF BLOCK CONTEXT CONTRIBUTION CHECK scan-synthetic-0001
artifact: valid
session_replay: matched
context_selection: frame_stride=1 total=2 selected=2
context_immutable: yes
depth_source: replay-depth-context
observation_sequence: 0
voxel: global=(8, -1, -1) block=(1, -1, -1) local=(0, 7, 7) row=1 array=(1, 7, 7, 0) storage_flat=1016
world_xyz_m: (1.062500000, -0.062500000, -0.062500000)
camera_xyz_m: (0.062500000, 0.062500000, 1.062500000)
projected_uv: (0.617647059, 0.617647059)
pixel_uv: (1, 1)
depth_decoded: yes
measured_depth_m: 1.000000000
signed_distance_m: -0.062500000
evaluation: contributes
proposed_delta: tsdf_sum=-0.125000000 weight=1
evaluation_replay_hashing: no
evaluation_depth_decoding: no
contributions_applied: 0
storage_mutated: no
voxel_observation_traversal_performed: no
voxel_address_traversal_performed: no
fusion_block_traversal_performed: no
full_fusion_performed: no
missing_blocks_created: no
artifact_written: no
context_persisted: no
plan_sha256: 372c7c5d49eff1a30317ceb8b67cb3c40683f1d9d2a6179a049772f763d6f79d
replay_digest_sha256: dc001ae0ca01004a21ad227b12d57a7350bbc23904040453ca73fd955ef050b8
```

This command still performs source I/O before evaluation: it replay-verifies
the plan, builds the full selected-observation context, decodes its ready depth
frames, and allocates replay-matched temporary storage. The two evaluation
labels apply only to the final one-observation scalar stage. That stage reads
the copied pose and immutable metric depth from the context, performs no replay
hashing or depth decode, evaluates no second observation or voxel, applies no
delta, and leaves storage unchanged. `depth_decoded: yes` means the selected
context record already contains decoded metric depth; it does not contradict
`evaluation_depth_decoding: no`.

The result exactly matches the session-backed fixture contribution, including
its `-0.125` sum delta and weight one. This proves numerical and diagnostic
parity for one address and one observation only.

Apply that accepted prepared contribution to exactly one temporary slot:

```powershell
.\.venv\Scripts\python.exe -m spatialforge reconstruct tsdf-block-context-contribution-apply `
  outputs\progress-blocks.sftplan `
  tests\fixtures\minimal.vgsession `
  --observation-sequence 0 `
  --voxel 8 -1 -1
```

Relevant output:

```text
TSDF BLOCK CONTEXT CONTRIBUTION APPLY CHECK scan-synthetic-0001
artifact: valid
session_replay: matched
context_selection: frame_stride=1 total=2 selected=2
context_immutable: yes
depth_source: replay-depth-context
observation_sequence: 0
voxel: global=(8, -1, -1) block=(1, -1, -1) local=(0, 7, 7) row=1 array=(1, 7, 7, 0) storage_flat=1016
world_xyz_m: (1.062500000, -0.062500000, -0.062500000)
camera_xyz_m: (0.062500000, 0.062500000, 1.062500000)
projected_uv: (0.617647059, 0.617647059)
pixel_uv: (1, 1)
depth_decoded: yes
measured_depth_m: 1.000000000
signed_distance_m: -0.062500000
evaluation: contributes
proposed_delta: tsdf_sum=-0.125000000 weight=1
evaluation_replay_hashing: no
evaluation_depth_decoding: no
slot_before: tsdf_sum=0.000000000 weight=0
applied_delta: tsdf_sum=-0.125000000 weight=1
slot_after: tsdf_sum=-0.125000000 weight=1
storage_before: nonzero_sums=0 nonzero_weights=0 unknown_voxels=4096
storage_after: nonzero_sums=1 nonzero_weights=1 unknown_voxels=4095
context_provenance: matched
application_source_freshness: construction-time-context
application_session_replay: no
application_replay_hashing: no
application_source_io: no
application_depth_access: no
contributions_evaluated: 1
contributions_applied: 1
storage_slots_updated: 1
voxel_observation_traversal_performed: no
voxel_address_traversal_performed: no
fusion_block_traversal_performed: no
ray_traversal_performed: no
full_fusion_performed: no
missing_blocks_created: no
artifact_written: no
storage_persisted: no
context_persisted: no
plan_sha256: 372c7c5d49eff1a30317ceb8b67cb3c40683f1d9d2a6179a049772f763d6f79d
replay_digest_sha256: dc001ae0ca01004a21ad227b12d57a7350bbc23904040453ca73fd955ef050b8
```

The command's setup stages still perform source work: the allocator
replay-verifies the plan, and context construction replay-brackets the
snapshot and decodes each ready selected depth frame once. The evaluation and
application labels describe only the two scalar stages after that setup.
Application accepts the frozen contribution and context, verifies their
plan/replay provenance against the destination storage, and changes only the
addressed slot. It does not replay or hash the session, read source files, or
access depth.

`application_source_freshness: construction-time-context` is a deliberate
boundary. The context proves the source state captured by its completed build;
the scalar updater does not detect later session-folder changes. It retains
the existing accumulator, overflow, exact-write, receipt, and rollback
guards, but has no post-write source-replay check to trigger rollback. The
existing session-backed updater remains unchanged. The exact contract is
documented in
[`docs/tsdf-context-voxel-update.md`](docs/tsdf-context-voxel-update.md).

Traverse every prepared selected observation for that same one voxel:

```powershell
.\.venv\Scripts\python.exe -m spatialforge reconstruct tsdf-block-context-voxel-traverse `
  outputs\progress-blocks.sftplan `
  tests\fixtures\minimal.vgsession `
  --voxel 8 -1 -1
```

Relevant output:

```text
TSDF BLOCK CONTEXT VOXEL TRAVERSAL CHECK scan-synthetic-0001
artifact: valid
session_replay: matched
context_selection: frame_stride=1 total=2 selected=2
context_immutable: yes
depth_source: replay-depth-context
voxel: global=(8, -1, -1) block=(1, -1, -1) local=(0, 7, 7) row=1 array=(1, 7, 7, 0) storage_flat=1016
selection: frame_stride=1 total=2 selected=2
slot_before: tsdf_sum=0.000000000 weight=0
observation[0]: sequence=0 status=contributes delta_sum=-0.125000000 delta_weight=1
observation[1]: sequence=1 status=contributes delta_sum=-0.125000000 delta_weight=1
status_counts: contributes=2
accumulated_delta: tsdf_sum=-0.250000000 weight=2
slot_after: tsdf_sum=-0.250000000 weight=2
storage_before: nonzero_sums=0 nonzero_weights=0 unknown_voxels=4096
storage_after: nonzero_sums=1 nonzero_weights=1 unknown_voxels=4095
context_provenance: matched
traversal_source_freshness: construction-time-context
traversal_session_replay: no
traversal_replay_hashing: no
traversal_source_io: no
traversal_depth_decoding: no
traversal_prepared_depth_access: yes
contributions_evaluated: 2
contributions_applied: 2
contributions_skipped: 0
duplicate_observation_applications: 0
storage_slots_updated: 1
voxel_observation_traversal_performed: yes
voxel_address_traversal_performed: no
fusion_block_traversal_performed: no
ray_traversal_performed: no
full_fusion_performed: no
missing_blocks_created: no
artifact_written: no
storage_persisted: no
context_persisted: no
plan_sha256: 372c7c5d49eff1a30317ceb8b67cb3c40683f1d9d2a6179a049772f763d6f79d
replay_digest_sha256: dc001ae0ca01004a21ad227b12d57a7350bbc23904040453ca73fd955ef050b8
```

The command still performs setup work before the traversal: temporary block
allocation replay-verifies the plan, and context construction replay-brackets
the snapshot and decodes each ready selected frame once. The traversal itself
accepts no `ScanSession`; it reads prepared metric depth while evaluating ready
records but performs no new replay, hashing, source I/O, or decoding. It
evaluates every selected record before the first write, applies accepted
contributions in canonical order with sequential float64 accumulation, and
restores the whole target slot after a caught application-phase failure. The
empty-target precondition remains a coarse duplicate guard rather than a
persistent observation ledger.

`traversal_prepared_depth_access: yes` is intentional: ready observations must
sample the metric depth already stored in the immutable context. It does not
contradict `traversal_source_io: no` or `traversal_depth_decoding: no`.

This proves context-backed observation traversal for exactly one caller-chosen
voxel. It is also the child primitive used by the one-selected-block checkpoint
below. See [`docs/tsdf-voxel-traversal.md`](docs/tsdf-voxel-traversal.md).

Traverse every voxel in exactly one selected planned block:

```powershell
.\.venv\Scripts\python.exe -m spatialforge reconstruct tsdf-block-context-block-traverse `
  outputs\progress-blocks.sftplan `
  tests\fixtures\minimal.vgsession `
  --block 1 -1 -1
```

Key output for the committed fixture is:

```text
block: index=(1, -1, -1) row=1 resolution=8 voxel_slots=512
storage_flat_range: 512..1023
address_order: local-flat-x-fastest local_flat=0..511
first_voxel: global=(8, -8, -8) local=(0, 0, 0) array=(1, 0, 0, 0) storage_flat=512
last_voxel: global=(15, -1, -1) local=(7, 7, 7) array=(1, 7, 7, 7) storage_flat=1023
block_before: nonzero_sums=0 nonzero_weights=0 unknown_voxels=512
block_after: nonzero_sums=102 nonzero_weights=102 unknown_voxels=410
storage_before: nonzero_sums=0 nonzero_weights=0 unknown_voxels=4096
storage_after: nonzero_sums=102 nonzero_weights=102 unknown_voxels=3994
status_counts: contributes=204 projection-outside-image=424 behind-truncation=396
block_weight_sum_after: 204
block_max_weight_after: 2
traversal_session_replay: no
traversal_replay_hashing: no
traversal_source_io: no
traversal_depth_decoding: no
traversal_prepared_depth_access: yes
voxel_addresses_traversed: 512
voxel_transcripts_retained: 512
voxel_observation_traversals: 512
contributions_evaluated: 1024
contributions_applied: 204
contributions_skipped: 820
storage_slots_updated: 102
blocks_traversed: 1
additional_blocks_visited: 0
voxel_observation_traversal_performed: yes
voxel_address_traversal_performed: yes
selected_block_traversal_performed: yes
fusion_block_traversal_performed: yes
fusion_block_traversal_scope: selected-planned-block-only
multiple_block_traversal_performed: no
planned_block_set_traversal_performed: no
free_space_coverage_planned: no
full_fusion_performed: no
missing_blocks_created: no
caught_failure_rollback_scope: selected-block
artifact_written: no
storage_persisted: no
```

The traversal derives all local-flat positions `0..511`, with X changing
fastest, and retains a complete one-voxel receipt for every address. Each child
evaluates both selected observations before changing its own slot. Across the
block, this yields 1,024 contribution outcomes: 204 accepted and applied, 820
skipped, and 102 updated voxel slots. Every other allocated block remains
unchanged.

The selected block must be canonically empty before the first child starts. A
caught failure at a later address restores the complete selected block to its
starting bytes. This is bounded one-block in-process rollback under exclusive
access, not crash atomicity, thread safety, persistence, or an observation
ledger. Evaluate-all-before-apply remains a per-voxel guarantee; earlier
voxels may already be updated when a later voxel is evaluated.

The command's setup replay-verifies allocation and builds the immutable depth
context. The later block traversal itself performs no replay, hashing, source
I/O, or depth decoding, although ready evaluations read prepared metric depth.
It targets no second block, creates no missing block, makes no free-space,
visibility, or culling decision, performs no plan-wide/full fusion, and writes
no artifact. The exact contract is documented in
[`docs/tsdf-context-block-traversal.md`](docs/tsdf-context-block-traversal.md).

The `fusion_block_traversal_performed: yes` label means the contribution path
covered all 512 slots of this one selected block. Its adjacent scope and
multiple-block labels are why that does not mean full fusion.

Traversing the plan's existing block rows is the next checkpoint. Designing
camera-to-surface free-space coverage, an explicit observation/idempotency
policy, and complete fusion remain separate later phases.

Both TSDF artifacts use the same `.sftsdf` contract. Mesh the sparse result
directly:

```powershell
.\.venv\Scripts\python.exe -m spatialforge reconstruct triangle-mesh `
  outputs/progress-sparse.sftsdf `
  outputs/progress-sparse-mesh.ply
```

Expected:

```text
cells: total=9 eligible=1 active=1
skipped_cells: unknown=8 exact_zero=0
mesh: vertices=9 triangles=8 boundary_edges=8
```

All mesh vertices lie on `X=1.0 m`; its eight triangles cover a
`0.5 m x 0.5 m` square and face the positive/free-space side. The exact
meshing contract is documented in `docs/triangle-mesh.md`.

For a visual check, open `outputs/minimal.ply` from the point-cloud command in
a PLY viewer. You can also open `outputs/progress-sparse-mesh.ply` in a viewer
that supports PLY faces. Commands refuse to overwrite outputs, so delete an
old diagnostic or choose a new filename before rerunning it.

The format and coordinate conventions are documented in
[`docs/scan-session-v0.md`](docs/scan-session-v0.md). TUM-specific conversion
rules are in [`docs/tum-import.md`](docs/tum-import.md), and the current
reconstruction steps are in
[`docs/known-pose-point-cloud.md`](docs/known-pose-point-cloud.md) and
[`docs/reference-tsdf.md`](docs/reference-tsdf.md). Automatic volume selection
is in [`docs/automatic-tsdf-bounds.md`](docs/automatic-tsdf-bounds.md), sparse
accumulation is in [`docs/sparse-tsdf.md`](docs/sparse-tsdf.md), surface
TSDF block planning is in
[`docs/tsdf-block-plan.md`](docs/tsdf-block-plan.md),
empty block allocation is in
[`docs/tsdf-block-storage.md`](docs/tsdf-block-storage.md),
signed voxel addressing is in
[`docs/tsdf-voxel-addressing.md`](docs/tsdf-voxel-addressing.md),
single-observation voxel evaluation is in
[`docs/tsdf-voxel-contribution.md`](docs/tsdf-voxel-contribution.md),
single-slot temporary voxel mutation is in
[`docs/tsdf-voxel-update.md`](docs/tsdf-voxel-update.md),
context-bound single-slot mutation is in
[`docs/tsdf-context-voxel-update.md`](docs/tsdf-context-voxel-update.md),
single-voxel traversal across selected observations is in
[`docs/tsdf-voxel-traversal.md`](docs/tsdf-voxel-traversal.md),
single selected-block context traversal is in
[`docs/tsdf-context-block-traversal.md`](docs/tsdf-context-block-traversal.md),
immutable selected-observation replay/depth preparation is in
[`docs/tsdf-replay-depth-context.md`](docs/tsdf-replay-depth-context.md),
surface extraction is in [`docs/surface-points.md`](docs/surface-points.md),
reference triangle meshing is in
[`docs/triangle-mesh.md`](docs/triangle-mesh.md), and overall status is in
[`docs/roadmap.md`](docs/roadmap.md).
