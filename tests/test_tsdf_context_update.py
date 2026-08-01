from __future__ import annotations

import io
import json
import shutil
import tempfile
import unittest
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np

from spatialforge import (
    MAX_TSDF_VOXEL_WEIGHT,
    allocate_empty_tsdf_blocks,
    apply_tsdf_voxel_contribution,
    apply_tsdf_voxel_contribution_from_context,
    build_tsdf_replay_depth_context,
    evaluate_tsdf_voxel_contribution,
    evaluate_tsdf_voxel_contribution_from_context,
    locate_tsdf_voxel,
)
from spatialforge.cli import main
from spatialforge.errors import TsdfError
from spatialforge.model import ScanSession
from spatialforge.session_loader import load_scan_session
from spatialforge.tsdf_block_plan import plan_tsdf_blocks
from spatialforge.tsdf_block_plan_loader import (
    TsdfBlockPlan,
    load_tsdf_block_plan,
)
from spatialforge.tsdf_block_storage import TsdfBlockStorage
from spatialforge.tsdf_replay_depth_context import (
    TsdfReplayDepthContext,
    TsdfReplayDepthObservation,
    TsdfReplayDepthStatus,
)
from spatialforge.tsdf_voxel_address import TsdfVoxelAddress
TEST_ROOT = Path(__file__).resolve().parent
FIXTURE = TEST_ROOT / "fixtures" / "minimal.vgsession"
PLAN_ARGUMENTS = {
    "voxel_size_m": 0.125,
    "truncation_m": 0.5,
}
PLAN_SHA256 = "372c7c5d49eff1a30317ceb8b67cb3c40683f1d9d2a6179a049772f763d6f79d"
REPLAY_SHA256 = "dc001ae0ca01004a21ad227b12d57a7350bbc23904040453ca73fd955ef050b8"

_FORBIDDEN_APPLICATION_TARGETS = (
    "spatialforge.tsdf_voxel_update.replay_session",
    "spatialforge.tsdf_voxel_contribution.replay_session",
    "spatialforge.tsdf_replay_depth_context.replay_session",
    "spatialforge.tsdf_voxel_contribution._read_depth",
    "spatialforge.tsdf_replay_depth_context._read_depth",
    "spatialforge.tsdf_voxel_contribution._sample_path",
    "spatialforge.tsdf_replay_depth_context._sample_path",
    "spatialforge.tsdf_voxel_contribution."
    "_validate_reconstruction_contract",
    "spatialforge.tsdf_replay_depth_context."
    "_validate_reconstruction_contract",
    "spatialforge.replay._file_digest",
    "hashlib.sha256",
    "pathlib.Path.open",
    "PIL.Image.open",
    "spatialforge.tsdf_voxel_contribution."
    "evaluate_tsdf_voxel_contribution",
    "spatialforge.tsdf_voxel_contribution."
    "evaluate_tsdf_voxel_contribution_from_context",
    "spatialforge.tsdf_replay_depth_context."
    "build_tsdf_replay_depth_context",
    "spatialforge.tsdf_voxel_update.apply_tsdf_voxel_contribution",
    "spatialforge.tsdf_voxel_traversal."
    "traverse_tsdf_voxel_observations",
    "spatialforge.tsdf_block_storage.allocate_empty_tsdf_blocks",
)

_FORBIDDEN_CONTEXT_APPLY_CLI_TARGETS = (
    "spatialforge.cli.evaluate_tsdf_voxel_contribution",
    "spatialforge.cli.apply_tsdf_voxel_contribution",
    "spatialforge.cli.traverse_tsdf_voxel_observations",
    "spatialforge.cli.reconstruct_point_cloud",
    "spatialforge.cli.integrate_tsdf",
    "spatialforge.cli.integrate_sparse_tsdf",
    "spatialforge.cli.infer_tsdf_bounds",
    "spatialforge.cli.plan_tsdf_blocks",
    "spatialforge.cli.extract_surface_points",
    "spatialforge.cli.extract_triangle_mesh",
    "spatialforge.tsdf._integrate_tsdf",
    "spatialforge.tsdf_block_plan._plan_observation_blocks",
    "spatialforge.tsdf._write_output_without_overwrite",
    "spatialforge.tsdf_block_plan._write_plan_without_overwrite",
    "spatialforge.surface._write_surface_ply",
    "spatialforge.mesh._write_mesh_ply",
)


def copy_fixture(parent: Path, name: str = "case.vgsession") -> Path:
    target = parent / name
    shutil.copytree(FIXTURE, target)
    return target


def create_plan(
    parent: Path,
    *,
    session_path: Path = FIXTURE,
    name: str = "fixture.sftplan",
    frame_stride: int = 1,
) -> Path:
    output = parent / name
    plan_tsdf_blocks(
        load_scan_session(session_path),
        output,
        frame_stride=frame_stride,
        **PLAN_ARGUMENTS,
    )
    return output


def load_case(
    parent: Path,
    *,
    session_path: Path = FIXTURE,
    plan_name: str = "fixture.sftplan",
    frame_stride: int = 1,
) -> tuple[
    TsdfBlockPlan,
    ScanSession,
    TsdfBlockStorage,
    TsdfReplayDepthContext,
]:
    plan = load_tsdf_block_plan(
        create_plan(
            parent,
            session_path=session_path,
            name=plan_name,
            frame_stride=frame_stride,
        )
    )
    session = load_scan_session(session_path)
    storage = allocate_empty_tsdf_blocks(plan, session)
    context = build_tsdf_replay_depth_context(plan, session)
    return plan, session, storage, context


def require_address(
    storage: TsdfBlockStorage,
    global_index_xyz: tuple[int, int, int],
) -> TsdfVoxelAddress:
    address = locate_tsdf_voxel(storage, global_index_xyz)
    if address is None:
        raise AssertionError(
            f"expected planned address for {global_index_xyz}"
        )
    return address


def remove_first_record(path: Path) -> None:
    records = path.read_text(encoding="utf-8").splitlines()[1:]
    path.write_text(
        "".join(record + "\n" for record in records),
        encoding="utf-8",
    )


def set_first_pose_translation_x(session_path: Path, value: float) -> None:
    pose_path = session_path / "streams" / "poses.jsonl"
    records = [
        json.loads(line)
        for line in pose_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    records[0]["T_world_camera"][3] = value
    pose_path.write_text(
        "".join(
            json.dumps(record, separators=(",", ":")) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def tree_snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def storage_layout_snapshot(storage: TsdfBlockStorage) -> tuple[object, ...]:
    return (
        storage.block_indices,
        id(storage.tsdf_sums),
        id(storage.weights),
        storage.tsdf_sums.shape,
        storage.weights.shape,
        storage.tsdf_sums.strides,
        storage.weights.strides,
        storage.tsdf_sums.dtype.str,
        storage.weights.dtype.str,
        bool(storage.tsdf_sums.flags.c_contiguous),
        bool(storage.weights.flags.c_contiguous),
        bool(storage.tsdf_sums.flags.writeable),
        bool(storage.weights.flags.writeable),
        id(storage.tsdf_sums.base),
        id(storage.weights.base),
        int(storage.tsdf_sums.ctypes.data),
        int(storage.weights.ctypes.data),
    )


def storage_bytes(storage: TsdfBlockStorage) -> tuple[bytes, bytes]:
    return storage.tsdf_sums.tobytes(), storage.weights.tobytes()


def context_snapshot(context: TsdfReplayDepthContext) -> tuple[object, ...]:
    observations: list[object] = []
    for observation in context.observations:
        depth = observation.depth_m
        observations.append(
            (
                observation.observation_sequence,
                observation.status,
                observation.t_world_camera,
                None
                if depth is None
                else (
                    depth.shape,
                    depth.dtype.str,
                    depth.strides,
                    bool(depth.flags.c_contiguous),
                    bool(depth.flags.owndata),
                    bool(depth.flags.writeable),
                    depth.tobytes(),
                ),
            )
        )
    return (
        context.session_id,
        context.source_plan_digest_sha256,
        context.replay_digest_sha256,
        context.frame_stride,
        context.total_observations,
        context.selected_observation_sequences,
        context.camera,
        context.depth_scale_m,
        tuple(observations),
        context.valid_depth_samples,
        context.invalid_depth_samples,
    )


@contextmanager
def forbidden_application_calls():
    with ExitStack() as stack:
        calls: list[MagicMock] = []
        for target in _FORBIDDEN_APPLICATION_TARGETS:
            calls.append(stack.enter_context(patch(target)))
        yield calls


@contextmanager
def forbidden_context_apply_cli_calls():
    with ExitStack() as stack:
        calls: list[MagicMock] = []
        for target in _FORBIDDEN_CONTEXT_APPLY_CLI_TARGETS:
            calls.append(stack.enter_context(patch(target)))
        yield calls


class _CountingTuple(tuple):
    def __new__(cls, values):
        instance = super().__new__(cls, values)
        instance.iterations = 0
        instance.lookups = 0
        return instance

    def __iter__(self):
        self.iterations += 1
        return super().__iter__()

    def __getitem__(self, index):
        self.lookups += 1
        return super().__getitem__(index)


class TsdfContextUpdateTests(unittest.TestCase):
    def assert_only_flat_value_changed(
        self,
        before: bytes,
        after: bytes,
        flat_index: int,
        dtype: np.dtype,
        expected_value: float | int,
    ) -> None:
        start = flat_index * dtype.itemsize
        end = start + dtype.itemsize
        self.assertEqual(after[:start], before[:start])
        self.assertEqual(after[end:], before[end:])
        self.assertEqual(
            after[start:end],
            np.asarray(expected_value, dtype=dtype).tobytes(),
        )

    def assert_apply_fails_unchanged(
        self,
        tracked_storage: TsdfBlockStorage,
        storage: object,
        contribution: object,
        context: object,
        expected_message: str,
    ) -> None:
        before_layout = storage_layout_snapshot(tracked_storage)
        before_bytes = storage_bytes(tracked_storage)
        with (
            patch.object(
                TsdfReplayDepthObservation,
                "depth_m",
                new_callable=PropertyMock,
                side_effect=AssertionError(
                    "context update accessed prepared depth"
                ),
            ),
            forbidden_application_calls() as forbidden,
        ):
            with self.assertRaises(TsdfError) as raised:
                apply_tsdf_voxel_contribution_from_context(
                    storage,  # type: ignore[arg-type]
                    contribution,  # type: ignore[arg-type]
                    context,  # type: ignore[arg-type]
                )
        self.assertIn(expected_message, str(raised.exception))
        self.assertEqual(
            storage_layout_snapshot(tracked_storage),
            before_layout,
        )
        self.assertEqual(storage_bytes(tracked_storage), before_bytes)
        for forbidden_call in forbidden:
            forbidden_call.assert_not_called()

    def test_fixture_matches_legacy_exactly_with_one_o1_slot_update(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            plan, session, legacy_storage, context = load_case(temporary_root)
            context_storage = allocate_empty_tsdf_blocks(plan, session)
            legacy_address = require_address(
                legacy_storage,
                (8, -1, -1),
            )
            context_address = require_address(
                context_storage,
                (8, -1, -1),
            )
            self.assertEqual(context_address, legacy_address)

            for storage in (legacy_storage, context_storage):
                storage.tsdf_sums[0, 0, 0, 0] = 0.375
                storage.weights[0, 0, 0, 0] = 7

            legacy_contribution = evaluate_tsdf_voxel_contribution(
                legacy_storage,
                legacy_address,
                session,
                0,
            )
            context_contribution = (
                evaluate_tsdf_voxel_contribution_from_context(
                    context_storage,
                    context_address,
                    context,
                    0,
                )
            )
            self.assertEqual(context_contribution, legacy_contribution)
            legacy_receipt = apply_tsdf_voxel_contribution(
                legacy_storage,
                legacy_contribution,
                session,
            )

            counted_observations = _CountingTuple(context.observations)
            counted_context = replace(
                context,
                observations=counted_observations,
            )
            context_before = context_snapshot(counted_context)
            counted_observations.iterations = 0
            counted_observations.lookups = 0
            before_layout = storage_layout_snapshot(context_storage)
            before_sums, before_weights = storage_bytes(context_storage)
            before_tree = tree_snapshot(temporary_root)

            with (
                patch.object(
                    TsdfReplayDepthObservation,
                    "depth_m",
                    new_callable=PropertyMock,
                    side_effect=AssertionError(
                        "context update accessed prepared depth"
                    ),
                ) as depth_property,
                forbidden_application_calls() as forbidden,
            ):
                context_receipt = (
                    apply_tsdf_voxel_contribution_from_context(
                        context_storage,
                        context_contribution,
                        counted_context,
                    )
                )
            application_iterations = counted_observations.iterations
            application_lookups = counted_observations.lookups
            after_sums, after_weights = storage_bytes(context_storage)
            after_tree = tree_snapshot(temporary_root)
            context_after = context_snapshot(counted_context)

        self.assertEqual(context_receipt, legacy_receipt)
        self.assertIs(context_receipt.contribution, context_contribution)
        self.assertEqual(
            (
                context_receipt.tsdf_sum_before,
                context_receipt.weight_before,
                context_receipt.tsdf_sum_after,
                context_receipt.weight_after,
            ),
            (0.0, 0, -0.125, 1),
        )
        self.assertEqual(storage_bytes(context_storage), storage_bytes(legacy_storage))
        self.assertEqual(
            storage_layout_snapshot(context_storage),
            before_layout,
        )
        self.assert_only_flat_value_changed(
            before_sums,
            after_sums,
            1016,
            np.dtype(np.float64),
            -0.125,
        )
        self.assert_only_flat_value_changed(
            before_weights,
            after_weights,
            1016,
            np.dtype(np.uint32),
            1,
        )
        self.assertEqual(
            float(context_storage.tsdf_sums[0, 0, 0, 0]),
            0.375,
        )
        self.assertEqual(int(context_storage.weights[0, 0, 0, 0]), 7)
        self.assertEqual(context_before, context_after)
        self.assertEqual(after_tree, before_tree)
        self.assertEqual(application_iterations, 0)
        self.assertEqual(application_lookups, 1)
        depth_property.assert_not_called()
        for forbidden_call in forbidden:
            forbidden_call.assert_not_called()

    def test_nonempty_target_repeats_with_exact_legacy_parity(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            plan, session, legacy_storage, context = load_case(temporary_root)
            context_storage = allocate_empty_tsdf_blocks(plan, session)
            legacy_address = require_address(legacy_storage, (8, -1, -1))
            context_address = require_address(context_storage, (8, -1, -1))
            index = context_address.array_index_bzyx
            for storage in (legacy_storage, context_storage):
                storage.tsdf_sums[index] = -0.125
                storage.weights[index] = 1

            legacy_contribution = evaluate_tsdf_voxel_contribution(
                legacy_storage,
                legacy_address,
                session,
                0,
            )
            context_contribution = (
                evaluate_tsdf_voxel_contribution_from_context(
                    context_storage,
                    context_address,
                    context,
                    0,
                )
            )
            legacy_receipt = apply_tsdf_voxel_contribution(
                legacy_storage,
                legacy_contribution,
                session,
            )
            with forbidden_application_calls() as forbidden:
                context_receipt = (
                    apply_tsdf_voxel_contribution_from_context(
                        context_storage,
                        context_contribution,
                        context,
                    )
                )

        self.assertEqual(context_receipt, legacy_receipt)
        self.assertEqual(
            (
                context_receipt.tsdf_sum_before,
                context_receipt.weight_before,
                context_receipt.tsdf_sum_after,
                context_receipt.weight_after,
            ),
            (-0.125, 1, -0.25, 2),
        )
        self.assertEqual(storage_bytes(context_storage), storage_bytes(legacy_storage))
        for forbidden_call in forbidden:
            forbidden_call.assert_not_called()

    def test_rejects_types_provenance_selection_and_address_atomically(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            plan, session, storage, context = load_case(temporary_root)
            address = require_address(storage, (8, -1, -1))
            contribution = evaluate_tsdf_voxel_contribution_from_context(
                storage,
                address,
                context,
                0,
            )
            skipped = evaluate_tsdf_voxel_contribution_from_context(
                storage,
                require_address(storage, (0, 7, -1)),
                context,
                0,
            )
            forged_address = replace(
                address,
                storage_flat_index=address.storage_flat_index + 1,
            )
            stride_context = replace(
                context,
                frame_stride=2,
                selected_observation_sequences=(0,),
                observations=(context.observations[0],),
                valid_depth_samples=4,
            )
            cases = (
                (
                    "storage-type",
                    object(),
                    contribution,
                    context,
                    "allocated TsdfBlockStorage",
                ),
                (
                    "contribution-type",
                    storage,
                    object(),
                    context,
                    "TsdfVoxelContribution",
                ),
                (
                    "context-type",
                    storage,
                    contribution,
                    object(),
                    "TsdfReplayDepthContext",
                ),
                (
                    "skipped",
                    storage,
                    skipped,
                    context,
                    "accepted finite contribution",
                ),
                (
                    "address",
                    storage,
                    replace(contribution, address=forged_address),
                    context,
                    "address does not match",
                ),
                (
                    "contribution-plan",
                    storage,
                    replace(
                        contribution,
                        source_plan_digest_sha256="0" * 64,
                    ),
                    context,
                    "source plan does not match destination",
                ),
                (
                    "contribution-replay",
                    storage,
                    replace(
                        contribution,
                        replay_digest_sha256="0" * 64,
                    ),
                    context,
                    "replay digest does not match destination",
                ),
                (
                    "context-plan",
                    storage,
                    contribution,
                    replace(
                        context,
                        source_plan_digest_sha256="0" * 64,
                    ),
                    "source plan digest",
                ),
                (
                    "context-replay",
                    storage,
                    contribution,
                    replace(context, replay_digest_sha256="0" * 64),
                    "replay digest",
                ),
                (
                    "context-session",
                    storage,
                    contribution,
                    replace(context, session_id="different-session"),
                    "session_id",
                ),
                (
                    "context-selection",
                    storage,
                    contribution,
                    stride_context,
                    "selection",
                ),
                (
                    "outside-context",
                    storage,
                    replace(contribution, observation_sequence=2),
                    context,
                    "outside replay/depth context",
                ),
            )
            for (
                name,
                changed_storage,
                changed_contribution,
                changed_context,
                message,
            ) in cases:
                with self.subTest(name=name):
                    self.assert_apply_fails_unchanged(
                        storage,
                        changed_storage,
                        changed_contribution,
                        changed_context,
                        message,
                    )

            sample_plan = replace(
                plan,
                valid_depth_points=7,
                invalid_depth_samples=1,
            )
            sample_storage = TsdfBlockStorage(
                source_plan=sample_plan,
                block_indices=storage.block_indices,
                tsdf_sums=storage.tsdf_sums.copy(),
                weights=storage.weights.copy(),
            )
            self.assert_apply_fails_unchanged(
                sample_storage,
                sample_storage,
                contribution,
                context,
                "sample counts",
            )

            _, _, stride_storage, selected_context = load_case(
                temporary_root,
                plan_name="stride.sftplan",
                frame_stride=2,
            )
            selected_address = require_address(
                stride_storage,
                (8, -1, -1),
            )
            selected_contribution = (
                evaluate_tsdf_voxel_contribution_from_context(
                    stride_storage,
                    selected_address,
                    selected_context,
                    0,
                )
            )
            self.assert_apply_fails_unchanged(
                stride_storage,
                stride_storage,
                replace(selected_contribution, observation_sequence=1),
                selected_context,
                "not selected by context",
            )

            missing_fixture = copy_fixture(
                temporary_root,
                "missing.vgsession",
            )
            remove_first_record(
                missing_fixture / "streams" / "depth.jsonl"
            )
            (
                missing_plan,
                _,
                missing_storage,
                missing_context,
            ) = load_case(
                temporary_root,
                session_path=missing_fixture,
                plan_name="missing.sftplan",
            )
            self.assertEqual(
                missing_context.observations[0].status,
                TsdfReplayDepthStatus.MISSING_DEPTH,
            )
            missing_address = require_address(
                missing_storage,
                (8, -1, -1),
            )
            forged_ready = replace(
                contribution,
                address=missing_address,
                source_plan_digest_sha256=(
                    missing_plan.artifact_digest_sha256
                ),
                replay_digest_sha256=missing_plan.replay_digest_sha256,
            )
            self.assert_apply_fails_unchanged(
                missing_storage,
                missing_storage,
                forged_ready,
                missing_context,
                "requires a ready observation",
            )

    def test_rejects_noncanonical_storage_layout_atomically(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            plan, session, source, context = load_case(temporary_root)
            address = require_address(source, (8, -1, -1))
            contribution = evaluate_tsdf_voxel_contribution_from_context(
                source,
                address,
                context,
                0,
            )

            for name, array_name in (
                ("sums-readonly", "tsdf_sums"),
                ("weights-readonly", "weights"),
            ):
                with self.subTest(name=name):
                    storage = allocate_empty_tsdf_blocks(plan, session)
                    getattr(storage, array_name).setflags(write=False)
                    self.assert_apply_fails_unchanged(
                        storage,
                        storage,
                        contribution,
                        context,
                        "writable sum and weight arrays",
                    )

            backing = np.zeros(source.tsdf_sums.nbytes, dtype=np.uint8)
            aliased = TsdfBlockStorage(
                source_plan=plan,
                block_indices=source.block_indices,
                tsdf_sums=np.ndarray(
                    shape=source.tsdf_sums.shape,
                    dtype=np.float64,
                    buffer=backing,
                ),
                weights=np.ndarray(
                    shape=source.weights.shape,
                    dtype=np.uint32,
                    buffer=backing,
                ),
            )
            self.assert_apply_fails_unchanged(
                aliased,
                aliased,
                contribution,
                context,
                "non-overlapping storage arrays",
            )

            wrong_shape = allocate_empty_tsdf_blocks(plan, session)
            wrong_shape.tsdf_sums.shape = (wrong_shape.tsdf_sums.size,)
            self.assert_apply_fails_unchanged(
                wrong_shape,
                wrong_shape,
                contribution,
                context,
                "canonical",
            )

    def test_rejects_corrupt_prestates_and_uint32_max_atomically(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            plan, session, source, context = load_case(temporary_root)
            address = require_address(source, (8, -1, -1))
            contribution = evaluate_tsdf_voxel_contribution_from_context(
                source,
                address,
                context,
                0,
            )
            index = address.array_index_bzyx

            def nan_sum(storage: TsdfBlockStorage) -> None:
                storage.tsdf_sums[index] = np.nan

            def infinite_sum(storage: TsdfBlockStorage) -> None:
                storage.tsdf_sums[index] = np.inf

            def negative_zero(storage: TsdfBlockStorage) -> None:
                storage.tsdf_sums[index] = -0.0

            def nonzero_unknown(storage: TsdfBlockStorage) -> None:
                storage.tsdf_sums[index] = 0.25

            def outside_envelope(storage: TsdfBlockStorage) -> None:
                storage.tsdf_sums[index] = 1.25
                storage.weights[index] = 1

            def maximum_weight(storage: TsdfBlockStorage) -> None:
                storage.weights[index] = np.uint32(MAX_TSDF_VOXEL_WEIGHT)

            cases = (
                ("nan", nan_sum, "sum must be finite"),
                ("infinity", infinite_sum, "sum must be finite"),
                ("negative-zero", negative_zero, "canonical positive zero"),
                (
                    "nonzero-unknown",
                    nonzero_unknown,
                    "canonical positive zero",
                ),
                (
                    "outside-envelope",
                    outside_envelope,
                    "exceeds its weight envelope",
                ),
                (
                    "uint32-maximum",
                    maximum_weight,
                    "cannot exceed uint32 maximum",
                ),
            )
            for name, corrupt, message in cases:
                with self.subTest(name=name):
                    storage = allocate_empty_tsdf_blocks(plan, session)
                    corrupt(storage)
                    self.assert_apply_fails_unchanged(
                        storage,
                        storage,
                        contribution,
                        context,
                        message,
                    )

    def test_postwrite_byte_mismatch_is_detected_and_rolled_back(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            _, _, storage, context = load_case(Path(temporary_directory))
            address = require_address(storage, (8, -1, -1))
            contribution = evaluate_tsdf_voxel_contribution_from_context(
                storage,
                address,
                context,
                0,
            )
            before_layout = storage_layout_snapshot(storage)
            before_bytes = storage_bytes(storage)
            import spatialforge.tsdf_voxel_update as update_module

            actual_layout_identity = update_module._storage_layout_identity
            calls = 0

            def corrupt_target_on_verification(candidate):
                nonlocal calls
                calls += 1
                if calls == 2:
                    candidate.weights[address.array_index_bzyx] = 2
                return actual_layout_identity(candidate)

            with (
                forbidden_application_calls() as forbidden,
                patch(
                    "spatialforge.tsdf_voxel_update."
                    "_storage_layout_identity",
                    side_effect=corrupt_target_on_verification,
                ),
            ):
                with self.assertRaises(TsdfError) as raised:
                    apply_tsdf_voxel_contribution_from_context(
                        storage,
                        contribution,
                        context,
                    )

        self.assertIn(
            "could not verify its scalar writes",
            str(raised.exception),
        )
        self.assertEqual(calls, 3)
        self.assertEqual(storage_layout_snapshot(storage), before_layout)
        self.assertEqual(storage_bytes(storage), before_bytes)
        self.assertEqual(float(storage.tsdf_sums[address.array_index_bzyx]), 0.0)
        self.assertFalse(np.signbit(storage.tsdf_sums[address.array_index_bzyx]))
        self.assertEqual(int(storage.weights[address.array_index_bzyx]), 0)
        for forbidden_call in forbidden:
            forbidden_call.assert_not_called()

    def test_unexpected_postwrite_failure_is_wrapped_and_rolled_back(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            _, _, storage, context = load_case(Path(temporary_directory))
            address = require_address(storage, (8, -1, -1))
            contribution = evaluate_tsdf_voxel_contribution_from_context(
                storage,
                address,
                context,
                0,
            )
            before_layout = storage_layout_snapshot(storage)
            before_bytes = storage_bytes(storage)
            import spatialforge.tsdf_voxel_update as update_module

            actual_layout_identity = update_module._storage_layout_identity
            calls = 0

            def fail_on_verification(candidate):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("injected postwrite failure")
                return actual_layout_identity(candidate)

            with (
                forbidden_application_calls() as forbidden,
                patch(
                    "spatialforge.tsdf_voxel_update."
                    "_storage_layout_identity",
                    side_effect=fail_on_verification,
                ),
            ):
                with self.assertRaises(TsdfError) as raised:
                    apply_tsdf_voxel_contribution_from_context(
                        storage,
                        contribution,
                        context,
                    )

        self.assertIn(
            "cannot apply context-bound TSDF voxel contribution: "
            "injected postwrite failure",
            str(raised.exception),
        )
        self.assertIsInstance(raised.exception.__cause__, RuntimeError)
        self.assertEqual(calls, 3)
        self.assertEqual(storage_layout_snapshot(storage), before_layout)
        self.assertEqual(storage_bytes(storage), before_bytes)
        for forbidden_call in forbidden:
            forbidden_call.assert_not_called()

    def test_context_snapshot_applies_after_session_sources_change(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            session_path = copy_fixture(temporary_root)
            _, _, storage, context = load_case(
                temporary_root,
                session_path=session_path,
            )
            address = require_address(storage, (8, -1, -1))
            contribution = evaluate_tsdf_voxel_contribution_from_context(
                storage,
                address,
                context,
                0,
            )
            context_before = context_snapshot(context)
            depth_path = session_path / "data" / "depth" / "000000.pgm"
            depth_path.write_bytes(b"source changed after context build\n")
            set_first_pose_translation_x(session_path, 9.0)
            changed_tree = tree_snapshot(temporary_root)

            with (
                patch.object(
                    TsdfReplayDepthObservation,
                    "depth_m",
                    new_callable=PropertyMock,
                    side_effect=AssertionError(
                        "context update accessed prepared depth"
                    ),
                ),
                forbidden_application_calls() as forbidden,
            ):
                receipt = apply_tsdf_voxel_contribution_from_context(
                    storage,
                    contribution,
                    context,
                )
            after_tree = tree_snapshot(temporary_root)
            context_after = context_snapshot(context)

        self.assertEqual(
            (
                receipt.tsdf_sum_before,
                receipt.weight_before,
                receipt.tsdf_sum_after,
                receipt.weight_after,
            ),
            (0.0, 0, -0.125, 1),
        )
        self.assertEqual(after_tree, changed_tree)
        self.assertEqual(context_after, context_before)
        for forbidden_call in forbidden:
            forbidden_call.assert_not_called()


class TsdfContextUpdateCliTests(unittest.TestCase):
    def test_cli_exactly_reports_one_isolated_context_update(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            session_path = copy_fixture(temporary_root)
            plan_path = create_plan(
                temporary_root,
                session_path=session_path,
            )
            before_tree = tree_snapshot(temporary_root)
            stdout = io.StringIO()
            stderr = io.StringIO()
            captured_storage: list[TsdfBlockStorage] = []
            isolated_calls: list[list[MagicMock]] = []

            actual_allocate = allocate_empty_tsdf_blocks
            actual_build = build_tsdf_replay_depth_context
            actual_evaluate = evaluate_tsdf_voxel_contribution_from_context
            actual_apply = apply_tsdf_voxel_contribution_from_context

            def capture_allocation(plan, session):
                storage = actual_allocate(plan, session)
                captured_storage.append(storage)
                return storage

            def apply_with_isolation(storage, contribution, context):
                with (
                    patch.object(
                        TsdfReplayDepthObservation,
                        "depth_m",
                        new_callable=PropertyMock,
                        side_effect=AssertionError(
                            "CLI context update accessed prepared depth"
                        ),
                    ),
                    forbidden_application_calls() as forbidden,
                ):
                    result = actual_apply(storage, contribution, context)
                isolated_calls.append(forbidden)
                return result

            with (
                forbidden_context_apply_cli_calls() as forbidden_cli,
                patch(
                    "spatialforge.cli.allocate_empty_tsdf_blocks",
                    side_effect=capture_allocation,
                ) as allocator,
                patch(
                    "spatialforge.cli.build_tsdf_replay_depth_context",
                    wraps=actual_build,
                ) as builder,
                patch(
                    "spatialforge.cli."
                    "evaluate_tsdf_voxel_contribution_from_context",
                    wraps=actual_evaluate,
                ) as evaluator,
                patch(
                    "spatialforge.cli."
                    "apply_tsdf_voxel_contribution_from_context",
                    side_effect=apply_with_isolation,
                ) as updater,
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                exit_code = main(
                    [
                        "reconstruct",
                        "tsdf-block-context-contribution-apply",
                        str(plan_path),
                        str(session_path),
                        "--observation-sequence",
                        "0",
                        "--voxel",
                        "8",
                        "-1",
                        "-1",
                    ]
                )
            after_tree = tree_snapshot(temporary_root)

        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(after_tree, before_tree)
        allocator.assert_called_once()
        builder.assert_called_once()
        evaluator.assert_called_once()
        updater.assert_called_once()
        self.assertEqual(len(isolated_calls), 1)
        for forbidden_call in isolated_calls[0]:
            forbidden_call.assert_not_called()
        for forbidden_call in forbidden_cli:
            forbidden_call.assert_not_called()
        self.assertEqual(len(captured_storage), 1)
        storage = captured_storage[0]
        address = require_address(storage, (8, -1, -1))
        self.assertEqual(address.storage_flat_index, 1016)
        self.assertEqual(
            np.flatnonzero(storage.tsdf_sums.reshape(-1)).tolist(),
            [1016],
        )
        self.assertEqual(
            np.flatnonzero(storage.weights.reshape(-1)).tolist(),
            [1016],
        )
        self.assertEqual(
            float(storage.tsdf_sums[address.array_index_bzyx]),
            -0.125,
        )
        self.assertEqual(int(storage.weights[address.array_index_bzyx]), 1)

        expected_output = (
            "TSDF BLOCK CONTEXT CONTRIBUTION APPLY CHECK "
            "scan-synthetic-0001\n"
            "artifact: valid\n"
            "session_replay: matched\n"
            "context_selection: frame_stride=1 total=2 selected=2\n"
            "context_immutable: yes\n"
            "depth_source: replay-depth-context\n"
            "observation_sequence: 0\n"
            "voxel: global=(8, -1, -1) block=(1, -1, -1) "
            "local=(0, 7, 7) row=1 array=(1, 7, 7, 0) "
            "storage_flat=1016\n"
            "world_xyz_m: (1.062500000, -0.062500000, -0.062500000)\n"
            "camera_xyz_m: (0.062500000, 0.062500000, 1.062500000)\n"
            "projected_uv: (0.617647059, 0.617647059)\n"
            "pixel_uv: (1, 1)\n"
            "depth_decoded: yes\n"
            "measured_depth_m: 1.000000000\n"
            "signed_distance_m: -0.062500000\n"
            "evaluation: contributes\n"
            "proposed_delta: tsdf_sum=-0.125000000 weight=1\n"
            "evaluation_replay_hashing: no\n"
            "evaluation_depth_decoding: no\n"
            "slot_before: tsdf_sum=0.000000000 weight=0\n"
            "applied_delta: tsdf_sum=-0.125000000 weight=1\n"
            "slot_after: tsdf_sum=-0.125000000 weight=1\n"
            "storage_before: nonzero_sums=0 nonzero_weights=0 "
            "unknown_voxels=4096\n"
            "storage_after: nonzero_sums=1 nonzero_weights=1 "
            "unknown_voxels=4095\n"
            "context_provenance: matched\n"
            "application_source_freshness: construction-time-context\n"
            "application_session_replay: no\n"
            "application_replay_hashing: no\n"
            "application_source_io: no\n"
            "application_depth_access: no\n"
            "contributions_evaluated: 1\n"
            "contributions_applied: 1\n"
            "storage_slots_updated: 1\n"
            "voxel_observation_traversal_performed: no\n"
            "voxel_address_traversal_performed: no\n"
            "fusion_block_traversal_performed: no\n"
            "ray_traversal_performed: no\n"
            "full_fusion_performed: no\n"
            "missing_blocks_created: no\n"
            "artifact_written: no\n"
            "storage_persisted: no\n"
            "context_persisted: no\n"
            f"plan_sha256: {PLAN_SHA256}\n"
            f"replay_digest_sha256: {REPLAY_SHA256}\n"
        )
        self.assertEqual(stdout.getvalue(), expected_output)


if __name__ == "__main__":
    unittest.main()
