"""Vectorised block evaluation, pinned to the scalar contribution path.

The scalar evaluator in ``tsdf_voxel_contribution`` is the reference
definition of the projective rule. These tests require the vector path to
reproduce it exactly rather than approximately: the same status for every
voxel, the same integer weight delta, and a ``tsdf_sum_delta`` with the same
float64 bit pattern.
"""

from __future__ import annotations

import io
import math
import shutil
import tempfile
import unittest
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from spatialforge import (
    TSDF_CONTRIBUTION_STATUS_ORDER,
    TsdfBlockContributionField,
    allocate_empty_tsdf_blocks,
    build_tsdf_replay_depth_context,
    compose_tsdf_global_voxel_index,
    evaluate_tsdf_block_contributions_from_context,
    evaluate_tsdf_voxel_contribution_from_context,
    load_tsdf_block_plan,
    locate_tsdf_voxel,
)
from spatialforge.cli import main
from spatialforge.errors import TsdfError
from spatialforge.model import CameraCalibration, ScanSession
from spatialforge.session_loader import load_scan_session
from spatialforge.tsdf_block_contributions import (
    _block_voxel_centres_world_m,
    _evaluate_ready_voxels,
    _freeze,
)
from spatialforge.tsdf_block_plan import plan_tsdf_blocks
from spatialforge.tsdf_block_plan_loader import TsdfBlockPlan
from spatialforge.tsdf_block_storage import (
    TSDF_BLOCK_VOXELS,
    TsdfBlockStorage,
)
from spatialforge.tsdf_block_traversal import _local_index_from_flat
from spatialforge.tsdf_plan_traversal import (
    traverse_tsdf_plan_blocks_from_context,
)
from spatialforge.tsdf_replay_depth_context import (
    TsdfReplayDepthContext,
    TsdfReplayDepthStatus,
)
from spatialforge.tsdf_voxel_address import TsdfVoxelAddress
from spatialforge.tsdf_voxel_contribution import (
    TsdfContributionStatus,
    _evaluate_metric_observation,
)

from tests.heavy_fixtures import shared_room_case

TEST_ROOT = Path(__file__).resolve().parent
FIXTURE = TEST_ROOT / "fixtures" / "minimal.vgsession"
PLAN_ARGUMENTS = {
    "voxel_size_m": 0.125,
    "truncation_m": 0.5,
}
SELECTED_BLOCK = (1, -1, -1)
PLAN_SHA256 = (
    "372c7c5d49eff1a30317ceb8b67cb3c40683f1d9d2a6179a049772f763d6f79d"
)
REPLAY_SHA256 = (
    "dc001ae0ca01004a21ad227b12d57a7350bbc23904040453ca73fd955ef050b8"
)

# Recorded by the ledgered fusion run in docs/real-scale-validation.md:
# 351 blocks x 512 voxels x 20 observations, of which this many were accepted
# and applied to storage.
ROOM_VOXEL_OBSERVATIONS = 3_594_240
ROOM_ACCEPTED_CONTRIBUTIONS = 1_127_112

_FORBIDDEN_EVALUATION_TARGETS = (
    "spatialforge.tsdf_voxel_contribution.replay_session",
    "spatialforge.tsdf_replay_depth_context.replay_session",
    "spatialforge.tsdf_voxel_contribution._read_depth",
    "spatialforge.tsdf_replay_depth_context._read_depth",
    "spatialforge.tsdf_voxel_contribution._sample_path",
    "spatialforge.tsdf_replay_depth_context._sample_path",
    "spatialforge.replay._file_digest",
    "hashlib.sha256",
    "pathlib.Path.open",
    "PIL.Image.open",
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
) -> Path:
    output = parent / name
    plan_tsdf_blocks(
        load_scan_session(session_path),
        output,
        frame_stride=1,
        **PLAN_ARGUMENTS,
    )
    return output


def load_case(
    parent: Path,
    *,
    session_path: Path = FIXTURE,
    plan_name: str = "fixture.sftplan",
) -> tuple[
    TsdfBlockPlan,
    ScanSession,
    TsdfBlockStorage,
    TsdfReplayDepthContext,
]:
    plan = load_tsdf_block_plan(
        create_plan(parent, session_path=session_path, name=plan_name)
    )
    session = load_scan_session(session_path)
    storage = allocate_empty_tsdf_blocks(plan, session)
    context = build_tsdf_replay_depth_context(plan, session)
    return plan, session, storage, context


def remove_first_record(path: Path) -> None:
    remaining = path.read_text(encoding="utf-8").splitlines()[1:]
    path.write_text(
        "".join(line + "\n" for line in remaining),
        encoding="utf-8",
    )


def tree_snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def storage_bytes(storage: TsdfBlockStorage) -> tuple[bytes, bytes]:
    return storage.tsdf_sums.tobytes(), storage.weights.tobytes()


@contextmanager
def forbidden_calls(targets: tuple[str, ...]):
    """Patch calls the vector evaluator must never make, and check them."""

    with ExitStack() as stack:
        patches = {
            target: stack.enter_context(patch(target))
            for target in targets
        }
        yield
        for target, mock in patches.items():
            if mock.called:
                raise AssertionError(f"{target} was called")


_ROOM_STORAGE: TsdfBlockStorage | None = None


def room_case() -> tuple[
    TsdfBlockPlan,
    TsdfBlockStorage,
    TsdfReplayDepthContext,
]:
    """The shared room scan, with storage this module only ever reads.

    Allocation replay-verifies the plan, so one storage is cached rather
    than rebuilt per test. Evaluation never writes to it, and a test in
    this module pins that.
    """

    global _ROOM_STORAGE
    case = shared_room_case()
    if _ROOM_STORAGE is None:
        _ROOM_STORAGE = allocate_empty_tsdf_blocks(case.plan, case.session)
    return case.plan, _ROOM_STORAGE, case.context


def scalar_block_reference(
    storage: TsdfBlockStorage,
    block_index_xyz: tuple[int, int, int],
    context: TsdfReplayDepthContext,
    observation_sequence: int,
) -> tuple[
    tuple[TsdfContributionStatus, ...],
    tuple[str, ...],
    tuple[int, ...],
]:
    """Re-derive one block one voxel at a time with the scalar evaluator."""

    statuses: list[TsdfContributionStatus] = []
    sum_hexes: list[str] = []
    weights: list[int] = []
    for local_flat_index in range(TSDF_BLOCK_VOXELS):
        global_index_xyz = compose_tsdf_global_voxel_index(
            block_index_xyz,
            _local_index_from_flat(local_flat_index),
        )
        address = locate_tsdf_voxel(storage, global_index_xyz)
        if address is None:
            raise AssertionError(
                f"expected planned address for {global_index_xyz}"
            )
        contribution = evaluate_tsdf_voxel_contribution_from_context(
            storage,
            address,
            context,
            observation_sequence,
        )
        statuses.append(contribution.status)
        sum_hexes.append(
            (
                0.0
                if contribution.tsdf_sum_delta is None
                else contribution.tsdf_sum_delta
            ).hex()
        )
        weights.append(contribution.weight_delta)
    return tuple(statuses), tuple(sum_hexes), tuple(weights)


def field_as_reference(
    field: TsdfBlockContributionField,
) -> tuple[
    tuple[TsdfContributionStatus, ...],
    tuple[str, ...],
    tuple[int, ...],
]:
    return (
        field.voxel_statuses,
        tuple(
            float(value).hex()
            for value in field.tsdf_sum_deltas.tolist()
        ),
        tuple(int(value) for value in field.weight_deltas.tolist()),
    )


class VectorBlockContributionTests(unittest.TestCase):
    def test_every_block_and_observation_matches_the_scalar_path(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            plan, _, storage, context = load_case(Path(temporary_dir))
            for block_index_xyz in plan.active_blocks:
                for sequence in context.selected_observation_sequences:
                    with self.subTest(block=block_index_xyz, seq=sequence):
                        field = (
                            evaluate_tsdf_block_contributions_from_context(
                                storage,
                                block_index_xyz,
                                context,
                                sequence,
                            )
                        )
                        self.assertEqual(
                            field_as_reference(field),
                            scalar_block_reference(
                                storage,
                                block_index_xyz,
                                context,
                                sequence,
                            ),
                        )

    def test_accepted_total_matches_the_fusing_plan_traversal(self) -> None:
        """The accept rule, pinned to the path that actually fuses."""

        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            temporary_root = Path(temporary_dir)
            plan, session, storage, context = load_case(temporary_root)
            accepted = 0
            evaluated = 0
            for block_index_xyz in plan.active_blocks:
                for sequence in context.selected_observation_sequences:
                    field = evaluate_tsdf_block_contributions_from_context(
                        storage,
                        block_index_xyz,
                        context,
                        sequence,
                    )
                    accepted += field.contributing_count
                    evaluated += field.evaluated_count
            receipt = traverse_tsdf_plan_blocks_from_context(
                allocate_empty_tsdf_blocks(plan, session),
                context,
            )

        self.assertEqual(evaluated, receipt.evaluated_count)
        self.assertEqual(evaluated, 8192)
        self.assertEqual(accepted, receipt.applied_count)
        self.assertEqual(accepted, 1168)

    def test_room_fixture_blocks_match_the_scalar_path(self) -> None:
        plan, storage, context = room_case()
        blocks = plan.active_blocks
        sampled = (
            blocks[0],
            blocks[len(blocks) // 3],
            blocks[len(blocks) // 2],
            blocks[-1],
        )
        for block_index_xyz in sampled:
            for sequence in (0, 7, 19):
                with self.subTest(block=block_index_xyz, seq=sequence):
                    field = evaluate_tsdf_block_contributions_from_context(
                        storage,
                        block_index_xyz,
                        context,
                        sequence,
                    )
                    self.assertEqual(
                        field_as_reference(field),
                        scalar_block_reference(
                            storage,
                            block_index_xyz,
                            context,
                            sequence,
                        ),
                    )

    def test_room_fixture_reproduces_the_recorded_fusion_totals(self) -> None:
        """The whole room plan, against the numbers its fusion run produced."""

        plan, storage, context = room_case()
        evaluated = 0
        accepted = 0
        for block_index_xyz in plan.active_blocks:
            for sequence in context.selected_observation_sequences:
                field = evaluate_tsdf_block_contributions_from_context(
                    storage,
                    block_index_xyz,
                    context,
                    sequence,
                )
                evaluated += field.evaluated_count
                accepted += field.contributing_count

        self.assertEqual(evaluated, ROOM_VOXEL_OBSERVATIONS)
        self.assertEqual(accepted, ROOM_ACCEPTED_CONTRIBUTIONS)

    def test_missing_inputs_skip_every_voxel_like_the_scalar_path(
        self,
    ) -> None:
        cases = (
            (
                "depth",
                ("streams/depth.jsonl",),
                TsdfReplayDepthStatus.MISSING_DEPTH,
                TsdfContributionStatus.MISSING_DEPTH,
            ),
            (
                "pose",
                ("streams/poses.jsonl",),
                TsdfReplayDepthStatus.MISSING_POSE,
                TsdfContributionStatus.MISSING_POSE,
            ),
            (
                "both",
                ("streams/depth.jsonl", "streams/poses.jsonl"),
                TsdfReplayDepthStatus.MISSING_DEPTH_AND_POSE,
                TsdfContributionStatus.MISSING_DEPTH_AND_POSE,
            ),
        )
        for name, references, prepared, expected_status in cases:
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory(
                    dir=TEST_ROOT
                ) as temporary_dir:
                    temporary_root = Path(temporary_dir)
                    session_path = copy_fixture(
                        temporary_root,
                        f"{name}.vgsession",
                    )
                    for reference in references:
                        remove_first_record(session_path / reference)
                    plan, _, storage, context = load_case(
                        temporary_root,
                        session_path=session_path,
                        plan_name=f"{name}.sftplan",
                    )
                    field = evaluate_tsdf_block_contributions_from_context(
                        storage,
                        plan.active_blocks[0],
                        context,
                        0,
                    )
                    reference_result = scalar_block_reference(
                        storage,
                        plan.active_blocks[0],
                        context,
                        0,
                    )

                self.assertEqual(field.observation_status, prepared)
                self.assertEqual(
                    field.status_counts,
                    ((expected_status, TSDF_BLOCK_VOXELS),),
                )
                self.assertEqual(field.contributing_count, 0)
                self.assertEqual(field_as_reference(field), reference_result)

    def test_invalid_depth_samples_match_the_scalar_path(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            temporary_root = Path(temporary_dir)
            session_path = copy_fixture(temporary_root, "invalid.vgsession")
            (session_path / "data" / "depth" / "000000.pgm").write_text(
                "P2\n2 2\n65535\n1000 1000\n1000 0\n",
                encoding="ascii",
            )
            plan, _, storage, context = load_case(
                temporary_root,
                session_path=session_path,
                plan_name="invalid.sftplan",
            )
            found = []
            for block_index_xyz in plan.active_blocks:
                field = evaluate_tsdf_block_contributions_from_context(
                    storage,
                    block_index_xyz,
                    context,
                    0,
                )
                self.assertEqual(
                    field_as_reference(field),
                    scalar_block_reference(
                        storage,
                        block_index_xyz,
                        context,
                        0,
                    ),
                )
                found.extend(
                    status
                    for status, _ in field.status_counts
                    if status is TsdfContributionStatus.DEPTH_INVALID
                )

        self.assertTrue(found, "expected the zeroed pixel to be sampled")


class VectorStatusLadderTests(unittest.TestCase):
    """Statuses the committed fixtures never reach, pinned pairwise.

    ``minimal.vgsession`` and the room scan between them only produce
    ``contributes``, ``projection-outside-image`` and ``behind-truncation``.
    The remaining reachable rungs need a deliberately hostile pose or
    camera, so they are compared against the scalar evaluator's own private
    metric-frame entry point rather than through a session.
    """

    truncation_m = 0.5
    voxel_size_m = 0.125

    def camera(self, **overrides: object) -> CameraCalibration:
        defaults = {
            "id": "camera-depth",
            "model": "pinhole",
            "width": 4,
            "height": 4,
            "fx": 2.0,
            "fy": 2.0,
            "cx": 1.5,
            "cy": 1.5,
            "distortion_model": "none",
            "distortion_coefficients": (),
            "t_rig_camera": tuple(
                float(value)
                for value in (
                    1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 1, 0,
                    0, 0, 0, 1,
                )
            ),
        }
        defaults.update(overrides)
        return CameraCalibration(**defaults)  # type: ignore[arg-type]

    def compare(
        self,
        camera: CameraCalibration,
        transform: tuple[float, ...],
        depth_m: np.ndarray,
        block_index_xyz: tuple[int, int, int] = (0, 0, 0),
    ) -> tuple[TsdfContributionStatus, ...]:
        world_xyz_m = _block_voxel_centres_world_m(
            block_index_xyz,
            self.voxel_size_m,
        )
        status_codes, sum_deltas, weight_deltas = _evaluate_ready_voxels(
            camera,
            transform,
            depth_m,
            self.truncation_m,
            world_xyz_m,
        )
        address = TsdfVoxelAddress(
            global_index_xyz=(0, 0, 0),
            block_index_xyz=(0, 0, 0),
            local_index_xyz=(0, 0, 0),
            block_row=0,
            local_flat_index=0,
            storage_flat_index=0,
        )
        statuses: list[TsdfContributionStatus] = []
        for local_flat_index in range(TSDF_BLOCK_VOXELS):
            point = tuple(
                float(axis[local_flat_index]) for axis in world_xyz_m
            )
            expected = _evaluate_metric_observation(
                address,
                0,
                PLAN_SHA256,
                REPLAY_SHA256,
                self.truncation_m,
                camera,
                transform,
                depth_m,
                point,
            )
            self.assertIs(
                expected.status,
                TSDF_CONTRIBUTION_STATUS_ORDER[
                    int(status_codes[local_flat_index])
                ],
                f"status differs at local flat {local_flat_index}",
            )
            expected_sum = (
                0.0
                if expected.tsdf_sum_delta is None
                else expected.tsdf_sum_delta
            )
            self.assertEqual(
                expected_sum.hex(),
                float(sum_deltas[local_flat_index]).hex(),
                f"sum differs at local flat {local_flat_index}",
            )
            self.assertEqual(
                expected.weight_delta,
                int(weight_deltas[local_flat_index]),
                f"weight differs at local flat {local_flat_index}",
            )
            statuses.append(expected.status)
        return tuple(statuses)

    def test_camera_point_nonfinite_matches(self) -> None:
        transform = tuple(
            float(value)
            for value in (
                1, 1, 1, -1e308,
                1, 1, 1, -1e308,
                1, 1, 1, -1e308,
                0, 0, 0, 1,
            )
        )
        depth_m = np.full((4, 4), 1.0, dtype=np.float64)
        statuses = self.compare(self.camera(), transform, depth_m)
        self.assertEqual(
            set(statuses),
            {TsdfContributionStatus.CAMERA_POINT_NONFINITE},
        )

    def test_camera_z_nonpositive_and_contributing_split(self) -> None:
        # The camera sits inside the block looking along +z, so the near half
        # of the block is behind it and the far half is in front.
        transform = tuple(
            float(value)
            for value in (
                1, 0, 0, 0.5,
                0, 1, 0, 0.5,
                0, 0, 1, 0.5,
                0, 0, 0, 1,
            )
        )
        depth_m = np.full((4, 4), 0.3, dtype=np.float64)
        statuses = self.compare(self.camera(), transform, depth_m)
        self.assertIn(
            TsdfContributionStatus.CAMERA_Z_NONPOSITIVE,
            statuses,
        )
        self.assertIn(TsdfContributionStatus.CONTRIBUTES, statuses)

    def test_projection_nonfinite_matches(self) -> None:
        transform = tuple(
            float(value)
            for value in (
                1, 0, 0, 0.0,
                0, 1, 0, 0.0,
                0, 0, 1, 0.0,
                0, 0, 0, 1,
            )
        )
        depth_m = np.full((4, 4), 1.0, dtype=np.float64)
        camera = self.camera(fx=1e308, fy=1e308)
        statuses = self.compare(camera, transform, depth_m)
        self.assertIn(
            TsdfContributionStatus.PROJECTION_NONFINITE,
            statuses,
        )

    def test_depth_invalid_matches(self) -> None:
        transform = tuple(
            float(value)
            for value in (
                1, 0, 0, 0.0,
                0, 1, 0, 0.0,
                0, 0, 1, -1.0,
                0, 0, 0, 1,
            )
        )
        depth_m = np.zeros((4, 4), dtype=np.float64)
        depth_m[0, 0] = math.nan
        statuses = self.compare(self.camera(), transform, depth_m)
        self.assertEqual(
            set(statuses),
            {TsdfContributionStatus.DEPTH_INVALID},
        )

    def test_signed_distance_is_unreachable_after_the_earlier_gates(
        self,
    ) -> None:
        """A finite positive depth minus a finite positive z stays finite."""

        transform = tuple(
            float(value)
            for value in (
                1, 0, 0, 0.0,
                0, 1, 0, 0.0,
                0, 0, 1, -1.0,
                0, 0, 0, 1,
            )
        )
        depth_m = np.full((4, 4), 1e308, dtype=np.float64)
        statuses = self.compare(self.camera(), transform, depth_m)
        self.assertNotIn(
            TsdfContributionStatus.SIGNED_DISTANCE_NONFINITE,
            statuses,
        )


class VectorFieldReceiptTests(unittest.TestCase):
    def field(self, storage, context) -> TsdfBlockContributionField:
        return evaluate_tsdf_block_contributions_from_context(
            storage,
            SELECTED_BLOCK,
            context,
            0,
        )

    def test_receipt_is_frozen_with_immutable_arrays(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            _, _, storage, context = load_case(Path(temporary_dir))
            field = self.field(storage, context)

        with self.assertRaises(FrozenInstanceError):
            field.block_row = 3  # type: ignore[misc]
        for array in (
            field.status_codes,
            field.tsdf_sum_deltas,
            field.weight_deltas,
        ):
            self.assertFalse(array.flags.writeable)
            self.assertFalse(array.flags.owndata)
            self.assertTrue(array.flags.c_contiguous)
            self.assertEqual(array.shape, (TSDF_BLOCK_VOXELS,))
            with self.assertRaises(ValueError):
                array[0] = 1

    def test_counts_are_derived_from_the_retained_arrays(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            _, _, storage, context = load_case(Path(temporary_dir))
            field = self.field(storage, context)

        self.assertEqual(field.voxel_count, TSDF_BLOCK_VOXELS)
        self.assertEqual(field.evaluated_count, TSDF_BLOCK_VOXELS)
        self.assertEqual(
            field.contributing_count + field.skipped_count,
            TSDF_BLOCK_VOXELS,
        )
        self.assertEqual(field.weight_delta_total, field.contributing_count)
        self.assertEqual(
            sum(count for _, count in field.status_counts),
            TSDF_BLOCK_VOXELS,
        )
        self.assertEqual(
            field.status_counts,
            (
                (TsdfContributionStatus.CONTRIBUTES, 102),
                (TsdfContributionStatus.PROJECTION_OUTSIDE_IMAGE, 212),
                (TsdfContributionStatus.BEHIND_TRUNCATION, 198),
            ),
        )
        self.assertEqual(len(field.voxel_statuses), TSDF_BLOCK_VOXELS)
        self.assertEqual(field.observation_sequence, 0)
        self.assertEqual(field.block_index_xyz, SELECTED_BLOCK)
        self.assertEqual(field.source_plan_digest_sha256, PLAN_SHA256)
        self.assertEqual(field.replay_digest_sha256, REPLAY_SHA256)

    def test_inconsistent_arrays_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            _, _, storage, context = load_case(Path(temporary_dir))
            field = self.field(storage, context)

        cases = {
            "weight_deltas": _freeze(
                np.zeros(TSDF_BLOCK_VOXELS, dtype=np.uint32)
            ),
            "tsdf_sum_deltas": _freeze(
                np.full(TSDF_BLOCK_VOXELS, 0.5, dtype=np.float64)
            ),
            "status_codes": _freeze(
                np.full(
                    TSDF_BLOCK_VOXELS,
                    len(TSDF_CONTRIBUTION_STATUS_ORDER),
                    dtype=np.uint8,
                )
            ),
        }
        for name, value in cases.items():
            with self.subTest(field=name):
                with self.assertRaises(TsdfError):
                    replace(field, **{name: value})

    def test_writeable_arrays_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            _, _, storage, context = load_case(Path(temporary_dir))
            field = self.field(storage, context)

        with self.assertRaises(TsdfError) as caught:
            replace(field, status_codes=np.array(field.status_codes))
        self.assertIn("immutable", str(caught.exception))

    def test_unprepared_observation_cannot_carry_ready_statuses(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            _, _, storage, context = load_case(Path(temporary_dir))
            field = self.field(storage, context)

        with self.assertRaises(TsdfError) as caught:
            replace(
                field,
                observation_status=TsdfReplayDepthStatus.MISSING_POSE,
            )
        self.assertIn("preparation status", str(caught.exception))


class VectorEvaluationIsolationTests(unittest.TestCase):
    def test_evaluation_leaves_storage_and_the_session_untouched(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            temporary_root = Path(temporary_dir)
            session_path = copy_fixture(temporary_root)
            plan, _, storage, context = load_case(
                temporary_root,
                session_path=session_path,
            )
            before_tree = tree_snapshot(temporary_root)
            before_bytes = storage_bytes(storage)
            for sequence in context.selected_observation_sequences:
                evaluate_tsdf_block_contributions_from_context(
                    storage,
                    plan.active_blocks[0],
                    context,
                    sequence,
                )
            after_bytes = storage_bytes(storage)
            after_tree = tree_snapshot(temporary_root)

        self.assertEqual(before_bytes, after_bytes)
        self.assertEqual(before_tree, after_tree)

    def test_evaluation_performs_no_replay_hashing_or_depth_io(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            _, _, storage, context = load_case(Path(temporary_dir))
            with forbidden_calls(_FORBIDDEN_EVALUATION_TARGETS):
                field = evaluate_tsdf_block_contributions_from_context(
                    storage,
                    SELECTED_BLOCK,
                    context,
                    0,
                )

        self.assertEqual(field.contributing_count, 102)

    def test_one_address_resolution_replaces_five_hundred_and_twelve(
        self,
    ) -> None:
        """The point of the checkpoint: no per-voxel Python in the hot path."""

        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            _, _, storage, context = load_case(Path(temporary_dir))
            with (
                patch(
                    "spatialforge.tsdf_block_contributions.locate_tsdf_voxel",
                    wraps=locate_tsdf_voxel,
                ) as locator,
                patch(
                    "spatialforge.tsdf_voxel_contribution."
                    "evaluate_tsdf_voxel_contribution_from_context"
                ) as scalar,
            ):
                evaluate_tsdf_block_contributions_from_context(
                    storage,
                    SELECTED_BLOCK,
                    context,
                    0,
                )

        self.assertEqual(locator.call_count, 1)
        scalar.assert_not_called()


class VectorEvaluationFailureTests(unittest.TestCase):
    def test_unplanned_block_is_refused(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            _, _, storage, context = load_case(Path(temporary_dir))
            with self.assertRaises(TsdfError) as caught:
                evaluate_tsdf_block_contributions_from_context(
                    storage,
                    (-1, 0, 0),
                    context,
                    0,
                )

        self.assertIn("is not planned", str(caught.exception))

    def test_invalid_arguments_are_refused(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            _, _, storage, context = load_case(Path(temporary_dir))
            cases = (
                (
                    "storage",
                    (None, SELECTED_BLOCK, context, 0),
                    "TsdfBlockStorage",
                ),
                (
                    "block",
                    (storage, (1, -1), context, 0),
                    "expected a tuple of 3 integers",
                ),
                (
                    "context",
                    (storage, SELECTED_BLOCK, None, 0),
                    "TsdfReplayDepthContext",
                ),
                (
                    "sequence-type",
                    (storage, SELECTED_BLOCK, context, "0"),
                    "expected an integer",
                ),
                (
                    "sequence-negative",
                    (storage, SELECTED_BLOCK, context, -1),
                    "non-negative",
                ),
                (
                    "sequence-range",
                    (storage, SELECTED_BLOCK, context, 99),
                    "outside context range",
                ),
            )
            for name, arguments, expected in cases:
                with self.subTest(case=name):
                    with self.assertRaises(TsdfError) as caught:
                        evaluate_tsdf_block_contributions_from_context(
                            *arguments
                        )
                    self.assertIn(expected, str(caught.exception))

    def test_unselected_observation_sequence_is_refused(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            temporary_root = Path(temporary_dir)
            plan_path = temporary_root / "stride.sftplan"
            plan_tsdf_blocks(
                load_scan_session(FIXTURE),
                plan_path,
                frame_stride=2,
                **PLAN_ARGUMENTS,
            )
            plan = load_tsdf_block_plan(plan_path)
            session = load_scan_session(FIXTURE)
            storage = allocate_empty_tsdf_blocks(plan, session)
            context = build_tsdf_replay_depth_context(plan, session)
            with self.assertRaises(TsdfError) as caught:
                evaluate_tsdf_block_contributions_from_context(
                    storage,
                    plan.active_blocks[0],
                    context,
                    1,
                )

        self.assertIn("frame_stride=2", str(caught.exception))


class VectorEvaluationCliTests(unittest.TestCase):
    def test_cli_reports_the_block_and_its_scalar_parity(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            temporary_root = Path(temporary_dir)
            plan_path = create_plan(temporary_root)
            before_tree = tree_snapshot(temporary_root)
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                exit_code = main(
                    [
                        "reconstruct",
                        "tsdf-block-context-block-contributions",
                        str(plan_path),
                        str(FIXTURE),
                        "--block",
                        "1",
                        "-1",
                        "-1",
                        "--observation-sequence",
                        "0",
                    ]
                )
            after_tree = tree_snapshot(temporary_root)

        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(before_tree, after_tree)
        self.assertEqual(
            stdout.getvalue(),
            "TSDF BLOCK CONTEXT BLOCK CONTRIBUTIONS CHECK "
            "scan-synthetic-0001\n"
            "artifact: valid\n"
            "session_replay: matched\n"
            "context_selection: frame_stride=1 total=2 selected=2\n"
            "context_immutable: yes\n"
            "depth_source: replay-depth-context\n"
            "block: index=(1, -1, -1) row=1 resolution=8 voxel_slots=512\n"
            "observation: sequence=0 status=ready\n"
            "evaluation_order: local-flat-x-fastest local_flat=0..511\n"
            "evaluation_path: vectorised\n"
            "evaluation_precision: float64\n"
            "contributions_evaluated: 512\n"
            "contributions_contributing: 102\n"
            "contributions_skipped: 410\n"
            "status_counts: contributes=102 "
            "projection-outside-image=212 behind-truncation=198\n"
            "weight_delta_total: 102\n"
            "scalar_reference_evaluations: 512\n"
            "scalar_reference_status_mismatches: 0\n"
            "scalar_reference_sum_mismatches: 0\n"
            "scalar_reference_weight_mismatches: 0\n"
            "scalar_reference_parity: bit-identical\n"
            "context_provenance: matched\n"
            "evaluation_source_freshness: construction-time-context\n"
            "evaluation_session_replay: no\n"
            "evaluation_replay_hashing: no\n"
            "evaluation_source_io: no\n"
            "evaluation_depth_decoding: no\n"
            "storage_before: nonzero_sums=0 nonzero_weights=0 "
            "unknown_voxels=4096\n"
            "storage_after: nonzero_sums=0 nonzero_weights=0 "
            "unknown_voxels=4096\n"
            "blocks_evaluated: 1\n"
            "additional_blocks_visited: 0\n"
            "observations_evaluated: 1\n"
            "contributions_applied: 0\n"
            "storage_slots_updated: 0\n"
            "storage_mutated: no\n"
            "fusion_performed: no\n"
            "ray_traversal_performed: no\n"
            "free_space_coverage_planned: no\n"
            "plan_expanded: no\n"
            "missing_blocks_created: no\n"
            "artifact_written: no\n"
            "storage_persisted: no\n"
            "context_persisted: no\n"
            f"plan_sha256: {PLAN_SHA256}\n"
            f"replay_digest_sha256: {REPLAY_SHA256}\n",
        )

    def test_cli_unplanned_block_fails_actionably(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            temporary_root = Path(temporary_dir)
            plan_path = create_plan(temporary_root)
            before_tree = tree_snapshot(temporary_root)
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                exit_code = main(
                    [
                        "reconstruct",
                        "tsdf-block-context-block-contributions",
                        str(plan_path),
                        str(FIXTURE),
                        "--block",
                        "-1",
                        "0",
                        "0",
                        "--observation-sequence",
                        "0",
                    ]
                )
            after_tree = tree_snapshot(temporary_root)

        self.assertEqual(exit_code, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(before_tree, after_tree)
        error = stderr.getvalue()
        self.assertIn("TSDF BLOCK CONTEXT BLOCK CONTRIBUTIONS FAILED", error)
        self.assertIn("block (-1, 0, 0) is not planned", error)
        self.assertNotIn("Traceback", error)


if __name__ == "__main__":
    unittest.main()
