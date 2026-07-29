from __future__ import annotations

import io
import json
import shutil
import tempfile
import unittest
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from spatialforge import (
    allocate_empty_tsdf_blocks,
    compose_tsdf_global_voxel_index,
    evaluate_tsdf_voxel_contribution,
    locate_tsdf_voxel,
)
from spatialforge.cli import main
from spatialforge.errors import PointCloudError, TsdfError
from spatialforge.replay import replay_session
from spatialforge.session_loader import load_scan_session
from spatialforge.tsdf import (
    _DenseTsdfAccumulator,
    _integrate_voxel_chunk,
)
from spatialforge.tsdf_block_plan import plan_tsdf_blocks
from spatialforge.tsdf_block_plan_loader import (
    TsdfBlockPlan,
    load_tsdf_block_plan,
)
from spatialforge.tsdf_block_storage import (
    TSDF_BLOCK_VOXELS,
    TsdfBlockStorage,
)
from spatialforge.tsdf_voxel_address import (
    TsdfVoxelAddress,
)
from spatialforge.tsdf_voxel_contribution import (
    TsdfContributionStatus,
    TsdfVoxelContribution,
)


TEST_ROOT = Path(__file__).resolve().parent
FIXTURE = TEST_ROOT / "fixtures" / "minimal.vgsession"
PLAN_ARGUMENTS = {
    "voxel_size_m": 0.125,
    "truncation_m": 0.5,
}
PLAN_SHA256 = "372c7c5d49eff1a30317ceb8b67cb3c40683f1d9d2a6179a049772f763d6f79d"
REPLAY_SHA256 = "dc001ae0ca01004a21ad227b12d57a7350bbc23904040453ca73fd955ef050b8"

_FORBIDDEN_RECONSTRUCTION_TARGETS = (
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
) -> tuple[TsdfBlockPlan, object, TsdfBlockStorage]:
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
    return plan, session, storage


def require_address(
    storage: TsdfBlockStorage,
    global_index: tuple[int, int, int],
) -> TsdfVoxelAddress:
    address = locate_tsdf_voxel(storage, global_index)
    if address is None:
        raise AssertionError(f"expected planned address for {global_index}")
    return address


def storage_for_blocks(
    plan: TsdfBlockPlan,
    session,
    block_indices: tuple[tuple[int, int, int], ...],
) -> TsdfBlockStorage:
    minimum = tuple(
        min(block[axis] for block in block_indices)
        for axis in range(3)
    )
    maximum = tuple(
        max(block[axis] for block in block_indices)
        for axis in range(3)
    )
    changed_plan = replace(
        plan,
        surface_blocks=(block_indices[0],),
        active_blocks=block_indices,
        planned_voxel_slots=len(block_indices) * TSDF_BLOCK_VOXELS,
        min_block_index=minimum,
        max_block_index=maximum,
    )
    return allocate_empty_tsdf_blocks(changed_plan, session)


def remove_first_record(path: Path) -> None:
    remaining = path.read_text(encoding="utf-8").splitlines()[1:]
    path.write_text(
        "".join(line + "\n" for line in remaining),
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


@contextmanager
def forbidden_reconstruction_calls():
    with ExitStack() as stack:
        mocks: list[MagicMock] = []
        for target in _FORBIDDEN_RECONSTRUCTION_TARGETS:
            mocks.append(stack.enter_context(patch(target)))
        yield mocks


class TsdfVoxelContributionTests(unittest.TestCase):
    def test_fixture_voxel_has_exact_read_only_contribution(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            _, session, storage = load_case(Path(temporary_directory))
            address = require_address(storage, (8, -1, -1))
            contribution = evaluate_tsdf_voxel_contribution(
                storage,
                address,
                session,  # type: ignore[arg-type]
                0,
            )
            repeated = evaluate_tsdf_voxel_contribution(
                storage,
                address,
                session,  # type: ignore[arg-type]
                0,
            )

        self.assertIsInstance(contribution, TsdfVoxelContribution)
        self.assertEqual(contribution, repeated)
        self.assertIs(contribution.address, address)
        self.assertEqual(contribution.observation_sequence, 0)
        self.assertIs(
            contribution.status,
            TsdfContributionStatus.CONTRIBUTES,
        )
        self.assertTrue(contribution.contributes)
        self.assertEqual(
            contribution.world_xyz_m,
            (1.0625, -0.0625, -0.0625),
        )
        self.assertEqual(
            contribution.camera_xyz_m,
            (0.0625, 0.0625, 1.0625),
        )
        assert contribution.projected_uv is not None
        self.assertAlmostEqual(contribution.projected_uv[0], 21.0 / 34.0)
        self.assertAlmostEqual(contribution.projected_uv[1], 21.0 / 34.0)
        self.assertEqual(contribution.pixel_uv, (1, 1))
        self.assertTrue(contribution.depth_decoded)
        self.assertEqual(contribution.measured_depth_m, 1.0)
        self.assertEqual(contribution.signed_distance_m, -0.0625)
        self.assertEqual(contribution.tsdf_sum_delta, -0.125)
        self.assertEqual(contribution.weight_delta, 1)
        self.assertEqual(address.block_index_xyz, (1, -1, -1))
        self.assertEqual(address.local_index_xyz, (0, 7, 7))
        self.assertEqual(address.block_row, 1)
        self.assertEqual(address.array_index_bzyx, (1, 7, 7, 0))
        self.assertEqual(address.storage_flat_index, 1016)

        with self.assertRaises(FrozenInstanceError):
            contribution.weight_delta = 9  # type: ignore[misc]
        self.assertFalse(hasattr(contribution, "__dict__"))

    def test_exact_surface_band_cases_match_dense_reference_math(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            fixture = copy_fixture(temporary_root)
            set_first_pose_translation_x(fixture, -0.0625)
            plan, session, storage = load_case(
                temporary_root,
                session_path=fixture,
                plan_name="shifted.sftplan",
            )
            replay = replay_session(session)  # type: ignore[arg-type]
            observation = replay.observations[0]
            camera = session.calibrations["camera-rgb"]  # type: ignore[union-attr]
            transform = tuple(observation.pose.data["T_world_camera"])  # type: ignore[union-attr]
            depth_metres = np.ones((camera.height, camera.width), dtype=np.float64)

            cases = (
                (1, 0.75, 1.0, TsdfContributionStatus.CONTRIBUTES),
                (3, 0.50, 1.0, TsdfContributionStatus.CONTRIBUTES),
                (5, 0.25, 0.5, TsdfContributionStatus.CONTRIBUTES),
                (7, 0.00, 0.0, TsdfContributionStatus.CONTRIBUTES),
                (9, -0.25, -0.5, TsdfContributionStatus.CONTRIBUTES),
                (11, -0.50, -1.0, TsdfContributionStatus.CONTRIBUTES),
                (12, -0.625, None, TsdfContributionStatus.BEHIND_TRUNCATION),
            )
            for (
                global_x,
                expected_signed_distance,
                expected_delta,
                expected_status,
            ) in cases:
                with self.subTest(global_x=global_x):
                    address = require_address(
                        storage,
                        (global_x, -1, -1),
                    )
                    contribution = evaluate_tsdf_voxel_contribution(
                        storage,
                        address,
                        session,  # type: ignore[arg-type]
                        0,
                    )
                    self.assertIs(contribution.status, expected_status)
                    self.assertEqual(
                        contribution.signed_distance_m,
                        expected_signed_distance,
                    )
                    self.assertEqual(
                        contribution.tsdf_sum_delta,
                        expected_delta,
                    )
                    self.assertEqual(
                        contribution.weight_delta,
                        1 if expected_delta is not None else 0,
                    )

                    half_voxel = plan.voxel_size_m * 0.5
                    origin = tuple(
                        component - half_voxel
                        for component in contribution.world_xyz_m
                    )
                    accumulator = _DenseTsdfAccumulator(1)
                    updates = _integrate_voxel_chunk(
                        0,
                        1,
                        depth_metres,
                        camera,
                        transform,
                        origin,  # type: ignore[arg-type]
                        (1, 1, 1),
                        plan.voxel_size_m,
                        plan.truncation_m,
                        accumulator,
                    )
                    indices, sums, weights = accumulator.observed_arrays()
                    if expected_delta is None:
                        self.assertEqual(updates, 0)
                        self.assertEqual(indices.size, 0)
                        self.assertEqual(sums.size, 0)
                        self.assertEqual(weights.size, 0)
                    else:
                        self.assertEqual(updates, 1)
                        self.assertEqual(indices.tolist(), [0])
                        self.assertEqual(sums.tolist(), [expected_delta])
                        self.assertEqual(weights.tolist(), [1])

    def test_missing_depth_and_pose_have_stable_status_without_decoding(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            cases = (
                (
                    "depth",
                    ("streams/depth.jsonl",),
                    TsdfContributionStatus.MISSING_DEPTH,
                ),
                (
                    "pose",
                    ("streams/poses.jsonl",),
                    TsdfContributionStatus.MISSING_POSE,
                ),
                (
                    "depth-and-pose",
                    (
                        "streams/depth.jsonl",
                        "streams/poses.jsonl",
                    ),
                    TsdfContributionStatus.MISSING_DEPTH_AND_POSE,
                ),
            )
            for name, relative_paths, expected_status in cases:
                with self.subTest(name=name):
                    fixture = copy_fixture(
                        temporary_root,
                        f"{name}.vgsession",
                    )
                    for relative_path in relative_paths:
                        remove_first_record(fixture / relative_path)
                    _, session, storage = load_case(
                        temporary_root,
                        session_path=fixture,
                        plan_name=f"{name}.sftplan",
                    )
                    global_index = compose_tsdf_global_voxel_index(
                        storage.block_indices[0],
                        (0, 0, 0),
                    )
                    address = require_address(storage, global_index)
                    with patch(
                        "spatialforge.tsdf_voxel_contribution._read_depth"
                    ) as depth_reader:
                        contribution = evaluate_tsdf_voxel_contribution(
                            storage,
                            address,
                            session,  # type: ignore[arg-type]
                            0,
                        )
                        depth_reader.assert_not_called()

                    self.assertIs(contribution.status, expected_status)
                    self.assertFalse(contribution.contributes)
                    self.assertFalse(contribution.depth_decoded)
                    self.assertIsNone(contribution.camera_xyz_m)
                    self.assertIsNone(contribution.projected_uv)
                    self.assertIsNone(contribution.pixel_uv)
                    self.assertIsNone(contribution.measured_depth_m)
                    self.assertIsNone(contribution.signed_distance_m)
                    self.assertIsNone(contribution.tsdf_sum_delta)
                    self.assertEqual(contribution.weight_delta, 0)

    def test_behind_camera_out_of_view_and_invalid_depth_statuses_are_stable(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            plan, session, storage = load_case(Path(temporary_directory))

            out_of_view_address = require_address(storage, (0, 7, -1))
            with patch(
                "spatialforge.tsdf_voxel_contribution._read_depth",
                return_value=(1000, 1000, 1000, 1000),
            ) as depth_reader:
                out_of_view = evaluate_tsdf_voxel_contribution(
                    storage,
                    out_of_view_address,
                    session,  # type: ignore[arg-type]
                    0,
                )
                depth_reader.assert_called_once()

            with patch(
                "spatialforge.tsdf_voxel_contribution._read_depth",
                side_effect=PointCloudError("corrupt depth frame"),
            ):
                with self.assertRaises(TsdfError) as raised:
                    evaluate_tsdf_voxel_contribution(
                        storage,
                        out_of_view_address,
                        session,  # type: ignore[arg-type]
                        0,
                    )
            self.assertIn("corrupt depth frame", str(raised.exception))

            behind_storage = storage_for_blocks(
                plan,
                session,
                ((-1, -1, -1),),
            )
            behind_address = require_address(
                behind_storage,
                (-1, -1, -1),
            )
            with patch(
                "spatialforge.tsdf_voxel_contribution._read_depth",
                return_value=(1000, 1000, 1000, 1000),
            ) as depth_reader:
                behind = evaluate_tsdf_voxel_contribution(
                    behind_storage,
                    behind_address,
                    session,  # type: ignore[arg-type]
                    0,
                )
                depth_reader.assert_called_once()

            accepted_address = require_address(storage, (8, -1, -1))
            with patch(
                "spatialforge.tsdf_voxel_contribution._read_depth",
                return_value=(1000, 1000, 1000, 0),
            ) as depth_reader:
                invalid_depth = evaluate_tsdf_voxel_contribution(
                    storage,
                    accepted_address,
                    session,  # type: ignore[arg-type]
                    0,
                )
                depth_reader.assert_called_once()

        self.assertIs(
            out_of_view.status,
            TsdfContributionStatus.PROJECTION_OUTSIDE_IMAGE,
        )
        self.assertTrue(out_of_view.depth_decoded)
        self.assertIsNotNone(out_of_view.camera_xyz_m)
        self.assertIsNotNone(out_of_view.projected_uv)
        self.assertIsNone(out_of_view.pixel_uv)
        self.assertEqual(out_of_view.weight_delta, 0)

        self.assertIs(
            behind.status,
            TsdfContributionStatus.CAMERA_Z_NONPOSITIVE,
        )
        self.assertTrue(behind.depth_decoded)
        self.assertIsNotNone(behind.camera_xyz_m)
        assert behind.camera_xyz_m is not None
        self.assertLess(behind.camera_xyz_m[2], 0.0)
        self.assertIsNone(behind.projected_uv)
        self.assertEqual(behind.weight_delta, 0)

        self.assertIs(
            invalid_depth.status,
            TsdfContributionStatus.DEPTH_INVALID,
        )
        self.assertTrue(invalid_depth.depth_decoded)
        self.assertEqual(invalid_depth.pixel_uv, (1, 1))
        self.assertIsNone(invalid_depth.measured_depth_m)
        self.assertIsNone(invalid_depth.signed_distance_m)
        self.assertIsNone(invalid_depth.tsdf_sum_delta)
        self.assertEqual(invalid_depth.weight_delta, 0)

    def test_storage_address_session_and_sequence_validation_is_strict(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            plan, session, storage = load_case(temporary_root)
            address = require_address(storage, (8, -1, -1))
            cases = (
                (
                    "storage",
                    object(),
                    address,
                    session,
                    0,
                    "allocated TsdfBlockStorage",
                ),
                (
                    "address",
                    storage,
                    object(),
                    session,
                    0,
                    "TsdfVoxelAddress",
                ),
                (
                    "session",
                    storage,
                    address,
                    object(),
                    0,
                    "loaded ScanSession",
                ),
                (
                    "sequence-bool",
                    storage,
                    address,
                    session,
                    True,
                    "expected an integer",
                ),
                (
                    "sequence-float",
                    storage,
                    address,
                    session,
                    0.0,
                    "expected an integer",
                ),
                (
                    "sequence-negative",
                    storage,
                    address,
                    session,
                    -1,
                    "non-negative integer",
                ),
                (
                    "sequence-past-end",
                    storage,
                    address,
                    session,
                    2,
                    "outside replay range",
                ),
                (
                    "session-id",
                    storage,
                    address,
                    replace(session, session_id="different-session"),
                    0,
                    "session_id does not match",
                ),
                (
                    "forged-address",
                    storage,
                    replace(
                        address,
                        storage_flat_index=address.storage_flat_index + 1,
                    ),
                    session,
                    0,
                    "does not match the allocated block storage",
                ),
            )
            for (
                name,
                changed_storage,
                changed_address,
                changed_session,
                sequence,
                expected_message,
            ) in cases:
                with self.subTest(name=name):
                    with self.assertRaises(TsdfError) as raised:
                        evaluate_tsdf_voxel_contribution(  # type: ignore[arg-type]
                            changed_storage,
                            changed_address,
                            changed_session,
                            sequence,
                        )
                    self.assertIn(
                        expected_message,
                        str(raised.exception),
                    )

            _, stride_session, stride_storage = load_case(
                temporary_root,
                plan_name="stride.sftplan",
                frame_stride=2,
            )
            stride_address = require_address(
                stride_storage,
                (8, -1, -1),
            )
            with self.assertRaises(TsdfError) as raised:
                evaluate_tsdf_voxel_contribution(
                    stride_storage,
                    stride_address,
                    stride_session,  # type: ignore[arg-type]
                    1,
                )
            self.assertIn("not selected", str(raised.exception))

            invalid_plan_cases = (
                (
                    replace(plan, voxel_size_m=-0.125),
                    "voxel_size_m",
                ),
                (
                    replace(plan, truncation_m=0.0),
                    "truncation_m",
                ),
                (
                    replace(plan, truncation_m=0.0625),
                    "greater than or equal",
                ),
                (
                    replace(plan, block_extent_m=2.0),
                    "block_extent_m",
                ),
                (
                    replace(plan, frame_stride=0),
                    "frame_stride",
                ),
            )
            for changed_plan, expected_message in invalid_plan_cases:
                with self.subTest(plan_error=expected_message):
                    changed_storage = TsdfBlockStorage(
                        source_plan=changed_plan,
                        block_indices=storage.block_indices,
                        tsdf_sums=storage.tsdf_sums,
                        weights=storage.weights,
                    )
                    with self.assertRaises(TsdfError) as raised:
                        evaluate_tsdf_voxel_contribution(
                            changed_storage,
                            address,
                            session,  # type: ignore[arg-type]
                            0,
                        )
                    self.assertIn(
                        expected_message,
                        str(raised.exception),
                    )

            with self.assertRaises(TsdfError) as raised:
                TsdfVoxelContribution(
                    address=address,
                    observation_sequence=0,
                    status=TsdfContributionStatus.CONTRIBUTES,
                    world_xyz_m=(1.0625, -0.0625, -0.0625),
                )
            self.assertIn(
                "internally inconsistent",
                str(raised.exception),
            )

            changed_fixture = copy_fixture(
                temporary_root,
                "changed.vgsession",
            )
            _, changed_session, changed_storage = load_case(
                temporary_root,
                session_path=changed_fixture,
                plan_name="changed.sftplan",
            )
            changed_address = require_address(
                changed_storage,
                (8, -1, -1),
            )
            depth_path = (
                changed_fixture / "data" / "depth" / "000000.pgm"
            )
            depth_path.write_text(
                depth_path.read_text(encoding="ascii").replace(
                    "1000",
                    "999",
                    1,
                ),
                encoding="ascii",
            )
            with self.assertRaises(TsdfError) as raised:
                evaluate_tsdf_voxel_contribution(
                    changed_storage,
                    changed_address,
                    changed_session,  # type: ignore[arg-type]
                    0,
                )
            self.assertIn("replay digest", str(raised.exception))

    def test_evaluation_preserves_exact_storage_bytes_identities_and_inputs(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            fixture = copy_fixture(temporary_root)
            _, session, storage = load_case(
                temporary_root,
                session_path=fixture,
            )
            accepted_address = require_address(storage, (8, -1, -1))
            skipped_address = require_address(storage, (0, 7, -1))
            storage.tsdf_sums[1, 2, 3, 4] = -0.75
            storage.weights[1, 2, 3, 4] = 9
            before_tree = tree_snapshot(temporary_root)
            before_storage = (
                storage.block_indices,
                id(storage.tsdf_sums),
                id(storage.weights),
                storage.tsdf_sums.shape,
                storage.weights.shape,
                storage.tsdf_sums.tobytes(),
                storage.weights.tobytes(),
                storage.payload_bytes,
                storage.nonzero_sum_count,
                storage.nonzero_weight_count,
            )
            with patch(
                "spatialforge.tsdf_block_storage.np.zeros"
            ) as zeros:
                accepted = evaluate_tsdf_voxel_contribution(
                    storage,
                    accepted_address,
                    session,  # type: ignore[arg-type]
                    0,
                )
                skipped = evaluate_tsdf_voxel_contribution(
                    storage,
                    skipped_address,
                    session,  # type: ignore[arg-type]
                    0,
                )
                zeros.assert_not_called()
            after_storage = (
                storage.block_indices,
                id(storage.tsdf_sums),
                id(storage.weights),
                storage.tsdf_sums.shape,
                storage.weights.shape,
                storage.tsdf_sums.tobytes(),
                storage.weights.tobytes(),
                storage.payload_bytes,
                storage.nonzero_sum_count,
                storage.nonzero_weight_count,
            )
            after_tree = tree_snapshot(temporary_root)

        self.assertTrue(accepted.contributes)
        self.assertFalse(skipped.contributes)
        self.assertEqual(before_storage, after_storage)
        self.assertEqual(before_tree, after_tree)
        self.assertEqual(storage.block_count, 8)
        self.assertEqual(storage.voxel_slots, 4096)


class TsdfVoxelContributionCliTests(unittest.TestCase):
    def test_cli_reports_exact_fixture_contribution_read_only(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            session_path = copy_fixture(temporary_root)
            plan_path = create_plan(
                temporary_root,
                session_path=session_path,
            )
            before = tree_snapshot(temporary_root)
            stdout = io.StringIO()
            stderr = io.StringIO()
            actual_zeros = np.zeros
            from spatialforge import tsdf_voxel_contribution as contribution_module

            actual_read_depth = contribution_module._read_depth
            with (
                forbidden_reconstruction_calls() as forbidden,
                patch(
                    "spatialforge.tsdf_block_storage.np.zeros",
                    wraps=actual_zeros,
                ) as zeros,
                patch(
                    "spatialforge.tsdf_voxel_contribution._read_depth",
                    wraps=actual_read_depth,
                ) as depth_reader,
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                exit_code = main(
                    [
                        "reconstruct",
                        "tsdf-block-contribution",
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
            after = tree_snapshot(temporary_root)

        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(before, after)
        self.assertEqual(zeros.call_count, 2)
        depth_reader.assert_called_once()
        for forbidden_call in forbidden:
            forbidden_call.assert_not_called()
        output = stdout.getvalue()
        self.assertIn("TSDF BLOCK CONTRIBUTION CHECK", output)
        self.assertIn("artifact: valid", output)
        self.assertIn("session_replay: matched", output)
        self.assertIn("observation_sequence: 0", output)
        self.assertIn(
            "voxel: global=(8, -1, -1) block=(1, -1, -1) "
            "local=(0, 7, 7) row=1 array=(1, 7, 7, 0) "
            "storage_flat=1016",
            output,
        )
        self.assertIn(
            "world_xyz_m: (1.062500000, -0.062500000, -0.062500000)",
            output,
        )
        self.assertIn(
            "camera_xyz_m: (0.062500000, 0.062500000, 1.062500000)",
            output,
        )
        self.assertIn(
            "projected_uv: (0.617647059, 0.617647059)",
            output,
        )
        self.assertIn("pixel_uv: (1, 1)", output)
        self.assertIn("depth_decoded: yes", output)
        self.assertIn("measured_depth_m: 1.000000000", output)
        self.assertIn("signed_distance_m: -0.062500000", output)
        self.assertIn("evaluation: contributes", output)
        self.assertIn(
            "proposed_delta: tsdf_sum=-0.125000000 weight=1",
            output,
        )
        self.assertIn("contributions_applied: 0", output)
        self.assertIn("fusion_performed: no", output)
        self.assertIn("storage_mutated: no", output)
        self.assertIn("missing_blocks_created: no", output)
        self.assertIn("artifact_written: no", output)
        self.assertIn(PLAN_SHA256, output)
        self.assertIn(REPLAY_SHA256, output)

    def test_cli_unplanned_voxel_fails_before_evaluation_read_only(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            session_path = copy_fixture(temporary_root)
            plan_path = create_plan(
                temporary_root,
                session_path=session_path,
            )
            before = tree_snapshot(temporary_root)
            stdout = io.StringIO()
            stderr = io.StringIO()
            actual_zeros = np.zeros
            with (
                forbidden_reconstruction_calls() as forbidden,
                patch(
                    "spatialforge.tsdf_block_storage.np.zeros",
                    wraps=actual_zeros,
                ) as zeros,
                patch(
                    "spatialforge.cli.evaluate_tsdf_voxel_contribution"
                ) as evaluator,
                patch(
                    "spatialforge.tsdf_voxel_contribution._read_depth"
                ) as depth_reader,
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                exit_code = main(
                    [
                        "reconstruct",
                        "tsdf-block-contribution",
                        str(plan_path),
                        str(session_path),
                        "--voxel",
                        "-1",
                        "0",
                        "0",
                    ]
                )
                evaluator.assert_not_called()
                depth_reader.assert_not_called()
            after = tree_snapshot(temporary_root)

        self.assertEqual(exit_code, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(before, after)
        self.assertEqual(zeros.call_count, 2)
        for forbidden_call in forbidden:
            forbidden_call.assert_not_called()
        error = stderr.getvalue()
        self.assertIn("TSDF BLOCK CONTRIBUTION FAILED", error)
        self.assertIn("is not in a planned block", error)
        self.assertNotIn("Traceback", error)

    def test_cli_invalid_sequence_fails_before_depth_decode_read_only(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            session_path = copy_fixture(temporary_root)
            plan_path = create_plan(
                temporary_root,
                session_path=session_path,
            )
            before = tree_snapshot(temporary_root)
            stdout = io.StringIO()
            stderr = io.StringIO()
            actual_zeros = np.zeros
            with (
                forbidden_reconstruction_calls() as forbidden,
                patch(
                    "spatialforge.tsdf_block_storage.np.zeros",
                    wraps=actual_zeros,
                ) as zeros,
                patch(
                    "spatialforge.tsdf_voxel_contribution._read_depth"
                ) as depth_reader,
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                exit_code = main(
                    [
                        "reconstruct",
                        "tsdf-block-contribution",
                        str(plan_path),
                        str(session_path),
                        "--observation-sequence",
                        "2",
                        "--voxel",
                        "8",
                        "-1",
                        "-1",
                    ]
                )
                depth_reader.assert_not_called()
            after = tree_snapshot(temporary_root)

        self.assertEqual(exit_code, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(before, after)
        self.assertEqual(zeros.call_count, 2)
        for forbidden_call in forbidden:
            forbidden_call.assert_not_called()
        error = stderr.getvalue()
        self.assertIn("TSDF BLOCK CONTRIBUTION FAILED", error)
        self.assertIn("outside replay range", error)
        self.assertNotIn("Traceback", error)


if __name__ == "__main__":
    unittest.main()
