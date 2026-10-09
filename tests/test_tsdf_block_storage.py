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

from spatialforge import allocate_empty_tsdf_blocks
from spatialforge.cli import main
from spatialforge.errors import TsdfError
from spatialforge.session_loader import load_scan_session
from spatialforge.tsdf_block_plan import (
    MAX_BLOCK_INDEX,
    MAX_PLANNED_BLOCKS,
    TSDF_BLOCK_RESOLUTION,
    plan_tsdf_blocks,
)
from spatialforge.tsdf_block_plan_loader import load_tsdf_block_plan
from spatialforge.tsdf_block_storage import (
    MAX_TSDF_BLOCK_STORAGE_BLOCKS,
    MAX_TSDF_BLOCK_STORAGE_BYTES,
    TSDF_BLOCK_STORAGE_BYTES_PER_VOXEL,
    TSDF_BLOCK_VOXELS,
    TSDF_SUM_DTYPE,
    TSDF_WEIGHT_DTYPE,
    _preflight_storage_plan,
)


TEST_ROOT = Path(__file__).resolve().parent
FIXTURE = TEST_ROOT / "fixtures" / "minimal.vgsession"
PLAN_ARGUMENTS = {
    "voxel_size_m": 0.125,
    "truncation_m": 0.5,
}
EXPECTED_ACTIVE_BLOCKS = (
    (0, -1, -1),
    (1, -1, -1),
    (0, 0, -1),
    (1, 0, -1),
    (0, -1, 0),
    (1, -1, 0),
    (0, 0, 0),
    (1, 0, 0),
)
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
    "spatialforge.point_cloud._read_depth",
    "spatialforge.tsdf._read_depth",
    "spatialforge.tsdf._integrate_tsdf",
    "spatialforge.tsdf_block_plan._read_depth",
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
) -> Path:
    output = parent / name
    plan_tsdf_blocks(
        load_scan_session(session_path),
        output,
        **PLAN_ARGUMENTS,
    )
    return output


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


class TsdfBlockStorageTests(unittest.TestCase):
    def test_fixture_allocates_exact_canonical_zero_storage(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            plan = load_tsdf_block_plan(create_plan(temporary_root))
            session = load_scan_session(FIXTURE)
            storage = allocate_empty_tsdf_blocks(plan, session)

        self.assertIs(storage.source_plan, plan)
        self.assertEqual(storage.block_indices, EXPECTED_ACTIVE_BLOCKS)
        self.assertEqual(storage.block_count, 8)
        self.assertEqual(storage.block_resolution, 8)
        self.assertEqual(storage.voxel_slots, 4096)
        self.assertEqual(storage.voxel_slots, plan.planned_voxel_slots)
        self.assertEqual(storage.tsdf_sums.shape, (8, 8, 8, 8))
        self.assertEqual(storage.weights.shape, (8, 8, 8, 8))
        self.assertEqual(storage.tsdf_sums.dtype, TSDF_SUM_DTYPE)
        self.assertEqual(storage.weights.dtype, TSDF_WEIGHT_DTYPE)
        self.assertTrue(storage.tsdf_sums.flags.c_contiguous)
        self.assertTrue(storage.weights.flags.c_contiguous)
        self.assertEqual(storage.tsdf_sum_bytes, 32_768)
        self.assertEqual(storage.weight_bytes, 16_384)
        self.assertEqual(storage.payload_bytes, 49_152)
        self.assertEqual(storage.nonzero_sum_count, 0)
        self.assertEqual(storage.nonzero_weight_count, 0)
        self.assertEqual(storage.unknown_voxel_count, 4096)
        self.assertTrue(np.all(storage.tsdf_sums == 0.0))
        self.assertTrue(np.all(storage.weights == 0))

        flat_cases = (
            ((0, 0, 0, 0), 0),
            ((0, 0, 0, 7), 7),
            ((0, 0, 1, 0), 8),
            ((0, 1, 0, 0), 64),
            ((0, 7, 7, 7), 511),
            ((1, 0, 0, 0), 512),
            ((7, 7, 7, 7), 4095),
        )
        flattened = storage.tsdf_sums.reshape(-1)
        for coordinates, expected_flat_index in flat_cases:
            with self.subTest(coordinates=coordinates):
                row, z_index, y_index, x_index = coordinates
                formula_index = (
                    row * TSDF_BLOCK_VOXELS
                    + x_index
                    + TSDF_BLOCK_RESOLUTION
                    * (
                        y_index
                        + TSDF_BLOCK_RESOLUTION * z_index
                    )
                )
                self.assertEqual(formula_index, expected_flat_index)
                storage.tsdf_sums[coordinates] = expected_flat_index + 1.0
                self.assertEqual(
                    flattened[expected_flat_index],
                    expected_flat_index + 1.0,
                )
                storage.tsdf_sums[coordinates] = 0.0

    def test_forged_plan_storage_fields_fail_before_verification_or_allocation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            plan = load_tsdf_block_plan(create_plan(Path(temporary_directory)))
            session = load_scan_session(FIXTURE)
            cases = (
                (
                    "wrong-object",
                    object(),
                    "loaded TsdfBlockPlan",
                ),
                (
                    "resolution",
                    replace(plan, block_resolution=4),
                    "block resolution 8",
                ),
                (
                    "empty",
                    replace(
                        plan,
                        active_blocks=(),
                        planned_voxel_slots=0,
                    ),
                    "at least one active block",
                ),
                (
                    "boolean-index",
                    replace(
                        plan,
                        active_blocks=((True, 0, 0),),  # type: ignore[arg-type]
                        planned_voxel_slots=TSDF_BLOCK_VOXELS,
                    ),
                    "expected an integer",
                ),
                (
                    "out-of-range-index",
                    replace(
                        plan,
                        active_blocks=((MAX_BLOCK_INDEX + 1, 0, 0),),
                        planned_voxel_slots=TSDF_BLOCK_VOXELS,
                    ),
                    "outside signed 32-bit",
                ),
                (
                    "duplicate",
                    replace(
                        plan,
                        active_blocks=((0, 0, 0), (0, 0, 0)),
                        planned_voxel_slots=2 * TSDF_BLOCK_VOXELS,
                    ),
                    "unique, strictly x-fastest",
                ),
                (
                    "unordered",
                    replace(
                        plan,
                        active_blocks=((1, 0, 0), (0, 0, 0)),
                        planned_voxel_slots=2 * TSDF_BLOCK_VOXELS,
                    ),
                    "unique, strictly x-fastest",
                ),
                (
                    "slot-count",
                    replace(
                        plan,
                        planned_voxel_slots=plan.planned_voxel_slots + 1,
                    ),
                    "voxel slots do not match",
                ),
            )

            for name, changed_plan, expected_message in cases:
                with self.subTest(name=name):
                    with (
                        patch(
                            "spatialforge.tsdf_block_storage."
                            "verify_tsdf_block_plan_replay"
                        ) as verifier,
                        patch(
                            "spatialforge.tsdf_block_storage.np.zeros"
                        ) as zeros,
                    ):
                        with self.assertRaises(TsdfError) as raised:
                            allocate_empty_tsdf_blocks(  # type: ignore[arg-type]
                                changed_plan,
                                session,
                            )
                        verifier.assert_not_called()
                        zeros.assert_not_called()
                    self.assertIn(
                        expected_message,
                        str(raised.exception),
                    )

    def test_replay_mismatch_fails_before_numeric_allocation(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            plan = load_tsdf_block_plan(create_plan(Path(temporary_directory)))
            changed_plan = replace(
                plan,
                replay_digest_sha256="0" * 64,
            )
            with patch(
                "spatialforge.tsdf_block_storage.np.zeros"
            ) as zeros:
                with self.assertRaises(TsdfError) as raised:
                    allocate_empty_tsdf_blocks(
                        changed_plan,
                        load_scan_session(FIXTURE),
                    )
                zeros.assert_not_called()

        self.assertIn("replay digest", str(raised.exception))

    def test_exact_payload_cap_is_preflighted_without_large_arrays(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            plan = load_tsdf_block_plan(create_plan(Path(temporary_directory)))

        # Storage holds exactly what the planner will plan.
        self.assertEqual(MAX_TSDF_BLOCK_STORAGE_BLOCKS, MAX_PLANNED_BLOCKS)
        self.assertEqual(MAX_TSDF_BLOCK_STORAGE_BLOCKS, 250_000)
        self.assertEqual(MAX_TSDF_BLOCK_STORAGE_BYTES, 1_536_000_000)
        allowed_blocks = tuple(
            (block_x, 0, 0)
            for block_x in range(MAX_TSDF_BLOCK_STORAGE_BLOCKS)
        )
        allowed_plan = replace(
            plan,
            active_blocks=allowed_blocks,
            planned_voxel_slots=(
                MAX_TSDF_BLOCK_STORAGE_BLOCKS * TSDF_BLOCK_VOXELS
            ),
        )
        block_indices, shape, payload_bytes = _preflight_storage_plan(
            allowed_plan
        )

        self.assertIs(block_indices, allowed_blocks)
        self.assertEqual(shape, (250_000, 8, 8, 8))
        self.assertEqual(payload_bytes, 1_536_000_000)
        self.assertLessEqual(
            payload_bytes,
            MAX_TSDF_BLOCK_STORAGE_BYTES,
        )

        rejected_count = MAX_TSDF_BLOCK_STORAGE_BLOCKS + 1
        rejected_blocks = allowed_blocks + ((rejected_count - 1, 0, 0),)
        rejected_plan = replace(
            plan,
            active_blocks=rejected_blocks,
            planned_voxel_slots=rejected_count * TSDF_BLOCK_VOXELS,
        )
        rejected_bytes = (
            rejected_count
            * TSDF_BLOCK_VOXELS
            * TSDF_BLOCK_STORAGE_BYTES_PER_VOXEL
        )
        self.assertEqual(rejected_bytes, 1_536_006_144)
        with self.assertRaises(TsdfError) as raised:
            _preflight_storage_plan(rejected_plan)

        self.assertIn(
            "1536006144 numeric payload bytes", str(raised.exception)
        )
        self.assertIn("(250000 blocks)", str(raised.exception))

    def test_memory_error_is_translated_after_partial_buffer_allocation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            plan = load_tsdf_block_plan(create_plan(Path(temporary_directory)))
            successful_sums = np.zeros(
                (8, 8, 8, 8),
                dtype=TSDF_SUM_DTYPE,
                order="C",
            )
            with patch(
                "spatialforge.tsdf_block_storage.np.zeros",
                side_effect=(
                    successful_sums,
                    MemoryError("synthetic allocation failure"),
                ),
            ) as zeros:
                with self.assertRaises(TsdfError) as raised:
                    allocate_empty_tsdf_blocks(
                        plan,
                        load_scan_session(FIXTURE),
                    )

        self.assertEqual(zeros.call_count, 2)
        self.assertIsInstance(raised.exception.__cause__, MemoryError)
        self.assertIn(
            "cannot allocate empty TSDF block storage",
            str(raised.exception),
        )

    def test_allocations_have_independent_mutable_buffers_and_frozen_metadata(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            plan = load_tsdf_block_plan(create_plan(Path(temporary_directory)))
            session = load_scan_session(FIXTURE)
            first = allocate_empty_tsdf_blocks(plan, session)
            second = allocate_empty_tsdf_blocks(plan, session)

        self.assertTrue(first.tsdf_sums.flags.writeable)
        self.assertTrue(first.weights.flags.writeable)
        self.assertFalse(np.shares_memory(first.tsdf_sums, first.weights))
        self.assertFalse(np.shares_memory(first.tsdf_sums, second.tsdf_sums))
        self.assertFalse(np.shares_memory(first.weights, second.weights))

        first.tsdf_sums[0, 0, 0, 0] = -0.25
        first.weights[0, 0, 0, 0] = 1
        self.assertEqual(first.nonzero_sum_count, 1)
        self.assertEqual(first.nonzero_weight_count, 1)
        self.assertEqual(first.unknown_voxel_count, 4095)
        self.assertEqual(second.tsdf_sums[0, 0, 0, 0], 0.0)
        self.assertEqual(second.weights[0, 0, 0, 0], 0)

        with self.assertRaises(FrozenInstanceError):
            first.block_indices = ()  # type: ignore[misc]
        with self.assertRaises(TypeError):
            first.block_indices[0] = (9, 9, 9)  # type: ignore[index]
        self.assertFalse(hasattr(first, "__dict__"))


class TsdfBlockStorageCliTests(unittest.TestCase):
    def test_cli_reports_allocation_and_is_strictly_read_only(self) -> None:
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
            with (
                forbidden_reconstruction_calls() as forbidden,
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                exit_code = main(
                    [
                        "reconstruct",
                        "tsdf-block-allocate",
                        str(plan_path),
                        str(session_path),
                    ]
                )
            after = tree_snapshot(temporary_root)

        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(before, after)
        for forbidden_call in forbidden:
            forbidden_call.assert_not_called()
        output = stdout.getvalue()
        self.assertIn("TSDF BLOCK ALLOCATION CHECK", output)
        self.assertIn("artifact: valid", output)
        self.assertIn("session_replay: matched", output)
        self.assertIn("depth_decoded: no", output)
        self.assertIn("geometry_recomputed: no", output)
        self.assertIn("fusion_performed: no", output)
        self.assertIn("artifact_written: no", output)
        self.assertIn(
            "allocation: blocks=8 resolution=8 voxel_slots=4096",
            output,
        )
        self.assertIn(
            "layout: shape=(8, 8, 8, 8) "
            "axes=block-z-y-x x_fastest=yes",
            output,
        )
        self.assertIn(
            "dtypes: tsdf_sums=float64 weights=uint32",
            output,
        )
        self.assertIn(
            "zero_state: nonzero_sums=0 nonzero_weights=0 "
            "unknown_voxels=4096",
            output,
        )
        self.assertIn(
            "payload_bytes: tsdf_sums=32768 weights=16384 total=49152",
            output,
        )
        self.assertIn(
            "block_rows: first=(0, -1, -1) last=(1, 0, 0)",
            output,
        )
        self.assertIn(PLAN_SHA256, output)
        self.assertIn(REPLAY_SHA256, output)

    def test_cli_replay_failure_is_read_only_and_allocates_nothing(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            session_path = copy_fixture(temporary_root)
            plan_path = create_plan(
                temporary_root,
                session_path=session_path,
            )
            document = json.loads(plan_path.read_text(encoding="ascii"))
            document["replay_digest_sha256"] = "0" * 64
            plan_path.write_text(
                json.dumps(
                    document,
                    ensure_ascii=True,
                    allow_nan=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="ascii",
            )
            before = tree_snapshot(temporary_root)
            stdout = io.StringIO()
            stderr = io.StringIO()
            with (
                forbidden_reconstruction_calls() as forbidden,
                patch(
                    "spatialforge.tsdf_block_storage.np.zeros"
                ) as zeros,
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                exit_code = main(
                    [
                        "reconstruct",
                        "tsdf-block-allocate",
                        str(plan_path),
                        str(session_path),
                    ]
                )
                zeros.assert_not_called()
            after = tree_snapshot(temporary_root)

        self.assertEqual(exit_code, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(before, after)
        for forbidden_call in forbidden:
            forbidden_call.assert_not_called()
        error = stderr.getvalue()
        self.assertIn("TSDF BLOCK ALLOCATION FAILED", error)
        self.assertIn("replay digest", error)
        self.assertNotIn("Traceback", error)


if __name__ == "__main__":
    unittest.main()
