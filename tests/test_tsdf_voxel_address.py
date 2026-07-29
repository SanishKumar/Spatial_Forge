from __future__ import annotations

import io
import shutil
import tempfile
import unittest
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from dataclasses import FrozenInstanceError, replace
from itertools import product
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from spatialforge import (
    allocate_empty_tsdf_blocks,
    compose_tsdf_global_voxel_index,
    locate_tsdf_voxel,
)
from spatialforge.cli import main
from spatialforge.errors import TsdfError
from spatialforge.session_loader import load_scan_session
from spatialforge.tsdf_block_plan import (
    MAX_BLOCK_INDEX,
    MIN_BLOCK_INDEX,
    TSDF_BLOCK_RESOLUTION,
    plan_tsdf_blocks,
)
from spatialforge.tsdf_block_plan_loader import (
    TsdfBlockPlan,
    load_tsdf_block_plan,
)
from spatialforge.tsdf_block_storage import (
    TSDF_BLOCK_VOXELS,
    TSDF_SUM_DTYPE,
    TSDF_WEIGHT_DTYPE,
    TsdfBlockStorage,
)
from spatialforge.tsdf_voxel_address import (
    MAX_TSDF_GLOBAL_VOXEL_INDEX,
    MIN_TSDF_GLOBAL_VOXEL_INDEX,
    TsdfVoxelAddress,
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


def allocate_fixture_storage(
    parent: Path,
) -> tuple[TsdfBlockPlan, TsdfBlockStorage]:
    plan = load_tsdf_block_plan(create_plan(parent))
    storage = allocate_empty_tsdf_blocks(
        plan,
        load_scan_session(FIXTURE),
    )
    return plan, storage


def allocate_storage_for_blocks(
    plan: TsdfBlockPlan,
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
    return allocate_empty_tsdf_blocks(
        changed_plan,
        load_scan_session(FIXTURE),
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


class TsdfVoxelAddressTests(unittest.TestCase):
    def test_fixture_addresses_pin_rows_local_array_and_flat_indices(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            _, storage = allocate_fixture_storage(Path(temporary_directory))

        cases = (
            (
                (0, -8, -8),
                (0, -1, -1),
                (0, 0, 0),
                0,
                0,
                0,
            ),
            (
                (7, -1, -1),
                (0, -1, -1),
                (7, 7, 7),
                0,
                511,
                511,
            ),
            (
                (8, -8, -8),
                (1, -1, -1),
                (0, 0, 0),
                1,
                0,
                512,
            ),
            (
                (0, 0, -8),
                (0, 0, -1),
                (0, 0, 0),
                2,
                0,
                1024,
            ),
            (
                (0, -8, 0),
                (0, -1, 0),
                (0, 0, 0),
                4,
                0,
                2048,
            ),
            (
                (0, 0, 0),
                (0, 0, 0),
                (0, 0, 0),
                6,
                0,
                3072,
            ),
            (
                (8, 0, 0),
                (1, 0, 0),
                (0, 0, 0),
                7,
                0,
                3584,
            ),
            (
                (15, 7, 7),
                (1, 0, 0),
                (7, 7, 7),
                7,
                511,
                4095,
            ),
        )

        first_address: TsdfVoxelAddress | None = None
        flattened = storage.tsdf_sums.reshape(-1)
        for position, (
            global_index,
            block_index,
            local_index,
            block_row,
            local_flat_index,
            storage_flat_index,
        ) in enumerate(cases):
            with self.subTest(global_index=global_index):
                address = locate_tsdf_voxel(storage, global_index)
                self.assertIsInstance(address, TsdfVoxelAddress)
                assert address is not None
                if first_address is None:
                    first_address = address
                self.assertEqual(address.global_index_xyz, global_index)
                self.assertEqual(address.block_index_xyz, block_index)
                self.assertEqual(address.local_index_xyz, local_index)
                self.assertEqual(address.block_row, block_row)
                self.assertEqual(
                    address.array_index_bzyx,
                    (
                        block_row,
                        local_index[2],
                        local_index[1],
                        local_index[0],
                    ),
                )
                self.assertEqual(
                    address.local_flat_index,
                    local_flat_index,
                )
                self.assertEqual(
                    address.storage_flat_index,
                    storage_flat_index,
                )
                self.assertEqual(
                    compose_tsdf_global_voxel_index(
                        address.block_index_xyz,
                        address.local_index_xyz,
                    ),
                    global_index,
                )

                sentinel = float(position + 1)
                storage.tsdf_sums[address.array_index_bzyx] = sentinel
                self.assertEqual(flattened[storage_flat_index], sentinel)
                storage.tsdf_sums[address.array_index_bzyx] = 0.0

        assert first_address is not None
        with self.assertRaises(FrozenInstanceError):
            first_address.block_row = 9  # type: ignore[misc]
        self.assertFalse(hasattr(first_address, "__dict__"))

    def test_signed_boundaries_use_floor_division_on_both_sides_of_zero(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            plan, _ = allocate_fixture_storage(Path(temporary_directory))
            storage = allocate_storage_for_blocks(
                plan,
                (
                    (-2, 0, 0),
                    (-1, 0, 0),
                    (0, 0, 0),
                    (1, 0, 0),
                ),
            )

        cases = (
            (-9, (-2, 0, 0), (7, 0, 0), 0, 7, 7),
            (-8, (-1, 0, 0), (0, 0, 0), 1, 0, 512),
            (-1, (-1, 0, 0), (7, 0, 0), 1, 7, 519),
            (0, (0, 0, 0), (0, 0, 0), 2, 0, 1024),
            (7, (0, 0, 0), (7, 0, 0), 2, 7, 1031),
            (8, (1, 0, 0), (0, 0, 0), 3, 0, 1536),
        )
        for (
            global_x,
            block_index,
            local_index,
            block_row,
            local_flat,
            storage_flat,
        ) in cases:
            with self.subTest(global_x=global_x):
                address = locate_tsdf_voxel(storage, (global_x, 0, 0))
                self.assertIsNotNone(address)
                assert address is not None
                self.assertEqual(address.block_index_xyz, block_index)
                self.assertEqual(address.local_index_xyz, local_index)
                self.assertEqual(address.block_row, block_row)
                self.assertEqual(address.local_flat_index, local_flat)
                self.assertEqual(address.storage_flat_index, storage_flat)

    def test_signed_extremes_are_exact_and_outside_range_is_rejected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            plan, _ = allocate_fixture_storage(Path(temporary_directory))
            storage = allocate_storage_for_blocks(
                plan,
                (
                    (MIN_BLOCK_INDEX, MIN_BLOCK_INDEX, MIN_BLOCK_INDEX),
                    (MAX_BLOCK_INDEX, MAX_BLOCK_INDEX, MAX_BLOCK_INDEX),
                ),
            )

        self.assertEqual(MIN_TSDF_GLOBAL_VOXEL_INDEX, -17_179_869_184)
        self.assertEqual(MAX_TSDF_GLOBAL_VOXEL_INDEX, 17_179_869_183)
        minimum_global = (MIN_TSDF_GLOBAL_VOXEL_INDEX,) * 3
        maximum_global = (MAX_TSDF_GLOBAL_VOXEL_INDEX,) * 3
        minimum_address = locate_tsdf_voxel(storage, minimum_global)
        maximum_address = locate_tsdf_voxel(storage, maximum_global)

        self.assertIsNotNone(minimum_address)
        self.assertIsNotNone(maximum_address)
        assert minimum_address is not None
        assert maximum_address is not None
        self.assertEqual(minimum_address.block_row, 0)
        self.assertEqual(minimum_address.local_index_xyz, (0, 0, 0))
        self.assertEqual(minimum_address.storage_flat_index, 0)
        self.assertEqual(maximum_address.block_row, 1)
        self.assertEqual(maximum_address.local_index_xyz, (7, 7, 7))
        self.assertEqual(maximum_address.storage_flat_index, 1023)
        self.assertEqual(
            compose_tsdf_global_voxel_index(
                minimum_address.block_index_xyz,
                minimum_address.local_index_xyz,
            ),
            minimum_global,
        )
        self.assertEqual(
            compose_tsdf_global_voxel_index(
                maximum_address.block_index_xyz,
                maximum_address.local_index_xyz,
            ),
            maximum_global,
        )
        self.assertIsNone(locate_tsdf_voxel(storage, (0, 0, 0)))

        invalid_globals = (
            (MIN_TSDF_GLOBAL_VOXEL_INDEX - 1, 0, 0),
            (MAX_TSDF_GLOBAL_VOXEL_INDEX + 1, 0, 0),
            (0, MIN_TSDF_GLOBAL_VOXEL_INDEX - 1, 0),
            (0, MAX_TSDF_GLOBAL_VOXEL_INDEX + 1, 0),
            (0, 0, MIN_TSDF_GLOBAL_VOXEL_INDEX - 1),
            (0, 0, MAX_TSDF_GLOBAL_VOXEL_INDEX + 1),
        )
        for global_index in invalid_globals:
            with self.subTest(global_index=global_index):
                with self.assertRaises(TsdfError) as raised:
                    locate_tsdf_voxel(storage, global_index)
                self.assertIn("expected an integer in", str(raised.exception))

    def test_compose_and_locate_roundtrip_every_fixture_voxel_slot(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            _, storage = allocate_fixture_storage(Path(temporary_directory))

        seen_storage_flats: set[int] = set()
        for block_row, block_index in enumerate(EXPECTED_ACTIVE_BLOCKS):
            for local_index in product(range(8), repeat=3):
                global_index = compose_tsdf_global_voxel_index(
                    block_index,
                    local_index,
                )
                address = locate_tsdf_voxel(storage, global_index)
                self.assertIsNotNone(address)
                assert address is not None
                local_x, local_y, local_z = local_index
                expected_local_flat = (
                    local_x
                    + 8 * (local_y + 8 * local_z)
                )
                expected_storage_flat = (
                    block_row * TSDF_BLOCK_VOXELS
                    + expected_local_flat
                )
                self.assertEqual(address.block_row, block_row)
                self.assertEqual(address.local_index_xyz, local_index)
                self.assertEqual(
                    address.local_flat_index,
                    expected_local_flat,
                )
                self.assertEqual(
                    address.storage_flat_index,
                    expected_storage_flat,
                )
                seen_storage_flats.add(address.storage_flat_index)

        self.assertEqual(seen_storage_flats, set(range(4096)))

    def test_unplanned_misses_do_not_mutate_or_allocate_storage(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            _, storage = allocate_fixture_storage(Path(temporary_directory))

        storage.tsdf_sums[2, 3, 4, 5] = -0.75
        storage.weights[2, 3, 4, 5] = 9
        before = (
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
        misses = (
            (-1, 0, 0),
            (16, 0, 0),
            (0, 8, 0),
            (0, 0, 8),
        )
        with patch("spatialforge.tsdf_block_storage.np.zeros") as zeros:
            for global_index in misses:
                with self.subTest(global_index=global_index):
                    self.assertIsNone(
                        locate_tsdf_voxel(storage, global_index)
                    )
                    self.assertIsNone(
                        locate_tsdf_voxel(storage, global_index)
                    )
            zeros.assert_not_called()
        after = (
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

        self.assertEqual(before, after)
        self.assertEqual(storage.block_count, 8)
        self.assertEqual(storage.voxel_slots, 4096)

    def test_input_types_and_component_ranges_are_strict(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            _, storage = allocate_fixture_storage(Path(temporary_directory))

        locate_cases = (
            ("list", [0, 0, 0], "tuple of 3 integers"),
            ("arity", (0, 0), "tuple of 3 integers"),
            ("boolean", (True, 0, 0), "expected an integer"),
            ("float", (0.0, 0, 0), "expected an integer"),
            (
                "below",
                (MIN_TSDF_GLOBAL_VOXEL_INDEX - 1, 0, 0),
                "expected an integer in",
            ),
            (
                "above",
                (MAX_TSDF_GLOBAL_VOXEL_INDEX + 1, 0, 0),
                "expected an integer in",
            ),
        )
        for name, global_index, expected_message in locate_cases:
            with self.subTest(api="locate", name=name):
                with self.assertRaises(TsdfError) as raised:
                    locate_tsdf_voxel(  # type: ignore[arg-type]
                        storage,
                        global_index,
                    )
                self.assertIn(expected_message, str(raised.exception))

        with self.assertRaises(TsdfError) as raised:
            locate_tsdf_voxel(  # type: ignore[arg-type]
                object(),
                (0, 0, 0),
            )
        self.assertIn("allocated TsdfBlockStorage", str(raised.exception))

        compose_cases = (
            (
                "block-list",
                [0, 0, 0],
                (0, 0, 0),
                "block_index_xyz: expected a tuple",
            ),
            (
                "block-bool",
                (True, 0, 0),
                (0, 0, 0),
                "block_index_xyz[0]: expected an integer",
            ),
            (
                "block-below",
                (MIN_BLOCK_INDEX - 1, 0, 0),
                (0, 0, 0),
                "block_index_xyz[0]: expected an integer in",
            ),
            (
                "block-above",
                (MAX_BLOCK_INDEX + 1, 0, 0),
                (0, 0, 0),
                "block_index_xyz[0]: expected an integer in",
            ),
            (
                "local-float",
                (0, 0, 0),
                (0.0, 0, 0),
                "local_index_xyz[0]: expected an integer",
            ),
            (
                "local-below",
                (0, 0, 0),
                (-1, 0, 0),
                "local_index_xyz[0]: expected an integer in",
            ),
            (
                "local-above",
                (0, 0, 0),
                (8, 0, 0),
                "local_index_xyz[0]: expected an integer in",
            ),
        )
        for (
            name,
            block_index,
            local_index,
            expected_message,
        ) in compose_cases:
            with self.subTest(api="compose", name=name):
                with self.assertRaises(TsdfError) as raised:
                    compose_tsdf_global_voxel_index(  # type: ignore[arg-type]
                        block_index,
                        local_index,
                    )
                self.assertIn(expected_message, str(raised.exception))

    def test_mutated_layout_and_direct_forged_storage_are_rejected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            plan, storage = allocate_fixture_storage(Path(temporary_directory))

        shape = (8, 8, 8, 8)
        valid_sums = np.zeros(shape, dtype=TSDF_SUM_DTYPE)
        valid_weights = np.zeros(shape, dtype=TSDF_WEIGHT_DTYPE)
        noncontiguous_sums = np.zeros(
            (8, 8, 8, 16),
            dtype=TSDF_SUM_DTYPE,
        )[..., ::2]
        self.assertEqual(noncontiguous_sums.shape, shape)
        self.assertFalse(noncontiguous_sums.flags.c_contiguous)
        direct_cases = (
            (
                "rows",
                tuple(reversed(plan.active_blocks)),
                valid_sums,
                valid_weights,
                "rows must match",
            ),
            (
                "sum-shape",
                plan.active_blocks,
                np.zeros((4096,), dtype=TSDF_SUM_DTYPE),
                valid_weights,
                "canonical (8, 8, 8, 8) shape",
            ),
            (
                "sum-dtype",
                plan.active_blocks,
                np.zeros(shape, dtype=np.float32),
                valid_weights,
                "sums must use float64",
            ),
            (
                "weight-dtype",
                plan.active_blocks,
                valid_sums,
                np.zeros(shape, dtype=np.int64),
                "weights must use uint32",
            ),
            (
                "contiguity",
                plan.active_blocks,
                noncontiguous_sums,
                valid_weights,
                "must be C-contiguous",
            ),
        )
        for (
            name,
            block_indices,
            sums,
            weights,
            expected_message,
        ) in direct_cases:
            with self.subTest(name=name):
                with self.assertRaises(TsdfError) as raised:
                    TsdfBlockStorage(
                        source_plan=plan,
                        block_indices=block_indices,
                        tsdf_sums=sums,
                        weights=weights,
                    )
                self.assertIn(expected_message, str(raised.exception))

        storage.tsdf_sums.shape = (storage.voxel_slots,)
        with self.assertRaises(TsdfError) as raised:
            locate_tsdf_voxel(storage, (0, 0, 0))
        self.assertIn(
            "canonical (8, 8, 8, 8) shape",
            str(raised.exception),
        )


class TsdfVoxelAddressCliTests(unittest.TestCase):
    def test_cli_reports_planned_and_unplanned_queries_read_only(self) -> None:
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
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                exit_code = main(
                    [
                        "reconstruct",
                        "tsdf-block-address",
                        str(plan_path),
                        str(session_path),
                        "--voxel",
                        "0",
                        "-8",
                        "-8",
                        "--voxel",
                        "8",
                        "0",
                        "0",
                        "--voxel",
                        "-1",
                        "0",
                        "0",
                    ]
                )
            after = tree_snapshot(temporary_root)

        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(before, after)
        self.assertEqual(zeros.call_count, 2)
        for forbidden_call in forbidden:
            forbidden_call.assert_not_called()
        output = stdout.getvalue()
        self.assertIn("TSDF BLOCK ADDRESS CHECK", output)
        self.assertIn("artifact: valid", output)
        self.assertIn("session_replay: matched", output)
        self.assertIn("depth_decoded: no", output)
        self.assertIn("geometry_recomputed: no", output)
        self.assertIn("fusion_performed: no", output)
        self.assertIn("storage_mutated: no", output)
        self.assertIn("addressing_created_blocks: no", output)
        self.assertIn("artifact_written: no", output)
        self.assertIn("allocation: blocks=8 voxel_slots=4096", output)
        self.assertIn(
            "queries: requested=3 resolved=2 unplanned=1",
            output,
        )
        self.assertIn(
            "voxel[0]: status=planned global=(0, -8, -8) "
            "block=(0, -1, -1) local=(0, 0, 0) row=0 "
            "array=(0, 0, 0, 0) local_flat=0 storage_flat=0",
            output,
        )
        self.assertIn(
            "voxel[1]: status=planned global=(8, 0, 0) "
            "block=(1, 0, 0) local=(0, 0, 0) row=7 "
            "array=(7, 0, 0, 0) local_flat=0 storage_flat=3584",
            output,
        )
        self.assertIn(
            "voxel[2]: status=unplanned global=(-1, 0, 0)",
            output,
        )

    def test_cli_out_of_range_query_fails_actionably_and_read_only(
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
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                exit_code = main(
                    [
                        "reconstruct",
                        "tsdf-block-address",
                        str(plan_path),
                        str(session_path),
                        "--voxel",
                        str(MAX_TSDF_GLOBAL_VOXEL_INDEX + 1),
                        "0",
                        "0",
                    ]
                )
            after = tree_snapshot(temporary_root)

        self.assertEqual(exit_code, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(before, after)
        self.assertEqual(zeros.call_count, 2)
        for forbidden_call in forbidden:
            forbidden_call.assert_not_called()
        error = stderr.getvalue()
        self.assertIn("TSDF BLOCK ADDRESS FAILED", error)
        self.assertIn("global_index_xyz[0]", error)
        self.assertIn("expected an integer in", error)
        self.assertNotIn("Traceback", error)


if __name__ == "__main__":
    unittest.main()
