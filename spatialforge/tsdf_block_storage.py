"""Bounded empty TSDF block storage backed by verified candidate plans."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .errors import TsdfError
from .model import ScanSession
from .tsdf_block_plan import (
    MAX_BLOCK_INDEX,
    MAX_PLANNED_BLOCKS,
    MIN_BLOCK_INDEX,
    TSDF_BLOCK_RESOLUTION,
)
from .tsdf_block_plan_loader import (
    TsdfBlockPlan,
    verify_tsdf_block_plan_replay,
)

TSDF_BLOCK_VOXELS = TSDF_BLOCK_RESOLUTION**3
TSDF_SUM_DTYPE = np.dtype(np.float64)
TSDF_WEIGHT_DTYPE = np.dtype(np.uint32)
TSDF_BLOCK_STORAGE_BYTES_PER_VOXEL = (
    TSDF_SUM_DTYPE.itemsize + TSDF_WEIGHT_DTYPE.itemsize
)
# Storage holds any plan the planner will write, and nothing larger.
# The two limits used to be set separately, and a plan could be
# accepted by one stage only to be refused by the next.
MAX_TSDF_BLOCK_STORAGE_BLOCKS = MAX_PLANNED_BLOCKS
MAX_TSDF_BLOCK_STORAGE_BYTES = (
    MAX_TSDF_BLOCK_STORAGE_BLOCKS
    * TSDF_BLOCK_VOXELS
    * TSDF_BLOCK_STORAGE_BYTES_PER_VOXEL
)

_BlockIndex = tuple[int, int, int]


@dataclass(frozen=True, slots=True, eq=False)
class TsdfBlockStorage:
    """Canonical block topology with mutable empty accumulator buffers."""

    source_plan: TsdfBlockPlan
    block_indices: tuple[_BlockIndex, ...]
    tsdf_sums: np.ndarray
    weights: np.ndarray

    def __post_init__(self) -> None:
        block_indices, shape, payload_bytes = _preflight_storage_plan(
            self.source_plan
        )
        if self.block_indices != block_indices:
            raise TsdfError(
                "TSDF block storage rows must match the source plan exactly"
            )
        _validate_storage_arrays(
            self.tsdf_sums,
            self.weights,
            shape,
            payload_bytes,
        )

    @property
    def block_count(self) -> int:
        return len(self.block_indices)

    @property
    def block_resolution(self) -> int:
        return self.source_plan.block_resolution

    @property
    def voxel_slots(self) -> int:
        return int(self.tsdf_sums.size)

    @property
    def tsdf_sum_bytes(self) -> int:
        return int(self.tsdf_sums.nbytes)

    @property
    def weight_bytes(self) -> int:
        return int(self.weights.nbytes)

    @property
    def payload_bytes(self) -> int:
        return self.tsdf_sum_bytes + self.weight_bytes

    @property
    def nonzero_sum_count(self) -> int:
        return int(np.count_nonzero(self.tsdf_sums))

    @property
    def nonzero_weight_count(self) -> int:
        return int(np.count_nonzero(self.weights))

    @property
    def unknown_voxel_count(self) -> int:
        return self.voxel_slots - self.nonzero_weight_count


def allocate_empty_tsdf_blocks(
    plan: TsdfBlockPlan,
    session: ScanSession,
) -> TsdfBlockStorage:
    """Replay-verify a plan and allocate zeroed block accumulator buffers."""

    block_indices, shape, payload_bytes = _preflight_storage_plan(plan)
    verify_tsdf_block_plan_replay(plan, session)

    try:
        tsdf_sums = np.zeros(shape, dtype=TSDF_SUM_DTYPE, order="C")
        weights = np.zeros(shape, dtype=TSDF_WEIGHT_DTYPE, order="C")
    except (MemoryError, ValueError) as error:
        raise TsdfError(
            "cannot allocate empty TSDF block storage within the "
            f"{MAX_TSDF_BLOCK_STORAGE_BYTES}-byte reference limit"
        ) from error

    return TsdfBlockStorage(
        source_plan=plan,
        block_indices=block_indices,
        tsdf_sums=tsdf_sums,
        weights=weights,
    )


def _preflight_storage_plan(
    plan: TsdfBlockPlan,
) -> tuple[tuple[_BlockIndex, ...], tuple[int, int, int, int], int]:
    if not isinstance(plan, TsdfBlockPlan):
        raise TsdfError(
            "empty TSDF block storage requires a loaded TsdfBlockPlan"
        )
    if plan.block_resolution != TSDF_BLOCK_RESOLUTION:
        raise TsdfError(
            "TSDF block storage requires block resolution "
            f"{TSDF_BLOCK_RESOLUTION}"
        )
    block_indices = plan.active_blocks
    if not isinstance(block_indices, tuple) or not block_indices:
        raise TsdfError(
            "TSDF block storage requires at least one active block"
        )
    if len(block_indices) > MAX_TSDF_BLOCK_STORAGE_BLOCKS:
        required = (
            len(block_indices)
            * TSDF_BLOCK_VOXELS
            * TSDF_BLOCK_STORAGE_BYTES_PER_VOXEL
        )
        raise TsdfError(
            "empty TSDF block storage requires "
            f"{required} numeric payload bytes for "
            f"{len(block_indices)} blocks; the maximum is "
            f"{MAX_TSDF_BLOCK_STORAGE_BYTES} bytes "
            f"({MAX_TSDF_BLOCK_STORAGE_BLOCKS} blocks), the most a plan "
            "may hold"
        )

    previous_key: tuple[int, int, int] | None = None
    for position, block in enumerate(block_indices):
        if not isinstance(block, tuple) or len(block) != 3:
            raise TsdfError(
                f"active block {position}: expected an integer triplet"
            )
        for axis, component in enumerate(block):
            if isinstance(component, bool) or not isinstance(component, int):
                raise TsdfError(
                    f"active block {position}[{axis}]: expected an integer"
                )
            if component < MIN_BLOCK_INDEX or component > MAX_BLOCK_INDEX:
                raise TsdfError(
                    f"active block {position}[{axis}]: outside signed "
                    "32-bit block range"
                )
        key = (block[2], block[1], block[0])
        if previous_key is not None and key <= previous_key:
            raise TsdfError(
                "TSDF block storage requires unique, strictly "
                "x-fastest active blocks"
            )
        previous_key = key

    voxel_slots = len(block_indices) * TSDF_BLOCK_VOXELS
    if plan.planned_voxel_slots != voxel_slots:
        raise TsdfError(
            "TSDF block storage voxel slots do not match the active blocks: "
            f"{plan.planned_voxel_slots} != {voxel_slots}"
        )
    # Within the block limit checked above, so within the byte limit.
    payload_bytes = voxel_slots * TSDF_BLOCK_STORAGE_BYTES_PER_VOXEL
    shape = (
        len(block_indices),
        TSDF_BLOCK_RESOLUTION,
        TSDF_BLOCK_RESOLUTION,
        TSDF_BLOCK_RESOLUTION,
    )
    return block_indices, shape, payload_bytes


def _validate_storage_arrays(
    tsdf_sums: np.ndarray,
    weights: np.ndarray,
    expected_shape: tuple[int, int, int, int],
    expected_payload_bytes: int,
) -> None:
    if not isinstance(tsdf_sums, np.ndarray):
        raise TsdfError("TSDF block sum storage must be a NumPy array")
    if not isinstance(weights, np.ndarray):
        raise TsdfError("TSDF block weight storage must be a NumPy array")
    if tsdf_sums.shape != expected_shape or weights.shape != expected_shape:
        raise TsdfError(
            "TSDF block storage arrays do not match the canonical "
            f"{expected_shape} shape"
        )
    if tsdf_sums.dtype != TSDF_SUM_DTYPE:
        raise TsdfError("TSDF block sums must use float64 storage")
    if weights.dtype != TSDF_WEIGHT_DTYPE:
        raise TsdfError("TSDF block weights must use uint32 storage")
    if not tsdf_sums.flags.c_contiguous or not weights.flags.c_contiguous:
        raise TsdfError("TSDF block storage arrays must be C-contiguous")
    if int(tsdf_sums.nbytes + weights.nbytes) != expected_payload_bytes:
        raise TsdfError(
            "TSDF block storage numeric payload size is inconsistent"
        )
