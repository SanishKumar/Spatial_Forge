"""Deterministic signed voxel addressing for planned TSDF block storage."""

from __future__ import annotations

from dataclasses import dataclass

from .errors import TsdfError
from .tsdf_block_plan import (
    MAX_BLOCK_INDEX,
    MIN_BLOCK_INDEX,
    TSDF_BLOCK_RESOLUTION,
)
from .tsdf_block_storage import (
    TSDF_BLOCK_STORAGE_BYTES_PER_VOXEL,
    TSDF_BLOCK_VOXELS,
    TsdfBlockStorage,
    _validate_storage_arrays,
)

MIN_TSDF_GLOBAL_VOXEL_INDEX = MIN_BLOCK_INDEX * TSDF_BLOCK_RESOLUTION
MAX_TSDF_GLOBAL_VOXEL_INDEX = (
    MAX_BLOCK_INDEX * TSDF_BLOCK_RESOLUTION
    + TSDF_BLOCK_RESOLUTION
    - 1
)

_Index3 = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class TsdfVoxelAddress:
    """A planned global voxel's immutable block and array address."""

    global_index_xyz: _Index3
    block_index_xyz: _Index3
    local_index_xyz: _Index3
    block_row: int
    local_flat_index: int
    storage_flat_index: int

    @property
    def array_index_bzyx(self) -> tuple[int, int, int, int]:
        local_x, local_y, local_z = self.local_index_xyz
        return (self.block_row, local_z, local_y, local_x)


def locate_tsdf_voxel(
    storage: TsdfBlockStorage,
    global_index_xyz: _Index3,
) -> TsdfVoxelAddress | None:
    """Resolve a valid global voxel index, or return None for a sparse miss."""

    if not isinstance(storage, TsdfBlockStorage):
        raise TsdfError(
            "TSDF voxel addressing requires allocated TsdfBlockStorage"
        )
    _validate_addressable_storage(storage)
    global_index, block_index, local_index = (
        _split_tsdf_global_voxel_index(global_index_xyz)
    )
    block_row = _find_block_row(storage.block_indices, block_index)
    if block_row is None:
        return None

    local_x, local_y, local_z = local_index
    local_flat_index = (
        local_x
        + TSDF_BLOCK_RESOLUTION
        * (
            local_y
            + TSDF_BLOCK_RESOLUTION * local_z
        )
    )
    storage_flat_index = (
        block_row * TSDF_BLOCK_VOXELS + local_flat_index
    )
    if storage_flat_index < 0 or storage_flat_index >= storage.voxel_slots:
        raise TsdfError("resolved TSDF voxel address is outside storage")
    return TsdfVoxelAddress(
        global_index_xyz=global_index,
        block_index_xyz=block_index,
        local_index_xyz=local_index,
        block_row=block_row,
        local_flat_index=local_flat_index,
        storage_flat_index=storage_flat_index,
    )


def compose_tsdf_global_voxel_index(
    block_index_xyz: _Index3,
    local_index_xyz: _Index3,
) -> _Index3:
    """Compose a signed block index and local voxel index into global XYZ."""

    block_index = _require_integer_triplet(
        block_index_xyz,
        "block_index_xyz",
        minimum=MIN_BLOCK_INDEX,
        maximum=MAX_BLOCK_INDEX,
    )
    local_index = _require_integer_triplet(
        local_index_xyz,
        "local_index_xyz",
        minimum=0,
        maximum=TSDF_BLOCK_RESOLUTION - 1,
    )
    return tuple(
        block_index[axis] * TSDF_BLOCK_RESOLUTION + local_index[axis]
        for axis in range(3)
    )  # type: ignore[return-value]


def _split_tsdf_global_voxel_index(
    global_index_xyz: _Index3,
) -> tuple[_Index3, _Index3, _Index3]:
    global_index = _require_integer_triplet(
        global_index_xyz,
        "global_index_xyz",
        minimum=MIN_TSDF_GLOBAL_VOXEL_INDEX,
        maximum=MAX_TSDF_GLOBAL_VOXEL_INDEX,
    )
    block_components: list[int] = []
    local_components: list[int] = []
    for component in global_index:
        block_component, local_component = divmod(
            component,
            TSDF_BLOCK_RESOLUTION,
        )
        block_components.append(block_component)
        local_components.append(local_component)
    block_index = (
        block_components[0],
        block_components[1],
        block_components[2],
    )
    local_index = (
        local_components[0],
        local_components[1],
        local_components[2],
    )
    return global_index, block_index, local_index


def _find_block_row(
    block_indices: tuple[_Index3, ...],
    target: _Index3,
) -> int | None:
    target_key = (target[2], target[1], target[0])
    lower = 0
    upper = len(block_indices)
    while lower < upper:
        middle = (lower + upper) // 2
        candidate = block_indices[middle]
        candidate_key = (
            candidate[2],
            candidate[1],
            candidate[0],
        )
        if candidate_key < target_key:
            lower = middle + 1
        else:
            upper = middle
    if lower < len(block_indices) and block_indices[lower] == target:
        return lower
    return None


def _validate_addressable_storage(storage: TsdfBlockStorage) -> None:
    expected_shape = (
        storage.block_count,
        TSDF_BLOCK_RESOLUTION,
        TSDF_BLOCK_RESOLUTION,
        TSDF_BLOCK_RESOLUTION,
    )
    expected_payload_bytes = (
        storage.block_count
        * TSDF_BLOCK_VOXELS
        * TSDF_BLOCK_STORAGE_BYTES_PER_VOXEL
    )
    _validate_storage_arrays(
        storage.tsdf_sums,
        storage.weights,
        expected_shape,
        expected_payload_bytes,
    )


def _require_integer_triplet(
    value: object,
    label: str,
    *,
    minimum: int,
    maximum: int,
) -> _Index3:
    if not isinstance(value, tuple) or len(value) != 3:
        raise TsdfError(f"{label}: expected a tuple of 3 integers")
    for axis, component in enumerate(value):
        if isinstance(component, bool) or not isinstance(component, int):
            raise TsdfError(f"{label}[{axis}]: expected an integer")
        if component < minimum or component > maximum:
            raise TsdfError(
                f"{label}[{axis}]: expected an integer in "
                f"[{minimum}, {maximum}]"
            )
    return value
