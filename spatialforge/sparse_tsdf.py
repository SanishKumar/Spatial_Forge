"""Deterministic sparse-accumulator TSDF reference backend."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from .model import ScanSession
from .tsdf import TsdfReport, _integrate_tsdf


class _SparseTsdfAccumulator:
    """Store sums and weights only for voxel indices that receive updates."""

    __slots__ = ("total_voxels", "_voxels")

    def __init__(self, total_voxels: int) -> None:
        self.total_voxels = total_voxels
        self._voxels: dict[int, tuple[float, int]] = {}

    @property
    def stored_voxels(self) -> int:
        return len(self._voxels)

    def add(
        self,
        flat_indices: np.ndarray,
        tsdf_values: np.ndarray,
    ) -> None:
        if flat_indices.shape != tsdf_values.shape:
            raise AssertionError("TSDF index and value arrays must have equal shape")

        for raw_index, raw_value in zip(
            flat_indices.tolist(),
            tsdf_values.tolist(),
            strict=True,
        ):
            flat_index = int(raw_index)
            if flat_index < 0 or flat_index >= self.total_voxels:
                raise AssertionError("TSDF update index is outside the volume")
            previous_sum, previous_weight = self._voxels.get(
                flat_index,
                (0.0, 0),
            )
            self._voxels[flat_index] = (
                previous_sum + float(raw_value),
                previous_weight + 1,
            )

    def observed_arrays(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ordered_indices = sorted(self._voxels)
        observed_indices = np.asarray(ordered_indices, dtype=np.int64)
        observed_sums = np.asarray(
            [self._voxels[index][0] for index in ordered_indices],
            dtype=np.float64,
        )
        observed_weights = np.asarray(
            [self._voxels[index][1] for index in ordered_indices],
            dtype=np.uint32,
        )
        return observed_indices, observed_sums, observed_weights


def integrate_sparse_tsdf(
    session: ScanSession,
    output: str | Path,
    *,
    origin_world_m: Sequence[float],
    dimensions: Sequence[int],
    voxel_size_m: float,
    truncation_m: float,
    frame_stride: int = 1,
    expected_replay_digest_sha256: str | None = None,
) -> TsdfReport:
    """Integrate with sparse in-memory sums/weights and dense traversal."""

    return _integrate_tsdf(
        session,
        output,
        origin_world_m=origin_world_m,
        dimensions=dimensions,
        voxel_size_m=voxel_size_m,
        truncation_m=truncation_m,
        frame_stride=frame_stride,
        expected_replay_digest_sha256=expected_replay_digest_sha256,
        accumulator_factory=_SparseTsdfAccumulator,
    )
