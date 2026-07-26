"""Immutable value objects for validated scan sessions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class CameraCalibration:
    id: str
    model: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion_model: str
    distortion_coefficients: tuple[float, ...]
    t_rig_camera: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class StreamDefinition:
    name: str
    kind: str
    index: str
    calibration_id: str | None = None
    depth_scale_m: float | None = None
    aligned_to: str | None = None
    transform: str | None = None


@dataclass(frozen=True, slots=True)
class StreamSample:
    stream: str
    id: str
    timestamp_ns: int
    data: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ScanSession:
    root: Path
    session_id: str
    schema_version: str
    created_at_utc: str
    calibrations: Mapping[str, CameraCalibration]
    stream_definitions: Mapping[str, StreamDefinition]
    streams: Mapping[str, tuple[StreamSample, ...]]

    @property
    def duration_ns(self) -> int:
        timestamps = [
            sample.timestamp_ns
            for samples in self.streams.values()
            for sample in samples
        ]
        if not timestamps:
            return 0
        return max(timestamps) - min(timestamps)


@dataclass(frozen=True, slots=True)
class Observation:
    sequence: int
    rgb: StreamSample
    depth: StreamSample | None
    pose: StreamSample | None
    imu: tuple[StreamSample, ...]
