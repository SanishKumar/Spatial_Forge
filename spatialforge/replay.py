"""Deterministic ScanSession replay around calibrated RGB observations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .errors import SessionReplayError
from .model import Observation, ScanSession, StreamSample


@dataclass(frozen=True, slots=True)
class ReplayResult:
    observations: tuple[Observation, ...]
    digest_sha256: str


def replay_session(session: ScanSession) -> ReplayResult:
    """Associate optional streams to RGB frames and hash the canonical result."""

    depth_by_timestamp = _by_timestamp(session.streams.get("depth", ()))
    pose_by_timestamp = _by_timestamp(session.streams.get("pose", ()))
    imu_samples = session.streams.get("imu", ())
    content_digests: dict[str, str] = {}

    observations: list[Observation] = []
    imu_index = 0
    previous_rgb_timestamp: int | None = None

    for sequence, rgb in enumerate(session.streams["rgb"]):
        imu_window: list[StreamSample] = []
        while (
            imu_index < len(imu_samples)
            and imu_samples[imu_index].timestamp_ns <= rgb.timestamp_ns
        ):
            imu = imu_samples[imu_index]
            if (
                previous_rgb_timestamp is None
                or imu.timestamp_ns > previous_rgb_timestamp
            ):
                imu_window.append(imu)
            imu_index += 1

        observations.append(
            Observation(
                sequence=sequence,
                rgb=rgb,
                depth=depth_by_timestamp.get(rgb.timestamp_ns),
                pose=pose_by_timestamp.get(rgb.timestamp_ns),
                imu=tuple(imu_window),
            )
        )
        previous_rgb_timestamp = rgb.timestamp_ns

    canonical = {
        "schema": "spatialforge.replay",
        "schema_version": session.schema_version,
        "session_id": session.session_id,
        "created_at_utc": session.created_at_utc,
        "calibrations": {
            key: asdict(session.calibrations[key])
            for key in sorted(session.calibrations)
        },
        "stream_definitions": {
            key: asdict(session.stream_definitions[key])
            for key in sorted(session.stream_definitions)
        },
        "streams": {
            key: [
                _canonical_sample(
                    sample, session.root, content_digests
                )
                for sample in session.streams[key]
            ]
            for key in sorted(session.streams)
        },
        "observations": [
            _canonical_observation(
                observation, session.root, content_digests
            )
            for observation in observations
        ],
    }
    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()

    return ReplayResult(
        observations=tuple(observations),
        digest_sha256=digest,
    )


def _by_timestamp(
    samples: tuple[StreamSample, ...],
) -> dict[int, StreamSample]:
    return {sample.timestamp_ns: sample for sample in samples}


def _canonical_observation(
    observation: Observation,
    root: Path,
    content_digests: dict[str, str],
) -> dict[str, Any]:
    return {
        "sequence": observation.sequence,
        "rgb": _canonical_sample(
            observation.rgb, root, content_digests
        ),
        "depth": _canonical_optional_sample(
            observation.depth, root, content_digests
        ),
        "pose": _canonical_optional_sample(
            observation.pose, root, content_digests
        ),
        "imu": [
            _canonical_sample(sample, root, content_digests)
            for sample in observation.imu
        ],
    }


def _canonical_optional_sample(
    sample: StreamSample | None,
    root: Path,
    content_digests: dict[str, str],
) -> dict[str, Any] | None:
    if sample is None:
        return None
    return _canonical_sample(sample, root, content_digests)


def _canonical_sample(
    sample: StreamSample,
    root: Path,
    content_digests: dict[str, str],
) -> dict[str, Any]:
    canonical = {
        "stream": sample.stream,
        "id": sample.id,
        "timestamp_ns": sample.timestamp_ns,
        "data": _thaw_json(sample.data),
    }
    if sample.stream in {"rgb", "depth"}:
        reference = str(sample.data["path"])
        if reference not in content_digests:
            content_digests[reference] = _file_digest(root, reference)
        canonical["content_sha256"] = content_digests[reference]
    return canonical


def _file_digest(root: Path, reference: str) -> str:
    path = root.joinpath(*PurePosixPath(reference).parts)
    digest = hashlib.sha256()
    try:
        with path.open("rb") as input_file:
            while chunk := input_file.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise SessionReplayError(
            f"cannot read sensor payload {reference!r}: {error}"
        ) from error
    return digest.hexdigest()


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _thaw_json(child)
            for key, child in value.items()
        }
    if isinstance(value, tuple):
        return [_thaw_json(child) for child in value]
    return value
