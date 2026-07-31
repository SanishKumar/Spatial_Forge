"""Immutable replay/depth snapshots for plan-selected TSDF observations."""

from __future__ import annotations

import math
import re
from dataclasses import InitVar, dataclass, field
from enum import StrEnum

import numpy as np

from .errors import PointCloudError, TsdfError
from .model import CameraCalibration, ScanSession
from .point_cloud import (
    _read_depth,
    _sample_path,
    _validate_reconstruction_contract,
)
from .replay import replay_session
from .tsdf_block_plan_loader import TsdfBlockPlan

MAX_TSDF_REPLAY_DEPTH_CONTEXT_BYTES = 512 * 1024 * 1024

_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


class TsdfReplayDepthStatus(StrEnum):
    """Stable preparation states for one plan-selected observation."""

    READY = "ready"
    MISSING_DEPTH = "missing-depth"
    MISSING_POSE = "missing-pose"
    MISSING_DEPTH_AND_POSE = "missing-depth-and-pose"


@dataclass(frozen=True, slots=True, eq=False)
class TsdfReplayDepthObservation:
    """One selected observation with an optional immutable metric depth."""

    observation_sequence: int
    status: TsdfReplayDepthStatus
    t_world_camera: tuple[float, ...] | None
    depth_m: InitVar[np.ndarray | None] = None
    _depth_payload: bytes | None = field(init=False, repr=False)
    _depth_shape: tuple[int, int] | None = field(init=False, repr=False)

    def __post_init__(self, depth_m: np.ndarray | None) -> None:
        if (
            isinstance(self.observation_sequence, bool)
            or not isinstance(self.observation_sequence, int)
            or self.observation_sequence < 0
        ):
            raise TsdfError(
                "TSDF replay/depth observation sequence is invalid"
            )
        if not isinstance(self.status, TsdfReplayDepthStatus):
            raise TsdfError(
                "TSDF replay/depth observation status is invalid"
            )
        if self.t_world_camera is not None and (
            not isinstance(self.t_world_camera, tuple)
            or len(self.t_world_camera) != 16
            or any(
                not _is_finite_number(component)
                for component in self.t_world_camera
            )
        ):
            raise TsdfError(
                "TSDF replay/depth observation pose must contain 16 "
                "finite values"
            )
        depth_payload: bytes | None = None
        depth_shape: tuple[int, int] | None = None
        if depth_m is not None:
            depth_payload = _validate_immutable_depth(depth_m)
            depth_shape = (int(depth_m.shape[0]), int(depth_m.shape[1]))
        object.__setattr__(self, "_depth_payload", depth_payload)
        object.__setattr__(self, "_depth_shape", depth_shape)

        if self.status is TsdfReplayDepthStatus.READY:
            if self.t_world_camera is None or depth_payload is None:
                raise TsdfError(
                    "ready TSDF replay/depth observation requires pose "
                    "and depth"
                )
        elif self.status is TsdfReplayDepthStatus.MISSING_DEPTH:
            if self.t_world_camera is None or depth_payload is not None:
                raise TsdfError(
                    "missing-depth TSDF observation requires only a pose"
                )
        elif self.t_world_camera is not None or depth_payload is not None:
            raise TsdfError(
                "pose-missing TSDF replay/depth observation cannot retain "
                "pose or depth"
            )

    @property
    def depth_decoded(self) -> bool:
        return self._depth_payload is not None

    @property
    def depth_sample_count(self) -> int:
        if self._depth_payload is None:
            return 0
        return len(self._depth_payload) // np.dtype(np.float64).itemsize

    @property
    def depth_payload_bytes(self) -> int:
        if self._depth_payload is None:
            return 0
        return len(self._depth_payload)


def _observation_depth_m(
    observation: TsdfReplayDepthObservation,
) -> np.ndarray | None:
    if observation._depth_payload is None:
        return None
    if observation._depth_shape is None:
        raise AssertionError("decoded depth must retain its canonical shape")
    return np.frombuffer(
        observation._depth_payload,
        dtype=np.float64,
    ).reshape(observation._depth_shape)


TsdfReplayDepthObservation.depth_m = property(  # type: ignore[assignment]
    _observation_depth_m
)


@dataclass(frozen=True, slots=True, eq=False)
class TsdfReplayDepthContext:
    """Frozen construction-time snapshot of selected replay/depth inputs."""

    session_id: str
    source_plan_digest_sha256: str
    replay_digest_sha256: str
    frame_stride: int
    total_observations: int
    selected_observation_sequences: tuple[int, ...]
    camera: CameraCalibration
    depth_scale_m: float
    observations: tuple[TsdfReplayDepthObservation, ...]
    valid_depth_samples: int
    invalid_depth_samples: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.session_id, str)
            or not _IDENTIFIER.fullmatch(self.session_id)
        ):
            raise TsdfError("TSDF replay/depth context session_id is invalid")
        if not _is_sha256(self.source_plan_digest_sha256):
            raise TsdfError(
                "TSDF replay/depth context source plan digest is invalid"
            )
        if not _is_sha256(self.replay_digest_sha256):
            raise TsdfError(
                "TSDF replay/depth context replay digest is invalid"
            )
        for value, label in (
            (self.frame_stride, "frame stride"),
            (self.total_observations, "total observations"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise TsdfError(
                    f"TSDF replay/depth context {label} must be positive"
                )
        expected_sequences = tuple(
            range(0, self.total_observations, self.frame_stride)
        )
        if (
            not isinstance(self.selected_observation_sequences, tuple)
            or any(
                isinstance(sequence, bool)
                or not isinstance(sequence, int)
                or sequence < 0
                for sequence in self.selected_observation_sequences
            )
            or self.selected_observation_sequences != expected_sequences
        ):
            raise TsdfError(
                "TSDF replay/depth context selected observations are not "
                "the complete canonical stride order"
            )
        _validate_context_camera(self.camera)
        if not _is_finite_number(self.depth_scale_m) or self.depth_scale_m <= 0:
            raise TsdfError(
                "TSDF replay/depth context depth scale must be positive"
            )
        if (
            not isinstance(self.observations, tuple)
            or len(self.observations) != len(expected_sequences)
        ):
            raise TsdfError(
                "TSDF replay/depth context observations do not match "
                "selection"
            )
        for sequence, observation in zip(
            expected_sequences,
            self.observations,
            strict=True,
        ):
            if (
                not isinstance(observation, TsdfReplayDepthObservation)
                or observation.observation_sequence != sequence
            ):
                raise TsdfError(
                    "TSDF replay/depth observation order is inconsistent"
                )
            if observation.depth_m is not None and observation.depth_m.shape != (
                self.camera.height,
                self.camera.width,
            ):
                raise TsdfError(
                    "TSDF replay/depth frame shape does not match camera"
                )

        for value, label in (
            (self.valid_depth_samples, "valid depth samples"),
            (self.invalid_depth_samples, "invalid depth samples"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise TsdfError(
                    f"TSDF replay/depth context {label} is invalid"
                )
        actual_valid, actual_invalid = _count_context_depth(self.observations)
        if (
            self.valid_depth_samples != actual_valid
            or self.invalid_depth_samples != actual_invalid
        ):
            raise TsdfError(
                "TSDF replay/depth context depth counters are inconsistent"
            )
        if self.depth_payload_bytes > MAX_TSDF_REPLAY_DEPTH_CONTEXT_BYTES:
            raise TsdfError(
                "TSDF replay/depth context exceeds retained depth limit"
            )

    @property
    def selected_observation_count(self) -> int:
        return len(self.observations)

    @property
    def ready_observation_count(self) -> int:
        return sum(
            observation.status is TsdfReplayDepthStatus.READY
            for observation in self.observations
        )

    @property
    def depth_frames_decoded(self) -> int:
        return sum(
            observation.depth_decoded
            for observation in self.observations
        )

    @property
    def depth_sample_count(self) -> int:
        return sum(
            observation.depth_sample_count
            for observation in self.observations
        )

    @property
    def depth_payload_bytes(self) -> int:
        return sum(
            observation.depth_payload_bytes
            for observation in self.observations
        )

    @property
    def status_counts(
        self,
    ) -> tuple[tuple[TsdfReplayDepthStatus, int], ...]:
        return tuple(
            (status, count)
            for status in TsdfReplayDepthStatus
            if (
                count := sum(
                    observation.status is status
                    for observation in self.observations
                )
            )
        )


def build_tsdf_replay_depth_context(
    plan: TsdfBlockPlan,
    session: ScanSession,
) -> TsdfReplayDepthContext:
    """Build one bounded immutable selected-observation depth snapshot."""

    if not isinstance(plan, TsdfBlockPlan):
        raise TsdfError(
            "TSDF replay/depth context requires a loaded TsdfBlockPlan"
        )
    if not isinstance(session, ScanSession):
        raise TsdfError(
            "TSDF replay/depth context requires a loaded ScanSession"
        )
    _validate_context_plan(plan)
    if plan.session_id != session.session_id:
        raise TsdfError(
            "TSDF block plan session_id does not match the loaded session"
        )
    try:
        camera, depth_scale_m = _validate_reconstruction_contract(session)
    except PointCloudError as error:
        raise TsdfError(str(error)) from error

    starting_replay = replay_session(session)
    if starting_replay.digest_sha256 != plan.replay_digest_sha256:
        raise TsdfError(
            "TSDF block plan replay digest does not match current session "
            "inputs; regenerate the plan"
        )
    if len(starting_replay.observations) != plan.total_observations:
        raise TsdfError(
            "TSDF block plan total observations do not match current replay"
        )
    selected_sequences = tuple(
        range(0, plan.total_observations, plan.frame_stride)
    )
    if len(selected_sequences) != plan.selected_observations:
        raise TsdfError(
            "TSDF block plan selected observations do not match its stride"
        )
    selected = tuple(
        starting_replay.observations[sequence]
        for sequence in selected_sequences
    )
    _validate_plan_associations(plan, selected, camera)

    retained_depth_bytes = (
        plan.paired_observations
        * camera.width
        * camera.height
        * np.dtype(np.float64).itemsize
    )
    if retained_depth_bytes > MAX_TSDF_REPLAY_DEPTH_CONTEXT_BYTES:
        raise TsdfError(
            "TSDF replay/depth context requires "
            f"{retained_depth_bytes} retained depth bytes; maximum is "
            f"{MAX_TSDF_REPLAY_DEPTH_CONTEXT_BYTES}"
        )

    prepared: list[TsdfReplayDepthObservation] = []
    valid_depth_samples = 0
    invalid_depth_samples = 0
    try:
        for observation in selected:
            status = _classify_observation(observation)
            transform = _prepared_transform(observation, status)
            depth_m: np.ndarray | None = None
            if status is TsdfReplayDepthStatus.READY:
                if observation.depth is None:
                    raise AssertionError("ready observation must have depth")
                depth_path = _sample_path(
                    session,
                    observation.depth.data,
                    "depth",
                )
                raw_depth = np.asarray(
                    _read_depth(
                        depth_path,
                        camera.width,
                        camera.height,
                    ),
                    dtype=np.float64,
                ).reshape((camera.height, camera.width))
                with np.errstate(over="ignore", invalid="ignore"):
                    np.multiply(
                        raw_depth,
                        depth_scale_m,
                        out=raw_depth,
                    )
                payload = raw_depth.tobytes(order="C")
                depth_m = np.frombuffer(
                    payload,
                    dtype=np.float64,
                ).reshape((camera.height, camera.width))
                frame_valid = int(
                    np.count_nonzero(
                        np.isfinite(depth_m) & (depth_m > 0.0)
                    )
                )
                frame_invalid = int(depth_m.size) - frame_valid
                valid_depth_samples += frame_valid
                invalid_depth_samples += frame_invalid
            prepared.append(
                TsdfReplayDepthObservation(
                    observation_sequence=observation.sequence,
                    status=status,
                    t_world_camera=transform,
                    depth_m=depth_m,
                )
            )
    except PointCloudError as error:
        raise TsdfError(str(error)) from error
    except (MemoryError, ValueError) as error:
        raise TsdfError(
            f"cannot build TSDF replay/depth context: {error}"
        ) from error

    if (
        valid_depth_samples != plan.valid_depth_points
        or invalid_depth_samples != plan.invalid_depth_samples
    ):
        raise TsdfError(
            "TSDF replay/depth context sample counts do not match the plan"
        )
    ending_replay = replay_session(session)
    if (
        ending_replay.digest_sha256 != starting_replay.digest_sha256
        or ending_replay.digest_sha256 != plan.replay_digest_sha256
    ):
        raise TsdfError(
            "session inputs changed while building TSDF replay/depth context"
        )

    return TsdfReplayDepthContext(
        session_id=session.session_id,
        source_plan_digest_sha256=plan.artifact_digest_sha256,
        replay_digest_sha256=plan.replay_digest_sha256,
        frame_stride=plan.frame_stride,
        total_observations=plan.total_observations,
        selected_observation_sequences=selected_sequences,
        camera=camera,
        depth_scale_m=depth_scale_m,
        observations=tuple(prepared),
        valid_depth_samples=valid_depth_samples,
        invalid_depth_samples=invalid_depth_samples,
    )


def _validate_immutable_depth(depth_m: np.ndarray) -> bytes:
    if type(depth_m) is not np.ndarray:
        raise TsdfError(
            "TSDF replay/depth frame must be a base NumPy array"
        )
    if depth_m.ndim != 2 or depth_m.dtype != np.dtype(np.float64):
        raise TsdfError(
            "TSDF replay/depth frame must be a 2D float64 array"
        )
    if (
        not depth_m.flags.c_contiguous
        or depth_m.flags.owndata
        or depth_m.flags.writeable
        or (payload := _immutable_bytes_base(depth_m)) is None
        or len(payload) != depth_m.nbytes
    ):
        raise TsdfError(
            "TSDF replay/depth frame must be immutable C-contiguous "
            "bytes-backed storage"
        )
    return payload


def _validate_context_camera(camera: object) -> None:
    if not isinstance(camera, CameraCalibration):
        raise TsdfError(
            "TSDF replay/depth context requires camera calibration"
        )
    if not isinstance(camera.id, str) or not _IDENTIFIER.fullmatch(camera.id):
        raise TsdfError(
            "TSDF replay/depth context camera id is invalid"
        )
    if camera.model != "pinhole":
        raise TsdfError(
            "TSDF replay/depth context requires pinhole calibration"
        )
    if (
        isinstance(camera.width, bool)
        or not isinstance(camera.width, int)
        or camera.width < 1
        or isinstance(camera.height, bool)
        or not isinstance(camera.height, int)
        or camera.height < 1
    ):
        raise TsdfError(
            "TSDF replay/depth context camera dimensions are invalid"
        )
    if (
        not _is_finite_number(camera.fx)
        or camera.fx <= 0
        or not _is_finite_number(camera.fy)
        or camera.fy <= 0
        or not _is_finite_number(camera.cx)
        or not 0 <= camera.cx < camera.width
        or not _is_finite_number(camera.cy)
        or not 0 <= camera.cy < camera.height
    ):
        raise TsdfError(
            "TSDF replay/depth context camera intrinsics are invalid"
        )
    if (
        camera.distortion_model != "none"
        or not isinstance(camera.distortion_coefficients, tuple)
        or camera.distortion_coefficients
    ):
        raise TsdfError(
            "TSDF replay/depth context requires undistorted calibration"
        )
    if (
        not isinstance(camera.t_rig_camera, tuple)
        or len(camera.t_rig_camera) != 16
        or any(
            not _is_finite_number(component)
            for component in camera.t_rig_camera
        )
        or not _is_rigid_transform(camera.t_rig_camera)
    ):
        raise TsdfError(
            "TSDF replay/depth context camera transform must be rigid"
        )


def _is_rigid_transform(transform: tuple[float, ...]) -> bool:
    tolerance = 1e-5
    if any(
        abs(actual - expected) > tolerance
        for actual, expected in zip(
            transform[12:16],
            (0.0, 0.0, 0.0, 1.0),
            strict=True,
        )
    ):
        return False

    rotation = (
        (transform[0], transform[1], transform[2]),
        (transform[4], transform[5], transform[6]),
        (transform[8], transform[9], transform[10]),
    )
    for row_index in range(3):
        for column_index in range(3):
            dot = sum(
                rotation[row_index][axis]
                * rotation[column_index][axis]
                for axis in range(3)
            )
            expected = 1.0 if row_index == column_index else 0.0
            if abs(dot - expected) > tolerance:
                return False

    determinant = (
        rotation[0][0]
        * (
            rotation[1][1] * rotation[2][2]
            - rotation[1][2] * rotation[2][1]
        )
        - rotation[0][1]
        * (
            rotation[1][0] * rotation[2][2]
            - rotation[1][2] * rotation[2][0]
        )
        + rotation[0][2]
        * (
            rotation[1][0] * rotation[2][1]
            - rotation[1][1] * rotation[2][0]
        )
    )
    return abs(determinant - 1.0) <= tolerance


def _immutable_bytes_base(array: np.ndarray) -> bytes | None:
    base: object = array
    while isinstance(base, np.ndarray):
        base = base.base
    if isinstance(base, bytes):
        return base
    return None


def _count_context_depth(
    observations: tuple[TsdfReplayDepthObservation, ...],
) -> tuple[int, int]:
    valid = 0
    invalid = 0
    for observation in observations:
        if observation.depth_m is None:
            continue
        frame_valid = int(
            np.count_nonzero(
                np.isfinite(observation.depth_m)
                & (observation.depth_m > 0.0)
            )
        )
        valid += frame_valid
        invalid += int(observation.depth_m.size) - frame_valid
    return valid, invalid


def _validate_context_plan(plan: TsdfBlockPlan) -> None:
    if not _is_sha256(plan.artifact_digest_sha256):
        raise TsdfError("TSDF block plan artifact digest is invalid")
    if not _is_sha256(plan.replay_digest_sha256):
        raise TsdfError("TSDF block plan replay digest is invalid")
    for value, label in (
        (plan.frame_stride, "frame_stride"),
        (plan.total_observations, "total_observations"),
        (plan.selected_observations, "selected_observations"),
        (plan.paired_observations, "paired_observations"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise TsdfError(
                f"TSDF block plan {label} must be a positive integer"
            )
    for value, label in (
        (plan.skipped_missing_depth, "skipped_missing_depth"),
        (plan.skipped_missing_pose, "skipped_missing_pose"),
        (plan.valid_depth_points, "valid_depth_points"),
        (plan.invalid_depth_samples, "invalid_depth_samples"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise TsdfError(
                f"TSDF block plan {label} must be a nonnegative integer"
            )


def _validate_plan_associations(
    plan: TsdfBlockPlan,
    selected: tuple[object, ...],
    camera: CameraCalibration,
) -> None:
    paired = 0
    missing_depth = 0
    missing_pose = 0
    for observation in selected:
        depth_missing = getattr(observation, "depth", None) is None
        pose_missing = getattr(observation, "pose", None) is None
        missing_depth += int(depth_missing)
        missing_pose += int(pose_missing)
        paired += int(not depth_missing and not pose_missing)
    if (
        paired != plan.paired_observations
        or missing_depth != plan.skipped_missing_depth
        or missing_pose != plan.skipped_missing_pose
    ):
        raise TsdfError(
            "TSDF block plan observation associations do not match replay"
        )
    expected_samples = paired * camera.width * camera.height
    if (
        plan.valid_depth_points + plan.invalid_depth_samples
        != expected_samples
    ):
        raise TsdfError(
            "TSDF block plan depth sample counts do not match camera/replay"
        )


def _classify_observation(observation: object) -> TsdfReplayDepthStatus:
    depth_missing = getattr(observation, "depth", None) is None
    pose_missing = getattr(observation, "pose", None) is None
    if depth_missing and pose_missing:
        return TsdfReplayDepthStatus.MISSING_DEPTH_AND_POSE
    if depth_missing:
        return TsdfReplayDepthStatus.MISSING_DEPTH
    if pose_missing:
        return TsdfReplayDepthStatus.MISSING_POSE
    return TsdfReplayDepthStatus.READY


def _prepared_transform(
    observation: object,
    status: TsdfReplayDepthStatus,
) -> tuple[float, ...] | None:
    if status in (
        TsdfReplayDepthStatus.MISSING_POSE,
        TsdfReplayDepthStatus.MISSING_DEPTH_AND_POSE,
    ):
        return None
    pose = getattr(observation, "pose", None)
    if pose is None:
        raise AssertionError("prepared observation must have a pose")
    transform = tuple(float(value) for value in pose.data["T_world_camera"])
    if len(transform) != 16 or any(
        not math.isfinite(value) for value in transform
    ):
        raise TsdfError(
            "T_world_camera must contain 16 finite numeric components"
        )
    return transform
