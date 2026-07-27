"""Known-pose RGB-D back-projection and deterministic colored PLY export."""

from __future__ import annotations

import hashlib
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Sequence, TextIO

from PIL import Image, UnidentifiedImageError

from .errors import PointCloudError
from .model import CameraCalibration, Observation, ScanSession
from .replay import replay_session


@dataclass(frozen=True, slots=True)
class PointCloudReport:
    session_id: str
    output: Path
    total_observations: int
    selected_observations: int
    integrated_frames: int
    skipped_missing_depth: int
    skipped_missing_pose: int
    invalid_depth_samples: int
    points_written: int
    replay_digest_sha256: str
    output_digest_sha256: str


def reconstruct_point_cloud(
    session: ScanSession,
    output: str | Path,
    *,
    frame_stride: int = 1,
    pixel_stride: int = 1,
) -> PointCloudReport:
    """Back-project aligned RGB-D observations using known camera poses."""

    _validate_stride(frame_stride, "frame_stride")
    _validate_stride(pixel_stride, "pixel_stride")
    output_path = Path(output).resolve()
    if output_path.suffix.lower() != ".ply":
        raise PointCloudError("output filename must end in .ply")
    if output_path.exists():
        raise PointCloudError(f"output already exists: {output_path}")

    camera, depth_scale_m = _validate_reconstruction_contract(session)
    replay = replay_session(session)
    selected = tuple(
        observation
        for observation in replay.observations
        if observation.sequence % frame_stride == 0
    )

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        body_handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="ascii",
            newline="\n",
            prefix=f".{output_path.stem}-",
            suffix=".vertices",
            dir=output_path.parent,
            delete=False,
        )
        body_path = Path(body_handle.name)
    except OSError as error:
        raise PointCloudError(
            f"cannot create point-cloud staging file: {error}"
        ) from error

    integrated_frames = 0
    skipped_missing_depth = 0
    skipped_missing_pose = 0
    invalid_depth_samples = 0
    points_written = 0
    output_digest_sha256 = ""
    final_temporary_path: Path | None = None

    try:
        with body_handle:
            for observation in selected:
                missing_depth = observation.depth is None
                missing_pose = observation.pose is None
                if missing_depth:
                    skipped_missing_depth += 1
                if missing_pose:
                    skipped_missing_pose += 1
                if missing_depth or missing_pose:
                    continue

                frame_points, frame_invalid = _write_observation_vertices(
                    session,
                    observation,
                    camera,
                    depth_scale_m,
                    pixel_stride,
                    body_handle,
                )
                integrated_frames += 1
                invalid_depth_samples += frame_invalid
                points_written += frame_points

        if integrated_frames == 0:
            raise PointCloudError(
                "no selected RGB observation has both exact depth and pose"
            )
        if points_written == 0:
            raise PointCloudError(
                "selected frames contain no positive depth samples"
            )

        final_temporary_path = _assemble_ply(
            output_path,
            body_path,
            session.session_id,
            replay.digest_sha256,
            frame_stride,
            pixel_stride,
            integrated_frames,
            points_written,
        )
        output_digest_sha256 = _sha256_file(final_temporary_path)
        _publish_without_overwrite(final_temporary_path, output_path)
        _remove_staging_file(final_temporary_path)
        final_temporary_path = None
    except PointCloudError:
        raise
    except OSError as error:
        raise PointCloudError(f"cannot write point cloud: {error}") from error
    finally:
        _remove_staging_file(body_path)
        if final_temporary_path is not None:
            _remove_staging_file(final_temporary_path)

    return PointCloudReport(
        session_id=session.session_id,
        output=output_path,
        total_observations=len(replay.observations),
        selected_observations=len(selected),
        integrated_frames=integrated_frames,
        skipped_missing_depth=skipped_missing_depth,
        skipped_missing_pose=skipped_missing_pose,
        invalid_depth_samples=invalid_depth_samples,
        points_written=points_written,
        replay_digest_sha256=replay.digest_sha256,
        output_digest_sha256=output_digest_sha256,
    )


def _validate_reconstruction_contract(
    session: ScanSession,
) -> tuple[CameraCalibration, float]:
    rgb_definition = session.stream_definitions.get("rgb")
    depth_definition = session.stream_definitions.get("depth")
    if rgb_definition is None:
        raise PointCloudError("session has no RGB stream definition")
    if depth_definition is None:
        raise PointCloudError("session has no depth stream definition")
    if depth_definition.aligned_to != "rgb":
        raise PointCloudError(
            "depth must declare aligned_to='rgb' for this reconstruction step"
        )
    if depth_definition.depth_scale_m is None:
        raise PointCloudError("depth stream has no depth_scale_m")
    if rgb_definition.calibration_id is None:
        raise PointCloudError("RGB stream has no calibration_id")

    camera = session.calibrations[rgb_definition.calibration_id]
    if camera.model != "pinhole":
        raise PointCloudError("only pinhole RGB calibration is supported")
    if camera.distortion_model != "none":
        raise PointCloudError(
            "distorted RGB calibration is unsupported until undistortion exists"
        )
    return camera, depth_definition.depth_scale_m


def _write_observation_vertices(
    session: ScanSession,
    observation: Observation,
    camera: CameraCalibration,
    depth_scale_m: float,
    pixel_stride: int,
    output: TextIO,
) -> tuple[int, int]:
    if observation.depth is None or observation.pose is None:
        raise AssertionError("caller must filter incomplete observations")

    rgb_path = _sample_path(session, observation.rgb.data, "RGB")
    depth_path = _sample_path(session, observation.depth.data, "depth")
    rgb_pixels = _read_rgb(rgb_path, camera.width, camera.height)
    depth_pixels = _read_depth(depth_path, camera.width, camera.height)
    transform = tuple(observation.pose.data["T_world_camera"])

    points_written = 0
    invalid_depth_samples = 0
    lines: list[str] = []

    for v in range(0, camera.height, pixel_stride):
        row_offset = v * camera.width
        for u in range(0, camera.width, pixel_stride):
            pixel_index = row_offset + u
            raw_depth = depth_pixels[pixel_index]
            if raw_depth <= 0:
                invalid_depth_samples += 1
                continue

            z_camera = raw_depth * depth_scale_m
            if not math.isfinite(z_camera) or z_camera <= 0:
                invalid_depth_samples += 1
                continue
            x_camera = (u - camera.cx) * z_camera / camera.fx
            y_camera = (v - camera.cy) * z_camera / camera.fy
            x_world, y_world, z_world = _transform_point(
                transform, x_camera, y_camera, z_camera
            )
            if not all(
                math.isfinite(value)
                for value in (x_world, y_world, z_world)
            ):
                raise PointCloudError(
                    "projection produced a non-finite coordinate for "
                    f"RGB sample {observation.rgb.id!r} at pixel ({u}, {v})"
                )
            red, green, blue = rgb_pixels[pixel_index]
            lines.append(
                f"{_format_coordinate(x_world)} "
                f"{_format_coordinate(y_world)} "
                f"{_format_coordinate(z_world)} "
                f"{red} {green} {blue}\n"
            )
            points_written += 1

        if len(lines) >= 8192:
            output.writelines(lines)
            lines.clear()

    if lines:
        output.writelines(lines)
    return points_written, invalid_depth_samples


def _sample_path(
    session: ScanSession,
    sample_data: Any,
    label: str,
) -> Path:
    reference = sample_data.get("path")
    if not isinstance(reference, str):
        raise PointCloudError(f"{label} sample has no path")
    path = session.root.joinpath(*PurePosixPath(reference).parts)
    if not path.is_file():
        raise PointCloudError(f"{label} file disappeared after validation")
    return path


def _read_rgb(
    path: Path,
    expected_width: int,
    expected_height: int,
) -> tuple[tuple[int, int, int], ...]:
    try:
        with Image.open(path) as image:
            image.load()
            _validate_dimensions(
                path, image.size, expected_width, expected_height
            )
            converted = image.convert("RGB")
            return tuple(converted.get_flattened_data())
    except (
        OSError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
    ) as error:
        raise PointCloudError(f"cannot decode RGB image {path.name}: {error}") from error


def _read_depth(
    path: Path,
    expected_width: int,
    expected_height: int,
) -> tuple[int, ...]:
    try:
        with Image.open(path) as image:
            image.load()
            _validate_dimensions(
                path, image.size, expected_width, expected_height
            )
            if image.mode != "I" and not image.mode.startswith("I;16"):
                raise PointCloudError(
                    f"depth image {path.name} must be 16-bit integer, "
                    f"received mode {image.mode!r}"
                )
            return tuple(
                int(value) for value in image.get_flattened_data()
            )
    except PointCloudError:
        raise
    except (
        OSError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
    ) as error:
        raise PointCloudError(
            f"cannot decode depth image {path.name}: {error}"
        ) from error


def _validate_dimensions(
    path: Path,
    actual: tuple[int, int],
    expected_width: int,
    expected_height: int,
) -> None:
    expected = (expected_width, expected_height)
    if actual != expected:
        raise PointCloudError(
            f"image {path.name} has dimensions {actual[0]}x{actual[1]}, "
            f"expected {expected_width}x{expected_height}"
        )


def _transform_point(
    transform: Sequence[float],
    x: float,
    y: float,
    z: float,
) -> tuple[float, float, float]:
    return (
        transform[0] * x
        + transform[1] * y
        + transform[2] * z
        + transform[3],
        transform[4] * x
        + transform[5] * y
        + transform[6] * z
        + transform[7],
        transform[8] * x
        + transform[9] * y
        + transform[10] * z
        + transform[11],
    )


def _assemble_ply(
    output_path: Path,
    body_path: Path,
    session_id: str,
    replay_digest: str,
    frame_stride: int,
    pixel_stride: int,
    integrated_frames: int,
    points_written: int,
) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}-",
        suffix=".ply",
        dir=output_path.parent,
    )
    temporary_path = Path(temporary_name)
    header = "\n".join(
        [
            "ply",
            "format ascii 1.0",
            f"comment spatialforge_session {session_id}",
            f"comment spatialforge_replay_sha256 {replay_digest}",
            "comment coordinates metres world_x_forward world_y_left world_z_up",
            f"comment spatialforge_frame_stride {frame_stride}",
            f"comment spatialforge_pixel_stride {pixel_stride}",
            f"comment spatialforge_frames_integrated {integrated_frames}",
            f"element vertex {points_written}",
            "property double x",
            "property double y",
            "property double z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "end_header",
            "",
        ]
    ).encode("ascii")

    try:
        with open(descriptor, "wb", closefd=True) as output_file:
            output_file.write(header)
            with body_path.open("rb") as body_file:
                shutil.copyfileobj(body_file, output_file)
    except OSError:
        _remove_staging_file(temporary_path)
        raise
    return temporary_path


def _format_coordinate(value: float) -> str:
    if abs(value) < 0.5e-9:
        value = 0.0
    return f"{value:.9f}"


def _validate_stride(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise PointCloudError(f"{label} must be a positive integer")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as input_file:
            while chunk := input_file.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise PointCloudError(f"cannot hash output point cloud: {error}") from error
    return digest.hexdigest()


def _publish_without_overwrite(staging_path: Path, output_path: Path) -> None:
    try:
        os.link(staging_path, output_path)
    except FileExistsError as error:
        raise PointCloudError(
            "output appeared while reconstructing; refusing to overwrite: "
            f"{output_path}"
        ) from error
    except OSError as error:
        raise PointCloudError(
            f"cannot publish point cloud without overwriting: {error}"
        ) from error


def _remove_staging_file(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass
