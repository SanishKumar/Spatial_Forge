"""Command-line entry point for the first SpatialForge milestone."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from .errors import (
    PointCloudError,
    SessionReplayError,
    SessionValidationError,
    TumImportError,
)
from .model import ScanSession
from .point_cloud import reconstruct_point_cloud
from .replay import replay_session
from .session_loader import STREAM_ORDER, load_scan_session
from .tum_importer import import_tum_dataset


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)

    if arguments.command == "reconstruct":
        return _run_point_cloud(
            arguments.path,
            arguments.output,
            arguments.frame_stride,
            arguments.pixel_stride,
        )
    return _run_scan(arguments)


def _run_scan(arguments: argparse.Namespace) -> int:
    if arguments.scan_command == "import-tum":
        return _run_tum_import(arguments.source, arguments.output)

    try:
        session = load_scan_session(arguments.path)
    except SessionValidationError as error:
        _print_validation_failure(arguments.path, error, sys.stderr)
        return 2

    if arguments.scan_command == "validate":
        _print_validation_summary(session, sys.stdout)
        return 0

    try:
        replay = replay_session(session)
    except SessionReplayError as error:
        print(f"REPLAY FAILED {session.session_id}", file=sys.stderr)
        print(f"- {error}", file=sys.stderr)
        return 3
    print(f"REPLAY {session.session_id}", file=sys.stdout)
    for observation in replay.observations:
        depth_id = observation.depth.id if observation.depth else "-"
        pose_id = observation.pose.id if observation.pose else "-"
        print(
            f"{observation.sequence:06d} "
            f"timestamp_ns={observation.rgb.timestamp_ns} "
            f"rgb={observation.rgb.id} "
            f"depth={depth_id} "
            f"pose={pose_id} "
            f"imu_samples={len(observation.imu)}",
            file=sys.stdout,
        )
    print(f"observations: {len(replay.observations)}", file=sys.stdout)
    print(f"digest_sha256: {replay.digest_sha256}", file=sys.stdout)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spatialforge",
        description="Validate, replay, and process SpatialForge sensor sessions.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    scan = commands.add_parser(
        "scan",
        help="Work with a folder-backed .vgsession.",
    )
    scan_commands = scan.add_subparsers(dest="scan_command", required=True)

    validate = scan_commands.add_parser(
        "validate",
        help="Validate and summarize a session.",
    )
    validate.add_argument("path", type=Path)

    replay = scan_commands.add_parser(
        "replay",
        help="Replay RGB observations deterministically.",
    )
    replay.add_argument("path", type=Path)

    import_tum = scan_commands.add_parser(
        "import-tum",
        help="Convert an extracted TUM RGB-D folder into a .vgsession.",
    )
    import_tum.add_argument("source", type=Path)
    import_tum.add_argument("output", type=Path)

    reconstruct = commands.add_parser(
        "reconstruct",
        help="Run known-pose geometric reconstruction steps.",
    )
    reconstruct_commands = reconstruct.add_subparsers(
        dest="reconstruct_command",
        required=True,
    )
    point_cloud = reconstruct_commands.add_parser(
        "point-cloud",
        help="Back-project known-pose RGB-D frames to a colored PLY.",
    )
    point_cloud.add_argument("path", type=Path)
    point_cloud.add_argument("output", type=Path)
    point_cloud.add_argument(
        "--frame-stride",
        type=_positive_integer,
        default=1,
        help="Integrate every Nth RGB observation (default: 1).",
    )
    point_cloud.add_argument(
        "--pixel-stride",
        type=_positive_integer,
        default=1,
        help="Sample every Nth image row and column (default: 1).",
    )
    return parser


def _run_point_cloud(
    path: Path,
    output: Path,
    frame_stride: int,
    pixel_stride: int,
) -> int:
    try:
        session = load_scan_session(path)
        report = reconstruct_point_cloud(
            session,
            output,
            frame_stride=frame_stride,
            pixel_stride=pixel_stride,
        )
    except SessionValidationError as error:
        print(f"RECONSTRUCT FAILED {path}", file=sys.stderr)
        for problem in error.errors:
            print(f"- {problem}", file=sys.stderr)
        return 2
    except (PointCloudError, SessionReplayError) as error:
        print(f"RECONSTRUCT FAILED {path}", file=sys.stderr)
        print(f"- {error}", file=sys.stderr)
        return 2

    print(f"POINT CLOUD {report.session_id}", file=sys.stdout)
    print(
        "frames: "
        f"total={report.total_observations} "
        f"selected={report.selected_observations} "
        f"integrated={report.integrated_frames}",
        file=sys.stdout,
    )
    print(
        "skipped: "
        f"missing_depth={report.skipped_missing_depth} "
        f"missing_pose={report.skipped_missing_pose}",
        file=sys.stdout,
    )
    print(
        f"invalid_depth_samples: {report.invalid_depth_samples}",
        file=sys.stdout,
    )
    print(f"points: {report.points_written}", file=sys.stdout)
    print(f"output: {report.output}", file=sys.stdout)
    print(f"output_sha256: {report.output_digest_sha256}", file=sys.stdout)
    return 0


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected a positive integer") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def _run_tum_import(source: Path, output: Path) -> int:
    try:
        report = import_tum_dataset(source, output)
        session = load_scan_session(report.output)
        replay = replay_session(session)
    except (TumImportError, SessionValidationError, SessionReplayError) as error:
        print(f"IMPORT FAILED {source}", file=sys.stderr)
        if isinstance(error, SessionValidationError):
            for problem in error.errors:
                print(f"- {problem}", file=sys.stderr)
        else:
            print(f"- {error}", file=sys.stderr)
        return 2

    print(f"IMPORTED {report.session_id}", file=sys.stdout)
    print(f"output: {report.output}", file=sys.stdout)
    print(
        "source: "
        f"rgb={report.source_rgb_count} "
        f"depth={report.source_depth_count} "
        f"poses={report.source_pose_count}",
        file=sys.stdout,
    )
    print(
        "matched: "
        f"rgb_depth={report.matched_rgbd_count} "
        f"poses={report.matched_pose_count}",
        file=sys.stdout,
    )
    print(
        "unmatched: "
        f"rgb={report.unmatched_rgb_count} "
        f"depth={report.unmatched_depth_count} "
        f"poses={report.unmatched_pose_count}",
        file=sys.stdout,
    )
    print(f"digest_sha256: {replay.digest_sha256}", file=sys.stdout)
    return 0


def _print_validation_summary(session: ScanSession, output: TextIO) -> None:
    # Kept local to the CLI so the domain model remains presentation-agnostic.
    calibration_ids = ", ".join(sorted(session.calibrations))
    sensors = " ".join(
        f"{name}={'yes' if name in session.streams else 'no'}"
        for name in STREAM_ORDER
    )
    counts = " ".join(
        f"{name}={len(session.streams.get(name, ()))}"
        for name in STREAM_ORDER
    )

    print(f"VALID {session.session_id}", file=output)
    print(f"schema_version: {session.schema_version}", file=output)
    print(f"calibration_ids: {calibration_ids}", file=output)
    print(f"sensors: {sensors}", file=output)
    print(f"samples: {counts}", file=output)
    print(f"time_span_ns: {session.duration_ns}", file=output)
    print(f"duration_s: {session.duration_ns / 1_000_000_000:.9f}", file=output)


def _print_validation_failure(
    path: Path,
    error: SessionValidationError,
    output: TextIO,
) -> None:
    print(f"INVALID {path}", file=output)
    for problem in error.errors:
        print(f"- {problem}", file=output)
