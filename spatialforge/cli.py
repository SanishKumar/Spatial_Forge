"""Command-line entry point for SpatialForge's incremental milestones."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from .errors import (
    MeshExtractionError,
    PointCloudError,
    SessionReplayError,
    SessionValidationError,
    SurfaceExtractionError,
    TsdfError,
    TumImportError,
)
from .mesh import extract_triangle_mesh
from .model import ScanSession
from .point_cloud import reconstruct_point_cloud
from .replay import replay_session
from .session_loader import STREAM_ORDER, load_scan_session
from .sparse_tsdf import integrate_sparse_tsdf
from .surface import extract_surface_points
from .tsdf import _validate_tsdf_output, integrate_tsdf
from .tsdf_block_plan import (
    _validate_tsdf_block_plan_output,
    plan_tsdf_blocks,
)
from .tsdf_block_plan_loader import (
    load_tsdf_block_plan,
    verify_tsdf_block_plan_replay,
)
from .tsdf_block_storage import allocate_empty_tsdf_blocks
from .tsdf_bounds import infer_tsdf_bounds
from .tsdf_replay_depth_context import build_tsdf_replay_depth_context
from .tsdf_voxel_address import locate_tsdf_voxel
from .tsdf_voxel_contribution import evaluate_tsdf_voxel_contribution
from .tsdf_voxel_traversal import traverse_tsdf_voxel_observations
from .tsdf_voxel_update import apply_tsdf_voxel_contribution
from .tum_importer import import_tum_dataset


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)

    if arguments.command == "reconstruct":
        if arguments.reconstruct_command == "point-cloud":
            return _run_point_cloud(
                arguments.path,
                arguments.output,
                arguments.frame_stride,
                arguments.pixel_stride,
            )
        if arguments.reconstruct_command == "tsdf":
            return _run_tsdf(
                arguments.path,
                arguments.output,
                arguments.origin,
                arguments.dimensions,
                arguments.voxel_size_m,
                arguments.truncation_m,
                arguments.frame_stride,
            )
        if arguments.reconstruct_command == "tsdf-sparse":
            return _run_sparse_tsdf(
                arguments.path,
                arguments.output,
                arguments.origin,
                arguments.dimensions,
                arguments.voxel_size_m,
                arguments.truncation_m,
                arguments.frame_stride,
            )
        if arguments.reconstruct_command == "tsdf-block-plan":
            return _run_tsdf_block_plan(
                arguments.path,
                arguments.output,
                arguments.voxel_size_m,
                arguments.truncation_m,
                arguments.frame_stride,
            )
        if arguments.reconstruct_command == "tsdf-block-plan-verify":
            return _run_tsdf_block_plan_verify(
                arguments.plan,
                arguments.session,
            )
        if (
            arguments.reconstruct_command
            == "tsdf-block-replay-context"
        ):
            return _run_tsdf_block_replay_context(
                arguments.plan,
                arguments.session,
            )
        if arguments.reconstruct_command == "tsdf-block-allocate":
            return _run_tsdf_block_allocate(
                arguments.plan,
                arguments.session,
            )
        if arguments.reconstruct_command == "tsdf-block-address":
            return _run_tsdf_block_address(
                arguments.plan,
                arguments.session,
                arguments.voxels,
            )
        if arguments.reconstruct_command == "tsdf-block-contribution":
            return _run_tsdf_block_contribution(
                arguments.plan,
                arguments.session,
                arguments.observation_sequence,
                arguments.voxel,
            )
        if (
            arguments.reconstruct_command
            == "tsdf-block-contribution-apply"
        ):
            return _run_tsdf_block_contribution_apply(
                arguments.plan,
                arguments.session,
                arguments.observation_sequence,
                arguments.voxel,
            )
        if arguments.reconstruct_command == "tsdf-block-voxel-traverse":
            return _run_tsdf_block_voxel_traverse(
                arguments.plan,
                arguments.session,
                arguments.voxel,
            )
        if arguments.reconstruct_command == "tsdf-auto":
            return _run_auto_tsdf(
                arguments.path,
                arguments.output,
                arguments.voxel_size_m,
                arguments.truncation_m,
                arguments.frame_stride,
            )
        if arguments.reconstruct_command == "surface-points":
            return _run_surface_points(arguments.path, arguments.output)
        return _run_triangle_mesh(arguments.path, arguments.output)
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

    tsdf = reconstruct_commands.add_parser(
        "tsdf",
        help="Integrate a fixed-bounds reference TSDF diagnostic.",
    )
    tsdf.add_argument("path", type=Path)
    tsdf.add_argument("output", type=Path)
    tsdf.add_argument(
        "--origin",
        type=float,
        nargs=3,
        required=True,
        metavar=("X", "Y", "Z"),
        help="World-space volume origin in metres.",
    )
    tsdf.add_argument(
        "--dimensions",
        type=_positive_integer,
        nargs=3,
        required=True,
        metavar=("NX", "NY", "NZ"),
        help="Voxel counts along world X, Y, and Z.",
    )
    tsdf.add_argument(
        "--voxel-size-m",
        type=float,
        default=0.05,
        help="Voxel edge length in metres (default: 0.05).",
    )
    tsdf.add_argument(
        "--truncation-m",
        type=float,
        default=0.10,
        help="Signed-distance truncation in metres (default: 0.10).",
    )
    tsdf.add_argument(
        "--frame-stride",
        type=_positive_integer,
        default=1,
        help="Integrate every Nth RGB observation (default: 1).",
    )

    sparse_tsdf = reconstruct_commands.add_parser(
        "tsdf-sparse",
        help="Dense-traversal TSDF with a sparse in-memory accumulator.",
    )
    sparse_tsdf.add_argument("path", type=Path)
    sparse_tsdf.add_argument("output", type=Path)
    sparse_tsdf.add_argument(
        "--origin",
        type=float,
        nargs=3,
        required=True,
        metavar=("X", "Y", "Z"),
        help="World-space volume origin in metres.",
    )
    sparse_tsdf.add_argument(
        "--dimensions",
        type=_positive_integer,
        nargs=3,
        required=True,
        metavar=("NX", "NY", "NZ"),
        help="Voxel counts along world X, Y, and Z.",
    )
    sparse_tsdf.add_argument(
        "--voxel-size-m",
        type=float,
        default=0.05,
        help="Voxel edge length in metres (default: 0.05).",
    )
    sparse_tsdf.add_argument(
        "--truncation-m",
        type=float,
        default=0.10,
        help="Signed-distance truncation in metres (default: 0.10).",
    )
    sparse_tsdf.add_argument(
        "--frame-stride",
        type=_positive_integer,
        default=1,
        help="Integrate every Nth RGB observation (default: 1).",
    )

    block_plan = reconstruct_commands.add_parser(
        "tsdf-block-plan",
        help=(
            "Plan candidate surface-neighborhood blocks; "
            "does not fuse TSDF values."
        ),
    )
    block_plan.add_argument("path", type=Path)
    block_plan.add_argument("output", type=Path)
    block_plan.add_argument(
        "--voxel-size-m",
        type=float,
        default=0.05,
        help="Voxel edge length in metres (default: 0.05).",
    )
    block_plan.add_argument(
        "--truncation-m",
        type=float,
        default=0.10,
        help="Surface-neighborhood half-width in metres (default: 0.10).",
    )
    block_plan.add_argument(
        "--frame-stride",
        type=_positive_integer,
        default=1,
        help="Plan from every Nth RGB observation (default: 1).",
    )

    block_plan_verify = reconstruct_commands.add_parser(
        "tsdf-block-plan-verify",
        help=(
            "Strictly validate a block plan and match its replay digest; "
            "does not replan or fuse."
        ),
        description=(
            "Strictly validate a block plan and match its recorded replay "
            "digest. This read-only check does not replan geometry or fuse "
            "TSDF values."
        ),
    )
    block_plan_verify.add_argument(
        "plan",
        type=Path,
        help="Existing .sftplan artifact to validate.",
    )
    block_plan_verify.add_argument(
        "session",
        type=Path,
        help="Current .vgsession directory to match.",
    )

    block_replay_context = reconstruct_commands.add_parser(
        "tsdf-block-replay-context",
        help=(
            "Build an immutable replay/depth context from a matched plan; "
            "does not allocate TSDF storage or evaluate voxels."
        ),
        description=(
            "Strictly load and replay-match a block plan, then decode each "
            "ready selected depth frame once into an immutable in-memory "
            "context. The context is discarded when this read-only "
            "diagnostic exits."
        ),
    )
    block_replay_context.add_argument(
        "plan",
        type=Path,
        help="Existing .sftplan artifact to validate.",
    )
    block_replay_context.add_argument(
        "session",
        type=Path,
        help="Current .vgsession directory to replay and match.",
    )

    block_allocate = reconstruct_commands.add_parser(
        "tsdf-block-allocate",
        help=(
            "Allocate empty in-memory TSDF blocks from a replay-matched plan; "
            "does not decode depth, fuse, or write an artifact."
        ),
        description=(
            "Strictly load and replay-match a block plan, then allocate "
            "zeroed in-memory TSDF sum and weight buffers. The buffers are "
            "discarded when this read-only diagnostic exits."
        ),
    )
    block_allocate.add_argument(
        "plan",
        type=Path,
        help="Existing .sftplan artifact whose active blocks are allocated.",
    )
    block_allocate.add_argument(
        "session",
        type=Path,
        help="Current .vgsession directory to replay-match before allocation.",
    )

    block_address = reconstruct_commands.add_parser(
        "tsdf-block-address",
        help=(
            "Resolve signed global voxel indices in replay-matched block "
            "storage; does not create blocks, mutate state, or fuse."
        ),
        description=(
            "Allocate replay-matched empty block storage, then resolve one "
            "or more signed global voxel indices into planned block rows and "
            "local array indices. Valid sparse misses are reported without "
            "creating blocks."
        ),
    )
    block_address.add_argument(
        "plan",
        type=Path,
        help="Existing .sftplan artifact whose active blocks are addressed.",
    )
    block_address.add_argument(
        "session",
        type=Path,
        help="Current .vgsession directory to replay-match before addressing.",
    )
    block_address.add_argument(
        "--voxel",
        dest="voxels",
        type=int,
        nargs=3,
        action="append",
        required=True,
        metavar=("GX", "GY", "GZ"),
        help="Signed global voxel XYZ to resolve; may be repeated.",
    )

    block_contribution = reconstruct_commands.add_parser(
        "tsdf-block-contribution",
        help=(
            "Evaluate one planned voxel against one selected replay "
            "observation without applying a contribution."
        ),
        description=(
            "Strictly load and replay-match a block plan, address one "
            "planned voxel, then evaluate that voxel against one selected "
            "replay observation. Exact depth and pose produce a projective "
            "result; missing inputs produce a skip diagnostic. No returned "
            "sum or weight delta is applied."
        ),
    )
    block_contribution.add_argument(
        "plan",
        type=Path,
        help="Existing .sftplan artifact whose grid parameters are used.",
    )
    block_contribution.add_argument(
        "session",
        type=Path,
        help="Current .vgsession directory to replay-match and sample.",
    )
    block_contribution.add_argument(
        "--observation-sequence",
        type=int,
        default=0,
        help="Zero-based replay observation sequence (default: 0).",
    )
    block_contribution.add_argument(
        "--voxel",
        type=int,
        nargs=3,
        required=True,
        metavar=("GX", "GY", "GZ"),
        help="One planned signed global voxel XYZ to evaluate.",
    )

    block_contribution_apply = reconstruct_commands.add_parser(
        "tsdf-block-contribution-apply",
        help=(
            "Evaluate and apply one accepted contribution to one temporary "
            "planned voxel slot."
        ),
        description=(
            "Strictly load and replay-match a block plan, allocate temporary "
            "storage, evaluate one selected observation at one planned "
            "voxel, and apply that accepted contribution to exactly one "
            "slot. The mutated storage is discarded and no artifact is "
            "written."
        ),
    )
    block_contribution_apply.add_argument(
        "plan",
        type=Path,
        help="Existing .sftplan artifact whose temporary storage is updated.",
    )
    block_contribution_apply.add_argument(
        "session",
        type=Path,
        help="Current .vgsession directory to replay-bind through mutation.",
    )
    block_contribution_apply.add_argument(
        "--observation-sequence",
        type=int,
        default=0,
        help="Zero-based replay observation sequence (default: 0).",
    )
    block_contribution_apply.add_argument(
        "--voxel",
        type=int,
        nargs=3,
        required=True,
        metavar=("GX", "GY", "GZ"),
        help="One planned signed global voxel XYZ to evaluate and update.",
    )

    block_voxel_traverse = reconstruct_commands.add_parser(
        "tsdf-block-voxel-traverse",
        help=(
            "Evaluate every plan-selected observation for one temporary "
            "planned voxel and apply accepted contributions."
        ),
        description=(
            "Strictly load and replay-match a block plan, allocate temporary "
            "storage, evaluate every observation selected by the plan for "
            "one planned voxel, and apply accepted contributions in "
            "canonical sequence order. No other voxel is evaluated and no "
            "artifact is written."
        ),
    )
    block_voxel_traverse.add_argument(
        "plan",
        type=Path,
        help="Existing .sftplan artifact whose temporary storage is updated.",
    )
    block_voxel_traverse.add_argument(
        "session",
        type=Path,
        help="Current .vgsession directory to replay-bind through traversal.",
    )
    block_voxel_traverse.add_argument(
        "--voxel",
        type=int,
        nargs=3,
        required=True,
        metavar=("GX", "GY", "GZ"),
        help="One planned signed global voxel XYZ to traverse and update.",
    )

    auto_tsdf = reconstruct_commands.add_parser(
        "tsdf-auto",
        help="Infer world-aligned bounds and integrate a reference TSDF.",
    )
    auto_tsdf.add_argument("path", type=Path)
    auto_tsdf.add_argument("output", type=Path)
    auto_tsdf.add_argument(
        "--voxel-size-m",
        type=float,
        default=0.05,
        help="Voxel edge length in metres (default: 0.05).",
    )
    auto_tsdf.add_argument(
        "--truncation-m",
        type=float,
        default=0.10,
        help="Bounds padding and TSDF truncation in metres (default: 0.10).",
    )
    auto_tsdf.add_argument(
        "--frame-stride",
        type=_positive_integer,
        default=1,
        help="Use every Nth RGB observation (default: 1).",
    )

    surface_points = reconstruct_commands.add_parser(
        "surface-points",
        help="Extract deterministic zero-crossing points from a reference TSDF.",
    )
    surface_points.add_argument("path", type=Path)
    surface_points.add_argument("output", type=Path)

    triangle_mesh = reconstruct_commands.add_parser(
        "triangle-mesh",
        help="Extract a deterministic reference triangle mesh from a TSDF.",
    )
    triangle_mesh.add_argument("path", type=Path)
    triangle_mesh.add_argument("output", type=Path)
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


def _run_tsdf(
    path: Path,
    output: Path,
    origin: Sequence[float],
    dimensions: Sequence[int],
    voxel_size_m: float,
    truncation_m: float,
    frame_stride: int,
) -> int:
    try:
        session = load_scan_session(path)
        report = integrate_tsdf(
            session,
            output,
            origin_world_m=origin,
            dimensions=dimensions,
            voxel_size_m=voxel_size_m,
            truncation_m=truncation_m,
            frame_stride=frame_stride,
        )
    except SessionValidationError as error:
        print(f"TSDF FAILED {path}", file=sys.stderr)
        for problem in error.errors:
            print(f"- {problem}", file=sys.stderr)
        return 2
    except (TsdfError, SessionReplayError) as error:
        print(f"TSDF FAILED {path}", file=sys.stderr)
        print(f"- {error}", file=sys.stderr)
        return 2

    print(f"TSDF {report.session_id}", file=sys.stdout)
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
    print(f"invalid_depth_pixels: {report.invalid_depth_pixels}", file=sys.stdout)
    print(
        "voxels: "
        f"total={report.total_voxels} "
        f"observed={report.observed_voxels} "
        f"fused={report.fused_voxels}",
        file=sys.stdout,
    )
    print(
        f"voxel_updates: {report.voxel_updates} "
        f"max_weight={report.max_weight}",
        file=sys.stdout,
    )
    print(f"output: {report.output}", file=sys.stdout)
    print(f"output_sha256: {report.output_digest_sha256}", file=sys.stdout)
    return 0


def _run_sparse_tsdf(
    path: Path,
    output: Path,
    origin: Sequence[float],
    dimensions: Sequence[int],
    voxel_size_m: float,
    truncation_m: float,
    frame_stride: int,
) -> int:
    try:
        session = load_scan_session(path)
        report = integrate_sparse_tsdf(
            session,
            output,
            origin_world_m=origin,
            dimensions=dimensions,
            voxel_size_m=voxel_size_m,
            truncation_m=truncation_m,
            frame_stride=frame_stride,
        )
    except SessionValidationError as error:
        print(f"SPARSE TSDF FAILED {path}", file=sys.stderr)
        for problem in error.errors:
            print(f"- {problem}", file=sys.stderr)
        return 2
    except (TsdfError, SessionReplayError) as error:
        print(f"SPARSE TSDF FAILED {path}", file=sys.stderr)
        print(f"- {error}", file=sys.stderr)
        return 2

    print(f"SPARSE TSDF {report.session_id}", file=sys.stdout)
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
    print(f"invalid_depth_pixels: {report.invalid_depth_pixels}", file=sys.stdout)
    print(
        "voxels: "
        f"total={report.total_voxels} "
        f"observed={report.observed_voxels} "
        f"fused={report.fused_voxels}",
        file=sys.stdout,
    )
    print(
        f"voxel_updates: {report.voxel_updates} "
        f"max_weight={report.max_weight}",
        file=sys.stdout,
    )
    print(
        f"storage: sparse accumulator_entries={report.observed_voxels}",
        file=sys.stdout,
    )
    print(f"output: {report.output}", file=sys.stdout)
    print(f"output_sha256: {report.output_digest_sha256}", file=sys.stdout)
    return 0


def _run_tsdf_block_plan(
    path: Path,
    output: Path,
    voxel_size_m: float,
    truncation_m: float,
    frame_stride: int,
) -> int:
    try:
        _validate_tsdf_block_plan_output(output)
        session = load_scan_session(path)
        report = plan_tsdf_blocks(
            session,
            output,
            voxel_size_m=voxel_size_m,
            truncation_m=truncation_m,
            frame_stride=frame_stride,
        )
    except SessionValidationError as error:
        print(f"TSDF BLOCK PLAN FAILED {path}", file=sys.stderr)
        for problem in error.errors:
            print(f"- {problem}", file=sys.stderr)
        return 2
    except (TsdfError, SessionReplayError) as error:
        print(f"TSDF BLOCK PLAN FAILED {path}", file=sys.stderr)
        print(f"- {error}", file=sys.stderr)
        return 2

    print(f"TSDF BLOCK PLAN {report.session_id}", file=sys.stdout)
    print(
        "frames: "
        f"total={report.total_observations} "
        f"selected={report.selected_observations} "
        f"paired={report.paired_observations}",
        file=sys.stdout,
    )
    print(
        "skipped: "
        f"missing_depth={report.skipped_missing_depth} "
        f"missing_pose={report.skipped_missing_pose}",
        file=sys.stdout,
    )
    print(
        "depth_samples: "
        f"valid={report.valid_depth_points} "
        f"invalid={report.invalid_depth_samples}",
        file=sys.stdout,
    )
    print(
        "grid: "
        f"voxel_size_m={report.voxel_size_m:.9f} "
        f"block_resolution={report.block_resolution} "
        f"block_extent_m={report.block_extent_m:.9f}",
        file=sys.stdout,
    )
    print(
        "candidate_blocks: "
        f"surface={report.surface_block_count} "
        f"active={report.active_block_count} "
        f"halo={report.halo_block_count} "
        f"voxel_slots={report.planned_voxel_slots}",
        file=sys.stdout,
    )
    print(
        "block_bounds: "
        f"min={report.min_block_index} "
        f"max={report.max_block_index}",
        file=sys.stdout,
    )
    print(f"output: {report.output}", file=sys.stdout)
    print(f"output_sha256: {report.output_digest_sha256}", file=sys.stdout)
    return 0


def _run_tsdf_block_plan_verify(
    plan_path: Path,
    session_path: Path,
) -> int:
    try:
        plan = load_tsdf_block_plan(plan_path)
        session = load_scan_session(session_path)
        verify_tsdf_block_plan_replay(plan, session)
    except SessionValidationError as error:
        print(
            f"TSDF BLOCK PLAN VERIFY FAILED {plan_path}",
            file=sys.stderr,
        )
        for problem in error.errors:
            print(f"- {problem}", file=sys.stderr)
        return 2
    except (TsdfError, SessionReplayError) as error:
        print(
            f"TSDF BLOCK PLAN VERIFY FAILED {plan_path}",
            file=sys.stderr,
        )
        print(f"- {error}", file=sys.stderr)
        return 2

    print(f"TSDF BLOCK PLAN CHECK {plan.session_id}", file=sys.stdout)
    print("artifact: valid", file=sys.stdout)
    print("session_replay: matched", file=sys.stdout)
    print("geometry_recomputed: no", file=sys.stdout)
    print(
        "frames: "
        f"total={plan.total_observations} "
        f"selected={plan.selected_observations} "
        f"paired={plan.paired_observations}",
        file=sys.stdout,
    )
    print(
        "candidate_blocks: "
        f"surface={plan.surface_block_count} "
        f"active={plan.active_block_count} "
        f"halo={plan.halo_block_count} "
        f"voxel_slots={plan.planned_voxel_slots}",
        file=sys.stdout,
    )
    print(f"plan: {plan.path}", file=sys.stdout)
    print(
        f"plan_sha256: {plan.artifact_digest_sha256}",
        file=sys.stdout,
    )
    print(
        f"replay_digest_sha256: {plan.replay_digest_sha256}",
        file=sys.stdout,
    )
    return 0


def _run_tsdf_block_replay_context(
    plan_path: Path,
    session_path: Path,
) -> int:
    try:
        plan = load_tsdf_block_plan(plan_path)
        session = load_scan_session(session_path)
        context = build_tsdf_replay_depth_context(plan, session)
    except SessionValidationError as error:
        print(
            f"TSDF BLOCK REPLAY DEPTH CONTEXT FAILED {plan_path}",
            file=sys.stderr,
        )
        for problem in error.errors:
            print(f"- {problem}", file=sys.stderr)
        return 2
    except (TsdfError, SessionReplayError) as error:
        print(
            f"TSDF BLOCK REPLAY DEPTH CONTEXT FAILED {plan_path}",
            file=sys.stderr,
        )
        print(f"- {error}", file=sys.stderr)
        return 2

    print(
        f"TSDF BLOCK REPLAY DEPTH CONTEXT CHECK {context.session_id}",
        file=sys.stdout,
    )
    print("artifact: valid", file=sys.stdout)
    print("session_replay: matched", file=sys.stdout)
    print(
        "selection: "
        f"frame_stride={context.frame_stride} "
        f"total={context.total_observations} "
        f"selected={len(context.selected_observation_sequences)}",
        file=sys.stdout,
    )
    for index, observation in enumerate(context.observations):
        print(
            f"observation[{index}]: "
            f"sequence={observation.observation_sequence} "
            f"status={observation.status.value} "
            "depth_decoded="
            f"{'yes' if observation.depth_m is not None else 'no'}",
            file=sys.stdout,
        )
    print(
        "status_counts: "
        + " ".join(
            f"{status.value}={count}"
            for status, count in context.status_counts
        ),
        file=sys.stdout,
    )
    print(
        f"ready_observations: {context.ready_observation_count}",
        file=sys.stdout,
    )
    print(
        f"depth_frames_decoded: {context.depth_frames_decoded}",
        file=sys.stdout,
    )
    print(
        "depth_layout: "
        f"shape=({context.camera.height}, {context.camera.width}) "
        "dtype=float64 "
        f"samples={context.depth_sample_count} "
        f"payload_bytes={context.depth_payload_bytes}",
        file=sys.stdout,
    )
    print("context_immutable: yes", file=sys.stdout)
    print("tsdf_storage_allocated: no", file=sys.stdout)
    print("voxel_evaluation_performed: no", file=sys.stdout)
    print("voxel_observation_traversal_performed: no", file=sys.stdout)
    print("voxel_address_traversal_performed: no", file=sys.stdout)
    print("fusion_block_traversal_performed: no", file=sys.stdout)
    print("ray_traversal_performed: no", file=sys.stdout)
    print("full_fusion_performed: no", file=sys.stdout)
    print("artifact_written: no", file=sys.stdout)
    print("context_persisted: no", file=sys.stdout)
    print(
        f"plan_sha256: {context.source_plan_digest_sha256}",
        file=sys.stdout,
    )
    print(
        f"replay_digest_sha256: {context.replay_digest_sha256}",
        file=sys.stdout,
    )
    return 0


def _run_tsdf_block_allocate(
    plan_path: Path,
    session_path: Path,
) -> int:
    try:
        plan = load_tsdf_block_plan(plan_path)
        session = load_scan_session(session_path)
        storage = allocate_empty_tsdf_blocks(plan, session)
    except SessionValidationError as error:
        print(
            f"TSDF BLOCK ALLOCATION FAILED {plan_path}",
            file=sys.stderr,
        )
        for problem in error.errors:
            print(f"- {problem}", file=sys.stderr)
        return 2
    except (TsdfError, SessionReplayError) as error:
        print(
            f"TSDF BLOCK ALLOCATION FAILED {plan_path}",
            file=sys.stderr,
        )
        print(f"- {error}", file=sys.stderr)
        return 2

    print(
        f"TSDF BLOCK ALLOCATION CHECK {plan.session_id}",
        file=sys.stdout,
    )
    print("artifact: valid", file=sys.stdout)
    print("session_replay: matched", file=sys.stdout)
    print("depth_decoded: no", file=sys.stdout)
    print("geometry_recomputed: no", file=sys.stdout)
    print("fusion_performed: no", file=sys.stdout)
    print("artifact_written: no", file=sys.stdout)
    print(
        "allocation: "
        f"blocks={storage.block_count} "
        f"resolution={storage.block_resolution} "
        f"voxel_slots={storage.voxel_slots}",
        file=sys.stdout,
    )
    print(
        f"layout: shape={storage.tsdf_sums.shape} "
        "axes=block-z-y-x x_fastest=yes",
        file=sys.stdout,
    )
    print(
        "dtypes: "
        f"tsdf_sums={storage.tsdf_sums.dtype.name} "
        f"weights={storage.weights.dtype.name}",
        file=sys.stdout,
    )
    print(
        "zero_state: "
        f"nonzero_sums={storage.nonzero_sum_count} "
        f"nonzero_weights={storage.nonzero_weight_count} "
        f"unknown_voxels={storage.unknown_voxel_count}",
        file=sys.stdout,
    )
    print(
        "payload_bytes: "
        f"tsdf_sums={storage.tsdf_sum_bytes} "
        f"weights={storage.weight_bytes} "
        f"total={storage.payload_bytes}",
        file=sys.stdout,
    )
    print(
        "block_rows: "
        f"first={storage.block_indices[0]} "
        f"last={storage.block_indices[-1]}",
        file=sys.stdout,
    )
    print(
        f"plan_sha256: {plan.artifact_digest_sha256}",
        file=sys.stdout,
    )
    print(
        f"replay_digest_sha256: {plan.replay_digest_sha256}",
        file=sys.stdout,
    )
    return 0


def _run_tsdf_block_address(
    plan_path: Path,
    session_path: Path,
    voxels: list[list[int]],
) -> int:
    try:
        plan = load_tsdf_block_plan(plan_path)
        session = load_scan_session(session_path)
        storage = allocate_empty_tsdf_blocks(plan, session)
        before = (
            storage.block_indices,
            id(storage.tsdf_sums),
            id(storage.weights),
            storage.nonzero_sum_count,
            storage.nonzero_weight_count,
            storage.payload_bytes,
        )
        results = tuple(
            (
                tuple(raw_voxel),
                locate_tsdf_voxel(storage, tuple(raw_voxel)),
            )
            for raw_voxel in voxels
        )
        after = (
            storage.block_indices,
            id(storage.tsdf_sums),
            id(storage.weights),
            storage.nonzero_sum_count,
            storage.nonzero_weight_count,
            storage.payload_bytes,
        )
        if after != before:
            raise AssertionError("TSDF voxel addressing mutated block storage")
    except SessionValidationError as error:
        print(
            f"TSDF BLOCK ADDRESS FAILED {plan_path}",
            file=sys.stderr,
        )
        for problem in error.errors:
            print(f"- {problem}", file=sys.stderr)
        return 2
    except (TsdfError, SessionReplayError) as error:
        print(
            f"TSDF BLOCK ADDRESS FAILED {plan_path}",
            file=sys.stderr,
        )
        print(f"- {error}", file=sys.stderr)
        return 2

    resolved = sum(address is not None for _, address in results)
    print(f"TSDF BLOCK ADDRESS CHECK {plan.session_id}", file=sys.stdout)
    print("artifact: valid", file=sys.stdout)
    print("session_replay: matched", file=sys.stdout)
    print("depth_decoded: no", file=sys.stdout)
    print("geometry_recomputed: no", file=sys.stdout)
    print("fusion_performed: no", file=sys.stdout)
    print("storage_mutated: no", file=sys.stdout)
    print("addressing_created_blocks: no", file=sys.stdout)
    print("artifact_written: no", file=sys.stdout)
    print(
        "allocation: "
        f"blocks={storage.block_count} "
        f"voxel_slots={storage.voxel_slots}",
        file=sys.stdout,
    )
    print(
        "queries: "
        f"requested={len(results)} "
        f"resolved={resolved} "
        f"unplanned={len(results) - resolved}",
        file=sys.stdout,
    )
    for position, (global_index, address) in enumerate(results):
        if address is None:
            print(
                f"voxel[{position}]: status=unplanned "
                f"global={global_index}",
                file=sys.stdout,
            )
            continue
        print(
            f"voxel[{position}]: status=planned "
            f"global={address.global_index_xyz} "
            f"block={address.block_index_xyz} "
            f"local={address.local_index_xyz} "
            f"row={address.block_row} "
            f"array={address.array_index_bzyx} "
            f"local_flat={address.local_flat_index} "
            f"storage_flat={address.storage_flat_index}",
            file=sys.stdout,
        )
    print(
        f"plan_sha256: {plan.artifact_digest_sha256}",
        file=sys.stdout,
    )
    print(
        f"replay_digest_sha256: {plan.replay_digest_sha256}",
        file=sys.stdout,
    )
    return 0


def _run_tsdf_block_contribution(
    plan_path: Path,
    session_path: Path,
    observation_sequence: int,
    voxel: list[int],
) -> int:
    try:
        plan = load_tsdf_block_plan(plan_path)
        session = load_scan_session(session_path)
        storage = allocate_empty_tsdf_blocks(plan, session)
        address = locate_tsdf_voxel(storage, tuple(voxel))
        if address is None:
            raise TsdfError(
                f"global voxel {tuple(voxel)} is not in a planned block"
            )
        before = (
            storage.block_indices,
            id(storage.tsdf_sums),
            id(storage.weights),
            storage.nonzero_sum_count,
            storage.nonzero_weight_count,
            storage.payload_bytes,
        )
        contribution = evaluate_tsdf_voxel_contribution(
            storage,
            address,
            session,
            observation_sequence,
        )
        after = (
            storage.block_indices,
            id(storage.tsdf_sums),
            id(storage.weights),
            storage.nonzero_sum_count,
            storage.nonzero_weight_count,
            storage.payload_bytes,
        )
        if after != before:
            raise AssertionError(
                "TSDF contribution evaluation mutated block storage"
            )
    except SessionValidationError as error:
        print(
            f"TSDF BLOCK CONTRIBUTION FAILED {plan_path}",
            file=sys.stderr,
        )
        for problem in error.errors:
            print(f"- {problem}", file=sys.stderr)
        return 2
    except (TsdfError, SessionReplayError) as error:
        print(
            f"TSDF BLOCK CONTRIBUTION FAILED {plan_path}",
            file=sys.stderr,
        )
        print(f"- {error}", file=sys.stderr)
        return 2

    print(
        f"TSDF BLOCK CONTRIBUTION CHECK {plan.session_id}",
        file=sys.stdout,
    )
    print("artifact: valid", file=sys.stdout)
    print("session_replay: matched", file=sys.stdout)
    print(
        f"observation_sequence: {contribution.observation_sequence}",
        file=sys.stdout,
    )
    print(
        "voxel: "
        f"global={address.global_index_xyz} "
        f"block={address.block_index_xyz} "
        f"local={address.local_index_xyz} "
        f"row={address.block_row} "
        f"array={address.array_index_bzyx} "
        f"storage_flat={address.storage_flat_index}",
        file=sys.stdout,
    )
    print(
        f"world_xyz_m: {_format_triplet(contribution.world_xyz_m)}",
        file=sys.stdout,
    )
    if contribution.camera_xyz_m is not None:
        print(
            f"camera_xyz_m: {_format_triplet(contribution.camera_xyz_m)}",
            file=sys.stdout,
        )
    if contribution.projected_uv is not None:
        print(
            "projected_uv: "
            f"({contribution.projected_uv[0]:.9f}, "
            f"{contribution.projected_uv[1]:.9f})",
            file=sys.stdout,
        )
    if contribution.pixel_uv is not None:
        print(f"pixel_uv: {contribution.pixel_uv}", file=sys.stdout)
    print(
        "depth_decoded: "
        f"{'yes' if contribution.depth_decoded else 'no'}",
        file=sys.stdout,
    )
    if contribution.measured_depth_m is not None:
        print(
            f"measured_depth_m: {contribution.measured_depth_m:.9f}",
            file=sys.stdout,
        )
    if contribution.signed_distance_m is not None:
        print(
            f"signed_distance_m: {contribution.signed_distance_m:.9f}",
            file=sys.stdout,
        )
    print(f"evaluation: {contribution.status.value}", file=sys.stdout)
    if contribution.contributes:
        print(
            "proposed_delta: "
            f"tsdf_sum={contribution.tsdf_sum_delta:.9f} "
            f"weight={contribution.weight_delta}",
            file=sys.stdout,
        )
    else:
        print(
            "proposed_delta: tsdf_sum=none weight=0",
            file=sys.stdout,
        )
    print("contributions_applied: 0", file=sys.stdout)
    print("fusion_performed: no", file=sys.stdout)
    print("storage_mutated: no", file=sys.stdout)
    print("missing_blocks_created: no", file=sys.stdout)
    print("artifact_written: no", file=sys.stdout)
    print(
        f"plan_sha256: {plan.artifact_digest_sha256}",
        file=sys.stdout,
    )
    print(
        f"replay_digest_sha256: {plan.replay_digest_sha256}",
        file=sys.stdout,
    )
    return 0


def _run_tsdf_block_contribution_apply(
    plan_path: Path,
    session_path: Path,
    observation_sequence: int,
    voxel: list[int],
) -> int:
    try:
        plan = load_tsdf_block_plan(plan_path)
        session = load_scan_session(session_path)
        storage = allocate_empty_tsdf_blocks(plan, session)
        address = locate_tsdf_voxel(storage, tuple(voxel))
        if address is None:
            raise TsdfError(
                f"global voxel {tuple(voxel)} is not in a planned block"
            )
        contribution = evaluate_tsdf_voxel_contribution(
            storage,
            address,
            session,
            observation_sequence,
        )
        storage_before = (
            storage.nonzero_sum_count,
            storage.nonzero_weight_count,
            storage.unknown_voxel_count,
        )
        receipt = apply_tsdf_voxel_contribution(
            storage,
            contribution,
            session,
        )
        storage_after = (
            storage.nonzero_sum_count,
            storage.nonzero_weight_count,
            storage.unknown_voxel_count,
        )
    except SessionValidationError as error:
        print(
            f"TSDF BLOCK CONTRIBUTION APPLY FAILED {plan_path}",
            file=sys.stderr,
        )
        for problem in error.errors:
            print(f"- {problem}", file=sys.stderr)
        return 2
    except (TsdfError, SessionReplayError) as error:
        print(
            f"TSDF BLOCK CONTRIBUTION APPLY FAILED {plan_path}",
            file=sys.stderr,
        )
        print(f"- {error}", file=sys.stderr)
        return 2

    print(
        f"TSDF BLOCK CONTRIBUTION APPLY CHECK {plan.session_id}",
        file=sys.stdout,
    )
    print("artifact: valid", file=sys.stdout)
    print("session_replay: matched", file=sys.stdout)
    print(
        f"observation_sequence: {contribution.observation_sequence}",
        file=sys.stdout,
    )
    print(
        "voxel: "
        f"global={address.global_index_xyz} "
        f"block={address.block_index_xyz} "
        f"local={address.local_index_xyz} "
        f"row={address.block_row} "
        f"array={address.array_index_bzyx} "
        f"storage_flat={address.storage_flat_index}",
        file=sys.stdout,
    )
    print(f"evaluation: {contribution.status.value}", file=sys.stdout)
    print(
        "slot_before: "
        f"tsdf_sum={receipt.tsdf_sum_before:.9f} "
        f"weight={receipt.weight_before}",
        file=sys.stdout,
    )
    print(
        "applied_delta: "
        f"tsdf_sum={contribution.tsdf_sum_delta:.9f} "
        f"weight={contribution.weight_delta}",
        file=sys.stdout,
    )
    print(
        "slot_after: "
        f"tsdf_sum={receipt.tsdf_sum_after:.9f} "
        f"weight={receipt.weight_after}",
        file=sys.stdout,
    )
    print(
        "storage_before: "
        f"nonzero_sums={storage_before[0]} "
        f"nonzero_weights={storage_before[1]} "
        f"unknown_voxels={storage_before[2]}",
        file=sys.stdout,
    )
    print(
        "storage_after: "
        f"nonzero_sums={storage_after[0]} "
        f"nonzero_weights={storage_after[1]} "
        f"unknown_voxels={storage_after[2]}",
        file=sys.stdout,
    )
    print("contributions_evaluated: 1", file=sys.stdout)
    print("contributions_applied: 1", file=sys.stdout)
    print("storage_slots_updated: 1", file=sys.stdout)
    print("fusion_block_traversal_performed: no", file=sys.stdout)
    print("ray_traversal_performed: no", file=sys.stdout)
    print("full_fusion_performed: no", file=sys.stdout)
    print("missing_blocks_created: no", file=sys.stdout)
    print("artifact_written: no", file=sys.stdout)
    print("storage_persisted: no", file=sys.stdout)
    print(
        f"plan_sha256: {plan.artifact_digest_sha256}",
        file=sys.stdout,
    )
    print(
        f"replay_digest_sha256: {plan.replay_digest_sha256}",
        file=sys.stdout,
    )
    return 0


def _run_tsdf_block_voxel_traverse(
    plan_path: Path,
    session_path: Path,
    voxel: list[int],
) -> int:
    try:
        plan = load_tsdf_block_plan(plan_path)
        session = load_scan_session(session_path)
        storage = allocate_empty_tsdf_blocks(plan, session)
        address = locate_tsdf_voxel(storage, tuple(voxel))
        if address is None:
            raise TsdfError(
                f"global voxel {tuple(voxel)} is not in a planned block"
            )
        storage_before = (
            storage.nonzero_sum_count,
            storage.nonzero_weight_count,
            storage.unknown_voxel_count,
        )
        receipt = traverse_tsdf_voxel_observations(
            storage,
            address,
            session,
        )
        storage_after = (
            storage.nonzero_sum_count,
            storage.nonzero_weight_count,
            storage.unknown_voxel_count,
        )
    except SessionValidationError as error:
        print(
            f"TSDF BLOCK VOXEL TRAVERSAL FAILED {plan_path}",
            file=sys.stderr,
        )
        for problem in error.errors:
            print(f"- {problem}", file=sys.stderr)
        return 2
    except (TsdfError, SessionReplayError) as error:
        print(
            f"TSDF BLOCK VOXEL TRAVERSAL FAILED {plan_path}",
            file=sys.stderr,
        )
        print(f"- {error}", file=sys.stderr)
        return 2

    print(
        f"TSDF BLOCK VOXEL TRAVERSAL CHECK {plan.session_id}",
        file=sys.stdout,
    )
    print("artifact: valid", file=sys.stdout)
    print("session_replay: matched", file=sys.stdout)
    print(
        "voxel: "
        f"global={address.global_index_xyz} "
        f"block={address.block_index_xyz} "
        f"local={address.local_index_xyz} "
        f"row={address.block_row} "
        f"array={address.array_index_bzyx} "
        f"storage_flat={address.storage_flat_index}",
        file=sys.stdout,
    )
    print(
        "selection: "
        f"frame_stride={receipt.frame_stride} "
        f"total={receipt.total_observations} "
        f"selected={receipt.evaluated_count}",
        file=sys.stdout,
    )
    print(
        "slot_before: "
        f"tsdf_sum={receipt.tsdf_sum_before:.9f} "
        f"weight={receipt.weight_before}",
        file=sys.stdout,
    )
    for position, contribution in enumerate(receipt.contributions):
        if contribution.contributes:
            delta_sum = f"{contribution.tsdf_sum_delta:.9f}"
        else:
            delta_sum = "none"
        print(
            f"observation[{position}]: "
            f"sequence={contribution.observation_sequence} "
            f"status={contribution.status.value} "
            f"delta_sum={delta_sum} "
            f"delta_weight={contribution.weight_delta}",
            file=sys.stdout,
        )
    print(
        "status_counts: "
        + " ".join(
            f"{status.value}={count}"
            for status, count in receipt.status_counts
        ),
        file=sys.stdout,
    )
    print(
        "accumulated_delta: "
        f"tsdf_sum={receipt.tsdf_sum_delta:.9f} "
        f"weight={receipt.weight_delta}",
        file=sys.stdout,
    )
    print(
        "slot_after: "
        f"tsdf_sum={receipt.tsdf_sum_after:.9f} "
        f"weight={receipt.weight_after}",
        file=sys.stdout,
    )
    print(
        "storage_before: "
        f"nonzero_sums={storage_before[0]} "
        f"nonzero_weights={storage_before[1]} "
        f"unknown_voxels={storage_before[2]}",
        file=sys.stdout,
    )
    print(
        "storage_after: "
        f"nonzero_sums={storage_after[0]} "
        f"nonzero_weights={storage_after[1]} "
        f"unknown_voxels={storage_after[2]}",
        file=sys.stdout,
    )
    print(
        f"contributions_evaluated: {receipt.evaluated_count}",
        file=sys.stdout,
    )
    print(
        f"contributions_applied: {receipt.applied_count}",
        file=sys.stdout,
    )
    print(
        f"contributions_skipped: {receipt.skipped_count}",
        file=sys.stdout,
    )
    print("duplicate_observation_applications: 0", file=sys.stdout)
    print(
        f"storage_slots_updated: {receipt.storage_slots_updated}",
        file=sys.stdout,
    )
    print("voxel_observation_traversal_performed: yes", file=sys.stdout)
    print("voxel_address_traversal_performed: no", file=sys.stdout)
    print("fusion_block_traversal_performed: no", file=sys.stdout)
    print("ray_traversal_performed: no", file=sys.stdout)
    print("full_fusion_performed: no", file=sys.stdout)
    print("missing_blocks_created: no", file=sys.stdout)
    print("artifact_written: no", file=sys.stdout)
    print("storage_persisted: no", file=sys.stdout)
    print(
        f"plan_sha256: {receipt.source_plan_digest_sha256}",
        file=sys.stdout,
    )
    print(
        f"replay_digest_sha256: {receipt.replay_digest_sha256}",
        file=sys.stdout,
    )
    return 0


def _run_auto_tsdf(
    path: Path,
    output: Path,
    voxel_size_m: float,
    truncation_m: float,
    frame_stride: int,
) -> int:
    try:
        _validate_tsdf_output(output)
        session = load_scan_session(path)
        bounds = infer_tsdf_bounds(
            session,
            voxel_size_m=voxel_size_m,
            truncation_m=truncation_m,
            frame_stride=frame_stride,
        )
        report = integrate_tsdf(
            session,
            output,
            origin_world_m=bounds.origin_world_m,
            dimensions=bounds.dimensions_xyz,
            voxel_size_m=voxel_size_m,
            truncation_m=truncation_m,
            frame_stride=frame_stride,
            expected_replay_digest_sha256=bounds.replay_digest_sha256,
        )
    except SessionValidationError as error:
        print(f"AUTO TSDF FAILED {path}", file=sys.stderr)
        for problem in error.errors:
            print(f"- {problem}", file=sys.stderr)
        return 2
    except (TsdfError, SessionReplayError) as error:
        print(f"AUTO TSDF FAILED {path}", file=sys.stderr)
        print(f"- {error}", file=sys.stderr)
        return 2

    print(f"AUTO TSDF {report.session_id}", file=sys.stdout)
    print(
        "bounds_frames: "
        f"total={bounds.total_observations} "
        f"selected={bounds.selected_observations} "
        f"paired={bounds.paired_observations}",
        file=sys.stdout,
    )
    print(
        "bounds_depth: "
        f"valid={bounds.valid_depth_points} "
        f"invalid={bounds.invalid_depth_samples}",
        file=sys.stdout,
    )
    print(
        "bounds_skipped: "
        f"missing_depth={bounds.skipped_missing_depth} "
        f"missing_pose={bounds.skipped_missing_pose}",
        file=sys.stdout,
    )
    print(
        "surface_min_m: "
        f"{_format_triplet(bounds.surface_min_world_m)}",
        file=sys.stdout,
    )
    print(
        "surface_max_m: "
        f"{_format_triplet(bounds.surface_max_world_m)}",
        file=sys.stdout,
    )
    print(
        "volume: "
        f"origin={_format_triplet(bounds.origin_world_m)} "
        f"dimensions={bounds.dimensions_xyz} "
        f"voxels={bounds.total_voxels}",
        file=sys.stdout,
    )
    print(
        "integration: "
        f"observed={report.observed_voxels} "
        f"fused={report.fused_voxels} "
        f"updates={report.voxel_updates} "
        f"max_weight={report.max_weight}",
        file=sys.stdout,
    )
    print(f"output: {report.output}", file=sys.stdout)
    print(f"output_sha256: {report.output_digest_sha256}", file=sys.stdout)
    return 0


def _format_triplet(values: Sequence[float]) -> str:
    formatted = []
    for value in values:
        normalized = 0.0 if abs(value) < 0.5e-9 else value
        formatted.append(f"{normalized:.9f}")
    return f"({', '.join(formatted)})"


def _run_surface_points(path: Path, output: Path) -> int:
    try:
        report = extract_surface_points(path, output)
    except SurfaceExtractionError as error:
        print(f"SURFACE EXTRACTION FAILED {path}", file=sys.stderr)
        print(f"- {error}", file=sys.stderr)
        return 2

    print(f"SURFACE POINTS {report.session_id}", file=sys.stdout)
    print(
        "voxels: "
        f"total={report.total_voxels} "
        f"observed={report.observed_voxels}",
        file=sys.stdout,
    )
    print(f"observed_edges: {report.observed_edges}", file=sys.stdout)
    print(
        "crossings: "
        f"x={report.crossing_x_points} "
        f"y={report.crossing_y_points} "
        f"z={report.crossing_z_points}",
        file=sys.stdout,
    )
    crossing_points = (
        report.crossing_x_points
        + report.crossing_y_points
        + report.crossing_z_points
    )
    print(
        "points: "
        f"exact_zero={report.exact_zero_points} "
        f"crossing={crossing_points} "
        f"total={report.points_written}",
        file=sys.stdout,
    )
    print(f"output: {report.output}", file=sys.stdout)
    print(f"output_sha256: {report.output_digest_sha256}", file=sys.stdout)
    return 0


def _run_triangle_mesh(path: Path, output: Path) -> int:
    try:
        report = extract_triangle_mesh(path, output)
    except MeshExtractionError as error:
        print(f"TRIANGLE MESH FAILED {path}", file=sys.stderr)
        print(f"- {error}", file=sys.stderr)
        return 2

    print(f"TRIANGLE MESH {report.session_id}", file=sys.stdout)
    print(
        "voxels: "
        f"total={report.total_voxels} "
        f"observed={report.observed_voxels}",
        file=sys.stdout,
    )
    print(
        "cells: "
        f"total={report.total_cells} "
        f"eligible={report.eligible_cells} "
        f"active={report.active_cells}",
        file=sys.stdout,
    )
    print(
        "skipped_cells: "
        f"unknown={report.skipped_unknown_cells} "
        f"exact_zero={report.skipped_exact_zero_cells}",
        file=sys.stdout,
    )
    print(
        "mesh: "
        f"vertices={report.vertices_written} "
        f"triangles={report.triangles_written} "
        f"boundary_edges={report.boundary_edges}",
        file=sys.stdout,
    )
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
