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
from .tsdf_block_traversal import traverse_tsdf_block_voxels_from_context
from .tsdf_plan_traversal import (
    MAX_TSDF_PLAN_TRAVERSAL_OUTCOMES,
    traverse_tsdf_plan_blocks_from_context,
)
from .tsdf_observation_block_rays import (
    MAX_TSDF_OBSERVATION_BLOCK_RAY_OUTCOMES,
    trace_tsdf_observation_block_rays_from_context,
)
from .tsdf_plan_block_ray_survey import (
    MAX_TSDF_PLAN_BLOCK_RAY_SURVEY_OUTCOMES,
    survey_tsdf_plan_block_rays_from_context,
)
from .tsdf_pixel_footprint_coverage import (
    MAX_TSDF_PIXEL_FOOTPRINT_CANDIDATE_BLOCKS,
    evaluate_tsdf_pixel_footprint_coverage_from_context,
)
from .tsdf_observation_footprint import (
    MAX_TSDF_OBSERVATION_FOOTPRINT_CANDIDATE_BLOCKS,
    survey_tsdf_observation_pixel_footprints_from_context,
)
from .tsdf_plan_footprint_survey import (
    MAX_TSDF_PLAN_FOOTPRINT_CANDIDATE_BLOCKS,
    survey_tsdf_plan_pixel_footprints_from_context,
)
from .tsdf_block_cross_view import (
    MAX_TSDF_BLOCK_CROSS_VIEW_OUTCOMES,
    classify_tsdf_block_voxels_across_observations_from_context,
)
from .tsdf_domain_cross_view import (
    MAX_TSDF_COVERAGE_DOMAIN_CROSS_VIEW_OUTCOMES,
    sweep_tsdf_coverage_domain_cross_view_from_context,
)
from .tsdf_plan_expansion import (
    propose_tsdf_plan_expansion_from_domain,
)
from .tsdf_expanded_plan import write_tsdf_expanded_block_plan
from .tsdf_voxel_cross_view import (
    MAX_TSDF_VOXEL_CROSS_VIEW_OBSERVATIONS,
    classify_tsdf_voxel_across_observations_from_context,
)
from .tsdf_voxel_sampling import (
    classify_tsdf_voxel_sampling_from_context,
)
from .tsdf_bounds import infer_tsdf_bounds
from .tsdf_replay_depth_context import (
    TsdfReplayDepthStatus,
    build_tsdf_replay_depth_context,
)
from .tsdf_voxel_address import (
    TsdfVoxelAddress,
    compose_tsdf_global_voxel_index,
    locate_tsdf_voxel,
)
from .tsdf_voxel_contribution import (
    TsdfVoxelContribution,
    evaluate_tsdf_voxel_contribution,
    evaluate_tsdf_voxel_contribution_from_context,
)
from .tsdf_voxel_traversal import (
    traverse_tsdf_voxel_observations,
    traverse_tsdf_voxel_observations_from_context,
)
from .tsdf_voxel_update import (
    apply_tsdf_voxel_contribution,
    apply_tsdf_voxel_contribution_from_context,
)
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
            == "tsdf-block-context-contribution"
        ):
            return _run_tsdf_block_context_contribution(
                arguments.plan,
                arguments.session,
                arguments.observation_sequence,
                arguments.voxel,
            )
        if (
            arguments.reconstruct_command
            == "tsdf-block-context-contribution-apply"
        ):
            return _run_tsdf_block_context_contribution_apply(
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
        if (
            arguments.reconstruct_command
            == "tsdf-block-context-voxel-traverse"
        ):
            return _run_tsdf_block_context_voxel_traverse(
                arguments.plan,
                arguments.session,
                arguments.voxel,
            )
        if (
            arguments.reconstruct_command
            == "tsdf-block-context-block-traverse"
        ):
            return _run_tsdf_block_context_block_traverse(
                arguments.plan,
                arguments.session,
                arguments.block,
            )
        if (
            arguments.reconstruct_command
            == "tsdf-block-context-plan-traverse"
        ):
            return _run_tsdf_block_context_plan_traverse(
                arguments.plan,
                arguments.session,
            )
        if (
            arguments.reconstruct_command
            == "tsdf-block-context-observation-rays"
        ):
            return _run_tsdf_block_context_observation_rays(
                arguments.plan,
                arguments.session,
                arguments.observation_sequence,
            )
        if (
            arguments.reconstruct_command
            == "tsdf-block-context-plan-rays"
        ):
            return _run_tsdf_block_context_plan_rays(
                arguments.plan,
                arguments.session,
            )
        if (
            arguments.reconstruct_command
            == "tsdf-block-context-pixel-footprint"
        ):
            return _run_tsdf_block_context_pixel_footprint(
                arguments.plan,
                arguments.session,
                arguments.observation_sequence,
                arguments.pixel,
            )
        if (
            arguments.reconstruct_command
            == "tsdf-block-context-observation-footprint"
        ):
            return _run_tsdf_block_context_observation_footprint(
                arguments.plan,
                arguments.session,
                arguments.observation_sequence,
            )
        if (
            arguments.reconstruct_command
            == "tsdf-block-context-plan-footprint"
        ):
            return _run_tsdf_block_context_plan_footprint(
                arguments.plan,
                arguments.session,
            )
        if (
            arguments.reconstruct_command
            == "tsdf-block-context-voxel-sampling"
        ):
            return _run_tsdf_block_context_voxel_sampling(
                arguments.plan,
                arguments.session,
                arguments.observation_sequence,
                arguments.voxel,
            )
        if (
            arguments.reconstruct_command
            == "tsdf-block-context-voxel-cross-view"
        ):
            return _run_tsdf_block_context_voxel_cross_view(
                arguments.plan,
                arguments.session,
                arguments.voxel,
            )
        if (
            arguments.reconstruct_command
            == "tsdf-block-context-block-cross-view"
        ):
            return _run_tsdf_block_context_block_cross_view(
                arguments.plan,
                arguments.session,
                arguments.block,
            )
        if (
            arguments.reconstruct_command
            == "tsdf-block-context-domain-cross-view"
        ):
            return _run_tsdf_block_context_domain_cross_view(
                arguments.plan,
                arguments.session,
            )
        if (
            arguments.reconstruct_command
            == "tsdf-block-context-plan-expansion"
        ):
            return _run_tsdf_block_context_plan_expansion(
                arguments.plan,
                arguments.session,
            )
        if arguments.reconstruct_command == "tsdf-block-plan-expand":
            return _run_tsdf_block_plan_expand(
                arguments.plan,
                arguments.session,
                arguments.output,
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
        if arguments.reconstruct_command == "triangle-mesh":
            return _run_triangle_mesh(arguments.path, arguments.output)
        raise AssertionError(
            "unrouted reconstruct command "
            f"{arguments.reconstruct_command!r}"
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

    block_context_contribution = reconstruct_commands.add_parser(
        "tsdf-block-context-contribution",
        help=(
            "Evaluate one planned voxel from one prepared replay/depth "
            "observation without evaluation-time source I/O."
        ),
        description=(
            "Strictly load a replay-matched block plan, allocate temporary "
            "storage, prepare the selected-observation replay/depth context, "
            "then evaluate exactly one selected observation at one planned "
            "voxel from that context. Evaluation does not replay the "
            "session, decode depth, apply a contribution, or write an "
            "artifact."
        ),
    )
    block_context_contribution.add_argument(
        "plan",
        type=Path,
        help="Existing .sftplan artifact whose grid parameters are used.",
    )
    block_context_contribution.add_argument(
        "session",
        type=Path,
        help="Current .vgsession directory used to prepare the context.",
    )
    block_context_contribution.add_argument(
        "--observation-sequence",
        type=int,
        default=0,
        help="Zero-based plan-selected observation sequence (default: 0).",
    )
    block_context_contribution.add_argument(
        "--voxel",
        type=int,
        nargs=3,
        required=True,
        metavar=("GX", "GY", "GZ"),
        help="One planned signed global voxel XYZ to evaluate.",
    )

    block_context_contribution_apply = reconstruct_commands.add_parser(
        "tsdf-block-context-contribution-apply",
        help=(
            "Evaluate from a prepared replay/depth context and apply one "
            "accepted contribution to one temporary voxel slot."
        ),
        description=(
            "Strictly load a replay-matched block plan, allocate temporary "
            "storage, prepare the replay/depth context, evaluate exactly "
            "one selected observation at one planned voxel, and apply that "
            "accepted contribution to exactly one slot using construction-"
            "time context provenance. The scalar evaluation and application "
            "do not replay the session, decode depth, traverse, or write an "
            "artifact."
        ),
    )
    block_context_contribution_apply.add_argument(
        "plan",
        type=Path,
        help="Existing .sftplan artifact whose temporary storage is updated.",
    )
    block_context_contribution_apply.add_argument(
        "session",
        type=Path,
        help="Current .vgsession directory used to prepare the context.",
    )
    block_context_contribution_apply.add_argument(
        "--observation-sequence",
        type=int,
        default=0,
        help="Zero-based plan-selected observation sequence (default: 0).",
    )
    block_context_contribution_apply.add_argument(
        "--voxel",
        type=int,
        nargs=3,
        required=True,
        metavar=("GX", "GY", "GZ"),
        help="One planned signed global voxel XYZ to evaluate and update.",
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

    block_context_voxel_traverse = reconstruct_commands.add_parser(
        "tsdf-block-context-voxel-traverse",
        help=(
            "Use one prepared replay/depth context to evaluate every "
            "selected observation for one temporary planned voxel."
        ),
        description=(
            "Strictly load a replay-matched block plan, allocate temporary "
            "storage, prepare one immutable replay/depth context, evaluate "
            "every selected context observation for one planned voxel, and "
            "apply accepted contributions in canonical sequence order. "
            "The traversal performs no session replay, source I/O, or depth "
            "decoding, and no artifact is written."
        ),
    )
    block_context_voxel_traverse.add_argument(
        "plan",
        type=Path,
        help="Existing .sftplan artifact whose temporary storage is updated.",
    )
    block_context_voxel_traverse.add_argument(
        "session",
        type=Path,
        help="Current .vgsession directory used to prepare the context.",
    )
    block_context_voxel_traverse.add_argument(
        "--voxel",
        type=int,
        nargs=3,
        required=True,
        metavar=("GX", "GY", "GZ"),
        help="One planned signed global voxel XYZ to traverse and update.",
    )

    block_context_block_traverse = reconstruct_commands.add_parser(
        "tsdf-block-context-block-traverse",
        help=(
            "Use one prepared replay/depth context to traverse all 512 "
            "voxels in one selected planned block."
        ),
        description=(
            "Strictly load a replay-matched block plan, allocate temporary "
            "storage, prepare one immutable replay/depth context, and "
            "traverse every voxel in exactly one selected planned block in "
            "canonical local-flat order. No second block, ray, or free-space "
            "coverage is traversed, and no artifact is written."
        ),
    )
    block_context_block_traverse.add_argument(
        "plan",
        type=Path,
        help="Existing .sftplan artifact whose temporary storage is updated.",
    )
    block_context_block_traverse.add_argument(
        "session",
        type=Path,
        help="Current .vgsession directory used to prepare the context.",
    )
    block_context_block_traverse.add_argument(
        "--block",
        type=int,
        nargs=3,
        required=True,
        metavar=("BX", "BY", "BZ"),
        help="Exactly one planned signed block XYZ to traverse.",
    )

    block_context_plan_traverse = reconstruct_commands.add_parser(
        "tsdf-block-context-plan-traverse",
        help=(
            "Use one prepared replay/depth context to traverse every "
            "existing planned block row."
        ),
        description=(
            "Strictly load a replay-matched block plan, allocate temporary "
            "storage, prepare one immutable replay/depth context, and "
            "traverse every already-existing plan row in canonical block "
            "and local-flat order. No missing block, ray, or free-space "
            "coverage is added, and no artifact is written."
        ),
    )
    block_context_plan_traverse.add_argument(
        "plan",
        type=Path,
        help="Existing .sftplan artifact whose complete rows are traversed.",
    )
    block_context_plan_traverse.add_argument(
        "session",
        type=Path,
        help="Current .vgsession directory used to prepare the context.",
    )

    block_context_observation_rays = reconstruct_commands.add_parser(
        "tsdf-block-context-observation-rays",
        help=(
            "Trace one prepared observation's pixel-center block rays."
        ),
        description=(
            "Strictly load a replay-matched block plan, prepare one "
            "immutable replay/depth context, and trace closed camera-to-"
            "measured-surface centerline segments for exactly one selected "
            "observation. This diagnostic reports existing and unplanned "
            "block coordinates; it does not expand the plan, allocate TSDF "
            "storage, fuse voxels, or write an artifact."
        ),
    )
    block_context_observation_rays.add_argument(
        "plan",
        type=Path,
        help="Existing .sftplan artifact used only for geometry/provenance.",
    )
    block_context_observation_rays.add_argument(
        "session",
        type=Path,
        help="Current .vgsession directory used to prepare the context.",
    )
    block_context_observation_rays.add_argument(
        "--observation-sequence",
        type=int,
        required=True,
        help="Exactly one nonnegative plan-selected observation sequence.",
    )

    block_context_plan_rays = reconstruct_commands.add_parser(
        "tsdf-block-context-plan-rays",
        help=(
            "Aggregate every selected observation's pixel-center block rays."
        ),
        description=(
            "Strictly load a replay-matched block plan, prepare one "
            "immutable replay/depth context, and trace the closed camera-to-"
            "measured-surface centerline segments of every plan-selected "
            "observation in canonical order. This diagnostic reports the "
            "combined existing and unplanned coverage with per-block "
            "observation support; it does not expand the plan, allocate "
            "TSDF storage, fuse voxels, or write an artifact."
        ),
    )
    block_context_plan_rays.add_argument(
        "plan",
        type=Path,
        help="Existing .sftplan artifact used only for geometry/provenance.",
    )
    block_context_plan_rays.add_argument(
        "session",
        type=Path,
        help="Current .vgsession directory used to prepare the context.",
    )

    block_context_pixel_footprint = reconstruct_commands.add_parser(
        "tsdf-block-context-pixel-footprint",
        help=(
            "Cover one prepared pixel's conservative nearest-pixel wedge."
        ),
        description=(
            "Strictly load a replay-matched block plan, prepare one "
            "immutable replay/depth context, and conservatively cover the "
            "nearest-pixel sampling wedge of exactly one pixel of one "
            "selected observation. This diagnostic reports a superset of "
            "the blocks that could contain voxel centres sampling that "
            "measurement; it does not expand the plan, allocate TSDF "
            "storage, fuse voxels, or write an artifact."
        ),
    )
    block_context_pixel_footprint.add_argument(
        "plan",
        type=Path,
        help="Existing .sftplan artifact used only for geometry/provenance.",
    )
    block_context_pixel_footprint.add_argument(
        "session",
        type=Path,
        help="Current .vgsession directory used to prepare the context.",
    )
    block_context_pixel_footprint.add_argument(
        "--observation-sequence",
        type=int,
        required=True,
        help="Exactly one nonnegative plan-selected observation sequence.",
    )
    block_context_pixel_footprint.add_argument(
        "--pixel",
        type=int,
        nargs=2,
        required=True,
        metavar=("U", "V"),
        help="Exactly one nonnegative image pixel column and row.",
    )

    block_context_observation_footprint = reconstruct_commands.add_parser(
        "tsdf-block-context-observation-footprint",
        help=(
            "Cover every pixel's conservative wedge for one observation."
        ),
        description=(
            "Strictly load a replay-matched block plan, prepare one "
            "immutable replay/depth context, and conservatively cover the "
            "nearest-pixel sampling wedge of every pixel of exactly one "
            "selected observation. The union is partitioned into existing "
            "and unplanned coordinates and is a superset of that "
            "observation's centreline ray coverage; it does not expand the "
            "plan, allocate TSDF storage, fuse voxels, or write an artifact."
        ),
    )
    block_context_observation_footprint.add_argument(
        "plan",
        type=Path,
        help="Existing .sftplan artifact used only for geometry/provenance.",
    )
    block_context_observation_footprint.add_argument(
        "session",
        type=Path,
        help="Current .vgsession directory used to prepare the context.",
    )
    block_context_observation_footprint.add_argument(
        "--observation-sequence",
        type=int,
        required=True,
        help="Exactly one nonnegative plan-selected observation sequence.",
    )

    block_context_plan_footprint = reconstruct_commands.add_parser(
        "tsdf-block-context-plan-footprint",
        help=(
            "Cover every selected observation's conservative pixel wedges."
        ),
        description=(
            "Strictly load a replay-matched block plan, prepare one "
            "immutable replay/depth context, and conservatively cover the "
            "nearest-pixel sampling wedges of every pixel of every "
            "plan-selected observation. The union is partitioned into "
            "existing and unplanned coordinates and contains the "
            "corresponding centreline ray coverage; it does not approve "
            "coverage, expand the plan, allocate TSDF storage, fuse voxels, "
            "or write an artifact."
        ),
    )
    block_context_plan_footprint.add_argument(
        "plan",
        type=Path,
        help="Existing .sftplan artifact used only for geometry/provenance.",
    )
    block_context_plan_footprint.add_argument(
        "session",
        type=Path,
        help="Current .vgsession directory used to prepare the context.",
    )

    block_context_voxel_sampling = reconstruct_commands.add_parser(
        "tsdf-block-context-voxel-sampling",
        help=(
            "Classify how one observation samples one voxel centre."
        ),
        description=(
            "Strictly load a replay-matched block plan, prepare one "
            "immutable replay/depth context, and classify exactly one "
            "signed global voxel centre against exactly one selected "
            "observation. This separates observed free space from the "
            "surface band and from occluded space, and accepts unplanned "
            "voxels; it does not expand the plan, allocate TSDF storage, "
            "fuse voxels, or write an artifact."
        ),
    )
    block_context_voxel_sampling.add_argument(
        "plan",
        type=Path,
        help="Existing .sftplan artifact used only for geometry/provenance.",
    )
    block_context_voxel_sampling.add_argument(
        "session",
        type=Path,
        help="Current .vgsession directory used to prepare the context.",
    )
    block_context_voxel_sampling.add_argument(
        "--observation-sequence",
        type=int,
        required=True,
        help="Exactly one nonnegative plan-selected observation sequence.",
    )
    block_context_voxel_sampling.add_argument(
        "--voxel",
        type=int,
        nargs=3,
        required=True,
        metavar=("X", "Y", "Z"),
        help="Exactly one signed global voxel XYZ, planned or not.",
    )

    block_context_voxel_cross_view = reconstruct_commands.add_parser(
        "tsdf-block-context-voxel-cross-view",
        help=(
            "Combine every selected observation's verdict for one voxel."
        ),
        description=(
            "Strictly load a replay-matched block plan, prepare one "
            "immutable replay/depth context, and combine the per-observation "
            "sampling verdicts of every plan-selected observation for "
            "exactly one signed global voxel centre. Surface-band evidence "
            "outranks free space, which outranks occlusion; occlusion and "
            "missing input never become free space. This does not carve free "
            "space, expand the plan, allocate TSDF storage, fuse voxels, or "
            "write an artifact."
        ),
    )
    block_context_voxel_cross_view.add_argument(
        "plan",
        type=Path,
        help="Existing .sftplan artifact used only for geometry/provenance.",
    )
    block_context_voxel_cross_view.add_argument(
        "session",
        type=Path,
        help="Current .vgsession directory used to prepare the context.",
    )
    block_context_voxel_cross_view.add_argument(
        "--voxel",
        type=int,
        nargs=3,
        required=True,
        metavar=("X", "Y", "Z"),
        help="Exactly one signed global voxel XYZ, planned or not.",
    )

    block_context_block_cross_view = reconstruct_commands.add_parser(
        "tsdf-block-context-block-cross-view",
        help=(
            "Resolve every voxel of one block across all observations."
        ),
        description=(
            "Strictly load a replay-matched block plan, prepare one "
            "immutable replay/depth context, and combine every selected "
            "observation's verdict for all 512 voxels of exactly one signed "
            "block, planned or not. This reports the block's carvable "
            "free-space voxels alongside surface, occluded and unseen "
            "counts; it does not carve, expand the plan, allocate TSDF "
            "storage, fuse voxels, or write an artifact."
        ),
    )
    block_context_block_cross_view.add_argument(
        "plan",
        type=Path,
        help="Existing .sftplan artifact used only for geometry/provenance.",
    )
    block_context_block_cross_view.add_argument(
        "session",
        type=Path,
        help="Current .vgsession directory used to prepare the context.",
    )
    block_context_block_cross_view.add_argument(
        "--block",
        type=int,
        nargs=3,
        required=True,
        metavar=("BX", "BY", "BZ"),
        help="Exactly one signed block XYZ, planned or not.",
    )

    block_context_domain_cross_view = reconstruct_commands.add_parser(
        "tsdf-block-context-domain-cross-view",
        help=(
            "Resolve every voxel of the surveyed coverage domain."
        ),
        description=(
            "Strictly load a replay-matched block plan, prepare one "
            "immutable replay/depth context, survey the conservative "
            "footprint coverage of every selected observation, and resolve "
            "every voxel of that covered domain across all observations. "
            "This reports the whole-scan carvable free-space set and how "
            "much of it falls in blocks the plan does not have; it does not "
            "carve, expand the plan, allocate TSDF storage, fuse voxels, or "
            "write an artifact."
        ),
    )
    block_context_domain_cross_view.add_argument(
        "plan",
        type=Path,
        help="Existing .sftplan artifact used only for geometry/provenance.",
    )
    block_context_domain_cross_view.add_argument(
        "session",
        type=Path,
        help="Current .vgsession directory used to prepare the context.",
    )

    block_context_plan_expansion = reconstruct_commands.add_parser(
        "tsdf-block-context-plan-expansion",
        help=(
            "Propose the blocks an expanded plan would hold."
        ),
        description=(
            "Strictly load a replay-matched block plan, prepare one "
            "immutable replay/depth context, survey the conservative "
            "footprint coverage, resolve every covered voxel, and propose "
            "the canonical expanded block set: the source plan plus every "
            "covered block that carries at least one observed voxel. This "
            "is a read-only proposal; it does not write a plan, allocate "
            "TSDF storage, fuse voxels, or write any artifact."
        ),
    )
    block_context_plan_expansion.add_argument(
        "plan",
        type=Path,
        help="Existing .sftplan artifact to propose an expansion for.",
    )
    block_context_plan_expansion.add_argument(
        "session",
        type=Path,
        help="Current .vgsession directory used to prepare the context.",
    )

    block_plan_expand = reconstruct_commands.add_parser(
        "tsdf-block-plan-expand",
        help="Write the approved expanded block set as a new .sftplan.",
        description=(
            "Strictly load a replay-matched block plan, resolve its "
            "conservative coverage domain, approve the evidence-bearing "
            "blocks, and write the merged block set as a new .sftplan. The "
            "source plan is never modified and the output must not already "
            "exist. The written plan records that free space came from "
            "conservative nearest-pixel footprint coverage, together with "
            "the source plan digest and the approval rule."
        ),
    )
    block_plan_expand.add_argument(
        "plan",
        type=Path,
        help="Existing .sftplan artifact to expand. Never modified.",
    )
    block_plan_expand.add_argument(
        "session",
        type=Path,
        help="Current .vgsession directory used to prepare the context.",
    )
    block_plan_expand.add_argument(
        "output",
        type=Path,
        help="New .sftplan path to write. Must not already exist.",
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
    _print_tsdf_voxel_contribution(contribution, address)
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


def _run_tsdf_block_context_contribution(
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
        context = build_tsdf_replay_depth_context(plan, session)
        before = (
            storage.block_indices,
            id(storage.tsdf_sums),
            id(storage.weights),
            storage.nonzero_sum_count,
            storage.nonzero_weight_count,
            storage.payload_bytes,
        )
        contribution = evaluate_tsdf_voxel_contribution_from_context(
            storage,
            address,
            context,
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
                "TSDF context contribution evaluation mutated block storage"
            )
    except SessionValidationError as error:
        print(
            f"TSDF BLOCK CONTEXT CONTRIBUTION FAILED {plan_path}",
            file=sys.stderr,
        )
        for problem in error.errors:
            print(f"- {problem}", file=sys.stderr)
        return 2
    except (TsdfError, SessionReplayError) as error:
        print(
            f"TSDF BLOCK CONTEXT CONTRIBUTION FAILED {plan_path}",
            file=sys.stderr,
        )
        print(f"- {error}", file=sys.stderr)
        return 2

    print(
        f"TSDF BLOCK CONTEXT CONTRIBUTION CHECK {plan.session_id}",
        file=sys.stdout,
    )
    print("artifact: valid", file=sys.stdout)
    print("session_replay: matched", file=sys.stdout)
    print(
        "context_selection: "
        f"frame_stride={context.frame_stride} "
        f"total={context.total_observations} "
        f"selected={context.selected_observation_count}",
        file=sys.stdout,
    )
    print("context_immutable: yes", file=sys.stdout)
    print("depth_source: replay-depth-context", file=sys.stdout)
    _print_tsdf_voxel_contribution(contribution, address)
    print("evaluation_replay_hashing: no", file=sys.stdout)
    print("evaluation_depth_decoding: no", file=sys.stdout)
    print("contributions_applied: 0", file=sys.stdout)
    print("storage_mutated: no", file=sys.stdout)
    print("voxel_observation_traversal_performed: no", file=sys.stdout)
    print("voxel_address_traversal_performed: no", file=sys.stdout)
    print("fusion_block_traversal_performed: no", file=sys.stdout)
    print("full_fusion_performed: no", file=sys.stdout)
    print("missing_blocks_created: no", file=sys.stdout)
    print("artifact_written: no", file=sys.stdout)
    print("context_persisted: no", file=sys.stdout)
    print(
        f"plan_sha256: {plan.artifact_digest_sha256}",
        file=sys.stdout,
    )
    print(
        f"replay_digest_sha256: {plan.replay_digest_sha256}",
        file=sys.stdout,
    )
    return 0


def _print_tsdf_voxel_contribution(
    contribution: TsdfVoxelContribution,
    address: TsdfVoxelAddress,
) -> None:
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


def _run_tsdf_block_context_contribution_apply(
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
        context = build_tsdf_replay_depth_context(plan, session)
        contribution = evaluate_tsdf_voxel_contribution_from_context(
            storage,
            address,
            context,
            observation_sequence,
        )
        storage_before = (
            storage.nonzero_sum_count,
            storage.nonzero_weight_count,
            storage.unknown_voxel_count,
        )
        receipt = apply_tsdf_voxel_contribution_from_context(
            storage,
            contribution,
            context,
        )
        storage_after = (
            storage.nonzero_sum_count,
            storage.nonzero_weight_count,
            storage.unknown_voxel_count,
        )
    except SessionValidationError as error:
        print(
            f"TSDF BLOCK CONTEXT CONTRIBUTION APPLY FAILED {plan_path}",
            file=sys.stderr,
        )
        for problem in error.errors:
            print(f"- {problem}", file=sys.stderr)
        return 2
    except (TsdfError, SessionReplayError) as error:
        print(
            f"TSDF BLOCK CONTEXT CONTRIBUTION APPLY FAILED {plan_path}",
            file=sys.stderr,
        )
        print(f"- {error}", file=sys.stderr)
        return 2

    print(
        "TSDF BLOCK CONTEXT CONTRIBUTION APPLY CHECK "
        f"{plan.session_id}",
        file=sys.stdout,
    )
    print("artifact: valid", file=sys.stdout)
    print("session_replay: matched", file=sys.stdout)
    print(
        "context_selection: "
        f"frame_stride={context.frame_stride} "
        f"total={context.total_observations} "
        f"selected={context.selected_observation_count}",
        file=sys.stdout,
    )
    print("context_immutable: yes", file=sys.stdout)
    print("depth_source: replay-depth-context", file=sys.stdout)
    _print_tsdf_voxel_contribution(contribution, address)
    print("evaluation_replay_hashing: no", file=sys.stdout)
    print("evaluation_depth_decoding: no", file=sys.stdout)
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
    print("context_provenance: matched", file=sys.stdout)
    print(
        "application_source_freshness: construction-time-context",
        file=sys.stdout,
    )
    print("application_session_replay: no", file=sys.stdout)
    print("application_replay_hashing: no", file=sys.stdout)
    print("application_source_io: no", file=sys.stdout)
    print("application_depth_access: no", file=sys.stdout)
    print("contributions_evaluated: 1", file=sys.stdout)
    print("contributions_applied: 1", file=sys.stdout)
    print("storage_slots_updated: 1", file=sys.stdout)
    print("voxel_observation_traversal_performed: no", file=sys.stdout)
    print("voxel_address_traversal_performed: no", file=sys.stdout)
    print("fusion_block_traversal_performed: no", file=sys.stdout)
    print("ray_traversal_performed: no", file=sys.stdout)
    print("full_fusion_performed: no", file=sys.stdout)
    print("missing_blocks_created: no", file=sys.stdout)
    print("artifact_written: no", file=sys.stdout)
    print("storage_persisted: no", file=sys.stdout)
    print("context_persisted: no", file=sys.stdout)
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


def _run_tsdf_block_context_block_traverse(
    plan_path: Path,
    session_path: Path,
    block: list[int],
) -> int:
    try:
        plan = load_tsdf_block_plan(plan_path)
        session = load_scan_session(session_path)
        storage = allocate_empty_tsdf_blocks(plan, session)
        block_index_xyz = tuple(block)
        first_global_index = compose_tsdf_global_voxel_index(
            block_index_xyz,
            (0, 0, 0),
        )
        first_address = locate_tsdf_voxel(storage, first_global_index)
        if first_address is None:
            raise TsdfError(
                f"TSDF block {block_index_xyz} is not planned in "
                "destination storage"
            )
        block_row = first_address.block_row
        block_before = (
            int((storage.tsdf_sums[block_row] != 0.0).sum()),
            int((storage.weights[block_row] != 0).sum()),
            int((storage.weights[block_row] == 0).sum()),
        )
        storage_before = (
            storage.nonzero_sum_count,
            storage.nonzero_weight_count,
            storage.unknown_voxel_count,
        )
        context = build_tsdf_replay_depth_context(plan, session)
        receipt = traverse_tsdf_block_voxels_from_context(
            storage,
            block_index_xyz,
            context,
        )
        block_after = (
            int((storage.tsdf_sums[block_row] != 0.0).sum()),
            int((storage.weights[block_row] != 0).sum()),
            int((storage.weights[block_row] == 0).sum()),
        )
        storage_after = (
            storage.nonzero_sum_count,
            storage.nonzero_weight_count,
            storage.unknown_voxel_count,
        )
    except SessionValidationError as error:
        print(
            f"TSDF BLOCK CONTEXT BLOCK TRAVERSAL FAILED {plan_path}",
            file=sys.stderr,
        )
        for problem in error.errors:
            print(f"- {problem}", file=sys.stderr)
        return 2
    except (TsdfError, SessionReplayError) as error:
        print(
            f"TSDF BLOCK CONTEXT BLOCK TRAVERSAL FAILED {plan_path}",
            file=sys.stderr,
        )
        print(f"- {error}", file=sys.stderr)
        return 2

    first_voxel = receipt.voxel_receipts[0].address
    last_voxel = receipt.voxel_receipts[-1].address
    print(
        f"TSDF BLOCK CONTEXT BLOCK TRAVERSAL CHECK {plan.session_id}",
        file=sys.stdout,
    )
    print("artifact: valid", file=sys.stdout)
    print("session_replay: matched", file=sys.stdout)
    print(
        "context_selection: "
        f"frame_stride={context.frame_stride} "
        f"total={context.total_observations} "
        f"selected={context.selected_observation_count}",
        file=sys.stdout,
    )
    print("context_immutable: yes", file=sys.stdout)
    print("depth_source: replay-depth-context", file=sys.stdout)
    print(
        "block: "
        f"index={receipt.block_index_xyz} "
        f"row={receipt.block_row} "
        f"resolution={receipt.block_resolution} "
        f"voxel_slots={receipt.voxel_count}",
        file=sys.stdout,
    )
    print(
        "storage_flat_range: "
        f"{first_voxel.storage_flat_index}.."
        f"{last_voxel.storage_flat_index}",
        file=sys.stdout,
    )
    print(
        "address_order: local-flat-x-fastest "
        f"local_flat=0..{receipt.voxel_count - 1}",
        file=sys.stdout,
    )
    print(
        "first_voxel: "
        f"global={first_voxel.global_index_xyz} "
        f"local={first_voxel.local_index_xyz} "
        f"array={first_voxel.array_index_bzyx} "
        f"storage_flat={first_voxel.storage_flat_index}",
        file=sys.stdout,
    )
    print(
        "last_voxel: "
        f"global={last_voxel.global_index_xyz} "
        f"local={last_voxel.local_index_xyz} "
        f"array={last_voxel.array_index_bzyx} "
        f"storage_flat={last_voxel.storage_flat_index}",
        file=sys.stdout,
    )
    print(
        "selection: "
        f"frame_stride={receipt.frame_stride} "
        f"total={receipt.total_observations} "
        f"selected={len(receipt.selected_observation_sequences)}",
        file=sys.stdout,
    )
    print(
        "block_before: "
        f"nonzero_sums={block_before[0]} "
        f"nonzero_weights={block_before[1]} "
        f"unknown_voxels={block_before[2]}",
        file=sys.stdout,
    )
    print(
        "block_after: "
        f"nonzero_sums={block_after[0]} "
        f"nonzero_weights={block_after[1]} "
        f"unknown_voxels={block_after[2]}",
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
        "status_counts: "
        + " ".join(
            f"{status.value}={count}"
            for status, count in receipt.status_counts
        ),
        file=sys.stdout,
    )
    print(
        f"block_weight_sum_after: {receipt.weight_delta}",
        file=sys.stdout,
    )
    print(
        f"block_max_weight_after: {receipt.maximum_weight_after}",
        file=sys.stdout,
    )
    print("context_provenance: matched", file=sys.stdout)
    print(
        "traversal_source_freshness: construction-time-context",
        file=sys.stdout,
    )
    print("traversal_session_replay: no", file=sys.stdout)
    print("traversal_replay_hashing: no", file=sys.stdout)
    print("traversal_source_io: no", file=sys.stdout)
    print("traversal_depth_decoding: no", file=sys.stdout)
    print(
        "traversal_prepared_depth_access: "
        f"{'yes' if receipt.prepared_depth_accessed else 'no'}",
        file=sys.stdout,
    )
    print(
        f"voxel_addresses_traversed: {receipt.voxel_address_count}",
        file=sys.stdout,
    )
    print(
        f"voxel_transcripts_retained: {len(receipt.voxel_receipts)}",
        file=sys.stdout,
    )
    print(
        "voxel_observation_traversals: "
        f"{receipt.voxel_observation_traversal_count}",
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
    print(
        f"storage_slots_updated: {receipt.storage_slots_updated}",
        file=sys.stdout,
    )
    print("blocks_traversed: 1", file=sys.stdout)
    print("additional_blocks_visited: 0", file=sys.stdout)
    print("voxel_observation_traversal_performed: yes", file=sys.stdout)
    print("voxel_address_traversal_performed: yes", file=sys.stdout)
    print("selected_block_traversal_performed: yes", file=sys.stdout)
    print("fusion_block_traversal_performed: yes", file=sys.stdout)
    print(
        "fusion_block_traversal_scope: selected-planned-block-only",
        file=sys.stdout,
    )
    print("multiple_block_traversal_performed: no", file=sys.stdout)
    print("planned_block_set_traversal_performed: no", file=sys.stdout)
    print("free_space_coverage_planned: no", file=sys.stdout)
    print("ray_traversal_performed: no", file=sys.stdout)
    print("full_fusion_performed: no", file=sys.stdout)
    print("missing_blocks_created: no", file=sys.stdout)
    print("caught_failure_rollback_scope: selected-block", file=sys.stdout)
    print("artifact_written: no", file=sys.stdout)
    print("storage_persisted: no", file=sys.stdout)
    print("context_persisted: no", file=sys.stdout)
    print(
        f"plan_sha256: {receipt.source_plan_digest_sha256}",
        file=sys.stdout,
    )
    print(
        f"replay_digest_sha256: {receipt.replay_digest_sha256}",
        file=sys.stdout,
    )
    return 0


def _run_tsdf_block_context_plan_traverse(
    plan_path: Path,
    session_path: Path,
) -> int:
    try:
        plan = load_tsdf_block_plan(plan_path)
        session = load_scan_session(session_path)
        storage = allocate_empty_tsdf_blocks(plan, session)
        storage_before = (
            storage.nonzero_sum_count,
            storage.nonzero_weight_count,
            storage.unknown_voxel_count,
        )
        context = build_tsdf_replay_depth_context(plan, session)
        receipt = traverse_tsdf_plan_blocks_from_context(storage, context)
        storage_after = (
            storage.nonzero_sum_count,
            storage.nonzero_weight_count,
            storage.unknown_voxel_count,
        )
    except SessionValidationError as error:
        print(
            f"TSDF BLOCK CONTEXT PLAN TRAVERSAL FAILED {plan_path}",
            file=sys.stderr,
        )
        for problem in error.errors:
            print(f"- {problem}", file=sys.stderr)
        return 2
    except (TsdfError, SessionReplayError) as error:
        print(
            f"TSDF BLOCK CONTEXT PLAN TRAVERSAL FAILED {plan_path}",
            file=sys.stderr,
        )
        print(f"- {error}", file=sys.stderr)
        return 2

    first_block = receipt.block_receipts[0]
    last_block = receipt.block_receipts[-1]
    first_voxel = first_block.voxel_receipts[0].address
    last_voxel = last_block.voxel_receipts[-1].address
    print(
        f"TSDF BLOCK CONTEXT PLAN TRAVERSAL CHECK {plan.session_id}",
        file=sys.stdout,
    )
    print("artifact: valid", file=sys.stdout)
    print("session_replay: matched", file=sys.stdout)
    print(
        "context_selection: "
        f"frame_stride={context.frame_stride} "
        f"total={context.total_observations} "
        f"selected={context.selected_observation_count}",
        file=sys.stdout,
    )
    print("context_immutable: yes", file=sys.stdout)
    print("depth_source: replay-depth-context", file=sys.stdout)
    print(
        "plan_blocks: "
        f"active={plan.active_block_count} "
        f"surface={plan.surface_block_count} "
        f"halo={plan.halo_block_count} "
        f"resolution={receipt.block_resolution} "
        f"voxel_slots={receipt.voxel_count}",
        file=sys.stdout,
    )
    print(
        "block_order: plan-canonical-x-fastest "
        f"rows=0..{receipt.block_count - 1}",
        file=sys.stdout,
    )
    print(
        "voxel_order: block-row-then-local-flat-x-fastest "
        "local_flat=0..511",
        file=sys.stdout,
    )
    print(
        "first_block: "
        f"index={first_block.block_index_xyz} "
        f"row={first_block.block_row} "
        f"storage_flat_range={first_voxel.storage_flat_index}.."
        f"{first_block.voxel_receipts[-1].address.storage_flat_index}",
        file=sys.stdout,
    )
    print(
        "last_block: "
        f"index={last_block.block_index_xyz} "
        f"row={last_block.block_row} "
        f"storage_flat_range="
        f"{last_block.voxel_receipts[0].address.storage_flat_index}.."
        f"{last_voxel.storage_flat_index}",
        file=sys.stdout,
    )
    print(
        "first_voxel: "
        f"global={first_voxel.global_index_xyz} "
        f"local={first_voxel.local_index_xyz} "
        f"array={first_voxel.array_index_bzyx} "
        f"storage_flat={first_voxel.storage_flat_index}",
        file=sys.stdout,
    )
    print(
        "last_voxel: "
        f"global={last_voxel.global_index_xyz} "
        f"local={last_voxel.local_index_xyz} "
        f"array={last_voxel.array_index_bzyx} "
        f"storage_flat={last_voxel.storage_flat_index}",
        file=sys.stdout,
    )
    print(
        "selection: "
        f"frame_stride={receipt.frame_stride} "
        f"total={receipt.total_observations} "
        f"selected={len(receipt.selected_observation_sequences)}",
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
        "status_counts: "
        + " ".join(
            f"{status.value}={count}"
            for status, count in receipt.status_counts
        ),
        file=sys.stdout,
    )
    print(
        f"plan_weight_sum_after: {receipt.weight_delta}",
        file=sys.stdout,
    )
    print(
        f"plan_max_weight_after: {receipt.maximum_weight_after}",
        file=sys.stdout,
    )
    print("context_provenance: matched", file=sys.stdout)
    print(
        "traversal_source_freshness: construction-time-context",
        file=sys.stdout,
    )
    print("traversal_session_replay: no", file=sys.stdout)
    print("traversal_replay_hashing: no", file=sys.stdout)
    print("traversal_source_io: no", file=sys.stdout)
    print("traversal_depth_decoding: no", file=sys.stdout)
    print(
        "traversal_prepared_depth_access: "
        f"{'yes' if receipt.prepared_depth_accessed else 'no'}",
        file=sys.stdout,
    )
    print(
        "traversal_workload: "
        f"retained_outcomes={receipt.evaluated_count} "
        f"maximum={MAX_TSDF_PLAN_TRAVERSAL_OUTCOMES}",
        file=sys.stdout,
    )
    print(
        f"blocks_traversed: {receipt.block_count}",
        file=sys.stdout,
    )
    print(
        f"block_transcripts_retained: {len(receipt.block_receipts)}",
        file=sys.stdout,
    )
    print(
        f"voxel_addresses_traversed: {receipt.voxel_address_count}",
        file=sys.stdout,
    )
    print(
        f"voxel_transcripts_retained: {receipt.voxel_count}",
        file=sys.stdout,
    )
    print(
        "voxel_observation_traversals: "
        f"{receipt.voxel_observation_traversal_count}",
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
    print(
        f"storage_slots_updated: {receipt.storage_slots_updated}",
        file=sys.stdout,
    )
    print("voxel_observation_traversal_performed: yes", file=sys.stdout)
    print("voxel_address_traversal_performed: yes", file=sys.stdout)
    print("fusion_block_traversal_performed: yes", file=sys.stdout)
    print(
        "fusion_block_traversal_scope: existing-plan-block-set-only",
        file=sys.stdout,
    )
    print(
        "multiple_block_traversal_performed: "
        f"{'yes' if receipt.block_count > 1 else 'no'}",
        file=sys.stdout,
    )
    print("planned_block_set_traversal_performed: yes", file=sys.stdout)
    print("all_existing_plan_blocks_traversed: yes", file=sys.stdout)
    print("unplanned_blocks_visited: 0", file=sys.stdout)
    print("free_space_coverage_planned: no", file=sys.stdout)
    print("ray_traversal_performed: no", file=sys.stdout)
    print("full_fusion_performed: no", file=sys.stdout)
    print("missing_blocks_created: no", file=sys.stdout)
    print(
        "caught_failure_rollback_scope: complete-planned-storage",
        file=sys.stdout,
    )
    print("artifact_written: no", file=sys.stdout)
    print("storage_persisted: no", file=sys.stdout)
    print("context_persisted: no", file=sys.stdout)
    print(
        f"plan_sha256: {receipt.source_plan_digest_sha256}",
        file=sys.stdout,
    )
    print(
        f"replay_digest_sha256: {receipt.replay_digest_sha256}",
        file=sys.stdout,
    )
    return 0


def _run_tsdf_block_context_observation_rays(
    plan_path: Path,
    session_path: Path,
    observation_sequence: int,
) -> int:
    try:
        plan = load_tsdf_block_plan(plan_path)
        session = load_scan_session(session_path)
        context = build_tsdf_replay_depth_context(plan, session)
        receipt = trace_tsdf_observation_block_rays_from_context(
            plan,
            context,
            observation_sequence,
        )
    except SessionValidationError as error:
        print(
            f"TSDF BLOCK CONTEXT OBSERVATION RAY TRACE FAILED {plan_path}",
            file=sys.stderr,
        )
        for problem in error.errors:
            print(f"- {problem}", file=sys.stderr)
        return 2
    except (TsdfError, SessionReplayError) as error:
        print(
            f"TSDF BLOCK CONTEXT OBSERVATION RAY TRACE FAILED {plan_path}",
            file=sys.stderr,
        )
        print(f"- {error}", file=sys.stderr)
        return 2

    print(
        f"TSDF BLOCK CONTEXT OBSERVATION RAY TRACE CHECK {plan.session_id}",
        file=sys.stdout,
    )
    print("artifact: valid", file=sys.stdout)
    print("session_replay: matched", file=sys.stdout)
    print(
        "context_selection: "
        f"frame_stride={context.frame_stride} "
        f"total={context.total_observations} "
        f"selected={context.selected_observation_count}",
        file=sys.stdout,
    )
    print("context_immutable: yes", file=sys.stdout)
    print("depth_source: replay-depth-context", file=sys.stdout)
    print(
        "observation: "
        f"sequence={receipt.observation_sequence} "
        f"status={receipt.observation_status.value}",
        file=sys.stdout,
    )
    if receipt.camera_origin_world_m is None:
        print("camera_origin_world_m: none", file=sys.stdout)
    else:
        print(
            "camera_origin_world_m: "
            f"{_format_triplet(receipt.camera_origin_world_m)}",
            file=sys.stdout,
        )
    print(
        "image: "
        f"width={receipt.image_size[0]} "
        f"height={receipt.image_size[1]}",
        file=sys.stdout,
    )
    print(
        "pixel_outcomes: "
        f"total={receipt.pixel_count} "
        f"traversed={receipt.traversed_ray_count} "
        f"depth_invalid={receipt.invalid_depth_count}",
        file=sys.stdout,
    )
    print(
        "ray_block_visits: "
        f"total={receipt.block_visit_count} "
        f"unique={len(receipt.covered_block_indices)} "
        f"duplicate={receipt.duplicate_block_visit_count} "
        f"maximum_per_ray={receipt.maximum_blocks_per_ray}",
        file=sys.stdout,
    )
    print(
        "coverage_blocks: "
        f"total={len(receipt.covered_block_indices)} "
        f"nonterminal={len(receipt.nonterminal_block_indices)} "
        f"surface_endpoint={len(receipt.surface_block_indices)}",
        file=sys.stdout,
    )
    print(
        "coverage_partition: "
        f"existing_plan={len(receipt.existing_plan_block_indices)} "
        f"unplanned={len(receipt.unplanned_block_indices)}",
        file=sys.stdout,
    )
    print("coverage_order: canonical-x-fastest", file=sys.stdout)
    if receipt.covered_block_indices:
        print(
            f"first_covered_block: {receipt.covered_block_indices[0]}",
            file=sys.stdout,
        )
        print(
            f"last_covered_block: {receipt.covered_block_indices[-1]}",
            file=sys.stdout,
        )
    else:
        print("first_covered_block: none", file=sys.stdout)
        print("last_covered_block: none", file=sys.stdout)
    traversed_rays = tuple(
        ray for ray in receipt.ray_receipts if ray.traversed
    )
    if traversed_rays:
        first_ray = traversed_rays[0]
        last_ray = traversed_rays[-1]
        print(
            "first_ray: "
            f"pixel={first_ray.pixel_uv} "
            f"depth_m={first_ray.measured_depth_m:.9f} "
            f"blocks={len(first_ray.block_indices)} "
            f"surface={first_ray.surface_world_m}",
            file=sys.stdout,
        )
        print(
            "last_ray: "
            f"pixel={last_ray.pixel_uv} "
            f"depth_m={last_ray.measured_depth_m:.9f} "
            f"blocks={len(last_ray.block_indices)} "
            f"surface={last_ray.surface_world_m}",
            file=sys.stdout,
        )
    else:
        print("first_ray: none", file=sys.stdout)
        print("last_ray: none", file=sys.stdout)
    print("context_provenance: matched", file=sys.stdout)
    print(
        "trace_source_freshness: construction-time-context",
        file=sys.stdout,
    )
    print("trace_session_replay: no", file=sys.stdout)
    print("trace_replay_hashing: no", file=sys.stdout)
    print("trace_source_io: no", file=sys.stdout)
    print("trace_depth_decoding: no", file=sys.stdout)
    print(
        "trace_prepared_depth_access: "
        f"{'yes' if receipt.prepared_depth_accessed else 'no'}",
        file=sys.stdout,
    )
    print(
        "trace_workload: "
        f"retained_outcomes={receipt.retained_outcome_count} "
        f"maximum={MAX_TSDF_OBSERVATION_BLOCK_RAY_OUTCOMES}",
        file=sys.stdout,
    )
    print(
        "coverage_scope: one-prepared-observation-only",
        file=sys.stdout,
    )
    print(
        "visibility_rule: positive-finite-depth-stops-at-measured-surface",
        file=sys.stdout,
    )
    print(
        "block_traversal_rule: closed-half-open-grid-thin-dda-"
        "simultaneous-exact-ties",
        file=sys.stdout,
    )
    print(
        "ray_traversal_performed: "
        f"{'yes' if receipt.traversed_ray_count else 'no'}",
        file=sys.stdout,
    )
    print(
        "centerline_ray_coverage_computed: "
        f"{'yes' if receipt.prepared_depth_accessed else 'no'}",
        file=sys.stdout,
    )
    print(
        "conservative_nearest_pixel_free_space_coverage_proven: no",
        file=sys.stdout,
    )
    print("multiple_observation_coverage_computed: no", file=sys.stdout)
    print("plan_expanded: no", file=sys.stdout)
    print("missing_blocks_created: no", file=sys.stdout)
    print("storage_allocated: no", file=sys.stdout)
    print("storage_mutated: no", file=sys.stdout)
    print("full_fusion_performed: no", file=sys.stdout)
    print("artifact_written: no", file=sys.stdout)
    print("context_persisted: no", file=sys.stdout)
    print(
        f"plan_sha256: {receipt.source_plan_digest_sha256}",
        file=sys.stdout,
    )
    print(
        f"replay_digest_sha256: {receipt.replay_digest_sha256}",
        file=sys.stdout,
    )
    return 0


def _run_tsdf_block_context_plan_rays(
    plan_path: Path,
    session_path: Path,
) -> int:
    try:
        plan = load_tsdf_block_plan(plan_path)
        session = load_scan_session(session_path)
        context = build_tsdf_replay_depth_context(plan, session)
        receipt = survey_tsdf_plan_block_rays_from_context(plan, context)
    except SessionValidationError as error:
        print(
            f"TSDF BLOCK CONTEXT PLAN RAY SURVEY FAILED {plan_path}",
            file=sys.stderr,
        )
        for problem in error.errors:
            print(f"- {problem}", file=sys.stderr)
        return 2
    except (TsdfError, SessionReplayError) as error:
        print(
            f"TSDF BLOCK CONTEXT PLAN RAY SURVEY FAILED {plan_path}",
            file=sys.stderr,
        )
        print(f"- {error}", file=sys.stderr)
        return 2

    status_counts = dict(receipt.observation_status_counts)
    print(
        f"TSDF BLOCK CONTEXT PLAN RAY SURVEY CHECK {plan.session_id}",
        file=sys.stdout,
    )
    print("artifact: valid", file=sys.stdout)
    print("session_replay: matched", file=sys.stdout)
    print(
        "context_selection: "
        f"frame_stride={context.frame_stride} "
        f"total={context.total_observations} "
        f"selected={context.selected_observation_count}",
        file=sys.stdout,
    )
    print("context_immutable: yes", file=sys.stdout)
    print("depth_source: replay-depth-context", file=sys.stdout)
    print(
        "observations: "
        f"selected={receipt.observation_count} "
        f"traced={receipt.traced_observation_count} "
        + " ".join(
            f"{status.value.replace('-', '_')}="
            f"{status_counts.get(status, 0)}"
            for status in TsdfReplayDepthStatus
        ),
        file=sys.stdout,
    )
    print(
        "observation_order: canonical-frame-stride "
        f"sequences={receipt.selected_observation_sequences[0]}.."
        f"{receipt.selected_observation_sequences[-1]}",
        file=sys.stdout,
    )
    first_observation = receipt.observation_receipts[0]
    last_observation = receipt.observation_receipts[-1]
    print(
        "first_observation: "
        f"sequence={first_observation.observation_sequence} "
        f"status={first_observation.observation_status.value} "
        f"covered_blocks={len(first_observation.covered_block_indices)}",
        file=sys.stdout,
    )
    print(
        "last_observation: "
        f"sequence={last_observation.observation_sequence} "
        f"status={last_observation.observation_status.value} "
        f"covered_blocks={len(last_observation.covered_block_indices)}",
        file=sys.stdout,
    )
    print(
        "image: "
        f"width={receipt.image_size[0]} "
        f"height={receipt.image_size[1]}",
        file=sys.stdout,
    )
    print(
        "pixel_outcomes: "
        f"total={receipt.pixel_count} "
        f"traversed={receipt.traversed_ray_count} "
        f"depth_invalid={receipt.invalid_depth_count}",
        file=sys.stdout,
    )
    print(
        "ray_block_visits: "
        f"total={receipt.block_visit_count} "
        f"unique={len(receipt.covered_block_indices)} "
        f"duplicate={receipt.duplicate_block_visit_count} "
        f"maximum_per_ray={receipt.maximum_blocks_per_ray}",
        file=sys.stdout,
    )
    print(
        "coverage_blocks: "
        f"total={len(receipt.covered_block_indices)} "
        f"nonterminal={len(receipt.nonterminal_block_indices)} "
        f"surface_endpoint={len(receipt.surface_block_indices)}",
        file=sys.stdout,
    )
    print(
        "coverage_partition: "
        f"existing_plan={len(receipt.existing_plan_block_indices)} "
        f"unplanned={len(receipt.unplanned_block_indices)}",
        file=sys.stdout,
    )
    print(
        "coverage_support: "
        f"multi_observation={len(receipt.multi_observation_block_indices)} "
        f"maximum_observations={receipt.maximum_block_observation_count}",
        file=sys.stdout,
    )
    print("coverage_order: canonical-x-fastest", file=sys.stdout)
    if receipt.covered_block_indices:
        print(
            f"first_covered_block: {receipt.covered_block_indices[0]}",
            file=sys.stdout,
        )
        print(
            f"last_covered_block: {receipt.covered_block_indices[-1]}",
            file=sys.stdout,
        )
    else:
        print("first_covered_block: none", file=sys.stdout)
        print("last_covered_block: none", file=sys.stdout)
    print("context_provenance: matched", file=sys.stdout)
    print(
        "survey_source_freshness: construction-time-context",
        file=sys.stdout,
    )
    print("survey_session_replay: no", file=sys.stdout)
    print("survey_replay_hashing: no", file=sys.stdout)
    print("survey_source_io: no", file=sys.stdout)
    print("survey_depth_decoding: no", file=sys.stdout)
    print(
        "survey_prepared_depth_access: "
        f"{'yes' if receipt.prepared_depth_accessed else 'no'}",
        file=sys.stdout,
    )
    print(
        "survey_workload: "
        f"retained_outcomes={receipt.retained_outcome_count} "
        f"maximum={MAX_TSDF_PLAN_BLOCK_RAY_SURVEY_OUTCOMES}",
        file=sys.stdout,
    )
    print(
        "coverage_scope: all-plan-selected-observations",
        file=sys.stdout,
    )
    print(
        "visibility_rule: positive-finite-depth-stops-at-measured-surface",
        file=sys.stdout,
    )
    print(
        "block_traversal_rule: closed-half-open-grid-thin-dda-"
        "simultaneous-exact-ties",
        file=sys.stdout,
    )
    print(
        "ray_traversal_performed: "
        f"{'yes' if receipt.traversed_ray_count else 'no'}",
        file=sys.stdout,
    )
    print(
        "centerline_ray_coverage_computed: "
        f"{'yes' if receipt.prepared_depth_accessed else 'no'}",
        file=sys.stdout,
    )
    print(
        "multiple_observation_coverage_computed: "
        f"{'yes' if receipt.traced_observation_count > 1 else 'no'}",
        file=sys.stdout,
    )
    print("all_selected_observations_surveyed: yes", file=sys.stdout)
    print(
        "conservative_nearest_pixel_free_space_coverage_proven: no",
        file=sys.stdout,
    )
    print("coverage_approved_for_expansion: no", file=sys.stdout)
    print("plan_expanded: no", file=sys.stdout)
    print("missing_blocks_created: no", file=sys.stdout)
    print("storage_allocated: no", file=sys.stdout)
    print("storage_mutated: no", file=sys.stdout)
    print("full_fusion_performed: no", file=sys.stdout)
    print("artifact_written: no", file=sys.stdout)
    print("context_persisted: no", file=sys.stdout)
    print(
        f"plan_sha256: {receipt.source_plan_digest_sha256}",
        file=sys.stdout,
    )
    print(
        f"replay_digest_sha256: {receipt.replay_digest_sha256}",
        file=sys.stdout,
    )
    return 0


def _run_tsdf_block_context_pixel_footprint(
    plan_path: Path,
    session_path: Path,
    observation_sequence: int,
    pixel: list[int],
) -> int:
    try:
        plan = load_tsdf_block_plan(plan_path)
        session = load_scan_session(session_path)
        context = build_tsdf_replay_depth_context(plan, session)
        receipt = evaluate_tsdf_pixel_footprint_coverage_from_context(
            plan,
            context,
            observation_sequence,
            tuple(pixel),
        )
    except SessionValidationError as error:
        print(
            f"TSDF BLOCK CONTEXT PIXEL FOOTPRINT FAILED {plan_path}",
            file=sys.stderr,
        )
        for problem in error.errors:
            print(f"- {problem}", file=sys.stderr)
        return 2
    except (TsdfError, SessionReplayError) as error:
        print(
            f"TSDF BLOCK CONTEXT PIXEL FOOTPRINT FAILED {plan_path}",
            file=sys.stderr,
        )
        print(f"- {error}", file=sys.stderr)
        return 2

    print(
        f"TSDF BLOCK CONTEXT PIXEL FOOTPRINT CHECK {plan.session_id}",
        file=sys.stdout,
    )
    print("artifact: valid", file=sys.stdout)
    print("session_replay: matched", file=sys.stdout)
    print(
        "context_selection: "
        f"frame_stride={context.frame_stride} "
        f"total={context.total_observations} "
        f"selected={context.selected_observation_count}",
        file=sys.stdout,
    )
    print("context_immutable: yes", file=sys.stdout)
    print("depth_source: replay-depth-context", file=sys.stdout)
    print(
        "observation: "
        f"sequence={receipt.observation_sequence} "
        f"status={receipt.observation_status.value}",
        file=sys.stdout,
    )
    print(
        "pixel: "
        f"uv={receipt.pixel_uv} "
        f"image={receipt.image_size[0]}x{receipt.image_size[1]}",
        file=sys.stdout,
    )
    print(f"footprint_status: {receipt.status.value}", file=sys.stdout)
    if receipt.camera_origin_world_m is None:
        print("camera_origin_world_m: none", file=sys.stdout)
    else:
        print(
            "camera_origin_world_m: "
            f"{_format_triplet(receipt.camera_origin_world_m)}",
            file=sys.stdout,
        )
    if receipt.measured_depth_m is None:
        print("measured_depth_m: none", file=sys.stdout)
    else:
        print(
            f"measured_depth_m: {receipt.measured_depth_m:.9f}",
            file=sys.stdout,
        )
    print(
        "sampling_rule: nearest-pixel-half-open-unit-square",
        file=sys.stdout,
    )
    print(
        "wedge_rule: apex-to-measured-depth-convex-pyramid",
        file=sys.stdout,
    )
    print(
        "coverage_rule: conservative-plane-superset-of-half-open-cells",
        file=sys.stdout,
    )
    if receipt.footprint_corners_world_m:
        print(
            "first_footprint_corner_world_m: "
            f"{_format_triplet(receipt.footprint_corners_world_m[0])}",
            file=sys.stdout,
        )
        print(
            "last_footprint_corner_world_m: "
            f"{_format_triplet(receipt.footprint_corners_world_m[-1])}",
            file=sys.stdout,
        )
    else:
        print("first_footprint_corner_world_m: none", file=sys.stdout)
        print("last_footprint_corner_world_m: none", file=sys.stdout)
    print(
        "candidate_blocks: "
        f"total={receipt.candidate_block_count} "
        f"min={receipt.candidate_min_block_index} "
        f"max={receipt.candidate_max_block_index}",
        file=sys.stdout,
    )
    print(
        "coverage_blocks: "
        f"covered={receipt.covered_block_count} "
        f"rejected={receipt.rejected_candidate_count}",
        file=sys.stdout,
    )
    print(
        "centerline_blocks: "
        f"total={receipt.centerline_block_count} "
        f"footprint_only={len(receipt.footprint_only_block_indices)}",
        file=sys.stdout,
    )
    print("centerline_contained_in_coverage: yes", file=sys.stdout)
    print(
        "widens_centerline_coverage: "
        f"{'yes' if receipt.widens_centerline_coverage else 'no'}",
        file=sys.stdout,
    )
    print(
        "coverage_partition: "
        f"existing_plan={len(receipt.existing_plan_block_indices)} "
        f"unplanned={len(receipt.unplanned_block_indices)}",
        file=sys.stdout,
    )
    print("coverage_order: canonical-x-fastest", file=sys.stdout)
    if receipt.covered_block_indices:
        print(
            f"first_covered_block: {receipt.covered_block_indices[0]}",
            file=sys.stdout,
        )
        print(
            f"last_covered_block: {receipt.covered_block_indices[-1]}",
            file=sys.stdout,
        )
    else:
        print("first_covered_block: none", file=sys.stdout)
        print("last_covered_block: none", file=sys.stdout)
    print("context_provenance: matched", file=sys.stdout)
    print(
        "coverage_source_freshness: construction-time-context",
        file=sys.stdout,
    )
    print("coverage_session_replay: no", file=sys.stdout)
    print("coverage_replay_hashing: no", file=sys.stdout)
    print("coverage_source_io: no", file=sys.stdout)
    print("coverage_depth_decoding: no", file=sys.stdout)
    print(
        "coverage_prepared_depth_access: "
        f"{'yes' if receipt.prepared_depth_accessed else 'no'}",
        file=sys.stdout,
    )
    print(
        "coverage_workload: "
        f"candidate_blocks={receipt.candidate_block_count} "
        f"maximum={MAX_TSDF_PIXEL_FOOTPRINT_CANDIDATE_BLOCKS}",
        file=sys.stdout,
    )
    print("coverage_scope: one-prepared-pixel-only", file=sys.stdout)
    print(
        "visibility_rule: positive-finite-depth-stops-at-measured-surface",
        file=sys.stdout,
    )
    print(
        "conservative_nearest_pixel_footprint_coverage_computed: "
        f"{'yes' if receipt.covered else 'no'}",
        file=sys.stdout,
    )
    print("per_voxel_sampling_proof_computed: no", file=sys.stdout)
    print("occlusion_rule_defined: no", file=sys.stdout)
    print("visibility_culling_rule_defined: no", file=sys.stdout)
    print("multi_pixel_coverage_computed: no", file=sys.stdout)
    print("multi_observation_coverage_computed: no", file=sys.stdout)
    print("plan_expanded: no", file=sys.stdout)
    print("missing_blocks_created: no", file=sys.stdout)
    print("storage_allocated: no", file=sys.stdout)
    print("storage_mutated: no", file=sys.stdout)
    print("full_fusion_performed: no", file=sys.stdout)
    print("artifact_written: no", file=sys.stdout)
    print("context_persisted: no", file=sys.stdout)
    print(
        f"plan_sha256: {receipt.source_plan_digest_sha256}",
        file=sys.stdout,
    )
    print(
        f"replay_digest_sha256: {receipt.replay_digest_sha256}",
        file=sys.stdout,
    )
    return 0


def _run_tsdf_block_context_voxel_sampling(
    plan_path: Path,
    session_path: Path,
    observation_sequence: int,
    voxel: list[int],
) -> int:
    try:
        plan = load_tsdf_block_plan(plan_path)
        session = load_scan_session(session_path)
        context = build_tsdf_replay_depth_context(plan, session)
        receipt = classify_tsdf_voxel_sampling_from_context(
            plan,
            context,
            observation_sequence,
            tuple(voxel),
        )
    except SessionValidationError as error:
        print(
            f"TSDF BLOCK CONTEXT VOXEL SAMPLING FAILED {plan_path}",
            file=sys.stderr,
        )
        for problem in error.errors:
            print(f"- {problem}", file=sys.stderr)
        return 2
    except (TsdfError, SessionReplayError) as error:
        print(
            f"TSDF BLOCK CONTEXT VOXEL SAMPLING FAILED {plan_path}",
            file=sys.stderr,
        )
        print(f"- {error}", file=sys.stderr)
        return 2

    print(
        f"TSDF BLOCK CONTEXT VOXEL SAMPLING CHECK {plan.session_id}",
        file=sys.stdout,
    )
    print("artifact: valid", file=sys.stdout)
    print("session_replay: matched", file=sys.stdout)
    print(
        "context_selection: "
        f"frame_stride={context.frame_stride} "
        f"total={context.total_observations} "
        f"selected={context.selected_observation_count}",
        file=sys.stdout,
    )
    print("context_immutable: yes", file=sys.stdout)
    print("depth_source: replay-depth-context", file=sys.stdout)
    print(
        "observation: "
        f"sequence={receipt.observation_sequence} "
        f"status={receipt.observation_status.value}",
        file=sys.stdout,
    )
    print(
        "voxel: "
        f"global={receipt.global_index_xyz} "
        f"block={receipt.block_index_xyz} "
        f"local={receipt.local_index_xyz}",
        file=sys.stdout,
    )
    print(
        "voxel_block_planned: "
        f"{'yes' if receipt.planned_block else 'no'}",
        file=sys.stdout,
    )
    print(
        "grid: "
        f"voxel_size_m={receipt.voxel_size_m:.9f} "
        f"truncation_m={receipt.truncation_m:.9f} "
        f"block_resolution={receipt.block_resolution}",
        file=sys.stdout,
    )
    print(
        f"world_xyz_m: {_format_triplet(receipt.world_xyz_m)}",
        file=sys.stdout,
    )
    if receipt.camera_xyz_m is None:
        print("camera_xyz_m: none", file=sys.stdout)
    else:
        print(
            f"camera_xyz_m: {_format_triplet(receipt.camera_xyz_m)}",
            file=sys.stdout,
        )
    if receipt.projected_uv is None:
        print("projected_uv: none", file=sys.stdout)
    else:
        print(
            "projected_uv: "
            f"({receipt.projected_uv[0]:.9f}, "
            f"{receipt.projected_uv[1]:.9f})",
            file=sys.stdout,
        )
    print(
        f"sampled_pixel: {receipt.pixel_uv}"
        if receipt.pixel_uv is not None
        else "sampled_pixel: none",
        file=sys.stdout,
    )
    if receipt.measured_depth_m is None:
        print("measured_depth_m: none", file=sys.stdout)
    else:
        print(
            f"measured_depth_m: {receipt.measured_depth_m:.9f}",
            file=sys.stdout,
        )
    if receipt.signed_distance_m is None:
        print("signed_distance_m: none", file=sys.stdout)
    else:
        print(
            f"signed_distance_m: {receipt.signed_distance_m:.9f}",
            file=sys.stdout,
        )
    if receipt.truncated_tsdf_value is None:
        print("truncated_tsdf_value: none", file=sys.stdout)
    else:
        print(
            f"truncated_tsdf_value: {receipt.truncated_tsdf_value:.9f}",
            file=sys.stdout,
        )
    print(f"sampling_status: {receipt.status.value}", file=sys.stdout)
    print(
        "sampling_rule: nearest-pixel-floor-projected-plus-half",
        file=sys.stdout,
    )
    print(
        "free_space_rule: signed-distance-above-positive-truncation",
        file=sys.stdout,
    )
    print(
        "occluded_rule: signed-distance-below-negative-truncation",
        file=sys.stdout,
    )
    print(
        f"voxel_sampled: {'yes' if receipt.sampled else 'no'}",
        file=sys.stdout,
    )
    print(
        f"voxel_observed: {'yes' if receipt.observed else 'no'}",
        file=sys.stdout,
    )
    print(
        "inside_sampling_wedge: "
        f"{'yes' if receipt.inside_sampling_wedge else 'no'}",
        file=sys.stdout,
    )
    print(
        "reference_evaluator_accepts: "
        f"{'yes' if receipt.contributes_to_reference_tsdf else 'no'}",
        file=sys.stdout,
    )
    print("context_provenance: matched", file=sys.stdout)
    print(
        "sampling_source_freshness: construction-time-context",
        file=sys.stdout,
    )
    print("sampling_session_replay: no", file=sys.stdout)
    print("sampling_replay_hashing: no", file=sys.stdout)
    print("sampling_source_io: no", file=sys.stdout)
    print("sampling_depth_decoding: no", file=sys.stdout)
    print(
        "sampling_prepared_depth_access: "
        f"{'yes' if receipt.prepared_depth_accessed else 'no'}",
        file=sys.stdout,
    )
    print("sampling_scope: one-voxel-one-observation-only", file=sys.stdout)
    print("unplanned_voxels_accepted: yes", file=sys.stdout)
    print("multi_observation_sampling_computed: no", file=sys.stdout)
    print("cross_view_occlusion_rule_defined: no", file=sys.stdout)
    print("visibility_culling_rule_defined: no", file=sys.stdout)
    print("free_space_carving_applied: no", file=sys.stdout)
    print("plan_expanded: no", file=sys.stdout)
    print("missing_blocks_created: no", file=sys.stdout)
    print("storage_allocated: no", file=sys.stdout)
    print("storage_mutated: no", file=sys.stdout)
    print("full_fusion_performed: no", file=sys.stdout)
    print("artifact_written: no", file=sys.stdout)
    print("context_persisted: no", file=sys.stdout)
    print(
        f"plan_sha256: {receipt.source_plan_digest_sha256}",
        file=sys.stdout,
    )
    print(
        f"replay_digest_sha256: {receipt.replay_digest_sha256}",
        file=sys.stdout,
    )
    return 0


def _run_tsdf_block_context_observation_footprint(
    plan_path: Path,
    session_path: Path,
    observation_sequence: int,
) -> int:
    try:
        plan = load_tsdf_block_plan(plan_path)
        session = load_scan_session(session_path)
        context = build_tsdf_replay_depth_context(plan, session)
        receipt = survey_tsdf_observation_pixel_footprints_from_context(
            plan,
            context,
            observation_sequence,
        )
    except SessionValidationError as error:
        print(
            f"TSDF BLOCK CONTEXT OBSERVATION FOOTPRINT FAILED {plan_path}",
            file=sys.stderr,
        )
        for problem in error.errors:
            print(f"- {problem}", file=sys.stderr)
        return 2
    except (TsdfError, SessionReplayError) as error:
        print(
            f"TSDF BLOCK CONTEXT OBSERVATION FOOTPRINT FAILED {plan_path}",
            file=sys.stderr,
        )
        print(f"- {error}", file=sys.stderr)
        return 2

    print(
        f"TSDF BLOCK CONTEXT OBSERVATION FOOTPRINT CHECK {plan.session_id}",
        file=sys.stdout,
    )
    print("artifact: valid", file=sys.stdout)
    print("session_replay: matched", file=sys.stdout)
    print(
        "context_selection: "
        f"frame_stride={context.frame_stride} "
        f"total={context.total_observations} "
        f"selected={context.selected_observation_count}",
        file=sys.stdout,
    )
    print("context_immutable: yes", file=sys.stdout)
    print("depth_source: replay-depth-context", file=sys.stdout)
    print(
        "observation: "
        f"sequence={receipt.observation_sequence} "
        f"status={receipt.observation_status.value}",
        file=sys.stdout,
    )
    if receipt.camera_origin_world_m is None:
        print("camera_origin_world_m: none", file=sys.stdout)
    else:
        print(
            "camera_origin_world_m: "
            f"{_format_triplet(receipt.camera_origin_world_m)}",
            file=sys.stdout,
        )
    print(
        "image: "
        f"width={receipt.image_size[0]} "
        f"height={receipt.image_size[1]}",
        file=sys.stdout,
    )
    print(
        "pixel_outcomes: "
        f"total={receipt.pixel_count} "
        f"covered={receipt.covered_pixel_count} "
        f"depth_invalid={receipt.depth_invalid_count}",
        file=sys.stdout,
    )
    print(
        "candidate_blocks: "
        f"total={receipt.candidate_block_count} "
        f"rejected={receipt.rejected_candidate_count}",
        file=sys.stdout,
    )
    print(
        "pixel_block_visits: "
        f"total={receipt.block_visit_count} "
        f"unique={len(receipt.covered_block_indices)} "
        f"duplicate={receipt.duplicate_block_visit_count} "
        f"maximum_per_pixel={receipt.maximum_blocks_per_pixel}",
        file=sys.stdout,
    )
    print(
        "coverage_blocks: "
        f"total={len(receipt.covered_block_indices)} "
        f"centerline={len(receipt.centerline_block_indices)} "
        f"footprint_only={len(receipt.footprint_only_block_indices)}",
        file=sys.stdout,
    )
    print("centerline_contained_in_coverage: yes", file=sys.stdout)
    print(
        "widens_centerline_coverage: "
        f"{'yes' if receipt.widens_centerline_coverage else 'no'}",
        file=sys.stdout,
    )
    print(
        "coverage_partition: "
        f"existing_plan={len(receipt.existing_plan_block_indices)} "
        f"unplanned={len(receipt.unplanned_block_indices)}",
        file=sys.stdout,
    )
    print(
        "coverage_support: "
        f"maximum_pixels_per_block={receipt.maximum_block_pixel_count}",
        file=sys.stdout,
    )
    print("coverage_order: canonical-x-fastest", file=sys.stdout)
    if receipt.covered_block_indices:
        print(
            f"first_covered_block: {receipt.covered_block_indices[0]}",
            file=sys.stdout,
        )
        print(
            f"last_covered_block: {receipt.covered_block_indices[-1]}",
            file=sys.stdout,
        )
    else:
        print("first_covered_block: none", file=sys.stdout)
        print("last_covered_block: none", file=sys.stdout)
    print("context_provenance: matched", file=sys.stdout)
    print(
        "footprint_source_freshness: construction-time-context",
        file=sys.stdout,
    )
    print("footprint_session_replay: no", file=sys.stdout)
    print("footprint_replay_hashing: no", file=sys.stdout)
    print("footprint_source_io: no", file=sys.stdout)
    print("footprint_depth_decoding: no", file=sys.stdout)
    print(
        "footprint_prepared_depth_access: "
        f"{'yes' if receipt.prepared_depth_accessed else 'no'}",
        file=sys.stdout,
    )
    print(
        "footprint_workload: "
        f"candidate_blocks={receipt.candidate_block_count} "
        f"maximum={MAX_TSDF_OBSERVATION_FOOTPRINT_CANDIDATE_BLOCKS}",
        file=sys.stdout,
    )
    print(
        "coverage_scope: one-prepared-observation-all-pixels",
        file=sys.stdout,
    )
    print(
        "sampling_rule: nearest-pixel-half-open-unit-square",
        file=sys.stdout,
    )
    print(
        "coverage_rule: conservative-plane-superset-of-half-open-cells",
        file=sys.stdout,
    )
    print(
        "conservative_nearest_pixel_footprint_coverage_computed: "
        f"{'yes' if receipt.covered_pixel_count else 'no'}",
        file=sys.stdout,
    )
    print("multi_observation_coverage_computed: no", file=sys.stdout)
    print("per_voxel_verdict_applied: no", file=sys.stdout)
    print("occlusion_rule_defined: no", file=sys.stdout)
    print("visibility_culling_rule_defined: no", file=sys.stdout)
    print("coverage_approved_for_expansion: no", file=sys.stdout)
    print("free_space_carving_applied: no", file=sys.stdout)
    print("plan_expanded: no", file=sys.stdout)
    print("missing_blocks_created: no", file=sys.stdout)
    print("storage_allocated: no", file=sys.stdout)
    print("storage_mutated: no", file=sys.stdout)
    print("full_fusion_performed: no", file=sys.stdout)
    print("artifact_written: no", file=sys.stdout)
    print("context_persisted: no", file=sys.stdout)
    print(
        f"plan_sha256: {receipt.source_plan_digest_sha256}",
        file=sys.stdout,
    )
    print(
        f"replay_digest_sha256: {receipt.replay_digest_sha256}",
        file=sys.stdout,
    )
    return 0


def _run_tsdf_block_context_plan_footprint(
    plan_path: Path,
    session_path: Path,
) -> int:
    try:
        plan = load_tsdf_block_plan(plan_path)
        session = load_scan_session(session_path)
        context = build_tsdf_replay_depth_context(plan, session)
        receipt = survey_tsdf_plan_pixel_footprints_from_context(
            plan,
            context,
        )
    except SessionValidationError as error:
        print(
            f"TSDF BLOCK CONTEXT PLAN FOOTPRINT FAILED {plan_path}",
            file=sys.stderr,
        )
        for problem in error.errors:
            print(f"- {problem}", file=sys.stderr)
        return 2
    except (TsdfError, SessionReplayError) as error:
        print(
            f"TSDF BLOCK CONTEXT PLAN FOOTPRINT FAILED {plan_path}",
            file=sys.stderr,
        )
        print(f"- {error}", file=sys.stderr)
        return 2

    status_counts = dict(receipt.observation_status_counts)
    print(
        f"TSDF BLOCK CONTEXT PLAN FOOTPRINT CHECK {plan.session_id}",
        file=sys.stdout,
    )
    print("artifact: valid", file=sys.stdout)
    print("session_replay: matched", file=sys.stdout)
    print(
        "context_selection: "
        f"frame_stride={context.frame_stride} "
        f"total={context.total_observations} "
        f"selected={context.selected_observation_count}",
        file=sys.stdout,
    )
    print("context_immutable: yes", file=sys.stdout)
    print("depth_source: replay-depth-context", file=sys.stdout)
    print(
        "observations: "
        f"selected={receipt.observation_count} "
        f"surveyed={receipt.surveyed_observation_count} "
        + " ".join(
            f"{status.value.replace('-', '_')}="
            f"{status_counts.get(status, 0)}"
            for status in TsdfReplayDepthStatus
        ),
        file=sys.stdout,
    )
    print(
        "observation_order: canonical-frame-stride "
        f"sequences={receipt.selected_observation_sequences[0]}.."
        f"{receipt.selected_observation_sequences[-1]}",
        file=sys.stdout,
    )
    print(
        "image: "
        f"width={receipt.image_size[0]} "
        f"height={receipt.image_size[1]}",
        file=sys.stdout,
    )
    print(
        "pixel_outcomes: "
        f"total={receipt.pixel_count} "
        f"covered={receipt.covered_pixel_count} "
        f"depth_invalid={receipt.depth_invalid_count}",
        file=sys.stdout,
    )
    print(
        "candidate_blocks: "
        f"total={receipt.candidate_block_count} "
        f"rejected={receipt.rejected_candidate_count}",
        file=sys.stdout,
    )
    print(
        "pixel_block_visits: "
        f"total={receipt.block_visit_count} "
        f"unique={len(receipt.covered_block_indices)} "
        f"duplicate={receipt.duplicate_block_visit_count} "
        f"maximum_per_pixel={receipt.maximum_blocks_per_pixel}",
        file=sys.stdout,
    )
    print(
        "coverage_blocks: "
        f"total={len(receipt.covered_block_indices)} "
        f"centerline={len(receipt.centerline_block_indices)} "
        f"footprint_only={len(receipt.footprint_only_block_indices)}",
        file=sys.stdout,
    )
    print("centerline_contained_in_coverage: yes", file=sys.stdout)
    print(
        "widens_centerline_coverage: "
        f"{'yes' if receipt.widens_centerline_coverage else 'no'}",
        file=sys.stdout,
    )
    print(
        "coverage_partition: "
        f"existing_plan={len(receipt.existing_plan_block_indices)} "
        f"unplanned={len(receipt.unplanned_block_indices)}",
        file=sys.stdout,
    )
    print(
        "coverage_support: "
        f"multi_observation={len(receipt.multi_observation_block_indices)} "
        f"maximum_observations={receipt.maximum_block_observation_count}",
        file=sys.stdout,
    )
    print("coverage_order: canonical-x-fastest", file=sys.stdout)
    if receipt.covered_block_indices:
        print(
            f"first_covered_block: {receipt.covered_block_indices[0]}",
            file=sys.stdout,
        )
        print(
            f"last_covered_block: {receipt.covered_block_indices[-1]}",
            file=sys.stdout,
        )
    else:
        print("first_covered_block: none", file=sys.stdout)
        print("last_covered_block: none", file=sys.stdout)
    print("context_provenance: matched", file=sys.stdout)
    print(
        "footprint_source_freshness: construction-time-context",
        file=sys.stdout,
    )
    print("footprint_session_replay: no", file=sys.stdout)
    print("footprint_replay_hashing: no", file=sys.stdout)
    print("footprint_source_io: no", file=sys.stdout)
    print("footprint_depth_decoding: no", file=sys.stdout)
    print(
        "footprint_prepared_depth_access: "
        f"{'yes' if receipt.prepared_depth_accessed else 'no'}",
        file=sys.stdout,
    )
    print(
        "footprint_workload: "
        f"candidate_blocks={receipt.candidate_block_count} "
        f"maximum={MAX_TSDF_PLAN_FOOTPRINT_CANDIDATE_BLOCKS}",
        file=sys.stdout,
    )
    print(
        "coverage_scope: all-plan-selected-observations-all-pixels",
        file=sys.stdout,
    )
    print(
        "sampling_rule: nearest-pixel-half-open-unit-square",
        file=sys.stdout,
    )
    print(
        "coverage_rule: conservative-plane-superset-of-half-open-cells",
        file=sys.stdout,
    )
    print(
        "conservative_nearest_pixel_footprint_coverage_computed: "
        f"{'yes' if receipt.covered_pixel_count else 'no'}",
        file=sys.stdout,
    )
    print(
        "multiple_observation_coverage_computed: "
        f"{'yes' if receipt.surveyed_observation_count > 1 else 'no'}",
        file=sys.stdout,
    )
    print("all_selected_observations_surveyed: yes", file=sys.stdout)
    print("per_voxel_verdict_applied: no", file=sys.stdout)
    print("carvable_free_space_set_computed: no", file=sys.stdout)
    print("occlusion_rule_defined: no", file=sys.stdout)
    print("visibility_culling_rule_defined: no", file=sys.stdout)
    print("coverage_approved_for_expansion: no", file=sys.stdout)
    print("free_space_carving_applied: no", file=sys.stdout)
    print("plan_expanded: no", file=sys.stdout)
    print("missing_blocks_created: no", file=sys.stdout)
    print("storage_allocated: no", file=sys.stdout)
    print("storage_mutated: no", file=sys.stdout)
    print("full_fusion_performed: no", file=sys.stdout)
    print("artifact_written: no", file=sys.stdout)
    print("context_persisted: no", file=sys.stdout)
    print(
        f"plan_sha256: {receipt.source_plan_digest_sha256}",
        file=sys.stdout,
    )
    print(
        f"replay_digest_sha256: {receipt.replay_digest_sha256}",
        file=sys.stdout,
    )
    return 0


def _run_tsdf_block_context_voxel_cross_view(
    plan_path: Path,
    session_path: Path,
    voxel: list[int],
) -> int:
    try:
        plan = load_tsdf_block_plan(plan_path)
        session = load_scan_session(session_path)
        context = build_tsdf_replay_depth_context(plan, session)
        receipt = classify_tsdf_voxel_across_observations_from_context(
            plan,
            context,
            tuple(voxel),
        )
    except SessionValidationError as error:
        print(
            f"TSDF BLOCK CONTEXT VOXEL CROSS-VIEW FAILED {plan_path}",
            file=sys.stderr,
        )
        for problem in error.errors:
            print(f"- {problem}", file=sys.stderr)
        return 2
    except (TsdfError, SessionReplayError) as error:
        print(
            f"TSDF BLOCK CONTEXT VOXEL CROSS-VIEW FAILED {plan_path}",
            file=sys.stderr,
        )
        print(f"- {error}", file=sys.stderr)
        return 2

    print(
        f"TSDF BLOCK CONTEXT VOXEL CROSS-VIEW CHECK {plan.session_id}",
        file=sys.stdout,
    )
    print("artifact: valid", file=sys.stdout)
    print("session_replay: matched", file=sys.stdout)
    print(
        "context_selection: "
        f"frame_stride={context.frame_stride} "
        f"total={context.total_observations} "
        f"selected={context.selected_observation_count}",
        file=sys.stdout,
    )
    print("context_immutable: yes", file=sys.stdout)
    print("depth_source: replay-depth-context", file=sys.stdout)
    print(
        "voxel: "
        f"global={receipt.global_index_xyz} "
        f"block={receipt.block_index_xyz} "
        f"local={receipt.local_index_xyz}",
        file=sys.stdout,
    )
    print(
        "voxel_block_planned: "
        f"{'yes' if receipt.planned_block else 'no'}",
        file=sys.stdout,
    )
    print(
        "grid: "
        f"voxel_size_m={receipt.voxel_size_m:.9f} "
        f"truncation_m={receipt.truncation_m:.9f} "
        f"block_resolution={receipt.block_resolution}",
        file=sys.stdout,
    )
    print(
        f"world_xyz_m: {_format_triplet(receipt.world_xyz_m)}",
        file=sys.stdout,
    )
    print(
        "observations: "
        f"selected={receipt.observation_count} "
        f"surface_band={receipt.surface_band_count} "
        f"free_space={receipt.free_space_count} "
        f"occluded={receipt.occluded_count} "
        f"unseen={receipt.unseen_count}",
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
    print(f"cross_view_verdict: {receipt.verdict.value}", file=sys.stdout)
    print(
        "verdict_precedence: surface-then-free-space-then-occluded-then-"
        "unseen",
        file=sys.stdout,
    )
    print(
        "occlusion_rule: carries-no-evidence-never-becomes-free-space",
        file=sys.stdout,
    )
    print(
        "contributing_observations: "
        f"{receipt.contributing_observation_sequences}",
        file=sys.stdout,
    )
    print(
        f"wedge_observations: {receipt.wedge_observation_sequences}",
        file=sys.stdout,
    )
    print(
        f"reference_weight: {receipt.reference_weight}",
        file=sys.stdout,
    )
    print(
        f"reference_tsdf_sum: {receipt.reference_tsdf_sum:.9f}",
        file=sys.stdout,
    )
    if receipt.reference_tsdf_value is None:
        print("reference_tsdf_value: none", file=sys.stdout)
    else:
        print(
            f"reference_tsdf_value: {receipt.reference_tsdf_value:.9f}",
            file=sys.stdout,
        )
    print(
        f"voxel_observed: {'yes' if receipt.observed else 'no'}",
        file=sys.stdout,
    )
    print(
        "carvable_free_space: "
        f"{'yes' if receipt.carvable_free_space else 'no'}",
        file=sys.stdout,
    )
    print("context_provenance: matched", file=sys.stdout)
    print(
        "cross_view_source_freshness: construction-time-context",
        file=sys.stdout,
    )
    print("cross_view_session_replay: no", file=sys.stdout)
    print("cross_view_replay_hashing: no", file=sys.stdout)
    print("cross_view_source_io: no", file=sys.stdout)
    print("cross_view_depth_decoding: no", file=sys.stdout)
    print(
        "cross_view_workload: "
        f"retained_observations={receipt.observation_count} "
        f"maximum={MAX_TSDF_VOXEL_CROSS_VIEW_OBSERVATIONS}",
        file=sys.stdout,
    )
    print("cross_view_scope: one-voxel-all-observations", file=sys.stdout)
    print("unplanned_voxels_accepted: yes", file=sys.stdout)
    print("multi_voxel_cross_view_computed: no", file=sys.stdout)
    print("visibility_culling_rule_defined: no", file=sys.stdout)
    print("confidence_or_sensor_weighting_applied: no", file=sys.stdout)
    print("free_space_carving_applied: no", file=sys.stdout)
    print("plan_expanded: no", file=sys.stdout)
    print("missing_blocks_created: no", file=sys.stdout)
    print("storage_allocated: no", file=sys.stdout)
    print("storage_mutated: no", file=sys.stdout)
    print("full_fusion_performed: no", file=sys.stdout)
    print("artifact_written: no", file=sys.stdout)
    print("context_persisted: no", file=sys.stdout)
    print(
        f"plan_sha256: {receipt.source_plan_digest_sha256}",
        file=sys.stdout,
    )
    print(
        f"replay_digest_sha256: {receipt.replay_digest_sha256}",
        file=sys.stdout,
    )
    return 0


def _run_tsdf_block_plan_expand(
    plan_path: Path,
    session_path: Path,
    output: Path,
) -> int:
    try:
        _validate_tsdf_block_plan_output(output)
        plan = load_tsdf_block_plan(plan_path)
        session = load_scan_session(session_path)
        context = build_tsdf_replay_depth_context(plan, session)
        coverage = survey_tsdf_plan_pixel_footprints_from_context(
            plan,
            context,
        )
        domain = sweep_tsdf_coverage_domain_cross_view_from_context(
            plan,
            context,
            coverage,
        )
        proposal = propose_tsdf_plan_expansion_from_domain(plan, domain)
        report = write_tsdf_expanded_block_plan(plan, proposal, output)
    except SessionValidationError as error:
        print(
            f"TSDF BLOCK PLAN EXPAND FAILED {plan_path}",
            file=sys.stderr,
        )
        for problem in error.errors:
            print(f"- {problem}", file=sys.stderr)
        return 2
    except (TsdfError, SessionReplayError) as error:
        print(
            f"TSDF BLOCK PLAN EXPAND FAILED {plan_path}",
            file=sys.stderr,
        )
        print(f"- {error}", file=sys.stderr)
        return 2

    print(
        f"TSDF BLOCK PLAN EXPAND {report.session_id}",
        file=sys.stdout,
    )
    print("source_artifact: valid", file=sys.stdout)
    print("session_replay: matched", file=sys.stdout)
    print(
        "approval_rule: covered-block-with-at-least-one-observed-voxel",
        file=sys.stdout,
    )
    print(
        "coverage_domain: "
        f"blocks={proposal.domain_block_count} "
        f"approved={proposal.approved_block_count} "
        f"rejected={proposal.rejected_block_count}",
        file=sys.stdout,
    )
    print(
        "source_plan: "
        f"blocks={report.source_block_count} "
        f"surface={report.surface_block_count}",
        file=sys.stdout,
    )
    print(
        "expanded_plan: "
        f"blocks={report.expanded_block_count} "
        f"added={report.added_block_count} "
        f"voxel_slots={report.expanded_voxel_slots}",
        file=sys.stdout,
    )
    print(
        "block_bounds: "
        f"min={report.min_block_index} max={report.max_block_index}",
        file=sys.stdout,
    )
    print(
        f"free_space_rule: {report.free_space_rule}",
        file=sys.stdout,
    )
    print("surface_blocks_retained: yes", file=sys.stdout)
    print("source_plan_mutated: no", file=sys.stdout)
    print("source_plan_overwritten: no", file=sys.stdout)
    print("storage_allocated: no", file=sys.stdout)
    print("full_fusion_performed: no", file=sys.stdout)
    print(f"output: {report.output}", file=sys.stdout)
    print(
        f"source_plan_sha256: {report.source_plan_sha256}",
        file=sys.stdout,
    )
    print(
        f"replay_digest_sha256: {report.replay_digest_sha256}",
        file=sys.stdout,
    )
    print(
        f"output_sha256: {report.output_digest_sha256}",
        file=sys.stdout,
    )
    return 0


def _run_tsdf_block_context_plan_expansion(
    plan_path: Path,
    session_path: Path,
) -> int:
    try:
        plan = load_tsdf_block_plan(plan_path)
        session = load_scan_session(session_path)
        context = build_tsdf_replay_depth_context(plan, session)
        coverage = survey_tsdf_plan_pixel_footprints_from_context(
            plan,
            context,
        )
        domain = sweep_tsdf_coverage_domain_cross_view_from_context(
            plan,
            context,
            coverage,
        )
        proposal = propose_tsdf_plan_expansion_from_domain(plan, domain)
    except SessionValidationError as error:
        print(
            f"TSDF BLOCK CONTEXT PLAN EXPANSION FAILED {plan_path}",
            file=sys.stderr,
        )
        for problem in error.errors:
            print(f"- {problem}", file=sys.stderr)
        return 2
    except (TsdfError, SessionReplayError) as error:
        print(
            f"TSDF BLOCK CONTEXT PLAN EXPANSION FAILED {plan_path}",
            file=sys.stderr,
        )
        print(f"- {error}", file=sys.stderr)
        return 2

    print(
        f"TSDF BLOCK CONTEXT PLAN EXPANSION CHECK {plan.session_id}",
        file=sys.stdout,
    )
    print("artifact: valid", file=sys.stdout)
    print("session_replay: matched", file=sys.stdout)
    print(
        "context_selection: "
        f"frame_stride={context.frame_stride} "
        f"total={context.total_observations} "
        f"selected={context.selected_observation_count}",
        file=sys.stdout,
    )
    print("context_immutable: yes", file=sys.stdout)
    print("depth_source: replay-depth-context", file=sys.stdout)
    print(
        "coverage_source: conservative-pixel-footprint-survey",
        file=sys.stdout,
    )
    print(
        "approval_rule: covered-block-with-at-least-one-observed-voxel",
        file=sys.stdout,
    )
    print(
        "source_plan: "
        f"blocks={proposal.source_block_count} "
        f"voxel_slots={proposal.source_voxel_slots}",
        file=sys.stdout,
    )
    print(
        "coverage_domain: "
        f"blocks={proposal.domain_block_count} "
        f"approved={proposal.approved_block_count} "
        f"rejected={proposal.rejected_block_count}",
        file=sys.stdout,
    )
    print(
        "domain_voxels: "
        f"observed={domain.observed_voxel_count} "
        f"carvable_free_space={domain.carvable_free_space_voxel_count}",
        file=sys.stdout,
    )
    print(
        "proposed_plan: "
        f"blocks={proposal.expanded_block_count} "
        f"voxel_slots={proposal.expanded_voxel_slots}",
        file=sys.stdout,
    )
    print(
        "proposed_delta: "
        f"added={proposal.added_block_count} "
        f"retained={proposal.retained_block_count} "
        f"removed={proposal.removed_block_count} "
        f"added_voxel_slots={proposal.added_voxel_slots}",
        file=sys.stdout,
    )
    print("block_order: canonical-x-fastest", file=sys.stdout)
    if proposal.added_block_indices:
        print(
            f"first_added_block: {proposal.added_block_indices[0]}",
            file=sys.stdout,
        )
        print(
            f"last_added_block: {proposal.added_block_indices[-1]}",
            file=sys.stdout,
        )
    else:
        print("first_added_block: none", file=sys.stdout)
        print("last_added_block: none", file=sys.stdout)
    print(
        "source_blocks_retained: "
        f"{'yes' if proposal.retained_block_count == plan.active_block_count else 'no'}",
        file=sys.stdout,
    )
    print(
        "expands_plan: "
        f"{'yes' if proposal.expands_plan else 'no'}",
        file=sys.stdout,
    )
    print("context_provenance: matched", file=sys.stdout)
    print(
        "expansion_source_freshness: construction-time-context",
        file=sys.stdout,
    )
    print("expansion_session_replay: no", file=sys.stdout)
    print("expansion_source_io: no", file=sys.stdout)
    print("expansion_depth_decoding: no", file=sys.stdout)
    print("expansion_scope: proposal-only", file=sys.stdout)
    print("evidence_threshold_applied: no", file=sys.stdout)
    print("confidence_or_sensor_weighting_applied: no", file=sys.stdout)
    print("free_space_carving_applied: no", file=sys.stdout)
    print("plan_written: no", file=sys.stdout)
    print("plan_expanded_on_disk: no", file=sys.stdout)
    print("source_plan_mutated: no", file=sys.stdout)
    print("missing_blocks_created: no", file=sys.stdout)
    print("storage_allocated: no", file=sys.stdout)
    print("storage_mutated: no", file=sys.stdout)
    print("full_fusion_performed: no", file=sys.stdout)
    print("artifact_written: no", file=sys.stdout)
    print(
        f"plan_sha256: {proposal.source_plan_digest_sha256}",
        file=sys.stdout,
    )
    print(
        f"replay_digest_sha256: {proposal.replay_digest_sha256}",
        file=sys.stdout,
    )
    return 0


def _run_tsdf_block_context_domain_cross_view(
    plan_path: Path,
    session_path: Path,
) -> int:
    try:
        plan = load_tsdf_block_plan(plan_path)
        session = load_scan_session(session_path)
        context = build_tsdf_replay_depth_context(plan, session)
        coverage = survey_tsdf_plan_pixel_footprints_from_context(
            plan,
            context,
        )
        receipt = sweep_tsdf_coverage_domain_cross_view_from_context(
            plan,
            context,
            coverage,
        )
    except SessionValidationError as error:
        print(
            f"TSDF BLOCK CONTEXT DOMAIN CROSS-VIEW FAILED {plan_path}",
            file=sys.stderr,
        )
        for problem in error.errors:
            print(f"- {problem}", file=sys.stderr)
        return 2
    except (TsdfError, SessionReplayError) as error:
        print(
            f"TSDF BLOCK CONTEXT DOMAIN CROSS-VIEW FAILED {plan_path}",
            file=sys.stderr,
        )
        print(f"- {error}", file=sys.stderr)
        return 2

    print(
        f"TSDF BLOCK CONTEXT DOMAIN CROSS-VIEW CHECK {plan.session_id}",
        file=sys.stdout,
    )
    print("artifact: valid", file=sys.stdout)
    print("session_replay: matched", file=sys.stdout)
    print(
        "context_selection: "
        f"frame_stride={context.frame_stride} "
        f"total={context.total_observations} "
        f"selected={context.selected_observation_count}",
        file=sys.stdout,
    )
    print("context_immutable: yes", file=sys.stdout)
    print("depth_source: replay-depth-context", file=sys.stdout)
    print(
        "coverage_source: conservative-pixel-footprint-survey",
        file=sys.stdout,
    )
    print(
        "coverage_domain: "
        f"blocks={receipt.block_count} "
        f"existing_plan={len(receipt.existing_plan_block_indices)} "
        f"unplanned={len(receipt.unplanned_block_indices)}",
        file=sys.stdout,
    )
    print(
        "domain_voxels: "
        f"total={receipt.voxel_count} "
        f"observations={receipt.observation_count}",
        file=sys.stdout,
    )
    print("block_order: canonical-x-fastest", file=sys.stdout)
    print(
        "voxel_order: canonical-local-flat-x-fastest local_flat=0..511",
        file=sys.stdout,
    )
    print(
        "voxel_verdicts: "
        f"surface={receipt.surface_voxel_count} "
        f"free_space={receipt.free_space_voxel_count} "
        f"occluded={receipt.occluded_voxel_count} "
        f"unseen={receipt.unseen_voxel_count}",
        file=sys.stdout,
    )
    print(
        f"observed_voxels: {receipt.observed_voxel_count}",
        file=sys.stdout,
    )
    print(
        "carvable_free_space_voxels: "
        f"total={receipt.carvable_free_space_voxel_count} "
        f"in_plan={receipt.planned_carvable_voxel_count} "
        f"unplanned={receipt.unplanned_carvable_voxel_count}",
        file=sys.stdout,
    )
    print(
        "carvable_blocks: "
        f"total={len(receipt.carvable_blocks)} "
        f"unplanned={len(receipt.unplanned_carvable_blocks)}",
        file=sys.stdout,
    )
    print(
        f"reference_weight_total: {receipt.reference_weight_total}",
        file=sys.stdout,
    )
    print(
        f"reference_tsdf_sum_total: {receipt.reference_tsdf_sum_total:.9f}",
        file=sys.stdout,
    )
    print(
        "planned_reference_weight_total: "
        f"{receipt.planned_reference_weight_total}",
        file=sys.stdout,
    )
    print(
        "planned_observed_voxels: "
        f"{receipt.planned_observed_voxel_count}",
        file=sys.stdout,
    )
    print(
        f"maximum_voxel_weight: {receipt.maximum_voxel_weight}",
        file=sys.stdout,
    )
    print(
        "verdict_precedence: surface-then-free-space-then-occluded-then-"
        "unseen",
        file=sys.stdout,
    )
    print(
        "occlusion_rule: carries-no-evidence-never-becomes-free-space",
        file=sys.stdout,
    )
    print("context_provenance: matched", file=sys.stdout)
    print(
        "cross_view_source_freshness: construction-time-context",
        file=sys.stdout,
    )
    print("cross_view_session_replay: no", file=sys.stdout)
    print("cross_view_replay_hashing: no", file=sys.stdout)
    print("cross_view_source_io: no", file=sys.stdout)
    print("cross_view_depth_decoding: no", file=sys.stdout)
    print(
        "cross_view_workload: "
        f"retained_outcomes={receipt.retained_outcome_count} "
        f"maximum={MAX_TSDF_COVERAGE_DOMAIN_CROSS_VIEW_OUTCOMES}",
        file=sys.stdout,
    )
    print(
        "cross_view_scope: surveyed-coverage-domain-all-voxels",
        file=sys.stdout,
    )
    print("unplanned_blocks_accepted: yes", file=sys.stdout)
    print(
        "whole_scan_carvable_set_computed: "
        f"{'yes' if receipt.carvable_free_space_voxel_count else 'no'}",
        file=sys.stdout,
    )
    print("confidence_or_sensor_weighting_applied: no", file=sys.stdout)
    print("visibility_culling_rule_defined: no", file=sys.stdout)
    print("coverage_approved_for_expansion: no", file=sys.stdout)
    print("free_space_carving_applied: no", file=sys.stdout)
    print("plan_expanded: no", file=sys.stdout)
    print("missing_blocks_created: no", file=sys.stdout)
    print("storage_allocated: no", file=sys.stdout)
    print("storage_mutated: no", file=sys.stdout)
    print("full_fusion_performed: no", file=sys.stdout)
    print("artifact_written: no", file=sys.stdout)
    print(
        f"plan_sha256: {receipt.source_plan_digest_sha256}",
        file=sys.stdout,
    )
    print(
        f"replay_digest_sha256: {receipt.replay_digest_sha256}",
        file=sys.stdout,
    )
    return 0


def _run_tsdf_block_context_block_cross_view(
    plan_path: Path,
    session_path: Path,
    block: list[int],
) -> int:
    try:
        plan = load_tsdf_block_plan(plan_path)
        session = load_scan_session(session_path)
        context = build_tsdf_replay_depth_context(plan, session)
        receipt = classify_tsdf_block_voxels_across_observations_from_context(
            plan,
            context,
            tuple(block),
        )
    except SessionValidationError as error:
        print(
            f"TSDF BLOCK CONTEXT BLOCK CROSS-VIEW FAILED {plan_path}",
            file=sys.stderr,
        )
        for problem in error.errors:
            print(f"- {problem}", file=sys.stderr)
        return 2
    except (TsdfError, SessionReplayError) as error:
        print(
            f"TSDF BLOCK CONTEXT BLOCK CROSS-VIEW FAILED {plan_path}",
            file=sys.stderr,
        )
        print(f"- {error}", file=sys.stderr)
        return 2

    print(
        f"TSDF BLOCK CONTEXT BLOCK CROSS-VIEW CHECK {plan.session_id}",
        file=sys.stdout,
    )
    print("artifact: valid", file=sys.stdout)
    print("session_replay: matched", file=sys.stdout)
    print(
        "context_selection: "
        f"frame_stride={context.frame_stride} "
        f"total={context.total_observations} "
        f"selected={context.selected_observation_count}",
        file=sys.stdout,
    )
    print("context_immutable: yes", file=sys.stdout)
    print("depth_source: replay-depth-context", file=sys.stdout)
    print(
        "block: "
        f"index={receipt.block_index_xyz} "
        f"resolution={receipt.block_resolution} "
        f"voxels={receipt.voxel_count}",
        file=sys.stdout,
    )
    print(
        "block_planned: "
        f"{'yes' if receipt.planned_block else 'no'}",
        file=sys.stdout,
    )
    print(
        "grid: "
        f"voxel_size_m={receipt.voxel_size_m:.9f} "
        f"truncation_m={receipt.truncation_m:.9f}",
        file=sys.stdout,
    )
    print(
        "voxel_order: canonical-local-flat-x-fastest local_flat=0..511",
        file=sys.stdout,
    )
    print(
        "voxel_verdicts: "
        f"surface={receipt.surface_voxel_count} "
        f"free_space={receipt.free_space_voxel_count} "
        f"occluded={receipt.occluded_voxel_count} "
        f"unseen={receipt.unseen_voxel_count}",
        file=sys.stdout,
    )
    print(
        f"observed_voxels: {receipt.observed_voxel_count}",
        file=sys.stdout,
    )
    print(
        "carvable_free_space_voxels: "
        f"{receipt.carvable_free_space_voxel_count}",
        file=sys.stdout,
    )
    carvable = receipt.carvable_free_space_voxel_indices
    if carvable:
        print(f"first_carvable_voxel: {carvable[0]}", file=sys.stdout)
        print(f"last_carvable_voxel: {carvable[-1]}", file=sys.stdout)
    else:
        print("first_carvable_voxel: none", file=sys.stdout)
        print("last_carvable_voxel: none", file=sys.stdout)
    print(
        f"reference_weight_total: {receipt.reference_weight_total}",
        file=sys.stdout,
    )
    print(
        f"reference_tsdf_sum_total: {receipt.reference_tsdf_sum_total:.9f}",
        file=sys.stdout,
    )
    print(
        f"maximum_voxel_weight: {receipt.maximum_voxel_weight}",
        file=sys.stdout,
    )
    print(
        "verdict_precedence: surface-then-free-space-then-occluded-then-"
        "unseen",
        file=sys.stdout,
    )
    print(
        "occlusion_rule: carries-no-evidence-never-becomes-free-space",
        file=sys.stdout,
    )
    print("context_provenance: matched", file=sys.stdout)
    print(
        "cross_view_source_freshness: construction-time-context",
        file=sys.stdout,
    )
    print("cross_view_session_replay: no", file=sys.stdout)
    print("cross_view_replay_hashing: no", file=sys.stdout)
    print("cross_view_source_io: no", file=sys.stdout)
    print("cross_view_depth_decoding: no", file=sys.stdout)
    print(
        "cross_view_workload: "
        f"retained_outcomes={receipt.retained_outcome_count} "
        f"maximum={MAX_TSDF_BLOCK_CROSS_VIEW_OUTCOMES}",
        file=sys.stdout,
    )
    print(
        "cross_view_scope: one-block-all-voxels-all-observations",
        file=sys.stdout,
    )
    print("unplanned_blocks_accepted: yes", file=sys.stdout)
    print(
        "carvable_free_space_set_computed: "
        f"{'yes' if receipt.carves_free_space else 'no'}",
        file=sys.stdout,
    )
    print("multi_block_cross_view_computed: no", file=sys.stdout)
    print("confidence_or_sensor_weighting_applied: no", file=sys.stdout)
    print("visibility_culling_rule_defined: no", file=sys.stdout)
    print("free_space_carving_applied: no", file=sys.stdout)
    print("plan_expanded: no", file=sys.stdout)
    print("missing_blocks_created: no", file=sys.stdout)
    print("storage_allocated: no", file=sys.stdout)
    print("storage_mutated: no", file=sys.stdout)
    print("full_fusion_performed: no", file=sys.stdout)
    print("artifact_written: no", file=sys.stdout)
    print(
        f"plan_sha256: {receipt.source_plan_digest_sha256}",
        file=sys.stdout,
    )
    print(
        f"replay_digest_sha256: {receipt.replay_digest_sha256}",
        file=sys.stdout,
    )
    return 0


def _run_tsdf_block_context_voxel_traverse(
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
        context = build_tsdf_replay_depth_context(plan, session)
        storage_before = (
            storage.nonzero_sum_count,
            storage.nonzero_weight_count,
            storage.unknown_voxel_count,
        )
        receipt = traverse_tsdf_voxel_observations_from_context(
            storage,
            address,
            context,
        )
        storage_after = (
            storage.nonzero_sum_count,
            storage.nonzero_weight_count,
            storage.unknown_voxel_count,
        )
    except SessionValidationError as error:
        print(
            f"TSDF BLOCK CONTEXT VOXEL TRAVERSAL FAILED {plan_path}",
            file=sys.stderr,
        )
        for problem in error.errors:
            print(f"- {problem}", file=sys.stderr)
        return 2
    except (TsdfError, SessionReplayError) as error:
        print(
            f"TSDF BLOCK CONTEXT VOXEL TRAVERSAL FAILED {plan_path}",
            file=sys.stderr,
        )
        print(f"- {error}", file=sys.stderr)
        return 2

    print(
        f"TSDF BLOCK CONTEXT VOXEL TRAVERSAL CHECK {plan.session_id}",
        file=sys.stdout,
    )
    print("artifact: valid", file=sys.stdout)
    print("session_replay: matched", file=sys.stdout)
    print(
        "context_selection: "
        f"frame_stride={context.frame_stride} "
        f"total={context.total_observations} "
        f"selected={context.selected_observation_count}",
        file=sys.stdout,
    )
    print("context_immutable: yes", file=sys.stdout)
    print("depth_source: replay-depth-context", file=sys.stdout)
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
    print("context_provenance: matched", file=sys.stdout)
    print(
        "traversal_source_freshness: construction-time-context",
        file=sys.stdout,
    )
    print("traversal_session_replay: no", file=sys.stdout)
    print("traversal_replay_hashing: no", file=sys.stdout)
    print("traversal_source_io: no", file=sys.stdout)
    print("traversal_depth_decoding: no", file=sys.stdout)
    prepared_depth_access = any(
        contribution.depth_decoded
        for contribution in receipt.contributions
    )
    print(
        "traversal_prepared_depth_access: "
        f"{'yes' if prepared_depth_access else 'no'}",
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
    print("context_persisted: no", file=sys.stdout)
    print(
        f"plan_sha256: {receipt.source_plan_digest_sha256}",
        file=sys.stdout,
    )
    print(
        f"replay_digest_sha256: {receipt.replay_digest_sha256}",
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
