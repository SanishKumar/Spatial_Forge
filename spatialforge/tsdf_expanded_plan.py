"""Serialization of an approved expanded TSDF block plan."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from .errors import TsdfError
from .tsdf_block_plan import (
    MAX_PLANNED_BLOCKS,
    TSDF_BLOCK_PLAN_SCHEMA,
    TSDF_BLOCK_PLAN_SCHEMA_VERSION,
    TSDF_BLOCK_RESOLUTION,
    TSDF_FREE_SPACE_RULE_FOOTPRINT,
    _clean_float,
    _validate_tsdf_block_plan_output,
    _write_plan_without_overwrite,
)
from .tsdf_block_plan_loader import TsdfBlockPlan
from .tsdf_plan_expansion import TsdfPlanExpansionProposal

_Index3 = tuple[int, int, int]

EXPANSION_APPROVAL_RULE = "covered-block-with-at-least-one-observed-voxel"


@dataclass(frozen=True, slots=True)
class TsdfExpandedPlanReport:
    """Immutable report for one written expanded block plan."""

    session_id: str
    output: Path
    source_plan_sha256: str
    replay_digest_sha256: str
    source_block_count: int
    expanded_block_count: int
    added_block_count: int
    surface_block_count: int
    expanded_voxel_slots: int
    min_block_index: _Index3
    max_block_index: _Index3
    free_space_rule: str
    output_digest_sha256: str


def write_tsdf_expanded_block_plan(
    plan: TsdfBlockPlan,
    proposal: TsdfPlanExpansionProposal,
    output: str | Path,
) -> TsdfExpandedPlanReport:
    """Write the proposal's expanded block set as a new `.sftplan`.

    The source plan is never modified and never overwritten: the output must
    be a new path. Every field except the active-block set, the counts derived
    from it, the free-space rule and the expansion provenance is carried
    through from the source plan unchanged.
    """

    if not isinstance(plan, TsdfBlockPlan):
        raise TsdfError(
            "TSDF expanded plan requires a loaded TsdfBlockPlan"
        )
    if not isinstance(proposal, TsdfPlanExpansionProposal):
        raise TsdfError(
            "TSDF expanded plan requires a TsdfPlanExpansionProposal"
        )
    if (
        proposal.source_plan_digest_sha256 != plan.artifact_digest_sha256
        or proposal.replay_digest_sha256 != plan.replay_digest_sha256
    ):
        raise TsdfError(
            "TSDF expansion proposal provenance does not match the block plan"
        )
    if proposal.source_plan_block_indices != plan.active_blocks:
        raise TsdfError(
            "TSDF expansion proposal source blocks do not match the block plan"
        )
    if (
        proposal.frame_stride != plan.frame_stride
        or proposal.total_observations != plan.total_observations
        or proposal.block_resolution != plan.block_resolution
    ):
        raise TsdfError(
            "TSDF expansion proposal selection does not match the block plan"
        )

    output_path = _validate_tsdf_block_plan_output(output)
    if output_path == plan.path.resolve():
        raise TsdfError(
            "TSDF expanded plan must not overwrite its source plan"
        )

    active_blocks = proposal.expanded_block_indices
    if not active_blocks or len(active_blocks) > MAX_PLANNED_BLOCKS:
        raise TsdfError("TSDF expanded plan active blocks are invalid")
    if not set(plan.surface_blocks).issubset(set(active_blocks)):
        raise TsdfError(
            "TSDF expanded plan must retain every surface block as active"
        )
    minimum = tuple(
        min(block[axis] for block in active_blocks) for axis in range(3)
    )
    maximum = tuple(
        max(block[axis] for block in active_blocks) for axis in range(3)
    )
    document = {
        "schema": TSDF_BLOCK_PLAN_SCHEMA,
        "schema_version": TSDF_BLOCK_PLAN_SCHEMA_VERSION,
        "session_id": plan.session_id,
        "replay_digest_sha256": plan.replay_digest_sha256,
        "grid": {
            "world_anchor_m": [0.0, 0.0, 0.0],
            "voxel_size_m": _clean_float(plan.voxel_size_m),
            "block_resolution": TSDF_BLOCK_RESOLUTION,
            "block_extent_m": _clean_float(plan.block_extent_m),
            "block_bounds": "lower-inclusive-upper-exclusive",
            "index_order": "x-fastest-then-y-then-z",
            "coordinate_rounding": (
                "floor-with-multiply-back-boundary-correction"
            ),
        },
        "activation": {
            "truncation_m": _clean_float(plan.truncation_m),
            "rule": "outward-conservative-half-open-l-infinity-cover",
            "endpoint_rounding": (
                "floor-ceil-with-multiply-back-outward-correction"
            ),
            "free_space_rule": TSDF_FREE_SPACE_RULE_FOOTPRINT,
        },
        "planning": {
            "frame_stride": plan.frame_stride,
            "total_observations": plan.total_observations,
            "selected_observations": plan.selected_observations,
            "paired_observations": plan.paired_observations,
            "skipped_missing_depth": plan.skipped_missing_depth,
            "skipped_missing_pose": plan.skipped_missing_pose,
            "valid_depth_points": plan.valid_depth_points,
            "invalid_depth_samples": plan.invalid_depth_samples,
            "surface_blocks": len(plan.surface_blocks),
            "active_blocks": len(active_blocks),
            "halo_blocks": len(active_blocks) - len(plan.surface_blocks),
            "planned_voxel_slots": (
                len(active_blocks) * TSDF_BLOCK_RESOLUTION**3
            ),
            "min_block_index": list(minimum),
            "max_block_index": list(maximum),
        },
        "surface_blocks": [list(index) for index in plan.surface_blocks],
        "active_blocks": [list(index) for index in active_blocks],
        "expansion": {
            "source_plan_sha256": plan.artifact_digest_sha256,
            "approval_rule": EXPANSION_APPROVAL_RULE,
            "added_blocks": proposal.added_block_count,
        },
    }

    try:
        encoded = (
            json.dumps(
                document,
                ensure_ascii=True,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise TsdfError(
            f"cannot encode expanded TSDF block plan: {error}"
        ) from error

    _write_plan_without_overwrite(output_path, encoded)

    return TsdfExpandedPlanReport(
        session_id=plan.session_id,
        output=output_path,
        source_plan_sha256=plan.artifact_digest_sha256,
        replay_digest_sha256=plan.replay_digest_sha256,
        source_block_count=len(plan.active_blocks),
        expanded_block_count=len(active_blocks),
        added_block_count=proposal.added_block_count,
        surface_block_count=len(plan.surface_blocks),
        expanded_voxel_slots=len(active_blocks) * TSDF_BLOCK_RESOLUTION**3,
        min_block_index=minimum,  # type: ignore[arg-type]
        max_block_index=maximum,  # type: ignore[arg-type]
        free_space_rule=TSDF_FREE_SPACE_RULE_FOOTPRINT,
        output_digest_sha256=sha256(encoded).hexdigest(),
    )
