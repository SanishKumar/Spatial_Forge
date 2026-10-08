"""Strict loading and replay verification for TSDF block plans."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import PointCloudError, TsdfError
from .model import ScanSession
from .point_cloud import _validate_reconstruction_contract
from .replay import replay_session
from .tsdf_block_plan import (
    MAX_BLOCK_INDEX,
    MAX_PLANNED_BLOCKS,
    MIN_BLOCK_INDEX,
    TSDF_BLOCK_PLAN_SCHEMA,
    TSDF_BLOCK_PLAN_SCHEMA_VERSIONS,
    TSDF_EXPANDED_BLOCK_PLAN_SCHEMA_VERSION,
    TSDF_EXPANSION_APPROVAL_RULES,
    TSDF_FREE_SPACE_RULE_NOT_PLANNED,
    TSDF_FREE_SPACE_RULES,
    TSDF_BLOCK_RESOLUTION,
)

MAX_TSDF_BLOCK_PLAN_BYTES = 32 * 1024 * 1024

_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_JSON_NESTING = 128
# root, grid, activation, planning, and the expanded plan's optional
# expansion provenance object.
_MAX_JSON_OBJECTS = 5
_JSON_ARRAY_OVERHEAD = 5
_JSON_COMMA_OVERHEAD = 64
_INDEX_ORDER = "x-fastest-then-y-then-z"
_BLOCK_BOUNDS = "lower-inclusive-upper-exclusive"
_COORDINATE_ROUNDING = "floor-with-multiply-back-boundary-correction"
_ACTIVATION_RULE = "outward-conservative-half-open-l-infinity-cover"
_ENDPOINT_ROUNDING = "floor-ceil-with-multiply-back-outward-correction"

_BlockIndex = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class TsdfBlockPlan:
    path: Path
    artifact_digest_sha256: str
    session_id: str
    replay_digest_sha256: str
    voxel_size_m: float
    block_resolution: int
    block_extent_m: float
    truncation_m: float
    free_space_rule: str
    expanded_from_plan_sha256: str | None
    frame_stride: int
    total_observations: int
    selected_observations: int
    paired_observations: int
    skipped_missing_depth: int
    skipped_missing_pose: int
    valid_depth_points: int
    invalid_depth_samples: int
    surface_blocks: tuple[_BlockIndex, ...]
    active_blocks: tuple[_BlockIndex, ...]
    planned_voxel_slots: int
    min_block_index: _BlockIndex
    max_block_index: _BlockIndex

    @property
    def surface_block_count(self) -> int:
        return len(self.surface_blocks)

    @property
    def active_block_count(self) -> int:
        return len(self.active_blocks)

    @property
    def halo_block_count(self) -> int:
        return self.active_block_count - self.surface_block_count


def load_tsdf_block_plan(path: str | Path) -> TsdfBlockPlan:
    """Load a structurally valid immutable `.sftplan` artifact."""

    input_path = Path(path).resolve()
    if input_path.suffix.lower() != ".sftplan":
        raise TsdfError("block plan filename must end in .sftplan")
    if not input_path.is_file():
        raise TsdfError(f"TSDF block plan does not exist: {input_path}")

    try:
        with input_path.open("rb") as input_file:
            encoded = input_file.read(MAX_TSDF_BLOCK_PLAN_BYTES + 1)
        if len(encoded) > MAX_TSDF_BLOCK_PLAN_BYTES:
            raise TsdfError(
                f"TSDF block plan exceeds the maximum of "
                f"{MAX_TSDF_BLOCK_PLAN_BYTES}"
            )
    except TsdfError:
        raise
    except OSError as error:
        raise TsdfError(f"cannot read TSDF block plan: {error}") from error
    input_digest = hashlib.sha256(encoded).hexdigest()

    try:
        text = encoded.decode("ascii")
    except UnicodeError as error:
        raise TsdfError("TSDF block plan must be ASCII JSON") from error
    _reject_excessive_json_structure(text)
    try:
        document = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except _DuplicateJsonKey as error:
        raise TsdfError(
            f"TSDF block plan contains duplicate JSON key {error.key!r}"
        ) from error
    except RecursionError as error:
        raise TsdfError("TSDF block plan JSON is nested too deeply") from error
    except (json.JSONDecodeError, ValueError) as error:
        raise TsdfError(f"TSDF block plan is invalid JSON: {error}") from error

    root = _require_object(document, "block_plan")
    _reject_unknown_fields(
        root,
        {
            "schema",
            "schema_version",
            "session_id",
            "replay_digest_sha256",
            "grid",
            "activation",
            "planning",
            "surface_blocks",
            "active_blocks",
            "expansion",
        },
        "block_plan",
    )
    _require_exact_string(
        root,
        "schema",
        TSDF_BLOCK_PLAN_SCHEMA,
        "block_plan",
    )
    schema_version = _require_enum_string(
        root,
        "schema_version",
        TSDF_BLOCK_PLAN_SCHEMA_VERSIONS,
        "block_plan",
    )
    session_id = _require_string(root, "session_id", "block_plan")
    if not _IDENTIFIER.fullmatch(session_id):
        raise TsdfError("block_plan.session_id: invalid identifier")
    replay_digest = _require_string(
        root,
        "replay_digest_sha256",
        "block_plan",
    )
    if not _SHA256.fullmatch(replay_digest):
        raise TsdfError(
            "block_plan.replay_digest_sha256: expected 64 lowercase "
            "hexadecimal characters"
        )

    (
        voxel_size,
        block_resolution,
        block_extent,
    ) = _load_grid(root)
    truncation, free_space_rule = _load_activation(root, voxel_size)
    expansion = _load_expansion(root)
    planning = _load_planning(root)
    _validate_plan_kind(
        schema_version,
        free_space_rule,
        expansion,
        planning,
    )
    expanded_from = None if expansion is None else expansion[0]
    surface_blocks = _load_block_list(
        _require_field(root, "surface_blocks", "block_plan"),
        "block_plan.surface_blocks",
        planning["surface_blocks"],
    )
    active_blocks = _load_block_list(
        _require_field(root, "active_blocks", "block_plan"),
        "block_plan.active_blocks",
        planning["active_blocks"],
    )
    _validate_block_relationships(
        surface_blocks,
        active_blocks,
        planning,
        block_resolution,
    )

    return TsdfBlockPlan(
        path=input_path,
        artifact_digest_sha256=input_digest,
        session_id=session_id,
        replay_digest_sha256=replay_digest,
        voxel_size_m=voxel_size,
        block_resolution=block_resolution,
        block_extent_m=block_extent,
        truncation_m=truncation,
        free_space_rule=free_space_rule,
        expanded_from_plan_sha256=expanded_from,
        frame_stride=planning["frame_stride"],
        total_observations=planning["total_observations"],
        selected_observations=planning["selected_observations"],
        paired_observations=planning["paired_observations"],
        skipped_missing_depth=planning["skipped_missing_depth"],
        skipped_missing_pose=planning["skipped_missing_pose"],
        valid_depth_points=planning["valid_depth_points"],
        invalid_depth_samples=planning["invalid_depth_samples"],
        surface_blocks=surface_blocks,
        active_blocks=active_blocks,
        planned_voxel_slots=planning["planned_voxel_slots"],
        min_block_index=planning["min_block_index"],
        max_block_index=planning["max_block_index"],
    )


def verify_tsdf_block_plan_replay(
    plan: TsdfBlockPlan,
    session: ScanSession,
) -> None:
    """Verify that a valid plan is bound to the current replay inputs."""

    if plan.session_id != session.session_id:
        raise TsdfError(
            "TSDF block plan session_id does not match the loaded session: "
            f"{plan.session_id!r} != {session.session_id!r}"
        )
    replay = replay_session(session)
    if plan.replay_digest_sha256 != replay.digest_sha256:
        raise TsdfError(
            "TSDF block plan replay digest does not match current session "
            "inputs; regenerate the plan"
        )
    try:
        camera, _ = _validate_reconstruction_contract(session)
    except PointCloudError as error:
        raise TsdfError(str(error)) from error

    selected = tuple(
        observation
        for observation in replay.observations
        if observation.sequence % plan.frame_stride == 0
    )
    paired = 0
    missing_depth = 0
    missing_pose = 0
    for observation in selected:
        depth_missing = observation.depth is None
        pose_missing = observation.pose is None
        if depth_missing:
            missing_depth += 1
        if pose_missing:
            missing_pose += 1
        if not depth_missing and not pose_missing:
            paired += 1

    expected = {
        "total_observations": len(replay.observations),
        "selected_observations": len(selected),
        "paired_observations": paired,
        "skipped_missing_depth": missing_depth,
        "skipped_missing_pose": missing_pose,
    }
    for field, expected_value in expected.items():
        actual_value = getattr(plan, field)
        if actual_value != expected_value:
            raise TsdfError(
                f"TSDF block plan {field} does not match current replay: "
                f"{actual_value} != {expected_value}"
            )
    expected_depth_samples = paired * camera.width * camera.height
    actual_depth_samples = (
        plan.valid_depth_points + plan.invalid_depth_samples
    )
    if actual_depth_samples != expected_depth_samples:
        raise TsdfError(
            "TSDF block plan depth sample counts do not match current "
            f"camera/replay: {actual_depth_samples} != "
            f"{expected_depth_samples}"
        )


def _load_grid(
    root: dict[str, Any],
) -> tuple[float, int, float]:
    grid = _require_object(
        _require_field(root, "grid", "block_plan"),
        "block_plan.grid",
    )
    _reject_unknown_fields(
        grid,
        {
            "world_anchor_m",
            "voxel_size_m",
            "block_resolution",
            "block_extent_m",
            "block_bounds",
            "index_order",
            "coordinate_rounding",
        },
        "block_plan.grid",
    )
    anchor = _require_number_triplet(
        _require_field(grid, "world_anchor_m", "block_plan.grid"),
        "block_plan.grid.world_anchor_m",
    )
    if anchor != (0.0, 0.0, 0.0) or any(
        math.copysign(1.0, component) < 0.0 for component in anchor
    ):
        raise TsdfError(
            "block_plan.grid.world_anchor_m: expected [0, 0, 0]"
        )
    voxel_size = _require_positive_number(
        _require_field(grid, "voxel_size_m", "block_plan.grid"),
        "block_plan.grid.voxel_size_m",
    )
    block_resolution = _require_positive_integer(
        _require_field(grid, "block_resolution", "block_plan.grid"),
        "block_plan.grid.block_resolution",
    )
    if block_resolution != TSDF_BLOCK_RESOLUTION:
        raise TsdfError(
            "block_plan.grid.block_resolution: "
            f"expected {TSDF_BLOCK_RESOLUTION}, received {block_resolution}"
        )
    block_extent = _require_positive_number(
        _require_field(grid, "block_extent_m", "block_plan.grid"),
        "block_plan.grid.block_extent_m",
    )
    expected_extent = voxel_size * block_resolution
    if not math.isfinite(expected_extent) or block_extent != expected_extent:
        raise TsdfError(
            "block_plan.grid.block_extent_m: expected "
            f"{expected_extent!r}, received {block_extent!r}"
        )
    _require_exact_string(
        grid,
        "block_bounds",
        _BLOCK_BOUNDS,
        "block_plan.grid",
    )
    _require_exact_string(
        grid,
        "index_order",
        _INDEX_ORDER,
        "block_plan.grid",
    )
    _require_exact_string(
        grid,
        "coordinate_rounding",
        _COORDINATE_ROUNDING,
        "block_plan.grid",
    )
    return voxel_size, block_resolution, block_extent


def _load_activation(
    root: dict[str, Any],
    voxel_size_m: float,
) -> tuple[float, str]:
    activation = _require_object(
        _require_field(root, "activation", "block_plan"),
        "block_plan.activation",
    )
    _reject_unknown_fields(
        activation,
        {
            "truncation_m",
            "rule",
            "endpoint_rounding",
            "free_space_rule",
        },
        "block_plan.activation",
    )
    truncation = _require_positive_number(
        _require_field(
            activation,
            "truncation_m",
            "block_plan.activation",
        ),
        "block_plan.activation.truncation_m",
    )
    if truncation < voxel_size_m:
        raise TsdfError(
            "block_plan.activation.truncation_m: "
            "must be at least voxel_size_m"
        )
    _require_exact_string(
        activation,
        "rule",
        _ACTIVATION_RULE,
        "block_plan.activation",
    )
    _require_exact_string(
        activation,
        "endpoint_rounding",
        _ENDPOINT_ROUNDING,
        "block_plan.activation",
    )
    free_space_rule = _require_enum_string(
        activation,
        "free_space_rule",
        TSDF_FREE_SPACE_RULES,
        "block_plan.activation",
    )
    return truncation, free_space_rule


def _load_planning(root: dict[str, Any]) -> dict[str, Any]:
    planning = _require_object(
        _require_field(root, "planning", "block_plan"),
        "block_plan.planning",
    )
    positive_names = (
        "total_observations",
        "selected_observations",
        "paired_observations",
        "valid_depth_points",
        "surface_blocks",
        "active_blocks",
        "planned_voxel_slots",
    )
    nonnegative_names = (
        "skipped_missing_depth",
        "skipped_missing_pose",
        "invalid_depth_samples",
        "halo_blocks",
    )
    values: dict[str, Any] = {
        "frame_stride": _require_positive_integer(
            _require_field(
                planning,
                "frame_stride",
                "block_plan.planning",
            ),
            "block_plan.planning.frame_stride",
        )
    }
    for name in positive_names:
        values[name] = _require_positive_integer(
            _require_field(planning, name, "block_plan.planning"),
            f"block_plan.planning.{name}",
        )
    for name in nonnegative_names:
        values[name] = _require_nonnegative_integer(
            _require_field(planning, name, "block_plan.planning"),
            f"block_plan.planning.{name}",
        )
    values["min_block_index"] = _require_block_index(
        _require_field(
            planning,
            "min_block_index",
            "block_plan.planning",
        ),
        "block_plan.planning.min_block_index",
    )
    values["max_block_index"] = _require_block_index(
        _require_field(
            planning,
            "max_block_index",
            "block_plan.planning",
        ),
        "block_plan.planning.max_block_index",
    )
    _reject_unknown_fields(
        planning,
        {
            "frame_stride",
            *positive_names,
            *nonnegative_names,
            "min_block_index",
            "max_block_index",
        },
        "block_plan.planning",
    )

    total = values["total_observations"]
    selected = values["selected_observations"]
    paired = values["paired_observations"]
    if selected > total:
        raise TsdfError(
            "block_plan.planning: selected observations exceed total"
        )
    expected_selected = ((total - 1) // values["frame_stride"]) + 1
    if selected != expected_selected:
        raise TsdfError(
            "block_plan.planning.selected_observations: "
            f"expected {expected_selected}, received {selected}"
        )
    if paired > selected:
        raise TsdfError(
            "block_plan.planning: paired observations exceed selected"
        )
    skipped = selected - paired
    missing_depth = values["skipped_missing_depth"]
    missing_pose = values["skipped_missing_pose"]
    if not (
        max(missing_depth, missing_pose)
        <= skipped
        <= missing_depth + missing_pose
    ):
        raise TsdfError(
            "block_plan.planning: missing-depth/pose counts are "
            "inconsistent with paired observations"
        )
    if values["valid_depth_points"] + values["invalid_depth_samples"] < paired:
        raise TsdfError(
            "block_plan.planning: depth sample counts are too small "
            "for paired observations"
        )
    if values["surface_blocks"] > values["active_blocks"]:
        raise TsdfError(
            "block_plan.planning: surface blocks exceed active blocks"
        )
    if values["surface_blocks"] > values["valid_depth_points"]:
        raise TsdfError(
            "block_plan.planning: surface blocks exceed valid depth points"
        )
    if values["active_blocks"] > MAX_PLANNED_BLOCKS:
        raise TsdfError(
            "block_plan.planning.active_blocks: "
            f"maximum is {MAX_PLANNED_BLOCKS}"
        )
    if values["halo_blocks"] != (
        values["active_blocks"] - values["surface_blocks"]
    ):
        raise TsdfError(
            "block_plan.planning.halo_blocks: inconsistent block counts"
        )
    return values


def _load_block_list(
    value: Any,
    label: str,
    expected_count: int,
) -> tuple[_BlockIndex, ...]:
    if not isinstance(value, list):
        raise TsdfError(f"{label}: expected an array")
    if len(value) != expected_count:
        raise TsdfError(
            f"{label}: expected {expected_count} entries, received {len(value)}"
        )
    if len(value) > MAX_PLANNED_BLOCKS:
        raise TsdfError(f"{label}: maximum is {MAX_PLANNED_BLOCKS} entries")

    blocks: list[_BlockIndex] = []
    previous_key: tuple[int, int, int] | None = None
    for position, raw_index in enumerate(value):
        block = _require_block_index(raw_index, f"{label}[{position}]")
        key = (block[2], block[1], block[0])
        if previous_key is not None and key <= previous_key:
            raise TsdfError(
                f"{label}: entries must be unique and strictly "
                "x-fastest ordered"
            )
        previous_key = key
        blocks.append(block)
    return tuple(blocks)


def _validate_block_relationships(
    surface_blocks: tuple[_BlockIndex, ...],
    active_blocks: tuple[_BlockIndex, ...],
    planning: dict[str, Any],
    block_resolution: int,
) -> None:
    active_set = set(active_blocks)
    if not set(surface_blocks).issubset(active_set):
        raise TsdfError(
            "block_plan.surface_blocks: every surface block must be active"
        )
    planned_voxel_slots = len(active_blocks) * block_resolution**3
    if planning["planned_voxel_slots"] != planned_voxel_slots:
        raise TsdfError(
            "block_plan.planning.planned_voxel_slots: expected "
            f"{planned_voxel_slots}, received "
            f"{planning['planned_voxel_slots']}"
        )

    minimum = tuple(
        min(block[axis] for block in active_blocks)
        for axis in range(3)
    )
    maximum = tuple(
        max(block[axis] for block in active_blocks)
        for axis in range(3)
    )
    if planning["min_block_index"] != minimum:
        raise TsdfError(
            "block_plan.planning.min_block_index: "
            f"expected {minimum}, received {planning['min_block_index']}"
        )
    if planning["max_block_index"] != maximum:
        raise TsdfError(
            "block_plan.planning.max_block_index: "
            f"expected {maximum}, received {planning['max_block_index']}"
        )


def _require_field(
    value: dict[str, Any],
    field: str,
    label: str,
) -> Any:
    if field not in value:
        raise TsdfError(f"{label}: missing required field {field!r}")
    return value[field]


def _require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TsdfError(f"{label}: expected an object")
    return value


def _reject_unknown_fields(
    value: dict[str, Any],
    allowed: set[str],
    label: str,
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise TsdfError(f"{label}: unknown field(s): {', '.join(unknown)}")


def _require_string(
    value: dict[str, Any],
    field: str,
    label: str,
) -> str:
    result = _require_field(value, field, label)
    if not isinstance(result, str):
        raise TsdfError(f"{label}.{field}: expected a string")
    return result


def _require_exact_string(
    value: dict[str, Any],
    field: str,
    expected: str,
    label: str,
) -> None:
    result = _require_string(value, field, label)
    if result != expected:
        raise TsdfError(
            f"{label}.{field}: expected {expected!r}, received {result!r}"
        )


def _require_enum_string(
    value: dict[str, Any],
    field: str,
    permitted: tuple[str, ...],
    label: str,
) -> str:
    result = _require_string(value, field, label)
    if result not in permitted:
        expected = ", ".join(repr(item) for item in permitted)
        raise TsdfError(
            f"{label}.{field}: expected one of {expected}, "
            f"received {result!r}"
        )
    return result


def _load_expansion(
    root: dict[str, Any],
) -> tuple[str, int, str] | None:
    """Load expansion provenance: source digest, blocks added, rule."""

    if "expansion" not in root:
        return None
    expansion = _require_object(
        _require_field(root, "expansion", "block_plan"),
        "block_plan.expansion",
    )
    _reject_unknown_fields(
        expansion,
        {"source_plan_sha256", "approval_rule", "added_blocks"},
        "block_plan.expansion",
    )
    source_digest = _require_string(
        expansion,
        "source_plan_sha256",
        "block_plan.expansion",
    )
    if not _SHA256.fullmatch(source_digest):
        raise TsdfError(
            "block_plan.expansion.source_plan_sha256: expected 64 lowercase "
            "hexadecimal characters"
        )
    approval_rule = _require_enum_string(
        expansion,
        "approval_rule",
        tuple(TSDF_EXPANSION_APPROVAL_RULES.values()),
        "block_plan.expansion",
    )
    added = _require_field(expansion, "added_blocks", "block_plan.expansion")
    if isinstance(added, bool) or not isinstance(added, int) or added < 0:
        raise TsdfError(
            "block_plan.expansion.added_blocks: expected a nonnegative "
            "integer"
        )
    return source_digest, added, approval_rule


def _validate_plan_kind(
    schema_version: str,
    free_space_rule: str,
    expansion: tuple[str, int, str] | None,
    planning: dict[str, Any],
) -> None:
    """Refuse a plan whose version, free-space rule and provenance disagree.

    A surface/truncation plan and an expanded plan are different claims.
    Each is one schema version, one free-space rule, and either no
    provenance or exactly one. A file that mixes them describes neither.
    """

    if schema_version == TSDF_EXPANDED_BLOCK_PLAN_SCHEMA_VERSION:
        if expansion is None:
            raise TsdfError(
                "block_plan.expansion: required by schema version "
                f"{schema_version!r}, which is an expanded plan"
            )
        if free_space_rule not in TSDF_EXPANSION_APPROVAL_RULES:
            raise TsdfError(
                "block_plan.activation.free_space_rule: schema version "
                f"{schema_version!r} is an expanded plan and requires "
                "one of "
                + ", ".join(
                    repr(rule) for rule in TSDF_EXPANSION_APPROVAL_RULES
                )
                + f", received {free_space_rule!r}"
            )
        expected_approval = TSDF_EXPANSION_APPROVAL_RULES[free_space_rule]
        if expansion[2] != expected_approval:
            raise TsdfError(
                "block_plan.expansion.approval_rule: free-space rule "
                f"{free_space_rule!r} approves by "
                f"{expected_approval!r}, received {expansion[2]!r}"
            )
        added_blocks = expansion[1]
        if added_blocks > planning["halo_blocks"]:
            raise TsdfError(
                "block_plan.expansion.added_blocks: "
                f"{added_blocks} blocks added, but only "
                f"{planning['halo_blocks']} active blocks are not "
                "surface blocks"
            )
        return
    if expansion is not None:
        raise TsdfError(
            "block_plan.expansion: not permitted by schema version "
            f"{schema_version!r}; an expanded plan is version "
            f"{TSDF_EXPANDED_BLOCK_PLAN_SCHEMA_VERSION!r}"
        )
    if free_space_rule != TSDF_FREE_SPACE_RULE_NOT_PLANNED:
        raise TsdfError(
            "block_plan.activation.free_space_rule: schema version "
            f"{schema_version!r} plans no free space and requires "
            f"{TSDF_FREE_SPACE_RULE_NOT_PLANNED!r}, received "
            f"{free_space_rule!r}"
        )


def _require_number_triplet(
    value: Any,
    label: str,
) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise TsdfError(f"{label}: expected an array of 3 finite numbers")
    converted = tuple(
        _require_finite_number(component, f"{label}[{index}]")
        for index, component in enumerate(value)
    )
    return converted  # type: ignore[return-value]


def _require_block_index(value: Any, label: str) -> _BlockIndex:
    if not isinstance(value, list) or len(value) != 3:
        raise TsdfError(f"{label}: expected an array of 3 integers")
    converted: list[int] = []
    for index, component in enumerate(value):
        if isinstance(component, bool) or not isinstance(component, int):
            raise TsdfError(f"{label}[{index}]: expected an integer")
        if component < MIN_BLOCK_INDEX or component > MAX_BLOCK_INDEX:
            raise TsdfError(
                f"{label}[{index}]: outside signed 32-bit block range"
            )
        converted.append(component)
    return (converted[0], converted[1], converted[2])


def _require_finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TsdfError(f"{label}: expected a finite number")
    try:
        converted = float(value)
    except (OverflowError, ValueError) as error:
        raise TsdfError(f"{label}: expected a finite number") from error
    if not math.isfinite(converted):
        raise TsdfError(f"{label}: expected a finite number")
    return converted


def _require_positive_number(value: Any, label: str) -> float:
    converted = _require_finite_number(value, label)
    if converted <= 0.0:
        raise TsdfError(f"{label}: expected a positive number")
    return converted


def _require_nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TsdfError(f"{label}: expected a nonnegative integer")
    return value


def _require_positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise TsdfError(f"{label}: expected a positive integer")
    return value


class _DuplicateJsonKey(ValueError):
    def __init__(self, key: str) -> None:
        self.key = key
        super().__init__(key)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKey(key)
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON number {value!r} is not allowed")


def _reject_excessive_json_structure(text: str) -> None:
    depth = 0
    arrays = 0
    objects = 0
    commas = 0
    in_string = False
    escaped = False
    for character in text:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "[":
            arrays += 1
            if arrays > 2 * MAX_PLANNED_BLOCKS + _JSON_ARRAY_OVERHEAD:
                raise TsdfError(
                    "TSDF block plan JSON contains too many arrays"
                )
            depth += 1
            if depth > _MAX_JSON_NESTING:
                raise TsdfError("TSDF block plan JSON is nested too deeply")
        elif character == "{":
            objects += 1
            if objects > _MAX_JSON_OBJECTS:
                raise TsdfError(
                    "TSDF block plan JSON contains too many objects"
                )
            depth += 1
            if depth > _MAX_JSON_NESTING:
                raise TsdfError("TSDF block plan JSON is nested too deeply")
        elif character in "]}":
            depth -= 1
        elif character == ",":
            commas += 1
            if commas > 6 * MAX_PLANNED_BLOCKS + _JSON_COMMA_OVERHEAD:
                raise TsdfError(
                    "TSDF block plan JSON contains too many separators"
                )
