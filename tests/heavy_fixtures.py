"""Shared, cached fixtures for the expensive coverage/verdict test paths.

Resolving a coverage domain costs a conservative footprint survey over every
selected observation plus a cross-view sweep over every covered voxel. Several
test modules need the same resolutions, and rebuilding them per test dominated
the suite's runtime.

Everything handed out here is frozen: the plan, the replay/depth context and
every receipt are immutable dataclasses, and no test writes into the shared
session or plan directory. One resolution per case is therefore shared across
the whole run rather than per module.

Cases are keyed by ``(raw_depth, frame_stride)``. ``raw_depth=0`` means the
pristine committed fixture; any other value copies the session and rewrites
its depth files, which changes their bytes and therefore the replay and plan
digests. Tests that assert committed digests must use the pristine case.
"""

from __future__ import annotations

import atexit
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from spatialforge import build_tsdf_replay_depth_context, load_tsdf_block_plan
from spatialforge.session_loader import load_scan_session
from spatialforge.tsdf_block_plan import plan_tsdf_blocks
from spatialforge.tsdf_block_plan_loader import TsdfBlockPlan
from spatialforge.tsdf_domain_cross_view import (
    TsdfCoverageDomainCrossViewReceipt,
    sweep_tsdf_coverage_domain_cross_view_from_context,
)
from spatialforge.tsdf_plan_footprint_survey import (
    TsdfPlanFootprintSurveyReceipt,
    survey_tsdf_plan_pixel_footprints_from_context,
)
from spatialforge.tsdf_replay_depth_context import TsdfReplayDepthContext

TESTS_ROOT = Path(__file__).resolve().parent
FIXTURE = TESTS_ROOT / "fixtures" / "minimal.vgsession"
PLAN_ARGUMENTS = {
    "voxel_size_m": 0.125,
    "truncation_m": 0.5,
}


@dataclass(slots=True)
class SharedCase:
    """One prepared case: plan and context always, coverage on demand.

    The coverage survey and domain sweep together cost several seconds, and
    plenty of tests need only the plan and context. They are therefore built
    lazily on first access and then reused, so a module that never touches
    ``coverage`` or ``domain`` never pays for them.
    """

    root: Path
    session_path: Path
    plan_path: Path
    plan: TsdfBlockPlan
    context: TsdfReplayDepthContext
    _coverage: TsdfPlanFootprintSurveyReceipt | None = None
    _domain: TsdfCoverageDomainCrossViewReceipt | None = None

    @property
    def coverage(self) -> TsdfPlanFootprintSurveyReceipt:
        if self._coverage is None:
            self._coverage = survey_tsdf_plan_pixel_footprints_from_context(
                self.plan,
                self.context,
            )
        return self._coverage

    @property
    def domain(self) -> TsdfCoverageDomainCrossViewReceipt:
        if self._domain is None:
            self._domain = (
                sweep_tsdf_coverage_domain_cross_view_from_context(
                    self.plan,
                    self.context,
                    self.coverage,
                )
            )
        return self._domain


_CASES: dict[tuple[int, int], SharedCase] = {}
# ``mkdtemp`` rather than ``TemporaryDirectory``: the latter registers its own
# interpreter-shutdown finalizer, which races the atexit cleanup below and
# emits a ResourceWarning that ``-W error`` turns into a failure.
_ROOTS: list[Path] = []


def _write_depth(session_path: Path, raw_depth: int) -> None:
    payload = (
        "P2\n2 2\n65535\n"
        f"{raw_depth} {raw_depth}\n{raw_depth} {raw_depth}\n"
    )
    for filename in ("000000.pgm", "000001.pgm"):
        (session_path / "data" / "depth" / filename).write_text(
            payload,
            encoding="ascii",
        )


def shared_case(raw_depth: int = 0, frame_stride: int = 1) -> SharedCase:
    """Return the cached resolution for one case, building it on demand."""

    key = (raw_depth, frame_stride)
    cached = _CASES.get(key)
    if cached is not None:
        return cached

    root = Path(tempfile.mkdtemp(dir=TESTS_ROOT))
    _ROOTS.append(root)
    if raw_depth:
        session_path = root / "case.vgsession"
        shutil.copytree(FIXTURE, session_path)
        _write_depth(session_path, raw_depth)
    else:
        session_path = FIXTURE

    plan_path = root / "fixture.sftplan"
    plan_tsdf_blocks(
        load_scan_session(session_path),
        plan_path,
        frame_stride=frame_stride,
        **PLAN_ARGUMENTS,
    )
    plan = load_tsdf_block_plan(plan_path)
    context = build_tsdf_replay_depth_context(
        plan,
        load_scan_session(session_path),
    )
    case = SharedCase(
        root=root,
        session_path=session_path,
        plan_path=plan_path,
        plan=plan,
        context=context,
    )
    _CASES[key] = case
    return case


@atexit.register
def _release_shared_cases() -> None:
    _CASES.clear()
    while _ROOTS:
        shutil.rmtree(_ROOTS.pop(), ignore_errors=True)
