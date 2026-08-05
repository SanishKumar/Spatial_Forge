from __future__ import annotations

import io
import shutil
import tempfile
import unittest
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path
from unittest.mock import MagicMock, patch

from spatialforge import build_tsdf_replay_depth_context, load_tsdf_block_plan
from spatialforge.cli import main
from spatialforge.errors import TsdfError
from spatialforge.session_loader import load_scan_session
from spatialforge.tsdf_block_plan import plan_tsdf_blocks
from spatialforge.tsdf_block_plan_loader import TsdfBlockPlan
from spatialforge.tsdf_domain_cross_view import (
    TsdfCoverageDomainCrossViewReceipt,
    sweep_tsdf_coverage_domain_cross_view_from_context,
)
from spatialforge.tsdf_plan_expansion import (
    TsdfPlanExpansionProposal,
    propose_tsdf_plan_expansion_from_domain,
)
from spatialforge.tsdf_plan_footprint_survey import (
    survey_tsdf_plan_pixel_footprints_from_context,
)
from spatialforge.tsdf_replay_depth_context import TsdfReplayDepthContext

from tests.heavy_fixtures import shared_case


TEST_ROOT = Path(__file__).resolve().parent
FIXTURE = TEST_ROOT / "fixtures" / "minimal.vgsession"
PLAN_ARGUMENTS = {
    "voxel_size_m": 0.125,
    "truncation_m": 0.5,
}
PLAN_SHA256 = (
    "372c7c5d49eff1a30317ceb8b67cb3c40683f1d9d2a6179a049772f763d6f79d"
)
REPLAY_SHA256 = (
    "dc001ae0ca01004a21ad227b12d57a7350bbc23904040453ca73fd955ef050b8"
)

_FORBIDDEN_CLI_TARGETS = (
    "spatialforge.cli.allocate_empty_tsdf_blocks",
    "spatialforge.cli.locate_tsdf_voxel",
    "spatialforge.cli.apply_tsdf_voxel_contribution",
    "spatialforge.cli.apply_tsdf_voxel_contribution_from_context",
    "spatialforge.cli.traverse_tsdf_voxel_observations",
    "spatialforge.cli.traverse_tsdf_voxel_observations_from_context",
    "spatialforge.cli.traverse_tsdf_block_voxels_from_context",
    "spatialforge.cli.traverse_tsdf_plan_blocks_from_context",
    "spatialforge.cli.reconstruct_point_cloud",
    "spatialforge.cli.integrate_tsdf",
    "spatialforge.cli.integrate_sparse_tsdf",
    "spatialforge.cli.infer_tsdf_bounds",
    "spatialforge.cli.plan_tsdf_blocks",
    "spatialforge.cli.extract_surface_points",
    "spatialforge.cli.extract_triangle_mesh",
    "spatialforge.tsdf._write_output_without_overwrite",
    "spatialforge.tsdf_block_plan._write_plan_without_overwrite",
    "spatialforge.surface._write_surface_ply",
    "spatialforge.mesh._write_mesh_ply",
)


def copy_fixture(parent: Path, name: str = "case.vgsession") -> Path:
    target = parent / name
    shutil.copytree(FIXTURE, target)
    return target


def create_plan(
    parent: Path,
    *,
    session_path: Path = FIXTURE,
    name: str = "fixture.sftplan",
    frame_stride: int = 1,
) -> Path:
    output = parent / name
    plan_tsdf_blocks(
        load_scan_session(session_path),
        output,
        frame_stride=frame_stride,
        **PLAN_ARGUMENTS,
    )
    return output


def load_case(
    parent: Path,
    *,
    session_path: Path = FIXTURE,
    frame_stride: int = 1,
    name: str = "fixture.sftplan",
) -> tuple[TsdfBlockPlan, TsdfReplayDepthContext]:
    plan = load_tsdf_block_plan(
        create_plan(
            parent,
            session_path=session_path,
            frame_stride=frame_stride,
            name=name,
        )
    )
    session = load_scan_session(session_path)
    context = build_tsdf_replay_depth_context(plan, session)
    return plan, context


def resolve_domain(
    plan: TsdfBlockPlan,
    context: TsdfReplayDepthContext,
) -> TsdfCoverageDomainCrossViewReceipt:
    coverage = survey_tsdf_plan_pixel_footprints_from_context(plan, context)
    return sweep_tsdf_coverage_domain_cross_view_from_context(
        plan,
        context,
        coverage,
    )


def tree_snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def plan_snapshot(plan: TsdfBlockPlan) -> tuple[tuple[str, object], ...]:
    return tuple((field.name, getattr(plan, field.name)) for field in fields(plan))


@contextmanager
def forbidden_cli_calls():
    with ExitStack() as stack:
        mocks: list[MagicMock] = []
        for target in _FORBIDDEN_CLI_TARGETS:
            mocks.append(stack.enter_context(patch(target, create=True)))
        yield mocks


class TsdfPlanExpansionTests(unittest.TestCase):
    def test_fixture_plan_already_covers_its_own_domain(self) -> None:
        case = shared_case()
        temporary_root, plan, domain = case.root, case.plan, case.domain
        before_tree = tree_snapshot(temporary_root)
        before_plan = plan_snapshot(plan)

        proposal = propose_tsdf_plan_expansion_from_domain(plan, domain)
        repeated = propose_tsdf_plan_expansion_from_domain(plan, domain)
        after_tree = tree_snapshot(temporary_root)

        self.assertIsInstance(proposal, TsdfPlanExpansionProposal)
        self.assertEqual(proposal, repeated)
        self.assertEqual(proposal.source_plan_digest_sha256, PLAN_SHA256)
        self.assertEqual(proposal.replay_digest_sha256, REPLAY_SHA256)
        self.assertEqual(proposal.source_block_count, 8)
        self.assertEqual(proposal.domain_block_count, 8)
        self.assertEqual(proposal.approved_block_count, 8)
        self.assertEqual(proposal.rejected_block_count, 0)
        self.assertEqual(proposal.expanded_block_count, 8)
        self.assertEqual(proposal.added_block_count, 0)
        self.assertEqual(proposal.added_block_indices, ())
        self.assertEqual(proposal.removed_block_count, 0)
        self.assertEqual(proposal.source_voxel_slots, 4096)
        self.assertEqual(proposal.expanded_voxel_slots, 4096)
        self.assertEqual(proposal.added_voxel_slots, 0)
        self.assertFalse(proposal.expands_plan)
        self.assertEqual(
            proposal.expanded_block_indices,
            plan.active_blocks,
        )
        self.assertEqual(plan_snapshot(plan), before_plan)
        self.assertEqual(after_tree, before_tree)

    def test_far_domain_adds_free_space_blocks_and_prunes_evidence_free_ones(
        self,
    ) -> None:
        case = shared_case(3000, 2)
        temporary_root, plan, domain = case.root, case.plan, case.domain
        before_plan = plan_snapshot(plan)
        before_tree = tree_snapshot(temporary_root)

        proposal = propose_tsdf_plan_expansion_from_domain(plan, domain)
        after_tree = tree_snapshot(temporary_root)

        self.assertEqual(proposal.source_block_count, 32)
        self.assertEqual(proposal.domain_block_count, 52)
        self.assertEqual(proposal.approved_block_count, 40)
        self.assertEqual(proposal.rejected_block_count, 12)
        self.assertEqual(proposal.expanded_block_count, 40)
        self.assertEqual(proposal.added_block_count, 8)
        self.assertEqual(proposal.source_voxel_slots, 16384)
        self.assertEqual(proposal.expanded_voxel_slots, 20480)
        self.assertEqual(proposal.added_voxel_slots, 4096)
        self.assertTrue(proposal.expands_plan)
        self.assertEqual(
            set(proposal.added_block_indices) & set(plan.active_blocks),
            set(),
        )
        # Conservative coverage deliberately over-includes; the per-voxel
        # verdict prunes the blocks its wedges only graze.
        self.assertEqual(
            set(proposal.rejected_block_indices) & set(plan.active_blocks),
            set(),
        )
        self.assertEqual(plan_snapshot(plan), before_plan)
        self.assertEqual(after_tree, before_tree)

    def test_approval_rule_is_exactly_one_observed_voxel(self) -> None:
        case = shared_case(3000, 2)
        plan, domain = case.plan, case.domain
        proposal = propose_tsdf_plan_expansion_from_domain(plan, domain)

        approved = set(proposal.approved_block_indices)
        rejected = set(proposal.rejected_block_indices)
        observed_blocks = set()
        for receipt in domain.block_receipts:
            if receipt.observed_voxel_count > 0:
                observed_blocks.add(receipt.block_index_xyz)
                self.assertIn(receipt.block_index_xyz, approved)
                self.assertNotIn(receipt.block_index_xyz, rejected)
            else:
                self.assertIn(receipt.block_index_xyz, rejected)
                self.assertNotIn(receipt.block_index_xyz, approved)
                self.assertEqual(receipt.surface_voxel_count, 0)
                self.assertEqual(receipt.free_space_voxel_count, 0)
                self.assertEqual(
                    receipt.carvable_free_space_voxel_count,
                    0,
                )
        self.assertEqual(approved, observed_blocks)
        self.assertEqual(
            approved | rejected,
            set(domain.domain_block_indices),
        )

    def test_expansion_never_drops_a_source_plan_block(self) -> None:
        cases = ((0, 1, "near"), (3000, 2, "far"))
        for raw_depth, frame_stride, name in cases:
            with self.subTest(name=name):
                case = shared_case(raw_depth, frame_stride)
                plan, domain = case.plan, case.domain
                proposal = propose_tsdf_plan_expansion_from_domain(
                    plan,
                    domain,
                )

                expanded = set(proposal.expanded_block_indices)
                self.assertTrue(set(plan.active_blocks) <= expanded)
                self.assertEqual(proposal.removed_block_count, 0)
                self.assertEqual(
                    proposal.retained_block_count,
                    len(plan.active_blocks),
                )
                self.assertEqual(
                    expanded,
                    set(plan.active_blocks)
                    | set(proposal.approved_block_indices),
                )
                self.assertEqual(
                    proposal.expanded_block_count,
                    proposal.source_block_count + proposal.added_block_count,
                )

    def test_proposal_is_bound_to_its_own_coverage_domain(self) -> None:
        case = shared_case()
        temporary_root, plan, domain = case.root, case.plan, case.domain
        foreign_domain = shared_case(3000, 2).domain
        before_tree = tree_snapshot(temporary_root)

        with self.assertRaises(TsdfError) as foreign:
            propose_tsdf_plan_expansion_from_domain(plan, foreign_domain)
        with self.assertRaises(TsdfError) as domain_type:
            propose_tsdf_plan_expansion_from_domain(
                plan,
                object(),  # type: ignore[arg-type]
            )
        with self.assertRaises(TsdfError) as plan_type:
            propose_tsdf_plan_expansion_from_domain(
                object(),  # type: ignore[arg-type]
                domain,
            )
        after_tree = tree_snapshot(temporary_root)

        self.assertIn("provenance does not match", str(foreign.exception))
        self.assertIn(
            "TsdfCoverageDomainCrossViewReceipt",
            str(domain_type.exception),
        )
        self.assertIn("TsdfBlockPlan", str(plan_type.exception))
        self.assertEqual(after_tree, before_tree)

    def test_proposals_are_frozen_slotted_and_strict(self) -> None:
        case = shared_case()
        plan, domain = case.plan, case.domain
        proposal = propose_tsdf_plan_expansion_from_domain(plan, domain)

        self.assertFalse(hasattr(proposal, "__dict__"))
        with self.assertRaises(FrozenInstanceError):
            proposal.frame_stride = 2  # type: ignore[misc]

        for arguments in (
            {"expanded_block_indices": proposal.source_plan_block_indices[:4]},
            {
                "expanded_block_indices": tuple(
                    reversed(proposal.expanded_block_indices)
                )
            },
            {"approved_block_indices": ()},
            {"rejected_block_indices": proposal.domain_block_indices},
            {"source_plan_block_indices": ()},
            {"domain_block_indices": ()},
            {"block_resolution": 7},
            {"frame_stride": 0},
            {"total_observations": 0},
            {"source_plan_digest_sha256": "0"},
            {"replay_digest_sha256": "0"},
        ):
            with self.subTest(arguments=arguments):
                with self.assertRaises(TsdfError):
                    replace(proposal, **arguments)

    def test_late_proposal_failure_leaves_inputs_unchanged(self) -> None:
        case = shared_case()
        temporary_root, plan, domain = case.root, case.plan, case.domain
        before_tree = tree_snapshot(temporary_root)
        before_plan = plan_snapshot(plan)
        injected = RuntimeError("injected final expansion failure")

        with patch(
            "spatialforge.tsdf_plan_expansion.TsdfPlanExpansionProposal",
            side_effect=injected,
        ):
            with self.assertRaises(TsdfError) as raised:
                propose_tsdf_plan_expansion_from_domain(plan, domain)
        after_tree = tree_snapshot(temporary_root)

        self.assertIs(raised.exception.__cause__, injected)
        self.assertIn(
            "injected final expansion failure",
            str(raised.exception),
        )
        self.assertEqual(plan_snapshot(plan), before_plan)
        self.assertEqual(after_tree, before_tree)


class TsdfPlanExpansionCliTests(unittest.TestCase):
    def test_cli_reports_exact_read_only_expansion_proposal(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            session_path = copy_fixture(temporary_root)
            plan_path = create_plan(
                temporary_root,
                session_path=session_path,
            )
            before_tree = tree_snapshot(temporary_root)
            stdout = io.StringIO()
            stderr = io.StringIO()
            actual = propose_tsdf_plan_expansion_from_domain

            with (
                forbidden_cli_calls() as forbidden,
                patch(
                    "spatialforge.cli."
                    "propose_tsdf_plan_expansion_from_domain",
                    wraps=actual,
                ) as proposer,
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                exit_code = main(
                    [
                        "reconstruct",
                        "tsdf-block-context-plan-expansion",
                        str(plan_path),
                        str(session_path),
                    ]
                )
            after_tree = tree_snapshot(temporary_root)

        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(after_tree, before_tree)
        proposer.assert_called_once()
        for forbidden_call in forbidden:
            forbidden_call.assert_not_called()

        output = stdout.getvalue()
        for expected in (
            "TSDF BLOCK CONTEXT PLAN EXPANSION CHECK scan-synthetic-0001\n",
            "coverage_source: conservative-pixel-footprint-survey\n",
            "approval_rule: covered-block-with-at-least-one-observed-voxel\n",
            "source_plan: blocks=8 voxel_slots=4096\n",
            "coverage_domain: blocks=8 approved=8 rejected=0\n",
            "domain_voxels: observed=584 carvable_free_space=24\n",
            "proposed_plan: blocks=8 voxel_slots=4096\n",
            "proposed_delta: added=0 retained=8 removed=0 "
            "added_voxel_slots=0\n",
            "source_blocks_retained: yes\n",
            "expands_plan: no\n",
            "expansion_scope: proposal-only\n",
            "evidence_threshold_applied: no\n",
            "plan_written: no\n",
            "plan_expanded_on_disk: no\n",
            "source_plan_mutated: no\n",
            "storage_allocated: no\n",
            "storage_mutated: no\n",
            "full_fusion_performed: no\n",
            "artifact_written: no\n",
            f"plan_sha256: {PLAN_SHA256}\n",
            f"replay_digest_sha256: {REPLAY_SHA256}\n",
        ):
            self.assertIn(expected, output)

    def test_cli_reports_failure_without_writing_or_traceback(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            session_path = copy_fixture(temporary_root)
            plan_path = create_plan(
                temporary_root,
                session_path=session_path,
            )
            before_tree = tree_snapshot(temporary_root)
            stdout = io.StringIO()
            stderr = io.StringIO()

            with (
                patch(
                    "spatialforge.cli."
                    "propose_tsdf_plan_expansion_from_domain",
                    side_effect=TsdfError("injected expansion failure"),
                ) as proposer,
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                exit_code = main(
                    [
                        "reconstruct",
                        "tsdf-block-context-plan-expansion",
                        str(plan_path),
                        str(session_path),
                    ]
                )
            after_tree = tree_snapshot(temporary_root)

        self.assertEqual(exit_code, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(after_tree, before_tree)
        proposer.assert_called_once()
        error = stderr.getvalue()
        self.assertIn("TSDF BLOCK CONTEXT PLAN EXPANSION FAILED", error)
        self.assertIn("injected expansion failure", error)
        self.assertNotIn("Traceback", error)


if __name__ == "__main__":
    unittest.main()
