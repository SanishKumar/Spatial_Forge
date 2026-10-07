from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

from pathlib import Path
from unittest.mock import patch

from spatialforge import (
    allocate_empty_tsdf_blocks,
    load_tsdf_block_plan,
    verify_tsdf_block_plan_replay,
)
from spatialforge.cli import main
from spatialforge.errors import TsdfError
from spatialforge.session_loader import load_scan_session
from spatialforge.tsdf_block_plan import (
    TSDF_BLOCK_PLAN_SCHEMA_VERSION,
    TSDF_EXPANDED_BLOCK_PLAN_SCHEMA_VERSION,
    TSDF_FREE_SPACE_RULE_FOOTPRINT,
    TSDF_FREE_SPACE_RULE_NOT_PLANNED,
)
from spatialforge.tsdf_expanded_plan import (
    TsdfExpandedPlanReport,
    write_tsdf_expanded_block_plan,
)
from spatialforge.tsdf_plan_expansion import (
    propose_tsdf_plan_expansion_from_domain,
)

from tests.heavy_fixtures import shared_case

TEST_ROOT = Path(__file__).resolve().parent
FIXTURE = TEST_ROOT / "fixtures" / "minimal.vgsession"


def proposal_for(raw_depth: int = 3000, frame_stride: int = 2):
    case = shared_case(raw_depth, frame_stride)
    return case, propose_tsdf_plan_expansion_from_domain(
        case.plan,
        case.domain,
    )


class TsdfExpandedPlanTests(unittest.TestCase):
    def test_expanded_plan_round_trips_and_verifies(self) -> None:
        case, proposal = proposal_for()
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            output = Path(temporary) / "expanded.sftplan"
            source_before = case.plan_path.read_bytes()

            report = write_tsdf_expanded_block_plan(
                case.plan,
                proposal,
                output,
            )
            reloaded = load_tsdf_block_plan(output)
            verify_tsdf_block_plan_replay(
                reloaded,
                load_scan_session(case.session_path),
            )
            storage = allocate_empty_tsdf_blocks(
                reloaded,
                load_scan_session(case.session_path),
            )
            source_after = case.plan_path.read_bytes()

        self.assertIsInstance(report, TsdfExpandedPlanReport)
        self.assertEqual(report.source_block_count, 32)
        self.assertEqual(report.expanded_block_count, 40)
        self.assertEqual(report.added_block_count, 8)
        self.assertEqual(report.expanded_voxel_slots, 20480)
        self.assertEqual(report.free_space_rule, TSDF_FREE_SPACE_RULE_FOOTPRINT)

        self.assertEqual(
            reloaded.active_blocks,
            proposal.expanded_block_indices,
        )
        self.assertEqual(reloaded.surface_blocks, case.plan.surface_blocks)
        self.assertEqual(reloaded.planned_voxel_slots, 20480)
        self.assertEqual(
            reloaded.free_space_rule,
            TSDF_FREE_SPACE_RULE_FOOTPRINT,
        )
        self.assertEqual(
            reloaded.expanded_from_plan_sha256,
            case.plan.artifact_digest_sha256,
        )
        self.assertEqual(
            reloaded.artifact_digest_sha256,
            report.output_digest_sha256,
        )
        self.assertEqual(storage.voxel_slots, 20480)
        self.assertEqual(source_after, source_before)

    def test_carried_fields_match_the_source_plan(self) -> None:
        case, proposal = proposal_for()
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            output = Path(temporary) / "expanded.sftplan"
            write_tsdf_expanded_block_plan(case.plan, proposal, output)
            reloaded = load_tsdf_block_plan(output)

        for field in (
            "session_id",
            "replay_digest_sha256",
            "voxel_size_m",
            "block_resolution",
            "block_extent_m",
            "truncation_m",
            "frame_stride",
            "total_observations",
            "selected_observations",
            "paired_observations",
            "skipped_missing_depth",
            "skipped_missing_pose",
            "valid_depth_points",
            "invalid_depth_samples",
            "surface_blocks",
        ):
            with self.subTest(field=field):
                self.assertEqual(
                    getattr(reloaded, field),
                    getattr(case.plan, field),
                )
        self.assertNotEqual(
            reloaded.active_blocks,
            case.plan.active_blocks,
        )
        self.assertNotEqual(
            reloaded.free_space_rule,
            case.plan.free_space_rule,
        )
        self.assertEqual(
            case.plan.free_space_rule,
            TSDF_FREE_SPACE_RULE_NOT_PLANNED,
        )
        self.assertIsNone(case.plan.expanded_from_plan_sha256)

    def test_writing_is_deterministic(self) -> None:
        case, proposal = proposal_for()
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            root = Path(temporary)
            first = write_tsdf_expanded_block_plan(
                case.plan,
                proposal,
                root / "a.sftplan",
            )
            second = write_tsdf_expanded_block_plan(
                case.plan,
                proposal,
                root / "b.sftplan",
            )
            self.assertEqual(
                (root / "a.sftplan").read_bytes(),
                (root / "b.sftplan").read_bytes(),
            )
        self.assertEqual(
            first.output_digest_sha256,
            second.output_digest_sha256,
        )

    def test_source_plan_is_never_overwritten(self) -> None:
        case, proposal = proposal_for()
        with self.assertRaises(TsdfError) as same_path:
            write_tsdf_expanded_block_plan(
                case.plan,
                proposal,
                case.plan_path,
            )
        self.assertIn("already exists", str(same_path.exception))

        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            output = Path(temporary) / "expanded.sftplan"
            write_tsdf_expanded_block_plan(case.plan, proposal, output)
            with self.assertRaises(TsdfError) as existing:
                write_tsdf_expanded_block_plan(case.plan, proposal, output)
            self.assertIn("already exists", str(existing.exception))

    def test_output_suffix_and_provenance_are_preflighted(self) -> None:
        case, proposal = proposal_for()
        other_case, other_proposal = proposal_for(0, 1)
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            root = Path(temporary)
            cases = (
                (
                    "plan-type",
                    object(),
                    proposal,
                    root / "a.sftplan",
                    "TsdfBlockPlan",
                ),
                (
                    "proposal-type",
                    case.plan,
                    object(),
                    root / "b.sftplan",
                    "TsdfPlanExpansionProposal",
                ),
                (
                    "foreign-proposal",
                    case.plan,
                    other_proposal,
                    root / "c.sftplan",
                    "provenance does not match",
                ),
                (
                    "suffix",
                    case.plan,
                    proposal,
                    root / "d.json",
                    ".sftplan",
                ),
            )
            for name, plan, candidate, output, message in cases:
                with self.subTest(name=name):
                    with self.assertRaises(TsdfError) as raised:
                        write_tsdf_expanded_block_plan(
                            plan,  # type: ignore[arg-type]
                            candidate,  # type: ignore[arg-type]
                            output,
                        )
                    self.assertIn(message, str(raised.exception))
                    self.assertFalse(output.exists())
        self.assertIsNotNone(other_case.plan)

    def test_expanded_plan_is_rejected_against_a_different_session(
        self,
    ) -> None:
        case, proposal = proposal_for()
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            output = Path(temporary) / "expanded.sftplan"
            write_tsdf_expanded_block_plan(case.plan, proposal, output)
            reloaded = load_tsdf_block_plan(output)
            with self.assertRaises(Exception) as raised:
                verify_tsdf_block_plan_replay(
                    reloaded,
                    load_scan_session(FIXTURE),
                )
        self.assertTrue(str(raised.exception))

    def test_write_failure_leaves_no_partial_artifact(self) -> None:
        case, proposal = proposal_for()
        injected = OSError("injected write failure")
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            root = Path(temporary)
            output = root / "expanded.sftplan"
            with patch(
                "spatialforge.tsdf_expanded_plan."
                "_write_plan_without_overwrite",
                side_effect=injected,
            ):
                with self.assertRaises(OSError):
                    write_tsdf_expanded_block_plan(
                        case.plan,
                        proposal,
                        output,
                    )
            self.assertFalse(output.exists())
            self.assertEqual(list(root.iterdir()), [])


class TsdfExpandedPlanCliTests(unittest.TestCase):
    def test_cli_writes_and_reports_the_expanded_plan(self) -> None:
        case = shared_case(3000, 2)
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            output = Path(temporary) / "expanded.sftplan"
            stdout = io.StringIO()
            stderr = io.StringIO()
            source_before = case.plan_path.read_bytes()

            with redirect_stdout(stdout), redirect_stderr(stderr):
                exit_code = main(
                    [
                        "reconstruct",
                        "tsdf-block-plan-expand",
                        str(case.plan_path),
                        str(case.session_path),
                        str(output),
                    ]
                )
            written = output.exists()
            reloaded = load_tsdf_block_plan(output)
            source_after = case.plan_path.read_bytes()

        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr.getvalue(), "")
        self.assertTrue(written)
        self.assertEqual(source_after, source_before)
        self.assertEqual(len(reloaded.active_blocks), 40)

        output_text = stdout.getvalue()
        for expected in (
            "TSDF BLOCK PLAN EXPAND scan-synthetic-0001\n",
            "approval_rule: covered-block-with-at-least-one-observed-voxel\n",
            "coverage_domain: blocks=52 approved=40 rejected=12\n",
            "source_plan: blocks=32 surface=4\n",
            "expanded_plan: blocks=40 added=8 voxel_slots=20480\n",
            "free_space_rule: conservative-nearest-pixel-footprint\n",
            "surface_blocks_retained: yes\n",
            "source_plan_mutated: no\n",
            "source_plan_overwritten: no\n",
            "storage_allocated: no\n",
            "full_fusion_performed: no\n",
        ):
            self.assertIn(expected, output_text)

    def test_cli_refuses_an_existing_output_without_writing(self) -> None:
        case = shared_case(3000, 2)
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            output = Path(temporary) / "expanded.sftplan"
            output.write_text("occupied", encoding="ascii")
            stdout = io.StringIO()
            stderr = io.StringIO()

            with redirect_stdout(stdout), redirect_stderr(stderr):
                exit_code = main(
                    [
                        "reconstruct",
                        "tsdf-block-plan-expand",
                        str(case.plan_path),
                        str(case.session_path),
                        str(output),
                    ]
                )
            preserved = output.read_text(encoding="ascii")

        self.assertEqual(exit_code, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(preserved, "occupied")
        error = stderr.getvalue()
        self.assertIn("TSDF BLOCK PLAN EXPAND FAILED", error)
        self.assertIn("already exists", error)
        self.assertNotIn("Traceback", error)


class TsdfPlanFreeSpaceRuleTests(unittest.TestCase):
    def test_loader_rejects_an_unknown_free_space_rule(self) -> None:
        case, proposal = proposal_for()
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            output = Path(temporary) / "expanded.sftplan"
            write_tsdf_expanded_block_plan(case.plan, proposal, output)
            text = output.read_text(encoding="ascii")
            tampered = Path(temporary) / "tampered.sftplan"
            tampered.write_text(
                text.replace(
                    TSDF_FREE_SPACE_RULE_FOOTPRINT,
                    "carve-everything",
                ),
                encoding="ascii",
            )
            with self.assertRaises(TsdfError) as raised:
                load_tsdf_block_plan(tampered)

        self.assertIn("free_space_rule", str(raised.exception))
        self.assertIn("expected one of", str(raised.exception))

    def test_loader_rejects_tampered_expansion_provenance(self) -> None:
        case, proposal = proposal_for()
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            output = Path(temporary) / "expanded.sftplan"
            write_tsdf_expanded_block_plan(case.plan, proposal, output)
            text = output.read_text(encoding="ascii")
            cases = (
                (
                    "rule",
                    text.replace(
                        "covered-block-with-at-least-one-observed-voxel",
                        "anything-goes",
                    ),
                    "approval_rule",
                ),
                (
                    "digest",
                    text.replace(
                        case.plan.artifact_digest_sha256,
                        "zz" + case.plan.artifact_digest_sha256[2:],
                    ),
                    "source_plan_sha256",
                ),
            )
            for name, tampered_text, message in cases:
                with self.subTest(name=name):
                    tampered = Path(temporary) / f"{name}.sftplan"
                    tampered.write_text(tampered_text, encoding="ascii")
                    with self.assertRaises(TsdfError) as raised:
                        load_tsdf_block_plan(tampered)
                    self.assertIn(message, str(raised.exception))

    def test_original_plans_still_declare_no_free_space(self) -> None:
        # TsdfBlockPlan itself is a plain frozen record; the strict checks
        # live in the loader, which the tampering tests above exercise.
        case = shared_case()
        self.assertEqual(
            case.plan.free_space_rule,
            TSDF_FREE_SPACE_RULE_NOT_PLANNED,
        )
        self.assertIsNone(case.plan.expanded_from_plan_sha256)


def _rewrite(path: Path, document: dict) -> None:
    path.write_bytes(
        (json.dumps(document, indent=2, sort_keys=True) + "\n").encode(
            "ascii"
        )
    )


class TsdfPlanKindTests(unittest.TestCase):
    """A plan is one kind or the other, and its fields must agree."""

    def test_each_kind_declares_its_own_schema_version(self) -> None:
        case, proposal = proposal_for()
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            output = Path(temporary) / "expanded.sftplan"
            write_tsdf_expanded_block_plan(case.plan, proposal, output)
            expanded = json.loads(output.read_bytes())
        original = json.loads(case.plan_path.read_bytes())

        self.assertEqual(TSDF_BLOCK_PLAN_SCHEMA_VERSION, "0.1.0")
        self.assertEqual(TSDF_EXPANDED_BLOCK_PLAN_SCHEMA_VERSION, "0.2.0")
        self.assertEqual(original["schema_version"], "0.1.0")
        self.assertNotIn("expansion", original)
        self.assertEqual(expanded["schema_version"], "0.2.0")
        self.assertIn("expansion", expanded)

    def test_loader_refuses_plans_that_contradict_themselves(self) -> None:
        case, proposal = proposal_for()
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            folder = Path(temporary)
            output = folder / "expanded.sftplan"
            write_tsdf_expanded_block_plan(case.plan, proposal, output)

            def original() -> dict:
                return json.loads(case.plan_path.read_bytes())

            def expanded() -> dict:
                return json.loads(output.read_bytes())

            # Rewriting a genuine plan without changing it must still
            # load, or the refusals below would prove nothing about the
            # one field each of them changes.
            controls = (
                ("original", original()),
                ("expanded", expanded()),
            )
            for name, document in controls:
                with self.subTest(control=name):
                    path = folder / f"control-{name}.sftplan"
                    _rewrite(path, document)
                    load_tsdf_block_plan(path)

            def with_version(document: dict, version: str) -> dict:
                document["schema_version"] = version
                return document

            def with_rule(document: dict, rule: str) -> dict:
                document["activation"]["free_space_rule"] = rule
                return document

            def without_expansion(document: dict) -> dict:
                del document["expansion"]
                return document

            def with_expansion(document: dict) -> dict:
                document["expansion"] = expanded()["expansion"]
                return document

            def with_added(document: dict, added: int) -> dict:
                document["expansion"]["added_blocks"] = added
                return document

            halo = expanded()["planning"]["halo_blocks"]
            cases = (
                (
                    "expanded-under-the-original-version",
                    with_version(expanded(), "0.1.0"),
                    "expansion: not permitted by schema version '0.1.0'",
                ),
                (
                    "expanded-without-provenance",
                    without_expansion(expanded()),
                    "expansion: required by schema version '0.2.0'",
                ),
                (
                    "expanded-claiming-no-free-space",
                    with_rule(expanded(), TSDF_FREE_SPACE_RULE_NOT_PLANNED),
                    "free_space_rule: schema version '0.2.0'",
                ),
                (
                    "expanded-adding-more-than-it-holds",
                    with_added(expanded(), halo + 1),
                    f"added_blocks: {halo + 1} blocks added, but only {halo}",
                ),
                (
                    "original-under-the-expanded-version",
                    with_version(original(), "0.2.0"),
                    "expansion: required by schema version '0.2.0'",
                ),
                (
                    "original-claiming-free-space",
                    with_rule(original(), TSDF_FREE_SPACE_RULE_FOOTPRINT),
                    "free_space_rule: schema version '0.1.0'",
                ),
                (
                    "original-with-provenance",
                    with_expansion(original()),
                    "expansion: not permitted by schema version '0.1.0'",
                ),
                (
                    "a-version-nobody-wrote",
                    with_version(expanded(), "0.3.0"),
                    "schema_version: expected one of",
                ),
            )
            for name, document, message in cases:
                with self.subTest(name=name):
                    path = folder / f"{name}.sftplan"
                    _rewrite(path, document)
                    with self.assertRaises(TsdfError) as raised:
                        load_tsdf_block_plan(path)
                    self.assertIn(message, str(raised.exception))

    def test_an_expansion_may_add_every_block_that_is_not_surface(self) -> None:
        case, proposal = proposal_for()
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            folder = Path(temporary)
            output = folder / "expanded.sftplan"
            write_tsdf_expanded_block_plan(case.plan, proposal, output)
            document = json.loads(output.read_bytes())
            document["expansion"]["added_blocks"] = document["planning"][
                "halo_blocks"
            ]
            path = folder / "boundary.sftplan"
            _rewrite(path, document)
            reloaded = load_tsdf_block_plan(path)

        self.assertEqual(
            reloaded.expanded_from_plan_sha256,
            case.plan.artifact_digest_sha256,
        )


if __name__ == "__main__":
    unittest.main()
