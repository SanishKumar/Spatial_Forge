from __future__ import annotations

import copy
import hashlib
import io
import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from unittest.mock import patch

from spatialforge.cli import main
from spatialforge.errors import TsdfError
from spatialforge.session_loader import load_scan_session
from spatialforge.tsdf_block_plan import plan_tsdf_blocks
from spatialforge.tsdf_block_plan_loader import (
    _MAX_JSON_OBJECTS,
    load_tsdf_block_plan,
    verify_tsdf_block_plan_replay,
)


TEST_ROOT = Path(__file__).resolve().parent
FIXTURE = TEST_ROOT / "fixtures" / "minimal.vgsession"
PLAN_ARGUMENTS = {
    "voxel_size_m": 0.125,
    "truncation_m": 0.5,
}
PLAN_SHA256 = "372c7c5d49eff1a30317ceb8b67cb3c40683f1d9d2a6179a049772f763d6f79d"
REPLAY_SHA256 = "dc001ae0ca01004a21ad227b12d57a7350bbc23904040453ca73fd955ef050b8"


def copy_fixture(parent: Path, name: str = "case.vgsession") -> Path:
    target = parent / name
    shutil.copytree(FIXTURE, target)
    return target


def create_plan(
    parent: Path,
    *,
    session_path: Path = FIXTURE,
    name: str = "fixture.sftplan",
) -> Path:
    output = parent / name
    plan_tsdf_blocks(
        load_scan_session(session_path),
        output,
        **PLAN_ARGUMENTS,
    )
    return output


def read_document(path: Path) -> dict:
    return json.loads(path.read_text(encoding="ascii"))


def write_document(path: Path, document: dict) -> None:
    path.write_text(
        json.dumps(
            document,
            ensure_ascii=True,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="ascii",
    )


class TsdfBlockPlanLoaderTests(unittest.TestCase):
    def test_generated_plan_loads_immutably_with_exact_artifact_digest(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            path = create_plan(temporary_root)
            encoded = path.read_bytes()
            plan = load_tsdf_block_plan(path)

        self.assertEqual(plan.artifact_digest_sha256, PLAN_SHA256)
        self.assertEqual(
            plan.artifact_digest_sha256,
            hashlib.sha256(encoded).hexdigest(),
        )
        self.assertEqual(plan.session_id, "scan-synthetic-0001")
        self.assertEqual(plan.replay_digest_sha256, REPLAY_SHA256)
        self.assertEqual(plan.voxel_size_m, 0.125)
        self.assertEqual(plan.block_resolution, 8)
        self.assertEqual(plan.block_extent_m, 1.0)
        self.assertEqual(plan.truncation_m, 0.5)
        self.assertEqual(plan.surface_block_count, 4)
        self.assertEqual(plan.active_block_count, 8)
        self.assertEqual(plan.halo_block_count, 4)
        self.assertEqual(plan.planned_voxel_slots, 4096)
        self.assertEqual(plan.min_block_index, (0, -1, -1))
        self.assertEqual(plan.max_block_index, (1, 0, 0))
        self.assertIsInstance(plan.surface_blocks, tuple)
        self.assertIsInstance(plan.surface_blocks[0], tuple)
        with self.assertRaises(FrozenInstanceError):
            plan.session_id = "changed"  # type: ignore[misc]
        with self.assertRaises(TypeError):
            plan.active_blocks[0] = (9, 9, 9)  # type: ignore[index]
        self.assertFalse(hasattr(plan, "__dict__"))

    def test_malformed_bytes_json_duplicates_and_nonfinite_values_fail(
        self,
    ) -> None:
        malformed_cases = (
            ("non-ascii", b"\xff", "ASCII JSON"),
            ("syntax", b"{", "invalid JSON"),
            (
                "duplicate-root",
                (
                    b'{"schema":"spatialforge.tsdf-block-plan",'
                    b'"schema":"spatialforge.tsdf-block-plan"}'
                ),
                "duplicate JSON key",
            ),
            (
                "duplicate-nested",
                (
                    b'{"grid":{"voxel_size_m":0.125,'
                    b'"voxel_size_m":0.125}}'
                ),
                "duplicate JSON key",
            ),
            (
                "nan",
                b'{"voxel_size_m":NaN}',
                "non-finite",
            ),
            (
                "infinity",
                b'{"voxel_size_m":Infinity}',
                "non-finite",
            ),
            (
                "overflow-number",
                b'{"grid":{"voxel_size_m":1e400}}',
                "missing required field",
            ),
            (
                "top-level-array",
                b"[]",
                "expected an object",
            ),
            (
                "deeply-nested",
                ("[" * 1100 + "0" + "]" * 1100).encode("ascii"),
                "nested too deeply",
            ),
        )
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            for name, encoded, message in malformed_cases:
                with self.subTest(name=name):
                    path = temporary_root / f"{name}.sftplan"
                    path.write_bytes(encoded)
                    with self.assertRaises(TsdfError) as raised:
                        load_tsdf_block_plan(path)
                    self.assertIn(message, str(raised.exception))

    def test_structural_limits_reject_memory_amplification_before_parse(
        self,
    ) -> None:
        cases = (
            (
                "arrays",
                "[" + ",".join("[]" for _ in range(8)) + "]",
                "too many arrays",
            ),
            (
                # One more object than a plan may legitimately contain:
                # root, grid, activation, planning, and the expanded plan's
                # optional expansion provenance.
                "objects",
                "["
                + ",".join("{}" for _ in range(_MAX_JSON_OBJECTS + 1))
                + "]",
                "too many objects",
            ),
            (
                "separators",
                "[" + ",".join("0" for _ in range(72)) + "]",
                "too many separators",
            ),
        )
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            with (
                patch(
                    "spatialforge.tsdf_block_plan_loader.MAX_PLANNED_BLOCKS",
                    1,
                ),
                patch(
                    "spatialforge.tsdf_block_plan_loader.json.loads",
                ) as json_loader,
            ):
                for name, text, message in cases:
                    with self.subTest(name=name):
                        path = temporary_root / f"{name}.sftplan"
                        path.write_text(text, encoding="ascii")
                        with self.assertRaises(TsdfError) as raised:
                            load_tsdf_block_plan(path)
                        self.assertIn(message, str(raised.exception))
                        json_loader.assert_not_called()

    def test_unknown_missing_exact_and_numeric_schema_fields_fail(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            canonical = create_plan(temporary_root, name="canonical.sftplan")
            base = read_document(canonical)

            cases: list[tuple[str, dict, str]] = []

            unknown_root = copy.deepcopy(base)
            unknown_root["unexpected"] = True
            cases.append(("unknown-root", unknown_root, "unknown field"))

            unknown_grid = copy.deepcopy(base)
            unknown_grid["grid"]["unexpected"] = 1
            cases.append(("unknown-grid", unknown_grid, "unknown field"))

            missing_schema = copy.deepcopy(base)
            del missing_schema["schema"]
            cases.append(("missing-schema", missing_schema, "missing required field"))

            wrong_schema = copy.deepcopy(base)
            wrong_schema["schema"] = "spatialforge.other"
            cases.append(("wrong-schema", wrong_schema, "expected"))

            bad_anchor = copy.deepcopy(base)
            bad_anchor["grid"]["world_anchor_m"] = [1.0, 0.0, 0.0]
            cases.append(("bad-anchor", bad_anchor, "world_anchor_m"))

            bool_resolution = copy.deepcopy(base)
            bool_resolution["grid"]["block_resolution"] = True
            cases.append(
                ("bool-resolution", bool_resolution, "expected a positive integer")
            )

            wrong_resolution = copy.deepcopy(base)
            wrong_resolution["grid"]["block_resolution"] = 4
            cases.append(("wrong-resolution", wrong_resolution, "expected 8"))

            wrong_extent = copy.deepcopy(base)
            wrong_extent["grid"]["block_extent_m"] = 2.0
            cases.append(("wrong-extent", wrong_extent, "block_extent_m"))

            wrong_rule = copy.deepcopy(base)
            wrong_rule["activation"]["rule"] = "other"
            cases.append(("wrong-rule", wrong_rule, "activation.rule"))

            short_truncation = copy.deepcopy(base)
            short_truncation["activation"]["truncation_m"] = 0.01
            cases.append(
                ("short-truncation", short_truncation, "at least voxel_size_m")
            )

            for name, document, message in cases:
                with self.subTest(name=name):
                    path = temporary_root / f"{name}.sftplan"
                    write_document(path, document)
                    with self.assertRaises(TsdfError) as raised:
                        load_tsdf_block_plan(path)
                    self.assertIn(message, str(raised.exception))

            overflow = temporary_root / "overflow.sftplan"
            overflow_text = canonical.read_text(encoding="ascii").replace(
                '"voxel_size_m": 0.125',
                '"voxel_size_m": 1e400',
                1,
            )
            overflow.write_text(overflow_text, encoding="ascii")
            with self.assertRaises(TsdfError) as raised:
                load_tsdf_block_plan(overflow)
            self.assertIn("finite number", str(raised.exception))

    def test_block_lists_and_all_derived_metadata_are_strict(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            canonical = create_plan(temporary_root, name="canonical.sftplan")
            base = read_document(canonical)

            cases: list[tuple[str, dict, str]] = []

            unordered = copy.deepcopy(base)
            unordered["active_blocks"][0], unordered["active_blocks"][1] = (
                unordered["active_blocks"][1],
                unordered["active_blocks"][0],
            )
            cases.append(("unordered", unordered, "strictly x-fastest ordered"))

            duplicate = copy.deepcopy(base)
            duplicate["active_blocks"][1] = duplicate["active_blocks"][0]
            cases.append(("duplicate", duplicate, "unique"))

            bool_coordinate = copy.deepcopy(base)
            bool_coordinate["active_blocks"][0][0] = True
            cases.append(("bool-coordinate", bool_coordinate, "expected an integer"))

            out_of_range = copy.deepcopy(base)
            out_of_range["active_blocks"][-1][0] = 2**31
            cases.append(("out-of-range", out_of_range, "signed 32-bit"))

            missing_surface = copy.deepcopy(base)
            missing_surface["surface_blocks"][0] = [-1, -1, -1]
            cases.append(("surface-subset", missing_surface, "must be active"))

            bad_list_count = copy.deepcopy(base)
            bad_list_count["planning"]["active_blocks"] = 7
            bad_list_count["planning"]["halo_blocks"] = 3
            cases.append(("list-count", bad_list_count, "expected 7 entries"))

            bad_halo = copy.deepcopy(base)
            bad_halo["planning"]["halo_blocks"] = 5
            cases.append(("halo-count", bad_halo, "halo_blocks"))

            bad_slots = copy.deepcopy(base)
            bad_slots["planning"]["planned_voxel_slots"] = 4095
            cases.append(("voxel-slots", bad_slots, "planned_voxel_slots"))

            bad_minimum = copy.deepcopy(base)
            bad_minimum["planning"]["min_block_index"] = [-1, -1, -1]
            cases.append(("minimum", bad_minimum, "min_block_index"))

            bad_maximum = copy.deepcopy(base)
            bad_maximum["planning"]["max_block_index"] = [2, 0, 0]
            cases.append(("maximum", bad_maximum, "max_block_index"))

            bad_selected = copy.deepcopy(base)
            bad_selected["planning"]["selected_observations"] = 1
            bad_selected["planning"]["paired_observations"] = 1
            bad_selected["planning"]["valid_depth_points"] = 4
            cases.append(("selected-count", bad_selected, "expected 2"))

            bad_missing = copy.deepcopy(base)
            bad_missing["planning"]["paired_observations"] = 1
            cases.append(("missing-count", bad_missing, "missing-depth/pose"))

            for name, document, message in cases:
                with self.subTest(name=name):
                    path = temporary_root / f"{name}.sftplan"
                    write_document(path, document)
                    with self.assertRaises(TsdfError) as raised:
                        load_tsdf_block_plan(path)
                    self.assertIn(message, str(raised.exception))

            with patch(
                "spatialforge.tsdf_block_plan_loader.MAX_PLANNED_BLOCKS",
                7,
            ):
                with self.assertRaises(TsdfError) as raised:
                    load_tsdf_block_plan(canonical)
            self.assertIn("maximum is 7", str(raised.exception))

    def test_file_size_suffix_and_missing_inputs_are_bounded(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            canonical = create_plan(temporary_root, name="canonical.sftplan")
            encoded = canonical.read_bytes()

            with patch(
                "spatialforge.tsdf_block_plan_loader.MAX_TSDF_BLOCK_PLAN_BYTES",
                len(encoded),
            ):
                load_tsdf_block_plan(canonical)

            with patch(
                "spatialforge.tsdf_block_plan_loader.MAX_TSDF_BLOCK_PLAN_BYTES",
                len(encoded) - 1,
            ):
                with self.assertRaises(TsdfError) as raised:
                    load_tsdf_block_plan(canonical)
            self.assertIn("maximum", str(raised.exception))

            with self.assertRaises(TsdfError) as raised:
                load_tsdf_block_plan(temporary_root / "wrong.json")
            self.assertIn("end in .sftplan", str(raised.exception))

            with self.assertRaises(TsdfError) as raised:
                load_tsdf_block_plan(temporary_root / "missing.sftplan")
            self.assertIn("does not exist", str(raised.exception))

            oversized = temporary_root / "oversized.sftplan"
            oversized.write_bytes(encoded + b" ")
            with patch(
                "spatialforge.tsdf_block_plan_loader.MAX_TSDF_BLOCK_PLAN_BYTES",
                len(encoded),
            ):
                with self.assertRaises(TsdfError) as raised:
                    load_tsdf_block_plan(oversized)
            self.assertIn("maximum", str(raised.exception))


class TsdfBlockPlanReplayVerificationTests(unittest.TestCase):
    def test_replay_verification_succeeds_without_geometry_or_fusion(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            path = create_plan(temporary_root)
            before = path.read_bytes()
            plan = load_tsdf_block_plan(path)
            session = load_scan_session(FIXTURE)
            with (
                patch("spatialforge.tsdf_block_plan._plan_observation_blocks") as planner,
                patch("spatialforge.tsdf.integrate_tsdf") as dense_fusion,
                patch("spatialforge.sparse_tsdf.integrate_sparse_tsdf") as sparse_fusion,
            ):
                result = verify_tsdf_block_plan_replay(plan, session)
                planner.assert_not_called()
                dense_fusion.assert_not_called()
                sparse_fusion.assert_not_called()
            after = path.read_bytes()

        self.assertIsNone(result)
        self.assertEqual(before, after)

    def test_session_digest_association_and_pixel_capacity_mismatches_fail(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            path = create_plan(temporary_root)
            plan = load_tsdf_block_plan(path)
            session = load_scan_session(FIXTURE)

            cases = (
                (
                    "session",
                    replace(plan, session_id="different-session"),
                    "session_id",
                ),
                (
                    "digest",
                    replace(plan, replay_digest_sha256="0" * 64),
                    "replay digest",
                ),
                (
                    "total",
                    replace(plan, total_observations=3),
                    "total_observations",
                ),
                (
                    "selected",
                    replace(plan, selected_observations=1),
                    "selected_observations",
                ),
                (
                    "paired",
                    replace(plan, paired_observations=1),
                    "paired_observations",
                ),
                (
                    "missing-depth",
                    replace(plan, skipped_missing_depth=1),
                    "skipped_missing_depth",
                ),
                (
                    "missing-pose",
                    replace(plan, skipped_missing_pose=1),
                    "skipped_missing_pose",
                ),
                (
                    "pixel-capacity",
                    replace(plan, valid_depth_points=9),
                    "depth sample",
                ),
            )
            for name, changed, message in cases:
                with self.subTest(name=name):
                    with self.assertRaises(TsdfError) as raised:
                        verify_tsdf_block_plan_replay(changed, session)
                    self.assertIn(message, str(raised.exception))

            changed_fixture = copy_fixture(temporary_root, "changed.vgsession")
            changed_session = load_scan_session(changed_fixture)
            depth_path = changed_fixture / "data" / "depth" / "000000.pgm"
            depth_path.write_text(
                depth_path.read_text(encoding="ascii").replace("1000", "999", 1),
                encoding="ascii",
            )
            with self.assertRaises(TsdfError) as raised:
                verify_tsdf_block_plan_replay(plan, changed_session)
            self.assertIn("replay digest", str(raised.exception))


class TsdfBlockPlanVerifyCliTests(unittest.TestCase):
    def test_cli_verifies_read_only_without_fusion(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            path = create_plan(temporary_root)
            before = path.read_bytes()
            stdout = io.StringIO()
            with (
                patch("spatialforge.cli.plan_tsdf_blocks") as planner,
                patch("spatialforge.cli.integrate_tsdf") as dense_fusion,
                patch("spatialforge.cli.integrate_sparse_tsdf") as sparse_fusion,
                redirect_stdout(stdout),
            ):
                exit_code = main(
                    [
                        "reconstruct",
                        "tsdf-block-plan-verify",
                        str(path),
                        str(FIXTURE),
                    ]
                )
                planner.assert_not_called()
                dense_fusion.assert_not_called()
                sparse_fusion.assert_not_called()
            after = path.read_bytes()

        self.assertEqual(exit_code, 0)
        self.assertEqual(before, after)
        self.assertIn("TSDF BLOCK PLAN CHECK", stdout.getvalue())
        self.assertIn("artifact: valid", stdout.getvalue())
        self.assertIn("session_replay: matched", stdout.getvalue())
        self.assertIn("geometry_recomputed: no", stdout.getvalue())
        self.assertIn(PLAN_SHA256, stdout.getvalue())
        self.assertIn(REPLAY_SHA256, stdout.getvalue())

    def test_cli_mismatch_is_actionable_and_read_only(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            fixture = copy_fixture(temporary_root)
            path = create_plan(temporary_root, session_path=fixture)
            before = path.read_bytes()
            depth_path = fixture / "data" / "depth" / "000000.pgm"
            depth_path.write_text(
                depth_path.read_text(encoding="ascii").replace("1000", "999", 1),
                encoding="ascii",
            )
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                exit_code = main(
                    [
                        "reconstruct",
                        "tsdf-block-plan-verify",
                        str(path),
                        str(fixture),
                    ]
                )
            after = path.read_bytes()

        self.assertEqual(exit_code, 2)
        self.assertEqual(before, after)
        self.assertIn("TSDF BLOCK PLAN VERIFY FAILED", stderr.getvalue())
        self.assertIn("replay digest", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
