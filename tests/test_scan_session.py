from __future__ import annotations

import io
import json
import shutil
import tempfile
import unittest
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Iterator

from spatialforge.cli import main
from spatialforge.errors import SessionValidationError
from spatialforge.replay import replay_session
from spatialforge.session_loader import load_scan_session


TEST_ROOT = Path(__file__).resolve().parent
FIXTURE = TEST_ROOT / "fixtures" / "minimal.vgsession"


@contextmanager
def copied_fixture() -> Iterator[Path]:
    with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
        target = Path(temporary_directory) / "case.vgsession"
        shutil.copytree(FIXTURE, target)
        yield target


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2) + "\n",
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(value, separators=(",", ":")) + "\n"
            for value in values
        ),
        encoding="utf-8",
    )


class ScanSessionTests(unittest.TestCase):
    def test_minimal_session_loads_with_expected_summary(self) -> None:
        session = load_scan_session(FIXTURE)

        self.assertEqual(session.session_id, "scan-synthetic-0001")
        self.assertEqual(session.schema_version, "0.1.0")
        self.assertEqual(set(session.calibrations), {"camera-rgb", "camera-depth"})
        self.assertEqual(len(session.streams["rgb"]), 2)
        self.assertEqual(len(session.streams["depth"]), 2)
        self.assertEqual(len(session.streams["imu"]), 3)
        self.assertEqual(len(session.streams["pose"]), 2)
        self.assertEqual(session.duration_ns, 33_333_333)

    def test_replay_is_deterministic_and_associates_optional_streams(self) -> None:
        session = load_scan_session(FIXTURE)

        first = replay_session(session)
        second = replay_session(session)

        self.assertEqual(first, second)
        self.assertEqual(len(first.observations), 2)
        self.assertEqual(first.observations[0].depth.id, "depth-000000")
        self.assertEqual(first.observations[0].pose.id, "pose-000000")
        self.assertEqual(
            [sample.id for sample in first.observations[0].imu],
            ["imu-000000"],
        )
        self.assertEqual(
            [sample.id for sample in first.observations[1].imu],
            ["imu-000001", "imu-000002"],
        )
        self.assertEqual(
            first.digest_sha256,
            "dc001ae0ca01004a21ad227b12d57a7350bbc23904040453ca73fd955ef050b8",
        )

    def test_loaded_session_cannot_be_mutated_after_validation(self) -> None:
        session = load_scan_session(FIXTURE)

        with self.assertRaises(TypeError):
            session.streams["rgb"] = ()  # type: ignore[index]
        with self.assertRaises(TypeError):
            session.streams["rgb"][0].data["timestamp_ns"] = 42  # type: ignore[index]

    def test_replay_digest_changes_when_sensor_bytes_change(self) -> None:
        with copied_fixture() as fixture:
            before = replay_session(load_scan_session(fixture))
            rgb_path = fixture / "data" / "rgb" / "000001.ppm"
            rgb_path.write_text(
                rgb_path.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            after = replay_session(load_scan_session(fixture))

        self.assertNotEqual(before.digest_sha256, after.digest_sha256)

    def test_replay_digest_includes_trailing_unassociated_imu(self) -> None:
        with copied_fixture() as fixture:
            before = replay_session(load_scan_session(fixture))
            path = fixture / "streams" / "imu.jsonl"
            records = read_jsonl(path)
            records.append(
                {
                    "id": "imu-000003",
                    "timestamp_ns": 50_000_000,
                    "accelerometer_m_s2": [0.0, 0.0, 9.80665],
                    "gyroscope_rad_s": [0.0, 0.0, 0.0],
                }
            )
            write_jsonl(path, records)
            after = replay_session(load_scan_session(fixture))

        self.assertEqual(len(after.observations[-1].imu), 2)
        self.assertNotEqual(before.digest_sha256, after.digest_sha256)

    def test_missing_exact_depth_association_is_explicit(self) -> None:
        with copied_fixture() as fixture:
            path = fixture / "streams" / "depth.jsonl"
            records = read_jsonl(path)
            records[1]["timestamp_ns"] = 30_000_000
            write_jsonl(path, records)

            replay = replay_session(load_scan_session(fixture))

        self.assertIsNone(replay.observations[1].depth)

    def test_missing_intrinsic_is_rejected_with_field_path(self) -> None:
        with copied_fixture() as fixture:
            path = fixture / "calibration" / "cameras.json"
            calibration = read_json(path)
            del calibration["cameras"][0]["intrinsics"]["fx"]
            write_json(path, calibration)

            with self.assertRaises(SessionValidationError) as raised:
                load_scan_session(fixture)

        self.assertIn(
            "calibration.cameras[0].intrinsics.fx: expected a finite number",
            raised.exception.errors,
        )

    def test_missing_sensor_file_is_rejected(self) -> None:
        with copied_fixture() as fixture:
            (fixture / "data" / "rgb" / "000001.ppm").unlink()

            with self.assertRaises(SessionValidationError) as raised:
                load_scan_session(fixture)

        self.assertTrue(
            any(
                "streams.rgb[line 2].path: referenced file does not exist"
                in error
                for error in raised.exception.errors
            )
        )

    def test_non_monotonic_timestamp_is_rejected(self) -> None:
        with copied_fixture() as fixture:
            path = fixture / "streams" / "rgb.jsonl"
            records = read_jsonl(path)
            records[1]["timestamp_ns"] = 0
            write_jsonl(path, records)

            with self.assertRaises(SessionValidationError) as raised:
                load_scan_session(fixture)

        self.assertTrue(
            any(
                "timestamp_ns: must be strictly increasing" in error
                for error in raised.exception.errors
            )
        )

    def test_non_rigid_pose_is_rejected(self) -> None:
        with copied_fixture() as fixture:
            path = fixture / "streams" / "poses.jsonl"
            records = read_jsonl(path)
            records[1]["T_world_camera"][0] = 2
            write_jsonl(path, records)

            with self.assertRaises(SessionValidationError) as raised:
                load_scan_session(fixture)

        self.assertTrue(
            any(
                "rotation must be orthonormal" in error
                for error in raised.exception.errors
            )
        )

    def test_non_positive_depth_scale_is_rejected(self) -> None:
        with copied_fixture() as fixture:
            path = fixture / "manifest.json"
            manifest = read_json(path)
            manifest["streams"]["depth"]["depth_scale_m"] = 0
            write_json(path, manifest)

            with self.assertRaises(SessionValidationError) as raised:
                load_scan_session(fixture)

        self.assertIn(
            "manifest.streams.depth.depth_scale_m: must be positive",
            raised.exception.errors,
        )

    def test_huge_number_is_a_validation_error_not_a_crash(self) -> None:
        with copied_fixture() as fixture:
            path = fixture / "calibration" / "cameras.json"
            calibration = read_json(path)
            calibration["cameras"][0]["intrinsics"]["fx"] = 10**400
            write_json(path, calibration)

            with self.assertRaises(SessionValidationError) as raised:
                load_scan_session(fixture)

        self.assertIn(
            "calibration.cameras[0].intrinsics.fx: expected a finite number",
            raised.exception.errors,
        )

    def test_parent_path_reference_is_rejected(self) -> None:
        with copied_fixture() as fixture:
            path = fixture / "manifest.json"
            manifest = read_json(path)
            manifest["calibration"] = "../outside.json"
            write_json(path, manifest)

            with self.assertRaises(SessionValidationError) as raised:
                load_scan_session(fixture)

        self.assertIn(
            "manifest.calibration: path must be relative and remain in the session",
            raised.exception.errors,
        )


class CliTests(unittest.TestCase):
    def test_validate_command_prints_deterministic_summary(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = main(["scan", "validate", str(FIXTURE)])

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            output.getvalue(),
            "\n".join(
                [
                    "VALID scan-synthetic-0001",
                    "schema_version: 0.1.0",
                    "calibration_ids: camera-depth, camera-rgb",
                    "sensors: rgb=yes depth=yes imu=yes pose=yes",
                    "samples: rgb=2 depth=2 imu=3 pose=2",
                    "time_span_ns: 33333333",
                    "duration_s: 0.033333333",
                    "",
                ]
            ),
        )

    def test_replay_command_is_identical_across_runs(self) -> None:
        first_output = io.StringIO()
        second_output = io.StringIO()

        with redirect_stdout(first_output):
            first_exit_code = main(["scan", "replay", str(FIXTURE)])
        with redirect_stdout(second_output):
            second_exit_code = main(["scan", "replay", str(FIXTURE)])

        self.assertEqual(first_exit_code, 0)
        self.assertEqual(second_exit_code, 0)
        self.assertEqual(first_output.getvalue(), second_output.getvalue())
        self.assertIn("depth=depth-000001", first_output.getvalue())
        self.assertIn("pose=pose-000001", first_output.getvalue())
        self.assertIn("digest_sha256: ", first_output.getvalue())

    def test_invalid_session_returns_nonzero_and_actionable_error(self) -> None:
        with copied_fixture() as fixture:
            path = fixture / "manifest.json"
            manifest = read_json(path)
            manifest["streams"]["depth"]["depth_scale_m"] = -1
            write_json(path, manifest)

            error_output = io.StringIO()
            with redirect_stderr(error_output):
                exit_code = main(["scan", "validate", str(fixture)])

        self.assertEqual(exit_code, 2)
        self.assertIn("INVALID", error_output.getvalue())
        self.assertIn(
            "manifest.streams.depth.depth_scale_m: must be positive",
            error_output.getvalue(),
        )


if __name__ == "__main__":
    unittest.main()
