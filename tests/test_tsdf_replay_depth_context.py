from __future__ import annotations

import hashlib
import io
import shutil
import tempfile
import unittest
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from spatialforge import build_tsdf_replay_depth_context
from spatialforge.cli import main
from spatialforge.errors import PointCloudError, TsdfError
from spatialforge.replay import replay_session
from spatialforge.session_loader import load_scan_session
from spatialforge.tsdf_block_plan import plan_tsdf_blocks
from spatialforge.tsdf_block_plan_loader import load_tsdf_block_plan
from spatialforge.tsdf_replay_depth_context import (
    MAX_TSDF_REPLAY_DEPTH_CONTEXT_BYTES,
    TsdfReplayDepthContext,
    TsdfReplayDepthObservation,
    TsdfReplayDepthStatus,
)


TEST_ROOT = Path(__file__).resolve().parent
FIXTURE = TEST_ROOT / "fixtures" / "minimal.vgsession"
PLAN_ARGUMENTS = {
    "voxel_size_m": 0.125,
    "truncation_m": 0.5,
}
PLAN_SHA256 = "372c7c5d49eff1a30317ceb8b67cb3c40683f1d9d2a6179a049772f763d6f79d"
STRIDE_TWO_PLAN_SHA256 = (
    "22c88a7929b30af0a8b6977fd992b679a852c1346eff4e0693262d5ed9d7ca75"
)
REPLAY_SHA256 = "dc001ae0ca01004a21ad227b12d57a7350bbc23904040453ca73fd955ef050b8"
DEPTH_PAYLOAD_SHA256 = (
    "c914e8188e43fff1c96e25283e15b252af0d9f39b469f2d1518915802c756d18",
    "3fe60f1a71c87b621133b02bb866868de3a3f9f4f0d47ffc963bba3df60c7c78",
)

_FORBIDDEN_CONTEXT_CLI_TARGETS = (
    "spatialforge.cli.allocate_empty_tsdf_blocks",
    "spatialforge.cli.locate_tsdf_voxel",
    "spatialforge.cli.evaluate_tsdf_voxel_contribution",
    "spatialforge.cli.apply_tsdf_voxel_contribution",
    "spatialforge.cli.traverse_tsdf_voxel_observations",
    "spatialforge.cli.reconstruct_point_cloud",
    "spatialforge.cli.integrate_tsdf",
    "spatialforge.cli.integrate_sparse_tsdf",
    "spatialforge.cli.infer_tsdf_bounds",
    "spatialforge.cli.plan_tsdf_blocks",
    "spatialforge.cli.extract_surface_points",
    "spatialforge.cli.extract_triangle_mesh",
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
    plan_name: str = "fixture.sftplan",
    frame_stride: int = 1,
):
    plan = load_tsdf_block_plan(
        create_plan(
            parent,
            session_path=session_path,
            name=plan_name,
            frame_stride=frame_stride,
        )
    )
    return plan, load_scan_session(session_path)


def remove_first_record(path: Path) -> None:
    remaining = path.read_text(encoding="utf-8").splitlines()[1:]
    path.write_text(
        "".join(line + "\n" for line in remaining),
        encoding="utf-8",
    )


def tree_snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def immutable_bytes_base(array: np.ndarray) -> object:
    current: object = array
    while isinstance(current, np.ndarray) and current.base is not None:
        current = current.base
    return current


@contextmanager
def forbidden_context_cli_calls():
    with ExitStack() as stack:
        mocks: list[MagicMock] = []
        for target in _FORBIDDEN_CONTEXT_CLI_TARGETS:
            mocks.append(stack.enter_context(patch(target)))
        yield mocks


class TsdfReplayDepthContextTests(unittest.TestCase):
    def test_exact_fixture_builds_canonical_immutable_context(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            plan, session = load_case(Path(temporary_directory))
            import spatialforge.tsdf_replay_depth_context as context_module

            actual_replay = context_module.replay_session
            actual_read_depth = context_module._read_depth
            with (
                patch(
                    "spatialforge.tsdf_replay_depth_context.replay_session",
                    wraps=actual_replay,
                ) as replay,
                patch(
                    "spatialforge.tsdf_replay_depth_context._read_depth",
                    wraps=actual_read_depth,
                ) as depth_reader,
            ):
                context = build_tsdf_replay_depth_context(plan, session)

        self.assertIsInstance(context, TsdfReplayDepthContext)
        self.assertEqual(context.session_id, "scan-synthetic-0001")
        self.assertEqual(context.source_plan_digest_sha256, PLAN_SHA256)
        self.assertEqual(context.replay_digest_sha256, REPLAY_SHA256)
        self.assertEqual(context.frame_stride, 1)
        self.assertEqual(context.total_observations, 2)
        self.assertEqual(context.selected_observation_sequences, (0, 1))
        self.assertEqual(context.selected_observation_count, 2)
        self.assertEqual(context.ready_observation_count, 2)
        self.assertEqual(context.depth_frames_decoded, 2)
        self.assertEqual(context.depth_sample_count, 8)
        self.assertEqual(context.depth_payload_bytes, 64)
        self.assertEqual(context.valid_depth_samples, 8)
        self.assertEqual(context.invalid_depth_samples, 0)
        self.assertEqual(context.depth_scale_m, 0.001)
        self.assertEqual((context.camera.width, context.camera.height), (2, 2))
        self.assertEqual(
            context.status_counts,
            ((TsdfReplayDepthStatus.READY, 2),),
        )
        self.assertEqual(
            tuple(record.observation_sequence for record in context.observations),
            (0, 1),
        )
        self.assertEqual(
            tuple(record.status for record in context.observations),
            (TsdfReplayDepthStatus.READY, TsdfReplayDepthStatus.READY),
        )
        self.assertEqual(
            context.observations[0].t_world_camera,
            (
                0.0,
                0.0,
                1.0,
                0.0,
                -1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ),
        )
        self.assertEqual(context.observations[1].t_world_camera[3], 0.05)
        self.assertTrue(
            all(
                isinstance(value, float)
                for record in context.observations
                for value in record.t_world_camera or ()
            )
        )

        expected_values = (1.0, 0.9500000000000001)
        arrays: list[np.ndarray] = []
        for position, (record, expected_value) in enumerate(
            zip(context.observations, expected_values, strict=True)
        ):
            self.assertTrue(record.depth_decoded)
            self.assertIs(type(record.depth_m), np.ndarray)
            depth = record.depth_m
            assert depth is not None
            arrays.append(depth)
            self.assertEqual(depth.shape, (2, 2))
            self.assertEqual(depth.dtype, np.dtype(np.float64))
            self.assertTrue(depth.flags.c_contiguous)
            self.assertFalse(depth.flags.owndata)
            self.assertFalse(depth.flags.writeable)
            self.assertIsInstance(immutable_bytes_base(depth), bytes)
            np.testing.assert_array_equal(
                depth,
                np.full((2, 2), expected_value, dtype=np.float64),
            )
            self.assertEqual(
                hashlib.sha256(depth.tobytes(order="C")).hexdigest(),
                DEPTH_PAYLOAD_SHA256[position],
            )
            with self.assertRaises(ValueError):
                depth[0, 0] = 7.0
            with self.assertRaises(ValueError):
                depth.setflags(write=True)
            with self.assertRaises(FrozenInstanceError):
                setattr(
                    record,
                    "status",
                    TsdfReplayDepthStatus.MISSING_DEPTH,
                )
            self.assertFalse(hasattr(record, "__dict__"))
        self.assertFalse(np.shares_memory(arrays[0], arrays[1]))
        with self.assertRaises(FrozenInstanceError):
            context.frame_stride = 2  # type: ignore[misc]
        self.assertFalse(hasattr(context, "__dict__"))

        self.assertEqual(replay.call_count, 2)
        self.assertEqual(depth_reader.call_count, 2)
        self.assertEqual(
            [call.args[0].name for call in depth_reader.call_args_list],
            ["000000.pgm", "000001.pgm"],
        )
        self.assertTrue(
            all(call.args[1:] == (2, 2) for call in depth_reader.call_args_list)
        )

    def test_stride_two_selects_and_decodes_only_sequence_zero(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            plan, session = load_case(
                Path(temporary_directory),
                frame_stride=2,
            )
            import spatialforge.tsdf_replay_depth_context as context_module

            actual_read_depth = context_module._read_depth
            with patch(
                "spatialforge.tsdf_replay_depth_context._read_depth",
                wraps=actual_read_depth,
            ) as depth_reader:
                context = build_tsdf_replay_depth_context(plan, session)

        self.assertEqual(plan.artifact_digest_sha256, STRIDE_TWO_PLAN_SHA256)
        self.assertEqual(context.frame_stride, 2)
        self.assertEqual(context.total_observations, 2)
        self.assertEqual(context.selected_observation_sequences, (0,))
        self.assertEqual(context.selected_observation_count, 1)
        self.assertEqual(context.ready_observation_count, 1)
        self.assertEqual(context.depth_frames_decoded, 1)
        self.assertEqual(context.depth_sample_count, 4)
        self.assertEqual(context.depth_payload_bytes, 32)
        self.assertEqual(context.valid_depth_samples, 4)
        self.assertEqual(context.invalid_depth_samples, 0)
        self.assertEqual(depth_reader.call_count, 1)
        self.assertEqual(depth_reader.call_args.args[0].name, "000000.pgm")
        np.testing.assert_array_equal(
            context.observations[0].depth_m,
            np.ones((2, 2), dtype=np.float64),
        )

    def test_missing_inputs_have_stable_status_and_suppress_decode(self) -> None:
        cases = (
            (
                "missing-depth",
                ("streams/depth.jsonl",),
                TsdfReplayDepthStatus.MISSING_DEPTH,
                True,
            ),
            (
                "missing-pose",
                ("streams/poses.jsonl",),
                TsdfReplayDepthStatus.MISSING_POSE,
                False,
            ),
            (
                "missing-both",
                ("streams/depth.jsonl", "streams/poses.jsonl"),
                TsdfReplayDepthStatus.MISSING_DEPTH_AND_POSE,
                False,
            ),
        )
        for name, references, expected_status, keeps_transform in cases:
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory(
                    dir=TEST_ROOT
                ) as temporary_directory:
                    temporary_root = Path(temporary_directory)
                    fixture = copy_fixture(
                        temporary_root,
                        f"{name}.vgsession",
                    )
                    for reference in references:
                        remove_first_record(fixture / reference)
                    plan, session = load_case(
                        temporary_root,
                        session_path=fixture,
                        plan_name=f"{name}.sftplan",
                    )
                    import spatialforge.tsdf_replay_depth_context as context_module

                    actual_read_depth = context_module._read_depth
                    with patch(
                        "spatialforge.tsdf_replay_depth_context._read_depth",
                        wraps=actual_read_depth,
                    ) as depth_reader:
                        context = build_tsdf_replay_depth_context(plan, session)

                first, second = context.observations
                self.assertEqual(first.observation_sequence, 0)
                self.assertEqual(first.status, expected_status)
                self.assertIsNone(first.depth_m)
                self.assertFalse(first.depth_decoded)
                if keeps_transform:
                    self.assertIsNotNone(first.t_world_camera)
                    self.assertEqual(first.t_world_camera[15], 1.0)
                else:
                    self.assertIsNone(first.t_world_camera)
                self.assertEqual(second.status, TsdfReplayDepthStatus.READY)
                self.assertTrue(second.depth_decoded)
                self.assertEqual(context.selected_observation_sequences, (0, 1))
                self.assertEqual(context.ready_observation_count, 1)
                self.assertEqual(context.depth_frames_decoded, 1)
                self.assertEqual(context.depth_sample_count, 4)
                self.assertEqual(context.depth_payload_bytes, 32)
                self.assertEqual(context.valid_depth_samples, 4)
                self.assertEqual(context.invalid_depth_samples, 0)
                self.assertEqual(depth_reader.call_count, 1)
                self.assertEqual(
                    depth_reader.call_args.args[0].name,
                    "000001.pgm",
                )
                self.assertEqual(
                    dict(context.status_counts),
                    {
                        expected_status: 1,
                        TsdfReplayDepthStatus.READY: 1,
                    },
                )

    def test_context_byte_limit_is_preflighted_before_decode(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            plan, session = load_case(Path(temporary_directory))
            huge_camera = replace(
                session.calibrations["camera-rgb"],
                width=8192,
                height=8192,
            )
            expected_samples = 2 * huge_camera.width * huge_camera.height
            changed_plan = replace(
                plan,
                valid_depth_points=expected_samples,
                invalid_depth_samples=0,
            )
            expected_bytes = expected_samples * np.dtype(np.float64).itemsize
            self.assertGreater(
                expected_bytes,
                MAX_TSDF_REPLAY_DEPTH_CONTEXT_BYTES,
            )

            with (
                patch(
                    "spatialforge.tsdf_replay_depth_context."
                    "_validate_reconstruction_contract",
                    return_value=(huge_camera, 0.001),
                ),
                patch(
                    "spatialforge.tsdf_replay_depth_context._read_depth"
                ) as depth_reader,
            ):
                with self.assertRaises(TsdfError) as raised:
                    build_tsdf_replay_depth_context(changed_plan, session)

        self.assertTrue(
            "maximum" in str(raised.exception)
            or "limit" in str(raised.exception)
        )
        depth_reader.assert_not_called()

    def test_plan_and_replay_metadata_failures_precede_decode(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            plan, session = load_case(Path(temporary_directory))
            cases = (
                (replace(plan, session_id="different-session"), "session_id"),
                (
                    replace(plan, replay_digest_sha256="0" * 64),
                    "replay digest",
                ),
                (replace(plan, total_observations=3), "total observations"),
                (replace(plan, selected_observations=1), "selected observations"),
                (replace(plan, paired_observations=1), "associations"),
                (
                    replace(plan, skipped_missing_depth=1),
                    "associations",
                ),
                (
                    replace(plan, skipped_missing_pose=1),
                    "associations",
                ),
                (replace(plan, valid_depth_points=9), "depth sample"),
            )
            for changed_plan, message in cases:
                with self.subTest(message=message):
                    with patch(
                        "spatialforge.tsdf_replay_depth_context._read_depth"
                    ) as depth_reader:
                        with self.assertRaises(TsdfError) as raised:
                            build_tsdf_replay_depth_context(
                                changed_plan,
                                session,
                            )
                    self.assertIn(message, str(raised.exception))
                    depth_reader.assert_not_called()

    def test_decode_failure_aborts_without_filesystem_mutation(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            plan, session = load_case(temporary_root)
            before_tree = tree_snapshot(temporary_root)
            import spatialforge.tsdf_replay_depth_context as context_module

            actual_read_depth = context_module._read_depth

            def fail_second_depth(
                path: Path,
                width: int,
                height: int,
            ):
                if path.name == "000001.pgm":
                    raise PointCloudError("injected second depth failure")
                return actual_read_depth(path, width, height)

            with (
                patch(
                    "spatialforge.tsdf_replay_depth_context.replay_session",
                    wraps=context_module.replay_session,
                ) as replay,
                patch(
                    "spatialforge.tsdf_replay_depth_context._read_depth",
                    side_effect=fail_second_depth,
                ) as depth_reader,
            ):
                with self.assertRaises(TsdfError) as raised:
                    build_tsdf_replay_depth_context(plan, session)
            after_tree = tree_snapshot(temporary_root)

        self.assertIn("injected second depth failure", str(raised.exception))
        self.assertEqual(replay.call_count, 1)
        self.assertEqual(depth_reader.call_count, 2)
        self.assertEqual(after_tree, before_tree)

    def test_ending_replay_change_rejects_completed_decodes(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            plan, session = load_case(Path(temporary_directory))
            matched = replay_session(session)
            changed = replace(matched, digest_sha256="0" * 64)
            import spatialforge.tsdf_replay_depth_context as context_module

            actual_read_depth = context_module._read_depth
            with (
                patch(
                    "spatialforge.tsdf_replay_depth_context.replay_session",
                    side_effect=(matched, changed),
                ) as replay,
                patch(
                    "spatialforge.tsdf_replay_depth_context._read_depth",
                    wraps=actual_read_depth,
                ) as depth_reader,
            ):
                with self.assertRaises(TsdfError) as raised:
                    build_tsdf_replay_depth_context(plan, session)

        self.assertIn("changed", str(raised.exception))
        self.assertEqual(replay.call_count, 2)
        self.assertEqual(depth_reader.call_count, 2)

    def test_repeated_builds_are_byte_identical_and_independent(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            plan, session = load_case(temporary_root)
            before_tree = tree_snapshot(temporary_root)
            first = build_tsdf_replay_depth_context(plan, session)
            second = build_tsdf_replay_depth_context(plan, session)
            after_tree = tree_snapshot(temporary_root)

        self.assertEqual(after_tree, before_tree)
        self.assertEqual(
            first.selected_observation_sequences,
            second.selected_observation_sequences,
        )
        for left, right in zip(
            first.observations,
            second.observations,
            strict=True,
        ):
            self.assertIsNot(left, right)
            self.assertEqual(left.status, right.status)
            assert left.depth_m is not None and right.depth_m is not None
            self.assertIsNot(left.depth_m, right.depth_m)
            self.assertFalse(np.shares_memory(left.depth_m, right.depth_m))
            self.assertEqual(left.depth_m.tobytes(), right.depth_m.tobytes())

    def test_retrieved_depth_shape_mutation_does_not_change_record(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            plan, session = load_case(Path(temporary_directory))
            context = build_tsdf_replay_depth_context(plan, session)

        record = context.observations[0]
        retrieved = record.depth_m
        assert retrieved is not None
        self.assertEqual(retrieved.shape, (2, 2))
        retrieved.shape = (4,)
        self.assertEqual(retrieved.shape, (4,))

        fresh = record.depth_m
        assert fresh is not None
        self.assertIsNot(fresh, retrieved)
        self.assertEqual(fresh.shape, (2, 2))
        self.assertEqual(fresh.dtype, np.dtype(np.float64))
        self.assertTrue(fresh.flags.c_contiguous)
        self.assertFalse(fresh.flags.owndata)
        self.assertFalse(fresh.flags.writeable)
        self.assertIsInstance(immutable_bytes_base(fresh), bytes)
        np.testing.assert_array_equal(
            fresh,
            np.ones((2, 2), dtype=np.float64),
        )
        with self.assertRaises(ValueError):
            fresh[0, 0] = 7.0
        with self.assertRaises(ValueError):
            fresh.setflags(write=True)

    def test_partial_invalid_depth_is_preserved_and_counted(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            temporary_root = Path(temporary_directory)
            fixture = copy_fixture(temporary_root, "invalid.vgsession")
            (fixture / "data" / "depth" / "000000.pgm").write_text(
                "P2\n2 2\n65535\n0 1000\n1000 1000\n",
                encoding="ascii",
            )
            plan, session = load_case(
                temporary_root,
                session_path=fixture,
                plan_name="invalid.sftplan",
            )
            context = build_tsdf_replay_depth_context(plan, session)

        self.assertEqual(
            tuple(record.status for record in context.observations),
            (TsdfReplayDepthStatus.READY, TsdfReplayDepthStatus.READY),
        )
        self.assertEqual(context.valid_depth_samples, 7)
        self.assertEqual(context.invalid_depth_samples, 1)
        self.assertEqual(context.depth_sample_count, 8)
        np.testing.assert_array_equal(
            context.observations[0].depth_m,
            np.asarray(((0.0, 1.0), (1.0, 1.0)), dtype=np.float64),
        )

    def test_value_objects_reject_inconsistent_state(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            plan, session = load_case(Path(temporary_directory))
            context = build_tsdf_replay_depth_context(plan, session)
        ready = context.observations[0]
        assert ready.depth_m is not None

        invalid_observations = (
            lambda: replace(ready, observation_sequence=-1),
            lambda: replace(ready, status="ready"),  # type: ignore[arg-type]
            lambda: replace(ready, t_world_camera=None),
            lambda: replace(ready, depth_m=None),
            lambda: replace(
                ready,
                depth_m=np.array(ready.depth_m, copy=True),
            ),
        )
        for position, construct_invalid in enumerate(invalid_observations):
            with self.subTest(observation=position):
                with self.assertRaises(TsdfError):
                    construct_invalid()

        wrong_shape = TsdfReplayDepthObservation(
            observation_sequence=0,
            status=TsdfReplayDepthStatus.READY,
            t_world_camera=ready.t_world_camera,
            depth_m=np.frombuffer(
                np.ones((1, 4), dtype=np.float64).tobytes(),
                dtype=np.float64,
            ).reshape((1, 4)),
        )

        invalid_contexts = (
            {"source_plan_digest_sha256": "0"},
            {"selected_observation_sequences": (0,)},
            {"observations": context.observations[::-1]},
            {"observations": (wrong_shape, context.observations[1])},
            {"valid_depth_samples": 7},
            {"invalid_depth_samples": -1},
        )
        for changes in invalid_contexts:
            with self.subTest(context=changes):
                with self.assertRaises(TsdfError):
                    replace(context, **changes)

    def test_observation_rejects_subview_of_larger_immutable_payload(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            plan, session = load_case(Path(temporary_directory))
            context = build_tsdf_replay_depth_context(plan, session)

        oversized_payload = np.arange(8, dtype=np.float64).tobytes()
        full = np.frombuffer(oversized_payload, dtype=np.float64)
        subview = full[:4].reshape((2, 2))
        self.assertEqual(subview.shape, (2, 2))
        self.assertEqual(subview.dtype, np.dtype(np.float64))
        self.assertTrue(subview.flags.c_contiguous)
        self.assertFalse(subview.flags.owndata)
        self.assertFalse(subview.flags.writeable)
        self.assertEqual(subview.nbytes, 32)
        self.assertIs(immutable_bytes_base(subview), oversized_payload)
        self.assertEqual(len(oversized_payload), 64)

        with self.assertRaises(TsdfError):
            TsdfReplayDepthObservation(
                observation_sequence=0,
                status=TsdfReplayDepthStatus.READY,
                t_world_camera=context.observations[0].t_world_camera,
                depth_m=subview,
            )

    def test_context_rejects_unsupported_or_malformed_camera(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_directory:
            plan, session = load_case(Path(temporary_directory))
            context = build_tsdf_replay_depth_context(plan, session)

        camera = context.camera
        wrong_final_row = camera.t_rig_camera[:-1] + (0.0,)
        non_orthonormal = (
            2.0,
            *camera.t_rig_camera[1:],
        )
        reflected_rotation = (
            -camera.t_rig_camera[0],
            -camera.t_rig_camera[1],
            -camera.t_rig_camera[2],
            camera.t_rig_camera[3],
            *camera.t_rig_camera[4:],
        )
        camera_cases = (
            ("identifier", replace(camera, id="Bad Camera")),
            ("model", replace(camera, model="fisheye")),
            (
                "distortion-model",
                replace(
                    camera,
                    distortion_model="opencv-radtan",
                    distortion_coefficients=(0.0, 0.0, 0.0, 0.0, 0.0),
                ),
            ),
            (
                "none-with-coefficients",
                replace(camera, distortion_coefficients=(0.1,)),
            ),
            ("fx-zero", replace(camera, fx=0.0)),
            ("fx-boolean", replace(camera, fx=True)),
            ("fy-nonfinite", replace(camera, fy=float("inf"))),
            ("cx-negative", replace(camera, cx=-0.1)),
            ("cx-nonfinite", replace(camera, cx=float("nan"))),
            ("cy-outside", replace(camera, cy=float(camera.height))),
            (
                "transform-length",
                replace(camera, t_rig_camera=camera.t_rig_camera[:-1]),
            ),
            (
                "transform-nonfinite",
                replace(
                    camera,
                    t_rig_camera=(float("nan"), *camera.t_rig_camera[1:]),
                ),
            ),
            (
                "transform-final-row",
                replace(camera, t_rig_camera=wrong_final_row),
            ),
            (
                "transform-orthonormality",
                replace(camera, t_rig_camera=non_orthonormal),
            ),
            (
                "transform-determinant",
                replace(camera, t_rig_camera=reflected_rotation),
            ),
        )
        for name, invalid_camera in camera_cases:
            with self.subTest(name=name):
                with self.assertRaises(TsdfError):
                    replace(context, camera=invalid_camera)


class TsdfReplayDepthContextCliTests(unittest.TestCase):
    def test_cli_reports_exact_isolated_read_only_context(self) -> None:
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
            contexts: list[TsdfReplayDepthContext] = []
            actual_build = build_tsdf_replay_depth_context
            import spatialforge.tsdf_replay_depth_context as context_module

            actual_read_depth = context_module._read_depth

            def capture_context(plan, session):
                context = actual_build(plan, session)
                contexts.append(context)
                return context

            with (
                forbidden_context_cli_calls() as forbidden,
                patch(
                    "spatialforge.cli.build_tsdf_replay_depth_context",
                    side_effect=capture_context,
                ) as builder,
                patch(
                    "spatialforge.tsdf_replay_depth_context._read_depth",
                    wraps=actual_read_depth,
                ) as depth_reader,
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                exit_code = main(
                    [
                        "reconstruct",
                        "tsdf-block-replay-context",
                        str(plan_path),
                        str(session_path),
                    ]
                )
            after_tree = tree_snapshot(temporary_root)

        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(after_tree, before_tree)
        builder.assert_called_once()
        self.assertEqual(depth_reader.call_count, 2)
        self.assertEqual(len(contexts), 1)
        self.assertEqual(contexts[0].depth_payload_bytes, 64)
        for forbidden_call in forbidden:
            forbidden_call.assert_not_called()

        expected_output = (
            "TSDF BLOCK REPLAY DEPTH CONTEXT CHECK scan-synthetic-0001\n"
            "artifact: valid\n"
            "session_replay: matched\n"
            "selection: frame_stride=1 total=2 selected=2\n"
            "observation[0]: sequence=0 status=ready depth_decoded=yes\n"
            "observation[1]: sequence=1 status=ready depth_decoded=yes\n"
            "status_counts: ready=2\n"
            "ready_observations: 2\n"
            "depth_frames_decoded: 2\n"
            "depth_layout: shape=(2, 2) dtype=float64 samples=8 "
            "payload_bytes=64\n"
            "context_immutable: yes\n"
            "tsdf_storage_allocated: no\n"
            "voxel_evaluation_performed: no\n"
            "voxel_observation_traversal_performed: no\n"
            "voxel_address_traversal_performed: no\n"
            "fusion_block_traversal_performed: no\n"
            "ray_traversal_performed: no\n"
            "full_fusion_performed: no\n"
            "artifact_written: no\n"
            "context_persisted: no\n"
            f"plan_sha256: {PLAN_SHA256}\n"
            f"replay_digest_sha256: {REPLAY_SHA256}\n"
        )
        self.assertEqual(stdout.getvalue(), expected_output)


if __name__ == "__main__":
    unittest.main()
