"""The vectorised block planner, pinned to the per-pixel reference.

``_plan_observation_blocks_scalar`` defines block planning one depth sample
at a time. The vectorised planner must select exactly the same blocks and
count exactly the same samples, which makes the strongest available check a
simple one: the ``.sftplan`` written through either path is the same file,
byte for byte.
"""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from spatialforge import plan_tsdf_blocks
from spatialforge import tsdf_block_plan as planner
from spatialforge.errors import PointCloudError, TsdfError
from spatialforge.point_cloud import (
    _read_depth,
    _read_depth_array,
    _validate_reconstruction_contract,
)
from spatialforge.replay import replay_session
from spatialforge.session_loader import load_scan_session

from tests.room_fixture import room_session

TEST_ROOT = Path(__file__).resolve().parent
FIXTURE = TEST_ROOT / "fixtures" / "minimal.vgsession"

# Voxel size, truncation and stride. Chosen to cover the reference fixture
# values, off-grid sizes, a truncation equal to the voxel (the narrowest legal
# band), and a band several blocks wide.
FIXTURE_CASES = (
    (0.125, 0.5, 1),
    (0.125, 0.5, 2),
    (0.05, 0.05, 1),
    (0.01, 0.2, 1),
)
ROOM_CASES = (
    (0.04, 0.12, 1),
    (0.02, 0.02, 3),
    (0.0173, 0.0519, 1),
    (0.03, 0.5, 2),
)


def plan_bytes(
    session_path: Path,
    output: Path,
    voxel_size_m: float,
    truncation_m: float,
    frame_stride: int,
) -> tuple[bytes, object]:
    report = plan_tsdf_blocks(
        load_scan_session(session_path),
        output,
        voxel_size_m=voxel_size_m,
        truncation_m=truncation_m,
        frame_stride=frame_stride,
    )
    return output.read_bytes(), report


def reference_only():
    """Force every frame down the per-pixel reference path."""

    return patch(
        "spatialforge.tsdf_block_plan._plan_observation_blocks_vector",
        return_value=None,
    )


def paired_observations(session_path: Path):
    session = load_scan_session(session_path)
    camera, depth_scale_m = _validate_reconstruction_contract(session)
    observations = [
        observation
        for observation in replay_session(session).observations
        if observation.depth is not None and observation.pose is not None
    ]
    return session, camera, depth_scale_m, observations


class VectorPlannerParityTests(unittest.TestCase):
    def assert_same_plan(self, session_path: Path, cases) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            temporary_root = Path(temporary_dir)
            for position, (voxel, truncation, stride) in enumerate(cases):
                with self.subTest(
                    voxel=voxel,
                    truncation=truncation,
                    stride=stride,
                ):
                    vector, vector_report = plan_bytes(
                        session_path,
                        temporary_root / f"vector-{position}.sftplan",
                        voxel,
                        truncation,
                        stride,
                    )
                    with reference_only():
                        scalar, scalar_report = plan_bytes(
                            session_path,
                            temporary_root / f"scalar-{position}.sftplan",
                            voxel,
                            truncation,
                            stride,
                        )
                    self.assertEqual(vector, scalar)
                    self.assertEqual(
                        vector_report.output_digest_sha256,
                        scalar_report.output_digest_sha256,
                    )
                    self.assertEqual(
                        vector_report.active_blocks,
                        scalar_report.active_blocks,
                    )

    def test_fixture_plans_are_byte_identical_to_the_reference(self) -> None:
        self.assert_same_plan(FIXTURE, FIXTURE_CASES)

    def test_room_plans_are_byte_identical_to_the_reference(self) -> None:
        self.assert_same_plan(room_session(), ROOM_CASES)

    def test_every_room_frame_matches_blocks_and_counts(self) -> None:
        session, camera, depth_scale_m, observations = paired_observations(
            room_session()
        )
        for voxel, truncation, _ in ROOM_CASES:
            extent = voxel * planner.TSDF_BLOCK_RESOLUTION
            for observation in observations:
                with self.subTest(
                    voxel=voxel,
                    sequence=observation.sequence,
                ):
                    planned = planner._plan_observation_blocks_vector(
                        session,
                        observation,
                        camera,
                        depth_scale_m,
                        extent,
                        truncation,
                    )
                    surface: set = set()
                    active: set = set()
                    counts = planner._plan_observation_blocks_scalar(
                        session,
                        observation,
                        camera,
                        depth_scale_m,
                        extent,
                        truncation,
                        surface,
                        active,
                    )
                    self.assertIsNotNone(planned)
                    self.assertEqual(planned[0], surface)
                    self.assertEqual(planned[1], active)
                    self.assertEqual(planned[2:], counts)

    def test_clean_frames_never_reach_the_per_pixel_loop(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            with patch(
                "spatialforge.tsdf_block_plan."
                "_plan_observation_blocks_scalar"
            ) as scalar:
                plan_bytes(
                    room_session(),
                    Path(temporary_dir) / "room.sftplan",
                    0.04,
                    0.12,
                    1,
                )

        scalar.assert_not_called()


class VectorPlannerDeferralTests(unittest.TestCase):
    """Irregular frames are not handled twice; the reference owns them."""

    def test_nonfinite_coordinates_raise_the_reference_error(self) -> None:
        """An overflowing sample must report the reference's exact pixel.

        A validated session cannot overflow on its own, so the frame is
        driven directly: a depth scale that puts every sample near the top
        of the float64 range, and a camera translated just as far along the
        axis its depth maps to, so their sum is infinite.
        """

        session, camera, _, observations = paired_observations(FIXTURE)
        observation = observations[0]
        transform = list(observation.pose.data["T_world_camera"])
        transform[3] = 1e308
        overflowing = replace(
            observation,
            pose=replace(
                observation.pose,
                data={**observation.pose.data, "T_world_camera": transform},
            ),
        )
        arguments = (session, overflowing, camera, 1e305, 1.0, 0.5)

        self.assertIsNone(planner._plan_observation_blocks_vector(*arguments))
        with self.assertRaises(TsdfError) as reference:
            planner._plan_observation_blocks_scalar(*arguments, set(), set())
        surface: set = set()
        active: set = set()
        with self.assertRaises(TsdfError) as dispatched:
            planner._plan_observation_blocks(*arguments, surface, active)

        self.assertEqual(str(dispatched.exception), str(reference.exception))
        self.assertIn("non-finite coordinate", str(reference.exception))
        self.assertIn("pixel (0, 0)", str(reference.exception))
        self.assertEqual((surface, active), (set(), set()))

    def test_a_span_too_wide_to_expand_defers_without_changing_the_plan(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            temporary_root = Path(temporary_dir)
            expected, _ = plan_bytes(
                FIXTURE,
                temporary_root / "normal.sftplan",
                0.125,
                0.5,
                1,
            )
            with (
                patch(
                    "spatialforge.tsdf_block_plan.MAX_VECTOR_SPAN_OFFSETS",
                    1,
                ),
                patch(
                    "spatialforge.tsdf_block_plan."
                    "_plan_observation_blocks_scalar",
                    wraps=planner._plan_observation_blocks_scalar,
                ) as scalar,
            ):
                deferred, _ = plan_bytes(
                    FIXTURE,
                    temporary_root / "deferred.sftplan",
                    0.125,
                    0.5,
                    1,
                )

        self.assertEqual(deferred, expected)
        self.assertEqual(scalar.call_count, 2)

    def test_a_deferred_frame_sees_the_plan_untouched(self) -> None:
        """The vector attempt must not leak partial work into the sets."""

        session, camera, depth_scale_m, observations = paired_observations(
            FIXTURE
        )
        surface = {(9, 9, 9)}
        active = {(9, 9, 9)}
        seen: list[tuple[set, set]] = []

        def record(*arguments):
            seen.append((set(arguments[6]), set(arguments[7])))
            return (0, 0)

        with (
            patch(
                "spatialforge.tsdf_block_plan.MAX_VECTOR_SPAN_OFFSETS",
                1,
            ),
            patch(
                "spatialforge.tsdf_block_plan."
                "_plan_observation_blocks_scalar",
                side_effect=record,
            ),
        ):
            planner._plan_observation_blocks(
                session,
                observations[0],
                camera,
                depth_scale_m,
                1.0,
                0.5,
                surface,
                active,
            )

        self.assertEqual(seen, [({(9, 9, 9)}, {(9, 9, 9)})])


class DepthArrayDecodingTests(unittest.TestCase):
    def test_array_decoder_matches_the_reference_decoder(self) -> None:
        rng = np.random.default_rng(20260806)
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            temporary_root = Path(temporary_dir)
            samples = rng.integers(0, 65536, size=(7, 11), dtype=np.uint16)
            samples[0, 0] = 0
            samples[-1, -1] = 65535

            png = temporary_root / "depth.png"
            Image.fromarray(samples).save(png)
            pgm = temporary_root / "depth.pgm"
            pgm.write_text(
                "P2\n11 7\n65535\n"
                + "\n".join(
                    " ".join(str(int(value)) for value in row)
                    for row in samples
                )
                + "\n",
                encoding="ascii",
            )
            binary = temporary_root / "depth-binary.pgm"
            binary.write_bytes(
                b"P5\n11 7\n65535\n" + samples.astype(">u2").tobytes()
            )
            fixture_depth = FIXTURE / "data" / "depth" / "000000.pgm"

            for path, width, height in (
                (png, 11, 7),
                (pgm, 11, 7),
                (binary, 11, 7),
                (fixture_depth, 2, 2),
            ):
                with self.subTest(path=path.name):
                    reference = _read_depth(path, width, height)
                    decoded = _read_depth_array(path, width, height)
                    self.assertEqual(decoded.dtype, np.int64)
                    self.assertEqual(decoded.shape, (width * height,))
                    self.assertEqual(tuple(decoded.tolist()), reference)
            self.assertEqual(
                tuple(_read_depth_array(png, 11, 7).tolist()),
                tuple(int(value) for value in samples.reshape(-1)),
            )

    def test_array_decoder_refuses_what_the_reference_refuses(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            temporary_root = Path(temporary_dir)
            eight_bit = temporary_root / "eight.png"
            Image.fromarray(np.zeros((2, 2), dtype=np.uint8)).save(eight_bit)
            garbage = temporary_root / "garbage.png"
            garbage.write_bytes(b"not an image")
            fixture_depth = FIXTURE / "data" / "depth" / "000000.pgm"

            for path, width, height in (
                (eight_bit, 2, 2),
                (garbage, 2, 2),
                (fixture_depth, 3, 2),
            ):
                with self.subTest(path=path.name):
                    with self.assertRaises(PointCloudError) as reference:
                        _read_depth(path, width, height)
                    with self.assertRaises(PointCloudError) as decoded:
                        _read_depth_array(path, width, height)
                    self.assertEqual(
                        str(decoded.exception),
                        str(reference.exception),
                    )


class UniqueRowTests(unittest.TestCase):
    def test_packed_keys_agree_with_a_row_wise_sort(self) -> None:
        rng = np.random.default_rng(7)
        for columns, low, high in (
            (3, -40, 40),
            (6, -5, 9),
            (3, -(2**31), 2**31 - 1),
            (1, 0, 3),
        ):
            with self.subTest(columns=columns, low=low):
                rows = rng.integers(
                    low,
                    high,
                    size=(5_000, columns),
                    dtype=np.int64,
                )
                rows[::7] = rows[0]
                expected = {tuple(row) for row in rows.tolist()}
                actual = planner._unique_rows(rows)
                self.assertEqual(
                    {tuple(row) for row in actual.tolist()},
                    expected,
                )
                self.assertEqual(len(actual), len(expected))

    def test_ranges_too_wide_to_pack_fall_back_to_the_row_sort(self) -> None:
        rows = np.array(
            [
                [-(2**31), 2**31 - 1, -(2**31)],
                [2**31 - 1, -(2**31), 2**31 - 1],
                [-(2**31), 2**31 - 1, -(2**31)],
            ],
            dtype=np.int64,
        )
        with patch(
            "spatialforge.tsdf_block_plan.np.unique",
            wraps=np.unique,
        ) as unique:
            actual = planner._unique_rows(rows)

        self.assertEqual(len(actual), 2)
        self.assertTrue(
            any(call.kwargs.get("axis") == 0 for call in unique.call_args_list)
        )


if __name__ == "__main__":
    unittest.main()
