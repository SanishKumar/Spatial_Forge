"""The depth-comparison report, on noise whose properties are planted.

The report exists to say whether a dataset's noisy depth is quantised and
whether it is biased. Both answers are checked on sequences made here with
a known rule: rounded to the nearest disparity level, rounded up, and not
quantised at all. If the report could not tell those three apart, what it
says about anyone else's files would mean nothing.
"""

from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
from PIL import Image

from tools.depth_noise_report import (
    KINECT_DISPARITY_CONSTANT,
    compare_sequences,
    main,
    smooth_mask,
)
from tools.simulate_kinect_noise import apply_depth_noise

TEST_ROOT = Path(__file__).resolve().parent
SHAPE = (60, 80)
FRAMES = 6


def clean_depth(index: int) -> np.ndarray:
    """A gently sloping wall between 1.2 and 3.4 m, different each frame."""

    rows, columns = np.mgrid[0:SHAPE[0], 0:SHAPE[1]]
    return 1.2 + 0.3 * index + 0.0015 * columns + 0.001 * rows


def write_sequence(root: Path, name: str, frames: list[np.ndarray]) -> Path:
    folder = root / name
    (folder / "depth").mkdir(parents=True)
    lines = []
    for index, depth_m in enumerate(frames):
        raw = np.floor(depth_m * 5000.0 + 0.5).astype(np.uint16)
        Image.fromarray(raw).save(folder / "depth" / f"{index}.png")
        lines.append(f"{index} depth/{index}.png\n")
    (folder / "depth.txt").write_bytes("".join(lines).encode("ascii"))
    return folder


def degraded(rule: str, index: int) -> np.ndarray:
    """One frame of noise under a named rule, from a fixed seed."""

    exact = clean_depth(index)
    noise = np.random.default_rng([41, index]).normal(0.0, 1.0 / 6.0, SHAPE)
    zero = np.zeros(SHAPE)
    if rule == "nearest":
        return apply_depth_noise(exact, zero, zero, noise)
    levels = KINECT_DISPARITY_CONSTANT / (exact * 100.0)
    if rule == "up":
        return KINECT_DISPARITY_CONSTANT / np.ceil(levels + noise) / 100.0
    if rule == "unquantised":
        return KINECT_DISPARITY_CONSTANT / (levels + noise) / 100.0
    raise AssertionError(rule)


class PlantedNoiseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = tempfile.TemporaryDirectory(dir=TEST_ROOT)
        cls.addClassCleanup(cls.directory.cleanup)
        root = Path(cls.directory.name)
        cls.clean = write_sequence(
            root, "clean", [clean_depth(i) for i in range(FRAMES)]
        )
        cls.results = {
            rule: compare_sequences(
                cls.clean,
                write_sequence(
                    root, rule, [degraded(rule, i) for i in range(FRAMES)]
                ),
                frame_stride=1,
            )
            for rule in ("nearest", "up", "unquantised")
        }

    def test_rounding_to_the_nearest_level_is_unbiased(self) -> None:
        core = self.results["nearest"]["core_disparity_difference"]
        self.assertAlmostEqual(core["median"], 0.0, delta=0.03)
        self.assertAlmostEqual(core["mean"], 0.0, delta=0.03)

    def test_rounding_up_shows_as_half_a_level_towards_the_camera(
        self,
    ) -> None:
        result = self.results["up"]
        core = result["core_disparity_difference"]
        self.assertAlmostEqual(core["median"], 0.5, delta=0.03)
        self.assertAlmostEqual(core["mean"], 0.5, delta=0.03)
        for band in result["bands"]:
            # Nearer by about half of whatever one level is at that depth.
            self.assertLess(band["median_depth_difference_mm"], 0.0)
            self.assertAlmostEqual(
                band["median_depth_difference_mm"],
                -0.5 * band["one_level_mm"],
                delta=0.25 * band["one_level_mm"],
            )

    def test_quantised_depth_is_told_from_depth_that_is_not(self) -> None:
        for rule in ("nearest", "up"):
            self.assertLess(
                self.results[rule]["median_distance_to_whole_disparity"], 0.05
            )
        self.assertAlmostEqual(
            self.results["unquantised"]["median_distance_to_whole_disparity"],
            0.25,
            delta=0.03,
        )
        # Not quantised, and not biased either.
        self.assertAlmostEqual(
            self.results["unquantised"]["core_disparity_difference"]["median"],
            0.0,
            delta=0.03,
        )

    def test_the_spread_is_the_noise_that_was_drawn(self) -> None:
        # One sixth of a level of Gaussian noise, with nothing added by
        # rounding because nothing is rounded.
        self.assertAlmostEqual(
            self.results["unquantised"]["core_disparity_difference"]["std"],
            1.0 / 6.0,
            delta=0.02,
        )
        for rule in ("nearest", "up", "unquantised"):
            self.assertEqual(self.results[rule]["frames"], FRAMES)
            self.assertGreater(self.results[rule]["core_fraction"], 0.999)

    def test_a_level_is_the_size_it_should_be(self) -> None:
        for band in self.results["nearest"]["bands"]:
            centre = (band["from_m"] + band["to_m"]) / 2.0
            expected = 10.0 * (centre * 100.0) ** 2 / KINECT_DISPARITY_CONSTANT
            self.assertAlmostEqual(
                band["one_level_mm"], expected, delta=0.35 * expected
            )
        at_two_metres = 10.0 * 200.0**2 / KINECT_DISPARITY_CONSTANT
        self.assertAlmostEqual(at_two_metres, 11.4, places=1)


class CountingTests(unittest.TestCase):
    def test_lost_and_grossly_wrong_pixels_are_counted_exactly(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            root = Path(temporary)
            exact = np.full(SHAPE, 2.0)
            exact[0, 0] = 0.0
            broken = exact.copy()
            broken[10, 10:20] = 0.0          # ten pixels lost
            broken[20, 10:14] = 2.5          # four half a metre out
            result = compare_sequences(
                write_sequence(root, "clean", [exact]),
                write_sequence(root, "noisy", [broken]),
                frame_stride=1,
            )
        present = SHAPE[0] * SHAPE[1] - 1
        self.assertEqual(result["pixels_with_exact_depth"], present)
        self.assertAlmostEqual(result["lost_fraction"], 10 / present)
        self.assertAlmostEqual(
            result["beyond_100mm_fraction"], 4 / (present - 10)
        )
        self.assertEqual(result["median_absolute_difference_mm"], 0.0)
        # Four pixels far outside the core, none of them in the statistics.
        self.assertLess(result["core_fraction"], 1.0)
        self.assertEqual(result["core_disparity_difference"]["median"], 0.0)

    def test_smooth_regions_stop_at_edges_and_holes(self) -> None:
        depth = np.full((7, 9), 2.0)
        depth[:, 5:] = 2.5
        depth[1, 1] = 0.0
        mask = smooth_mask(depth, 0.005)
        # Either side of the step, and everything around the hole.
        self.assertFalse(bool(mask[3, 4]))
        self.assertFalse(bool(mask[3, 5]))
        self.assertFalse(bool(mask[:3, :3].any()))
        self.assertTrue(bool(mask[4:, :4].all()))
        self.assertTrue(bool(mask[:, 6:].all()))
        # A slope of 1 mm a pixel is smooth at 5 mm and not at 1.5 mm.
        ramp = 2.0 + 0.001 * np.mgrid[0:7, 0:9][1]
        self.assertTrue(bool(smooth_mask(ramp, 0.005)[3, 4]))
        self.assertFalse(bool(smooth_mask(ramp, 0.0015)[3, 4]))

    def test_frames_are_paired_by_timestamp_not_by_position(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            root = Path(temporary)
            clean = write_sequence(
                root, "clean", [np.full(SHAPE, 2.0), np.full(SHAPE, 3.0)]
            )
            noisy = write_sequence(root, "noisy", [np.full(SHAPE, 3.0)])
            # The noisy sequence's only frame is the clean one's second.
            (noisy / "depth.txt").write_bytes(b"1.0 depth/0.png\n")
            result = compare_sequences(clean, noisy, frame_stride=1)
        self.assertEqual(result["frames"], 1)
        self.assertEqual(result["median_absolute_difference_mm"], 0.0)


class CommandTests(unittest.TestCase):
    def test_command_line_prints_the_three_answers(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            root = Path(temporary)
            clean = write_sequence(
                root, "clean", [clean_depth(i) for i in range(2)]
            )
            noisy = write_sequence(
                root, "noisy", [degraded("up", i) for i in range(2)]
            )
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [str(clean), str(noisy), "--frame-stride", "1"]
                )
        printed = stdout.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("frames compared: 2 of 2 shared", printed)
        self.assertIn("(0.25 if not quantised)", printed)
        self.assertIn("noisy minus exact, in levels: median +0.5", printed)

    def test_unusable_input_is_refused(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            root = Path(temporary)
            clean = write_sequence(root, "clean", [np.full(SHAPE, 2.0)])
            noisy = write_sequence(root, "noisy", [np.full(SHAPE, 2.0)])

            def refused(message: str, *folders: Path, **kw) -> None:
                with self.assertRaises(SystemExit) as raised:
                    compare_sequences(*folders, **kw)
                self.assertIn(message, str(raised.exception))

            refused("smoothness tolerance", clean, noisy, smooth_m=0.0)
            refused("disparity constant", clean, noisy, constant=-1.0)
            refused("missing", clean, root / "absent")
            other = write_sequence(root, "other", [np.full((30, 40), 2.0)])
            refused("differ in size", clean, other)
            (other / "depth.txt").write_bytes(b"7 depth/0.png\n")
            refused("no timestamp in common", clean, other)
            (other / "depth.txt").write_bytes(b"0 depth/0.png extra\n")
            refused("expected 'timestamp relative/path'", clean, other)
            (other / "depth.txt").write_bytes(b"x depth/0.png\n")
            refused("timestamp is not a number", clean, other)
            (other / "depth.txt").write_bytes(b"0 ../clean/depth/0.png\n")
            refused("must stay inside", clean, other)
            (other / "depth.txt").write_bytes(
                b"0 depth/0.png\n0.0 depth/0.png\n"
            )
            refused("repeated timestamp", clean, other)
            Image.new("L", (SHAPE[1], SHAPE[0]), 9).save(
                noisy / "depth" / "0.png"
            )
            refused("must be 16-bit", clean, noisy)
            empty = write_sequence(root, "empty", [np.zeros(SHAPE)])
            refused("no pixel has depth in both", clean, empty)
            rough = np.full(SHAPE, 2.0)
            rough[::2, :] = 2.5
            refused(
                "no smooth region",
                write_sequence(root, "rough", [rough]),
                write_sequence(root, "rough-noisy", [rough]),
            )
            with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
                main([str(clean), str(noisy), "--frame-stride", "0"])


if __name__ == "__main__":
    unittest.main()
