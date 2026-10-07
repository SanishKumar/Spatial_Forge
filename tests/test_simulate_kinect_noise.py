"""The depth-noise simulator, checked against the equation it claims.

A sensor-noise result is only as good as the noise. If the simulator were
off by a unit, or quietly drew the same noise for every frame, the
reconstruction would still come out and the numbers would still look
plausible. So the equation is checked on inputs where its answer can be
worked out by hand, its randomness is checked as a distribution, and the
files the tool writes are rebuilt independently from the same seed.
"""

from __future__ import annotations

import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
from PIL import Image

from tools.simulate_kinect_noise import (
    KINECT_DISPARITY_CONSTANT,
    SIGMA_DISPARITY,
    SIGMA_SHIFT_PIXELS,
    apply_depth_noise,
    draw_noise,
    main,
    simulate_kinect_noise,
)

TEST_ROOT = Path(__file__).resolve().parent


def still(shape: tuple[int, int]) -> np.ndarray:
    return np.zeros(shape)


def quantised(depth_m: float, extra_disparity: float = 0.0) -> float:
    """Equation 3 for one pixel with no shift, by hand."""

    disparity = KINECT_DISPARITY_CONSTANT / (depth_m * 100.0)
    return (
        KINECT_DISPARITY_CONSTANT
        / math.floor(disparity + extra_disparity + 0.5)
        / 100.0
    )


class EquationTests(unittest.TestCase):
    def test_noise_free_draws_leave_only_the_quantisation(self) -> None:
        depth = np.full((4, 5), 2.0)
        result = apply_depth_noise(
            depth, still(depth.shape), still(depth.shape), still(depth.shape)
        )
        # 35130 / 200 cm is a disparity of 175.65, which rounds to 176.
        self.assertEqual(quantised(2.0), 35130.0 / 176.0 / 100.0)
        np.testing.assert_array_equal(result, np.full((4, 5), quantised(2.0)))

    def test_the_depth_steps_are_a_kinects(self) -> None:
        # Neighbouring whole disparities, as depths: 11 mm apart at 2 m
        # (disparities 176 and 175) and 73 mm apart at 5 m (70 and 69).
        for depth_m, step_mm in ((2.0, 11.4), (5.0, 72.7)):
            disparity = round(KINECT_DISPARITY_CONSTANT / (depth_m * 100.0))
            step = 1000.0 * (
                KINECT_DISPARITY_CONSTANT / (disparity - 1) / 100.0
                - KINECT_DISPARITY_CONSTANT / disparity / 100.0
            )
            self.assertAlmostEqual(step, step_mm, delta=0.1)
        # And the function lands every depth on one of those levels.
        depth = np.linspace(0.5, 5.0, 60).reshape((6, 10))
        result = apply_depth_noise(
            depth, still(depth.shape), still(depth.shape), still(depth.shape)
        )
        disparities = KINECT_DISPARITY_CONSTANT / (result * 100.0)
        np.testing.assert_allclose(
            disparities, np.round(disparities), atol=1e-9
        )
        self.assertLess(float(np.abs(result - depth).max()), 0.04)

    def test_disparity_noise_moves_the_level(self) -> None:
        depth = np.full((3, 3), 2.0)
        for extra in (-1.0, 0.3, 0.9, 2.6):
            with self.subTest(extra=extra):
                result = apply_depth_noise(
                    depth,
                    still(depth.shape),
                    still(depth.shape),
                    np.full(depth.shape, extra),
                )
                np.testing.assert_array_equal(
                    result, np.full((3, 3), quantised(2.0, extra))
                )
        # 175.65 stays at 176 until the noise passes +0.85.
        self.assertEqual(quantised(2.0, 0.3), quantised(2.0))
        self.assertEqual(quantised(2.0, 0.9), 35130.0 / 177.0 / 100.0)
        self.assertEqual(quantised(2.0, -1.0), 35130.0 / 175.0 / 100.0)

    def test_a_shifted_pixel_reads_the_interpolated_depth(self) -> None:
        rows, columns = np.mgrid[0:6, 0:8]
        ramp = 1.0 + 0.01 * columns + 0.02 * rows
        # A constant large enough that rounding the disparity is far below
        # what is being checked.
        result = apply_depth_noise(
            ramp,
            np.full(ramp.shape, 0.25),
            np.full(ramp.shape, -0.5),
            still(ramp.shape),
            constant=1e12,
        )
        expected = 1.0 + 0.01 * np.clip(columns + 0.25, 0, 7) + 0.02 * np.clip(
            rows - 0.5, 0, 5
        )
        np.testing.assert_allclose(result, expected, rtol=1e-9)
        # Off the edge of the image means at the edge, not beyond it.
        far = apply_depth_noise(
            ramp,
            np.full(ramp.shape, 50.0),
            np.full(ramp.shape, -50.0),
            still(ramp.shape),
            constant=1e12,
        )
        np.testing.assert_allclose(far, np.full(ramp.shape, 1.07), rtol=1e-9)

    def test_missing_depth_stays_missing_and_spreads_to_what_reads_it(
        self,
    ) -> None:
        depth = np.full((5, 5), 2.0)
        depth[2, 2] = 0.0
        untouched = apply_depth_noise(
            depth, still(depth.shape), still(depth.shape), still(depth.shape)
        )
        self.assertEqual(untouched[2, 2], 0.0)
        self.assertEqual(int(np.count_nonzero(untouched == 0.0)), 1)
        # Every pixel now reads half a pixel to its right, so the pixel to
        # the left of the hole interpolates across it and has no depth.
        shifted = apply_depth_noise(
            depth,
            np.full(depth.shape, 0.5),
            still(depth.shape),
            still(depth.shape),
        )
        self.assertEqual(shifted[2, 1], 0.0)
        self.assertEqual(shifted[2, 2], 0.0)
        self.assertGreater(shifted[2, 3], 0.0)
        self.assertGreater(shifted[1, 1], 0.0)

    def test_a_disparity_below_one_is_no_depth(self) -> None:
        depth = np.full((2, 2), 2.0)
        result = apply_depth_noise(
            depth,
            still(depth.shape),
            still(depth.shape),
            np.full(depth.shape, -500.0),
        )
        np.testing.assert_array_equal(result, np.zeros((2, 2)))

    def test_unusable_inputs_are_refused(self) -> None:
        depth = np.full((3, 3), 2.0)
        zero = still(depth.shape)
        with self.assertRaisesRegex(ValueError, "at least 2 by 2"):
            apply_depth_noise(np.ones((1, 5)), zero, zero, zero)
        with self.assertRaisesRegex(ValueError, "shift_y must have"):
            apply_depth_noise(depth, zero, np.zeros((2, 2)), zero)
        for constant in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "disparity constant"):
                apply_depth_noise(depth, zero, zero, zero, constant=constant)


class DrawTests(unittest.TestCase):
    def test_noise_has_the_stated_spread(self) -> None:
        shift_x, shift_y, disparity = draw_noise(5, 0, (300, 400))
        for field, sigma in (
            (shift_x, SIGMA_SHIFT_PIXELS),
            (shift_y, SIGMA_SHIFT_PIXELS),
            (disparity, SIGMA_DISPARITY),
        ):
            self.assertAlmostEqual(float(field.mean()), 0.0, delta=0.01)
            self.assertAlmostEqual(float(field.std()), sigma, delta=0.01)
        # The three fields are separate draws, not one reused.
        self.assertLess(
            abs(float(np.corrcoef(shift_x.ravel(), shift_y.ravel())[0, 1])),
            0.02,
        )

    def test_each_frame_and_each_seed_has_its_own_noise(self) -> None:
        first = draw_noise(5, 0, (8, 8))
        again = draw_noise(5, 0, (8, 8))
        other_frame = draw_noise(5, 1, (8, 8))
        other_seed = draw_noise(6, 0, (8, 8))
        for a, b in zip(first, again):
            np.testing.assert_array_equal(a, b)
        for a, b in zip(first, other_frame):
            self.assertFalse(bool(np.array_equal(a, b)))
        for a, b in zip(first, other_seed):
            self.assertFalse(bool(np.array_equal(a, b)))

    def test_a_flat_wall_lands_on_the_levels_the_noise_predicts(self) -> None:
        # At 2 m the disparity is 175.65. With sd = 1/6 it rounds to 176
        # unless the noise is below -0.15, nine tenths of a deviation.
        depth = np.full((200, 200), 2.0)
        result = apply_depth_noise(depth, *draw_noise(9, 0, depth.shape))
        disparities = np.round(KINECT_DISPARITY_CONSTANT / (result * 100.0))
        below = 0.5 * (1.0 + math.erf(-0.9 / math.sqrt(2.0)))
        self.assertAlmostEqual(
            float(np.mean(disparities == 175.0)), below, delta=0.01
        )
        self.assertAlmostEqual(
            float(np.mean(disparities == 176.0)), 1.0 - below, delta=0.01
        )


def write_sequence(root: Path, frames: int = 3) -> Path:
    source = root / "clean"
    (source / "rgb").mkdir(parents=True)
    (source / "depth").mkdir()
    rows, columns = np.mgrid[0:12, 0:16]
    depth = np.round((1.5 + 0.02 * columns + 0.03 * rows) * 5000.0).astype(
        np.uint16
    )
    depth[0, 0] = 0
    listing = []
    for index in range(frames):
        # Every frame has the same depth, so any difference between the
        # noisy frames is the noise.
        Image.fromarray(depth).save(source / "depth" / f"{index}.png")
        Image.new("RGB", (16, 12), (10 * index, 20, 30)).save(
            source / "rgb" / f"{index}.png"
        )
        listing.append(index)
    (source / "rgb.txt").write_bytes(
        "".join(f"{i} rgb/{i}.png\n" for i in listing).encode("ascii")
    )
    (source / "depth.txt").write_bytes(
        b"# timestamp filename\n"
        + "".join(f"{i} depth/{i}.png\n" for i in listing).encode("ascii")
    )
    (source / "groundtruth.txt").write_bytes(b"0 0 0 0 0 0 0 1\n")
    return source


def decoded(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image).copy()


class SequenceTests(unittest.TestCase):
    def test_the_written_depth_is_the_equation_on_the_seeded_draws(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            root = Path(temporary)
            source = write_sequence(root)
            counts = simulate_kinect_noise(source, root / "noisy", seed=7)
            clean = decoded(source / "depth" / "0.png").astype(np.float64)
            written = [
                decoded(root / "noisy" / "depth" / f"{index}.png")
                for index in range(3)
            ]
            description = json.loads(
                (root / "noisy" / "noise.json").read_bytes()
            )
            copied = {
                name: (root / "noisy" / name).read_bytes()
                == (source / name).read_bytes()
                for name in (
                    "rgb.txt",
                    "depth.txt",
                    "groundtruth.txt",
                    "rgb/0.png",
                    "rgb/2.png",
                )
            }
            leftovers = sorted(path.name for path in root.iterdir())

        for index, image in enumerate(written):
            self.assertEqual(image.dtype, np.uint16)
            expected = np.floor(
                apply_depth_noise(
                    clean / 5000.0, *draw_noise(7, index, clean.shape)
                )
                * 5000.0
                + 0.5
            )
            np.testing.assert_array_equal(image, expected)
        # Same clean depth in, different noise per frame out.
        self.assertFalse(bool(np.array_equal(written[0], written[1])))
        self.assertTrue(bool(np.any(written[0] != clean)))
        self.assertEqual(int(written[0][0, 0]), 0)
        self.assertTrue(all(copied.values()), copied)
        self.assertEqual(counts["frames"], 3)
        self.assertEqual(counts["valid_depth_before"], 3 * (12 * 16 - 1))
        self.assertLessEqual(
            counts["valid_depth_after"], counts["valid_depth_before"]
        )
        self.assertEqual(description["seed"], 7)
        self.assertEqual(description["frames"], 3)
        self.assertEqual(description["disparity_constant"], 35130.0)
        self.assertEqual(description["sigma_shift_pixels"], 0.5)
        self.assertEqual(leftovers, ["clean", "noisy"])

    def test_the_same_seed_gives_the_same_depth_and_another_does_not(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            root = Path(temporary)
            source = write_sequence(root, frames=2)
            simulate_kinect_noise(source, root / "a", seed=3)
            simulate_kinect_noise(source, root / "b", seed=3)
            simulate_kinect_noise(source, root / "c", seed=4)
            images = {
                name: [
                    decoded(root / name / "depth" / f"{index}.png")
                    for index in range(2)
                ]
                for name in "abc"
            }
        for index in range(2):
            np.testing.assert_array_equal(
                images["a"][index], images["b"][index]
            )
            self.assertFalse(
                bool(np.array_equal(images["a"][index], images["c"][index]))
            )

    def test_zero_sigmas_leave_only_quantisation(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            root = Path(temporary)
            source = write_sequence(root, frames=1)
            simulate_kinect_noise(
                source, root / "noisy", sigma_shift=0.0, sigma_disparity=0.0
            )
            clean = decoded(source / "depth" / "0.png").astype(np.float64)
            noisy = decoded(root / "noisy" / "depth" / "0.png").astype(
                np.float64
            )
        valid = clean > 0
        self.assertTrue(bool(np.all(noisy[~valid] == 0)))
        # Between 1.5 and 2.2 m the levels are 6 to 14 mm apart, so no
        # depth moves by more than half of that: 7 mm, or 35 raw units.
        self.assertLessEqual(
            float(np.abs(noisy[valid] - clean[valid]).max()), 37.0
        )
        self.assertTrue(bool(np.any(noisy[valid] != clean[valid])))

    def test_command_line(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            root = Path(temporary)
            source = write_sequence(root, frames=2)
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        str(source),
                        str(root / "noisy"),
                        "--seed",
                        "11",
                        "--sigma-disparity",
                        "0.5",
                    ]
                )
            description = json.loads(
                (root / "noisy" / "noise.json").read_bytes()
            )
        self.assertEqual(exit_code, 0)
        self.assertIn("frames: 2;", stdout.getvalue())
        self.assertEqual(description["seed"], 11)
        self.assertEqual(description["sigma_disparity"], 0.5)


class RefusalTests(unittest.TestCase):
    def test_unusable_sequences_and_outputs_are_refused(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            root = Path(temporary)
            source = write_sequence(root, frames=1)

            def refused(message: str, *, output: Path | None = None, **kw):
                with self.assertRaises(SystemExit) as raised:
                    simulate_kinect_noise(
                        source,
                        root / "noisy" if output is None else output,
                        **kw,
                    )
                self.assertIn(message, str(raised.exception))
                self.assertFalse((root / "noisy").exists())
                # No staging directory is left behind either.
                self.assertEqual(
                    [
                        path.name
                        for path in root.iterdir()
                        if path.name.startswith(".")
                    ],
                    [],
                )

            existing = root / "existing"
            existing.mkdir()
            refused("already exists", output=existing)
            refused("into the source", output=source / "noisy")
            refused("into the source", output=source)
            refused("sigma_shift must be", sigma_shift=-1.0)
            refused("sigma_disparity must be", sigma_disparity=float("nan"))
            refused("disparity constant", constant=0.0)

            listing = (source / "depth.txt").read_bytes()
            (source / "depth.txt").write_bytes(b"0 ../depth/0.png\n")
            refused("must be relative and stay inside")
            (source / "depth.txt").write_bytes(b"0 depth/9.png\n")
            refused("no such file")
            (source / "depth.txt").write_bytes(b"0 depth/0.png extra\n")
            refused("expected 'timestamp relative/path'")
            (source / "depth.txt").write_bytes(
                b"0 depth/0.png\n1 depth/0.png\n"
            )
            refused("listed more than once")
            (source / "depth.txt").write_bytes(b"0 rgb/0.png\n")
            refused("both colour and depth")
            (source / "depth.txt").write_bytes(listing)

            Image.new("L", (16, 12), 9).save(source / "depth" / "0.png")
            refused("must be 16-bit")
            (source / "depth" / "0.png").unlink()
            Image.new("L", (16, 12), 9).save(source / "depth" / "0.pgm")
            (source / "depth.txt").write_bytes(b"0 depth/0.pgm\n")
            refused("must be 16-bit PNG")
            (source / "depth.txt").unlink()
            refused("missing")
            with self.assertRaisesRegex(SystemExit, "must be a directory"):
                simulate_kinect_noise(root / "absent", root / "noisy")


if __name__ == "__main__":
    unittest.main()
