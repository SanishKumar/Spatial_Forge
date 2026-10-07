"""Give a clean depth sequence the noise of a Kinect, reproducibly.

A reconstruction measured on exact depth says what the engine adds to
perfect input. It does not say what a sensor costs. This takes a sequence
in the TUM layout and writes a copy whose depth images carry simulated
sensor noise, so that the same scene, the same poses and the same ground
truth can be measured again with only the depth changed.

The model is the one the ICL-NUIM paper uses for its own noisy sequences
(Handa et al., 2014, equation 3, after Barron and Malik, 2013):

    Z'(x, y) = 35130 / floor(35130 / Z(x + nx, y + ny) + N(0, sd^2) + 0.5)

Each pixel reads the true depth a fraction of a pixel away from where it
should, with ``(nx, ny)`` normal with standard deviation ``ss = 1/2`` and
the depth interpolated bilinearly. That depth is turned into a disparity,
normal noise with ``sd = 1/6`` is added, and the disparity is rounded to a
whole number before being turned back into depth.

35130 is the Kinect's baseline times its focal length in eighths of a
pixel, with depth in centimetres. The paper does not state the unit; it is
fixed by what the model then predicts. In centimetres, neighbouring
disparity levels are 11 mm apart at 2 m, and a Kinect is reported to resolve
about 1 cm there (Khoshelham and Elberink, 2012). In metres they would be a
tenth of a millimetre apart and the rounding would do nothing. The
dataset's own noisy files settle it: their disparities, computed this way,
are whole numbers.

Two things this is not. It is not ICL-NUIM's published noisy sequence: the
paper also displaces points along their normals by an amount it gives no
parameters for, and that step is not applied here. Nor does it reproduce
the half-level offset towards the camera that those files carry and the
printed equation does not; ``depth_noise_report.py`` measures both. And it
is not a sensor: no missing returns, no edge fringing, no rolling shutter.

Noise is drawn from a generator seeded with the given seed and the frame's
index, so a frame's noise does not depend on which other frames exist, and
the decoded depth is the same on every machine for a given NumPy.

Usage:

    python tools/simulate_kinect_noise.py SOURCE OUTPUT [--seed N]
        [--sigma-shift S] [--sigma-disparity S] [--disparity-constant K]

``SOURCE`` holds ``rgb.txt``, ``depth.txt`` and 16-bit PNG depth at 5000
units per metre. ``OUTPUT`` must not exist. Colour images, timestamps and
poses are copied unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

if __package__ in (None, ""):  # run as a script rather than imported
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools._output import non_negative_int  # noqa: E402

KINECT_DISPARITY_CONSTANT = 35130.0
SIGMA_SHIFT_PIXELS = 0.5
SIGMA_DISPARITY = 1.0 / 6.0
DEPTH_UNITS_PER_METRE = 5000.0
_CENTIMETRES_PER_METRE = 100.0
_MAX_RAW_DEPTH = 65535


def apply_depth_noise(
    depth_m: np.ndarray,
    shift_x: np.ndarray,
    shift_y: np.ndarray,
    disparity_noise: np.ndarray,
    *,
    constant: float = KINECT_DISPARITY_CONSTANT,
) -> np.ndarray:
    """Equation 3 for one image and one draw of its three noise fields.

    Depth is in metres, zero where there is none. A pixel whose shifted
    reading would interpolate across a missing depth, or whose disparity
    rounds to less than one, has no depth in the result.
    """

    depth_m = np.asarray(depth_m, dtype=np.float64)
    if depth_m.ndim != 2 or min(depth_m.shape) < 2:
        raise ValueError("depth must be an image of at least 2 by 2 pixels")
    for name, field in (
        ("shift_x", shift_x),
        ("shift_y", shift_y),
        ("disparity_noise", disparity_noise),
    ):
        if np.shape(field) != depth_m.shape:
            raise ValueError(f"{name} must have the shape of the depth image")
    if not (math.isfinite(constant) and constant > 0.0):
        raise ValueError("the disparity constant must be finite and positive")

    height, width = depth_m.shape
    rows, columns = np.mgrid[0:height, 0:width]
    # A reading that would fall off the image is taken at its edge.
    x = np.clip(columns + shift_x, 0.0, width - 1.0)
    y = np.clip(rows + shift_y, 0.0, height - 1.0)
    x0 = np.minimum(np.floor(x).astype(np.int64), width - 2)
    y0 = np.minimum(np.floor(y).astype(np.int64), height - 2)
    along_x = x - x0
    along_y = y - y0
    corners = (
        (depth_m[y0, x0], (1.0 - along_y) * (1.0 - along_x)),
        (depth_m[y0, x0 + 1], (1.0 - along_y) * along_x),
        (depth_m[y0 + 1, x0], along_y * (1.0 - along_x)),
        (depth_m[y0 + 1, x0 + 1], along_y * along_x),
    )
    valid = np.ones(depth_m.shape, dtype=bool)
    shifted = np.zeros(depth_m.shape)
    for corner, weight in corners:
        present = np.isfinite(corner) & (corner > 0.0)
        # A missing neighbour only matters to a reading that
        # actually draws on it: an unshifted pixel beside a hole
        # still has its own depth.
        valid &= present | (weight == 0.0)
        shifted += weight * np.where(present, corner, 0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        disparity = constant / (shifted * _CENTIMETRES_PER_METRE)
        rounded = np.floor(disparity + disparity_noise + 0.5)
        valid &= np.isfinite(rounded) & (rounded >= 1.0)
        noisy = constant / rounded / _CENTIMETRES_PER_METRE
    return np.where(valid, noisy, 0.0)


def draw_noise(
    seed: int,
    frame_index: int,
    shape: tuple[int, int],
    *,
    sigma_shift: float = SIGMA_SHIFT_PIXELS,
    sigma_disparity: float = SIGMA_DISPARITY,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The three noise fields of one frame: x shift, y shift, disparity."""

    generator = np.random.default_rng([seed, frame_index])
    return (
        generator.normal(0.0, sigma_shift, size=shape),
        generator.normal(0.0, sigma_shift, size=shape),
        generator.normal(0.0, sigma_disparity, size=shape),
    )


def _relative_file(source: Path, reference: str, where: str) -> str:
    relative = PurePosixPath(reference)
    if relative.is_absolute() or ".." in relative.parts or "\\" in reference:
        raise SystemExit(
            f"{where}: path must be relative and stay inside the "
            f"sequence: {reference!r}"
        )
    if not (source / relative).is_file():
        raise SystemExit(f"{where}: no such file: {reference!r}")
    return relative.as_posix()


def _listed_files(source: Path, name: str) -> list[str]:
    path = source / name
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    files = []
    for number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        if len(fields) != 2:
            raise SystemExit(
                f"{name}:{number}: expected 'timestamp relative/path'"
            )
        files.append(_relative_file(source, fields[1], f"{name}:{number}"))
    if len(set(files)) != len(files):
        raise SystemExit(f"{name}: a file is listed more than once")
    return files


def _read_raw_depth(path: Path) -> np.ndarray:
    if path.suffix.lower() != ".png":
        raise SystemExit(f"{path}: depth images must be 16-bit PNG files")
    with Image.open(path) as image:
        if image.mode not in ("I;16", "I;16B", "I;16L", "I"):
            raise SystemExit(
                f"{path}: depth images must be 16-bit, single channel"
            )
        raw = np.asarray(image)
    if raw.ndim != 2 or raw.min() < 0 or raw.max() > _MAX_RAW_DEPTH:
        raise SystemExit(f"{path}: not a 16-bit depth image")
    return raw.astype(np.float64)


def simulate_kinect_noise(
    source: Path,
    output: Path,
    *,
    seed: int = 0,
    sigma_shift: float = SIGMA_SHIFT_PIXELS,
    sigma_disparity: float = SIGMA_DISPARITY,
    constant: float = KINECT_DISPARITY_CONSTANT,
) -> dict[str, int]:
    """Write the noisy copy; returns counts of what was written."""

    source = Path(source).resolve()
    output = Path(output).resolve()
    for name, value in (
        ("sigma_shift", sigma_shift),
        ("sigma_disparity", sigma_disparity),
    ):
        if not (math.isfinite(value) and value >= 0.0):
            raise SystemExit(f"{name} must be finite and not negative")
    if not (math.isfinite(constant) and constant > 0.0):
        raise SystemExit("the disparity constant must be finite and positive")
    if not source.is_dir():
        raise SystemExit(f"source must be a directory: {source}")
    if output == source or output.is_relative_to(source):
        raise SystemExit(
            f"output would write into the source sequence: {output}"
        )
    if output.exists():
        raise SystemExit(
            f"output already exists, refusing to overwrite: {output}"
        )

    colour_files = _listed_files(source, "rgb.txt")
    depth_files = _listed_files(source, "depth.txt")
    if set(colour_files) & set(depth_files):
        raise SystemExit("a file is listed as both colour and depth")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent)
    )
    valid_before = valid_after = 0
    try:
        for name in ("rgb.txt", "depth.txt", "groundtruth.txt"):
            if (source / name).is_file():
                shutil.copyfile(source / name, staging / name)
        for relative in colour_files:
            (staging / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / relative, staging / relative)
        for index, relative in enumerate(depth_files):
            raw = _read_raw_depth(source / relative)
            noisy_m = apply_depth_noise(
                raw / DEPTH_UNITS_PER_METRE,
                *draw_noise(
                    seed,
                    index,
                    raw.shape,
                    sigma_shift=sigma_shift,
                    sigma_disparity=sigma_disparity,
                ),
                constant=constant,
            )
            encoded = np.floor(noisy_m * DEPTH_UNITS_PER_METRE + 0.5)
            encoded[encoded > _MAX_RAW_DEPTH] = 0.0
            valid_before += int(np.count_nonzero(raw > 0))
            valid_after += int(np.count_nonzero(encoded > 0))
            (staging / relative).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(encoded.astype(np.uint16)).save(
                staging / relative, format="PNG"
            )
        description = {
            "model": (
                "ICL-NUIM depth noise, equation 3: sub-pixel shift, "
                "disparity noise, disparity quantisation"
            ),
            "seed": seed,
            "sigma_shift_pixels": sigma_shift,
            "sigma_disparity": sigma_disparity,
            "disparity_constant": constant,
            "depth_units_per_metre": DEPTH_UNITS_PER_METRE,
            "frames": len(depth_files),
            "numpy": np.__version__,
        }
        (staging / "noise.json").write_bytes(
            (json.dumps(description, indent=2, sort_keys=True) + "\n").encode(
                "utf-8"
            )
        )
        if output.exists():
            raise SystemExit(
                f"output appeared while running; refusing to overwrite: "
                f"{output}"
            )
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)

    return {
        "frames": len(depth_files),
        "valid_depth_before": valid_before,
        "valid_depth_after": valid_after,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=non_negative_int, default=0)
    parser.add_argument(
        "--sigma-shift", type=float, default=SIGMA_SHIFT_PIXELS
    )
    parser.add_argument(
        "--sigma-disparity", type=float, default=SIGMA_DISPARITY
    )
    parser.add_argument(
        "--disparity-constant", type=float, default=KINECT_DISPARITY_CONSTANT
    )
    arguments = parser.parse_args(argv)

    counts = simulate_kinect_noise(
        arguments.source,
        arguments.output,
        seed=arguments.seed,
        sigma_shift=arguments.sigma_shift,
        sigma_disparity=arguments.sigma_disparity,
        constant=arguments.disparity_constant,
    )
    print(f"wrote {Path(arguments.output).resolve()}")
    print(
        f"frames: {counts['frames']}; depth pixels with a value: "
        f"{counts['valid_depth_before']} before, "
        f"{counts['valid_depth_after']} after"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
