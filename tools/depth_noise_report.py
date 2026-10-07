"""Measure how a noisy depth sequence differs from the clean one it came from.

A reconstruction from noisy depth can only be read if the noise is known,
and a dataset's description of its noise is not the same thing as the noise
in its files. This compares two sequences of the same frames, one exact and
one degraded, pixel by pixel, and reports what the degradation actually is.

The comparison is made in disparity, because that is where a depth camera's
noise lives: ``35130 / depth`` with depth in centimetres, the unit in which
a Kinect measures in whole numbers. Three questions are answered.

Is the noisy depth quantised? A quantised sensor's disparities are whole
numbers. The median distance from each noisy disparity to the nearest whole
number is 0.25 for depth that is not quantised and close to zero for depth
that is.

Is it biased? In smooth regions, where a pixel's neighbours all have the
same depth to within a few millimetres and reading a neighbour by mistake
changes nothing, the difference between noisy and exact disparity is the
sensor's own error. Rounding to the nearest level leaves a median of zero.
Rounding up leaves half a level, which is a few millimetres near the camera
and two centimetres across a room, always towards the camera.

How much of it is gross? The share of pixels more than three levels out,
and the share with no depth at all.

Usage:

    python tools/depth_noise_report.py CLEAN NOISY [--frame-stride N]
        [--smooth-mm MM] [--disparity-constant K]

Both are folders in the TUM layout with a ``depth.txt`` and 16-bit PNG
depth at 5000 units per metre. Frames are paired by timestamp.
"""

from __future__ import annotations

import argparse
import math
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

if __package__ in (None, ""):  # run as a script rather than imported
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools._output import positive_int  # noqa: E402

KINECT_DISPARITY_CONSTANT = 35130.0
DEPTH_UNITS_PER_METRE = 5000.0
_CENTIMETRES_PER_METRE = 100.0
# A pixel is in the "core" of the noise if it is within this many
# disparity levels of the truth; beyond it something else has happened.
CORE_LEVELS = 3.0
_DEPTH_BANDS_M = ((0.0, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 4.0), (4.0, 8.0))


def disparity(depth_m: np.ndarray, constant: float) -> np.ndarray:
    return constant / (depth_m * _CENTIMETRES_PER_METRE)


def smooth_mask(depth_m: np.ndarray, tolerance_m: float) -> np.ndarray:
    """Pixels whose 3x3 neighbourhood is all present and nearly one depth.

    There a reading taken from a neighbouring pixel is the same reading, so
    what differs between the two sequences is not a lateral shift.
    """

    padded = np.pad(depth_m, 1, mode="edge")
    height, width = depth_m.shape
    low = np.full(depth_m.shape, np.inf)
    high = np.full(depth_m.shape, -np.inf)
    for offset_y in (0, 1, 2):
        for offset_x in (0, 1, 2):
            window = padded[
                offset_y:offset_y + height, offset_x:offset_x + width
            ]
            low = np.minimum(low, window)
            high = np.maximum(high, window)
    return (low > 0.0) & (high - low < tolerance_m)


def _listed_depth(folder: Path) -> dict[Decimal, Path]:
    listing = folder / "depth.txt"
    if not listing.is_file():
        raise SystemExit(f"missing {listing}")
    frames: dict[Decimal, Path] = {}
    for number, line in enumerate(
        listing.read_text(encoding="utf-8").splitlines(), start=1
    ):
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        if len(fields) != 2:
            raise SystemExit(
                f"{listing}:{number}: expected 'timestamp relative/path'"
            )
        try:
            stamp = Decimal(fields[0])
        except InvalidOperation:
            raise SystemExit(
                f"{listing}:{number}: timestamp is not a number"
            ) from None
        relative = PurePosixPath(fields[1])
        if relative.is_absolute() or ".." in relative.parts:
            raise SystemExit(
                f"{listing}:{number}: path must stay inside the sequence"
            )
        if stamp in frames:
            raise SystemExit(f"{listing}:{number}: repeated timestamp")
        frames[stamp] = folder / relative
    return frames


def _read_depth_m(path: Path) -> np.ndarray:
    try:
        with Image.open(path) as image:
            if image.mode not in ("I;16", "I;16B", "I;16L", "I"):
                raise SystemExit(
                    f"{path}: depth images must be 16-bit, single channel"
                )
            raw = np.asarray(image)
    except OSError as error:
        raise SystemExit(f"{path}: cannot read depth image: {error}") from None
    return raw.astype(np.float64) / DEPTH_UNITS_PER_METRE


def compare_sequences(
    clean: Path,
    noisy: Path,
    *,
    frame_stride: int = 20,
    smooth_m: float = 0.005,
    constant: float = KINECT_DISPARITY_CONSTANT,
) -> dict[str, object]:
    """Statistics of ``noisy`` against ``clean`` over their shared frames."""

    if not (math.isfinite(smooth_m) and smooth_m > 0.0):
        raise SystemExit("the smoothness tolerance must be finite and positive")
    if not (math.isfinite(constant) and constant > 0.0):
        raise SystemExit("the disparity constant must be finite and positive")
    clean_frames = _listed_depth(Path(clean))
    noisy_frames = _listed_depth(Path(noisy))
    shared = sorted(set(clean_frames) & set(noisy_frames))
    if not shared:
        raise SystemExit("the two sequences have no timestamp in common")
    chosen = shared[::frame_stride]

    pixels = both = lost = far = 0
    absolute_mm: list[np.ndarray] = []
    off_level: list[np.ndarray] = []
    smooth_difference: list[np.ndarray] = []
    smooth_depth: list[np.ndarray] = []
    smooth_depth_difference: list[np.ndarray] = []
    for stamp in chosen:
        exact = _read_depth_m(clean_frames[stamp])
        degraded = _read_depth_m(noisy_frames[stamp])
        if exact.shape != degraded.shape:
            raise SystemExit(
                f"frame {stamp}: the two depth images differ in size"
            )
        present = exact > 0.0
        valid = present & (degraded > 0.0)
        pixels += int(np.count_nonzero(present))
        both += int(np.count_nonzero(valid))
        lost += int(np.count_nonzero(present & (degraded <= 0.0)))
        difference_mm = 1000.0 * (degraded[valid] - exact[valid])
        far += int(np.count_nonzero(np.abs(difference_mm) > 100.0))
        absolute_mm.append(np.abs(difference_mm))
        levels = disparity(degraded[valid], constant)
        off_level.append(np.abs(levels - np.round(levels)))
        calm = valid & smooth_mask(exact, smooth_m)
        smooth_difference.append(
            disparity(degraded[calm], constant)
            - disparity(exact[calm], constant)
        )
        smooth_depth.append(exact[calm])
        smooth_depth_difference.append(
            1000.0 * (degraded[calm] - exact[calm])
        )
    if both == 0:
        raise SystemExit("no pixel has depth in both sequences")

    absolute = np.concatenate(absolute_mm)
    difference = np.concatenate(smooth_difference)
    depth = np.concatenate(smooth_depth)
    depth_difference = np.concatenate(smooth_depth_difference)
    if len(difference) == 0:
        raise SystemExit(
            "no smooth region is shared by the two sequences; raise "
            "--smooth-mm"
        )
    core = np.abs(difference) < CORE_LEVELS

    bands = []
    for low, high in _DEPTH_BANDS_M:
        chosen_band = core & (depth >= low) & (depth < high)
        count = int(np.count_nonzero(chosen_band))
        if count < 1_000:
            continue
        centre_cm = float(np.median(depth[chosen_band])) * 100.0
        bands.append(
            {
                "from_m": low,
                "to_m": high,
                "pixels": count,
                "median_disparity_difference": float(
                    np.median(difference[chosen_band])
                ),
                "median_depth_difference_mm": float(
                    np.median(depth_difference[chosen_band])
                ),
                "one_level_mm": 10.0 * centre_cm * centre_cm / constant,
            }
        )

    return {
        "frames": len(chosen),
        "frames_shared": len(shared),
        "pixels_with_exact_depth": pixels,
        "lost_fraction": lost / pixels,
        "median_absolute_difference_mm": float(np.median(absolute)),
        "p95_absolute_difference_mm": float(np.percentile(absolute, 95)),
        "beyond_100mm_fraction": far / both,
        "median_distance_to_whole_disparity": float(
            np.median(np.concatenate(off_level))
        ),
        "smooth_pixels": int(len(difference)),
        "core_fraction": float(np.count_nonzero(core) / len(difference)),
        "core_disparity_difference": {
            "mean": float(difference[core].mean()),
            "median": float(np.median(difference[core])),
            "std": float(difference[core].std()),
            "p25": float(np.percentile(difference[core], 25)),
            "p75": float(np.percentile(difference[core], 75)),
        },
        "bands": bands,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("clean", type=Path)
    parser.add_argument("noisy", type=Path)
    parser.add_argument("--frame-stride", type=positive_int, default=20)
    parser.add_argument("--smooth-mm", type=float, default=5.0)
    parser.add_argument(
        "--disparity-constant", type=float, default=KINECT_DISPARITY_CONSTANT
    )
    arguments = parser.parse_args(argv)

    result = compare_sequences(
        arguments.clean,
        arguments.noisy,
        frame_stride=arguments.frame_stride,
        smooth_m=arguments.smooth_mm / 1000.0,
        constant=arguments.disparity_constant,
    )
    core = result["core_disparity_difference"]
    print(
        f"frames compared: {result['frames']} of {result['frames_shared']} "
        "shared"
    )
    print(
        "all pixels: median |depth difference| "
        f"{result['median_absolute_difference_mm']:.1f} mm, p95 "
        f"{result['p95_absolute_difference_mm']:.1f} mm; more than 100 mm "
        f"out: {100 * result['beyond_100mm_fraction']:.2f}%; depth lost: "
        f"{100 * result['lost_fraction']:.2f}%"
    )
    print(
        "quantisation: median distance from a noisy disparity to a whole "
        f"number is {result['median_distance_to_whole_disparity']:.3f} "
        "(0.25 if not quantised)"
    )
    print(
        f"smooth regions: {result['smooth_pixels']} pixels, "
        f"{100 * result['core_fraction']:.1f}% within "
        f"{CORE_LEVELS:g} levels of the exact disparity"
    )
    print(
        "  disparity, noisy minus exact, in levels: median "
        f"{core['median']:+.3f}, mean {core['mean']:+.3f}, std "
        f"{core['std']:.3f}, quartiles {core['p25']:+.3f} to "
        f"{core['p75']:+.3f}"
    )
    print("  by depth, median over the core:")
    for band in result["bands"]:
        print(
            f"    {band['from_m']:.1f}-{band['to_m']:.1f} m: "
            f"{band['median_disparity_difference']:+.3f} levels, "
            f"{band['median_depth_difference_mm']:+.1f} mm "
            f"(one level here is {band['one_level_mm']:.1f} mm)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
