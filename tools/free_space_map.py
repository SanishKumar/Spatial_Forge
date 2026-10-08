"""Draw where a fused volume says there is room to stand.

A volume fused from a surface plan knows where surfaces are and nothing
else. One fused from a plan expanded into observed free space also knows
where the scan looked and found nothing, which is the question a map for
moving through a place has to answer, and it can tell that apart from where
the scan never looked.

This reads a ``.sftvol`` and rules on every vertical column of voxels
between two heights:

    occupied   some voxel in the band is at or behind a surface
    free       every voxel in the band was observed, and all of them lie
               in front of every surface that was seen
    unknown    neither: no surface, but part of the band was never observed

"Observed" asks for at least ``--min-weight`` observations, the number a
mesh asks of a voxel. "In front of" is a mean truncated distance above
zero. A column is free only if the whole band is known to be, so the map
errs toward unknown: one unobserved voxel is enough.

The camera path can be drawn over it. A camera can be in an occupied column
without being inside anything: it may have passed above a table. What a
camera cannot have been is behind a surface, so the tool looks up the voxel
each camera was in and says how many of those are free, unseen, or, which
would be a contradiction, occupied.

Usage:

    python tools/free_space_map.py VOLUME.sftvol OUTPUT.png
        --from-m LOW --to-m HIGH [--up-axis z] [--min-weight N]
        [--session SESSION] [--pixels-per-voxel N]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

if __package__ in (None, ""):  # run as a script rather than imported
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spatialforge import (  # noqa: E402
    load_scan_session,
    load_tsdf_block_volume,
    replay_session,
)
from spatialforge.errors import TsdfError  # noqa: E402
from tools._output import positive_int, publishing, reserve_output  # noqa: E402

UNKNOWN, FREE, OCCUPIED = 0, 1, 2
_COLOURS = {
    UNKNOWN: (150, 152, 156),
    FREE: (246, 246, 242),
    OCCUPIED: (28, 38, 64),
}
_PATH_COLOUR = (226, 106, 24)
_INK = (20, 20, 20)
_PAPER = (255, 255, 255)
# A slab of the volume is laid out densely to be read column by column.
_MAX_SLAB_VOXELS = 200_000_000
_AXES = {"x": 0, "y": 1, "z": 2}


def classify_columns(
    values: np.ndarray,
    weights: np.ndarray,
    *,
    min_weight: int,
) -> np.ndarray:
    """Rule on each column of a slab whose last axis is height.

    ``values`` is the mean truncated distance where a voxel was observed
    and anything at all where it was not; ``weights`` says which is which.
    """

    known = weights >= min_weight
    occupied = (known & (values <= 0.0)).any(axis=-1)
    free = known.all(axis=-1) & ~occupied
    result = np.full(values.shape[:-1], UNKNOWN, dtype=np.uint8)
    result[free] = FREE
    result[occupied] = OCCUPIED
    return result


def band_slab(volume, up_axis: int, low_m: float, high_m: float):
    """The volume's voxels between two heights, as dense arrays.

    Returns values and weights of shape ``(a, b, height)``, where ``a`` and
    ``b`` run along the two other axes in x, y, z order, and the global
    voxel index of the slab's first voxel on each of the three axes.
    """

    voxel = volume.voxel_size_m
    # Voxel g covers [g, g + 1) * voxel; keep those whose centre is in band.
    first = math.ceil(low_m / voxel - 0.5)
    last = math.floor(high_m / voxel - 0.5)
    if last < first:
        raise SystemExit(
            "no voxel centre lies between those heights; the band is "
            f"narrower than a {1000 * voxel:g} mm voxel"
        )
    blocks = np.asarray(volume.block_indices, dtype=np.int64)
    across = [axis for axis in range(3) if axis != up_axis]
    low = blocks.min(axis=0) * 8
    high = blocks.max(axis=0) * 8 + 7
    origin = [int(low[across[0]]), int(low[across[1]]), first]
    shape = (
        int(high[across[0]] - low[across[0]] + 1),
        int(high[across[1]] - low[across[1]] + 1),
        last - first + 1,
    )
    if shape[0] * shape[1] * shape[2] > _MAX_SLAB_VOXELS:
        raise SystemExit(
            f"the band holds {shape[0]} x {shape[1]} x {shape[2]} voxels; "
            f"the most this tool lays out is {_MAX_SLAB_VOXELS}"
        )
    values = np.zeros(shape, dtype=np.float64)
    weights = np.zeros(shape, dtype=np.uint32)
    for row, block in enumerate(blocks):
        base = block * 8
        start = max(first, int(base[up_axis]))
        stop = min(last, int(base[up_axis]) + 7)
        if stop < start:
            continue
        # Storage is [z, y, x]; bring it to [x, y, z], then the band's axes.
        sums = np.transpose(volume.tsdf_sums[row], (2, 1, 0))
        counts = np.transpose(volume.weights[row], (2, 1, 0))
        order = (across[0], across[1], up_axis)
        sums = np.transpose(sums, order)
        counts = np.transpose(counts, order)
        inside = slice(start - int(base[up_axis]), stop - int(base[up_axis]) + 1)
        a = int(base[across[0]]) - origin[0]
        b = int(base[across[1]]) - origin[1]
        target = (
            slice(a, a + 8),
            slice(b, b + 8),
            slice(start - first, stop - first + 1),
        )
        seen = counts[:, :, inside]
        weights[target] = seen
        values[target] = np.divide(
            sums[:, :, inside],
            seen,
            out=np.zeros((8, 8, stop - start + 1)),
            where=seen > 0,
        )
    return values, weights, origin, across


def camera_centres_m(session_path: Path) -> np.ndarray:
    """Where each posed camera was, in world metres."""

    replay = replay_session(load_scan_session(session_path))
    centres = np.array(
        [
            [
                float(observation.pose.data["T_world_camera"][3 + 4 * axis])
                for axis in range(3)
            ]
            for observation in replay.observations
            if observation.pose is not None
        ]
    )
    if not len(centres):
        raise SystemExit("the session has no posed frame to draw")
    return centres


def camera_columns(centres_m: np.ndarray, voxel_size_m: float, origin, across):
    """The column each camera is in, as slab indices."""

    columns = np.floor(centres_m[:, across] / voxel_size_m).astype(np.int64)
    return columns - np.array(origin[:2])


def camera_voxel_verdicts(
    volume,
    centres_m: np.ndarray,
    *,
    min_weight: int,
) -> np.ndarray:
    """The volume's verdict on the one voxel each camera was in."""

    rows = {block: row for row, block in enumerate(volume.block_indices)}
    voxels = np.floor(centres_m / volume.voxel_size_m).astype(np.int64)
    verdicts = np.full(len(voxels), UNKNOWN, dtype=np.uint8)
    for index, voxel in enumerate(voxels):
        row = rows.get(tuple(int(value) for value in voxel // 8))
        if row is None:
            continue
        x, y, z = (int(value) for value in voxel % 8)
        weight = int(volume.weights[row, z, y, x])
        if weight >= min_weight:
            behind = volume.tsdf_sums[row, z, y, x] <= 0.0
            verdicts[index] = OCCUPIED if behind else FREE
    return verdicts


def draw(
    columns: np.ndarray,
    *,
    voxel_size_m: float,
    pixels_per_voxel: int,
    path: np.ndarray | None,
) -> Image.Image:
    """The map, first axis to the right and second axis up, with a key."""

    colours = np.empty(columns.shape + (3,), dtype=np.uint8)
    for kind, colour in _COLOURS.items():
        colours[columns == kind] = colour
    # Rows of an image run downward; the second axis should run up.
    picture = np.transpose(colours, (1, 0, 2))[::-1]
    picture = np.repeat(
        np.repeat(picture, pixels_per_voxel, axis=0), pixels_per_voxel, axis=1
    )
    height, width = picture.shape[:2]
    text = max(11, round(min(width, height) * 0.032))
    margin = round(text * 0.8)
    key = round(text * 2.4)
    image = Image.new("RGB", (width, height + key), _PAPER)
    image.paste(Image.fromarray(picture, mode="RGB"), (0, 0))
    canvas = ImageDraw.Draw(image)
    if path is not None and len(path):
        points = [
            (
                (float(a) + 0.5) * pixels_per_voxel,
                height - (float(b) + 0.5) * pixels_per_voxel,
            )
            for a, b in path
        ]
        canvas.line(
            points, fill=_PATH_COLOUR, width=max(1, pixels_per_voxel // 2)
        )
    try:
        font = ImageFont.load_default(size=text)
    except (TypeError, OSError):  # a Pillow without scalable default
        font = ImageFont.load_default()
    cursor = margin
    baseline = height + (key - text) // 2
    entries = [("free", _COLOURS[FREE]), ("occupied", _COLOURS[OCCUPIED])]
    entries.append(("unknown", _COLOURS[UNKNOWN]))
    if path is not None:
        entries.append(("camera path", _PATH_COLOUR))
    for label, colour in entries:
        canvas.rectangle(
            (cursor, baseline, cursor + text, baseline + text),
            fill=colour,
            outline=_INK,
        )
        cursor += round(text * 1.4)
        canvas.text((cursor, baseline - 1), label, fill=_INK, font=font)
        cursor += round(canvas.textlength(label, font=font) + text * 1.2)
    # One metre, drawn to the scale of the map.
    metre = round(pixels_per_voxel / voxel_size_m)
    if metre <= width - cursor - 2 * margin:
        right = width - margin
        middle = baseline + text // 2
        canvas.line((right - metre, middle, right, middle), fill=_INK, width=2)
        for end in (right - metre, right):
            canvas.line(
                (end, middle - text // 3, end, middle + text // 3),
                fill=_INK,
                width=2,
            )
        label = "1 m"
        canvas.text(
            (right - metre - canvas.textlength(label, font=font) - text // 2,
             baseline - 1),
            label,
            fill=_INK,
            font=font,
        )
    return image


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("volume", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--from-m", type=float, required=True)
    parser.add_argument("--to-m", type=float, required=True)
    parser.add_argument("--up-axis", choices=sorted(_AXES), default="z")
    parser.add_argument("--min-weight", type=positive_int, default=3)
    parser.add_argument("--session", type=Path, default=None)
    parser.add_argument("--pixels-per-voxel", type=positive_int, default=3)
    arguments = parser.parse_args(argv)
    if not (
        math.isfinite(arguments.from_m)
        and math.isfinite(arguments.to_m)
        and arguments.from_m < arguments.to_m
    ):
        parser.error("--from-m must be finite and below --to-m")

    output = reserve_output(arguments.output, ".png")
    try:
        volume = load_tsdf_block_volume(arguments.volume)
    except TsdfError as error:
        raise SystemExit(str(error)) from error
    up_axis = _AXES[arguments.up_axis]
    values, weights, origin, across = band_slab(
        volume, up_axis, arguments.from_m, arguments.to_m
    )
    columns = classify_columns(
        values, weights, min_weight=arguments.min_weight
    )

    voxel = volume.voxel_size_m
    column_area = voxel * voxel
    names = "xyz"
    print(
        f"volume: {volume.block_count} blocks at {1000 * voxel:g} mm, "
        f"{volume.observed_voxel_count} observed voxels"
    )
    print(
        f"band: {values.shape[2]} voxels along {names[up_axis]}, centres "
        f"from {(origin[2] + 0.5) * voxel:.3f} m to "
        f"{(origin[2] + values.shape[2] - 0.5) * voxel:.3f} m"
    )
    print(
        f"map: {columns.shape[0]} x {columns.shape[1]} columns along "
        f"{names[across[0]]} and {names[across[1]]}, "
        f"{columns.size * column_area:.2f} m2"
    )
    for label, kind in (
        ("free", FREE),
        ("occupied", OCCUPIED),
        ("unknown", UNKNOWN),
    ):
        count = int(np.count_nonzero(columns == kind))
        print(
            f"{label}: {count} columns, {count * column_area:.2f} m2 "
            f"({100 * count / columns.size:.1f}%)"
        )

    path = None
    if arguments.session is not None:
        centres = camera_centres_m(arguments.session)
        path = camera_columns(centres, voxel, origin, across)
        own = camera_voxel_verdicts(
            volume, centres, min_weight=arguments.min_weight
        )
        print(
            f"cameras: {len(centres)} posed; the voxel each was in is free "
            f"for {int(np.count_nonzero(own == FREE))}, unseen for "
            f"{int(np.count_nonzero(own == UNKNOWN))} and behind a surface "
            f"for {int(np.count_nonzero(own == OCCUPIED))}"
        )

    image = draw(
        columns,
        voxel_size_m=voxel,
        pixels_per_voxel=arguments.pixels_per_voxel,
        path=path,
    )
    with publishing(output, ".png") as temporary:
        image.save(temporary, format="PNG")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
