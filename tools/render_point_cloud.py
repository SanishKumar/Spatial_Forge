"""Render a SpatialForge XYZ point-cloud PLY to a still and an orbit GIF.

The reconstruction has been a table of numbers for its whole life, because the
triangle mesher refuses volumes with non-manifold vertices and real data
produces them. Surface-point extraction succeeds on the same volumes, so this
renders those points directly: a z-buffered splat with an orbiting camera, in
numpy and PIL, with no plotting dependency.

Usage:

    python tools/render_point_cloud.py INPUT.ply OUTPUT_PREFIX
        [--size N] [--frames N] [--elevation DEG] [--point-size N]

Writes OUTPUT_PREFIX.png (one still) and OUTPUT_PREFIX.gif (an orbit).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Perceptually ordered stops, dark blue through green to yellow. Sampled so a
# height ramp stays readable when the GIF is quantised to 255 colours.
_COLOUR_STOPS = np.array(
    [
        (68, 1, 84),
        (72, 40, 120),
        (62, 74, 137),
        (49, 104, 142),
        (38, 130, 142),
        (31, 158, 137),
        (53, 183, 121),
        (109, 205, 89),
        (180, 222, 44),
        (253, 231, 37),
    ],
    dtype=np.float64,
)
_BACKGROUND = np.array((14, 16, 22), dtype=np.float64)
_SUPERSAMPLE = 2


def read_xyz_ply(path: Path) -> np.ndarray:
    """Read the ASCII XYZ PLY this project writes; return an (N, 3) array."""

    lines = path.read_text(encoding="ascii").splitlines()
    try:
        header_end = next(
            index
            for index, line in enumerate(lines)
            if line.strip() == "end_header"
        )
    except StopIteration:
        raise SystemExit(f"{path}: no end_header, not an ASCII PLY")
    rows = []
    for line in lines[header_end + 1:]:
        parts = line.split()
        if len(parts) >= 3:
            rows.append((float(parts[0]), float(parts[1]), float(parts[2])))
    if not rows:
        raise SystemExit(f"{path}: no vertices")
    return np.array(rows, dtype=np.float64)


def colour_ramp(values: np.ndarray) -> np.ndarray:
    """Map values in [0, 1] onto the stop table with linear interpolation."""

    scaled = np.clip(values, 0.0, 1.0) * (len(_COLOUR_STOPS) - 1)
    lower = np.floor(scaled).astype(np.int64)
    upper = np.minimum(lower + 1, len(_COLOUR_STOPS) - 1)
    fraction = (scaled - lower)[:, None]
    return (
        _COLOUR_STOPS[lower] * (1.0 - fraction)
        + _COLOUR_STOPS[upper] * fraction
    )


def _camera_basis(
    points: np.ndarray,
    azimuth_degrees: float,
    elevation_degrees: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    centre = (points.min(axis=0) + points.max(axis=0)) / 2.0
    radius = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0))) / 2
    distance = radius * 2.6

    azimuth = np.radians(azimuth_degrees)
    elevation = np.radians(elevation_degrees)
    eye = centre + distance * np.array(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ]
    )
    forward = centre - eye
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return eye, right, up, forward


def fit_focal(
    points: np.ndarray,
    azimuths: list[float],
    elevation_degrees: float,
    resolution: int,
    fill: float,
) -> float:
    """One focal length that frames the cloud at every orbit position.

    Fitting per frame would make the subject pulse as it rotates, so the
    tightest fit that still contains the worst-case azimuth wins.
    """

    worst = 0.0
    for azimuth in azimuths:
        eye, right, up, forward = _camera_basis(
            points,
            azimuth,
            elevation_degrees,
        )
        relative = points - eye
        depth = relative @ forward
        visible = depth > 1e-6
        if not np.any(visible):
            continue
        extent = np.maximum(
            np.abs(relative[visible] @ right) / depth[visible],
            np.abs(relative[visible] @ up) / depth[visible],
        )
        worst = max(worst, float(np.percentile(extent, 99.9)))
    if worst <= 0.0:
        raise SystemExit("cannot frame the cloud")
    return (resolution / 2.0) * fill / worst


def render(
    points: np.ndarray,
    colours: np.ndarray,
    azimuth_degrees: float,
    elevation_degrees: float,
    size: int,
    point_size: int,
    focal: float,
) -> Image.Image:
    """Z-buffered splat of the cloud from one orbit position."""

    resolution = size * _SUPERSAMPLE
    azimuth = np.radians(azimuth_degrees)
    eye, right, up, forward = _camera_basis(
        points,
        azimuth_degrees,
        elevation_degrees,
    )
    relative = points - eye
    camera_x = relative @ right
    camera_y = relative @ up
    camera_z = relative @ forward
    in_front = camera_z > 1e-6
    if not np.any(in_front):
        raise SystemExit("every point fell behind the camera")

    centre_pixel = resolution / 2.0
    u = focal * camera_x[in_front] / camera_z[in_front] + centre_pixel
    v = -focal * camera_y[in_front] / camera_z[in_front] + centre_pixel
    depth = camera_z[in_front]
    visible = colours[in_front]

    canvas = np.repeat(
        _BACKGROUND[None, None, :],
        resolution,
        axis=0,
    ).repeat(resolution, axis=1)
    z_buffer = np.full((resolution, resolution), np.inf)

    # Nearest points last so they overwrite; the explicit z test then only has
    # to break ties inside one splat footprint.
    order = np.argsort(-depth)
    u = u[order]
    v = v[order]
    depth = depth[order]
    visible = visible[order]

    half = point_size * _SUPERSAMPLE // 2
    base_x = np.round(u).astype(np.int64)
    base_y = np.round(v).astype(np.int64)
    for offset_y in range(-half, half + 1):
        for offset_x in range(-half, half + 1):
            if offset_x * offset_x + offset_y * offset_y > half * half + 1:
                continue
            x = base_x + offset_x
            y = base_y + offset_y
            inside = (
                (x >= 0) & (x < resolution) & (y >= 0) & (y < resolution)
            )
            if not np.any(inside):
                continue
            xi = x[inside]
            yi = y[inside]
            di = depth[inside]
            nearer = di < z_buffer[yi, xi]
            if not np.any(nearer):
                continue
            z_buffer[yi[nearer], xi[nearer]] = di[nearer]
            canvas[yi[nearer], xi[nearer]] = visible[inside][nearer]

    image = Image.fromarray(canvas.astype(np.uint8), mode="RGB")
    return image.resize((size, size), Image.LANCZOS)


def crop_to_content(
    images: list[Image.Image],
    margin_fraction: float = 0.04,
) -> list[Image.Image]:
    """Crop every image to one square box containing all their content.

    A per-image crop would make the subject drift and breathe as it orbits,
    so the union of every frame's occupied pixels is cropped identically.
    """

    background = _BACKGROUND.astype(np.uint8)
    left = top = 10**9
    right = bottom = -1
    for image in images:
        occupied = np.any(np.asarray(image) != background, axis=2)
        columns = np.flatnonzero(np.any(occupied, axis=0))
        rows = np.flatnonzero(np.any(occupied, axis=1))
        if not len(columns) or not len(rows):
            continue
        left = min(left, int(columns[0]))
        right = max(right, int(columns[-1]))
        top = min(top, int(rows[0]))
        bottom = max(bottom, int(rows[-1]))
    if right < 0:
        return images

    width, height = images[0].size
    centre_x = (left + right) / 2.0
    centre_y = (top + bottom) / 2.0
    half = max(right - left, bottom - top) / 2.0
    half *= 1.0 + margin_fraction * 2.0
    half = min(half, centre_x, centre_y, width - centre_x, height - centre_y)
    box = (
        int(round(centre_x - half)),
        int(round(centre_y - half)),
        int(round(centre_x + half)),
        int(round(centre_y + half)),
    )
    return [image.crop(box).resize((width, height), Image.LANCZOS)
            for image in images]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output_prefix", type=Path)
    parser.add_argument("--size", type=int, default=720)
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--elevation", type=float, default=16.0)
    parser.add_argument("--point-size", type=int, default=4)
    parser.add_argument("--still-azimuth", type=float, default=35.0)
    parser.add_argument("--fill", type=float, default=0.92)
    arguments = parser.parse_args(argv)

    points = read_xyz_ply(arguments.input)
    extent = points.max(axis=0) - points.min(axis=0)
    print(
        f"{len(points)} points, extent "
        f"{extent[0]:.2f} x {extent[1]:.2f} x {extent[2]:.2f} m"
    )

    height = points[:, 2]
    span = float(height.max() - height.min())
    normalised = (
        (height - height.min()) / span
        if span > 0
        else np.zeros(len(points))
    )
    colours = colour_ramp(normalised)

    orbit = [
        360.0 * index / arguments.frames
        for index in range(arguments.frames)
    ]
    focal = fit_focal(
        points,
        orbit + [arguments.still_azimuth],
        arguments.elevation,
        arguments.size * _SUPERSAMPLE,
        arguments.fill,
    )

    still = crop_to_content(
        [
            render(
                points,
                colours,
                arguments.still_azimuth,
                arguments.elevation,
                arguments.size,
                arguments.point_size,
                focal,
            )
        ]
    )[0]
    still_path = arguments.output_prefix.with_suffix(".png")
    still.save(still_path)
    print(f"wrote {still_path}")

    frames = [
        frame.convert("P", palette=Image.ADAPTIVE, colors=255)
        for frame in crop_to_content(
            [
                render(
                    points,
                    colours,
                    azimuth,
                    arguments.elevation,
                    arguments.size,
                    arguments.point_size,
                    focal,
                )
                for azimuth in orbit
            ]
        )
    ]
    gif_path = arguments.output_prefix.with_suffix(".gif")
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=70,
        loop=0,
        optimize=True,
    )
    print(f"wrote {gif_path} ({len(frames)} frames)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
