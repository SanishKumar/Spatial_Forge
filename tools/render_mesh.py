"""Render a SpatialForge triangle-mesh PLY to a shaded still and animation.

``render_point_cloud.py`` exists because, for a long time, the only thing a
reconstruction could be turned into was a cloud of surface points from a
coarse dense volume. Sparse volumes now persist and mesh, so this renders
the measured surface itself: a software triangle rasteriser with a z-buffer
and smooth shading, in NumPy and PIL, with no rendering dependency.

Colour is optional and comes from the scan. With ``--session`` each vertex
is projected into the scan's RGB frames and takes the colour it has in the
frames that actually see it -- in front of the camera, inside the image,
and at the depth the frame measured there. That is a visualisation of the
geometry, not part of it: the mesh file carries no colour, and nothing here
feeds back into the reconstruction.

Usage:

    python tools/render_mesh.py MESH.ply OUTPUT_PREFIX
        [--session SCAN.vgsession] [--colour-stride N]
        [--size N] [--frames N] [--sweep DEG] [--azimuth DEG]
        [--elevation DEG]

Writes OUTPUT_PREFIX.png (one still) and OUTPUT_PREFIX.gif (an animation).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

if __package__ in (None, ""):  # run as a script rather than imported
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools._output import (  # noqa: E402
    positive_int,
    publishing,
    reserve_output,
    unit_fraction,
)
from tools.render_point_cloud import (  # noqa: E402
    _BACKGROUND,
    _SUPERSAMPLE,
    _camera_basis,
    crop_to_content,
    fit_focal,
)

_CLAY = np.array((205.0, 208.0, 214.0))
_UNSEEN = np.array((120.0, 124.0, 132.0))
_BACK_TINT = np.array((0.34, 0.38, 0.50))
_AMBIENT = 0.30
# Triangles are rasterised in groups by the side of their pixel bounding
# box, so that a few large triangles do not set the cost for all the rest.
_BUCKET_SIDES = (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
_COVERAGE_TOLERANCE = 1e-9
_COLOUR_DEPTH_TOLERANCE_M = 0.03


def read_mesh_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the binary little-endian triangle PLY the block mesher writes."""

    encoded = path.read_bytes()
    marker = b"end_header\n"
    end = encoded.find(marker)
    if not encoded.startswith(b"ply\n") or end < 0:
        raise SystemExit(f"{path}: not a PLY file")
    header = encoded[:end].decode("ascii", errors="replace").splitlines()
    if "format binary_little_endian 1.0" not in header:
        raise SystemExit(
            f"{path}: expected a binary little-endian PLY; ASCII point "
            "clouds are rendered by render_point_cloud.py"
        )
    elements = [line.split() for line in header if line.startswith("element")]
    properties = [line for line in header if line.startswith("property")]
    if (
        len(elements) != 2
        or elements[0][1] != "vertex"
        or elements[1][1] != "face"
        or properties
        != [
            "property double x",
            "property double y",
            "property double z",
            "property list uchar int vertex_indices",
        ]
    ):
        raise SystemExit(
            f"{path}: expected double x, y, z vertices and a triangle list"
        )
    vertex_count = int(elements[0][2])
    face_count = int(elements[1][2])
    start = end + len(marker)
    face_start = start + vertex_count * 24
    if face_start + face_count * 13 != len(encoded):
        raise SystemExit(f"{path}: payload size does not match its header")
    if vertex_count == 0 or face_count == 0:
        raise SystemExit(f"{path}: mesh is empty")
    vertices = np.frombuffer(
        encoded,
        dtype="<f8",
        count=vertex_count * 3,
        offset=start,
    ).reshape((vertex_count, 3))
    records = np.frombuffer(
        encoded,
        dtype=[("count", "u1"), ("indices", "<i4", (3,))],
        count=face_count,
        offset=face_start,
    )
    if bool(np.any(records["count"] != 3)):
        raise SystemExit(f"{path}: every face must be a triangle")
    faces = records["indices"].astype(np.int64)
    if int(faces.min()) < 0 or int(faces.max()) >= vertex_count:
        raise SystemExit(f"{path}: face index outside the vertex list")
    if not bool(np.all(np.isfinite(vertices))):
        raise SystemExit(f"{path}: non-finite vertex coordinate")
    return vertices.astype(np.float64), faces


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals, following the mesh's own winding."""

    first = vertices[faces[:, 0]]
    face_normal = np.cross(
        vertices[faces[:, 1]] - first,
        vertices[faces[:, 2]] - first,
    )
    normals = np.zeros_like(vertices)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normal)
    length = np.linalg.norm(normals, axis=1)
    flat = length == 0.0
    normals[flat] = (0.0, 0.0, 1.0)
    length[flat] = 1.0
    return normals / length[:, None]


def scan_vertex_colours(
    session_path: Path,
    vertices: np.ndarray,
    normals: np.ndarray,
    frame_stride: int,
) -> tuple[np.ndarray, int, int]:
    """Colour each vertex from the RGB frames that genuinely observe it.

    A frame contributes to a vertex only if the vertex projects inside the
    image, faces the camera, and lies at the depth that frame measured at
    that pixel. The depth test is what stops a wall being painted with the
    colour of the desk in front of it.
    """

    from spatialforge.point_cloud import (
        _read_depth_array,
        _sample_path,
        _validate_reconstruction_contract,
    )
    from spatialforge.replay import replay_session
    from spatialforge.session_loader import load_scan_session

    session = load_scan_session(session_path)
    camera, depth_scale_m = _validate_reconstruction_contract(session)
    total = np.zeros((len(vertices), 3))
    weight = np.zeros(len(vertices))
    frames_used = 0
    for observation in replay_session(session).observations:
        if (
            observation.sequence % frame_stride
            or observation.depth is None
            or observation.pose is None
        ):
            continue
        matrix = [
            float(value) for value in observation.pose.data["T_world_camera"]
        ]
        delta = vertices - np.array([matrix[3], matrix[7], matrix[11]])
        camera_x = (
            matrix[0] * delta[:, 0]
            + matrix[4] * delta[:, 1]
            + matrix[8] * delta[:, 2]
        )
        camera_y = (
            matrix[1] * delta[:, 0]
            + matrix[5] * delta[:, 1]
            + matrix[9] * delta[:, 2]
        )
        camera_z = (
            matrix[2] * delta[:, 0]
            + matrix[6] * delta[:, 1]
            + matrix[10] * delta[:, 2]
        )
        in_front = camera_z > 1e-6
        safe_z = np.where(in_front, camera_z, 1.0)
        column = np.floor(camera.fx * camera_x / safe_z + camera.cx + 0.5)
        row = np.floor(camera.fy * camera_y / safe_z + camera.cy + 0.5)
        visible = (
            in_front
            & (column >= 0)
            & (column < camera.width)
            & (row >= 0)
            & (row < camera.height)
        )
        # Facing: the vertex normal points into free space, towards
        # whichever cameras saw this side of the surface.
        distance = np.linalg.norm(delta, axis=1)
        facing = -np.einsum("ij,ij->i", normals, delta) / np.where(
            distance > 0.0,
            distance,
            1.0,
        )
        visible &= facing > 0.15
        index = np.flatnonzero(visible)
        if index.size == 0:
            continue

        depth_m = (
            _read_depth_array(
                _sample_path(session, observation.depth.data, "depth"),
                camera.width,
                camera.height,
            )
            .astype(np.float64)
            .reshape((camera.height, camera.width))
            * depth_scale_m
        )
        pixel_row = row[index].astype(np.int64)
        pixel_column = column[index].astype(np.int64)
        measured = depth_m[pixel_row, pixel_column]
        agrees = (
            np.abs(measured - camera_z[index]) < _COLOUR_DEPTH_TOLERANCE_M
        )
        index = index[agrees]
        if index.size == 0:
            continue
        with Image.open(
            _sample_path(session, observation.rgb.data, "rgb")
        ) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float64)
        sample = rgb[pixel_row[agrees], pixel_column[agrees]]
        # Head-on, close views are the sharpest, so they count for more.
        contribution = facing[index] / (camera_z[index] ** 2)
        total[index] += sample * contribution[:, None]
        weight[index] += contribution
        frames_used += 1

    seen = weight > 0.0
    colours = np.repeat(_UNSEEN[None, :], len(vertices), axis=0)
    colours[seen] = total[seen] / weight[seen, None]
    return colours, frames_used, int(np.count_nonzero(seen))


def rasterise(
    screen: np.ndarray,
    depth: np.ndarray,
    faces: np.ndarray,
    colours: np.ndarray,
    resolution: int,
) -> np.ndarray:
    """Z-buffered, smoothly interpolated triangles on a square canvas.

    Back faces are drawn, tinted, rather than culled: a scan is an open
    surface, and seeing the reverse of a wall is more honest than seeing
    through it.
    """

    canvas = np.repeat(
        _BACKGROUND[None, :],
        resolution * resolution,
        axis=0,
    )
    z_buffer = np.full(resolution * resolution, np.inf)

    in_front = np.all(depth[faces] > 1e-6, axis=1)
    faces = faces[in_front]
    x = screen[faces, 0]
    y = screen[faces, 1]
    area = (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0]) - (
        x[:, 2] - x[:, 0]
    ) * (y[:, 1] - y[:, 0])
    left = np.maximum(np.ceil(x.min(axis=1)), 0).astype(np.int64)
    right = np.minimum(np.floor(x.max(axis=1)), resolution - 1).astype(
        np.int64
    )
    top = np.maximum(np.ceil(y.min(axis=1)), 0).astype(np.int64)
    bottom = np.minimum(np.floor(y.max(axis=1)), resolution - 1).astype(
        np.int64
    )
    drawable = (area != 0.0) & (right >= left) & (bottom >= top)
    faces, area = faces[drawable], area[drawable]
    x, y = x[drawable], y[drawable]
    left, right, top, bottom = (
        left[drawable],
        right[drawable],
        top[drawable],
        bottom[drawable],
    )
    inverse_depth = 1.0 / depth[faces]
    # The mesh winds toward free space; in image coordinates, where y runs
    # downwards, that makes a triangle seen from its free side negative.
    front = area < 0.0
    side = np.maximum(right - left, bottom - top) + 1

    previous = 0
    for limit in _BUCKET_SIDES:
        bucket = np.flatnonzero((side > previous) & (side <= limit))
        previous = limit
        if bucket.size == 0:
            continue
        for offset_y in range(limit):
            rows = top[bucket] + offset_y
            row_active = bucket[rows <= bottom[bucket]]
            if row_active.size == 0:
                break
            for offset_x in range(limit):
                columns = left[row_active] + offset_x
                active = row_active[columns <= right[row_active]]
                if active.size == 0:
                    break
                sample_x = (left[active] + offset_x).astype(np.float64)
                sample_y = (top[active] + offset_y).astype(np.float64)
                ax, ay = x[active], y[active]
                weight_0 = (
                    (ax[:, 1] - sample_x) * (ay[:, 2] - sample_y)
                    - (ax[:, 2] - sample_x) * (ay[:, 1] - sample_y)
                ) / area[active]
                weight_1 = (
                    (ax[:, 2] - sample_x) * (ay[:, 0] - sample_y)
                    - (ax[:, 0] - sample_x) * (ay[:, 2] - sample_y)
                ) / area[active]
                weight_2 = 1.0 - weight_0 - weight_1
                inside = (
                    (weight_0 >= -_COVERAGE_TOLERANCE)
                    & (weight_1 >= -_COVERAGE_TOLERANCE)
                    & (weight_2 >= -_COVERAGE_TOLERANCE)
                )
                if not bool(np.any(inside)):
                    continue
                active = active[inside]
                weights = np.stack(
                    [weight_0[inside], weight_1[inside], weight_2[inside]],
                    axis=1,
                )
                sample_depth = 1.0 / np.einsum(
                    "ij,ij->i",
                    weights,
                    inverse_depth[active],
                )
                pixel = (
                    top[active] + offset_y
                ) * resolution + left[active] + offset_x

                # Nearest sample per pixel within this batch, then against
                # what is already on the canvas. Ties resolve by triangle
                # order, so the image does not depend on sort stability.
                order = np.lexsort((active, sample_depth, pixel))
                pixel = pixel[order]
                first = np.flatnonzero(np.diff(pixel, prepend=-1) != 0)
                chosen = order[first]
                pixel = pixel[first]
                nearer = sample_depth[chosen] < z_buffer[pixel]
                chosen = chosen[nearer]
                pixel = pixel[nearer]
                if pixel.size == 0:
                    continue
                triangle = active[chosen]
                shade = np.einsum(
                    "ij,ijk->ik",
                    weights[chosen],
                    colours[faces[triangle]],
                )
                shade[~front[triangle]] *= _BACK_TINT
                z_buffer[pixel] = sample_depth[chosen]
                canvas[pixel] = shade
    return canvas.reshape((resolution, resolution, 3))


def render(
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    base_colours: np.ndarray,
    azimuth_degrees: float,
    elevation_degrees: float,
    size: int,
    focal: float,
) -> Image.Image:
    """One shaded frame from one camera position."""

    resolution = size * _SUPERSAMPLE
    eye, right, up, forward = _camera_basis(
        vertices,
        azimuth_degrees,
        elevation_degrees,
    )
    relative = vertices - eye
    camera_z = relative @ forward
    safe_z = np.where(camera_z > 1e-6, camera_z, 1.0)
    centre = (resolution - 1) / 2.0
    screen = np.stack(
        [
            focal * (relative @ right) / safe_z + centre,
            -focal * (relative @ up) / safe_z + centre,
        ],
        axis=1,
    )
    # A headlight raised and offset slightly, so surfaces facing the viewer
    # are bright and curvature still reads; lit from both sides.
    light = -forward + 0.35 * right + 0.45 * up
    light /= np.linalg.norm(light)
    intensity = _AMBIENT + (1.0 - _AMBIENT) * np.abs(normals @ light)
    canvas = rasterise(
        screen,
        camera_z,
        faces,
        base_colours * intensity[:, None],
        resolution,
    )
    image = Image.fromarray(
        np.clip(canvas, 0.0, 255.0).astype(np.uint8),
        mode="RGB",
    )
    return image.resize((size, size), Image.LANCZOS)


def animation_azimuths(
    centre: float,
    frames: int,
    sweep: float | None,
) -> list[float]:
    """A full orbit, or a there-and-back sweep either side of ``centre``."""

    if sweep is None:
        return [centre + 360.0 * index / frames for index in range(frames)]
    if frames == 1:
        return [centre]
    return [
        centre + sweep * float(np.sin(2.0 * np.pi * index / frames))
        for index in range(frames)
    ]


def encode_gif(frames: list[Image.Image], target: Path) -> None:
    """Write frames against one shared palette.

    A palette chosen per frame makes flat regions shimmer as it changes
    from frame to frame, so one is built from every frame at once.
    """

    width, height = frames[0].size
    sample_step = max(1, len(frames) // 8)
    sampled = frames[::sample_step]
    montage = Image.new("RGB", (width * len(sampled), height))
    for index, frame in enumerate(sampled):
        montage.paste(frame, (index * width, 0))
    palette = montage.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    indexed = [
        frame.quantize(palette=palette, dither=Image.Dither.FLOYDSTEINBERG)
        for frame in frames
    ]
    indexed[0].save(
        target,
        format="GIF",
        save_all=True,
        append_images=indexed[1:],
        duration=70,
        loop=0,
        optimize=False,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output_prefix", type=Path)
    parser.add_argument("--session", type=Path, default=None)
    parser.add_argument("--colour-stride", type=positive_int, default=8)
    parser.add_argument("--size", type=positive_int, default=720)
    parser.add_argument("--frames", type=positive_int, default=36)
    parser.add_argument("--azimuth", type=float, default=35.0)
    parser.add_argument("--elevation", type=float, default=20.0)
    parser.add_argument("--sweep", type=float, default=None)
    parser.add_argument("--fill", type=unit_fraction, default=0.92)
    arguments = parser.parse_args(argv)

    if not -89.0 <= arguments.elevation <= 89.0:
        raise SystemExit(
            "--elevation must be within [-89, 89] degrees; at +/-90 the "
            "camera basis is undefined"
        )
    if arguments.sweep is not None and not 0.0 < arguments.sweep <= 180.0:
        raise SystemExit("--sweep must be within (0, 180] degrees")

    protected = [arguments.input]
    if arguments.session is not None:
        protected.append(arguments.session)
    still_path = reserve_output(
        arguments.output_prefix.with_suffix(".png"),
        ".png",
        protected=protected,
    )
    gif_path = reserve_output(
        arguments.output_prefix.with_suffix(".gif"),
        ".gif",
        protected=protected,
    )

    vertices, faces = read_mesh_ply(arguments.input)
    extent = vertices.max(axis=0) - vertices.min(axis=0)
    print(
        f"{len(vertices)} vertices, {len(faces)} triangles, extent "
        f"{extent[0]:.2f} x {extent[1]:.2f} x {extent[2]:.2f} m"
    )
    normals = vertex_normals(vertices, faces)
    if arguments.session is None:
        colours = np.repeat(_CLAY[None, :], len(vertices), axis=0)
        print("colour: uniform")
    else:
        colours, frames_used, seen = scan_vertex_colours(
            arguments.session,
            vertices,
            normals,
            arguments.colour_stride,
        )
        print(
            f"colour: scan RGB from {frames_used} frames, "
            f"{seen} of {len(vertices)} vertices observed"
        )

    azimuths = animation_azimuths(
        arguments.azimuth,
        arguments.frames,
        arguments.sweep,
    )
    focal = fit_focal(
        vertices,
        azimuths + [arguments.azimuth],
        arguments.elevation,
        arguments.size * _SUPERSAMPLE,
        arguments.fill,
    )

    def frame(azimuth: float) -> Image.Image:
        return render(
            vertices,
            faces,
            normals,
            colours,
            azimuth,
            arguments.elevation,
            arguments.size,
            focal,
        )

    still = crop_to_content([frame(arguments.azimuth)])[0]
    with publishing(still_path, ".png") as temporary:
        still.save(temporary, format="PNG")
    print(f"wrote {still_path}")

    animation = crop_to_content([frame(azimuth) for azimuth in azimuths])
    with publishing(gif_path, ".gif") as temporary:
        encode_gif(animation, temporary)
    print(f"wrote {gif_path} ({len(animation)} frames)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
