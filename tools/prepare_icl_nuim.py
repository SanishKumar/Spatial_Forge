"""Rewrite an ICL-NUIM sequence as a right-handed TUM RGB-D folder.

ICL-NUIM publishes its sequences "in TUM RGB-D format", but two things stop
the TUM importer from reading one as it stands.

The layout differs: one ``associations.txt`` instead of ``rgb.txt`` and
``depth.txt``, and a trajectory named ``*.gt.freiburg``.

The geometry differs, and this is the part that matters. The scenes were
rendered by POV-Ray in a left-handed frame, and the dataset says so with a
negative focal length: ``fy = -480``. With that sign the camera's y axis
points up the image. Dropping the sign and keeping the poses, which is the
easy thing to do, does not give an error. It gives a self-consistent
reconstruction of a room that does not exist: the mirror image of each
frame, placed by poses that belong to the unmirrored one.

The conversion is a change of basis on both sides of every pose. Write the
published pose as ``x_world = R x_camera + t`` in the left-handed frames,
``S = diag(1, -1, 1)`` for the flip that makes the camera y axis point down,
and ``F = diag(1, 1, -1)`` for a flip of the world z axis. Then

    R' = F R S        t' = F t

is a proper rotation and a translation: a right-handed camera, x right, y
down, z forward, with ``fy = +480``, in a right-handed world. Flipping any
one world axis would do that much. Flipping z is the choice that leaves the
result within about a degree, and a translation, of the frame of the
dataset's ground-truth room model; ``surface_accuracy_report.py`` fits that
remaining rigid motion rather than assuming it.

For a unit quaternion the whole conversion is a permutation,

    (qx, qy, qz, qw)  ->  (qw, qz, qy, qx)        tz -> -tz

so nothing is recomputed and no rounding is introduced: the output carries
the dataset's own digits.

Frames without a pose are left out, since nothing downstream can place
them. In the published sequences that is frame 0.

Usage:

    python tools/prepare_icl_nuim.py SOURCE OUTPUT

``SOURCE`` is an extracted ``*_frei_png`` sequence. ``OUTPUT`` must not
exist. The result imports with:

    python -m spatialforge scan import-tum OUTPUT SESSION.vgsession
        --fx 481.2 --fy 480 --cx 319.5 --cy 239.5
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath

if __package__ in (None, ""):  # run as a script rather than imported
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# What the importer must be told about the camera, with the sign of fy
# already absorbed into the poses written here.
ICL_NUIM_INTRINSICS = {"fx": 481.2, "fy": 480.0, "cx": 319.5, "cy": 239.5}


def convert_pose(fields: list[str]) -> list[str]:
    """One ``tx ty tz qx qy qz qw`` row, left-handed to right-handed."""

    tx, ty, tz, qx, qy, qz, qw = fields
    return [tx, ty, _negated(tz), qw, qz, qy, qx]


def prepare_icl_nuim(source: Path, output: Path) -> dict[str, int]:
    """Write the TUM-layout folder; returns counts of what was written."""

    source = Path(source).resolve()
    output = Path(output).resolve()
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

    frames = _read_associations(source)
    poses = _read_trajectory(source)
    # Matched by value, so that 1 and 1.0 are the same instant; written
    # with the frame's own token.
    posed = [frame for frame in frames if Decimal(frame[0]) in poses]
    if not posed:
        raise SystemExit(
            "no frame in associations.txt has a pose in the trajectory"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent)
    )
    try:
        (staging / "rgb").mkdir()
        (staging / "depth").mkdir()
        rgb_lines = []
        depth_lines = []
        pose_lines = []
        for stamp, depth_path, rgb_path in posed:
            for kind, relative, lines in (
                ("rgb", rgb_path, rgb_lines),
                ("depth", depth_path, depth_lines),
            ):
                name = f"{kind}/{PurePosixPath(relative).name}"
                shutil.copyfile(source / relative, staging / name)
                lines.append(f"{stamp} {name}")
            pose_lines.append(
                " ".join([stamp, *convert_pose(poses[Decimal(stamp)])])
            )
        _write_lines(staging / "rgb.txt", rgb_lines)
        _write_lines(staging / "depth.txt", depth_lines)
        _write_lines(staging / "groundtruth.txt", pose_lines)
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
        "frames": len(posed),
        "frames_without_pose": len(frames) - len(posed),
        "poses_without_frame": len(poses) - len(posed),
    }


def _negated(token: str) -> str:
    value = _number(token, "translation")
    if value == 0:
        return "0"
    return str(-value)


def _number(token: str, what: str) -> Decimal:
    try:
        value = Decimal(token)
    except InvalidOperation:
        raise SystemExit(f"{what} is not a number: {token!r}") from None
    if not value.is_finite():
        raise SystemExit(f"{what} is not finite: {token!r}")
    return value


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


def _read_associations(source: Path) -> list[tuple[str, str, str]]:
    """``(timestamp, depth path, rgb path)`` per frame, in file order."""

    path = source / "associations.txt"
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    frames = []
    seen: set[Decimal] = set()
    names: set[str] = set()
    for number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        where = f"associations.txt:{number}"
        if len(fields) != 4:
            raise SystemExit(
                f"{where}: expected 'timestamp depth timestamp rgb'"
            )
        depth_stamp, depth_path, rgb_stamp, rgb_path = fields
        if depth_stamp != rgb_stamp:
            raise SystemExit(
                f"{where}: depth and rgb timestamps differ; this is not "
                "the pre-associated ICL-NUIM layout"
            )
        instant = _number(depth_stamp, f"{where}: timestamp")
        if instant in seen:
            raise SystemExit(f"{where}: repeated timestamp {depth_stamp}")
        seen.add(instant)
        depth_path = _relative_file(source, depth_path, where)
        rgb_path = _relative_file(source, rgb_path, where)
        for kind, relative in (("depth", depth_path), ("rgb", rgb_path)):
            name = f"{kind}/{PurePosixPath(relative).name}"
            if name in names:
                raise SystemExit(f"{where}: repeated file name {name}")
            names.add(name)
        frames.append((depth_stamp, depth_path, rgb_path))
    return frames


def _read_trajectory(source: Path) -> dict[Decimal, list[str]]:
    candidates = sorted(source.glob("*.gt.freiburg"))
    if len(candidates) != 1:
        raise SystemExit(
            f"expected exactly one *.gt.freiburg trajectory in {source}, "
            f"found {len(candidates)}"
        )
    poses: dict[Decimal, list[str]] = {}
    for number, line in enumerate(
        candidates[0].read_text(encoding="utf-8").splitlines(), start=1
    ):
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        where = f"{candidates[0].name}:{number}"
        if len(fields) != 8:
            raise SystemExit(
                f"{where}: expected 'timestamp tx ty tz qx qy qz qw'"
            )
        for token in fields:
            _number(token, where)
        instant = Decimal(fields[0])
        if instant in poses:
            raise SystemExit(f"{where}: repeated timestamp {fields[0]}")
        poses[instant] = fields[1:]
    return poses


def _write_lines(path: Path, lines: list[str]) -> None:
    # Bytes, so the folder is the same on every platform.
    path.write_bytes(("\n".join(lines) + "\n").encode("ascii"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args(argv)

    counts = prepare_icl_nuim(arguments.source, arguments.output)
    print(f"wrote {Path(arguments.output).resolve()}")
    print(
        f"frames: {counts['frames']} "
        f"(left out, no pose: {counts['frames_without_pose']}; "
        f"poses with no frame: {counts['poses_without_frame']})"
    )
    print(
        "import with: --fx {fx} --fy {fy} --cx {cx} --cy {cy}".format(
            **ICL_NUIM_INTRINSICS
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
