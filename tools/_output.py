"""Fail-closed output publishing and provenance for the analysis tools.

The library refuses to overwrite its artifacts and publishes them by linking
a fully written temporary sibling into place, so a failed run leaves nothing
behind and a half-written file can never be mistaken for a result. The tools
that produce published numbers and pictures need the same guarantee, and they
additionally need to refuse writing over the very inputs they were given.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent


def positive_int(value: str) -> int:
    """argparse type: an integer of at least one."""

    try:
        parsed = int(value)
    except ValueError:
        raise ValueError(f"expected an integer, got {value!r}") from None
    if parsed < 1:
        raise ValueError(f"expected a positive integer, got {parsed}")
    return parsed


def non_negative_int(value: str) -> int:
    """argparse type: an integer of at least zero."""

    try:
        parsed = int(value)
    except ValueError:
        raise ValueError(f"expected an integer, got {value!r}") from None
    if parsed < 0:
        raise ValueError(f"expected a non-negative integer, got {parsed}")
    return parsed


def unit_fraction(value: str) -> float:
    """argparse type: a float in (0, 1]."""

    try:
        parsed = float(value)
    except ValueError:
        raise ValueError(f"expected a number, got {value!r}") from None
    if not 0.0 < parsed <= 1.0:
        raise ValueError(f"expected a fraction in (0, 1], got {parsed}")
    return parsed


def reserve_output(
    output: Path,
    suffix: str,
    *,
    protected: Sequence[Path] = (),
) -> Path:
    """Resolve an output path, refusing to overwrite or clobber an input."""

    resolved = Path(output).resolve()
    if resolved.suffix.lower() != suffix:
        raise SystemExit(f"output filename must end in {suffix}: {resolved}")
    # Checked before mere existence so that aiming an output at an input
    # reports why it is wrong rather than the incidental fact that inputs
    # tend to already exist.
    for guard in protected:
        guarded = Path(guard).resolve()
        if resolved == guarded or resolved.is_relative_to(guarded):
            raise SystemExit(
                f"output would write over an input: {resolved} is inside "
                f"or equal to {guarded}"
            )
    if resolved.exists():
        raise SystemExit(
            f"output already exists, refusing to overwrite: {resolved}"
        )
    return resolved


@contextmanager
def publishing(target: Path, suffix: str) -> Iterator[Path]:
    """Yield a temporary sibling, then link it into place on success.

    ``os.link`` fails rather than overwrites if the target appeared while the
    run was in progress, and the temporary file is removed either way, so a
    failed run publishes nothing.
    """

    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{target.stem}-",
        suffix=suffix,
        dir=target.parent,
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        yield temporary
        try:
            os.link(temporary, target)
        except FileExistsError as error:
            raise SystemExit(
                "output appeared while running; refusing to overwrite: "
                f"{target}"
            ) from error
        except OSError as error:
            raise SystemExit(f"cannot publish {target}: {error}") from error
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _git(*arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=REPOSITORY_ROOT,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def source_state() -> tuple[str | None, bool | None]:
    """Return the current commit and whether the working tree is clean.

    Both are ``None`` when git is unavailable, which is itself worth
    recording: a manifest that cannot name its source should say so rather
    than imply a clean checkout.
    """

    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return (
        commit.strip() or None if commit is not None else None,
        status.strip() == "" if status is not None else None,
    )
