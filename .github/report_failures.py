"""Turn a failed unittest run into GitHub annotations.

A red job normally says only "Process completed with exit code 1", and the
log behind it needs a signed-in session to read. This prints each failing
test's traceback as a workflow error annotation, so the reason is visible on
the run summary itself.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

MAX_ANNOTATIONS = 9
MAX_MESSAGE_CHARACTERS = 6_000
SEPARATOR = "=" * 70


def escape(text: str) -> str:
    return (
        text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    )


def main() -> int:
    output = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
    blocks = [
        block.strip()
        for block in output.split(SEPARATOR)[1:]
        if re.match(r"\s*(ERROR|FAIL):", block)
    ]
    if not blocks:
        tail = "\n".join(output.splitlines()[-60:])
        print(f"::error title=Test run failed::{escape(tail)}")
        return 0
    for block in blocks[:MAX_ANNOTATIONS]:
        title = block.splitlines()[0][:200]
        body = block[-MAX_MESSAGE_CHARACTERS:]
        print(f"::error title={escape(title)}::{escape(body)}")
    if len(blocks) > MAX_ANNOTATIONS:
        print(
            "::error title=More failures::"
            f"{len(blocks) - MAX_ANNOTATIONS} further failing tests omitted"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
