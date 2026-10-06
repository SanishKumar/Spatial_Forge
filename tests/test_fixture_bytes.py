"""The committed fixtures must reach every checkout byte for byte.

Replay digests hash the fixture files directly, so a checkout that rewrites
their line endings computes different digests from the same commit and every
pinned digest in the suite fails for a reason that has nothing to do with the
code. Git does exactly that on Windows by default (``core.autocrlf=true``)
unless ``.gitattributes`` marks the fixtures as not-text.
"""

from __future__ import annotations

import unittest
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent / "fixtures"


class FixtureByteTests(unittest.TestCase):
    def test_no_fixture_has_had_its_line_endings_rewritten(self) -> None:
        files = sorted(path for path in FIXTURES.rglob("*") if path.is_file())
        self.assertTrue(files, "no fixture files found")
        for path in files:
            with self.subTest(fixture=path.relative_to(FIXTURES).as_posix()):
                self.assertNotIn(
                    b"\r",
                    path.read_bytes(),
                    "fixture contains carriage returns: Git rewrote its "
                    "line endings on checkout, so every digest derived "
                    "from it will differ. .gitattributes must mark "
                    "tests/fixtures/** as -text.",
                )


if __name__ == "__main__":
    unittest.main()
