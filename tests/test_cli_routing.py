from __future__ import annotations

import argparse
import io
import unittest
from contextlib import redirect_stderr, redirect_stdout

from spatialforge.cli import _build_parser, main


def subcommand_names(parser: argparse.ArgumentParser, group: str) -> tuple[str, ...]:
    for action in parser._actions:
        choices = getattr(action, "choices", None)
        if choices and group in choices:
            child = choices[group]
            for child_action in child._actions:
                child_choices = getattr(child_action, "choices", None)
                if child_choices:
                    return tuple(sorted(child_choices))
    raise AssertionError(f"no {group} subcommands found")


class CliRoutingTests(unittest.TestCase):
    """Guard the parser and the dispatch chain against drifting apart.

    ``main`` routes on the subcommand string. A parser entry with no matching
    branch used to fall through to the last handler, which would then fail on
    the wrong arguments; an unrouted command now raises instead. These tests
    keep both halves in step.
    """

    def test_every_registered_subcommand_is_routed(self) -> None:
        parser = _build_parser()
        for group in ("reconstruct", "scan"):
            for name in subcommand_names(parser, group):
                with self.subTest(group=group, command=name):
                    stdout = io.StringIO()
                    stderr = io.StringIO()
                    with self.assertRaises(SystemExit) as raised:
                        with (
                            redirect_stdout(stdout),
                            redirect_stderr(stderr),
                        ):
                            main([group, name, "--help"])
                    self.assertEqual(raised.exception.code, 0)
                    self.assertIn(name, stdout.getvalue())

    def test_unrouted_reconstruct_command_raises(self) -> None:
        arguments = argparse.Namespace(
            command="reconstruct",
            reconstruct_command="not-a-real-command",
        )
        with patch_parser(arguments):
            with self.assertRaises(AssertionError) as raised:
                main([])
        self.assertIn("unrouted reconstruct command", str(raised.exception))
        self.assertIn("not-a-real-command", str(raised.exception))

    def test_reconstruct_subcommands_are_unique_and_nonempty(self) -> None:
        parser = _build_parser()
        names = subcommand_names(parser, "reconstruct")
        self.assertEqual(len(names), len(set(names)))
        self.assertGreater(len(names), 20)
        for name in names:
            self.assertTrue(name.strip())
            self.assertEqual(name, name.strip().lower())


class _StubParser:
    def __init__(self, arguments: argparse.Namespace) -> None:
        self._arguments = arguments

    def parse_args(self, argv: object) -> argparse.Namespace:
        return self._arguments


def patch_parser(arguments: argparse.Namespace):
    from unittest.mock import patch

    return patch(
        "spatialforge.cli._build_parser",
        return_value=_StubParser(arguments),
    )


if __name__ == "__main__":
    unittest.main()
