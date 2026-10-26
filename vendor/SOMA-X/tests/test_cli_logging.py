# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import ast
import logging
from pathlib import Path

import pytest

from tools.logging_utils import add_logging_args, log_level_from_args

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = REPO_ROOT / "tools"
INTERNAL_TOOLS_ROOT = TOOLS_ROOT / "internal"


def _is_json_stdout(node: ast.Call) -> bool:
    for arg in node.args:
        if not isinstance(arg, ast.Call):
            continue
        if not isinstance(arg.func, ast.Attribute):
            continue
        if not isinstance(arg.func.value, ast.Name):
            continue
        if arg.func.value.id == "json" and arg.func.attr == "dumps":
            return True
    return False


def _is_pose_converter_json_stdout(path: Path, node: ast.Call) -> bool:
    return path.name == "pose_converter.py" and _is_json_stdout(node)


def test_public_cli_tools_do_not_call_print_except_machine_readable_stdout():
    offenders = []
    for path in sorted(TOOLS_ROOT.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            if node.func.id != "print":
                continue
            if _is_pose_converter_json_stdout(path, node):
                continue
            offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")

    assert offenders == []


def test_internal_cli_tools_do_not_call_print_except_machine_readable_stdout():
    offenders = []
    for path in sorted(INTERNAL_TOOLS_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            if node.func.id != "print":
                continue
            if _is_json_stdout(node):
                continue
            offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")

    assert offenders == []


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        ([], logging.INFO),
        (["--verbose"], logging.DEBUG),
        (["--quiet"], logging.WARNING),
        (["--log-level", "ERROR"], logging.ERROR),
    ],
)
def test_logging_args_resolve_expected_levels(argv, expected):
    parser = argparse.ArgumentParser()
    add_logging_args(parser)
    args = parser.parse_args(argv)

    assert log_level_from_args(args) == expected


def test_logging_args_are_mutually_exclusive():
    parser = argparse.ArgumentParser()
    add_logging_args(parser)

    with pytest.raises(SystemExit):
        parser.parse_args(["--verbose", "--quiet"])
