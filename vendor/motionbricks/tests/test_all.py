# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collect the tests in every folder under tests, so one command runs them all.

Unittest discovery skips plain folders, so this module discovers each folder that
holds tests:

    python -m unittest discover -s tests
"""

import unittest
from pathlib import Path


def load_tests(loader, tests, pattern):
    pattern = pattern or "test*.py"
    root = Path(__file__).resolve().parent
    for folder in sorted({path.parent for path in root.rglob(pattern)} - {root}):
        # A separate loader leaves the outer discovery untouched and keeps its -k filters.
        folder_loader = unittest.TestLoader()
        folder_loader.testNamePatterns = loader.testNamePatterns
        tests.addTests(folder_loader.discover(str(folder), pattern, str(folder)))
    return tests
