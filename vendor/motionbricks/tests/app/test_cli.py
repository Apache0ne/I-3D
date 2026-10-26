# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Output levels shared by the apps."""

import argparse
import logging
import unittest

import warp as wp

from app.utils.cli import configure_logging


class ConfigureLoggingTests(unittest.TestCase):
    def setUp(self):
        root = logging.getLogger()
        names = ("__main__", "app", "motionbricks")
        state = (root.level, list(root.handlers), {name: logging.getLogger(name).level for name in names})
        self.addCleanup(self._restore, *state)
        self.addCleanup(setattr, wp.config, "log_level", wp.config.log_level)

    @staticmethod
    def _restore(level, handlers, levels):
        root = logging.getLogger()
        for handler in list(root.handlers):
            root.removeHandler(handler)
        for handler in handlers:
            root.addHandler(handler)
        root.setLevel(level)
        for name, value in levels.items():
            logging.getLogger(name).setLevel(value)

    def test_quiet_default_and_verbose_levels(self):
        for quiet, verbose, level in (
            (True, False, logging.WARNING),
            (False, False, logging.INFO),
            (False, True, logging.DEBUG),
        ):
            with self.subTest(quiet=quiet, verbose=verbose):
                configure_logging(argparse.Namespace(quiet=quiet, verbose=verbose))
                for name in ("__main__", "app.scene_preview", "motionbricks.io.bvh"):
                    self.assertEqual(logging.getLogger(name).getEffectiveLevel(), level)
                self.assertEqual(logging.getLogger("other.library").getEffectiveLevel(), logging.WARNING)
                self.assertEqual(wp.config.log_level >= wp.LOG_WARNING, not verbose)
                wp.config.log_level = wp.LOG_INFO


if __name__ == "__main__":
    unittest.main()
