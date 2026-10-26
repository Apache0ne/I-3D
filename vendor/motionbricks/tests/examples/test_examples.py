# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The example scripts compile, and with a GPU they run for both rigs.

Running them loads the real engines, so it needs an NVIDIA GPU. The first run builds the engines
into .trt-cache, which takes several minutes. Set the MOTIONBRICKS_ENGINE_TESTS environment
variable to 1 to run them.
"""

import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_EXAMPLES = _REPO / "docs" / "examples"


class ExampleSourceTests(unittest.TestCase):
    def test_every_example_compiles(self):
        for path in sorted(_EXAMPLES.glob("*.py")):
            with self.subTest(example=path.name):
                compile(path.read_text(encoding="utf-8"), str(path), "exec")


@unittest.skipUnless(os.environ.get("MOTIONBRICKS_ENGINE_TESTS") == "1", "set MOTIONBRICKS_ENGINE_TESTS=1 to run")
class ExampleRunTests(unittest.TestCase):
    def run_example(self, name, model):
        result = subprocess.run(
            [sys.executable, "-u", str(_EXAMPLES / name), "--model", model],
            cwd=_REPO, capture_output=True, text=True, timeout=1800,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def test_generate_once_chains_two_generations(self):
        for model in ("soma", "unitree_g1"):
            with self.subTest(model=model):
                lines = re.findall(r"generated (\d+) frames", self.run_example("generate_once.py", model))
                self.assertEqual(len(lines), 2)
                for frames in lines:
                    # One generation makes 18 to 58 valid frames.
                    self.assertTrue(18 <= int(frames) <= 58)

    def test_locomotion_loop_turns_left(self):
        for model in ("soma", "unitree_g1"):
            with self.subTest(model=model):
                output = self.run_example("locomotion_loop.py", model)
                roots = [
                    [float(value) for value in match]
                    for match in re.findall(r"root \[\s*(-?[\d.]+)\s+(-?[\d.]+)\s*\]", output)
                ]
                # It walks along +X first, then bends toward +Y.
                self.assertGreater(roots[3][0], 1.5)
                self.assertGreater(roots[-1][1], 3.0)


if __name__ == "__main__":
    unittest.main()
