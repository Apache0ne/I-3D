# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import unittest

import numpy as np

from motionbricks.utils.math_utils import (
    critical_spring_angle,
    critical_spring_damper_exact,
    wrap_angle,
)


class MathUtilsTests(unittest.TestCase):
    def test_critical_spring_converges_to_goal(self) -> None:
        current = 0.0
        rate = 0.0
        for _ in range(240):
            current, rate = critical_spring_damper_exact(
                current,
                rate,
                goal=3.0,
                halflife=0.4,
                delta_time=1.0 / 60.0,
            )

        self.assertAlmostEqual(current, 3.0, delta=2.0e-4)
        self.assertAlmostEqual(rate, 0.0, delta=2.0e-4)

    def test_angle_spring_uses_the_shortest_arc(self) -> None:
        current_angle = math.radians(179.0)
        desired_angle = math.radians(-179.0)
        desired_direction = np.array(
            [math.cos(desired_angle), math.sin(desired_angle), 0.0],
            dtype=np.float32,
        )

        next_angle, _next_rate = critical_spring_angle(
            current_angle,
            current_rate=0.0,
            desired_direction=desired_direction,
            halflife=0.2,
            delta_time=1.0 / 60.0,
        )

        angular_step = wrap_angle(next_angle - current_angle)
        self.assertGreater(angular_step, 0.0)
        self.assertLess(angular_step, math.radians(2.0))


if __name__ == "__main__":
    unittest.main()
