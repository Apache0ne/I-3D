# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The bundled Unitree G1 robot matches the joint layout of the G1 model."""

import unittest

from motionbricks.io.utils import get_unitree_g1_model
from motionbricks.model.unitree_g1.contract import UNITREE_G1_N1_CONTRACT


class UnitreeG1RobotTests(unittest.TestCase):
    def test_robot_joints_follow_the_model_order(self):
        # Generated poses are written straight into joint_q, as the free root followed by one angle per model joint.
        robot = get_unitree_g1_model(device="cpu", load_visuals=False)
        joint_names = list(UNITREE_G1_N1_CONTRACT.joint_names)
        self.assertEqual([label.rsplit("/", 1)[-1] for label in robot.joint_label[1:]], joint_names)
        self.assertEqual(robot.joint_coord_count, 7 + len(joint_names))


if __name__ == "__main__":
    unittest.main()
