# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for target-pose runtime contracts and utilities."""

import unittest

import numpy as np
import warp as wp

from motionbricks.animation.skeleton import Skeleton
from motionbricks.runtime.target_pose import (
    IKPin,
    TargetPoseIKCorrector,
    TwoBoneIKChainConfig,
    adjust_target_dof_speed,
    adjust_target_pose_speed,
)


def _transform(x: float, y: float, z: float) -> np.ndarray:
    return np.array([x, y, z, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)


class TargetPoseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        wp.init()
        cls.previous_device = wp.get_device()
        wp.set_device("cpu")
        cls.local_values = np.stack(
            [
                _transform(0.0, 0.0, 0.0),
                _transform(0.0, 0.0, 0.0),
                _transform(1.0, 0.0, 0.0),
                _transform(1.0, 0.0, 0.0),
                _transform(0.0, -3.0, 0.0),
                _transform(1.0, 0.0, 0.0),
                _transform(1.0, 0.0, 0.0),
            ]
        )
        cls.skeleton = Skeleton(
            ["Base", "RootA", "MidA", "TipA", "RootB", "MidB", "TipB"],
            [-1, 0, 1, 2, 0, 4, 5],
            cls.local_values,
        )
        cls.ik_chains = (
            TwoBoneIKChainConfig(
                IKPin.LEFT_HAND,
                1,
                2,
                3,
                (0.0, 1.0, 0.0),
            ),
            TwoBoneIKChainConfig(
                IKPin.RIGHT_HAND,
                4,
                5,
                6,
                (0.0, 1.0, 0.0),
            ),
        )

    @classmethod
    def tearDownClass(cls) -> None:
        wp.set_device(cls.previous_device)

    def test_adjust_speed_scales_all_planar_velocities_and_preserves_pose_motion(self) -> None:
        poses = np.zeros((4, 2, 7), dtype=np.float32)
        poses[..., 6] = 1.0
        # Unequal steps, lateral motion, and height changes must survive scaling.
        poses[:, 0, :3] = [[2, -1, 0.8], [2.02, -0.97, 0.86],
                           [2.09, -0.98, 0.78], [2.12, -0.91, 0.9]]
        poses[:, 1, :3] = [[0.1, 0.2, 0.3], [0.2, 0.1, 0.4],
                           [0.3, 0.4, 0.2], [0.2, 0.3, 0.1]]
        angles = np.arange(8, dtype=np.float32).reshape(4, 2) * 0.1
        poses[..., 5] = np.sin(angles)
        poses[..., 6] = np.cos(angles)
        for device in wp.get_devices():
            for count in (2, 3, 4):
                source = poses[:count]
                velocities = np.diff(source[:, 0, :2], axis=0) * 30
                source_speed = float(np.linalg.norm(velocities.mean(axis=0)))
                for scale in (0.0, 0.5, 1.0, 2.0):
                    with self.subTest(device=str(device), count=count, scale=scale), wp.ScopedDevice(device):
                        target_poses = wp.array(source, dtype=wp.transform)
                        speed = source_speed * scale

                        adjust_target_pose_speed(target_poses, speed)

                        actual = target_poses.numpy()
                        actual_velocities = np.diff(actual[:, 0, :2], axis=0) * 30
                        np.testing.assert_allclose(actual_velocities, velocities * scale, atol=8e-6)
                        self.assertAlmostEqual(float(np.linalg.norm(actual_velocities.mean(axis=0))),
                                               speed, delta=8e-6)
                        np.testing.assert_array_equal(actual[0], source[0])
                        np.testing.assert_array_equal(actual[:, 0, 2:], source[:, 0, 2:])
                        np.testing.assert_array_equal(actual[:, 1:], source[:, 1:])

    def test_adjust_speed_leaves_zero_mean_planar_velocity_unchanged(self) -> None:
        poses = np.zeros((4, 1, 7), dtype=np.float32)
        poses[..., 6] = 1.0
        poses[:, 0, 2] = [0.8, 0.86, 0.78, 0.9]
        for device in wp.get_devices():
            for xy in ([[2, -1]] * 4, [[2, -1], [2.1, -0.9], [1.9, -1.1], [2, -1]]):
                poses[:, 0, :2] = xy
                for speed in (0.0, 4.0):
                    with self.subTest(device=str(device), xy=xy, speed=speed), wp.ScopedDevice(device):
                        target_poses = wp.array(poses, dtype=wp.transform)

                        adjust_target_pose_speed(target_poses, speed)

                        np.testing.assert_array_equal(target_poses.numpy(), poses)

    def test_adjust_dof_speed_applies_the_pose_rule_and_keeps_joint_angles(self) -> None:
        roots = np.zeros((4, 7), dtype=np.float32)
        roots[:, :3] = [[2, -1, 0.8], [2.02, -0.97, 0.86], [2.09, -0.98, 0.78], [2.12, -0.91, 0.9]]
        angles = np.arange(4, dtype=np.float32) * 0.1
        roots[:, 5] = np.sin(angles)
        roots[:, 6] = np.cos(angles)
        stationary = roots.copy()
        stationary[:, :2] = [2, -1]
        joint_angles = np.linspace(-1.0, 1.0, 12, dtype=np.float32).reshape(4, 3)
        for device in wp.get_devices():
            for window, source in (("moving", roots), ("stationary", stationary)):
                for count in (2, 3, 4):
                    for speed in (0.0, 1.5, 4.0):
                        with (
                            self.subTest(device=str(device), window=window, count=count, speed=speed),
                            wp.ScopedDevice(device),
                        ):
                            target_poses = wp.array(source[:count, None], dtype=wp.transform)
                            target_joint_q = wp.array(
                                np.concatenate([source[:count], joint_angles[:count]], axis=1), dtype=wp.float32,
                            )

                            adjust_target_pose_speed(target_poses, speed)
                            adjust_target_dof_speed(target_joint_q, speed)

                            actual = target_joint_q.numpy()
                            np.testing.assert_array_equal(actual[:, :7], target_poses.numpy()[:, 0])
                            np.testing.assert_array_equal(actual[:, 7:], joint_angles[:count])

    def test_ik_corrector_applies_only_captured_pins(self) -> None:
        corrector = TargetPoseIKCorrector(self.skeleton, self.ik_chains)
        target_values = self.local_values.copy()
        target_values[3] = _transform(0.0, 1.0, 0.0)
        target_values[6] = _transform(0.0, 1.0, 0.0)
        target_pose = wp.array(target_values, dtype=wp.transform)
        current_pose = wp.array(self.local_values, dtype=wp.transform)
        world_before = wp.empty(7, dtype=wp.transform)
        self.skeleton.compute_world_transforms(current_pose, world_before)

        corrector.capture_pins(target_pose, IKPin.LEFT_HAND)
        corrector.apply(current_pose, 1.0)

        corrected_world = wp.empty(7, dtype=wp.transform)
        self.skeleton.compute_world_transforms(current_pose, corrected_world)
        corrected_values = corrected_world.numpy()
        np.testing.assert_allclose(
            corrected_values[3, :3],
            [1.0, 1.0, 0.0],
            atol=1.0e-4,
        )
        np.testing.assert_allclose(
            corrected_values[4:7],
            world_before.numpy()[4:7],
            atol=1.0e-6,
        )
        self.assertEqual(corrector.pin_mask, IKPin.LEFT_HAND)

        # A target without pins clears them, and the pose stays as sampled.
        corrector.capture_pins(target_pose, IKPin.NONE)
        uncorrected = wp.array(self.local_values, dtype=wp.transform)
        corrector.apply(uncorrected, 1.0)
        np.testing.assert_array_equal(uncorrected.numpy(), self.local_values)


if __name__ == "__main__":
    unittest.main()
