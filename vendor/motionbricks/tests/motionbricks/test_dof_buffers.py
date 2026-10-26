# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DOF clip sampling, DOF tokens and Newton playback on synthetic data."""

import unittest

import newton
import numpy as np
import warp as wp

from motionbricks.model.unitree_g1.buffers import DOFToken
from motionbricks.robotics.dof_animation_buffer import DOFAnimationBuffer


def _poses(count):
    poses = np.zeros((count, 9), dtype=np.float32)
    poses[:, 0] = np.arange(count)
    poses[:, 6] = 1.0
    poses[:, 7] = np.arange(count) * 0.25
    poses[:, 8] = -poses[:, 7]
    return poses


def _assert_transform(test, actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    np.testing.assert_allclose(actual[:3], expected[:3], atol=2e-6)
    test.assertAlmostEqual(abs(float(actual[3:] @ expected[3:])), 1.0, places=5)


class DOFBufferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.previous_device = wp.get_device()
        wp.set_device("cpu")

    @classmethod
    def tearDownClass(cls):
        wp.set_device(cls.previous_device)

    def test_token_reuses_storage_and_replaces_padding_on_each_update(self):
        poses = _poses(4)
        source = wp.array(poses, dtype=wp.float32)
        token = DOFToken(2)
        pointer = token.poses.ptr
        self.assertEqual(token.num_poses, 0)
        for count in (4, 2, 1, 3):
            with self.subTest(count=count):
                token.update_from_poses(source[0] if count == 1 else source[:count])
                self.assertEqual(token.num_poses, count)
                self.assertEqual(token.poses.ptr, pointer)
                expected = np.concatenate((poses[:count], np.repeat(poses[count - 1:count], 4 - count, axis=0)))
                np.testing.assert_array_equal(token.poses.numpy(), expected)
        np.testing.assert_array_equal(source.numpy(), poses)

    def test_sampling_places_the_root_clamps_time_and_writes_only_the_robot_view(self):
        poses = np.full((6, 9), np.nan, dtype=np.float32)
        poses[:2, :3] = [[1, 2, 3], [5, 6, 7]]
        poses[:2, 3:7] = [
            wp.quat_from_axis_angle(wp.vec3(0, 0, 1), float(np.deg2rad(170))),
            wp.quat_from_axis_angle(wp.vec3(0, 0, 1), float(np.deg2rad(-170))),
        ]
        poses[:2, 7:] = [[-2.8, 1.2], [2.8, -0.4]]
        source = wp.array(poses, dtype=wp.float32)
        animation = DOFAnimationBuffer(("a", "b"), 2, source, num_frames=2)
        joint_q = wp.full(27, -100.0, dtype=wp.float32)
        robot_view = joint_q[9:18]
        offset = wp.transform(wp.vec3(10, -4, 0), wp.quat_from_axis_angle(wp.vec3(0, 1, 0), 0.3))

        animation.sample(0.25, robot_view, root_offset=offset)
        actual = robot_view.numpy()
        expected_root = offset * wp.transform(wp.vec3(3, 4, 5), wp.quat_from_axis_angle(wp.vec3(0, 0, 1), np.pi))
        _assert_transform(self, actual[:7], expected_root)
        np.testing.assert_allclose(actual[7:], [0, 0.4], atol=1e-6)
        for num_frames, time, frame in ((2, -1.0, 0), (2, 10.0, 1), (1, -1.0, 0), (1, 10.0, 0)):
            animation.num_frames = num_frames
            animation.sample(time, robot_view)
            np.testing.assert_allclose(robot_view.numpy(), poses[frame], atol=1e-6)

        window = _poses(13)
        clip = DOFAnimationBuffer(("a", "b"), 120, wp.array(window, dtype=wp.float32))
        for frame in range(4):
            clip.sample(frame / 30, robot_view)
            np.testing.assert_allclose(robot_view.numpy(), window[frame * 4], atol=1e-6)

        np.testing.assert_array_equal(joint_q.numpy()[:9], -100)
        np.testing.assert_array_equal(joint_q.numpy()[18:], -100)
        self.assertEqual(animation.poses.ptr, source.ptr)
        np.testing.assert_array_equal(source.numpy(), poses)

    def test_make_loopable_moves_the_root_along_a_straight_line(self):
        # The root makes a quarter turn and ends 0.3 m from its start.
        poses = _poses(3)
        poses[:, 0] = (0.0, 1.0, 0.3)
        for frame, angle in enumerate((0.0, 0.25 * np.pi, 0.5 * np.pi)):
            poses[frame, 3:7] = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), angle)
        animation = DOFAnimationBuffer(("a", "b"), 30, wp.array(poses, dtype=wp.float32))

        animation.make_loopable()

        corrected = animation.poses.numpy()
        # The middle frame moves by half the end gap, in the same direction as the gap.
        np.testing.assert_allclose(corrected[1, :3], [0.85, 0.0, 0.0], atol=1e-6)
        _assert_transform(self, corrected[2, :7], corrected[0, :7])

    def test_copy_keeps_the_frames_in_use_in_new_storage(self):
        poses = wp.array(_poses(3), dtype=wp.float32)
        contacts = wp.array(np.arange(12, dtype=np.float32).reshape(3, 4), dtype=wp.float32)
        animation = DOFAnimationBuffer(("a", "b"), 30, poses, num_frames=2, foot_contacts=contacts)

        copy = animation.copy()

        self.assertEqual(copy.joint_names, ("a", "b"))
        self.assertEqual((copy.capacity, copy.num_frames, copy.sample_rate), (2, 2, 30.0))
        self.assertNotEqual(copy.poses.ptr, poses.ptr)
        np.testing.assert_array_equal(copy.poses.numpy(), poses.numpy()[:2])
        np.testing.assert_array_equal(copy.foot_contacts.numpy(), contacts.numpy()[:2])
        self.assertIsNone(DOFAnimationBuffer(("a", "b"), 30, poses).copy().foot_contacts)

    def test_valid_count_controls_duration_and_hides_stale_frames(self):
        animation = DOFAnimationBuffer(
            ("a", "b"), 30, wp.empty((6, 9), dtype=wp.float32), num_frames=0,
        )
        pointer = animation.poses.ptr
        self.assertEqual(animation.capacity, 6)
        output = wp.empty(9, dtype=wp.float32)
        self.assertEqual(animation.num_frames, 0)
        self.assertEqual(animation.duration, 0)
        with self.assertRaisesRegex(ValueError, "empty"):
            animation.sample(0, output)
        poses = _poses(6)
        animation.poses.assign(poses)
        for count in (6, 2, 1):
            animation.num_frames = count
            self.assertEqual(animation.poses.ptr, pointer)
            self.assertAlmostEqual(animation.duration, (count - 1) / 30)
            animation.sample(100, output)
            np.testing.assert_array_equal(output.numpy(), poses[count - 1])
        animation.num_frames = 0
        with self.assertRaisesRegex(ValueError, "empty"):
            animation.sample(0, output)
        for invalid in (-1, animation.capacity + 1):
            with self.assertRaises(ValueError):
                animation.num_frames = invalid
        self.assertEqual(animation.num_frames, 0)

    def test_samples_directly_into_newton_joint_q_and_evaluates_fk(self):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        root = builder.add_link(label="root", mass=1, inertia=wp.mat33(1.0))
        arm = builder.add_link(label="arm", mass=1, inertia=wp.mat33(1.0))
        leg = builder.add_link(label="leg", mass=1, inertia=wp.mat33(1.0))
        joints = [builder.add_joint_free(root, label="free")]
        joints.append(builder.add_joint_revolute(
            root, arm, axis=wp.vec3(0, 0, 1),
            parent_xform=wp.transform(wp.vec3(1, 0, 0), wp.quat_identity()), label="robot/elbow",
        ))
        joints.append(builder.add_joint_revolute(
            root, leg, axis=wp.vec3(0, 1, 0),
            parent_xform=wp.transform(wp.vec3(0, 0, -1), wp.quat_identity()), label="robot/knee",
        ))
        builder.add_articulation(joints)
        model = builder.finalize()
        state = model.state()
        poses = _poses(2)
        poses[:, :3] = [[4, -2, 1], [6, 2, 3]]
        poses[:, 3:7] = wp.quat_rpy(0.2, -0.3, 0.7)
        poses[:, 7:] = [[0.25, -0.75], [0.75, -0.25]]
        clip = DOFAnimationBuffer(("elbow", "knee"), 2, wp.array(poses, dtype=wp.float32))
        offset = wp.transform(wp.vec3(3, -1, 0.6), wp.quat_rpy(0.0, 0.0, 0.3))

        clip.sample(0.25, model.joint_q, root_offset=offset)
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)

        position = poses[:, :3].mean(axis=0)
        hinges = poses[:, 7:].mean(axis=0)
        root_transform = offset * wp.transform(wp.vec3(*position), wp.quat(*poses[0, 3:7]))
        sample = model.joint_q.numpy()
        _assert_transform(self, sample[:7], root_transform)
        np.testing.assert_allclose(sample[7:], hinges, atol=1e-6)
        body_transforms = state.body_q.numpy()
        _assert_transform(self, body_transforms[root], root_transform)
        _assert_transform(self, body_transforms[arm], root_transform * wp.transform(
            wp.vec3(1, 0, 0), wp.quat_from_axis_angle(wp.vec3(0, 0, 1), float(hinges[0])),
        ))
        _assert_transform(self, body_transforms[leg], root_transform * wp.transform(
            wp.vec3(0, 0, -1), wp.quat_from_axis_angle(wp.vec3(0, 1, 0), float(hinges[1])),
        ))


if __name__ == "__main__":
    unittest.main()
