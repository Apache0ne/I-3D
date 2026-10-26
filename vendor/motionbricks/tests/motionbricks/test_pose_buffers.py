# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

from motionbricks.runtime.motion_recorder import MotionRecorder
from motionbricks.animation.animation_buffer import AnimationBuffer
from motionbricks.animation.skeleton import Skeleton
from motionbricks.model.soma.buffers import PoseToken
from motionbricks.utils import pose_utils


def _pose(x: float) -> np.ndarray:
    return np.array([[x, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32)


class PoseBufferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        wp.init()
        if not wp.is_cuda_available():
            raise unittest.SkipTest("CUDA is required for MotionBricks pose buffers.")
        cls.previous_device = wp.get_device()
        cls.device = wp.get_device("cuda:0")
        wp.set_device(cls.device)
        cls.skeleton = Skeleton(
            ["Root"],
            [-1],
            _pose(0.0),
        )

    @classmethod
    def tearDownClass(cls) -> None:
        wp.set_device(cls.previous_device)

    def test_pose_token_reuses_storage(self) -> None:
        token = PoseToken(1)
        storage_ptr = token.poses.ptr
        self.assertEqual(token.num_poses, 0)

        token.update_from_poses(wp.array(np.stack([_pose(2.0)]), dtype=wp.transform))
        wp.synchronize_device(self.device)
        self.assertEqual(token.num_poses, 1)
        np.testing.assert_allclose(token.poses.numpy()[:, 0, 0], 2.0)

        token.update_from_poses(wp.array(np.stack([_pose(1.0), _pose(3.0)]), dtype=wp.transform))
        wp.synchronize_device(self.device)
        self.assertEqual(token.poses.ptr, storage_ptr)
        self.assertEqual(token.num_poses, 2)
        np.testing.assert_allclose(token.poses.numpy()[:, 0, 0], [1.0, 3.0, 3.0, 3.0])

        with self.assertRaisesRegex(ValueError, "requires 1 to 4 poses"):
            token.update_from_poses(wp.array(np.stack([_pose(0.0)] * 5), dtype=wp.transform))

    def test_runtime_root_offset_moves_only_the_root_joint(self) -> None:
        skeleton = Skeleton(["Root", "Child"], [-1, 0], np.concatenate([_pose(0.0), _pose(0.0)]))
        frames = np.stack([
            np.concatenate([_pose(1.0), _pose(5.0)]),
            np.concatenate([_pose(3.0), _pose(7.0)]),
        ])
        animation = AnimationBuffer(skeleton, 30.0, wp.array(frames, dtype=wp.transform))
        output_pose = wp.empty(2, dtype=wp.transform)
        root_offset = wp.transform(
            wp.vec3(10.0, 20.0, 0.0),
            wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.5 * np.pi),
        )

        animation.sample(1.0 / 60.0, output_pose, root_offset=root_offset)
        wp.synchronize_device(self.device)

        expected_root = wp.mul(root_offset, wp.transform(wp.vec3(2.0, 0.0, 0.0), wp.quat_identity()))
        np.testing.assert_allclose(output_pose.numpy()[0], np.asarray(expected_root), atol=1.0e-6)
        np.testing.assert_allclose(output_pose.numpy()[1], _pose(6.0)[0], atol=1.0e-6)

    def test_make_loopable_distributes_end_to_start_delta(self) -> None:
        frames = wp.array(
            np.stack([_pose(0.0), _pose(1.0), _pose(3.0), _pose(100.0), _pose(200.0)]),
            dtype=wp.transform,
        )
        animation = AnimationBuffer(self.skeleton, 30.0, frames, num_frames=3)

        animation.make_loopable()
        wp.synchronize_device(self.device)

        corrected = animation.poses.numpy()
        np.testing.assert_allclose(corrected[0], _pose(0.0))
        np.testing.assert_allclose(corrected[1], _pose(-0.5))
        np.testing.assert_allclose(corrected[2], _pose(0.0))
        np.testing.assert_array_equal(corrected[3:], np.stack([_pose(100.0), _pose(200.0)]))

        animation.num_frames = 1
        animation.make_loopable()
        np.testing.assert_allclose(animation.poses.numpy()[0], _pose(0.0))

    def test_make_loopable_moves_the_root_along_a_straight_line(self) -> None:
        # The root makes a quarter turn and ends 0.3 m from its start.
        frames = np.array([
            [[x, 0.0, 0.0, *wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), angle)]]
            for x, angle in ((0.0, 0.0), (1.0, 0.25 * np.pi), (0.3, 0.5 * np.pi))
        ], dtype=np.float32)
        animation = AnimationBuffer(self.skeleton, 30.0, wp.array(frames, dtype=wp.transform))

        animation.make_loopable()

        corrected = animation.poses.numpy()[:, 0]
        # The middle frame moves by half the end gap, in the same direction as the gap.
        np.testing.assert_allclose(corrected[1, :3], [0.85, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(corrected[2, :3], corrected[0, :3], atol=1e-6)
        self.assertAlmostEqual(abs(float(corrected[2, 3:] @ corrected[0, 3:])), 1.0, places=6)

    def test_copy_keeps_the_frames_in_use_in_new_storage(self) -> None:
        frames = wp.array(np.stack([_pose(0.0), _pose(1.0), _pose(9.0)]), dtype=wp.transform)
        contacts = wp.array(np.arange(12, dtype=np.float32).reshape(3, 4), dtype=wp.float32)
        animation = AnimationBuffer(self.skeleton, 30.0, frames, num_frames=2, foot_contacts=contacts)

        copy = animation.copy()

        self.assertIs(copy.skeleton, self.skeleton)
        self.assertEqual((copy.capacity, copy.num_frames, copy.sample_rate), (2, 2, 30.0))
        self.assertNotEqual(copy.poses.ptr, frames.ptr)
        np.testing.assert_array_equal(copy.poses.numpy(), frames.numpy()[:2])
        np.testing.assert_array_equal(copy.foot_contacts.numpy(), contacts.numpy()[:2])
        self.assertIsNone(AnimationBuffer(self.skeleton, 30.0, frames).copy().foot_contacts)

    def test_delta_pose_is_right_multiplicative(self) -> None:
        transform = wp.transform(
            wp.vec3(1.0, 2.0, 3.0),
            wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.5),
        )
        target_transform = wp.transform(
            wp.vec3(4.0, 2.0, 1.0),
            wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -0.25),
        )
        pose = wp.array([transform], dtype=wp.transform)
        target_pose = wp.array([target_transform], dtype=wp.transform)
        delta_pose = wp.empty_like(pose)

        pose_utils.compute_delta_pose(pose, target_pose, delta_pose)
        pose_utils.apply_delta_pose(pose, delta_pose, 1.0)
        wp.synchronize_device(self.device)

        expected_delta = wp.mul(
            wp.transform_inverse(transform),
            target_transform,
        )
        np.testing.assert_allclose(
            delta_pose.numpy()[0],
            np.asarray(expected_delta),
            atol=1.0e-5,
        )
        np.testing.assert_allclose(
            pose.numpy()[0],
            np.asarray(target_transform),
            atol=1.0e-5,
        )

    def test_history_keeps_latest_pose_at_active_end(self) -> None:
        motion_history = MotionRecorder(self.skeleton.reference_local_transforms, capacity=3)
        device_poses = []
        for frame in range(4):
            device_poses.append(
                wp.array(_pose(float(frame)), dtype=wp.transform)
            )
            motion_history.record(float(frame + 1), device_poses[-1])
        wp.synchronize_device(self.device)

        np.testing.assert_allclose(motion_history.delta_times, [3.0, 4.0, 0.0])
        np.testing.assert_allclose(
            motion_history.poses.numpy()[:, 0, 0],
            [1.0, 2.0, 3.0],
        )

    def test_history_resamples_variable_frame_deltas_at_model_rate(self) -> None:
        motion_history = MotionRecorder(self.skeleton.reference_local_transforms, capacity=16)
        timestamp = 0.0
        device_poses = []
        for delta_time in (0.020, 0.015, 0.030, 0.025, 0.020):
            timestamp += delta_time
            device_poses.append(
                wp.array(_pose(timestamp), dtype=wp.transform)
            )
            motion_history.record(delta_time, device_poses[-1])

        token = PoseToken(1)
        token.update_from_recorder(motion_history)
        wp.synchronize_device(self.device)

        expected = timestamp + np.arange(-3, 1, dtype=np.float32) / 30.0
        np.testing.assert_allclose(
            token.poses.numpy()[:, 0, 0],
            expected,
            atol=1.0e-6,
        )

    def test_animation_buffer_tracks_valid_frames_in_reusable_storage(self) -> None:
        frames = wp.array(
            np.stack([_pose(0.0), _pose(1.0), _pose(2.0), _pose(999.0)]),
            dtype=wp.transform,
        )
        output = AnimationBuffer(self.skeleton, 30.0, frames, num_frames=0)
        self.assertEqual(output.capacity, 4)
        self.assertEqual(output.duration, 0)
        sampled = wp.empty(1, dtype=wp.transform)
        with self.assertRaisesRegex(ValueError, "empty"):
            output.sample(0, sampled)
        for count in (3, 2, 1):
            output.num_frames = count
            self.assertEqual(output.poses.ptr, frames.ptr)
            self.assertAlmostEqual(output.duration, (count - 1) / 30)
            output.sample(10, sampled)
            np.testing.assert_array_equal(sampled.numpy(), _pose(float(count - 1)))
        output.num_frames = 3

        output.sample(1.0 / 60.0, sampled)
        wp.synchronize_device(self.device)

        self.assertEqual(output.num_frames, 3)
        self.assertTrue(output.poses.device.is_cuda)
        self.assertAlmostEqual(float(sampled.numpy()[0, 0]), 0.5, places=6)
        for invalid in (-1, output.capacity + 1):
            with self.assertRaises(ValueError):
                output.num_frames = invalid
        self.assertEqual(output.num_frames, 3)
        output.num_frames = 0
        with self.assertRaisesRegex(ValueError, "empty"):
            output.sample(0, sampled)


if __name__ == "__main__":
    unittest.main()
