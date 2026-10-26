# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

from motionbricks.animation.skeleton import Skeleton
from motionbricks.animation.animation_buffer import AnimationBuffer
from motionbricks.model.soma.buffers import PoseToken
from motionbricks.runtime.motion_player import GeneratedMotionPlayer
from motionbricks.runtime.motion_recorder import MotionRecorder


def _pose(x: float = 0.0) -> np.ndarray:
    return np.array([x, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)


class GeneratedMotionPlayerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        wp.init()
        cls.previous_device = wp.get_device()
        wp.set_device("cpu")

    @classmethod
    def tearDownClass(cls) -> None:
        wp.set_device(cls.previous_device)

    def test_transition_deadline_does_not_limit_sampling_to_trimmed_frames(self) -> None:
        player = GeneratedMotionPlayer.__new__(GeneratedMotionPlayer)
        player.engine = SimpleNamespace(stream=None)
        skeleton = Skeleton(["Hips"], [-1], np.stack([_pose()]))
        contacts = np.repeat(np.arange(8, dtype=np.float32)[:, None] / 10.0, 4, axis=1)
        player.output = AnimationBuffer(
            skeleton, 30.0, wp.empty((8, skeleton.num_joints), dtype=wp.transform),
            num_frames=0, foot_contacts=wp.array(contacts, dtype=wp.float32),
        )
        frames = np.stack([_pose(float(x)) for x in (0, 1, 2, 3, 4, 5, 999, 999)])[:, None]
        player.output.poses.assign(frames)
        player.output.num_frames = 6
        destination_pose = wp.empty(1, dtype=wp.transform)
        destination_contacts = wp.empty(4, dtype=wp.float32)

        for early_frames, last_frame in ((0, 5), (1, 4), (2, 3), (5, 0), (20, 0)):
            with self.subTest(early_frames=early_frames):
                player.transition_early_by_frames = early_frames
                player.playback_time = 0.0
                self.assertAlmostEqual(player.time_to_transition, last_frame / 30.0)

                player.sample(4.5 / 30.0, destination_pose, destination_contacts)
                np.testing.assert_allclose(destination_pose.numpy()[0], _pose(4.5))
                np.testing.assert_allclose(destination_contacts.numpy(), 0.45, atol=1e-6)
                self.assertAlmostEqual(player.playback_time, 4.5 / 30.0)
                self.assertAlmostEqual(player.time_to_transition, max(0.0, (last_frame - 4.5) / 30.0))

                # Sampling is still bounded by valid data, never unused buffer capacity.
                player.sample(10.0, destination_pose, destination_contacts)
                np.testing.assert_allclose(destination_pose.numpy()[0], frames[5, 0])
                np.testing.assert_allclose(destination_contacts.numpy(), 0.5, atol=1e-6)
                self.assertEqual(player.time_to_transition, 0.0)

    def test_seed_history_uses_supplied_pose_and_refreshes_context(self) -> None:
        player = GeneratedMotionPlayer.__new__(GeneratedMotionPlayer)
        player.engine = SimpleNamespace(stream=None)
        skeleton = Skeleton(["Hips"], [-1], np.stack([_pose()]))
        player.motion_history = MotionRecorder(skeleton.reference_local_transforms, capacity=3)
        player.context = PoseToken(skeleton.num_joints)
        player.record_pose(1.0 / 30.0, wp.array(np.stack([_pose(1.0)]), dtype=wp.transform))
        player.refresh_context()
        seed = np.stack([_pose(2.0)])

        player.seed_history_from_pose(wp.array(seed, dtype=wp.transform))

        np.testing.assert_array_equal(player.motion_history.poses.numpy(), np.repeat(seed[None], 3, axis=0))
        np.testing.assert_array_equal(player.motion_history.delta_times, np.zeros(3, dtype=np.float32))
        np.testing.assert_array_equal(player.context.poses.numpy(), np.repeat(seed[None], 4, axis=0))


if __name__ == "__main__":
    unittest.main()
