# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Skeleton previews draw each bone between a joint and its parent for every active pose."""

import unittest
from unittest.mock import Mock

import numpy as np
import warp as wp

from motionbricks.animation.skeleton import Skeleton
from motionbricks.renderers.skeleton_renderer import SkeletonRenderer


def _local(position, angle=0.0):
    return [*position, *wp.quat_from_axis_angle(wp.vec3(0, 0, 1), angle)]


class SkeletonRendererTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.previous_device = wp.get_device()
        wp.set_device("cpu")

    @classmethod
    def tearDownClass(cls):
        wp.set_device(cls.previous_device)

    def test_batch_draws_placed_bones_and_hides_zero_poses(self):
        skeleton = Skeleton(["Hips", "Spine", "Head"], [-1, 0, 1], np.tile(_local((0, 0, 0)), (3, 1)))
        poses = np.full((3, 3, 7), np.nan, dtype=np.float32)
        poses[0] = [_local((0, 0, 1)), _local((0, 0, 0.5)), _local((0, 0, 0.3))]
        poses[1] = [_local((2, 3, 1), 0.5 * np.pi), _local((0.5, 0, 0)), _local((0.3, 0, 0))]
        local_poses = wp.array(poses, dtype=wp.transform)
        placement = wp.transform(wp.vec3(10, 0, 0), wp.quat_from_axis_angle(wp.vec3(0, 0, 1), 0.5 * np.pi))
        colors = wp.array([wp.vec3(0.2, 0.75, 1.0), wp.vec3(1.0, 0.5, 0.2)], dtype=wp.vec3)
        renderer = SkeletonRenderer(skeleton, max_instances=1)
        viewer = Mock()

        renderer.draw_batch(viewer, local_poses, 2, placement, colors, "target")

        joints = [[[10, 0, 1], [10, 0, 1.5], [10, 0, 1.8]], [[7, 2, 1], [6.5, 2, 1], [6.2, 2, 1]]]
        np.testing.assert_allclose(renderer.world_poses.numpy()[:2, :3, :3], joints, atol=1e-6)
        _, starts, ends, line_colors = viewer.log_lines.call_args.args
        # Each bone draws 16 lines, one pyramid from the joint and one from its parent.
        self.assertEqual(starts.shape, (2 * 2 * 16,))
        starts = starts.numpy().reshape(2, 2, 16, 3)
        for pose in range(2):
            for bone, (parent, child) in enumerate(((0, 1), (1, 2))):
                np.testing.assert_allclose(starts[pose, bone, 0], joints[pose][child], atol=1e-6)
                np.testing.assert_allclose(starts[pose, bone, 8], joints[pose][parent], atol=1e-6)
        self.assertTrue(np.isfinite(ends.numpy()).all())
        pose_colors = np.repeat(colors.numpy()[:, None], 32, axis=1)
        np.testing.assert_allclose(line_colors.numpy().reshape(2, 32, 3), pose_colors)
        np.testing.assert_array_equal(local_poses.numpy(), poses)
        renderer.draw_batch(viewer, local_poses, 0, placement, colors, "target")
        self.assertTrue(viewer.log_lines.call_args.kwargs["hidden"])


if __name__ == "__main__":
    unittest.main()
