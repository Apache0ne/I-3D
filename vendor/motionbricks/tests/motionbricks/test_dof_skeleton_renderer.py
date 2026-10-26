# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Debug FK must honor valid pose counts without changing the animated model."""

import unittest
from unittest.mock import Mock

import newton
import numpy as np
import warp as wp

from motionbricks.renderers.skeleton_renderer import DOFSkeletonRenderer


def _model():
    """A free root, a revolute arm, and a tip fixed one metre along the arm."""
    builder = newton.ModelBuilder()
    root = builder.add_link(mass=1, inertia=wp.mat33(np.eye(3)))
    arm = builder.add_link(mass=1, inertia=wp.mat33(np.eye(3)))
    tip = builder.add_link(mass=1, inertia=wp.mat33(np.eye(3)))
    joints = [builder.add_joint_free(root)]
    # Co-located joint origins are common on robots and must draw finite lines.
    joints.append(builder.add_joint_revolute(root, arm, axis=wp.vec3(0, 0, 1)))
    joints.append(builder.add_joint_fixed(
        arm, tip, parent_xform=wp.transform(wp.vec3(1, 0, 0), wp.quat_identity()),
    ))
    builder.add_articulation(joints)
    return builder.finalize(), root, tip


class DOFSkeletonRendererTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.previous_device = wp.get_device()
        wp.set_device("cpu")

    @classmethod
    def tearDownClass(cls):
        wp.set_device(cls.previous_device)

    def test_batch_fk_preserves_playback_grows_storage_and_hides_zero_poses(self):
        model, root, tip = _model()
        playback = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, playback)
        original_q = model.joint_q.numpy().copy()
        original_body_q = playback.body_q.numpy().copy()
        poses = np.full((4, model.joint_coord_count), np.nan, dtype=np.float32)
        poses[:2] = 0
        poses[0, :7] = [0, 0, 1, 0, 0, 0, 1]
        poses[1, :7] = [2, 3, 1, 0, 0, np.sqrt(0.5), np.sqrt(0.5)]
        poses[1, 7] = -np.pi / 2
        joint_q = wp.array(poses, dtype=wp.float32)
        placement = wp.transform(wp.vec3(10, 0, 0), wp.quat_from_axis_angle(wp.vec3(0, 0, 1), 0.5 * np.pi))
        renderer = DOFSkeletonRenderer(model, max_instances=1)
        viewer = Mock()
        colors = wp.array([wp.vec3(0.2, 0.75, 1.0), wp.vec3(1.0, 0.5, 0.2)], dtype=wp.vec3)
        renderer.draw_batch(viewer, joint_q, 2, placement, colors, "target")
        bodies = renderer.world_poses.numpy()
        np.testing.assert_allclose(bodies[:2, root, :3], [[10, 0, 1], [7, 2, 1]], atol=1e-6)
        np.testing.assert_allclose(bodies[:2, tip, :3], [[10, 1, 1], [7, 3, 1]], atol=1e-6)
        _, starts, ends, line_colors = viewer.log_lines.call_args.args
        self.assertTrue(np.isfinite(starts.numpy()).all())
        self.assertTrue(np.isfinite(ends.numpy()).all())
        # Each pose's bones use that pose's color.
        half = len(line_colors) // 2
        np.testing.assert_allclose(line_colors.numpy()[:half], np.tile([0.2, 0.75, 1.0], (half, 1)))
        np.testing.assert_allclose(line_colors.numpy()[half:], np.tile([1.0, 0.5, 0.2], (half, 1)))
        np.testing.assert_array_equal(model.joint_q.numpy(), original_q)
        np.testing.assert_array_equal(playback.body_q.numpy(), original_body_q)
        np.testing.assert_array_equal(joint_q.numpy(), poses)
        renderer.draw_batch(viewer, joint_q, 0, placement, colors, "target")
        self.assertTrue(viewer.log_lines.call_args.kwargs["hidden"])


if __name__ == "__main__":
    unittest.main()
