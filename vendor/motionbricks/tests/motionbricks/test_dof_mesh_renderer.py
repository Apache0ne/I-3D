# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Robot previews and picking must share geometry and leave playback untouched."""

import unittest
from unittest.mock import Mock

import newton
import numpy as np
import warp as wp

from motionbricks.renderers.dof_mesh_renderer import DOFMeshRenderer


class DOFMeshRendererTests(unittest.TestCase):
    def test_fk_mesh_scale_and_picking_leave_playback_untouched(self):
        with wp.ScopedDevice("cpu"):
            builder = newton.ModelBuilder()
            root = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
            arm = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
            builder.add_articulation([
                builder.add_joint_free(root),
                builder.add_joint_revolute(
                    root, arm, axis=wp.vec3(0, 0, 1),
                    parent_xform=wp.transform(0, 2, 0, 0, 0, 0, 1),
                ),
            ])
            mesh = newton.Mesh(
                np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
                np.array([0, 1, 2], dtype=np.int32), compute_inertia=False,
            )
            builder.add_shape_mesh(
                arm, mesh=mesh, scale=(2, 3, 1), xform=wp.transform(1, 0, 0, 0, 0, 0, 1),
            )
            builder.add_shape_box(root, xform=wp.transform(0, 0, 2, 0, 0, 0, 1))
            # An invisible collision shape sits closer to the picking rays.
            builder.add_shape_box(
                root, xform=wp.transform(0, 0, 3, 0, 0, 0, 1), hx=3, hy=3, hz=0.2,
                cfg=newton.ModelBuilder.ShapeConfig(is_visible=False),
            )
            model = builder.finalize()
            playback = model.state()
            newton.eval_fk(model, model.joint_q, model.joint_qd, playback)
            original_q = model.joint_q.numpy().copy()
            original_bodies = playback.body_q.numpy().copy()
            original_scales = model.shape_scale.numpy().copy()
            pose = wp.array([3, 4, 1, 0, 0, np.sqrt(0.5), np.sqrt(0.5), -np.pi / 2], dtype=wp.float32)
            renderer = DOFMeshRenderer(model)
            viewer = Mock()

            # Mesh instances carry the mesh scale. Box sizes live in the shared geometry.
            renderer.draw_pose(viewer, pose, wp.transform_identity(), wp.vec3(0.2, 0.8, 0.4), "start")
            scales = [call.args[3].numpy() for call in viewer.log_instances.call_args_list]
            np.testing.assert_allclose(scales, [[[2, 3, 1]], [[1, 1, 1]]])

            placement = wp.transform(wp.vec3(10, -1, 2), wp.quat_rpy(0.2, -0.3, 0.7))
            for origin, limit, expected in (
                ((2.4, 4.6, 5), 20.0, 4.0),  # Scaled, articulated triangle.
                ((3, 4, 10), 20.0, 6.5),     # Visible box, ignoring the nearer collision shape.
                ((3.8, 6.7, 5), 20.0, 20.0),  # Inside the triangle's bounds but outside its surface.
                ((2.4, 4.6, 5), 2.0, 2.0),  # Hit beyond the requested distance.
            ):
                with self.subTest(origin=origin, limit=limit):
                    distance = renderer.raycast_pose(
                        pose, placement, wp.transform_point(placement, wp.vec3(*origin)),
                        wp.transform_vector(placement, wp.vec3(0, 0, -1)), limit,
                    )
                    self.assertAlmostEqual(distance, expected, places=5)

            np.testing.assert_array_equal(model.joint_q.numpy(), original_q)
            np.testing.assert_array_equal(playback.body_q.numpy(), original_bodies)
            np.testing.assert_array_equal(model.shape_scale.numpy(), original_scales)


if __name__ == "__main__":
    unittest.main()
