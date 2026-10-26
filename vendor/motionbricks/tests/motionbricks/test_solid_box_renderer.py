# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import Mock

import numpy as np
import warp as wp

from motionbricks.renderers.solid_box_renderer import SolidBoxRenderer


class SolidBoxRendererTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        wp.init()
        cls.previous_device = wp.get_device()
        wp.set_device("cpu")

    @classmethod
    def tearDownClass(cls) -> None:
        wp.set_device(cls.previous_device)

    def test_box_picking_uses_rotation_full_dimensions_and_positive_ray_distances(self):
        size = wp.vec3(2, 4, 6)
        for rotation, origin, direction, limit, expected in (
            (0.0, (3, 4, 10), (0, 0, -1), 100, 2),
            (0.0, (3, 4, 10), (0, 0, 1), 100, 100),
            (0.0, (5, 4, 10), (0, 0, -1), 100, 100),
            (0.0, (3, 4, 10), (0, 0, -1), 1, 1),
            (0.0, (3, 4, 5), (0, 1, 0), 100, 2),
            (wp.pi / 2, (10, 4, 5), (-1, 0, 0), 100, 5),
            (wp.pi / 2, (3, 5.5, 10), (0, 0, -1), 100, 100),
        ):
            with self.subTest(rotation=rotation, origin=origin, direction=direction, limit=limit):
                transform = wp.transform(wp.vec3(3, 4, 5), wp.quat_rpy(0.0, 0.0, rotation))
                distance = SolidBoxRenderer.raycast(transform, size, wp.vec3(*origin), wp.vec3(*direction), limit)
                self.assertAlmostEqual(distance, expected, places=5)

    def test_drawn_box_matches_the_pick_volume(self) -> None:
        renderer = SolidBoxRenderer(device="cpu")
        viewer = Mock()
        size = wp.vec3(2.0, 3.0, 4.0)

        renderer.draw_batch(viewer, [wp.transform_identity()], [size], [wp.vec3(0.2, 0.4, 0.8)], "scene")

        half_extents = viewer.log_geo.call_args.args[2]
        scales = viewer.log_instances.call_args.args[3].numpy()
        np.testing.assert_allclose(2.0 * np.asarray(half_extents) * scales, [size])


if __name__ == "__main__":
    unittest.main()
