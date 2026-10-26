# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Foot contact buffers, outline builders, and the outline renderer."""

import unittest
from unittest.mock import Mock

import newton
import numpy as np
import warp as wp

from motionbricks.animation.animation_buffer import AnimationBuffer
from motionbricks.animation.mesh import SkeletalMesh, SkinnedMesh
from motionbricks.animation.skeleton import Skeleton
from motionbricks.io.utils import get_soma_human_model, get_unitree_g1_model
from motionbricks.model.soma.contract import SOMA_N1_FOOT_CONTACT_JOINTS
from motionbricks.model.unitree_g1.contract import (
    UNITREE_G1_N1_FOOT_CONTACT_BODIES,
    UNITREE_G1_N1_FOOT_CONTACT_OFFSETS,
)
from motionbricks.renderers.foot_contact_renderer import (
    ContactOutline,
    FootContactRenderer,
    body_sole_outlines,
    skeletal_mesh_sole_outlines,
)
from motionbricks.robotics.dof_animation_buffer import DOFAnimationBuffer


def _slab(forwards, sides, heights):
    """Return the vertices (forward, side, height) and triangles of a box sliced at each forward position."""
    vertices = np.array([[f, s, h] for f in forwards for s in sides for h in heights], dtype=np.float32)
    triangles = []
    for slice_index in range(len(forwards) - 1):
        a, b = 4 * slice_index, 4 * (slice_index + 1)
        for h in (0, 1):
            triangles += [a + h, b + h, b + 2 + h, a + h, b + 2 + h, a + 2 + h]
    return vertices, np.array(triangles, dtype=np.int32)


class FootContactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.previous_device = wp.get_device()
        wp.set_device("cpu")

    @classmethod
    def tearDownClass(cls):
        wp.set_device(cls.previous_device)

    def test_buffers_validate_their_contacts_and_blend_them_within_the_valid_frames(self):
        contacts = np.array([[0.0, 0.2, 0.4, 0.6], [1.0, 0.8, 0.6, 0.4], [0.5, 0.5, 0.5, 0.5]], dtype=np.float32)
        skeleton = Skeleton(["Root"], [-1], np.array([[0, 0, 0, 0, 0, 0, 1]], dtype=np.float32))
        buffers = (
            AnimationBuffer(
                skeleton, 30.0, wp.zeros((3, 1), dtype=wp.transform),
                foot_contacts=wp.array(contacts, dtype=wp.float32),
            ),
            DOFAnimationBuffer(
                ("hinge",), 30.0, wp.zeros((3, 8), dtype=wp.float32),
                foot_contacts=wp.array(contacts, dtype=wp.float32),
            ),
        )
        output = wp.empty(4, dtype=wp.float32)
        for buffer in buffers:
            with self.subTest(buffer=type(buffer).__name__):
                buffer.num_frames = 2
                buffer.sample_foot_contacts(0.5 / 30.0, output)
                np.testing.assert_allclose(output.numpy(), [0.5, 0.5, 0.5, 0.5], atol=1e-6)
                # Sampling stops at the last valid frame, not at the end of the storage.
                buffer.sample_foot_contacts(10.0, output)
                np.testing.assert_allclose(output.numpy(), contacts[1])

        with self.assertRaisesRegex(ValueError, "foot_contacts"):
            AnimationBuffer(
                skeleton, 30.0, wp.zeros((3, 1), dtype=wp.transform), foot_contacts=wp.zeros((2, 4), dtype=wp.float32),
            )
        with self.assertRaisesRegex(ValueError, "foot_contacts"):
            DOFAnimationBuffer(
                ("hinge",), 30.0, wp.zeros((3, 8), dtype=wp.float32), foot_contacts=wp.zeros((3, 3), dtype=wp.float32),
            )
        clip = DOFAnimationBuffer(("hinge",), 30.0, wp.zeros((3, 8), dtype=wp.float32))
        with self.assertRaisesRegex(ValueError, "Cannot sample foot contacts"):
            clip.sample_foot_contacts(0.0, output)

    def test_outlines_close_each_loop_and_show_contacts_above_the_threshold(self):
        square = np.array([[0, 0, 0], [0.1, 0, 0], [0.1, 0.1, 0], [0, 0.1, 0]], dtype=np.float32)
        outlines = [
            ContactOutline(0, 0, square),
            ContactOutline(1, 0, square - [0, 0, 0.1]),
            ContactOutline(2, 1, square),
            ContactOutline(3, 1, square),
        ]
        transforms = wp.array(
            [wp.transform_identity(), wp.transform(wp.vec3(1.0, 0.0, 0.5), wp.quat_identity())], dtype=wp.transform,
        )
        renderer = FootContactRenderer("outlines", outlines)
        self.assertEqual(renderer.threshold, FootContactRenderer.DEFAULT_THRESHOLD)
        viewer = Mock()

        renderer.draw(viewer, transforms, wp.array([0.0, 0.6, 0.3, 1.0], dtype=wp.float32))

        _, starts, ends, colors = viewer.log_lines.call_args.args
        starts, ends, colors = (array.numpy().reshape(4, 4, 3) for array in (starts, ends, colors))
        # Contacts at or below the threshold collapse their segments.
        np.testing.assert_allclose(ends[0], starts[0])
        np.testing.assert_allclose(ends[2], starts[2])
        # A drawn outline is closed and follows its transform.
        np.testing.assert_allclose(starts[3], square + [1.0, 0.0, 0.5], atol=1e-6)
        np.testing.assert_allclose(ends[3], np.roll(square, -1, axis=0) + [1.0, 0.0, 0.5], atol=1e-6)
        # Points below the floor rise to just above it.
        np.testing.assert_allclose(starts[1][:, 2], 0.003, atol=1e-6)
        np.testing.assert_allclose(colors[1], [[1.0, 0.4, 0.0]] * 4, atol=1e-6)
        np.testing.assert_allclose(colors[3], [[1.0, 0.0, 0.0]] * 4, atol=1e-6)
        self.assertIs(viewer.log_lines.call_args.kwargs["hidden"], False)

        # A negative threshold still hides a contact of 0.
        renderer.threshold = -1.0
        renderer.draw(viewer, transforms, wp.array([0.0, 0.05, -0.2, 1.0], dtype=wp.float32))
        starts, ends, _ = (array.numpy().reshape(4, 4, 3) for array in viewer.log_lines.call_args.args[1:])
        drawn = [bool(np.any(ends[contact] != starts[contact])) for contact in range(4)]
        self.assertEqual(drawn, [False, True, False, True])

        renderer.hide(viewer)
        renderer.hide(viewer)
        hidden = [call for call in viewer.log_lines.call_args_list if call.kwargs["hidden"]]
        self.assertEqual(len(hidden), 1)

    def test_body_outlines_surround_the_foot_mesh_and_grow_away_from_the_split(self):
        builder = newton.ModelBuilder()
        vertices, triangles = _slab(np.linspace(-0.05, 0.15, 9), (-0.04, 0.04), (-0.03, 0.02))
        for name in ("left_foot", "right_foot"):
            body = builder.add_body(label=name, mass=1.0)
            builder.add_shape_mesh(body, mesh=newton.Mesh(vertices, triangles, compute_inertia=False))
        model = builder.finalize(device="cpu")

        heel_and_toe = ((0.0, 0.0, 0.0), (0.1, 0.0, 0.0))
        outlines = body_sole_outlines(
            model, ("left_foot",) * 2 + ("right_foot",) * 2, heel_and_toe * 2, margin=0.008, rings=1,
        )

        self.assertEqual(
            [(outline.contact, outline.transform) for outline in outlines], [(0, 0), (1, 0), (2, 1), (3, 1)],
        )
        heel, toe = outlines[0].points, outlines[1].points
        # The outline grows 8 mm at its ends and sides. It splits halfway to the toe point, at x = 0.05.
        np.testing.assert_allclose([heel[:, 0].min(), heel[:, 0].max()], [-0.058, 0.05], atol=1e-6)
        np.testing.assert_allclose([toe[:, 0].min(), toe[:, 0].max()], [0.05, 0.158], atol=1e-6)
        np.testing.assert_allclose([heel[:, 1].min(), heel[:, 1].max()], [-0.048, 0.048], atol=1e-6)
        # It sits 4 mm above the sole.
        np.testing.assert_allclose(np.concatenate([heel, toe])[:, 2], -0.026, atol=1e-6)

        # The second ring runs 5 mm beyond the first.
        rings = body_sole_outlines(model, ("left_foot",) * 2, heel_and_toe, margin=0.01, rings=2, spacing=0.005)
        self.assertEqual([outline.contact for outline in rings], [0, 1, 0, 1])
        np.testing.assert_allclose(
            [rings[0].points[:, 0].min(), rings[2].points[:, 0].min()], [-0.06, -0.065], atol=1e-6,
        )

    def test_skinned_outlines_split_at_the_toe_joint_in_the_bind_pose(self):
        # The bind pose is Y up and faces +Z. Its joint frames are rotated, so outline points must change frame.
        positions = {"Hips": (0.0, 1.0, 0.0), "LeftFoot": (0.1, 0.08, -0.05), "LeftToeBase": (0.1, 0.02, 0.05)}
        skeleton = Skeleton(list(positions), [-1, 0, 1], np.tile([0, 0, 0, 0, 0, 0, 1], (3, 1)).astype(np.float32))
        bind = [wp.transform(wp.vec3(*position), wp.quat_rpy(0.2, -0.3, 0.4)) for position in positions.values()]
        inverse_bind = [wp.transform_inverse(transform) for transform in bind]
        foot, foot_triangles = _slab(np.linspace(-0.15, 0.10, 6), (0.05, 0.15), (0.0, 0.06))
        leg, leg_triangles = _slab((-0.3, 0.3), (0.0, 0.2), (0.5, 1.0))
        # Slab coordinates (forward, side, height) become (Z, X, Y) in the bind pose.
        points = np.concatenate([foot, leg])[:, [1, 2, 0]]
        # The toe joint moves the front of the foot. The hips move the leg, which no outline includes.
        joints = np.concatenate([np.where(foot[:, 0] < 0.025, 1, 2), np.zeros(len(leg), dtype=int)]).astype(np.int32)
        # The foot part stores one influence per vertex. The leg part stores two.
        foot_mesh = SkinnedMesh(
            points[:len(foot)], foot_triangles, joints[:len(foot)], np.ones(len(foot), dtype=np.float32),
        )
        leg_joints = np.tile([0, 2], len(leg)).astype(np.int32)
        leg_weights = np.tile([0.7, 0.3], len(leg)).astype(np.float32)
        leg_mesh = SkinnedMesh(points[len(foot):], leg_triangles, leg_joints, leg_weights)
        skinned = SkeletalMesh([foot_mesh, leg_mesh], skeleton, inverse_bind)

        outlines = skeletal_mesh_sole_outlines(skinned, ["LeftFoot", "LeftToeBase"], margin=0.008, rings=1)

        self.assertEqual([(outline.contact, outline.transform) for outline in outlines], [(0, 1), (1, 2)])
        renderer = FootContactRenderer("foot", outlines)
        viewer = Mock()
        # The pose stands 1 m above the floor, so no outline point is raised to it.
        lift = wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity())
        lifted = wp.array([wp.mul(lift, joint) for joint in bind], dtype=wp.transform)
        renderer.draw(viewer, lifted, wp.ones(4, dtype=wp.float32))
        # World joint transforms place the local outlines back around the original foot.
        starts = viewer.log_lines.call_args.args[1].numpy()
        heel, toe = np.split(starts, [len(outlines[0].points)])
        np.testing.assert_allclose([heel[:, 2].min(), heel[:, 2].max()], [0.842, 1.05], atol=1e-6)
        np.testing.assert_allclose([toe[:, 2].min(), toe[:, 2].max()], [1.05, 1.108], atol=1e-6)
        np.testing.assert_allclose([heel[:, 0].min(), heel[:, 0].max()], [0.042, 0.158], atol=1e-6)
        np.testing.assert_allclose(np.concatenate([heel, toe])[:, 1], 0.004, atol=1e-6)

    def test_contract_contact_points_outline_both_shipped_rigs(self):
        soma = skeletal_mesh_sole_outlines(get_soma_human_model(), SOMA_N1_FOOT_CONTACT_JOINTS, rings=1)
        g1 = body_sole_outlines(
            get_unitree_g1_model(), UNITREE_G1_N1_FOOT_CONTACT_BODIES, UNITREE_G1_N1_FOOT_CONTACT_OFFSETS, rings=1,
        )
        for outlines in (soma, g1):
            self.assertEqual([outline.contact for outline in outlines], [0, 1, 2, 3])
        # G1 splits halfway between its ankle roll joint and the toe point 14 cm ahead.
        np.testing.assert_allclose([g1[0].points[:, 0].max(), g1[1].points[:, 0].min()], [0.07, 0.07], atol=1e-6)

    def test_contacts_and_samples_stay_on_the_pose_device(self):
        if not wp.is_cuda_available():
            self.skipTest("A second device needs CUDA.")
        skeleton = Skeleton(["Root"], [-1], np.array([[0, 0, 0, 0, 0, 0, 1]], dtype=np.float32))
        on_gpu = wp.zeros((3, 4), dtype=wp.float32, device="cuda:0")
        with self.assertRaisesRegex(ValueError, "device"):
            AnimationBuffer(skeleton, 30.0, wp.zeros((3, 1), dtype=wp.transform), foot_contacts=on_gpu)
        with self.assertRaisesRegex(ValueError, "device"):
            DOFAnimationBuffer(("hinge",), 30.0, wp.zeros((3, 8), dtype=wp.float32), foot_contacts=on_gpu)
        contacts = wp.zeros((3, 4), dtype=wp.float32)
        buffer = DOFAnimationBuffer(("hinge",), 30.0, wp.zeros((3, 8), dtype=wp.float32), foot_contacts=contacts)
        with self.assertRaisesRegex(ValueError, "device"):
            buffer.sample_foot_contacts(0.0, wp.empty(4, dtype=wp.float32, device="cuda:0"))


if __name__ == "__main__":
    unittest.main()
