# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

from motionbricks.runtime.pivot import Axis, ConeSide, ConstraintType, OperationOrder, Pivot


def _character(desired, position=(0.0, 0.0, 0.0)):
    start = wp.vec3(*position)
    end = start + wp.vec3(*desired) if math.isfinite(math.hypot(*desired)) else start
    return SimpleNamespace(position=position, desired_velocity=desired, probe_segment=(start, end))


class PivotAlignmentTests(unittest.TestCase):
    def test_local_z_alignment_preserves_position_and_authored_axis(self):
        parent = wp.transform(wp.vec3(3.0, -2.0, 1.0), wp.quat_rpy(0.3, -0.4, 0.6))
        local = wp.transform(wp.vec3(0.5, 0.1, 0.2), wp.quat_rpy(-0.2, 0.1, 0.5))
        upright = wp.transform(wp.vec3(1.0, 2.0, 3.0), wp.quat_rpy(0.0, 0.0, 0.7))
        for world in (upright, wp.mul(parent, local)):
            # A world-space approach with a known heading in the authored frame.
            # Its local-Z component should not affect the chosen turn.
            desired = wp.quat_rotate(world.q, wp.vec3(3.0, 4.0, 5.0))
            for motion in ((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.6, 0.8)):
                with self.subTest(world=world, motion=motion):
                    pivot = Pivot(motion_direction=motion, rotation_axis=Axis.Z)
                    aligned = pivot.aligned_transform(world, world, _character(desired))
                    result_direction = np.asarray(wp.quat_rotate(
                        wp.quat_inverse(world.q), wp.quat_rotate(aligned.q, wp.vec3(*motion)),
                    ))
                    np.testing.assert_allclose(
                        result_direction[:2] / np.linalg.norm(result_direction[:2]), [0.6, 0.8], atol=1e-6,
                    )
                    np.testing.assert_allclose(result_direction[2], motion[2], atol=1e-6, rtol=0.0)
                    np.testing.assert_array_equal(aligned.p, world.p)
                    # The relative rotation must use local Z only, even with a tilted parent.
                    delta = wp.mul(wp.quat_inverse(world.q), aligned.q)
                    np.testing.assert_allclose(np.asarray(delta)[:2], [0.0, 0.0], atol=1e-6)

    def test_no_rotation_or_undefined_heading_keeps_the_current_transform(self):
        for rotation in (wp.quat_rpy(0.0, 0.0, 0.7), wp.quat_rpy(0.4, -0.6, 0.7)):
            world = wp.transform(wp.vec3(3.0, 2.0, 1.0), rotation)
            for axis, motion, local_desired in (
                (Axis.NONE, (0, 1, 0), (1, 0, 0)),
                (Axis.X, (0, 1, 0), (1, 0, 0)),
                (Axis.X, (1, 0, 0), (0, 1, 0)),
                (Axis.Y, (1, 0, 0), (0, 1, 0)),
                (Axis.Y, (0, 1, 0), (1, 0, 0)),
                (Axis.Z, (0, 1, 0), (0, 0, 0)),
                (Axis.Z, (0, 1, 0), (0, 0, 1)),
                (Axis.Z, (0, 0, 1), (1, 0, 0)),
                (Axis.Z, (0, 1, 0), (float("nan"), 0, 0)),
            ):
                with self.subTest(rotation=rotation, axis=axis, motion=motion, desired=local_desired):
                    pivot = Pivot(motion_direction=motion, rotation_axis=axis)
                    desired = wp.quat_rotate(rotation, wp.vec3(*local_desired))
                    np.testing.assert_array_equal(pivot.aligned_transform(world, world, _character(desired)), world)
            np.testing.assert_array_equal(
                Pivot().aligned_transform(wp.transform_identity(), world, _character((1, 0, 0))), world,
            )

    def test_line_projection_uses_a_finite_approach_and_fixed_limits(self):
        reference = wp.transform(wp.vec3(2.0, 3.0, 4.0), wp.quat_rpy(0.3, -0.4, 0.5))
        current = wp.transform(wp.vec3(7.0, 8.0, 9.0), wp.quat_rpy(0.2, 0.1, 0.8))
        pivot = Pivot(
            translation_axis=Axis.X, translation_constraint=ConstraintType.LINE_PROJECTION,
            line_min_offset=-1.0, line_max_offset=1.0,
        )
        for position, velocity, expected_offset in (
            ((0.25, -2, 0), (0, 1, 0), 0.25),
            ((0, -2, 0), (0.5, 1, 0), 0.5),  # Stops short of the line.
            ((0, -2, 0), (1, 2, 0), 1.0),  # Same direction, enough speed to reach it.
            ((0, -2, 0), (2, 1, 0), 1.0),
            ((2, -2, 0), (-2, 1, 0), 0.0),
            ((0.25, 2, 0), (0, 1, 0), 0.25),
            ((0.25, -2, 1), (0, 1, 0), 0.25),
            ((0.25, -2, 0), (0, 0, 0), 0.25),
            ((0.25, -2, 0), (float("nan"), 0, 0), 0.25),
        ):
            with self.subTest(position=position, velocity=velocity):
                character = _character(
                    wp.quat_rotate(reference.q, wp.vec3(*velocity)),
                    wp.transform_point(reference, wp.vec3(*position)),
                )
                result = pivot.aligned_transform(reference, current, character)
                expected = wp.transform_point(reference, wp.vec3(expected_offset, 0.0, 0.0))
                np.testing.assert_allclose(result.p, expected, atol=1e-6)
                np.testing.assert_array_equal(result.q, current.q)
                current = result  # Limits must not follow previous adjustments.

        # The caller's interaction probe can extend beyond one second of motion.
        character = _character((0.25, 0.5, 0.0), (0.0, -2.0, 0.0))
        character.probe_segment = (wp.vec3(0.0, -2.0, 0.0), wp.vec3(1.0, 0.0, 0.0))
        result = pivot.aligned_transform(wp.transform_identity(), wp.transform_identity(), character)
        np.testing.assert_allclose(result.p, [1.0, 0.0, 0.0], atol=1e-6)

        # Parallel segments can have several equally close points.
        result = pivot.aligned_transform(reference, current, _character(
            wp.quat_rotate(reference.q, wp.vec3(1.0, 0.0, 0.0)),
            wp.transform_point(reference, wp.vec3(0.25, -2.0, 0.0)),
        ))
        local_point = wp.transform_point(wp.transform_inverse(reference), result.p)
        self.assertGreaterEqual(local_point[0], 0.25 - 1e-6)
        self.assertLessEqual(local_point[0], 1.0 + 1e-6)
        np.testing.assert_allclose(np.asarray(local_point)[1:], [0.0, 0.0], atol=1e-6)

        pivot.line_min_offset = pivot.line_max_offset = 0.4
        result = pivot.aligned_transform(reference, current, _character((1, 0, 0)))
        np.testing.assert_allclose(result.p, wp.transform_point(reference, wp.vec3(0.4, 0.0, 0.0)), atol=1e-6)

        pivot.translation_constraint = ConstraintType.UNCONSTRAINED
        result = pivot.aligned_transform(reference, current, _character(
            wp.quat_rotate(reference.q, wp.vec3(2.0, 1.0, 0.0)),
            wp.transform_point(reference, wp.vec3(0.0, -2.0, 0.0)),
        ))
        np.testing.assert_allclose(result.p, wp.transform_point(reference, wp.vec3(4.0, 0.0, 0.0)), atol=1e-5)

    def test_operation_order_projects_onto_the_authored_or_rotated_line_without_drift(self):
        for reference in (
            wp.transform_identity(),
            wp.transform(wp.vec3(2.0, 3.0, 4.0), wp.quat_rpy(0.3, -0.4, 0.5)),
        ):
            # In the authored frame, this approach crosses y = 0 at x = -0.5.
            character = _character(
                wp.quat_rotate(reference.q, wp.vec3(2.0, 2.0, 0.0)),
                wp.transform_point(reference, wp.vec3(-1.5, -1.0, 0.0)),
            )
            for constraint, rotated_position, heading in (
                (ConstraintType.UNCONSTRAINED, (-0.25, 0.25, 0.0), (math.sqrt(0.5), math.sqrt(0.5), 0.0)),
                (ConstraintType.CONE, (-0.3169873, 0.1830127, 0.0), (0.5, math.sqrt(0.75), 0.0)),
            ):
                for order, expected_position in (
                    (OperationOrder.TRANSLATION_THEN_ROTATION, (-0.5, 0.0, 0.0)),
                    (OperationOrder.ROTATION_THEN_TRANSLATION, rotated_position),
                ):
                    with self.subTest(reference=reference, constraint=constraint, order=order):
                        pivot = Pivot(
                            operation_order=order,
                            translation_axis=Axis.X,
                            translation_constraint=ConstraintType.LINE_PROJECTION,
                            line_min_offset=-1.0, line_max_offset=1.0,
                            rotation_axis=Axis.Z, rotation_constraint=constraint,
                            cone_half_angle_degrees=30.0,
                        )
                        current = wp.transform(wp.vec3(7.0, 8.0, 9.0), wp.quat_rpy(0.2, 0.1, 0.8))
                        for _ in range(3):
                            current = pivot.aligned_transform(reference, current, character)
                            np.testing.assert_allclose(
                                current.p, wp.transform_point(reference, wp.vec3(*expected_position)), atol=1e-6,
                            )
                            np.testing.assert_allclose(
                                wp.quat_rotate(current.q, wp.vec3(0, 1, 0)),
                                wp.quat_rotate(reference.q, wp.vec3(*heading)), atol=1e-6,
                            )

    def test_rotation_first_still_translates_when_rotation_is_disabled_or_heading_is_undefined(self):
        reference = wp.transform_identity()
        current = wp.transform(wp.vec3(7.0, 8.0, 9.0), wp.quat_rpy(0.0, 0.0, -math.pi / 2))
        character = _character((0.0, 0.0, 0.0), (0.4, -0.25, 0.0))
        for axis, expected in (
            (Axis.NONE, (0.4, 0.0, 0.0)),
            (Axis.Z, (0.0, -0.25, 0.0)),
        ):
            with self.subTest(axis=axis):
                pivot = Pivot(
                    operation_order=OperationOrder.ROTATION_THEN_TRANSLATION,
                    translation_axis=Axis.X,
                    translation_constraint=ConstraintType.LINE_PROJECTION,
                    rotation_axis=axis,
                )
                result = pivot.aligned_transform(reference, current, character)
                np.testing.assert_allclose(result.p, expected, atol=1e-6)
                np.testing.assert_array_equal(result.q, current.q)

    def test_cone_limits_forward_mirrored_and_both_without_drifting(self):
        reference = wp.transform(wp.vec3(2.0, 3.0, 4.0), wp.quat_rpy(0.3, -0.4, 0.5))
        for axis, motion, sideways, normal in (
            (Axis.X, wp.vec3(0, 1, 0), wp.vec3(0, 0, 1), wp.vec3(1, 0, 0)),
            (Axis.Y, wp.vec3(0, 0, 1), wp.vec3(1, 0, 0), wp.vec3(0, 1, 0)),
            (Axis.Z, wp.vec3(1, 0, 0), wp.vec3(0, 1, 0), wp.vec3(0, 0, 1)),
        ):
            current = wp.transform(wp.vec3(7.0, 8.0, 9.0), reference.q)
            pivot = Pivot(motion_direction=tuple(motion), rotation_axis=axis, rotation_constraint=ConstraintType.CONE)
            for side, half_angle, approach, expected in (
                (ConeSide.FORWARD, 45, 20, 20),
                (ConeSide.FORWARD, 45, 100, 45),
                (ConeSide.FORWARD, 45, -100, -45),
                (ConeSide.MIRRORED, 45, 170, 170),
                (ConeSide.MIRRORED, 45, 10, 135),
                (ConeSide.MIRRORED, 45, -10, -135),
                (ConeSide.BOTH, 45, 100, 135),
                (ConeSide.BOTH, 45, -100, -135),
                (ConeSide.BOTH, 45, 70, 45),
                (ConeSide.BOTH, 0, 179, 180),
                (ConeSide.FORWARD, 180, -175, -175),
            ):
                with self.subTest(axis=axis, side=side, half_angle=half_angle, approach=approach):
                    pivot.cone_side, pivot.cone_half_angle_degrees = side, half_angle
                    approach, expected = math.radians(approach), math.radians(expected)
                    desired = wp.quat_rotate(
                        reference.q, math.cos(approach) * motion + math.sin(approach) * sideways + 0.4 * normal,
                    )
                    result = pivot.aligned_transform(reference, current, _character(desired))
                    np.testing.assert_allclose(
                        wp.quat_rotate(result.q, motion),
                        wp.quat_rotate(reference.q, math.cos(expected) * motion + math.sin(expected) * sideways),
                        atol=1e-6,
                    )
                    np.testing.assert_allclose(
                        wp.quat_rotate(result.q, normal), wp.quat_rotate(reference.q, normal), atol=1e-6,
                    )
                    np.testing.assert_array_equal(result.p, current.p)
                    current = result


if __name__ == "__main__":
    unittest.main()
