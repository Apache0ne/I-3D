# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Indexed pose/DOF blending on synthetic data, independent of models and clips."""

import itertools
import unittest

import numpy as np
import warp as wp

from motionbricks.runtime.blend_spaces import (
    BlendSample,
    BlendSpace,
    BlendSpaceType,
    directional_2d_dof_blend,
    directional_2d_pose_blend,
    linear_1d_dof_blend,
    linear_1d_pose_blend,
    prepare_directional_poses,
)


def _space(space_type, values, *, data_indices=None, device=None):
    space = BlendSpace(space_type, device=device)
    indices = range(len(values)) if data_indices is None else data_indices
    for value, index in zip(values, indices):
        space.add_target(wp.vec2(*value), index)
    space.compile()
    return space


def _rotation(degrees):
    half_angle = np.deg2rad(degrees) * 0.5
    return np.array([0, 0, np.sin(half_angle), np.cos(half_angle)], dtype=np.float32)


def _sources(frame_count=3):
    # Unregistered entries deliberately have invalid data; zero weights must skip them.
    poses = np.full((5, frame_count, 2, 7), np.nan, dtype=np.float32)
    dofs = np.full((5, frame_count, 9), np.nan, dtype=np.float32)
    for target, position, angle, hinges in ((1, 2, 0, (-2.8, 1.2)), (4, 10, 120, (2.8, -0.4))):
        for frame in range(frame_count):
            for joint in range(2):
                poses[target, frame, joint, :3] = [position + frame + joint, 2 * frame, 1 + joint]
                poses[target, frame, joint, 3:] = _rotation(angle + frame * 10 + joint * 20)
            dofs[target, frame, :7] = poses[target, frame, 0]
            dofs[target, frame, 7:] = np.asarray(hinges) + frame * 0.1
    return poses, dofs


def _sample(indices, weights):
    sample = BlendSample()
    sample.data_indices = wp.array(indices, dtype=wp.int32)
    sample.indices = wp.vec3i(*(list(range(len(indices))) + [0] * (3 - len(indices))))
    sample.weights = wp.vec3(*(list(weights) + [0.0] * (3 - len(weights))))
    sample.uniform_weight = 0.0
    return sample


def _assert_rotations(actual, expected):
    np.testing.assert_allclose(np.linalg.norm(actual, axis=-1), 1, atol=2e-6)
    np.testing.assert_allclose(np.abs(np.sum(actual * expected, axis=-1)), 1, atol=2e-6)


class AnimationBlendingTests(unittest.TestCase):
    device = "cpu"

    @classmethod
    def setUpClass(cls):
        wp.init()
        if cls.device == "cuda:0" and not wp.is_cuda_available():
            raise unittest.SkipTest("CUDA is not available.")
        cls.previous_device = wp.get_device()
        wp.set_device(cls.device)

    @classmethod
    def tearDownClass(cls):
        wp.set_device(cls.previous_device)

    def test_same_linear_weights_blend_pose_and_dof_windows_with_slerp(self):
        poses, dofs = _sources()
        pose_data = wp.array(poses, dtype=wp.transform)
        dof_data = wp.array(dofs, dtype=wp.float32)
        # Registration order is deliberately different from the parameter order.
        space = _space(BlendSpaceType.LINEAR_1D, ((1, 0), (0, 0), (2, 0)), data_indices=(4, 1, 3))
        pose_output = wp.array(np.full((5, 2, 7), 99, dtype=np.float32), dtype=wp.transform)
        dof_output = wp.full((5, 9), 99.0, dtype=wp.float32)

        for blend in (0.25, 0.0, 0.75, 1.0):
            sample = space.evaluate(wp.vec2(blend, 0))
            linear_1d_pose_blend(pose_data, sample, pose_output)
            linear_1d_dof_blend(dof_data, sample, dof_output)
            expected_poses = poses[1] * (1 - blend) + poses[4] * blend
            for frame in range(3):
                for joint in range(2):
                    expected_poses[frame, joint, 3:] = _rotation(blend * 120 + frame * 10 + joint * 20)
            actual_poses = pose_output.numpy()
            actual_dofs = dof_output.numpy()
            np.testing.assert_allclose(actual_poses[:3, :, :3], expected_poses[:, :, :3], atol=1e-6)
            _assert_rotations(actual_poses[:3, :, 3:], expected_poses[:, :, 3:])
            np.testing.assert_allclose(actual_dofs[:3, :3], expected_poses[:, 0, :3], atol=1e-6)
            _assert_rotations(actual_dofs[:3, 3:7], expected_poses[:, 0, 3:])
            expected_hinges = dofs[1, :, 7:] * (1 - blend) + dofs[4, :, 7:] * blend
            np.testing.assert_allclose(actual_dofs[:3, 7:], expected_hinges, atol=1e-6)
            np.testing.assert_array_equal(actual_poses[3:], 99)
            np.testing.assert_array_equal(actual_dofs[3:], 99)
        np.testing.assert_array_equal(pose_data.numpy(), poses)
        np.testing.assert_array_equal(dof_data.numpy(), dofs)

    def test_additive_blending_preserves_quaternion_accumulation_and_reference_fallback(self):
        poses, dofs = _sources()
        poses[4, ..., 3:] *= -1
        dofs[4, :, 3:7] *= -1
        pose_reference = np.zeros((2, 7), dtype=np.float32)
        pose_reference[:, 0] = 4
        pose_reference[:, 3:] = _rotation(40)
        dof_reference = np.concatenate([pose_reference[0], [0.3, -0.5]]).astype(np.float32)
        pose_output = wp.array(np.full((5, 2, 7), 99, dtype=np.float32), dtype=wp.transform)
        dof_output = wp.full((5, 9), 99.0, dtype=wp.float32)

        prepared = wp.array(poses, dtype=wp.transform)
        reference = wp.array(pose_reference, dtype=wp.transform)
        prepare_directional_poses(prepared, reference)
        for first, second in ((0.75, 0.25), (0.25, 0.25), (0.0, 0.0)):
            sample = _sample([1, 3, 4], [first, 0, second])
            directional_2d_pose_blend(prepared, sample, reference, pose_output)
            directional_2d_dof_blend(wp.array(dofs, dtype=wp.float32), sample,
                                wp.array(dof_reference, dtype=wp.float32), dof_output)

            missing = 1 - first - second
            expected = poses[1] * first + poses[4] * second + pose_reference * missing
            rotations = poses[1, ..., 3:] * first - poses[4, ..., 3:] * second + pose_reference[:, 3:] * missing
            rotations /= np.linalg.norm(rotations, axis=-1, keepdims=True)
            actual_poses = pose_output.numpy()
            actual_dofs = dof_output.numpy()
            np.testing.assert_allclose(actual_poses[:3, :, :3], expected[..., :3], atol=1e-6)
            _assert_rotations(actual_poses[:3, :, 3:], rotations)
            np.testing.assert_allclose(actual_dofs[:3, :3], expected[:, 0, :3], atol=1e-6)
            _assert_rotations(actual_dofs[:3, 3:7], rotations[:, 0])
            expected_hinges = dofs[1, :, 7:] * first + dofs[4, :, 7:] * second + dof_reference[7:] * missing
            np.testing.assert_allclose(actual_dofs[:3, 7:], expected_hinges, atol=1e-6)
            np.testing.assert_array_equal(actual_poses[3:], 99)
            np.testing.assert_array_equal(actual_dofs[3:], 99)

        # Three far apart root rotations align each sign with the running sum.
        # The 120 degree target comes last and flips.
        data = np.zeros((3, 1, 8), dtype=np.float32)
        data[:, 0, 3:7] = [_rotation(angle) for angle in (0, 120, -120)]
        data[:, 0, 7] = [0, 1, 2]
        source = wp.array(data, dtype=wp.float32)
        output = wp.empty((1, 8), dtype=wp.float32)
        directional_2d_dof_blend(source, _sample([0, 2, 1], [0.2, 0.5, 0.3]), source[0][0], output)
        expected = data[0, 0, 3:7] * 0.2 + data[2, 0, 3:7] * 0.5 - data[1, 0, 3:7] * 0.3
        _assert_rotations(output.numpy()[0, 3:7], expected / np.linalg.norm(expected))
        self.assertAlmostEqual(float(output.numpy()[0, 7]), 1.3, places=6)

    def test_more_than_three_targets_share_the_same_pose_and_dof_root_math(self):
        space = _space(
            BlendSpaceType.DIRECTIONAL_2D, ((1, 0), (0, 1), (-1, 0), (0, -1)),
            data_indices=(5, 0, 4, 2),
        )
        poses = np.zeros((7, 2, 1, 7), dtype=np.float32)
        poses[..., 6] = 1
        dofs = np.zeros((7, 2, 8), dtype=np.float32)
        samples = zip((5, 0, 4, 2), ((1, 0), (0, 1), (-1, 0), (0, -1)), (0, 30, -45, 60))
        for target, (index, key, angle) in enumerate(samples):
            poses[index, :, 0, 0] = target * 4
            poses[index, :, 0, 3:] = _rotation(angle)
            dofs[index, :, :7] = poses[index, :, 0]
            dofs[index, :, 7] = target
        pose_output = wp.empty((2, 1), dtype=wp.transform)
        dof_output = wp.empty((2, 8), dtype=wp.float32)
        sample = space.evaluate(wp.vec2(0.25, 0.25))

        prepared = wp.array(poses, dtype=wp.transform)
        reference = wp.array(poses[0, 0], dtype=wp.transform)
        prepare_directional_poses(prepared, reference)
        directional_2d_pose_blend(prepared, sample, reference, pose_output)
        directional_2d_dof_blend(wp.array(dofs, dtype=wp.float32), sample,
                            wp.array(dofs[0, 0], dtype=wp.float32), dof_output)

        expected_weights = np.array([0.375, 0.375, 0.125, 0.125])
        expected_position = expected_weights @ [0, 4, 8, 12]
        expected_rotation = expected_weights @ np.array([_rotation(angle) for angle in (0, 30, -45, 60)])
        expected_rotation /= np.linalg.norm(expected_rotation)
        np.testing.assert_allclose(pose_output.numpy()[:, 0, 0], expected_position)
        _assert_rotations(pose_output.numpy()[:, 0, 3:], expected_rotation)
        np.testing.assert_allclose(dof_output.numpy()[:, :7], pose_output.numpy()[:, 0], atol=1e-6)
        np.testing.assert_allclose(dof_output.numpy()[:, 7], expected_weights @ [0, 1, 2, 3])

    def test_skeletal_directional_signs_use_each_default_joint_and_ignore_sample_order(self):
        poses = np.zeros((3, 1, 2, 7), dtype=np.float32)
        rotations = np.array([_rotation(angle) for angle in (0, 120, -120)])
        poses[:, 0, :, 3:] = rotations[:, None]
        poses[1, ..., 3:] *= -1
        poses[:, 0, :, 0] = np.arange(3)[:, None]
        reference = np.zeros((2, 7), dtype=np.float32)
        reference[:, 3:] = [_rotation(0), _rotation(80)]
        prepared = wp.array(poses, dtype=wp.transform)
        reference_array = wp.array(reference, dtype=wp.transform)
        prepare_directional_poses(prepared, reference_array)
        expected_rotations = []
        for joint in range(2):
            signed = rotations.copy()
            signed[signed @ reference[joint, 3:] < 0] *= -1
            total = np.array([0.2, 0.3, 0.5]) @ signed
            expected_rotations.append(total / np.linalg.norm(total))
        keys = ((0, 0), (1, 0), (0, 1))
        output = wp.empty((1, 2), dtype=wp.transform)
        for order in itertools.permutations(range(3)):
            space = _space(
                BlendSpaceType.DIRECTIONAL_2D, [keys[index] for index in order],
                data_indices=order,
            )
            directional_2d_pose_blend(prepared, space.evaluate(wp.vec2(0.3, 0.5)), reference_array, output)
            _assert_rotations(output.numpy()[0, :, 3:], np.asarray(expected_rotations))
            np.testing.assert_allclose(output.numpy()[0, :, 0], 1.3)
        np.testing.assert_array_equal(prepared.numpy()[..., :3], poses[..., :3])
        _assert_rotations(prepared.numpy()[..., 3:], poses[..., 3:])

    def test_output_layout_errors_are_reported_before_launch(self):
        poses, dofs = _sources()
        sample = _sample([1], [1])
        with self.assertRaisesRegex(ValueError, "capacity"):
            linear_1d_pose_blend(wp.array(poses, dtype=wp.transform), sample,
                            wp.empty((2, 2), dtype=wp.transform))
        with self.assertRaisesRegex(ValueError, "components"):
            linear_1d_dof_blend(wp.array(dofs, dtype=wp.float32), sample,
                           wp.empty((3, 8), dtype=wp.float32))


class CUDAAnimationBlendingTests(AnimationBlendingTests):
    device = "cuda:0"

    def test_compilation_and_both_blenders_use_the_data_device(self):
        with wp.ScopedDevice("cpu"):
            device = wp.get_device(self.device)
            poses, dofs = _sources(frame_count=1)
            pose_data = wp.array(poses, dtype=wp.transform, device=device)
            dof_data = wp.array(dofs, dtype=wp.float32, device=device)
            pose_output = wp.empty((1, 2), dtype=wp.transform, device=device)
            dof_output = wp.empty((1, 9), dtype=wp.float32, device=device)
            pose_reference = wp.array(poses[1, 0], dtype=wp.transform, device=device)
            dof_reference = wp.array(dofs[1, 0], dtype=wp.float32, device=device)
            space = _space(BlendSpaceType.DIRECTIONAL_2D, ((1, 0), (0, 1)), data_indices=(1, 4), device=device)
            sample = space.evaluate(wp.vec2(0, 0))
            prepare_directional_poses(pose_data, pose_reference)
            directional_2d_pose_blend(pose_data, sample, pose_reference, pose_output)
            directional_2d_dof_blend(dof_data, sample, dof_reference, dof_output)
            np.testing.assert_allclose(pose_output.numpy()[0, 0, 0], 6.0)
            np.testing.assert_allclose(dof_output.numpy()[0, :7], pose_output.numpy()[0, 0], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
