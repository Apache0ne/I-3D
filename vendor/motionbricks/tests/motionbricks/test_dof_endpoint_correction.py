# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Endpoint correction mathematics, independent of motions and robot assets."""

import unittest

import numpy as np
from scipy.spatial.transform import Rotation
import warp as wp

from motionbricks.model.unitree_g1 import kernels


def _motion(count):
    t = np.linspace(0, 1, count)
    poses = np.zeros((count, 9), dtype=np.float32)
    poses[:, :3] = np.column_stack((t, t * -.2, .8 + np.sin(t) * .02))
    poses[:, 3:7] = Rotation.from_euler("xyz", np.column_stack((t * .2, t * -.15, t * .3 + 3.))).as_quat()
    poses[:, 7:] = np.column_stack((2.8 + t * .1, np.sin(t * 3)))
    return poses


def _reference(raw, start, end, fade_frames=6.0):
    weights = np.linspace(0, 1, len(raw))[:, None]
    expected = raw.astype(np.float64)
    expected[:, :3] += (1 - weights) * (start[:3] - raw[0, :3])
    expected[:, :3] += weights * (end[:3] - raw[-1, :3])
    distance = np.arange(len(raw), dtype=np.float64)[:, None]
    phase = np.clip(distance / min(fade_frames, (len(raw) - 1) / 2), 0, 1)
    falloff = np.where(phase < .5, 1 - 2*phase**2, 2*(1 - phase)**2)
    expected[:, 7:] += falloff * (start[7:] - raw[0, 7:])
    expected[:, 7:] += falloff[::-1] * (end[7:] - raw[-1, 7:])
    rotations = Rotation.from_quat(raw[:, 3:7])
    start_delta = rotations[0].inv() * Rotation.from_quat(start[3:7])
    end_delta = rotations[-1].inv() * Rotation.from_quat(end[3:7])
    expected[:, 3:7] = (
        rotations * Rotation.from_rotvec((1 - weights) * start_delta.as_rotvec())
        * Rotation.from_rotvec(weights * end_delta.as_rotvec())
    ).as_quat()
    return expected


class DOFEndpointCorrectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.previous_device = wp.get_device()
        wp.set_device("cpu")

    @classmethod
    def tearDownClass(cls):
        wp.set_device(cls.previous_device)

    def _correct(self, source, context, target, count, fade_frames=6.0):
        generated = wp.array(source, dtype=wp.float32)
        context_device = wp.array(context, dtype=wp.float32)
        target_device = wp.array(target, dtype=wp.float32)
        frame_count = wp.array([count], dtype=wp.int32)
        deltas = wp.empty((2, source.shape[1]), dtype=wp.float32)
        wp.launch(kernels.compute_endpoint_deltas_kernel, dim=2,
                  inputs=[frame_count, context_device, target_device, generated], outputs=[deltas])
        wp.launch(kernels.apply_endpoint_deltas_kernel, dim=len(source),
                  inputs=[frame_count, deltas, fade_frames], outputs=[generated])
        np.testing.assert_array_equal(context_device.numpy(), context)
        np.testing.assert_array_equal(target_device.numpy(), target)
        return generated.numpy()

    def _assert_poses(self, actual, expected):
        np.testing.assert_allclose(actual[:, :3], expected[:, :3], atol=3e-6)
        np.testing.assert_allclose(actual[:, 7:], expected[:, 7:], atol=3e-6)
        error = Rotation.from_quat(actual[:, 3:7]).inv() * Rotation.from_quat(expected[:, 3:7])
        self.assertLess(float(error.magnitude().max()), 3e-6)
        np.testing.assert_allclose(np.linalg.norm(actual[:, 3:7], axis=1), 1, atol=2e-7)

    def test_matches_endpoints_and_localizes_deltas_without_wrapping_hinges(self):
        for count in (18, 58):
            with self.subTest(count=count):
                raw = _motion(count)
                context, target = _motion(4), _motion(4)
                context[-1, :3] += [.1, -.2, .03]
                target[0, :3] += [1.5, .1, -.04]
                context[-1, 3:7] = -Rotation.from_euler("xyz", [.4, .3, -3.0]).as_quat()
                target[0, 3:7] = Rotation.from_euler("xyz", [-.5, .2, 3.1]).as_quat()
                context[-1, 7:] = [-2.8, 1.5]
                target[0, 7:] = [-2.6, .3]
                # Other guide rows and the spare output capacity must not affect correction.
                context[:-1] = np.nan
                target[1:] = np.nan
                source = np.concatenate((raw, np.full((3, 9), np.nan, dtype=np.float32)))
                actual = self._correct(source, context, target, count)
                expected = _reference(raw, context[-1], target[0])
                self._assert_poses(actual[:count], expected)
                self._assert_poses(actual[[0, count - 1]], np.stack((context[-1], target[0])))
                # Crossing +/-pi uses a scalar offset, not an angle wrap. The
                # unmodified interior keeps the predictor's positive hinge.
                np.testing.assert_array_equal(actual[6:count - 6, 7:], raw[6:count - 6, 7:])
                self.assertAlmostEqual(actual[3, 7], (raw[3, 7] + .5*(context[-1, 7] - raw[0, 7])), places=6)
                np.testing.assert_array_equal(actual[count:], source[count:])

    def test_fade_duration_sets_the_corrected_region(self):
        raw, context, target = _motion(26), _motion(4), _motion(4)
        context[-1, 7:] = raw[0, 7:] + [1.0, -2.0]
        target[0, 7:] = raw[-1, 7:] + [3.0, -4.0]
        for fade_frames in (3.0, 6.0, 9.0, 12.0):
            with self.subTest(fade_frames=fade_frames):
                actual = self._correct(raw, context, target, len(raw), fade_frames)
                self._assert_poses(actual, _reference(raw, context[-1], target[0], fade_frames))
                span = int(fade_frames)
                np.testing.assert_array_equal(actual[span:-span, 7:], raw[span:-span, 7:])
                start_offsets = actual[:span + 1, 7] - raw[:span + 1, 7]
                self.assertTrue(np.all(np.diff(start_offsets) <= 0))
                self.assertAlmostEqual(float(start_offsets[0]), 1.0, places=6)
                self.assertEqual(float(start_offsets[-1]), 0.0)
                np.testing.assert_allclose(start_offsets + start_offsets[::-1], 1.0, atol=3e-7)

    def test_short_clips_keep_both_endpoints_without_overlapping_fades(self):
        for count in (2, 3, 8, 13):
            with self.subTest(count=count):
                raw, context, target = _motion(count), _motion(4), _motion(4)
                context[-1, 7:] -= [2, 3]
                target[0, 7:] += [4, 5]
                actual = self._correct(raw, context, target, count)
                self._assert_poses(actual, _reference(raw, context[-1], target[0]))
                self._assert_poses(actual[[0, -1]], np.stack((context[-1], target[0])))
                if count % 2 == 1:
                    np.testing.assert_array_equal(actual[count // 2, 7:], raw[count // 2, 7:])

    def test_quaternion_signs_do_not_change_the_corrected_motion(self):
        source, context, target = _motion(18), _motion(4), _motion(4)
        source[:, 3:7] = Rotation.from_euler("z", np.deg2rad(179)).as_quat()
        context[-1, 3:7] = Rotation.from_euler("z", np.deg2rad(-179)).as_quat()
        target[0, 3:7] = Rotation.from_euler("z", np.deg2rad(177)).as_quat()
        expected = self._correct(source, context, target, len(source))
        source[::2, 3:7] *= -1
        context[:, 3:7] *= -1
        target[:, 3:7] *= -1
        actual = self._correct(source, context, target, len(source))
        self._assert_poses(actual, expected)
        step_angles = (Rotation.from_quat(actual[1:, 3:7]) * Rotation.from_quat(actual[:-1, 3:7]).inv()).magnitude()
        self.assertLess(float(step_angles.max()), np.deg2rad(1))


if __name__ == "__main__":
    unittest.main()
