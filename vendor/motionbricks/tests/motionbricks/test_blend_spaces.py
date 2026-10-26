# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

from motionbricks.runtime.blend_spaces import BlendSpace, BlendSpaceType


def _space(space_type, values, *, data_indices=None):
    space = BlendSpace(space_type)
    indices = range(len(values)) if data_indices is None else data_indices
    for value, index in zip(values, indices):
        space.add_target(wp.vec2(*value), index)
    space.compile()
    return space


def _weights(sample):
    result = np.full(sample.data_indices.size, sample.uniform_weight)
    for index, weight in zip(sample.indices, sample.weights):
        result[index] += weight
    return result


class BlendSpaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.previous_device = wp.get_device()
        wp.set_device("cpu")

    @classmethod
    def tearDownClass(cls):
        wp.set_device(cls.previous_device)

    def test_linear_sorts_keys_and_selects_the_ordered_pair(self):
        space = _space(
            BlendSpaceType.LINEAR_1D, ((2, 100), (0, -100), (10, 0)),
            data_indices=(7, 4, 1),
        )
        pointer = space.data_indices.ptr
        for value, expected in (
            (-1.0, [0.0, 1.0, 0.0]),
            (0.5, [0.25, 0.75, 0.0]),
            (2.0, [1.0, 0.0, 0.0]),
            (4.0, [0.75, 0.0, 0.25]),
            (20.0, [0.0, 0.0, 1.0]),
        ):
            np.testing.assert_allclose(_weights(space.evaluate(wp.vec2(value, 999))), expected)
        sample = space.evaluate(wp.vec2(0.5, 0))
        self.assertEqual(tuple(sample.indices)[:2], (1, 0))
        np.testing.assert_allclose(tuple(sample.weights)[:2], (0.75, 0.25))
        np.testing.assert_array_equal(space.data_indices.numpy(), [7, 4, 1])
        self.assertEqual(space.data_indices.ptr, pointer)
        self.assertEqual(space.data_indices.device, wp.get_device("cpu"))

    def test_directional_uses_center_surrounding_targets_and_clamps(self):
        space = _space(
            BlendSpaceType.DIRECTIONAL_2D, ((0, 0), (1, 0), (0, 1), (-1, 0), (0, -1)),
            data_indices=(0, 2, 4, 6, 8),
        )
        for value, expected in (
            ((0.0, 0.0), [1, 0, 0, 0, 0]),
            ((0.25, 0.25), [0.5, 0.25, 0.25, 0, 0]),
            ((2.0, 2.0), [0, 0.5, 0.5, 0, 0]),
            ((1.0, 0.0), [0, 1, 0, 0, 0]),
            ((0.0, -1.0), [0, 0, 0, 0, 1]),
            ((-1.0, -0.0), [0, 0, 0, 1, 0]),
            ((-0.25, -0.5), [0.25, 0, 0, 0.25, 0.5]),
        ):
            np.testing.assert_allclose(_weights(space.evaluate(wp.vec2(*value))), expected)
        np.testing.assert_array_equal(space.data_indices.numpy(), [0, 2, 4, 6, 8])

    def test_directional_preparation_handles_unequal_radii_and_unsorted_keys(self):
        space = _space(
            BlendSpaceType.DIRECTIONAL_2D, ((0, -0.5), (0, 3), (-1, 0), (2, 0), (0, 0)),
        )
        for value, expected in (
            ((0.5, 0.75), [0, 0.25, 0, 0.25, 0.5]),
            ((4, 6), [0, 0.5, 0, 0.5, 0]),
            ((-0.25, -0.125), [0.25, 0, 0.25, 0, 0.5]),
        ):
            sample = space.evaluate(wp.vec2(*value))
            np.testing.assert_allclose(_weights(sample), expected)
            self.assertEqual(tuple(sample.indices), tuple(sorted(sample.indices)))

    def test_directional_falls_back_for_open_and_collinear_sectors(self):
        for keys in (((0, 0), (1, 0)), ((0, 0), (1, 0), (-1, 0))):
            space = _space(BlendSpaceType.DIRECTIONAL_2D, keys)
            for value in ((0, 1), (0, -1)):
                np.testing.assert_array_equal(_weights(space.evaluate(wp.vec2(*value))), [1] + [0] * (len(keys) - 1))
        space = _space(BlendSpaceType.DIRECTIONAL_2D, ((1, 0), (-1, 0)))
        np.testing.assert_allclose(_weights(space.evaluate(wp.vec2(0, 1))), [0.5, 0.5])

    def test_evaluation_requires_compiling_all_registered_targets(self):
        space = BlendSpace(BlendSpaceType.LINEAR_1D)
        space.add_target(wp.vec2(0, 0), 4)
        with self.assertRaisesRegex(RuntimeError, "compile"):
            space.evaluate(wp.vec2(0, 0))
        space.compile()
        np.testing.assert_array_equal(_weights(space.evaluate(wp.vec2(0, 0))), [1])
        space.add_target(wp.vec2(1, 0), 8)
        with self.assertRaisesRegex(RuntimeError, "compile"):
            space.evaluate(wp.vec2(0.25, 0))
        space.compile()
        np.testing.assert_allclose(_weights(space.evaluate(wp.vec2(0.25, 0))), [0.75, 0.25])

    def test_invalid_layouts_are_rejected_during_preparation(self):
        for space_type, values, indices, message in (
            (BlendSpaceType.LINEAR_1D, (), None, "without targets"),
            (BlendSpaceType.LINEAR_1D, ((0, 0),), (-1,), "non-negative"),
            (BlendSpaceType.LINEAR_1D, ((0, 0), (0, 1)), None, "must be unique"),
            (BlendSpaceType.DIRECTIONAL_2D, ((0, 0), (0, 0)), None, "only one center"),
            (BlendSpaceType.DIRECTIONAL_2D, ((1, 0), (2, 0)), None, "distinct directions"),
            (BlendSpaceType.DIRECTIONAL_2D, ((1, float("nan")),), None, "finite"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                _space(space_type, values, data_indices=indices)


if __name__ == "__main__":
    unittest.main()
