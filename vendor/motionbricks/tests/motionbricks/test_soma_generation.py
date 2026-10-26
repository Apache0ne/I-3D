# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SOMA generation with a controlled predictor in place of TensorRT."""

import unittest

import numpy as np
from scipy.spatial.transform import Rotation
import warp as wp

from motionbricks.animation.animation_buffer import AnimationBuffer
from motionbricks.animation.skeleton import Skeleton
from motionbricks.inference.errors import TensorContractError
from motionbricks.io.usd import load_skeleton_from_usd
from motionbricks.io.utils import SOMA_SKELETON_PRIM_PATH, get_asset_file
from motionbricks.model.config import MotionBricksInferenceConfig
from motionbricks.model.soma.engine import MotionBricksSOMAEngine
from motionbricks.utils.space_conversion_utils import FacingDirectionType, SpaceConverter


def _transform(values) -> wp.transform:
    return wp.transform(wp.vec3(*values[:3]), wp.quat(*values[3:7]))


def _config(tokens: int, correct: bool) -> MotionBricksInferenceConfig:
    return MotionBricksInferenceConfig(allowed_token_counts=(tokens,), correct_generated_diff=correct)


class _ControlledSOMAEngine(MotionBricksSOMAEngine):
    """Exercise the real generation path, replacing only the TensorRT backend."""

    def _initialize(self, build_options, *, cache_dir, force_rebuild, progress_monitor):
        self._tensor_specs = {spec.name: spec for spec in self.contract.tensor_specs}
        self._buffers = {
            spec.name: wp.zeros(
                spec.shape, dtype=wp.float32 if spec.dtype == np.dtype(np.float32) else wp.int32, device=self.device,
            )
            for spec in self.contract.tensor_specs
        }
        self.tokens = 8

    def _enqueue_tensorrt(self):
        with wp.ScopedStream(self.stream):
            wp.copy(self.device_tensor("pred_global_motions"), self.prediction)
            self.device_tensor("pred_num_tokens").assign([self.tokens])


class SOMAGenerationEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        if not wp.is_cuda_available():
            raise unittest.SkipTest("CUDA is unavailable")
        cls.previous_device = wp.get_device()
        wp.set_device("cuda:0")
        cls.skeleton = load_skeleton_from_usd(
            get_asset_file("human", "soma_base_skel_minimal_no_root.usd"), SOMA_SKELETON_PRIM_PATH,
        )

    @classmethod
    def tearDownClass(cls):
        wp.set_device(cls.previous_device)

    def setUp(self):
        self.model_to_runtime = SpaceConverter(FacingDirectionType.MUJOCO).convert_transform(wp.transform_identity())
        self.engine = _ControlledSOMAEngine(
            "controlled.onnx", model_to_runtime=self.model_to_runtime, bake_cuda_graph=False,
        )
        self.addCleanup(self.engine.close)
        # Hips move forward in the aligned model space. Every joint keeps an identity global rotation.
        prediction = np.zeros((1, 64, 366), dtype=np.float32)
        prediction[0, :, 0:3] = np.column_stack((0.02 * np.arange(64), np.full(64, 0.9), 0.05 * np.arange(64)))
        rotations = 5 + 3 * (len(self.engine.contract.joint_names) - 1)
        for joint in range(len(self.engine.contract.joint_names)):
            prediction[0, :, rotations + 6 * joint: rotations + 6 * joint + 6] = [1, 0, 0, 0, 1, 0]
        self.prediction = prediction
        self.engine.prediction = wp.array(prediction, dtype=wp.float32)

        self.output = self.engine.create_output(self.skeleton)
        self.context = self.engine.create_input(self.skeleton)
        self.target = self.engine.create_input(self.skeleton)
        reference = self.skeleton.reference_local_transforms.numpy()
        hips = _transform(reference[0])
        poses = np.repeat(reference[None], 8, axis=0)
        for frame in range(8):
            # Walk along the model's forward axis, with the target poses 1.5 m further ahead.
            step = wp.vec3(0.0, 0.0, 0.04 * frame + (1.5 if frame >= 4 else 0.0))
            moved = wp.transform(wp.add(hips.p, step), hips.q)
            poses[frame, 0] = np.asarray(wp.mul(self.model_to_runtime, moved))
        with wp.ScopedStream(self.engine.stream):
            self.context.update_from_poses(wp.array(poses[:4], dtype=wp.transform), stream=self.engine.stream)
            self.target.update_from_poses(wp.array(poses[4:], dtype=wp.transform), stream=self.engine.stream)
        self.poses = poses

    def generate(self, tokens: int, correct: bool, context=None) -> np.ndarray:
        self.engine.tokens = tokens
        self.engine.generate_animation(
            context or self.context, self.target, self.output, config=_config(tokens, correct),
        )
        return self.output.poses.numpy()[:self.output.num_frames].copy()

    def test_generation_trims_padding_and_places_the_hips_in_runtime_space(self):
        pointer = self.output.poses.ptr
        for tokens in (16, 6, 8):
            with self.subTest(tokens=tokens):
                frames = self.generate(tokens, correct=False)
                self.assertEqual(self.output.num_frames, tokens * 4 - 6)
                self.assertEqual(self.output.poses.ptr, pointer)
                # The first output frame is the raw frame after the three padding frames.
                to_runtime = wp.mul(self.model_to_runtime, _transform(self.engine._alignment.numpy()[0]))
                expected = [wp.transform_point(to_runtime, wp.vec3(*self.prediction[0, 3 + frame, :3]))
                            for frame in range(len(frames))]
                np.testing.assert_allclose(frames[:, 0, :3], expected, atol=1e-5)

    def test_generation_copies_each_output_frames_foot_contacts_clamped(self):
        contacts = np.arange(64 * 4, dtype=np.float32).reshape(64, 4) / 128.0 - 0.5
        self.prediction[0, :, -4:] = contacts
        self.engine.prediction = wp.array(self.prediction, dtype=wp.float32)
        self.generate(8, correct=False)
        frames = self.output.num_frames
        # Output frame 0 is the raw frame after the three padding frames.
        np.testing.assert_array_equal(self.output.foot_contacts.numpy()[:frames], np.clip(contacts[3:3 + frames], 0, 1))

    def test_correction_meets_context_and_target_poses_and_can_be_turned_off(self):
        raw = self.generate(8, correct=False)
        corrected = self.generate(8, correct=True)
        for frame, pose in ((corrected[0], self.poses[3]), (corrected[-1], self.poses[4])):
            np.testing.assert_allclose(frame[:, :3], pose[:, :3], atol=1e-4)
            angles = (Rotation.from_quat(frame[:, 3:]).inv() * Rotation.from_quat(pose[:, 3:])).magnitude()
            self.assertLess(float(angles.max()), 1e-3)
        self.assertGreater(float(np.abs(raw[0, 0, :3] - self.poses[3, 0, :3]).max()), 1e-2)
        np.testing.assert_array_equal(self.generate(8, correct=False), raw)
        self.assertEqual(self.engine.timer.milliseconds["post_process"], 0.0)

    def test_invalid_model_lengths_raise_without_publishing_frames(self):
        for tokens in (5, 17, 7):
            with self.subTest(tokens=tokens):
                self.generate(8, correct=False)
                self.engine.tokens = tokens
                with self.assertRaises(TensorContractError):
                    self.engine.generate_animation(self.context, self.target, self.output, config=_config(8, False))
                self.assertEqual(self.output.num_frames, 0)

    def test_buffers_must_match_the_model(self):
        for create in (self.engine.create_input, self.engine.create_output):
            with self.subTest(create=create.__name__), self.assertRaisesRegex(ValueError, "missing model joints"):
                create(Skeleton(["Hips"], [-1], [wp.transform_identity()]))
        short_context = self.engine.create_input(self.skeleton)
        with wp.ScopedStream(self.engine.stream):
            short_context.update_from_poses(wp.array(self.poses[:2], dtype=wp.transform), stream=self.engine.stream)
        with self.assertRaisesRegex(ValueError, "Context token must contain 4 poses"):
            self.generate(8, correct=False, context=short_context)
        empty_target = self.engine.create_input(self.skeleton)
        with self.assertRaisesRegex(ValueError, "Target token must contain 1 to 4 poses"):
            self.engine.generate_animation(self.context, empty_target, self.output)
        no_contacts = AnimationBuffer(self.skeleton, 30.0, wp.empty_like(self.output.poses), num_frames=0)
        with self.assertRaisesRegex(ValueError, "foot_contacts"):
            self.engine.generate_animation(self.context, self.target, no_contacts)


if __name__ == "__main__":
    unittest.main()
