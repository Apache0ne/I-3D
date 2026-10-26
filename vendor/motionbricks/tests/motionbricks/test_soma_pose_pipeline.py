# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
import unittest

import numpy as np
import warp as wp

from motionbricks.io.usd import load_skeleton_from_usd
from motionbricks.io.utils import SOMA_SKELETON_PRIM_PATH, get_asset_file, get_soma_human_model
from motionbricks.model.contract import MODEL_CONTEXT_AND_TARGET_POSES, MODEL_POSES_PER_TOKEN, valid_frame_count
from motionbricks.model.soma import kernels
from motionbricks.model.soma.contract import SOMA_N1_CONTRACT
from motionbricks.runtime.target_pose import IKPin, TargetPoseIKCorrector, create_soma_target_pose_ik_rig
from motionbricks.utils import pose_utils
from motionbricks.utils.space_conversion_utils import FacingDirectionType, SpaceConverter


def _as_transform(values: np.ndarray) -> wp.transform:
    return wp.transform(wp.vec3(*values[:3]), wp.quat(*values[3:]))


class SOMAPosePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        wp.init()
        cls.previous_device = wp.get_device()
        wp.set_device("cpu")
        cls.skeleton = load_skeleton_from_usd(
            get_asset_file("human", "soma_base_skel_minimal_no_root.usd"),
            SOMA_SKELETON_PRIM_PATH,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        wp.set_device(cls.previous_device)

    def test_usd_rest_pose_defines_the_runtime_skeleton_and_model_joints(self) -> None:
        skeleton = self.skeleton

        self.assertEqual(skeleton.num_joints, 77)
        self.assertEqual(skeleton.joint_names[:3], ["Hips", "Spine1", "Spine2"])
        np.testing.assert_array_equal(skeleton.parent_indices.numpy()[:3], [-1, 0, 1])
        np.testing.assert_allclose(skeleton.up_axis, [0.0, 1.0, 0.0])
        np.testing.assert_allclose(skeleton.forward_axis, [0.0, 0.0, 1.0])

        model_indices = [skeleton.joint_index(name) for name in SOMA_N1_CONTRACT.joint_names]
        self.assertTrue(all(index >= 0 for index in model_indices))
        self.assertEqual(len(set(model_indices)), len(model_indices))
        thumb_end = skeleton.joint_index("LeftHandThumbEnd")
        self.assertEqual(skeleton.joint_name(skeleton.joint_parent(thumb_end)), "LeftHandThumb3")

    def test_mesh_holds_relaxed_hands_and_a_fingers_mask(self) -> None:
        mesh = get_soma_human_model()
        finger = re.compile(r"(Left|Right)Hand(Thumb|Index|Middle|Ring|Pinky)")
        fingers = [index for index, name in enumerate(mesh.skeleton.joint_names) if finger.match(name)]
        self.assertEqual(len(fingers), 48)
        weights = mesh.override_pose_masks["Fingers"].numpy()
        np.testing.assert_array_equal(np.flatnonzero(weights), fingers)
        np.testing.assert_array_equal(weights[fingers], 1.0)

        # The reference pose keeps its own fingers, so only the drawn mesh curls them.
        relaxed = mesh.override_poses["RelaxedHands"].numpy()[fingers, 3:]
        rest = mesh.skeleton.reference_local_transforms.numpy()[fingers, 3:]
        turns = np.degrees(2 * np.arccos(np.clip(np.abs(np.sum(relaxed * rest, axis=1)), 0, 1)))
        self.assertGreater(turns.max(), 20)

    def test_target_pose_ik_corrector_builds_on_the_usd_skeleton(self) -> None:
        corrector = TargetPoseIKCorrector(self.skeleton, create_soma_target_pose_ik_rig(self.skeleton))
        corrector.capture_pins(self.skeleton.reference_local_transforms, IKPin.ALL)
        self.assertEqual(corrector.pin_mask, IKPin.ALL)

    def test_decoder_restores_world_displacement_for_generated_frames(self) -> None:
        num_model_joints = len(SOMA_N1_CONTRACT.joint_names)
        num_full_joints = self.skeleton.num_joints
        frame_count = valid_frame_count(6)
        features = np.zeros((1, 64, 366), dtype=np.float32)
        features[:, :, 0] = 2.0
        features[:, :, 1] = 1.25
        features[:, :, 2] = 3.0

        position_offset = 5
        rotation_offset = position_offset + (num_model_joints - 1) * 3
        for joint in range(num_model_joints):
            features[:, :, rotation_offset + joint * 6] = 1.0
            features[:, :, rotation_offset + joint * 6 + 4] = 1.0

        predicted = wp.array(features, dtype=wp.float32)
        predicted_tokens = wp.array([6], dtype=wp.int32)
        model_to_pose = [self.skeleton.joint_index(name) for name in SOMA_N1_CONTRACT.joint_names]
        pose_to_model_values = np.full(num_full_joints, -1, dtype=np.int32)
        for model_joint, pose_joint in enumerate(model_to_pose):
            parent = self.skeleton.joint_parent(pose_joint)
            if parent == -1 or pose_to_model_values[parent] >= 0:
                pose_to_model_values[pose_joint] = model_joint
        pose_to_model = wp.array(pose_to_model_values, dtype=wp.int32)
        reference = self.skeleton.reference_local_transforms
        alignment = wp.array([wp.transform(wp.vec3(100.0, 0.0, -200.0), wp.quat_identity())], dtype=wp.transform)
        converter = SpaceConverter(FacingDirectionType.MAYA)
        model_to_runtime = converter.convert_transform(wp.transform_identity())
        output_capacity = frame_count + 2
        sentinel = np.asarray(wp.transform(wp.vec3(-9.0, -8.0, -7.0), wp.quat_identity()), dtype=np.float32)
        initial_output = np.tile(sentinel, (output_capacity, num_full_joints, 1))
        output_local = wp.array(initial_output, dtype=wp.transform)
        output_count = wp.empty(1, dtype=wp.int32)
        output_contacts = wp.empty((output_capacity, 4), dtype=wp.float32)

        wp.launch(kernels.decode_motionbricks_output_kernel, dim=output_capacity, inputs=[
            num_model_joints, num_full_joints, self.skeleton.parent_indices,
            pose_to_model, reference, alignment, model_to_runtime, predicted, predicted_tokens,
        ], outputs=[output_local, output_contacts, output_count])

        self.assertEqual(int(output_count.numpy()[0]), frame_count)
        expected_hips = wp.mul(model_to_runtime, wp.transform(wp.vec3(102.0, 1.25, -197.0), wp.quat_identity()))
        decoded_values = output_local.numpy()
        np.testing.assert_allclose(decoded_values[0, 0], np.asarray(expected_hips), atol=1.0e-5)
        np.testing.assert_allclose(decoded_values[frame_count:], initial_output[frame_count:], atol=1.0e-6)
        reference_values = reference.numpy()
        output_values = decoded_values[0]
        for name in ("LeftHandThumbEnd", "LeftHandMiddleEnd", "RightHandThumbEnd", "RightHandMiddleEnd"):
            joint = self.skeleton.joint_index(name)
            self.assertEqual(pose_to_model_values[joint], -1)
            np.testing.assert_allclose(output_values[joint], reference_values[joint])

    def test_endpoint_correction_does_not_arc_hips_translation(self) -> None:
        frame_count = 3
        identity = np.asarray(wp.transform_identity(), dtype=np.float32)
        context_values = np.tile(identity, (4, 1, 1))
        target_values = context_values.copy()
        output_values = np.tile(identity, (frame_count, 1, 1))

        start_rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.75)
        end_rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -0.75)
        context_values[-1, 0] = np.asarray(wp.transform(wp.vec3(10.0, 0.0, 1.2), start_rotation))
        target_values[0, 0] = np.asarray(wp.transform(wp.vec3(12.0, 0.0, 1.4), end_rotation))
        output_values[:, 0, :3] = ((10.0, 0.0, 1.0), (11.0, 0.0, 1.0), (12.0, 0.0, 1.0))

        context = wp.array(context_values, dtype=wp.transform)
        target = wp.array(target_values, dtype=wp.transform)
        output = wp.array(output_values, dtype=wp.transform)
        output_count = wp.array([frame_count], dtype=wp.int32)
        deltas = [wp.empty(1, dtype=wp.transform), wp.empty(1, dtype=wp.transform)]

        wp.launch(kernels.compute_endpoint_deltas_kernel, dim=1,
                  inputs=[output_count, context, target, output], outputs=deltas)
        wp.launch(kernels.apply_endpoint_deltas_kernel, dim=(frame_count, 1),
                  inputs=[output_count, *deltas], outputs=[output])

        corrected = output.numpy()[:, 0]
        np.testing.assert_allclose(corrected[0], context_values[-1, 0], atol=1.0e-5)
        np.testing.assert_allclose(corrected[-1], target_values[0, 0], atol=1.0e-5)
        np.testing.assert_allclose(
            corrected[:, :3], [[10.0, 0.0, 1.2], [11.0, 0.0, 1.3], [12.0, 0.0, 1.4]], atol=1.0e-5,
        )

    def test_encoder_recenters_newton_poses_in_model_space(self) -> None:
        skeleton = self.skeleton
        reference = skeleton.reference_local_transforms.numpy()
        context_poses = np.tile(reference[None, :, :], (MODEL_POSES_PER_TOKEN, 1, 1))
        target_poses = np.tile(reference[None, :, :], (MODEL_POSES_PER_TOKEN, 1, 1))
        planar_origin = np.array([10_000.0, 0.0, -20_000.0], dtype=np.float32)
        context_poses[:, 0, :3] += planar_origin
        target_poses[:, 0, :3] += planar_origin
        target_poses[:, 0, 2] += 2.0

        converter = SpaceConverter(FacingDirectionType.MAYA)
        model_to_runtime = converter.convert_transform(wp.transform_identity())
        for poses in (context_poses, target_poses):
            for frame in range(poses.shape[0]):
                poses[frame, 0] = np.asarray(wp.mul(model_to_runtime, _as_transform(poses[frame, 0])))
        runtime_to_model = wp.transform_inverse(model_to_runtime)

        num_model_joints = len(SOMA_N1_CONTRACT.joint_names)
        model_to_pose = [skeleton.joint_index(name) for name in SOMA_N1_CONTRACT.joint_names]
        model_indices = wp.array(model_to_pose, dtype=wp.int32)
        right_leg, left_leg = skeleton.joint_index("RightLeg"), skeleton.joint_index("LeftLeg")
        context_local = wp.array(context_poses, dtype=wp.transform)
        target_local = wp.array(target_poses, dtype=wp.transform)
        guide_world = wp.empty((MODEL_CONTEXT_AND_TARGET_POSES, skeleton.num_joints), dtype=wp.transform)
        alignment = wp.empty(1, dtype=wp.transform)
        specs = {spec.name: spec for spec in SOMA_N1_CONTRACT.tensor_specs}
        global_root = wp.zeros(specs["global_root_values"].shape, dtype=wp.float32)
        local_root = wp.zeros(specs["local_root_values"].shape, dtype=wp.float32)
        local_pose = wp.zeros(specs["local_poses"].shape, dtype=wp.float32)

        pose_utils.compute_world_poses(
            context_local, skeleton.parent_indices, guide_world[:MODEL_POSES_PER_TOKEN], runtime_to_model,
        )
        pose_utils.compute_world_poses(
            target_local, skeleton.parent_indices, guide_world[MODEL_POSES_PER_TOKEN:], runtime_to_model,
        )
        wp.launch(kernels.compute_guide_alignment_kernel, dim=1, inputs=[
            right_leg, left_leg, wp.vec2(1.0, 0.0), guide_world,
        ], outputs=[alignment])
        wp.launch(kernels.encode_motionbricks_inputs_kernel, dim=MODEL_CONTEXT_AND_TARGET_POSES, inputs=[
            num_model_joints, right_leg, left_leg, model_indices, guide_world, alignment,
        ], outputs=[global_root, local_root, local_pose])

        global_values = global_root.numpy()[0]
        local_values = local_root.numpy()[0]
        pose_values = local_pose.numpy()[0]
        guide_values = guide_world.numpy()
        context = MODEL_POSES_PER_TOKEN - 1
        target = MODEL_POSES_PER_TOKEN

        np.testing.assert_allclose(global_values[:MODEL_POSES_PER_TOKEN][:, [0, 2]], 0.0, atol=1.0e-5)
        np.testing.assert_allclose(
            global_values[MODEL_POSES_PER_TOKEN:][:, [0, 2]],
            np.tile([2.0, 0.0], (MODEL_POSES_PER_TOKEN, 1)),
            atol=1.0e-5,
        )
        self.assertAlmostEqual(float(global_values[context, 1]), float(guide_values[context, 0, 1]), places=5)
        self.assertAlmostEqual(global_values[target, 0] - global_values[context, 0], 2.0, places=5)
        np.testing.assert_allclose(local_values[context, :3], 0.0, atol=1.0e-5)
        self.assertTrue(np.isfinite(pose_values).all())
        rotations = pose_values[:, 3 * (num_model_joints - 1):].reshape(
            MODEL_CONTEXT_AND_TARGET_POSES, num_model_joints, 6,
        )
        np.testing.assert_allclose(np.linalg.norm(rotations[..., :3], axis=-1), 1.0, atol=1.0e-5)
        np.testing.assert_allclose(np.linalg.norm(rotations[..., 3:], axis=-1), 1.0, atol=1.0e-5)

    def test_pose_utils_accept_single_and_batched_pose_storage(self) -> None:
        reference = self.skeleton.reference_local_transforms.numpy()
        local_values = np.repeat(reference[None], 2, axis=0)
        local_values[1, 0, 0] += 1.0

        parent_indices = self.skeleton.parent_indices
        local_poses = wp.array(local_values, dtype=wp.transform)
        world_poses = wp.empty_like(local_poses)
        round_trip = wp.empty_like(local_poses)
        pose_utils.compute_world_poses(local_poses, parent_indices, world_poses)
        pose_utils.compute_local_poses(world_poses, parent_indices, round_trip)
        np.testing.assert_allclose(round_trip.numpy(), local_values, atol=1.0e-5)

        single_world = wp.empty(self.skeleton.num_joints, dtype=wp.transform)
        pose_utils.compute_world_poses(local_poses[1], parent_indices, single_world)
        np.testing.assert_allclose(single_world.numpy(), world_poses.numpy()[1], atol=1.0e-5)


if __name__ == "__main__":
    unittest.main()
