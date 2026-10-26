# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import warp as wp

from app.runtime.common.clip_registry import ClipAsset, ClipRegistry
from app.runtime.common.io.smart_primitive_document import (
    SmartPrimitiveDocumentError,
    SmartPrimitiveAssetPool,
    SmartPrimitiveDocument,
    TargetPoseComponent,
    build_runtime_smart_primitive_data,
    save_smart_primitive_document,
)
from app.runtime.common.smart_primitive_behaviors import (
    SittingBehavior,
    SmartPrimitiveBehavior,
    VelocityMatchingSequenceBehavior,
    behavior_types,
)
from app.scene_preview import ScenePreview, _RuntimeSceneSmartPrimitive
from app.runtime.common.io.scene_document import SceneDefinition, SmartPrimitiveInstance, save_scene_document
from motionbricks.animation.animation_buffer import AnimationBuffer
from motionbricks.animation.skeleton import Skeleton
from motionbricks.model.contract import MODEL_POSES_PER_TOKEN, ModelRig
from motionbricks.runtime.pivot import Axis, ConeSide, ConstraintType
from motionbricks.runtime.smart_primitive import (
    CharacterProperties,
    InteractionBox,
    SmartPrimitive,
    SmartPrimitiveAsset,
    compute_interaction_box_overlaps,
)
from motionbricks.runtime.target_pose import (
    IKPin,
    TargetPoseInferenceParams,
)


def _transform(values) -> wp.transform:
    return wp.transform(wp.vec3(*values[:3]), wp.quat(*values[3:]))


def _transform_values(transform: wp.transform) -> np.ndarray:
    return np.asarray(
        [float(transform[index]) for index in range(7)],
        dtype=np.float32,
    )


def _interaction_box_array(
    position,
    half_extents=(0.5, 0.5, 0.5),
    rotation=None,
) -> wp.array:
    box = InteractionBox()
    box.transform = wp.transform(
        wp.vec3(*position),
        rotation if rotation is not None else wp.quat(0.0, 0.0, 0.0, 1.0),
    )
    box.half_extents = wp.vec3(*half_extents)
    return wp.array([box], dtype=InteractionBox, device="cpu")


class RuntimeSmartPrimitiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        wp.init()
        cls.previous_device = wp.get_device()
        wp.set_device("cpu")

    @classmethod
    def tearDownClass(cls) -> None:
        wp.set_device(cls.previous_device)

    def _build_authoring(self):
        reference_pose = np.asarray(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        skeleton = Skeleton(
            ["Hips", "Chest"],
            [-1, 0],
            reference_pose,
        )
        self.skeleton = skeleton
        clip_poses = np.repeat(reference_pose[None, ...], 10, axis=0)
        clip_poses[:, 0, 0] = 2.0 + np.arange(10, dtype=np.float32)
        clip_poses[:, 0, 1] = 0.5
        clip_poses[:, 0, 2] = 4.0 + 0.5 * np.arange(10, dtype=np.float32)
        animation = AnimationBuffer(skeleton, 30.0, wp.array(clip_poses, dtype=wp.transform))
        clip = ClipAsset(Path("runtime_test.bvh"), animation)

        model_to_runtime = wp.transform(
            wp.vec3(0.0, 0.0, 0.0),
            wp.quat_rpy(0.5 * wp.pi, 0.0, 0.0),
        )
        authoring = SmartPrimitiveDocument("soma", skeleton.reference_local_transforms, model_to_runtime)
        authoring.root.local_xform = wp.transform(
            wp.vec3(1.0, 2.0, 3.0),
            wp.quat_rpy(0.0, 0.0, 0.2),
        )

        approach = authoring.add_node()
        approach.name = "Approach"
        approach.local_xform = wp.transform(
            wp.vec3(4.0, 0.0, 0.0),
            wp.quat_rpy(0.0, 0.1, 0.0),
        )
        approach_target = authoring.add_target_pose(approach)
        approach_target.set_clip(clip)
        approach_target.set_time(1 / 30.0)
        approach_target.set_num_poses_to_use(3)
        approach_target.transition_early_by_frames = 7
        approach_target.ik_pin_mask = IKPin.LEFT_HAND | IKPin.RIGHT_FOOT

        contact = authoring.add_node(approach)
        contact.name = "Contact"
        contact.local_xform = wp.transform(
            wp.vec3(0.0, 5.0, 0.0),
            wp.quat_rpy(0.1, 0.0, 0.0),
        )
        contact_target = authoring.add_target_pose(contact)
        contact_target.set_clip(clip)
        contact_target.set_time(2 / 30.0)
        contact_target.constrain_hips(True)
        contact_target.set_num_poses_to_use(2)
        contact_target.transition_early_by_frames = 11
        contact_target.set_token_range(7, 9)
        contact_target.ik_pin_mask = IKPin.RIGHT_HAND | IKPin.LEFT_FOOT

        box = authoring.add_interaction_box(contact)
        box.set_dimensions(2.0, 4.0, 6.0)
        sequence = authoring.add_sequence()
        sequence.name = "Contact Then Approach"
        authoring.add_sequence_node(sequence, contact)
        authoring.add_sequence_node(sequence, approach)
        return authoring, approach, approach_target, contact, contact_target

    def _build_data(self):
        authoring, approach, approach_target, contact, contact_target = self._build_authoring()
        data = build_runtime_smart_primitive_data(authoring)
        return data, authoring, approach, approach_target, contact, contact_target

    def test_prepared_data_preserves_target_and_sequence_settings(self) -> None:
        data, *_ = self._build_data()
        self.assertEqual(data.rig, "soma")
        self.assertEqual(data.target_poses.shape, (8, 2, 7))
        self.assertEqual(data.target_inference_params, (
            TargetPoseInferenceParams(
                transition_early_by_frames=7, num_poses_to_use=3,
                ik_pin_mask=int(IKPin.LEFT_HAND | IKPin.RIGHT_FOOT),
                allowed_token_counts=tuple(range(6, 15)),
            ),
            TargetPoseInferenceParams(
                transition_early_by_frames=11, num_poses_to_use=2,
                ik_pin_mask=int(IKPin.RIGHT_HAND | IKPin.LEFT_FOOT),
                allowed_token_counts=(7, 8, 9),
            ),
        ))
        np.testing.assert_array_equal(data.node_parent_indices, [-1, 0, 1])
        np.testing.assert_allclose(data.box_half_extents, [[1, 2, 3]])
        self.assertEqual(len(data.sequences), 1)
        sequence = data.sequences[0]
        self.assertEqual(sequence.name, "Contact Then Approach")
        self.assertEqual(sequence.node_indices, (2, 1))
        self.assertAlmostEqual(sequence.start_speed, np.sqrt(30**2 + 15**2), places=4)

    def test_runtime_retrieves_newton_local_and_world_target_poses(self) -> None:
        data, authoring, approach, approach_target, contact, contact_target = (
            self._build_data()
        )
        runtime = SmartPrimitive(SmartPrimitiveAsset(data, device="cpu"))
        output = wp.empty((6, 2), dtype=wp.transform, device="cpu")
        for node, component in ((approach, approach_target), (contact, contact_target)):
            node_index = runtime.node_index(node.name)
            sampled = component.sampled_poses.numpy()
            if component.hips_constrained:
                np.testing.assert_allclose(sampled[0, 0, [0, 2]], 0.0, atol=1.0e-6)

            converted_hips = [
                wp.mul(authoring.model_to_runtime, _transform(sampled[frame, 0]))
                for frame in range(4)
            ]
            runtime.get_local_target_poses(node_index, output)
            local_output = output.numpy()[:4]
            for frame in range(4):
                expected = wp.mul(node.local_xform, converted_hips[frame])
                np.testing.assert_allclose(
                    local_output[frame, 0],
                    _transform_values(expected),
                    atol=1.0e-5,
                )
            np.testing.assert_allclose(local_output[:, 1], sampled[:, 1], atol=1.0e-6)

            runtime.get_world_target_poses(node_index, output)
            world_output = output.numpy()[:4]
            node_world = authoring.world_xform(node)
            for frame in range(4):
                expected = wp.mul(node_world, converted_hips[frame])
                np.testing.assert_allclose(
                    world_output[frame, 0],
                    _transform_values(expected),
                    atol=1.0e-5,
                )
            np.testing.assert_allclose(world_output[:, 1], sampled[:, 1], atol=1.0e-6)

    def test_json_asset_pool_preserves_runtime_order_and_shares_clips(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            authoring, approach, approach_target, contact, contact_target = self._build_authoring()
            clip_path = root / "source.bvh"
            clip_path.write_text(
                "HIERARCHY\nROOT Hips\n{\n  OFFSET 0 0 0\n"
                "  CHANNELS 6 Xposition Yposition Zposition Xrotation Yrotation Zrotation\n"
                "  JOINT Chest\n  {\n    OFFSET 0 100 0\n"
                "    CHANNELS 3 Xrotation Yrotation Zrotation\n  }\n}\n"
                "MOTION\nFrames: 10\nFrame Time: 0.03333333333333333\n"
                + "\n".join(f"{200 + 100*i} 50 {400 + 50*i} 0 0 0 0 0 0" for i in range(10))
                + "\n",
                encoding="utf-8",
            )
            clip = ClipAsset(clip_path, approach_target.clip.animation)
            approach_target.clip = contact_target.clip = clip
            approach_target.show_trajectory = True
            # A parent created last must precede its children at runtime.
            parent = authoring.add_node()
            parent.name = "Late Parent"
            parent.local_xform = wp.transform(
                wp.vec3(-2.0, 3.0, 1.0), wp.quat_rpy(0.2, 0.0, -0.4),
            )
            authoring.reparent(approach, parent)
            authoring.add_pivot(parent)
            authoring.set_sequence_pivot(authoring.sequences[0], parent)
            variation = authoring.add_sequence()
            variation.name = "Return"
            for node in (approach, contact, approach):
                authoring.add_sequence_node(variation, node)
            authoring.sequences.reverse()
            camera = {
                "pos": [1, 2, 3], "pivot": [0, 0, 0], "pitch": -20, "yaw": 45,
                "fov": 45, "near": 0.01, "far": 1000,
            }
            document_path = save_smart_primitive_document(authoring, root / "first", approach.id, camera)
            other_path = save_smart_primitive_document(authoring, root / "second", contact.id, camera)
            expected_asset = SmartPrimitiveAsset(build_runtime_smart_primitive_data(authoring), device="cpu")
            expected_runtime = SmartPrimitive(expected_asset)
            registry = ClipRegistry(ModelRig.SOMA, skeleton=self.skeleton)
            pool = SmartPrimitiveAssetPool(registry, authoring.reference_pose, authoring.model_to_runtime)

            with patch.object(TargetPoseComponent, "_update_trajectory_samples") as sample_trajectory:
                runtime = pool.instantiate(document_path)
                second = pool.instantiate(document_path)
                other = pool.instantiate(other_path)
            sample_trajectory.assert_not_called()
            self.assertIs(runtime.asset, second.asset)
            self.assertIsNot(runtime.asset, other.asset)

            self.assertEqual(
                tuple(runtime.node_name(index) for index in range(runtime.node_count)),
                ("Root", "Late Parent", "Approach", "Contact"),
            )
            np.testing.assert_array_equal(runtime.node_parent_indices.numpy(), [-1, 0, 1, 2])
            np.testing.assert_allclose(
                runtime.node_local_transforms.numpy(), expected_runtime.node_local_transforms.numpy(),
            )
            self.assertEqual(runtime.pivot_node_indices, expected_runtime.pivot_node_indices)
            for node_index in runtime.pivot_node_indices:
                self.assertEqual(runtime.pivot(node_index), expected_runtime.pivot(node_index))
            self.assertEqual(
                [runtime.sequence_name(index) for index in range(runtime.sequence_count)],
                ["Return", "Contact Then Approach"],
            )
            output = wp.empty((12, 2), dtype=wp.transform, device="cpu")
            expected_output = wp.empty_like(output)
            for index in range(runtime.sequence_count):
                self.assertEqual(
                    runtime.target_names_for_sequence(index), expected_runtime.target_names_for_sequence(index),
                )
                self.assertAlmostEqual(
                    runtime.sequence_start_speed(index), expected_runtime.sequence_start_speed(index), places=4,
                )
                self.assertEqual(
                    runtime.sequence_pivot_node_index(index), expected_runtime.sequence_pivot_node_index(index),
                )
                params = runtime.get_world_target_poses_for_sequence(index, output)
                self.assertEqual(params, expected_runtime.get_world_target_poses_for_sequence(index, expected_output))
                frames = len(params) * MODEL_POSES_PER_TOKEN
                np.testing.assert_allclose(output.numpy()[:frames], expected_output.numpy()[:frames], atol=1.0e-5)
            boxes = wp.empty(runtime.interaction_box_count, dtype=InteractionBox, device="cpu")
            expected_boxes = wp.empty_like(boxes)
            runtime.get_world_interaction_boxes(boxes)
            expected_runtime.get_world_interaction_boxes(expected_boxes)
            for field in ("transform", "half_extents"):
                np.testing.assert_allclose(boxes.numpy()[field], expected_boxes.numpy()[field], atol=1.0e-5)

    def test_velocity_matching_behavior_selects_the_closest_sequence_speed(
        self,
    ) -> None:
        authoring, approach, _, _, contact_target = self._build_authoring()
        contact_target.set_num_poses_to_use(1)
        alternate = authoring.add_sequence()
        alternate.name = "Approach Only"
        authoring.add_sequence_node(alternate, approach)
        data = build_runtime_smart_primitive_data(authoring)
        runtime = VelocityMatchingSequenceBehavior(SmartPrimitiveAsset(data, device="cpu"))
        output = wp.empty(
            (
                runtime.max_sequence_length * MODEL_POSES_PER_TOKEN,
                runtime.pose_size,
            ),
            dtype=wp.transform,
            device="cpu",
        )
        expected = wp.empty_like(output)
        stopped = CharacterProperties(
            position=np.zeros(3, dtype=np.float32),
            velocity=np.zeros(3, dtype=np.float32),
            facing=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            desired_velocity=np.zeros(3, dtype=np.float32),
            probe_segment=(np.zeros(3), np.zeros(3)),
        )
        self.assertEqual(runtime.sequence_start_speed(0), 0.0)
        alternate_speed = runtime.sequence_start_speed(1)

        # Speeds above every sequence pick the fastest one. A tie keeps the first.
        for speed, sequence in (
            (0.0, 0), (alternate_speed, 1), (2.0 * alternate_speed, 1), (0.5 * alternate_speed, 0),
        ):
            with self.subTest(speed=speed):
                properties = replace(stopped, velocity=np.array([speed, 0.0, 0.0]))
                params = runtime.get_targets(properties, output)
                self.assertEqual(params, runtime.get_world_target_poses_for_sequence(sequence, expected))
                frames = len(params) * MODEL_POSES_PER_TOKEN
                np.testing.assert_allclose(output.numpy()[:frames], expected.numpy()[:frames])

    def test_every_behavior_subclass_is_found_by_class_name(self) -> None:
        class _TestLeaningBehavior(SmartPrimitiveBehavior):
            pass

        class _TestLongSittingBehavior(SittingBehavior):
            pass

        types = behavior_types()
        self.assertIs(types["SittingBehavior"], SittingBehavior)
        self.assertIs(types["_TestLeaningBehavior"], _TestLeaningBehavior)
        self.assertIs(types["_TestLongSittingBehavior"], _TestLongSittingBehavior)

    def test_node_transforms_can_be_set_in_local_or_world_space(self) -> None:
        data, _, approach, _, contact, _ = self._build_data()
        runtime = SmartPrimitive(SmartPrimitiveAsset(data, device="cpu"))
        world_transforms_output = wp.empty(
            runtime.node_count + 2,
            dtype=wp.transform,
            device="cpu",
        )

        new_approach_local = wp.transform(
            wp.vec3(-2.0, 3.0, 4.0),
            wp.quat_rpy(0.2, -0.3, 0.4),
        )
        runtime.set_node_local_transform(
            runtime.node_index(approach.name),
            new_approach_local,
        )
        runtime.compute_world_transforms(world_transforms_output)
        world_transforms_numpy = world_transforms_output.numpy()
        root = _transform(runtime.node_local_transforms.numpy()[0])
        expected_approach_world = wp.mul(root, new_approach_local)
        np.testing.assert_allclose(
            world_transforms_numpy[runtime.node_index(approach.name)],
            _transform_values(expected_approach_world),
            atol=1.0e-5,
        )

        desired_contact_world = wp.transform(
            wp.vec3(8.0, 9.0, 10.0),
            wp.quat_rpy(-0.1, 0.2, -0.3),
        )
        runtime.set_node_world_transform(
            runtime.node_index(contact.name),
            desired_contact_world,
        )
        runtime.compute_world_transforms(world_transforms_output)
        np.testing.assert_allclose(
            world_transforms_output.numpy()[runtime.node_index(contact.name)],
            _transform_values(desired_contact_world),
            atol=1.0e-5,
        )

        expected_local = wp.mul(
            wp.transform_inverse(expected_approach_world),
            desired_contact_world,
        )
        np.testing.assert_allclose(
            runtime.node_local_transforms.numpy()[runtime.node_index(contact.name)],
            _transform_values(expected_local),
            atol=1.0e-5,
        )

    def test_interaction_box_returns_collision_ready_box(self) -> None:
        data, authoring, _, _, contact, _ = self._build_data()
        runtime = SmartPrimitive(SmartPrimitiveAsset(data, device="cpu"))
        node_index = runtime.node_index(contact.name)
        output_box = wp.empty(3, dtype=InteractionBox, device="cpu")

        runtime.get_local_interaction_box(node_index, output_box)
        local_box = output_box.numpy()
        np.testing.assert_allclose(
            local_box["transform"][0],
            _transform_values(contact.local_xform),
            atol=1.0e-6,
        )
        np.testing.assert_allclose(local_box["half_extents"][0], [1.0, 2.0, 3.0])

        runtime.get_world_interaction_box(node_index, output_box)
        world_box = output_box.numpy()
        np.testing.assert_allclose(
            world_box["transform"][0],
            _transform_values(authoring.world_xform(contact)),
            atol=1.0e-5,
        )

        runtime.get_world_interaction_boxes(output_box)
        world_boxes = output_box.numpy()
        np.testing.assert_allclose(
            world_boxes["transform"][0],
            _transform_values(authoring.world_xform(contact)),
            atol=1.0e-5,
        )
        np.testing.assert_allclose(
            world_boxes["half_extents"][0],
            [1.0, 2.0, 3.0],
        )

    def test_interaction_box_overlap_helper_processes_one_batch(self) -> None:
        near_box = InteractionBox()
        near_box.transform = wp.transform_identity()
        near_box.half_extents = wp.vec3(0.5, 0.5, 0.5)
        far_box = InteractionBox()
        far_box.transform = wp.transform(
            wp.vec3(3.0, 0.0, 0.0),
            wp.quat_identity(),
        )
        far_box.half_extents = wp.vec3(0.5, 0.5, 0.5)
        boxes = wp.array(
            [near_box, far_box],
            dtype=InteractionBox,
            device="cpu",
        )
        results = wp.empty(3, dtype=wp.int32, device="cpu")
        for character_box, expected in (
            (_interaction_box_array((0.0, 0.0, 0.0)), [1, 0]),
            (_interaction_box_array((1.0, 0.0, 0.0)), [1, 0]),  # Face contact.
            (_interaction_box_array((2.0, 0.0, 0.0)), [0, 1]),
            (_interaction_box_array((5.0, 0.0, 0.0)), [0, 0]),
            (_interaction_box_array(
                (0.0, 0.0, 0.0), half_extents=(0.75, 0.25, 0.5),
                rotation=wp.quat_rpy(0.0, 0.0, 0.25 * wp.pi),
            ), [1, 0]),
        ):
            compute_interaction_box_overlaps(boxes, character_box, results)
            np.testing.assert_array_equal(results.numpy()[:2], expected)
        with self.assertRaisesRegex(ValueError, "at least 2 results"):
            compute_interaction_box_overlaps(
                boxes,
                character_box,
                wp.empty(1, dtype=wp.int32, device="cpu"),
            )

    def test_interaction_box_overlap_is_exact_for_parallel_boxes_on_every_device(self) -> None:
        # Walking along an axis keeps the probe box parallel to an overhead box.
        overhead = InteractionBox()
        overhead.transform = wp.transform(
            wp.vec3(3.1405947, 12.562379, 1.5375934), wp.quat(0.0, 0.0, -0.70710677, 0.70710677),
        )
        overhead.half_extents = wp.vec3(0.19628717, 0.19628717, 0.39257434)
        for device in wp.get_devices():
            for half_height, expected in ((0.63, 1), (0.55, 0)):
                with self.subTest(device=str(device), half_height=half_height):
                    probe = InteractionBox()
                    probe.transform = wp.transform(
                        wp.vec3(3.0434444, 12.853794, half_height), wp.quat(0.0, 0.0, -0.7071068, 0.70710677),
                    )
                    probe.half_extents = wp.vec3(0.395, 0.5872061, half_height)
                    results = wp.empty(1, dtype=wp.int32, device=device)
                    compute_interaction_box_overlaps(
                        wp.array([overhead], dtype=InteractionBox, device=device),
                        wp.array([probe], dtype=InteractionBox, device=device),
                        results,
                    )
                    self.assertEqual(results.numpy()[0], expected)

    def test_scene_preview_checks_a_scene_before_building_its_engine(self) -> None:
        camera = {
            "pos": [1, 2, 3], "pivot": [0, 0, 0], "pitch": -20, "yaw": 45,
            "fov": 45, "near": 0.01, "far": 1000,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            authoring = SmartPrimitiveDocument(
                ModelRig.SOMA, wp.array([wp.transform_identity()], dtype=wp.transform), wp.transform_identity(),
            )
            reference = save_smart_primitive_document(authoring, root / "soma", authoring.root.id, camera)
            preview = ScenePreview.__new__(ScenePreview)
            current = object()
            preview._scene_session = current
            preview._rig_for = Mock()
            for reference_path, behavior, message in (
                (
                    reference, "SittingBehavior",
                    "soma.smart_primitive.json was saved for the 'soma' rig, but this scene uses 'unitree_g1'",
                ),
                (reference, "unknown", "uses unknown behavior 'unknown'"),
                (None, "SittingBehavior", "has no reference"),
            ):
                with self.subTest(message=message):
                    scene = SceneDefinition(rig=ModelRig.UNITREE_G1)
                    instance = scene.add_smart_primitive()
                    instance.reference, instance.behavior = reference_path, behavior
                    preview._pending_scene_path = save_scene_document(scene, root / "scene", 1, camera)
                    preview.scene_error = None
                    preview._load_pending_scene()
                    self.assertIn(message, preview.scene_error)
                    self.assertIs(preview._scene_session, current)
            preview._rig_for.assert_not_called()

    def test_scene_preview_draws_target_poses_in_the_instance_color(self) -> None:
        data, *_ = self._build_data()
        definition = SmartPrimitiveInstance(id=1, name="Primitive", target_pose_color=wp.vec3(0.2, 0.55, 1.0))
        instance = _RuntimeSceneSmartPrimitive(
            definition, SmartPrimitive(SmartPrimitiveAsset(data, device="cpu")), 0,
            self.skeleton.reference_local_transforms, Mock(), None,
        )
        np.testing.assert_array_equal(instance.target_pose_colors.numpy(), [list(definition.target_pose_color)])

    def test_scene_preview_interacts_with_the_nearest_overlapping_primitive(self) -> None:
        data, *_ = self._build_data()
        instances = tuple(
            _RuntimeSceneSmartPrimitive(
                SmartPrimitiveInstance(
                    id=index + 1,
                    name=f"Primitive {index}",
                    transform=wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity()),
                ),
                SmartPrimitive(SmartPrimitiveAsset(data, device="cpu")),
                index,
                self.skeleton.reference_local_transforms,
                Mock(),
                None,
            )
            for index, x in enumerate((0.0, 1.0, 10.0))
        )
        character = SimpleNamespace(
            engine=SimpleNamespace(stream=None),
            # The probe touches the first two interaction boxes but not the third.
            probe_box=_interaction_box_array((4.5, 5.0, 0.0)),
            character_ground_position=np.array([9.0, 0.0, 0.0], dtype=np.float32),
        )
        preview = ScenePreview.__new__(ScenePreview)
        preview._active_interaction = None
        preview._scene_session = SimpleNamespace(
            character=character,
            smart_primitives=instances,
            interaction_boxes=wp.empty(3, dtype=InteractionBox),
            interaction_overlap_results=wp.empty(3, dtype=wp.int32),
            interaction_box_count=3,
        )

        # The third root is the closest, but only primitives whose boxes overlap the probe count.
        preview._update_smart_primitive_interaction()
        self.assertEqual([instance.interactable for instance in instances], [True, True, False])
        self.assertIs(preview._nearest_interactable, instances[1])
        character.character_ground_position = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        preview._update_smart_primitive_interaction()
        self.assertIs(preview._nearest_interactable, instances[0])

    def test_prepared_pivot_rules_keep_limits_fixed(self) -> None:
        authoring, approach, _, contact, _ = self._build_authoring()
        approach.name = "Rail"
        authoring.root.local_xform = approach.local_xform = wp.transform_identity()
        rules = authoring.add_pivot(approach).pivot
        rules.translation_axis = Axis.X
        rules.translation_constraint = ConstraintType.LINE_PROJECTION
        rules.line_min_offset, rules.line_max_offset = -1.0, 1.0
        rules.rotation_axis = Axis.Z
        rules.rotation_constraint = ConstraintType.CONE
        rules.cone_side = ConeSide.BOTH
        authoring.set_sequence_pivot(authoring.sequences[0], approach)
        authoring.add_sequence_node(authoring.add_sequence(), contact)  # No new alignment.

        primitive = SmartPrimitive(SmartPrimitiveAsset(build_runtime_smart_primitive_data(authoring), device="cpu"))
        node = primitive.sequence_pivot_node_index(0)
        self.assertEqual(node, primitive.node_index("Rail"))
        self.assertNotEqual(node, approach.id)
        self.assertEqual(primitive.pivot_node_indices, (node,))
        self.assertEqual(primitive.sequence_pivot_node_index(1), -1)
        self.assertEqual(primitive.pivot(node), rules)
        self.assertIsNot(primitive.pivot(node), rules)

        properties = CharacterProperties(
            position=np.zeros(3), velocity=np.zeros(3), facing=np.array([1, 0, 0]),
            desired_velocity=np.array([0, 1, 0]),
            probe_segment=(np.zeros(3), np.zeros(3)),
        )
        transforms = wp.empty(primitive.node_count, dtype=wp.transform, device="cpu")
        for x, expected in ((0.75, 0.75), (-4.0, -1.0), (4.0, 1.0)):
            properties.position[:] = (x, -3.0, 0.0)
            properties.probe_segment[0][:] = (x, -2.85, 0.0)
            properties.probe_segment[1][:] = (x, -2.0, 0.0)
            primitive.align_sequence_pivot(0, properties)
            primitive.compute_world_transforms(transforms)
            np.testing.assert_allclose(transforms.numpy()[node, :3], [expected, 0, 0], atol=1e-6)

        properties.desired_velocity[:] = (1, 0, 0)
        primitive.align_sequence_pivot(0, properties)
        primitive.compute_world_transforms(transforms)
        aligned = transforms.numpy()[node].copy()
        np.testing.assert_allclose(
            wp.quat_rotate(wp.quat(*aligned[3:]), wp.vec3(0, 1, 0)),
            [np.sqrt(0.5), np.sqrt(0.5), 0.0], atol=1e-6,
        )
        properties.desired_velocity[:] = (-1, 0, 0)
        primitive.align_sequence_pivot(1, properties)
        primitive.compute_world_transforms(transforms)
        np.testing.assert_array_equal(transforms.numpy()[node], aligned)

    def test_preparation_rejects_duplicate_names_and_targets_without_clips(self) -> None:
        authoring, approach, _, contact, _ = self._build_authoring()
        contact.name = approach.name
        with self.assertRaisesRegex(
            SmartPrimitiveDocumentError,
            "must be unique",
        ):
            build_runtime_smart_primitive_data(authoring)

        empty_authoring = SmartPrimitiveDocument(
            authoring.rig,
            authoring.reference_pose,
            authoring.model_to_runtime,
        )
        node = empty_authoring.add_node()
        empty_authoring.add_target_pose(node)
        with self.assertRaisesRegex(
            SmartPrimitiveDocumentError,
            "has no clip",
        ):
            build_runtime_smart_primitive_data(empty_authoring)

        sequence_authoring = SmartPrimitiveDocument(
            authoring.rig,
            authoring.reference_pose,
            authoring.model_to_runtime,
        )
        sequence_authoring.add_sequence()
        with self.assertRaisesRegex(
            SmartPrimitiveDocumentError,
            "must contain at least one Target Pose",
        ):
            build_runtime_smart_primitive_data(sequence_authoring)

    def test_empty_primitive_keeps_its_pose_size_and_writes_no_boxes(self) -> None:
        authoring, _, _, _, _ = self._build_authoring()
        empty = SmartPrimitiveDocument(
            authoring.rig,
            authoring.reference_pose,
            authoring.model_to_runtime,
        )
        data = build_runtime_smart_primitive_data(empty)
        runtime = SmartPrimitive(SmartPrimitiveAsset(data, device="cpu"))

        self.assertEqual(runtime.pose_size, self.skeleton.num_joints)
        runtime.get_world_interaction_boxes(
            wp.empty(0, dtype=InteractionBox, device="cpu")
        )

    def test_asset_pool_rejects_invalid_pivot_rules(self) -> None:
        authoring, *_ = self._build_authoring()
        pivot_only = SmartPrimitiveDocument(authoring.rig, authoring.reference_pose, authoring.model_to_runtime)
        pivot_only.add_pivot(pivot_only.root)
        camera = {
            "pos": [1, 2, 3], "pivot": [0, 0, 0], "pitch": -20, "yaw": 45,
            "fov": 45, "near": 0.01, "far": 1000,
        }
        pool = SmartPrimitiveAssetPool(
            ClipRegistry(ModelRig.SOMA, skeleton=self.skeleton), authoring.reference_pose, authoring.model_to_runtime,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = save_smart_primitive_document(pivot_only, Path(directory) / "pivot", pivot_only.root.id, camera)
            original = path.read_text("utf-8")
            for field, value, message in (
                ("motion_direction", [0, 0, 0], "non-zero"),
                ("operation_order", "bad", "valid OperationOrder"),
                ("translation_axis", "bad", "valid Axis"),
                ("line_min_offset", 2.0, "Min Offset <= Max Offset"),
                ("cone_half_angle_degrees", 181, "between 0 and 180"),
            ):
                with self.subTest(field=field):
                    document = json.loads(original)
                    document["root"]["components"][0][field] = value
                    path.write_text(json.dumps(document), encoding="utf-8")
                    with self.assertRaisesRegex(SmartPrimitiveDocumentError, message):
                        pool.load(path)


if __name__ == "__main__":
    unittest.main()
