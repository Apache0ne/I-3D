# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import warp as wp

from app.runtime.common.clip_registry import ClipAsset, ClipRegistry
from app.runtime.common.io.smart_primitive_document import (
    SmartPrimitiveDocument,
    build_runtime_smart_primitive_data,
)
from app.runtime.soma.character import SOMACharacter
from app.runtime.common.motion import AnimationClip
from app.runtime.soma.presets import DEFAULT_LOCOMOTION_MODES
from app.runtime.common.state_machine import StateMachine
from app.runtime.common.smart_primitive_behaviors import (
    SmartPrimitiveBehavior,
    VelocityMatchingSequenceBehavior,
)
from app.scene_preview import ScenePreview
from app.utils.input import InputManager
from motionbricks.animation.animation_buffer import AnimationBuffer
from motionbricks.runtime.motion_recorder import MotionRecorder
from motionbricks.animation.skeleton import Skeleton
from motionbricks.model.contract import ModelRig
from motionbricks.runtime.motion_player import GeneratedMotionPlayer
from motionbricks.runtime.pivot import Axis
from motionbricks.runtime.smart_primitive import SmartPrimitiveAsset


class _SequenceClipCycle(SmartPrimitiveBehavior):
    """Drive the real interaction state through sequences and baked playback."""

    required_sequences = ("Entry", "Exit")
    clip_paths = (Path("synthetic_loop.bvh"),)

    def enter(self):
        self.interaction_state.play_target_pose_sequence("Entry", on_complete=self._play_loop)

    def _play_loop(self):
        self.interaction_state.play_clip(self.clip_paths[0], loop=True)

    def interact(self):
        if self.interaction_state.is_looping:
            self.interaction_state.play_target_pose_sequence("Exit")


class _OrderedTargets(SmartPrimitiveBehavior):
    """Exercise a multi-target selection, independent of hierarchy order."""

    def get_targets(self, properties, output, stream=None):
        first = self.get_world_target_poses_for_sequence(
            self.sequence_index("Exit"), output[:4], stream,
        )[0]
        second = self.get_world_target_poses_for_sequence(
            self.sequence_index("Entry"), output[4:12], stream,
        )
        return replace(first, transition_early_by_frames=7), *second


class SmartPrimitiveInteractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.previous_device = wp.get_device()
        wp.set_device("cpu")
        reference = np.array([
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0.5, 0, 0, 0, 0, 1],
        ], dtype=np.float32)
        cls.skeleton = Skeleton(["Hips", "Child"], [-1, 0], reference)
        poses = np.repeat(reference[None], 12, axis=0)
        poses[:4, 0, 0] = np.arange(4, dtype=np.float32) / 30.0
        poses[4:8, 0, 0] = 1.0
        poses[8:, 0, 0] = 2.0
        clip = ClipAsset(
            Path("synthetic_targets.bvh"), AnimationBuffer(cls.skeleton, 30.0, wp.array(poses, dtype=wp.transform)),
        )
        authoring = SmartPrimitiveDocument("soma", cls.skeleton.reference_local_transforms, wp.transform_identity())
        guide = authoring.add_node()
        guide.name = "Guide"
        pivot = authoring.add_pivot(guide).pivot
        pivot.motion_direction = (-1.0, 0.0, 0.0)
        pivot.rotation_axis = Axis.Z
        unrelated = authoring.add_node()
        unrelated.name = "Unrelated"

        targets = []
        # Approach keeps the default token range.
        for name, frame, poses_to_use, early_frames, tokens in (
            ("Approach", 0, 4, 3, (6, 14)), ("Hold", 4, 1, 0, (8, 8)), ("Depart", 8, 4, 0, (7, 12)),
        ):
            node = authoring.add_node(guide)
            node.name = name
            target = authoring.add_target_pose(node)
            target.set_clip(clip)
            target.set_time(frame / clip.animation.sample_rate)
            target.set_num_poses_to_use(poses_to_use)
            target.transition_early_by_frames = early_frames
            target.set_token_range(*tokens)
            targets.append(node)
        entry = authoring.add_sequence()
        entry.name = "Entry"
        for node in targets[:2]:
            authoring.add_sequence_node(entry, node)
        authoring.set_sequence_pivot(entry, guide)
        exit_sequence = authoring.add_sequence()
        exit_sequence.name = "Exit"
        authoring.add_sequence_node(exit_sequence, targets[2])
        cls.asset = SmartPrimitiveAsset(build_runtime_smart_primitive_data(authoring), device="cpu")

    @classmethod
    def tearDownClass(cls):
        wp.set_device(cls.previous_device)

    def setUp(self):
        self.machine = StateMachine(
            tuple(SimpleNamespace(definition=mode) for mode in DEFAULT_LOCOMOTION_MODES),
            standing_idle_clips=(object(),),
            crouching_idle_clips=(object(),),
            pose=self.skeleton.reference_local_transforms,
        )
        self.machine.standing_idle_state.enter = Mock()
        self.machine.standing_locomotion_state.enter = Mock()
        self.generated = []
        self.character = SimpleNamespace(
            skeleton=self.skeleton,
            clip_registry=ClipRegistry(ModelRig.SOMA, skeleton=self.skeleton),
            current_pose=wp.empty_like(self.skeleton.reference_local_transforms),
            engine=SimpleNamespace(device=wp.get_device("cpu"), stream=None),
            movement=SimpleNamespace(
                input_speed=0.0,
                current_facing=np.array([1.0, 0.0, 0.0], dtype=np.float32),
                desired_velocity=np.zeros(3, dtype=np.float32),
            ),
            character_ground_position=np.zeros(3, dtype=np.float32),
            actual_velocity=np.zeros(3, dtype=np.float32),
            probe_segment=(np.zeros(3), np.zeros(3)),
            clear_ik_pins=Mock(),
            face_pose=Mock(),
            time_to_transition=Mock(return_value=0.5),
            start_clip=Mock(),
            clip_ends_within=Mock(return_value=False),
            generate_to_target_poses=Mock(side_effect=self._record_generated_target),
        )
        self.cycle = _SequenceClipCycle(self.asset)
        self.state = self.machine.interaction_state
        poses = np.repeat(self.skeleton.reference_local_transforms.numpy()[None], 3, axis=0)
        poses[1, 1, 2] = 0.2
        self.baked_clip = AnimationClip(
            animation=AnimationBuffer(self.skeleton, 30.0, wp.array(poses, dtype=wp.transform)),
            source_ground=wp.transform_identity(),
            duration=2.0 / 30.0,
            root_offset=wp.transform_identity(),
            root_positions=poses[:, 0, :3].copy(),
        )
        # Supply a real buffer without reading a BVH or any bundled motion assets.
        self.character.load_clip = Mock(return_value=self.baked_clip)

    def _record_generated_target(self, poses, params):
        self.generated.append((poses.numpy().copy(), params))
        self.character.time_to_transition.return_value = 0.5

    def reach_target(self):
        self.character.time_to_transition.return_value = 0.003
        self.state.update(self.character, self.machine, 0.016)

    def test_sequence_alignment_survives_clip_and_unaligned_followup(self):
        primitive = _SequenceClipCycle(self.asset)
        self.character.movement.current_facing[:] = (0.0, 1.0, 0.0)
        self.character.movement.desired_velocity[:] = (1.0, 0.0, 0.0)
        pivot = primitive.sequence_pivot_node_index(primitive.sequence_index("Entry"))
        primitive.set_node_world_transform(
            0, wp.transform(wp.vec3(3.0, -2.0, 0.4), wp.quat_rpy(0.0, 0.0, 0.6)),
        )
        primitive.set_node_local_transform(
            pivot, wp.transform(wp.vec3(0.3, -0.1, 0.2), wp.quat_rpy(0.0, 0.0, 0.4)),
        )
        transforms = wp.empty(primitive.node_count, dtype=wp.transform, device="cpu")
        primitive.compute_world_transforms(transforms)
        before = transforms.numpy().copy()

        self.machine.start_interaction(self.character, primitive)

        primitive.compute_world_transforms(transforms)
        aligned = transforms.numpy().copy()
        np.testing.assert_allclose(
            wp.quat_rotate(wp.quat(*aligned[pivot, 3:]), wp.vec3(1.0, 0.0, 0.0)),
            [-1.0, 0.0, 0.0], atol=1e-6,
        )
        np.testing.assert_allclose(aligned[pivot, :3], before[pivot, :3], atol=1e-6)
        for node in (0, primitive.node_index("Unrelated")):
            np.testing.assert_allclose(aligned[node], before[node], atol=1e-6)
        expected = wp.empty((8, self.skeleton.num_joints), dtype=wp.transform, device="cpu")
        primitive.get_world_target_poses_for_sequence(
            primitive.sequence_index("Entry"), expected,
        )
        np.testing.assert_allclose(self.generated[0][0], expected.numpy()[:4])

        self.reach_target()
        np.testing.assert_allclose(self.generated[1][0], expected.numpy()[4:])
        self.reach_target()
        self.assertTrue(self.state.is_looping)
        self.character.movement.current_facing[:] = (1.0, 0.0, 0.0)
        self.character.movement.desired_velocity[:] = (0.0, 1.0, 0.0)
        self.machine.interact()
        primitive.compute_world_transforms(transforms)
        np.testing.assert_allclose(transforms.numpy()[pivot], aligned[pivot], atol=1e-6)
        primitive.get_world_target_poses_for_sequence(
            primitive.sequence_index("Exit"), expected[:4],
        )
        np.testing.assert_allclose(self.generated[2][0], expected.numpy()[:4])

        self.reach_target()
        self.machine.start_interaction(self.character, primitive)
        primitive.compute_world_transforms(transforms)
        realigned = transforms.numpy()[pivot]
        np.testing.assert_allclose(
            wp.quat_rotate(wp.quat(*realigned[3:]), wp.vec3(1.0, 0.0, 0.0)),
            [0.0, -1.0, 0.0], atol=1e-6,
        )
        np.testing.assert_allclose(realigned[:3], before[pivot, :3], atol=1e-6)

    def test_velocity_matching_uses_the_authored_motion_direction(self):
        primitive = VelocityMatchingSequenceBehavior(self.asset)
        primitive.set_node_world_transform(
            0, wp.transform(wp.vec3(3.0, -2.0, 0.4), wp.quat_rpy(0.0, 0.0, 0.6)),
        )
        # Measured speed selects the moving Entry sequence; desired velocity
        # independently determines how its authored -X direction aligns.
        self.character.actual_velocity[:] = (1.0, 0.0, 0.0)
        self.character.movement.desired_velocity[:] = (3.0, 4.0, 0.0)
        self.machine.start_interaction(self.character, primitive)
        transforms = wp.empty(primitive.node_count, dtype=wp.transform, device="cpu")
        primitive.compute_world_transforms(transforms)
        pivot = transforms.numpy()[primitive.sequence_pivot_node_index(primitive.sequence_index("Entry"))]
        np.testing.assert_allclose(
            wp.quat_rotate(wp.quat(*pivot[3:]), wp.vec3(-1.0, 0.0, 0.0)),
            [0.6, 0.8, 0.0], atol=1e-6,
        )

    def test_sequence_to_looping_clip_to_sequence_waits_for_each_boundary(self):
        self.machine.start_interaction(self.character, self.cycle)
        self.machine.interact()  # Input during the entry sequence is not queued.
        self.state.update(self.character, self.machine, 0.016)
        self.assertFalse(self.state.is_looping)
        self.assertEqual(self.generated[0][1].num_poses_to_use, 4)
        self.assertEqual(self.generated[0][1].allowed_token_counts, tuple(range(6, 15)))
        self.character.start_clip.assert_not_called()

        self.reach_target()
        self.assertEqual(self.generated[1][1].num_poses_to_use, 1)
        self.assertEqual(self.generated[1][1].allowed_token_counts, (8,))
        self.assertFalse(self.state.is_looping)
        self.character.start_clip.assert_not_called()
        self.reach_target()
        self.assertTrue(self.state.is_looping)
        self.character.clear_ik_pins.assert_called()
        self.character.start_clip.assert_called_once_with(
            self.baked_clip,
            loop=True, align_to_current_pose=True,
        )
        self.character.movement.input_speed = 2.0
        for _ in range(3):
            self.reach_target()
        self.assertTrue(self.machine.is_interacting)
        self.assertEqual(len(self.generated), 2)

        self.machine.interact()
        self.machine.interact()  # Repeated input during the exit sequence is ignored.
        self.assertFalse(self.state.is_looping)
        self.assertEqual(len(self.generated), 3)
        self.assertEqual(self.generated[2][1].num_poses_to_use, 4)
        self.assertEqual(self.generated[2][1].allowed_token_counts, tuple(range(7, 13)))
        self.reach_target()
        self.reach_target()  # No stale completion callback may run again.
        self.assertIs(self.machine.state, self.machine.standing_locomotion_state)
        self.machine.standing_locomotion_state.enter.assert_called_once()
        self.assertIsNone(self.cycle._interaction_state)
        self.character.clear_ik_pins.assert_called()

    def test_default_behavior_preserves_target_order_and_inference_settings(self):
        primitive = _OrderedTargets(self.asset)
        self.machine.start_interaction(self.character, primitive)
        self.state.update(self.character, self.machine, 0.016)
        self.assertEqual(len(self.generated), 1)
        self.reach_target()
        self.assertEqual(len(self.generated), 2)
        self.assertTrue(self.machine.is_interacting)
        self.reach_target()
        self.assertEqual(len(self.generated), 3)
        self.assertTrue(self.machine.is_interacting)
        self.assertEqual(
            [(params.num_poses_to_use, params.transition_early_by_frames, params.allowed_token_counts)
             for _, params in self.generated],
            [(4, 7, tuple(range(7, 13))), (4, 3, tuple(range(6, 15))), (1, 0, (8,))],
        )
        expected = wp.empty((12, self.skeleton.num_joints), dtype=wp.transform, device="cpu")
        primitive.get_world_target_poses_for_sequence(primitive.sequence_index("Exit"), expected[:4])
        primitive.get_world_target_poses_for_sequence(primitive.sequence_index("Entry"), expected[4:])
        for index, (poses, _) in enumerate(self.generated):
            np.testing.assert_allclose(poses, expected.numpy()[index * 4 : (index + 1) * 4])
        self.reach_target()
        self.assertIs(self.machine.state, self.machine.standing_idle_state)
        self.machine.standing_idle_state.enter.assert_called_once()

    def test_non_looping_clip_returns_to_idle_when_its_end_is_within_the_step(self):
        self.machine.start_interaction(self.character, self.cycle)
        self.cycle.interaction_state.play_clip(self.cycle.clip_paths[0], loop=False)
        self.assertFalse(self.state.is_looping)
        self.character.clear_ik_pins.assert_called()
        for _ in range(3):
            self.reach_target()  # A generated target's deadline cannot end this clip.
        self.assertTrue(self.machine.is_interacting)
        self.assertEqual(len(self.generated), 1)

        self.character.clip_ends_within.return_value = True
        self.state.update(self.character, self.machine, 0.016)
        self.state.update(self.character, self.machine, 0.016)
        self.assertIs(self.machine.state, self.machine.standing_idle_state)
        self.machine.standing_idle_state.enter.assert_called_once()
        self.character.face_pose.assert_called_once_with()
        self.assertIsNone(self.cycle._interaction_state)

    def test_sequence_to_one_shot_clip_to_sequence_preserves_callbacks(self):
        completed = Mock()
        self.cycle.enter = lambda: self.state.play_target_pose_sequence(
            "Entry", on_complete=lambda: self.state.play_clip(
                self.cycle.clip_paths[0], loop=False,
                on_complete=lambda: self.state.play_target_pose_sequence("Exit", on_complete=completed),
            ),
        )
        self.machine.start_interaction(self.character, self.cycle)
        self.reach_target()
        self.character.start_clip.assert_not_called()
        self.reach_target()
        self.character.start_clip.assert_called_once_with(
            self.baked_clip, loop=False, align_to_current_pose=True,
        )
        self.character.clear_ik_pins.assert_called()
        completed.assert_not_called()
        self.state.update(self.character, self.machine, 0.016)
        self.assertEqual(len(self.generated), 2)
        self.character.clip_ends_within.return_value = True
        self.state.update(self.character, self.machine, 0.016)
        self.assertEqual(self.generated[-1][1].num_poses_to_use, 4)
        self.assertEqual(len(self.generated), 3)
        self.state.update(self.character, self.machine, 0.016)
        completed.assert_not_called()
        self.reach_target()
        self.reach_target()
        completed.assert_called_once_with()

    def test_exit_cancels_pending_callbacks_and_allows_reusing_the_primitive(self):
        self.machine.start_interaction(self.character, self.cycle)
        with self.assertRaisesRegex(RuntimeError, "another is active"):
            self.machine.start_interaction(self.character, self.cycle)
        self.machine.resume_movement(self.character)
        self.reach_target()
        self.character.start_clip.assert_not_called()
        self.assertIsNone(self.cycle._interaction_state)
        self.assertIsNone(self.state.primitive)
        self.character.clear_ik_pins.assert_called()
        self.machine.start_interaction(self.character, self.cycle)
        self.reach_target()
        self.reach_target()
        self.assertTrue(self.state.is_looping)
        completed = Mock()
        self.state.play_clip(self.cycle.clip_paths[0], loop=False, on_complete=completed)
        self.machine.resume_movement(self.character)
        self.character.clip_ends_within.return_value = True
        self.reach_target()
        completed.assert_not_called()
        self.assertIsNone(self.cycle._interaction_state)
        self.assertIsNone(self.state.primitive)

    def test_failed_entry_releases_the_primitive_and_restores_movement(self):
        self.character.generate_to_target_poses.side_effect = RuntimeError("inference failed")
        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            self.machine.start_interaction(self.character, self.cycle)
        self.assertIsNone(self.cycle._interaction_state)
        self.assertFalse(self.machine.is_interacting)
        self.character.clear_ik_pins.assert_called()
        self.assertIs(self.machine.state, self.machine.standing_idle_state)

    def test_viewer_routes_v_and_controller_x_once_and_rearms_auto_on_exit(self):
        instance = SimpleNamespace(
            definition=SimpleNamespace(id=10, name="TestInteraction"),
            runtime=self.cycle,
            interactable=True,
        )
        preview = ScenePreview.__new__(ScenePreview)
        preview._scene_session = SimpleNamespace(
            state_machine=self.machine, character=self.character,
            smart_primitives=(instance,),
        )
        key = SimpleNamespace(down=False)
        controller = SimpleNamespace(a=False, x=False, y=False, b=False)
        preview._input = InputManager(lambda _: key.down)
        preview._input._controller = controller
        preview._interaction_down = False
        preview._active_interaction = None
        preview._nearest_interactable = instance
        preview._auto_interaction_blocked = set()
        preview.auto_interaction = True
        preview._dispatch_interaction_input()
        key.down = True
        preview._dispatch_interaction_input()  # A press during entry must not skip the loop.
        self.reach_target()
        self.reach_target()
        for _ in range(3):
            preview._dispatch_interaction_input()
        self.assertTrue(self.state.is_looping)
        preview._nearest_interactable = None  # Manual input still reaches the active primitive.
        controller.x = True
        key.down = False
        preview._dispatch_interaction_input()  # Changing devices while held is not a new press.
        self.assertTrue(self.state.is_looping)
        controller.x = False
        preview._dispatch_interaction_input()
        controller.x = True
        preview._dispatch_interaction_input()
        self.assertFalse(self.state.is_looping)
        self.assertEqual(len(self.generated), 3)
        self.reach_target()
        preview._nearest_interactable = instance
        for _ in range(3):
            preview._dispatch_interaction_input()
        self.assertFalse(self.machine.is_interacting)
        self.assertIs(self.machine.state, self.machine.standing_idle_state)
        self.assertIsNone(preview._active_interaction)
        instance.interactable = False
        preview._nearest_interactable = None
        preview._dispatch_interaction_input()
        instance.interactable = True
        preview._nearest_interactable = instance
        preview._dispatch_interaction_input()
        self.assertTrue(self.machine.is_interacting)
        self.assertEqual(len(self.generated), 4)

    def test_clip_playback_preserves_alignment_history_and_the_correct_ending(self):
        source = wp.transform(wp.vec3(2.0, 1.0, -3.0), wp.quat_rpy(0.4, 0.2, 0.5))
        reached = wp.transform(wp.vec3(8.0, -2.0, 1.7), wp.quat_rpy(0.1, 0.0, 1.9))
        child = wp.transform(wp.vec3(0.5, 0.0, 0.0), wp.quat_identity())
        local = np.array([[list(source), list(child)]] * 3, dtype=np.float32)
        local[1, 0, 0] += 0.3
        local[1, 1, 2] += 0.25
        local[2, 0, 0] += 0.6
        local[2, 1, 2] += 0.2
        skeleton = SimpleNamespace(
            num_joints=2,
            reference_local_transforms=wp.array(local[0], dtype=wp.transform),
        )
        for loop in (False, True):
            with self.subTest(loop=loop):
                animation = AnimationBuffer(skeleton, 30.0, wp.array(local, dtype=wp.transform))
                character = SOMACharacter.__new__(SOMACharacter)
                character.engine = SimpleNamespace(stream=None)
                character.clip_registry = ClipRegistry(ModelRig.SOMA, skeleton=skeleton)
                character.model_to_runtime = wp.transform_identity()
                with patch.object(character.clip_registry, "load", return_value=ClipAsset(Path("test.bvh"), animation)):
                    clip = character.load_clip(Path("test.bvh"), loop=loop)
                prepared = clip.animation.poses.numpy().copy()
                if loop:
                    np.testing.assert_allclose(prepared[-1], prepared[0], atol=1e-6)
                else:
                    np.testing.assert_array_equal(prepared, local)

                character.current_pose = wp.array([reached, child], dtype=wp.transform)
                character.current_foot_contacts = wp.full(4, 0.5, dtype=wp.float32)
                character.character_ground_position = np.array([8.0, -2.0, 0.0], dtype=np.float32)
                character.actual_velocity = np.zeros(3, dtype=np.float32)
                character.state_machine = SimpleNamespace(update=Mock())
                character._update_probe_box = Mock()
                player = GeneratedMotionPlayer.__new__(GeneratedMotionPlayer)
                player.engine = character.engine
                player.motion_history = MotionRecorder(skeleton.reference_local_transforms, capacity=8)
                player.record_pose(1.0 / 30.0, character.current_pose)
                character.player = player
                history_before = player.motion_history.poses.numpy().copy()
                times_before = player.motion_history.delta_times.copy()

                character.start_clip(clip, loop=loop, align_to_current_pose=True)

                np.testing.assert_allclose(character.current_pose.numpy()[0], list(reached), atol=1e-6)
                np.testing.assert_allclose(character.current_pose.numpy()[1], list(child), atol=1e-6)
                np.testing.assert_array_equal(player.motion_history.poses.numpy(), history_before)
                np.testing.assert_array_equal(player.motion_history.delta_times, times_before)
                self.assertFalse(character.clip_ends_within(0.0))
                character.step(1.0 / 30.0, None)
                self.assertGreater(float(character.current_pose.numpy()[1, 2]), 0.1)
                # Baked clips have no predicted contacts.
                np.testing.assert_array_equal(character.current_foot_contacts.numpy(), 0.0)
                self.assertFalse(character.clip_ends_within(0.0))

                character.step(1.0 / 30.0, None)
                self.assertEqual(character.clip_ends_within(0.0), not loop)
                end_frame = 0 if loop else -1
                expected_hips = wp.mul(
                    clip.root_offset, wp.transform(*prepared[end_frame, 0]),
                )
                for _ in range(2):
                    np.testing.assert_allclose(
                        character.current_pose.numpy()[0], list(expected_hips), atol=1e-6,
                    )
                    np.testing.assert_allclose(
                        character.current_pose.numpy()[1], prepared[end_frame, 1], atol=1e-6,
                    )
                    # Looping wraps; one-shot playback holds its original final pose.
                    character.step(clip.duration, None)
                np.testing.assert_array_equal(animation.poses.numpy(), local)


if __name__ == "__main__":
    unittest.main()
