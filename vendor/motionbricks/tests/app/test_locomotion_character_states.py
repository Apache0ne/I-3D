# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import warp as wp

from app.runtime.soma.character import SOMACharacter
from app.runtime.common.clip_registry import ClipRegistry
from app.runtime.common.io.smart_primitive_document import SmartPrimitiveAssetPool
from app.runtime.common.motion import (
    LocomotionCommand,
    LocomotionMode,
    LocomotionModeRuntime,
    LocomotionSettings,
    MovementControl,
    PlaybackSource,
    load_locomotion_mode,
    ground_direction,
)
from app.runtime.common.states import IdleState, LocomotionState
from app.runtime.soma.target_pose_blender import SOMATargetPoseBlender
from app.runtime.unitree_g1.character import UnitreeG1Character
from motionbricks.animation.skeleton import Skeleton
from motionbricks.model.contract import ModelRig
from motionbricks.runtime.blend_spaces import BlendSpace, BlendSpaceType
from motionbricks.runtime.motion_player import GeneratedMotionPlayer
from motionbricks.runtime.target_pose import IKPin, TargetPoseInferenceParams


def _mode(
    name: str,
    blend_space_type: BlendSpaceType,
    max_speed: float,
    *,
    default_speed: float = 2.3,
    minimum_speed: float = 0.0,
    transition_early_by_frames: int = 8,
    allowed_token_counts: tuple[int, ...] = tuple(range(8, 15)),
    velocity_scale: float = 1.0,
    position_scale: float = 1.0,
) -> LocomotionMode:
    return LocomotionMode(
        name=name,
        smart_primitive_path=Path("unused.smart_primitive.json"),
        blend_space_type=blend_space_type,
        target_values=(("Idle", (0.0, 0.0)),),
        max_speed=max_speed,
        default_speed=default_speed,
        minimum_speed=minimum_speed,
        transition_early_by_frames=transition_early_by_frames,
        allowed_token_counts=allowed_token_counts,
        velocity_scale=velocity_scale,
        position_scale=position_scale,
    )


def _locomotion_state(
    mode: LocomotionMode,
    settings: LocomotionSettings = LocomotionSettings(),
    runtime=None,
) -> LocomotionState:
    runtime = runtime or SimpleNamespace(definition=mode, blend_space=Mock())
    pose = wp.empty(2, dtype=wp.transform, device="cpu")
    return LocomotionState("Locomotion", (runtime,), mode.name, pose, settings=settings)


def _idle_state(*clips: object) -> IdleState:
    return IdleState("Idle", clips)


class CharacterStateTests(unittest.TestCase):
    def test_idle_selection_excludes_the_previous_clip(self) -> None:
        clips = (object(), object())
        state = _idle_state(*clips)

        self.assertIs(state.choose_clip(SimpleNamespace(current_clip=clips[0])), clips[1])
        self.assertIs(state.choose_clip(SimpleNamespace(current_clip=clips[1])), clips[0])

    def test_locomotion_generation_uses_active_mode_early_transition_and_token_counts(self) -> None:
        state = _locomotion_state(
            _mode(
                "linear",
                BlendSpaceType.LINEAR_1D,
                4.5,
                transition_early_by_frames=5,
                allowed_token_counts=(6, 7, 8),
            )
        )
        state._blended_target_poses = object()
        state._target_poses = object()
        state.evaluate_current_target_poses = Mock()
        character = SimpleNamespace(
            movement=SimpleNamespace(desired_velocity=np.array([2.0, 0.0, 0.0], dtype=np.float32)),
            generate_to_target_poses=Mock(),
        )

        state._generate(character)

        target_poses, params = (
            character.generate_to_target_poses.call_args.args
        )
        self.assertIs(target_poses, state._target_poses)
        self.assertEqual(
            params,
            TargetPoseInferenceParams(
                transition_early_by_frames=5,
                num_poses_to_use=4,
                allowed_token_counts=(6, 7, 8),
            ),
        )

    def test_locomotion_mode_rejects_invalid_early_transition_token_counts_and_speeds(self) -> None:
        for options, message in (
            ({"transition_early_by_frames": -1}, "cannot be negative"),
            ({"allowed_token_counts": ()}, "cannot be empty"),
            ({"allowed_token_counts": (5, 6)}, "must be 6 to 16"),
            ({"allowed_token_counts": (16, 17)}, "must be 6 to 16"),
            ({"default_speed": 4.6}, "default_speed"),
            ({"minimum_speed": -0.1}, "minimum_speed"),
            ({"minimum_speed": 2.4}, "minimum_speed"),
            ({"minimum_speed": 4.6}, "minimum_speed"),
        ):
            with self.subTest(**options):
                with self.assertRaisesRegex(ValueError, message):
                    _mode("linear", BlendSpaceType.LINEAR_1D, 4.5, **options)


class PoseFacingTests(unittest.TestCase):
    def _character(self, character_type, current_pose):
        character = character_type.__new__(character_type)
        character.current_pose = current_pose
        character.movement = MovementControl(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        character.movement.current_facing_rate = 0.4
        return character

    def test_soma_faces_along_the_hips_y_axis_on_the_ground(self) -> None:
        # The hips turn 30 degrees and tilt forward, as when bending to stand up.
        rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi / 6.0) * wp.quat_from_axis_angle(
            wp.vec3(1.0, 0.0, 0.0), -0.4,
        )
        pose = wp.array([wp.transform(wp.vec3(1.0, 2.0, 0.9), rotation)], dtype=wp.transform, device="cpu")
        character = self._character(SOMACharacter, pose)

        np.testing.assert_allclose(character.pose_facing(), [-0.5, np.sqrt(0.75), 0.0], atol=1.0e-6)

    def test_g1_faces_along_the_pelvis_x_axis_and_face_pose_updates_every_facing(self) -> None:
        rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 2.0) * wp.quat_from_axis_angle(
            wp.vec3(0.0, 1.0, 0.0), 0.3,
        )
        joint_q = np.zeros(36, dtype=np.float32)
        joint_q[:7] = [0.0, 0.0, 0.75, *rotation]
        character = self._character(UnitreeG1Character, wp.array(joint_q, dtype=wp.float32, device="cpu"))
        movement = character.movement
        current_facing = movement.current_facing

        character.face_pose()

        expected = [np.cos(2.0), np.sin(2.0), 0.0]
        self.assertIs(movement.current_facing, current_facing)
        for facing in (movement.current_facing, movement.desired_facing, movement.last_input_facing):
            np.testing.assert_allclose(facing, expected, atol=1.0e-6)
        self.assertAlmostEqual(movement.current_facing_angle, 2.0, places=6)
        self.assertEqual(movement.current_facing_rate, 0.0)


class GroundDirectionTests(unittest.TestCase):
    def test_returns_a_unit_xy_direction_or_its_fallback(self) -> None:
        for direction, fallback, expected in (
            ((3.0, 4.0, 8.0), (1.0, 0.0, 0.0), (0.6, 0.8, 0.0)),
            ((0.0, 0.0, 0.0), (0.0, -3.0, 4.0), (0.0, -1.0, 0.0)),
        ):
            with self.subTest(direction=direction, fallback=fallback):
                result = ground_direction(np.array(direction, dtype=np.float32), np.array(fallback, dtype=np.float32))
                np.testing.assert_allclose(result, expected)
        with self.assertRaisesRegex(ValueError, "no valid ground direction"):
            ground_direction(np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32))


class LocomotionModeSwitchTests(unittest.TestCase):
    def test_mode_switch_reduces_speed_only_when_the_new_mode_is_not_faster(self) -> None:
        linear = _mode("linear", BlendSpaceType.LINEAR_1D, 4.5)
        directional = _mode("directional", BlendSpaceType.DIRECTIONAL_2D, 4.5)
        crouch = _mode("crouch", BlendSpaceType.DIRECTIONAL_2D, 3.6)
        settings = LocomotionSettings(maximum_direction_change_speed_reduction=0.25)
        for previous, current, speed, expected_speed in (
            (linear, directional, 4.5, 3.375),
            (directional, crouch, 4.5, 3.375),
            (crouch, directional, 3.6, 3.6),
        ):
            with self.subTest(previous=previous.name, current=current.name):
                movement = MovementControl(np.array([0.0, 1.0, 0.0], dtype=np.float32), settings)
                movement.desired_facing = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                movement.desired_speed = speed

                movement._ease_mode_switch(current, previous)

                self.assertAlmostEqual(movement.desired_speed, expected_speed)
                self.assertLessEqual(movement.desired_speed, current.max_speed)
                np.testing.assert_array_equal(movement.current_facing, movement.desired_facing)


class LocomotionTargetEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.previous_device = wp.get_device()
        wp.set_device("cpu")

    @classmethod
    def tearDownClass(cls):
        wp.set_device(cls.previous_device)

    def _targets(self):
        poses = np.zeros((3, 4, 2, 7), dtype=np.float32)
        poses[..., 6] = 1
        poses[0, :, 0, 0] = np.arange(4) * 0.1
        poses[1, :, 0, 0] = np.arange(4) * 0.3
        poses[0, :, 0, 2] = 1
        poses[1, :, 0, 2] = 3
        poses[0, :, 1, 0] = 0.5
        poses[1, :, 1, 0] = 1.5
        return poses, Skeleton(["Hips", "Child"], [-1, 0], poses[0, 0])

    def test_modes_blend_then_scale_speed_and_place_only_the_root(self):
        for space_type, speed, facing, height, child_x in (
            (BlendSpaceType.LINEAR_1D, 2.25, (0, 1, 0), 2, 1),
            (BlendSpaceType.DIRECTIONAL_2D, 4.5, (1, 0, 0), 3, 1.5),
        ):
            with self.subTest(space_type=space_type):
                poses, skeleton = self._targets()
                mode = _mode("test", space_type, 4.5, velocity_scale=0.9, position_scale=0.8)
                keys = ((0, 0), (1, 0)) if space_type is BlendSpaceType.LINEAR_1D else ((0, 0), (1, 0), (0, -1))
                space = BlendSpace(space_type)
                for index, key in enumerate(keys):
                    space.add_target(wp.vec2(*key), index)
                space.compile()
                runtime = LocomotionModeRuntime(
                    mode, space, wp.array(poses, dtype=wp.transform),
                )
                state = _locomotion_state(mode, runtime=runtime)
                direction = np.asarray(facing, dtype=np.float32)
                character = SimpleNamespace(
                    target_pose_blender=SOMATargetPoseBlender(skeleton.reference_local_transforms),
                    engine=SimpleNamespace(stream=None),
                    character_ground_position=np.array([10, 2, 0], dtype=np.float32),
                    movement=SimpleNamespace(current_facing=direction, desired_velocity=direction * speed),
                )
                blended = wp.empty((4, 2), dtype=wp.transform)
                target = wp.empty_like(blended)

                state.evaluate_current_target_poses(character, blended, target)

                actual = target.numpy()
                ground = character.character_ground_position + direction * speed * 0.9 * 0.8
                expected_root = ground + [0, 0, height]
                for frame in range(4):
                    np.testing.assert_allclose(actual[frame, 0, :3],
                                               expected_root + direction * speed * 0.9 * frame / 30, atol=1e-6)
                np.testing.assert_allclose(actual[:, 1, 0], child_x)
                np.testing.assert_array_equal(actual[:, 1, 1:3], 0)
                root_rotation = wp.quat(*actual[0, 0, 3:])
                np.testing.assert_allclose(wp.quat_rotate(root_rotation, wp.vec3(1, 0, 0)), direction, atol=1e-6)
                np.testing.assert_array_equal(runtime.target_poses.numpy(), poses)

    def test_mode_loader_keeps_named_target_windows_in_caller_owned_storage(self):
        poses, skeleton = self._targets()
        poses[1, :, :, 3:] *= -1
        source = wp.array(poses, dtype=wp.transform)
        mode = LocomotionMode(
            "test", Path("unused.smart_primitive.json"), BlendSpaceType.LINEAR_1D,
            (("Fast", (1, 0)), ("Slow", (0, 0))), max_speed=4.5, default_speed=2.3,
        )

        def copy_target(node_index, output):
            wp.copy(output, source[node_index])
            return TargetPoseInferenceParams(0, 4)

        primitive = SimpleNamespace(
            rig=ModelRig.SOMA,
            node_index=lambda name: {"Fast": 1, "Slow": 0}[name],
            has_target_pose=lambda index: True,
            get_local_target_poses=copy_target,
        )
        pool = SmartPrimitiveAssetPool(
            ClipRegistry(ModelRig.SOMA, skeleton=skeleton),
            skeleton.reference_local_transforms, wp.transform_identity(),
        )
        with patch.object(pool, "instantiate", return_value=primitive):
            runtime = load_locomotion_mode(mode, pool)
            directional = load_locomotion_mode(replace(mode, blend_space_type=BlendSpaceType.DIRECTIONAL_2D), pool)
        source.zero_()

        np.testing.assert_array_equal(runtime.target_poses.numpy(), poses[[1, 0]])
        expected = poses[[1, 0]].copy()
        expected[0, :, :, 3:] *= -1
        np.testing.assert_array_equal(directional.target_poses.numpy(), expected)
        np.testing.assert_array_equal(runtime.blend_space.data_indices.numpy(), [0, 1])
        sample = runtime.blend_space.evaluate(wp.vec2(0.25, 0))
        self.assertEqual(tuple(sample.indices)[:2], (1, 0))
        np.testing.assert_allclose(tuple(sample.weights)[:2], [0.75, 0.25])


class CharacterVelocityTests(unittest.TestCase):
    def test_analog_movement_scales_the_mode_maximum_speed_with_a_floor(self) -> None:
        for movement, minimum_speed, expected_speed in (
            ((0.5, 0.0, 0.0), 0.0, 2.25),
            ((0.05, 0.0, 0.0), 1.84, 1.84),
            (None, 1.84, 0.0),
        ):
            with self.subTest(movement=movement, minimum_speed=minimum_speed):
                mode = _mode("linear", BlendSpaceType.LINEAR_1D, 4.5, minimum_speed=minimum_speed)
                control = MovementControl(np.array([1.0, 0.0, 0.0], dtype=np.float32))
                control.input_speed = 1.0

                control._read_input(LocomotionCommand(
                    facing_direction=np.array([1.0, 0.0, 0.0], dtype=np.float32),
                    movement=None if movement is None else np.array(movement, dtype=np.float32),
                    use_max_speed=True,
                ), mode)

                self.assertAlmostEqual(control.input_speed, expected_speed)
                np.testing.assert_allclose(control.input_velocity_direction, [1.0, 0.0, 0.0])

    def test_realign_uses_measured_velocity_and_preserves_input(self) -> None:
        mode = _mode("linear", BlendSpaceType.LINEAR_1D, 4.5)
        velocity = np.array([0.0, 3.0, 0.0], dtype=np.float32)
        movement = MovementControl(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        movement.input_speed = movement.desired_speed = 4.5
        movement.desired_speed_rate = movement.desired_velocity_direction_rate = movement.current_facing_rate = 1.0
        movement.desired_velocity[:] = (4.5, 0.0, 0.0)

        movement.realign(velocity, mode)

        self.assertAlmostEqual(movement.desired_speed, 3.0)
        np.testing.assert_allclose(movement.desired_velocity, velocity)
        np.testing.assert_allclose(movement.current_facing, [0.0, 1.0, 0.0])
        self.assertEqual(movement.input_speed, 4.5)
        np.testing.assert_allclose(movement.input_velocity_direction, [1.0, 0.0, 0.0])


class CharacterGeneratedMotionTests(unittest.TestCase):
    def test_regeneration_happens_before_sampling_the_full_timestep(self) -> None:
        delta_time = 1.0 / 64.0
        for remaining_time, early_frames in ((0.003, 0), (delta_time, 0), (0.020, 0), (0.003, 20)):
            with self.subTest(remaining_time=remaining_time, early_frames=early_frames):
                player = GeneratedMotionPlayer.__new__(GeneratedMotionPlayer)
                player.engine = SimpleNamespace(stream=None)
                player.output = SimpleNamespace(
                    sample_rate=30.0, num_frames=31, sample=Mock(), sample_foot_contacts=Mock(),
                )
                player.transition_early_by_frames = 0
                player.playback_time = 1.0 - remaining_time
                trajectory = np.array(
                    [[0.0, frame / 10.0, 0.0] for frame in range(31)], dtype=np.float32,
                )
                player._generated_root_positions = trajectory.copy()
                player.record_pose = Mock()
                character = SOMACharacter.__new__(SOMACharacter)
                character.engine = player.engine
                character.player = player
                character._playback_source = PlaybackSource.GENERATED
                character.current_pose = object()
                character.current_foot_contacts = object()
                character.character_ground_position = player.current_root_position()
                character.actual_velocity = np.array([0.0, 3.0, 0.0], dtype=np.float32)
                character.movement = MovementControl(np.array([0.0, 1.0, 0.0], dtype=np.float32))
                character.movement.input_speed = character.movement.desired_speed = 3.0
                character._target_pose_ik_corrector = None
                character._update_probe_box = Mock()
                start_position = character.character_ground_position.copy()
                locomotion = _locomotion_state(_mode("linear", BlendSpaceType.LINEAR_1D, 4.5))
                locomotion._force_regeneration = False
                locomotion._previous_generation_desired_velocity[:] = character.actual_velocity

                def regenerate(current_character):
                    # Generation uses the current pose/velocity, not a clamped endpoint.
                    np.testing.assert_allclose(current_character.character_ground_position, start_position)
                    np.testing.assert_allclose(current_character.actual_velocity, [0.0, 3.0, 0.0])
                    player.record_pose.assert_not_called()
                    player.playback_time = 0.0
                    # Even a deadline at time zero leaves the generated frames available.
                    player.output.num_frames = 18
                    player.transition_early_by_frames = early_frames
                    player._generated_root_positions = trajectory[:18] + start_position

                locomotion._generate = Mock(side_effect=regenerate)
                character.state_machine = SimpleNamespace(
                    update=lambda current, dt, command: locomotion.update(current, None, dt),
                )

                character.step(delta_time, None)

                self.assertEqual(locomotion._generate.call_count, int(remaining_time <= delta_time))
                expected_time = delta_time if remaining_time <= delta_time else 1.0 - remaining_time + delta_time
                self.assertAlmostEqual(player.playback_time, expected_time)
                np.testing.assert_allclose(
                    character.character_ground_position, start_position + [0.0, 3.0 * delta_time, 0.0],
                    atol=1e-6,
                )
                np.testing.assert_allclose(character.actual_velocity, [0.0, 3.0, 0.0], atol=1e-4)
                player.record_pose.assert_called_once_with(delta_time, character.current_pose)

    def test_clip_transition_does_not_end_early(self) -> None:
        character = SOMACharacter.__new__(SOMACharacter)
        character.character_ground_position = np.zeros(3, dtype=np.float32)
        character.actual_velocity = np.zeros(3, dtype=np.float32)
        character.movement = MovementControl(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        character._align_clip = Mock()
        character._clip_target_poses = np.zeros(
            (4, 1, 7), dtype=np.float32
        )
        character.engine = SimpleNamespace(stream=None)
        character.generate_to_target_poses = Mock()
        clip = SimpleNamespace(animation=Mock(), root_offset=None)

        character.generate_clip_transition(clip)

        target_poses, params, source = (
            character.generate_to_target_poses.call_args.args
        )
        self.assertIs(target_poses, character._clip_target_poses)
        self.assertEqual(
            params,
            TargetPoseInferenceParams(
                transition_early_by_frames=0,
                num_poses_to_use=4,
            ),
        )
        self.assertIs(source, PlaybackSource.CLIP_TRANSITION)

    def test_target_generation_captures_ik_targets_from_the_first_target_pose(self) -> None:
        character = SOMACharacter.__new__(SOMACharacter)
        character.engine = SimpleNamespace(stream=None)
        character._target_pose_ik_corrector = Mock()
        character.player = Mock()
        character.current_pose = object()
        character.current_foot_contacts = object()
        character._apply_target_pose_ik = Mock()
        character._update_character_position = Mock()
        target_poses = np.arange(4 * 77 * 7, dtype=np.float32).reshape(4, 77, 7)
        inference_params = TargetPoseInferenceParams(
            transition_early_by_frames=2, num_poses_to_use=2,
            ik_pin_mask=IKPin.LEFT_HAND | IKPin.RIGHT_FOOT,
        )

        character.generate_to_target_poses(target_poses, inference_params)

        captured_pose, captured_mask = (
            character._target_pose_ik_corrector.capture_pins.call_args.args
        )
        np.testing.assert_array_equal(captured_pose, target_poses[0])
        self.assertEqual(captured_mask, inference_params.ik_pin_mask)

    def test_target_pose_ik_uses_a_late_smooth_ramp(self) -> None:
        character = SOMACharacter.__new__(SOMACharacter)
        character.player = SimpleNamespace(
            playback_duration=1.0,
            time_to_transition=0.5,
        )
        character._playback_source = PlaybackSource.GENERATED

        self.assertEqual(character._target_pose_ik_weight(), 0.0)

        character.player.time_to_transition = 0.25
        self.assertAlmostEqual(character._target_pose_ik_weight(), 0.5)

        character.player.time_to_transition = 0.0
        self.assertEqual(character._target_pose_ik_weight(), 1.0)

    def test_each_target_sets_the_pins_that_target_pose_ik_applies(self) -> None:
        character = SOMACharacter.__new__(SOMACharacter)
        character.engine = SimpleNamespace(stream=None)
        character._target_pose_ik_corrector = Mock()
        character.player = Mock(playback_duration=1.0, time_to_transition=0.0)
        character.current_pose = object()
        character.current_foot_contacts = object()
        character._update_character_position = Mock()
        target_poses = np.zeros((4, 1, 7), dtype=np.float32)

        character.generate_to_target_poses(target_poses, TargetPoseInferenceParams(0, 4, ik_pin_mask=IKPin.LEFT_HAND))
        # A clip transition pins nothing, so the earlier target's pins stop applying.
        character.generate_to_target_poses(
            target_poses, TargetPoseInferenceParams(0, 4), PlaybackSource.CLIP_TRANSITION,
        )

        corrector = character._target_pose_ik_corrector
        masks = [call.args[1] for call in corrector.capture_pins.call_args_list]
        self.assertEqual(masks, [IKPin.LEFT_HAND, IKPin.NONE])
        corrector.apply.assert_called_with(character.current_pose, 1.0, stream=None)


if __name__ == "__main__":
    unittest.main()
