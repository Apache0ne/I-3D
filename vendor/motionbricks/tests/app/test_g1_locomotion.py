# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic DOF history, CSV targets, and locomotion, idle and interaction lifecycle tests."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
import warp as wp

from app.runtime.common.clip_registry import ClipRegistry
from app.runtime.common.interaction import InteractionProbe
from app.runtime.common.motion import (
    CROUCH_LOCOMOTION_MODE, DIRECTIONAL_LOCOMOTION_MODE, LINEAR_LOCOMOTION_MODE,
    LocomotionCommand, LocomotionMode, LocomotionModeRuntime, PlaybackSource, load_locomotion_mode,
)
from app.runtime.common.smart_primitive_behaviors import SmartPrimitiveBehavior
from app.runtime.common.io.smart_primitive_document import (
    PlaybackMode, SmartPrimitiveAssetPool, SmartPrimitiveDocument, save_smart_primitive_document,
)
from app.runtime.unitree_g1.character import UnitreeG1Character
from app.runtime.common.states import LocomotionState
from app.runtime.unitree_g1.target_pose_blender import G1TargetPoseBlender
from motionbricks.io.csv import save_csv
from motionbricks.model.config import MotionBricksInferenceConfig
from motionbricks.model.contract import ModelRig
from motionbricks.model.unitree_g1.buffers import DOFToken
from motionbricks.robotics.dof_animation_buffer import DOFAnimationBuffer
from motionbricks.utils.math_utils import project_to_plane
from motionbricks.runtime.motion_recorder import MotionRecorder
from motionbricks.runtime.motion_player import GeneratedDOFMotionPlayer
from motionbricks.runtime.blend_spaces import BlendSpace, BlendSpaceType
from motionbricks.runtime.target_pose import TargetPoseInferenceParams


# Test-owned keys and speeds keep these cases independent of demo tuning.
LINEAR_LOCOMOTION = LocomotionMode(
    LINEAR_LOCOMOTION_MODE, Path("unused.json"), BlendSpaceType.LINEAR_1D,
    (("Idle", (0, 0)), ("Walk", (.3, 0)), ("Jog", (.6, 0)), ("Run", (1, 0))),
    max_speed=4., default_speed=1.5, minimum_speed=.5, transition_early_by_frames=10,
)
DIRECTIONAL_LOCOMOTION = replace(
    LINEAR_LOCOMOTION, name=DIRECTIONAL_LOCOMOTION_MODE,
    blend_space_type=BlendSpaceType.DIRECTIONAL_2D,
    target_values=(("Idle", (0, 0)), ("Forward", (1, 0)), ("Backward", (-1, 0)),
                   ("Left", (0, 1)), ("Right", (0, -1))),
    max_speed=2., default_speed=1.,
)
CROUCH_LOCOMOTION = replace(
    DIRECTIONAL_LOCOMOTION, name=CROUCH_LOCOMOTION_MODE,
    max_speed=1., default_speed=.5, minimum_speed=.3,
)
_CAMERA = {"pos": [0.0, -3.0, 1.0], "pivot": [0.0, 0.0, 1.0], "pitch": -10.0, "yaw": 90.0,
           "fov": 45.0, "near": 0.1, "far": 100.0}


def _poses(times):
    times = np.asarray(times, dtype=np.float32)
    data = np.zeros((len(times), 8), dtype=np.float32)
    data[:, 0] = 2.0 + times
    data[:, 1] = -1.0 + times * 0.5
    data[:, 2] = 0.8 + times * 0.1
    data[:, 5] = np.sin(times * 0.4)
    data[:, 6] = np.cos(times * 0.4)
    data[:, 7] = times * 0.7
    return data


def _facing_minus_y(poses):
    """Turn each root a quarter turn clockwise, so the clip starts facing -Y like the real G1 clips."""
    turn = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -0.5 * np.pi)
    for row in poses:
        row[3:7] = wp.mul(turn, wp.quat(*row[3:7]))
    return poses


def _reference_pose():
    return wp.array(_poses([0])[0], dtype=wp.float32)


class _Engine:
    """Deterministic endpoint interpolation; no ONNX or robot assets needed."""

    def __init__(self):
        self.device = wp.get_device("cpu")
        self.stream = None
        self.contract = SimpleNamespace(joint_names=("hinge",), sample_rate=30.0)
        self.contexts = []
        self.targets = []
        self.configs = []

    def create_input(self):
        return DOFToken(1, device=self.device)

    def create_output(self):
        return DOFAnimationBuffer(
            ("hinge",), 30.0, wp.full((58, 8), 999.0, dtype=wp.float32, device=self.device), num_frames=0,
            foot_contacts=wp.zeros((58, 4), dtype=wp.float32, device=self.device),
        )

    def generate_animation(self, context, target, output, *, config):
        self.contexts.append(context.poses.numpy().copy())
        self.targets.append(target.poses.numpy()[: target.num_poses].copy())
        self.configs.append(config)
        start, end = self.contexts[-1][-1], target.poses.numpy()[0]
        values = np.linspace(start, end, 26, dtype=np.float32)
        for frame in range(26):
            values[frame, 3:7] = wp.quat_slerp(wp.quat(*start[3:7]), wp.quat(*end[3:7]), frame / 25)
        output.poses[:26].assign(values)
        output.foot_contacts[:26].assign(np.repeat(np.linspace(0.1, 0.9, 26, dtype=np.float32)[:, None], 4, axis=1))
        output.num_frames = 26


class _SequenceLoop(SmartPrimitiveBehavior):
    """Enter through a sequence, loop a clip until interact(), then leave through another sequence."""

    required_sequences = ("Entry", "Exit")

    def enter(self):
        self.interaction_state.play_target_pose_sequence("Entry", on_complete=self._loop)

    def _loop(self):
        self.interaction_state.play_clip(self.clip_paths[0], loop=True)

    def interact(self):
        if self.interaction_state.is_looping:
            self.interaction_state.play_target_pose_sequence("Exit")


class _PlayOnce(SmartPrimitiveBehavior):
    """Play one clip once, then return to movement."""

    def enter(self):
        self.interaction_state.play_clip(self.clip_paths[0])


class G1LocomotionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.previous_device = wp.get_device()
        wp.set_device("cpu")

    @classmethod
    def tearDownClass(cls):
        wp.set_device(cls.previous_device)

    def setUp(self):
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.source = DOFAnimationBuffer(("hinge",), 120.0, wp.array(_poses(np.arange(121) / 120), dtype=wp.float32))
        self.csv_path = self.root / "source.csv"
        save_csv(str(self.csv_path), self.source)
        idle_data = _facing_minus_y(_poses(np.arange(121) / 120))
        idle_data[:, 2] += 0.03 * np.sin(np.arange(121) / 120 * 2 * np.pi)
        idle_data[:, 7] += 0.2 * np.sin(np.arange(121) / 120 * 2 * np.pi)
        self.idle_path = self.root / "idle.csv"
        save_csv(
            str(self.idle_path), DOFAnimationBuffer(("hinge",), 120, wp.array(idle_data, dtype=wp.float32)),
        )
        crouch_data = _facing_minus_y(_poses(np.arange(121) / 120))
        crouch_data[:, 2] -= 0.4
        crouch_data[:, 7] += 1.0
        crouch_buffer = DOFAnimationBuffer(("hinge",), 120, wp.array(crouch_data, dtype=wp.float32))
        crouch_path = self.root / "crouch.csv"
        save_csv(str(crouch_path), crouch_buffer)
        self.crouch_idle_path = self.root / "crouch_idle.csv"
        save_csv(str(self.crouch_idle_path), crouch_buffer)

        self.clip_registry = ClipRegistry(
            ModelRig.UNITREE_G1, joint_names=("hinge",), device="cpu",
        )
        self.selection_times = (0.014, 0.2, 0.4, 0.6)
        linear_targets = [
            (name, self.csv_path, self.selection_times[index], index > 0)
            for index, name in enumerate(("Idle", "Walk", "Jog", "Run"))
        ]
        self.mode = replace(LINEAR_LOCOMOTION, smart_primitive_path=self._save_targets("linear", linear_targets))
        directional_modes = []
        for definition, path in ((DIRECTIONAL_LOCOMOTION, self.csv_path), (CROUCH_LOCOMOTION, crouch_path)):
            targets = [(name, path, index * 0.15, True) for index, (name, _) in enumerate(definition.target_values)]
            directional_modes.append(replace(
                definition, smart_primitive_path=self._save_targets(definition.name, targets),
            ))
        self.modes = (self.mode, *directional_modes)
        self.asset_pool = SmartPrimitiveAssetPool(
            self.clip_registry, _reference_pose(), wp.transform_identity(),
        )

    def _save_targets(self, name, targets):
        """Save targets that start at the origin facing +X, as the bundled G1 documents do."""
        document = SmartPrimitiveDocument("unitree_g1", _reference_pose(), wp.transform_identity())
        for node_name, path, sample_time, use_frames in targets:
            node = document.add_node()
            node.name = node_name
            target = document.add_target_pose(node)
            target.set_clip(self.clip_registry.load(path))
            if use_frames:
                target.playback_mode = PlaybackMode.FRAMES
                target.set_time(round(sample_time * 120) / 120)
            else:
                target.set_time(sample_time)
            target.constrain_hips(True)
            root = target.pose.numpy()
            heading = project_to_plane(wp.transform(wp.vec3(*root[:3]), wp.quat(*root[3:7])), wp.vec3(0.0, 0.0, 1.0))
            node.local_xform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_inverse(heading.q))
        return save_smart_primitive_document(document, self.root / name, document.root.id, _CAMERA)

    def _character(self, initial_mode=LINEAR_LOCOMOTION.name, **options):
        engine = _Engine()
        # Ensure sampling honors a single-embodiment view in a larger flat array.
        storage = wp.full(18, -99.0, dtype=wp.float32)
        joint_q = storage[5:13]
        joint_q.assign(_poses([0])[0])
        character = UnitreeG1Character(
            engine, current_pose=joint_q, asset_pool=self.asset_pool, modes=self.modes, initial_mode=initial_mode,
            idle_paths=(self.idle_path,), crouch_idle_paths=(self.crouch_idle_path,), **options,
        )
        return character, engine, storage

    def _save_interaction(self):
        """Save an Entry sequence of two targets and an Exit sequence of one, all from the source clip."""
        document = SmartPrimitiveDocument("unitree_g1", _reference_pose(), wp.transform_identity())
        clip = self.clip_registry.load(self.csv_path)
        nodes = {}
        for name, frame, poses_to_use in (("Approach", 24, 4), ("Hold", 60, 1), ("Depart", 96, 4)):
            nodes[name] = document.add_node()
            nodes[name].name = name
            target = document.add_target_pose(nodes[name])
            target.set_clip(clip)
            target.playback_mode = PlaybackMode.FRAMES
            target.set_time(frame / 120)
            target.set_num_poses_to_use(poses_to_use)
        for name, targets in (("Entry", ("Approach", "Hold")), ("Exit", ("Depart",))):
            sequence = document.add_sequence()
            sequence.name = name
            for target in targets:
                document.add_sequence_node(sequence, nodes[target])
        return save_smart_primitive_document(document, self.root / "interaction", document.root.id, _CAMERA)

    def _step(self, character, engine, command, delta_time=0.03):
        """Step once and check that every new generation continues from the pose shown before the step."""
        before = character.current_pose.numpy().copy()
        generations = len(engine.contexts)
        character.step(delta_time, command)
        for context in engine.contexts[generations:]:
            np.testing.assert_allclose(context[-1], before, atol=1e-6)
        return before

    def _step_until(self, character, engine, command, done, limit=200):
        """Step until done() is true and return the pose shown before the last step."""
        for _ in range(limit):
            before = self._step(character, engine, command)
            if done():
                return before
        self.fail("The character never reached the expected state.")

    def test_selection_modes_sampling_rate_and_root_normalization(self):
        primitive = self.asset_pool.instantiate(self.mode.smart_primitive_path)
        self.assertEqual(primitive.rig, "unitree_g1")
        self.assertEqual(primitive.asset.rig, "unitree_g1")
        self.assertEqual(primitive.pose_size, self.source.num_coordinates)
        runtime = load_locomotion_mode(self.mode, self.asset_pool)
        targets = runtime.target_poses.numpy()
        for index, start in enumerate(self.selection_times):
            times = start + np.arange(4) / 30
            np.testing.assert_allclose(targets[index, :, 7], times * 0.7, atol=2e-7)
            np.testing.assert_allclose(targets[index, :, 2], 0.8 + times * 0.1, atol=1e-6)
            np.testing.assert_allclose(targets[index, 0, :2], 0.0, atol=3e-7)
            np.testing.assert_allclose(targets[index, 0, 3:7], [0, 0, 0, 1], atol=3e-7)

    def test_modes_and_idles_share_source_clips_without_looping_the_source(self):
        source = self.clip_registry.load(self.csv_path).animation
        original = source.poses.numpy().copy()
        load_locomotion_mode(self.mode, self.asset_pool)
        load_locomotion_mode(self.modes[1], self.asset_pool)
        character = UnitreeG1Character.__new__(UnitreeG1Character)
        character.engine = SimpleNamespace(stream=None)
        character.clip_registry = self.clip_registry
        character.model_to_runtime = wp.transform_identity()
        looped = character.load_clip(self.csv_path, loop=True)
        second_loop = character.load_clip(self.csv_path, loop=True)
        once = character.load_clip(self.csv_path, loop=False)
        np.testing.assert_array_equal(source.poses.numpy(), original)
        # A clip played once uses the shared source directly.
        self.assertIs(once.animation, source)
        self.assertIsNot(looped.animation.poses, source.poses)
        self.assertIsNot(second_loop.animation.poses, looped.animation.poses)
        np.testing.assert_allclose(looped.animation.poses.numpy()[-1], original[0], atol=1e-6)

    def test_dof_target_reports_root_velocity_and_constrained_trajectory(self):
        document = SmartPrimitiveDocument("unitree_g1", _reference_pose(), wp.transform_identity())
        target = document.add_target_pose(document.add_node())
        target.set_clip(self.clip_registry.load(self.csv_path))
        target.set_time(0.5)
        target.constrain_hips(True)

        # The clip's root moves at (1, 0.5, 0.1) m/s; the constraint keeps only its height.
        np.testing.assert_allclose([float(target.velocity[axis]) for axis in range(3)], [1.0, 0.5, 0.1], atol=1e-4)
        times = np.clip(0.5 + (np.arange(21) - 10) * target.trajectory_scale / 30, 0.0, 1.0)
        trajectory = target.trajectory_transforms.numpy()
        np.testing.assert_allclose(trajectory[:, 0], times - 0.5, atol=1e-5)
        np.testing.assert_allclose(trajectory[:, 1], 0.5 * (times - 0.5), atol=1e-5)
        np.testing.assert_allclose(trajectory[:, 2], 0.8 + 0.1 * times, atol=1e-5)

    def test_locomotion_targets_must_use_all_four_frames(self):
        document = SmartPrimitiveDocument("unitree_g1", _reference_pose(), wp.transform_identity())
        for node_name, _ in LINEAR_LOCOMOTION.target_values:
            node = document.add_node()
            node.name = node_name
            target = document.add_target_pose(node)
            target.set_clip(self.clip_registry.load(self.csv_path))
        # Interactions may use fewer frames, but locomotion blends whole four-frame windows.
        target.set_num_poses_to_use(1)
        path = save_smart_primitive_document(
            document, self.root / "one_frame", document.root.id, _CAMERA,
        )

        with self.assertRaisesRegex(ValueError, "'Run' must use 4 poses"):
            load_locomotion_mode(replace(LINEAR_LOCOMOTION, smart_primitive_path=path), self.asset_pool)

    def test_modes_blend_then_scale_speed_and_place_only_the_root(self):
        # Nonuniform steps catch changes to the authored velocity pattern.
        progress = np.array([0, 0.2, 0.7, 1.0], dtype=np.float32)
        height_motion = np.array([0, 0.06, -0.02, 0.1], dtype=np.float32)
        hinge_motion = np.array([0, 0.1, 0.15, 0.2], dtype=np.float32)
        for definition, speed, facing, height, hinge in (
            (LINEAR_LOCOMOTION, 2.25, (0, 1, 0), 2, 1),
            (DIRECTIONAL_LOCOMOTION, 4.5, (1, 0, 0), 3, 1.5),
        ):
            for device in wp.get_devices():
                with self.subTest(mode=definition.name, device=str(device)), wp.ScopedDevice(device):
                    mode = replace(definition, max_speed=4.5, velocity_scale=0.9, position_scale=0.8)
                    keys = ((0, 0), (1, 0)) if definition is LINEAR_LOCOMOTION else ((0, 0), (1, 0), (0, -1))
                    space = BlendSpace(mode.blend_space_type)
                    for index, key in enumerate(keys):
                        space.add_target(wp.vec2(*key), index)
                    space.compile()
                    data = np.zeros((len(keys), 4, 8), dtype=np.float32)
                    data[..., 6] = 1
                    data[0, :, 0] = progress * 0.3
                    data[1, :, 0] = progress * 0.9
                    data[0, :, 2] = 1 + height_motion
                    data[1, :, 2] = 3 + height_motion
                    data[0, :, 7] = 0.5 + hinge_motion
                    data[1, :, 7] = 1.5 + hinge_motion
                    runtime = LocomotionModeRuntime(
                        mode, space, wp.array(data, dtype=wp.float32),
                    )
                    state = LocomotionState("test", (runtime,), mode.name, wp.empty((1, 8), dtype=wp.float32))
                    direction = np.asarray(facing, dtype=np.float32)
                    character = SimpleNamespace(
                        engine=SimpleNamespace(device=device, stream=None),
                        target_pose_blender=G1TargetPoseBlender(),
                        character_ground_position=np.array([10, 2, 0], dtype=np.float32),
                        movement=SimpleNamespace(current_facing=direction, desired_velocity=direction * speed),
                    )
                    blended = wp.empty((4, 8), dtype=wp.float32)
                    targets = wp.empty_like(blended)

                    state.evaluate_current_target_poses(character, blended, targets)

                    result = targets.numpy()
                    ground = character.character_ground_position + direction * speed * 0.9 * 0.8
                    expected_xy = ground[:2] + progress[:, None] * direction[:2] * speed * 0.9 * 0.1
                    np.testing.assert_allclose(result[:, :2], expected_xy, atol=2e-6)
                    np.testing.assert_allclose(result[:, 2], height + height_motion, atol=3e-7)
                    yaw = wp.quat_from_axis_angle(wp.vec3(0, 0, 1), np.arctan2(direction[1], direction[0]))
                    np.testing.assert_allclose(result[:, 3:7], np.tile(yaw, (4, 1)), atol=2e-7)
                    np.testing.assert_allclose(result[:, 7], hinge + hinge_motion, atol=2e-7)
                    np.testing.assert_array_equal(runtime.target_poses.numpy(), data)

    def test_directional_targets_follow_facing_and_blend_diagonal_dof_motion(self):
        mode = replace(DIRECTIONAL_LOCOMOTION, max_speed=2.0)
        height = 0.8
        for device in wp.get_devices():
            with self.subTest(device=str(device)), wp.ScopedDevice(device):
                space = BlendSpace(mode.blend_space_type)
                data = np.zeros((5, 4, 8), dtype=np.float32)
                data[:, :, 2] = height
                data[:, :, 6] = 1
                for index, (_, value) in enumerate(mode.target_values):
                    space.add_target(wp.vec2(*value), index)
                    data[index, :, :2] = np.arange(4)[:, None] / 30 * np.array(value) * 2
                    data[index, :, 7] = index + np.arange(4) * 0.1
                space.compile()
                runtime = LocomotionModeRuntime(mode, space, wp.array(data, dtype=wp.float32))
                state = LocomotionState("test", (runtime,), mode.name, wp.empty((1, 8), dtype=wp.float32))
                character = SimpleNamespace(
                    engine=SimpleNamespace(device=device, stream=None),
                    target_pose_blender=G1TargetPoseBlender(),
                    character_ground_position=np.array([10, 20, 0], dtype=np.float32),
                    movement=SimpleNamespace(
                        current_facing=np.array([0, 1, 0], dtype=np.float32),
                        desired_velocity=np.zeros(3, dtype=np.float32),
                    ),
                )
                blended = wp.empty((4, 8), dtype=wp.float32)
                targets = wp.empty_like(blended)
                # Facing +Y puts left at -X. A request half forward and half left
                # uses three positive weights, including the idle.
                for velocity, weights in (
                    ((0, 0, 0), (1, 0, 0, 0, 0)),
                    ((0, 2, 0), (0, 1, 0, 0, 0)),
                    ((0, -2, 0), (0, 0, 1, 0, 0)),
                    ((-2, 0, 0), (0, 0, 0, 1, 0)),
                    ((2, 0, 0), (0, 0, 0, 0, 1)),
                    ((-0.5, 0.5, 0), (0.5, 0.25, 0, 0.25, 0)),
                ):
                    with self.subTest(velocity=velocity):
                        character.movement.desired_velocity[:] = velocity
                        state.evaluate_current_target_poses(character, blended, targets)
                        actual = targets.numpy()
                        expected_root = character.character_ground_position + velocity
                        expected_root[2] = height
                        np.testing.assert_allclose(
                            actual[:, :3], expected_root + np.arange(4)[:, None] / 30 * velocity, atol=3e-6,
                        )
                        np.testing.assert_allclose(
                            actual[:, 7], np.dot(weights, np.arange(5)) + np.arange(4) * 0.1, atol=2e-7,
                        )
                        np.testing.assert_allclose(
                            actual[:, 3:7], np.tile([0, 0, np.sqrt(0.5), np.sqrt(0.5)], (4, 1)), atol=2e-7,
                        )
                np.testing.assert_array_equal(runtime.target_poses.numpy(), data)

    def test_idle_mode_switches_generate_between_standing_and_crouching_clips(self):
        character, engine, _ = self._character()
        machine = character.state_machine
        self.assertFalse(machine.is_interacting)
        facing = np.array([1, 0, 0], dtype=np.float32)
        character.step(0.03, LocomotionCommand(facing, None, requested_mode=DIRECTIONAL_LOCOMOTION.name))
        self.assertEqual(len(engine.contexts), 0)
        self.assertIs(machine.state, machine.standing_idle_state)
        for mode, idle in ((CROUCH_LOCOMOTION, machine.crouching_idle_state),
                           (LINEAR_LOCOMOTION, machine.standing_idle_state),
                           (CROUCH_LOCOMOTION, machine.crouching_idle_state)):
            before = character.current_pose.numpy().copy()
            character.step(0.03, LocomotionCommand(facing, None, requested_mode=mode.name))
            self.assertIs(machine.state, idle)
            self.assertEqual(machine.active_locomotion_state.active_mode.name, mode.name)
            self.assertIs(character._playback_source, PlaybackSource.CLIP_TRANSITION)
            np.testing.assert_allclose(engine.contexts[-1][-1], before, atol=1e-6)
            self.assertIn(character.current_clip, idle.clips)
        for _ in range(40):
            character.step(0.03, LocomotionCommand(facing, None))
        self.assertIs(character._playback_source, PlaybackSource.CLIP)
        self.assertLess(character.current_pose.numpy()[2], 0.6)

    def test_moving_mode_switches_reuse_displayed_history_and_matching_idle(self):
        character, engine, storage = self._character()
        machine = character.state_machine
        facing = np.array([1, 0, 0], dtype=np.float32)
        for mode in (LINEAR_LOCOMOTION, DIRECTIONAL_LOCOMOTION, CROUCH_LOCOMOTION,
                     DIRECTIONAL_LOCOMOTION, LINEAR_LOCOMOTION):
            before = character.current_pose.numpy().copy()
            generations = len(engine.contexts)
            character.step(0.03, LocomotionCommand(facing, facing, requested_mode=mode.name))
            self.assertTrue(machine.is_locomoting)
            self.assertEqual(machine.active_locomotion_state.active_mode.name, mode.name)
            self.assertGreater(len(engine.contexts), generations)
            np.testing.assert_allclose(engine.contexts[-1][-1], before, atol=1e-6)
            for _ in range(4):
                character.step(0.03, LocomotionCommand(facing, facing))
        for mode in (LINEAR_LOCOMOTION, CROUCH_LOCOMOTION):
            character.step(0.03, LocomotionCommand(facing, facing, requested_mode=mode.name))
            character.step(0.03, LocomotionCommand(facing, None))
            idle = machine.crouching_idle_state if mode is CROUCH_LOCOMOTION else machine.standing_idle_state
            self.assertIs(machine.state, idle)
            self.assertIn(character.current_clip, idle.clips)
        np.testing.assert_array_equal(storage.numpy()[:5], -99.0)
        np.testing.assert_array_equal(storage.numpy()[13:], -99.0)

    def test_initial_crouch_uses_the_crouching_idle_without_generation(self):
        character, engine, _ = self._character(CROUCH_LOCOMOTION.name)
        self.assertIs(character.state_machine.state, character.state_machine.crouching_idle_state)
        self.assertLess(character.current_pose.numpy()[2], 0.6)
        self.assertEqual(len(engine.contexts), 0)

    def test_loop_closes_root_and_hinges_without_touching_capacity(self):
        data = _poses([0, 0.25, 0.5, 0.75, 1.0, 1.25])
        # Nontrivial intermediate motion should survive the endpoint correction.
        data[2, 2] += 0.15
        data[2, 7] += 0.2
        # The last quaternion can use the opposite sign for the same rotation.
        data[4, 3:7] *= -1
        animation = DOFAnimationBuffer(("hinge",), 30, wp.array(data, dtype=wp.float32), num_frames=5)
        animation.make_loopable()
        result = animation.poses.numpy()
        np.testing.assert_allclose(result[0], data[0], atol=1e-7)
        np.testing.assert_allclose(result[4, :3], result[0, :3], atol=1e-6)
        self.assertAlmostEqual(abs(float(result[4, 3:7] @ result[0, 3:7])), 1.0, places=6)
        np.testing.assert_allclose(result[4, 7:], result[0, 7:], atol=1e-6)
        np.testing.assert_array_equal(result[5], data[5])
        self.assertAlmostEqual(result[2, 2], data[0, 2] + 0.15, places=6)
        self.assertAlmostEqual(result[2, 7], 0.2, places=6)
        for count in (0, 1):
            animation.num_frames = count
            animation.make_loopable()
            np.testing.assert_array_equal(animation.poses.numpy(), result)

    def test_history_resamples_elapsed_time_and_root_rotation_at_30_hz(self):
        reference = wp.array(_poses([0])[0], dtype=wp.float32)
        history = MotionRecorder(reference, capacity=16)
        token = DOFToken(1)
        token.update_from_recorder(history)
        np.testing.assert_allclose(token.poses.numpy(), _poses([0, 0, 0, 0]))
        elapsed = 0.0
        for dt in (0.017, 0.009, 0.032, 0.025, 0.041, 0.013, 0.027, 0.008, 0.03):
            elapsed += dt
            reference.assign(_poses([elapsed])[0])
            history.record(dt, reference)
        token.update_from_recorder(history)
        np.testing.assert_allclose(token.poses.numpy(), _poses(elapsed + np.arange(-3, 1) / 30), atol=4e-6)
        self.assertEqual(token.num_poses, 4)

    def test_generated_player_uses_target_token_policy_and_valid_output(self):
        engine = _Engine()
        reference = wp.array(_poses([0])[0], dtype=wp.float32)
        player = GeneratedDOFMotionPlayer(engine, reference)
        player.seed_history_from_pose(reference)
        target = wp.array(_poses([1, 1.03, 1.06, 1.1]), dtype=wp.float32)
        # A target without its own token counts keeps those of the config.
        config = MotionBricksInferenceConfig(allowed_token_counts=(9,))
        player.generate(target, TargetPoseInferenceParams(4, 4), config=config)
        self.assertEqual(engine.configs[-1].allowed_token_counts, (9,))
        player.generate(target, TargetPoseInferenceParams(4, 4, allowed_token_counts=(6, 7)))
        self.assertEqual(engine.configs[-1].allowed_token_counts, (6, 7))
        self.assertAlmostEqual(player.time_to_transition, 21 / 30)
        contacts = wp.empty(4, dtype=wp.float32)
        player.sample(100.0, reference, contacts)
        np.testing.assert_allclose(reference.numpy(), target.numpy()[0], atol=1e-6)
        np.testing.assert_allclose(contacts.numpy(), 0.9, atol=1e-6)
        np.testing.assert_allclose(player.current_root_position(), target.numpy()[0, :3])

    def test_current_foot_contacts_follow_generated_motion_and_read_zero_during_clips(self):
        character, _, _ = self._character()
        facing = np.array([1, 0, 0], dtype=np.float32)
        character.step(0.03, LocomotionCommand(facing, None))
        self.assertIs(character._playback_source, PlaybackSource.CLIP)
        np.testing.assert_array_equal(character.current_foot_contacts.numpy(), 0.0)
        idle_clip = character.current_clip
        expected = wp.empty(4, dtype=wp.float32, device="cpu")

        # Starting generated motion or a clip updates the contacts before the next step.
        target = wp.array(_poses([1, 1.03, 1.06, 1.1]), dtype=wp.float32)
        character.generate_to_target_poses(target, TargetPoseInferenceParams(4, 4))
        character.player.output.sample_foot_contacts(character.player.playback_time, expected)
        self.assertGreater(expected.numpy().min(), 0.0)
        np.testing.assert_allclose(character.current_foot_contacts.numpy(), expected.numpy())
        character.start_clip(idle_clip, loop=True)
        np.testing.assert_array_equal(character.current_foot_contacts.numpy(), 0.0)

        character.step(0.03, LocomotionCommand(facing, facing))
        self.assertIs(character._playback_source, PlaybackSource.GENERATED)
        character.player.output.sample_foot_contacts(character.player.playback_time, expected)
        np.testing.assert_allclose(character.current_foot_contacts.numpy(), expected.numpy())

    def test_idle_locomotion_stop_restart_and_clip_boundary_keep_played_history(self):
        character, engine, storage = self._character()
        facing = np.array([1, 0, 0], dtype=np.float32)
        idle = LocomotionCommand(facing, None)
        moving = LocomotionCommand(facing, facing)
        character.step(0.137, idle)
        before = character.current_pose.numpy().copy()
        character.step(0.02, moving)
        np.testing.assert_allclose(engine.contexts[-1][-1], before, atol=1e-6)
        self.assertEqual(character.state_machine.state.name, "Locomotion")
        for _ in range(8):
            character.step(0.03, moving)
        character.step(0.02, idle)
        self.assertIs(character._playback_source, PlaybackSource.CLIP_TRANSITION)
        character.step(0.03, idle)
        before = character.current_pose.numpy().copy()
        character.step(0.02, moving)
        self.assertIs(character._playback_source, PlaybackSource.GENERATED)
        np.testing.assert_allclose(engine.contexts[-1][-1], before, atol=1e-6)
        character.step(0.02, idle)
        while character.time_to_transition() > 0.04:
            character.step(0.031, idle)
        character.step(0.05, idle)
        # The transition ended at the idle clip's first pose, so the clip starts from zero.
        self.assertIs(character._playback_source, PlaybackSource.CLIP)
        self.assertAlmostEqual(character._clip_playback_time, 0.05)
        expected = wp.empty(8, dtype=wp.float32)
        clip = character.current_clip
        clip.animation.sample(0.05, expected, root_offset=clip.root_offset)
        np.testing.assert_allclose(character.current_pose.numpy(), expected.numpy(), atol=1e-6)
        character._clip_playback_time = clip.duration - 0.005
        character.step(0.012, idle)
        self.assertAlmostEqual(character._clip_playback_time, 0.007)
        before = character.current_pose.numpy().copy()
        character.step(0.015, moving)
        np.testing.assert_allclose(engine.contexts[-1][-1], before, atol=1e-6)
        np.testing.assert_array_equal(storage.numpy()[:5], -99.0)
        np.testing.assert_array_equal(storage.numpy()[13:], -99.0)

    def test_interaction_sequences_and_loop_continue_from_the_displayed_pose(self):
        character, engine, storage = self._character()
        machine = character.state_machine
        state = machine.interaction_state
        primitive = self.asset_pool.instantiate(self._save_interaction(), _SequenceLoop)
        primitive.clip_paths = (self.idle_path,)
        expected = wp.empty((12, 8), dtype=wp.float32)
        primitive.get_world_target_poses_for_sequence(primitive.sequence_index("Entry"), expected[:8])
        primitive.get_world_target_poses_for_sequence(primitive.sequence_index("Exit"), expected[8:])
        expected = expected.numpy()
        facing = np.array([1, 0, 0], dtype=np.float32)
        idle = LocomotionCommand(facing, None)
        self._step(character, engine, idle)

        before = character.current_pose.numpy().copy()
        machine.start_interaction(character, primitive)
        self.assertTrue(machine.is_interacting)
        np.testing.assert_allclose(engine.contexts[-1][-1], before, atol=1e-6)
        np.testing.assert_allclose(engine.targets[-1], expected[:4], atol=1e-6)

        reached = self._step_until(character, engine, idle, lambda: state.is_looping)
        self.assertEqual(len(engine.targets), 2)
        np.testing.assert_allclose(engine.targets[-1], expected[4:5], atol=1e-6)
        # The loop starts from the reached root, including its height and turn.
        clip = character.current_clip
        loop_start = clip.animation.poses.numpy()[0]
        self.assertGreater(abs(reached[2] - loop_start[2]), 0.01)
        self.assertLess(abs(float(reached[3:7] @ loop_start[3:7])), 0.999)
        first = wp.empty(8, dtype=wp.float32)
        clip.animation.sample(0.0, first, root_offset=clip.root_offset)
        first = first.numpy()
        np.testing.assert_allclose(first[:3], reached[:3], atol=1e-5)
        self.assertAlmostEqual(abs(float(first[3:7] @ reached[3:7])), 1.0, places=6)
        self.assertAlmostEqual(character._clip_playback_time, 0.03)
        sampled = wp.empty(8, dtype=wp.float32)
        clip.animation.sample(0.03, sampled, root_offset=clip.root_offset)
        np.testing.assert_allclose(character.current_pose.numpy(), sampled.numpy(), atol=1e-6)

        # The loop keeps playing past its duration without generating.
        generations = len(engine.contexts)
        for _ in range(40):
            self._step(character, engine, idle)
        self.assertTrue(state.is_looping)
        self.assertEqual(len(engine.contexts), generations)
        self.assertLess(character._clip_playback_time, 0.5 * clip.duration)

        before = character.current_pose.numpy().copy()
        machine.interact()
        self.assertFalse(state.is_looping)
        np.testing.assert_allclose(engine.contexts[-1][-1], before, atol=1e-6)
        np.testing.assert_allclose(engine.targets[-1], expected[8:], atol=1e-6)
        self._step_until(character, engine, idle, lambda: not machine.is_interacting)
        self.assertIs(machine.state, machine.standing_idle_state)
        self.assertIs(character._playback_source, PlaybackSource.CLIP_TRANSITION)
        self.assertIsNone(primitive._interaction_state)

        self._step(character, engine, LocomotionCommand(facing, facing))
        self.assertTrue(machine.is_locomoting)
        self.assertEqual(engine.configs[-1].allowed_token_counts, tuple(range(8, 15)))
        np.testing.assert_array_equal(storage.numpy()[:5], -99.0)
        np.testing.assert_array_equal(storage.numpy()[13:], -99.0)

    def test_clip_started_during_an_idle_transition_plays_once_from_its_first_frame(self):
        character, engine, _ = self._character()
        machine = character.state_machine
        primitive = self.asset_pool.instantiate(self._save_interaction(), _PlayOnce)
        primitive.clip_paths = (self.csv_path,)
        facing = np.array([1, 0, 0], dtype=np.float32)
        idle = LocomotionCommand(facing, None)
        for _ in range(5):
            self._step(character, engine, LocomotionCommand(facing, facing))
        self._step(character, engine, idle)
        self.assertIs(character._playback_source, PlaybackSource.CLIP_TRANSITION)
        self.assertGreater(character.time_to_transition(), 0.03)

        machine.start_interaction(character, primitive)
        self.assertEqual(character._clip_playback_time, 0.0)
        self._step(character, engine, idle)
        self.assertAlmostEqual(character._clip_playback_time, 0.03)
        clip = character.current_clip
        sampled = wp.empty(8, dtype=wp.float32)
        clip.animation.sample(0.03, sampled, root_offset=clip.root_offset)
        np.testing.assert_allclose(character.current_pose.numpy(), sampled.numpy(), atol=1e-6)

        # The interaction ends during the step that reaches the clip's end.
        self._step_until(character, engine, idle, lambda: character.clip_ends_within(0.03))
        self.assertTrue(machine.is_interacting)
        self._step(character, engine, idle)
        self.assertFalse(machine.is_interacting)
        self.assertIs(machine.state, machine.standing_idle_state)
        self.assertIs(character._playback_source, PlaybackSource.CLIP_TRANSITION)

    def test_initial_transform_places_and_turns_the_starting_idle(self):
        yaw = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.5 * np.pi)
        character, engine, _ = self._character(
            initial_transform=wp.transform(wp.vec3(3.0, -2.0, 0.4), yaw),
        )
        np.testing.assert_allclose(character.character_ground_position, [3.0, -2.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(character.movement.current_facing, [0.0, 1.0, 0.0], atol=1e-6)
        self.assertAlmostEqual(character.movement.current_facing_angle, 0.5 * np.pi, places=6)
        self.assertAlmostEqual(character.movement.desired_velocity_angle, 0.5 * np.pi, places=6)
        pose = character.current_pose.numpy()
        np.testing.assert_allclose(pose[:3], [3.0, -2.0, 0.8], atol=1e-5)
        self.assertAlmostEqual(abs(float(pose[3:7] @ np.asarray(yaw))), 1.0, places=6)
        near, _ = character.probe_segment
        np.testing.assert_allclose(near, [3.0, -2.0 + character.interaction_probe.near_distance, 0.0], atol=1e-6)
        self.assertEqual(len(engine.contexts), 0)

    def test_interaction_box_follows_desired_velocity_and_crouch_height(self):
        probe = InteractionProbe(
            box_width=0.8, box_min_depth=0.6, box_height=1.25, near_distance=0.1, lookahead_time=1.0,
            extra_lookahead_min_speed=1.5, extra_lookahead_time=0.25,
        )
        character, _, _ = self._character(interaction_probe=probe)
        # Standing still, the box has its minimum depth.
        box = character.probe_box.numpy()
        np.testing.assert_allclose(box["transform"][0, :3], [0.4, 0.0, 0.625], atol=1e-6)
        np.testing.assert_allclose(box["half_extents"][0], [0.4, 0.3, 0.625], atol=1e-6)

        facing = np.array([1, 0, 0], dtype=np.float32)
        character.step(0.03, LocomotionCommand(facing, None, requested_mode=CROUCH_LOCOMOTION.name))
        self.assertEqual(character.state_machine.probe_box_height_scale, 0.6)
        character.movement.desired_velocity[:] = (0.0, 2.0, 0.0)
        character.movement.desired_velocity_direction[:] = (0.0, 1.0, 0.0)
        character._update_probe_box()
        ground = character.character_ground_position
        near, far = character.probe_segment
        np.testing.assert_allclose(near, ground + [0.0, 0.1, 0.0], atol=1e-6)
        np.testing.assert_allclose(far, ground + [0.0, 2.125, 0.0], atol=1e-6)
        box = character.probe_box.numpy()
        np.testing.assert_allclose(box["transform"][0, :3], ground + [0.0, 1.1125, 0.375], atol=1e-6)
        np.testing.assert_allclose(box["transform"][0, 3:], [0.0, 0.0, 0.0, 1.0], atol=1e-6)
        np.testing.assert_allclose(box["half_extents"][0], [0.4, 1.0125, 0.375], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
