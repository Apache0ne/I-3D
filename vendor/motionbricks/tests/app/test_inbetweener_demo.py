# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from itertools import product
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import newton
import numpy as np
import warp as wp

from app.inbetweener_demo import InbetweenerDemo, PoseEntry, sample_pose_window
from app.authoring.common import InlineRename
from app.runtime.common.clip_registry import ClipRegistry
from motionbricks.animation.animation_buffer import AnimationBuffer
from motionbricks.animation.skeleton import Skeleton
from motionbricks.model.contract import ModelRig
from motionbricks.model.soma.buffers import PoseToken
from motionbricks.model.unitree_g1.buffers import DOFToken
from motionbricks.renderers.foot_contact_renderer import (
    ContactOutline,
    DOFFootContactRenderer,
    SkeletonFootContactRenderer,
)
from motionbricks.robotics.dof_animation_buffer import DOFAnimationBuffer
from motionbricks.utils.space_conversion_utils import FacingDirectionType, SpaceConverter


_G1_JOINTS = ("hip", "knee")


class _RecordingEngine:
    """Deterministic output with unused capacity to catch context carry-over errors."""

    stream = None

    def __init__(self):
        self.calls = []
        self.target_counts = []
        self.token_counts = []

    def generate_animation(self, context, target, output, *, config):
        self.calls.append((context.poses.numpy().copy(), target.poses.numpy().copy()))
        self.target_counts.append(target.num_poses)
        self.token_counts.append(config.allowed_token_counts)
        frames = np.repeat(context.poses.numpy()[-1:], output.capacity, axis=0)
        # Both pose layouts start with the root position.
        root_x = frames.reshape(output.capacity, -1)[:, 0]
        root_x[:6] = np.linspace(root_x[0], target.poses.numpy()[0].reshape(-1)[0], 6)
        root_x[6:] = 999.0
        output.poses.assign(frames)
        output.num_frames = 6


class InbetweenerDemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.previous_device = wp.get_device()
        wp.set_device("cpu")
        reference = np.array([
            [0, 1, 0, 0, 0, 0, 1],
            [0, 0.2, 0, 0, 0, 0, 1],
        ], dtype=np.float32)
        cls.skeleton = Skeleton(["Hips", "Child"], [-1, 0], reference)

    @classmethod
    def tearDownClass(cls):
        wp.set_device(cls.previous_device)

    def make_entry(self, sample_rate=60.0, sample_time=0.5):
        num_frames = int(sample_rate) + 1
        frames = np.repeat(self.skeleton.reference_local_transforms.numpy()[None], num_frames, axis=0)
        times = np.arange(num_frames) / sample_rate
        frames[:, 0, 0] = 10 + times * 3
        frames[:, 0, 1] = 1 + times
        frames[:, 0, 2] = 20 - times * 6
        clip = AnimationBuffer(self.skeleton, sample_rate, wp.array(frames, dtype=wp.transform))
        return PoseEntry(0, "Start", Path("test.bvh"), clip, sample_time, wp.transform_identity())

    def make_dof_entry(self, sample_rate=120.0, sample_time=0.5):
        # Z-up root motion and joint angles, like a G1 CSV clip.
        num_frames = int(sample_rate) + 1
        times = np.arange(num_frames) / sample_rate
        frames = np.zeros((num_frames, 7 + len(_G1_JOINTS)), dtype=np.float32)
        frames[:, :3] = np.array([10, 20, 1]) + times[:, None] * [3, -6, 1]
        frames[:, 6] = 1.0
        frames[:, 7:] = times[:, None] * [2, -1]
        clip = DOFAnimationBuffer(_G1_JOINTS, sample_rate, wp.array(frames, dtype=wp.float32))
        return PoseEntry(0, "Start", Path("test.csv"), clip, sample_time, wp.transform_identity())

    def make_demo(self, g1=False):
        # Exercise app behavior without opening a GL window or loading TensorRT.
        demo = InbetweenerDemo.__new__(InbetweenerDemo)
        demo.engine = _RecordingEngine()
        demo.viewer = Mock()
        demo.mesh_renderer = Mock()
        demo.skeleton_renderer = Mock()
        demo.playback_color = None if g1 else wp.vec3(0.7, 0.7, 0.7)
        demo.velocity_renderer = Mock()
        demo.foot_contact_renderer = Mock()
        demo.show_foot_contacts = False
        demo.state = Mock()
        demo.selected_index = 0
        demo._next_id = 1
        demo._rename = InlineRename()
        demo.error = None
        if g1:
            demo.clip_registry = ClipRegistry(ModelRig.UNITREE_G1, joint_names=_G1_JOINTS)
            demo.model_to_runtime = wp.transform_identity()
            demo.pose_entries = [self.make_dof_entry()]
            demo.context = DOFToken(len(_G1_JOINTS))
            demo.target = DOFToken(len(_G1_JOINTS))
            demo.output = DOFAnimationBuffer(
                _G1_JOINTS, 30.0, wp.empty((20, 7 + len(_G1_JOINTS)), dtype=wp.float32), num_frames=0,
            )
        else:
            demo.clip_registry = ClipRegistry(ModelRig.SOMA, skeleton=self.skeleton)
            converter = SpaceConverter(FacingDirectionType.MUJOCO)
            demo.model_to_runtime = converter.convert_transform(wp.transform_identity())
            demo.pose_entries = [self.make_entry()]
            demo.context = PoseToken(2)
            demo.target = PoseToken(2)
            demo.output = AnimationBuffer(
                self.skeleton, 30.0, wp.empty((20, self.skeleton.num_joints), dtype=wp.transform), num_frames=0,
            )
        demo.current_pose = wp.empty_like(demo.context.poses[0])
        demo.refresh_poses()
        return demo

    def test_windows_resample_source_fps_on_each_side_of_visible_pose(self):
        # The conversion moves BVH X, Y (height), and Z to scene Y, Z, and X.
        conversion = SpaceConverter(FacingDirectionType.MAYA).convert_transform(wp.transform_identity())
        for source_rate in (15, 30, 60):
            for context in (True, False):
                with self.subTest(source_rate=source_rate, context=context):
                    entry = self.make_entry(source_rate)
                    window = wp.empty((4, self.skeleton.num_joints), dtype=wp.transform)
                    sample_pose_window(entry, conversion, window, anchor=3 if context else 0)
                    poses = window.numpy()
                    offsets = (np.arange(4) - (3 if context else 0)) / 30
                    np.testing.assert_allclose(poses[:, 0, 0], offsets * -6, atol=1e-5)
                    np.testing.assert_allclose(poses[:, 0, 1], offsets * 3, atol=1e-5)
                    np.testing.assert_allclose(poses[:, 0, 2], 1.5 + offsets, atol=1e-5)
                    np.testing.assert_allclose(poses[:, 1, 1], 0.2)

    def test_dof_windows_keep_height_rotation_and_joint_angles(self):
        placement = wp.transform(wp.vec3(4, 5, 0), wp.quat_from_axis_angle(wp.vec3(0, 0, 1), 0.4))
        rotation = wp.quat_from_axis_angle(wp.vec3(0, 0, 1), 0.7)
        for source_rate, context in product((15, 30, 120), (True, False)):
            with self.subTest(source_rate=source_rate, context=context):
                entry = self.make_dof_entry(source_rate)
                frames = entry.clip.poses.numpy()
                frames[:, 3:7] = np.asarray(rotation)
                entry.clip.poses.assign(frames)
                entry.placement = placement
                window = wp.empty((4, 7 + len(_G1_JOINTS)), dtype=wp.float32)
                sample_pose_window(entry, wp.transform_identity(), window, anchor=3 if context else 0)
                poses = window.numpy()
                offsets = (np.arange(4) - (3 if context else 0)) / 30
                for pose, offset in zip(poses, offsets):
                    # Ground motion is relative to the visible pose. Height and rotation are kept.
                    root = wp.transform(wp.vec3(3 * offset, -6 * offset, 1.5 + offset), rotation)
                    np.testing.assert_allclose(pose[:7], np.asarray(wp.mul(placement, root)), atol=3e-6)
                np.testing.assert_allclose(poses[:, 7:], (0.5 + offsets[:, None]) * [2, -1], atol=1e-6)

    def test_hips_constraint_preserves_rotation_height_and_local_children(self):
        entry = self.make_entry()
        frames = entry.clip.poses.numpy()
        frames[:, 0, 3:] = np.asarray(wp.quat_from_axis_angle(wp.vec3(0, 1, 0), 0.7))
        entry.clip.poses.assign(frames)
        entry.placement = wp.transform(
            wp.vec3(4, 5, 2), wp.quat_from_axis_angle(wp.vec3(0, 0, 1), 0.4)
        )
        conversion = SpaceConverter(FacingDirectionType.MAYA).convert_transform(wp.transform_identity())
        window = wp.empty((4, self.skeleton.num_joints), dtype=wp.transform)
        sample_pose_window(entry, conversion, window, anchor=0)
        poses = window.numpy()
        root = wp.transform(wp.vec3(0, 1.5, 0), wp.quat(*frames[30, 0, 3:]))
        expected = wp.mul(entry.placement, wp.mul(conversion, root))
        np.testing.assert_allclose(poses[0, 0], np.asarray(expected), atol=3e-6)
        np.testing.assert_allclose(poses[:, 1], frames[30:34, 1], atol=1e-6)
        self.assertAlmostEqual(float(np.linalg.norm(poses[1, 0, :3] - poses[0, 0, :3])),
                               float(np.linalg.norm([3, 1, -6]) / 30), places=5)

    def test_list_edits_keep_the_start_first_and_update_the_ground_route(self):
        demo = self.make_demo()
        for _ in range(3):
            demo.add_target()
        positions = np.array([[0, 0], [2, 1], [5, -1], [9, 2]])
        for entry, (x, y) in zip(demo.pose_entries, positions):
            entry.placement = wp.transform(wp.vec3(float(x), float(y), 0), wp.quat_identity())
        # The start cannot be removed, moved or displaced.
        demo.selected_index = 0
        demo.remove_target()
        demo.move_target(0, 1)
        demo.move_target(1, 0)
        self.assertEqual([entry.id for entry in demo.pose_entries], [0, 1, 2, 3])
        demo.move_target(3, 1)
        self.assertEqual([entry.id for entry in demo.pose_entries], [0, 3, 1, 2])
        self.assertEqual(demo.selected_index, 1)
        ordered_positions = positions[[0, 3, 1, 2]]
        np.testing.assert_allclose(demo.order_starts.numpy()[:, :2], ordered_positions[:-1])
        np.testing.assert_allclose(demo.order_ends.numpy()[:, :2], ordered_positions[1:])
        rings = demo.ring_starts.numpy().reshape(len(demo.pose_entries), -1, 3)
        np.testing.assert_allclose(rings[:, :, :2].mean(axis=1), ordered_positions, atol=1e-6)
        demo.move_target(1, 3)
        self.assertEqual([entry.id for entry in demo.pose_entries], [0, 1, 2, 3])
        for _ in range(3):
            demo.remove_target()
        self.assertEqual(demo.order_starts.shape, (0,))
        self.assertEqual(demo.order_ends.shape, (0,))

    def test_generation_follows_list_and_carries_last_four_valid_output_frames(self):
        demo = self.make_demo()
        demo.add_target()
        demo.add_target()
        demo.move_target(2, 1)
        demo.pose_entries[1].min_tokens, demo.pose_entries[1].max_tokens = 6, 8
        demo.pose_entries[2].min_tokens = demo.pose_entries[2].max_tokens = 16
        initial_context = demo.pose_entries[0].sampled_poses.numpy()
        demo.play()
        demo.step(0.0)
        self.assertEqual(len(demo.engine.calls), 1)
        np.testing.assert_allclose(demo.engine.calls[0][0], initial_context)
        np.testing.assert_allclose(demo.engine.calls[0][1], demo.pose_entries[1].sampled_poses.numpy())
        demo.step(demo.output.duration / 2)
        self.assertEqual(len(demo.engine.calls), 1)
        expected_context = demo.output.poses.numpy()[2:6].copy()
        demo.step(demo.output.duration)
        self.assertEqual(len(demo.engine.calls), 1)
        demo.step(0.0)
        self.assertEqual(len(demo.engine.calls), 2)
        self.assertEqual(demo.engine.token_counts, [(6, 7, 8), (16,)])
        np.testing.assert_allclose(demo.engine.calls[1][0], expected_context)
        np.testing.assert_allclose(demo.engine.calls[1][1], demo.pose_entries[2].sampled_poses.numpy())
        demo.step(demo.output.duration)
        self.assertFalse(demo.playing)
        self.assertFalse(demo.has_segment)
        demo.step(10.0)
        self.assertEqual(len(demo.engine.calls), 2)

    def test_target_frame_count_controls_conditioning_and_stop_velocity(self):
        for g1, count in product((False, True), (1, 4)):
            with self.subTest(g1=g1, count=count):
                demo = self.make_demo(g1)
                demo.add_target()
                demo.pose_entries[1].num_poses_to_use = count
                demo.refresh_poses()
                demo.play()
                demo.step(0.0)
                self.assertEqual(demo.engine.target_counts, [count])
                self.assertEqual(demo.context.num_poses, 4)
                expected = demo.pose_entries[1].sampled_poses.numpy()
                expected[count:] = expected[count - 1]
                np.testing.assert_allclose(demo.engine.calls[0][1], expected)
                if count == 1:
                    np.testing.assert_array_equal(demo.pose_entries[1].velocity, [0, 0, 0])
                else:
                    self.assertAlmostEqual(float(np.linalg.norm(demo.pose_entries[1].velocity)),
                                           float(np.linalg.norm([3, 1, -6])), places=4)

    def test_early_transition_still_plays_the_four_poses_of_the_next_context(self):
        # The output has six frames, and a segment plays at least four of them.
        cases = ((0, 5), (1, 4), (2, 3), (4, 3), (20, 3))
        for g1, (early_frames, last_frame) in product((False, True), cases):
            with self.subTest(g1=g1, early_frames=early_frames):
                demo = self.make_demo(g1)
                demo.add_target()
                demo.add_target()
                demo.pose_entries[1].transition_early_by_frames = early_frames
                demo.play()
                demo.step(0.0)
                np.testing.assert_allclose(demo.engine.calls[0][0], demo.pose_entries[0].sampled_poses.numpy())
                played = demo.output.poses.numpy()[:last_frame + 1].copy()
                demo.step(10.0)
                self.assertAlmostEqual(demo.playback_time, last_frame / 30.0)
                np.testing.assert_allclose(demo.current_pose.numpy(), played[-1])
                demo.step(0.0)
                np.testing.assert_allclose(demo.engine.calls[1][0], played[-4:])

    def test_pause_and_edit_reset_do_not_reuse_stale_motion(self):
        demo = self.make_demo()
        demo.add_target()
        demo.play()
        demo.step(0.0)
        pose = demo.current_pose.numpy().copy()
        demo.playing = False
        demo.step(10.0)
        np.testing.assert_allclose(demo.current_pose.numpy(), pose)
        demo.playing = True
        demo.step(demo.output.duration / 2)
        self.assertGreater(demo.playback_time, 0)
        demo.pose_entries[0].placement = wp.transform(wp.vec3(9, 0, 0), wp.quat_identity())
        demo.refresh_poses()
        self.assertFalse(demo.playing)
        self.assertFalse(demo.has_segment)
        self.assertEqual(demo.next_target_index, 1)
        np.testing.assert_allclose(demo.current_pose.numpy(), demo.pose_entries[0].sampled_poses.numpy()[-1])
        demo.play()
        demo.step(0.0)
        np.testing.assert_allclose(demo.engine.calls[-1][0], demo.pose_entries[0].sampled_poses.numpy())

    def test_failed_clip_load_keeps_the_original_entry(self):
        demo = self.make_demo()
        demo._pending_clip = demo.pose_entries[0]
        original = demo.pose_entries[0].clip
        with patch("app.authoring.common.choose_file", return_value="bad.bvh"), \
             patch.object(demo.clip_registry, "load", side_effect=ValueError("Invalid BVH")):
            demo._choose_pending_clip()
        self.assertIs(demo.pose_entries[0].clip, original)
        self.assertEqual(demo.pose_entries[0].path, Path("test.bvh"))
        self.assertIn("Invalid BVH", demo.error)

    def test_mesh_colors_and_pose_ids_survive_reordering_and_removal(self):
        demo = self.make_demo()
        demo.add_target()
        demo.add_target()
        colors = {0: (0.2, 0.8, 0.4), 1: (0.7, 0.1, 0.2), 2: (0.3, 0.2, 0.9)}
        for entry in demo.pose_entries:
            entry.color = colors[entry.id]
        demo.move_target(2, 1)
        demo.render()
        calls = demo.mesh_renderer.draw_pose.call_args_list
        self.assertEqual([call.args[4] for call in calls], [0, 2, 1])
        for index, call in enumerate(calls):
            np.testing.assert_allclose(call.args[1].numpy(), demo.pose_entries[index].preview_pose.numpy())
            np.testing.assert_allclose(np.asarray(call.args[3]), colors[call.args[4]])
        demo.mesh_renderer.reset_mock()
        demo.selected_index = 2
        demo.remove_target()
        demo.mesh_renderer.clear.assert_called_once_with(demo.viewer)
        demo.render()
        self.assertEqual([call.args[4] for call in demo.mesh_renderer.draw_pose.call_args_list], [0, 2])

    def test_ground_rings_select_poses_with_hidden_meshes(self):
        demo = self.make_demo()
        demo.viewer.camera.far = 100.0
        demo.add_target()
        demo.add_target()
        for entry in demo.pose_entries:
            entry.show_mesh = False
        demo.pose_entries[2].placement = wp.transform(4, 5, 0, 0, 0, 0, 1)
        demo.refresh_poses()
        down = wp.vec3(0, 0, -1)
        demo.selected_index = 0
        demo._select_pose(wp.vec3(4.3, 5, 2), down)
        self.assertEqual(demo.pose_entries[demo.selected_index].id, 2)
        # The empty center of a ring misses.
        demo.selected_index = 0
        demo._select_pose(wp.vec3(4, 5, 2), down)
        self.assertEqual(demo.selected_index, 0)
        demo.mesh_renderer.raycast_pose.assert_not_called()

    def test_gizmo_edit_updates_placement_and_resets_playback(self):
        demo = self.make_demo()
        demo.add_target()
        entry = demo.pose_entries[demo.selected_index]
        entry.placement = wp.transform(wp.vec3(1, 2, 3), wp.quat_identity())
        demo.refresh_poses()
        demo.play()
        demo.step(0.0)
        rotation = wp.quat_from_axis_angle(wp.vec3(0, 0, 1), 0.5)

        def drag_gizmo():
            gizmo = demo.viewer.log_gizmo.call_args.args[1]
            gizmo[:] = wp.transform(wp.vec3(4, 5, 3), rotation)

        demo.viewer.end_frame.side_effect = drag_gizmo
        demo.render()

        np.testing.assert_allclose(entry.placement.p, [4, 5, 3])
        np.testing.assert_allclose(entry.placement.q, rotation)
        self.assertFalse(demo.playing)
        self.assertFalse(demo.has_segment)

    def test_generate_survives_idle_gizmo_redraw_after_pose_rotation(self):
        for angle in (90, 180, 270, -180, 360):
            with self.subTest(angle=angle):
                demo = self.make_demo()
                demo.add_target()
                entry = demo.pose_entries[demo.selected_index]
                entry.placement = wp.transform(
                    wp.vec3(2, 1, 0), wp.quat_rpy(0.0, 0.0, wp.radians(float(angle))),
                )
                entry.euler_degrees = [0, 0, angle]
                demo.refresh_poses()

                def redraw_gizmo():
                    # Newton converts through a matrix even without a user edit.
                    # This can return -q, which is the same rotation as q.
                    gizmo = demo.viewer.log_gizmo.call_args.args[1]
                    gizmo[:] = wp.transform_from_matrix(wp.transform_to_matrix(gizmo))

                demo.viewer.end_frame.side_effect = redraw_gizmo
                demo.play()
                for _ in range(3):
                    demo.render()
                    demo.step(1 / 60)

                self.assertEqual(len(demo.engine.calls), 1)
                self.assertTrue(demo.playing)
                self.assertGreater(demo.playback_time, 0.0)

    def test_playback_mesh_hides_only_where_a_visible_preview_shows_its_pose(self):
        for early_frames, preview_visible in ((1, True), (2, True), (0, False), (0, True)):
            with self.subTest(early_frames=early_frames, preview_visible=preview_visible):
                demo = self.make_demo()
                demo.add_target()
                demo.add_target()
                demo.pose_entries[1].transition_early_by_frames = early_frames
                demo.pose_entries[1].show_mesh = preview_visible

                def playback_drawn():
                    demo.mesh_renderer.reset_mock()
                    demo.render()
                    return -1 in [call.args[4] for call in demo.mesh_renderer.draw_pose.call_args_list]

                # The visible Start mesh shows the first frame.
                demo.play()
                demo.step(0.0)
                self.assertFalse(playback_drawn())
                demo.step(demo.output.duration / 2)
                self.assertTrue(playback_drawn())
                demo.step(10.0)
                boundary_pose = demo.current_pose.numpy().copy()

                # The end of one segment and frame zero of the next must agree
                # on visibility: a preview replaces only a reached target.
                for generate_next in (False, True):
                    if generate_next:
                        demo.step(0.0)
                    self.assertEqual(playback_drawn(), early_frames > 0 or not preview_visible)
                    np.testing.assert_allclose(demo.current_pose.numpy(), boundary_pose)

    def test_contacts_follow_the_final_pose_without_a_mesh_draw_and_respect_the_threshold(self):
        demo = self.make_demo()
        demo.rig = ModelRig.SOMA
        points = np.array([[0, 0, 0], [0.1, 0, 0], [0, 0, 0.1]], dtype=np.float32)
        demo.foot_contact_renderer = SkeletonFootContactRenderer(
            "contacts", self.skeleton, [ContactOutline(0, 1, points)],
        )
        demo.foot_contacts = wp.empty(4, dtype=wp.float32)
        demo.output.foot_contacts = wp.full((demo.output.capacity, 4), 0.75, dtype=wp.float32)
        demo.show_foot_contacts = True
        demo.foot_contact_threshold = 0.5
        demo.add_target()
        demo.play()
        demo.step(0.0)
        demo.step(demo.output.duration)

        demo._draw_foot_contacts()

        root, child = (wp.transform(*pose) for pose in demo.current_pose.numpy())
        expected = np.array([wp.transform_point(wp.mul(root, child), wp.vec3(*point)) for point in points])
        _, starts, ends, _ = demo.viewer.log_lines.call_args.args
        np.testing.assert_allclose(starts.numpy(), expected, atol=1e-6)
        np.testing.assert_allclose(ends.numpy(), np.roll(expected, -1, axis=0), atol=1e-6)
        demo.foot_contact_threshold = 0.8
        demo._draw_foot_contacts()
        np.testing.assert_allclose(ends.numpy(), starts.numpy())
        demo.reset()
        demo._draw_foot_contacts()
        self.assertTrue(demo.viewer.log_lines.call_args.kwargs["hidden"])

    def test_g1_contacts_follow_the_final_pose_without_a_mesh_draw(self):
        demo = self.make_demo(g1=True)
        demo.rig = ModelRig.UNITREE_G1
        # A free root, then the hip and knee hinges, matching the fixture's joint_q.
        builder = newton.ModelBuilder()
        pelvis, thigh, shin = (builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3))) for _ in range(3))
        builder.add_articulation([
            builder.add_joint_free(pelvis),
            builder.add_joint_revolute(pelvis, thigh, axis=wp.vec3(0.0, 0.0, 1.0)),
            builder.add_joint_revolute(thigh, shin, axis=wp.vec3(0.0, 0.0, 1.0)),
        ])
        model = builder.finalize()
        points = np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0]], dtype=np.float32)
        demo.foot_contact_renderer = DOFFootContactRenderer("contacts", model, [ContactOutline(0, shin, points)])
        demo.foot_contacts = wp.empty(4, dtype=wp.float32)
        demo.output.foot_contacts = wp.full((demo.output.capacity, 4), 0.75, dtype=wp.float32)
        demo.show_foot_contacts = True
        demo.foot_contact_threshold = 0.5
        demo.add_target()
        demo.play()
        demo.step(0.0)
        demo.step(demo.output.duration)

        demo._draw_foot_contacts()

        pose = demo.current_pose.numpy()
        root = wp.transform(wp.vec3(*pose[:3]), wp.quat(*pose[3:7]))
        bend = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(pose[7] + pose[8]))
        shin_transform = wp.mul(root, wp.transform(wp.vec3(0.0, 0.0, 0.0), bend))
        expected = np.array([wp.transform_point(shin_transform, wp.vec3(*point)) for point in points])
        np.testing.assert_allclose(demo.viewer.log_lines.call_args.args[1].numpy(), expected, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
