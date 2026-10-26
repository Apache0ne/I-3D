# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import newton
import numpy as np
import warp as wp
from imgui_bundle import imgui

from app.authoring.common import InlineRename, draw_clip_scrubber
from app.runtime.common.clip_registry import ClipAsset, ClipRegistry
from app.runtime.common.io.document_io import DocumentError
from app.runtime.common.io.smart_primitive_document import (
    SmartPrimitiveAssetPool,
    InteractionBoxComponent,
    PivotComponent,
    PlaybackMode,
    SmartPrimitiveDocument,
    SmartPrimitiveDocumentError,
    TargetPoseComponent,
    load_smart_primitive_document,
    load_smart_primitive_rig,
    save_smart_primitive_document,
)
from app.smart_primitive_editor import (
    _PIVOT_ARROW_LIFT,
    SmartPrimitiveEditor,
    _HipsTrajectoryRenderer,
    _IKPinRenderer,
    _InteractionBoxRenderer,
    _PivotRenderer,
    _Rig,
)
from motionbricks.animation.animation_buffer import AnimationBuffer
from motionbricks.animation.mesh import SkeletalMesh, SkinnedMesh
from motionbricks.animation.skeleton import Skeleton
from motionbricks.io.csv import save_csv
from motionbricks.model.contract import ModelRig
from motionbricks.renderers.skeletal_mesh_renderer import SkeletalMeshRenderer
from motionbricks.renderers.vector_arrow_renderer import VectorArrowRenderer
from motionbricks.robotics.dof_animation_buffer import DOFAnimationBuffer
from motionbricks.runtime.pivot import Axis, ConeSide, ConstraintType, OperationOrder
from motionbricks.runtime.target_pose import IKPin, IK_PINS


# BVH clips store height along Y.
_BVH_UP_INDEX = 1


def _hips_pose(x: float) -> np.ndarray:
    return np.array(
        [[x, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def _camera_state() -> dict:
    return {
        "pos": [1.0, 2.0, 3.0],
        "pivot": [4.0, 5.0, 6.0],
        "pitch": 7.0,
        "yaw": 8.0,
        "fov": 45.0,
        "near": 0.01,
        "far": 1000.0,
    }


class _ArrowViewer:
    def __init__(self) -> None:
        self.starts = None
        self.ends = None
        self.lines = {}

    def log_arrows(self, name, starts, ends, colors, hidden=False) -> None:
        self.starts = starts.numpy().copy()
        self.ends = ends.numpy().copy()

    def log_lines(self, name, starts, ends, colors, hidden=False) -> None:
        self.lines[name] = (starts.numpy().copy(), ends.numpy().copy())


class _MeshViewer:
    def __init__(self) -> None:
        self.points = {}

    def log_mesh(self, name, points, indices, **options) -> None:
        self.points[name] = points.numpy().copy()


class _ShapeViewer:
    def __init__(self) -> None:
        self.shapes = {}

    def log_shapes(
        self,
        name,
        geo_type,
        geo_scale,
        xforms,
        colors=None,
        materials=None,
        geo_thickness=0.0,
        geo_is_solid=True,
        geo_src=None,
        hidden=False,
    ) -> None:
        self.shapes[name] = (geo_type, xforms.numpy().copy(), geo_is_solid)


class _EmptyClipRegistry:
    rig = ModelRig.SOMA


class _TitleRenderer:
    def set_title(self, title: str) -> None:
        pass


class _CameraVector(tuple):
    def __new__(cls, x: float, y: float, z: float):
        return super().__new__(cls, (x, y, z))


class _Camera:
    def __init__(self, state: dict) -> None:
        self.pos = _CameraVector(*state["pos"])
        self.pivot = _CameraVector(*state["pivot"])
        self.pitch = state["pitch"]
        self.yaw = state["yaw"]
        self.fov = state["fov"]
        self.near = state["near"]
        self.far = state["far"]


class _TitleViewer:
    def __init__(self) -> None:
        self.renderer = _TitleRenderer()
        self.camera = _Camera(_camera_state())


class _ClearableRenderers:
    def __init__(self) -> None:
        self.clear_count = 0

    def clear(self, viewer) -> None:
        self.clear_count += 1


def _create_imgui_context():
    """Create an ImGui context that draws editor windows without a display."""
    context = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename("")
    io.display_size = (1000.0, 700.0)
    io.delta_time = 1.0 / 60.0
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    return context


def _loading_editor(rig) -> SmartPrimitiveEditor:
    """An editor without a window, ready to open a document."""
    editor = SmartPrimitiveEditor.__new__(SmartPrimitiveEditor)
    editor.viewer = _TitleViewer()
    editor.interaction_box_renderer = _ClearableRenderers()
    editor.pivot_renderer = _ClearableRenderers()
    editor.rig = rig
    editor.authoring = SmartPrimitiveDocument(rig.clip_registry.rig, rig.reference_pose, rig.model_to_runtime)
    editor.selected_id = editor.authoring.root.id
    editor._document_path = None
    editor._status = object()
    editor._node_rename = InlineRename()
    editor._gizmo_item_id = None
    editor._gizmo_xform = None
    return editor


class _InputIdRecordingUi:
    def __init__(self, ui) -> None:
        self._ui = ui
        self.input_ids = {}

    def __getattr__(self, name):
        return getattr(self._ui, name)

    def input_float(self, label, *args, **kwargs):
        result = self._ui.input_float(label, *args, **kwargs)
        self.input_ids[label] = self._ui.get_item_id()
        return result


class SmartPrimitiveUiIntegrationTests(unittest.TestCase):
    def test_shared_scrubber_uses_source_frames_and_snaps_only_when_requested(self):
        clip = SimpleNamespace(num_frames=61, sample_rate=60.0)
        for animation, initial_time, show_frames, toggle, scrub_value, expected_time, expected_mode in (
            (clip, 0.126, False, True, None, 8 / 60, True),
            (clip, 0.126, False, False, None, 0.126, False),
            (clip, 0.5, True, True, None, 0.5, False),
            (clip, 0.0, True, False, 30, 0.5, True),
            (clip, 0.0, False, False, 0.4, 0.4, False),
            (None, 0.0, True, False, None, 0.0, True),
        ):
            with self.subTest(animation=animation, initial_time=initial_time, show_frames=show_frames, toggle=toggle):
                ui = Mock()
                ui.button.return_value = toggle
                ui.slider_int.side_effect = ui.slider_float.side_effect = (
                    lambda _label, value, *_, **__: (
                        scrub_value is not None, value if scrub_value is None else scrub_value,
                    )
                )
                changed, sample_time, mode = draw_clip_scrubber(ui, animation, initial_time, show_frames, 1.0)
                self.assertAlmostEqual(sample_time, expected_time)
                self.assertEqual(mode, expected_mode)
                self.assertEqual(changed, expected_time != initial_time)

    def test_target_pose_ik_pins_turn_on_and_off_in_the_mask(self) -> None:
        component = TargetPoseComponent.__new__(TargetPoseComponent)
        component.ik_pin_mask = IKPin.NONE

        for pin in IK_PINS:
            component.set_ik_pin_enabled(pin, True)
        self.assertEqual(component.ik_pin_mask, IKPin.ALL)

        component.set_ik_pin_enabled(IKPin.RIGHT_HAND, False)
        self.assertEqual(component.ik_pin_mask, IKPin.ALL & ~IKPin.RIGHT_HAND)

    def test_target_pose_token_range_defaults_and_clamps(self) -> None:
        component = TargetPoseComponent.__new__(TargetPoseComponent)
        component.set_token_range(6, 14)
        for requested, expected in (
            ((7, 9), (7, 9)), ((2, 20), (6, 16)), ((10, 8), (10, 10)), ((17, 3), (16, 16)),
        ):
            with self.subTest(requested=requested):
                component.set_token_range(*requested)
                self.assertEqual((component.min_tokens, component.max_tokens), expected)

    def test_transform_fields_have_node_specific_imgui_ids(self) -> None:
        context = _create_imgui_context()
        try:
            authoring = SmartPrimitiveDocument(
                "soma",
                SimpleNamespace(),
                wp.transform_identity(),
            )
            child = authoring.add_node()
            editor = SmartPrimitiveEditor.__new__(SmartPrimitiveEditor)
            editor.authoring = authoring
            editor.selected_id = authoring.root.id
            editor._gizmo_item_id = None
            editor._gizmo_xform = None
            editor._status = None
            ui = _InputIdRecordingUi(imgui)

            field_ids = []
            for selected in (authoring.root, child):
                editor.selected_id = selected.id
                ui.input_ids = {}
                imgui.new_frame()
                editor._draw_properties_window(ui, 1.0)
                imgui.render()
                field_ids.append(ui.input_ids.copy())

            self.assertEqual(field_ids[0].keys(), field_ids[1].keys())
            self.assertTrue(field_ids[0])
            for label in field_ids[0]:
                with self.subTest(label=label):
                    self.assertNotEqual(field_ids[0][label], field_ids[1][label])
        finally:
            imgui.destroy_context(context)


class SmartPrimitiveEditorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        wp.init()
        if not wp.is_cuda_available():
            raise unittest.SkipTest("CUDA is required for editor pose sampling.")
        cls.previous_device = wp.get_device()
        cls.device = wp.get_device("cuda:0")
        wp.set_device(cls.device)

        cls.skeleton = Skeleton(
            ["Hips"],
            [-1],
            _hips_pose(0.0),
        )
        # Interval velocities are 1, 3, and 5 meters per second.
        positions = (0.0, 1.0 / 30.0, 4.0 / 30.0, 9.0 / 30.0)
        poses = wp.array(
            np.stack([_hips_pose(position) for position in positions]),
            dtype=wp.transform,
        )
        animation = AnimationBuffer(cls.skeleton, 30.0, poses)
        cls.clip = ClipAsset(Path("velocity_test.bvh"), animation)

        trajectory_positions = np.arange(61, dtype=np.float32) / 30.0
        trajectory_poses = wp.array(
            np.stack([_hips_pose(position) for position in trajectory_positions]),
            dtype=wp.transform,
        )
        trajectory_animation = AnimationBuffer(cls.skeleton, 30.0, trajectory_poses)
        cls.trajectory_clip = ClipAsset(
            Path("trajectory_test.bvh"),
            trajectory_animation,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        wp.set_device(cls.previous_device)

    def setUp(self) -> None:
        self.component = TargetPoseComponent(7, self.skeleton.reference_local_transforms, _BVH_UP_INDEX)
        self.component.set_clip(self.clip)

    def test_target_pose_velocity_requires_two_poses(self) -> None:
        self.component.set_num_poses_to_use(1)
        np.testing.assert_array_equal(
            [float(self.component.velocity[index]) for index in range(3)],
            np.zeros(3),
        )
        self.assertEqual(self.component.velocity_magnitude, 0.0)
        self.assertFalse(self.component.has_velocity)

        for frame, num_poses, expected in ((0, 2, 1.0), (0, 3, 1.0), (0, 4, 1.0), (1, 3, 3.0)):
            with self.subTest(frame=frame, num_poses=num_poses):
                self.component.set_time(frame / 30.0)
                self.component.set_num_poses_to_use(num_poses)
                self.assertAlmostEqual(
                    float(self.component.velocity[0]),
                    expected,
                    places=5,
                )
                self.assertAlmostEqual(
                    self.component.velocity_magnitude,
                    expected,
                    places=5,
                )
                self.assertTrue(self.component.has_velocity)

    def test_trajectory_is_refreshed_after_edits(self) -> None:
        self.component.set_clip(self.trajectory_clip)
        self.component.set_time(30 / 30.0)
        trajectory = self.component.trajectory_transforms
        np.testing.assert_allclose(
            trajectory.numpy()[:, 0], np.arange(10, 51, 2) / 30.0, atol=1.0e-6,
        )

        # The preview uses the latest settings after several edits.
        self.component.set_time(45 / 30.0)
        self.component.constrain_hips(True)
        self.component.set_trajectory_scale(1.0)
        self.assertIs(self.component.trajectory_transforms, trajectory)
        np.testing.assert_allclose(
            trajectory.numpy()[:, 0], np.arange(-10, 11) / 30.0, atol=1.0e-6,
        )

    def test_target_pose_trajectory_scale_controls_sample_interval(self) -> None:
        self.component.set_clip(self.trajectory_clip)
        self.component.set_time(30 / 30.0)
        trajectory_buffer = self.component.trajectory_transforms
        sample_offsets = np.arange(-10, 11, dtype=np.float32)

        for scale in (1.0, 2.0, 3.0, 4.0):
            with self.subTest(scale=scale):
                self.component.set_trajectory_scale(scale)
                expected_frames = np.clip(
                    30.0 + sample_offsets * scale,
                    0.0,
                    60.0,
                )
                np.testing.assert_allclose(
                    self.component.trajectory_transforms.numpy()[:, 0],
                    expected_frames / 30.0,
                    atol=1.0e-6,
                )
                self.assertIs(
                    self.component.trajectory_transforms,
                    trajectory_buffer,
                )

        for invalid_scale in (0.99, 4.01, float("nan")):
            with (
                self.subTest(invalid_scale=invalid_scale),
                self.assertRaisesRegex(
                    ValueError,
                    "Trajectory scale",
                ),
            ):
                self.component.set_trajectory_scale(invalid_scale)

    def test_sampling_reuses_allocated_pose_buffers(self) -> None:
        display_pose = self.component.pose
        sampled_poses = self.component.sampled_poses
        trajectory = self.component.trajectory_transforms

        self.component.set_time(1 / 30.0)
        self.component.constrain_hips(True)
        self.component.set_clip(self.trajectory_clip)

        self.assertIs(self.component.pose, display_pose)
        self.assertIs(self.component.sampled_poses, sampled_poses)
        self.assertIs(self.component.trajectory_transforms, trajectory)

    def test_velocity_arrow_uses_displayed_hips_and_node_world_transform(self) -> None:
        self.component.set_num_poses_to_use(4)
        renderer = VectorArrowRenderer("target_pose_velocity", wp.vec3(1.0, 0.0, 0.0))
        viewer = _ArrowViewer()
        world_xform = wp.transform(
            wp.vec3(10.0, 20.0, 30.0),
            wp.quat_rpy(0.0, 0.0, 0.5 * wp.pi),
        )

        renderer.draw(
            viewer,
            self.component.id,
            world_xform,
            self.component.hips_position,
            self.component.velocity,
        )

        np.testing.assert_allclose(viewer.starts[0], [10.0, 20.0, 30.0], atol=1.0e-5)
        np.testing.assert_allclose(viewer.ends[0], [10.0, 21.0, 30.0], atol=1.0e-5)

    def test_interaction_box_renderer_draws_only_box_edges(self) -> None:
        component = InteractionBoxComponent(12)
        component.set_dimensions(2.0, 4.0, 6.0)
        self.assertEqual(
            (component.width, component.depth, component.height),
            (2.0, 4.0, 6.0),
        )
        renderer = _InteractionBoxRenderer()
        viewer = _ArrowViewer()
        world_xform = wp.transform(
            wp.vec3(1.0, 2.0, 3.0),
            wp.quat_identity(),
        )

        renderer.draw(viewer, component, world_xform)

        starts, ends = viewer.lines["/interaction_box_12"]
        points = np.concatenate((starts, ends), axis=0)
        self.assertEqual(starts.shape, (12, 3))
        np.testing.assert_allclose(points.min(axis=0), [0.0, 0.0, 0.0])
        np.testing.assert_allclose(points.max(axis=0), [2.0, 4.0, 6.0])
        np.testing.assert_allclose(points.mean(axis=0), world_xform.p)

        component.set_dimensions(8.0, 10.0, 12.0)
        renderer.draw(viewer, component, world_xform)
        resized_points = np.concatenate(viewer.lines["/interaction_box_12"], axis=0)
        np.testing.assert_allclose(resized_points.mean(axis=0), world_xform.p)

    def test_ik_pin_renderer_draws_selected_frame_zero_joints_as_spheres(
        self,
    ) -> None:
        joint_names = [
            "Hips",
            "LeftArm",
            "LeftForeArm",
            "LeftHand",
            "RightArm",
            "RightForeArm",
            "RightHand",
            "LeftLeg",
            "LeftShin",
            "LeftFoot",
            "RightLeg",
            "RightShin",
            "RightFoot",
        ]
        tip_positions = {
            "LeftHand": (1.0, 0.0, 0.0),
            "RightHand": (2.0, 0.0, 0.0),
            "LeftFoot": (0.0, 3.0, 0.0),
            "RightFoot": (0.0, 4.0, 0.0),
        }
        reference_pose = np.array(
            [
                [*tip_positions.get(name, (0.0, 0.0, 0.0)), 0.0, 0.0, 0.0, 1.0]
                for name in joint_names
            ],
            dtype=np.float32,
        )
        skeleton = Skeleton(
            joint_names,
            [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11],
            reference_pose,
        )
        component = TargetPoseComponent(19, skeleton.reference_local_transforms, _BVH_UP_INDEX)
        component.set_ik_pin_enabled(IKPin.LEFT_HAND, True)
        component.set_ik_pin_enabled(IKPin.RIGHT_FOOT, True)
        renderer = _IKPinRenderer(skeleton)
        viewer = _ShapeViewer()
        world_xform = wp.transform(
            wp.vec3(10.0, 20.0, 30.0),
            wp.quat_identity(),
        )

        renderer.draw(viewer, component, world_xform)

        geo_type, transforms, is_solid = viewer.shapes["/target_pose_ik_pins_19"]
        self.assertEqual(geo_type, newton.GeoType.SPHERE)
        self.assertTrue(is_solid)
        np.testing.assert_allclose(
            transforms[:, :3],
            [[11.0, 20.0, 30.0], [10.0, 24.0, 30.0]],
            atol=1.0e-5,
        )

    def test_trajectory_renderer_draws_path_and_coordinate_frames(self) -> None:
        self.component.set_clip(self.trajectory_clip)
        self.component.set_time(30 / 30.0)
        renderer = _HipsTrajectoryRenderer()
        viewer = _ArrowViewer()
        world_xform = wp.transform(
            wp.vec3(10.0, 20.0, 30.0),
            wp.quat_rpy(0.0, 0.0, 0.5 * wp.pi),
        )

        renderer.draw(
            viewer,
            self.component.id,
            world_xform,
            self.component.trajectory_transforms,
        )

        path_starts, path_ends = viewer.lines["/target_pose_trajectory_7"]
        expected_positions = np.arange(10, 51, 2, dtype=np.float32) / 30.0
        self.assertEqual(path_starts.shape, (20, 3))
        np.testing.assert_allclose(path_starts[:, 0], 10.0, atol=1.0e-5)
        np.testing.assert_allclose(
            path_starts[:, 1],
            20.0 + expected_positions[:-1],
            atol=1.0e-5,
        )
        np.testing.assert_allclose(
            path_ends[:, 1],
            20.0 + expected_positions[1:],
            atol=1.0e-5,
        )

        frame_starts, frame_ends = viewer.lines["/target_pose_trajectory_frames_7"]
        self.assertEqual(frame_starts.shape, (63, 3))
        np.testing.assert_allclose(frame_starts[:3], [[10.0, 20.0 + 10.0 / 30.0, 30.0]] * 3, atol=1.0e-5)
        # The X, Y and Z lines of each frame follow the node's turned axes.
        axes = frame_ends[:3] - frame_starts[:3]
        np.testing.assert_allclose(
            axes / np.linalg.norm(axes, axis=1, keepdims=True),
            [
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            atol=1.0e-5,
        )

    def test_mesh_renderer_uses_precomputed_inverse_bind_transforms(self) -> None:
        skinned_mesh = SkinnedMesh(
            np.array([[2.0, 0.0, 0.0]], dtype=np.float32),
            np.array([0, 0, 0], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.array([1.0], dtype=np.float32),
        )
        inverse_bind_transform = wp.transform(
            wp.vec3(-1.0, 0.0, 0.0),
            wp.quat_identity(),
        )
        skeletal_mesh = SkeletalMesh(
            [skinned_mesh],
            self.skeleton,
            [inverse_bind_transform],
        )
        renderer = SkeletalMeshRenderer(skeletal_mesh)
        viewer = _MeshViewer()
        posed_transforms = wp.array(_hips_pose(3.0), dtype=wp.transform)

        renderer.draw_pose(
            viewer,
            posed_transforms,
            wp.transform_identity(),
            wp.vec3(0.2, 0.4, 0.6),
            11,
        )

        np.testing.assert_allclose(
            skeletal_mesh.inverse_bind_transforms.numpy()[0, :3],
            [-1.0, 0.0, 0.0],
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            viewer.points["/skeletal_mesh_11_0"],
            [[4.0, 0.0, 0.0]],
            atol=1.0e-6,
        )

    def test_hierarchy_traversal_drives_visibility_cycles_and_removal(self) -> None:
        authoring = SmartPrimitiveDocument(
            "soma",
            self.skeleton.reference_local_transforms,
            wp.transform_identity(),
        )
        branch = authoring.add_node()
        sibling = authoring.add_node()
        leaf = authoring.add_node(branch)

        self.assertEqual(
            list(authoring.iter_nodes()),
            [authoring.root, branch, leaf, sibling],
        )
        self.assertEqual(
            list(authoring.iter_ancestors(leaf)),
            [leaf, branch, authoring.root],
        )

        branch.visible = False
        self.assertFalse(authoring.is_visible(leaf))
        self.assertFalse(authoring.can_reparent(branch, leaf))

        authoring.remove_node(branch)
        self.assertEqual(list(authoring.iter_nodes()), [authoring.root, sibling])
        self.assertIsNone(authoring.find_node(branch.id))
        self.assertIsNone(authoring.find_node(leaf.id))

    def test_node_accepts_one_component_of_each_type(self) -> None:
        authoring = SmartPrimitiveDocument(
            "soma",
            self.skeleton.reference_local_transforms,
            wp.transform_identity(),
        )
        node = authoring.add_node()

        box = authoring.add_interaction_box(node)
        target_pose = authoring.add_target_pose(node)
        pivot = authoring.add_pivot(node)

        self.assertIs(authoring.add_interaction_box(node), box)
        self.assertIs(authoring.add_target_pose(node), target_pose)
        self.assertIs(authoring.add_pivot(node), pivot)
        box.set_dimensions(0.0, 0.0, 0.0)
        self.assertEqual((box.width, box.depth, box.height), (0.0, 0.0, 0.0))
        # Components list in a fixed order, whatever order they were added in.
        self.assertEqual(node.components, (target_pose, box, pivot))
        self.assertEqual(list(authoring.iter_interaction_boxes()), [(node, box)])
        self.assertEqual(list(authoring.iter_target_poses()), [(node, target_pose)])
        self.assertEqual(list(authoring.iter_pivots()), [(node, pivot)])

    def test_pivot_round_trip_and_reference_cleanup_preserve_target_steps(self) -> None:
        authoring = SmartPrimitiveDocument("soma", self.skeleton.reference_local_transforms, wp.transform_identity())
        parent = authoring.add_node()
        node = authoring.add_node(parent)
        component = authoring.add_pivot(node)
        component.pivot.set_motion_direction((0.0, -4.0, 3.0))
        component.pivot.operation_order = OperationOrder.ROTATION_THEN_TRANSLATION
        component.pivot.translation_axis = Axis.Y
        component.pivot.translation_constraint = ConstraintType.LINE_PROJECTION
        component.pivot.line_min_offset = 1.25
        component.pivot.line_max_offset = 2.5
        component.pivot.rotation_axis = Axis.X
        component.pivot.rotation_constraint = ConstraintType.CONE
        component.pivot.cone_half_angle_degrees = 32.5
        component.pivot.cone_side = ConeSide.BOTH
        component.show_motion_direction = component.show_rotation_constraint = False
        target = authoring.add_node()
        authoring.add_target_pose(target)
        first, second = authoring.add_sequence(), authoring.add_sequence()
        for sequence in (first, second):
            authoring.add_sequence_node(sequence, target)
            authoring.set_sequence_pivot(sequence, node)
        without_pivot = authoring.add_sequence()
        node.name = "Renamed approach guide"

        with tempfile.TemporaryDirectory() as directory:
            path = save_smart_primitive_document(
                authoring, Path(directory) / "pivot", node.id, _camera_state(),
            )
            loaded, _, _ = load_smart_primitive_document(
                _EmptyClipRegistry(),
                self.skeleton.reference_local_transforms,
                authoring.model_to_runtime,
                path,
            )
        loaded_node = loaded.find_node(node.id)
        loaded_component = loaded_node.pivot
        self.assertIsInstance(loaded_component, PivotComponent)
        np.testing.assert_allclose(loaded_component.pivot.motion_direction, (0.0, -0.8, 0.6))
        self.assertIs(loaded_component.pivot.operation_order, OperationOrder.ROTATION_THEN_TRANSLATION)
        self.assertIs(loaded_component.pivot.translation_axis, Axis.Y)
        self.assertIs(loaded_component.pivot.rotation_axis, Axis.X)
        self.assertIs(loaded_component.pivot.translation_constraint, ConstraintType.LINE_PROJECTION)
        self.assertEqual(loaded_component.pivot.line_min_offset, 1.25)
        self.assertEqual(loaded_component.pivot.line_max_offset, 2.5)
        self.assertIs(loaded_component.pivot.rotation_constraint, ConstraintType.CONE)
        self.assertEqual(loaded_component.pivot.cone_half_angle_degrees, 32.5)
        self.assertIs(loaded_component.pivot.cone_side, ConeSide.BOTH)
        self.assertEqual((
            loaded_component.show_motion_direction,
            loaded_component.show_translation_constraint,
            loaded_component.show_rotation_constraint,
        ), (False, True, False))
        self.assertEqual(loaded_node.name, node.name)
        self.assertEqual([s.pivot_node_id for s in loaded.sequences], [node.id, node.id, None])
        self.assertEqual([s.node_ids for s in loaded.sequences], [[target.id], [target.id], []])

        # Removing the component clears references but leaves target steps alone.
        loaded.remove_component(loaded_node, loaded_component)
        self.assertEqual([s.pivot_node_id for s in loaded.sequences], [None, None, None])
        self.assertEqual(loaded.sequences[0].node_ids, [target.id])
        # Removing an ancestor must clear references to its descendant pivots too.
        loaded.add_pivot(loaded_node)
        loaded.set_sequence_pivot(loaded.sequences[0], loaded_node)
        loaded.set_sequence_pivot(loaded.sequences[1], loaded_node)
        loaded.remove_node(loaded.find_node(parent.id))
        self.assertEqual([s.pivot_node_id for s in loaded.sequences], [None, None, None])
        self.assertEqual(loaded.sequences[1].node_ids, [target.id])
        self.assertIsNone(without_pivot.pivot_node_id)

    def test_pivot_preview_uses_node_space_and_selected_local_axis(self) -> None:
        # The preview must not inherit the model coordinate conversion.
        model_conversion = wp.transform(wp.vec3(0.0), wp.quat_rpy(wp.pi / 2, 0.0, 0.0))
        authoring = SmartPrimitiveDocument("soma", self.skeleton.reference_local_transforms, model_conversion)
        parent = authoring.add_node()
        parent.local_xform = wp.transform(wp.vec3(2.0, 3.0, 1.0), wp.quat_rpy(0.0, 0.0, wp.pi / 2))
        node = authoring.add_node(parent)
        node.local_xform = wp.transform(wp.vec3(0.0), wp.quat_rpy(0.0, float(np.arcsin(0.6)), 0.0))
        component = authoring.add_pivot(node)
        pivot = component.pivot
        pivot.translation_constraint = ConstraintType.LINE_PROJECTION
        pivot.line_min_offset, pivot.line_max_offset = -2.0, 0.75
        pivot.rotation_constraint = ConstraintType.CONE
        editor = SmartPrimitiveEditor.__new__(SmartPrimitiveEditor)
        editor.authoring = authoring
        editor.viewer = _ArrowViewer()
        editor.pivot_renderer = _PivotRenderer()
        line_name = f"/pivot_translation_constraint_{component.id}"
        cone_name = f"/pivot_rotation_constraint_{component.id}"
        origin = np.array([2.0, 3.0, 1.0])
        # The 45 degree cone boundaries lie in the plane across the selected axis.
        edge = 1.5 / np.sqrt(2.0)
        for axis, motion, arrow_end, line, normal, forward in (
            (
                Axis.X, (0, 1, 0), (1, 3, 1), [(2, 1.4, 2.2), (2, 3.6, 0.55)], (0, 0.8, -0.6),
                [(-edge, 0.6 * edge, 0.8 * edge), (-edge, -0.6 * edge, -0.8 * edge)],
            ),
            (
                Axis.Y, (0, 0, 1), (2, 3.6, 1.8), [(4, 3, 1), (1.25, 3, 1)], (-1, 0, 0),
                [(0, 1.4 * edge, 0.2 * edge), (0, -0.2 * edge, 1.4 * edge)],
            ),
            (
                Axis.Z, (1, 0, 0), (2, 3.8, 0.4), [(2, 1.8, -0.6), (2, 3.45, 1.6)], (0, 0.6, 0.8),
                [(-edge, 0.8 * edge, -0.6 * edge), (edge, 0.8 * edge, -0.6 * edge)],
            ),
        ):
            pivot.translation_axis = pivot.rotation_axis = axis
            pivot.set_motion_direction(motion)
            mirrored = [tuple(-value for value in boundary) for boundary in forward]
            for side, expected in (
                (ConeSide.BOTH, forward + mirrored),
                (ConeSide.FORWARD, forward),
                (ConeSide.MIRRORED, mirrored),
            ):
                with self.subTest(axis=axis, side=side):
                    pivot.cone_side = side
                    editor._draw_pivots()
                    # The arrow sits just above its pivot, clear of the ground plane.
                    lift = np.array([0.0, 0.0, _PIVOT_ARROW_LIFT])
                    np.testing.assert_allclose(editor.viewer.starts[0], origin + lift, atol=1e-6)
                    np.testing.assert_allclose(editor.viewer.ends[0], np.add(arrow_end, lift), atol=1e-6)
                    line_starts, line_ends = editor.viewer.lines[line_name]
                    np.testing.assert_allclose([line_starts[0], line_ends[0]], line, atol=1e-6)
                    starts, ends = editor.viewer.lines[cone_name]
                    rays = np.all(np.isclose(starts, origin), axis=1)
                    actual = ends[rays] - origin
                    # Compare each expected boundary without depending on draw order.
                    self.assertEqual(len(actual), len(expected))
                    for boundary in expected:
                        self.assertTrue(np.any(np.all(np.isclose(actual, boundary, atol=1e-6), axis=1)))
                    np.testing.assert_allclose((ends - origin) @ normal, 0.0, atol=1e-6)

        previous_direction = pivot.motion_direction
        for invalid in ((0.0, 0.0, 0.0), (float("nan"), 1.0, 0.0), (0.0, float("inf"), 0.0)):
            with self.assertRaisesRegex(ValueError, "finite and non-zero"):
                pivot.set_motion_direction(invalid)
            self.assertEqual(pivot.motion_direction, previous_direction)

    def test_sequences_order_reuse_and_clean_up_target_pose_nodes(self) -> None:
        authoring = SmartPrimitiveDocument(
            "soma",
            self.skeleton.reference_local_transforms,
            wp.transform_identity(),
        )
        entry = authoring.add_node()
        action = authoring.add_node()
        unused = authoring.add_node()
        entry_target = authoring.add_target_pose(entry)
        authoring.add_target_pose(action)

        primary = authoring.add_sequence()
        variation = authoring.add_sequence()
        authoring.add_sequence_node(primary, entry)
        authoring.add_sequence_node(primary, action)
        authoring.add_sequence_node(variation, entry)
        authoring.add_sequence_node(variation, action)
        authoring.add_sequence_node(variation, entry)
        authoring.move_sequence_node(variation, 2, 0)

        self.assertEqual(primary.node_ids, [entry.id, action.id])
        self.assertEqual(variation.node_ids, [entry.id, entry.id, action.id])
        self.assertEqual(
            list(authoring.sequences_referencing(entry)),
            [primary, variation],
        )
        with self.assertRaisesRegex(ValueError, "must own a Target Pose"):
            authoring.add_sequence_node(primary, unused)

        authoring.remove_component(entry, entry_target)
        self.assertEqual(primary.node_ids, [action.id])
        self.assertEqual(variation.node_ids, [action.id])

        authoring.remove_node(action)
        self.assertEqual(primary.node_ids, [])
        self.assertEqual(variation.node_ids, [])

        authoring.remove_sequence(primary)
        self.assertNotIn(primary, authoring.sequences)

    def _build_authored_scene(self, clip: ClipAsset):
        model_to_runtime = wp.transform(
            wp.vec3(0.0, 0.0, 0.0),
            wp.quat_rpy(0.5 * wp.pi, 0.0, 0.0),
        )
        authoring = SmartPrimitiveDocument("soma", self.skeleton.reference_local_transforms, model_to_runtime)
        authoring.root.name = "Authored Root"
        authoring.root.local_xform = wp.transform(
            wp.vec3(1.0, 2.0, 3.0),
            wp.quat_rpy(0.1, 0.2, 0.3),
        )
        authoring.root.euler_degrees = [5.0, 10.0, 15.0]
        authoring.root.translation_locks = [True, False, True]

        branch = authoring.add_node()
        branch.name = "Approach"
        branch.visible = False
        branch.rotation_locks = [False, True, False]
        branch.local_xform = wp.transform(
            wp.vec3(4.0, 5.0, 6.0),
            wp.quat_rpy(-0.2, 0.4, -0.6),
        )
        sibling = authoring.add_node()
        sibling.name = "Exit"
        leaf = authoring.add_node(branch)
        leaf.name = "Contact"
        leaf.euler_degrees = [-20.0, 30.0, 220.0]

        first_target = authoring.add_target_pose(branch)
        first_target.set_clip(clip)
        first_target.playback_mode = PlaybackMode.FRAMES
        first_target.set_time(2 / clip.animation.sample_rate)
        first_target.constrain_hips(True)
        first_target.set_num_poses_to_use(3)
        first_target.transition_early_by_frames = 7
        first_target.ik_pin_mask = IKPin.LEFT_HAND | IKPin.RIGHT_FOOT
        first_target.show_velocity = True
        first_target.show_trajectory = True
        first_target.set_trajectory_scale(3.5)
        first_target.show_mesh = False
        first_target.show_skeletons = True
        first_target.color = wp.vec3(0.1, 0.2, 0.3)

        interaction_box = authoring.add_interaction_box(branch)
        interaction_box.set_dimensions(2.5, 3.5, 4.5)
        interaction_box.color = wp.vec3(0.4, 0.5, 0.6)

        second_target = authoring.add_target_pose(leaf)
        second_target.set_clip(clip)
        second_target.playback_mode = PlaybackMode.TIME
        second_target.set_time(1.0 / 30.0)
        second_target.set_num_poses_to_use(2)
        second_target.transition_early_by_frames = 11
        second_target.set_token_range(9, 9)
        second_target.ik_pin_mask = IKPin.RIGHT_HAND
        second_target.color = wp.vec3(0.7, 0.8, 0.9)

        primary_sequence = authoring.add_sequence()
        primary_sequence.name = "Primary"
        authoring.add_sequence_node(primary_sequence, branch)
        authoring.add_sequence_node(primary_sequence, leaf)
        variation_sequence = authoring.add_sequence()
        variation_sequence.name = "Variation"
        authoring.add_sequence_node(variation_sequence, leaf)
        authoring.add_sequence_node(variation_sequence, branch)

        return authoring, leaf.id

    def test_authoring_document_round_trip_preserves_complete_scene(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            clip_path = temporary_path / "clips" / "shared.bvh"
            clip_path.parent.mkdir()
            clip_path.touch()
            clip = ClipAsset(clip_path.resolve(), self.clip.animation)
            authoring, selected_id = self._build_authored_scene(clip)

            document_path = save_smart_primitive_document(
                authoring,
                temporary_path / "documents" / "locomotion",
                selected_id,
                _camera_state(),
            )
            self.assertEqual(
                document_path.name,
                "locomotion.smart_primitive.json",
            )
            saved_document = json.loads(document_path.read_text("utf-8"))
            self.assertEqual(
                saved_document["format"],
                "motionbricks.smart_primitive.authoring",
            )
            self.assertEqual(saved_document["schema_version"], 1)
            self.assertEqual(saved_document["name"], "locomotion")
            self.assertEqual(saved_document["rig"], "soma")
            self.assertEqual(
                saved_document["root"]["children"][0]["components"][0]["clip"],
                "../clips/shared.bvh",
            )

            registry = ClipRegistry(ModelRig.SOMA, skeleton=self.skeleton)
            with patch(
                "app.runtime.common.clip_registry.bvh_utils.load_bvh",
                return_value=clip.animation,
            ) as load_bvh:
                loaded, loaded_selected_id, loaded_camera_state = (
                    load_smart_primitive_document(
                        registry,
                        self.skeleton.reference_local_transforms,
                        authoring.model_to_runtime,
                        document_path,
                    )
                )
            load_bvh.assert_called_once_with(clip.path, self.skeleton)
            loaded_targets = [component for _, component in loaded.iter_target_poses()]
            self.assertIs(loaded_targets[0].clip, loaded_targets[1].clip)
            self.assertIs(loaded_targets[0].clip.animation, clip.animation)
            round_trip_path = save_smart_primitive_document(
                loaded,
                document_path.parent / "round_trip.smart_primitive.json",
                loaded_selected_id,
                loaded_camera_state,
            )
            round_trip_document = json.loads(round_trip_path.read_text("utf-8"))
            self.assertEqual(saved_document["editor"], round_trip_document["editor"])
            self.assertEqual(
                saved_document["sequences"],
                round_trip_document["sequences"],
            )
            self.assertEqual(saved_document["root"], round_trip_document["root"])

            existing_ids = {
                item_id
                for node in loaded.iter_nodes()
                for item_id in [node.id, *[item.id for item in node.components]]
            }
            existing_ids.update(sequence.id for sequence in loaded.sequences)
            added_node = loaded.add_node()
            self.assertEqual(added_node.id, max(existing_ids) + 1)

            saved_document["editor"]["selected_node_id"] = 9999
            renamed_path = document_path.with_name("renamed.smart_primitive.json")
            renamed_path.write_text(json.dumps(saved_document), encoding="utf-8")
            renamed, selected_id, _ = load_smart_primitive_document(
                registry,
                self.skeleton.reference_local_transforms,
                authoring.model_to_runtime,
                renamed_path,
            )
            self.assertEqual(selected_id, renamed.root.id)
            self.assertEqual(len(list(renamed.iter_target_poses())), 2)
            self.assertEqual(renamed.sequences, authoring.sequences)

    def test_authoring_document_rejects_invalid_sequence_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            authoring = SmartPrimitiveDocument(
                "soma",
                self.skeleton.reference_local_transforms,
                wp.transform_identity(),
            )
            node = authoring.add_node()
            authoring.add_target_pose(node)
            sequence = authoring.add_sequence()
            authoring.add_sequence_node(sequence, node)
            document_path = save_smart_primitive_document(
                authoring,
                temporary_path / "sequences",
                node.id,
                _camera_state(),
            )
            document = json.loads(document_path.read_text("utf-8"))

            invalid_documents = (
                (
                    "must identify a node with a Pivot component",
                    lambda value: value["sequences"][0].__setitem__("pivot_node_id", 999),
                ),
                (
                    "must identify a node with a Pivot component",
                    lambda value: value["sequences"][0].__setitem__("pivot_node_id", node.id),
                ),
                (
                    "does not identify a hierarchy node",
                    lambda value: value["sequences"][0]["node_ids"].__setitem__(
                        0, 999
                    ),
                ),
                (
                    "Duplicate ID",
                    lambda value: value["sequences"][0].__setitem__(
                        "id", value["root"]["id"]
                    ),
                ),
            )
            for message, mutate in invalid_documents:
                with self.subTest(message=message):
                    invalid_document = json.loads(json.dumps(document))
                    mutate(invalid_document)
                    document_path.write_text(
                        json.dumps(invalid_document),
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(
                        SmartPrimitiveDocumentError,
                        message,
                    ):
                        load_smart_primitive_document(
                            _EmptyClipRegistry(),
                            self.skeleton.reference_local_transforms,
                            authoring.model_to_runtime,
                            document_path,
                        )

    def test_authoring_document_requires_valid_trajectory_scale(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            authoring = SmartPrimitiveDocument(
                "soma",
                self.skeleton.reference_local_transforms,
                wp.transform_identity(),
            )
            authoring.add_target_pose(authoring.root)
            document_path = save_smart_primitive_document(
                authoring,
                Path(temporary_directory) / "trajectory_scale",
                authoring.root.id,
                _camera_state(),
            )
            document = json.loads(document_path.read_text("utf-8"))
            document["root"]["components"][0]["trajectory_scale"] = 4.1
            document_path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(
                SmartPrimitiveDocumentError,
                r"\.trajectory_scale must be 1\.0 to 4\.0",
            ):
                load_smart_primitive_document(
                    _EmptyClipRegistry(),
                    self.skeleton.reference_local_transforms,
                    authoring.model_to_runtime,
                    document_path,
                )

    def test_authoring_document_requires_valid_ik_pin_mask(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            authoring = SmartPrimitiveDocument(
                "soma",
                self.skeleton.reference_local_transforms,
                wp.transform_identity(),
            )
            authoring.add_target_pose(authoring.root)
            document_path = save_smart_primitive_document(
                authoring,
                Path(temporary_directory) / "ik_pin_mask",
                authoring.root.id,
                _camera_state(),
            )
            document = json.loads(document_path.read_text("utf-8"))
            document["root"]["components"][0]["ik_pin_mask"] = 1 << 4
            document_path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(
                SmartPrimitiveDocumentError,
                r"\.ik_pin_mask contains unsupported bits",
            ):
                load_smart_primitive_document(
                    _EmptyClipRegistry(),
                    self.skeleton.reference_local_transforms,
                    authoring.model_to_runtime,
                    document_path,
                )

    def test_authoring_document_requires_valid_token_range(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            authoring = SmartPrimitiveDocument(
                "soma",
                self.skeleton.reference_local_transforms,
                wp.transform_identity(),
            )
            component = authoring.add_target_pose(authoring.root)
            self.assertEqual((component.min_tokens, component.max_tokens), (6, 14))
            document_path = save_smart_primitive_document(
                authoring,
                Path(temporary_directory) / "tokens",
                authoring.root.id,
                _camera_state(),
            )
            saved = json.loads(document_path.read_text("utf-8"))
            for tokens, message in (
                ((5, 14), r"\.min_tokens must be at least 6"),
                ((6, 17), r"\.max_tokens must be at most 16"),
                ((10, 9), r"\.max_tokens must be at least 10"),
            ):
                with self.subTest(tokens=tokens):
                    document = json.loads(json.dumps(saved))
                    target = document["root"]["components"][0]
                    target["min_tokens"], target["max_tokens"] = tokens
                    document_path.write_text(json.dumps(document), encoding="utf-8")
                    with self.assertRaisesRegex(DocumentError, message):
                        load_smart_primitive_document(
                            _EmptyClipRegistry(),
                            self.skeleton.reference_local_transforms,
                            authoring.model_to_runtime,
                            document_path,
                        )

    def test_authoring_document_rejects_multiple_interaction_boxes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            authoring = SmartPrimitiveDocument(
                "soma",
                self.skeleton.reference_local_transforms,
                wp.transform_identity(),
            )
            authoring.add_interaction_box(authoring.root)
            document_path = save_smart_primitive_document(
                authoring,
                Path(temporary_directory) / "duplicate_box",
                authoring.root.id,
                _camera_state(),
            )
            document = json.loads(document_path.read_text("utf-8"))
            duplicate_document = json.loads(json.dumps(document))
            duplicate = duplicate_document["root"]["components"][0].copy()
            duplicate["id"] += 1
            duplicate_document["root"]["components"].append(duplicate)
            document_path.write_text(
                json.dumps(duplicate_document),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                SmartPrimitiveDocumentError,
                "more than one Interaction Box",
            ):
                load_smart_primitive_document(
                    _EmptyClipRegistry(),
                    self.skeleton.reference_local_transforms,
                    authoring.model_to_runtime,
                    document_path,
                )

            document["root"]["components"][0]["width"] = -0.1
            document_path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(
                SmartPrimitiveDocumentError,
                r"\.width cannot be negative",
            ):
                load_smart_primitive_document(
                    _EmptyClipRegistry(),
                    self.skeleton.reference_local_transforms,
                    authoring.model_to_runtime,
                    document_path,
                )

    def test_new_scene_resets_authoring_document_and_interaction_state(self) -> None:
        authoring = SmartPrimitiveDocument(
            "soma",
            self.skeleton.reference_local_transforms,
            wp.transform_identity(),
        )
        selected = authoring.add_node()
        authoring.add_target_pose(selected)

        editor = SmartPrimitiveEditor.__new__(SmartPrimitiveEditor)
        editor.viewer = _TitleViewer()
        editor.rig = SimpleNamespace(target_pose_renderers=_ClearableRenderers())
        editor.interaction_box_renderer = _ClearableRenderers()
        editor.pivot_renderer = _ClearableRenderers()
        editor.authoring = authoring
        editor.selected_id = selected.id
        editor._document_path = Path("existing.smart_primitive.json")
        editor._status = object()

        editor._new_document()

        self.assertIsNot(editor.authoring, authoring)
        self.assertEqual(list(editor.authoring.iter_nodes()), [editor.authoring.root])
        self.assertEqual(editor.selected_id, editor.authoring.root.id)
        self.assertIsNone(editor._document_path)
        self.assertIsNone(editor._status)

    def test_save_as_chooses_new_current_document(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            authoring = SmartPrimitiveDocument(
                "soma",
                self.skeleton.reference_local_transforms,
                wp.transform_identity(),
            )
            original_path = save_smart_primitive_document(
                authoring,
                temporary_path / "original.smart_primitive.json",
                authoring.root.id,
                _camera_state(),
            )
            requested_path = temporary_path / "copy"

            editor = SmartPrimitiveEditor.__new__(SmartPrimitiveEditor)
            editor.viewer = _TitleViewer()
            editor.authoring = authoring
            editor.selected_id = authoring.root.id
            editor._document_path = original_path
            editor._status = object()

            with patch("app.authoring.common.choose_file", return_value=str(requested_path)) as chooser:
                editor._save_as()

            copy_path = requested_path.with_name("copy.smart_primitive.json").resolve()
            self.assertEqual(
                chooser.call_args,
                call(
                    "Save Smart Primitive", "Smart Primitive documents", ".smart_primitive.json",
                    original_path, save=True,
                ),
            )
            self.assertTrue(original_path.is_file())
            self.assertTrue(copy_path.is_file())
            self.assertEqual(editor._document_path, copy_path)
            self.assertIsNone(editor._status)

    def test_successful_load_replaces_scene_without_persistent_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            authoring = SmartPrimitiveDocument(
                "soma",
                self.skeleton.reference_local_transforms,
                wp.transform_identity(),
            )
            selected = authoring.add_node()
            selected.name = "Loaded Node"
            document_path = save_smart_primitive_document(
                authoring,
                Path(temporary_directory) / "loaded.smart_primitive.json",
                selected.id,
                _camera_state(),
            )

            rig = _Rig(
                _EmptyClipRegistry(),
                self.skeleton.reference_local_transforms,
                wp.transform_identity(),
                _ClearableRenderers(),
            )
            editor = _loading_editor(rig)
            other_camera_state = _camera_state()
            other_camera_state["pos"] = [-10.0, -20.0, -30.0]
            editor.viewer.camera = _Camera(other_camera_state)

            editor._open_document(document_path)

            self.assertEqual(editor.selected.name, "Loaded Node")
            self.assertEqual(editor._document_path, document_path.resolve())
            self.assertIsNone(editor._status)
            # Opening with the same character reuses its rig resources.
            self.assertIs(editor.rig, rig)
            self.assertEqual(editor._capture_camera_state(), _camera_state())

    def _bvh_rig(self) -> _Rig:
        # BVH clips are Y-up; a quarter turn about X makes them Z-up.
        y_up_to_z_up = wp.transform(wp.vec3(0.0), wp.quat_rpy(0.5 * wp.pi, 0.0, 0.0))
        return _Rig(
            ClipRegistry(ModelRig.SOMA, skeleton=self.skeleton),
            self.skeleton.reference_local_transforms,
            y_up_to_z_up,
            _ClearableRenderers(),
        )

    @staticmethod
    def _csv_rig(folder: Path) -> tuple[_Rig, Path]:
        """A one-hinge robot and a CSV clip that walks it forward."""
        frames = np.zeros((31, 8), dtype=np.float32)
        frames[:, 0] = np.arange(31) / 120.0
        frames[:, 2] = 0.8
        frames[:, 6] = 1.0
        clip_path = folder / "walk.csv"
        save_csv(str(clip_path), DOFAnimationBuffer(("hinge",), 120.0, wp.array(frames, dtype=wp.float32)))
        rig = _Rig(
            ClipRegistry(ModelRig.UNITREE_G1, joint_names=("hinge",)),
            wp.array(frames[0], dtype=wp.float32),
            wp.transform_identity(),
            _ClearableRenderers(),
        )
        return rig, clip_path

    def test_clips_must_match_the_character_and_then_fix_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            csv_rig, csv_path = self._csv_rig(Path(temporary_directory))
            bvh_rig = self._bvh_rig()
            editor = SmartPrimitiveEditor.__new__(SmartPrimitiveEditor)
            editor.viewer = SimpleNamespace()
            editor._load_rig = Mock(return_value=csv_rig)
            editor.rig = bvh_rig
            editor.authoring = SmartPrimitiveDocument(
                bvh_rig.clip_registry.rig, bvh_rig.reference_pose, bvh_rig.model_to_runtime,
            )
            editor._status = None
            editor._choose_clip = lambda: str(csv_path)
            target = editor.authoring.add_target_pose(editor.authoring.add_node())

            # SOMA loads only BVH clips, and refusing a CSV clip builds no other rig.
            with patch("app.smart_primitive_editor.draw_clip_selector", return_value=True):
                editor._draw_clip_selector(None, target)
            self.assertIn("loads .bvh clips", editor._status)
            self.assertIsNone(target.clip)
            editor._load_rig.assert_not_called()

            # Without clips, another character re-lays out every target pose.
            editor._use_rig(editor._rig_for(ModelRig.UNITREE_G1))
            editor._load_rig.assert_called_once_with(ModelRig.UNITREE_G1)
            self.assertIs(editor.rig, csv_rig)
            self.assertEqual(editor.authoring.rig, "unitree_g1")
            self.assertEqual(bvh_rig.target_pose_renderers.clear_count, 1)
            self.assertEqual(editor.authoring.up_index, 2)
            np.testing.assert_array_equal(target.pose.numpy(), csv_rig.reference_pose.numpy())
            with patch("app.smart_primitive_editor.draw_clip_selector", return_value=True):
                editor._draw_clip_selector(None, target)
            self.assertEqual(target.clip.path, csv_path.resolve())

            # A clip fixes the character, so the document refuses another one.
            with self.assertRaisesRegex(ValueError, "Cannot change the rig"):
                editor._use_rig(bvh_rig)
            self.assertIs(editor.rig, csv_rig)

    def test_load_selects_the_saved_rig_even_without_clips(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            folder = Path(temporary_directory)
            csv_rig, _ = self._csv_rig(folder)
            authoring = SmartPrimitiveDocument(
                csv_rig.clip_registry.rig, csv_rig.reference_pose, csv_rig.model_to_runtime,
            )
            authoring.add_target_pose(authoring.add_node())
            document_path = save_smart_primitive_document(
                authoring, folder / "empty", authoring.root.id, _camera_state(),
            )
            bvh_rig = self._bvh_rig()
            editor = _loading_editor(bvh_rig)
            editor._load_rig = Mock(return_value=csv_rig)

            editor._open_document(document_path)

            editor._load_rig.assert_called_once_with(ModelRig.UNITREE_G1)

            self.assertEqual(load_smart_primitive_rig(document_path), "unitree_g1")
            self.assertIsNone(editor._status)
            self.assertIs(editor.rig, csv_rig)
            self.assertEqual(editor.authoring.rig, "unitree_g1")
            self.assertEqual(bvh_rig.target_pose_renderers.clear_count, 1)
            [(_, target)] = editor.authoring.iter_target_poses()
            self.assertIsNone(target.clip)
            np.testing.assert_array_equal(target.pose.numpy(), csv_rig.reference_pose.numpy())

    def test_asset_pool_uses_registry_rig_before_loading_clips(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            clip_path = temporary_path / "clip.bvh"
            clip_path.touch()
            clip = ClipAsset(clip_path.resolve(), self.clip.animation)
            authoring = SmartPrimitiveDocument(
                "soma", self.skeleton.reference_local_transforms, wp.transform_identity(),
            )
            authoring.add_target_pose(authoring.root).set_clip(clip)
            document_path = save_smart_primitive_document(
                authoring, temporary_path / "jump", authoring.root.id, _camera_state(),
            )
            clip_registry = ClipRegistry(
                ModelRig.UNITREE_G1, joint_names=("hinge",),
            )
            pool = SmartPrimitiveAssetPool(
                clip_registry, wp.array([0, 0, 0, 0, 0, 0, 1, 0], dtype=wp.float32), wp.transform_identity(),
            )

            with patch.object(clip_registry, "load") as load_clip:
                with self.assertRaisesRegex(
                    SmartPrimitiveDocumentError,
                    r"jump\.smart_primitive\.json was saved for the 'soma' rig, but this app uses 'unitree_g1'",
                ):
                    pool.load(document_path)
            load_clip.assert_not_called()


if __name__ == "__main__":
    unittest.main()
