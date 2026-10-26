# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from pathlib import Path
import unittest

import numpy as np
import warp as wp

from app.runtime.common.io.scene_document import SceneDefinition
from app.runtime.common.clip_registry import ClipAsset
from app.runtime.common.io.smart_primitive_document import SmartPrimitiveDocument, build_runtime_smart_primitive_data
from app.inbetweener_demo import InbetweenerDemo, PoseEntry
from app.smart_primitive_editor import SmartPrimitiveEditor
from app.scene_editor import SceneEditor, _CharacterStartRenderer, _SceneSmartPrimitiveView
from motionbricks.animation.animation_buffer import AnimationBuffer
from motionbricks.animation.mesh import SkeletalMesh, SkinnedMesh
from motionbricks.animation.skeleton import Skeleton
from motionbricks.renderers.skeletal_mesh_renderer import SkeletalMeshRenderer
from motionbricks.renderers.solid_box_renderer import SolidBoxRenderer
from motionbricks.renderers.wire_box_renderer import WireBoxRenderer
from motionbricks.runtime.smart_primitive import SmartPrimitive, SmartPrimitiveAsset


def _renderer():
    reference = np.array([[0, 0, 0, 0, 0, 0, 1], [0, 1, 0, 0, 0, 0, 1]], dtype=np.float32)
    skeleton = Skeleton(["Hips", "Child"], [-1, 0], reference)
    parts = [
        SkinnedMesh(
            np.array([[0, 0, z], [1, 0, z], [0, 1, z]], dtype=np.float32),
            np.array([0, 1, 2], dtype=np.int32),
            np.ones(3, dtype=np.int32), np.ones(3, dtype=np.float32),
        )
        for z in (-2, 0)
    ]
    # Like the SOMA mesh, it has the relaxed-hands override pose and finger mask.
    mesh = SkeletalMesh(
        parts, skeleton, [wp.transform_identity(), wp.transform(0, -1, 0, 0, 0, 0, 1)],
        override_poses={"RelaxedHands": wp.clone(skeleton.reference_local_transforms)},
        override_pose_masks={"Fingers": wp.zeros(2, dtype=wp.float32)},
    )
    return SkeletalMeshRenderer(mesh)


class MeshPickingTests(unittest.TestCase):
    def test_scene_picks_instance_roots_and_cubes_by_depth_and_geometry_visibility(self):
        with wp.ScopedDevice("cpu"):
            mesh = _renderer().skeletal_mesh
            authoring = SmartPrimitiveDocument(
                "soma", mesh.skeleton.reference_local_transforms, wp.transform_identity(),
            )
            frames = np.repeat(mesh.skeleton.reference_local_transforms.numpy()[None], 4, axis=0)
            animation = AnimationBuffer(mesh.skeleton, 30.0, wp.array(frames, dtype=wp.transform))
            clip = ClipAsset(Path("selection.bvh"), animation)
            first, second = authoring.add_node(), authoring.add_node()
            second.local_xform = wp.transform(4, 0, 0, 0, 0, 0, 1)
            for node in (first, second):
                authoring.add_target_pose(node).set_clip(clip)
            box_node = authoring.add_node()
            box_node.local_xform = wp.transform(wp.vec3(-3, 0, 1), wp.quat_rpy(0.0, 0.0, 0.6))
            authoring.add_interaction_box(box_node).set_dimensions(2, 4, 2)
            sequence = authoring.add_sequence()
            authoring.add_sequence_node(sequence, second)
            asset = SmartPrimitiveAsset(build_runtime_smart_primitive_data(authoring), device="cpu")
            editor = SceneEditor.__new__(SceneEditor)
            editor.viewer = SimpleNamespace(camera=SimpleNamespace(far=100))
            editor.scene = SceneDefinition()
            editor.solid_box_renderer = SolidBoxRenderer()
            editor.character_start_renderer = _CharacterStartRenderer()
            editor._runtime_views = {}
            instances = [editor.scene.add_smart_primitive(), editor.scene.add_smart_primitive()]
            for index, instance in enumerate(instances):
                instance.reference = Path("selection.smart_primitive.json")
                instance.transform = wp.transform(index * 8, 0, 1, 0, 0, 0, 1)
                runtime = SmartPrimitive(asset)
                editor._runtime_views[instance.id] = _SceneSmartPrimitiveView(
                    instance, runtime, mesh, mesh.skeleton.reference_local_transforms, instance.transform,
                )
            editor.selected_item_id = editor.scene.character_start.id
            direction = wp.vec3(0, 0, -1)
            for x, owner in ((0.2, instances[0]), (4.2, instances[0]), (8.2, instances[1])):
                editor._select_scene_item(wp.vec3(x, 0.2, 5), direction)
                self.assertEqual(editor.selected_item_id, owner.id)
            cube = editor.scene.add_solid_cube()
            cube.transform = wp.transform(0.2, 0.2, 3, 0, 0, 0, 1)
            editor._select_scene_item(wp.vec3(0.2, 0.2, 5), direction)
            self.assertEqual(editor.selected_item_id, cube.id)
            cube.transform = wp.transform(0.2, 0.2, -3, 0, 0, 0, 1)
            editor._select_scene_item(wp.vec3(0.2, 0.2, 5), direction)
            self.assertEqual(editor.selected_item_id, instances[0].id)
            view = editor._runtime_views[instances[0].id]
            view.target_pose_sequence_index = 0
            editor._select_scene_item(wp.vec3(0.2, 0.2, 5), direction)
            self.assertEqual(editor.selected_item_id, cube.id)
            editor._select_scene_item(wp.vec3(4.2, 0.2, 5), direction)
            self.assertEqual(editor.selected_item_id, instances[0].id)
            instances[0].show_target_poses = False
            editor.selected_item_id = editor.scene.character_start.id
            editor._select_scene_item(wp.vec3(4.2, 0.2, 5), direction)
            self.assertEqual(editor.selected_item_id, editor.scene.character_start.id)

            # Wire boxes select their instance root even when its target poses are hidden.
            box_world = instances[0].transform * box_node.local_xform
            origin = wp.transform_point(box_world, wp.vec3(0, 2, 5))
            direction = wp.transform_vector(box_world, wp.vec3(0, 0, -1))
            editor._select_scene_item(origin, direction)
            self.assertEqual(editor.selected_item_id, instances[0].id)
            instances[0].show_interaction_boxes = False
            editor.selected_item_id = cube.id
            editor._select_scene_item(origin, direction)
            self.assertEqual(editor.selected_item_id, cube.id)

            instances[0].show_interaction_boxes = True
            instances[0].transform = wp.transform(wp.vec3(-5, 2, 4), wp.quat_rpy(0.4, -0.3, 0.7))
            view.set_transform(instances[0].transform)
            editor._select_scene_item(origin, direction)
            self.assertEqual(editor.selected_item_id, cube.id)
            box_world = instances[0].transform * box_node.local_xform
            origin = wp.transform_point(box_world, wp.vec3(0, 2, 5))
            direction = wp.transform_vector(box_world, wp.vec3(0, 0, -1))
            editor._select_scene_item(origin, direction)
            self.assertEqual(editor.selected_item_id, instances[0].id)

    def test_wire_box_picking_follows_transformed_resized_edges_and_leaves_faces_open(self):
        placement = wp.transform(wp.vec3(3, 4, 5), wp.quat_rpy(0.3, -0.4, 0.9))
        for origin, direction, limit, expected in (
            ((0.25, 2, 8), (0, 0, -1), 12.3, 5.0),
            ((1, 0.25, 8), (0, 0, -1), 12.3, 5.0),
            ((6, 2, 0.25), (-1, 0, 0), 12.3, 5.0),
            ((0, 0, 8), (0, 0, -1), 12.3, None),
            ((1, 0.25, 8), (0, 0, 1), 12.3, None),
            ((1, 0.25, 8), (0, 0, -1), 4.0, None),
        ):
            with self.subTest(origin=origin, direction=direction, limit=limit):
                distance = WireBoxRenderer.raycast(
                    placement, wp.vec3(2, 4, 6), wp.transform_point(placement, wp.vec3(*origin)),
                    wp.transform_vector(placement, wp.vec3(*direction)), limit,
                )
                if expected is None:
                    self.assertEqual(distance, limit)
                else:
                    self.assertAlmostEqual(distance, expected, delta=0.1)
        origin = wp.transform_point(placement, wp.vec3(0.25, 2, 8))
        direction = wp.transform_vector(placement, wp.vec3(0, 0, -1))
        self.assertEqual(WireBoxRenderer.raycast(placement, wp.vec3(2, 3, 6), origin, direction, 12.3), 12.3)
        # Authored boxes may be flat; collapsed edges must not create invalid rotations.
        self.assertAlmostEqual(
            WireBoxRenderer.raycast(placement, wp.vec3(2, 4, 0), origin, direction), 8.0, delta=0.1,
        )

    def test_scene_picks_transformed_start_lines_without_selecting_gaps_or_occluded_geometry(self):
        with wp.ScopedDevice("cpu"):
            editor = SceneEditor.__new__(SceneEditor)
            editor.viewer = SimpleNamespace(camera=SimpleNamespace(far=100))
            editor.scene = SceneDefinition()
            editor._runtime_views = {}
            editor.character_start_renderer = _CharacterStartRenderer()
            editor.solid_box_renderer = SolidBoxRenderer()
            start = editor.scene.character_start
            placement = wp.transform(wp.vec3(3, -2, 1), wp.quat_rpy(0.3, 0.4, 0.9))
            start.transform = placement
            cube = editor.scene.add_solid_cube()
            cube.width = cube.depth = 2.0
            cube.transform = placement * wp.transform(0, 0, -2, 0, 0, 0, 1)
            direction = wp.transform_vector(placement, wp.vec3(0, 0, -1))
            # Circle, arrow shaft, and arrowhead select the marker ahead of a cube.
            for x, y in ((0.0, 0.35), (0.5, 0.0), (0.71, 0.06)):
                with self.subTest(x=x, y=y):
                    editor.selected_item_id = cube.id
                    origin = wp.transform_point(placement, wp.vec3(x, y, 5))
                    editor._select_scene_item(origin, direction)
                    self.assertEqual(editor.selected_item_id, start.id)

            cube.transform = placement * wp.transform(0, 0, 2, 0, 0, 0, 1)
            editor._select_scene_item(origin, direction)
            self.assertEqual(editor.selected_item_id, cube.id)

            cube.transform = wp.transform(20, 0, 0, 0, 0, 0, 1)
            for point in ((0, 0.15, 5), (0.5, 0.3, 5)):
                editor._select_scene_item(wp.transform_point(placement, wp.vec3(*point)), direction)
                self.assertEqual(editor.selected_item_id, cube.id)

            # Picking uses the edited transform, not the marker's previous placement.
            start.transform = wp.transform(10, 0, 0, 0, 0, 0, 1) * placement
            editor._select_scene_item(origin, direction)
            self.assertEqual(editor.selected_item_id, cube.id)
            editor._select_scene_item(origin + wp.vec3(10, 0, 0), direction)
            self.assertEqual(editor.selected_item_id, start.id)

    def test_raycast_follows_skinned_pose_placement_and_uses_nearest_mesh_part(self):
        wp.init()
        for device in wp.get_devices():
            with self.subTest(device=device.alias), wp.ScopedDevice(device):
                renderer = _renderer()
                pose = renderer.skeletal_mesh.skeleton.reference_local_transforms.numpy().copy()
                pose[1, 0] = 1
                transforms = wp.array(pose, dtype=wp.transform)
                placement = wp.transform(wp.vec3(3, 0, 0), wp.quat_rpy(0.0, 0.0, wp.pi / 2))
                ray = wp.vec3(0, 0, -1)
                self.assertAlmostEqual(renderer.raycast_pose(transforms, placement, wp.vec3(2.8, 1.2, 5), ray), 5)
                # Moving a joint must invalidate the old surface, even though the picker reuses its mesh.
                pose[1, 0] = 4
                transforms.assign(pose)
                self.assertEqual(renderer.raycast_pose(transforms, placement, wp.vec3(2.8, 1.2, 5), ray, 12.3), 12.3)
                self.assertAlmostEqual(renderer.raycast_pose(transforms, placement, wp.vec3(2.8, 4.2, 5), ray), 5)
                self.assertEqual(renderer.raycast_pose(transforms, placement, wp.vec3(2.8, 4.2, 5), ray, 4), 4)
                # Inside the bounding box but outside the triangle is not a hit.
                self.assertEqual(renderer.raycast_pose(transforms, placement, wp.vec3(2.1, 4.9, 5), ray, 10), 10)

    def test_demo_selects_nearest_current_pose_after_reordering_and_preserves_selection_on_miss(self):
        with wp.ScopedDevice("cpu"):
            demo = InbetweenerDemo.__new__(InbetweenerDemo)
            demo.viewer = SimpleNamespace(camera=SimpleNamespace(far=100))
            demo.mesh_renderer = _renderer()
            skeleton = demo.mesh_renderer.skeletal_mesh.skeleton
            clip = AnimationBuffer(
                skeleton, 30.0, wp.array(skeleton.reference_local_transforms.numpy()[None], dtype=wp.transform),
            )
            demo.pose_entries = [
                PoseEntry(identifier, str(identifier), Path("synthetic.bvh"), clip,
                          0.0, wp.transform_identity())
                for identifier in (42, 8, 17)
            ]
            poses = np.repeat(
                demo.mesh_renderer.skeletal_mesh.skeleton.reference_local_transforms.numpy()[None], 3, axis=0,
            )
            poses[:, 0, 2] = [0, 2, -2]
            for index, (entry, pose) in enumerate(zip(demo.pose_entries, poses)):
                entry.preview_pose = wp.array(pose, dtype=wp.transform)
                # Keep the ground rings away from these rays. They have their own test.
                entry.root_position = np.array([10 + 2 * index, 0, 0], dtype=np.float32)
            demo._refresh_order_markers()
            demo.selected_index = 0
            origin, direction = wp.vec3(0.2, 0.2, 5), wp.vec3(0, 0, -1)
            demo._select_pose(origin, direction)
            self.assertEqual(demo.pose_entries[demo.selected_index].id, 8)
            demo.pose_entries = [demo.pose_entries[i] for i in (0, 2, 1)]
            demo._refresh_order_markers()
            demo._select_pose(origin, direction)
            self.assertEqual(demo.pose_entries[demo.selected_index].id, 8)
            demo._select_pose(wp.vec3(5, 5, 5), direction)
            self.assertEqual(demo.pose_entries[demo.selected_index].id, 8)
            demo.pose_entries[demo.selected_index].show_mesh = False
            demo._select_pose(origin, direction)
            self.assertEqual(demo.pose_entries[demo.selected_index].id, 42)

    def test_editor_selects_owning_node_and_ignores_hidden_meshes_and_ancestors(self):
        with wp.ScopedDevice("cpu"):
            editor = SmartPrimitiveEditor.__new__(SmartPrimitiveEditor)
            renderer = _renderer()
            editor.rig = SimpleNamespace(target_pose_renderers=SimpleNamespace(mesh=renderer))
            editor.viewer = SimpleNamespace(camera=SimpleNamespace(far=100))
            editor.authoring = SmartPrimitiveDocument(
                "soma", renderer.skeletal_mesh.skeleton.reference_local_transforms, wp.transform(3, 0, 0, 0, 0, 0, 1),
            )
            parent = editor.authoring.add_node()
            parent.local_xform = wp.transform(0, 4, 0, 0, 0, 0, 1)
            far = editor.authoring.add_node(parent)
            near = editor.authoring.add_node(parent)
            near.local_xform = wp.transform(0, 0, 2, 0, 0, 0, 1)
            editor.authoring.add_target_pose(far)
            component = editor.authoring.add_target_pose(near)
            editor.selected_id = editor.authoring.root.id
            origin, direction = wp.vec3(3.2, 4.2, 5), wp.vec3(0, 0, -1)
            editor._select_scene_item(origin, direction)
            self.assertEqual(editor.selected_id, near.id)
            self.assertNotEqual(editor.selected_id, component.id)
            component.show_mesh = False
            editor._select_scene_item(origin, direction)
            self.assertEqual(editor.selected_id, far.id)
            parent.visible = False
            editor.selected_id = editor.authoring.root.id
            editor._select_scene_item(origin, direction)
            self.assertEqual(editor.selected_id, editor.authoring.root.id)

    def test_editor_picks_wire_box_owner_with_hierarchy_visibility_and_mesh_clickthrough(self):
        with wp.ScopedDevice("cpu"):
            editor = SmartPrimitiveEditor.__new__(SmartPrimitiveEditor)
            renderer = _renderer()
            editor.rig = SimpleNamespace(target_pose_renderers=SimpleNamespace(mesh=renderer))
            editor.interaction_box_renderer = WireBoxRenderer()
            editor.viewer = SimpleNamespace(camera=SimpleNamespace(far=100))
            editor.authoring = SmartPrimitiveDocument(
                "soma", renderer.skeletal_mesh.skeleton.reference_local_transforms, wp.transform(3, 0, 0, 0, 0, 0, 1),
            )
            parent = editor.authoring.add_node()
            parent.local_xform = wp.transform(wp.vec3(2, 4, 1), wp.quat_rpy(0.3, -0.2, 0.8))
            pose_node = editor.authoring.add_node(parent)
            editor.authoring.add_target_pose(pose_node)
            box_node = editor.authoring.add_node(parent)
            box_node.local_xform = wp.transform(3.2, 0.2, 1, 0, 0, 0, 1)
            box = editor.authoring.add_interaction_box(box_node)
            box.set_dimensions(0.4, 0.4, 2)
            editor.selected_id = parent.id
            direction = wp.transform_vector(parent.local_xform, wp.vec3(0, 0, -1))
            interior = wp.transform_point(parent.local_xform, wp.vec3(3.2, 0.2, 5))
            editor._select_scene_item(interior, direction)
            self.assertEqual(editor.selected_id, pose_node.id)

            edge = wp.transform_point(parent.local_xform, wp.vec3(3.2, 0.4, 5))
            editor._select_scene_item(edge, direction)
            self.assertEqual(editor.selected_id, box_node.id)
            self.assertNotEqual(editor.selected_id, box.id)
            box_node.visible = False
            editor._select_scene_item(edge, direction)
            self.assertEqual(editor.selected_id, pose_node.id)
            box_node.visible = True
            box_node.local_xform = wp.transform(3.2, 0.2, -4, 0, 0, 0, 1)
            editor._select_scene_item(edge, direction)
            self.assertEqual(editor.selected_id, pose_node.id)


if __name__ == "__main__":
    unittest.main()
