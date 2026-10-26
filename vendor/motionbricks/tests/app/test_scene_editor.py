# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import newton
import numpy as np
import warp as wp

from app.authoring.common import InlineRename
from app.runtime.common.clip_registry import ClipRegistry
from app.runtime.common.io.scene_document import SceneDefinition, load_scene_document, save_scene_document
from app.runtime.common.io.smart_primitive_document import SmartPrimitiveDocument, save_smart_primitive_document
from app.scene_editor import SceneEditor, _CharacterStartRenderer
from motionbricks.animation.mesh import SkeletalMesh, SkinnedMesh
from motionbricks.animation.skeleton import Skeleton
from motionbricks.io.csv import save_csv
from motionbricks.model.contract import ModelRig
from motionbricks.renderers.dof_mesh_renderer import DOFMeshRenderer
from motionbricks.renderers.skeletal_mesh_renderer import SkeletalMeshRenderer
from motionbricks.renderers.solid_box_renderer import SolidBoxRenderer
from motionbricks.robotics.dof_animation_buffer import DOFAnimationBuffer


_CAMERA = {"pos": [0, -3, 2], "pivot": [0, 0, 1], "pitch": -15, "yaw": 90,
           "fov": 45, "near": 0.01, "far": 100}


class SceneEditorRigTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(wp.ScopedDevice("cpu"))
        self.folder = Path(self.enterContext(TemporaryDirectory()))
        skeleton = Skeleton(["Hips"], [-1], [wp.transform_identity()])
        mesh = SkinnedMesh(
            np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
            np.array([0, 1, 2], dtype=np.int32),
            np.zeros(3, dtype=np.int32), np.ones(3, dtype=np.float32),
        )
        # Like the SOMA mesh, it has the relaxed-hands override pose and finger mask.
        soma = SkeletalMesh(
            [mesh], skeleton, [wp.transform_identity()],
            override_poses={"RelaxedHands": wp.clone(skeleton.reference_local_transforms)},
            override_pose_masks={"Fingers": wp.zeros(1, dtype=wp.float32)},
        )
        builder = newton.ModelBuilder()
        root = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        arm = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_articulation([
            builder.add_joint_free(root),
            builder.add_joint_revolute(root, arm, axis=wp.vec3(0, 0, 1)),
        ])
        builder.add_shape_box(root, hx=0.2, hy=0.2, hz=0.2)
        self.robot = builder.finalize()
        self.load_soma = self.enterContext(
            patch("app.runtime.common.rig.io_utils.get_soma_human_model", return_value=soma),
        )
        self.load_g1 = self.enterContext(
            patch("app.runtime.common.rig.io_utils.get_unitree_g1_model", return_value=self.robot),
        )
        self.enterContext(
            patch("app.runtime.common.rig.UNITREE_G1_N1_CONTRACT", SimpleNamespace(joint_names=("hinge",))),
        )

        bvh = self.folder / "walk.bvh"
        bvh.write_text(
            "HIERARCHY\nROOT Hips\n{\n OFFSET 0 0 0\n"
            " CHANNELS 6 Xposition Yposition Zposition Xrotation Yrotation Zrotation\n}\n"
            "MOTION\nFrames: 2\nFrame Time: 0.03333333333333333\n"
            "0 100 0 0 0 0\n10 100 0 0 0 0\n", encoding="utf-8",
        )
        self.soma_path = self._save_primitive(
            ClipRegistry(ModelRig.SOMA, skeleton=skeleton), skeleton.reference_local_transforms, bvh,
        )
        csv = self.folder / "walk.csv"
        frames = np.tile([0, 0, 1, 0, 0, 0, 1, 0.4], (4, 1)).astype(np.float32)
        save_csv(csv, DOFAnimationBuffer(("hinge",), 120, wp.array(frames, dtype=wp.float32)))
        self.g1_path = self._save_primitive(
            ClipRegistry(ModelRig.UNITREE_G1, joint_names=("hinge",)),
            self.robot.joint_q, csv,
        )
        self.editor = SceneEditor.__new__(SceneEditor)
        self.editor.device = wp.get_device("cpu")
        self.editor._rig_assets = None
        self.editor._asset_pool = None
        self.editor._runtime_views = {}
        self.editor.scene = SceneDefinition()
        self.editor.viewer = SimpleNamespace(objects={}, lines={})

    def _save_primitive(self, registry, reference_pose, clip_path):
        document = SmartPrimitiveDocument(registry.rig, reference_pose, wp.transform_identity())
        document.add_target_pose(document.add_node()).set_clip(registry.load(clip_path))
        return save_smart_primitive_document(document, self.folder / registry.rig, document.root.id, _CAMERA)

    def _save_unknown_rig_primitive(self):
        path = self.folder / "invalid.smart_primitive.json"
        document = json.loads(self.g1_path.read_text(encoding="utf-8"))
        document["rig"] = "robot"
        path.write_text(json.dumps(document), encoding="utf-8")
        return path

    def test_children_follow_their_parent_in_previews_picking_and_deletion(self):
        editor = self.editor
        editor._gizmo_item_id = editor._gizmo_transform = editor._gizmo_source_transform = None
        editor._transform_locks = {}
        editor._rename = InlineRename()
        editor.solid_box_renderer = SolidBoxRenderer()
        editor.character_start_renderer = _CharacterStartRenderer()
        editor.viewer.camera = SimpleNamespace(far=100.0)
        jump, child = editor.scene.add_smart_primitive(), editor.scene.add_smart_primitive()
        platform = editor.scene.add_solid_cube()
        for instance in (jump, child):
            editor._assign_reference(instance, self.soma_path)
        editor._set_item_transform(child, wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity()))
        editor._set_item_transform(platform, wp.transform(wp.vec3(0.0, 3.0, 0.5), wp.quat_identity()))
        editor._set_parent(child.id, jump.id)
        editor._set_parent(platform.id, jump.id)
        self.assertEqual(editor.selected_item_id, platform.id)

        editor._set_item_transform(jump, wp.transform(wp.vec3(10.0, 0.0, 0.0), wp.quat_identity()))
        nodes = wp.empty(editor._runtime_views[child.id].runtime.node_count, dtype=wp.transform)
        editor._runtime_views[child.id].runtime.compute_world_transforms(nodes)
        np.testing.assert_allclose(nodes.numpy()[0][:3], [11.0, 0.0, 0.0], atol=1e-6)
        editor.selected_item_id = editor.scene.character_start.id
        editor._select_scene_item(wp.vec3(10.0, 3.0, 5.0), wp.vec3(0.0, 0.0, -1.0))
        self.assertEqual(editor.selected_item_id, platform.id)
        # Picking in the view opens the tree down to the picked child.
        self.assertEqual(editor._reveal_item_ids, {platform.id, jump.id})

        editor._delete_item(jump.id)
        self.assertEqual(editor._runtime_views, {})
        self.assertEqual((editor.scene.smart_primitives, editor.scene.solid_cubes), ([], []))
        self.assertEqual(editor.selected_item_id, editor.scene.character_start.id)

    def test_g1_scene_uses_dof_previews_shares_assets_and_never_loads_soma(self):
        scene = SceneDefinition(rig=ModelRig.UNITREE_G1)
        for _ in range(2):
            scene.add_smart_primitive().reference = self.g1_path
        scene_path = save_scene_document(scene, self.folder / "g1", 1, _CAMERA)
        loaded, _, _ = load_scene_document(scene_path)
        views = list(self.editor._create_runtime_views(loaded).values())
        for view in views:
            self.assertIsInstance(view.mesh_renderer, DOFMeshRenderer)
            self.assertEqual(view.target_poses.dtype, wp.float32)
        self.assertIs(views[0].runtime.asset, views[1].runtime.asset)

        original_pose = self.robot.joint_q.numpy().copy()
        views[0].set_transform(wp.transform(5, 0, 0, 0, 0, 0, 1))
        views[0].runtime.get_world_target_poses(1, views[0].target_poses)
        np.testing.assert_allclose(views[0].target_poses.numpy()[:, :3], [[5, 0, 1]] * 4)
        np.testing.assert_allclose(views[0].target_poses.numpy()[:, 7], 0.4)
        self.assertAlmostEqual(views[0].raycast(wp.vec3(5, 0, 3), wp.vec3(0, 0, -1), 100), 1.8, places=5)
        views[1].runtime.get_world_target_poses(1, views[1].target_poses)
        np.testing.assert_allclose(views[1].target_poses.numpy()[:, 0], 0)
        np.testing.assert_array_equal(self.robot.joint_q.numpy(), original_pose)

        reopened = list(self.editor._create_runtime_views(loaded).values())
        for previous, current in zip(views, reopened):
            self.assertIs(previous.runtime.asset, current.runtime.asset)
        self.load_soma.assert_not_called()
        self.load_g1.assert_called_once_with()

    def test_a_reference_for_another_rig_is_rejected_and_the_previous_view_kept(self):
        instance = self.editor.scene.add_smart_primitive()
        self.editor._assign_reference(instance, self.soma_path)
        self.assertIsNone(self.editor._status)
        view = self.editor._runtime_views[instance.id]
        self.assertIsInstance(view.mesh_renderer, SkeletalMeshRenderer)
        self.assertEqual(view.target_poses.dtype, wp.transform)

        for path, message in (
            (self.g1_path, "was saved for the 'unitree_g1' rig, but this scene uses 'soma'"),
            (self._save_unknown_rig_primitive(), "unknown rig 'robot'"),
        ):
            with self.subTest(reference=path.name):
                self.editor._assign_reference(instance, path)
                self.assertEqual(instance.reference, self.soma_path)
                self.assertIs(self.editor._runtime_views[instance.id], view)
                self.assertIn(message, self.editor._status)
        self.load_g1.assert_not_called()

    def test_the_first_reference_chooses_the_character(self):
        first = self.editor.scene.add_smart_primitive()
        second = self.editor.scene.add_smart_primitive()
        self.editor._assign_reference(first, self._save_unknown_rig_primitive())
        self.assertIn("unknown rig 'robot'", self.editor._status)
        self.assertIsNone(first.reference)
        self.assertIs(self.editor.scene.rig, ModelRig.SOMA)

        self.editor._assign_reference(first, self.g1_path)
        self.assertIsNone(self.editor._status)
        self.assertIs(self.editor.scene.rig, ModelRig.UNITREE_G1)
        self.assertIsInstance(self.editor._runtime_views[first.id].mesh_renderer, DOFMeshRenderer)

        self.editor._assign_reference(second, self.soma_path)
        self.assertIn("was saved for the 'soma' rig, but this scene uses 'unitree_g1'", self.editor._status)
        self.assertIsNone(second.reference)
        self.assertNotIn(second.id, self.editor._runtime_views)
        self.assertIs(self.editor.scene.rig, ModelRig.UNITREE_G1)
        self.load_soma.assert_not_called()

    def test_loading_a_scene_with_another_rigs_reference_keeps_the_current_scene(self):
        scene = SceneDefinition(rig=ModelRig.UNITREE_G1)
        scene.add_smart_primitive().reference = self.g1_path
        scene.add_smart_primitive().reference = self.soma_path
        scene_path = save_scene_document(scene, self.folder / "mixed", 1, _CAMERA)
        current = self.editor.scene

        self.editor._open_document(scene_path)

        self.assertIs(self.editor.scene, current)
        self.assertIn(
            "soma.smart_primitive.json was saved for the 'soma' rig, but this scene uses 'unitree_g1'",
            self.editor._status,
        )
        self.load_soma.assert_not_called()
        self.load_g1.assert_not_called()

    def test_opening_a_file_without_the_scene_suffix_keeps_the_current_scene(self):
        saved = save_scene_document(SceneDefinition(), self.folder / "renamed", 1, _CAMERA)
        renamed = saved.rename(saved.with_name("renamed.json"))
        current = self.editor.scene

        self.editor._open_document(renamed)

        self.assertIs(self.editor.scene, current)
        self.assertIn("Scene documents must end in .scene.json", self.editor._status)


if __name__ == "__main__":
    unittest.main()
