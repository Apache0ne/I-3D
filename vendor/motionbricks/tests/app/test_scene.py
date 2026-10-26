# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import warp as wp

from app.runtime.common.io.document_io import DocumentError
from app.runtime.common.io.scene_document import (
    SCENE_DOCUMENT,
    CharacterStart,
    SceneDefinition,
    SceneDocumentError,
    check_reference_rigs,
    load_scene_document,
    save_scene_document,
)
from app.runtime.common.io.smart_primitive_document import SmartPrimitiveDocument, save_smart_primitive_document
from motionbricks.model.contract import ModelRig


def _camera_state() -> dict:
    return {
        "pos": [1.0, 2.0, 3.0],
        "pivot": [4.0, 5.0, 6.0],
        "pitch": -20.0,
        "yaw": 45.0,
        "fov": 60.0,
        "near": 0.01,
        "far": 1000.0,
    }


def _transform_values(transform: wp.transform) -> np.ndarray:
    return np.asarray([float(transform[index]) for index in range(7)])


def _turn(angle: float) -> wp.quat:
    return wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), angle)


def _assert_same_placement(test, actual: wp.transform, expected: wp.transform) -> None:
    actual, expected = _transform_values(actual), _transform_values(expected)
    np.testing.assert_allclose(actual[:3], expected[:3], atol=1.0e-5)
    test.assertGreater(abs(float(np.dot(actual[3:], expected[3:]))), 1.0 - 1.0e-6)


class SceneHierarchyTests(unittest.TestCase):
    def test_parenting_keeps_placements_and_branches_follow_their_parent(self) -> None:
        scene = SceneDefinition()
        jump, platform, step = scene.add_smart_primitive(), scene.add_solid_cube(), scene.add_solid_cube()
        jump.transform = wp.transform(wp.vec3(2.0, 0.0, 0.0), _turn(0.5 * wp.pi))
        platform.transform = wp.transform(wp.vec3(3.0, 1.0, 0.5), _turn(0.3))
        step.transform = wp.transform(wp.vec3(3.0, 2.0, 0.25), wp.quat_identity())
        before = {item.id: scene.world_transform(item) for item in (jump, platform, step)}

        self.assertTrue(scene.set_parent(platform, jump.id))
        self.assertTrue(scene.set_parent(step, platform.id))
        for item in (jump, platform, step):
            _assert_same_placement(self, scene.world_transform(item), before[item.id])
        np.testing.assert_allclose(_transform_values(platform.transform)[:3], [1.0, -1.0, 0.5], atol=1.0e-6)
        self.assertEqual([item.id for item in scene.children(None)], [jump.id])
        self.assertEqual([item.id for item in scene.subtree(jump)], [jump.id, platform.id, step.id])

        # Moving and turning the root carries the whole branch through both levels.
        jump.transform = wp.transform(wp.vec3(5.0, 0.0, 0.0), wp.quat_identity())
        _assert_same_placement(
            self, scene.world_transform(platform), wp.transform(wp.vec3(6.0, -1.0, 0.5), _turn(0.3 - 0.5 * wp.pi)),
        )
        _assert_same_placement(
            self, scene.world_transform(step), wp.transform(wp.vec3(7.0, -1.0, 0.25), _turn(-0.5 * wp.pi)),
        )

        start = scene.character_start
        for item, parent_id in (
            (jump, step.id), (platform, platform.id), (start, jump.id),
            (platform, start.id), (platform, 999), (platform, jump.id),
        ):
            with self.subTest(item=item.id, parent_id=parent_id):
                self.assertFalse(scene.set_parent(item, parent_id))

        world = scene.world_transform(step)
        self.assertTrue(scene.set_parent(step, None))
        self.assertIsNone(step.parent_id)
        _assert_same_placement(self, step.transform, world)

    def test_removing_an_item_removes_everything_under_it(self) -> None:
        scene = SceneDefinition()
        jump, other = scene.add_smart_primitive(), scene.add_smart_primitive()
        platform, step = scene.add_solid_cube(), scene.add_solid_cube()
        self.assertEqual(len({item.id for item in (scene.character_start, jump, other, platform, step)}), 5)
        scene.set_parent(platform, jump.id)
        scene.set_parent(step, platform.id)

        self.assertEqual(scene.remove_item(scene.character_start.id), [])
        self.assertEqual(scene.remove_item(jump.id), [jump.id, platform.id, step.id])
        self.assertEqual(scene.smart_primitives, [other])
        self.assertEqual(scene.solid_cubes, [])

    def test_hierarchy_round_trips_and_flattens_into_world_transforms(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            scene = SceneDefinition()
            jump, platform = scene.add_smart_primitive(), scene.add_solid_cube()
            jump.transform = wp.transform(wp.vec3(2.0, 0.0, 0.0), _turn(0.5 * wp.pi))
            platform.transform = wp.transform(wp.vec3(3.0, 1.0, 0.5), wp.quat_identity())
            scene.set_parent(platform, jump.id)
            path = save_scene_document(scene, Path(directory) / "nested", platform.id, _camera_state())

            loaded, _, _ = load_scene_document(path)
            loaded_platform = loaded.find_item(platform.id)
            self.assertEqual(loaded_platform.parent_id, jump.id)
            self.assertEqual([item.id for item in loaded.children(None)], [jump.id])
            np.testing.assert_allclose(
                _transform_values(loaded_platform.transform), _transform_values(platform.transform), atol=1.0e-6,
            )
            flat = loaded.flatten()
            self.assertTrue(all(item.parent_id is None for item in (*flat.smart_primitives, *flat.solid_cubes)))
            _assert_same_placement(self, flat.find_item(platform.id).transform, loaded.world_transform(loaded_platform))
            self.assertEqual(loaded_platform.parent_id, jump.id)


class SceneDocumentTests(unittest.TestCase):
    def test_path_normalization_uses_scene_json_suffix(self) -> None:
        for path in ("demo", "demo.json", "demo.scene"):
            self.assertEqual(SCENE_DOCUMENT.normalize_path(path), Path("demo.scene.json"))

    def test_round_trip_preserves_scene_and_editor_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "jump.smart_primitive.json"
            reference.touch()
            scene = SceneDefinition(
                rig=ModelRig.UNITREE_G1,
                character_start=CharacterStart(
                    transform=wp.transform(
                        wp.vec3(1.0, 2.0, 3.0),
                        wp.quat_rpy(0.1, 0.2, 0.3),
                    ),
                    euler_degrees=[5.0, 10.0, 15.0],
                )
            )
            instance = scene.add_smart_primitive()
            instance.name = "Jump"
            instance.transform = wp.transform(
                wp.vec3(5.0, 6.0, 0.0),
                wp.quat_rpy(0.0, 0.0, 1.0),
            )
            instance.euler_degrees = [0.0, 0.0, 57.2958]
            instance.reference = reference
            instance.show_target_poses = False
            instance.show_interaction_boxes = True
            instance.target_pose_color = wp.vec3(0.2, 0.6, 0.9)
            instance.behavior = "VelocityMatchingSequenceBehavior"
            unassigned = scene.add_smart_primitive()
            cube = scene.add_solid_cube()
            cube.name = "Platform"
            cube.transform = wp.transform(
                wp.vec3(-2.0, 3.0, 1.0),
                wp.quat_rpy(0.1, 0.0, -0.2),
            )
            cube.euler_degrees = [5.72958, 0.0, -11.45916]
            cube.width = 4.0
            cube.depth = 2.0
            cube.height = 0.5
            cube.color = wp.vec3(0.2, 0.4, 0.8)

            document_path = save_scene_document(
                scene,
                root / "demo",
                cube.id,
                _camera_state(),
            )
            loaded, selected_item_id, camera = load_scene_document(document_path)

            self.assertEqual(document_path.name, "demo.scene.json")
            self.assertEqual(json.loads(document_path.read_text("utf-8"))["rig"], "unitree_g1")
            self.assertIs(loaded.rig, ModelRig.UNITREE_G1)
            self.assertEqual(selected_item_id, cube.id)
            self.assertEqual(camera, _camera_state())
            np.testing.assert_allclose(
                _transform_values(loaded.character_start.transform),
                _transform_values(scene.character_start.transform),
            )
            self.assertEqual(len(loaded.smart_primitives), 2)
            self.assertIsNone(loaded.find_item(unassigned.id).reference)
            loaded_instance = loaded.smart_primitives[0]
            self.assertEqual(loaded_instance.name, "Jump")
            self.assertEqual(loaded_instance.reference, reference.resolve())
            self.assertFalse(loaded_instance.show_target_poses)
            self.assertTrue(loaded_instance.show_interaction_boxes)
            self.assertEqual(loaded_instance.target_pose_color, instance.target_pose_color)
            self.assertEqual(
                loaded_instance.behavior,
                "VelocityMatchingSequenceBehavior",
            )
            np.testing.assert_allclose(
                _transform_values(loaded_instance.transform),
                _transform_values(instance.transform),
            )
            self.assertEqual(len(loaded.solid_cubes), 1)
            loaded_cube = loaded.solid_cubes[0]
            self.assertEqual(loaded_cube.name, "Platform")
            self.assertEqual(
                (loaded_cube.width, loaded_cube.depth, loaded_cube.height),
                (4.0, 2.0, 0.5),
            )
            self.assertEqual(loaded_cube.color, cube.color)
            np.testing.assert_allclose(
                _transform_values(loaded_cube.transform),
                _transform_values(cube.transform),
            )

            document = json.loads(document_path.read_text("utf-8"))
            document["editor"]["selected_item_id"] = 9999
            renamed_path = root / "renamed.scene.json"
            renamed_path.write_text(json.dumps(document), encoding="utf-8")
            renamed, selected_item_id, _ = load_scene_document(renamed_path)
            self.assertEqual(selected_item_id, renamed.character_start.id)
            self.assertEqual(renamed.smart_primitives[0].reference, reference.resolve())
            self.assertEqual(renamed.solid_cubes[0].name, cube.name)

    def test_every_reference_must_use_the_scene_rig(self) -> None:
        with tempfile.TemporaryDirectory() as directory, wp.ScopedDevice("cpu"):
            root = Path(directory)
            references = {}
            for rig, reference_pose in (
                (ModelRig.SOMA, wp.array([wp.transform_identity()], dtype=wp.transform)),
                (ModelRig.UNITREE_G1, wp.array([0, 0, 0, 0, 0, 0, 1, 0], dtype=wp.float32)),
            ):
                document = SmartPrimitiveDocument(rig, reference_pose, wp.transform_identity())
                references[rig] = save_smart_primitive_document(
                    document, root / rig, document.root.id, _camera_state(),
                )
            scene = SceneDefinition(rig=ModelRig.UNITREE_G1)
            scene.add_smart_primitive()
            scene.add_smart_primitive().reference = references[ModelRig.UNITREE_G1]
            check_reference_rigs(scene)

            scene.add_smart_primitive().reference = references[ModelRig.SOMA]
            with self.assertRaisesRegex(
                SceneDocumentError,
                r"soma\.smart_primitive\.json was saved for the 'soma' rig, but this scene uses 'unitree_g1'",
            ):
                check_reference_rigs(scene)

    def test_scene_rejects_non_smart_primitive_references(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene = SceneDefinition()
            reference = root / "unrelated.json"
            reference.touch()
            scene.add_smart_primitive().reference = reference
            with self.assertRaisesRegex(SceneDocumentError, "must reference"):
                save_scene_document(scene, root / "invalid", 1, _camera_state())

    def test_references_are_relative_so_moved_folders_still_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            original = Path(directory) / "original"
            reference = original / "smart_primitives" / "asset.smart_primitive.json"
            reference.parent.mkdir(parents=True)
            reference.touch()
            scene = SceneDefinition()
            instance = scene.add_smart_primitive()
            instance.reference = reference
            document_path = save_scene_document(
                scene, original / "scenes" / "relative", instance.id, _camera_state(),
            )
            document = json.loads(document_path.read_text(encoding="utf-8"))
            self.assertEqual(
                document["smart_primitives"][0]["reference"],
                "../smart_primitives/asset.smart_primitive.json",
            )

            moved = original.rename(Path(directory) / "moved")
            loaded, _, _ = load_scene_document(moved / "scenes" / document_path.name)
            self.assertEqual(
                loaded.smart_primitives[0].reference,
                (moved / "smart_primitives" / reference.name).resolve(),
            )

    def test_loader_and_saver_reject_invalid_documents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "asset.smart_primitive.json"
            reference.touch()
            scene = SceneDefinition()
            instance = scene.add_smart_primitive()
            first, second = scene.add_solid_cube(), scene.add_solid_cube()
            scene.set_parent(second, first.id)
            path = save_scene_document(scene, root / "invalid", instance.id, _camera_state())
            original = path.read_text(encoding="utf-8")

            # A reference on another Windows drive is saved as an absolute path.
            document = json.loads(original)
            document["smart_primitives"][0]["reference"] = reference.resolve().as_posix()
            path.write_text(json.dumps(document), encoding="utf-8")
            loaded, _, _ = load_scene_document(path)
            self.assertEqual(loaded.smart_primitives[0].reference, reference.resolve())

            for location, value, message in (
                (("rig",), "robot", "invalid.scene.json uses unknown rig 'robot'"),
                (("unexpected",), True, "unexpected"),
                (("editor", "camera", "far"), 0.001, "camera.far"),
                (("character_start", "transform", "rotation_xyzw"), [0, 0, 0, 2], "non-unit quaternion"),
                (("smart_primitives", 0, "target_pose_color"), [0, 0, 1.1], "target_pose_color"),
                (("smart_primitives", 0, "reference"), "", "reference must be a non-empty string"),
                (("smart_primitives", 0, "reference"), "   ", "reference must be a non-empty string"),
                (
                    ("smart_primitives", 0, "reference"), "missing.smart_primitive.json",
                    "Could not find the Smart Primitive",
                ),
                (("solid_cubes", 0, "parent_id"), 999, "as its parent"),
                (("solid_cubes", 0, "parent_id"), scene.character_start.id, "as its parent"),
                (("solid_cubes", 0, "parent_id"), first.id, "under itself"),
                (("solid_cubes", 0, "parent_id"), second.id, "under itself"),
            ):
                with self.subTest(location=location, value=value):
                    document = json.loads(original)
                    *keys, field = location
                    container = document
                    for key in keys:
                        container = container[key]
                    container[field] = value
                    path.write_text(json.dumps(document), encoding="utf-8")
                    with self.assertRaisesRegex(DocumentError, message):
                        load_scene_document(path)

            first.parent_id = second.id
            with self.assertRaisesRegex(SceneDocumentError, "under itself"):
                save_scene_document(scene, path, 1, _camera_state())

    def test_save_rejects_items_the_loader_would_refuse(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for add_item, field, value, message in (
                (SceneDefinition.add_solid_cube, "height", 0.0, "height must be greater than zero"),
                (SceneDefinition.add_solid_cube, "name", "", "name cannot be empty"),
                (SceneDefinition.add_smart_primitive, "name", " ", "name cannot be empty"),
            ):
                with self.subTest(field=field, value=value):
                    scene = SceneDefinition()
                    item = add_item(scene)
                    saved = save_scene_document(scene, Path(directory) / "scene", item.id, _camera_state())
                    original = saved.read_bytes()
                    setattr(item, field, value)
                    with self.assertRaisesRegex(SceneDocumentError, message):
                        save_scene_document(scene, saved, item.id, _camera_state())
                    # A refused save leaves the existing file as it was.
                    self.assertEqual(saved.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
