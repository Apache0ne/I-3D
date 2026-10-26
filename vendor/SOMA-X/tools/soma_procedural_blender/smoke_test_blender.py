# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Headless Blender smoke test for the SOMA procedural add-on."""

import logging
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix

PLUGIN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PLUGIN_ROOT.parents[2]
DEFINITION_PATH = REPO_ROOT / "assets" / "SOMA_procedural_transforms.json"
sys.path.insert(0, str(PLUGIN_ROOT))

logger = logging.getLogger(__name__)


def _clear_scene() -> None:
    if bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def _create_armature():
    from soma_procedural_blender.definition import load_definition

    definition = load_definition(DEFINITION_PATH)
    armature_data = bpy.data.armatures.new("SOMA_TestRigData")
    armature_object = bpy.data.objects.new("SOMA_TestRig", armature_data)
    bpy.context.collection.objects.link(armature_object)
    bpy.ops.object.select_all(action="DESELECT")
    armature_object.select_set(True)
    bpy.context.view_layer.objects.active = armature_object
    bpy.ops.object.mode_set(mode="EDIT")

    for index, bone_name in enumerate(
        (*definition.public_joint_names, *definition.twist_joint_names)
    ):
        edit_bone = armature_data.edit_bones.new(bone_name)
        x = float(index) * 0.02
        edit_bone.head = (x, 0.0, 0.0)
        edit_bone.tail = (x, 0.1, 0.0)

    bpy.ops.object.mode_set(mode="POSE")
    return armature_object


def _set_pose_matrix(
    armature_object,
    bone_name: str,
    translation: tuple[float, float, float],
    angle: float,
) -> None:
    pose_bone = armature_object.pose.bones[bone_name]
    pose_bone.matrix = Matrix.Translation(translation) @ Matrix.Rotation(angle, 4, "X")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from soma_procedural_blender.blender_reference import configure_armature, evaluate_armature

    _clear_scene()
    armature_object = _create_armature()
    configure_armature(armature_object, definition_path=DEFINITION_PATH)

    _set_pose_matrix(armature_object, "LeftArm", (0.0, 0.0, 0.0), 0.6)
    _set_pose_matrix(armature_object, "LeftForeArm", (3.0, 0.0, 0.0), 0.3)
    bpy.context.view_layer.update()

    written = evaluate_armature(armature_object)
    bpy.context.view_layer.update()

    matrix = armature_object.pose.bones["LeftArmTwist1"].matrix
    twist_x = math.atan2(matrix[2][1], matrix[1][1])
    translation_x = matrix[0][3]
    if not math.isclose(twist_x, -0.5850001, rel_tol=1e-5, abs_tol=1e-5):
        raise AssertionError(f"unexpected LeftArmTwist1 rotation: {twist_x}")
    if not math.isclose(translation_x, 1.56, rel_tol=1e-5, abs_tol=1e-5):
        raise AssertionError(f"unexpected LeftArmTwist1 translation: {translation_x}")
    logger.info(
        f"BLENDER_PLUGIN_SMOKE_TEST_OK outputs={len(written)} twist_x={twist_x:.7f} tx={translation_x:.7f}"
    )


if __name__ == "__main__":
    main()
