# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Override poses load from USD and only change the drawn mesh, never the poses passed in."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import warp as wp
from pxr import Gf, Usd, UsdGeom, UsdSkel

from motionbricks.animation.mesh import SkeletalMesh, SkinnedMesh
from motionbricks.animation.skeleton import Skeleton
from motionbricks.io.usd import load_skeletal_mesh_with_skeleton_from_usd
from motionbricks.renderers.skeletal_mesh_renderer import SkeletalMeshRenderer


def _drawn_points(renderer, pose):
    viewer = Mock()
    renderer.draw_pose(viewer, pose, wp.transform_identity(), wp.vec3(1, 1, 1), "pose")
    return viewer.log_mesh.call_args.args[1].numpy()


def _finger_mesh():
    """Return a two joint mesh with a Curled override pose and Finger and HalfFinger masks."""
    reference = np.array([[0, 0, 0, 0, 0, 0, 1], [0, 1, 0, 0, 0, 0, 1]], dtype=np.float32)
    skeleton = Skeleton(["Hips", "Finger"], [-1, 0], reference)
    # One vertex follows the finger, which is bound one unit above the hips.
    part = SkinnedMesh(
        np.array([[1, 1, 0]], dtype=np.float32), np.array([0, 0, 0], dtype=np.int32),
        np.array([1], dtype=np.int32), np.array([1], dtype=np.float32),
    )
    inverse_binds = [wp.transform_identity(), wp.transform(0, -1, 0, 0, 0, 0, 1)]
    # The override pose curls the finger a quarter turn. Its hips are far away, but no mask selects them.
    turn = wp.quat_from_axis_angle(wp.vec3(0, 0, 1), 0.5 * np.pi)
    curled = wp.array([[9, 9, 9, 0, 0, 0, 1], [0, 1, 0, *turn]], dtype=wp.transform)
    masks = {"Finger": skeleton.joint_mask(["Finger"]), "HalfFinger": skeleton.joint_mask(["Finger"], 0.5)}
    return SkeletalMesh([part], skeleton, inverse_binds, override_poses={"Curled": curled}, override_pose_masks=masks)


def _write_character(path, frame=None):
    """Write a three joint character in centimeters with one partial SkelAnimation.

    The animation holds default values, or only time samples at frame on a single frame stage.
    """
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    time = Usd.TimeCode.Default()
    if frame is not None:
        stage.SetStartTimeCode(frame)
        stage.SetEndTimeCode(frame)
        time = Usd.TimeCode(frame)
    UsdSkel.Root.Define(stage, "/Character")
    skeleton = UsdSkel.Skeleton.Define(stage, "/Character/Skeleton")
    skeleton.CreateJointsAttr(["Hips", "Hips/Finger", "Hips/Finger/Tip"])
    # The joints stack up from the hips, 10 cm and then 5 cm apart.
    rest_heights, bind_heights = (100, 10, 5), (100, 110, 115)
    skeleton.CreateRestTransformsAttr([Gf.Matrix4d().SetTranslate(Gf.Vec3d(0, y, 0)) for y in rest_heights])
    skeleton.CreateBindTransformsAttr([Gf.Matrix4d().SetTranslate(Gf.Vec3d(0, y, 0)) for y in bind_heights])
    curled = UsdSkel.Animation.Define(stage, "/Character/Skeleton/Curled")
    curled.CreateJointsAttr(["Hips/Finger"])
    curled.CreateTranslationsAttr().Set([Gf.Vec3f(0, 8, 0)], time)
    curled.CreateRotationsAttr().Set([Gf.Quatf(np.sqrt(0.5), 0, 0, np.sqrt(0.5))], time)
    curled.CreateScalesAttr().Set([Gf.Vec3h(1, 1, 1)], time)
    mesh = UsdGeom.Mesh.Define(stage, "/Character/Mesh")
    mesh.CreatePointsAttr([(0, 110, 0), (1, 110, 0), (0, 111, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    binding = UsdSkel.BindingAPI.Apply(mesh.GetPrim())
    binding.CreateSkeletonRel().SetTargets([skeleton.GetPath()])
    binding.CreateJointIndicesPrimvar(False, 1).Set([1, 1, 1])
    binding.CreateJointWeightsPrimvar(False, 1).Set([1, 1, 1])
    stage.Save()


class OverridePoseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.previous_device = wp.get_device()
        wp.set_device("cpu")

    @classmethod
    def tearDownClass(cls):
        wp.set_device(cls.previous_device)

    def test_joint_mask_weights_only_the_named_joints(self):
        skeleton = Skeleton(["Hips", "Hand", "Finger"], [-1, 0, 1], np.tile([0, 0, 0, 0, 0, 0, 1], (3, 1)))
        np.testing.assert_allclose(skeleton.joint_mask(["Finger", "Hips"], 0.25).numpy(), [0.25, 0, 0.25])
        np.testing.assert_allclose(skeleton.joint_mask(["Toe", "Hand"]).numpy(), [0, 1, 0])
        np.testing.assert_allclose(skeleton.joint_mask(["Hand"], 1.5).numpy(), [0, 1, 0])
        np.testing.assert_allclose(skeleton.joint_mask(["Hand"], -0.5).numpy(), [0, 0, 0])
        if wp.is_cuda_available():
            # The mask lives with the skeleton, whatever the current device is.
            with wp.ScopedDevice("cuda:0"):
                self.assertEqual(skeleton.joint_mask(["Hand"]).device, skeleton.reference_local_transforms.device)

    def test_override_pose_blends_each_joint_by_its_mask_weight(self):
        mesh = _finger_mesh()
        renderer = SkeletalMeshRenderer(mesh)
        # The animated pose moves the hips and stretches the finger.
        pose = wp.array([[5, 0, 0, 0, 0, 0, 1], [0, 3, 0, 0, 0, 0, 1]], dtype=wp.transform)
        animated = pose.numpy().copy()
        np.testing.assert_allclose(_drawn_points(renderer, pose), [[6, 3, 0]], atol=1e-6)

        renderer.set_override_pose("Curled", mask="Finger")
        np.testing.assert_allclose(_drawn_points(renderer, pose), [[5, 2, 0]], atol=1e-6)
        # Half weight puts the finger halfway up and turns it an eighth of a turn.
        renderer.set_override_pose("Curled", mask="HalfFinger")
        np.testing.assert_allclose(_drawn_points(renderer, pose), [[5 + np.sqrt(0.5), 2 + np.sqrt(0.5), 0]], atol=1e-6)
        # Weights outside 0 to 1 are clamped, and a NaN weight counts as 0.
        for weight, expected in ((1.5, [5, 2, 0]), (-0.5, [6, 3, 0]), (np.nan, [6, 3, 0])):
            with self.subTest(weight=weight):
                mesh.override_pose_masks["Odd"] = wp.array([0.0, weight], dtype=wp.float32)
                renderer.set_override_pose("Curled", mask="Odd")
                np.testing.assert_allclose(_drawn_points(renderer, pose), [expected], atol=1e-6)
        renderer.set_override_pose(None)
        np.testing.assert_allclose(_drawn_points(renderer, pose), [[6, 3, 0]], atol=1e-6)
        np.testing.assert_array_equal(pose.numpy(), animated)

    def test_override_pose_selection_rejects_masks_that_do_not_fit(self):
        mesh = _finger_mesh()
        masks = mesh.override_pose_masks
        masks["Short"] = wp.zeros(1, dtype=wp.float32)
        masks["Double"] = wp.zeros(2, dtype=wp.float64)
        renderer = SkeletalMeshRenderer(mesh)
        renderer.set_override_pose("Curled", mask="Finger")
        for name in ("Short", "Double"):
            with self.subTest(mask=name), self.assertRaises(ValueError):
                renderer.set_override_pose("Curled", mask=name)
        with self.assertRaises(ValueError):
            renderer.set_override_pose("Curled")
        with self.assertRaises(KeyError):
            renderer.set_override_pose("Missing", mask="Finger")

        # A rejected selection keeps the previous one.
        pose = wp.array([[5, 0, 0, 0, 0, 0, 1], [0, 3, 0, 0, 0, 0, 1]], dtype=wp.transform)
        np.testing.assert_allclose(_drawn_points(renderer, pose), [[5, 2, 0]], atol=1e-6)
        if wp.is_cuda_available():
            # Drawing with a mask on another device would crash, so selection rejects it.
            masks["Elsewhere"] = wp.zeros(2, dtype=wp.float32, device="cuda:0")
            with self.assertRaises(ValueError):
                renderer.set_override_pose("Curled", mask="Elsewhere")

    def test_usd_skel_animations_load_as_full_override_poses_in_meters(self):
        # Exporters write a pose as default values or as a single time sample.
        for frame in (None, 1):
            with self.subTest(frame=frame), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "character.usda"
                _write_character(path, frame)
                mesh = load_skeletal_mesh_with_skeleton_from_usd(path, "/Character/Mesh", "/Character/Skeleton")

                self.assertEqual(list(mesh.override_poses), ["Curled"])
                pose = mesh.override_poses["Curled"].numpy()
                rest = mesh.skeleton.reference_local_transforms.numpy()
                np.testing.assert_allclose(pose[1], [0, 0.08, 0, 0, 0, np.sqrt(0.5), np.sqrt(0.5)], atol=1e-6)
                # Joints the animation does not list keep their rest transforms.
                np.testing.assert_allclose(pose[[0, 2]], rest[[0, 2]])
                np.testing.assert_allclose(rest[:, :3], [[0, 1, 0], [0, 0.1, 0], [0, 0.05, 0]], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
