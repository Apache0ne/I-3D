# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""USD loading reports a stage that cannot open, or missing skinning data, as ValueError."""

import tempfile
import unittest
from pathlib import Path

from pxr import Gf, Usd, UsdGeom, UsdSkel

from motionbricks.io.usd import load_skeletal_mesh_with_skeleton_from_usd, load_skeleton_from_usd


def _write_character(path, *, bind_transforms=True, joint_indices=True):
    """Write a one joint character with a one triangle skinned mesh."""
    stage = Usd.Stage.CreateNew(str(path))
    UsdSkel.Root.Define(stage, "/Root")
    skeleton = UsdSkel.Skeleton.Define(stage, "/Root/Skeleton")
    skeleton.CreateJointsAttr(["Hips"])
    skeleton.CreateRestTransformsAttr([Gf.Matrix4d(1.0)])
    if bind_transforms:
        skeleton.CreateBindTransformsAttr([Gf.Matrix4d(1.0)])
    mesh = UsdGeom.Mesh.Define(stage, "/Root/Mesh")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    binding = UsdSkel.BindingAPI.Apply(mesh.GetPrim())
    binding.CreateSkeletonRel().SetTargets([skeleton.GetPath()])
    if joint_indices:
        binding.CreateJointIndicesPrimvar(False, 1).Set([0, 0, 0])
    binding.CreateJointWeightsPrimvar(False, 1).Set([1, 1, 1])
    stage.Save()


class UsdLoadingTests(unittest.TestCase):
    def test_a_missing_file_raises_value_error_from_both_loaders(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.usda"
            with self.assertRaisesRegex(ValueError, "Could not open USD stage"):
                load_skeleton_from_usd(missing, "/Root/Skeleton")
            with self.assertRaisesRegex(ValueError, "Could not open USD stage"):
                load_skeletal_mesh_with_skeleton_from_usd(missing, "/Root/Mesh", "/Root/Skeleton")

    def test_missing_skinning_data_raises_value_error_with_the_reason(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            _write_character(folder / "complete.usda")
            mesh = load_skeletal_mesh_with_skeleton_from_usd(folder / "complete.usda", "/Root/Mesh", "/Root/Skeleton")
            self.assertEqual(len(mesh.skinned_meshes), 1)
            with self.assertRaisesRegex(ValueError, "Mesh prim not found"):
                load_skeletal_mesh_with_skeleton_from_usd(folder / "complete.usda", "/Root/Missing", "/Root/Skeleton")
            for name, options, message in (
                ("no_binds", {"bind_transforms": False}, "needs one bind transform per joint"),
                ("no_indices", {"joint_indices": False}, "needs jointIndices and jointWeights"),
            ):
                with self.subTest(case=name):
                    path = folder / f"{name}.usda"
                    _write_character(path, **options)
                    with self.assertRaisesRegex(ValueError, message):
                        load_skeletal_mesh_with_skeleton_from_usd(path, "/Root/Mesh", "/Root/Skeleton")


if __name__ == "__main__":
    unittest.main()
