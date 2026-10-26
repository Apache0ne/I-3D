# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source clips stay bound to a demo's rig and device."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from app.runtime.common.clip_registry import ClipLoadError, ClipRegistry
from motionbricks.animation.skeleton import Skeleton
from motionbricks.io.csv import CSV_SAMPLE_RATE, load_csv, save_csv
from motionbricks.model.contract import ModelRig
from motionbricks.robotics.dof_animation_buffer import DOFAnimationBuffer


class ClipRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()

    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)

    def test_rejects_loading_configuration_for_another_rig(self):
        with wp.ScopedDevice("cpu"):
            skeleton = Skeleton(["Hips"], [-1], [wp.transform_identity()])
        soma = {"skeleton": skeleton}
        g1 = {"joint_names": ("hinge",)}
        for rig, configuration in (
            (ModelRig.SOMA, g1),
            (ModelRig.UNITREE_G1, soma),
            (ModelRig.SOMA, {**soma, **g1}),
            (ModelRig.UNITREE_G1, {**g1, **soma}),
        ):
            with self.subTest(rig=rig, configuration=configuration):
                with self.assertRaises(ValueError):
                    ClipRegistry(rig, **configuration)

    def test_bvh_uses_each_registrys_skeleton_and_device(self):
        path = self.root / "source.bvh"
        path.write_text(
            "HIERARCHY\nROOT Hips\n{\n  OFFSET 0 0 0\n"
            "  CHANNELS 6 Xposition Yposition Zposition Xrotation Yrotation Zrotation\n"
            "}\nMOTION\nFrames: 2\nFrame Time: 0.03333333333333333\n"
            "100 200 300 0 0 0\n400 500 600 0 0 0\n",
            encoding="utf-8",
        )
        previous = None
        for device in wp.get_devices():
            for extra_joint_height in (1.0, 2.0):
                with self.subTest(device=device, height=extra_joint_height):
                    with wp.ScopedDevice(device):
                        skeleton = Skeleton(["Hips", "Extra"], [-1, 0], [
                            wp.transform_identity(),
                            wp.transform(wp.vec3(0, extra_joint_height, 0), wp.quat_identity()),
                        ])
                    registry = ClipRegistry(ModelRig.SOMA, skeleton=skeleton)
                    # Loading must follow the registry, even with a different current device.
                    with wp.ScopedDevice("cpu"):
                        clip = registry.load(path)
                        self.assertIs(registry.load(path.parent / "unused" / ".." / path.name), clip)
                    self.assertIs(clip.animation.skeleton, skeleton)
                    self.assertEqual(clip.animation.poses.device, device)
                    poses = clip.animation.poses.numpy()
                    np.testing.assert_allclose(poses[:, 0, :3], [[1, 2, 3], [4, 5, 6]])
                    np.testing.assert_allclose(poses[:, 1, 1], extra_joint_height)
                    self.assertIsNot(clip, previous)
                    previous = clip

    def test_csv_preserves_joint_order_at_the_csv_sample_rate(self):
        path = self.root / "source.csv"
        with wp.ScopedDevice("cpu"):
            data = np.zeros((3, 9), dtype=np.float32)
            data[:, 6] = 1.0
            data[:, 7:] = [[0.1, -0.2], [0.2, -0.3], [0.3, -0.4]]
            source = DOFAnimationBuffer(("hip", "knee"), 120.0, wp.array(data, dtype=wp.float32))
            save_csv(str(path), source)
        registry = ClipRegistry(ModelRig.UNITREE_G1, joint_names=("hip", "knee"), device="cpu")
        with patch("app.runtime.common.clip_registry.load_csv", wraps=load_csv) as read_csv:
            clip = registry.load(path)
            self.assertIs(clip, registry.load(path))
            self.assertEqual(read_csv.call_count, 1)
        self.assertEqual(clip.animation.sample_rate, CSV_SAMPLE_RATE)
        np.testing.assert_allclose(clip.animation.poses.numpy(), data, atol=1e-7)
        wrong_order = ClipRegistry(ModelRig.UNITREE_G1, joint_names=("knee", "hip"), device="cpu")
        with self.assertRaisesRegex(ClipLoadError, "joint order"):
            wrong_order.load(path)
        with self.assertRaisesRegex(ClipLoadError, r"loads \.csv clips"):
            registry.load(self.root / "source.bvh")

    def test_failed_load_does_not_cache_a_missing_or_incompatible_clip(self):
        path = self.root / "source.csv"
        registry = ClipRegistry(ModelRig.UNITREE_G1, joint_names=("hinge",), device="cpu")
        with self.assertRaisesRegex(ClipLoadError, "Could not find clip"):
            registry.load(path)
        header = "Frame,root_translateX,root_translateY,root_translateZ,root_rotateX,root_rotateY,root_rotateZ,"
        path.write_text(header + "wrong_joint\n0,0,0,0,0,0,0,30\n", encoding="utf-8")
        with self.assertRaisesRegex(ClipLoadError, "joint order"):
            registry.load(path)
        path.write_text(header + "hinge\n0,0,0,0,0,0,0,30\n", encoding="utf-8")
        clip = registry.load(path)
        self.assertEqual(clip.animation.joint_names, ("hinge",))


if __name__ == "__main__":
    unittest.main()
