# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BVH loading contracts, including channel order and downstream consumers."""

import itertools
from pathlib import Path
import tempfile
import unittest

import numpy as np
import warp as wp

from motionbricks.animation.skeleton import Skeleton
from motionbricks.io.bvh import load_bvh, load_skeleton_from_bvh


def _write_motion(directory, order="ZXY", frames=2, root_angles=None):
    # Position channels are deliberately interleaved and out of XYZ order.
    # End Site offsets must not overwrite the preceding joint's offset.
    hierarchy = f"""HIERARCHY
ROOT actor:rig:Root
{{
  OFFSET 1 2 3
  CHANNELS 6 Zposition {order[0]}rotation Xposition {order[1]}rotation Yposition {order[2]}rotation
  JOINT actor:rig:Spine
  {{
    OFFSET 0 50 0
    CHANNELS 3 Yrotation Zrotation Xrotation
    JOINT actor:rig:Tip
    {{
      OFFSET 0 25 0
      CHANNELS 0
      End Site
      {{
        OFFSET 100 200 300
      }}
    }}
  }}
  JOINT actor:rig:Sibling
  {{
    OFFSET -10 0 0
    CHANNELS 3 Xposition Yposition Zposition
  }}
}}
MOTION
Frames: {frames}
Frame Time: 0.008333333333333333
"""
    samples = np.array([
        [300, 20, 100, -35, -200, 65, 15, 25, -40, -10, 5, 7],
        [-600, 55, -400, -10, 500, -15, -65, 10, 35, -12, 6, 8],
    ], dtype=np.float32)[:frames]
    if root_angles is not None:
        samples[:, [1, 3, 5]] = root_angles[:frames]
    path = directory / "motion.BVH"
    with path.open("w", encoding="utf-8") as target:
        target.write(hierarchy)
        np.savetxt(target, samples, fmt="%.6f")
    return path


class _BVHTestCase(unittest.TestCase):
    device = "cpu"

    @classmethod
    def setUpClass(cls) -> None:
        wp.init()
        if cls.device == "cuda:0" and not wp.is_cuda_available():
            raise unittest.SkipTest("CUDA is not available.")
        cls.previous_device = wp.get_device()
        wp.set_device(cls.device)

    @classmethod
    def tearDownClass(cls) -> None:
        wp.set_device(cls.previous_device)

    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)


class BVHTests(_BVHTestCase):
    def test_authored_channels_and_world_poses(self) -> None:
        angle_cases = {
            "ordinary": [(20, -35, 65), (55, -10, -15)],
            "zero": [(0, 0, 0), (0, 0, 0)],
            "gimbal": [(30, 90, -45), (-120, -90, 15)],
            "multi_turn": [(180, -180, 360), (-360, 720, -720)],
        }
        for axes, (case, root_angles) in itertools.product(
            itertools.permutations("XYZ"), angle_cases.items()
        ):
            order = "".join(axes)
            with self.subTest(order=order, angles=case):
                path = _write_motion(self.directory, order, root_angles=root_angles)
                animation = load_bvh(path)
                skeleton = animation.skeleton
                self.assertEqual(skeleton.joint_names, ["Root", "Spine", "Tip", "Sibling"])
                np.testing.assert_array_equal(skeleton.parent_indices.numpy(), [-1, 0, 1, 0])
                self.assertEqual(animation.num_frames, 2)
                self.assertEqual(animation.sample_rate, 120.0)
                self.assertEqual(animation.poses.dtype, wp.transform)
                self.assertEqual(animation.poses.shape, (2, 4))
                self.assertEqual(animation.poses.device, wp.get_device(self.device))
                reference = skeleton.reference_local_transforms.numpy()
                np.testing.assert_allclose(reference[:, :3], [
                    [.01, .02, .03], [0, .5, 0], [0, .25, 0], [-.1, 0, 0],
                ], atol=1e-7)
                np.testing.assert_array_equal(reference[:, 3:], np.tile([0, 0, 0, 1], (4, 1)))

                positions = [
                    [[1, -2, 3], [0, .5, 0], [0, .25, 0], [-.1, .05, .07]],
                    [[-4, 5, -6], [0, .5, 0], [0, .25, 0], [-.12, .06, .08]],
                ]
                spine_angles = [(15, 25, -40), (-65, 10, 35)]
                local = animation.poses.numpy()
                world_transforms = wp.empty_like(animation.poses)
                for frame in range(animation.num_frames):
                    animation.skeleton.compute_world_transforms(animation.poses[frame], world_transforms[frame])
                actual_world = world_transforms.numpy()

                for frame, angles in enumerate(root_angles):
                    # Compose the authored order with fixed-convention RPY calls,
                    # independently of the loader's axis-angle conversion.
                    angles_by_axis = dict(zip(order, angles))
                    x = wp.quat_rpy(wp.radians(float(angles_by_axis["X"])), 0.0, 0.0)
                    y = wp.quat_rpy(0.0, wp.radians(float(angles_by_axis["Y"])), 0.0)
                    z = wp.quat_rpy(0.0, 0.0, wp.radians(float(angles_by_axis["Z"])))
                    root_rotation = {
                        "XYZ": x * y * z,
                        "XZY": x * z * y,
                        "YXZ": y * x * z,
                        "YZX": y * z * x,
                        "ZXY": z * x * y,
                        "ZYX": z * y * x,
                    }[order]
                    sy, sz, sx = (wp.radians(float(angle)) for angle in spine_angles[frame])
                    spine_rotation = (
                        wp.quat_rpy(0.0, sy, 0.0)
                        * wp.quat_rpy(0.0, 0.0, sz)
                        * wp.quat_rpy(sx, 0.0, 0.0)
                    )
                    rotations = [
                        root_rotation, spine_rotation, wp.quat_identity(), wp.quat_identity()
                    ]
                    expected_local = [
                        wp.transform(position, rotation)
                        for position, rotation in zip(positions[frame], rotations, strict=True)
                    ]

                    # Compose the known hierarchy, including the separate root child.
                    root, spine, tip, sibling = expected_local
                    world_spine = wp.transform_multiply(root, spine)
                    expected_world = [
                        root,
                        world_spine,
                        wp.transform_multiply(world_spine, tip),
                        wp.transform_multiply(root, sibling),
                    ]
                    for actual_pose, expected_pose in (
                        (local[frame], expected_local),
                        (actual_world[frame], expected_world),
                    ):
                        for actual, expected in zip(actual_pose, expected_pose, strict=True):
                            rotation = wp.quat(actual[3:])
                            self.assertAlmostEqual(wp.length(rotation), 1.0, delta=1e-6)
                            np.testing.assert_allclose(actual[:3], expected.p, rtol=0, atol=1e-6)
                            np.testing.assert_allclose(
                                wp.quat_to_matrix(rotation),
                                wp.quat_to_matrix(expected.q),
                                atol=1e-6,
                            )

    def test_skeleton_only_load(self) -> None:
        path = _write_motion(self.directory)
        expected = load_bvh(path).skeleton
        hierarchy = path.read_text(encoding="utf-8").split("MOTION", 1)[0]
        path.write_text(
            hierarchy + "MOTION\nFrames: invalid\nFrame Time: invalid\nnot motion data\n",
            encoding="utf-8",
        )

        skeleton = load_skeleton_from_bvh(path)
        self.assertEqual(skeleton.joint_names, expected.joint_names)
        np.testing.assert_array_equal(
            skeleton.parent_indices.numpy(), expected.parent_indices.numpy()
        )
        np.testing.assert_array_equal(
            skeleton.reference_local_transforms.numpy(),
            expected.reference_local_transforms.numpy(),
        )

    def test_single_frame_motion(self) -> None:
        animation = load_bvh(_write_motion(self.directory, frames=1))
        self.assertEqual(animation.num_frames, 1)
        self.assertEqual(animation.poses.shape, (1, 4))

    def test_namespace_preservation_and_remapping(self) -> None:
        path = _write_motion(self.directory)
        stripped = load_bvh(path)
        stripped_skeleton = stripped.skeleton
        animation = load_bvh(path, remove_namespace=False)
        skeleton = animation.skeleton
        self.assertEqual(skeleton.joint_names, [
            "actor:rig:Root", "actor:rig:Spine", "actor:rig:Tip", "actor:rig:Sibling",
        ])
        np.testing.assert_array_equal(
            skeleton.parent_indices.numpy(), stripped_skeleton.parent_indices.numpy()
        )
        np.testing.assert_array_equal(
            skeleton.reference_local_transforms.numpy(),
            stripped_skeleton.reference_local_transforms.numpy(),
        )
        poses = animation.poses.numpy()
        np.testing.assert_array_equal(poses, stripped.poses.numpy())

        reference = skeleton.reference_local_transforms.numpy()[[0, 3, 1]]
        target = Skeleton(
            ["actor:rig:Root", "actor:rig:Extra", "actor:rig:Spine"], [-1, 0, 0], reference
        )
        remapped = load_bvh(path, target, remove_namespace=False)
        self.assertIs(remapped.skeleton, target)
        remapped_poses = remapped.poses.numpy()
        np.testing.assert_array_equal(remapped_poses[:, 0], poses[:, 0])
        np.testing.assert_array_equal(remapped_poses[:, 2], poses[:, 1])
        np.testing.assert_array_equal(remapped_poses[:, 1], np.tile(reference[1], (2, 1)))

    def test_channel_names_are_case_insensitive(self) -> None:
        for case in ("lower", "mixed"):
            with self.subTest(case=case):
                path = _write_motion(self.directory)
                expected = load_bvh(path)
                lines = path.read_text(encoding="utf-8").splitlines()
                for index, line in enumerate(lines):
                    tokens = line.split()
                    if tokens and tokens[0] == "CHANNELS":
                        tokens[2:] = [
                            channel.lower() if case == "lower" or channel_index % 2 == 0
                            else channel.upper()
                            for channel_index, channel in enumerate(tokens[2:])
                        ]
                        lines[index] = " ".join(tokens)
                path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                animation = load_bvh(path)
                np.testing.assert_array_equal(
                    animation.poses.numpy(), expected.poses.numpy()
                )

    def test_unsupported_channels_are_ignored(self) -> None:
        path = _write_motion(self.directory)
        expected = load_bvh(path)
        expected_skeleton = expected.skeleton
        text = path.read_text(encoding="utf-8").replace(
            "CHANNELS 6 Zposition Zrotation Xposition Xrotation Yposition Yrotation",
            "CHANNELS 9 Scale Zposition Zrotation Xposition Confidence Xrotation Yposition Yrotation Xscale",
        ).replace("CHANNELS 0", "CHANNELS 1 UserData")
        header, frame_time, motion = text.partition("Frame Time: 0.008333333333333333\n")
        rows = []
        for line in motion.splitlines():
            values = line.split()
            # Extra values surround the root channels and belong to the tip as well.
            rows.append(" ".join([
                "101", *values[:3], "202", *values[3:6], "303",
                *values[6:9], "404", *values[9:],
            ]))
        path.write_text(header + frame_time + "\n".join(rows) + "\n", encoding="utf-8")

        animation = load_bvh(path)
        skeleton = animation.skeleton
        self.assertEqual(skeleton.joint_names, expected_skeleton.joint_names)
        np.testing.assert_array_equal(
            skeleton.parent_indices.numpy(), expected_skeleton.parent_indices.numpy()
        )
        np.testing.assert_array_equal(
            skeleton.reference_local_transforms.numpy(),
            expected_skeleton.reference_local_transforms.numpy(),
        )
        np.testing.assert_array_equal(
            animation.poses.numpy(), expected.poses.numpy()
        )

    def test_utf8_joint_names(self) -> None:
        path = _write_motion(self.directory)
        expected = load_bvh(path)
        root_name = "Racine_\u00e9_\u9aa8"
        text = path.read_text(encoding="utf-8").replace("actor:rig:Root", f"actor:rig:{root_name}")
        path.write_text(text, encoding="utf-8")
        animation = load_bvh(path, remove_namespace=False)
        skeleton = animation.skeleton
        self.assertEqual(skeleton.joint_names, [
            f"actor:rig:{root_name}", "actor:rig:Spine", "actor:rig:Tip", "actor:rig:Sibling",
        ])
        np.testing.assert_array_equal(
            animation.poses.numpy(), expected.poses.numpy()
        )

    def test_mismatched_channel_count_is_rejected(self) -> None:
        path = _write_motion(self.directory)
        text = path.read_text(encoding="utf-8").replace(
            "CHANNELS 3 Xposition Yposition Zposition",
            "CHANNELS 2 Xposition Yposition Zposition",
        )
        path.write_text(text, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "BVH joint 'Sibling' declares 2 channels but lists 3"):
            load_bvh(path)

    def test_malformed_motion_is_rejected(self) -> None:
        for invalid in ("frames", "channels", "frame_time"):
            with self.subTest(invalid=invalid):
                path = _write_motion(self.directory)
                text = path.read_text(encoding="utf-8")
                if invalid == "frames":
                    text = text.replace("Frames: 2", "Frames: 3")
                    message = "Expected 3 frames and 12 channels"
                elif invalid == "channels":
                    text = text.replace(
                        "CHANNELS 3 Xposition Yposition Zposition", "CHANNELS 2 Xposition Yposition"
                    )
                    message = "Expected 2 frames and 11 channels"
                else:
                    text = text.replace("Frame Time: 0.008333333333333333\n", "")
                    message = "BVH frame time is missing"
                path.write_text(text, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    load_bvh(path)


class BVHCUDATests(_BVHTestCase):
    """Run the Warp pose conversion on CUDA."""

    device = "cuda:0"
    test_authored_channels_and_world_poses = BVHTests.test_authored_channels_and_world_poses


if __name__ == "__main__":
    unittest.main()
