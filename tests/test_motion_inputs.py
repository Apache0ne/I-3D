"""CPU-only regression for Blender-posed SOMA → MotionBricks token tensors."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from i3d.pose_math import soma_to_character_matrix
from scripts.generate_motion import load_inputs


class MotionInputTests(unittest.TestCase):
    def test_two_authored_poses_produce_30_joint_soma_transforms(self):
        # Deliberately synthetic. Checks invertible coordinate math and
        # 30-joint tensor contract; never claims the actual GPU ONNX ran.
        names = ["Hips"] + [f"Joint{i:02d}" for i in range(1, 30)]
        parents = [-1] + [0] * 29
        soma_bind = np.repeat(np.eye(4)[None, :, :], 30, axis=0)
        for i in range(30):
            soma_bind[i, :3, 3] = [0.01 * i, 0.7 + 0.005 * i, 0.02 * i]
        transform = soma_to_character_matrix(1.0, [0.1, 0.0, -0.1], "+X")
        blender_bind = [transform @ m for m in soma_bind]

        def frame(xshift):
            delta = np.eye(4)
            delta[0, 3] = xshift
            return {
                "world_pose": {
                    name: (delta @ blender_bind[i]).tolist()
                    for i, name in enumerate(names)
                }
            }

        poses = {
            "joint_names": names,
            "armature_world": np.eye(4).tolist(),
            "armature_rest": {
                name: blender_bind[i].tolist()
                for i, name in enumerate(names)
            },
            "poses": [frame(0.0), frame(0.25)],
        }
        reference = {
            "joint_names": names, "parents": parents,
            "rest_world_soma": soma_bind.tolist(),
        }
        bridge = {
            "template_scale": 1.0, "template_translation": [0.1, 0.0, -0.1],
            "forward": "+X",
        }
        with tempfile.TemporaryDirectory() as d:
            paths = [Path(d) / name for name in ("poses.json", "ref.json", "bridge.json")]
            for p, value in zip(paths, (poses, reference, bridge)):
                p.write_text(json.dumps(value), encoding="utf-8")
            tokens, output_names = load_inputs(*paths)
        self.assertEqual(output_names, names)
        self.assertEqual(tokens.shape, (2, 30, 7))
        self.assertTrue(np.isfinite(tokens).all())
        np.testing.assert_allclose(
            np.linalg.norm(tokens[:, :, 3:7], axis=-1), 1,
            atol=1e-5,
        )
        np.testing.assert_allclose(tokens[0, 0, :3], soma_bind[0, :3, 3], atol=1e-5)
        self.assertGreater(float(np.linalg.norm(tokens[1, 0, :3]-tokens[0, 0, :3])), .2)

    def test_reject_mismatched_joint_names(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            names = ["Hips"]+[f"Joint{i}" for i in range(1, 30)]
            (p/"poses.json").write_text(json.dumps({"joint_names": ["Wrong"]+names[1:]}))
            (p/"ref.json").write_text(json.dumps({"joint_names": names, "parents": [-1]*30}))
            (p/"bridge.json").write_text("{}")
            with self.assertRaises(ValueError):
                load_inputs(p/"poses.json", p/"ref.json", p/"bridge.json")


if __name__ == "__main__":
    unittest.main()
