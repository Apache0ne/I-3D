# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np
import warp as wp

from motionbricks.io.csv import load_csv, save_csv


_ROOT_FIELDS = ["Frame", "root_translateX", "root_translateY", "root_translateZ",
                "root_rotateX", "root_rotateY", "root_rotateZ"]


class CSVTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)

    def write_clip(self, *, single=False):
        path = self.directory / "clip.csv"
        rows = [[10, 100, -250, 80, 20, -35, 65, 90, -45],
                [11, 200, -200, 90, 0, 0, 90, 60, -30]]
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(_ROOT_FIELDS + ["elbow", "knee"])
            writer.writerows(rows[:1] if single else rows)
        return path

    def test_converts_units_and_root_euler_in_file_order(self):
        clip = load_csv(self.write_clip(), sample_rate=60, device="cpu")
        values = clip.poses.numpy()
        self.assertEqual(clip.joint_names, ("elbow", "knee"))
        self.assertEqual(clip.sample_rate, 60)
        np.testing.assert_allclose(values[:, :3], [[1, -2.5, 0.8], [2, -2, 0.9]], atol=1e-6)
        np.testing.assert_allclose(values[:, 7:], np.deg2rad([[90, -45], [60, -30]]), atol=1e-6)
        expected = np.array(wp.quat_rpy(*np.deg2rad([20, -35, 65])))
        self.assertAlmostEqual(abs(float(values[0, 3:7] @ expected)), 1.0, places=6)

    def test_single_frame(self):
        clip = load_csv(self.write_clip(single=True), device="cpu")
        self.assertEqual(clip.num_frames, 1)
        self.assertEqual(clip.duration, 0)

    def test_rejects_non_finite_coordinates(self):
        path = self.directory / "invalid.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(_ROOT_FIELDS + ["hinge"])
            writer.writerow([0.0] * 7 + [float("nan")])
        with self.assertRaisesRegex(ValueError, "coordinates must be finite"):
            load_csv(path, device="cpu")

    def test_existing_writer_round_trip_preserves_coordinate_order(self):
        original = load_csv(self.write_clip(), device="cpu")
        path = self.directory / "saved.csv"
        save_csv(path, original)
        loaded = load_csv(path, device="cpu")
        self.assertEqual(loaded.joint_names, original.joint_names)
        a, b = original.poses.numpy(), loaded.poses.numpy()
        np.testing.assert_allclose(a[:, :3], b[:, :3], atol=1e-6)
        np.testing.assert_allclose(a[:, 7:], b[:, 7:], atol=1e-6)
        np.testing.assert_allclose(np.abs(np.sum(a[:, 3:7] * b[:, 3:7], axis=1)), 1, atol=1e-6)

    def test_writer_exports_only_valid_frames(self):
        clip = load_csv(self.write_clip(), device="cpu")
        clip.num_frames = 1
        clip.poses[1:].fill_(float("nan"))
        path = self.directory / "valid.csv"
        save_csv(path, clip)
        loaded = load_csv(path, device="cpu")
        self.assertEqual(loaded.num_frames, 1)
        np.testing.assert_allclose(loaded.poses.numpy(), clip.poses.numpy()[:1], atol=1e-6)
        clip.num_frames = 0
        with self.assertRaisesRegex(ValueError, "at least one frame in use"):
            save_csv(path, clip)


if __name__ == "__main__":
    unittest.main()
