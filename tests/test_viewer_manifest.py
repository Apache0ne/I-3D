"""CPU regression for viewer discovery of partial and complete model files."""
from __future__ import annotations

import json
from pathlib import Path
import struct
import tempfile
import unittest

from scripts.viewer_server import manifest


def write_minimal_glb(path: Path):
    document = {
        "asset": {"version": "2.0"},
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
    }
    payload = json.dumps(document).encode("utf-8")
    payload += b" " * ((4 - len(payload) % 4) % 4)
    path.write_bytes(
        struct.pack("<4sII", b"glTF", 2, len(payload) + 20)
        + struct.pack("<I4s", len(payload), b"JSON") + payload
    )


class ViewerManifestTests(unittest.TestCase):
    def test_partial_glb_does_not_appear_as_ready(self):
        with tempfile.TemporaryDirectory() as d:
            work = Path(d)
            geometry = work / "01_geometry.glb"
            geometry.write_text("v 0 0 0\n", encoding="utf-8")
            m = manifest(work)
            self.assertFalse(m["stages"][0]["available"])
            self.assertEqual(len(m["assets"]), 0)

            write_minimal_glb(geometry)
            m = manifest(work)
            self.assertTrue(m["stages"][0]["available"])
            self.assertEqual(m["stages"][0]["asset"]["filename"] if
                             "filename" in m["stages"][0]["asset"] else
                             m["stages"][0]["asset"]["name"], geometry.name)

            geometry.write_bytes(b"glTF")
            m = manifest(work)
            self.assertFalse(m["stages"][0]["available"])

    def test_only_generated_model_stage_is_ready(self):
        with tempfile.TemporaryDirectory() as d:
            work = Path(d)
            write_minimal_glb(work / "02_textured.glb")
            stages = manifest(work)["stages"]
            self.assertEqual([s["id"] for s in stages if s["available"]], ["paint"])


if __name__ == "__main__":
    unittest.main()
