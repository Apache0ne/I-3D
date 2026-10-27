"""CPU-only regression tests for the binary GLB / Hunyuan Paint contracts."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

from i3d.artifacts import (
    cache_valid,
    obj_intermediate_for_glb,
    required_binary_asset,
    sha256,
    validate_glb,
    write_stamp,
)

ROOT = Path(__file__).resolve().parents[1]


def minimal_glb(path: Path) -> None:
    """Small valid GLB *container*. Deliberately not a renderable GPU model."""
    payload = json.dumps({
        "asset": {"version": "2.0"},
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
    }, separators=(",", ":")).encode()
    payload += b" " * ((4 - len(payload) % 4) % 4)
    blob = struct.pack("<4sII", b"glTF", 2, 20 + len(payload))
    blob += struct.pack("<I4s", len(payload), b"JSON") + payload
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)


class ArtifactTests(unittest.TestCase):
    def test_glb_vs_obj_and_length_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            asset = Path(d) / "one.glb"
            minimal_glb(asset)
            self.assertEqual(validate_glb(asset), asset)
            asset.write_text("v 0 0 0\nf 1 2 3\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_glb(asset)
            minimal_glb(asset)
            binary = bytearray(asset.read_bytes())
            binary[8:12] = struct.pack("<I", len(binary) + 4)
            asset.write_bytes(binary)
            with self.assertRaises(ValueError):
                validate_glb(asset)

    def test_obj_stage_and_paint_output_suffix(self):
        with tempfile.TemporaryDirectory() as d:
            final = Path(d) / "painted.glb"
            self.assertEqual(obj_intermediate_for_glb(final), final.with_suffix(".obj"))
            with self.assertRaises(ValueError):
                obj_intermediate_for_glb(final.with_suffix(".ply"))

    def test_cache_checks_parameters_and_file_integrity(self):
        with tempfile.TemporaryDirectory() as d:
            asset = Path(d) / "painted.glb"
            minimal_glb(asset)
            p = {"source_sha256": "abc", "resolution": 512}
            self.assertFalse(cache_valid(asset, p))
            write_stamp(asset, p)
            self.assertTrue(cache_valid(asset, p))
            self.assertFalse(cache_valid(asset, dict(p, resolution=1024)))
            with asset.open("ab") as stream:
                stream.write(b"JUNK")
            self.assertFalse(cache_valid(asset, p))

    def test_reject_git_lfs_pointer(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "checkpoint.usd"
            p.write_text("version https://git-lfs.github.com/spec/v1\n" + "x" * 2000)
            with self.assertRaises(ValueError):
                required_binary_asset(p)


class PaintAdapterContractTests(unittest.TestCase):
    """Mock only the HEAVY model; exercise our real adapter as a subprocess."""

    def run_adapter(self, *, write_glb: bool):
        work = tempfile.TemporaryDirectory()
        root = Path(work.name)
        (root / "scripts").mkdir()
        (root / "i3d").mkdir()
        (root / "vendor" / "Hunyuan3D-2.1" / "hy3dpaint" / "ckpt").mkdir(parents=True)
        (root / "vendor" / "Hunyuan3D-2.1" / "hy3dpaint" / "cfgs").mkdir(parents=True)
        shutil.copy2(ROOT / "scripts" / "paint_hunyuan.py",
                     root / "scripts" / "paint_hunyuan.py")
        shutil.copy2(ROOT / "i3d" / "artifacts.py",
                     root / "i3d" / "artifacts.py")
        (root / "i3d" / "__init__.py").write_text("")
        vendor = root / "vendor" / "Hunyuan3D-2.1" / "hy3dpaint"
        (vendor / "ckpt" / "RealESRGAN_x4plus.pth").write_bytes(b"fake")
        (vendor / "cfgs" / "hunyuan-paint-pbr.yaml").write_text("mock")
        (vendor / "textureGenPipeline.py").write_text(
            "from pathlib import Path\n"
            "class Hunyuan3DPaintConfig:\n"
            "    def __init__(self, views, resolution): pass\n"
            "class Hunyuan3DPaintPipeline:\n"
            "    def __init__(self, conf): pass\n"
            "    def __call__(self, *, mesh_path, image_path, output_mesh_path, save_glb):\n"
            "        assert output_mesh_path.endswith('.obj'), output_mesh_path\n"
            "        assert save_glb is True\n"
            "        obj = Path(output_mesh_path)\n"
            "        obj.write_text('v 0 0 0\\n')\n"
            f"        if {write_glb!r}:\n"
            "            obj.with_suffix('.glb').write_bytes(Path(mesh_path).read_bytes())\n"
            "        return str(obj)\n"
        )
        source = root / "person.glb"
        minimal_glb(source)
        image = root / "source.png"
        image.write_bytes(b"synthetic-image")
        output = root / "textured.glb"
        process = subprocess.run(
            [sys.executable, str(root / "scripts" / "paint_hunyuan.py"),
             "--mesh", str(source), "--image", str(image), "--output", str(output)],
            text=True, capture_output=True,
        )
        return work, process, output

    def test_successful_mock_generates_genuine_glb(self):
        work, process, output = self.run_adapter(write_glb=True)
        try:
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            self.assertEqual(validate_glb(output), output)
            self.assertTrue(output.with_name("textured.i3d-working.obj").is_file())
        finally:
            work.cleanup()

    def test_missing_upstream_conversion_is_error_not_success(self):
        work, process, output = self.run_adapter(write_glb=False)
        try:
            self.assertNotEqual(process.returncode, 0)
            self.assertIn("Missing GLB", process.stderr)
            self.assertFalse(output.exists())
        finally:
            work.cleanup()


if __name__ == "__main__":
    unittest.main()
