"""CPU-only verification of browser RMBG PNG upload and stage contracts."""
from __future__ import annotations

import hashlib
import http.client
import json
from http.server import ThreadingHTTPServer
from pathlib import Path
import struct
import tempfile
import threading
import unittest
import zlib

from scripts.viewer_server import (
    PNG_SIGNATURE, validate_rmbg_png, save_rmbg_png, handler_for, manifest
)


def chunk(kind, body):
    crc = zlib.crc32(kind + body) & 0xFFFFFFFF
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", crc)


def png(width=2, height=2, rgba=(100, 200, 40, 255)):
    ihdr=struct.pack(">IIBBBBB",width,height,8,6,0,0,0)
    row=bytes((0,)) + bytes(rgba) * width
    # PNG rows are stored as raw bytes prefixed by filter 0.
    return (PNG_SIGNATURE + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT",zlib.compress(row*height))
            + chunk(b"IEND",b""))


class RMBGPNGTests(unittest.TestCase):
    def test_accept_valid_png_reject_corrupt_signature_crc_and_too_large_dimensions(self):
        image=png(3,2)
        self.assertEqual(validate_rmbg_png(image),(3,2))
        with self.assertRaisesRegex(ValueError,"signature"):
            validate_rmbg_png(b"bad" + image[3:])
        corrupt=bytearray(image);corrupt[18]^=7
        with self.assertRaisesRegex(ValueError,"checksum"):
            validate_rmbg_png(bytes(corrupt))
        with self.assertRaisesRegex(ValueError,"truncated"):
            validate_rmbg_png(b"")

    def test_metadata_sha256_and_stale_invalidation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            a,b=png(2,2),png(3,3)
            save_rmbg_png(root,"source",a,"patched_fp16")
            save_rmbg_png(root,"mask",a,"patched_fp16")
            result=save_rmbg_png(root,"cutout",a,"patched_fp16")
            self.assertTrue(result["ready"])
            meta=json.loads((root/"00_rmbg_metadata.json").read_text())
            self.assertEqual(meta["model"],"patched_fp16")
            self.assertEqual(meta["sha256"]["source"],hashlib.sha256(a).hexdigest())
            self.assertTrue(manifest(root)["background_removal_ready"])
            self.assertEqual(manifest(root)["original_image"]["name"],"00_source_input.png")

            save_rmbg_png(root,"source",b,"kn4666_fp16")
            self.assertFalse((root/"00_rmbg_metadata.json").exists())
            self.assertFalse((root/"00_background_removed.png").exists())
            self.assertFalse((root/"00_alpha_mask.png").exists())
            save_rmbg_png(root,"mask",b,"kn4666_fp16")
            with self.assertRaisesRegex(ValueError,"dimensions"):
                save_rmbg_png(root,"cutout",a,"kn4666_fp16")
            self.assertFalse((root/"00_background_removed.png").exists())

    def test_bad_model_and_bad_kind_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,"Unknown"):
                save_rmbg_png(d,"source",png(),"untrusted_model")
            with self.assertRaisesRegex(ValueError,"Unknown"):
                save_rmbg_png(d,"../escape",png(),"patched_fp16")


class UploadHTTPTests(unittest.TestCase):
    def test_real_http_png_post_route_and_action_guard(self):
        with tempfile.TemporaryDirectory() as d:
            server=ThreadingHTTPServer(("127.0.0.1",0),handler_for(Path(d)))
            thread=threading.Thread(target=server.serve_forever,daemon=True)
            thread.start()
            try:
                conn=http.client.HTTPConnection("127.0.0.1",server.server_port,timeout=5)
                content=png()
                headers={"Content-Type":"image/png","X-I3D-Action":"rmbg",
                         "X-I3D-Model":"patched_fp16"}
                conn.request("POST","/api/rmbg?kind=source",content,headers)
                response=conn.getresponse()
                body=json.loads(response.read())
                self.assertEqual(response.status,200)
                self.assertTrue(body["ok"])
                conn.request("POST","/api/rmbg?kind=mask",content,{"Content-Type":"image/png"})
                response=conn.getresponse()
                response.read()
                self.assertEqual(response.status,403)
                conn.close()
            finally:
                server.shutdown();server.server_close();thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
