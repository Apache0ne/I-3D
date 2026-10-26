# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from motionbricks.inference.cache import (
    TensorRTEngineCache,
    cache_key_for,
    make_cache_identity,
)


class TensorRTCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary_directory = TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.temporary_directory = Path(temporary_directory.name)

    def _key(
        self,
        *,
        onnx_sha256: str = "a" * 64,
        tensorrt_version: str = "10.8.0",
        device_fingerprint=None,
        **option_overrides,
    ):
        options = {"workspace_size_bytes": None, "builder_optimization_level": 3}
        options.update(option_overrides)
        identity = make_cache_identity(
            onnx_sha256=onnx_sha256,
            tensorrt_version=tensorrt_version,
            build_options=options,
            device_fingerprint=device_fingerprint
            or {"name": "GPU", "architecture": 89},
        )
        return cache_key_for(identity)

    def _metadata(self):
        metadata = make_cache_identity(
            onnx_sha256="a" * 64,
            tensorrt_version="10.8.0",
            build_options={"workspace_size_bytes": None, "builder_optimization_level": 3},
            device_fingerprint={"name": "GPU", "architecture": 89},
        )
        metadata["onnx_filename"] = "model.onnx"
        return metadata

    def test_key_is_deterministic_and_covers_build_environment(self) -> None:
        self.assertEqual(self._key(), self._key())
        self.assertNotEqual(self._key(), self._key(builder_optimization_level=4))
        self.assertNotEqual(self._key(), self._key(onnx_sha256="b" * 64))
        self.assertNotEqual(self._key(), self._key(tensorrt_version="10.9.0"))
        self.assertNotEqual(
            self._key(),
            self._key(
                device_fingerprint={
                    "name": "Another GPU",
                    "architecture": 90,
                }
            ),
        )

    def test_cache_round_trip_and_invalidation(self) -> None:
        cache = TensorRTEngineCache(self.temporary_directory)
        key = self._key()
        cache.store(
            "a model",
            key,
            b"serialized-plan",
            metadata=self._metadata(),
        )
        path = cache.path_for("a model", key)
        metadata_path = cache.metadata_path_for("a model", key)

        self.assertTrue(path.is_file())
        self.assertTrue(metadata_path.is_file())
        self.assertEqual(cache.load("a model", key), b"serialized-plan")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["onnx_sha256"], "a" * 64)
        self.assertEqual(metadata["cache_key"], key)
        self.assertEqual(metadata["plan_file"], path.name)
        cache.invalidate("a model", key)
        self.assertIsNone(cache.load("a model", key))
        self.assertFalse(metadata_path.exists())


if __name__ == "__main__":
    unittest.main()
