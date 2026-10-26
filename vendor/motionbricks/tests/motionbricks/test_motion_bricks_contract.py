# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from motionbricks.inference.errors import TensorContractError
from motionbricks.inference.types import TensorMode, TensorSpec
from motionbricks.model.config import MotionBricksInferenceConfig
from motionbricks.model.contract import (
    MODEL_CONTEXT_AND_TARGET_POSES,
    MODEL_MAX_TOKEN_COUNT,
    MODEL_MAX_VALID_FRAMES,
    MODEL_MIN_TOKEN_COUNT,
    MODEL_POSES_PER_TOKEN,
    load_model_manifest,
    valid_frame_count,
)
from motionbricks.model.soma.contract import (
    SOMA_N1_CONTRACT,
    SOMA_N1_FP16_ONNX_PATH,
    SOMA_N1_FP32_ONNX_PATH,
    SOMA_N1_MANIFEST_PATH,
)
from motionbricks.model.unitree_g1.contract import UNITREE_G1_N1_CONTRACT


class MotionBricksContractTests(unittest.TestCase):
    def test_tensor_layouts_follow_the_joint_counts(self) -> None:
        predicted_frames = MODEL_MAX_TOKEN_COUNT * MODEL_POSES_PER_TOKEN
        shapes = {spec.name: spec.shape for spec in SOMA_N1_CONTRACT.tensor_specs}
        joints = len(SOMA_N1_CONTRACT.joint_names)
        self.assertEqual(shapes["local_poses"], (1, MODEL_CONTEXT_AND_TARGET_POSES, 3 * (joints - 1) + 6 * joints))
        self.assertEqual(
            shapes["pred_global_motions"],
            (1, predicted_frames, 5 + 3 * (joints - 1) + 6 * joints + 3 * joints + 4),
        )

        shapes = {spec.name: spec.shape for spec in UNITREE_G1_N1_CONTRACT.tensor_specs}
        coordinates = 7 + len(UNITREE_G1_N1_CONTRACT.joint_names)
        self.assertEqual(shapes["context_qpos"], (1, MODEL_POSES_PER_TOKEN, coordinates))
        self.assertEqual(shapes["target_qpos"], (1, MODEL_POSES_PER_TOKEN, coordinates))
        self.assertEqual(shapes["pred_qpos"], (1, predicted_frames, coordinates))
        self.assertEqual(shapes["pred_foot_contacts"], (1, predicted_frames, 4))

    def test_contract_reports_mismatched_tensor(self) -> None:
        specs = {spec.name: spec for spec in SOMA_N1_CONTRACT.tensor_specs}
        specs["pred_num_tokens"] = TensorSpec(
            "pred_num_tokens",
            TensorMode.OUTPUT,
            (2,),
            np.dtype(np.int32),
        )
        with self.assertRaisesRegex(TensorContractError, "expected shape"):
            SOMA_N1_CONTRACT.validate_tensor_specs(specs)

    def test_token_range_is_enforced(self) -> None:
        MotionBricksInferenceConfig(allowed_token_counts=(MODEL_MIN_TOKEN_COUNT, MODEL_MAX_TOKEN_COUNT))
        for invalid in (MODEL_MIN_TOKEN_COUNT - 1, MODEL_MAX_TOKEN_COUNT + 1):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    MotionBricksInferenceConfig(allowed_token_counts=(invalid,))
                with self.assertRaises(ValueError):
                    valid_frame_count(invalid)

    def test_valid_frames_drop_three_boundary_frames_at_each_end(self) -> None:
        self.assertEqual(valid_frame_count(MODEL_MIN_TOKEN_COUNT), 18)
        self.assertEqual(MODEL_MAX_VALID_FRAMES, 58)

    def test_manifest_timing_must_match_the_model_constants(self) -> None:
        manifest = json.loads(SOMA_N1_MANIFEST_PATH.read_text(encoding="utf-8"))
        for field, value in (("min_pred_tokens", 8), ("max_pred_tokens", 14), ("sample_rate", 60.0)):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "manifest.json"
                path.write_text(json.dumps(dict(manifest, **{field: value})), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, f"{field} must be"):
                    load_model_manifest(path)

    def test_manifest_names_both_precisions_next_to_it(self) -> None:
        self.assertEqual(SOMA_N1_FP32_ONNX_PATH, SOMA_N1_MANIFEST_PATH.with_name("mmm_v6_soma_fp32.onnx"))
        self.assertEqual(SOMA_N1_FP16_ONNX_PATH, SOMA_N1_MANIFEST_PATH.with_name("mmm_v6_soma_fp16.onnx"))
        self.assertTrue(SOMA_N1_FP16_ONNX_PATH.is_file())


if __name__ == "__main__":
    unittest.main()
