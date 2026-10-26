# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check checkpoint validation, output preservation, and the tensors of exported models."""

import hashlib
from itertools import product
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import onnx
from onnx import TensorProto, helper

from motionbricks_onnx import checkpoints, export

_ONNX_ASSETS = Path(__file__).resolve().parents[3] / "motionbricks" / "assets" / "onnx"
_MODEL_MANIFESTS = {
    "soma": _ONNX_ASSETS / "soma" / "soma_n1.json",
    "g1": _ONNX_ASSETS / "unitree_g1" / "unitree_g1_n1.json",
}


def g1_graph():
    """A small, valid batched graph with the documented G1 interface."""
    inputs = [
        helper.make_tensor_value_info(name, dtype, ["N", *shape])
        for name, dtype, shape in (
            ("context_qpos", TensorProto.FLOAT, (4, 36)),
            ("target_qpos", TensorProto.FLOAT, (4, 36)),
            ("has_global_root_values", TensorProto.INT32, (8,)),
            ("has_local_root_values", TensorProto.INT32, (8,)),
            ("has_local_poses", TensorProto.INT32, (8,)),
            ("allowed_pred_num_tokens", TensorProto.INT32, (11,)),
        )
    ]
    outputs = [
        helper.make_tensor_value_info("pred_qpos", TensorProto.FLOAT, ["N", 64, 36]),
        helper.make_tensor_value_info("pred_num_tokens", TensorProto.INT32, ["N"]),
        helper.make_tensor_value_info(
            "pred_foot_contacts", TensorProto.FLOAT, ["N", 64, 4]
        ),
    ]
    nodes = [
        helper.make_node("Shape", ["context_qpos"], ["context_shape"]),
        helper.make_node("Gather", ["context_shape", "batch_index"], ["batch_shape"]),
        helper.make_node(
            "Concat", ["batch_shape", "pose_shape"], ["output_shape"], axis=0
        ),
        helper.make_node(
            "ConstantOfShape",
            ["output_shape"],
            ["pred_qpos"],
            value=helper.make_tensor("zero", TensorProto.FLOAT, [1], [0]),
        ),
        helper.make_node(
            "ConstantOfShape",
            ["batch_shape"],
            ["pred_num_tokens"],
            value=helper.make_tensor("zero", TensorProto.INT32, [1], [0]),
        ),
        helper.make_node(
            "Concat", ["batch_shape", "contact_shape"], ["contact_output_shape"], axis=0
        ),
        helper.make_node(
            "ConstantOfShape",
            ["contact_output_shape"],
            ["pred_foot_contacts"],
            value=helper.make_tensor("zero", TensorProto.FLOAT, [1], [0]),
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "test_g1",
        inputs,
        outputs,
        initializer=[
            helper.make_tensor("batch_index", TensorProto.INT64, [1], [0]),
            helper.make_tensor("pose_shape", TensorProto.INT64, [2], [64, 36]),
            helper.make_tensor("contact_shape", TensorProto.INT64, [2], [64, 4]),
        ],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


class ReleaseExportTests(unittest.TestCase):
    def test_missing_or_corrupt_checkpoints_reject_both_models_before_loading(self):
        original = b"original checkpoint"
        expected = dict(
            bytes=len(original), sha256=hashlib.sha256(original).hexdigest()
        )
        release = dict(root_directory="out", files=[dict(expected, path="model.safetensors")])
        # A corrupt file sits in the downloaded folder itself, or in its root_directory.
        # A missing file is a folder without the checkpoint, such as a wrong path.
        for kind, layout in product(("soma", "g1"), ("direct", "nested", "missing")):
            with (
                self.subTest(model=kind, layout=layout),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                if layout != "missing":
                    files = root if layout == "direct" else root / "out"
                    files.mkdir(exist_ok=True)
                    # Corrupt one byte while keeping the expected file size.
                    (files / "model.safetensors").write_bytes(b"X" + original[1:])
                output = root / "existing.onnx"
                output.write_bytes(b"keep this file")
                with (
                    patch.dict(export.CHECKPOINTS, {kind: release}),
                    patch.object(export, "check_environment"),
                    patch.object(
                        checkpoints,
                        "load_file",
                        side_effect=AssertionError("must not deserialize"),
                    ),
                    patch.object(
                        export,
                        "export_fp32",
                        side_effect=AssertionError("must not export"),
                    ),
                ):
                    message = "Missing checkpoint file" if layout == "missing" else "Checksum mismatch"
                    with self.assertRaisesRegex(ValueError, message):
                        export.export_release(f"{kind}_fp32", root, output, force=True)
                self.assertEqual(output.read_bytes(), b"keep this file")

    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing.onnx"
            output.write_bytes(b"keep this file")
            with self.assertRaisesRegex(ValueError, "already exists"):
                export.export_release("g1_fp32", Path("missing"), output)
            self.assertEqual(output.read_bytes(), b"keep this file")

    def test_force_preserves_output_when_export_is_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing.onnx"
            output.write_bytes(b"keep this file")

            def invalid_export(kind, model, path):
                path.write_bytes(b"not an ONNX graph")

            with (
                patch.object(export, "check_environment"),
                patch.object(export, "checkpoint_root", return_value=Path(directory)),
                patch.object(export, "load_sources", return_value={}),
                patch(
                    "motionbricks_onnx.models.g1.build_g1",
                    return_value=export.torch.nn.Identity(),
                ),
                patch.object(export, "export_fp32", side_effect=invalid_export),
            ):
                with self.assertRaisesRegex(ValueError, "Invalid ONNX model"):
                    export.export_release(
                        "g1_fp32", Path(directory), output, force=True
                    )
            self.assertEqual(output.read_bytes(), b"keep this file")
            self.assertEqual(list(Path(directory).iterdir()), [output])

    def test_valid_export_can_replace_different_onnx_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "model.onnx"
            graph = g1_graph()
            onnx.save(graph, output)
            previous = output.read_bytes()
            graph.producer_name = "another exporter serialization"
            with (
                patch.object(export, "check_environment"),
                patch.object(export, "checkpoint_root", return_value=Path(directory)),
                patch.object(export, "load_sources", return_value={}),
                patch(
                    "motionbricks_onnx.models.g1.build_g1",
                    return_value=export.torch.nn.Identity(),
                ),
                patch.object(
                    export,
                    "export_fp32",
                    side_effect=lambda kind, model, path: onnx.save(graph, path),
                ),
            ):
                export.export_release("g1_fp32", Path(directory), output, force=True)
            self.assertNotEqual(output.read_bytes(), previous)
            self.assertEqual(onnx.load(output).producer_name, graph.producer_name)
            self.assertEqual(list(Path(directory).iterdir()), [output])

    def test_incompatible_tensor_interfaces_are_rejected(self):
        for mismatch in (
            "name",
            "dtype",
            "rank",
            "shape",
            "fixed_batch",
            "different_batch",
        ):
            with (
                self.subTest(mismatch=mismatch),
                tempfile.TemporaryDirectory() as directory,
            ):
                graph = g1_graph()
                value = graph.graph.input[1]
                tensor = value.type.tensor_type
                if mismatch == "name":
                    value.name = "unexpected_target"
                elif mismatch == "dtype":
                    tensor.elem_type = TensorProto.FLOAT16
                elif mismatch == "rank":
                    del tensor.shape.dim[-1]
                elif mismatch == "shape":
                    tensor.shape.dim[-1].dim_value = 35
                elif mismatch == "fixed_batch":
                    tensor.shape.dim[0].dim_value = 1
                else:
                    tensor.shape.dim[0].dim_param = "different_batch"
                path = Path(directory) / "incompatible.onnx"
                onnx.save(graph, path)
                with self.assertRaisesRegex(ValueError, "ONNX|symbolic batch"):
                    export.validate_output(path, "g1")

    def test_bundled_models_have_the_tensors_the_runtime_expects(self):
        # The bundled models are what the exporter produces from the released checkpoints.
        # The runtime checks each TensorRT engine against the tensor specs in its manifest.
        for kind, manifest_path in _MODEL_MANIFESTS.items():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            filenames = [manifest["onnx_filename"]] + [
                variant["onnx_filename"] for variant in manifest["precision_variants"].values()
            ]
            for filename in filenames:
                path = manifest_path.parent / filename
                with self.subTest(model=filename):
                    if path.stat().st_size < 1024:
                        self.skipTest(f"Git LFS has not fetched {filename}.")
                    export.validate_output(path, kind)
                    graph = onnx.load(path).graph
                    tensors = {value.name: ("input", value) for value in graph.input}
                    tensors |= {value.name: ("output", value) for value in graph.output}
                    self.assertEqual(set(tensors), {spec["name"] for spec in manifest["tensor_specs"]})
                    for spec in manifest["tensor_specs"]:
                        mode, value = tensors[spec["name"]]
                        tensor = value.type.tensor_type
                        self.assertEqual(mode, spec["mode"], spec["name"])
                        self.assertEqual(
                            helper.tensor_dtype_to_np_dtype(tensor.elem_type), np.dtype(spec["dtype"]), spec["name"]
                        )
                        self.assertEqual(len(tensor.shape.dim), len(spec["shape"]), spec["name"])
                        # The batch dimension is symbolic. Some SOMA output sizes stay symbolic
                        # until TensorRT builds the engine.
                        for dimension, size in zip(tensor.shape.dim[1:], spec["shape"][1:]):
                            if not (mode == "output" and dimension.dim_param):
                                self.assertEqual(dimension.dim_value, size, spec["name"])

    def test_checkpoints_must_be_a_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoints.txt"
            path.write_bytes(b"not a folder")
            with self.assertRaisesRegex(ValueError, "must be a folder"):
                checkpoints.checkpoint_root(path, export.CHECKPOINTS["g1"])


if __name__ == "__main__":
    unittest.main()
