# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
import unittest
from pathlib import Path


class ImportBoundaryTests(unittest.TestCase):
    def assert_imports_without(self, source: str, excluded: tuple[str, ...]) -> None:
        # A fresh process prevents other test modules from hiding eager imports.
        program = (
            "import sys\n"
            + source
            + f"\nexcluded = {excluded!r}\n"
            + "unexpected = sorted(name for name in sys.modules if any(\n"
            + "    name == prefix or name.startswith(prefix + '.')\n"
            + "    for prefix in excluded))\n"
            + "assert not unexpected, unexpected\n"
        )
        result = subprocess.run(
            [sys.executable, "-B", "-c", program],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_model_contract_does_not_load_execution_backends(self) -> None:
        self.assert_imports_without(
            "from motionbricks.model.soma.contract import SOMA_N1_CONTRACT\n"
            "from motionbricks.model.config import MotionBricksInferenceConfig\n"
            "from motionbricks.inference.types import TensorMode, TensorSpec\n",
            ("tensorrt", "newton", "warp", "motionbricks.inference.engine"),
        )

    def test_common_runtime_does_not_load_rig_specific_runtime_or_backends(self) -> None:
        self.assert_imports_without(
            "from motionbricks.runtime.target_pose import (\n"
            "    IKPin, TargetPoseInferenceParams, create_soma_target_pose_ik_rig)\n"
            "from motionbricks.runtime.blend_spaces import BlendSpace\n"
            "assert 'app' not in sys.modules, 'target poses or blend spaces loaded app'\n"
            "from app.runtime.common.motion import (\n"
            "    AnimationClip, LocomotionCommand, LocomotionMode,\n"
            "    LocomotionModeRuntime, LocomotionSettings, PlaybackSource,\n"
            "    directional_blend_space_value, ground_direction)\n"
            "from app.runtime.common.state_machine import StateMachine\n"
            "from app.runtime.common.states import IdleState, InteractionState, LocomotionState, State\n"
            "from app.runtime.common.target_pose_blender import BaseTargetPoseBlender\n"
            "from app.runtime.common.interaction import InteractionProbe\n"
            "from app.runtime.common.character import BaseCharacter\n",
            (
                "app.runtime.soma",
                "app.runtime.unitree_g1",
                "app.authoring",
                "motionbricks.model.soma.engine",
                "motionbricks.inference.engine",
                "motionbricks.renderers",
                "tensorrt",
                "newton",
                "imgui_bundle",
                "pyglet",
            ),
        )

    def test_shared_behaviors_and_documents_do_not_load_rig_specific_runtime_or_applications(self) -> None:
        self.assert_imports_without(
            "import app.runtime.common.smart_primitive_behaviors\n"
            "import app.runtime.common.io.scene_document\n"
            "import app.runtime.common.io.smart_primitive_document\n",
            (
                "tensorrt",
                "imgui_bundle",
                "pyglet",
                "app.authoring",
                "app.smart_primitive_editor",
                "app.scene_editor",
                "app.scene_preview",
                "app.inbetweener_demo",
                "app.runtime.soma",
                "app.runtime.unitree_g1",
            ),
        )

    def test_all_runtime_modules_import_without_authoring_or_ui(self) -> None:
        self.assert_imports_without(
            "import importlib\n"
            "import pkgutil\n"
            "import app.runtime\n"
            "for module in pkgutil.walk_packages(app.runtime.__path__, 'app.runtime.'):\n"
            "    importlib.import_module(module.name)\n",
            (
                "app.authoring",
                "app.smart_primitive_editor",
                "app.scene_editor",
                "app.scene_preview",
                "app.inbetweener_demo",
                "imgui_bundle",
                "pyglet",
            ),
        )


if __name__ == "__main__":
    unittest.main()
