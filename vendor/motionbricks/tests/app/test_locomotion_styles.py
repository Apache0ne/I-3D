# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Both rigs offer the same locomotion styles, each blending the standing Idle target into its style."""

import unittest
from pathlib import Path

import warp as wp

from app.runtime.common.io.smart_primitive_document import SmartPrimitiveAssetPool
from app.runtime.common.motion import load_locomotion_mode, style_locomotion_mode
from app.runtime.common.rig import load_rig_assets
from app.runtime.soma.presets import DEFAULT_LOCOMOTION_MODES as SOMA_MODES
from app.runtime.soma.presets import STYLE_LOCOMOTION_MODES as SOMA_STYLES
from app.runtime.unitree_g1.presets import DEFAULT_LOCOMOTION_MODES as G1_MODES
from app.runtime.unitree_g1.presets import STYLE_LOCOMOTION_MODES as G1_STYLES
from motionbricks.model.contract import ModelRig


class LocomotionStyleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.previous_device = wp.get_device()
        wp.set_device("cpu")

    @classmethod
    def tearDownClass(cls):
        wp.set_device(cls.previous_device)

    def test_both_rigs_offer_the_same_styles_at_their_top_speed(self):
        labels = ["Skipping", "Vogue", "Injured Torso", "Injured Leg", "Zombie", "Combat"]
        for modes, styles in ((SOMA_MODES, SOMA_STYLES), (G1_MODES, G1_STYLES)):
            self.assertEqual([mode.label for mode in styles], labels)
            self.assertEqual(modes[-len(styles):], styles)
        self.assertEqual([mode.name for mode in SOMA_STYLES], [mode.name for mode in G1_STYLES])
        for mode in (*SOMA_STYLES, *G1_STYLES):
            with self.subTest(mode=mode.name):
                self.assertEqual(mode.target_values, (("Idle", (0.0, 0.0)), (mode.label, (1.0, 0.0))))
                self.assertEqual(mode.default_speed, mode.max_speed)
                self.assertLess(mode.minimum_speed, mode.max_speed)

    def test_style_helper_names_the_mode_and_passes_its_tuning_through(self):
        mode = style_locomotion_mode(
            "Injured Leg", Path("styles.smart_primitive.json"), minimum_speed=0.5, max_speed=0.75,
            velocity_scale=0.9, position_scale=1.2, transition_early_by_frames=3, allowed_token_counts=(6, 7, 8),
        )
        self.assertEqual((mode.name, mode.label), ("injured_leg_1d", "Injured Leg"))
        self.assertEqual((mode.velocity_scale, mode.position_scale, mode.transition_early_by_frames), (0.9, 1.2, 3))
        self.assertEqual(mode.allowed_token_counts, (6, 7, 8))

    def test_each_rig_loads_every_style_from_its_document(self):
        for rig, styles in ((ModelRig.SOMA, SOMA_STYLES), (ModelRig.UNITREE_G1, G1_STYLES)):
            assets = load_rig_assets(rig)
            pool = SmartPrimitiveAssetPool(assets.clip_registry, assets.reference_pose, assets.model_to_runtime)
            for mode in styles:
                with self.subTest(rig=rig.name, mode=mode.name):
                    runtime = load_locomotion_mode(mode, pool)
                    self.assertEqual(runtime.target_poses.shape[:2], (2, 4))


if __name__ == "__main__":
    unittest.main()
