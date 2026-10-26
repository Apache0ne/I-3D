# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import unittest

from app.runtime.common.motion import (
    CROUCH_LOCOMOTION_MODE, DIRECTIONAL_LOCOMOTION_MODE, LINEAR_LOCOMOTION_MODE,
)
from app.utils.input import InputManager, LocomotionModeInput, apply_radial_dead_zone


class _Controller:
    def __init__(self, name: str = "Test Controller") -> None:
        self.name = name
        self.leftx = 0.0
        self.lefty = 0.0
        self.rightx = 0.0
        self.righty = 0.0
        self.dpadx = 0.0
        self.dpady = 0.0
        self.a = False
        self.x = False
        self.y = False
        self.b = False
        self.open_count = 0
        self.close_count = 0

    def open(self) -> None:
        self.open_count += 1

    def close(self) -> None:
        self.close_count += 1


class _ControllerManager:
    def __init__(self, controllers=()) -> None:
        self.controllers = list(controllers)
        self.handler = None

    def get_controllers(self) -> list[_Controller]:
        return self.controllers

    def push_handlers(self, handler) -> None:
        self.handler = handler

    def remove_handlers(self, handler) -> None:
        if self.handler is handler:
            self.handler = None


class InputTests(unittest.TestCase):
    def test_radial_dead_zone_preserves_and_remaps_stick_magnitude(self) -> None:
        self.assertEqual(
            apply_radial_dead_zone(0.1, -0.1, 0.15),
            (0.0, 0.0),
        )
        x, y = apply_radial_dead_zone(0.575, 0.0, 0.15)
        self.assertAlmostEqual(x, 0.5)
        self.assertEqual(y, 0.0)

    def test_keyboard_movement_is_normalized_at_default_speed(self) -> None:
        for held_keys in ({"w", "d"}, {"up", "right"}, {"w", "right"}):
            with self.subTest(keys=sorted(held_keys)):
                input_manager = InputManager(lambda key: key in held_keys)

                frame = input_manager.poll()

                self.assertAlmostEqual(frame.movement[0], 1.0 / math.sqrt(2.0))
                self.assertAlmostEqual(frame.movement[1], 1.0 / math.sqrt(2.0))
                self.assertEqual(frame.camera_orbit, (0.0, 0.0))
                self.assertFalse(frame.use_max_speed)
                self.assertAlmostEqual(math.hypot(*frame.movement), 1.0)

    def test_arrow_keys_move_like_wasd(self) -> None:
        for arrow, letter in (("up", "w"), ("down", "s"), ("left", "a"), ("right", "d")):
            with self.subTest(key=arrow):
                arrow_frame = InputManager(lambda key: key == arrow).poll()
                letter_frame = InputManager(lambda key: key == letter).poll()
                self.assertEqual(arrow_frame.movement, letter_frame.movement)
                self.assertNotEqual(arrow_frame.movement, (0.0, 0.0))

    def test_controller_sticks_override_keyboard_and_use_max_speed(self) -> None:
        controller = _Controller()
        controller.leftx = 0.0
        controller.lefty = 0.5
        controller.rightx = 0.25
        controller.righty = -0.25
        manager = _ControllerManager((controller,))
        input_manager = InputManager(
            lambda key: key in {"w", "space"},
            manager,
            movement_dead_zone=0.2,
            camera_dead_zone=0.1,
        )

        frame = input_manager.poll()

        expected_speed_fraction = (0.5 - 0.2) / (1.0 - 0.2)
        self.assertEqual(frame.movement[0], 0.0)
        self.assertAlmostEqual(frame.movement[1], expected_speed_fraction)
        self.assertAlmostEqual(
            math.hypot(*frame.movement),
            expected_speed_fraction,
        )
        raw_camera_length = math.hypot(0.25, -0.25)
        camera_magnitude = (raw_camera_length - 0.1) / (1.0 - 0.1)
        expected_camera_component = camera_magnitude / math.sqrt(2.0)
        self.assertAlmostEqual(frame.camera_orbit[0], expected_camera_component)
        self.assertAlmostEqual(frame.camera_orbit[1], -expected_camera_component)
        self.assertTrue(frame.use_max_speed)
        self.assertEqual(controller.open_count, 1)

        input_manager.close()

        self.assertEqual(controller.close_count, 1)
        self.assertIsNone(manager.handler)

    def test_keyboard_remains_available_inside_controller_dead_zone(self) -> None:
        controller = _Controller()
        controller.leftx = 0.19
        controller.rightx = 0.19
        input_manager = InputManager(
            lambda key: key == "a",
            _ControllerManager((controller,)),
            movement_dead_zone=0.2,
            camera_dead_zone=0.2,
        )

        frame = input_manager.poll()

        self.assertEqual(frame.movement, (-1.0, 0.0))
        self.assertEqual(frame.camera_orbit, (0.0, 0.0))
        self.assertFalse(frame.use_max_speed)
        input_manager.close()

    def test_controller_hot_plug_selects_first_available_device(self) -> None:
        manager = _ControllerManager()
        input_manager = InputManager(lambda _key: False, manager)
        first = _Controller("First")
        second = _Controller("Second")
        first.leftx = 1.0
        second.lefty = 1.0

        manager.controllers.extend((first, second))
        input_manager.on_connect(first)
        input_manager.on_connect(second)
        self.assertEqual(input_manager.poll().movement, (1.0, 0.0))
        self.assertEqual(second.open_count, 0)

        manager.controllers.remove(first)
        input_manager.on_disconnect(first)

        self.assertEqual(input_manager.poll().movement, (0.0, 1.0))
        self.assertEqual(second.open_count, 1)
        input_manager.close()

    def test_i_key_or_x_button_interacts(self) -> None:
        held_keys = {"v"}
        controller = _Controller()
        input_manager = InputManager(lambda key: key in held_keys)
        input_manager._controller = controller
        self.assertFalse(input_manager.is_interaction_down())
        held_keys.add("i")
        self.assertTrue(input_manager.is_interaction_down())
        held_keys.clear()
        controller.x = True
        self.assertTrue(input_manager.is_interaction_down())


class LocomotionModeInputTests(unittest.TestCase):
    def test_keys_toggle_modes_like_their_buttons_once_per_press(self):
        held_keys = set()
        controller = _Controller()
        input_manager = InputManager(lambda key: key in held_keys)
        input_manager._controller = controller
        selector = LocomotionModeInput()
        current_mode = LINEAR_LOCOMOTION_MODE
        for buttons, keys, expected in (
            ("", "", None),
            ("y", "", DIRECTIONAL_LOCOMOTION_MODE),
            ("y", "", None),
            ("", "", None),
            ("", "o", LINEAR_LOCOMOTION_MODE),
            # Holding the key and its button together is still one press.
            ("y", "o", None),
            ("", "", None),
            ("b", "", CROUCH_LOCOMOTION_MODE),
            ("b", "", None),
            ("", "", None),
            ("", "p", DIRECTIONAL_LOCOMOTION_MODE),
            ("", "", None),
            ("", "p", CROUCH_LOCOMOTION_MODE),
            ("", "", None),
            ("", "o", LINEAR_LOCOMOTION_MODE),
        ):
            with self.subTest(mode=current_mode, buttons=buttons, keys=keys):
                controller.y, controller.b = "y" in buttons, "b" in buttons
                held_keys.clear()
                held_keys.update(keys)
                requested = selector.poll(input_manager, current_mode)
                self.assertEqual(requested, expected)
                if requested is not None:
                    current_mode = requested
        input_manager._controller = None
        held_keys.clear()
        self.assertIsNone(selector.poll(input_manager, current_mode))
        held_keys.add("p")
        self.assertEqual(selector.poll(input_manager, current_mode), CROUCH_LOCOMOTION_MODE)

    def test_bracket_or_a_cycles_styles_and_resumes_the_last_one(self):
        held_keys = set()
        controller = _Controller()
        input_manager = InputManager(lambda key: key in held_keys)
        input_manager._controller = controller
        selector = LocomotionModeInput(("skipping_1d", "vogue_1d", "zombie_1d"))
        current_mode = LINEAR_LOCOMOTION_MODE
        for button_a, keys, expected in (
            (False, "[", "skipping_1d"),
            (False, "[", None),
            (False, "", None),
            (True, "", "vogue_1d"),
            (False, "", None),
            (False, "[", "zombie_1d"),
            (False, "", None),
            (False, "[", "skipping_1d"),
            (False, "", None),
            (False, "[", "vogue_1d"),
            (False, "", None),
            (False, "o", LINEAR_LOCOMOTION_MODE),
            (False, "", None),
            (False, "p", CROUCH_LOCOMOTION_MODE),
            (False, "", None),
            # From any other mode, the last style used comes back.
            (False, "[", "vogue_1d"),
        ):
            with self.subTest(mode=current_mode, button_a=button_a, keys=keys):
                controller.a = button_a
                held_keys.clear()
                held_keys.update(keys)
                requested = selector.poll(input_manager, current_mode)
                self.assertEqual(requested, expected)
                if requested is not None:
                    current_mode = requested
        # Without style modes, the style key does nothing.
        held_keys.clear()
        held_keys.add("[")
        self.assertIsNone(LocomotionModeInput().poll(input_manager, LINEAR_LOCOMOTION_MODE))


if __name__ == "__main__":
    unittest.main()
