# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import warp as wp

from app.runtime.common.motion import (
    CROUCH_LOCOMOTION_MODE,
    DIRECTIONAL_LOCOMOTION_MODE,
    LINEAR_LOCOMOTION_MODE,
    LocomotionCommand,
    LocomotionMode,
)
from app.runtime.common.state_machine import StateMachine
from app.runtime.common.states import State
from motionbricks.runtime.blend_spaces import BlendSpaceType


def _command(requested_mode: str | None = None) -> LocomotionCommand:
    return LocomotionCommand(
        facing_direction=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        movement=None,
        requested_mode=requested_mode,
    )


class StateMachineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.modes = tuple(LocomotionMode(
            name, Path("unused.smart_primitive.json"), blend_type, (("Idle", (0, 0)),),
            max_speed=1., default_speed=1.,
        ) for name, blend_type in (
            (LINEAR_LOCOMOTION_MODE, BlendSpaceType.LINEAR_1D),
            (DIRECTIONAL_LOCOMOTION_MODE, BlendSpaceType.DIRECTIONAL_2D),
            (CROUCH_LOCOMOTION_MODE, BlendSpaceType.DIRECTIONAL_2D),
            ("zombie_1d", BlendSpaceType.LINEAR_1D),
            ("skipping_1d", BlendSpaceType.LINEAR_1D),
        ))
        self.machine = self._machine()

    def _machine(self, initial_mode: str = LINEAR_LOCOMOTION_MODE) -> StateMachine:
        return StateMachine(
            tuple(SimpleNamespace(definition=mode) for mode in self.modes),
            initial_mode,
            standing_idle_clips=(object(),),
            crouching_idle_clips=(object(),),
            pose=wp.empty(1, dtype=wp.transform, device="cpu"),
        )

    def test_transition_calls_exit_before_enter_and_skips_the_active_state(self) -> None:
        events = []
        character = object()
        previous = Mock(spec=State)
        next_state = Mock(spec=State)
        previous.exit.side_effect = lambda *_: events.append("exit")
        next_state.enter.side_effect = lambda *_: events.append("enter")
        self.machine._state = previous

        self.machine.transition_to(character, next_state)
        self.assertEqual(events, ["exit", "enter"])
        self.assertIs(self.machine.state, next_state)

        self.machine.transition_to(character, next_state)
        next_state.exit.assert_not_called()
        next_state.enter.assert_called_once()

    def test_mode_request_does_not_interrupt_interaction_and_selects_the_resume_pair(self) -> None:
        standing_mode, crouch_mode = self.modes[0], self.modes[2]
        character = SimpleNamespace(movement=Mock(input_speed=0.0))
        interaction = self.machine.interaction_state
        crouching = self.machine.crouching_locomotion_state
        crouching.enter = Mock()
        interaction.update = Mock()
        interaction.exit = Mock()
        self.machine._state = interaction
        command = _command(CROUCH_LOCOMOTION_MODE)

        self.machine.update(character, 0.1, command)

        self.assertIs(self.machine.state, interaction)
        self.assertIs(self.machine.active_locomotion_state, crouching)
        interaction.update.assert_called_once_with(character, self.machine, 0.1)
        # Movement keeps smoothing during the interaction, easing from the standing mode into Crouching.
        character.movement.update.assert_called_once_with(command, crouch_mode, 0.1, standing_mode)

        character.movement.input_speed = 2.0
        self.machine.resume_movement(character)

        self.assertIs(self.machine.state, crouching)
        interaction.exit.assert_called_once_with(character, self.machine)
        crouching.enter.assert_called_once_with(character, self.machine)

    def test_extra_modes_are_standing_styles_that_share_the_standing_idle(self) -> None:
        machine = self._machine(CROUCH_LOCOMOTION_MODE)
        standing = machine.standing_locomotion_state
        self.assertEqual(machine.style_modes, ("zombie_1d", "skipping_1d"))
        self.assertEqual(
            list(standing.mode_runtimes),
            [LINEAR_LOCOMOTION_MODE, DIRECTIONAL_LOCOMOTION_MODE, "zombie_1d", "skipping_1d"],
        )

        # Picking a style while crouched stands the character up into the standing idle.
        character = SimpleNamespace(movement=Mock(input_speed=0.0))
        machine.crouching_idle_state.exit = Mock()
        machine.standing_idle_state.enter = Mock()
        machine.standing_idle_state.update = Mock()
        self.assertIs(machine.state, machine.crouching_idle_state)

        machine.update(character, 0.1, _command("zombie_1d"))

        self.assertEqual(standing.active_mode.name, "zombie_1d")
        self.assertIs(machine.active_locomotion_state, standing)
        self.assertIs(machine.state, machine.standing_idle_state)


if __name__ == "__main__":
    unittest.main()
