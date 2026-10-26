# Scene Preview

[Scene Preview](../app/scene_preview.py) drives one character through a scene with continuous locomotion and interactions. It is a small character runtime built on MotionBricks. States decide the target poses. The engine generates toward them. The character plays the result and records it for the next generation.

<table>
  <tr>
    <th>SOMA</th>
    <th>Unitree G1</th>
  </tr>
  <tr>
    <td><img src="media/scene_preview_soma.gif" alt="Scene Preview with SOMA" width="100%"></td>
    <td><img src="media/scene_preview_g1.gif" alt="Scene Preview with the Unitree G1" width="100%"></td>
  </tr>
</table>

This page shows how to run the app, then explains its runtime for developers who want a character to move continuously and interact with objects in a scene. Every part is plain Python. The same code runs the SOMA and Unitree G1 characters.

[Part 4 of the guided tutorial](guided_tutorial.md#part-4-play-the-scene) plays the interaction scene that the tutorial builds. The sections below connect what you see there to the code.

## Run Scene Preview

```sh
uv run python -m app.scene_preview --scene assets/scenes/soma_gym.scene.json
uv run python -m app.scene_preview --scene assets/scenes/g1_gym.scene.json
```

- `--scene` opens a scene document. Its rig selects SOMA or the Unitree G1. The SOMA gym is the default.
- `--model-precision fp16` uses the mixed FP16 model. FP32 is the default. [TensorRT builds](runtime_api.md#tensorrt-builds) explains how FP16 and FP32 motion can differ.
- `--verbose` shows detailed log messages.

The gym scenes place a small jump, a vault, a bar to duck under, and a box to sit on. Every motion comes from the model, except for the idle and sitting loops that play from clips.

Things to try:

- Walk into each object. With **Auto Interact** on, the character jumps, vaults, ducks under the bar, or sits on the box as it arrives.
- Approach the same object from other angles and at other speeds. Pivots turn and slide each sequence toward the approach. The jumps and the bar pick their variation by speed.
- While seated, press **I** (**X** on a controller) to stand up.
- Turn off **Auto Interact** in the **Scene** section, then press **I** (**X** on a controller) near an object to start its interaction by hand.
- Switch to Crouching with **P** (**B** on a controller) and walk under the bar. Its raised interaction box only catches a standing character, so no interaction starts.
- Switch to Strafing with **O** (**Y** on a controller), or cycle through the style modes with **[** (**A** on a controller).
- Turn on the options under **Visibility** to see the poses and boxes behind each generation, such as **Show Context Poses** and **Show Probe Box**.
- Open the other gym with **...** next to **Scene**. The scene's rig switches the character between SOMA and the Unitree G1.

### Controls

| Action | Keyboard and mouse | Controller |
| --- | --- | --- |
| Move forward, left, backward, or right | **W**, **A**, **S**, and **D**, or the **arrow keys** | Left stick or D-pad |
| Run | Hold **Space** while moving | Push the left stick further |
| Interact manually, or stand up after sitting | **I** | **X** |
| Switch between the Forward and Strafing modes | **O** | **Y** |
| Switch between the Crouching and Strafing modes | **P** | **B** |
| Cycle through the style modes, such as Skipping and Zombie | **[** | **A** |
| Orbit the camera around the character | Drag with the **middle mouse button** | Right stick |

## Code map

These symbols carry the runtime's model-facing flow:

| Symbol | Responsibility |
| --- | --- |
| `ScenePreview.run()` in [scene_preview.py](../app/scene_preview.py) | Loads a pending scene between frames, then steps and draws each frame on the engine stream. |
| `ScenePreview.step()` | Reads input, advances the character, moves the camera, then updates interactions. |
| `ScenePreview.load_scene()` | Reads and checks a scene, loads its rig and Smart Primitives, then creates the character. If a scene opened from the window fails before its engine loads, the current scene keeps running and the window shows the error. |
| `ScenePreview._update_smart_primitive_interaction()` | Finds the nearest Smart Primitive whose interaction box overlaps the probe box. |
| `ScenePreview._dispatch_interaction_input()` | Starts an interaction, by hand or with **Auto Interact**, or passes a press to the running behavior. |
| `BaseCharacter.step()` in [character.py](../app/runtime/common/character.py) | Updates the state machine, plays the current pose, and records it for the next generation. |
| `BaseCharacter.generate_to_target_poses()` | Generates from the recorded poses toward placed target poses, then restarts playback. |
| `StateMachine.update()` in [state_machine.py](../app/runtime/common/state_machine.py) | Switches modes, smooths the movement, then updates the current state. |
| `IdleState`, `LocomotionState`, and `InteractionState` in [states.py](../app/runtime/common/states.py) | Decide when to generate, toward which targets, and when to hand over to another state. |
| `BaseTargetPoseBlender.evaluate()` in [target_pose_blender.py](../app/runtime/common/target_pose_blender.py) | Blends the locomotion targets for the desired velocity and places them ahead of the character. |
| `LocomotionMode` and `LocomotionSettings` in [motion.py](../app/runtime/common/motion.py) | Hold each mode's targets and speeds, how quickly movement responds to input, and when locomotion generates again. |
| `SmartPrimitiveBehavior` in [smart_primitive_behaviors.py](../app/runtime/common/smart_primitive_behaviors.py) | Base class of the behaviors that scenes choose, such as `VelocityMatchingSequenceBehavior` and `SittingBehavior`. |
| [app/runtime/soma](../app/runtime/soma), [app/runtime/unitree_g1](../app/runtime/unitree_g1) | Each rig's character, target pose blender, and presets. |

The rest of `scene_preview.py` handles the camera, the drawing, and the **Scene Preview** window.

## Main update loop

`ScenePreview.run()` loads a pending scene between frames. Loading can build another engine, so it runs outside the active engine's stream. `run()` then enters the engine stream and calls `step()` and `render()`.

Each `step()`:

1. Measures the time since the last frame.
2. Reads the keys and the controller, then orbits the camera.
3. Builds a `LocomotionCommand` from the movement relative to the camera, the run input, and any requested mode.
4. Calls `BaseCharacter.step()`.
5. Moves the camera with the character.
6. Tests the probe box against every interaction box.
7. Starts an interaction, or passes a press to the running behavior.

The character advances before the interaction test. An interaction that starts in this frame therefore generates its first segment from poses that were already recorded.

### The character update

`BaseCharacter.step()` advances the character by the frame time:

```python
self.state_machine.update(self, delta_time, command)
previous_ground_position = self.character_ground_position.copy()
self._sample_playback(delta_time)
self._update_character_velocity(previous_ground_position, delta_time)
self.player.record_pose(delta_time, self.current_pose)
self._update_probe_box()
```

1. The state machine updates the movement and the current state. That state may generate new motion or start a clip.
2. `_sample_playback()` writes the pose to show into `current_pose`. It samples generated motion or a clip.
3. `_update_character_velocity()` measures how far the character moved.
4. `record_pose()` stores the pose that was shown.
5. `_update_probe_box()` moves the probe box ahead of the character.

The character plays from one of three sources:

- `GENERATED`: motion generated toward target poses.
- `CLIP_TRANSITION`: motion generated toward the first frame of a clip.
- `CLIP`: the clip itself, played once or in a loop.

### Generate from recorded motion

Every state generates through `generate_to_target_poses()`. The character's motion player samples four context poses at 30 Hz from the recorded history, ending at the latest pose. It then calls the engine and restarts playback. These are the same calls as in the [Inbetweener](inbetweener.md#generation-flow).

The recorded history holds the poses that were actually shown, including clip frames and any [IK Pins](smart_primitive_editor.md#ik-pins) corrections. That is why a generation can start from a clip or from the middle of another generation. A state can start a generation at any time. Otherwise, the current segment plays until its transition point.

## State machine and states

![Scene Preview state machine with Idle, Locomotion, and Interaction states](media/state_machine.png)

`StateMachine` holds five states:

| State | On enter | On each update |
| --- | --- | --- |
| Idle, Crouching Idle | Generates a transition into an idle clip | Switches to locomotion when there is movement input. Loops the clip when the transition ends. |
| Locomotion, Crouching Locomotion | Generates toward the blended target | Switches to idle when the input stops. Generates again when needed. |
| Interaction | Starts the Smart Primitive's behavior | Generates toward the next target at each transition point, or waits for the clip to end. |

Idle and locomotion come in two pairs, standing and crouching. Standing locomotion has the Forward, Strafing, and style modes. Crouching locomotion has the Crouching mode, and its probe box is 60 percent as tall. When a mode key selects a mode of the other pair, the character moves to that pair's idle or locomotion state, depending on the input.

Each frame, `StateMachine.update()` switches the mode when the command asks for one. It then smooths the movement and updates the current state. Every switch exits the old state before it enters the new one. Movement keeps smoothing in every state, so it carries into the next one.

### Idle

Idle plays looping clips, and the model generates the way in:

1. Choose one of the rig's idle clips at random, other than the current one.
2. Place the clip one second ahead of the character along its actual velocity. Turn it to the facing of the last movement input.
3. Sample the clip's first four poses at 30 Hz into the target token. Generate a transition toward these poses.
4. When the transition ends, play the clip in a loop.

When a scene starts, the character begins directly in an idle loop, and its history is filled with that first pose.

### Interaction

`start_interaction()` enters the interaction state with a Smart Primitive's [behavior](../app/runtime/common/smart_primitive_behaviors.py). The behavior calls `finish()` when it is done. `resume_movement()` then picks locomotion if there is movement input, and idle otherwise. A sequence or clip played without an `on_complete` callback calls `finish()` when it ends.

In the [guided tutorial](guided_tutorial.md#inspect-the-runtime), the avoidance and jump interactions returned to locomotion as soon as their sequences ended. The sit interaction stayed in the interaction state while its clip looped.

## Continuous locomotion

<table>
  <tr>
    <th>SOMA</th>
    <th>Unitree G1</th>
  </tr>
  <tr>
    <td><img src="media/locomotion_targets_soma.gif" alt="SOMA walks, runs, and stops with context, target, and blend space poses shown" width="100%"></td>
    <td><img src="media/locomotion_targets_g1.gif" alt="Unitree G1 walks, runs, and stops with context, target, and blend space poses shown" width="100%"></td>
  </tr>
  <tr>
    <td><img src="media/locomotion_directional_soma.gif" alt="SOMA strafes in four directions, crouches, then stands back up, with context, target, and blend space poses shown" width="100%"></td>
    <td><img src="media/locomotion_directional_g1.gif" alt="Unitree G1 strafes in four directions, crouches, then stands back up, with context, target, and blend space poses shown" width="100%"></td>
  </tr>
</table>

*Top: in the Forward mode, each character walks to a full run, then stops into idle. Bottom: in the Strafing mode, each character faces one way while it moves forward, right, back, and left. It then switches to Crouching while moving, and back to Strafing before it stops. Context poses are purple, the current targets are blue, and the live blend space pose is green.*

Locomotion keeps generating toward a target placed ahead of the character. Each generation starts from the poses that were just shown, so the motion continues without a break. The movement input is relative to the camera. `MovementControl` smooths it into a desired velocity and a facing direction. In Strafing and Crouching, the character faces the camera direction.

### Locomotion modes

Each rig's `presets.py` defines its locomotion modes. A `LocomotionMode` names a Smart Primitive document of blend targets, the blend value of each target, and the speed limits. The blend targets hold four poses sampled from locomotion clips:

| Mode | Blend space | Targets |
| --- | --- | --- |
| Forward | 1D, by speed | Idle, Walk, Jog, Run |
| Strafing | 2D, by velocity relative to facing | Idle, Forward, Backward, Left, Right |
| Crouching | 2D, by velocity relative to facing | Idle, Forward, Backward, Left, Right |
| Styles | 1D, by speed | Idle, then Skipping, Vogue, Injured Torso, Injured Leg, Zombie, or Combat |

The six style modes share one document, `locomotion_style_variations_bs1_linear`. Each blends the Idle target at 0 into its style target at 1. A style mode's default speed is its maximum speed, so the style pose shows in full. Its targets also sit farther ahead, so it walks close to the speed of its clip. Its transition point sits nearer the end of each transition, so the motion gets closer to the style pose. From another mode, **[** (**A** on a controller) returns to the last style mode used. From a style mode, **O** (**Y** on a controller) returns to Forward, and **P** (**B** on a controller) switches to Crouching.

<table>
  <tr>
    <th>SOMA</th>
    <th>Unitree G1</th>
  </tr>
  <tr>
    <td><img src="media/locomotion_styles_soma.gif" alt="SOMA walks in each locomotion style, from Skipping to Combat" width="100%"></td>
    <td><img src="media/locomotion_styles_g1.gif" alt="Unitree G1 walks in each locomotion style, from Skipping to Combat" width="100%"></td>
  </tr>
</table>

*SOMA and the Unitree G1 walk in each of the six style modes, named in the top left*

### Blend and place the target

Each time locomotion generates, it:

1. Evaluates the blend space at the desired velocity. Blends the targets.
2. Scales the ground speed of the blended target to the desired speed times the mode's `velocity_scale`.
3. Places the blended target `position_scale` seconds of travel ahead of the character, along the scaled desired velocity. Turns it to the facing direction. Only the root moves.
4. Generates toward all four target poses, with the mode's **Transition Early By** and token counts.

`BaseTargetPoseBlender.evaluate()` runs the first three steps. Each rig overrides the blend, speed, and placement steps for its pose layout.

These are not classic animation blend spaces. A classic blend space, like the ones in game engines, mixes several clips as they play. The clips must stay in step, so the game engine synchronizes their foot phases and stretches them in time. Here the blend value mixes target poses sampled once from each clip. The model generates the motion toward the blended target, so there is nothing to synchronize or stretch. A rough blend is fine, because the model is robust to imperfect targets and locomotion generates again before each target is reached. Refer to Appendix D of the [paper](https://arxiv.org/abs/2604.24833).

### When locomotion generates

`LocomotionState.update()` decides each frame:

```python
if character.movement.input_speed <= 0.0:
    state_machine.transition_to_idle(character)
    return

if (
    self._force_regeneration
    or character.time_to_transition() <= delta_time
    or self._needs_regeneration(character.movement)
):
    self._generate(character)
```

Without movement input, the character switches to idle. Otherwise, locomotion generates in three cases:

- The mode changed, which sets `_force_regeneration`.
- Playback reached the transition point of the current segment.
- The input changed enough. `_needs_regeneration()` checks whether the character started moving, whether the speed changed by more than a fraction of the remaining speed range, or whether the direction turned past a threshold. `LocomotionSettings` in [motion.py](../app/runtime/common/motion.py) holds these thresholds.

These rules are one way to decide when to generate, not the only one. Other rules work too, such as generating on a fixed schedule or only at transition points. Try your own in `LocomotionState` and compare how the motion responds.

### See it in the viewer

Turn on **Show Target Poses**, **Show Context Poses**, and **Show Blend Space Poses** under **Visibility** in the **Scene Preview** window. They show the placed targets of the current generation, the context poses that the next generation would read, and the blend space target for the current frame. The blend space target shows only during locomotion. These are the options that you enabled in [Inspect the runtime](guided_tutorial.md#inspect-the-runtime) in the guided tutorial. **Show Foot Contacts** outlines the model's [foot contacts](runtime_api.md#foot-contacts) on the character's soles.

## Smart Primitive behaviors

### Find an interaction

The character carries a probe box that reaches ahead along the desired velocity. Each frame, Scene Preview tests it against every Smart Primitive interaction box and keeps the nearest overlapping instance. Interaction boxes only trigger behaviors. The model never sees them as collision geometry.

- **Auto Interact** is on by default. It starts an interaction once per approach.
- **I** (**X** on a controller) starts an interaction by hand. During an interaction, the same press goes to the running behavior, which is how `SittingBehavior` makes the character stand up.

### Run a behavior

`StateMachine.start_interaction()` enters the interaction state, which calls the behavior's `enter()`. The behavior then plays sequences and clips through the interaction state:

- `play_target_pose_sequence()` generates through the targets of a sequence in order. Each step keeps its **Poses to Use**, **Transition Early By**, and **Tokens to Generate** settings. A SOMA step can also pin hands or feet with [IK Pins](smart_primitive_editor.md#ik-pins).
- `play_clip()` plays a clip once or in a loop. By default, the clip's first root pose is aligned with the character's current root.
- `finish()` returns control to locomotion or idle.

At each transition point, `InteractionState.update()` generates toward the next target of the sequence. After the last target, it runs the sequence's `on_complete` callback. The **Transition Early By** value of each step sets its transition point.

### Bundled behaviors

Each Smart Primitive instance in a scene names its behavior class. `behavior_types()` in [smart_primitive_behaviors.py](../app/runtime/common/smart_primitive_behaviors.py) finds every `SmartPrimitiveBehavior` subclass by class name:

| Class | What it does |
| --- | --- |
| `VelocityMatchingSequenceBehavior` | Picks the sequence whose authored start speed is closest to the character's speed, aligns its pivot, and plays it |
| `SittingBehavior` | Plays Sit, loops the sitting clip, and plays Stand on the next press |

In [Part 3 of the guided tutorial](guided_tutorial.md#part-3-assemble-an-interaction-scene-with-avoidance-jump-and-sit), you chose `VelocityMatchingSequenceBehavior` for the avoidance and the jump. You chose `SittingBehavior` for the sit.

The bundled avoidance document in the gym scenes has a Walk and a Jog sequence. `VelocityMatchingSequenceBehavior` picks between them by speed:

![VelocityMatchingSequenceBehavior picks the Walk or Jog sequence by speed](media/velocity_matching_flow.png)

`SittingBehavior` plays the Sit sequence, then loops a sitting clip until the next press. The Stand sequence then generates the motion back to standing:

![SOMA sits down, holds, stands up, and walks off](media/sit_hold_soma.gif)
*SOMA sits down, holds, stands up, and walks off*

### Write a behavior

To add a behavior, subclass `SmartPrimitiveBehavior`, override `enter()` and `interact()`, and declare the sequences and clips it needs:

```python
class LeaningBehavior(SmartPrimitiveBehavior):
    """Lean on a wall, stay there, and step away when the user interacts again."""

    required_sequences = ("Lean", "StepAway")
    clip_paths = (LEAN_LOOP_CLIP,)

    def enter(self) -> None:
        self.interaction_state.play_target_pose_sequence("Lean", on_complete=self._hold)

    def _hold(self) -> None:
        self.interaction_state.play_clip(LEAN_LOOP_CLIP, loop=True, align_to_current_pose=True)

    def interact(self) -> None:
        if self.interaction_state.is_looping:
            self.interaction_state.play_target_pose_sequence("StepAway")
```

The class then appears in the Scene Editor's **Behavior** list. Define it in `smart_primitive_behaviors.py`, or import its module before scenes load. The clips a behavior declares load with the scene. Declared sequences are checked when the instance is created. The [Smart Primitive Editor](smart_primitive_editor.md#example-a-sitting-hold) page shows how to author the sequences for a hold.

The bundled behaviors and this example are one setup among many. The model only needs context and target poses, so how a behavior picks and times its targets is up to you. Use them as a starting point and explore your own ideas.

## Scenes

A `.scene.json` document records its rig, the CharacterStart, the Smart Primitive instances, and the Solid Cubes. Each instance references a Smart Primitive document, with its own placement, behavior, and visibility settings. Instances of the same document share one Smart Primitive asset. The [Scene Editor](scene_editor.md) creates scenes, and the files are plain JSON that you can also edit by hand.

`load_scene()` flattens the item hierarchy into world transforms, because parents only help editing. It checks every instance before it builds an engine. A new Smart Primitive item has an empty **Behavior**, so choose one before you save the scene.

## Explore and adapt the runtime

Scene Preview is a reference to learn from, not a framework you have to build on. Change its parts to see how the motion responds, then carry the ideas into your own engine or tools. Each part has its own place:

| Goal | Where to look |
| --- | --- |
| Add an interaction | Author a Smart Primitive document, and write a behavior class if the existing ones do not fit. See [Write a behavior](#write-a-behavior). |
| Change the locomotion poses | The blend target documents that each rig's `presets.py` names, such as `locomotion_bs1_linear`. |
| Change speeds or the transition point of locomotion | `LocomotionMode` in the rig's `presets.py`. |
| Change when locomotion generates | `LocomotionSettings` in [motion.py](../app/runtime/common/motion.py), or the rules in `LocomotionState`. |
| Change how far ahead interactions start | `INTERACTION_PROBE` in each rig's presets, [soma/presets.py](../app/runtime/soma/presets.py) and [unitree_g1/presets.py](../app/runtime/unitree_g1/presets.py). [interaction.py](../app/runtime/common/interaction.py) describes each value. |
| Build the same pattern in another engine | Record every pose you show, and build each context from the latest four at 30 Hz. Let your own states choose and place the targets. Generate again at each transition point, or whenever your states ask. [Chaining generations](runtime_api.md#chaining-generations) describes the pieces. |
| Reuse this runtime in another Python app | Create the rig's character, such as `SOMACharacter`, with an engine and a `SmartPrimitiveAssetPool`. Call its `step()` once per frame with a `LocomotionCommand`, then render `current_pose`. Start interactions with `state_machine.start_interaction()`, as `ScenePreview` does. |

## Further reference

- [Guided tutorial](guided_tutorial.md): the user-facing workflow that ends in this app.
- [Runtime API](runtime_api.md): engines, motion players, generation settings, and chaining.
- [Smart Primitive Editor](smart_primitive_editor.md): authoring the target poses, sequences, and pivots that behaviors play.
- [Scene Editor](scene_editor.md): placing Smart Primitive instances and choosing their behaviors.
- [Inbetweener](inbetweener.md): the same generation flow without a state machine.
