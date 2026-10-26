# Inbetweener

The [Inbetweener](../app/inbetweener_demo.py) is the smallest complete example of MotionBricks generation in this repository. It prepares context and target poses, generates a transition, and plays it up to its transition point. It then feeds the motion that was played back into the next generation.

<table>
  <tr>
    <th>SOMA</th>
    <th>Unitree G1</th>
  </tr>
  <tr>
    <td><img src="media/inbetweener_soma.gif" alt="The Inbetweener with SOMA" width="100%"></td>
    <td><img src="media/inbetweener_g1.gif" alt="The Inbetweener with Unitree G1" width="100%"></td>
  </tr>
</table>

This page shows how to run the app, then explains its code for developers who integrate MotionBricks into another application. The Newton viewer, its ImGui controls, and its renderers make the data flow visible. The model does not need them.

The [guided tutorial](guided_tutorial.md#part-1-generate-an-animation-in-the-inbetweener) uses this app to build an avoidance sequence from the Start, Entry, Duck-under, and End poses. The sections below connect those authoring actions to the code that prepares the model inputs and runs each generation.

## Run the Inbetweener

```sh
uv run python -m app.inbetweener_demo
```

- `--model soma` or `--model unitree_g1` picks the character to start with. SOMA is the default.
- `--model-precision fp16` uses the mixed FP16 model. FP32 is the default.
- `--verbose` shows detailed log messages.

The app opens with Start and **Target 1**, sampled from one walk clip at 1 s and 7 s. Target 1 stands 2 m from Start along -Y. Start is green, and targets are blue.

Things to try:

- Switch between SOMA and the Unitree G1 with the **Character** selector. Each switch starts over with Start, one target, and the default view.
- Move and turn a pose. Height and tilt are locked by default, so a pose stays on the ground.
- Pick another clip with **...** and drag the time slider to choose a pose from it.
- Add targets with **+**, drag them in the list to change their order, then press **Generate**.
- Tune **Poses to Use**, **Transition Early By (frames)**, and **Tokens to Generate** for each target. [How tutorial actions update model inputs](#how-tutorial-actions-update-model-inputs) explains each setting.
- Turn on **Show Foot Contacts**, beside **Generate**, to outline the foot contacts of generated motion.

### Controls

| Action | Control |
| --- | --- |
| Move the camera | **W**, **A**, **S**, and **D**, or the **arrow keys**. **Q** and **E** move it down and up. |
| Orbit around the camera pivot | Drag with the **middle mouse button** |
| Look around | Drag with the **left mouse button** |
| Pan | **Shift** + drag with the **middle mouse button** |
| Move toward or away from the camera pivot | **Ctrl** + drag with the **middle mouse button**, or the **mouse wheel** |
| Change the field of view | **Ctrl** + **mouse wheel** |
| Select a pose | Click its mesh or ground ring in the view, or click it in **Target Poses** |
| Move or turn the selection | Drag the gizmo, or edit the **Transform** fields |
| Reset or lock a transform | Right click **Translation** or **Rotation** to reset it. Right click an axis label to lock that axis. |

## Generation flow

The demo uses four essential runtime operations:

```python
context.update_from_poses(recent_poses)
target.update_from_poses(target_poses)
engine.generate_animation(context=context, target=target, output=output)
output.sample(time, current_pose)
```

The same pattern applies to SOMA and the G1. Their engine classes and pose layouts differ, but the calls are equivalent.

Each generation produces a short transition. The [Runtime API](runtime_api.md#generation-flow) explains how repeated generations become continuous motion.

## Code map

The model-facing code is concentrated in the following symbols in [`app/inbetweener_demo.py`](../app/inbetweener_demo.py):

| Symbol | Responsibility |
| --- | --- |
| `PoseEntry` | Stores one source clip, sample time, world placement, target settings, and four-pose sample window for the demo. |
| `sample_pose_window()` | Samples four poses at 30 Hz into a caller's array and places their root in world space. |
| `InbetweenerDemo.use_rig()` | Closes the previous rig's engine, creates one for SOMA or the G1, and resets the app to that rig. |
| `InbetweenerDemo._prepare_rig()` | Creates the reusable context token, target token, and output buffer, then sets up Start and the first target. |
| `InbetweenerDemo.refresh_poses()` | Resamples Start and every target, updates their previews and velocities, and returns to Start. |
| `InbetweenerDemo.reset()` | Loads Start's four poses into the context token. |
| `InbetweenerDemo.generate_next_segment()` | Loads the next target and calls `engine.generate_animation()` with the target's allowed token counts. |
| `InbetweenerDemo.step()` | Plays the current segment up to its transition point. There, it copies the last four played poses into the next context and moves to the next target. For each target, the first call runs `generate_next_segment()` instead. |
| `InbetweenerDemo._draw_foot_contacts()` | Samples the foot contacts of the generated motion and draws them on the character's soles. |

The remaining functions implement the interface, rendering, picking, and file dialogs. They show one authoring environment. An integration can replace them with game state, procedural targets, network input, or another animation tool.

## Initialize the rig and engine

`use_rig()` closes the previous rig's engine before it creates one for the selected rig, so only one engine holds GPU memory:

```python
if self.engine is not None:
    self.engine.synchronize()
    ...
    self.engine.close()
    self.engine = None
self.rig, self.engine = rig, create_engine(self.viewer, rig, self.fp16)
```

It then calls `_prepare_rig()`. That function uses `load_rig_assets()` to obtain a `RigAssets` with:

- The SOMA character or the G1 robot.
- A `ClipRegistry` for the rig.
- A reference pose with the correct buffer layout.
- The transform from source coordinates to world space.

`_prepare_rig()` creates the context token, target token, and output buffer each time a rig is selected. Every generation with that rig reuses these buffers:

```python
if rig is ModelRig.SOMA:
    skeleton = assets.model.skeleton
    self.context = engine.create_input(skeleton)
    self.target = engine.create_input(skeleton)
    self.output = engine.create_output(skeleton)
else:
    self.context = engine.create_input()
    self.target = engine.create_input()
    self.output = engine.create_output()
```

For SOMA, each pose holds one local joint transform per skeleton joint. For the G1, each pose holds 36 joint coordinates: the root position, the root's XYZW quaternion, and 29 joint angles. See the [SOMA](soma_model.md) and [Unitree G1](unitree_g1_model.md) model cards for the complete contracts. The Runtime API covers [pose layouts and model space](runtime_api.md#pose-layouts-and-model-space).

## Represent poses in the demo

`PoseEntry` is app state for this demo. The MotionBricks library does not need it. Each `PoseEntry` records:

- A clip and a sample time.
- A root placement in world space.
- One four-pose sample window.
- How many target poses to send to the model, set by **Poses to Use**.
- How many frames earlier the transition point moves, set by **Transition Early By (frames)**.
- The smallest and largest token count the model may choose, set by **Tokens to Generate**.
- Visualization state for the viewer.

The list always starts with Start. The targets follow it.

### Sample the four-pose window

`sample_pose_window()` converts a clip sample into the format that the model expects. It fills the `poses` array that the caller passes. It returns nothing. The keyword-only `anchor` argument picks the pose that sits at the sample time:

```python
def sample_pose_window(
    entry: PoseEntry, model_to_runtime: wp.transform, poses: wp.array, *, anchor: int,
) -> None:
    ...
    for frame in range(MODEL_POSES_PER_TOKEN):
        sample_time = entry.sample_time + (frame - anchor) / MODEL_SAMPLE_RATE
        entry.clip.sample(sample_time, poses[frame], root_offset=root_offset)
```

`MODEL_POSES_PER_TOKEN` is four, and `MODEL_SAMPLE_RATE` is 30 Hz.

- For Start, `refresh_poses()` passes `anchor=3`, so the selected sample is the last pose of the window. The three earlier poses describe the motion entering Start.
- For a target, it passes `anchor=0`, so the selected sample is the first pose of the window. Up to three following poses describe the desired motion after that target.

The [guided tutorial](guided_tutorial.md#inspect-the-target-and-context-poses) shows this difference. With **Show Skeleton(s)** on, the view shows the four poses of Start that end at its sample time. For the End target, it shows the **Poses to Use** poses that begin at its sample time. The `anchor` argument implements that difference.

### Place a pose in the scene

The demo removes the sampled root's ground offset and applies the pose's placement, which the gizmo and the **Transform** fields edit. It also converts source coordinates to world space:

```python
ground_offset = wp.transform(
    wp.vec3(-float(root[0]), -float(root[1]), 0.0),
    wp.quat_identity(),
)
root_offset = wp.mul(entry.placement, wp.mul(ground_offset, model_to_runtime))
```

Only the root placement changes. SOMA local joint transforms and G1 joint angles stay as sampled. For this reason, the [tutorial](guided_tutorial.md#add-the-intermediate-targets) could move and rotate Entry and Duck-under with the gizmo without re-authoring their body poses.

Locked axes keep their value. `apply_transform_locks()` applies the locks to gizmo edits, and the **Transform** fields skip locked axes.

`model_to_runtime` converts SOMA's Y up source coordinates to Z up world space. The G1 conversion is the identity. [Pose layouts and model space](runtime_api.md#pose-layouts-and-model-space) explains why the engine and clip sampling take the same transform.

## How tutorial actions update model inputs

### Select a clip and frame

Choosing a clip and a frame, such as the avoidance BVH clip at frame 199 for Entry in the [tutorial](guided_tutorial.md#assign-the-clip-and-frames), sets `PoseEntry.path`, `PoseEntry.clip`, and `PoseEntry.sample_time` in `_choose_pending_clip()` and `_draw_properties()`. Every change calls `refresh_poses()`:

```python
for index, entry in enumerate(self.pose_entries):
    anchor = MODEL_POSES_PER_TOKEN - 1 if index == 0 else 0
    entry.sampled_poses = wp.empty_like(self.context.poses)
    sample_pose_window(entry, self.model_to_runtime, entry.sampled_poses, anchor=anchor)
```

An integration does not need clips. It can load any finite poses with the correct SOMA or G1 layout into the context and target tokens. Clips are only a convenient source of target poses for this demo. For example, [Kimodo](https://github.com/nv-tlabs/kimodo) or [ARDY](https://github.com/nv-tlabs/ardy) could generate them instead.

### Add and order targets

**+** calls `add_target()`, as in the [tutorial](guided_tutorial.md#add-the-intermediate-targets). It copies the clip and sample time of the selection, places the target 1.5 m from the last target along -Y with the same rotation, and names it after its ID, as in **Target 2**. Dragging a target in the list calls `move_target()`.

The list order controls playback. `next_target_index` advances through `pose_entries`. Each target produces one generation. Start stays at index zero because it provides the first context rather than a target.

### Set Poses to Use

**Poses to Use** sets how many of a target's four poses inform the model. It decides whether a target sets only a posture or also describes motion:

```python
self.target.update_from_poses(entry.sampled_poses[:entry.num_poses_to_use], stream=self.engine.stream)
```

The accepted range is 1 to 4:

- One pose sets the destination. The model treats it as a stop with no velocity.
- A second pose adds velocity.
- A third and fourth pose add acceleration and momentum.

Start always provides all four context poses. No generation ends at Start. For this reason, **Poses to Use**, **Transition Early By (frames)**, and **Tokens to Generate** are disabled for Start.

### Set Transition Early By

**Transition Early By (frames)** lets the sequence move toward the next target before the current segment reaches its last frame, as in the [tutorial](guided_tutorial.md#assign-the-clip-and-frames):

```python
self.segment_end_frame = max(
    MODEL_POSES_PER_TOKEN - 1, self.output.num_frames - 1 - entry.transition_early_by_frames,
)
```

This setting does not change the target sent to the model. It moves the transition point earlier, so less of the generated motion plays before the next generation. The context for that generation therefore ends at the transition point. At least four frames always play, because the next context is the last four played frames.

This makes a target useful as a guide at runtime. The character heads toward it and moves on before it arrives. Use such a target to steer and place the character before it generates toward a more precise interaction pose, as Entry does before Duck-under in the [tutorial](guided_tutorial.md#assign-the-clip-and-frames).

### Set Tokens to Generate

**Tokens to Generate** sets the smallest and largest token count that the model may choose for the generation toward this target. Each motion token holds four raw frames. The **Raw frames:** readout shows the matching range. `generate_next_segment()` passes the range to the engine as allowed token counts:

```python
config=MotionBricksInferenceConfig(
    allowed_token_counts=tuple(range(entry.min_tokens, entry.max_tokens + 1)),
),
```

Each target starts with the range in [`DEFAULT_TOKEN_COUNTS`](../motionbricks/model/config.py), 8 to 14 tokens. `MotionBricksInferenceConfig` uses the same default. The model contract allows 6 to 16. Equal **Min** and **Max** values fix the length of the transition. A wide range lets the model choose a duration that suits the distance and the motion.

### Click Generate

The **Generate** button calls `play()`. The button is disabled until the list holds at least one target. `play()` first calls `reset()` and then starts playback:

```python
def play(self) -> None:
    self.reset()
    if len(self.pose_entries) > 1:
        self.playing = True
```

Every preview therefore begins from the Start context and moves through the current list of targets in order.

## One generation

`generate_next_segment()` performs one generation:

```python
entry = self.pose_entries[self.next_target_index]
self.target.update_from_poses(entry.sampled_poses[:entry.num_poses_to_use], stream=self.engine.stream)
self.engine.generate_animation(
    context=self.context, target=self.target, output=self.output,
    config=MotionBricksInferenceConfig(
        allowed_token_counts=tuple(range(entry.min_tokens, entry.max_tokens + 1)),
    ),
)
```

Either `reset()` or the previous segment's `step()` loaded the context. The target is the next one in the list. `generate_animation()` fills the reusable output buffer and sets `output.num_frames` to the number of valid frames.

The output buffer is allocated for the largest supported result. Only the first `output.num_frames` frames are valid for the current generation. The demo derives the segment's transition point from that value and the target's **Transition Early By** setting.

All GPU operations are ordered on `engine.stream`. The main loop enters `wp.ScopedStream(self.engine.stream)`, and individual buffer operations also receive `stream=self.engine.stream`. An integration should preserve this ordering. It should also synchronize the engine before it reads GPU results on the CPU.

### Show foot contacts

The **Show Foot Contacts** checkbox next to **Generate** draws the model's foot contacts for the generated motion. The slider beside it sets the threshold. `_draw_foot_contacts()` samples the contacts at the playback time with `output.sample_foot_contacts()`. It then outlines each sole whose contact is above the threshold. [foot_contacts.py](../motionbricks/animation/foot_contacts.py) defines the order and meaning of the four values.

## Playback and chained generation

The complete sequence follows three functions.

### 1. `reset()`: seed the first context

```python
start = self.pose_entries[0]
self.context.update_from_poses(start.sampled_poses, stream=self.engine.stream)
wp.copy(self.current_pose, start.preview_pose, stream=self.engine.stream)
```

Start's four poses become the context. The current displayed pose returns to the last pose of that window.

### 2. `generate_next_segment()`: generate toward one target

The demo copies the next target's poses to the target token and calls `generate_animation()`. It records the segment's transition point and samples its first pose.

### 3. `step()`: play the segment and preserve continuity

```python
end_time = self.segment_end_frame / MODEL_SAMPLE_RATE
self.playback_time = min(self.playback_time + delta_time, end_time)
self.output.sample(self.playback_time, self.current_pose, stream=self.engine.stream)
```

`step()` samples the generated motion at the app's update rate until it reaches the transition point. `run()` advances playback by the time since the last frame, capped at 0.1 s. On slower hardware, playback slows down instead of skipping ahead.

Before `step()` advances to the next target, it copies the last four poses that actually played:

```python
played_frames = self.segment_end_frame + 1
recent_poses = self.output.poses[played_frames - MODEL_POSES_PER_TOKEN:played_frames]
self.context.update_from_poses(recent_poses, stream=self.engine.stream)
```

In the [guided tutorial](guided_tutorial.md#generate-the-final-animation), the generated sequence moved from Start to Entry, Duck-under, and End. After each segment, `step()` made the last four played poses the context for the next target. This feedback loop joins independent generations into continuous motion.

## SOMA and Unitree G1

The **Character** selector shows that the integration flow does not depend on the rig:

| | SOMA | Unitree G1 |
| --- | --- | --- |
| Source clips | BVH | CSV |
| Engine input | `PoseToken` holding local joint transforms | `DOFToken` holding joint coordinates |
| Generated output | `AnimationBuffer` holding local joint transforms | `DOFAnimationBuffer` holding joint coordinates |
| Conversion to world space | `model_to_runtime` converts Y up source coordinates to Z up world space | Identity |
| Rendering in this demo | `SkeletalMeshRenderer` | `DOFMeshRenderer` |

When you select another rig, the demo defers the switch until the start of the next frame. `use_rig()` then closes the current engine and creates an engine for the selected rig. It calls `_prepare_rig()`, which creates the rig's buffers and resets the list to Start and one target.

The TensorRT plan cache keeps a switch back short: it loads the plan instead of building it again. A larger application that keeps several engines should key them by rig, precision, and any other build option that affects the TensorRT plan.

## Use this pattern in an integration

The Inbetweener runs the generation loop by hand, so each call stays visible. The [generation flow](runtime_api.md#generation-flow) in the Runtime API page lists the steps of the loop. For a larger real-time application, use `GeneratedMotionPlayer` or `GeneratedDOFMotionPlayer` from [motion_player.py](../motionbricks/runtime/motion_player.py) as examples. [Chaining generations](runtime_api.md#chaining-generations) describes them.

## Further reference

- [Guided tutorial](guided_tutorial.md): the user-facing workflow that uses this app.
- [Runtime API](runtime_api.md): engines, pose layouts, clips, generation settings, chaining, TensorRT builds, and logging.
- [ONNX inference and batching](onnx_inference.md): direct ONNX interfaces, batching, and alternative runtimes.
- [SOMA model card](soma_model.md): SOMA inputs, outputs, coordinates, and limitations.
- [Unitree G1 model card](unitree_g1_model.md): G1 joint layout, inputs, outputs, and limitations.
