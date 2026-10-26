# Smart Primitive Editor

A Smart Primitive describes an action as target poses placed around an object. The character does not play the action as a clip. MotionBricks generates motion through the target poses, starting from whatever the character is doing, so one Smart Primitive works from many approaches and speeds. The paper calls these smart objects when they describe interactions, and smart locomotion when they drive navigation.

In this repository, Smart Primitives define the interactions in Scene Preview, such as jumping, ducking, and sitting. They also hold the blend targets of every locomotion mode. The Smart Primitive Editor authors them.

<p align="center">
  <img src="media/smart_primitive_editor.png" alt="Smart Primitive Editor with the jump_0_5m Smart Primitive" width="100%">
</p>

```sh
uv run python -m app.smart_primitive_editor
```

Add `--verbose` for detailed log messages.

> **Note:** The Smart Primitive Editor is an example of how to author behaviors and interactions for the model. It is not a production tool. Known limitation: it has no undo or redo, so save your work often.

[Part 2 of the guided tutorial](guided_tutorial.md#part-2-build-a-smart-primitive) builds an avoidance Smart Primitive step by step. This page describes what each part of a Smart Primitive holds and how Scene Preview uses it.

## Controls

Use the **File** menu to create, open, and save documents. The editor has three panels: **Hierarchy**, **Sequences**, and **Properties**. Their controls are described with the parts they edit.

| Action | Control |
| --- | --- |
| Move the camera | **W**, **A**, **S**, and **D**, or the **arrow keys**. **Q** and **E** move it down and up. |
| Orbit around the camera pivot | Drag with the **middle mouse button** |
| Look around | Drag with the **left mouse button** |
| Pan | **Shift** + drag with the **middle mouse button** |
| Move toward or away from the camera pivot | **Ctrl** + drag with the **middle mouse button**, or the **mouse wheel** |
| Change the field of view | **Ctrl** + **mouse wheel** |
| Select a node | Click its pose or interaction box in the view, or click it in **Hierarchy** |
| Move or turn the selection | Drag the gizmo, or edit the **Transform** fields |
| Reset or lock a transform | Right click **Translation** or **Rotation** to reset it. Right click an axis label to lock that axis. |

The **Transform** values and the gizmo handles use the parent node's axes.

## What a Smart Primitive holds

A Smart Primitive is a `.smart_primitive.json` document for one rig, SOMA or Unitree G1. It holds:

- **Nodes** in a transform hierarchy under the root node.
- **Components** on the nodes: a **Target Pose**, an **Interaction Box**, or a **Pivot**. A node holds at most one of each kind.
- **Sequences** that list target pose nodes in the order the character reaches them.

The avoidance Smart Primitive from the tutorial shows how they fit together:

```text
Root
├── Interaction box       Interaction Box: where the interaction can start
└── Pivot                 Pivot: turns and slides the sequence toward the approach
    ├── Entry pose        Target Pose: a walking pose before the bar
    └── Duck-under pose   Target Pose: the pose under the bar

Sequences
  Sequence 1   Entry pose, Duck-under pose   Pivot: Pivot
```

The bundled documents are in [assets/smart_primitives](../assets/smart_primitives):

| Document | What it shows |
| --- | --- |
| `jump_0_5m` | A small jump with two variations |
| `jump_0_75m` | A vault with three variations, two with a hand on the cube |
| `avoidance` | Ducking under a bar, with walk and jog variations |
| `sit` | Sitting down, holding, and standing up |
| `locomotion_bs1_linear` | Idle, Walk, Jog, and Run targets for Forward |
| `locomotion_bs2_directional` | Idle, Forward, Backward, Left, and Right targets for Strafing |
| `locomotion_bs2_directional_crouch` | Idle, Forward, Backward, Left, and Right targets for Crouching |
| `locomotion_style_variations_bs1_linear` | Idle and six style targets, such as Skipping and Zombie, for the style modes |

## Node hierarchy

Each node has a name and a local transform relative to its parent. Moving a node moves everything below it. The root node stands for the Smart Primitive itself. A scene places each instance by moving its root, and the other nodes keep their authored placement relative to it.

The hierarchy also decides what moves together. Put the target poses that a pivot should move under the pivot node. Place interaction boxes and target poses around the object where the action happens.

In **Hierarchy**:

- **+** adds a child to the selected node. **-** deletes the selected node with its children, and the sequences drop their steps. Right click a node for the same actions.
- Double click a node to rename it. Drag it onto another node to change its parent.
- The checkbox hides a node and its children in the view.
- The **Character** selector sets the rig. It is locked once a target pose has a clip.

[Create the node hierarchy](guided_tutorial.md#create-the-node-hierarchy) in the tutorial builds the avoidance hierarchy.

## Components

To add a component, select a node, right click an empty area of **Properties**, and choose **Add** > **Components**. **Referenced by** lists the sequences that use the node.

### Target Pose

A Target Pose samples four poses at 30 Hz from a clip, starting at the chosen time. When a sequence uses the target, only the root of these poses moves to the node's world placement. Joint values stay as sampled, so one sampled target serves every placement.

| Setting | Meaning |
| --- | --- |
| Clip | The BVH clip for SOMA or the CSV clip for Unitree G1 |
| Seconds or Frames | The sample time. |
| Constrain Root | Moves the clip's root to the node's origin along the ground and keeps its height. Without it, the pose stays where it was in the clip, relative to the clip's origin. |
| IK Pins | SOMA only. Pulls the chosen hands or feet onto the target pose over the second half of the step. See [IK Pins](#ik-pins). |
| Poses to Use | How many of the four poses inform the model. See [Set Poses to Use](inbetweener.md#set-poses-to-use). |
| Transition Early By (frames) | Moves on to the next step this many frames before the generated motion ends. |
| Tokens to Generate | Limits the length the model can choose for the step. A Target Pose starts at 6 to 14 tokens, and the model allows 6 to 16. Set both values to the same number to request a fixed length. The **Raw frames:** readout shows the matching frame range. |

**Poses to Use**, **Transition Early By**, and **Tokens to Generate** work as they do for [Inbetweener targets](inbetweener.md#set-poses-to-use), except that the default token counts differ. **Speed** shows how fast the sampled poses move. The first target of a sequence sets the sequence's start speed. **Show Mesh**, **Show Skeleton(s)**, **Show Velocity**, and **Show Trajectory** change only the view.

[Configure the target poses](guided_tutorial.md#configure-the-target-poses) in the tutorial sets up the Entry and Duck-under targets.

#### IK Pins

**IK Pins** lists the left and right hands and feet. Pin one when it must land exactly where the target pose puts it. For example, two targets of the SOMA vault, `jump_0_75m`, pin the left hand.

During playback, two-bone IK pulls each pinned hand or foot to where it is in the step's first target pose. The pull starts halfway through the generated step and reaches full strength at its transition point. Each generation replaces the pins, so a step without pins clears them. They also clear when the sequence ends.

The character records the corrected pose, so the next generation starts from where the hand or foot actually was. Only SOMA has target pose IK. `TargetPoseIKCorrector` in [target_pose.py](../motionbricks/runtime/target_pose.py) implements it.

### Interaction Box

**Width**, **Depth**, and **Height** size the box around the node. In Scene Preview, an interaction can start when the character's probe box overlaps any of the Smart Primitive's interaction boxes. The boxes only start interactions. They are not collision geometry for the model.

A Smart Primitive can have several interaction boxes, one on each of several nodes. The probe box of a crouching character is shorter, so a raised interaction box only catches a standing character. The bundled `avoidance` document raises its box for this reason.

[Configure the Interaction Box](guided_tutorial.md#configure-the-interaction-box) in the tutorial sizes and places the avoidance box.

### Pivot

A pivot makes one sequence work from many approaches. When a sequence starts, its pivot node moves and turns toward the character's approach, and every target below it follows. Only the roots of the target poses move.

![A character arrives from the side, without a pivot and with the tutorial Pivot](media/tut_pivot_00_problem.gif)

*A character arrives at 90 degrees to the authored sequence. Without a pivot, on the left, it walks to the authored Entry pose and turns. With the tutorial Pivot, on the right, the target poses turn and slide onto its course.*

| Setting | Meaning |
| --- | --- |
| Motion Direction | The pivot's local direction that should line up with the approach |
| Translation | Slides the pivot along one local axis. **Line Projection** keeps it within the **Offsets**, measured from where it was authored. |
| Rotation | Turns the pivot around one local axis. **Cone** limits the turn to **Half Angle** on each side of the cone's center. **Cone Side** sets that center: **Motion Direction**, **Mirrored** for the opposite direction, or **Both** for either. |
| Operation Order | **Translation -> Rotation** slides first. **Rotation -> Translation** turns first, which also turns the line the pivot slides along. |

**Show Motion Direction**, **Show Translation Constraint**, and **Show Rotation Constraint** draw these settings in the view:

![Authored layout of the tutorial Pivot, top view](media/tut_pivot_01_layout.png)

*The tutorial Pivot from above. The yellow arrow is the Motion Direction, and the red dashed line is the range of the Line Projection.*

The pivot aligns once, when the sequence starts. Its limits are measured from the authored placement, never from a previous alignment, so a Smart Primitive does not drift over repeated interactions. A sequence without a pivot keeps its current placement.

[Configure the Pivot](guided_tutorial.md#configure-the-pivot) in the tutorial sets up a pivot. [Understanding the Pivot](guided_tutorial.md#understanding-the-pivot) shows how each setting behaves, compares six approaches, and explains why the aligned poses continue the incoming motion.

## Sequences

A sequence lists target pose nodes in the order the character reaches them. Each step generates toward one target, with that target's settings. A sequence can name a pivot node, which aligns the whole sequence to the approach before it starts.

Several sequences can share nodes. That is how one document holds variations of an action, such as the Walk and Jog sequences of the bundled `avoidance` document. The behavior that a scene chooses for the instance picks the sequence:

- `VelocityMatchingSequenceBehavior` picks the sequence whose start speed is closest to the character's speed.
- `SittingBehavior` plays the sequences named Sit and Stand.

[Scene Preview](scene_preview.md#smart-primitive-behaviors) describes the behaviors and how to write one.

In **Sequences**:

- **+** adds a sequence and starts renaming it. **-** deletes the selected one. Right click a sequence to rename or delete it.
- **Pivot** sets the sequence's pivot node, or **None**.
- **+ Add Target Pose** adds a step. You can also drag a node from **Hierarchy** onto the sequence header. Only nodes with a **Target Pose** become steps.
- Drag steps to reorder them, and click **x** to remove one.

[Define the sequence](guided_tutorial.md#define-the-sequence) in the tutorial orders the Entry and Duck-under steps. [Explore variations](guided_tutorial.md#explore-variations) adds more sequences to the same document.

## How a scene places the poses

For each step, Scene Preview:

1. Lets the behavior pick a sequence and align its pivot to the character's approach.
2. Computes each target node's world transform from the instance's placement in the scene and the node hierarchy.
3. Moves the root of the node's four sampled poses by that transform. Joint values stay as sampled.
4. Generates from the character's recent motion toward those poses.

Because only the root moves, one set of sampled poses serves every placement. The same document works for any instance in any scene, and for both jump directions of the same cube. The [Scene Editor](scene_editor.md) stores each instance's placement.

## Example: a sitting hold

A hold keeps the character in a pose until something happens. The sitting hold in [sit.smart_primitive.json](../assets/smart_primitives/soma/sit.smart_primitive.json) combines two sequences with a looping clip:

```text
Root
├── InteractionBox        Interaction Box
└── Pivot                 Pivot, turns around Z to face the approach
    └── Sit               Target Pose: sit_on_chair_loop, frame 0, 1 pose, Constrain Root
        ├── ToStand       Target Pose: sit_on_chair_stop at 1.99 s, 4 poses
        └── ToSit         Target Pose: sit_on_chair_start at 1.85 s, 4 poses

Sequences
  Sit     ToSit, Sit      Pivot: Pivot
  Stand   ToStand         Pivot: None
```

1. **Sit** first reaches ToSit, a pose from the sitting clip just before the character sits. Then it reaches Sit, which is the first frame of the looping clip. The pivot turns both targets toward the approach, so the character turns around and sits from any side.
2. When Sit completes, `SittingBehavior` plays [sit_on_chair_loop](../assets/motions/soma/sitting/sit_on_chair_loop.bvh) as a loop, aligned to the reached pose. Because the Sit target is the loop's first frame, the handoff from generated motion to the clip is smooth.
3. When you press **I** (**X** on a controller), the behavior plays **Stand**, which generates toward a standing pose from the stand-up clip. Stand has no pivot, so it keeps the placement chosen by Sit.
4. When Stand completes, the character returns to locomotion or idle.

<p align="center">
  <img src="media/sit_hold_soma.gif" alt="SOMA sits down, holds, stands up, and walks off" width="80%">
</p>

To build another hold, such as leaning on a wall:

1. Prepare a looping clip of the held pose.
2. Author an entry sequence that ends on the looping clip's first frame with **Constrain Root**, and an exit sequence that returns to a standing pose.
3. Add a pivot to the entry sequence so the hold adapts to the approach.
4. Write a behavior class like `SittingBehavior`. [Write a behavior](scene_preview.md#write-a-behavior) in Scene Preview shows an example.
5. Choose that class as the scene instance's **Behavior** in the [Scene Editor](scene_editor.md).

## Document format

Documents are plain JSON:

| Field | Contents |
| --- | --- |
| `format`, `schema_version` | `motionbricks.smart_primitive.authoring` and 1 |
| `name`, `rig` | The document name and `soma` or `unitree_g1` |
| `root` | The node tree. Each node has an `id`, a `name`, a `local_transform`, its `components`, and its `children`. |
| `sequences` | Each sequence has an `id`, a `name`, its `node_ids` in order, and a `pivot_node_id` or `null` |
| `editor` | The selected node and the camera. The editor restores them. |

Clip paths are relative to the document. Opening a document checks its rig against the clips. Scene Preview reads the same files, so there is no export step.

**File** > **Save** and **Save As...** write the document. Each component is saved with its `type`, such as `target_pose`, `interaction_box`, or `pivot`, and its settings. A Target Pose saves its clip and sample time, not the sampled poses. Loading samples the poses again from the clip.

## Further reference

- [Guided tutorial](guided_tutorial.md#part-2-build-a-smart-primitive): building the avoidance Smart Primitive step by step.
- [Scene Editor](scene_editor.md): placing Smart Primitive instances in a scene.
- [Scene Preview](scene_preview.md): the behaviors and states that play Smart Primitive sequences.
- [Runtime API](runtime_api.md): target settings, motion players, and chaining.
