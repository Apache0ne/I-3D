# Scene Editor

A scene brings Smart Primitives together into a space the character can play through. It places Smart Primitive instances and Solid Cubes, sets where the character starts, and names the behavior that runs each instance. [Scene Preview](scene_preview.md) opens the saved `.scene.json` document and plays it.

<p align="center">
  <img src="media/scene_editor.png" alt="Scene Editor with the SOMA gym" width="100%">
</p>

```sh
uv run python -m app.scene_editor
uv run python -m app.scene_editor assets/scenes/soma_gym.scene.json
```

The optional argument opens a scene at startup. Add `--verbose` for detailed log messages.

> **Note:** The Scene Editor is an example of how to author behaviors and interactions for the model. It is not a production tool. Known limitation: it has no undo or redo, so save your work often.

[Part 3 of the guided tutorial](guided_tutorial.md#part-3-assemble-an-interaction-scene-with-avoidance-jump-and-sit) assembles an interaction course step by step. This page describes what a scene holds and how Scene Preview uses it.

## Controls

Use the **File** menu to create, open, and save scenes. The editor has two panels: **Scene** and **Properties**. Their controls are described with the parts they edit.

| Action | Control |
| --- | --- |
| Move the camera | **W**, **A**, **S**, and **D**, or the **arrow keys**. **Q** and **E** move it down and up. |
| Orbit around the camera pivot | Drag with the **middle mouse button** |
| Look around | Drag with the **left mouse button** |
| Pan | **Shift** + drag with the **middle mouse button** |
| Move toward or away from the camera pivot | **Ctrl** + drag with the **middle mouse button**, or the **mouse wheel** |
| Change the field of view | **Ctrl** + **mouse wheel** |
| Select an item | Click it in the view, or click it in the **Scene** tree |
| Move or turn the selection | Drag the gizmo, or edit the **Transform** fields |
| Reset or lock a transform | Right click **Translation** or **Rotation** to reset it. Right click an axis label to lock that axis. |

The **Transform** values and the gizmo handles use the parent item's axes.

## What a scene holds

A scene is a `.scene.json` document for one rig, SOMA or Unitree G1. It holds:

- One CharacterStart: where the character appears and which way it faces.
- Smart Primitive instances: each places one Smart Primitive document in the scene and names the behavior that runs it.
- Solid Cubes: scenery, such as the box under a jump.

The SOMA gym shows how they fit together:

```text
CharacterStart
SmallJumpSP     jump_0_5m, VelocityMatchingSequenceBehavior
└── Box         Solid Cube to jump onto
MidJumpSP       jump_0_75m, VelocityMatchingSequenceBehavior
└── Wall        Solid Cube to vault over
SitSP           sit, SittingBehavior
└── Seat        Solid Cube to sit on
AvoidanceSP     avoidance, VelocityMatchingSequenceBehavior
└── Bar         Solid Cube to duck under
```

The bundled scenes are in [assets/scenes](../assets/scenes):

| Scene | What it shows |
| --- | --- |
| `soma_gym` | A small jump, a vault, a box to sit on, and a bar to duck under, for SOMA |
| `g1_gym` | The same layout for the Unitree G1, scaled to the robot's size |

## Scene tree

The **Scene** panel lists the items. Parents only help editing: moving a Smart Primitive moves the cubes under it. Scene Preview places every item by its world transform.

In **Scene**:

- **+** adds a Smart Primitive or a Solid Cube. **-** deletes the selected item and its children. Right click an item to delete it.
- Double click an item to rename it. Drag it onto a Smart Primitive or a Solid Cube to make it a child. To move it back to the top level, drop it below the tree or choose **Move to Root** in its right click menu.
- CharacterStart stays at the top level. It cannot be renamed, deleted, or moved under another item.
- The **Character** selector sets the scene's rig. While the scene has no references, it switches between SOMA and Unitree G1. To change the rig later, delete the Smart Primitives that hold references, or start a new scene.

## Items

Select an item to edit it in **Properties**.

### CharacterStart

CharacterStart sets where the character appears when Scene Preview opens the scene. Its arrow shows the direction the character faces. Every scene has exactly one.

### Smart Primitive

| Setting | Meaning |
| --- | --- |
| Reference | The `.smart_primitive.json` document to place. The first reference sets the scene's rig. Later references must use the same rig. |
| Show Target Poses | Draws the document's target poses in the scene: all sequences, or the one chosen beside it, in the chosen color |
| Show Interaction Boxes | Draws the document's interaction boxes |
| Behavior | Under **Runtime**, the class that runs the instance in Scene Preview, such as `VelocityMatchingSequenceBehavior` or `SittingBehavior`. The list shows every `SmartPrimitiveBehavior` subclass that is imported. |

Several instances can reference the same document. Each instance has its own placement and behavior. A Smart Primitive item starts with no reference and an empty **Behavior**. Scene Preview refuses a scene until every instance has both. [Write a behavior](scene_preview.md#write-a-behavior) in Scene Preview shows how to add a class to the list.

[Add the avoidance Smart Primitive](guided_tutorial.md#add-the-avoidance-smart-primitive) in the tutorial places an instance and chooses its behavior. [Complete the course](guided_tutorial.md#complete-the-course) adds the jump and the sit.

### Solid Cube

**Width**, **Depth**, **Height**, and **Color** set the cube. Cubes are scenery, such as the box under a jump. They are not collision geometry for the model. Put a cube under its Smart Primitive so the two move together.

## How Scene Preview uses a scene

Save the scene with **File** > **Save**, then open it in Scene Preview:

```sh
uv run python -m app.scene_preview --scene path/to/your.scene.json
```

Scene Preview flattens the item hierarchy into world transforms and checks every instance before it builds an engine. It then moves the root node of each instance's Smart Primitive to the instance's placement, and starts the character at CharacterStart. [Scenes](scene_preview.md#scenes) in Scene Preview describes the loading.

## Document format

Scene documents are plain JSON:

| Field | Contents |
| --- | --- |
| `format`, `schema_version` | `motionbricks.scene` and 1 |
| `name`, `rig` | The scene name, taken from the file name, and the rig, `soma` or `unitree_g1` |
| `character_start` | ID 1, its `transform`, and its `euler_degrees` |
| `smart_primitives` | Each instance has an `id`, a `name`, a `parent_id` or `null`, a `transform`, its `euler_degrees`, a `reference`, its `visibility`, a `target_pose_color`, and a `behavior` class name |
| `solid_cubes` | Each cube has an `id`, a `name`, a `parent_id` or `null`, a `transform`, its `euler_degrees`, its `dimensions`, and a `color` |
| `editor` | The selected item and the camera. The editor restores them. |

A `transform` holds a `translation` and a `rotation_xyzw` quaternion. It is relative to the parent item, or to world space at the top level. A `reference` is relative to the scene file, or absolute when the files are on different drives. Another program can write these files. It must use one rig for every referenced Smart Primitive.

## Further reference

- [Guided tutorial](guided_tutorial.md#part-3-assemble-an-interaction-scene-with-avoidance-jump-and-sit): assembling the tutorial scene step by step.
- [Smart Primitive Editor](smart_primitive_editor.md): authoring the Smart Primitive documents that a scene references.
- [Scene Preview](scene_preview.md): how a scene loads and how its behaviors run.
