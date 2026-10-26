# MotionBricks guided tutorial

This tutorial introduces the core concepts of MotionBricks and applies them to a complete SOMA workflow. The exercises progress from a single generation to a fully interactive scene:

1. **Fundamentals:** use the Inbetweener to rebuild a walking duck-under interaction from target poses.
2. **Smart Primitive:** turn the same poses into a reusable, pose-based interaction.
3. **Scene:** assemble several Smart Primitives into an interaction course.
4. **Scene Preview:** play the result in real time and inspect how it works.

All commands and file paths in this tutorial are relative to the repository root.

![tutorial results](media/tutorial_final_results.gif)

*Final result of the guided tutorial.*

> **Unitree G1:** The same workflow applies to the Unitree G1. Choose **Unitree G1** in the **Character** selector. Use G1 CSV clips and G1 Smart Primitive documents instead of the SOMA ones.


## High-level concepts

### Motion generation between poses

MotionBricks generates a transition from the character's recent motion to one or more target poses.

![Four context poses in green, generated poses, and a target pose in blue](media/core_idea.gif)

Each generation uses the following inputs:

- **Four context poses:** the character's most recent motion. Together, they describe the current pose and the velocity with which the character enters the transition.
- **One to four target poses:** the poses that the character must reach. A single target pose sets the posture, position, and facing where the character stops. A second pose adds velocity at the target. A third and fourth pose add acceleration and momentum, so the character keeps moving through the target.

The model determines the duration of the transition. It generates the root trajectory and the body motion. It also generates the foot placement and the changes in momentum required to reach the target.

The host application remains responsible for deciding **what should happen next**. MotionBricks determines **how the character moves from its current motion to that goal**.

This tutorial uses four example apps. They show how to integrate MotionBricks. They are not a production toolset. The tutorial explains how the model works. It also shows where the model's inputs and outputs fit in an animation system. With that understanding, developers can integrate MotionBricks into their own engines, tools, and pipelines.

### Key poses instead of transition clips

Traditional clip-based systems require authored transitions between many combinations of states. These clips must be selected, synchronized, and blended. With MotionBricks, the state machine selects target poses and places them in the scene instead.

![SOMA locomotion driven by context poses, target poses, and a blend space](media/locomotion_targets_soma.gif)
*Context poses are shown in purple, current target poses in blue, and the live blend space result in green.*

### Chaining generations

A single generation is intentionally short. Continuous motion comes from a loop. The application generates a transition and plays it up to the transition point. It records each pose it shows. The last four played poses become the context poses of the next generation.

![chaining animation generation](media/target_b_flow.png)

This loop is what we mean by autoregressive. Each generation starts from the motion that the previous generation produced. The model never sees the whole animation at once. It only sees the last four played poses and the next target.

This is the central integration pattern of both the Inbetweener and Scene Preview. Generated motion, locomotion changes, and transitions to or from clips can all contribute to the next context.

## Part 1: Generate an animation in the Inbetweener

The Inbetweener is the smallest complete example of the MotionBricks generation loop. It samples poses from clips and lets you place those poses in the view. It then generates motion through them in order.

In this part, you will create the following sequence of poses:

![target poses for avoidance](media/duck_under_flow.png)

### Launch the Inbetweener

Start the Inbetweener by running:

```sh
uv run python -m app.inbetweener_demo --model soma
```

The Newton viewer opens with the Inbetweener.

![Inbetweener opened in the Newton viewer](media/tut_inbetweener_01_open.png)

The interface has two main areas:

- **View:** displays the character, the current poses, and the generated motion.
- **Inbetweener window:** contains the **Character** selector, the **Target Poses** list, and the generation settings. Confirm that the **Character** selector shows **SOMA**.

To navigate, place the pointer over the view and use the following controls:

- **W**, **A**, **S**, and **D**, or the **arrow keys**, move the camera horizontally.
- **Q** and **E** move the camera down and up.
- **Middle mouse drag** orbits around the camera pivot.
- **Shift + middle mouse drag** pans.
- **Ctrl + middle mouse drag** moves toward or away from the camera pivot.
- **Mouse wheel** moves toward or away from the camera pivot. **Ctrl + mouse wheel** changes the camera field of view.

### Create and place the End target

In the **Target Poses** list, click **Target 1** to select it and display its properties:

![Target 1 selected with its properties](media/tut_inbetweener_02_Pose1.png)

Double click **Target 1** and rename it to **End**:

![End pose renamed](media/tut_inbetweener_03_End.png)

In the view, use the move gizmo on the blue pose to move the character along the green (Y) axis:

![move end pose](media/tut_inbetweener_05_EndMove.png)

Move the character slightly further:

![move character](media/tut_inbetweener_04_Move.png)

Click **Generate**:

![Generate button](media/tut_inbetweener_06_generate.png)

This generates your first MotionBricks transition:

![animation generation between two poses](media/tutorial_first_anim.gif)

### Inspect the target and context poses

With the **End** target still selected, turn on **Show Skeleton(s)** and **Show Velocity** under **Visibility**. **Show Mesh** is already on:

![visibility options enabled](media/tut_inbetweener_07_visibility.png)

The view now displays the four poses sampled from the clip for this target. Together, they define the target and its velocity. The red arrow represents the velocity of the target:

![target pose frames](media/tut_inbetweener_08_targetpose.png)

Select **Start** and enable the same options to inspect it:

![context pose frames](media/tut_inbetweener_09_startpose.png)

For **Start**, the view displays the pose at the selected frame and the three poses before it. The poses are sampled at 30 Hz, so in this 120 Hz BVH clip they are every fourth frame. These four poses are the context poses of the generation.

### Add the intermediate targets

Create two new targets by clicking the **+** button below the **Target Poses** list:

![two new targets added](media/tut_inbetweener_10_addposes.png)

Select **Target 2** and **Target 3**, and drag them up in the list:

![new targets moved up in the list](media/tut_inbetweener_11_moveposes.png)

Rename the targets as follows:

- **Entry**
- **Duck-under**

Select the **Entry** target and use the gizmo to place it between **Start** and **End**, at about one third of the distance:

![move entry pose](media/tut_inbetweener_12_movepose.png)

Select the **Duck-under** target and place it at about two thirds of the distance:

![move avoid pose](media/tut_inbetweener_13_movepose.png)

### Assign the clip and frames

Select the **Entry** target and click the **...** button next to its **Clip** field:

![clip selection button](media/tut_inbetweener_14_clip.png)

Browse to and open the following BVH clip:

```text
assets/motions/soma/avoidance/neutral_avoid_obstacle_bend_down_walk_ff_180_R_001__A543.bvh
```

The view now displays the clip on that target:

![new clip for first pose](media/tut_inbetweener_15_newclippose.png)

Below the **Clip** field, click the **Seconds** button to switch to the **Frames** slider:

![seconds button](media/tut_inbetweener_16_seconds.png)

Select frame **199**:

![switch to frames](media/tut_inbetweener_17_frames.png)

> **Note:** Ctrl + click the slider to type a value directly.

In the view, use the rotate gizmo (the largest circle) to orient the target so that it is consistent with the surrounding targets:

![orient pose](media/tut_inbetweener_18_rotate.png)

Apply the following target settings:

![Entry target settings](media/tut_inbetweener_19_settings.png)

- **Poses to Use:** 4, so the poses that start at the selected frame define the velocity, acceleration, and momentum at the target. See [Set Poses to Use](inbetweener.md#set-poses-to-use).
- **Transition Early By (frames):** 8. The transition point moves 8 frames earlier, so the next generation starts before the character reaches this target. This target then acts as a guide. It steers the character into the orientation and speed that the next target needs, and the context poses are still in motion, so the model carries that velocity forward. Use a smaller value when the motion must match the target as closely as it can, whether the target is a single pose that stops the character or four poses that describe a velocity and momentum to match. The character then plays all the way to the target. See [Chaining generations](#chaining-generations).
- **Tokens to Generate:** leave **Min** at 8 and **Max** at 14. This range sets the lengths the model can choose for the transition to this target. Each motion token yields four raw frames, so 8 to 14 motion tokens give 32 to 56 raw frames. The **Raw frames:** readout shows this range. A longer distance between targets needs more tokens to cover the ground. A shorter distance needs fewer tokens, so that the character does not add extra motion to fill the time.

Select the **Duck-under** target and repeat the same steps to assign the BVH clip:

```text
assets/motions/soma/avoidance/neutral_avoid_obstacle_bend_down_walk_ff_180_R_001__A543.bvh
```

Select frame **328** and apply the following settings to this target:

![Duck-under target settings](media/tut_inbetweener_20_settings.png)

Orient the character in the view as shown below:

![duck-under pose](media/tut_inbetweener_21_pose2.png)

### Generate the final animation

The list of targets is now complete. Click **Generate** to view the final animation:

![animation generation in real-time](media/tutorial_end_inbetweener.gif)

You can explore this setup further by changing the following:

- The position and orientation of the character for each target.
- The clip frame used for each target.
- The **Poses to Use** and **Transition Early By** values.
- The **Tokens to Generate** range, which sets how long each generated transition can be. The **Raw frames:** readout below it shows the matching length in frames.

> **Note:** These are the fundamental concepts for integrating MotionBricks into your own application. Your application logic defines the target poses in real time, and MotionBricks generates the motion that reaches them. For the concepts behind the Inbetweener code, see **[Inbetweener](inbetweener.md)** and the **[Runtime API](runtime_api.md)**.

## Part 2: Build a Smart Primitive

The remaining parts of this tutorial show how the example apps work together to build an interactive MotionBricks experience:

- **Smart Primitive Editor:** defines simple pose-based logic around interaction areas.
- **Scene Editor:** assembles Smart Primitives into an environment.
- **Scene Preview:** runs the result in real time. It combines locomotion, interaction detection, behaviors, and generated motion.

Together, these apps give developers a practical reference for the MotionBricks workflow. Developers can adapt each part to their own integrations.

### Launch the Smart Primitive Editor

Launch the Smart Primitive Editor:

```sh
uv run python -m app.smart_primitive_editor
```

The Newton viewer opens with the Smart Primitive Editor:

![Smart Primitive Editor opened in the Newton viewer](media/tut_smartprim_01_open.png)

Before you begin, confirm that the **Character** selector shows **SOMA**:

![SOMA character](media/tut_smartprim_02_SOMA.png)

> **Note:** The view controls are the same as in the Inbetweener. The Scene Editor uses them too. Scene Preview has its own [controls](#controls). There, **W**, **A**, **S**, and **D**, or the **arrow keys**, move the character. The camera follows the character.

### Create the node hierarchy

A Smart Primitive is built from a hierarchy of nodes, and each node has a defined role. Create the following hierarchy:

- Interaction box
- Pivot
  - Entry pose
  - Duck-under pose

Select the **Root** node and click the **+** button above the hierarchy:

![add node button](media/tut_smartprim_03_addnode.png)

Double click the new node and rename it to **Interaction box**:

![renamed interaction box](media/tut_smartprim_04_interactionbox.png)

To build the rest of the hierarchy, select a node and click **+** to add a child under it:

![Smart Primitive hierarchy](media/tut_smartprim_05_hierarchy.png)

The next steps define the role of each node in the Smart Primitive.

### Configure the Interaction Box

Select the **Interaction box** node and go to the **Properties** window. Right click an empty area of the window to display the available component types, and choose **Add** > **Components** > **Interaction Box**:

![add Interaction Box component](media/tut_smartprim_06_interactionbox.png)

Set **Width** and **Depth** to 1 m and **Height** to 2 m. Set the Z value of **Translation** to 1 m, so that the box stands on the ground:

![interaction box properties](media/tut_smartprim_07_interactionboxset.png)

You can also use the 3D gizmo to move the box up along the Z axis to 1 m:

![move interaction box](media/tut_smartprim_08_interactionboxtransf.png)

### Configure the Pivot

The target poses are authored at fixed positions in the Smart Primitive. They face the direction in which the authored sequence travels. At runtime, however, the character can reach the interaction area from any direction. Without a pivot, a character that arrives from the side first walks to the authored Entry pose. It then turns to follow the sequence, and it leaves the interaction in the authored direction instead of its own.

A pivot solves this problem. When the interaction starts, the pivot turns and slides the whole sequence of target poses to match the direction in which the character enters the interaction area. The character then passes through the interaction without leaving its course:

![A character arrives from the side, without a pivot and with the tutorial Pivot](media/tut_pivot_00_problem.gif)

*A character arrives at 90 degrees to the authored sequence. Without a pivot, on the left, it walks to the authored Entry pose and turns. With the tutorial Pivot, on the right, the target poses turn and slide onto its course. The character's path is an illustration. At runtime, MotionBricks generates the motion.*

To add the pivot, select the **Pivot** node in the hierarchy. Right click an empty area of the **Properties** window and choose **Add** > **Components** > **Pivot**:

![add pivot](media/tut_smartprim_09_addpivot.png)

The **Pivot** properties are added to the node:

![pivot properties](media/tut_smartprim_10_pivotprops.png)

In this tutorial, the character can arrive from any direction. It continues forward and lowers its body to pass under an obstacle in the interaction area. It then resumes its course. Only the poses required inside the interaction area need to be authored. No poses are needed outside of it. When the interaction starts, the pivot first turns the sequence to match the character's direction of approach. It then slides the sequence to the character's point of entry.

To achieve this behavior, set the **Pivot** properties to the following values:

![finished pivot properties](media/tut_smartprim_11_pivotpropsset.png)

| Setting | Value | Purpose |
| --- | --- | --- |
| Motion Direction | 0, 1, 0 | The pivot's local +Y axis is the direction in which the authored sequence travels. |
| Operation Order | Rotation -> Translation | The pivot turns first, and then slides along its turned translation line. |
| Translation | TranslateX, Line Projection, Min -0.50 m, Max 0.50 m | The pivot can slide sideways by up to 0.5 m in either direction to meet the character's path. |
| Rotation | RotateZ, Unconstrained | The pivot can turn around the vertical axis to face any direction of approach. |

Each setting offers other options, such as a **Cone** limit on the rotation or the other **Operation Order**. The [Understanding the Pivot](#understanding-the-pivot) section follows the save step. It explains each option in detail and illustrates its effect on the character's motion.

### Configure the target poses

Select the **Entry pose** node in the hierarchy. Right click an empty area of the **Properties** window and choose **Add** > **Components** > **Target Pose**:

![add target pose](media/tut_smartprim_12_addpose.png)

The **Target Pose** properties match the Inbetweener's from Part 1. The Smart Primitive Editor adds **Constrain Root**, **IK Pins**, and **Show Trajectory**. Its **Tokens to Generate** defaults to 6 to 14 instead of 8 to 14. Consecutive steps in a sequence are often placed close together, and the lower minimum lets the model generate a shorter transition between them:

![Target Pose properties](media/tut_smartprim_13_poseprops.png)

Click the **...** button next to **Clip** and open the clip from Part 1:

```text
assets/motions/soma/avoidance/neutral_avoid_obstacle_bend_down_walk_ff_180_R_001__A543.bvh
```

Set the target pose values as follows. Enable **Constrain Root**, which moves the clip's root to the node's origin along the ground and keeps its height:

![Entry pose properties set](media/tut_smartprim_14_posepropsset.png)

Use the 3D gizmo to rotate and position the **Entry pose** near the edge of the interaction box. Align the pose with the yellow arrow. This arrow shows the pivot's **Motion Direction**, which is the direction in which the authored sequence travels:

![pose transformations](media/tut_smartprim_15_posetransf.png)

Repeat the previous steps for the **Duck-under pose**. Set the following values, and enable **Constrain Root**:

![Duck-under pose properties set](media/tut_smartprim_16_pose2propsset.png)

Use the 3D gizmo to position and rotate the character so that it stands at the center of the box:

![second pose transformations](media/tut_smartprim_18_pose2transf.png)

### Define the sequence

Next, define the order in which the poses are used.

In the **Sequences** window, click the **+** button to add a new sequence:

![add sequence button](media/tut_smartprim_17_addsequence.png)

Set the sequence **Pivot** field to the **Pivot** node. The sequence then uses this pivot:

![sequence pivot](media/tut_smartprim_19_sequpivot.png)

Add the target poses in the order in which the action takes place. Click **+ Add Target Pose** and choose **Entry pose**. Click it again and choose **Duck-under pose**. You can also drag a node from the **Hierarchy** window onto the sequence header:

![sequence done](media/tut_smartprim_20_sequdone.png)

### Save the Smart Primitive

The Smart Primitive is now complete.

Choose **File** > **Save As...** and save the Smart Primitive document as:

```text
assets/smart_primitives/soma/SOMA_duck_tutorial.smart_primitive.json
```

### Understanding the Pivot

This section explains how the **Pivot** configured above adapts the authored poses to each approach. It also explains why the resulting poses continue the character's incoming motion.

The figures use the `SOMA_duck_tutorial` Smart Primitive saved above. They place the poses with the same pivot alignment code that Scene Preview runs, and they show the SOMA mesh used by the editors. The pink and green characters are the actual target poses sent to the model. The blue character stands in for the incoming character. It and the dashed path are illustrations only. At runtime, MotionBricks generates the path.

#### Why a pivot is needed

The target poses of a Smart Primitive are authored at fixed positions relative to the Smart Primitive. At runtime, however, the character can reach the interaction area from many positions, directions, and speeds. If the poses stayed at their authored placement, the first target could lie to one side of the character's path or face a different direction. The generated transition would then need an abrupt turn or sidestep to reach it.

A pivot adapts the placement of the entire sequence to the approach. The **Pivot** node is the parent of the target pose nodes. When the runtime moves and turns the pivot, every target pose follows it as a rigid group. Only the root of each sampled pose is transformed. The joint values stay exactly as sampled from the clip, so the authored body motion is preserved.

#### When the pivot is aligned

In Scene Preview, the pivot is aligned once, at the start of each sequence:

1. The character's probe box extends ahead of the character along its movement. The sequence starts when the probe box overlaps an interaction box of the Smart Primitive.
2. The behavior selects a sequence. `VelocityMatchingSequenceBehavior` selects the sequence whose first target speed is closest to the character's current speed.
3. If the sequence references a pivot, the pivot is aligned with the approach. The alignment uses the character's position and its desired velocity. It also uses the line that the probe box covers on the ground.
4. The target poses in world space are computed from the aligned hierarchy and sent to MotionBricks.

The limits are always measured from the authored placement of the pivot, never from its previous alignment. As a result, the Smart Primitive does not drift over repeated interactions. A sequence without a pivot keeps its current placement.

#### How each setting contributes

The following top view shows the authored layout of the tutorial Smart Primitive. It also shows the visual guides that the Smart Primitive Editor displays for the **Pivot**:

![Authored layout of the tutorial Pivot, top view](media/tut_pivot_01_layout.png)

*Authored layout, top view. The yellow arrow is the Motion Direction, and the red dashed line is the range of the Line Projection.*

| Setting | Role | Editor guide |
| --- | --- | --- |
| **Motion Direction** | The direction, in the pivot's local axes, in which the authored sequence travels. Alignment turns the pivot so that this direction matches the character's direction of travel. For this reason, the target poses were oriented along the yellow arrow. | Yellow arrow |
| **Rotation** | The local axis around which the pivot can turn. **RotateZ** turns it around the vertical axis. **Unconstrained** allows any heading. **Cone** limits the turn to **Half Angle** on each side of the cone center. **Cone Side** sets that center. **Motion Direction** centers the cone on the Motion Direction. **Mirrored** centers it on the opposite direction. **Both** allows either side. | Red arc (Cone only) |
| **Translation** | The local axis along which the pivot can slide. With **Line Projection**, the pivot moves to the point of the line, between the **Min** and **Max** offsets, that is closest to that line of the probe box. In practice, this is the point where the character's path crosses the line. **Unconstrained** allows any distance along the axis. | Red dashed line |
| **Operation Order** | **Rotation -> Translation** turns the pivot first. The turn also rotates the translation line around the authored origin. The pivot then slides along the turned line. **Translation -> Rotation** slides along the authored line first, and then turns. | None |

The **Show Motion Direction**, **Show Translation Constraint**, and **Show Rotation Constraint** options of the **Pivot** component show or hide the guides.

#### Effect on different approaches

The figure below compares the placement of the target poses for six approaches, seen from above. In each panel, the blue character is the incoming character at the moment the interaction starts. The pink and green characters are the Entry and Duck-under target poses. The faded dashed arrow marks the authored Motion Direction.

![Pivot alignment for six approaches](media/tut_pivot_02_cases.png)

1. **No pivot:** the poses stay where they were authored. A character that approaches at an angle must turn sharply to reach the Entry pose. It must then turn again to pass under the obstacle.
2. **Translation only:** the character approaches 0.4 m to the side of the center. The pivot slides 0.4 m along its line, so the poses move onto the character's path.
3. **Rotation only:** the character approaches at 35 degrees. The pivot turns 35 degrees around Z, so the poses face the character's direction of travel.
4. **Tutorial settings:** the character approaches at an angle and off center. The pivot turns first. It then slides along the turned line to the point where the character's path crosses it.
5. **Line Projection** limit: the character approaches 0.85 m to the side. The slide stops at the +0.5 m end of the line, and the generated transition covers the remaining 0.35 m.
6. **Cone limit:** the cone's **Half Angle** is 30 degrees, and the character approaches at 55 degrees. The turn stops at 30 degrees, and the generated transition covers the remaining 25 degrees.

The following animation uses the tutorial settings. It sweeps the approach across lateral offsets, then across approach angles, and finally across both at once. Each step is shown from above and in perspective, where the duck-under posture is clearly visible. The readout at the bottom reports the resulting turn and slide of the pivot:

![Pivot alignment animated across approach offsets and angles](media/tut_pivot_03_alignment.gif)
*Live pivot alignment with the tutorial settings. In the first phase, the slide is clamped at ±0.50 m when the offset exceeds the line.*

#### Continuity with the incoming motion

MotionBricks always generates motion from the four context poses toward the next target. These poses carry the character's current velocity. The pivot places the targets so that they are consistent with that context:

- **Position:** after translation, the Entry pose lies on the character's path. The character reaches it by continuing forward, without a sidestep.
- **Orientation:** after rotation, the four poses of the Entry target, and therefore its velocity, point along the character's direction of travel. The context poses and the target poses describe the same direction of motion. The generated transition is a natural continuation of the incoming walk.
- **Within the sequence:** all target poses move together. The spacing, orientation, and timing between the Entry and Duck-under poses stay exactly as authored. With **Transition Early By**, the character carries its momentum from one target to the next.
- **Exit:** the last pose turns by the same rotation as the rest of the sequence. The sequence therefore exits along the aligned Motion Direction. When the interaction ends, the character already moves in its original direction of travel, so locomotion resumes without a corrective turn.

In effect, the pivot re-expresses the authored poses as if the clip had been recorded along the character's actual path. The limits define the range over which this remains valid. In this Smart Primitive, the translation line runs across the direction of travel. The Duck-under pose lies close to the pivot origin. Turning the pivot therefore changes the angle at which the character crosses the interaction area, while the duck-under posture stays near its center. When you author your own Smart Primitives, choose offsets and cone angles that keep every allowed placement valid for the obstacle.

### Explore variations

Change the poses or add more poses to the sequence to make variations. You can also add sequences that respond to different speeds or incoming motions. To learn more about Smart Primitives and their editor, see the **[Smart Primitive Editor](smart_primitive_editor.md)** page.

For comparison, open the following Smart Primitive document in the Smart Primitive Editor. It is similar to the tutorial Smart Primitive and contains several variations of the avoidance:

`assets/smart_primitives/soma/avoidance.smart_primitive.json`

It extends the tutorial Smart Primitive with:

- A second branch, `Variation 1`.
- A `Jog` sequence sampled from another clip.
- The same pivot, with the same settings as in this tutorial, shared by the `Walk` and `Jog` sequences.
- Sequence selection by speed at runtime.
- A smaller interaction box raised above the ground, from about 1.46 m to 2.46 m. A standing character's probe box is 1.6 m tall, so it reaches the interaction box and starts the avoidance. When the character crouches, its probe box shrinks to about 0.96 m. The character can then move under the obstacle without starting the interaction. This refinement is outside the scope of this tutorial. The tutorial's interaction box extends from the ground to 2 m, so every locomotion mode starts the interaction.

![velocity matching flow](media/velocity_matching_flow.png)

This shows the flexibility of pose-based interactions. A faster approach does not require a new state machine branch or a set of transition clips. It only requires another target sequence. The behavior then selects the sequence that best matches the character's current speed.

## Part 3: Assemble an interaction scene with avoidance, jump, and sit

The scene uses the following Smart Primitives:

- `SOMA_duck_tutorial.smart_primitive.json`, created in Part 2.
- `jump_0_5m.smart_primitive.json`, bundled with the repository.
- `sit.smart_primitive.json`, bundled with the repository.

### Launch the Scene Editor

Launch the Scene Editor:

```sh
uv run python -m app.scene_editor
```

The Newton viewer opens with the Scene Editor:

![scene editor](media/tut_sceneedit_01_open.png)

Confirm that the **Character** selector shows **SOMA**.

Every Smart Primitive reference in a scene must use the same rig. The `CharacterStart` item defines where the character appears when Scene Preview starts the scene.

### Add the avoidance Smart Primitive

In the Scene Editor, add a new scene item by clicking the **+** button at the top:

![Scene window with the + button](media/tut_sceneedit_02_add.png)

Choose **Smart Primitive**:

![Add menu with Smart Primitive](media/tut_sceneedit_03_addsmartprim.png)

Double click the new item and rename it to **avoid**:

![Renamed Smart Primitive instance](media/tut_sceneedit_04_addsavoid.png)

In the **Properties** window:

- Set **Reference** to the Smart Primitive document created in Part 2.
- Set the runtime **Behavior** to `VelocityMatchingSequenceBehavior`.

![Smart Primitive instance properties](media/tut_sceneedit_05_primpropsset.png)

Use the 3D gizmo to rotate and position the Smart Primitive instance in the scene:

![Smart Primitive instance layout](media/tut_sceneedit_06_primmove.png)

To show the obstacle, add a **Solid Cube** item to the scene:

![add solid cube](media/tut_sceneedit_07_solidcube.png)

Use the 3D gizmo and the cube's properties to position and scale the cube so that it sits above the **Duck-under pose** of the Smart Primitive:

![adding obstacle cube](media/tut_sceneedit_08_obstacle.png)

### Complete the course

To complete the obstacle course, repeat the previous steps to add the following Smart Primitives:

- `assets/smart_primitives/soma/jump_0_5m.smart_primitive.json`
- `assets/smart_primitives/soma/sit.smart_primitive.json`

For the **Sit** Smart Primitive, select `SittingBehavior` as the runtime behavior.

Position and rotate the Smart Primitive instances in the scene:

![scene layout](media/tut_sceneedit_09_layout.png)

Add Solid Cubes to show the obstacles for these Smart Primitives:

![obstacles layout](media/tut_sceneedit_10_layoutobstacles.png)

### Save the scene

Choose **File** > **Save As...** and save the scene as:

```text
assets/scenes/soma_tutorial.scene.json
```

The scene stores its references relative to the scene file. Several instances can reference the same Smart Primitive document, each with its own placement and behavior. You can experiment further by adding more Smart Primitives and building your own layout.

## Part 4: Play the scene

Scene Preview is a compact real-time character demo. It combines the following parts:

- A state machine.
- Locomotion blend spaces.
- Interaction detection.
- Smart Primitive behaviors.
- Clip playback.
- The MotionBricks generation API.

### Launch Scene Preview

Run:

```sh
uv run python -m app.scene_preview --scene assets/scenes/soma_tutorial.scene.json
```

The Newton viewer opens with Scene Preview:

![Scene Preview opened with the tutorial scene](media/tut_sceneprev_01_start.png)

If the scene does not load, check three things. Every Smart Primitive reference must resolve. Every referenced Smart Primitive document must use SOMA, the scene's rig. Every instance must have a valid **Behavior**.

The character starts in an idle loop and is ready to explore the scene:

![final result](media/tutorial_end_scenepreview.gif)

### Controls

Use the following controls to move the character and orbit the camera:

| Action | Keyboard and mouse | Controller |
| --- | --- | --- |
| Move forward, left, backward, or right | **W**, **A**, **S**, and **D**, or the **arrow keys** | Left stick or D-pad |
| Run | Hold **Space** while moving | Push the left stick further |
| Interact manually, or stand up after sitting | **I** | **X** |
| Switch between the Forward and Strafing modes | **O** | **Y** |
| Switch between the Crouching and Strafing modes | **P** | **B** |
| Cycle through the style modes, such as Skipping and Zombie | **[** | **A** |
| Orbit the camera around the character | Drag with the **middle mouse button** | Right stick |

**Auto Interact** is enabled by default. Scene Preview starts the interaction of the nearest overlapping Smart Primitive instance once per approach. To start each interaction manually with **I** (**X** on a controller), disable **Auto Interact** in the **Scene Preview** window.

When the character is seated, press **I** (**X** on a controller) to run the `Stand` sequence and return to locomotion.

### Inspect the runtime

In the **Scene Preview** window, enable the following options:

- **Show Context Poses** displays the recent poses supplied to the next generation.
- **Show Target Poses** under **Character** displays the current target poses.
- **Show Blend Space Poses** displays the blended target pose where locomotion places it. When locomotion needs new motion, it generates toward this target.
- **Show Probe Box** displays the character's probe box, which reaches ahead along its movement.
- **Show Interaction Boxes** displays the areas where Smart Primitive behaviors can begin.

Watch **State:** in the **Status** section change from Locomotion to Interaction when the probe box reaches each interaction box. The interaction box turns from red to green while the probe box overlaps it. When the avoidance or jump interaction completes, the character returns to locomotion. Without movement input, it returns to idle. The sit interaction stays in its looping clip until the user requests the `Stand` sequence.

> **Note:** If the feet slide, or if the duck-under starts too early or too late, adjust the Smart Primitive in the Smart Primitive Editor:
>
> - **Distance between the target poses:** move the **Entry pose** farther from the **Duck-under pose** to give the character enough lead to lower its body before the obstacle. Move the poses closer together if the character adds extra steps to cover the distance.
> - **Transition Early By (frames):** adjust the value of the **Entry pose** and the **Duck-under pose**. A larger value moves the transition point earlier, which gives the character more lead into the duck-under. A smaller value lets the character reach each target more closely.
>
> Save the Smart Primitive, and then open the scene again in Scene Preview to check the result. Tuning these values is an iterative process. Repeat it until the motion through each interaction box looks right.

## Summary: what this workflow demonstrates

Two target poses from a single clip served first as direct generation goals and then as a reusable interaction. The bundled avoidance Smart Primitive adds a Jog variation without any change to the runtime state machine. The scene combines that interaction with Jump and Sit. Each one uses the same pose-based generation pattern.

This separation of responsibilities is the main integration model:

- The application or state machine selects the next goals.
- Smart Primitives organize location-specific goals and adapt them to each approach.
- Behaviors select and play the appropriate target sequence.
- Scenes place those interactions and their visible geometry.
- MotionBricks generates continuous motion from the character's current context.

Continue with the [Runtime API](runtime_api.md) to explore the reusable library calls behind the examples. The [Smart Primitive Editor](smart_primitive_editor.md), [Scene Editor](scene_editor.md), and [Scene Preview](scene_preview.md) pages describe each layer in more detail.
