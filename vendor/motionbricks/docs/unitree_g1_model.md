# MotionBricks Unitree G1 Model Card

## Model Description

MotionBricks Unitree G1 generates robot motion from recent poses and placed target poses. It produces short transitions that an application can join into continuous reference motion.

The [MotionBricks](https://nvlabs.github.io/motionbricks/) model is trained on motion capture retargeted to a 34-joint Unitree G1 skeleton. It reuses the original weights released in [GR00T Whole-Body Control](https://github.com/NVlabs/GR00T-WholeBodyControl/tree/main/motionbricks).

This card covers the G1 ONNX models in MotionBricks v0.2.0. Each ONNX model combines the pose tokenizer and decoder, pose predictor, and root predictor from `MotionBricks-G1-RP-v1`. It accepts and returns G1 joint coordinates, handling the conversion to skeletal features within the model. The FP32 and mixed FP16 variants share the same tensor interface. FP32 uses 32-bit floating-point precision. Mixed FP16 uses 16-bit precision for selected operations.

The SafeTensors checkpoint has its own model card and component interfaces. [SOMA models](soma_model.md) are documented separately. A training pipeline is not included.

This model is intended for commercial and non-commercial motion-generation applications.

### Where to Start

- To run the G1 demo, follow [Setup](../README.md#setup), then the commands under [Inference](#inference).
- To add generation to an application, use the [runtime API](runtime_api.md#engines).
- To export ONNX models from the SafeTensors checkpoint, follow the [exporter guide](../tools/onnx/README.md).

| Property | Value |
| --- | --- |
| Model family | MotionBricks |
| Task | Robot motion generation and inbetweening conditioned on poses |
| Robot | Unitree G1, 29 joint angles and a floating root |
| Skeleton | Unitree G1, 34 model joints |
| Parameters | About 195 million in the checkpoint |
| Public pose representation | Root XYZ, XYZW quaternion, and 29 joint angles |
| Model format | ONNX, opset 17 |
| Precisions | FP32 and mixed FP16 |
| Batch size | Variable `N`. The demos use `N=1`. |
| FPS | 30 |
| Context | Four recent poses |
| Targets | Up to four placed poses |
| Output length | 24 to 64 raw frames, or 18 to 58 after the engine removes the boundary frames |

## License and Terms of Use

Use of this model is governed by the [NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/). It permits commercial use and redistribution of modified models, subject to the agreement's terms.

- The accompanying software uses [Apache 2.0](../LICENSE).
- The listed Bones-derived human BVH and G1 CSV samples use the [NVIDIA Sample Data Evaluation License](../assets/motions/LICENSE.txt). They are for internal evaluation and testing of NVIDIA technologies. See the [sample notice](../assets/motions/README.md) for the exact files covered.
- Robot assets and third-party dependencies retain their [respective licenses](../licenses).

## Deployment Geography

Global.

## Intended Uses

**Use Case:** Generate pose-conditioned G1 reference motion for animation, robotics simulation, and downstream control experiments.

- Generate transitions between G1 poses or poses sampled from motion clips.
- Preview locomotion and steering in Newton by placing target poses.
- Produce kinematic reference motion for robot control experiments.
- Explore repeated generation that uses the previous output as the next context.

The output describes desired poses. A controller downstream is responsible for tracking those poses under robot dynamics and physical constraints.

This model is intended for developers and researchers building animation, robotics, and simulation workflows, both commercial and non-commercial. Generated motion is not validated for physical robot control.

## References

T. Wang, O. Dionne, M. De Ruyter, D. Minor, D. Rempe, K. Zhao, M. Petrovich, Y. Yuan, C. Li, Z. Luo, B. Robison, X. Blackwell, B. Antoniazzi, X. B. Peng, Y. Zhu, and S. Yuen. **MotionBricks: Scalable Real-Time Motions with Modular Latent Generative Model and Smart Primitives**. ACM Transactions on Graphics 45(4), 2026. [DOI 10.1145/3811334](https://doi.org/10.1145/3811334), [arXiv:2604.24833](https://arxiv.org/abs/2604.24833).

## Model Architecture

**Architecture Type:** Modular latent generative model.

**Network Architecture:** Transformers that predict the root motion and the motion tokens, with a learned motion decoder.

| Component | Role | Parameters |
| --- | --- | --- |
| Pose VQ-VAE (vector-quantized variational autoencoder) | Encodes tokens of four poses and decodes motion. Its codebook has 8 heads of 10 codes. | 23.8 million |
| Pose model | Masked transformer that predicts motion tokens, with 16 layers of width 1024 | 136.6 million |
| Root model | Predicts the duration and the root path | 34.1 million |

The ONNX model converts between G1 joint coordinates and the transforms of its 34-joint skeleton. The method is described in the [MotionBricks paper](https://arxiv.org/abs/2604.24833).

**Base Model:** The original G1 components were trained from scratch on RP1 motion retargeted to G1. This release reuses those weights without retraining. The ONNX export combines the components and adds the joint-coordinate interface.

## Inputs

**Input modality:** G1 joint coordinates, conditioning masks, and allowed token counts. Context and target poses are in G1 model space and sampled at 30 Hz.

**Input format:** Numeric ONNX tensors. Joint coordinates use FP32, and masks and token counts use INT32. The input tensor table defines each parameter's name, shape, and meaning.

`N` is the number of independent requests, and all inputs must use the same `N`. The model manifest `unitree_g1_n1.json` describes the tensors for `N=1`.

### Pose Representation

Every pose contains 36 FP32 joint coordinates:

```text
[root_x, root_y, root_z, root_qx, root_qy, root_qz, root_qw, 29 joint angles]
```

G1 model space is Z up, with +X forward. Positions use meters, and joint angles use radians. The root joint is the pelvis. Quaternion components use XYZW order. Joint angle order is defined by `joint_names` in [unitree_g1_n1.json](../motionbricks/assets/onnx/unitree_g1/unitree_g1_n1.json).

### Input Tensors

| Input | Shape | Type | Meaning |
| --- | --- | --- | --- |
| `context_qpos` | `[N,4,36]` | FP32 | Four recent context poses at 30 Hz |
| `target_qpos` | `[N,4,36]` | FP32 | Four placed target poses |
| `has_global_root_values` | `[N,8]` | INT32 | Mask for the root position and heading |
| `has_local_root_values` | `[N,8]` | INT32 | Mask for the root velocity and height |
| `has_local_poses` | `[N,8]` | INT32 | Mask for the body pose |
| `allowed_pred_num_tokens` | `[N,11]` | INT32 | Allowed token counts, 6 to 16 |

All six inputs are required. Masks contain 0 or 1, with context slots first, then target slots. At least one token count must be allowed. Entry `i` of `allowed_pred_num_tokens` allows a token count of `6 + i`. Each motion token represents four poses.

Pose values must be finite and root quaternions non-zero, including masked slots.

## Outputs

**Output modality:** G1 joint coordinates, predicted foot contacts, and the chosen token count.

**Output format:** Numeric ONNX tensors. Joint coordinates and foot contacts use FP32. The token count uses INT32. The output tensor table defines each parameter's name, shape, and meaning.

| Output | Shape | Type | Meaning |
| --- | --- | --- | --- |
| `pred_qpos` | `[N,64,36]` | FP32 | Decoded joint coordinates in the same space as the inputs |
| `pred_num_tokens` | `[N]` | INT32 | Token count the model chose |
| `pred_foot_contacts` | `[N,64,4]` | FP32 | Predicted foot contacts for each frame of `pred_qpos` |

The leading `4 * pred_num_tokens` frames contain generated motion. The raw output contains 24 to 64 frames. Removing three boundary frames at each end leaves 18 to 58 valid frames at 30 Hz.

Each frame of `pred_foot_contacts` holds four foot contacts for the left heel, left toe, right heel, and right toe. The heel contact points are at the ankle roll joints. Each toe contact point sits 0.14 m ahead of its ankle roll joint and 0.035 m below it, in the ankle roll link's local space. See [foot_contacts.py](../motionbricks/animation/foot_contacts.py) for their meaning.

The foot contacts are model predictions. They are not contact sensor readings and do not guarantee stable contact on a physical robot.

## Software Integration

| Component | Integration |
| --- | --- |
| Tested inference runtime | NVIDIA TensorRT 10.8 |
| Application framework | Python and NVIDIA Warp |
| Demo environment | Newton |
| Tested operating system | Windows |
| Documented deployment hardware | NVIDIA Ada Lovelace GeForce RTX 4090 |

The application supplies placed context and target poses. The engine aligns them for inference and returns generated motion in world space.

The table identifies the tested setup, not a complete compatibility matrix. Timings are reported only for the GeForce RTX 4090 under Windows. The checkpoint-to-ONNX exporter runs on CPU and is separate from this GPU inference path. A CPU-versus-GPU inference comparison is not provided.

The integration of foundation and fine-tuned models into AI systems requires additional testing using use-case-specific data to ensure safe and effective deployment. Following the V-model methodology, iterative testing and validation at both unit and system levels are essential to mitigate risks, meet technical and functional requirements, and ensure compliance with safety and ethical standards before deployment.

### Runtime Generation

`MotionBricksUnitreeG1Engine` runs the generation calls in [Engines](runtime_api.md#engines). Its `DOFToken` context and target hold joint coordinates as Warp FP32 arrays. It generates into the same `DOFAnimationBuffer` type that loaded CSV clips use.

The caller can sample directly into one robot's view of Newton `model.joint_q`, then call `newton.eval_fk` for playback. No joint names are remapped. Pass `model_to_runtime` when world space differs from G1 model space.

Before inference, the engine removes the ground position and heading of the last context pose. It keeps the root height and joint angles. After inference, it restores the world placement and removes three boundary frames at each end. By default, inference turns the last context pose to face +X in G1 model space.

The engine smooths the decoded root position over neighboring frames. This removes frame-to-frame jitter in the model's output and keeps the gait.

## Model Versions

**Model version:** `MotionBricks-G1-RP-v1`. The accompanying software release is v0.2.0.

| Precision | ONNX file |
| --- | --- |
| FP32 | `mmm_v6_g1_fp32.onnx` |
| Mixed FP16 | `mmm_v6_g1_fp16.onnx` |

Files are stored in [motionbricks/assets/onnx/unitree_g1](../motionbricks/assets/onnx/unitree_g1).

Motion inputs and outputs use FP32 in both variants. Masks and token counts use INT32. The [G1 checkpoint on Hugging Face](https://huggingface.co/nvidia/MotionBricks/tree/main/g1) contains the SafeTensors weights used by the [exporter](../tools/onnx/README.md).

## Training, Testing and Evaluation Datasets

### Training Dataset

**Name/Identifier:** Bones RigPlay 1 (RP1).

**Data Modality:** Other: human motion capture retargeted to a 34-joint Unitree G1 representation at 30 Hz. The model uses skeletal motion features rather than text descriptions for conditioning.

**Training Data Size:** 313,526 selected training clips. The source corpus contains approximately 350,000 RP1 clips and 700 hours of motion. The 700 hours describe the source corpus, not the selected training subset.

**Data Collection Method:** Automatic/Sensors. Bones captured human performances in a studio and supplied skeletal motion. NVIDIA used [SOMA Retargeter](https://github.com/NVIDIA/soma-retargeter) to retarget that motion to G1, then converted it into model features. This preparation is not direct robot recording.

**Labeling Method:** Automatic/Sensors for captured skeletal motion, followed by automated retargeting and feature preparation. Text annotations are not conditioning inputs for this motion-only model.

**Properties:** Time-varying retargeted skeletal motion for pose-conditioned robot generation.

**Linguistic Characteristics:** Not applicable to the motion-feature inputs. Associated text descriptions are dataset metadata, not model conditioning inputs.

The ONNX conversion does not retrain the original G1 weights or introduce a new training dataset.

### Testing Dataset

**Name, Size and Role:** We used 313,526 clips for training and a separate, selected set of 4,881 clips for validation and evaluation during training. No clips are shared between these sets. The 4,881 clips are a selection from the held-out pool, not the entire pool. A separate untouched final-test set, a performer-disjoint split, and the selected subset's duration are not documented.

**Data Collection Method:** Automatic/Sensors, from the same Bones studio motion-capture source as the training set. NVIDIA retargeted the human motion to the G1 representation.

**Labeling Method:** Automatic/Sensors for captured skeletal motion, followed by automated retargeting and feature preparation.

**Properties:** The validation inputs are 30 Hz skeletal-motion features on the 34-joint G1 representation. Text descriptions are selection metadata, not conditioning inputs. Linguistic characteristics are not applicable to these motion-only inputs. Disjoint clips alone do not rule out other forms of data leakage.

### Evaluation Dataset and Results

**Name/Size:** Evaluation uses the same 4,881-clip RP1 subset described under [Testing Dataset](#testing-dataset). It is not an additional independent test set.

**Benchmark Score:** A complete quantitative evaluation of these G1 ONNX models is not provided.

**Data Collection Method:** Automatic/Sensors, using the same studio-captured human motion retargeted to G1.

**Labeling Method:** Automatic/Sensors for captured skeletal motion, followed by automated retargeting and feature preparation.

**Properties:** Evaluation uses the same 30 Hz G1 skeletal-motion representation as training. Text descriptions are selection metadata, not conditioning inputs. Linguistic characteristics are not applicable to the motion-only inputs. The subset's duration is unspecified.

**Recorded Results:** Training measurements for the original G1 components are not reported.

The ONNX models were tested in the setup listed under [Software Integration](#software-integration). The GPU time and throughput of the TensorRT plans at several batch sizes are in [Inference performance](onnx_inference.md#inference-performance). Memory usage and demographic comparisons are not reported. A quality comparison between FP32 and mixed FP16 is also not reported.

The [MotionBricks paper](https://arxiv.org/abs/2604.24833) describes method-level experiments on its own models and datasets. This Newton integration has not been evaluated for tracking on a physical robot.

## Inference

FP32 is the default. Run the G1 gym in Scene Preview with either precision:

```sh
uv run python -m app.scene_preview --scene assets/scenes/g1_gym.scene.json --model-precision fp32
uv run python -m app.scene_preview --scene assets/scenes/g1_gym.scene.json --model-precision fp16
```

The G1 TensorRT configuration disables TensorFloat-32 (TF32) in both precisions. With TF32 enabled, a batch of one runs only about 2 percent faster, but joint angles can differ by several degrees on identical inputs. The differences grow across chained generations. Larger batches gain more, as [Inference performance](onnx_inference.md#inference-performance) shows. Pass `TensorRTBuildOptions(enable_tf32=True)` to allow it.

The mixed FP16 ONNX model uses FP16 for the neural predictors and the motion decoder. It keeps FP32 for feature normalization, motion geometry, LayerNormalization, Softmax, and Resize. The G1 forward kinematics, motion integration, and conversion between quaternions and joint angles also stay FP32. The demos build a TensorRT plan for `N=1`.

Mixed FP16 and FP32 can produce different motion, because rounding affects the choice of motion tokens.

**Acceleration Engine:** NVIDIA TensorRT 10.8.

**Test Hardware Set-Up:** NVIDIA Ada Lovelace GeForce RTX 4090 on Windows.

The exporter checks checkpoint integrity, ONNX graph validity, and the expected tensor interface. Those structural checks do not establish complete security coverage, safe physical behavior, or equivalent motion quality across precisions and backends.

## Limitations

- The demos run one request at a time with the same batched ONNX model available to users. Context and target tokens hold at most four poses each, so longer motion requires repeated generation. See [batching](onnx_inference.md).
- The joint coordinate interface is specific to the bundled G1 kinematics and joint order. Another robot or joint layout requires a matching conversion and model.
- Conditioning comes from poses and motion constraints. Text prompts and scene geometry are outside this model's input interface. The application places targets and handles obstacles.
- The model works best with target poses within its training distribution. Poses outside it can produce lower quality motion.
- The model generates kinematic poses. Dynamic balance, actuator limits, collision avoidance, and contact tracking require a controller downstream.
- Generated motion can vary with precision and inference runtime. Differences can accumulate when generated motion is fed back as context.
- This Newton integration has not been evaluated for tracking on a physical robot.

## Ethical Considerations

NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. Developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.

For more detail, see the [Bias](#bias-subcard), [Explainability](#explainability-subcard), [Privacy](#privacy-subcard) and [Safety and Security](#safety-and-security-subcard) subcards below.

Please report model quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).

## Bias Subcard

| Field | Response |
| --- | --- |
| Participation considerations from adversely impacted groups in model design and testing | Specific participation measures are not reported for the original G1 checkpoints or these exports. |
| Measures taken to mitigate unwanted bias | A measured bias-mitigation procedure is not reported. Evaluate motion behavior across the intended application's users, targets and motion conditions. |
| Bias Metric, if measured | No demographic subgroup or motion-coverage bias metric is reported for these exports. |

Retargeting motion to one robot skeleton does not establish equal quality across the human behaviors represented in the source data or across motion categories.

## Explainability Subcard

| Field | Response |
| --- | --- |
| Intended Task/Domain | Pose-conditioned Unitree G1 motion generation and inbetweening. |
| Model Type | Modular latent generative model with transformer predictors and a learned tokenizer and decoder. |
| Intended Users | Developers and researchers building robot-motion previews, interactive pose transitions and reference-motion workflows. |
| Output | Joint coordinates (root position, XYZW quaternion, and 29 joint angles), four foot contacts per frame, and the token count. These are kinematic references. |
| Describe how the model works | The ONNX model converts joint coordinates into skeletal features. Root and pose predictors generate motion codes and root motion conditioned on recent and target poses. A decoder reconstructs motion, and the ONNX model converts it back to joint coordinates. |
| Groups tested to produce comparable outcomes | Demographic subgroup comparisons are not reported. |
| Technical Limitations and Mitigation | Fixed G1 kinematics, short context, no scene geometry, and variation across backends and precisions. Validate targets, motion quality, physical limits, and downstream tracking for the intended application. |
| Verified to have met prescribed NVIDIA quality standards | Yes |
| Performance Metrics | See [Evaluation Dataset and Results](#evaluation-dataset-and-results) for the available measurements and their limits. |
| Potential Known Risks | Pose/contact artifacts, unsuitable motion from out-of-distribution constraints, and physical risk if a generated reference is treated as a safe control command. |
| Licensing | Use of this model is governed by the [NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/). See [License and Terms of Use](#license-and-terms-of-use) for separate software and sample-data terms. |

## Privacy Subcard

The datasets used to train this model were reviewed before release. Please review [NVIDIA’s Privacy Policy](https://www.nvidia.com/en-us/about-nvidia/privacy-policy/) for more information.

## Safety and Security Subcard

| Field | Response |
| --- | --- |
| Model Application Fields | Industrial/Machinery and Robotics, Media & Entertainment. Generated motion is not validated for physical robot control. |
| Life-critical impact | Generated reference motion is not a validated physical controller. Directly executing unsuitable motion on a robot can create physical risk. |
| Use Case Restrictions | Use of this model is governed by the [NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/). See [License and Terms of Use](#license-and-terms-of-use) for separate software and sample-data terms. |
| Model and dataset restrictions | The ONNX files do not redistribute the full training corpus. Dataset access and motion-sample use are governed separately from the model weights. |

The exporter checks artifact integrity and graph structure. Those checks are not security approval or assurance of safe behavior. Evaluate the model and its downstream control system together under the intended operating conditions before physical use.

Physical execution requires a downstream tracking controller, limit enforcement, and safety validation.
