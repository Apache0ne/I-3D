# MotionBricks SOMA Model Card

## Model Description

MotionBricks SOMA generates 3D human motion from recent motion and target poses. It predicts short transitions that an application joins into continuous animation, updating the targets as the character moves through a scene.

The [MotionBricks](https://nvlabs.github.io/motionbricks/) model is trained on Bones RigPlay 1 (RP1) motion prepared for the 30-joint [SOMA](https://github.com/NVlabs/SOMA-X) skeleton.

This card covers the SOMA ONNX models in MotionBricks v0.2.0. Each ONNX model combines the pose tokenizer and decoder, pose predictor, and root predictor from `MotionBricks-SOMA-RP-v1`. The FP32 and mixed FP16 variants share the same tensor interface. FP32 uses 32-bit floating-point precision. Mixed FP16 uses 16-bit precision for selected operations.

The SafeTensors checkpoint has its own model card and component interfaces. [Unitree G1 models](unitree_g1_model.md) are documented separately. A training pipeline is not included.

This model is intended for commercial and non-commercial motion-generation applications.

### Where to Start

- To run the demo, follow [Setup](../README.md#setup), then the commands under [Inference](#inference).
- To add generation to an application, use the [runtime API](runtime_api.md#engines).
- To export ONNX models from the SafeTensors checkpoint, follow the [exporter guide](../tools/onnx/README.md).

| Property | Value |
| --- | --- |
| Model family | MotionBricks |
| Task | Motion generation and inbetweening conditioned on poses |
| Skeleton | SOMA, 30 joints |
| Parameters | About 200 million in the checkpoint |
| Public pose representation | Root motion, joint positions, and 6D rotations |
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

**Use Case:** Generate pose-conditioned human motion for animation, robotics simulation, and interactive character applications.

- Generate transitions between authored or sampled human poses.
- Drive interactive locomotion by placing targets from the application.
- Animate approaches to objects and interactions described by target poses.
- Prototype character animation in games, media tools, and simulation scenes.

The application supplies target positions and orientations in the scene. Generated motion can become the context for the next generation.

This model is intended for developers and researchers building animation, robotics, and simulation workflows, both commercial and non-commercial. Generated motion is not validated for physical robot control.

## References

T. Wang, O. Dionne, M. De Ruyter, D. Minor, D. Rempe, K. Zhao, M. Petrovich, Y. Yuan, C. Li, Z. Luo, B. Robison, X. Blackwell, B. Antoniazzi, X. B. Peng, Y. Zhu, and S. Yuen. **MotionBricks: Scalable Real-Time Motions with Modular Latent Generative Model and Smart Primitives**. ACM Transactions on Graphics 45(4), 2026. [DOI 10.1145/3811334](https://doi.org/10.1145/3811334), [arXiv:2604.24833](https://arxiv.org/abs/2604.24833).

## Model Architecture

**Architecture Type:** Modular latent generative model.

**Network Architecture:** Transformers that predict the root motion and the motion tokens, with a learned motion decoder.

| Component | Role | Parameters |
| --- | --- | --- |
| Pose VQ-VAE (vector-quantized variational autoencoder) | Encodes tokens of four poses and decodes motion. Its codebook has 5 heads of 16 codes. | 23.4 million |
| Pose model | Masked transformer that predicts motion tokens, with 16 layers of width 1024 | 136.5 million |
| Root model | Predicts the duration and the root path | 40.4 million |

The method is described in the [MotionBricks paper](https://arxiv.org/abs/2604.24833).

**Base Model:** The SOMA components were trained from scratch on RP1 motion. The ONNX export combines those trained components into a deployment graph and converts precision where requested. Exporting does not retrain them or introduce a new training dataset.

## Inputs

**Input modality:** Poses as joint positions and 6D rotations, root motion constraints, and allowed token counts. The eight conditioning slots hold four context poses followed by four target poses, sampled at 30 Hz.

**Input format:** Numeric ONNX tensors. Motion values use FP32, and masks and token counts use INT32. The input tensor table defines each parameter's name, shape, and meaning.

`N` is the number of independent requests, and all inputs must use the same `N`. The model manifest `soma_n1.json` describes the tensors for `N=1`.

### Pose Representation

Poses describe joint positions and 6D rotations on the SOMA skeleton. The root joint is `Hips`. Root features provide position, heading, velocity, and height constraints. SOMA model space is Y up, with XZ as the ground plane. Positions use meters. Each 6D rotation holds the first two columns of a joint's rotation matrix in SOMA model space. Context and target poses share the same space. The model manifest [soma_n1.json](../motionbricks/assets/onnx/soma/soma_n1.json) records the joint order and each tensor's name, shape, and dtype.

### Input Tensors

| Input | Shape | Type | Meaning |
| --- | --- | --- | --- |
| `global_root_values` | `[N,8,5]` | FP32 | Root position (XYZ), then the cosine and sine of the heading |
| `has_global_root_values` | `[N,8]` | INT32 | Mask for the root position and heading |
| `local_root_values` | `[N,8,4]` | FP32 | Heading rate in rad/s, root ground velocity along X and Z in m/s, and root height |
| `has_local_root_values` | `[N,8]` | INT32 | Mask for the root velocity and height |
| `local_poses` | `[N,8,267]` | FP32 | Positions of the 29 non-root joints relative to the root's ground position (87 values), then a 6D rotation for each of the 30 joints (180 values) |
| `has_local_poses` | `[N,8]` | INT32 | Mask for the body pose |
| `allowed_pred_num_tokens` | `[N,11]` | INT32 | Allowed token counts, 6 to 16 |

All seven inputs are required. Masks contain 0 or 1, with context slots first, then target slots. At least one token count must be allowed. Entry `i` of `allowed_pred_num_tokens` allows a token count of `6 + i`. Each motion token represents four poses.

## Outputs

**Output modality:** Skeletal motion features with predicted foot contacts, and the chosen token count.

**Output format:** Numeric ONNX tensors. Motion features use FP32, and the token count uses INT32. The output tensor table defines each parameter's name, shape, and meaning.

| Output | Shape | Type | Meaning |
| --- | --- | --- | --- |
| `pred_global_motions` | `[N,64,366]` | FP32 | Per frame: root position and heading (5), joint positions (87), 6D rotations (180), joint velocities (90), and foot contacts (4), in SOMA model space |
| `pred_num_tokens` | `[N]` | INT32 | Token count the model chose |

The leading `4 * pred_num_tokens` frames contain generated motion. The raw output contains 24 to 64 frames. Removing three boundary frames at each end leaves 18 to 58 valid frames at 30 Hz. The engine decodes these features into the root position and joint rotations.

The last four features of each frame are foot contacts for the left heel, left toe, right heel, and right toe. Their joints are `LeftFoot`, `LeftToeBase`, `RightFoot`, and `RightToeBase`. See [foot_contacts.py](../motionbricks/animation/foot_contacts.py) for their meaning.

## Software Integration

| Component | Integration |
| --- | --- |
| Tested inference runtime | NVIDIA TensorRT 10.8 |
| Application framework | Python and NVIDIA Warp |
| Demo environment | Newton |
| Tested operating system | Windows |
| Documented deployment hardware | NVIDIA Ada Lovelace GeForce RTX 4090 |

The application supplies placed context and target poses. The engine handles coordinate conversion and returns generated motion in world space.

The table identifies the tested setup, not a complete compatibility matrix. Timings are reported only for the GeForce RTX 4090 under Windows. The checkpoint-to-ONNX exporter runs on CPU and is separate from this GPU inference path. A CPU-versus-GPU inference comparison is not provided.

The integration of foundation and fine-tuned models into AI systems requires additional testing using use-case-specific data to ensure safe and effective deployment. Following the V-model methodology, iterative testing and validation at both unit and system levels are essential to mitigate risks, meet technical and functional requirements, and ensure compliance with safety and ethical standards before deployment.

### Runtime Generation

`MotionBricksSOMAEngine` runs the generation calls in [Engines](runtime_api.md#engines). Its `PoseToken` context and target hold local joint transforms. It generates into the same `AnimationBuffer` type that loaded BVH clips use.

The skeleton passed to `create_input()` and `create_output()` must contain the model's joint names, with the root at index zero. The engine maps model joints to this skeleton. Pass `model_to_runtime` when world space differs from SOMA model space.

Before inference, the engine removes the ground position and heading of the last context pose. It keeps the root height. After inference, it restores the world placement and removes three boundary frames at each end. By default, inference turns the last context pose to face +X in SOMA model space.

## Model Versions

**Model version:** `MotionBricks-SOMA-RP-v1`. The accompanying software release is v0.2.0.

| Precision | ONNX file |
| --- | --- |
| FP32 | `mmm_v6_soma_fp32.onnx` |
| Mixed FP16 | `mmm_v6_soma_fp16.onnx` |

Files are stored in [motionbricks/assets/onnx/soma](../motionbricks/assets/onnx/soma).

Motion inputs and outputs use FP32 in both variants. Masks and token counts use INT32. The [SOMA checkpoint on Hugging Face](https://huggingface.co/nvidia/MotionBricks/tree/main/soma) contains the SafeTensors weights used by the [exporter](../tools/onnx/README.md).

## Training, Testing and Evaluation Datasets

### Training Dataset

**Name/Identifier:** Bones RigPlay 1 (RP1).

**Data Modality:** Other: human skeletal motion capture processed into the uniform 30-joint SOMA representation at 30 Hz. The training inputs are motion features, not text, images, audio, or video.

**Training Data Size:** 313,526 human-motion clips in the selected training subset. The source corpus contains approximately 350,000 clips and 700 hours of motion. The source-corpus duration is not the selected training-subset duration.

**Data Collection Method:** Automatic/Sensors. Bones recorded human performances in a motion-capture studio and supplied skeletal motion. NVIDIA processed that motion into a uniform SOMA representation. The source performances were not captured directly on the SOMA skeleton.

**Labeling Method:** Automatic/Sensors for captured skeletal motion, followed by automated preparation into uniform motion features. Pose and motion values are the learning data. The recorded runs do not consume text embeddings or text annotations.

**Properties:** Training uses uniform SOMA motion features, random first headings, and random crops of up to 15 seconds. Root motion and joint features provide the motion representation. Text embeddings and text augmentation are disabled. All three checkpoint components were trained in FP32. The training corpus is not distributed with these exports.

**Linguistic Characteristics:** Not applicable to the motion-feature inputs. Associated text descriptions are dataset metadata, not model conditioning inputs.

### Testing Dataset

**Name, Size and Role:** We used 313,526 clips for training and a separate, selected set of 4,881 clips for validation and evaluation during training. No clips are shared between these sets. The 4,881 clips are a selection from the held-out pool, not the entire pool. A separate untouched final-test set, a performer-disjoint split, and the selected subset's duration are not documented.

**Data Collection Method:** Automatic/Sensors, from the same Bones studio motion-capture source and processed uniform SOMA representation as the training set.

**Labeling Method:** Automatic/Sensors for the skeletal motion, followed by automated preparation into the same motion features. Text annotations are not consumed by the recorded validation path.

**Properties:** The validation inputs are 30 Hz skeletal-motion features in the same uniform SOMA representation as the training inputs. Linguistic characteristics are not applicable to these motion-only inputs. Disjoint clips alone do not rule out other forms of data leakage.

### Evaluation Dataset and Results

**Name/Size:** Evaluation uses the same 4,881-clip RP1 subset described under [Testing Dataset](#testing-dataset). It is not an additional independent test set.

**Benchmark Score:** A complete end-to-end benchmark of these SOMA ONNX models is not provided.

**Data Collection Method:** Automatic/Sensors. Bones captured the source motion in a studio.

**Labeling Method:** Automatic/Sensors for captured skeletal motion, followed by automated preparation into uniform SOMA motion features.

**Properties:** SOMA skeletal-motion features at 30 Hz, without text-conditioned labels. Linguistic characteristics are not applicable to these inputs. The subset's duration is not documented.

**Recorded Results:** Training measured tokenizer reconstruction and foot skating, root-predictor validation, and pose-predictor training loss. A complete pose-predictor validation result is not provided. These measurements do not evaluate the exported ONNX models.

The ONNX models were tested in the setup listed under [Software Integration](#software-integration). The GPU time and throughput of the TensorRT plans at several batch sizes are in [Inference performance](onnx_inference.md#inference-performance). Quantitative motion quality and an FP32-versus-mixed-FP16 quality comparison are not reported. The [MotionBricks paper](https://arxiv.org/abs/2604.24833) describes method-level experiments on its own models and datasets, not a benchmark of these SOMA ONNX models.

## Inference

FP32 is the default. Run the demo with either precision:

```sh
uv run python -m app.scene_preview --model-precision fp32
uv run python -m app.scene_preview --model-precision fp16
```

The SOMA TensorRT configuration permits TF32 for eligible FP32 operations.

The mixed FP16 ONNX model uses FP16 for the neural predictors and the motion decoder. It keeps FP32 for feature normalization, motion geometry, LayerNormalization, Softmax, and Resize. The demos build a TensorRT plan for `N=1`.

Mixed FP16 and FP32 can produce different motion, because rounding affects the choice of motion tokens.

**Acceleration Engine:** NVIDIA TensorRT 10.8.

**Test Hardware Set-Up:** NVIDIA Ada Lovelace GeForce RTX 4090 on Windows.

The exporter checks checkpoint integrity, ONNX graph validity, and the expected tensor interface. Those structural checks do not establish complete security coverage, safe physical behavior, or equivalent motion quality across precisions and backends.

## Limitations

- The demos run one request at a time with the same batched ONNX model available to users. Context and target tokens hold at most four poses each, so longer motion requires repeated generation. See [batching](onnx_inference.md).
- The model uses the SOMA joint layout and proportions. Other skeletons require an appropriate representation or a retargeting step.
- Conditioning comes from poses and motion constraints. Text prompts and scene geometry are outside this model's input interface. The application places targets and handles obstacles.
- The model works best with target poses within its training distribution. Poses outside it can produce lower quality motion.
- Generated motion can vary with precision and inference runtime. Differences can accumulate when generated motion is fed back as context.
- Kinematic motion generation does not establish dynamic feasibility, physical robot tracking, collision avoidance, or safe contact. Physical execution requires a separate controller and application-specific validation.

## Ethical Considerations

NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. Developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.

For more detailed information, see the [Bias](#bias-subcard), [Explainability](#explainability-subcard), [Privacy](#privacy-subcard) and [Safety and Security](#safety-and-security-subcard) subcards below.

Please report model quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).

## Bias Subcard

| Field | Response |
| --- | --- |
| Participation considerations from adversely impacted groups in model design and testing | Not reported. |
| Measures taken to mitigate unwanted bias | Specific measured mitigation is not reported. Applications should evaluate motion quality for their users and motion conditions. |
| Bias Metric, if measured | Comparisons of demographic, body, and motion coverage are not reported. |

The fixed SOMA representation does not establish equivalent quality across body proportions or motion categories. Precision conversion does not establish new subgroup-performance evidence.

## Explainability Subcard

| Field | Response |
| --- | --- |
| Intended Task/Domain | Pose-conditioned human skeletal motion generation and inbetweening. |
| Model Type | Modular latent generative model, with transformer predictors and learned pose tokenizer and decoder. |
| Intended Users | Developers and researchers building animation, robotics, and simulation workflows. |
| Output | Batched skeletal-motion features and selected token count through the two-output ONNX interface. The engine decodes features into the root position and joint rotations. |
| Describe how the model works | Recent motion and target constraints condition root and pose predictors. A tokenizer represents motion as discrete codes, and a decoder reconstructs skeletal motion. The combined graph includes the inference feature transformations. Applications can reuse output as context. |
| Groups tested to produce comparable outcomes | Demographic subgroup comparisons are not reported. |
| Technical Limitations and Mitigation | Fixed skeleton, short context, no scene-geometry input, and variation across backends and precisions. Validate application-specific motion, target placement, and any physical execution separately. |
| Verified to have met prescribed NVIDIA quality standards | Yes |
| Performance Metrics | See [Evaluation Dataset and Results](#evaluation-dataset-and-results) for the available measurements and their limits. |
| Potential Known Risks | Pose/contact artifacts, unwanted motion under out-of-distribution constraints, and unsafe physical interpretation if generated poses are treated as validated robot commands. |
| Licensing | Use of this model is governed by the [NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/). See [License and Terms of Use](#license-and-terms-of-use) for separate software and sample-data terms. |

## Privacy Subcard

The datasets used to train this model were reviewed before release. Please review [NVIDIA’s Privacy Policy](https://www.nvidia.com/en-us/about-nvidia/privacy-policy/) for more information.

## Safety and Security Subcard

| Field | Response |
| --- | --- |
| Model Application Fields | Industrial/Machinery and Robotics, Media & Entertainment. Generated motion is not validated for physical robot control. |
| Life-critical impact | Generated skeletal motion is not a validated physical controller. Misuse as robot commands can create physical risk. |
| Use Case Restrictions | Use of this model is governed by the [NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/). See [License and Terms of Use](#license-and-terms-of-use) for separate software and sample-data terms. |
| Model and dataset restrictions | These ONNX files do not distribute the RP1 training corpus. Access to the dataset is governed separately from the model weights. |

Evaluate generated motion in the intended application. Physical execution requires a separate tracking and safety system. Artifact-integrity and graph checks do not establish complete security or safe downstream behavior.
