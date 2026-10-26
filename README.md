# I-3D — Single Image → 3D Character → Rig → Two Poses → MotionBricks Animation

**Research implementation, October 2026.** One canonical textured mesh is created and rigged once. The user edits *the same 3D character* at two Blender frames; MotionBricks synthesizes the transition. There is no 2D pose editor, no duplicate mesh reconstruction, and no requirement to retrain MotionBricks.

## End-to-end pipeline

| Stage | Original implementation | Output |
|---|---|---|
| 1. Image/background removal | Hi3D demo background removal (rembg); BRIA RMBG-2.0 optional | Image with transparent subject |
| 2. Coarse geometry | Tencent Hunyuan3D-2mini Turbo (`demo/coarse_stage.py`) | Coarse 3D mesh |
| 3. Refined geometry | Hi3D v1.0 (Sparc3D-based refinement) | High-detail mesh |
| 4. UV/PBR textures | Hunyuan3D-Paint 2.1 | Textured GLB |
| 5. Auto-rigging | TokenRig / SkinTokens | Skinned GLB with predicted skeleton |
| 6. SOMA bridge **(original I-3D code)** | Actual NVIDIA SOMA reference USD + bone/weight remapping | SOMA 30-joint skinned GLB |
| 7. Author start/target | Blender poses keyframed at frames 1 and 2 | `poses.json` |
| 8. Generate motion | MotionBricks-SOMA-RP-v1, FP32/mixed FP16 ONNX | 30 FPS joint transforms NPZ |
| 9. Animate/export | Blender rest/pose transforms and glTF | Animated, textured GLB + editable `.blend` |

## Install upstream source and models

`python scripts/vendor_sources.py` clones the **complete upstream source trees** at their current commits into `vendor/`, preserving original LICENSE/NOTICE files. To pin exact commits, use `--revision-file pins.json`. Large model weights and checkpoints are deliberately not stored in this Git repository. They must be downloaded from the upstream model releases under their own licenses. Do not claim bundled NVIDIA sample motions are unrestricted; those have separate evaluation terms.

**Each upstream model needs its original tested environment.** Hi3D documents Python 3.10 / CUDA 12.4 and very high GPU memory for its 1536-resolution mode; TokenRig recommends >=14 GB VRAM; MotionBricks uses TensorRT. Do not install incompatible PyTorch stacks into a single venv. Set `I3D_HI3D_PYTHON`, `I3D_PAINT_PYTHON`, `I3D_TOKENRIG_PYTHON`, and the `--motion-python` option to the correct environment interpreters. Blender is a separate executable. Pretrained files are **not included**.

Git LFS assets must be pulled separately from the original projects, particularly MotionBricks `soma_base_skel_minimal_no_root.usd`, and the pretrained ONNX models. Hi3D's pipeline requires the coarse Hunyuan3D-2mini dependencies installed per the upstream Hi3D documentation. For Hunyuan3D-Paint, download the RealESRGAN checkpoint according to its README.

## Run

```bash
# Stage 1 — 2D image -> textured mesh -> rigged character -> SOMA bridge
python scripts/pipeline.py prepare --image person.png --output outputs/character --forward +X

# Stage 2 — manually open outputs/character/04_soma_character.glb in Blender.
# Keep one armature; pose frame 1 (Start), keyframe every bone;
# pose frame 2 (Target), keyframe every bone. Save character_poses.blend.

# Stage 3 — MotionBricks generation + animated GLB
python scripts/pipeline.py animate --blend character_poses.blend --output outputs/character
```

`--forward` is REQUIRED because single-image character reconstruction does not reliably reveal canonical facing. In Blender Z-up, +X/+Y/-X/-Y are allowed. Confirm the model visually faces this direction before running; incorrect forward selection causes a failed left/right test or misaligned bone weights. No mirror, negative scale or heuristic axis swapping is applied.

## Original bridge limitations

The `blender_bridge.py` does **not** pretend to use NVIDIA's topology-constrained `rig_soma_body_mesh.py` with an arbitrary Hi3D mesh. Instead, it aligns the actual SOMA reference rig with the mesh and remaps existing TokenRig weights to the nearest joints. This is an approximate **spatial skin-transfer** method; its accuracy depends on the canonical pose, the TokenRig joint positions, and the correct `--forward` selection. It rejects degenerate scales and gross misalignment, and writes `*.bridge.json` with mapping diagnostics. **Manual skin weight and joint review remains necessary before calling the result production-ready.** A-pose/T-pose inputs are preferable.

`export_soma_reference.py` reads the actual MotionBricks 30-joint USD bind transforms. It rejects Git LFS pointer files and implausible units. NVIDIA's alternative `tools/rig_soma_body_mesh.py` requires exactly 18,056 vertices with identical SOMA template topology and **must not** be applied directly to an arbitrary Hi3D mesh.

## Validation

```bash
pip install numpy
python -m unittest discover -s tests -v
```

These CPU tests cover: SOMA Y-up → Blender Z-up basis, positive determinants, non-mirrored left/right anatomy, non-reflective pose fitting, local/world forward kinematics, world-to-local conversion and quaternion rotation round-trips. Actual GPU end-to-end testing needs the licensed model weights, Blender and an NVIDIA GPU and has **not** been completed by this repository's CPU tests.

## Licenses and citations

**Licenses must be respected independently.** Credits are not a substitute for the original licenses, notices, weight terms or usage restrictions. When upstream code is copied into `vendor/`, leave the upstream copyright headers, LICENSE and NOTICE documents intact.

- **Hi3D v1.0** — OCEANO HK HOLDING LIMITED. [Code](https://github.com/hi3d-official/Hi3Dv1.0), [weights](https://huggingface.co/hi3d-official/Hi3Dv1.0), [Hi3D Open License Agreement v1.0](https://github.com/hi3d-official/Hi3Dv1.0/blob/main/LICENSE); commercial integrations may require **Powered by Hi3D 1.0** attribution and scale-based licensing.
- **Hunyuan3D-2 & 2mini** — Tencent. [Code](https://github.com/Tencent-Hunyuan/Hunyuan3D-2), [weights](https://huggingface.co/tencent/Hunyuan3D-2mini), model and code terms apply.
- **Hunyuan3D-Paint 2.1** — Tencent. [Code](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1), [weights](https://huggingface.co/tencent/Hunyuan3D-2.1), Tencent Hunyuan 3D 2.1 Community License (including territorial restrictions).
- **SkinTokens / TokenRig** — Zhang et al., *SkinTokens: A Learned Compact Representation for Unified Autoregressive Rigging* (2026), [paper](https://arxiv.org/abs/2602.04805), [code](https://github.com/VAST-AI-Research/SkinTokens), MIT License.
- **SOMA-X** — Saito et al., *SOMA: Unifying Parametric Human Body Models* (2026), [paper](https://arxiv.org/abs/2603.16858), [code](https://github.com/NVlabs/SOMA-X), Apache 2.0.
- **MotionBricks** — Wang et al., *MotionBricks: Scalable Real-Time Motions with Modular Latent Generative Model and Smart Primitives* (2026), [paper](https://arxiv.org/abs/2604.24833), [code](https://github.com/NVlabs/motionbricks), Apache 2.0 code; NVIDIA Open Model Agreement weights; motion sample data evaluation license.
- **BRIA RMBG-2.0** (optional) — [model](https://huggingface.co/briaai/RMBG-2.0), CC BY-NC 4.0 for noncommercial use; separate commercial licensing.

The original I-3D bridge source is newly written and should not be presented as an NVIDIA-endorsed or upstream component.
