# I-3D — Single Image → 3D Character → Rig → Two Poses → MotionBricks Animation

> **I-3D_web — isolated experimental WebGPU branch.** This is **not** `main`. New custom WGSL compute kernels, the WebGPU runtime, browser tests, and a GPU-resident depth/normal viewer are in [webgpu/](webgpu/README.md). The branch-specific [Colab launcher](https://colab.research.google.com/github/Apache0ne/I-3D/blob/I-3D_web/notebooks/I-3D_End_to_End_Colab.ipynb) clones `I-3D_web`. Existing full Hi3D/Paint/TokenRig/MotionBricks inference still uses CUDA/TensorRT: **full WebGPU model ports are not yet implemented**. Do not merge this branch into `main` without an explicit new request.

**Research implementation, October 2026.** One canonical textured mesh is created and rigged once. The user edits *the same 3D character* in the embedded Colab 3D Studio; the saved poses are converted into Blender keyframes headlessly and MotionBricks synthesizes the transition. There is no second 3D reconstruction or requirement to retrain MotionBricks.

## I-3D_web stage 0: BRIA RMBG 2.0 ONNX browser matting

The experimental WebGPU branch now has a standalone, real-model background-removal integration at [webgpu/rmbg.html](webgpu/rmbg.html), accessible from [the branch-specific Colab notebook](https://colab.research.google.com/github/Apache0ne/I-3D/blob/I-3D_web/notebooks/I-3D_End_to_End_Colab.ipynb) before installing CUDA models. It supports the user's [kn4666 BRIA RMBG 2.0 ONNX weights](https://huggingface.co/kn4666/bria-rmbg-2.0-web) as experimental variants, plus the [separately patched WebGPU FP16 export](https://huggingface.co/yamura4/RMBG-2.0-WebGPU) as the default. It downloads weights to the **local browser**, runs through **ONNX Runtime Web's WebGPU EP**, and produces original RGB, full-resolution transparent PNG and alpha-mask PNG.

Files go directly to the Colab viewer workspace through a bounded PNG endpoint, and the branch's geometry command accepts `--cutout`, validates its source/alpha SHA256 manifest and uses it for Hi3D geometry while retaining the original image for PBR painting. No alpha-model weights are committed to GitHub. **CC BY-NC 4.0 applies** to the BRIA weights, so this stage is for noncommercial research absent a separate license. The WebGPU model graph is handled by ONNX Runtime, not yet by the custom I-3D WGSL graph executor. **The browser model has not been tested end to end on actual hardware**; CPU and syntax checks passed.

## Simple Colab launcher (recommended)

**The notebook is only a launcher.** [Open I-3D in Google Colab](https://colab.research.google.com/github/Apache0ne/I-3D/blob/I-3D_web/notebooks/I-3D_End_to_End_Colab.ipynb). It clones this repository, installs through the repository CLI, and calls the original model, rigging, MotionBricks and viewer scripts. No dependency lists or inference implementations live in notebook cells.

The same commands work in a Colab terminal after cloning:

```bash
git clone https://github.com/Apache0ne/I-3D.git
cd I-3D
python scripts/colab.py setup --stage all --precision fp16
python scripts/colab.py prepare --image /content/person.png --forward +X --resolution 512
# Open the notebook's embedded 3D Studio, save START and TARGET poses.
python scripts/colab.py animate --precision fp16
python scripts/colab.py pack
```

All CUDA and Python setup is in `scripts/colab.py` and its helper scripts. Stages are restartable: `setup --stage hi3d`, `paint`, `tokenrig` or `motion` installs only a selected environment. `prepare` reuses only valid GLBs with matching content hashes, model settings and stage source code; changing the image or options automatically invalidates downstream outputs. `--force` regenerates the stages. Subprocess errors are emitted directly. All stages are **real model operations**, not placeholders, and remain dependent on weights, GPUs and upstream licenses. The external CUDA-13 adaptations and TensorRT engine are **not yet validated end to end**. Blender executes headlessly for rigging and GLB export; it does not require opening Blender's interface.

## Google Colab notebook

Open [I-3D_End_to_End_Colab.ipynb](notebooks/I-3D_End_to_End_Colab.ipynb) in [Google Colab](https://colab.research.google.com/github/Apache0ne/I-3D/blob/I-3D_web/notebooks/I-3D_End_to_End_Colab.ipynb). It installs separate model environments, downloads the original external model assets, walks through geometry, paint, auto-rigging, SOMA bridge, two manually authored 3D poses, and MotionBricks inference/export. Use an NVIDIA GPU; heavier Hi3D modes may exceed Colab GPU memory. The notebook has not been executed or validated in Colab.

## Interactive 3D Studio — no Blender GUI needed

The [Colab notebook](https://colab.research.google.com/github/Apache0ne/I-3D/blob/I-3D_web/notebooks/I-3D_End_to_End_Colab.ipynb) opens the new [Three.js production viewer](viewer/index.html) directly in a large output cell using [viewer_server.py](scripts/viewer_server.py). It reads all five generated GLBs and supporting output artifacts from the same Colab workspace.

**Viewport features:** PBR lighting, editable exposure, multiple light presets, scene background, wireframe, normal/depth/albedo/clay/skin-weight views, skeleton overlay, joint labels, mesh inspector, joint picking, technical metadata, ground/grid/axes/bounds, image reference overlay, fullscreen, orbit/pan/zoom, front/side/top camera presets, side-by-side stage comparison and GLB downloads. For animated GLBs, the viewer has play/pause, clip selection, timeline seeking, single-frame stepping, speed controls and looping. It can save screenshots as PNG and record browser-rendered video as WebM.

**In-Colab pose editor:** On the SOMA stage, select the actual 30-joint character bones and edit local rotations using the gizmo or numeric fields; move Hips in the right-handed **glTF Y-up** coordinate frame. Save **START** and **TARGET** in the viewer. They become `outputs/character/viewer_poses.json`, and [author_viewer_poses.py](scripts/author_viewer_poses.py) converts those poses with the explicit glTF Y-up → Blender Z-up rotation into a two-frame `.blend` file. The existing MotionBricks step then creates the animated GLB, which can immediately be viewed and recorded without leaving Colab. Blender runs **headlessly as an internal export dependency**; no Blender GUI or desktop installation is required by the user.

**Boundaries:** This is new original I-3D integration code. GPU inference and Three.js-to-Blender joint deformation on real generated characters have not yet been validated end-to-end; review orientation, proportions and skinning before relying on results. The viewer loads the Three.js MIT-licensed browser libraries from jsDelivr at runtime, and requires the browser to reach that CDN. It is served by the local Colab port proxy, not a hosted remote service.

## End-to-end pipeline

| Stage | Original implementation | Output |
|---|---|---|
| 1. Image/background removal | Hi3D demo background removal (rembg); BRIA RMBG-2.0 optional | Image with transparent subject |
| 2. Coarse geometry | Tencent Hunyuan3D-2mini Turbo (`demo/coarse_stage.py`) | Coarse 3D mesh |
| 3. Refined geometry | Hi3D v1.0 (Sparc3D-based refinement) | High-detail mesh |
| 4. UV/PBR textures | Hunyuan3D-Paint 2.1 | Textured GLB |
| 5. Auto-rigging | TokenRig / SkinTokens | Skinned GLB with predicted skeleton |
| 6. SOMA bridge **(original I-3D code)** | Actual NVIDIA SOMA reference USD + bone/weight remapping | SOMA 30-joint skinned GLB |
| 7. Author start/target | 3D Studio edits, converted to Blender keyframes headlessly | `viewer_poses.json` + `05_poses.json` |
| 8. Generate motion | MotionBricks-SOMA-RP-v1, FP32/mixed FP16 ONNX | 30 FPS joint transforms NPZ |
| 9. Animate/export | Blender rest/pose transforms and glTF | Animated, textured GLB + editable `.blend` |

## CUDA 13 Colab installation (October 2026)

The [updated Colab notebook](notebooks/I-3D_End_to_End_Colab.ipynb) uses CUDA **13.0** for its PyTorch model stages. Do **not** run the older CUDA-12 installation cells from a stale Colab tab. Restart the Colab runtime and reopen the notebook from `main`.

| Stage | Research environment | CUDA 13 status |
|---|---|---|
| Hi3D + Hunyuan3D-2mini | Python 3.10, PyTorch 2.11.0+cu130, third-party cu130 cumm/spconv wheels | **Experimental** (official Hi3D only validated cu124 / PyTorch 2.5) |
| Hunyuan3D-Paint 2.1 | Python 3.10, PyTorch 2.11.0+cu130, `cupy-cuda13x`, original rasterizer built with nvcc 13.0 | **Install path updated**, inference unverified |
| SkinTokens/TokenRig | Python 3.11, PyTorch 2.11.0+cu130 | **Install path updated**, inference unverified |
| MotionBricks-SOMA | Python 3.12, original NVIDIA TensorRT runtime | Upstream TensorRT dependency retained; **not** a native cu130 Torch stage |

I-3D install helpers:
- `python scripts/install_hi3d_cuda13.py --python /content/i3d_envs/hi3d/bin/python`
- `python scripts/install_paint_cuda13.py --python /content/i3d_envs/paint/bin/python`

CUDA 13 SpConv/cumm wheels are sourced from [fafraob/spconv-wheels](https://github.com/fafraob/spconv-wheels), an **unofficial third-party CUDA-13 rebuild** of upstream SpConv/cumm. The documented matching pair is SpConv 2.3.8 + cumm 0.8.2. Inspect its source, releases and licensing before running. We do not quietly fall back to CUDA-12 binaries; errors are reported for debugging. Original pretrained model weights are unchanged, and **this port has not yet passed a GPU end-to-end run**. The original Hi3D/Tencent licenses and TensorRT dependencies still apply.

## Install upstream source and models

`python scripts/vendor_sources.py` clones the **complete upstream source trees** at their current commits into `vendor/`, preserving original LICENSE/NOTICE files. To pin exact commits, use `--revision-file pins.json`. Large model weights and checkpoints are deliberately not stored in this Git repository. They must be downloaded from the upstream model releases under their own licenses. Do not claim bundled NVIDIA sample motions are unrestricted; those have separate evaluation terms.

**Each upstream model needs its original tested environment.** Hi3D officially documents Python 3.10 / CUDA 12.4 and very high GPU memory for its 1536-resolution mode; this I-3D notebook now experiments with CUDA-13 kernels instead (not Hi3D-verified). TokenRig recommends >=14 GB VRAM; MotionBricks uses TensorRT. Do not install incompatible PyTorch stacks into a single venv. Set `I3D_HI3D_PYTHON`, `I3D_PAINT_PYTHON`, `I3D_TOKENRIG_PYTHON`, and the `--motion-python` option to the correct environment interpreters. Blender is a separate executable. Pretrained files are **not included**.

MotionBricks Git LFS assets can be hydrated with `python scripts/hydrate_motionbricks.py --precision fp32`. This retrieves the original 30-joint reference USD and ONNX graph without committing weights to I-3D. Other model weights must be downloaded under their upstream terms. Hi3D's pipeline requires the coarse Hunyuan3D-2mini dependencies installed per the upstream Hi3D documentation. For Hunyuan3D-Paint, download the RealESRGAN checkpoint according to its README.

## Optional legacy manual-Blender workflow

The primary Colab workflow above does **not** require opening Blender. The older `scripts/pipeline.py` remains available when manually authored desktop Blender poses are preferred.

```bash
# Stage 1 — 2D image -> textured mesh -> rigged character -> SOMA bridge
python scripts/pipeline.py prepare --image person.png --output outputs/character --forward +X

# Stage 2 — manually open outputs/character/04_soma_character.glb in Blender.
# Keep one armature; edit its bones in POSE MODE (move the Hips pose bone
# for world travel; do not move/scale the Armature object). Keyframe all bones
# at frame 1 (Start) and frame 2 (Target). Save character_poses.blend.

# Stage 3 — MotionBricks generation + animated GLB
python scripts/pipeline.py animate --blend character_poses.blend --output outputs/character
```

`--forward` is REQUIRED because single-image character reconstruction does not reliably reveal canonical facing. In Blender Z-up, +X/+Y/-X/-Y are allowed. Confirm the model visually faces this direction before running; incorrect forward selection causes a failed left/right test or misaligned bone weights. No mirror, negative scale or heuristic axis swapping is applied.

## Original bridge limitations

The `blender_bridge.py` does **not** pretend to use NVIDIA's topology-constrained `rig_soma_body_mesh.py` with an arbitrary Hi3D mesh. Instead, it aligns the actual SOMA reference rig with the mesh and remaps existing TokenRig weights to the nearest joints. This is an approximate **spatial skin-transfer** method; its accuracy depends on the canonical pose, the TokenRig joint positions, and the correct `--forward` selection. It rejects degenerate scales and gross misalignment, and writes `*.bridge.json` with mapping diagnostics. **Manual skin weight and joint review remains necessary before calling the result production-ready.** A-pose/T-pose inputs are preferable.

`export_soma_reference.py` composes the actual MotionBricks 30-joint USD **local rest transforms into world rest transforms**, exactly as the original MotionBricks skeleton loader expects. USD bind transforms are different from local rest transforms and must not be substituted. It rejects Git LFS pointer files and implausible units. NVIDIA's alternative `tools/rig_soma_body_mesh.py` requires exactly 18,056 vertices with identical SOMA template topology and **must not** be applied directly to an arbitrary Hi3D mesh.

## Validation

```bash
pip install numpy
python -m unittest discover -s tests -v
```

These CPU tests cover: SOMA Y-up → Blender Z-up basis, positive determinants, non-mirrored left/right anatomy, non-reflective pose fitting, local/world forward kinematics, world-to-local conversion, quaternion round-trips, binary GLB validation, cache invalidation, a mocked Hunyuan Paint OBJ→GLB contract, and synthetic 30-joint MotionBricks input tensors. GitHub Actions also compiles the Python sources and smoke-tests the Colab CLI. Actual GPU end-to-end testing needs the licensed model weights, Blender and an NVIDIA GPU and has **not** been completed by this repository's CPU tests.

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